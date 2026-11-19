"""Local server of Borealis-5b-it for the Gate 0 spike (#68, docs/decisions/0010-gate0-borealis.md).

`python -m sound_scout.eval.gate0_borealis --model-dir D --host 127.0.0.1 --port P --threads N`
answers the four `llama-server` routes `sound_scout.eval.gate0` uses: `GET /health`,
`POST /apply-template`, `POST /tokenize`, `POST /v1/chat/completions`. The request plan, the rule
and the verdict stay in `gate0`; this module only turns one request into the model's input.

The model's own `from_pretrained` is not used: it fetches configs and a tokenizer from other
repositories and builds an fp32 skeleton (peak about 29 GB). The skeleton is built on `meta`
from constants of the revisions read for the Decision, the tensors are read one at a time from
`model.safetensors` and cast F16 -> bf16. `modeling_borealis.py` of the weights directory is
imported by path for its adapter and audio path; `trust_remote_code` is not used anywhere.

The socket binds to 127.0.0.1 first, then the network guard goes on: nothing else connects.
`torch` and `transformers` are imported only by `load_engine`, so the module and its request
contract are testable without them. A test with a fake engine proves that contract, never what
the model hears (CLAUDE.md).
"""

import argparse
import base64
import binascii
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import struct
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from sound_scout import config, guard

REPO_ID = "Vikhrmodels/Borealis-5b-it"
REVISION = "66e8899f74a7cfa8931dbdbb8a907534b927ddb4"
HOST = "127.0.0.1"
SAMPLE_RATE = 16_000
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}

# The stock `from_pretrained` adds <|start_of_audio|> and <|end_of_audio|> to the 151 669 tokens
# of the Qwen3-4B tokenizer; the model's own tokenizer.json names these ids differently.
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
IM_END_ID = 151645
PAD_ID = 151643
SYSTEM_PROMPT = "You are a helpful voice assistant. Listen to the audio and respond appropriately."
AUDIO_SLOT = "\x00audio\x00"

# Qwen/Qwen3-4B config.json at 1cfa9a7208912126459214e8b04321603b3df60c; the vocabulary size is
# the checkpoint's. The `text_config` of the model's config.json does not match the weights.
QWEN3_4B = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 40960,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "attention_bias": False,
    "tie_word_embeddings": True,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
}
# openai/whisper-large-v3 config.json and preprocessor_config.json at
# 06f233fe06e710322aca913c1bc4249a0d71fce1: the encoder and the log-mel front end.
WHISPER_ENCODER = {
    "d_model": 1280,
    "encoder_layers": 32,
    "encoder_attention_heads": 20,
    "encoder_ffn_dim": 5120,
    "num_mel_bins": 128,
    "max_source_positions": 1500,
    "activation_function": "gelu",
    "scale_embedding": False,
}
FEATURE_EXTRACTOR = {
    "feature_size": 128,
    "sampling_rate": SAMPLE_RATE,
    "hop_length": 160,
    "chunk_length": 30,
    "n_fft": 400,
    "padding_value": 0.0,
}
ADAPTER_FIELDS = ("downsample_factor", "audio_hidden_size", "llm_hidden_size")
CHECKPOINT_DTYPE = "F16"
EMBED_KEY = "llm.model.embed_tokens.weight"
LM_HEAD_KEY = "llm.lm_head.weight"
MAX_HEADER_BYTES = 100 << 20
MAX_BODY_BYTES = 64 << 20


class BadRequest(Exception):
    """The request does not fit the routes this server answers (HTTP 400)."""


class LoadError(Exception):
    """The weights do not fit the skeleton; the server refuses to start."""


# --- request -> model input ----------------------------------------------------------------


def render(user_text: str) -> str:
    """ChatML of Qwen3 as the stock `apply_chat_template` gives it without `enable_thinking`."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def user_parts(body: object) -> list[str | bytes]:
    """The parts of the one user message in order: text as `str`, the audio part as PCM bytes."""
    try:
        (message,) = body["messages"]
        role, content = message["role"], message["content"]
    except (KeyError, TypeError, ValueError):
        raise BadRequest("messages must hold exactly one message with role and content") from None
    if role != "user":
        raise BadRequest(f"the message role is {role!r}, expected 'user'")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list) or not content:
        raise BadRequest("content must be a string or a non-empty list of parts")
    parts: list[str | bytes] = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            if AUDIO_SLOT in part["text"]:
                raise BadRequest("text holds a reserved character sequence")
            parts.append(part["text"])
        elif kind == "input_audio" and isinstance(part.get("input_audio"), dict):
            parts.append(wav_pcm(part["input_audio"]))
        else:
            raise BadRequest(f"unsupported content part {kind!r}")
    if sum(isinstance(part, bytes) for part in parts) > 1:
        raise BadRequest("more than one audio part; the model takes one audio")
    return parts


def wav_pcm(audio: dict) -> bytes:
    """PCM s16le of a non-empty 16 kHz mono WAV sent as base64."""
    if audio.get("format") != "wav" or not isinstance(audio.get("data"), str):
        raise BadRequest("input_audio must be base64 WAV with format 'wav'")
    try:
        data = base64.b64decode(audio["data"], validate=True)
        with wave.open(io.BytesIO(data), "rb") as source:
            layout = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            pcm = source.readframes(source.getnframes())
    except (binascii.Error, wave.Error, EOFError) as error:
        raise BadRequest(f"input_audio is not a WAV: {error}") from None
    if layout != (1, 2, SAMPLE_RATE):
        raise BadRequest(f"WAV is {layout}, expected (1, 2, {SAMPLE_RATE})")
    if len(pcm) < 2:
        raise BadRequest("WAV holds no samples")
    return pcm


def text_of(body: object) -> str:
    """The rendered prompt of a text-only message (`/apply-template`)."""
    parts = user_parts(body)
    if any(isinstance(part, bytes) for part in parts):
        raise BadRequest("/apply-template takes text parts only")
    return render("\n".join(parts))


def split_reasoning(text: str) -> tuple[str, str | None]:
    """(content, reasoning): the text inside <think>…</think> leaves the content."""
    if "<think>" not in text and "</think>" not in text:
        return text, None
    if "<think>" not in text:
        reasoning, _, content = text.partition("</think>")
        return content.strip(), reasoning.strip()
    before, _, rest = text.partition("<think>")
    reasoning, _, after = rest.partition("</think>")
    return (before + after).strip(), reasoning.strip()


def complete(engine, body: object) -> dict:
    """One chat completion in the `llama-server` response form."""
    if not isinstance(body, dict):
        raise BadRequest("the body must be a JSON object")
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise BadRequest("max_tokens must be a positive integer")
    parts = user_parts(body)
    chat = render("\n".join(AUDIO_SLOT if isinstance(part, bytes) else part for part in parts))
    pcm = next((part for part in parts if isinstance(part, bytes)), None)
    started = time.monotonic()
    extra = {}
    if pcm is None:
        before, audio, after, positions = engine.tokenize(chat), None, [], 0
    else:
        prefix, suffix = chat.split(AUDIO_SLOT)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        audio, positions, mel_sha256 = engine.encode_audio(samples)
        before = [*engine.tokenize(prefix), AUDIO_START_ID]
        after = [AUDIO_END_ID, *engine.tokenize(suffix)]
        extra = {
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "mel_sha256": mel_sha256,
            "audio_positions": positions,
        }
    encode_ms = (time.monotonic() - started) * 1000
    generated = engine.generate(before, audio, after, max_tokens)
    generate_ms = (time.monotonic() - started) * 1000 - encode_ms
    stopped = bool(generated) and generated[-1] == IM_END_ID
    finish = "length" if not stopped and len(generated) >= max_tokens else "stop"
    text = engine.detokenize(generated[:-1] if stopped else generated)
    content, reasoning = split_reasoning(text)
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    prompt_tokens = len(before) + positions + len(after)
    response = {
        "object": "chat.completion",
        "model": f"borealis-{REVISION}",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(generated),
            "total_tokens": prompt_tokens + len(generated),
        },
        # Not llama-server's prompt_ms/predicted_ms: the LLM prefill runs inside generate.
        "timings": {
            "encode_ms": round(encode_ms, 3),
            "generate_ms": round(generate_ms, 3),
            "predicted_n": len(generated),
        },
    }
    if extra:
        response["borealis"] = extra
    if "response_format" in body:
        response["response_format"] = "not_enforced"
    if "chat_template_kwargs" in body:
        response["chat_template_kwargs"] = "ignored"
    return response


# --- HTTP ----------------------------------------------------------------------------------


class Server(HTTPServer):
    engine = None


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        if self.server.engine is None:
            return self._send(503, {"status": "loading"})
        self._send(200, {"status": "ok", "network_guard_active": guard.network_guard_active()})

    def do_POST(self) -> None:
        engine = self.server.engine
        if engine is None:
            return self._send(503, {"status": "loading"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                raise BadRequest("the body is empty or too large")
            body = json.loads(self.rfile.read(length))
            if self.path == "/apply-template":
                answer = {"prompt": text_of(body)}
            elif self.path == "/tokenize":
                if not isinstance(body, dict) or not isinstance(body.get("content"), str):
                    raise BadRequest("content must be a string")
                answer = {"tokens": engine.tokenize(body["content"])}
            elif self.path == "/v1/chat/completions":
                answer = complete(engine, body)
            else:
                return self._send(404, {"error": "not found"})
        except BadRequest as error:
            return self._send(400, {"error": str(error)})
        except ValueError as error:
            return self._send(400, {"error": f"the body is not JSON: {error}"})
        self._send(200, answer)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(host: str, port: int, load) -> int:
    """Bind, install the network guard or refuse to start, then load the model (`/health` 503)."""
    if host != HOST:
        print(f"gate0_borealis: host must be {HOST}, got {host!r}", file=sys.stderr)
        return 2
    os.environ.update(OFFLINE_ENV)
    try:
        settings = config.load()
    except config.ConfigError as error:
        print(f"gate0_borealis: {error}", file=sys.stderr)
        return 2
    server = Server((host, port), Handler)
    if not guard.install_network_guard(settings):
        server.server_close()
        print("gate0_borealis: the network guard is off ([network] allow = true)", file=sys.stderr)
        return 2
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        server.engine = load()
    except LoadError as error:
        print(f"gate0_borealis: {error}", file=sys.stderr)
        server.shutdown()
        return 3
    print("gate0_borealis: ready", file=sys.stderr, flush=True)
    threading.Event().wait()
    return 0


# --- the model -----------------------------------------------------------------------------


def read_header(path: Path) -> dict[str, dict]:
    """The tensor table of a safetensors file (name -> dtype, shape, offsets)."""
    with open(path, "rb") as source:
        (size,) = struct.unpack("<Q", source.read(8))
        if size > MAX_HEADER_BYTES:
            raise LoadError(f"{path.name}: header of {size} bytes")
        header = json.loads(source.read(size))
    header.pop("__metadata__", None)
    return header


def check_header(header: dict[str, dict], expected: dict[str, tuple[int, ...]]) -> None:
    """Every tensor is F16, the names are the skeleton's and every shape matches."""
    foreign = sorted(set(header) - set(expected))
    missing = sorted(set(expected) - set(header))
    if foreign or missing:
        raise LoadError(f"checkpoint keys differ: foreign {foreign[:5]}, missing {missing[:5]}")
    dtypes = sorted({entry["dtype"] for entry in header.values()})
    if dtypes != [CHECKPOINT_DTYPE]:
        raise LoadError(f"checkpoint dtypes are {dtypes}, expected [{CHECKPOINT_DTYPE!r}]")
    shapes = [name for name, entry in header.items() if tuple(entry["shape"]) != expected[name]]
    if shapes:
        raise LoadError(f"checkpoint shapes differ from the skeleton: {sorted(shapes)[:5]}")


def load_weights(model, path: Path) -> None:
    """Stream F16 tensors into a `meta` skeleton as bf16, one at a time; tie `lm_head`."""
    import torch
    from safetensors import safe_open

    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    check_header(read_header(path), expected)

    def read(name: str):
        # opened per tensor so the mapped pages of the file do not pile up in the RSS
        with safe_open(str(path), framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    for name in sorted(expected):
        if name == LM_HEAD_KEY:
            continue
        module_name, _, attribute = name.rpartition(".")
        tensor = read(name).to(torch.bfloat16)
        module = model.get_submodule(module_name)
        module._parameters[attribute] = torch.nn.Parameter(tensor, requires_grad=False)
    if not torch.equal(read(LM_HEAD_KEY), read(EMBED_KEY)):
        raise LoadError(f"{LM_HEAD_KEY} differs from {EMBED_KEY}; the weights are not tied")
    model.llm.lm_head.weight = model.llm.model.embed_tokens.weight


def load_engine(model_dir: Path, threads: int) -> "TorchEngine":
    os.environ.update(OFFLINE_ENV)
    import torch
    from transformers import (
        PretrainedConfig,
        Qwen2Tokenizer,
        Qwen3Config,
        Qwen3ForCausalLM,
        WhisperConfig,
        WhisperFeatureExtractor,
    )
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    torch.set_num_threads(threads)
    spec = importlib.util.spec_from_file_location(
        "modeling_borealis", model_dir / "modeling_borealis.py"
    )
    modeling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling)
    settings = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    # BorealisConfig of this revision fails the config validation of transformers 5: the
    # adapter needs only these three fields
    adapter = PretrainedConfig()
    for field in ADAPTER_FIELDS:
        setattr(adapter, field, settings[field])
    llm_config = Qwen3Config(**QWEN3_4B, vocab_size=settings["vocab_size"])
    with torch.device("meta"):
        model = modeling.BorealisForConditionalGeneration(adapter)
        model.encoder = WhisperEncoder(WhisperConfig(**WHISPER_ENCODER))
        model.llm = Qwen3ForCausalLM(llm_config)
    load_weights(model, model_dir / "model.safetensors")
    model.llm.model.rotary_emb = Qwen3RotaryEmbedding(config=llm_config)
    on_meta = [
        name
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if tensor.is_meta
    ]
    if on_meta:
        raise LoadError(f"tensors left on meta: {on_meta[:5]}")
    model.feature_extractor = WhisperFeatureExtractor(**FEATURE_EXTRACTOR)
    model.tokenizer = Qwen2Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return TorchEngine(model)


class TorchEngine:
    """The loaded model behind `complete`: tokenizer, audio path of the model, greedy decoding."""

    def __init__(self, model) -> None:
        self.model = model
        self.tokenizer = model.tokenizer

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def encode_audio(self, samples: np.ndarray):
        import torch

        with torch.inference_mode():
            mel = self.model.prepare_audio(torch.from_numpy(samples))
            mel_sha256 = hashlib.sha256(mel[0][0].float().numpy().tobytes()).hexdigest()
            embeddings, _, positions = self.model._process_audio(mel)
        return embeddings[0], positions[0], mel_sha256

    def generate(self, before: list[int], audio, after: list[int], max_new_tokens: int):
        import torch

        settings = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": IM_END_ID,
            "pad_token_id": PAD_ID,
        }
        torch.manual_seed(0)
        llm = self.model.llm
        with torch.inference_mode():
            if audio is None:
                ids = torch.tensor([before])
                out = llm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **settings)
                return out[0, ids.shape[1] :].tolist()
            embed = llm.get_input_embeddings()
            parts = [embed(torch.tensor(before)), audio, embed(torch.tensor(after))]
            inputs = torch.cat([part.to(embed.weight.dtype) for part in parts])[None]
            mask = torch.ones(inputs.shape[:2], dtype=torch.long)
            # with inputs_embeds and no input_ids, generate returns the new tokens only
            return llm.generate(inputs_embeds=inputs, attention_mask=mask, **settings)[0].tolist()


def version_line() -> str:
    versions = []
    for package in ("torch", "transformers", "safetensors"):
        try:
            versions.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{package} missing")
    return f"version: borealis-{REVISION} " + " ".join(versions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.gate0_borealis")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.version:
        print(version_line())
        return 0
    if args.model_dir is None or args.port is None:
        parser.error("--model-dir and --port are required")
    return serve(args.host, args.port, lambda: load_engine(args.model_dir, args.threads))


if __name__ == "__main__":
    sys.exit(main())
