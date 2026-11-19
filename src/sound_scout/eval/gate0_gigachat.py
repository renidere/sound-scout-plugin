"""Local server of GigaChat3.1-Audio-10B-A1.8B for the Gate 0 spike (#85,
docs/decisions/0011-gate0-gigachat-audio.md).

`python -m sound_scout.eval.gate0_gigachat --model-dir D --host 127.0.0.1 --port P --threads N`
answers the same four `llama-server` routes as `gate0_borealis` (`GET /health`,
`POST /apply-template`, `POST /tokenize`, `POST /v1/chat/completions`); the HTTP part, the
request parsing and the order bind -> network guard -> load are that module's. The request plan,
the rule and the verdict stay in `gate0`.

The model's own `from_pretrained` is not used: it needs `trust_remote_code`, loads the decoder
first and the encoder state dict in fp32 after it (peak about 26 GB). Here the encoder and the
adapter are built by the model's own classes, loaded by its own `load_encoder_and_adapter` and
cast to bf16 first; the decoder is the native `transformers` class `DeepseekV3ForCausalLM`,
built on `meta` and filled one tensor at a time. `modeling_gigachat_audio.py` runs only when its
SHA-256 is the pinned one, from the bytes that were checked (`load_modeling`).

`torch` and `transformers` are imported only when the model loads, so the request contract is
testable without them. A test with a fake engine proves that contract, never what the model
hears (CLAUDE.md).
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from sound_scout import guard
from sound_scout.eval import gate0_borealis as borealis
from sound_scout.eval.gate0_borealis import BadRequest, LoadError

REPO_ID = "ai-sage/GigaChat3.1-Audio-10B-A1.8B"
REVISION = "bf73d03a43bdf5118f5a4dbdc24ba6f56ac31cfb"
MODELING_FILE = "modeling_gigachat_audio.py"
MODELING_SHA256 = "ec96f0f9052b88616e427c9b75ed1cff6df98ed43176cdc11655ed6633b72e61"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
DECODER_DIR = "decoder"
DECODER_INDEX = "model.safetensors.index.json"
CHECKPOINT_DTYPES = frozenset({"BF16", "F32"})
# One file name for every clip: the template writes it into the prompt, the clip id must not
# reach the model.
AUDIO_NAME = "audio.wav"
SAMPLE_RATE = borealis.SAMPLE_RATE
HOST = borealis.HOST


# --- request -> model input ----------------------------------------------------------------


def messages_of(parts: list[str | bytes]) -> list[dict]:
    """The one user message in the model's chat form; audio parts become `AUDIO_NAME`."""
    content = [
        {"type": "audio", "path": AUDIO_NAME}
        if isinstance(part, bytes)
        else {"type": "text", "text": part}
        for part in parts
    ]
    return [{"role": "user", "content": content}]


def text_of(engine, body: object) -> str:
    """The rendered prompt of a text-only message (`/apply-template`)."""
    parts = borealis.user_parts(body)
    if any(isinstance(part, bytes) for part in parts):
        raise BadRequest("/apply-template takes text parts only")
    return engine.render(messages_of(parts))


def complete(engine, body: object) -> dict:
    """One chat completion in the `llama-server` response form."""
    if not isinstance(body, dict):
        raise BadRequest("the body must be a JSON object")
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise BadRequest("max_tokens must be a positive integer")
    parts = borealis.user_parts(body)
    pcm = next((part for part in parts if isinstance(part, bytes)), None)
    started = time.monotonic()
    samples = None if pcm is None else np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    prepared = engine.prepare(engine.render(messages_of(parts)), samples)
    prepare_ms = (time.monotonic() - started) * 1000
    generated = engine.generate(prepared, max_tokens)
    generate_ms = (time.monotonic() - started) * 1000 - prepare_ms
    finish = "length" if len(generated) >= max_tokens else "stop"
    prompt_tokens = len(prepared["input_ids"])
    response = {
        "object": "chat.completion",
        "model": f"gigachat-{REVISION}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": engine.decode(generated)},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(generated),
            "total_tokens": prompt_tokens + len(generated),
        },
        # prepare: template, tokens, log-mel; generate: encoder, prefill and decoding
        "timings": {
            "prepare_ms": round(prepare_ms, 3),
            "generate_ms": round(generate_ms, 3),
            "predicted_n": len(generated),
        },
    }
    if pcm is not None:
        response["gigachat"] = {
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "mel_sha256": prepared["mel_sha256"],
            "audio_positions": prepared["audio_positions"],
            "audio_block_tokens": prepared["audio_block_tokens"],
        }
    if "response_format" in body:
        response["response_format"] = "not_enforced"
    if "chat_template_kwargs" in body:
        response["chat_template_kwargs"] = "ignored"
    return response


def require_audio_slots(tokens: list[int], audio_token_id: int, has_audio: bool) -> None:
    """One audio-token slot for the waveform, none without it: the audio block expands that
    single slot, so a slot that came from the request text would take the waveform's place."""
    slots = list(tokens).count(audio_token_id)
    if slots != int(has_audio):
        raise BadRequest(f"the prompt holds {slots} audio-token slots, expected {int(has_audio)}")


class GigaServer(borealis.Server):
    complete = staticmethod(complete)
    text_of = staticmethod(text_of)


# --- the model -----------------------------------------------------------------------------


def load_modeling(model_dir: Path) -> types.ModuleType:
    """The model's code, run from the very bytes whose SHA-256 is `MODELING_SHA256`."""
    path = model_dir / MODELING_FILE
    try:
        source = path.read_bytes()
    except OSError as error:
        raise LoadError(f"{MODELING_FILE}: {error.strerror or error}") from None
    found = hashlib.sha256(source).hexdigest()
    if found != MODELING_SHA256:
        raise LoadError(f"{MODELING_FILE} is {found}, the pinned code is {MODELING_SHA256}")
    module = types.ModuleType("modeling_gigachat_audio")
    module.__file__ = str(path)
    # transformers reads the class source through sys.modules[cls.__module__].__file__
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def check_index(
    weight_map: dict[str, str],
    headers: dict[str, dict[str, dict]],
    expected: dict[str, tuple[int, ...]],
) -> None:
    """The index names exactly the skeleton's tensors; each shard header agrees in shape and
    holds BF16 or F32."""
    foreign = sorted(set(weight_map) - set(expected))
    missing = sorted(set(expected) - set(weight_map))
    if foreign or missing:
        raise LoadError(f"decoder keys differ: foreign {foreign[:5]}, missing {missing[:5]}")
    absent = sorted(name for name, shard in weight_map.items() if name not in headers[shard])
    if absent:
        raise LoadError(f"tensors missing from their shard: {absent[:5]}")
    entries = {name: headers[shard][name] for name, shard in weight_map.items()}
    dtypes = sorted({entry["dtype"] for entry in entries.values()} - CHECKPOINT_DTYPES)
    if dtypes:
        raise LoadError(f"decoder dtypes {dtypes} are not in {sorted(CHECKPOINT_DTYPES)}")
    shapes = sorted(
        name for name, entry in entries.items() if tuple(entry["shape"]) != expected[name]
    )
    if shapes:
        raise LoadError(f"decoder shapes differ from the skeleton: {shapes[:5]}")


def load_decoder(decoder_dir: Path):
    """`DeepseekV3ForCausalLM` on `meta`, filled tensor by tensor in bf16 anonymous memory."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, DeepseekV3ForCausalLM
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

    settings = AutoConfig.from_pretrained(decoder_dir, local_files_only=True)
    settings._attn_implementation = "sdpa"
    with torch.device("meta"):
        decoder = DeepseekV3ForCausalLM(settings)
    expected = {name: tuple(tensor.shape) for name, tensor in decoder.state_dict().items()}
    index = json.loads((decoder_dir / DECODER_INDEX).read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    headers = {shard: borealis.read_header(decoder_dir / shard) for shard in shards}
    check_index(weight_map, headers, expected)
    for name in sorted(weight_map, key=lambda key: (weight_map[key], key)):
        # opened per tensor so the mapped pages of the shard do not pile up in the RSS
        with safe_open(str(decoder_dir / weight_map[name]), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)
        tensor = tensor.clone() if tensor.dtype == torch.bfloat16 else tensor.to(torch.bfloat16)
        module_name, _, attribute = name.rpartition(".")
        module = decoder.get_submodule(module_name)
        if attribute in module._parameters:
            module._parameters[attribute] = torch.nn.Parameter(tensor, requires_grad=False)
        else:
            module._buffers[attribute] = tensor
    decoder.model.rotary_emb = DeepseekV3RotaryEmbedding(config=settings)
    on_meta = [
        name
        for name, tensor in [*decoder.named_parameters(), *decoder.named_buffers()]
        if tensor.is_meta
    ]
    if on_meta:
        raise LoadError(f"tensors left on meta: {on_meta[:5]}")
    return decoder.eval()


def load_processor(model_dir: Path, modeling: types.ModuleType):
    """The model's processor as its `from_pretrained` builds it, without `trust_remote_code`."""
    from transformers import AutoTokenizer

    settings = modeling.GigaChatAudioConfig.from_checkpoint_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / DECODER_DIR, local_files_only=True, **settings.tokenizer_kwargs
    )
    template = (model_dir / CHAT_TEMPLATE_FILE).read_text(encoding="utf-8")
    return settings, modeling.GigaChatAudioProcessor(tokenizer, settings, chat_template=template)


def load_engine(model_dir: Path, threads: int) -> "TorchEngine":
    os.environ.update(borealis.OFFLINE_ENV)
    import torch

    torch.set_num_threads(threads)
    modeling = load_modeling(model_dir)
    settings, processor = load_processor(model_dir, modeling)
    model = modeling.GigaChatAudioForCausalLM(settings)
    modeling.load_encoder_and_adapter(model.encoder, model.modality_adapter, model_dir)
    model.encoder.to(dtype=torch.bfloat16).eval()
    model.modality_adapter.to(dtype=torch.bfloat16).eval()
    gc.collect()
    model.language_model = load_decoder(model_dir / DECODER_DIR)
    model.eval()
    return TorchEngine(modeling, processor, model)


def offline(load):
    """`load`, then a refusal if anything in it tried to reach the network and the guard denied
    it. A denied `socket.bind` sends nothing and does not count: urllib3 (through
    `transformers` -> `huggingface_hub` -> `requests`) binds `::1` on import to probe IPv6."""

    def checked():
        engine = load()
        denied = [event for event in guard.denied_events() if event != "socket.bind"]
        if denied:
            raise LoadError(f"the load tried the network: {list(denied)[:5]}")
        return engine

    return checked


class TorchEngine:
    """The loaded model behind `complete`: the model's template, log-mel, audio block and
    greedy `generate`."""

    def __init__(self, modeling, processor, model) -> None:
        self.modeling = modeling
        self.processor = processor
        self.model = model

    def render(self, messages: list[dict]) -> str:
        return self.processor.render_prompt(messages, add_generation_prompt=True)[0]

    def tokenize(self, text: str) -> list[int]:
        return self.processor._encode_text(text)

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)

    def prepare(self, text: str, samples: np.ndarray | None) -> dict:
        """`prepare_for_inference` of the model with the waveform of the request."""
        tokens = self.tokenize(text)
        require_audio_slots(tokens, self.processor.audio_token_id, samples is not None)
        import torch

        ids = torch.tensor(tokens, dtype=torch.long)
        prepared = {"spectrograms": None, "spectrogram_lengths": None}
        if samples is not None:
            processor = self.processor
            mel = processor.feature_extractor.process_waveform(
                torch.from_numpy(samples), SAMPLE_RATE
            )
            lengths = torch.tensor([mel.shape[0]], dtype=torch.long)
            positions = processor._predict_audio_frames(lengths)
            boundary = processor.config.boundary_timings
            text_length = len(ids)
            ids = self.modeling.expand_audio_tokens(
                ids,
                processor.audio_token_id,
                positions,
                processor.timing_lut,
                boundary.framerate_ms,
                inter_spec=boundary.inter_timings if processor.use_inter_timings else None,
                durations=[len(samples) / SAMPLE_RATE] if boundary.add_durations else None,
            )
            prepared.update(
                spectrograms=mel.unsqueeze(0),
                spectrogram_lengths=lengths,
                mel_sha256=hashlib.sha256(mel.numpy().tobytes()).hexdigest(),
                audio_positions=int(positions[0]),
                audio_block_tokens=len(ids) - text_length + 1,
            )
        prepared["input_ids"] = ids.tolist()
        return prepared

    def generate(self, prepared: dict, max_new_tokens: int) -> list[int]:
        import torch

        ids = torch.tensor([prepared["input_ids"]], dtype=torch.long)
        torch.manual_seed(0)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                spectrograms=prepared["spectrograms"],
                spectrogram_lengths=prepared["spectrogram_lengths"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return out[0, ids.shape[1] :].tolist()


def version_line() -> str:
    versions = []
    for package in ("torch", "torchaudio", "transformers", "safetensors"):
        try:
            versions.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{package} missing")
    return f"version: gigachat-{REVISION} " + " ".join(versions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.gate0_gigachat")
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
    return borealis.serve(
        args.host,
        args.port,
        offline(lambda: load_engine(args.model_dir, args.threads)),
        server_class=GigaServer,
        name="gate0_gigachat",
    )


if __name__ == "__main__":
    sys.exit(main())
