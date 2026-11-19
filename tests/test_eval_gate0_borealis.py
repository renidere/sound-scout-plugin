"""The Borealis-5b-it server of the Gate 0 spike (#68): request contract and streaming loader.

A fake engine stands in for the model here: these tests prove how a `gate0` request becomes the
model's input and how the answer comes back, never what the model hears. That is only the real
run recorded in docs/decisions/0010-gate0-borealis.md.
"""

import base64
import hashlib
import io
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

from sound_scout.eval import gate0
from sound_scout.eval import gate0_borealis as borealis

SYSTEM = "You are a helpful voice assistant. Listen to the audio and respond appropriately."


class FakeEngine:
    """Characters as token ids; a fixed audio slot of 375 positions; scripted output ids."""

    def __init__(self, output: list[int] | None = None) -> None:
        self.output = output if output is not None else [*map(ord, "нет"), borealis.IM_END_ID]
        self.calls: list[dict] = []
        self.samples = None

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def detokenize(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i < 0x110000)

    def encode_audio(self, samples):
        self.samples = samples
        return "AUDIO", 375, "mel-sha"

    def generate(self, before, audio, after, max_new_tokens):
        self.calls.append({"before": before, "audio": audio, "after": after, "max": max_new_tokens})
        return self.output[:max_new_tokens]


def _wav(samples: np.ndarray, channels: int = 1, rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(gate0.pcm16(samples))
    return buffer.getvalue()


def _audio_part(data: bytes) -> dict:
    encoded = base64.b64encode(data).decode()
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


def _body(*parts: dict, max_tokens: int = 8, **extra) -> dict:
    message = {"role": "user", "content": list(parts)}
    return {"messages": [message], "max_tokens": max_tokens, **extra}


def _tone() -> np.ndarray:
    return 0.1 * np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000)


# --- request -> model input ----------------------------------------------------------------


def test_render_is_the_qwen3_chatml_with_the_stock_system_prompt_and_no_think_block() -> None:
    assert borealis.SYSTEM_PROMPT == SYSTEM
    assert borealis.render("вопрос") == (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        "<|im_start|>user\nвопрос<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_audio_part_goes_between_the_markers_before_the_text_and_counts_in_prompt_tokens() -> None:
    engine = FakeEngine()
    wav = _wav(_tone())
    pcm = gate0.wav_pcm(wav)
    answer = borealis.complete(engine, _body(_audio_part(wav), {"type": "text", "text": "Что?"}))
    (call,) = engine.calls
    head, tail = borealis.render(borealis.AUDIO_SLOT + "\nЧто?").split(borealis.AUDIO_SLOT)
    assert call["before"] == [*map(ord, head), borealis.AUDIO_START_ID]
    assert call["after"] == [borealis.AUDIO_END_ID, *map(ord, tail)]
    assert (borealis.AUDIO_START_ID, borealis.AUDIO_END_ID) == (151669, 151670)
    assert call["audio"] == "AUDIO" and call["max"] == 8
    assert answer["usage"]["prompt_tokens"] == len(call["before"]) + 375 + len(call["after"])
    assert answer["borealis"] == {
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "mel_sha256": "mel-sha",
        "audio_positions": 375,
    }
    expected = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    assert engine.samples.dtype == np.float32 and np.array_equal(engine.samples, expected)


def test_part_order_is_kept_text_before_audio() -> None:
    engine = FakeEngine()
    borealis.complete(
        engine, _body({"type": "text", "text": "Сначала"}, _audio_part(_wav(_tone())))
    )
    (call,) = engine.calls
    assert "".join(map(chr, call["before"][:-1])).endswith("user\nСначала\n")
    assert "".join(map(chr, call["after"][1:])).startswith("<|im_end|>")


def test_text_only_request_has_no_audio_and_counts_the_whole_template() -> None:
    engine = FakeEngine()
    answer = borealis.complete(engine, _body({"type": "text", "text": "Только текст"}))
    (call,) = engine.calls
    assert call["audio"] is None and call["after"] == []
    assert call["before"] == [*map(ord, borealis.render("Только текст"))]
    assert answer["usage"]["prompt_tokens"] == len(call["before"])
    assert "borealis" not in answer


def test_finish_reason_is_stop_at_eos_and_length_at_max_tokens() -> None:
    stopped = borealis.complete(FakeEngine(), _body({"type": "text", "text": "x"}))
    assert stopped["choices"][0]["finish_reason"] == "stop"
    assert stopped["choices"][0]["message"]["content"] == "нет"
    assert stopped["usage"]["completion_tokens"] == 4
    long = FakeEngine([ord("a")] * 20)
    cut = borealis.complete(long, _body({"type": "text", "text": "x"}, max_tokens=5))
    assert cut["choices"][0]["finish_reason"] == "length"
    assert cut["usage"]["completion_tokens"] == 5
    short = borealis.complete(FakeEngine([ord("a")]), _body({"type": "text", "text": "x"}))
    assert short["choices"][0]["finish_reason"] == "stop"


class SlowEngine(FakeEngine):
    def encode_audio(self, samples):
        time.sleep(0.05)
        return super().encode_audio(samples)

    def generate(self, before, audio, after, max_new_tokens):
        time.sleep(0.3)
        return super().generate(before, audio, after, max_new_tokens)


def test_timings_split_input_encoding_from_generation_with_the_prefill() -> None:
    answer = borealis.complete(SlowEngine(), _body(_audio_part(_wav(_tone()))))
    timings = answer["timings"]
    assert set(timings) == {"encode_ms", "generate_ms", "predicted_n"}
    assert 50 <= timings["encode_ms"] < 300
    assert timings["generate_ms"] >= 300
    assert timings["predicted_n"] == 4
    text_only = borealis.complete(SlowEngine(), _body({"type": "text", "text": "x"}))
    assert text_only["timings"]["encode_ms"] < 50
    assert text_only["timings"]["generate_ms"] >= 300


@pytest.mark.parametrize(
    ("text", "content", "reasoning"),
    [
        ("<think>думаю</think>\n\nда", "да", "думаю"),
        ("думаю</think>нет", "нет", "думаю"),
        ("<think>обрыв", "", "обрыв"),
        (" да ", " да ", None),
    ],
)
def test_think_block_goes_to_reasoning_content(text, content, reasoning) -> None:
    body = _body({"type": "text", "text": "x"}, max_tokens=64)
    answer = borealis.complete(FakeEngine([*map(ord, text)]), body)
    message = answer["choices"][0]["message"]
    assert message["content"] == content
    assert message.get("reasoning_content") == reasoning


def test_response_format_and_chat_template_kwargs_are_reported_as_not_applied() -> None:
    body = _body(
        {"type": "text", "text": "x"},
        response_format={"type": "json_object"},
        chat_template_kwargs={"enable_thinking": False},
    )
    answer = borealis.complete(FakeEngine(), body)
    assert answer["response_format"] == "not_enforced"
    assert answer["chat_template_kwargs"] == "ignored"


@pytest.mark.parametrize(
    "body",
    [
        _body(_audio_part(_wav(np.zeros(0))), {"type": "text", "text": "x"}),
        _body(
            _audio_part(_wav(_tone())), _audio_part(_wav(_tone())), {"type": "text", "text": "x"}
        ),
        _body(_audio_part(_wav(_tone(), channels=2)), {"type": "text", "text": "x"}),
        _body(_audio_part(_wav(_tone(), rate=48_000)), {"type": "text", "text": "x"}),
        _body({"type": "input_audio", "input_audio": {"data": "!!", "format": "wav"}}),
        _body({"type": "image_url", "image_url": {}}),
        _body({"type": "text", "text": "x"}, max_tokens=0),
        _body({"type": "text", "text": borealis.AUDIO_SLOT}),
        {"messages": [], "max_tokens": 8},
        {"messages": [{"role": "system", "content": "x"}], "max_tokens": 8},
    ],
    ids=[
        "empty WAV",
        "two audio parts",
        "stereo",
        "48 kHz",
        "not base64",
        "foreign part",
        "max_tokens 0",
        "reserved text",
        "no message",
        "not user",
    ],
)
def test_bad_requests_are_refused_before_the_model(body) -> None:
    engine = FakeEngine()
    with pytest.raises(borealis.BadRequest):
        borealis.complete(engine, body)
    assert engine.calls == [] and engine.samples is None


# --- HTTP routes ---------------------------------------------------------------------------


@pytest.fixture
def server():
    """The real handler on 127.0.0.1 with a fake engine, without the process network guard."""
    httpd = borealis.Server((borealis.HOST, 0), borealis.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _call(port: int, path: str, body: object = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method="GET" if body is None else "POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_routes_answer_like_llama_server_and_the_gate0_client_counts_text_tokens(server) -> None:
    port = server.server_address[1]
    assert _call(port, "/health")[0] == 503
    server.engine = FakeEngine()
    status, health = _call(port, "/health")
    assert status == 200 and health["status"] == "ok"
    message = {"role": "user", "content": [{"type": "text", "text": "Вопрос"}]}
    status, template = _call(port, "/apply-template", {"messages": [message]})
    assert status == 200 and template["prompt"] == borealis.render("Вопрос")
    status, tokens = _call(port, "/tokenize", {"content": "абв", "add_special": True})
    assert status == 200 and tokens["tokens"] == [ord("а"), ord("б"), ord("в")]
    assert gate0.Client(port).text_tokens("Вопрос") == len(borealis.render("Вопрос"))
    status, answer = _call(port, "/v1/chat/completions", _body(_audio_part(_wav(_tone()))))
    assert status == 200 and answer["choices"][0]["message"]["content"] == "нет"


def test_routes_refuse_audio_in_the_template_bad_bodies_and_unknown_paths(server) -> None:
    port = server.server_address[1]
    server.engine = FakeEngine()
    message = {"role": "user", "content": [_audio_part(_wav(_tone()))]}
    assert _call(port, "/apply-template", {"messages": [message]})[0] == 400
    assert _call(port, "/tokenize", {"content": 1})[0] == 400
    assert _call(port, "/v1/chat/completions", _body(_audio_part(_wav(np.zeros(0)))))[0] == 400
    assert _call(port, "/v1/models", {})[0] == 404
    assert _call(port, "/metrics")[0] == 404
    assert server.engine.calls == []


def test_server_refuses_a_host_other_than_loopback(capsys) -> None:
    assert borealis.main(["--model-dir", "/nowhere", "--host", "0.0.0.0", "--port", "1"]) == 2
    assert "127.0.0.1" in capsys.readouterr().err


def test_version_line_names_the_revision_and_the_runtime_packages(capsys) -> None:
    assert borealis.main(["--version"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"version: borealis-{borealis.REVISION} torch ")
    assert " transformers " in line and " safetensors " in line


GUARD_CHILD = r"""
import socket, sys
from sound_scout import guard
from sound_scout.eval import gate0_borealis as borealis

class Engine:
    pass

def load():
    try:
        socket.create_connection(("127.0.0.1", int(sys.argv[2])), timeout=2)
        print("connected", flush=True)
    except guard.NetworkDenied:
        print("denied", flush=True)
    except OSError as error:
        print(f"not denied: {type(error).__name__}", flush=True)
    return Engine()

sys.exit(borealis.serve("127.0.0.1", int(sys.argv[1]), load))
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_network_guard_goes_on_after_bind_and_before_the_model_loads(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "data").mkdir()
    toml = tmp_path / "c.toml"
    toml.write_text(f'allowed_roots = ["{tmp_path / "in"}"]\ndata_root = "{tmp_path / "data"}"\n')
    port, other = _free_port(), _free_port()
    child = subprocess.Popen(
        [sys.executable, "-c", GUARD_CHILD, str(port), str(other)],
        stdout=subprocess.PIPE,
        env={**os.environ, "SOUND_SCOUT_CONFIG": str(toml)},
    )
    try:
        deadline = time.monotonic() + 30
        status, health = None, {}
        while time.monotonic() < deadline and status != 200:
            try:
                status, health = _call(port, "/health")
            except OSError:
                time.sleep(0.1)
        assert status == 200 and health["network_guard_active"] is True
    finally:
        child.kill()
        out, _ = child.communicate()
    assert out.decode().split() == ["denied"]


def test_server_refuses_to_start_when_the_config_allows_network(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "data").mkdir()
    toml = tmp_path / "c.toml"
    toml.write_text(
        f'allowed_roots = ["{tmp_path / "in"}"]\ndata_root = "{tmp_path / "data"}"\n'
        "[network]\nallow = true\n"
    )
    port, other = _free_port(), _free_port()
    child = subprocess.run(
        [sys.executable, "-c", GUARD_CHILD, str(port), str(other)],
        capture_output=True,
        env={**os.environ, "SOUND_SCOUT_CONFIG": str(toml)},
        timeout=30,
    )
    assert child.returncode == 2
    assert child.stdout.decode() == ""
    assert "network guard is off" in child.stderr.decode()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


# --- the streaming loader ------------------------------------------------------------------


def _safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = 2 * int(np.prod(shape))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def test_header_check_passes_the_skeleton_and_refuses_foreign_dtype_and_shape(tmp_path) -> None:
    expected = {"encoder.w": (2, 3), "llm.w": (4,)}
    path = tmp_path / "model.safetensors"
    _safetensors(path, {"encoder.w": ("F16", [2, 3]), "llm.w": ("F16", [4])})
    borealis.check_header(borealis.read_header(path), expected)
    for tensors, message in (
        ({"encoder.w": ("F16", [2, 3]), "llm.w": ("F16", [4]), "x": ("F16", [1])}, "foreign"),
        ({"encoder.w": ("F16", [2, 3])}, "missing"),
        ({"encoder.w": ("F32", [2, 3]), "llm.w": ("F16", [4])}, "dtypes"),
        ({"encoder.w": ("F16", [3, 2]), "llm.w": ("F16", [4])}, "shapes"),
    ):
        _safetensors(path, tensors)
        with pytest.raises(borealis.LoadError, match=message):
            borealis.check_header(borealis.read_header(path), expected)


def _tiny(torch):
    """A Borealis-shaped model with tiny Whisper and Qwen3 parts."""
    from transformers import Qwen3Config, Qwen3ForCausalLM, WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    model = torch.nn.Module()
    model.encoder = WhisperEncoder(
        WhisperConfig(
            d_model=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            num_mel_bins=8,
            max_source_positions=10,
        )
    )
    model.llm = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=40,
            tie_word_embeddings=True,
        )
    )
    model.adapter = torch.nn.Linear(64, 16, bias=False)
    return model


@pytest.mark.slow
def test_streaming_loader_equals_a_naive_bf16_load_and_refuses_broken_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file

    torch.manual_seed(0)
    source = _tiny(torch)
    tensors = {name: t.detach().half().clone() for name, t in source.state_dict().items()}
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    with torch.device("meta"):
        skeleton = _tiny(torch)
    borealis.load_weights(skeleton, path)
    loaded = skeleton.state_dict()
    assert set(loaded) == set(tensors)
    for name, tensor in loaded.items():
        assert tensor.dtype == torch.bfloat16 and not tensor.is_meta
        assert torch.equal(tensor, tensors[name].to(torch.bfloat16)), name
    assert skeleton.llm.lm_head.weight is skeleton.llm.model.embed_tokens.weight

    broken = {
        "untied lm_head": {**tensors, borealis.LM_HEAD_KEY: tensors[borealis.LM_HEAD_KEY] + 1},
        "foreign key": {**tensors, "llm.extra": torch.zeros(2, dtype=torch.float16)},
        "wrong shape": {**tensors, "adapter.weight": torch.zeros(16, 63, dtype=torch.float16)},
    }
    for case, checkpoint in broken.items():
        save_file(checkpoint, str(path))
        with torch.device("meta"):
            skeleton = _tiny(torch)
        with pytest.raises(borealis.LoadError):
            borealis.load_weights(skeleton, path)
        assert case


@pytest.mark.slow
def test_torch_engine_generates_only_new_tokens_and_embeds_audio_in_place():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = _tiny(torch).to(torch.bfloat16).eval()
    model.tokenizer = None
    engine = borealis.TorchEngine(model)
    embed = model.llm.get_input_embeddings()
    with torch.inference_mode():
        audio = embed(torch.tensor([5, 6, 7]))
    as_text = engine.generate([1, 2, 5, 6, 7, 3], None, [], 4)
    as_audio = engine.generate([1, 2], audio, [3], 4)
    assert len(as_text) <= 4 and as_audio == as_text
    with torch.inference_mode():
        mel = torch.zeros(1, 8, 20, dtype=torch.bfloat16)
        encoded = model.encoder(input_features=mel).last_hidden_state
    assert encoded.shape == (1, 10, 16) and encoded.dtype == torch.bfloat16
