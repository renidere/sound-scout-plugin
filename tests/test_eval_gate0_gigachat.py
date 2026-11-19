"""The GigaChat3.1-Audio server of the Gate 0 spike (#85): request contract, pinned model code,
decoder index check and the offline start.

A fake engine stands in for the model here: these tests prove how a `gate0` request becomes the
model's input and how the answer comes back, never what the model hears. That is only the real
run recorded in docs/decisions/0011-gate0-gigachat-audio.md. No test needs the weights.
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
from sound_scout.eval import gate0_gigachat as gigachat


class FakeEngine:
    """Characters as token ids; an audio block of 79 ids; scripted output ids."""

    def __init__(self, output: list[int] | None = None) -> None:
        self.output = output if output is not None else [*map(ord, "нет")]
        self.rendered: list[list[dict]] = []
        self.prepared: list[dict] = []
        self.samples = None

    def render(self, messages: list[dict]) -> str:
        self.rendered.append(messages)
        return json.dumps(messages, ensure_ascii=False)

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def prepare(self, text: str, samples) -> dict:
        self.samples = samples
        ids = self.tokenize(text)
        prepared = {"input_ids": ids}
        if samples is not None:
            prepared.update(
                input_ids=ids + [0] * 78,
                mel_sha256="mel-sha",
                audio_positions=63,
                audio_block_tokens=79,
            )
        self.prepared.append(prepared)
        return prepared

    def generate(self, prepared: dict, max_new_tokens: int) -> list[int]:
        return self.output[:max_new_tokens]


def _wav(samples: np.ndarray, channels: int = 1, rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(gate0.pcm16(np.repeat(samples, channels)))
    return buffer.getvalue()


def _audio_part(data: bytes) -> dict:
    encoded = base64.b64encode(data).decode()
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _body(*parts: dict, max_tokens: int = 8, **extra) -> dict:
    message = {"role": "user", "content": list(parts)}
    return {"messages": [message], "max_tokens": max_tokens, **extra}


def _tone(hz: float = 440.0) -> np.ndarray:
    return 0.1 * np.sin(2 * np.pi * hz * np.arange(16_000) / 16_000)


# --- request -> model input ----------------------------------------------------------------


def test_parts_keep_their_order_and_every_clip_is_named_audio_wav() -> None:
    engine = FakeEngine()
    gigachat.complete(engine, _body(_audio_part(_wav(_tone())), _text("Что?")))
    gigachat.complete(engine, _body(_text("Сначала"), _audio_part(_wav(_tone(220.0)))))
    first, second = engine.rendered
    assert first == [
        {
            "role": "user",
            "content": [{"type": "audio", "path": "audio.wav"}, {"type": "text", "text": "Что?"}],
        }
    ]
    assert second[0]["content"] == [
        {"type": "text", "text": "Сначала"},
        {"type": "audio", "path": "audio.wav"},
    ]
    assert gigachat.AUDIO_NAME == "audio.wav"


def test_audio_answer_echoes_the_pcm_sent_and_counts_the_expanded_prompt() -> None:
    engine = FakeEngine()
    wav = _wav(_tone())
    pcm = gate0.wav_pcm(wav)
    answer = gigachat.complete(engine, _body(_audio_part(wav), _text("Что?")))
    (prepared,) = engine.prepared
    assert answer["usage"]["prompt_tokens"] == len(prepared["input_ids"])
    assert answer["usage"]["completion_tokens"] == 3
    assert answer["model"] == f"gigachat-{gigachat.REVISION}"
    assert answer["gigachat"] == {
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "mel_sha256": "mel-sha",
        "audio_positions": 63,
        "audio_block_tokens": 79,
    }
    expected = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    assert engine.samples.dtype == np.float32 and np.array_equal(engine.samples, expected)
    assert set(answer["timings"]) == {"prepare_ms", "generate_ms", "predicted_n"}


def test_text_only_request_has_no_audio_echo() -> None:
    engine = FakeEngine()
    answer = gigachat.complete(engine, _body(_text("Только текст")))
    assert engine.samples is None and "gigachat" not in answer
    assert answer["choices"][0]["message"]["content"] == "нет"


def test_finish_reason_is_length_at_max_tokens_and_stop_before_it() -> None:
    cut = gigachat.complete(FakeEngine([ord("a")] * 20), _body(_text("x"), max_tokens=5))
    assert cut["choices"][0]["finish_reason"] == "length"
    assert cut["usage"]["completion_tokens"] == 5
    stopped = gigachat.complete(FakeEngine([ord("a")]), _body(_text("x")))
    assert stopped["choices"][0]["finish_reason"] == "stop"
    empty = gigachat.complete(FakeEngine([]), _body(_text("x")))
    assert empty["choices"][0]["finish_reason"] == "stop"
    assert empty["choices"][0]["message"]["content"] == ""


def test_response_format_and_chat_template_kwargs_are_reported_as_not_applied() -> None:
    body = _body(
        _text("x"),
        response_format={"type": "json_object"},
        chat_template_kwargs={"enable_thinking": False},
    )
    answer = gigachat.complete(FakeEngine(), body)
    assert answer["response_format"] == "not_enforced"
    assert answer["chat_template_kwargs"] == "ignored"


@pytest.mark.parametrize(
    "body",
    [
        _body(_audio_part(_wav(_tone())), _audio_part(_wav(_tone())), _text("x")),
        _body(_audio_part(_wav(_tone(), rate=44_100)), _text("x")),
        _body(_audio_part(_wav(_tone(), channels=2)), _text("x")),
        _body(_audio_part(_wav(np.zeros(0))), _text("x")),
        _body({"type": "input_audio", "input_audio": {"data": "!!", "format": "wav"}}),
        _body(_text("x"), max_tokens=0),
        {"messages": [{"role": "system", "content": "x"}], "max_tokens": 8},
        [],
    ],
    ids=["two audio parts", "44.1 kHz", "stereo", "empty WAV", "not base64", "max_tokens 0",
         "not user", "not an object"],
)  # fmt: skip
def test_bad_requests_are_refused_before_the_model(body) -> None:
    engine = FakeEngine()
    with pytest.raises(borealis.BadRequest):
        gigachat.complete(engine, body)
    assert engine.prepared == [] and engine.samples is None


class SlotProcessor:
    audio_token_id = 7

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def _encode_text(self, text: str) -> list[int]:
        return list(self.ids)


@pytest.mark.parametrize(
    ("ids", "samples"),
    [([1, 7, 7, 2], np.zeros(160, np.float32)), ([1, 2], np.zeros(160, np.float32)),
     ([1, 7, 2], None)],
    ids=["two slots for one waveform", "no slot for the waveform", "a slot without audio"],
)  # fmt: skip
def test_prompt_audio_slots_other_than_one_per_waveform_are_refused_before_expansion(
    ids, samples
) -> None:
    engine = gigachat.TorchEngine(modeling=None, processor=SlotProcessor(ids), model=None)
    with pytest.raises(borealis.BadRequest, match="audio-token slots"):
        engine.prepare("prompt", samples)


def test_one_slot_with_the_waveform_and_none_without_it_pass_the_slot_check() -> None:
    gigachat.require_audio_slots([1, 7, 2], 7, has_audio=True)
    gigachat.require_audio_slots([1, 2], 7, has_audio=False)


# --- pinned model code ---------------------------------------------------------------------


def test_modeling_code_with_another_hash_is_refused_before_it_runs(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    code = f"open({str(marker)!r}, 'w').write('ran')\n"
    (tmp_path / gigachat.MODELING_FILE).write_text(code, encoding="utf-8")
    with pytest.raises(borealis.LoadError, match="pinned code"):
        gigachat.load_modeling(tmp_path)
    assert not marker.exists()
    with pytest.raises(borealis.LoadError, match=gigachat.MODELING_FILE):
        gigachat.load_modeling(tmp_path / "missing")


def test_modeling_code_with_the_pinned_hash_runs_from_the_checked_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    source = b"value = 42\nwhere = __file__\n"
    path = tmp_path / gigachat.MODELING_FILE
    path.write_bytes(source)
    monkeypatch.setattr(gigachat, "MODELING_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.delitem(sys.modules, "modeling_gigachat_audio", raising=False)
    module = gigachat.load_modeling(tmp_path)
    assert module.value == 42 and module.where == str(path)
    assert sys.modules["modeling_gigachat_audio"] is module
    assert not (tmp_path / "__pycache__").exists()


# --- the decoder index ---------------------------------------------------------------------


def _header(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> dict:
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = (4 if dtype == "F32" else 2) * int(np.prod(shape))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return borealis.read_header(path)


def test_index_check_passes_the_skeleton_and_refuses_foreign_keys_shapes_and_dtypes(
    tmp_path: Path,
) -> None:
    expected = {"model.a": (2, 3), "model.b": (4,), "lm_head.weight": (5, 2)}
    weight_map = {"model.a": "s1", "model.b": "s1", "lm_head.weight": "s2"}
    good = {
        "s1": _header(tmp_path / "s1", {"model.a": ("BF16", [2, 3]), "model.b": ("F32", [4])}),
        "s2": _header(tmp_path / "s2", {"lm_head.weight": ("BF16", [5, 2])}),
    }
    gigachat.check_index(weight_map, good, expected)
    cases = [
        ({**weight_map, "model.extra": "s1"}, good, "foreign"),
        ({"model.a": "s1", "model.b": "s1"}, good, "missing"),
        (
            weight_map,
            {**good, "s1": _header(tmp_path / "s3", {"model.a": ("BF16", [2, 3])})},
            "missing from their shard",
        ),
        (
            weight_map,
            {**good, "s2": _header(tmp_path / "s4", {"lm_head.weight": ("F16", [5, 2])})},
            "dtypes",
        ),
        (
            weight_map,
            {**good, "s2": _header(tmp_path / "s5", {"lm_head.weight": ("BF16", [2, 5])})},
            "shapes",
        ),
    ]
    for index, headers, message in cases:
        with pytest.raises(borealis.LoadError, match=message):
            gigachat.check_index(index, headers, expected)


@pytest.mark.slow
def test_streamed_decoder_equals_the_checkpoint_in_bf16_with_nothing_left_on_meta(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("4."):
        pytest.skip("the gate0-gigachat extra pins transformers 4.57")
    from safetensors.torch import save_file
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    settings = DeepseekV3Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        kv_lora_rank=8,
        q_lora_rank=None,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=8,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    source = DeepseekV3ForCausalLM(settings).to(torch.bfloat16)
    tensors = {name: t.detach().clone() for name, t in source.state_dict().items()}
    names = sorted(tensors)
    shards = {"model-1.safetensors": names[::2], "model-2.safetensors": names[1::2]}
    for shard, keys in shards.items():
        save_file({key: tensors[key] for key in keys}, str(tmp_path / shard))
    weight_map = {key: shard for shard, keys in shards.items() for key in keys}
    (tmp_path / gigachat.DECODER_INDEX).write_text(json.dumps({"weight_map": weight_map}))
    settings.save_pretrained(tmp_path)
    decoder = gigachat.load_decoder(tmp_path)
    loaded = decoder.state_dict()
    assert set(loaded) == set(tensors)
    for name, tensor in loaded.items():
        assert tensor.dtype == torch.bfloat16 and not tensor.is_meta, name
        assert torch.equal(tensor, tensors[name]), name
    assert decoder.model.rotary_emb.inv_freq.dtype == torch.float32
    with torch.inference_mode():
        logits = decoder(torch.tensor([[1, 2, 3]])).logits
    assert logits.shape == (1, 3, 40) and bool(torch.isfinite(logits).all())

    del weight_map[names[0]]
    (tmp_path / gigachat.DECODER_INDEX).write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(borealis.LoadError, match="missing"):
        gigachat.load_decoder(tmp_path)


# --- HTTP and the offline start ------------------------------------------------------------


@pytest.fixture
def server():
    """The real handler on 127.0.0.1 with a fake engine, without the process network guard."""
    httpd = gigachat.GigaServer((gigachat.HOST, 0), borealis.Handler)
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


def test_routes_answer_like_llama_server_with_the_model_template(server) -> None:
    port = server.server_address[1]
    assert _call(port, "/health")[0] == 503
    server.engine = FakeEngine()
    message = {"role": "user", "content": [_text("Вопрос")]}
    status, template = _call(port, "/apply-template", {"messages": [message]})
    assert status == 200 and template["prompt"] == FakeEngine().render([message])
    assert _call(port, "/tokenize", {"content": "аб"}) == (200, {"tokens": [ord("а"), ord("б")]})
    assert gate0.Client(port).text_tokens("Вопрос") == len(template["prompt"])
    status, answer = _call(port, "/v1/chat/completions", _body(_audio_part(_wav(_tone()))))
    assert status == 200 and answer["gigachat"]["audio_block_tokens"] == 79
    audio_message = {"role": "user", "content": [_audio_part(_wav(_tone()))]}
    assert _call(port, "/apply-template", {"messages": [audio_message]})[0] == 400


def test_server_refuses_a_host_other_than_loopback(capsys) -> None:
    assert gigachat.main(["--model-dir", "/nowhere", "--host", "0.0.0.0", "--port", "1"]) == 2
    assert "gate0_gigachat: host must be 127.0.0.1" in capsys.readouterr().err


def test_version_line_names_the_revision_and_the_runtime_packages(capsys) -> None:
    assert gigachat.main(["--version"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"version: gigachat-{gigachat.REVISION} torch ")
    for package in (" torchaudio ", " transformers ", " safetensors "):
        assert package in line


START_CHILD = r"""
import socket, sys
from sound_scout import guard
from sound_scout.eval import gate0_gigachat as gigachat

class Engine:
    pass

def load():
    try:
        if sys.argv[3] == "connect":
            socket.create_connection(("127.0.0.1", int(sys.argv[2])), timeout=2)
        if sys.argv[3] == "bind":
            socket.socket(socket.AF_INET6).bind(("::1", 0))
    except guard.NetworkDenied:
        pass
    print("loaded", flush=True)
    return Engine()

sys.exit(gigachat.borealis.serve(
    "127.0.0.1", int(sys.argv[1]), gigachat.offline(load),
    server_class=gigachat.GigaServer, name="gate0_gigachat",
))
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config(tmp_path: Path, allow: bool = False) -> Path:
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    toml = tmp_path / "c.toml"
    network = "[network]\nallow = true\n" if allow else ""
    toml.write_text(
        f'allowed_roots = ["{tmp_path / "in"}"]\ndata_root = "{tmp_path / "data"}"\n{network}'
    )
    return toml


def _start(tmp_path: Path, mode: str, allow: bool = False) -> tuple[int, subprocess.Popen]:
    port = _free_port()
    child = subprocess.Popen(
        [sys.executable, "-c", START_CHILD, str(port), str(_free_port()), mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "SOUND_SCOUT_CONFIG": str(_config(tmp_path, allow))},
    )
    return port, child


def test_a_load_that_tried_the_network_refuses_to_start_with_code_3(tmp_path: Path) -> None:
    _, child = _start(tmp_path, "connect")
    out, err = child.communicate(timeout=30)
    assert child.returncode == 3
    assert out.decode().split() == ["loaded"]
    assert "the load tried the network" in err.decode()


@pytest.mark.parametrize("mode", ["quiet", "bind"])
def test_a_load_that_sent_nothing_serves_with_the_guard_on(tmp_path: Path, mode) -> None:
    port, child = _start(tmp_path, mode)
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
        child.communicate()


def test_the_server_refuses_to_start_when_the_config_allows_network(tmp_path: Path) -> None:
    _, child = _start(tmp_path, "quiet", allow=True)
    out, err = child.communicate(timeout=30)
    assert child.returncode == 2 and out.decode() == ""
    assert "gate0_gigachat: the network guard is off" in err.decode()
