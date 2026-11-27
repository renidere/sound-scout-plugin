"""Forced choice of the stressed word (#205): the rule pin, the selection, the plan, the parser,
the pins, the run against a fake server and the analysis.

The fake server proves the contract of `forced_choice.run` only (what is sent, traced, stopped
and refused), never what the model hears. All audio is synthesized here.
"""

import dataclasses
import hashlib
import inspect
import io
import json
import socket
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config
from sound_scout.eval import forced_choice as fc
from sound_scout.eval import gate0, gpu_ext

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / fc.DECISION_DOC

# --- the rule ------------------------------------------------------------------------------


def test_rule_block_of_0036_is_pinned_and_names_the_files_and_the_template() -> None:
    document = DOC.read_text(encoding="utf-8")
    block = fc.rule_block(document)
    assert fc.rule_sha256(document) == fc.RULE_SHA256
    assert f"SHA-256 блока: `{fc.RULE_SHA256}`" in document
    assert block.startswith("```text\nFORCED-CHOICE-RULE v1 (sound-scout #205")
    assert fc.TEMPLATE in fc.flat(block)
    for pin in (fc.SET_SHA256, fc.PLAN_SHA256):
        assert pin in block and pin in document
    assert fc.check_rule(DOC) == fc.RULE_SHA256


def test_pins_of_0036_are_those_of_dependencies_md() -> None:
    dependencies = (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    document = DOC.read_text(encoding="utf-8")
    pins = fc.Pins()
    for sha in (*pins.weights.values(), pins.archive_sha256):
        assert sha in dependencies and sha in document
    assert pins.weights == gate0.QWEN25OMNI_WEIGHTS
    assert pins.server_sha256 in document
    sources = fc.Sources()
    for sha in (
        sources.set_manifest_sha256,
        sources.set_plan_sha256,
        sources.control_manifest_sha256,
        sources.pilot_provenance_sha256,
        sources.pilot_manifest_sha256,
    ):
        assert sha in document


def test_check_rule_refuses_an_edited_block_or_template(tmp_path: Path, monkeypatch) -> None:
    edited = tmp_path / "0036.md"
    edited.write_text(DOC.read_text("utf-8").replace("max_tokens 8", "max_tokens 9"), "utf-8")
    with pytest.raises(fc.ForcedChoiceError, match="SHA-256 is"):
        fc.check_rule(edited)
    monkeypatch.setattr(fc, "TEMPLATE", fc.TEMPLATE.replace("одной цифрой", "числом"))
    with pytest.raises(fc.ForcedChoiceError, match="template"):
        fc.check_rule(DOC)
    edited.write_text("no block", "utf-8")
    with pytest.raises(fc.ForcedChoiceError, match="one"):
        fc.check_rule(edited)


# --- a synthetic set -----------------------------------------------------------------------

WORDS = {"marked": "первое", "unmarked": "второе"}


def _wav(k: int, seconds: float = 0.3) -> bytes:
    t = np.arange(round(seconds * gate0.SAMPLE_RATE)) / gate0.SAMPLE_RATE
    return gate0.wav_bytes(gate0.pcm16(0.1 * np.sin(2 * np.pi * (200 + 10 * k) * t)))


def _clip_id(k: int) -> str:
    return "E-" + hashlib.sha256(f"test|{k}".encode()).hexdigest()[:8]


def _er(k: int, member: str, transcript: str = "первое второе слово", **extra) -> dict:
    return {
        "id": _clip_id(k),
        "block": "E-R",
        "member": member,
        "wav_sha256": hashlib.sha256(f"wav|{k}".encode()).hexdigest(),
        "transcript": transcript,
        "targets": dict(WORDS),
        **extra,
    }


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, data: object) -> str:
    return _write(path, json.dumps(data, ensure_ascii=False).encode("utf-8"))


def build_root(
    tmp_path: Path,
    *,
    utterances: int = 22,
    pilot_members: tuple[str, ...] = (),
    heldout_extra: tuple[str, ...] = (),
) -> tuple[config.Config, fc.Sources]:
    """A data root with a small `gpu-ext-149`, a controls' day and `pilot-1`: two utterances on
    each WAV, so one per WAV leaves `utterances`."""
    data = (tmp_path / "data").resolve()
    inputs = tmp_path / "in"
    inputs.mkdir()
    clips = []
    for k in range(2 * utterances):
        wav = _wav(k)
        name = f"clips/{_clip_id(k)}.wav"
        sha = _write(data / fc.SET_DIR / name, wav)
        clip = _er(k, f"rinco/u-{k // 2}.wav", file=name, duration_s=0.3)
        clip.update(wav_sha256=sha, pcm_sha256=hashlib.sha256(gate0.wav_pcm(wav)).hexdigest())
        clips.append(clip)
    manifest = {
        "clips": clips,
        "questions_sha256": gpu_ext.questions_sha256(),
        "prompt_sha256": gate0.PROMPT_SHA256,
    }
    set_manifest = _json(data / fc.SET_DIR / "manifest.json", manifest)
    set_plan = _json(data / fc.SET_DIR / "plan.json", {"entries": []})
    controls = []
    for k, clip_id in enumerate(fc.CONTROLS):
        wav = _wav(1000 + k, 0.5)
        sha = _write(data / fc.CONTROL_DIR / f"clips/{clip_id}.wav", wav)
        controls.append(
            {
                "id": clip_id,
                "file": f"clips/{clip_id}.wav",
                "wav_sha256": sha,
                "pcm_sha256": hashlib.sha256(gate0.wav_pcm(wav)).hexdigest(),
                "duration_s": 0.5,
            }
        )
    control_manifest = _json(data / fc.CONTROL_DIR / gate0.MANIFEST_FILE, {"clips": controls})
    provenance = _json(
        data / fc.PILOT_DIR / "provenance.json",
        {"clips": [{"id": m, "member": m, "sha256": "f" * 64} for m in pilot_members]},
    )
    records = [{"id": "h", "sha256": sha, "split": "heldout"} for sha in heldout_extra]
    pilot_manifest = _json(data / fc.PILOT_DIR / "manifest.json", {"records": records})
    config_ = config.Config(allowed_roots=(inputs.resolve(),), data_root=data)
    sources = fc.Sources(set_manifest, set_plan, control_manifest, provenance, pilot_manifest)
    return config_, sources


FAKE_SERVER = r"""#!{python}
import base64, json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

if "--version" in sys.argv:
    print("version: 1.0.0-dev (build 12359, commit b1b770d90)")
    sys.exit(0)
open(os.environ["FC_FAKE_MARKER"], "a").write("started\n")
port = int(sys.argv[sys.argv.index("--port") + 1])
behavior = json.loads(os.environ.get("FC_FAKE", "{}"))
count = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        global count
        count += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        parts = body["messages"][0]["content"]
        audio = [p for p in parts if p["type"] == "input_audio"]
        text = next(p["text"] for p in parts if p["type"] == "text")
        with open(os.environ["FC_FAKE_LOG"], "a") as out:
            out.write(json.dumps({"n": count, "audio": len(audio), "sizes":
                [len(base64.b64decode(p["input_audio"]["data"])) for p in audio]}) + "\n")
        if count == behavior.get("exit_at"):
            os._exit(1)
        time.sleep(behavior.get("sleep", 0))
        if count == behavior.get("sleep_at"):
            time.sleep(5)
        if "«сегодня вечером»" in text:
            answer = "0"
        elif audio:
            answer = "1" if "1 — «первое»" in text else "2"
        else:
            answer = behavior.get("text_answer", "1")
        finish = "length" if count == behavior.get("length_at") else "stop"
        self._send({
            "choices": [{"message": {"content": answer + "."}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 40 + 300 * len(audio), "completion_tokens": 2},
        })


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def build_pins(tmp_path: Path) -> fc.Pins:
    """Fake weights, a fake runtime archive and its unpacked tree holding the fake server."""
    models = tmp_path / "models"
    models.mkdir()
    weights = {}
    for name in (gate0.QWEN25OMNI_MODEL_FILE, gate0.QWEN25OMNI_MMPROJ_FILE):
        weights[name] = _write(models / name, name.encode())
    opt = tmp_path / "opt"
    llama = opt / "llama-t"
    llama.mkdir(parents=True)
    server = llama / gate0.SERVER_BINARY
    server.write_text(FAKE_SERVER.replace("{python}", sys.executable), encoding="utf-8")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)
    (llama / "libfake.so.1").write_bytes(b"lib")
    (llama / "libfake.so").symlink_to("libfake.so.1")
    archive = opt / "llama-t.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(llama, arcname="llama-t")
    return fc.Pins(
        model_dir=models,
        weights=weights,
        llama_dir=llama,
        archive=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        server_sha256=hashlib.sha256(server.read_bytes()).hexdigest(),
    )


@pytest.fixture
def root(tmp_path: Path) -> dict:
    config_, sources = build_root(tmp_path)
    pins = build_pins(tmp_path)
    return {"config": config_, "sources": sources, "pins": pins, "tmp": tmp_path}


# --- selection and plan --------------------------------------------------------------------


def test_select_takes_the_first_of_each_wav_in_order_and_is_deterministic() -> None:
    clips = [_er(k, f"w{k // 2}") for k in range(50)]
    clips.insert(3, {"block": "E-C", "id": "E-00000000"})
    chosen, counts = fc.select(clips, set(), set())
    assert [c["id"] for c in chosen] == [_clip_id(k) for k in range(0, 40, 2)]
    assert [c["position"] for c in chosen] == list(range(0, 40, 2))
    assert fc.select(clips, set(), set()) == (chosen, counts)
    assert counts["e_r"] == 50 and counts["second_of_wav"] == 25 and counts["valid"] == 25
    assert counts["selected"] == 20


def test_select_skips_pilot_1_by_member_or_sha_and_pairs_not_found_once() -> None:
    clips = [_er(k, f"w{k}") for k in range(24)]
    clips[0]["member"] = "pilot/q-1.wav"
    clips[1]["transcript"] = "первое первое второе"
    clips[2]["transcript"] = "первое слово"
    chosen, counts = fc.select(clips, {"pilot/q-1.wav"}, {clips[3]["wav_sha256"]})
    ids = [c["id"] for c in chosen]
    assert not {clips[k]["id"] for k in range(4)} & set(ids)
    assert ids == [clips[k]["id"] for k in range(4, 24)]
    assert counts["pilot_1"] == 2 and counts["targets_not_once"] == 2


def test_select_refuses_under_20_valid_and_names_the_shortfall() -> None:
    clips = [_er(k, f"w{k // 2}") for k in range(38)]
    with pytest.raises(fc.ForcedChoiceError, match="only 19 valid .* 1 missing"):
        fc.select(clips, set(), set())


def test_prepare_writes_86_requests_with_one_audio_each_and_no_inference(root) -> None:
    summary = fc.prepare(root["config"], pins=root["pins"], sources=root["sources"])
    assert summary["requests"] == 86 and summary["audio_requests"] == 46
    assert summary["counts"]["selected"] == 20 and summary["counts"]["valid"] == 22
    assert not (root["tmp"] / "started").exists()
    out = root["config"].data_root / fc.OUT_DIR
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    data, entries = fc.load_prepared(
        root["config"].data_root, summary["set_sha256"], summary["plan_sha256"]
    )
    assert [e["n"] for e in entries] == list(range(1, 87))
    assert [(e["clip"], e["order"]) for e in entries[:6]] == [
        (c, o) for c in fc.CONTROLS for o in fc.ORDERS
    ]
    assert {e["expected"] for e in entries[:6]} == {"0"}
    first = data["utterances"][0]["id"]
    assert [(e["clip"], e["modality"], e["order"], e["expected"]) for e in entries[6:10]] == [
        (first, "audio", "MU", "1"),
        (first, "audio", "UM", "2"),
        (first, "text", "MU", "1"),
        (first, "text", "UM", "2"),
    ]
    assert entries[6]["text"] == entries[8]["text"]
    assert "1 — «первое»; 2 — «второе»" in entries[6]["text"]
    assert "1 — «второе»; 2 — «первое»" in entries[7]["text"]
    assert "«сегодня вечером»" in entries[0]["text"]
    assert all(e["max_tokens"] == 8 for e in entries)


def test_build_plan_refuses_more_than_86_requests() -> None:
    utterances = [{"id": f"u{k}", "transcript": "a b", **WORDS} for k in range(21)]
    controls = [{"id": c} for c in fc.CONTROLS]
    assert len(fc.build_plan(utterances[:20], controls)) == 86
    with pytest.raises(fc.ForcedChoiceError, match="at most 86"):
        fc.build_plan(utterances, controls)


def test_prepare_refuses_other_content_in_place_and_keeps_identical_files(root) -> None:
    first = fc.prepare(root["config"], pins=root["pins"], sources=root["sources"])
    assert fc.prepare(root["config"], pins=root["pins"], sources=root["sources"]) == first
    path = root["config"].data_root / fc.OUT_DIR / fc.PLAN_FILE
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(fc.ForcedChoiceError, match="remove it first"):
        fc.prepare(root["config"], pins=root["pins"], sources=root["sources"])


def test_prepare_refuses_a_held_out_clip_and_a_changed_source(tmp_path: Path) -> None:
    config_, sources = build_root(tmp_path, utterances=21)
    control = config_.data_root / fc.CONTROL_DIR / "clips/S0.wav"
    heldout = hashlib.sha256(control.read_bytes()).hexdigest()
    records = [{"id": "h", "sha256": heldout, "split": "heldout"}]
    pilot = _json(config_.data_root / fc.PILOT_DIR / "manifest.json", {"records": records})
    with pytest.raises(fc.ForcedChoiceError, match="pilot-1 manifest.json differs"):
        fc.build_set(config_.data_root, sources)
    sources = dataclasses.replace(sources, pilot_manifest_sha256=pilot)
    with pytest.raises(fc.ForcedChoiceError, match="held-out"):
        fc.build_set(config_.data_root, sources)
    changed = dataclasses.replace(sources, control_manifest_sha256="0" * 64)
    with pytest.raises(fc.ForcedChoiceError, match="controls' manifest"):
        fc.build_set(config_.data_root, changed)
    changed = dataclasses.replace(sources, pilot_provenance_sha256="0" * 64)
    with pytest.raises(fc.ForcedChoiceError, match="pilot-1 provenance.json differs"):
        fc.build_set(config_.data_root, changed)
    with pytest.raises(gpu_ext.ExtError, match="manifest.json differs"):
        fc.build_set(config_.data_root, fc.Sources("0" * 64, "0" * 64, "0" * 64))


def test_prepare_excludes_utterances_of_wavs_that_pilot_1_holds(tmp_path: Path) -> None:
    config_, sources = build_root(tmp_path, pilot_members=("rinco/u-0.wav", "rinco/u-1.wav"))
    data, _ = fc.build_set(config_.data_root, sources)
    assert data["counts"]["pilot_1"] == 4 and data["counts"]["valid"] == 20
    assert _clip_id(0) not in [u["id"] for u in data["utterances"]]


# --- the parser ----------------------------------------------------------------------------


def _body(content, finish="stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_body("1"), "1"),
        (_body(" 2.\n"), "2"),
        (_body("«0»"), "0"),
        (_body("**1**"), "1"),
        (_body("1 — первое"), fc.INVALID),
        (_body("12"), fc.INVALID),
        (_body("3"), fc.INVALID),
        (_body("один"), fc.INVALID),
        (_body(""), fc.INVALID),
        (_body(None), fc.INVALID),
        (_body("1", "length"), fc.INVALID),
        ({"choices": []}, fc.INVALID),
        (None, fc.INVALID),
    ],
)
def test_parse_answer_takes_only_a_bare_digit(body, expected) -> None:
    assert fc.parse_answer(body) == expected


def test_outcome_keeps_every_kind_apart() -> None:
    sent = {"sent": True, "http_status": 200}
    assert fc.outcome_of(sent, _body("1"), "1") == "correct"
    assert fc.outcome_of(sent, _body("2"), "1") == "wrong"
    assert fc.outcome_of(sent, _body("0"), "1") == "zero"
    assert fc.outcome_of(sent, _body("0"), "0") == "correct"
    assert fc.outcome_of(sent, _body("x"), "1") == fc.INVALID
    assert fc.outcome_of({**sent, "limit": "request_timeout"}, None, "1") == "timeout"
    assert fc.outcome_of({**sent, "limit": "wall_limit"}, None, "1") == "timeout"
    assert fc.outcome_of({**sent, "limit": "rss_limit"}, None, "1") == "error"
    assert fc.outcome_of({"sent": True, "http_status": 500}, _body("1"), "1") == "error"
    assert fc.outcome_of({"sent": False}, None, "1") == "not_sent"
    assert fc.outcome_of(None, None, "1") == "not_sent"


# --- pins ----------------------------------------------------------------------------------


def _no_network_no_process(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a pin check must not open a socket or start a process")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_weight", "weight file Qwen2.5-Omni-7B-Q8_0.gguf is missing"),
        ("other_weight", "weight file mmproj-Qwen2.5-Omni-7B-f16.gguf has SHA-256"),
        ("missing_archive", "runtime archive llama-t.tar.gz is missing"),
        ("other_archive", "runtime archive has SHA-256"),
        ("tree", "runtime file llama-t/libfake.so.1 differs"),
        ("extra_file", "runtime file llama-t/extra differs"),
        ("server", "llama-server has SHA-256"),
    ],
)
def test_a_missing_or_other_pin_refuses_prepare_without_network(
    root, monkeypatch, change, message
) -> None:
    pins = root["pins"]
    if change == "missing_weight":
        (pins.model_dir / gate0.QWEN25OMNI_MODEL_FILE).unlink()
    elif change == "other_weight":
        (pins.model_dir / gate0.QWEN25OMNI_MMPROJ_FILE).write_bytes(b"other")
    elif change == "missing_archive":
        pins.archive.unlink()
    elif change == "other_archive":
        pins.archive.write_bytes(b"other")
    elif change == "tree":
        (pins.llama_dir / "libfake.so.1").write_bytes(b"other")
    elif change == "extra_file":
        (pins.llama_dir / "extra").write_bytes(b"x")
    else:
        pins = dataclasses.replace(pins, server_sha256="0" * 64)
    _no_network_no_process(monkeypatch)
    with pytest.raises(fc.ForcedChoiceError, match="nothing is downloaded") as refused:
        fc.prepare(root["config"], pins=pins, sources=root["sources"])
    assert message in str(refused.value)
    assert not (root["config"].data_root / fc.OUT_DIR).exists()
    found, problems = fc.check_pins(root["pins"] if change == "server" else pins)
    assert (problems == []) == (change == "server")


# --- the run against a fake server ---------------------------------------------------------


def _cgroup(root_dir: Path, memory_max: str, swap_max: str) -> Path:
    line = next(
        line for line in Path("/proc/self/cgroup").read_text().splitlines() if line[:3] == "0::"
    )
    directory = root_dir / line[3:].lstrip("/")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "memory.max").write_text(memory_max + "\n")
    (directory / "memory.swap.max").write_text(swap_max + "\n")
    return root_dir


@pytest.fixture
def ready(root, monkeypatch) -> dict:
    summary = fc.prepare(root["config"], pins=root["pins"], sources=root["sources"])
    tmp = root["tmp"]
    meminfo = tmp / "meminfo"
    meminfo.write_text("MemTotal: 64000000 kB\nMemAvailable: 30000000 kB\nSwapFree: 0 kB\n")
    monkeypatch.setenv("FC_FAKE_MARKER", str(tmp / "started"))
    monkeypatch.setenv("FC_FAKE_LOG", str(tmp / "received"))
    kwargs = {
        "pins": root["pins"],
        "set_sha256": summary["set_sha256"],
        "plan_sha256": summary["plan_sha256"],
        "cgroup_root": _cgroup(tmp / "cgroup", "18000000000", "0"),
        "meminfo": meminfo,
        "busy": lambda: [],
        "pin_cpus": lambda: [0, 1],
    }
    return {**root, "kwargs": kwargs}


def _received(ready) -> list[dict]:
    path = ready["tmp"] / "received"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _attempt(ready, name: str = "1") -> Path:
    return ready["config"].data_root / fc.OUT_DIR / fc.ATTEMPTS_DIR / name


def _trace(ready) -> list[dict]:
    text = (_attempt(ready) / fc.TRACE_FILE).read_text("utf-8")
    return [json.loads(line) for line in text.splitlines()]


@pytest.mark.slow
def test_full_run_sends_86_requests_once_with_one_audio_each_and_shows_the_gain(
    ready, monkeypatch
) -> None:
    monkeypatch.setenv("FC_FAKE", json.dumps({"text_answer": "1"}))
    result = fc.run(ready["config"], **ready["kwargs"])
    received = _received(ready)
    assert len(received) == 86
    assert sum(r["audio"] == 1 for r in received) == 46
    assert sum(r["audio"] == 0 for r in received) == 40
    assert all(r["audio"] <= 1 for r in received)
    record = json.loads((_attempt(ready) / fc.RUN_FILE).read_text())
    assert record["complete"] is True and record["stop_reason"] is None
    assert record["rule_sha256"] == fc.RULE_SHA256 and record["threads"] == 8
    argv = record["argv"]
    assert argv[argv.index("-t") + 1] == "8" and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert record["sampling"]["temperature"] == 0 and record["server_version"].endswith(
        "b1b770d90)"
    )
    request = json.loads((_attempt(ready) / fc.REQUESTS_DIR / "7.json").read_text())
    assert "data" not in request["messages"][0]["content"][0]["input_audio"]
    assert result["verdict"] == "audio advantage shown" and result["controls_passed"]
    assert result["conditions"]["audio"]["accuracy"]["share"] == 1.0
    assert result["conditions"]["text"]["accuracy"]["share"] == 0.5
    assert result["conditions"]["text"]["order_disagreement"]["share"] == 1.0
    assert result["conditions"]["text"]["answer_1_among_1_2"]["share"] == 1.0
    assert result["audio_gain"] == {"mean": 0.5, "ci": [0.5, 0.5]}
    assert result["audio_tokens"]["median"] == 300.0
    report = (_attempt(ready) / fc.REPORT_FILE).read_text("utf-8")
    analysis = (_attempt(ready) / fc.ANALYSIS_FILE).read_text("utf-8")
    assert "первое" in report and "первое" not in analysis
    assert fc.analyze_attempt(
        ready["config"],
        "1",
        set_sha256=ready["kwargs"]["set_sha256"],
        plan_sha256=ready["kwargs"]["plan_sha256"],
    ) == json.loads(analysis)


@pytest.mark.slow
def test_a_request_timeout_stops_the_run_without_a_retry_and_marks_it_incomplete(
    ready, monkeypatch
) -> None:
    monkeypatch.setenv("FC_FAKE", json.dumps({"sleep_at": 3}))
    result = fc.run(ready["config"], request_timeout=0.5, **ready["kwargs"])
    assert len(_received(ready)) == 3
    lines = _trace(ready)
    assert len(lines) == 86
    assert lines[2]["limit"] == "request_timeout" and lines[2]["outcome"] == "timeout"
    assert {line.get("not_sent") for line in lines[3:]} == {"request_timeout"}
    assert result["verdict"] == "incomplete" and result["stop_reason"] == "request_timeout"
    assert result["requests"] == {"planned": 86, "sent": 3}
    assert result["conditions"]["audio"]["outcomes"]["not_sent"]["count"] == 40


@pytest.mark.slow
def test_the_total_time_limit_stops_the_run_as_incomplete(ready, monkeypatch) -> None:
    monkeypatch.setenv("FC_FAKE", json.dumps({"sleep": 0.2}))
    result = fc.run(ready["config"], wall_limit=3.0, **ready["kwargs"])
    lines = _trace(ready)
    sent = [line for line in lines if line["sent"]]
    assert 0 < len(sent) < 86 and len(_received(ready)) == len(sent)
    assert lines[-1]["not_sent"] == "wall_limit" and result["stop_reason"] == "wall_limit"
    assert result["verdict"] == "incomplete"


@pytest.mark.slow
def test_a_server_exit_stops_the_run_and_the_rest_is_not_sent(ready, monkeypatch) -> None:
    monkeypatch.setenv("FC_FAKE", json.dumps({"exit_at": 9, "length_at": 2}))
    result = fc.run(ready["config"], **ready["kwargs"])
    lines = _trace(ready)
    assert lines[1]["outcome"] == fc.INVALID
    assert lines[8]["outcome"] == "error" and lines[8]["http_status"] is None
    assert {line.get("not_sent") for line in lines[9:]} == {"server_exited"}
    assert result["verdict"] == "incomplete" and not result["controls_passed"]
    assert result["note"] == fc.CONTROL_NOTE


@pytest.mark.slow
def test_the_rss_watchdog_stops_the_run_before_any_request(ready, monkeypatch) -> None:
    result = fc.run(ready["config"], rss_limit=1, **ready["kwargs"])
    assert _received(ready) == []
    assert result["stop_reason"] == "rss_limit" and result["verdict"] == "incomplete"
    assert {line["not_sent"] for line in _trace(ready)} == {"rss_limit"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("cgroup", "memory.max"),
        ("swap", "memory.swap.max"),
        ("busy", "other heavy work"),
        ("cpus", "at most 8 CPUs"),
        ("memory", "MemAvailable"),
        ("plan", "plan.json differs"),
        ("clip", "differs from its SHA-256"),
        ("weights", "nothing is downloaded"),
        ("version", "--version says"),
    ],
)
def test_run_refuses_before_the_server_starts(ready, change, message) -> None:
    kwargs = dict(ready["kwargs"])
    tmp = ready["tmp"]
    if change == "cgroup":
        kwargs["cgroup_root"] = _cgroup(tmp / "cg2", "19000000000", "0")
    elif change == "swap":
        kwargs["cgroup_root"] = _cgroup(tmp / "cg2", "18000000000", "max")
    elif change == "busy":
        kwargs["busy"] = lambda: ["pytest"]
    elif change == "cpus":
        kwargs["pin_cpus"] = lambda: list(range(9))
    elif change == "memory":
        kwargs["meminfo"].write_text("MemAvailable: 1000 kB\n")
    elif change == "plan":
        kwargs["plan_sha256"] = "0" * 64
    elif change == "clip":
        data = json.loads((ready["config"].data_root / fc.OUT_DIR / fc.SET_FILE).read_text())
        (ready["config"].data_root / data["utterances"][5]["file"]).write_bytes(_wav(999))
    elif change == "version":
        kwargs["pins"] = dataclasses.replace(kwargs["pins"], server_version="other")
    else:
        (kwargs["pins"].model_dir / gate0.QWEN25OMNI_MODEL_FILE).write_bytes(b"other")
    with pytest.raises((fc.ForcedChoiceError, gate0.Gate0Error), match=message):
        fc.run(ready["config"], **kwargs)
    assert not (tmp / "started").exists()
    assert not _attempt(ready).exists()


def test_run_refuses_a_second_attempt_without_the_owners_decision(ready) -> None:
    earlier = _attempt(ready)
    earlier.mkdir(parents=True)
    (earlier / fc.RUN_FILE).write_text("{}")
    kwargs = dict(ready["kwargs"])
    for decision in (None, "", "later"):
        with pytest.raises(fc.ForcedChoiceError, match="attempts \\['1'\\] already ran"):
            fc.run(ready["config"], owner_decision=decision, **kwargs)
    assert not (ready["tmp"] / "started").exists()
    assert not _attempt(ready, "2").exists()
    # with the owner's link the run goes on to its other checks: here the pins refuse it
    kwargs["pins"] = dataclasses.replace(kwargs["pins"], server_version="other")
    link = "https://github.com/renidere/sound-scout-plugin/issues/205#issuecomment-1"
    with pytest.raises(fc.ForcedChoiceError, match="--version says"):
        fc.run(ready["config"], owner_decision=link, **kwargs)
    assert not (ready["tmp"] / "started").exists()
    assert not _attempt(ready, "2").exists()


def test_run_refuses_under_another_rule_block(ready, tmp_path: Path) -> None:
    edited = tmp_path / "0036.md"
    edited.write_text(DOC.read_text("utf-8").replace("120 с на запрос", "600 с на запрос"))
    with pytest.raises(fc.ForcedChoiceError, match="SHA-256 is"):
        fc.run(ready["config"], decision=edited, **ready["kwargs"])
    assert not (ready["tmp"] / "started").exists()


def test_busy_processes_finds_a_prosody_run_on_the_host() -> None:
    code = "import time; time.sleep(30)"
    child = subprocess.Popen([sys.executable, "-c", code, "sound_scout.eval.prosody_confirm"])
    try:
        assert "sound_scout.eval.prosody_confirm" in fc.busy_processes()
    finally:
        child.kill()
        child.wait()


# --- the analysis --------------------------------------------------------------------------


def _plan(utterances: int = 20) -> list[dict]:
    items = [{"id": f"u{k}", "transcript": "a b", **WORDS} for k in range(utterances)]
    return fc.build_plan(items, [{"id": c} for c in fc.CONTROLS])


def _answers(entries, answer_of, *, stop_after=None) -> tuple[list, dict]:
    lines, bodies = [], {}
    for entry in entries:
        if stop_after is not None and entry["n"] > stop_after:
            lines.append({"n": entry["n"], "sent": False, "not_sent": "wall_limit"})
            continue
        answer = answer_of(entry)
        if answer == "timeout":
            lines.append({"n": entry["n"], "sent": True, "limit": "request_timeout"})
            continue
        lines.append({"n": entry["n"], "sent": True, "http_status": 200, "prompt_tokens": 50})
        bodies[entry["n"]] = _body(answer)
    return lines, bodies


def _analyze(entries, answer_of, stop=None, **kwargs) -> dict:
    lines, bodies = _answers(entries, answer_of, **kwargs)
    record = {"attempt": "1", "stop_reason": stop}
    return fc.analyze(record, entries, lines, bodies.get)


def _right(entry) -> str:
    return entry["expected"]


def _wrong(entry) -> str:
    return {"0": "1", "1": "2", "2": "1"}[entry["expected"]]


def test_bootstrap_is_the_preregistered_cluster_bootstrap_and_is_deterministic() -> None:
    audio = np.array([1, 0.5, 1, 0, 1, 1, 0.5, 1, 0, 1, 1, 1, 0.5, 0, 1, 1, 0.5, 1, 1, 0])
    text = np.array([0.5, 0.5, 0, 0, 1, 0.5, 0.5, 0, 0, 1, 0.5, 0.5, 0, 0, 1, 0.5, 0.5, 0, 0, 1])
    found = fc.bootstrap(audio, text)
    rng = np.random.default_rng(0)
    index = rng.integers(0, 20, size=(10_000, 20))
    for name, values in (
        ("audio", audio[index].mean(axis=1)),
        ("text", text[index].mean(axis=1)),
        ("gain", (audio[index] - text[index]).mean(axis=1)),
    ):
        low, high = np.percentile(values, [2.5, 97.5])
        assert found[name] == [round(float(low), 6), round(float(high), 6)]
    assert fc.bootstrap(audio, text) == found
    assert found["gain"][0] < float(np.mean(audio - text)) < found["gain"][1]
    assert fc.bootstrap(np.array([]), np.array([])) == {"audio": None, "text": None, "gain": None}


def test_analyze_shows_the_gain_only_when_the_run_is_complete_and_the_bounds_hold() -> None:
    entries = _plan()

    def audio_right(entry):
        if entry["kind"] == "control":
            return "0"
        return _right(entry) if entry["modality"] == "audio" else "0"

    shown = _analyze(entries, audio_right)
    assert shown["verdict"] == "audio advantage shown" and shown["controls_passed"]
    assert shown["note"] is None
    assert shown["audio_gain"] == {"mean": 1.0, "ci": [1.0, 1.0]}
    assert shown["conditions"]["text"]["outcomes"]["zero"]["share"] == 1.0
    assert shown["conditions"]["text"]["order_disagreement"]["share"] == 0.0
    assert shown["outcomes_all_utterances"]["zero"] == {"count": 40, "of": 80, "share": 0.5}

    same = _analyze(entries, _right)
    assert same["verdict"] == "not shown" and same["audio_gain"]["ci"] == [0.0, 0.0]

    late = _analyze(entries, audio_right, stop="wall_limit", stop_after=50)
    assert late["verdict"] == "incomplete" and late["complete"] is False
    assert late["conditions"]["audio"]["accuracy"]["of"] == 40
    assert late["conditions"]["audio"]["outcomes"]["not_sent"]["count"] == 18


def test_analyze_counts_order_bias_timeouts_and_failed_controls_in_the_denominator() -> None:
    entries = _plan()

    def position_1(entry):
        if entry["kind"] == "control":
            return "1" if entry["clip"] == "N1" else "0"
        if entry["clip"] == "u0" and entry["modality"] == "audio":
            return "timeout"
        return "1"

    result = _analyze(entries, position_1)
    audio, text = result["conditions"]["audio"], result["conditions"]["text"]
    assert audio["accuracy"] == {"count": 19, "of": 40, "share": 0.475}
    assert audio["accuracy_by_order"]["MU"]["share"] == 0.95
    assert audio["accuracy_by_order"]["UM"]["share"] == 0.0
    assert audio["outcomes"]["timeout"]["count"] == 2
    assert audio["order_disagreement"]["share"] == 1.0
    assert text["answer_1_among_1_2"]["share"] == 1.0
    assert result["scores"]["u0"] == {"audio": 0.0, "text": 0.5}
    assert result["verdict"] == "not shown"
    assert not result["controls_passed"] and result["note"] == fc.CONTROL_NOTE
    assert [c["answer"] for c in result["controls"]] == ["0", "0", "1", "1", "0", "0"]


def test_analyze_order_agreement_counts_two_zeros_as_agreement() -> None:
    entries = _plan()
    result = _analyze(entries, lambda e: "0" if e["modality"] == "text" else _wrong(e))
    assert result["conditions"]["text"]["order_disagreement"]["share"] == 0.0
    # wrong in both orders names the same (unmarked) word: agreement, accuracy 0
    assert result["conditions"]["audio"]["order_disagreement"]["share"] == 0.0
    assert result["conditions"]["audio"]["accuracy"]["share"] == 0.0
    assert result["verdict"] == "not shown"


def test_main_refuses_without_an_attempt(tmp_path: Path, monkeypatch, capsys) -> None:
    data = tmp_path / "data"
    (data / "gate0").mkdir(parents=True)
    inputs = tmp_path / "in"
    inputs.mkdir()
    cfg = tmp_path / "sound-scout.toml"
    cfg.write_text(
        f'allowed_roots = ["{inputs}"]\ndata_root = "{data}"\n[network]\nallow = false\n'
    )
    monkeypatch.setenv("SOUND_SCOUT_CONFIG", str(cfg))
    assert fc.main(["analyze"]) == 2
    assert "attempt - not found" in capsys.readouterr().err


def test_describe_0028_counts_p_answers_of_the_same_clips_only(tmp_path: Path) -> None:
    attempt = tmp_path / "gpu"
    attempt.mkdir()
    lines = [
        {"block": "E-R", "variant": "P", "clip": "u0", "target": "marked", "answer": "да"},
        {"block": "E-R", "variant": "P", "clip": "u0", "target": "unmarked", "answer": "да"},
        {"block": "E-R", "variant": "P-text", "about": "u0", "target": "marked", "answer": "нет"},
        {"block": "E-D", "variant": "P", "clip": "u0", "target": "marked", "answer": "да"},
        {"block": "E-R", "variant": "P", "clip": "x9", "target": "marked", "answer": "да"},
    ]
    (attempt / gate0.TRACE_FILE).write_text("".join(json.dumps(x) + "\n" for x in lines))
    found = fc.describe_0028(attempt, [{"id": "u0"}])
    assert found["answers"] == {
        "P-text/marked": {"нет": 1},
        "P/marked": {"да": 1},
        "P/unmarked": {"да": 1},
    }
    assert fc.describe_0028(tmp_path / "none", [{"id": "u0"}]) is None


def test_tree_check_reads_the_archive_without_extracting(tmp_path: Path) -> None:
    base = tmp_path / "b"
    base.mkdir()
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("top/file")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    assert fc._tree_differences(archive, base) == ["top/file"]
    (base / "top").mkdir()
    (base / "top/file").write_bytes(b"abc")
    assert fc._tree_differences(archive, base) == []


# --- the acceptance criteria of #205, stage 1 ----------------------------------------------


def test_criterion_1_rule_block_states_the_generation_parser_and_metrics_the_code_uses() -> None:
    block = fc.flat(fc.rule_block(DOC.read_text(encoding="utf-8")))
    assert fc.SAMPLING == {
        "temperature": 0,
        "seed": 0,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert fc.MAX_TOKENS == 8
    for text in (
        "temperature 0, seed 0, cache_prompt false, chat_template_kwargs.enable_thinking false, "
        "max_tokens 8",
        f"content без краёв {json.dumps(gate0._EDGES, ensure_ascii=False)} должен быть "
        "ровно 1, 2 или 0",
        "finish_reason length, пустой или отсутствующий content — invalid",
        "План — 86 запросов",
        "10000 повторов, numpy.random.default_rng(0)",
        "перцентили 2.5 и 97.5",
        "нижняя граница CI G > 0 и нижняя граница CI точности audio > 0.5",
        fc.CONTROL_NOTE,
    ):
        assert text in block, text
    assert (fc.BOOTSTRAP_REPEATS, fc.BOOTSTRAP_SEED, fc.CI_PERCENTILES) == (10_000, 0, (2.5, 97.5))
    assert fc.CHANCE == 0.5 and fc.MAX_REQUESTS == 86 and fc.UTTERANCES == 20
    assert fc.CONTROLS == ("S0", "N1", "T1") and fc.RUNTIME == "qwen25omni"


def test_criterion_5_limits_are_the_preregistered_8_cpus_18e9_no_swap_120s_75min() -> None:
    block = fc.flat(fc.rule_block(DOC.read_text(encoding="utf-8")))
    assert fc.MAX_CPUS == 8 and fc.THREADS == 8
    assert fc.MEMORY_CAP_BYTES == 18_000_000_000
    assert fc.RSS_LIMIT_BYTES == 17_500_000_000
    assert fc.MIN_AVAILABLE_BYTES == 14_000_000_000
    assert fc.REQUEST_TIMEOUT_S == 120 and fc.WALL_LIMIT_S == 75 * 60
    defaults = {
        name: parameter.default for name, parameter in inspect.signature(fc.run).parameters.items()
    }
    assert defaults["request_timeout"] == 120 and defaults["wall_limit"] == 4500
    assert defaults["rss_limit"] == 17_500_000_000
    for text in (
        "memory.max <= 18000000000, memory.swap.max = 0, не больше 8 CPU",
        "MemAvailable >= 14000000000",
        "сторож RSS сервера 17500000000; 120 с на запрос; 4500 с",
        "Повтора и повторной отправки нет",
    ):
        assert text in block, text
    with pytest.raises(gate0.Gate0Error, match="memory.max"):
        gate0.check_memory_cap({"memory_max": "max", "memory_swap_max": "0"}, fc.MEMORY_CAP_BYTES)


def test_criterion_3_prepare_refuses_under_20_valid_before_any_request(tmp_path: Path) -> None:
    config_, sources = build_root(tmp_path, utterances=19)
    pins = build_pins(tmp_path)
    with pytest.raises(fc.ForcedChoiceError, match="only 19 valid .* 1 missing"):
        fc.prepare(config_, pins=pins, sources=sources)
    assert not (config_.data_root / fc.OUT_DIR).exists()
    with pytest.raises(fc.ForcedChoiceError, match="run prepare first"):
        fc.run(config_, pins=pins)
    assert not (tmp_path / "started").exists()


def test_criterion_3_a_held_out_utterance_of_pilot_1_is_never_selected(tmp_path: Path) -> None:
    heldout = hashlib.sha256(_wav(0)).hexdigest()
    config_, sources = build_root(tmp_path, heldout_extra=(heldout,))
    data, plan = fc.build_set(config_.data_root, sources)
    ids = [u["id"] for u in data["utterances"]]
    assert _clip_id(0) not in ids and len(ids) == 20
    assert data["counts"]["pilot_1"] == 1
    assert all(c["wav_sha256"] != heldout for c in data["utterances"] + data["controls"])
    assert _clip_id(0) not in {e["clip"] for e in plan["entries"]}
    assert fc.build_set(config_.data_root, sources) == (data, plan)


@pytest.mark.slow
def test_criterion_4_each_audio_request_sends_its_own_clip_and_text_requests_none(
    ready, monkeypatch
) -> None:
    monkeypatch.setenv("FC_FAKE", json.dumps({}))
    fc.run(ready["config"], **ready["kwargs"])
    data = json.loads((ready["config"].data_root / fc.OUT_DIR / fc.SET_FILE).read_text())
    clips = {c["id"]: c for c in data["utterances"] + data["controls"]}
    lines = _trace(ready)
    received = _received(ready)
    assert [line["n"] for line in lines] == list(range(1, 87))
    assert [r["n"] for r in received] == list(range(1, 87))
    for line, got in zip(lines, received, strict=True):
        if line["modality"] == "audio":
            clip = clips[line["clip"]]
            wav = (ready["config"].data_root / clip["file"]).read_bytes()
            assert line["wav_sha256"] == clip["wav_sha256"]
            assert got["sizes"] == [len(wav)]
        else:
            assert "wav_sha256" not in line and got["sizes"] == []
    controls = [line for line in lines if line["kind"] == "control"]
    assert [(c["clip"], c["expected"]) for c in controls] == [
        (c, "0") for c in fc.CONTROLS for _ in fc.ORDERS
    ]
    assert all(c["modality"] == "audio" for c in controls)
    requests = sorted((_attempt(ready) / fc.REQUESTS_DIR).iterdir())
    assert len(requests) == 86


def test_criterion_6_gain_above_zero_with_audio_at_chance_is_not_shown() -> None:
    entries = _plan()

    def mu_only(entry):
        if entry["kind"] == "control":
            return "0"
        return "1" if entry["modality"] == "audio" else "0"

    result = _analyze(entries, mu_only)
    assert result["audio_gain"] == {"mean": 0.5, "ci": [0.5, 0.5]}
    assert result["conditions"]["audio"]["accuracy_ci"] == [0.5, 0.5]
    assert result["verdict"] == "not shown" and result["controls_passed"]


def test_criterion_6_invalid_answers_stay_in_the_denominator_and_break_order_agreement() -> None:
    entries = _plan()
    broken = {f"u{k}" for k in range(10)}

    def half_invalid(entry):
        if entry["kind"] == "control":
            return "0"
        if entry["modality"] == "audio" and entry["clip"] in broken:
            return "?"
        return _right(entry)

    result = _analyze(entries, half_invalid)
    audio = result["conditions"]["audio"]
    assert audio["accuracy"] == {"count": 20, "of": 40, "share": 0.5}
    assert audio["outcomes"]["invalid"] == {"count": 20, "of": 40, "share": 0.5}
    assert audio["order_disagreement"] == {"count": 10, "of": 20, "share": 0.5}
    assert result["outcomes_all_utterances"]["invalid"] == {"count": 20, "of": 80, "share": 0.25}
    assert result["conditions"]["text"]["accuracy"]["share"] == 1.0
    assert [c["outcome"] for c in result["controls"]] == ["correct"] * 6
    assert result["audio_gain"]["mean"] == -0.5 and result["verdict"] == "not shown"
