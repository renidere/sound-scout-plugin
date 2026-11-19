"""Gate 0 spike module: the rule, parsing, signals, manipulations, the run contract.

The fake server here proves the contract of `run_spike` (what is sent, traced and refused), never
what the model hears: that is only the real run recorded in docs/decisions/0007.
"""

import copy
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config, review
from sound_scout.eval import gate0
from sound_scout.smoke import speech_like_control

REPO = Path(__file__).resolve().parent.parent
RULE_DOC = f"{gate0.RULE_BEGIN}\n```text\nGATE0-RULE v1 (test)\nrule body\n```\n{gate0.RULE_END}\n"

# --- parsing, rule block, texts ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        ("Речь.", gate0.K_OPTIONS, "речь"),
        ("**Тишина**", gate0.K_OPTIONS, "тишина"),
        ("  «шум»!", gate0.K_OPTIONS, "шум"),
        ("речь и музыка", gate0.K_OPTIONS, gate0.INVALID),
        ("тона", gate0.K_OPTIONS, gate0.INVALID),
        ("", gate0.K_OPTIONS, gate0.INVALID),
        ("нет речи", gate0.P_OPTIONS, "нет речи"),
        ("Нет речи.", gate0.P_OPTIONS, "нет речи"),
        ("Нет.", gate0.P_OPTIONS, "нет"),
        ("Да", gate0.P_OPTIONS, "да"),
        ("да, нет", gate0.P_OPTIONS, gate0.INVALID),
        ("нет речи, но да", gate0.P_OPTIONS, gate0.INVALID),
        ("Не знаю", gate0.P_OPTIONS, gate0.INVALID),
    ],
)
def test_parse_closed_finds_exactly_one_whole_word_option(text, options, expected) -> None:
    assert gate0.parse_closed(text, options) == expected


def test_decision_document_rule_block_hash_is_the_one_written_next_to_it() -> None:
    document = (REPO / gate0.DECISION_DOC).read_text(encoding="utf-8")
    recorded = re.search(r"SHA-256 блока: `([0-9a-f]{64})`", document).group(1)
    assert gate0.rule_sha256(document) == recorded
    assert gate0.rule_block(document).startswith("```text\nGATE0-RULE v1 (sound-scout #23")


@pytest.mark.parametrize(
    "document",
    ["no markers", RULE_DOC + RULE_DOC, f"{gate0.RULE_BEGIN}\nsomething else\n{gate0.RULE_END}"],
)
def test_rule_block_refuses_a_missing_doubled_or_foreign_block(document) -> None:
    with pytest.raises(gate0.Gate0Error):
        gate0.rule_block(document)


def test_review_prompt_appends_trusted_fields_and_the_transcript_only_when_given() -> None:
    prompt = review.load_prompt()
    blind = gate0.review_prompt(prompt, "B1", 10.6)
    text = gate0.review_prompt(prompt, "B1", 10.6, "слова")
    assert blind.startswith(prompt.text)
    assert '"clip_id": "B1", "duration_s": 10.6, "task": null, "intent": null' in blind
    assert "Расшифровка" not in blind
    assert (
        text == prompt.text + "\n\nРасшифровка (вместо аудио): «слова»" + blind[len(prompt.text) :]
    )
    assert prompt.sha256 == gate0.PROMPT_SHA256


# --- signals and manipulations -------------------------------------------------------------


def _dbfs(samples: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _pcm_samples(samples: np.ndarray) -> np.ndarray:
    return np.frombuffer(gate0.pcm16(samples), dtype="<i2") / 32767


def test_silence_is_ten_seconds_of_zeros() -> None:
    pcm = gate0.pcm16(gate0.silence())
    assert len(pcm) == 2 * 10 * gate0.SAMPLE_RATE
    assert set(pcm) == {0}


def test_tone_peaks_at_440_hz_at_minus_20_dbfs() -> None:
    samples = _pcm_samples(gate0.tone())
    spectrum = np.abs(np.fft.rfft(samples))
    peak_hz = np.fft.rfftfreq(len(samples), 1 / gate0.SAMPLE_RATE)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(440.0, abs=0.2)
    assert 20 * np.log10(np.max(np.abs(samples))) == pytest.approx(-20.0, abs=0.01)
    assert abs(samples[0]) < 1e-3 and abs(samples[-1]) < 1e-3


def test_pink_noise_is_minus_30_dbfs_deterministic_and_falls_with_frequency() -> None:
    samples = _pcm_samples(gate0.pink_noise())
    assert _dbfs(samples) == pytest.approx(-30.0, abs=0.5)
    assert gate0.pcm16(gate0.pink_noise()) == gate0.pcm16(gate0.pink_noise())
    assert gate0.pcm16(gate0.pink_noise(seed=1)) != gate0.pcm16(gate0.pink_noise())
    power = np.abs(np.fft.rfft(samples)) ** 2
    frequencies = np.fft.rfftfreq(len(samples), 1 / gate0.SAMPLE_RATE)
    low = power[(frequencies > 100) & (frequencies < 200)].mean()
    high = power[(frequencies > 1000) & (frequencies < 2000)].mean()
    assert 10 * np.log10(low / high) == pytest.approx(10.0, abs=1.5)


def _median_f0(samples: np.ndarray, start: float, end: float) -> float:
    word = {"start": start, "end": end}
    return gate0.f0_medians(samples, [word])[0]


def test_pitch_manipulation_raises_the_word_by_four_semitones_and_leaves_the_rest() -> None:
    voice = speech_like_control(3.0, gate0.SAMPLE_RATE, f0=130.0)
    base, variant, scale = gate0.build_pair(voice, 1.0, 1.5, 4.0, 1.0)
    assert scale == 1.0
    assert len(base) == len(variant) == len(voice)
    inside = 12 * np.log2(_median_f0(variant, 1.05, 1.45) / _median_f0(base, 1.05, 1.45))
    assert inside == pytest.approx(4.0, abs=0.5)
    for start, end in ((0.2, 0.9), (1.7, 2.8)):
        outside = 12 * np.log2(_median_f0(variant, start, end) / _median_f0(base, start, end))
        assert outside == pytest.approx(0.0, abs=0.3)


def test_duration_manipulation_stretches_only_the_word_by_1_6() -> None:
    voice = speech_like_control(3.0, gate0.SAMPLE_RATE, f0=130.0)
    base, variant, _ = gate0.build_pair(voice, 1.0, 1.5, 0.0, 1.6)
    stretched = 0.5 + (len(variant) - len(base)) / gate0.SAMPLE_RATE
    assert stretched / 0.5 == pytest.approx(1.6, rel=0.05)
    assert _median_f0(variant, 0.2, 0.9) == pytest.approx(_median_f0(base, 0.2, 0.9), rel=0.02)


@pytest.mark.parametrize(("semitones", "duration_factor"), [(0.0, 1.6), (3.0, 1.3), (4.0, 1.0)])
def test_build_pair_gives_the_same_pcm_on_every_call(semitones, duration_factor) -> None:
    import parselmouth

    voice = speech_like_control(3.0, gate0.SAMPLE_RATE, f0=130.0)
    pairs = []
    for _ in range(3):
        base, variant, _ = gate0.build_pair(voice, 1.0, 1.5, semitones, duration_factor)
        pairs.append((base.tobytes(), variant.tobytes()))
        parselmouth.praat.run(
            "random_initializeSafelyAndUnpredictably ()\nx = randomUniform (0, 1)\n"
        )
    assert pairs[0] == pairs[1] == pairs[2]


def test_build_pair_without_the_seed_stretches_differently_each_call(monkeypatch) -> None:
    monkeypatch.setattr(
        gate0, "RESYNTHESIS_SEED_SCRIPT", "random_initializeSafelyAndUnpredictably ()\n"
    )
    voice = speech_like_control(3.0, gate0.SAMPLE_RATE, f0=130.0)
    first = gate0.build_pair(voice, 1.0, 1.5, 0.0, 1.6)[1]
    second = gate0.build_pair(voice, 1.0, 1.5, 0.0, 1.6)[1]
    assert len(first) == len(second) and first.tobytes() != second.tobytes()


def test_a_loud_pair_is_scaled_by_one_factor_to_the_peak_limit() -> None:
    voice = speech_like_control(2.0, gate0.SAMPLE_RATE, f0=130.0) * 2.0
    base, variant, scale = gate0.build_pair(voice, 0.8, 1.2, 4.0, 1.0)
    assert scale < 1.0
    assert max(np.max(np.abs(base)), np.max(np.abs(variant))) == pytest.approx(0.99, abs=1e-9)


def _word(start, end, probability=0.9, f0=120.0, text="слово"):
    return {
        "word": text,
        "start": start,
        "end": end,
        "probability": probability,
        "f0_median_hz": f0,
    }


def test_choose_target_takes_the_lowest_f0_inner_word_earlier_on_a_tie() -> None:
    words = [
        _word(0.0, 0.4, f0=60.0),  # first: never
        _word(0.5, 0.9, f0=110.0),
        _word(1.0, 1.4, f0=100.0),
        _word(1.5, 1.9, f0=100.0),  # tie, later
        _word(2.0, 2.1, f0=50.0),  # too short
        _word(2.2, 3.4, f0=50.0),  # too long
        _word(3.5, 3.9, probability=0.5, f0=50.0),  # unsure
        _word(4.0, 4.4, f0=None),  # unvoiced
        _word(4.5, 4.9, f0=40.0),  # last: never
    ]
    assert gate0.choose_target(words) == 2


def test_choose_target_is_none_without_a_candidate() -> None:
    assert gate0.choose_target([_word(0.0, 0.4), _word(0.5, 0.9)]) is None


def test_wav_pcm_returns_the_data_block_and_refuses_another_layout() -> None:
    pcm = gate0.pcm16(gate0.tone(0.1))
    assert gate0.wav_pcm(gate0.wav_bytes(pcm)) == pcm
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(gate0.SAMPLE_RATE)
        out.writeframes(pcm)
    with pytest.raises(gate0.Gate0Error):
        gate0.wav_pcm(buffer.getvalue())


def test_server_argv_is_local_with_the_protocol_settings_and_no_download_flag() -> None:
    argv = gate0.server_argv(Path("/opt/llama-server"), Path("/models"), 4321, 12)
    assert argv[0] == "/opt/llama-server"
    pairs = dict(zip(argv[1::2], argv[2::2], strict=False))
    assert argv[argv.index("-m") + 1] == "/models/gemma-4-E4B-it-Q8_0.gguf"
    assert argv[argv.index("--mmproj") + 1] == "/models/mmproj-gemma-4-E4B-it-BF16.gguf"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "4321"
    assert argv[argv.index("-c") + 1] == "4096"
    assert argv[argv.index("--parallel") + 1] == "1"
    assert argv[argv.index("-t") + 1] == "12"
    assert "--jinja" in argv and "--no-webui" in argv
    assert not gate0.DOWNLOAD_FLAGS.intersection(argv) and pairs


def test_dependencies_runtime_section_describes_the_llama_server_route_gate0_uses() -> None:
    dependencies = (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    runtime = dependencies.split("### 3.2. Runtime", 1)[1].split("\n### ", 1)[0]
    flat = " ".join(runtime.split())
    argv = gate0.server_argv(Path("/opt/llama-server"), Path("/models"), 4321, 12)
    assert gate0.SERVER_BINARY == "llama-server" and "--jinja" in argv
    assert argv[argv.index("--parallel") + 1] == "1"
    assert argv[argv.index("--host") + 1] == gate0.HOST
    assert gate0.SAMPLING["chat_template_kwargs"] == {"enable_thinking": False}
    assert "используется только `llama-server`" in flat
    assert "`llama-mtmd-cli` не используется" in flat
    for detail in (
        "Сервер запускается с `--jinja`",
        f"`POST {gate0.CHAT_ROUTE}`",
        f"`{gate0.HOST}`",
        "(`--parallel 1`)",
        "просьба подавить режим рассуждений, а не гарантия",
        "`reasoning_content`",
        "`input_audio` (WAV в base64",
        '"chat_template_kwargs": {"enable_thinking": false}',
        str(gate0.DECISION_DOC),
    ):
        assert detail in flat, detail
    for retired in ("--audio", "multimodal_data"):
        assert retired not in dependencies, retired
    mtmd_lines = [line for line in dependencies.splitlines() if "llama-mtmd-cli" in line]
    assert mtmd_lines and all("не используется" in line for line in mtmd_lines)


# --- the verdict rule ----------------------------------------------------------------------


def _synthetic_manifest() -> dict:
    pairs = [
        {"pair": i, "source": f"H0{i}", "base": f"B{i}", "variant": f"V{i}", "target": {"index": 2}}
        for i in (1, 2, 3)
    ]
    clips = [
        {"id": f"{kind}{i}", "duration_s": 10.0, "pair": i} for i in (1, 2, 3) for kind in "BV"
    ]
    clips += [{"id": clip, "duration_s": 10.0} for clip in gate0.CONTROLS]
    return {"mode": "synthetic", "pairs": pairs, "clips": clips}


def _line(variant, clip, answer, take=None, **fields):
    return {
        "n": 1,
        "variant": variant,
        "clip": clip,
        "take": take,
        "modality": "audio",
        "duration_s": 10.0,
        "sent": True,
        "http_status": 200,
        "finish_reason": "stop",
        "text_tokens": 30,
        "prompt_tokens": 280,
        "audio_tokens": 250,
        "answer": answer,
        **fields,
    }


EMPTY_REVIEW = {"valid": True, "observations": 0, "unlocalized_observations": 0}


def _held_lines() -> list[dict]:
    lines = [{"variant": "E0", "clip": "E0", "rejected": True, "server_calls": 0, "sent": False}]
    for clip in gate0.CONTROLS:
        lines += [
            _line("K1", clip, "тишина"),
            _line("K2", clip, "шум"),
            _line("P", clip, "нет речи"),
            _line("A", clip, dict(EMPTY_REVIEW)),
        ]
    for i in (1, 2, 3):
        lines += [_line("P", f"B{i}", "нет"), _line("P", f"V{i}", "да")]
    return lines


RECORD = {"preconditions": {"checks": {"weights": True, "prompt": True}}}


def test_synthetic_run_passing_controls_and_pairs_is_held_never_ready() -> None:
    result = gate0.decide(_synthetic_manifest(), RECORD, _held_lines())
    assert result["verdict"] == "HELD"
    assert result["preconditions"] == {"P1": True, "P2": True, "P3": True}
    assert result["C"] is True and result["S"] is True


@pytest.mark.parametrize("clip", gate0.CONTROLS)
@pytest.mark.parametrize(
    ("variant", "bad"),
    [
        ("K1", "речь"),
        ("K1", gate0.INVALID),
        ("K2", "речь"),
        ("P", "да"),
        ("P", gate0.INVALID),
        ("A", {"valid": True, "observations": 1, "unlocalized_observations": 0}),
        ("A", {"valid": True, "observations": 0, "unlocalized_observations": 1}),
        ("A", {"valid": False, "errors": ["$: response is not valid JSON"]}),
    ],
)
def test_any_failed_control_item_is_not_ready(clip, variant, bad) -> None:
    lines = _held_lines()
    line = next(item for item in lines if item["variant"] == variant and item["clip"] == clip)
    line["answer"] = bad
    result = gate0.decide(_synthetic_manifest(), RECORD, lines)
    assert result["verdict"] == "NOT READY"
    assert result["reason"].startswith("C:")


@pytest.mark.parametrize(
    "field", [{"finish_reason": "length"}, {"http_status": 500}, {"sent": False}]
)
def test_a_truncated_failed_or_unsent_control_request_does_not_pass(field) -> None:
    lines = _held_lines()
    next(item for item in lines if item["variant"] == "K1").update(field)
    assert gate0.decide(_synthetic_manifest(), RECORD, lines)["verdict"] == "NOT READY"


def test_only_one_distinguished_synthetic_pair_is_not_ready() -> None:
    lines = _held_lines()
    for line in lines:
        if line["clip"] in ("V2", "V3"):
            line["answer"] = "нет"
    result = gate0.decide(_synthetic_manifest(), RECORD, lines)
    assert result["verdict"] == "NOT READY"
    assert [pair["distinguished"] for pair in result["synthetic_pairs"]] == [True, False, False]


def test_two_distinguished_pairs_pass_and_a_pair_without_target_is_not_distinguished() -> None:
    manifest = _synthetic_manifest()
    manifest["pairs"][2]["target"] = None
    lines = [line for line in _held_lines() if line["clip"] not in ("B3", "V3")]
    result = gate0.decide(manifest, RECORD, lines)
    assert result["verdict"] == "HELD"
    assert result["synthetic_pairs"][2]["distinguished"] is False


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        (lambda lines, record: lines[0].update(rejected=False), "P1"),
        (lambda lines, record: lines[0].update(server_calls=1), "P1"),
        (lambda lines, record: record["preconditions"]["checks"].update(weights=False), "P2"),
        (lambda lines, record: record["preconditions"]["checks"].update(prompt=False), "P2"),
        (lambda lines, record: lines[1].update(audio_tokens=150), "P3"),
    ],
)
def test_a_broken_precondition_makes_the_run_invalid_and_not_ready(change, failed) -> None:
    lines, record = _held_lines(), copy.deepcopy(RECORD)
    change(lines, record)
    result = gate0.decide(_synthetic_manifest(), record, lines)
    assert result["verdict"] == "NOT READY"
    assert result["reason"] == f"протокол не выполнен: ['{failed}']"


def test_audio_route_falls_back_to_the_k1_slope_without_the_tokenizer() -> None:
    lines = [
        _line("K1", f"c{d}", "речь", duration_s=d, text_tokens=None, prompt_tokens=int(40 + 25 * d))
        for d in (5.0, 10.0, 15.0)
    ]
    route = gate0.audio_route(lines)
    assert route["method"] == "k1_slope" and route["passed"] is True
    assert route["slope"] == pytest.approx(25.0)
    for line in lines:
        line["prompt_tokens"] = 300
    assert gate0.audio_route(lines)["passed"] is False


def _k1_slope_lines() -> list[dict]:
    return [
        _line("K1", f"c{d}", "речь", n=int(d), duration_s=d, prompt_tokens=int(40 + 25 * d),
              audio_tokens=int(25 * d))
        for d in (5.0, 10.0, 15.0)
    ]  # fmt: skip


def test_one_untokenized_request_keeps_the_measured_failures_and_adds_the_k1_slope() -> None:
    lines = _k1_slope_lines() + [_line("A", "c20", {}, n=20, duration_s=20.0, text_tokens=None)]
    route = gate0.audio_route(lines)
    assert route["method"] == "tokenize+k1_slope" and route["untokenized"] == [20]
    assert route["passed"] is True and route["slope"] == pytest.approx(25.0)
    lines[0]["audio_tokens"] = 10
    route = gate0.audio_route(lines)
    assert route["passed"] is False and route["failing"] == [5]


def _natural_manifest() -> dict:
    pairs = [{"pair": j, "a": f"P{j}-a", "b": f"P{j}-b"} for j in (1, 2, 3)]
    return {"mode": "natural", "pairs": pairs, "clips": []}


def _natural_lines(correct_pairs=(1, 2, 3)) -> list[dict]:
    lines = [line for line in _held_lines() if line["clip"] in ("E0", *gate0.CONTROLS)]
    for line in lines:
        if line["variant"] == "P":
            line["take"] = "a"
    for j in (1, 2, 3):
        good = j in correct_pairs
        lines += [
            _line("P", f"P{j}-a", "да", take="a"),
            _line("P", f"P{j}-b", "нет", take="a"),
            _line("P", f"P{j}-b", "да" if good else "нет", take="b"),
            _line("P", f"P{j}-a", "нет", take="b"),
        ]
    return lines


def test_natural_rerun_with_two_of_three_pairs_and_clean_controls_is_ready() -> None:
    result = gate0.decide(_natural_manifest(), RECORD, _natural_lines(correct_pairs=(1, 3)))
    assert result["verdict"] == "READY"
    assert [pair["distinguished"] for pair in result["natural_pairs"]] == [True, False, True]


def test_natural_rerun_with_one_pair_or_a_failed_control_is_not_ready() -> None:
    assert (
        gate0.decide(_natural_manifest(), RECORD, _natural_lines(correct_pairs=(2,)))["verdict"]
        == "NOT READY"
    )
    lines = _natural_lines()
    next(line for line in lines if line["variant"] == "K2" and line["clip"] == "T1")["answer"] = (
        "речь"
    )
    assert gate0.decide(_natural_manifest(), RECORD, lines)["verdict"] == "NOT READY"


# --- prepare and the run against a fake server ---------------------------------------------

FAKE_SERVER = r"""#!{python}
import base64, ctypes, hashlib, io, json, os, sys, wave
from http.server import BaseHTTPRequestHandler, HTTPServer

pdeathsig = ctypes.c_int()
ctypes.CDLL(None).prctl(2, ctypes.byref(pdeathsig))  # PR_GET_PDEATHSIG
open(os.environ["GATE0_FAKE_MARKER"], "a").write(f"started {pdeathsig.value}\n")
port = int(sys.argv[sys.argv.index("--port") + 1])


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
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            return self._send({"prompt": "template"})
        if self.path == "/tokenize":
            return self._send({"tokens": [1] * 30})
        audio_tokens = 0
        if os.environ.get("GATE0_FAKE_BODIES"):
            with open(os.environ["GATE0_FAKE_BODIES"], "a") as out:
                out.write(json.dumps(body, ensure_ascii=False) + "\n")
        for part in body["messages"][0]["content"]:
            if part["type"] == "input_audio":
                data = base64.b64decode(part["input_audio"]["data"])
                with wave.open(io.BytesIO(data)) as source:
                    pcm = source.readframes(source.getnframes())
                audio_tokens = 25 * len(pcm) // 2 // 16000
                with open(os.environ["GATE0_FAKE_RECEIVED"], "a") as out:
                    out.write(hashlib.sha256(pcm).hexdigest() + "\n")
        content = (
            json.dumps({"observations": [], "unlocalized_observations": [],
                        "uncertainties": [], "limitations": []})
            if "response_format" in body else "Речь."
        )
        self._send({
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30 + audio_tokens, "completion_tokens": 2},
        })


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def _fake_transcribe(path, run, config_):
    samples = np.frombuffer(gate0.wav_pcm(Path(path).read_bytes()), dtype="<i2") / 32767
    words = [
        {"word": " раз", "start": 0.1, "end": 0.5, "probability": 0.9},
        {"word": " два,", "start": 1.0, "end": 1.5, "probability": 0.9},
        {"word": " три", "start": 2.0, "end": 2.6, "probability": 0.9},
    ]
    return {
        "samples": samples,
        "sha256": "0" * 64,
        "duration_s": len(samples) / gate0.SAMPLE_RATE,
        "words": words,
        "transcript": "раз два, три",
    }


@pytest.fixture
def spike(tmp_path: Path, monkeypatch) -> dict:
    """A prepared synthetic spike in a temp data root, a fake llama-server and fake weights."""
    inputs = tmp_path / "in"
    data = tmp_path / "data"
    inputs.mkdir()
    data.mkdir()
    config_ = config.Config(allowed_roots=(inputs.resolve(),), data_root=data.resolve())
    for name in gate0.SOURCE_FILES:
        voice = speech_like_control(3.0, gate0.SAMPLE_RATE, f0=130.0)
        (inputs / name).write_bytes(gate0.wav_bytes(gate0.pcm16(voice)))

    decision = tmp_path / "0007.md"
    decision.write_text(RULE_DOC, encoding="utf-8")
    manifest = gate0.prepare_clips(
        "2026-11-18", config_, decision=decision, transcribe=_fake_transcribe
    )
    llama = tmp_path / "llama"
    models = tmp_path / "models"
    llama.mkdir()
    models.mkdir()
    server = llama / gate0.SERVER_BINARY
    server.write_text(FAKE_SERVER.replace("{python}", sys.executable), encoding="utf-8")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)
    weights = {}
    for name in (gate0.MODEL_FILE, gate0.MMPROJ_FILE):
        (models / name).write_bytes(name.encode())
        weights[name] = hashlib.sha256(name.encode()).hexdigest()
    marker, received = tmp_path / "started", tmp_path / "received"
    monkeypatch.setenv("GATE0_FAKE_MARKER", str(marker))
    monkeypatch.setenv("GATE0_FAKE_RECEIVED", str(received))
    return {
        "config": config_,
        "manifest": manifest,
        "decision": decision,
        "kwargs": {
            "decision": decision,
            "model_dir": models,
            "llama_dir": llama,
            "weights": weights,
        },
        "marker": marker,
        "received": received,
        "out": data.resolve() / "2026-11-18",
    }


def test_prepare_builds_nine_clips_and_keeps_the_words_out_of_the_manifest(spike) -> None:
    manifest = spike["manifest"]
    assert [clip["id"] for clip in manifest["clips"]] == [
        "B1", "V1", "B2", "V2", "B3", "V3", "S0", "N1", "T1",
    ]  # fmt: skip
    assert all(pair["target"]["index"] == 2 for pair in manifest["pairs"])
    assert manifest["rule_sha256"] == gate0.rule_sha256(RULE_DOC)
    assert manifest["questions_sha256"] == gate0.questions_sha256()
    for clip in manifest["clips"]:
        data = (spike["out"] / clip["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == clip["wav_sha256"]
        assert hashlib.sha256(gate0.wav_pcm(data)).hexdigest() == clip["pcm_sha256"]
        assert clip.get("synthetic_manipulation", False) == clip["id"].startswith("V")
    text = (spike["out"] / gate0.MANIFEST_FILE).read_text(encoding="utf-8")
    assert "два" not in text and "раз" not in text
    words = json.loads((spike["out"] / gate0.WORDS_FILE).read_text(encoding="utf-8"))
    assert words["H01"]["target_word"] == "два"


def test_two_prepares_of_the_same_sources_give_the_same_pcm_and_record_the_seed(spike) -> None:
    again = gate0.prepare_clips(
        "2026-11-19", spike["config"], decision=spike["decision"], transcribe=_fake_transcribe
    )
    first = {clip["id"]: clip["pcm_sha256"] for clip in spike["manifest"]["clips"]}
    second = {clip["id"]: clip["pcm_sha256"] for clip in again["clips"]}
    assert first == second and len(first) == 9
    for manifest in (spike["manifest"], again):
        resynthesis = manifest["resynthesis"]
        assert resynthesis["praat_random_seed"] == gate0.RESYNTHESIS_SEED == 0
        assert resynthesis["seed_script"] == "random_initializeWithSeedUnsafelyButPredictably (0)"
        assert resynthesis["seeded"] == "before each resynthesis"
        assert resynthesis["praat_version"] and resynthesis["parselmouth_version"]
    saved = json.loads((spike["out"] / gate0.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert saved["resynthesis"] == again["resynthesis"]


def test_prepare_never_overwrites_a_prepared_day(spike) -> None:
    with pytest.raises(gate0.Gate0Error, match="never overwrites"):
        gate0.prepare_clips("2026-11-18", spike["config"], decision=spike["decision"])


def test_prepare_after_one_that_stopped_midway_asks_to_delete_the_day(spike) -> None:
    (spike["out"] / gate0.MANIFEST_FILE).unlink()
    with pytest.raises(gate0.Gate0Error, match=r"stopped midway; delete the directory 2026-11-18/"):
        gate0.prepare_clips("2026-11-18", spike["config"], decision=spike["decision"])


def test_prepare_refuses_a_clip_over_30_s_before_writing_any_clip(spike) -> None:
    config_, inputs = spike["config"], spike["config"].allowed_roots[0]
    long_voice = speech_like_control(31.0, gate0.SAMPLE_RATE, f0=130.0)
    (inputs / gate0.SOURCE_FILES[0]).write_bytes(gate0.wav_bytes(gate0.pcm16(long_voice)))
    with pytest.raises(gate0.Gate0Error, match=r"clips over 30.0 s: \['B1', 'V1'\]"):
        gate0.prepare_clips(
            "2026-11-19", config_, decision=spike["decision"], transcribe=_fake_transcribe
        )
    assert not (config_.data_root / "2026-11-19" / gate0.CLIPS_DIR).exists()


def test_plan_of_the_synthetic_run_is_the_42_fixed_requests(spike) -> None:
    words = json.loads((spike["out"] / gate0.WORDS_FILE).read_text(encoding="utf-8"))
    entries = gate0.plan(spike["manifest"], words, review.load_prompt())
    order = [e["variant"] for e in entries]
    assert (
        order == ["K1"] * 9 + ["K2"] * 9 + ["P"] * 9 + ["P-text"] * 3 + ["A"] * 9 + ["A-text"] * 3
    )
    assert [e["n"] for e in entries] == list(range(1, 43))
    controls = [e for e in entries if e["variant"] == "P" and e["clip"] in gate0.CONTROLS]
    assert all("«два»" in e["text"] for e in controls)
    assert all(e["modality"] == "text" and "«раз два, три»" in e["text"]
               for e in entries if e["variant"] in ("P-text", "A-text"))  # fmt: skip
    assert all(e["json_response"] == (e["variant"] in ("A", "A-text")) for e in entries)


def test_run_sends_each_request_once_traces_the_pcm_sent_and_refuses_e0(spike) -> None:
    result = gate0.run_spike("2026-11-18", spike["config"], **spike["kwargs"])
    attempt = spike["out"] / gate0.ATTEMPTS_DIR / "1"
    lines = [json.loads(t) for t in (attempt / gate0.TRACE_FILE).read_text().splitlines()]
    assert len(lines) == 43
    e0 = lines[0]
    assert e0["variant"] == "E0" and e0["rejected"] is True and e0["server_calls"] == 0
    audio = [line for line in lines[1:] if line["modality"] == "audio"]
    received = spike["received"].read_text().split()
    assert [line["pcm_sha256"] for line in audio] == received
    by_id = {clip["id"]: clip for clip in spike["manifest"]["clips"]}
    for line in audio:
        assert line["pcm_sha256"] == by_id[line["clip"]]["pcm_sha256"]
        assert line["audio_tokens"] == line["prompt_tokens"] - line["text_tokens"]
        assert line["rss_peak_bytes"] > 0
    request = json.loads((attempt / gate0.REQUESTS_DIR / "1.json").read_text())
    assert "data" not in request["messages"][0]["content"][0]["input_audio"]
    assert request["temperature"] == 0 and request["cache_prompt"] is False
    record = json.loads((attempt / gate0.RUN_FILE).read_text())
    assert record["stop_reason"] is None and record["server_returncode"] is not None
    assert result["verdict"] == "NOT READY" and result["reason"].startswith("C:")
    assert result["preconditions"] == {"P1": True, "P2": True, "P3": True}
    assert gate0.evaluate(attempt) == json.loads((attempt / gate0.VERDICT_FILE).read_text())
    # the server and its `--version` die with this process: no orphan llama-server
    assert spike["marker"].read_text().split("\n")[:-1] == ["started 9", "started 9"]


@pytest.mark.parametrize("broken", ["weights", "rule"])
def test_a_weight_or_rule_mismatch_stops_the_run_before_the_server_starts(spike, broken) -> None:
    kwargs = dict(spike["kwargs"])
    if broken == "weights":
        kwargs["weights"] = {name: "0" * 64 for name in kwargs["weights"]}
    else:
        spike["decision"].write_text(RULE_DOC.replace("rule body", "edited"), encoding="utf-8")
    with pytest.raises(gate0.Gate0Error, match="before the server start"):
        gate0.run_spike("2026-11-18", spike["config"], **kwargs)
    assert not spike["marker"].exists()
    attempt = spike["out"] / gate0.ATTEMPTS_DIR / "1"
    record = json.loads((attempt / gate0.RUN_FILE).read_text())
    assert record["stop_reason"] == "precondition_failed"
    assert record["preconditions"]["checks"][broken] is False
    verdict = json.loads((attempt / gate0.VERDICT_FILE).read_text())
    assert verdict["verdict"] == "NOT READY"
    assert verdict["reason"].startswith("протокол не выполнен")
    assert verdict["preconditions"]["P2"] is (broken != "weights")


@pytest.mark.parametrize("broken", ["missing", "no rule block"])
def test_an_unreadable_decision_document_stops_the_run_before_any_attempt_directory(
    spike, broken
) -> None:
    if broken == "missing":
        spike["decision"].unlink()
    else:
        spike["decision"].write_text("no rule here", encoding="utf-8")
    with pytest.raises(gate0.Gate0Error):
        gate0.run_spike("2026-11-18", spike["config"], **spike["kwargs"])
    assert not (spike["out"] / gate0.ATTEMPTS_DIR).exists()
    assert not spike["marker"].exists()


def test_verdict_of_a_refused_attempt_without_a_trace_is_not_ready(spike, monkeypatch, capsys):
    attempt = spike["out"] / gate0.ATTEMPTS_DIR / "1"
    attempt.mkdir(parents=True)
    record = {"stop_reason": "precondition_failed", "preconditions": {"checks": {"weights": False}}}
    (attempt / gate0.RUN_FILE).write_text(json.dumps(record))
    result = gate0.evaluate(attempt)
    assert result["verdict"] == "NOT READY"
    assert result["preconditions"] == {"P1": False, "P2": False, "P3": False}
    monkeypatch.setattr(gate0.config, "load", lambda: spike["config"])
    monkeypatch.setattr(gate0.guard, "install_network_guard", lambda config_: None)
    assert gate0.main(["verdict", "--date", "2026-11-18"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "NOT READY"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["verdict", "--date", "2026-11-18"], "attempt - not found"),
        (["verdict", "--date", "2026-11-18", "--attempt", "7"], "attempt 7 not found"),
        (["verdict", "--date", "2026-13-40"], "is not a date"),
        (["run", "--date", "2026-02-30"], "is not a date"),
    ],
)
def test_cli_verdict_without_an_attempt_or_a_date_exits_2(
    spike, monkeypatch, capsys, argv, message
):
    monkeypatch.setattr(gate0.config, "load", lambda: spike["config"])
    monkeypatch.setattr(gate0.guard, "install_network_guard", lambda config_: None)
    assert gate0.main(argv) == 2
    assert message in capsys.readouterr().err


def test_rss_over_the_limit_stops_the_server_and_marks_every_request_not_sent(spike) -> None:
    result = gate0.run_spike("2026-11-18", spike["config"], rss_limit=1, **spike["kwargs"])
    attempt = spike["out"] / gate0.ATTEMPTS_DIR / "1"
    lines = [json.loads(t) for t in (attempt / gate0.TRACE_FILE).read_text().splitlines()]
    assert all(line["not_sent"] == "rss_limit" for line in lines[1:])
    assert len(lines) == 43 and lines[0]["rejected"] is True
    record = json.loads((attempt / gate0.RUN_FILE).read_text())
    assert record["stop_reason"] == "rss_limit" and record["rss_exceeded_bytes"] > 1
    assert result["verdict"] == "NOT READY"


def test_natural_prepare_builds_six_takes_and_a_51_request_plan(tmp_path: Path) -> None:
    pairs_dir = tmp_path / "in" / "pairs"
    data = tmp_path / "data"
    pairs_dir.mkdir(parents=True)
    data.mkdir()
    config_ = config.Config(allowed_roots=((tmp_path / "in").resolve(),), data_root=data.resolve())
    for j in (1, 2, 3):
        for take in "ab":
            voice = speech_like_control(2.0, 48_000, f0=120.0)
            (pairs_dir / f"P{j}-{take}.wav").write_bytes(_wav48(voice))
    spec = {f"P{j}": {"text": "мы идём домой сегодня вечером", "a": 2, "b": 4} for j in (1, 2, 3)}
    (pairs_dir / gate0.PAIRS_FILE).write_text(json.dumps(spec, ensure_ascii=False))
    decision = tmp_path / "0007.md"
    decision.write_text(RULE_DOC, encoding="utf-8")
    manifest = gate0.prepare_clips("2026-11-19", config_, natural=pairs_dir, decision=decision)
    assert manifest["mode"] == "natural" and manifest["resynthesis"] is None
    assert [clip["id"] for clip in manifest["clips"]][:6] == [
        "P1-a", "P1-b", "P2-a", "P2-b", "P3-a", "P3-b",
    ]  # fmt: skip
    assert all(clip["duration_s"] == pytest.approx(2.0) for clip in manifest["clips"][:6])
    words = json.loads((data / "2026-11-19" / gate0.WORDS_FILE).read_text(encoding="utf-8"))
    entries = gate0.plan(manifest, words, review.load_prompt())
    assert len(entries) == 51
    p1 = [(e["clip"], e["take"]) for e in entries if e["variant"] == "P" and e["pair"] == 1]
    assert p1[:4] == [("P1-a", "a"), ("P1-b", "a"), ("P1-b", "b"), ("P1-a", "b")]
    assert sum(e["variant"] == "P-text" for e in entries) == 6


@pytest.mark.parametrize(
    "spec",
    [
        {"text": "а б в", "a": 2, "b": 2},
        {"text": "Мама мыла раму, мама", "a": 1, "b": 4},
        {"text": "ёж и ещё еж", "a": 1, "b": 4},
    ],
)
def test_natural_prepare_refuses_the_same_word_stressed_in_both_takes(tmp_path: Path, spec) -> None:
    pairs_dir = tmp_path / "in" / "pairs"
    pairs_dir.mkdir(parents=True)
    (tmp_path / "data").mkdir()
    config_ = config.Config(
        allowed_roots=((tmp_path / "in").resolve(),), data_root=(tmp_path / "data").resolve()
    )
    (pairs_dir / gate0.PAIRS_FILE).write_text(json.dumps({"P1": spec}))
    decision = tmp_path / "0007.md"
    decision.write_text(RULE_DOC, encoding="utf-8")
    with pytest.raises(gate0.Gate0Error, match="same word"):
        gate0.prepare_clips("2026-11-19", config_, natural=pairs_dir, decision=decision)


def _wav48(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(48_000)
        out.writeframes(gate0.pcm16(samples))
    return buffer.getvalue()


def test_cli_refuses_a_bad_date_with_exit_code_2(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "data").mkdir()
    toml = tmp_path / "c.toml"
    toml.write_text(f'allowed_roots = ["{tmp_path / "in"}"]\ndata_root = "{tmp_path / "data"}"\n')
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(toml))
    monkeypatch.chdir(REPO)
    # `run` installs no network guard, so the audit hook does not leak into this process
    assert gate0.main(["run", "--date", "18-11-2026"]) == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err
    assert os.listdir(tmp_path / "data") == []


# --- acceptance criteria of #23 that can be proven before the real run ---------------------

# SHA-256 of the GATE0-RULE v1 block of Decision 6986704763 on #23, computed from the comment
DECISION_RULE_SHA256 = "077e01eada34c88aa13aa1ed3033666d9bc5c50cb64457631302288ba9deec35"


def _preregistered() -> str:
    document = (REPO / gate0.DECISION_DOC).read_text(encoding="utf-8")
    return document[: document.index("\n## 4.")]


def test_criterion_a_document_preregisters_backend_audio_prompt_settings_and_cost() -> None:
    document = (REPO / gate0.DECISION_DOC).read_text(encoding="utf-8")
    before = _preregistered()
    dependencies = (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    assert gate0.MODEL_REPO in before and gate0.MODEL_REVISION in before
    for name, sha in gate0.WEIGHTS.items():
        assert name in before and sha in before and sha in dependencies
    spec_prompt = (REPO / "docs/spec/AUDIO_REVIEW_PROMPT.md").read_bytes()
    assert hashlib.sha256(spec_prompt).hexdigest() == gate0.PROMPT_SHA256
    assert gate0.PROMPT_SHA256 in before
    assert "16 kHz mono PCM s16 WAV" in before and "input_audio (WAV base64" in before
    clips = before[
        before.index("<!-- gate0-clips:begin -->") : before.index("<!-- gate0-clips:end")
    ]
    rows = [row for row in clips.splitlines() if re.match(r"\| `[BVSNT]\d` \|", row)]
    assert len(rows) == 9 and all(re.search(r"`[0-9a-f]{64}`", row) for row in rows)
    for setting in ("temperature 0", "seed 0", '"enable_thinking": false', "-c 4096"):
        assert setting in before
    assert "--parallel 1" in before
    assert gate0.SAMPLING["temperature"] == 0 and gate0.SAMPLING["seed"] == 0
    assert "**42**" in before and "локально, стоимость 0" in before
    assert gate0.rule_sha256(document) == DECISION_RULE_SHA256


def test_criterion_b_pairs_controls_and_transcript_only_requests_go_out_as_the_protocol_says(
    spike, tmp_path: Path, monkeypatch
) -> None:
    bodies_file = tmp_path / "bodies.jsonl"
    monkeypatch.setenv("GATE0_FAKE_BODIES", str(bodies_file))
    synthetic = [pair for pair in spike["manifest"]["pairs"] if pair["target"] is not None]
    by_id = {clip["id"]: clip for clip in spike["manifest"]["clips"]}
    assert len(synthetic) >= 2
    assert all(by_id[pair["variant"]]["synthetic_manipulation"] for pair in synthetic)
    gate0.run_spike("2026-11-18", spike["config"], **spike["kwargs"])

    bodies = [json.loads(line) for line in bodies_file.read_text(encoding="utf-8").splitlines()]
    words = json.loads((spike["out"] / gate0.WORDS_FILE).read_text(encoding="utf-8"))
    prompt = review.load_prompt()
    entries = gate0.plan(spike["manifest"], words, prompt)
    assert len(bodies) == len(entries)
    sent_pcm = set(spike["received"].read_text().split())
    for signal in (gate0.silence(), gate0.pink_noise(), gate0.tone()):
        assert hashlib.sha256(gate0.pcm16(signal)).hexdigest() in sent_pcm
    sent = {}
    for entry, body in zip(entries, bodies, strict=True):
        parts = body["messages"][0]["content"]
        kinds = [part["type"] for part in parts]
        assert kinds == (["input_audio", "text"] if entry["modality"] == "audio" else ["text"])
        assert parts[-1]["text"] == entry["text"]
        sent[entry["variant"], entry["clip"] or entry.get("about")] = parts[-1]["text"]
    for pair in spike["manifest"]["pairs"]:
        clip_id, transcript = pair["base"], words[pair["source"]]["transcript"]
        duration = by_id[clip_id]["duration_s"]
        assert sent["A", clip_id] == gate0.review_prompt(prompt, clip_id, duration)
        assert sent["A-text", clip_id] == gate0.review_prompt(prompt, clip_id, duration, transcript)


def test_criterion_a_document_reports_limitations_provenance_request_count_and_cost() -> None:
    document = (REPO / gate0.DECISION_DOC).read_text(encoding="utf-8")
    limits = " ".join(document[document.index("## 6.") : document.index("## 7.")].split())
    for topic in ("синтетические", "PSOLA", "Один говорящий", "не held-out", "Gate 1", "R11"):
        assert topic in limits, topic
    provenance = " ".join(document[document.index("## 9.") :].split())
    assert "`data/gate0/2026-11-18/`" in provenance and "вне git" in provenance
    for item in ("manifest.json", "run.json", "trace.jsonl", "responses/<n>.json", "verdict.json"):
        assert item in provenance, item
    assert "Запросов к модели — **42**" in provenance
    assert "Стоимость — **локально, 0**" in provenance


def test_criterion_d_document_states_the_rule_verdict_of_the_run_and_doctor_unchanged() -> None:
    document = (REPO / gate0.DECISION_DOC).read_text(encoding="utf-8")
    verdict = " ".join(document[document.index("## 5. Вердикт") : document.index("## 6.")].split())
    assert "**Вердикт: NOT READY**" in verdict
    assert "C: не прошли 6 из 12 пунктов контролей" in verdict
    assert "Предусловия P1–P3 выполнены" in verdict
    assert "`backend` остаётся `none`" in verdict
    assert "permission_required" in document[document.index("## 7.") : document.index("## 8.")]
    assert not re.search(r"(?m)^\*{0,2}Вердикт:?\s*(READY|HELD)", document)
    assert not re.search(r"\*\*Вердикт: (READY|HELD)", document)


def test_criterion_e_raw_outputs_are_gitignored_and_no_personal_path_is_committed() -> None:
    ignored = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", "data/gate0/2026-11-18/attempts/1/run.json"],
        capture_output=True,
        check=False,
    )
    if ignored.returncode > 1:
        pytest.skip("git is unavailable here")
    assert ignored.returncode == 0
    for path in (gate0.DECISION_DOC, Path("src/sound_scout/eval/gate0.py"), Path(__file__)):
        text = (REPO / path).read_text(encoding="utf-8")
        assert not re.search(r"/(home|Users)/|~[/]", text), path


# --- the borealis runtime of #68 (docs/decisions/0010-gate0-borealis.md) -------------------

BOREALIS_DOC = Path("docs/decisions/0010-gate0-borealis.md")


def _cgroup(root: Path, memory_max: str, swap_max: str) -> Path:
    """A fake cgroup v2 tree at the path of this process's own cgroup."""
    line = next(
        line for line in Path("/proc/self/cgroup").read_text().splitlines() if line[:3] == "0::"
    )
    directory = root / line[3:].lstrip("/")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "memory.max").write_text(memory_max + "\n")
    (directory / "memory.swap.max").write_text(swap_max + "\n")
    (directory / "memory.peak").write_text("123456\n")
    (directory / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    return root


def test_borealis_argv_is_local_with_host_port_threads_and_no_hub_id() -> None:
    argv = gate0.borealis_argv(Path("/models/borealis"), 4321, 12)
    assert argv[:3] == [sys.executable, "-m", "sound_scout.eval.gate0_borealis"]
    assert argv[argv.index("--model-dir") + 1] == "/models/borealis"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "4321"
    assert argv[argv.index("--threads") + 1] == "12"
    assert not any(hub in arg for arg in argv for hub in ("Vikhrmodels/", "Qwen", "openai/"))
    assert not gate0.DOWNLOAD_FLAGS.intersection(argv)
    llama = gate0.server_argv(Path("/opt/llama-server"), Path("/models"), 4321, 12)
    assert llama[1:3] == ["-m", "/models/gemma-4-E4B-it-Q8_0.gguf"]


@pytest.mark.skipif(not Path("/proc/self/cgroup").is_file(), reason="needs Linux cgroup v2")
@pytest.mark.parametrize(
    ("memory_max", "swap_max"),
    [("max", "0"), ("20000000001", "0"), ("21474836480", "0"), ("20000000000", "max")],
)
def test_borealis_run_without_the_memory_cap_is_refused_before_any_attempt(
    spike, tmp_path, memory_max, swap_max
) -> None:
    root = _cgroup(tmp_path / "cgroup", memory_max, swap_max)
    with pytest.raises(gate0.Gate0Error, match="MemoryMax=20000000000"):
        gate0.run_spike(
            "2026-11-18", spike["config"], runtime="borealis", cgroup_root=root, **spike["kwargs"]
        )
    assert not (spike["out"] / gate0.ATTEMPTS_DIR).exists()
    assert not spike["marker"].exists()


@pytest.mark.skipif(not Path("/proc/self/cgroup").is_file(), reason="needs Linux cgroup v2")
def test_borealis_weight_mismatch_stops_before_the_server_start(spike, tmp_path) -> None:
    root = _cgroup(tmp_path / "cgroup", "19999997952", "0")
    kwargs = {**spike["kwargs"], "weights": {"model.safetensors": "0" * 64}}
    with pytest.raises(gate0.Gate0Error, match="before the server start"):
        gate0.run_spike(
            "2026-11-18", spike["config"], runtime="borealis", cgroup_root=root, **kwargs
        )
    assert not spike["marker"].exists()
    record = json.loads((spike["out"] / gate0.ATTEMPTS_DIR / "1" / gate0.RUN_FILE).read_text())
    assert record["runtime"] == "borealis" and record["preconditions"]["checks"]["weights"] is False
    assert record["cgroup"]["memory_max"] == "19999997952"


@pytest.mark.skipif(not Path("/proc/self/cgroup").is_file(), reason="needs Linux cgroup v2")
def test_borealis_run_under_the_cap_records_the_cgroup_peak_and_the_same_plan(
    spike, tmp_path, monkeypatch
) -> None:
    root = _cgroup(tmp_path / "cgroup", "19999997952", "0")
    server = spike["kwargs"]["llama_dir"] / gate0.SERVER_BINARY
    seen = {}

    def argv(model_dir, port, threads):
        seen.update(model_dir=model_dir, threads=threads)
        return [str(server), "--port", str(port)]

    monkeypatch.setattr(gate0, "borealis_argv", argv)
    result = gate0.run_spike(
        "2026-11-18", spike["config"], runtime="borealis", cgroup_root=root, **spike["kwargs"]
    )
    attempt = spike["out"] / gate0.ATTEMPTS_DIR / "1"
    record = json.loads((attempt / gate0.RUN_FILE).read_text())
    assert record["runtime"] == "borealis" and seen["model_dir"] == spike["kwargs"]["model_dir"]
    assert record["cgroup"]["memory_peak"] == 123456
    assert record["cgroup"]["memory_events"]["oom_kill"] == 0
    assert record["server_version"].startswith(f"version: borealis-{gate0.BOREALIS_REVISION}")
    assert record["requests_planned"] == 42
    lines = (attempt / gate0.TRACE_FILE).read_text().splitlines()
    assert len(lines) == 43 and result["preconditions"] == {"P1": True, "P2": True, "P3": True}


def test_memory_usage_without_a_cgroup_is_empty_and_llama_runs_need_no_cap() -> None:
    assert gate0.memory_usage({"path": None}) == {"memory_peak": None, "memory_events": None}
    gate0.check_memory_cap({"memory_max": "20000000000", "memory_swap_max": "0"})
    with pytest.raises(gate0.Gate0Error):
        gate0.check_memory_cap({"memory_max": None, "memory_swap_max": None})


def test_unknown_runtime_is_refused(spike) -> None:
    with pytest.raises(gate0.Gate0Error, match="--runtime"):
        gate0.run_spike("2026-11-18", spike["config"], runtime="vllm", **spike["kwargs"])


@pytest.mark.skipif(not Path("/proc/self/cgroup").is_file(), reason="needs Linux /proc")
def test_other_inference_sees_a_running_borealis_server() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", gate0.BOREALIS_MODULE]
    )
    try:
        assert gate0.BOREALIS_MODULE in gate0._other_inference()
    finally:
        child.kill()
        child.wait()


def test_borealis_weights_are_the_table_of_dependencies_section_3_1() -> None:
    dependencies = (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    weights = dependencies.split("### 3.1.", 1)[1].split("\n### 3.2.", 1)[0]
    borealis = weights[weights.index("Borealis-5b-it") :]
    rows = dict(re.findall(r"^\| `([^`]+)` \| [\d ]+ \| `([0-9a-f]{64})` \|", borealis, re.M))
    assert rows == gate0.BOREALIS_WEIGHTS
    assert gate0.BOREALIS_REVISION in borealis


def test_borealis_decision_document_holds_the_same_rule_as_0007() -> None:
    borealis = (REPO / BOREALIS_DOC).read_text(encoding="utf-8")
    gemma = (REPO / gate0.DECISION_DOC).read_text(encoding="utf-8")
    assert gate0.rule_sha256(borealis) == gate0.rule_sha256(gemma) == DECISION_RULE_SHA256
    assert not re.search(r"/(home|Users)/|~[/]", borealis)


QUESTIONS_SHA256_0007 = "34871259fa7ad301b1cd5ee24182664e79c3b1b7439bb1741204038e56cced82"


def test_the_questions_hash_stays_the_one_0007_and_0010_recorded() -> None:
    assert gate0.questions_sha256() == QUESTIONS_SHA256_0007
    for name in (gate0.DECISION_DOC, BOREALIS_DOC):
        assert QUESTIONS_SHA256_0007 in (REPO / name).read_text(encoding="utf-8")
