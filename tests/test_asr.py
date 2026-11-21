"""Local ASR (issue #19): the stage contract with a test double, the hallucination rule, the
immutable raw transcript, the smoke receipt; `slow` tests with the real cached model.

The test double stands in for `asr._load_model` only: it proves the artifacts, statuses and
refusals, not that the tool hears anything. The real-model tests need the weights already in
the local Hugging Face cache (never downloaded here) and, for the recording, the path in
`SOUND_SCOUT_DEV_AUDIO`; without them they are skipped, with the reason."""

import dataclasses
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth
import pytest
from parselmouth.praat import call

from sound_scout import acoustics, asr, channels, doctor, guard, ingest, prepare, schema, smoke
from sound_scout._schema_messages import schema_errors
from sound_scout.asr import (
    INFERENCE_PARAMETERS,
    MODEL_UNAVAILABLE_REASON,
    SUSPECT_HALLUCINATION,
    AsrError,
    Transcription,
    flag_segments,
    hallucination_signals,
    read_raw,
    transcribe,
    write_corrected,
    write_smoke_receipt,
)
from sound_scout.config import Asr, Budgets, Config
from sound_scout.eval import corpus, gate0, synthetic
from sound_scout.runs import open_run

SR = asr.SAMPLE_RATE
IDENTITY = {"audio": None, "model": None, "preprocessing": None, "ranges": None}
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
REPO = "o/n"
FAKE_IMPLEMENTATION = f"{__name__}.FakeModel"


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class FakeSegment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[FakeWord] | None
    temperature: float | None = 0.0


@dataclass
class FakeInfo:
    language: str = "ru"
    language_probability: float = 1.0
    duration: float = 1.0
    duration_after_vad: float = 1.0
    all_language_probs: None = None
    transcription_options: dict | None = None
    vad_options: None = None


class FakeModel:
    """Returns the scripted segments; records what it was asked to do."""

    def __init__(self, segments: list[FakeSegment]) -> None:
        self.segments = segments
        self.calls: list[dict] = []

    def transcribe(self, audio: np.ndarray, **parameters):
        self.calls.append({"frames": len(audio), "dtype": str(audio.dtype), **parameters})
        return iter(self.segments), FakeInfo(duration=len(audio) / SR)


def word(start: float, end: float, text: str, probability: float = 0.9) -> FakeWord:
    return FakeWord(start, end, text, probability)


def segment(
    start: float,
    end: float,
    text: str,
    words: list[FakeWord] | None = None,
    *,
    avg_logprob: float = -0.2,
    compression_ratio: float = 1.0,
    no_speech_prob: float = 0.01,
) -> FakeSegment:
    if words is None:
        words = [word(start, end, text)]
    return FakeSegment(
        1, 0, start, end, text, [1, 2], avg_logprob, compression_ratio, no_speech_prob, words
    )


def speech_like(seconds: float = 2.0) -> np.ndarray:
    """Formant-synthesised vowels: the speech detector finds speech in them, unlike a tone."""
    return smoke.speech_like_control(seconds, SR)


def tone(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def spans(*intervals: tuple[float, float]) -> list[dict]:
    return [{"start": start, "end": end} for start, end in intervals]


def silence(seconds: float = 5.0) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def restore_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the offline variables so `transcribe` sets them, and have monkeypatch put the
    original state back after the test (`delenv` alone records nothing for an absent name)."""
    for name in guard.OFFLINE_ENV:
        monkeypatch.setenv(name, "unset-by-test")
        monkeypatch.delenv(name)


def fake_snapshot(hf_cache: Path, revision: str) -> Path:
    snapshot = hf_cache / f"models--{REPO.replace('/', '--')}" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"not a model")
    (snapshot / "config.json").write_bytes(b"{}")
    return snapshot


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A config whose ASR model is a fake snapshot in a fake HF cache, so the probe passes."""
    base = tmp_path.resolve()
    recordings, data_root, hf_cache = base / "recordings", base / "data", base / "hf"
    for directory in (recordings, data_root):
        directory.mkdir()
    fake_snapshot(hf_cache, REVISION)
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))
    restore_offline_env(monkeypatch)
    return Config(allowed_roots=(recordings,), data_root=data_root, asr=Asr(model=REPO))


@pytest.fixture
def run(config: Config):
    return open_run("req", IDENTITY, config.data_root)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch):
    """Install a `FakeModel` as the loaded model and return a setter for its segments."""
    holder: dict[str, FakeModel] = {}

    def install(segments: list[FakeSegment]) -> FakeModel:
        holder["model"] = FakeModel(segments)
        monkeypatch.setattr(asr, "_load_model", lambda snapshot_dir: holder["model"])
        return holder["model"]

    return install


def never_load(snapshot_dir: Path):
    raise AssertionError(f"the model must not be loaded, asked for {snapshot_dir}")


# --- stage contract with the test double ---------------------------------------------------


def test_transcribe_writes_raw_and_model_receipt_with_source_timecodes(
    config: Config, run, fake_model
) -> None:
    audio = speech_like(2.0)
    model = fake_model(
        [
            segment(0.0, 1.0, " Привет", [word(0.0, 0.5, " При", 0.8), word(0.5, 1.0, "вет", 0.7)]),
            segment(1.0, 2.0, " мир", [word(1.0, 2.0, " мир", 0.95)]),
        ]
    )

    result = transcribe(audio, run, config)

    assert result.status == "completed" and result.reason is None
    assert result.stage == {
        "requested": True,
        "status": "completed",
        "reason": None,
        "elapsed_ms": result.elapsed_ms,
    }
    assert result.flags == ()
    assert model.calls == [{"frames": len(audio), "dtype": "float32", **INFERENCE_PARAMETERS}]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert result.raw_path == run.path / "asr" / "raw.json"
    assert result.receipt_path == run.path / "asr" / "model-receipt.json"
    assert sorted(p.name for p in (run.path / "asr").iterdir()) == [
        "model-receipt.json",
        "raw.json",
    ]
    assert stat.S_IMODE((run.path / "asr").stat().st_mode) == 0o700
    assert stat.S_IMODE(result.raw_path.stat().st_mode) == 0o400

    raw = json.loads(result.raw_path.read_text(encoding="utf-8"))
    assert raw == result.raw
    assert raw["kind"] == "transcript_raw" and raw["run_id"] == run.run_id
    assert raw["model"] == {
        "repo": REPO,
        "revision": REVISION,
        "implementation": FAKE_IMPLEMENTATION,
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": asr.CPU_THREADS,
        "num_workers": 1,
        "local_files_only": True,
    }
    assert raw["inference"] == INFERENCE_PARAMETERS
    assert raw["inference"]["language"] == "ru"
    assert raw["audio"] == {
        "sample_rate": SR,
        "frames": len(audio),
        "duration_s": 2.0,
        "sha256": __import__("hashlib").sha256(audio.tobytes()).hexdigest(),
        "timescale": "source",
        "offset_s": 0.0,
    }
    assert raw["text"] == "Привет мир"
    assert [s["text"] for s in raw["segments"]] == [" Привет", " мир"]
    assert raw["segments"][0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": " При", "probability": 0.8},
        {"start": 0.5, "end": 1.0, "word": "вет", "probability": 0.7},
    ]
    assert (
        raw["segments"][1]["avg_logprob"] == -0.2 and raw["segments"][1]["no_speech_prob"] == 0.01
    )
    assert all(s[SUSPECT_HALLUCINATION] is False for s in raw["segments"])
    assert raw["flags"] == []
    assert raw["info"]["language"] == "ru" and raw["info"]["duration"] == 2.0
    assert set(raw["timing"]) == {"model_load_ms", "inference_ms"}
    assert raw["versions"]["faster-whisper"] == doctor.installed_version("faster-whisper")
    assert "ctranslate2" in raw["versions"] and raw["versions"]["sound-scout"]

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert (receipt["repo"], receipt["revision"]) == (REPO, REVISION)
    assert receipt["files"] == [
        {
            "name": "config.json",
            "bytes": 2,
            "sha256": __import__("hashlib").sha256(b"{}").hexdigest(),
        },
        {
            "name": "model.bin",
            "bytes": 11,
            "sha256": __import__("hashlib").sha256(b"not a model").hexdigest(),
        },
    ]
    assert receipt["versions"] == raw["versions"]
    assert (receipt["device"], receipt["compute_type"], receipt["local_files_only"]) == (
        "cpu",
        "int8",
        True,
    )
    assert receipt["offline_env"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    assert receipt["audio_uploaded"] is False and receipt["paid_inference_called"] is False
    assert not any("/" in str(v) for v in receipt["files"][0].values())
    assert str(run.path) not in result.receipt_path.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_transcribe_takes_the_mono16k_derivative_and_keeps_source_timecodes(
    config: Config, run, fake_model
) -> None:
    """A 48 kHz source with a tone from 1.25 s: the derivative is what the model gets, and a
    segment the model places at the tone's onset in that array lands at 1.25 s of the source."""
    rate, duration_s, onset_s = 48_000, 2.5, 1.25
    t = np.arange(int(duration_s * rate)) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    mono = np.where(t >= onset_s, tone, np.float32(0.0)).astype(np.float32)
    derivative = prepare.mono16k(mono, rate, run, config)
    heard_onset = int(np.argmax(np.abs(derivative.samples) > 0.05)) / SR
    model = fake_model([segment(heard_onset, duration_s, " тон")])

    result = transcribe(derivative.samples, run, config)

    assert result.status == "completed", result.reason
    assert model.calls[0]["frames"] == len(derivative.samples) == derivative.record["frames"]
    raw = json.loads(result.raw_path.read_text(encoding="utf-8"))
    assert raw["audio"]["sha256"] == derivative.sha256
    assert raw["audio"]["timescale"] == "source" and raw["audio"]["offset_s"] == 0.0
    assert abs(raw["audio"]["duration_s"] - duration_s) < 0.001
    assert abs(raw["segments"][0]["start"] - onset_s) < 0.001
    assert raw["segments"][0]["end"] == duration_s
    assert sorted(p.name for p in (run.path / "work").iterdir()) == ["mono16k.f32", "mono16k.json"]


def test_model_constructor_gets_the_snapshot_dir_cpu_int8_and_local_files_only(
    config: Config, run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real `_load_model` with `faster_whisper.WhisperModel` replaced by a recorder."""
    import faster_whisper

    constructed: list[tuple[tuple, dict]] = []
    model = FakeModel([segment(0.0, 1.0, " слово")])

    class RecordingWhisperModel:
        def __init__(self, *args, **kwargs) -> None:
            constructed.append((args, kwargs))

        def transcribe(self, audio, **parameters):
            return model.transcribe(audio, **parameters)

    monkeypatch.setattr(faster_whisper, "WhisperModel", RecordingWhisperModel)

    result = transcribe(speech_like(), run, config)

    assert result.status == "completed", result.reason
    assert constructed == [
        (
            (str(doctor.snapshot_dir(REPO, REVISION)),),
            {
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": asr.CPU_THREADS,
                "num_workers": 1,
                "local_files_only": True,
            },
        )
    ]
    opened = Path(constructed[0][0][0])
    assert opened.is_dir() and (opened / "model.bin").is_file()
    assert model.calls[0]["language"] == "ru" and model.calls[0]["task"] == "transcribe"
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_model_loader_switches_onnxruntime_telemetry_off_before_the_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """faster-whisper's VAD filter loads onnxruntime: telemetry is off by the time it can."""
    import faster_whisper

    seen: list[str | None] = []

    class RecordingWhisperModel:
        def __init__(self, *args, **kwargs) -> None:
            seen.append(os.environ.get("ORT_DISABLE_TELEMETRY"))

    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr(faster_whisper, "WhisperModel", RecordingWhisperModel)

    asr._load_model(tmp_path)

    assert seen == ["1"]


def test_missing_weights_under_hf_hub_offline_reach_neither_model_package_nor_network(
    config: Config, run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `_load_model` patched: the model package, the hub client and sockets stay idle."""
    import socket

    import faster_whisper
    import faster_whisper.utils
    import huggingface_hub

    empty_home = tmp_path / "empty-hf-home"
    empty_home.mkdir()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(empty_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def forbidden(what: str):
        def _fail(*args, **kwargs):
            pytest.fail(f"{what} was reached: {args} {kwargs}")

        return _fail

    monkeypatch.setattr(faster_whisper, "WhisperModel", forbidden("faster_whisper.WhisperModel"))
    monkeypatch.setattr(faster_whisper.utils, "download_model", forbidden("download_model"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden("snapshot_download"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden("hf_hub_download"))
    monkeypatch.setattr(socket.socket, "connect", forbidden("socket.connect"))
    monkeypatch.setattr(socket, "getaddrinfo", forbidden("socket.getaddrinfo"))
    monkeypatch.setattr(socket, "create_connection", forbidden("socket.create_connection"))

    result = transcribe(speech_like(), run, config)

    assert result.stage == {
        "requested": True,
        "status": "unavailable",
        "reason": MODEL_UNAVAILABLE_REASON,
        "elapsed_ms": result.elapsed_ms,
    }
    assert not (run.path / "asr").exists()
    assert list(empty_home.rglob("*")) == []
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_missing_weights_are_unavailable_without_any_load_or_download(
    config: Config, run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_home = tmp_path / "empty-hf-home"
    empty_home.mkdir()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(empty_home))
    monkeypatch.setattr(asr, "_load_model", never_load)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("download attempted")
    )
    before = sorted(p.name for p in run.path.iterdir())

    result = transcribe(speech_like(), run, config)

    assert result.status == "unavailable"
    assert result.reason == MODEL_UNAVAILABLE_REASON
    assert result.reason == "модель не установлена; загрузка требует разрешения владельца"
    assert (result.raw_path, result.receipt_path, result.raw, result.flags) == (
        None,
        None,
        None,
        (),
    )
    assert result.stage["status"] == "unavailable" and result.stage["reason"] == result.reason
    assert sorted(p.name for p in run.path.iterdir()) == before
    assert not (run.path / "asr").exists()
    assert list(empty_home.rglob("*")) == []
    assert doctor.component_status("asr", config, config.data_root)["status"] == "unavailable"


def test_other_snapshot_problems_keep_the_reason_of_the_probe(
    config: Config, run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights installed twice and no `asr.revision`: the stage says so, not «не установлена»."""
    monkeypatch.setattr(asr, "_load_model", never_load)
    fake_snapshot(Path(os.environ["HF_HUB_CACHE"]), OTHER_REVISION)

    result = transcribe(speech_like(), run, config)

    assert result.status == "unavailable"
    assert result.reason == f"several snapshots of {REPO} in the cache and no asr.revision set"
    assert result.reason != MODEL_UNAVAILABLE_REASON
    assert not (run.path / "asr").exists()

    missing = Config(
        allowed_roots=config.allowed_roots,
        data_root=config.data_root,
        asr=Asr(model=REPO, revision="fedcba9876"),
    )
    gone = transcribe(speech_like(), run, missing)
    assert gone.status == "unavailable"
    assert gone.reason == "asr.revision fedcba9876 is not in the local cache"


def test_missing_package_is_unavailable_with_the_package_named(
    config: Config, run, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asr, "_load_model", never_load)
    monkeypatch.setattr(
        doctor, "installed_version", lambda name: None if name == "ctranslate2" else "1.0"
    )

    result = transcribe(speech_like(), run, config)

    assert result.status == "unavailable"
    assert result.reason == "package ctranslate2 is not installed"
    assert not (run.path / "asr").exists()


def test_model_error_is_a_failed_stage_with_the_reason(
    config: Config, run, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(snapshot_dir: Path):
        raise RuntimeError("Unable to open file 'model.bin'")

    monkeypatch.setattr(asr, "_load_model", broken)

    result = transcribe(speech_like(), run, config)

    assert result.status == "failed"
    assert result.reason == "RuntimeError: Unable to open file 'model.bin'"
    assert result.raw is None and not (run.path / "asr").exists()


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.zeros((SR, 1), dtype=np.float32), "one-dimensional float32"),
        (np.zeros(SR, dtype=np.float64), "one-dimensional float32"),
        (np.zeros(0, dtype=np.float32), "empty"),
        (np.array([0.0, np.nan], dtype=np.float32), "finite"),
        (np.zeros(3 * SR + 1, dtype=np.float32), "limit is 48000 (3 s at 16000 Hz)"),
    ],
    ids=["2d", "float64", "empty", "nan", "too_long"],
)
def test_bad_audio_is_the_callers_error_before_any_load(
    config: Config, run, monkeypatch: pytest.MonkeyPatch, audio: np.ndarray, message: str
) -> None:
    monkeypatch.setattr(asr, "_load_model", never_load)
    short = Config(
        allowed_roots=config.allowed_roots,
        data_root=config.data_root,
        budgets=Budgets(max_duration_seconds=3),
        asr=config.asr,
    )

    with pytest.raises(AsrError) as error:
        transcribe(audio, run, short)

    assert error.value.code == "bad_audio" and message in error.value.message
    assert not (run.path / "asr").exists()


def test_empty_transcript_is_completed_with_no_segments_and_no_flags(
    config: Config, run, fake_model
) -> None:
    fake_model([])

    result = transcribe(silence(), run, config)

    assert result.status == "completed"
    assert result.raw["segments"] == [] and result.raw["text"] == ""
    assert result.raw["flags"] == [] and result.flags == ()


# --- the hallucination rule -----------------------------------------------------------------


def test_rule_flags_text_over_digital_silence_even_with_high_probabilities() -> None:
    confident = {
        "start": 0.0,
        "end": 4.62,
        "text": " Продолжение следует...",
        "avg_logprob": -0.078,
        "compression_ratio": 0.83,
        "no_speech_prob": 1.1e-10,
        "words": [{"start": 0.0, "end": 1.34, "word": " Продолжение", "probability": 0.886}],
    }

    assert hallucination_signals(confident, silence(5.0), []) == ["silent_span", "no_vad_speech"]
    flagged = flag_segments([confident], silence(5.0), [])
    assert flagged[0][SUSPECT_HALLUCINATION] is True
    assert flagged[0]["hallucination_signals"] == ["silent_span", "no_vad_speech"]
    assert flagged[0]["text"] == confident["text"], "the rule flags, it does not rewrite"


def test_rule_uses_the_signal_under_the_segment_not_the_whole_file() -> None:
    audio = np.concatenate([tone(2.0), silence(3.0)])
    speech = spans((0.0, 2.0))

    assert hallucination_signals({"start": 0.0, "end": 2.0, "text": " речь"}, audio, speech) == []
    assert hallucination_signals({"start": 2.5, "end": 4.9, "text": " эхо"}, audio, speech) == [
        "silent_span",
        "no_vad_speech",
    ]
    assert hallucination_signals({"start": 1.5, "end": 4.9, "text": " край"}, audio, speech) == []


def test_rule_flags_whispers_own_signals_and_ignores_empty_text() -> None:
    audio, speech = tone(3.0), spans((0.0, 3.0))
    base = {"start": 0.0, "end": 3.0, "text": " слово"}

    def signals(segment: dict, samples: np.ndarray = audio, found: list = speech) -> list[str]:
        return hallucination_signals(segment, samples, found)

    assert signals({**base, "no_speech_prob": 0.61}) == ["no_speech_prob"]
    assert signals({**base, "avg_logprob": -1.01}) == ["low_avg_logprob"]
    assert signals({**base, "compression_ratio": 2.5}) == ["high_compression_ratio"]
    assert signals(
        {**base, "no_speech_prob": 0.9, "avg_logprob": -2.0, "compression_ratio": 3.0},
        silence(3.0),
        [],
    ) == list(asr.HALLUCINATION_SIGNALS)
    assert signals({**base, "no_speech_prob": 0.6}) == []
    assert signals({**base, "text": "   "}, silence(3.0), []) == []
    assert signals({"start": 0.0, "end": 1.0}, silence(3.0), []) == []
    assert signals({**base, "end": 0.0}) == ["silent_span", "no_vad_speech"]


def test_rule_flags_text_the_speech_detector_does_not_cover() -> None:
    """Noise or a tone: a signal well above silence, Whisper's own signals normal (#135)."""
    audio = tone(4.0)
    base = {"start": 1.0, "end": 3.0, "text": " уверенно", "no_speech_prob": 0.0}

    assert hallucination_signals(base, audio, []) == ["no_vad_speech"]
    assert hallucination_signals(base, audio, spans((3.0, 4.0))) == ["no_vad_speech"]
    assert hallucination_signals(base, audio, spans((0.0, 1.09))) == ["no_vad_speech"]
    assert hallucination_signals(base, audio, spans((0.0, 1.1))) == []
    assert hallucination_signals(base, audio, spans((1.0, 1.06), (2.95, 4.0))) == []
    assert asr.MIN_VAD_SPEECH_S == 0.1


def test_quiet_but_not_silent_speech_is_not_flagged_by_the_signal_rule() -> None:
    quiet = (tone(2.0) * 0.01).astype(np.float32)  # about -53 dBFS RMS
    speech = spans((0.0, 2.0))

    assert hallucination_signals({"start": 0.0, "end": 2.0, "text": " тихо"}, quiet, speech) == []
    very_quiet = (tone(2.0) * 0.001).astype(np.float32)  # about -73 dBFS RMS
    assert hallucination_signals(
        {"start": 0.0, "end": 2.0, "text": " тихо"}, very_quiet, speech
    ) == ["silent_span"]


def test_transcribe_marks_silence_segments_with_the_rule(config: Config, run, fake_model) -> None:
    fake_model(
        [segment(0.0, 4.6, " Продолжение следует...", [word(0.0, 4.6, " Продолжение", 0.9)])]
    )

    result = transcribe(silence(5.0), run, config)

    assert result.status == "completed"
    assert result.flags == (SUSPECT_HALLUCINATION,)
    assert result.raw["flags"] == [SUSPECT_HALLUCINATION]
    assert result.raw["segments"][0][SUSPECT_HALLUCINATION] is True
    assert result.raw["segments"][0]["hallucination_signals"] == ["silent_span", "no_vad_speech"]
    assert result.raw["text"] == "Продолжение следует..."
    assert result.raw["hallucination_rule"] == asr.HALLUCINATION_RULE
    assert result.raw["speech_spans"] == []


def test_transcribe_without_the_speech_detector_is_unavailable_and_trusts_nothing(
    config: Config, run, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_detector(mono: np.ndarray, rate: int) -> list[dict]:
        raise acoustics.VadUnavailable("Silero VAD asset not found in faster_whisper")

    monkeypatch.setattr(acoustics, "speech_activity", no_detector)
    monkeypatch.setattr(asr, "_load_model", never_load)

    result = transcribe(speech_like(), run, config)

    assert result.status == "unavailable"
    assert result.reason == (
        f"{asr.VAD_UNAVAILABLE_REASON}: Silero VAD asset not found in faster_whisper"
    )
    assert result.raw is None and result.flags == ()
    assert not (run.path / "asr" / "raw.json").exists()


def test_transcribe_checks_text_against_the_detector_on_the_same_array(
    config: Config, run, fake_model
) -> None:
    """Text over a tone gets no trust; text over formant vowels keeps it (#135)."""
    audio = np.concatenate([speech_like(2.0), tone(3.0)])
    fake_model(
        [
            segment(0.0, 2.0, " речь", [word(0.0, 2.0, " речь", 0.9)]),
            segment(2.5, 4.9, " тон", [word(2.5, 4.9, " тон", 0.95)]),
        ]
    )

    result = transcribe(audio, run, config)

    trusted, flagged = asr.trusted_segments(result.raw)
    assert [s["text"] for s in trusted] == [" речь"]
    assert [s["hallucination_signals"] for s in flagged] == [["no_vad_speech"]]
    assert result.raw["speech_spans"] == acoustics.speech_activity(audio, SR)
    assert result.raw["speech_spans"][0]["start"] < 0.1


# --- the one rule of trust: trusted_segments / suspect_segments (docs/decisions/0005) ---------


def test_confident_phrase_over_silence_is_flagged_and_quiet_speech_is_trusted(
    config: Config, run, fake_model
) -> None:
    quiet = (speech_like(2.0) * 0.03).astype(np.float32)  # about -51 dBFS RMS, above the rule
    audio = np.concatenate([quiet, silence(3.0)])
    fake_model(
        [
            segment(0.0, 2.0, " тихая речь", [word(0.0, 2.0, " тихая", 0.4)]),
            segment(2.2, 2.2, "  ", []),
            segment(2.5, 4.9, " Уверенная фраза", [word(2.5, 4.9, " Уверенная", 0.95)]),
        ]
    )
    result = transcribe(audio, run, config)
    raw_path = run.path / "asr" / "raw.json"
    raw_bytes = raw_path.read_bytes()

    trusted, flagged = asr.trusted_segments(result.raw)

    assert [s["text"] for s in trusted] == [" тихая речь"]
    assert [s["text"] for s in flagged] == [" Уверенная фраза"]
    assert flagged[0]["hallucination_signals"] == ["silent_span", "no_vad_speech"]
    assert trusted[0] is result.raw["segments"][0] and flagged[0] is result.raw["segments"][2]
    assert asr.suspect_segments(result.raw, len(audio) / SR) == [
        {
            "start": 2.5,
            "end": 4.9,
            "signals": ["silent_span", "no_vad_speech"],
            "certainty": "uncertain",
        }
    ]
    assert raw_path.read_bytes() == raw_bytes, "the raw transcript is not changed (R05)"
    assert read_raw(run) == result.raw


def test_trust_rests_on_the_flag_not_on_word_probabilities() -> None:
    confident = {"text": " фраза", "suspect_hallucination": True, "words": [{"probability": 0.99}]}
    unsure = {"text": " слово", "suspect_hallucination": False, "words": [{"probability": 0.05}]}

    assert asr.trusted_segments({"segments": [confident, unsure]}) == ([unsure], [confident])
    assert asr.trusted_segments({"segments": []}) == ([], [])


@pytest.mark.parametrize(
    ("raw", "path"),
    [
        ({}, "segments must be a list"),
        ({"segments": {"0": "секрет"}}, "segments must be a list"),
        ({"segments": ["секрет"]}, "segments[0] must be an object"),
        ({"segments": [{"text": ["секрет"], "suspect_hallucination": False}]}, "segments[0].text"),
        ({"segments": [{"text": " секрет"}]}, "segments[0].suspect_hallucination"),
        (
            {"segments": [{"text": " секрет", "suspect_hallucination": "false"}]},
            "segments[0].suspect_hallucination must be a boolean",
        ),
        (
            {"segments": [{"text": "", "suspect_hallucination": False}, {"text": " секрет"}]},
            "segments[1].suspect_hallucination",
        ),
    ],
)
def test_a_missing_or_malformed_flag_is_an_error_naming_the_path_not_the_text(
    raw: dict, path: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        asr.trusted_segments(raw)
    assert path in str(excinfo.value)
    assert "секрет" not in str(excinfo.value)
    with pytest.raises(ValueError):
        asr.suspect_segments(raw, 5.0)


def test_suspect_segments_are_clamped_to_the_source_and_carry_no_text() -> None:
    def flagged(start: object, end: object, signals: object = ("silent_span",)) -> dict:
        return {
            "start": start,
            "end": end,
            "text": " секрет",
            "words": [{"word": " секрет"}],
            "hallucination_signals": list(signals) if isinstance(signals, tuple) else signals,
            "suspect_hallucination": True,
        }

    raw = {"segments": [flagged(-0.2, 1.0), flagged(4.9999999, 5.3), flagged(5.4, 5.6)]}
    items = asr.suspect_segments(raw, 5.0)

    assert items == [
        {"start": 0.0, "end": 1.0, "signals": ["silent_span"], "certainty": "uncertain"},
        {"start": 5.0, "end": 5.0, "signals": ["silent_span"], "certainty": "uncertain"},
        {"start": 5.0, "end": 5.0, "signals": ["silent_span"], "certainty": "uncertain"},
    ]
    assert "секрет" not in json.dumps(items, ensure_ascii=False)
    for bad, fragment in (
        (flagged("0", 1.0), "segments[0]: start and end must be numbers"),
        (flagged(0.0, True), "segments[0]: start and end must be numbers"),
        (flagged(0.0, 1.0, []), "segments[0].hallucination_signals"),
        (flagged(0.0, 1.0, ("echo",)), "segments[0].hallucination_signals"),
        (flagged(0.0, 1.0, "silent_span"), "segments[0].hallucination_signals"),
    ):
        with pytest.raises(ValueError, match=re.escape(fragment)):
            asr.suspect_segments({"segments": [bad]}, 5.0)


def test_signal_names_are_one_list_from_the_rule_to_the_contract() -> None:
    item = schema.SCHEMA["$defs"]["suspect_segment"]["properties"]["signals"]["items"]
    assert tuple(item["enum"]) == asr.HALLUCINATION_SIGNALS
    every = {"no_speech_prob": 0.9, "avg_logprob": -2.0, "compression_ratio": 3.0}
    signals = hallucination_signals(
        {"start": 0.0, "end": 1.0, "text": " x", **every}, silence(), []
    )
    assert tuple(signals) == asr.HALLUCINATION_SIGNALS


# --- the raw transcript is immutable; corrections are a separate artifact --------------------


def test_raw_transcript_cannot_be_rewritten_through_the_module(
    config: Config, run, fake_model
) -> None:
    fake_model([segment(0.0, 1.0, " раз")])
    first = transcribe(speech_like(), run, config)
    raw_bytes = first.raw_path.read_bytes()
    fake_model([segment(0.0, 1.0, " два")])

    with pytest.raises(AsrError) as error:
        transcribe(speech_like(), run, config)

    assert error.value.code == "transcript_exists"
    assert first.raw_path.read_bytes() == raw_bytes
    assert read_raw(run)["text"] == "раз"
    with pytest.raises(PermissionError):
        first.raw_path.write_text("x", encoding="utf-8")
    assert not hasattr(asr, "write_raw") and not hasattr(asr, "rewrite_raw")


def test_corrections_live_in_their_own_artifact_referencing_raw(
    config: Config, run, fake_model
) -> None:
    fake_model([segment(0.0, 1.0, " превет")])
    result = transcribe(speech_like(), run, config)
    raw_bytes = result.raw_path.read_bytes()
    correction = {
        "start": 0.0,
        "end": 1.0,
        "original": "превет",
        "corrected": "привет",
        "source": "human_verified",
        "evidence_refs": [{"type": "human_label", "ref": "label-1", "sha256": "ab" * 32}],
        "certainty": "clear",
        "note": None,
    }

    path = write_corrected(run, config, [correction])

    assert path == run.path / "transcript" / "corrected.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["kind"] == "transcript_corrections" and document["run_id"] == run.run_id
    assert document["raw"] == {"path": "asr/raw.json", "sha256": result.raw_sha256}
    assert document["corrections"] == [correction]
    assert result.raw_path.read_bytes() == raw_bytes
    assert read_raw(run)["text"] == "превет"
    with pytest.raises(AsrError) as error:
        write_corrected(run, config, [correction])
    assert error.value.code == "corrections_exist"


def test_corrections_need_a_raw_transcript_and_the_contract_shape(config: Config, run) -> None:
    good = {
        "start": 0.0,
        "end": 1.0,
        "original": "а",
        "corrected": "б",
        "source": "transcript",
        "evidence_refs": [{"type": "transcript", "ref": "raw", "sha256": "ab" * 32}],
        "certainty": "uncertain",
        "note": "x",
    }
    with pytest.raises(AsrError) as missing:
        write_corrected(run, config, [good])
    assert missing.value.code == "transcript_missing"
    with pytest.raises(AsrError) as missing_raw:
        read_raw(run)
    assert missing_raw.value.code == "transcript_missing"

    (run.path / "asr").mkdir()
    (run.path / "asr" / "raw.json").write_text("{}", encoding="utf-8")
    for bad, where in (
        ({**good, "source": "model_said_so"}, "source"),
        ({k: v for k, v in good.items() if k != "certainty"}, "certainty"),
        ({**good, "evidence_refs": []}, "evidence_refs"),
        ({**good, "start": 1.0, "end": 1.0}, "start must be less than end"),
        ({**good, "extra": 1}, "corrections[1]: Additional properties are not allowed"),
    ):
        with pytest.raises(AsrError) as error:
            write_corrected(run, config, [good, bad])
        assert error.value.code == "bad_correction" and where in error.value.message, where
    assert not (run.path / "transcript").exists()


CORRECTION_TEXT = "SECRET-слово-из-записи"
GOOD_CORRECTION = {
    "start": 0.0,
    "end": 1.0,
    "original": "а",
    "corrected": "б",
    "source": "transcript",
    "evidence_refs": [{"type": "transcript", "ref": "raw", "sha256": "ab" * 32}],
    "certainty": "uncertain",
    "note": None,
}


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (
            {**GOOD_CORRECTION, "original": [CORRECTION_TEXT]},
            "corrections[1].original: must be of type string",
        ),
        (
            {**GOOD_CORRECTION, "corrected": {"text": CORRECTION_TEXT}},
            "corrections[1].corrected: must be of type string",
        ),
        (
            {**GOOD_CORRECTION, "certainty": CORRECTION_TEXT},
            "corrections[1].certainty: must be one of ['clear', 'uncertain']",
        ),
        (
            {**GOOD_CORRECTION, "start": CORRECTION_TEXT},
            "corrections[1].start: must be of type number",
        ),
        (
            {**GOOD_CORRECTION, "note": [CORRECTION_TEXT]},
            "corrections[1].note: must be of type ['string', 'null']",
        ),
        ({**GOOD_CORRECTION, CORRECTION_TEXT: 1}, "corrections[1]: Additional properties are not"),
        (CORRECTION_TEXT, "corrections[1]: must be of type object"),
    ],
)
def test_bad_correction_names_path_and_rule_without_the_transcript_text(
    config: Config, run, bad: object, error: str
) -> None:
    (run.path / "asr").mkdir()
    (run.path / "asr" / "raw.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AsrError) as refused:
        write_corrected(run, config, [GOOD_CORRECTION, bad])
    assert refused.value.code == "bad_correction"
    assert refused.value.message.startswith(error), refused.value.message
    assert CORRECTION_TEXT not in refused.value.message
    assert CORRECTION_TEXT not in str(refused.value)
    assert not (run.path / "transcript" / "corrected.json").exists()


# --- the smoke receipt ------------------------------------------------------------------------


def test_smoke_receipt_is_refused_without_a_completed_unflagged_transcription(
    config: Config, run, fake_model
) -> None:
    for outcome in (
        Transcription("unavailable", MODEL_UNAVAILABLE_REASON, 1),
        Transcription("failed", "RuntimeError: x", 1),
    ):
        with pytest.raises(doctor.ReceiptRejected, match=outcome.status):
            write_smoke_receipt(outcome, config, test="t", input_label="silence")

    fake_model([segment(0.0, 4.6, " Продолжение следует...")])
    flagged = transcribe(silence(5.0), run, config)
    with pytest.raises(doctor.ReceiptRejected, match=SUSPECT_HALLUCINATION):
        write_smoke_receipt(flagged, config, test="t", input_label="silence")

    other = open_run("other", IDENTITY, config.data_root)
    fake_model([])
    empty = transcribe(speech_like(), other, config)
    with pytest.raises(doctor.ReceiptRejected, match="no words"):
        write_smoke_receipt(empty, config, test="t", input_label="tone")

    third = open_run("third", IDENTITY, config.data_root)
    fake_model([segment(0.0, 1.0, " слово")])
    completed = transcribe(speech_like(), third, config)
    other_model = Config(
        allowed_roots=config.allowed_roots, data_root=config.data_root, asr=Asr(model="x/y")
    )
    with pytest.raises(doctor.ReceiptRejected, match="configured"):
        write_smoke_receipt(completed, other_model, test="t", input_label="tone")
    other_revision = claimed_real(completed, revision=OTHER_REVISION)
    with pytest.raises(doctor.ReceiptRejected, match=f"{OTHER_REVISION}.*configured.*{REVISION}"):
        write_smoke_receipt(other_revision, config, test="t", input_label="tone")

    assert not (config.data_root / "receipts").exists()
    assert doctor.component_status("asr", config, config.data_root)["status"] == "unvalidated"


def test_smoke_receipt_is_refused_for_a_substituted_model(config: Config, run, fake_model) -> None:
    """A completed, unflagged stage from the test double never makes doctor `ready` for asr."""
    fake_model([segment(0.0, 1.0, " слово", [word(0.0, 1.0, " слово", 0.99)])])
    completed = transcribe(speech_like(), run, config)
    assert completed.status == "completed" and completed.flags == ()
    assert completed.raw["model"]["implementation"] == FAKE_IMPLEMENTATION

    with pytest.raises(doctor.ReceiptRejected, match="FakeModel.*substituted") as error:
        write_smoke_receipt(completed, config, test="t", input_label="tone")

    assert asr.MODEL_IMPLEMENTATION in str(error.value)
    assert not (config.data_root / "receipts").exists()
    assert doctor.component_status("asr", config, config.data_root)["status"] == "unvalidated"


def claimed_real(transcription: Transcription, **model_fields: object) -> Transcription:
    """The transcription with `raw["model"]` rewritten to claim the real model's class.

    A forged record for contract tests of the writer only; it proves the shape of the receipt,
    never that the component works (the slow dev-recording test does that)."""
    raw = transcription.raw
    model = {**raw["model"], "implementation": asr.MODEL_IMPLEMENTATION, **model_fields}
    return dataclasses.replace(transcription, raw={**raw, "model": model})


def test_smoke_receipt_record_carries_counts_and_labels_not_text_or_paths(
    config: Config, run, fake_model
) -> None:
    fake_model(
        [segment(0.0, 1.0, " секрет", [word(0.0, 0.5, " сек", 0.8), word(0.5, 1.0, "рет", 0.6)])]
    )
    completed = claimed_real(transcribe(speech_like(), run, config))

    path = write_smoke_receipt(
        completed, config, test="tests/test_asr.py::fake", input_label="synthetic tone, 2 s"
    )

    receipt = json.loads(path.read_text(encoding="utf-8"))
    smoke = receipt["smoke"]
    assert "секрет" not in path.read_text(encoding="utf-8")
    assert str(run.path) not in path.read_text(encoding="utf-8")
    assert smoke["test"] == "tests/test_asr.py::fake"
    assert smoke["input"] == "synthetic tone, 2 s"
    assert smoke["implementation"] == asr.MODEL_IMPLEMENTATION
    assert (smoke["segments"], smoke["words"]) == (1, 2)
    assert smoke["mean_word_probability"] == pytest.approx(0.7)
    assert smoke["min_word_probability"] == 0.6
    assert smoke["raw_sha256"] == completed.raw_sha256
    assert smoke["flags"] == [] and smoke["duration_s"] == 2.0
    assert set(smoke) >= {"elapsed_ms", "model_load_ms", "inference_ms", "audio_sha256"}
    assert doctor.component_status("asr", config, config.data_root)["status"] == "ready"


# --- the real model, offline, from the local cache ------------------------------------------


def real_config(tmp_path: Path, allowed: Path | None = None) -> Config:
    data_root = tmp_path / "data"
    data_root.mkdir()
    return Config(allowed_roots=(allowed or tmp_path,), data_root=data_root)


def skip_without_model(config: Config) -> None:
    _, reason = doctor.snapshot(config.asr.model, config.asr.revision)
    if reason is not None:
        pytest.skip(f"model not in the local cache: {reason}")
    _, missing = doctor.distribution_versions("asr")
    if missing is not None:
        pytest.skip(missing)


@pytest.mark.slow
def test_real_model_on_five_seconds_of_silence_is_empty_or_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = real_config(tmp_path)
    skip_without_model(config)
    restore_offline_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = open_run("silence", IDENTITY, config.data_root)

    result = transcribe(silence(5.0), run, config)

    assert result.status == "completed", result.reason
    raw = result.raw
    assert raw["model"]["repo"] == config.asr.model and raw["model"]["local_files_only"] is True
    assert raw["model"]["implementation"] == asr.MODEL_IMPLEMENTATION
    with_text = [s for s in raw["segments"] if s["text"].strip()]
    if not with_text:
        assert raw["text"] == "" and result.flags == ()
    else:
        assert result.flags == (SUSPECT_HALLUCINATION,)
        assert all(s[SUSPECT_HALLUCINATION] for s in with_text)
        assert all("silent_span" in s["hallucination_signals"] for s in with_text)
    trusted, flagged = asr.trusted_segments(raw)
    assert trusted == [], "no trusted text on silence (T05)"
    assert [id(s) for s in flagged] == [id(s) for s in with_text]
    item_schema = {"$ref": "#/$defs/suspect_segment", "$defs": schema.SCHEMA["$defs"]}
    items = asr.suspect_segments(raw, 5.0)
    assert len(items) == len(with_text)
    assert all(schema_errors(item_schema, item) == [] for item in items)
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert {f["name"] for f in receipt["files"]} >= {"model.bin", "config.json"}
    assert all(len(f["sha256"]) == 64 for f in receipt["files"])


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)
def test_real_model_transcribes_the_dev_recording_and_writes_the_smoke_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_audio = os.environ.get("SOUND_SCOUT_DEV_AUDIO")
    if not dev_audio:
        pytest.skip("SOUND_SCOUT_DEV_AUDIO is not set")
    recording = Path(dev_audio).resolve()
    if not recording.is_file():
        pytest.skip("SOUND_SCOUT_DEV_AUDIO does not name a file")
    config = real_config(tmp_path, allowed=recording.parent)
    skip_without_model(config)
    restore_offline_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = open_run("dev", IDENTITY, config.data_root)
    source = ingest.decode(recording, run, config)
    audio = channels.load(source)
    mono, _ = channels.choose_mono(audio, channels.report(audio))
    derivative = prepare.mono16k(mono, source.probe.sample_rate, run, config)
    mono16k = derivative.samples
    assert abs(len(mono16k) / SR - source.duration_s) < 0.001

    result = transcribe(mono16k, run, config)

    assert result.status == "completed", result.reason
    assert result.raw["audio"]["sha256"] == derivative.sha256
    words = [w for s in result.raw["segments"] for w in s["words"] or []]
    assert words, "a recording with speech gives words"
    assert all(0 <= w["start"] <= w["end"] <= source.duration_s + 0.1 for w in words)
    assert result.flags == (), [s["hallucination_signals"] for s in result.raw["segments"]]

    path = write_smoke_receipt(
        result,
        config,
        test=f"{__name__}::dev_recording",
        input_label=smoke.recording_label(source.duration_s),
    )

    assert path == config.data_root / "receipts" / "asr.json"
    text = path.read_text(encoding="utf-8")
    assert str(recording) not in text and recording.name not in text
    assert result.raw["text"] not in text
    assert json.loads(text)["smoke"]["input"] == f"development recording, {source.duration_s:.1f} s"
    assert doctor.component_status("asr", config, config.data_root)["status"] == "ready"
    assert "faster_whisper" in sys.modules


# --- noise, tone and rough voice on the real model (#135, docs/decisions/0021) ---------------

CORPUS_DIR_ENV = "SOUND_SCOUT_CORPUS_DIR"
# Files of the pilot-1 dev split, excluded from prosody-measure-1 (decision 0021 lists them).
RINCO_DEV_MEMBERS = (
    "INTON-EAF/2-Imperatives/i-1113.wav",
    "INTON-EAF/2-Imperatives/i-12124_2.wav",
    "INTON-EAF/3-Statements/s-221.WAV",
    "INTON-EAF/3-Statements/s-218.WAV",
    "INTON-EAF/1-Questions/q-1122.wav",
)
ROUGH_SENTENCES = (
    "Сегодня утром я долго не мог найти ключи от машины.",
    "Позвони мне вечером, когда закончишь работу, и мы всё обсудим.",
    "В субботу мы поедем на дачу и будем сажать картошку.",
)


def real_transcription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, audio):
    config = Config(allowed_roots=(tmp_path,), data_root=tmp_path / name)
    config.data_root.mkdir()
    skip_without_model(config)
    restore_offline_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = open_run(name, IDENTITY, config.data_root)
    result = transcribe(np.asarray(audio, dtype=np.float32), run, config)
    assert result.status == "completed", result.reason
    assert result.raw["model"]["implementation"] == asr.MODEL_IMPLEMENTATION
    return result


def as_clip(samples: np.ndarray) -> np.ndarray:
    """The samples as Gate 0 stores a clip: 16-bit PCM."""
    return np.frombuffer(gate0.pcm16(samples), "<i2").astype(np.float32) / 32767


@pytest.mark.slow
@pytest.mark.parametrize(
    "control", [gate0.pink_noise, gate0.tone], ids=["N1_pink_noise", "T1_tone_440_hz"]
)
def test_real_model_text_on_the_gate0_noise_and_tone_controls_is_never_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control
) -> None:
    result = real_transcription(tmp_path, monkeypatch, "control", as_clip(control()))

    with_text = [s for s in result.raw["segments"] if s["text"].strip()]
    trusted, flagged = asr.trusted_segments(result.raw)
    assert trusted == [], "no trusted text on a negative control (T05)"
    assert [id(s) for s in flagged] == [id(s) for s in with_text]
    assert all("no_vad_speech" in s["hallucination_signals"] for s in with_text)
    assert result.raw["speech_spans"] == []


def corpus_dir() -> Path:
    value = os.environ.get(CORPUS_DIR_ENV)
    if not value:
        pytest.skip(f"{CORPUS_DIR_ENV} is not set")
    return Path(value)


def piper_voice():
    pytest.importorskip("piper", reason="the corpus-tts extra is not installed")
    try:
        return synthetic.load_voice(corpus_dir())
    except synthetic.SyntheticError as error:
        pytest.skip(str(error))


def to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    sound = parselmouth.Sound(np.asarray(samples, dtype=np.float64), rate)
    return call(sound, "Resample", SR, 50).values[0]


def pulse_times(sound) -> np.ndarray:
    pulses = call(call(sound, "To Manipulation", 0.01, 40, 500), "Extract pulses")
    count = call(pulses, "Get number of points")
    return np.array([call(pulses, "Get time from index", i + 1) for i in range(count)])


def rough_voice(samples: np.ndarray, rate: int, kind: str) -> np.ndarray:
    """A rough copy of the voice: period jitter and shimmer, creak (half F0, 10 % jitter),
    a 30 Hz rattle with clipping, or breath noise at 5 dB SNR."""
    rng = np.random.default_rng(1)
    sound = parselmouth.Sound(np.asarray(samples, dtype=np.float64), rate)
    if kind in ("jitter_shimmer", "creak"):
        jitter = 0.03 if kind == "jitter_shimmer" else 0.10
        manipulation = call(sound, "To Manipulation", 0.01, 40, 500)
        if kind == "creak":
            tier = call(manipulation, "Extract pitch tier")
            call(tier, "Multiply frequencies", sound.xmin, sound.xmax, 0.5)
            call([tier, manipulation], "Replace pitch tier")
            sound = call(manipulation, "Get resynthesis (overlap-add)")
            manipulation = call(sound, "To Manipulation", 0.01, 40, 500)
        times = pulse_times(sound)
        periods = np.diff(times)
        voiced = periods < 0.025
        shifted = np.where(
            voiced, periods * (1 + jitter * rng.standard_normal(len(periods))), periods
        )
        jittered = np.clip(times[0] + np.concatenate([[0.0], np.cumsum(shifted)]), 0, sound.xmax)
        pulses = call("Create empty PointProcess", "pulses", sound.xmin, sound.xmax)
        for time in np.unique(jittered):
            call(pulses, "Add point", float(time))
        call([manipulation, pulses], "Replace pulses")
        sound = call(manipulation, "Get resynthesis (overlap-add)")
        out = sound.values[0].copy()
        edges = [round(t * rate) for t in pulse_times(sound)] + [len(out)]
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            if end - start < 0.025 * rate:
                out[start:end] *= 10 ** (3.0 * rng.standard_normal() / 20)
    elif kind == "rattle":
        t = np.arange(len(samples)) / rate
        out = samples * (0.6 + 0.4 * np.sin(2 * np.pi * 30.0 * t))
        out = np.clip(out / np.max(np.abs(out)), -0.3, 0.3)
    elif kind == "breathy":
        width = round(0.02 * rate)
        envelope = np.convolve(np.abs(samples), np.ones(width) / width, mode="same")
        noise = rng.standard_normal(len(samples)) * envelope
        noise *= np.sqrt(np.mean(samples**2) / np.mean(noise**2)) * 10 ** (-5.0 / 20)
        out = samples + noise
    else:
        out = np.asarray(samples, dtype=np.float64)
    return out / np.max(np.abs(out))


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["clean", "jitter_shimmer", "creak", "rattle", "breathy"])
@pytest.mark.parametrize("sentence", ROUGH_SENTENCES, ids=["s1", "s2", "s3"])
def test_real_model_keeps_trusted_text_for_rough_creaky_and_buzzy_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sentence: str, kind: str
) -> None:
    """Owner input on #135: a voice may rattle; the rule must not take its words away."""
    voice = piper_voice()
    chunks = list(voice.synthesize(sentence, synthetic._synthesis_config()))
    rate = chunks[0].sample_rate
    pause = np.zeros(round(0.5 * rate))
    spoken = np.concatenate([pause, *(c.audio_float_array for c in chunks), pause])
    audio = 0.5 * to_16k(rough_voice(spoken, rate, kind), rate)

    result = real_transcription(tmp_path, monkeypatch, kind, audio)

    trusted, flagged = asr.trusted_segments(result.raw)
    words = [w for s in trusted for w in s["words"] or []]
    assert len(words) >= len(sentence.split()), [s["hallucination_signals"] for s in flagged]
    assert result.raw["speech_spans"], "the detector hears the rough voice"


@pytest.mark.slow
@pytest.mark.parametrize("member", RINCO_DEV_MEMBERS, ids=lambda m: m.rsplit("/", 1)[1])
def test_real_model_trusts_every_segment_of_rinco_dev_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str
) -> None:
    archive = corpus_dir() / "sources" / "rinco" / corpus.RINCO_ARCHIVE
    if not archive.is_file():
        pytest.skip(f"{corpus.RINCO_ARCHIVE} is not in {CORPUS_DIR_ENV}")
    with zipfile.ZipFile(archive) as opened:
        source = tmp_path / "rinco.wav"
        source.write_bytes(opened.read(member))
    config = Config(allowed_roots=(tmp_path,), data_root=tmp_path / "decode")
    config.data_root.mkdir()
    decoded = ingest.decode(source, open_run("decode", IDENTITY, config.data_root), config)
    audio = channels.load(decoded)
    mono, _ = channels.choose_mono(audio, channels.report(audio))
    samples = prepare.mono16k(
        mono, decoded.probe.sample_rate, open_run("prep", IDENTITY, config.data_root), config
    ).samples

    result = real_transcription(tmp_path, monkeypatch, "rinco", samples)

    trusted, flagged = asr.trusted_segments(result.raw)
    assert trusted and flagged == [], [s["hallucination_signals"] for s in flagged]
