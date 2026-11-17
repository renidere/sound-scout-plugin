"""Local ASR (issue #19): the stage contract with a test double, the hallucination rule, the
immutable raw transcript, the smoke receipt; `slow` tests with the real cached model.

The test double stands in for `asr._load_model` only: it proves the artifacts, statuses and
refusals, not that the tool hears anything. The real-model tests need the weights already in
the local Hugging Face cache (never downloaded here) and, for the recording, the path in
`SOUND_SCOUT_DEV_AUDIO`; without them they are skipped, with the reason."""

import dataclasses
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sound_scout import asr, channels, doctor, guard, ingest
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
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


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

    assert hallucination_signals(confident, silence(5.0)) == ["silent_span"]
    flagged = flag_segments([confident], silence(5.0))
    assert flagged[0][SUSPECT_HALLUCINATION] is True
    assert flagged[0]["hallucination_signals"] == ["silent_span"]
    assert flagged[0]["text"] == confident["text"], "the rule flags, it does not rewrite"


def test_rule_uses_the_signal_under_the_segment_not_the_whole_file() -> None:
    audio = np.concatenate([speech_like(2.0), silence(3.0)])

    assert hallucination_signals({"start": 0.0, "end": 2.0, "text": " речь"}, audio) == []
    assert hallucination_signals({"start": 2.5, "end": 4.9, "text": " эхо"}, audio) == [
        "silent_span"
    ]
    assert hallucination_signals({"start": 1.5, "end": 4.9, "text": " край"}, audio) == []


def test_rule_flags_whispers_own_signals_and_ignores_empty_text() -> None:
    audio = speech_like(3.0)
    base = {"start": 0.0, "end": 3.0, "text": " слово"}

    assert hallucination_signals({**base, "no_speech_prob": 0.61}, audio) == ["no_speech_prob"]
    assert hallucination_signals({**base, "avg_logprob": -1.01}, audio) == ["low_avg_logprob"]
    assert hallucination_signals({**base, "compression_ratio": 2.5}, audio) == [
        "high_compression_ratio"
    ]
    assert hallucination_signals(
        {**base, "no_speech_prob": 0.9, "avg_logprob": -2.0, "compression_ratio": 3.0},
        silence(3.0),
    ) == ["silent_span", "no_speech_prob", "low_avg_logprob", "high_compression_ratio"]
    assert hallucination_signals({**base, "no_speech_prob": 0.6}, audio) == []
    assert hallucination_signals({**base, "text": "   "}, silence(3.0)) == []
    assert hallucination_signals({"start": 0.0, "end": 1.0}, silence(3.0)) == []
    assert hallucination_signals({**base, "end": 0.0}, audio) == ["silent_span"]


def test_quiet_but_not_silent_speech_is_not_flagged_by_the_signal_rule() -> None:
    quiet = (speech_like(2.0) * 0.01).astype(np.float32)  # about -53 dBFS RMS

    assert hallucination_signals({"start": 0.0, "end": 2.0, "text": " тихо"}, quiet) == []
    very_quiet = (speech_like(2.0) * 0.001).astype(np.float32)  # about -73 dBFS RMS
    assert hallucination_signals({"start": 0.0, "end": 2.0, "text": " тихо"}, very_quiet) == [
        "silent_span"
    ]


def test_transcribe_marks_silence_segments_with_the_rule(config: Config, run, fake_model) -> None:
    fake_model(
        [segment(0.0, 4.6, " Продолжение следует...", [word(0.0, 4.6, " Продолжение", 0.9)])]
    )

    result = transcribe(silence(5.0), run, config)

    assert result.status == "completed"
    assert result.flags == (SUSPECT_HALLUCINATION,)
    assert result.raw["flags"] == [SUSPECT_HALLUCINATION]
    assert result.raw["segments"][0][SUSPECT_HALLUCINATION] is True
    assert result.raw["segments"][0]["hallucination_signals"] == ["silent_span"]
    assert result.raw["text"] == "Продолжение следует..."
    assert result.raw["hallucination_rule"] == asr.HALLUCINATION_RULE


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
        "evidence_refs": [{"type": "human_label", "ref": "label-1"}],
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
        "evidence_refs": [{"type": "transcript", "ref": "raw"}],
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
        ({**good, "extra": 1}, "extra"),
    ):
        with pytest.raises(AsrError) as error:
            write_corrected(run, config, [good, bad])
        assert error.value.code == "bad_correction" and where in error.value.message, where
    assert not (run.path / "transcript").exists()


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
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert {f["name"] for f in receipt["files"]} >= {"model.bin", "config.json"}
    assert all(len(f["sha256"]) == 64 for f in receipt["files"])


def resample_16k_mono(mono: np.ndarray, sample_rate: int, tmp_path: Path) -> np.ndarray:
    """ffmpeg resample of the working mono to 16 kHz float32 (the test's derivative)."""
    source = tmp_path / "mono.f32"
    source.write_bytes(np.ascontiguousarray(mono, dtype="<f4").tobytes())
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            str(source),
            "-ar",
            str(SR),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype="<f4").copy()


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
    mono16k = resample_16k_mono(mono, source.probe.sample_rate, tmp_path)
    assert abs(len(mono16k) / SR - source.duration_s) < 0.01

    result = transcribe(mono16k, run, config)

    assert result.status == "completed", result.reason
    words = [w for s in result.raw["segments"] for w in s["words"] or []]
    assert words, "a recording with speech gives words"
    assert all(0 <= w["start"] <= w["end"] <= source.duration_s + 0.1 for w in words)
    assert result.flags == (), [s["hallucination_signals"] for s in result.raw["segments"]]

    path = write_smoke_receipt(
        result,
        config,
        test=f"{__name__}::dev_recording",
        input_label="development recording from SOUND_SCOUT_DEV_AUDIO",
    )

    assert path == config.data_root / "receipts" / "asr.json"
    text = path.read_text(encoding="utf-8")
    assert str(recording) not in text and recording.name not in text
    assert result.raw["text"] not in text
    assert doctor.component_status("asr", config, config.data_root)["status"] == "ready"
    assert "faster_whisper" in sys.modules
