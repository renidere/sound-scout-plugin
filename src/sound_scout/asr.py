"""Local ASR on faster-whisper: the raw transcript as an artifact, offline only (R03, R04, R05).

`transcribe` takes the working mono signal resampled to 16 kHz (a float32 array that covers
the whole source from its first sample, so a second here is a second of the original) and
runs the already installed model on the CPU in int8, language `ru`, with the inference
parameters of the owner's proven run (`INFERENCE_PARAMETERS`: no VAD filter, no prompt, no
conditioning on the previous text). It writes two files into `<run>/asr/` and nothing else:
`raw.json` — every segment and word the model produced, with timecodes, probabilities, the
inference parameters and the hash of the audio actually passed to the model — and
`model-receipt.json` — repo, revision, SHA-256 of each model file and the package versions.

Nothing is downloaded, ever: the snapshot is located in the local Hugging Face cache by
`doctor.snapshot` before the model package is even imported, and the model is opened by that
directory with `local_files_only=True` under `HF_HUB_OFFLINE=1`. Missing weights give the
stage `unavailable` with `MODEL_UNAVAILABLE_REASON`, any other snapshot problem (a configured
`asr.revision` that is not cached, several snapshots and no revision chosen) `unavailable` with
that reason as `doctor.snapshot` words it; an exception of the model load or the inference
gives `failed` with the reason; the array itself is checked up front and a bad one
is the caller's error (`AsrError`), not a stage status. The run's wall time is the pipeline's
to bound (R11): this call runs in-process.

The raw transcript is never rewritten (R05): `raw.json` is created exclusively, made
read-only, and a run that already has one refuses a second `transcribe`. A correction is a
separate artifact, `transcript/corrected.json`, written once by `write_corrected` with the
path and SHA-256 of the raw file it refers to and items in the shape of the result contract's
`correction`.

`suspect_hallucination` is a rule here, not a model opinion: a segment with text over a
span of the signal that is digital silence (RMS below `SILENT_SPAN_RMS_DBFS`), or with
Whisper's own no-speech, log-probability or compression-ratio signals past their thresholds,
is flagged in `segments[].suspect_hallucination`, and `flags` carries the flag for the whole
transcript. On this workstation the installed model answers 5 s of digital silence with a
confident phrase (word probabilities near 0.9), so the signal rule is the one that catches it.

`write_smoke_receipt` is the only path to the `asr` smoke receipt of `doctor`: it accepts a
completed, unflagged transcription with at least one word, produced by faster-whisper's own
`WhisperModel` on the configured snapshot, and hands a record without text, paths or test
doubles to `doctor.write_receipt`. `raw["model"]["implementation"]` records the class of the
model object that actually ran, so a substituted model (a test double) can complete a stage
but never certifies the component (CLAUDE.md: a mock proves a contract, not a capability).
"""

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from sound_scout import __version__, doctor, guard, paths, schema
from sound_scout.config import MIB, Config
from sound_scout.runs import Run

ASR_DIR_NAME = "asr"
RAW_FILE_NAME = "raw.json"
MODEL_RECEIPT_FILE_NAME = "model-receipt.json"
TRANSCRIPT_DIR_NAME = "transcript"
CORRECTED_FILE_NAME = "corrected.json"
RAW_FORMAT_VERSION = 1
SAMPLE_RATE = 16_000
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4
NUM_WORKERS = 1
INFERENCE_PARAMETERS: dict[str, object] = {
    "language": "ru",
    "task": "transcribe",
    "beam_size": 5,
    "word_timestamps": True,
    "vad_filter": False,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "initial_prompt": None,
    "prefix": None,
    "hotwords": None,
}
MODEL_UNAVAILABLE_REASON = "модель не установлена; загрузка требует разрешения владельца"
MODEL_IMPLEMENTATION = "faster_whisper.transcribe.WhisperModel"
SUSPECT_HALLUCINATION = "suspect_hallucination"
SILENT_SPAN_RMS_DBFS = -60.0
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
HALLUCINATION_RULE = {
    "silent_span_rms_dbfs": SILENT_SPAN_RMS_DBFS,
    "no_speech_threshold": NO_SPEECH_THRESHOLD,
    "logprob_threshold": LOGPROB_THRESHOLD,
    "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
}
TEXT_POSTPROCESSING = "segments joined with a space; words unchanged"
VERSIONED_DISTRIBUTIONS = ("faster-whisper", "ctranslate2", "numpy", "huggingface-hub")
DIR_MODE = 0o700
RAW_FILE_MODE = 0o400
_HASH_CHUNK = MIB

BAD_AUDIO = "bad_audio"
TRANSCRIPT_EXISTS = "transcript_exists"
TRANSCRIPT_MISSING = "transcript_missing"
CORRECTIONS_EXIST = "corrections_exist"
BAD_CORRECTION = "bad_correction"
REASON_CODES = (BAD_AUDIO, TRANSCRIPT_EXISTS, TRANSCRIPT_MISSING, CORRECTIONS_EXIST, BAD_CORRECTION)

_CORRECTION_VALIDATOR = Draft202012Validator(
    {"$ref": "#/$defs/correction", "$defs": schema.SCHEMA["$defs"]}
)


class AsrError(Exception):
    """The call is refused before any inference or write; `code` is one of `REASON_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Transcription:
    """Outcome of one `transcribe`: the stage status and, when completed, the artifacts."""

    status: str
    reason: str | None
    elapsed_ms: int
    raw_path: Path | None = None
    receipt_path: Path | None = None
    raw_sha256: str | None = None
    raw: dict | None = None

    @property
    def stage(self) -> dict[str, object]:
        """The `asr` entry of `stages` in the result contract."""
        return {
            "requested": True,
            "status": self.status,
            "reason": self.reason,
            "elapsed_ms": self.elapsed_ms,
        }

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(self.raw["flags"]) if self.raw is not None else ()


def transcribe(mono16k: np.ndarray, run: Run, config: Config) -> Transcription:
    """Run the local model on `mono16k` and write `asr/raw.json` and `asr/model-receipt.json`.

    `mono16k` is float32, one-dimensional, finite, at most `budgets.max_duration_seconds`
    long, and the whole source at 16 kHz: its timecodes are the source's. A run that already
    holds a raw transcript is refused (`AsrError`, `transcript_exists`).
    """
    started = time.monotonic()
    _require_audio(mono16k, config)
    asr_dir = paths.resolve_output(run.path, ASR_DIR_NAME, config)
    raw_path = asr_dir / RAW_FILE_NAME
    if os.path.lexists(raw_path):
        raise AsrError(
            TRANSCRIPT_EXISTS,
            f"{ASR_DIR_NAME}/{RAW_FILE_NAME} is already in the run; the raw transcript "
            "is not rewritten",
        )
    versions, package_reason = doctor.distribution_versions("asr")
    model, model_reason = doctor.snapshot(config.asr.model, config.asr.revision)
    if package_reason is not None:
        return Transcription("unavailable", package_reason, _elapsed_ms(started))
    if model_reason is not None:
        if model_reason.startswith(doctor.WEIGHTS_NOT_CACHED):
            model_reason = MODEL_UNAVAILABLE_REASON
        return Transcription("unavailable", model_reason, _elapsed_ms(started))
    snapshot_dir = doctor.snapshot_dir(model["repo"], model["revision"])
    os.environ.update(guard.OFFLINE_ENV)
    try:
        load_started = time.monotonic()
        loaded = _load_model(snapshot_dir)
        implementation = f"{type(loaded).__module__}.{type(loaded).__qualname__}"
        model_load_ms = _elapsed_ms(load_started)
        inference_started = time.monotonic()
        iterator, info = loaded.transcribe(mono16k, **INFERENCE_PARAMETERS)
        segments = [_json_safe(_record(segment)) for segment in iterator]
        inference_ms = _elapsed_ms(inference_started)
    except Exception as error:  # a stage boundary: the reason is reported, not raised (R11)
        return Transcription("failed", f"{type(error).__name__}: {error}", _elapsed_ms(started))
    segments = flag_segments(segments, mono16k)
    raw = {
        "format_version": RAW_FORMAT_VERSION,
        "kind": "transcript_raw",
        "run_id": run.run_id,
        "model": {
            "repo": model["repo"],
            "revision": model["revision"],
            "implementation": implementation,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "cpu_threads": CPU_THREADS,
            "num_workers": NUM_WORKERS,
            "local_files_only": True,
        },
        "inference": dict(INFERENCE_PARAMETERS),
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "frames": int(len(mono16k)),
            "duration_s": len(mono16k) / SAMPLE_RATE,
            "sha256": hashlib.sha256(np.ascontiguousarray(mono16k).tobytes()).hexdigest(),
            "timescale": "source",
            "offset_s": 0.0,
        },
        "info": _json_safe(_record(info)),
        "segments": segments,
        "text": " ".join(segment["text"].strip() for segment in segments).strip(),
        "flags": [SUSPECT_HALLUCINATION]
        if any(segment[SUSPECT_HALLUCINATION] for segment in segments)
        else [],
        "hallucination_rule": dict(HALLUCINATION_RULE),
        "text_postprocessing": TEXT_POSTPROCESSING,
        "timing": {"model_load_ms": model_load_ms, "inference_ms": inference_ms},
        "versions": _versions(versions),
    }
    receipt = _model_receipt(model, snapshot_dir, versions)
    raw_bytes = _json_bytes(raw)
    _write_artifacts(asr_dir, raw_bytes, _json_bytes(receipt), config)
    return Transcription(
        "completed",
        None,
        _elapsed_ms(started),
        raw_path=raw_path,
        receipt_path=asr_dir / MODEL_RECEIPT_FILE_NAME,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw=raw,
    )


def flag_segments(segments: list[dict], mono16k: np.ndarray) -> list[dict]:
    """Copy of `segments` with `hallucination_signals` and `suspect_hallucination` on each."""
    flagged = []
    for segment in segments:
        signals = hallucination_signals(segment, mono16k)
        flagged.append(
            {**segment, "hallucination_signals": signals, SUSPECT_HALLUCINATION: bool(signals)}
        )
    return flagged


def hallucination_signals(segment: Mapping[str, object], mono16k: np.ndarray) -> list[str]:
    """Why a segment's text is suspect; empty for a segment without text or without signals.

    `silent_span`: the signal under the segment is digital silence (RMS below
    `SILENT_SPAN_RMS_DBFS`), so there was nothing to transcribe. The other three are Whisper's
    own quality signals past the thresholds Whisper itself uses for a retry.
    """
    text = segment.get("text")
    if not isinstance(text, str) or not text.strip():
        return []
    signals = []
    rms_dbfs = _span_rms_dbfs(segment, mono16k)
    if rms_dbfs is None or rms_dbfs < SILENT_SPAN_RMS_DBFS:
        signals.append("silent_span")
    if _number(segment.get("no_speech_prob"), 0.0) > NO_SPEECH_THRESHOLD:
        signals.append("no_speech_prob")
    if _number(segment.get("avg_logprob"), 0.0) < LOGPROB_THRESHOLD:
        signals.append("low_avg_logprob")
    if _number(segment.get("compression_ratio"), 0.0) > COMPRESSION_RATIO_THRESHOLD:
        signals.append("high_compression_ratio")
    return signals


def read_raw(run: Run) -> dict:
    """The raw transcript of `run`; `AsrError` (`transcript_missing`) when there is none."""
    path = run.path / ASR_DIR_NAME / RAW_FILE_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise AsrError(
            TRANSCRIPT_MISSING, f"{ASR_DIR_NAME}/{RAW_FILE_NAME} is not in the run"
        ) from None


def write_corrected(run: Run, config: Config, corrections: list[Mapping[str, object]]) -> Path:
    """Write `transcript/corrected.json`: corrections next to the raw transcript, never in it.

    Each item has the shape of the result contract's `correction` (`AsrError`,
    `bad_correction` otherwise). The file names the raw transcript by path and SHA-256, is
    created once (`corrections_exist` on a repeat) and requires a raw transcript in the run.
    """
    raw_path = run.path / ASR_DIR_NAME / RAW_FILE_NAME
    try:
        raw_sha256 = _sha256_file(raw_path)
    except FileNotFoundError:
        raise AsrError(
            TRANSCRIPT_MISSING, f"{ASR_DIR_NAME}/{RAW_FILE_NAME} is not in the run"
        ) from None
    items = []
    for index, correction in enumerate(corrections):
        errors = sorted(_CORRECTION_VALIDATOR.iter_errors(correction), key=lambda e: e.json_path)
        if errors:
            raise AsrError(
                BAD_CORRECTION,
                f"corrections[{index}]{errors[0].json_path[1:]}: {errors[0].message}",
            )
        if correction["start"] >= correction["end"]:
            raise AsrError(BAD_CORRECTION, f"corrections[{index}]: start must be less than end")
        items.append(dict(correction))
    transcript_dir = paths.resolve_output(run.path, TRANSCRIPT_DIR_NAME, config)
    target = transcript_dir / CORRECTED_FILE_NAME
    if os.path.lexists(target):
        raise AsrError(
            CORRECTIONS_EXIST, f"{TRANSCRIPT_DIR_NAME}/{CORRECTED_FILE_NAME} is already in the run"
        )
    document = {
        "format_version": RAW_FORMAT_VERSION,
        "kind": "transcript_corrections",
        "run_id": run.run_id,
        "raw": {"path": f"{ASR_DIR_NAME}/{RAW_FILE_NAME}", "sha256": raw_sha256},
        "corrections": items,
    }
    _write_files(transcript_dir, {CORRECTED_FILE_NAME: _json_bytes(document)}, config)
    return target


def write_smoke_receipt(
    transcription: Transcription, config: Config, *, test: str, input_label: str
) -> Path:
    """Save the `asr` smoke receipt of `doctor` after a real, completed transcription.

    Refused (`doctor.ReceiptRejected`) unless the stage completed on the configured model and
    snapshot, with at least one word, without `suspect_hallucination`, and the model object
    that ran was faster-whisper's `WhisperModel` (`MODEL_IMPLEMENTATION`): a substituted model
    never gets a receipt. `test` names the test that ran, `input_label` describes the audio
    without a path. The record carries counts, probabilities and timing — no text, no path.
    """
    raw = transcription.raw
    if transcription.status != "completed" or raw is None:
        raise doctor.ReceiptRejected(
            f"asr stage is {transcription.status!r}: {transcription.reason}"
        )
    used = (raw["model"]["repo"], raw["model"]["revision"])
    expected, _ = doctor.snapshot(config.asr.model, config.asr.revision)
    if used != (expected["repo"], expected["revision"]):
        raise doctor.ReceiptRejected(
            f"transcription used model {used!r}, configured "
            f"{(expected['repo'], expected['revision'])!r}"
        )
    if transcription.flags:
        raise doctor.ReceiptRejected(f"transcription is flagged {list(transcription.flags)}")
    probabilities = [
        _number(word.get("probability"), 0.0)
        for segment in raw["segments"]
        for word in segment.get("words") or []
    ]
    if not probabilities:
        raise doctor.ReceiptRejected("transcription has no words")
    implementation = raw["model"].get("implementation")
    if implementation != MODEL_IMPLEMENTATION:
        raise doctor.ReceiptRejected(
            f"transcription came from {implementation!r}, not {MODEL_IMPLEMENTATION!r}; "
            "a substituted model proves no capability"
        )
    smoke = {
        "test": test,
        "input": input_label,
        "implementation": implementation,
        "duration_s": raw["audio"]["duration_s"],
        "audio_sha256": raw["audio"]["sha256"],
        "segments": len(raw["segments"]),
        "words": len(probabilities),
        "mean_word_probability": sum(probabilities) / len(probabilities),
        "min_word_probability": min(probabilities),
        "flags": list(transcription.flags),
        "elapsed_ms": transcription.elapsed_ms,
        **raw["timing"],
        "raw_sha256": transcription.raw_sha256,
    }
    return doctor.write_receipt("asr", config, smoke)


def _load_model(snapshot_dir: Path):
    """Open the snapshot directory with faster-whisper; the one place the package is imported."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        str(snapshot_dir),
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        num_workers=NUM_WORKERS,
        local_files_only=True,
    )


def _require_audio(mono16k: object, config: Config) -> None:
    if not isinstance(mono16k, np.ndarray) or mono16k.dtype != np.float32 or mono16k.ndim != 1:
        raise AsrError(BAD_AUDIO, "expected a one-dimensional float32 array at 16 kHz")
    if not len(mono16k):
        raise AsrError(BAD_AUDIO, "the array is empty")
    if not np.isfinite(mono16k).all():
        raise AsrError(BAD_AUDIO, "samples must be finite")
    limit = config.budgets.max_duration_seconds * SAMPLE_RATE
    if len(mono16k) > limit:
        raise AsrError(
            BAD_AUDIO,
            f"{len(mono16k)} frames, the limit is {limit} "
            f"({config.budgets.max_duration_seconds} s at {SAMPLE_RATE} Hz)",
        )


def _span_rms_dbfs(segment: Mapping[str, object], mono16k: np.ndarray) -> float | None:
    """RMS in dBFS of the signal under the segment; None for digital silence or an empty span."""
    start = max(0, int(math.floor(_number(segment.get("start"), 0.0) * SAMPLE_RATE)))
    end = min(len(mono16k), int(math.ceil(_number(segment.get("end"), 0.0) * SAMPLE_RATE)))
    span = mono16k[start:end]
    if not len(span):
        return None
    rms = float(np.sqrt(np.mean(np.square(span, dtype=np.float64))))
    return 20 * math.log10(rms) if rms > 0 else None


def _model_receipt(model: Mapping[str, object], snapshot_dir: Path, versions: Mapping) -> dict:
    files = [
        {"name": name, "bytes": size, "sha256": _sha256_file(snapshot_dir / name)}
        for name, size in sorted(model["files"].items())
    ]
    return {
        "receipt_version": RAW_FORMAT_VERSION,
        "repo": model["repo"],
        "revision": model["revision"],
        "files": files,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "local_files_only": True,
        "offline_env": {name: os.environ.get(name) for name in guard.OFFLINE_ENV},
        "network_guard_active": guard.network_guard_active(),
        "versions": _versions(versions),
        "platform": doctor.platform_info(),
        "audio_uploaded": False,
        "paid_inference_called": False,
    }


def _versions(installed: Mapping[str, str | None]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"sound-scout": __version__}
    for name in VERSIONED_DISTRIBUTIONS:
        versions[name] = installed.get(name) or doctor.installed_version(name)
    return versions


def _write_artifacts(asr_dir: Path, raw_bytes: bytes, receipt_bytes: bytes, config: Config) -> None:
    _write_files(
        asr_dir, {MODEL_RECEIPT_FILE_NAME: receipt_bytes, RAW_FILE_NAME: raw_bytes}, config
    )
    os.chmod(asr_dir / RAW_FILE_NAME, RAW_FILE_MODE)


def _write_files(directory: Path, files: Mapping[str, bytes], config: Config) -> None:
    """Create `directory` (0700) under the run and the files in it exclusively; undo on failure."""
    created: list[Path] = []
    try:
        try:
            os.mkdir(directory, DIR_MODE)
        except FileExistsError:
            pass
        else:
            created.append(directory)
            os.chmod(directory, DIR_MODE)
        for name, content in files.items():
            with paths.open_output(directory, name, config) as out:
                created.append(directory / name)
                out.write(content)
    except BaseException:
        for path in reversed(created):
            with contextlib.suppress(OSError):
                (os.rmdir if path == directory else os.unlink)(path)
        raise


def _record(value: object) -> object:
    """A faster-whisper result object as plain data: dataclasses become dicts, recursively."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _record(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _record(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_record(item) for item in value]
    return value


def _json_safe(value: object) -> object:
    """Non-finite floats become null (the contract forbids NaN/Infinity); numpy scalars, Python."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _json_bytes(document: Mapping[str, object]) -> bytes:
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _number(value: object, default: float) -> float:
    return (
        float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
