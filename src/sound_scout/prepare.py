"""The ASR input: the working mono resampled to 16 kHz as a recorded derivative (R02, R04).

`mono16k` takes the working mono signal `channels.choose_mono` produced, at its native sample
rate, and builds the one derivative `asr.transcribe` consumes: a float32 array at 16 kHz that
covers the whole source from its first sample, so a second of it is a second of the original and
the transcript's timecodes stay on the source timescale. The resampling is done by the system
`ffmpeg` (libswresample through the `aresample` filter, every parameter pinned in
`RESAMPLER_SETTINGS` so the record does not depend on the defaults of an ffmpeg version), run
like the decoder: `proc.run` with argv and no shell, the protocol whitelist, the subprocess wall
time and memory budgets, and the output capped at `budgets.max_duration_seconds` of 16 kHz
samples, so a longer signal is refused as `decoded_samples_exceeded` rather than cut. The result
goes into the run directory as `work/mono16k.f32` (raw little-endian float32, one channel) next
to `work/mono16k.json`, the record of the transformation: source and target rate, frames before
and after, SHA-256 of the source mono and of the derivative, the resampler and its parameters, the
ffmpeg version, the limits applied, and the same data as one `{"name", "settings"}` entry for
`provenance.transformations` of the result contract (`Mono16k.transformation`).

The derivative is the ASR input only: the channel report and `acoustics.measure` keep working on
the original mono (R04: measurements refer to the source signal), and nothing here applies gain,
trims or normalises. A derivative whose length differs from the source by `DURATION_TOLERANCE_S`
or more is refused (`duration_mismatch`) rather than kept. A repeated call for the same run and
the same mono returns the derivative already written, checked against the record by SHA-256,
without running ffmpeg; a different mono for a run that already holds a derivative is refused
(`source_mismatch`), the file is never overwritten. An ffmpeg failure, a timeout or an exceeded
sample cap is a `PrepareError` with a `code` from `REASON_CODES` and the reason, raised before
anything is written; a failure while writing removes what this call created, so no partial
`work/mono16k.f32` stays in the run.
"""

import contextlib
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sound_scout import paths
from sound_scout.config import MIB, Config, decoded_samples_limit
from sound_scout.ingest import (
    BYTES_PER_SAMPLE,
    DECODED_FORMAT,
    DECODED_SAMPLES_EXCEEDED,
    DECODER_UNAVAILABLE,
    DIR_MODE,
    SUBPROCESS_TIMEOUT,
    WORK_DIR_NAME,
)
from sound_scout.proc import RunResult, ffmpeg_argv, run
from sound_scout.runs import Run

MONO16K_FILE_NAME = "mono16k.f32"
RECORD_FILE_NAME = "mono16k.json"
SAMPLE_RATE = 16_000
CHANNELS = 1
RECORD_FORMAT_VERSION = 1
TRANSFORMATION_NAME = "resample_16k_mono"
RESAMPLER = "swr"
RESAMPLER_SETTINGS: dict[str, str | int | float] = {
    "filter_type": "kaiser",
    "kaiser_beta": 9,
    "filter_size": 32,
    "phase_shift": 10,
    "cutoff": 0.97,
    "linear_interp": 0,
    "exact_rational": 1,
    "dither_method": "none",
}
DURATION_TOLERANCE_S = 0.001
VERSION_OUTPUT_LIMIT = 64 * 1024
_SAMPLE_DTYPE = np.dtype("<f4")
_VERSION_LINE = re.compile(r"^ffmpeg version (\S+)")

BAD_AUDIO = "bad_audio"
RESAMPLE_FAILED = "resample_failed"
DURATION_MISMATCH = "duration_mismatch"
SOURCE_MISMATCH = "source_mismatch"
DERIVATIVE_CORRUPT = "derivative_corrupt"
REASON_CODES = (
    BAD_AUDIO,
    RESAMPLE_FAILED,
    DURATION_MISMATCH,
    SOURCE_MISMATCH,
    DERIVATIVE_CORRUPT,
    DECODED_SAMPLES_EXCEEDED,
    SUBPROCESS_TIMEOUT,
    DECODER_UNAVAILABLE,
)


class PrepareError(Exception):
    """The derivative could not be built or reused; `code` is one of `REASON_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Mono16k:
    """The derivative as `asr.transcribe` takes it, with its record; `reused` when not built now."""

    samples: np.ndarray
    path: Path
    record_path: Path
    record: dict
    reused: bool

    @property
    def sha256(self) -> str:
        return self.record["sha256"]

    @property
    def transformation(self) -> dict:
        """The `provenance.transformations` entry of the result contract."""
        return self.record["transformation"]


def mono16k(mono: np.ndarray, rate: int, run: Run, config: Config) -> Mono16k:
    """Build or reuse `work/mono16k.f32` and `work/mono16k.json` of `run` from the working mono.

    `mono` is the one-dimensional float32 signal of `channels.choose_mono` at `rate` Hz. Raises
    `PrepareError` for the signal, ffmpeg, the budgets and the derivative already in the run (the
    array itself is checked first, `bad_audio`), `paths.PathDenied` for a run directory outside
    the data root, and `OSError` when the run directory cannot be written.
    """
    _require_mono(mono, rate)
    source_bytes = np.ascontiguousarray(mono, dtype=_SAMPLE_DTYPE).tobytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    work_dir = paths.resolve_output(run.path, WORK_DIR_NAME, config)
    existing = _existing(work_dir, source_sha256, rate, len(mono))
    if existing is not None:
        return existing
    version = _ffmpeg_version(config)
    samples = _resample(source_bytes, rate, work_dir, config)
    frames = len(samples) // BYTES_PER_SAMPLE
    mismatch_s = frames / SAMPLE_RATE - len(mono) / rate
    if abs(mismatch_s) >= DURATION_TOLERANCE_S:
        raise PrepareError(
            DURATION_MISMATCH,
            f"ffmpeg produced {frames} frames at {SAMPLE_RATE} Hz for {len(mono)} frames at "
            f"{rate} Hz, {mismatch_s:+.4f} s against the source",
        )
    record = _record(
        run,
        rate,
        len(mono),
        source_sha256,
        frames,
        hashlib.sha256(samples).hexdigest(),
        mismatch_s,
        version,
        config,
    )
    _write(work_dir, samples, _json_bytes(record), config)
    return Mono16k(
        samples=np.array(np.frombuffer(samples, dtype=_SAMPLE_DTYPE)),
        path=work_dir / MONO16K_FILE_NAME,
        record_path=work_dir / RECORD_FILE_NAME,
        record=record,
        reused=False,
    )


def _require_mono(mono: object, rate: object) -> None:
    if not isinstance(mono, np.ndarray) or mono.dtype != np.float32 or mono.ndim != 1:
        raise PrepareError(BAD_AUDIO, "expected a one-dimensional float32 array")
    if not len(mono):
        raise PrepareError(BAD_AUDIO, "the array is empty")
    if not np.isfinite(mono).all():
        raise PrepareError(BAD_AUDIO, "samples must be finite")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise PrepareError(BAD_AUDIO, f"sample rate must be a positive integer, got {rate!r}")


def _existing(work_dir: Path, source_sha256: str, rate: int, source_frames: int) -> Mono16k | None:
    """The derivative already in the run, checked against its record; None when there is none."""
    path, record_path = work_dir / MONO16K_FILE_NAME, work_dir / RECORD_FILE_NAME
    present = {name: os.path.lexists(work_dir / name) for name in (path.name, record_path.name)}
    if not any(present.values()):
        return None
    if not all(present.values()):
        missing = next(name for name, found in present.items() if not found)
        raise PrepareError(
            DERIVATIVE_CORRUPT, f"{WORK_DIR_NAME}/{missing} is missing next to its counterpart"
        )
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        source = record["source"]
        recorded_source = (source["sha256"], source["sample_rate"], source["frames"])
        frames, sha256 = record["frames"], record["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PrepareError(
            DERIVATIVE_CORRUPT, f"{WORK_DIR_NAME}/{RECORD_FILE_NAME} is not usable: {error}"
        ) from None
    if recorded_source != (source_sha256, rate, source_frames):
        raise PrepareError(
            SOURCE_MISMATCH,
            f"{WORK_DIR_NAME}/{MONO16K_FILE_NAME} was built from another mono signal "
            f"({recorded_source[2]} frames at {recorded_source[1]} Hz); "
            "the derivative is not overwritten",
        )
    mismatch = PrepareError(
        DERIVATIVE_CORRUPT, f"{WORK_DIR_NAME}/{MONO16K_FILE_NAME} does not match its record"
    )
    try:
        if os.stat(path).st_size != frames * BYTES_PER_SAMPLE:
            raise mismatch
        samples = np.fromfile(path, dtype=_SAMPLE_DTYPE)
    except OSError as error:
        raise PrepareError(
            DERIVATIVE_CORRUPT, f"{WORK_DIR_NAME}/{MONO16K_FILE_NAME} is not readable: {error}"
        ) from None
    if len(samples) != frames or hashlib.sha256(samples.tobytes()).hexdigest() != sha256:
        raise mismatch
    return Mono16k(samples, path, record_path, record, reused=True)


def _ffmpeg_version(config: Config) -> str:
    result = _run_ffmpeg(["ffmpeg", "-version"], config, max_output_bytes=VERSION_OUTPUT_LIMIT)
    if result.timed_out:
        raise PrepareError(
            SUBPROCESS_TIMEOUT, f"ffmpeg exceeded {config.budgets.max_subprocess_seconds} s"
        )
    if result.returncode != 0:
        raise PrepareError(DECODER_UNAVAILABLE, f"ffmpeg -version failed: {_stderr_tail(result)}")
    match = _VERSION_LINE.match(result.stdout.decode("utf-8", errors="replace"))
    if match is None:
        raise PrepareError(DECODER_UNAVAILABLE, "ffmpeg -version gave unrecognized output")
    return match.group(1)


def _resample(source_bytes: bytes, rate: int, work_dir: Path, config: Config) -> bytes:
    """ffmpeg's float32 output at 16 kHz; the source is handed over as a temporary file."""
    samples_limit = decoded_samples_limit(config.budgets, SAMPLE_RATE, CHANNELS)
    source_name = f".{MONO16K_FILE_NAME}.source.{uuid.uuid4().hex}"
    created_dir = _ensure_work_dir(work_dir)
    try:
        with paths.open_output(work_dir, source_name, config) as out:
            out.write(source_bytes)
        result = _run_ffmpeg(
            ffmpeg_argv(
                "-v",
                "error",
                "-xerror",
                "-f",
                DECODED_FORMAT,
                "-ar",
                str(rate),
                "-ac",
                str(CHANNELS),
                "-i",
                str(work_dir / source_name),
                "-af",
                resample_filter(),
                "-f",
                DECODED_FORMAT,
                "-",
            ),
            config,
            max_output_bytes=samples_limit * BYTES_PER_SAMPLE,
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(work_dir / source_name)
        if created_dir:
            with contextlib.suppress(OSError):
                os.rmdir(work_dir)
    if result.truncated:
        raise PrepareError(
            DECODED_SAMPLES_EXCEEDED,
            f"more than {samples_limit} samples resampled "
            f"({config.budgets.max_duration_seconds} s at {SAMPLE_RATE} Hz)",
        )
    if result.timed_out:
        raise PrepareError(
            SUBPROCESS_TIMEOUT, f"ffmpeg exceeded {config.budgets.max_subprocess_seconds} s"
        )
    if result.returncode != 0:
        raise PrepareError(RESAMPLE_FAILED, f"ffmpeg failed: {_stderr_tail(result)}")
    if len(result.stdout) % BYTES_PER_SAMPLE or not result.stdout:
        raise PrepareError(RESAMPLE_FAILED, f"ffmpeg produced {len(result.stdout)} bytes")
    return result.stdout


def resample_filter() -> str:
    """The `aresample` filter with every resampler parameter spelled out."""
    settings = ":".join(f"{name}={value}" for name, value in RESAMPLER_SETTINGS.items())
    return f"aresample=osr={SAMPLE_RATE}:resampler={RESAMPLER}:{settings}"


def _run_ffmpeg(argv: list[str], config: Config, *, max_output_bytes: int) -> RunResult:
    budgets = config.budgets
    try:
        return run(
            argv,
            timeout=budgets.max_subprocess_seconds,
            max_output_bytes=max_output_bytes,
            memory_bytes=budgets.max_subprocess_memory_mib * MIB,
        )
    except FileNotFoundError:
        raise PrepareError(DECODER_UNAVAILABLE, "ffmpeg is not installed") from None


def _record(
    run: Run,
    rate: int,
    source_frames: int,
    source_sha256: str,
    frames: int,
    sha256: str,
    mismatch_s: float,
    ffmpeg_version: str,
    config: Config,
) -> dict:
    budgets = config.budgets
    return {
        "format_version": RECORD_FORMAT_VERSION,
        "kind": "mono16k",
        "run_id": run.run_id,
        "path": f"{WORK_DIR_NAME}/{MONO16K_FILE_NAME}",
        "format": DECODED_FORMAT,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "frames": frames,
        "duration_s": frames / SAMPLE_RATE,
        "sha256": sha256,
        "source": {
            "sample_rate": rate,
            "channels": CHANNELS,
            "frames": source_frames,
            "duration_s": source_frames / rate,
            "sha256": source_sha256,
        },
        "transformation": {
            "name": TRANSFORMATION_NAME,
            "settings": {
                "source_sample_rate": rate,
                "target_sample_rate": SAMPLE_RATE,
                "source_frames": source_frames,
                "frames": frames,
                "duration_mismatch_s": mismatch_s,
                "duration_tolerance_s": DURATION_TOLERANCE_S,
                "resampler": RESAMPLER,
                "resampler_settings": dict(RESAMPLER_SETTINGS),
                "filter": resample_filter(),
                "ffmpeg_version": ffmpeg_version,
                "gain_db": 0.0,
                "source_sha256": source_sha256,
                "sha256": sha256,
            },
        },
        "limits": {
            "max_duration_seconds": budgets.max_duration_seconds,
            "decoded_samples_limit": decoded_samples_limit(budgets, SAMPLE_RATE, CHANNELS),
            "max_subprocess_seconds": budgets.max_subprocess_seconds,
            "max_subprocess_memory_mib": budgets.max_subprocess_memory_mib,
        },
    }


def _ensure_work_dir(work_dir: Path) -> bool:
    """Create the private work directory; True when this call made it."""
    try:
        os.mkdir(work_dir, DIR_MODE)
    except FileExistsError:
        return False
    os.chmod(work_dir, DIR_MODE)
    return True


def _write(work_dir: Path, samples: bytes, record: bytes, config: Config) -> None:
    created: list[Path] = []
    try:
        if _ensure_work_dir(work_dir):
            created.append(work_dir)
        for name, content in ((MONO16K_FILE_NAME, samples), (RECORD_FILE_NAME, record)):
            with paths.open_output(work_dir, name, config) as out:
                created.append(work_dir / name)
                out.write(content)
    except BaseException:
        for path in reversed(created):
            with contextlib.suppress(OSError):
                (os.rmdir if path == work_dir else os.unlink)(path)
        raise


def _json_bytes(record: dict) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _stderr_tail(result: RunResult) -> str:
    text = result.stderr.decode("utf-8", errors="replace").strip()
    return text.splitlines()[-1] if text else f"exit code {result.returncode}"
