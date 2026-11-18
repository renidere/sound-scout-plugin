"""Clips of the decoded original and their listening copies (R02, R05, spec § 4).

`cut` takes an interval on the source timescale, pads it by `pad_ms` on both sides without
crossing the recording boundary, and writes the padded fragment of `work/decoded.f32` — the
signal as `ingest.decode` left it, every channel, native sample rate, no gain — as a WAV
(`clips/<id>.wav`, 32-bit PCM through the stdlib `wave` module, never a lossy codec) next to its
record `clips/<id>.json`: `source_start` and `source_end` in seconds on the source timescale,
rounded to whole source frames so that sample `i` of the clip is sample `first_frame + i` of
the original, the requested interval, `pad_ms`, the SHA-256 of the WAV and of the source it was
cut from. The clip ID names the frame span, `clip-<first_frame>-<last_frame>`, so the same
interval of the same source gives the same clip; a repeated `cut` returns the clip already in the
run, checked against its record, and never overwrites it. An interval outside `[0, duration]`,
with `start >= end`, shorter than a frame, or a negative `pad_ms` is refused (`bad_interval`)
before anything is written.

`listening_copy` makes the louder copy R02 allows for listening: the clip's samples times one
linear gain that brings the peak to `target_peak_dbfs` (between -120 and 0 dBFS, so no sample
clips and none vanishes below the 32-bit quantization step), written as `listen/<id>.wav` with a
record that carries `gain_db`, the peak before and after and the flag `for_listening_only: true`.
One listening copy per clip: a repeated call with the same target returns it; a call with another
target is refused (`clip_exists`) instead of returning a copy at a different level or overwriting
it. Measurements never read it: `acoustics.measure` and the channel report work on the decoded
original, and the clip itself keeps `gain_db` 0. A clip of digital silence has no finite gain and
is refused (`silent_clip`) rather than amplified into noise.

`artifacts` lists every clip and listening copy of a run as `artifacts[]` entries of the result
contract (`schema`): the path is relative to the run directory, `source_start`/`source_end` map
the file onto the original. Both files of a pair are created exclusively (`paths.open_output`,
mode 0600) in a private directory; a failure while writing removes what this call created.
"""

import contextlib
import hashlib
import io
import json
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sound_scout import paths
from sound_scout.config import Config
from sound_scout.ingest import (
    BYTES_PER_SAMPLE,
    DECODED_FILE_NAME,
    DIR_MODE,
    SOURCE_FILE_NAME,
    WORK_DIR_NAME,
)
from sound_scout.runs import Run

CLIPS_DIR_NAME = "clips"
LISTEN_DIR_NAME = "listen"
CLIP_KIND = "clip"
LISTENING_COPY_KIND = "listening_copy"
CLIP_ID_PREFIX = "clip-"
LISTEN_ID_PREFIX = "listen-"
DEFAULT_PAD_MS = 150
DEFAULT_TARGET_PEAK_DBFS = -1.0
MIN_TARGET_PEAK_DBFS = -120.0
RECORD_FORMAT_VERSION = 1
SAMPLE_FORMAT = "pcm_s32le"
SAMPLE_WIDTH = 4
_FULL_SCALE = 2**31
_SOURCE_DTYPE = np.dtype("<f4")
_WAV_DTYPE = np.dtype("<i4")

BAD_INTERVAL = "bad_interval"
BAD_TARGET = "bad_target"
SOURCE_MISSING = "source_missing"
SOURCE_CORRUPT = "source_corrupt"
CLIP_MISSING = "clip_missing"
CLIP_CORRUPT = "clip_corrupt"
CLIP_EXISTS = "clip_exists"
SILENT_CLIP = "silent_clip"
REASON_CODES = (
    BAD_INTERVAL,
    BAD_TARGET,
    SOURCE_MISSING,
    SOURCE_CORRUPT,
    CLIP_MISSING,
    CLIP_CORRUPT,
    CLIP_EXISTS,
    SILENT_CLIP,
)


class ClipError(Exception):
    """The clip or listening copy could not be made or reused; `code` is one of `REASON_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Clip:
    """A clip or a listening copy in the run with its record; `reused` when not written now."""

    id: str
    kind: str
    path: Path
    record_path: Path
    record: dict
    reused: bool

    @property
    def sha256(self) -> str:
        return self.record["sha256"]

    @property
    def source_start(self) -> float:
        return self.record["source_start"]

    @property
    def source_end(self) -> float:
        return self.record["source_end"]

    @property
    def gain_db(self) -> float:
        return self.record["gain_db"]

    @property
    def artifact(self) -> dict:
        """The `artifacts[]` entry of the result contract, path relative to the run directory."""
        return _artifact(self.record)


@dataclass(frozen=True)
class _Source:
    record: dict
    decoded_path: Path
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate


def cut(
    run: Run, start: float, end: float, pad_ms: int = DEFAULT_PAD_MS, *, config: Config
) -> Clip:
    """Write or reuse `clips/<id>.wav` and `clips/<id>.json` for `[start, end]` padded by `pad_ms`.

    `start` and `end` are seconds on the source timescale. Raises `ClipError` for the interval,
    a run without a decoded source, or a clip in the run that does not match its record;
    `paths.PathDenied` for a run directory outside the data root.
    """
    _require_pad(pad_ms)
    source = _load_source(run)
    _require_interval(start, end, source.duration_s)
    rate = source.sample_rate
    pad_frames = round(pad_ms * rate / 1000)
    first = max(0, round(start * rate) - pad_frames)
    last = min(source.frames, round(end * rate) + pad_frames)
    if last <= first:
        raise ClipError(BAD_INTERVAL, f"[{start}, {end}] is shorter than one frame at {rate} Hz")
    clip_id = f"{CLIP_ID_PREFIX}{first}-{last}"
    clips_dir = paths.resolve_output(run.path, CLIPS_DIR_NAME, config)
    existing = _existing(clips_dir, clip_id, CLIP_KIND, source)
    if existing is not None:
        return existing
    samples = _read_frames(source, first, last)
    pcm, clipped = _to_pcm(samples)
    wav = _wav_bytes(pcm, rate)
    record = {
        "format_version": RECORD_FORMAT_VERSION,
        "kind": CLIP_KIND,
        "id": clip_id,
        "run_id": run.run_id,
        "path": f"{CLIPS_DIR_NAME}/{clip_id}.wav",
        "format": "wav",
        "sample_format": SAMPLE_FORMAT,
        "sample_rate": rate,
        "channels": source.channels,
        "frames": last - first,
        "duration_s": (last - first) / rate,
        "sha256": hashlib.sha256(wav).hexdigest(),
        "source": _source_summary(source),
        "source_start": first / rate,
        "source_end": last / rate,
        "first_frame": first,
        "last_frame": last,
        "requested_start": float(start),
        "requested_end": float(end),
        "pad_ms": pad_ms,
        "gain_db": 0.0,
        "for_listening_only": False,
        "clipped_samples": clipped,
    }
    _write_pair(clips_dir, clip_id, wav, record, config)
    return Clip(
        clip_id,
        CLIP_KIND,
        clips_dir / f"{clip_id}.wav",
        clips_dir / f"{clip_id}.json",
        record,
        False,
    )


def listening_copy(
    run: Run,
    clip_id: str,
    target_peak_dbfs: float = DEFAULT_TARGET_PEAK_DBFS,
    *,
    config: Config,
) -> Clip:
    """Write or reuse `listen/<id>.wav` and `listen/<id>.json`: the clip at `target_peak_dbfs`.

    The gain is one linear factor applied to every sample and recorded as `gain_db`; the clip
    file is left as it is. Raises `ClipError` for a target at or above 0 dBFS or below
    `MIN_TARGET_PEAK_DBFS` (`bad_target`), a clip not in the run (`clip_missing`), a clip or copy
    that does not match its record (`clip_corrupt`), a copy already in the run made at another
    target (`clip_exists`) and a clip of digital silence (`silent_clip`).
    """
    if (
        not _finite_number(target_peak_dbfs)
        or target_peak_dbfs >= 0
        or target_peak_dbfs < MIN_TARGET_PEAK_DBFS
    ):
        raise ClipError(
            BAD_TARGET,
            f"target peak must be a number in [{MIN_TARGET_PEAK_DBFS}, 0) dBFS, "
            f"got {target_peak_dbfs!r}",
        )
    clips_dir = paths.resolve_output(run.path, CLIPS_DIR_NAME, config)
    clip = _load_clip(clips_dir, clip_id)
    samples, rate = _read_wav(clip.path.read_bytes())
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        raise ClipError(
            SILENT_CLIP, f"{clip.record['path']} is digital silence, no gain brings it to a level"
        )
    gain = 10 ** (target_peak_dbfs / 20) / peak
    pcm, _ = _to_pcm(samples * np.float32(gain))
    wav = _wav_bytes(pcm, rate)
    listen_id = f"{LISTEN_ID_PREFIX}{clip_id.removeprefix(CLIP_ID_PREFIX)}"
    listen_dir = paths.resolve_output(run.path, LISTEN_DIR_NAME, config)
    source = clip.record["source"]
    existing = _existing(listen_dir, listen_id, LISTENING_COPY_KIND, source)
    if existing is not None:
        if hashlib.sha256(wav).hexdigest() != existing.sha256:
            raise ClipError(
                CLIP_EXISTS,
                f"{existing.record['path']} was made at "
                f"{existing.record.get('target_peak_dbfs')} dBFS, not {target_peak_dbfs}; "
                "it is not overwritten",
            )
        return existing
    peak_after = float(np.max(np.abs(pcm))) / _FULL_SCALE
    record = {
        "format_version": RECORD_FORMAT_VERSION,
        "kind": LISTENING_COPY_KIND,
        "id": listen_id,
        "run_id": run.run_id,
        "path": f"{LISTEN_DIR_NAME}/{listen_id}.wav",
        "format": "wav",
        "sample_format": SAMPLE_FORMAT,
        "sample_rate": rate,
        "channels": clip.record["channels"],
        "frames": clip.record["frames"],
        "duration_s": clip.record["duration_s"],
        "sha256": hashlib.sha256(wav).hexdigest(),
        "clip": {"id": clip.id, "path": clip.record["path"], "sha256": clip.sha256},
        "source": dict(source),
        "source_start": clip.source_start,
        "source_end": clip.source_end,
        "gain_linear": gain,
        "gain_db": 20 * math.log10(gain),
        "target_peak_dbfs": float(target_peak_dbfs),
        "peak_dbfs_before": 20 * math.log10(peak),
        "peak_dbfs_after": 20 * math.log10(peak_after),
        "for_listening_only": True,
    }
    _write_pair(listen_dir, listen_id, wav, record, config)
    return Clip(
        listen_id,
        LISTENING_COPY_KIND,
        listen_dir / f"{listen_id}.wav",
        listen_dir / f"{listen_id}.json",
        record,
        False,
    )


def load(run: Run, clip_id: str, *, config: Config) -> Clip:
    """The clip `clip_id` of `run`, its WAV checked against the record.

    Raises `ClipError` (`clip_missing`, `clip_corrupt`); `paths.PathDenied` for a run directory
    outside the data root.
    """
    return _load_clip(paths.resolve_output(run.path, CLIPS_DIR_NAME, config), clip_id)


def pcm_f32le(clip: Clip) -> bytes:
    """The clip's samples as little-endian float32, channels interleaved: the backend payload."""
    samples, _ = _read_wav(clip.path.read_bytes())
    return np.ascontiguousarray(samples, dtype=_SOURCE_DTYPE).tobytes()


def artifacts(run: Run) -> list[dict]:
    """Every clip and listening copy of `run` as `artifacts[]` entries, sorted by id.

    Raises `ClipError` (`clip_corrupt`) when a record has no WAV next to it or is not readable.
    """
    entries = []
    for dir_name in (CLIPS_DIR_NAME, LISTEN_DIR_NAME):
        directory = run.path / dir_name
        if not directory.is_dir():
            continue
        for record_path in sorted(directory.glob("*.json")):
            record = _read_record(record_path)
            if not os.path.isfile(run.path / record["path"]):
                raise ClipError(CLIP_CORRUPT, f"{record['path']} is missing next to its record")
            entries.append(_artifact(record))
    return sorted(entries, key=lambda entry: entry["id"])


def _artifact(record: dict) -> dict:
    return {
        "id": record["id"],
        "kind": record["kind"],
        "path": record["path"],
        "sha256": record["sha256"],
        "source_start": record["source_start"],
        "source_end": record["source_end"],
    }


def _require_pad(pad_ms: object) -> None:
    if not isinstance(pad_ms, int) or isinstance(pad_ms, bool) or pad_ms < 0:
        raise ClipError(BAD_INTERVAL, f"pad_ms must be a non-negative integer, got {pad_ms!r}")


def _require_interval(start: object, end: object, duration_s: float) -> None:
    if not (_finite_number(start) and _finite_number(end)):
        raise ClipError(
            BAD_INTERVAL, f"start and end must be finite numbers, got {start!r}, {end!r}"
        )
    if start < 0:
        raise ClipError(BAD_INTERVAL, f"start must be >= 0, got {start}")
    if start >= end:
        raise ClipError(BAD_INTERVAL, f"start must be less than end, got {start} >= {end}")
    if end > duration_s:
        raise ClipError(BAD_INTERVAL, f"end {end} exceeds the source duration {duration_s}")


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _load_source(run: Run) -> _Source:
    record_path = run.path / SOURCE_FILE_NAME
    decoded_path = run.path / WORK_DIR_NAME / DECODED_FILE_NAME
    if not (record_path.is_file() and decoded_path.is_file()):
        raise ClipError(
            SOURCE_MISSING,
            f"the run holds no {SOURCE_FILE_NAME} and {WORK_DIR_NAME}/{DECODED_FILE_NAME}",
        )
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        rate, channels, frames = record["sample_rate"], record["channels"], record["frames"]
        record["sha256"], record["id"], record["decoded"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ClipError(SOURCE_CORRUPT, f"{SOURCE_FILE_NAME} is not usable: {error}") from None
    if decoded_path.stat().st_size != frames * channels * BYTES_PER_SAMPLE:
        raise ClipError(
            SOURCE_CORRUPT, f"{WORK_DIR_NAME}/{DECODED_FILE_NAME} does not hold {frames} frames"
        )
    return _Source(record, decoded_path, rate, channels, frames)


def _source_summary(source: _Source) -> dict:
    record = source.record
    return {
        "id": record["id"],
        "sha256": record["sha256"],
        "decoded_sha256": record["decoded"]["sha256"],
        "sample_rate": source.sample_rate,
        "channels": source.channels,
        "frames": source.frames,
        "duration_s": source.duration_s,
    }


def _read_frames(source: _Source, first: int, last: int) -> np.ndarray:
    """Frames `[first, last)` of the decoded original as a frames × channels float32 array."""
    channels = source.channels
    samples = np.fromfile(
        source.decoded_path,
        dtype=_SOURCE_DTYPE,
        count=(last - first) * channels,
        offset=first * channels * BYTES_PER_SAMPLE,
    )
    return samples.reshape(last - first, channels)


def _to_pcm(samples: np.ndarray) -> tuple[np.ndarray, int]:
    """32-bit PCM of float samples; the count of samples beyond full scale that were clamped."""
    clipped = int(np.count_nonzero(np.abs(samples) > 1.0))
    scaled = np.clip(samples.astype(np.float64) * _FULL_SCALE, -_FULL_SCALE, _FULL_SCALE - 1)
    return np.rint(scaled).astype(_WAV_DTYPE), clipped


def _wav_bytes(pcm: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(pcm.shape[1])
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(rate)
        out.writeframes(np.ascontiguousarray(pcm).tobytes())
    return buffer.getvalue()


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Float32 frames × channels and the sample rate of a WAV `_wav_bytes` wrote."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        params = wav.getparams()
        if params.sampwidth != SAMPLE_WIDTH:
            raise ClipError(
                CLIP_CORRUPT, f"expected {SAMPLE_WIDTH * 8}-bit PCM, got {params.sampwidth * 8}-bit"
            )
        frames = wav.readframes(params.nframes)
    pcm = np.frombuffer(frames, dtype=_WAV_DTYPE).reshape(params.nframes, params.nchannels)
    return (pcm.astype(np.float64) / _FULL_SCALE).astype(np.float32), params.framerate


def _existing(directory: Path, clip_id: str, kind: str, source: object) -> Clip | None:
    """The pair already in `directory`, checked against its record; None when there is none."""
    path, record_path = directory / f"{clip_id}.wav", directory / f"{clip_id}.json"
    present = {p.name: os.path.lexists(p) for p in (path, record_path)}
    if not any(present.values()):
        return None
    if not all(present.values()):
        missing = next(name for name, found in present.items() if not found)
        raise ClipError(
            CLIP_CORRUPT, f"{directory.name}/{missing} is missing next to its counterpart"
        )
    record = _read_record(record_path)
    recorded_source = record.get("source")
    expected = _source_summary(source) if isinstance(source, _Source) else source
    if record.get("kind") != kind or recorded_source != expected:
        raise ClipError(
            CLIP_EXISTS,
            f"{directory.name}/{path.name} was made from another source; it is not overwritten",
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
        raise ClipError(CLIP_CORRUPT, f"{directory.name}/{path.name} does not match its record")
    return Clip(clip_id, kind, path, record_path, record, reused=True)


def _load_clip(clips_dir: Path, clip_id: str) -> Clip:
    if not isinstance(clip_id, str) or not paths.OUTPUT_NAME.fullmatch(clip_id):
        raise ClipError(CLIP_MISSING, f"{clip_id!r} is not a clip id")
    path, record_path = clips_dir / f"{clip_id}.wav", clips_dir / f"{clip_id}.json"
    if not (os.path.lexists(path) and os.path.lexists(record_path)):
        raise ClipError(CLIP_MISSING, f"the run holds no clip {clip_id!r}")
    record = _read_record(record_path)
    if record.get("kind") != CLIP_KIND:
        raise ClipError(CLIP_CORRUPT, f"{CLIPS_DIR_NAME}/{record_path.name} is not a clip record")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
        raise ClipError(CLIP_CORRUPT, f"{CLIPS_DIR_NAME}/{path.name} does not match its record")
    return Clip(clip_id, CLIP_KIND, path, record_path, record, reused=True)


def _read_record(record_path: Path) -> dict:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise TypeError("record is not an object")
        for key in ("id", "kind", "path", "sha256", "source", "source_start", "source_end"):
            record[key]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ClipError(
            CLIP_CORRUPT, f"{record_path.parent.name}/{record_path.name} is not usable: {error}"
        ) from None
    return record


def _write_pair(directory: Path, clip_id: str, wav: bytes, record: dict, config: Config) -> None:
    created: list[Path] = []
    try:
        try:
            os.mkdir(directory, DIR_MODE)
        except FileExistsError:
            pass
        else:
            created.append(directory)
            os.chmod(directory, DIR_MODE)
        content = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        for name, data in ((f"{clip_id}.wav", wav), (f"{clip_id}.json", content)):
            with paths.open_output(directory, name, config) as out:
                created.append(directory / name)
                out.write(data)
    except BaseException:
        for path in reversed(created):
            with contextlib.suppress(OSError):
                (os.rmdir if path == directory else os.unlink)(path)
        raise
