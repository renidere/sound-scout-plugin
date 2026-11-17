"""Intake of one recording: probe, check the limits, decode into the run directory (R02, T01).

`probe` answers what the container declares: format, codec, sample rate, channels, duration and
size, through `ffprobe`. `decode` writes two files into the run directory and nothing else:
`source.json` (SHA-256 of the original, its metadata, the source ID, the limits applied and the
declared-vs-actual duration check) and `work/decoded.f32` — the first audio stream as raw
little-endian float32 PCM, interleaved, at the native sample rate and channel count. Nothing is
resampled, normalised or trimmed: sample `i` of channel `c` is time `i / sample_rate` of the
original, and a copy for ASR or listening is a separate derivative with its own record.

A path outside the policy (not under an allowed root, a denied name, a symlink) is refused by
`paths` with `PathDenied` and its `reason`; every refusal of the input itself or of the decoder
is an `IngestError` with a `code` from `REASON_CODES`. Both are raised before any file is
created, so a refused input leaves the run directory as `runs.open_run` made it; a failure
while writing removes what this call created. The limits come from `config.Budgets`: the size
of the original is checked on the open descriptor before anything reads it; the declared
duration and channel count after the probe; and the decoded samples are bounded by
`config.decoded_samples_limit` through `proc.run`'s output cap: the excess is drained and
dropped while the child keeps running under the wall-time budget, and the result is refused as
`decoded_samples_exceeded` rather than truncated, even when the child then ran out of time.
Each `ffprobe` and `ffmpeg` call runs under the subprocess wall time and memory budgets, and
`ffmpeg` runs with `-xerror`, so a decoding error anywhere in the stream fails the decode
instead of being skipped.

The original is read-only: it is opened through `paths.open_input` (descriptor, `O_RDONLY`),
hashed from that descriptor, and the decoder reads the resolved path; after decoding the file
is checked to be the one that was hashed (same inode, size, mtime and ctime, so a swap that
was renamed back or an mtime restored with `touch` still shows), otherwise the result is
refused as `input_changed`.
"""

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sound_scout import paths
from sound_scout.config import MIB, Config, decoded_samples_limit
from sound_scout.proc import RunResult, ffmpeg_argv, ffprobe_argv, run
from sound_scout.runs import Run

SOURCE_FILE_NAME = "source.json"
WORK_DIR_NAME = "work"
DECODED_FILE_NAME = "decoded.f32"
DECODED_FORMAT = "f32le"
BYTES_PER_SAMPLE = 4
DURATION_MISMATCH_TOLERANCE_S = 0.05
PROBE_OUTPUT_LIMIT = MIB
DIR_MODE = 0o700
_HASH_CHUNK = MIB
_RIFF_UNKNOWN_SIZES = (0, 0xFFFFFFFF)
_RIFF_MAX_CHUNKS = 64

INPUT_TOO_LARGE = "input_too_large"
NOT_AUDIO = "not_audio"
CORRUPTED = "corrupted"
TRUNCATED = "truncated"
EMPTY = "empty"
TOO_MANY_CHANNELS = "too_many_channels"
TOO_LONG = "too_long"
DECODED_SAMPLES_EXCEEDED = "decoded_samples_exceeded"
SUBPROCESS_TIMEOUT = "subprocess_timeout"
DECODER_UNAVAILABLE = "decoder_unavailable"
INPUT_CHANGED = "input_changed"
DERIVATIVES_EXIST = "derivatives_exist"
REASON_CODES = (
    INPUT_TOO_LARGE,
    NOT_AUDIO,
    CORRUPTED,
    TRUNCATED,
    EMPTY,
    TOO_MANY_CHANNELS,
    TOO_LONG,
    DECODED_SAMPLES_EXCEEDED,
    SUBPROCESS_TIMEOUT,
    DECODER_UNAVAILABLE,
    INPUT_CHANGED,
    DERIVATIVES_EXIST,
)


class IngestError(Exception):
    """The input is refused or the decoder failed; `code` is one of `REASON_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Probe:
    """What the container declares; `declared_duration_s` is None when it declares nothing."""

    container: str
    codec: str
    sample_rate: int
    channels: int
    declared_duration_s: float | None
    size_bytes: int


@dataclass(frozen=True)
class Source:
    """The decoded original as recorded in `source.json`; `duration_s` is from the samples."""

    id: str
    sha256: str
    probe: Probe
    frames: int
    duration_s: float
    duration_mismatch_s: float | None
    decoded_sha256: str
    decoded_path: Path
    record_path: Path

    @property
    def duration_mismatch(self) -> bool | None:
        if self.duration_mismatch_s is None:
            return None
        return abs(self.duration_mismatch_s) > DURATION_MISMATCH_TOLERANCE_S


def probe(path: str | os.PathLike[str], config: Config) -> Probe:
    """Probe an input: `paths.PathDenied` outside the policy, `IngestError` like `decode`."""
    resolved = paths.resolve_input(path, config)
    with paths.open_input(resolved, config) as file:
        return _probe_open(file, resolved, config)


def decode(path: str | os.PathLike[str], run: Run, config: Config) -> Source:
    """Decode an input into `run`: `source.json` and `work/decoded.f32`.

    Raises `paths.PathDenied` for a path outside the policy and `IngestError` for an input
    or decoder refusal, including derivatives already present in `run`.
    """
    resolved = paths.resolve_input(path, config)
    with paths.open_input(resolved, config) as file:
        opened = os.fstat(file.fileno())
        probed = _probe_open(file, resolved, config)
        sha256 = _sha256(file)
        samples = _decode_samples(resolved, probed, config)
        _require_unchanged(file, resolved, opened)
    frame_bytes = BYTES_PER_SAMPLE * probed.channels
    if len(samples) % frame_bytes:
        raise IngestError(CORRUPTED, "decoder output is not whole frames")
    frames = len(samples) // frame_bytes
    if frames == 0:
        raise IngestError(EMPTY, "decoder produced no samples")
    duration_s = frames / probed.sample_rate
    mismatch_s = None
    if probed.declared_duration_s is not None:
        mismatch_s = duration_s - probed.declared_duration_s
    source = Source(
        id=f"src-{sha256[:16]}",
        sha256=sha256,
        probe=probed,
        frames=frames,
        duration_s=duration_s,
        duration_mismatch_s=mismatch_s,
        decoded_sha256=hashlib.sha256(samples).hexdigest(),
        decoded_path=run.path / WORK_DIR_NAME / DECODED_FILE_NAME,
        record_path=run.path / SOURCE_FILE_NAME,
    )
    _write(source, samples, run, config)
    return source


def _probe_open(file: BinaryIO, resolved: Path, config: Config) -> Probe:
    size = os.fstat(file.fileno()).st_size
    limit = config.budgets.max_input_bytes
    if size > limit:
        raise IngestError(INPUT_TOO_LARGE, f"{size} bytes, the limit is {limit}")
    looks_like_wav = _riff_check(file, size)
    result = _run_decoder(
        ffprobe_argv(
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "a",
            str(resolved),
        ),
        config,
        max_output_bytes=PROBE_OUTPUT_LIMIT,
    )
    _require_in_time(result, "ffprobe", config)
    if result.returncode != 0 or result.truncated:
        code = CORRUPTED if looks_like_wav else NOT_AUDIO
        raise IngestError(code, f"ffprobe failed: {_stderr_tail(result)}")
    try:
        probed = _parse_probe(json.loads(result.stdout), size)
    except (ValueError, KeyError, TypeError) as error:
        raise IngestError(CORRUPTED, f"ffprobe output is not usable: {error}") from None
    if probed is None:
        raise IngestError(NOT_AUDIO, "no audio stream")
    if probed.channels > config.budgets.max_channels:
        raise IngestError(
            TOO_MANY_CHANNELS,
            f"{probed.channels} channels, the limit is {config.budgets.max_channels}",
        )
    max_duration = config.budgets.max_duration_seconds
    if probed.declared_duration_s is not None and probed.declared_duration_s > max_duration:
        raise IngestError(
            TOO_LONG,
            f"declared duration {probed.declared_duration_s} s, the limit is {max_duration} s",
        )
    return probed


def _parse_probe(data: dict[str, object], size: int) -> Probe | None:
    streams = data.get("streams") or []
    if not isinstance(streams, list) or not streams:
        return None
    stream, container = streams[0], data["format"]
    assert isinstance(stream, dict) and isinstance(container, dict)
    sample_rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    if sample_rate <= 0 or channels <= 0:
        raise ValueError(f"sample_rate {sample_rate}, channels {channels}")
    duration = stream.get("duration", container.get("duration"))
    return Probe(
        container=str(container["format_name"]),
        codec=str(stream["codec_name"]),
        sample_rate=sample_rate,
        channels=channels,
        declared_duration_s=None if duration is None else float(duration),
        size_bytes=size,
    )


def _riff_check(file: BinaryIO, size: int) -> bool:
    """Refuse a RIFF/WAVE file whose header or data chunk runs past the end; True when WAVE.

    ffprobe clamps the declared duration of a WAV to the bytes present, so a cut file looks
    like a short one to it; the container's own sizes are the only sign of the truncation.
    """
    file.seek(0)
    head = file.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return False
    riff_size = int.from_bytes(head[4:8], "little")
    if riff_size not in _RIFF_UNKNOWN_SIZES and riff_size + 8 > size:
        raise IngestError(TRUNCATED, f"RIFF header declares {riff_size + 8} bytes, file has {size}")
    offset = 12
    # A real WAV has a handful of chunks before `data`; a file of empty chunks is left to
    # ffprobe instead of being walked byte by byte.
    for _ in range(_RIFF_MAX_CHUNKS):
        if offset + 8 > size:
            break
        file.seek(offset)
        chunk = file.read(8)
        tag, length = chunk[:4], int.from_bytes(chunk[4:8], "little")
        if tag == b"data":
            available = size - offset - 8
            if length not in _RIFF_UNKNOWN_SIZES and length > available:
                raise IngestError(
                    TRUNCATED, f"data chunk declares {length} bytes, file has {available}"
                )
            return True
        offset += 8 + length + (length & 1)
    return True


def _decode_samples(resolved: Path, probed: Probe, config: Config) -> bytes:
    samples_limit = decoded_samples_limit(config.budgets, probed.sample_rate, probed.channels)
    result = _run_decoder(
        ffmpeg_argv(
            "-v",
            "error",
            "-xerror",
            "-i",
            str(resolved),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_f32le",
            "-f",
            DECODED_FORMAT,
            "-",
        ),
        config,
        max_output_bytes=samples_limit * BYTES_PER_SAMPLE,
    )
    if result.truncated:
        raise IngestError(
            DECODED_SAMPLES_EXCEEDED,
            f"more than {samples_limit} samples decoded "
            f"({config.budgets.max_duration_seconds} s at {probed.sample_rate} Hz, "
            f"{probed.channels} channels)",
        )
    _require_in_time(result, "ffmpeg", config)
    if result.returncode != 0:
        raise IngestError(CORRUPTED, f"ffmpeg failed: {_stderr_tail(result)}")
    return result.stdout


def _run_decoder(argv: list[str], config: Config, *, max_output_bytes: int) -> RunResult:
    budgets = config.budgets
    try:
        result = run(
            argv,
            timeout=budgets.max_subprocess_seconds,
            max_output_bytes=max_output_bytes,
            memory_bytes=budgets.max_subprocess_memory_mib * MIB,
        )
    except FileNotFoundError:
        raise IngestError(DECODER_UNAVAILABLE, f"{argv[0]} is not installed") from None
    return result


def _require_in_time(result: RunResult, program: str, config: Config) -> None:
    if result.timed_out:
        limit = config.budgets.max_subprocess_seconds
        raise IngestError(SUBPROCESS_TIMEOUT, f"{program} exceeded {limit} s")


def _require_unchanged(file: BinaryIO, resolved: Path, opened: os.stat_result) -> None:
    now = os.fstat(file.fileno())
    try:
        named = os.stat(resolved)
    except FileNotFoundError:
        raise IngestError(INPUT_CHANGED, "the input was removed while being decoded") from None
    same_file = (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino)
    same_content = (now.st_size, now.st_mtime_ns, now.st_ctime_ns) == (
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    if not (same_file and same_content):
        raise IngestError(INPUT_CHANGED, "the input changed while it was being decoded")


def _write(source: Source, samples: bytes, run: Run, config: Config) -> None:
    work_dir = paths.resolve_output(run.path, WORK_DIR_NAME, config)
    for existing in (work_dir / DECODED_FILE_NAME, run.path / SOURCE_FILE_NAME):
        if os.path.lexists(existing):
            raise IngestError(
                DERIVATIVES_EXIST, f"{existing.relative_to(run.path)} is already in the run"
            )
    created: list[Path] = []
    try:
        try:
            os.mkdir(work_dir, DIR_MODE)
        except FileExistsError:
            pass
        else:
            created.append(work_dir)
            os.chmod(work_dir, DIR_MODE)
        with paths.open_output(work_dir, DECODED_FILE_NAME, config) as out:
            created.append(work_dir / DECODED_FILE_NAME)
            out.write(samples)
        with paths.open_output(run.path, SOURCE_FILE_NAME, config) as out:
            created.append(run.path / SOURCE_FILE_NAME)
            out.write(_record(source, config))
    except BaseException:
        for path in reversed(created):
            with contextlib.suppress(OSError):
                (os.rmdir if path == work_dir else os.unlink)(path)
        raise


def _record(source: Source, config: Config) -> bytes:
    budgets = config.budgets
    probed = source.probe
    record = {
        "id": source.id,
        "sha256": source.sha256,
        "size_bytes": probed.size_bytes,
        "container": probed.container,
        "codec": probed.codec,
        "sample_rate": probed.sample_rate,
        "channels": probed.channels,
        "declared_duration_s": probed.declared_duration_s,
        "duration_s": source.duration_s,
        "frames": source.frames,
        "decoded": {
            "path": f"{WORK_DIR_NAME}/{DECODED_FILE_NAME}",
            "format": DECODED_FORMAT,
            "sample_rate": probed.sample_rate,
            "channels": probed.channels,
            "frames": source.frames,
            "sha256": source.decoded_sha256,
            "transformations": [],
        },
        "limits": {
            "max_input_bytes": budgets.max_input_bytes,
            "max_duration_seconds": budgets.max_duration_seconds,
            "max_channels": budgets.max_channels,
            "decoded_samples_limit": decoded_samples_limit(
                budgets, probed.sample_rate, probed.channels
            ),
            "max_subprocess_seconds": budgets.max_subprocess_seconds,
            "duration_mismatch_tolerance_s": DURATION_MISMATCH_TOLERANCE_S,
            "duration_mismatch_s": source.duration_mismatch_s,
            "duration_mismatch": source.duration_mismatch,
        },
    }
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(file: BinaryIO) -> str:
    file.seek(0)
    digest = hashlib.sha256()
    while chunk := file.read(_HASH_CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def _stderr_tail(result: RunResult) -> str:
    text = result.stderr.decode("utf-8", errors="replace").strip()
    return text.splitlines()[-1] if text else f"exit code {result.returncode}"
