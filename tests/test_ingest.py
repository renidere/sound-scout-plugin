import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from sound_scout import ingest, schema
from sound_scout.config import MAX_CHANNELS, MIB, Budgets, Config
from sound_scout.ingest import IngestError, Probe, decode, probe
from sound_scout.paths import PathDenied
from sound_scout.runs import Run, open_run
from tests.conftest import Encoder, require_encoders
from tests.test_proc import process_gone

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)
pytestmark = needs_ffmpeg

SAMPLE_RATE = 16_000
MP3_SAMPLE_RATE = 8_000
IDENTITY = {"audio": None, "model": None, "preprocessing": None, "ranges": None}


def sine(seconds: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write int16 samples shaped (frames,) or (frames, channels) as a PCM WAV."""
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(np.ascontiguousarray(samples, dtype=np.int16).tobytes())
    return path


def wav_header(frames: int, channels: int = 1, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A canonical 44-byte PCM16 WAV header declaring `frames` frames of data."""
    data_size = frames * channels * 2
    fmt = struct.pack(
        "<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16
    )
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVEfmt "
        + fmt
        + b"data"
        + struct.pack("<I", data_size)
    )


def write_two_bitrate_mp3(path: Path, head_seconds: float, tail_seconds: float) -> Path:
    """An MP3 whose declared duration is only an estimate: a 64 kbps head, an 8 kbps tail.

    Without a Xing/Info frame ffprobe estimates the duration from the bitrate of the first
    frames and the file size, so the head makes the estimate fall short of the audio present.
    """
    require_encoders("libmp3lame")
    parts = []
    for name, seconds, bitrate in (("head", head_seconds, "64k"), ("tail", tail_seconds, "8k")):
        part = path.with_suffix(f".{name}.mp3")
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                "-i", f"sine=frequency=220:sample_rate={MP3_SAMPLE_RATE}:duration={seconds}",
                "-ac", "1", "-c:a", "libmp3lame", "-b:a", bitrate,
                "-write_xing", "0", "-id3v2_version", "0", str(part),
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
        parts.append(part.read_bytes())
        part.unlink()
    path.write_bytes(b"".join(parts))
    return path


def write_damaged_mp3(path: Path) -> Path:
    """A 2 s MP3 with 600 bytes of garbage in the middle: frames before and after still decode."""
    require_encoders("libmp3lame")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=220:sample_rate={MP3_SAMPLE_RATE}:duration=2",
            "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k",
            "-write_xing", "0", "-id3v2_version", "0", str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    data = bytearray(path.read_bytes())
    middle = len(data) // 2
    data[middle : middle + 600] = b"\xaa" * 600
    path.write_bytes(data)
    return path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir_entries(run: Run) -> set[str]:
    return {entry.name for entry in run.path.iterdir()}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "recordings"
    root.mkdir()
    return root


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    data_root.mkdir()
    return data_root


@pytest.fixture
def config(root: Path, data_root: Path) -> Config:
    return Config(allowed_roots=(root,), data_root=data_root)


@pytest.fixture
def run(data_root: Path) -> Run:
    return open_run("req-1", IDENTITY, data_root)


@pytest.fixture
def tone(root: Path) -> Path:
    return write_wav(root / "tone.wav", sine(1.0))


@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """Put a shell script named `ffmpeg` first on PATH; ffprobe stays the real one."""

    def _install(script: str) -> Path:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        exe = bin_dir / "ffmpeg"
        exe.write_text(f"#!/bin/sh\n{script}\n")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        return exe

    return _install


@pytest.fixture
def declare_duration(monkeypatch: pytest.MonkeyPatch) -> Callable[[float | None], None]:
    """Make the probe report `seconds` as the declared duration; everything else stays real."""

    def _declare(seconds: float | None) -> None:
        real = ingest._parse_probe

        def lying(data: dict[str, object], size: int) -> Probe | None:
            probed = real(data, size)
            assert probed is not None
            return Probe(**{**probed.__dict__, "declared_duration_s": seconds})

        monkeypatch.setattr(ingest, "_parse_probe", lying)

    return _declare


# probe


def test_probe_reports_what_the_container_declares(tone: Path, config: Config) -> None:
    assert probe(tone, config) == Probe(
        container="wav",
        codec="pcm_s16le",
        sample_rate=SAMPLE_RATE,
        channels=1,
        declared_duration_s=1.0,
        size_bytes=tone.stat().st_size,
    )


def test_probe_takes_only_paths_inside_the_allowed_roots(tmp_path: Path, config: Config) -> None:
    outside = write_wav(tmp_path / "outside.wav", sine(0.1))
    with pytest.raises(PathDenied, match="outside allowed roots"):
        probe(outside, config)


# decode: the happy path


def test_decode_writes_source_json_and_decoded_f32(tone: Path, run: Run, config: Config) -> None:
    source = decode(tone, run, config)

    assert source.record_path == run.path / "source.json"
    assert source.decoded_path == run.path / "work" / "decoded.f32"
    assert source.sha256 == sha256_of(tone)
    assert source.id == f"src-{source.sha256[:16]}"
    assert source.frames == SAMPLE_RATE
    assert source.duration_s == 1.0
    assert source.duration_mismatch_s == 0.0
    assert source.duration_mismatch is False
    assert source.decoded_path.stat().st_size == SAMPLE_RATE * 4
    assert stat.S_IMODE(source.decoded_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(source.record_path.stat().st_mode) == 0o600

    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert record["id"] == source.id
    assert record["sha256"] == source.sha256
    assert record["size_bytes"] == tone.stat().st_size
    assert (record["container"], record["codec"]) == ("wav", "pcm_s16le")
    assert (record["sample_rate"], record["channels"]) == (SAMPLE_RATE, 1)
    assert (record["declared_duration_s"], record["duration_s"]) == (1.0, 1.0)
    assert record["frames"] == SAMPLE_RATE
    assert record["decoded"] == {
        "path": "work/decoded.f32",
        "format": "f32le",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "frames": SAMPLE_RATE,
        "sha256": sha256_of(source.decoded_path),
        "transformations": [],
    }
    assert record["limits"] == {
        "max_input_bytes": 100 * MIB,
        "max_duration_seconds": 180,
        "max_channels": 2,
        "decoded_samples_limit": 180 * SAMPLE_RATE,
        "max_subprocess_seconds": 60,
        "duration_mismatch_tolerance_s": 0.05,
        "duration_mismatch_s": 0.0,
        "duration_mismatch": False,
    }
    assert str(tone) not in source.record_path.read_text(encoding="utf-8")


def test_source_json_limits_share_their_names_with_the_result_contract(
    root: Path, run: Run, config: Config
) -> None:
    """docs/decisions/0002: ingest-only limits are the per-subprocess budget and the sample cap."""
    tone = write_wav(root / "tone.wav", sine(1.0))
    source = decode(tone, run, config)
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    result_limits = schema.SCHEMA["properties"]["limits"]["properties"]
    assert set(record["limits"]) - set(result_limits) == {
        "decoded_samples_limit",
        "max_subprocess_seconds",
    }
    assert set(result_limits) - set(record["limits"]) == {"worker_deadline_seconds"}


def test_decode_keeps_the_time_scale_of_the_original(root: Path, run: Run, config: Config) -> None:
    impulse_at = SAMPLE_RATE // 2
    samples = np.zeros(SAMPLE_RATE, dtype=np.int16)
    samples[impulse_at] = 32767
    samples[impulse_at + 1 :] = sine(0.5)[: SAMPLE_RATE - impulse_at - 1]
    path = write_wav(root / "late.wav", samples)

    source = decode(path, run, config)
    decoded = np.fromfile(source.decoded_path, dtype="<f4")

    assert len(decoded) == SAMPLE_RATE
    assert not decoded[:impulse_at].any()
    assert decoded[impulse_at] == pytest.approx(32767 / 32768)
    assert np.allclose(decoded, samples / 32768, atol=1e-6)


def test_decode_keeps_native_rate_and_channels_without_normalising(
    root: Path, run: Run, config: Config
) -> None:
    left = sine(0.5, sample_rate=44_100, amplitude=0.25)
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    path = write_wav(root / "stereo.wav", stereo, sample_rate=44_100)

    source = decode(path, run, config)
    decoded = np.fromfile(source.decoded_path, dtype="<f4").reshape(-1, 2)

    assert (source.probe.sample_rate, source.probe.channels) == (44_100, 2)
    assert source.frames == len(left)
    assert not decoded[:, 1].any()
    assert np.allclose(decoded[:, 0], left / 32768, atol=1e-6)
    assert decoded.max() == pytest.approx(0.25, abs=1e-3)


def test_original_is_opened_read_only_and_left_unchanged(
    tone: Path, run: Run, config: Config
) -> None:
    tone.chmod(0o444)
    before = (sha256_of(tone), tone.stat().st_mtime_ns, tone.stat().st_size)

    decode(tone, run, config)

    assert (sha256_of(tone), tone.stat().st_mtime_ns, tone.stat().st_size) == before
    assert stat.S_IMODE(tone.stat().st_mode) == 0o444


# the formats of R02: judged by content, one audio stream, padding within 30 ms


ENCODED_FORMATS = [
    pytest.param("mp3", "libmp3lame", "mp3", "mp3", SAMPLE_RATE, id="mp3"),
    pytest.param("ogg", "libopus", "ogg", "opus", 48_000, id="ogg-opus"),
    pytest.param("m4a", "aac", "mov,mp4,m4a,3gp,3g2,mj2", "aac", SAMPLE_RATE, id="m4a-aac"),
]


@pytest.mark.parametrize(("suffix", "encoder", "container", "codec", "rate"), ENCODED_FORMATS)
def test_encoded_formats_decode_within_30_ms_of_the_wav_original(
    tone: Path,
    encode: Encoder,
    run: Run,
    config: Config,
    suffix: str,
    encoder: str,
    container: str,
    codec: str,
    rate: int,
) -> None:
    """Opus is always carried at 48 kHz; the AAC decoder pads the signal to whole frames."""
    encoded = encode(tone, tone.with_suffix(f".{suffix}"), encoder)
    probed = probe(encoded, config)
    assert (probed.container, probed.codec, probed.sample_rate, probed.channels) == (
        container,
        codec,
        rate,
        1,
    )
    assert probed.declared_duration_s is not None

    source = decode(encoded, run, config)

    assert source.sha256 == sha256_of(encoded)
    assert source.duration_s == source.frames / rate
    with wave.open(str(tone), "rb") as original:
        original_duration_s = original.getnframes() / original.getframerate()
    assert abs(source.frames / rate - original_duration_s) <= 0.03
    assert source.decoded_path.stat().st_size == source.frames * 4
    assert source.duration_mismatch_s == pytest.approx(
        source.duration_s - probed.declared_duration_s
    )
    assert source.duration_mismatch is False
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert (record["container"], record["codec"]) == (container, codec)
    assert (record["sample_rate"], record["duration_s"]) == (rate, source.duration_s)
    assert record["limits"]["duration_mismatch_s"] == source.duration_mismatch_s
    assert record["limits"]["duration_mismatch"] is False
    assert abs(record["limits"]["duration_mismatch_s"]) <= 0.03


def test_supported_formats_is_the_r02_list() -> None:
    assert set(ingest.SUPPORTED_FORMATS) == {"wav", "mp3", "ogg", "mov,mp4,m4a,3gp,3g2,mj2"}
    assert ingest.SUPPORTED_FORMATS["mp3"] == {"mp3"}
    assert ingest.SUPPORTED_FORMATS["ogg"] == {"opus"}
    assert ingest.SUPPORTED_FORMATS["mov,mp4,m4a,3gp,3g2,mj2"] == {"aac"}
    assert all(codec.startswith("pcm_") for codec in ingest.SUPPORTED_FORMATS["wav"])
    assert "pcm_s16le" in ingest.SUPPORTED_FORMATS["wav"]


def test_the_extension_is_not_the_source_of_truth(
    root: Path, data_root: Path, config: Config, encode: Encoder
) -> None:
    """A WAV named `.mp3` is decoded as the WAV it is; an MP3 named `.wav` as an MP3."""
    wav_as_mp3 = write_wav(root / "voice.mp3", sine(1.0))
    assert (probe(wav_as_mp3, config).container, probe(wav_as_mp3, config).codec) == (
        "wav",
        "pcm_s16le",
    )
    source = decode(wav_as_mp3, open_run("req-wav", IDENTITY, data_root), config)
    assert (source.probe.container, source.probe.codec, source.frames) == (
        "wav",
        "pcm_s16le",
        SAMPLE_RATE,
    )
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert (record["container"], record["codec"]) == ("wav", "pcm_s16le")

    mp3_as_wav = encode(wav_as_mp3, root / "encoded.mp3", "libmp3lame").rename(root / "voice.wav")
    source = decode(mp3_as_wav, open_run("req-mp3", IDENTITY, data_root), config)
    assert (source.probe.container, source.probe.codec, source.frames) == (
        "mp3",
        "mp3",
        SAMPLE_RATE,
    )
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert (record["container"], record["codec"]) == ("mp3", "mp3")


# refusals: a reason code, nothing written


def assert_refused(path: Path, run: Run, config: Config, code: str, match: str) -> None:
    before = run_dir_entries(run)
    with pytest.raises(IngestError, match=match) as info:
        decode(path, run, config)
    assert info.value.code == code
    assert code in ingest.REASON_CODES
    assert run_dir_entries(run) == before == {"request.json", "manifest.json"}


def test_refuses_a_file_over_the_size_budget(root: Path, run: Run, config: Config) -> None:
    small = Config(config.allowed_roots, config.data_root, Budgets(max_input_bytes=1000))
    path = write_wav(root / "big.wav", sine(0.1))
    assert path.stat().st_size > 1000

    assert_refused(path, run, small, "input_too_large", "the limit is 1000")
    assert Budgets().max_input_bytes == 100 * MIB


def test_refuses_a_file_over_100_mib_with_the_default_budget(
    root: Path, run: Run, config: Config
) -> None:
    path = root / "huge.wav"
    with path.open("wb") as out:
        out.write(wav_header(frames=0))
        out.truncate(100 * MIB + 1)

    assert_refused(
        path, run, config, "input_too_large", f"{100 * MIB + 1} bytes, the limit is {100 * MIB}"
    )


def test_a_file_of_exactly_100_mib_passes_the_size_check(
    root: Path, run: Run, config: Config
) -> None:
    """The limit is "over 100 MiB": at the limit the size is fine and the length is what refuses."""
    path = root / "limit.wav"
    data_size = 100 * MIB - len(wav_header(frames=0))
    with path.open("wb") as out:
        out.write(wav_header(frames=data_size // 2))
        out.truncate(100 * MIB)

    assert_refused(path, run, config, "too_long", "declared duration 3276.79")


def test_refuses_a_declared_duration_over_180_s(root: Path, run: Run, config: Config) -> None:
    path = write_wav(root / "long.wav", np.zeros(181 * 8_000, dtype=np.int16), sample_rate=8_000)

    assert_refused(path, run, config, "too_long", "declared duration 181.0 s, the limit is 180 s")


def test_refuses_more_than_two_channels(root: Path, run: Run, config: Config) -> None:
    three = np.stack([sine(0.1)] * 3, axis=1)
    path = write_wav(root / "three.wav", three)

    assert_refused(path, run, config, "too_many_channels", "3 channels, the limit is 2")


def test_three_channels_are_refused_at_ingest_before_the_channel_report(
    root: Path, run: Run, config: Config
) -> None:
    """#49: with `max_channels` at the version limit a wider file never reaches `channels`."""
    at_limit = Config(config.allowed_roots, config.data_root, Budgets(max_channels=MAX_CHANNELS))
    path = write_wav(root / "three.wav", np.column_stack([sine(0.1)] * 3))

    assert_refused(
        path, run, at_limit, "too_many_channels", f"3 channels, the limit is {MAX_CHANNELS}"
    )
    assert not (run.path / "work").exists()


def test_refuses_a_file_that_is_not_audio(root: Path, run: Run, config: Config) -> None:
    path = root / "notes.wav"
    path.write_bytes(b"hello world " * 100)

    assert_refused(path, run, config, "not_audio", "ffprobe failed")


def test_refuses_a_wav_truncated_in_the_middle(tone: Path, run: Run, config: Config) -> None:
    data = tone.read_bytes()
    tone.write_bytes(data[: len(data) // 2])

    assert_refused(tone, run, config, "truncated", "declares 32044 bytes, file has 16022")


def test_refuses_a_header_declaring_more_than_the_file_holds(
    root: Path, run: Run, config: Config
) -> None:
    path = root / "header-only.wav"
    path.write_bytes(wav_header(frames=181 * SAMPLE_RATE))

    assert_refused(path, run, config, "truncated", "declares 5792044 bytes, file has 44")


def test_refuses_a_corrupted_wav_header(root: Path, run: Run, config: Config) -> None:
    path = root / "broken.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", 96) + b"WAVE" + bytes(range(92)))

    assert_refused(path, run, config, "corrupted", "ffprobe failed")


def test_refuses_an_mp3_damaged_in_the_middle_instead_of_skipping_the_damage(
    root: Path, run: Run, config: Config
) -> None:
    """Without `-xerror` ffmpeg logs the bad frames, exits 0 and hands over a shorter signal."""
    path = write_damaged_mp3(root / "damaged.mp3")
    assert probe(path, config).codec == "mp3"

    assert_refused(path, run, config, "corrupted", "ffmpeg failed: .*Invalid data")


@pytest.mark.parametrize(
    ("name", "encoder", "reason"),
    [
        pytest.param("tone.flac", "flac", "container 'flac' is not supported", id="flac"),
        pytest.param("tone.webm", "libopus", "container 'matroska,webm' is not", id="webm"),
        pytest.param("tone.ogg", "libvorbis", "codec 'vorbis' in container 'ogg'", id="ogg-vorbis"),
        pytest.param("tone.wav", "adpcm_ms", "codec 'adpcm_ms' in container 'wav'", id="wav-adpcm"),
    ],
)
def test_refuses_a_container_or_codec_outside_the_table(
    root: Path, run: Run, config: Config, encode: Encoder, name: str, encoder: str, reason: str
) -> None:
    path = encode(write_wav(root / "original.wav", sine(1.0)), root / name, encoder)

    with pytest.raises(IngestError, match=reason) as info:
        probe(path, config)
    assert info.value.code == "unsupported_format"
    assert_refused(path, run, config, "unsupported_format", reason)


def test_refuses_two_audio_streams(root: Path, run: Run, config: Config, encode: Encoder) -> None:
    wav = write_wav(root / "original.wav", sine(1.0))
    path = encode(wav, root / "two.m4a", "aac", "-i", str(wav), "-map", "0:a", "-map", "1:a")

    assert_refused(path, run, config, "unsupported_format", "2 audio streams, one is required")


def test_refuses_a_video_container_with_audio(
    root: Path, run: Run, config: Config, encode: Encoder
) -> None:
    require_encoders("mpeg4")
    wav = write_wav(root / "original.wav", sine(1.0))
    path = encode(
        wav, root / "clip.mp4", "aac",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
        "-map", "0:a", "-map", "1:v", "-c:v", "mpeg4", "-shortest",
    )  # fmt: skip

    assert_refused(
        path, run, config, "unsupported_format", r"video stream \(mpeg4\) next to the audio"
    )


@pytest.mark.parametrize(
    ("suffix", "encoder", "codec"), [("mp3", "libmp3lame", "mp3"), ("m4a", "aac", "aac")]
)
def test_embedded_cover_art_is_not_a_video_stream(
    root: Path, run: Run, config: Config, encode: Encoder, suffix: str, encoder: str, codec: str
) -> None:
    """Opus carries its cover in a tag, not a stream, so only MP3 and M4A can have this picture."""
    require_encoders("mjpeg")
    wav = write_wav(root / "original.wav", sine(1.0))
    path = encode(
        wav, root / f"with-cover.{suffix}", encoder,
        "-f", "lavfi", "-i", "color=c=red:s=16x16:r=1", "-map", "0:a", "-map", "1:v",
        "-frames:v", "1", "-c:v", "mjpeg", "-disposition:v", "attached_pic",
    )  # fmt: skip
    streams = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["streams"]
    assert [stream["codec_type"] for stream in streams] == ["audio", "video"]

    probed = probe(path, config)
    source = decode(path, run, config)

    assert (probed.codec, probed.channels) == (codec, 1)
    assert abs(source.duration_s - 1.0) <= 0.03
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert record["codec"] == codec


def test_a_video_only_file_is_not_audio(root: Path, run: Run, config: Config) -> None:
    require_encoders("mpeg4")
    path = root / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
            "-c:v", "mpeg4", str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip

    assert_refused(path, run, config, "not_audio", "no audio stream")


def test_refuses_a_wav_of_empty_chunks_without_walking_it_byte_by_byte(
    root: Path, run: Run, config: Config
) -> None:
    path = root / "zeros.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + MIB) + b"WAVE" + bytes(MIB))

    assert_refused(path, run, config, "corrupted", "ffprobe failed")


def test_refuses_an_empty_decode(root: Path, run: Run, config: Config) -> None:
    path = root / "empty.wav"
    path.write_bytes(wav_header(frames=0))

    assert_refused(path, run, config, "empty", "no samples")


def test_derivatives_go_only_into_a_run_dir_under_the_data_root(
    tone: Path, run: Run, config: Config, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    stray = Run(run.run_id, elsewhere, run.request_id, run.identity_hash, False, False, True)

    with pytest.raises(PathDenied, match="outside data root"):
        decode(tone, stray, config)

    assert not (elsewhere / "work").exists()
    assert {entry.name for entry in elsewhere.iterdir()} == set()


def test_work_dir_is_private(tone: Path, run: Run, config: Config) -> None:
    source = decode(tone, run, config)

    assert stat.S_IMODE(source.decoded_path.parent.stat().st_mode) == 0o700


def test_existing_derivatives_in_the_run_are_refused_not_overwritten(
    tone: Path, run: Run, config: Config
) -> None:
    source = decode(tone, run, config)
    before = (sha256_of(source.decoded_path), source.record_path.read_bytes())

    with pytest.raises(IngestError, match="work/decoded.f32 is already in the run") as info:
        decode(tone, run, config)

    assert info.value.code == "derivatives_exist"
    assert (sha256_of(source.decoded_path), source.record_path.read_bytes()) == before


def test_a_failed_write_leaves_no_orphan_derivative(
    tone: Path, run: Run, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open_output = ingest.paths.open_output

    def failing(run_dir: Path, name: str, config: Config):
        if name == "source.json":
            raise OSError("no space left on device")
        return real_open_output(run_dir, name, config)

    monkeypatch.setattr(ingest.paths, "open_output", failing)

    with pytest.raises(OSError, match="no space left"):
        decode(tone, run, config)

    assert run_dir_entries(run) == {"request.json", "manifest.json"}


def test_a_failed_write_keeps_a_work_dir_it_did_not_create(
    tone: Path, run: Run, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run.path / "work").mkdir(0o700)
    monkeypatch.setattr(ingest.paths, "open_output", lambda *args: (_ := 1 / 0))

    with pytest.raises(ZeroDivisionError):
        decode(tone, run, config)

    assert run_dir_entries(run) == {"request.json", "manifest.json", "work"}
    assert list((run.path / "work").iterdir()) == []


# the budgets of #14 bound the decoder


def test_decoder_timeout_is_reported_with_the_budget(
    tone: Path, run: Run, config: Config, fake_ffmpeg: Callable[[str], Path], tmp_path: Path
) -> None:
    pid_file = tmp_path / "pid"
    fake_ffmpeg(f"echo $$ > {pid_file}\nexec sleep 30")
    slow = Config(config.allowed_roots, config.data_root, Budgets(max_subprocess_seconds=1))

    assert_refused(tone, run, slow, "subprocess_timeout", "ffmpeg exceeded 1 s")
    assert process_gone(int(pid_file.read_text()))


def test_an_oversized_slow_decode_is_refused_for_its_samples_not_its_time(
    tone: Path, run: Run, config: Config, fake_ffmpeg: Callable[[str], Path], tmp_path: Path
) -> None:
    """The cap is hit first; the child then idles until the wall time kills it."""
    pid_file = tmp_path / "pid"
    fake_ffmpeg(f"echo $$ > {pid_file}\nhead -c {2 * SAMPLE_RATE * 4} /dev/zero\nexec sleep 30")
    tight = Config(
        config.allowed_roots,
        config.data_root,
        Budgets(max_duration_seconds=1, max_subprocess_seconds=1),
    )

    assert_refused(
        tone, run, tight, "decoded_samples_exceeded", f"more than {SAMPLE_RATE} samples decoded"
    )
    assert process_gone(int(pid_file.read_text()))


def test_decoded_samples_over_the_budget_are_refused_not_truncated(
    root: Path, run: Run, config: Config, declare_duration: Callable[[float | None], None]
) -> None:
    """The container claims 1 s, the real decoder produces 2 s: refused, not cut to 1 s."""
    path = write_wav(root / "two-seconds.wav", sine(2.0))
    declare_duration(1.0)
    short = Config(config.allowed_roots, config.data_root, Budgets(max_duration_seconds=1))

    assert_refused(
        path, run, short, "decoded_samples_exceeded", f"more than {SAMPLE_RATE} samples decoded"
    )


def test_refuses_an_actual_decode_over_180_s_with_the_default_budget(
    root: Path, run: Run, config: Config
) -> None:
    """ffprobe declares about 67 s for this file; the real decode yields over 186 s."""
    path = write_two_bitrate_mp3(root / "long.mp3", head_seconds=1.0, tail_seconds=185.0)
    declared = probe(path, config).declared_duration_s
    assert declared is not None and declared < 180

    assert_refused(
        path,
        run,
        config,
        "decoded_samples_exceeded",
        f"more than {180 * MP3_SAMPLE_RATE} samples decoded",
    )


def test_decoded_samples_exactly_at_the_budget_pass(tone: Path, run: Run, config: Config) -> None:
    exact = Config(config.allowed_roots, config.data_root, Budgets(max_duration_seconds=1))

    assert decode(tone, run, exact).frames == SAMPLE_RATE


def test_decoder_failure_after_a_good_probe_is_corrupted(
    tone: Path, run: Run, config: Config, fake_ffmpeg: Callable[[str], Path]
) -> None:
    fake_ffmpeg("echo 'Invalid data found when processing input' >&2\nexit 1")

    assert_refused(tone, run, config, "corrupted", "ffmpeg failed: Invalid data")


def test_missing_decoder_is_unavailable(
    tone: Path, run: Run, config: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    assert_refused(tone, run, config, "decoder_unavailable", "ffprobe is not installed")


def test_input_replaced_during_decoding_is_refused(
    tone: Path, run: Run, config: Config, fake_ffmpeg: Callable[[str], Path]
) -> None:
    replacement = write_wav(tone.parent / "other.wav", sine(1.0, amplitude=0.1))
    fake_ffmpeg(f"mv {replacement} {tone}\nhead -c {SAMPLE_RATE * 4} /dev/zero")

    assert_refused(tone, run, config, "input_changed", "changed while")


def test_input_swapped_and_restored_during_decoding_is_refused(
    tone: Path, run: Run, config: Config, fake_ffmpeg: Callable[[str], Path]
) -> None:
    """The decoder read another file; the original is back under its name with its mtime."""
    other = write_wav(tone.parent / "other.wav", sine(1.0, amplitude=0.1))
    aside = tone.parent / "aside.wav"
    fake_ffmpeg(
        f"mv {tone} {aside}\nmv {other} {tone}\nhead -c {SAMPLE_RATE * 4} /dev/zero\n"
        f"mv {aside} {tone}"
    )
    before = (tone.stat().st_ino, tone.stat().st_size, tone.stat().st_mtime_ns)

    assert_refused(tone, run, config, "input_changed", "changed while")
    assert (tone.stat().st_ino, tone.stat().st_size, tone.stat().st_mtime_ns) == before


def test_input_removed_during_decoding_is_refused(
    tone: Path, run: Run, config: Config, fake_ffmpeg: Callable[[str], Path]
) -> None:
    fake_ffmpeg(f"rm {tone}\nhead -c {SAMPLE_RATE * 4} /dev/zero")

    assert_refused(tone, run, config, "input_changed", "removed while")


# declared-vs-actual duration


def test_duration_mismatch_over_50_ms_is_flagged_in_source_json_and_limits(
    tone: Path, run: Run, config: Config, declare_duration: Callable[[float | None], None]
) -> None:
    declare_duration(1.5)

    source = decode(tone, run, config)

    assert (source.probe.declared_duration_s, source.duration_s) == (1.5, 1.0)
    assert source.duration_mismatch_s == -0.5
    assert source.duration_mismatch is True
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert (record["declared_duration_s"], record["duration_s"]) == (1.5, 1.0)
    assert record["limits"]["duration_mismatch_s"] == -0.5
    assert record["limits"]["duration_mismatch"] is True


def test_duration_mismatch_over_50_ms_is_flagged_for_a_really_decoded_file(
    root: Path, run: Run, config: Config
) -> None:
    path = write_two_bitrate_mp3(root / "mixed.mp3", head_seconds=1.0, tail_seconds=4.0)

    source = decode(path, run, config)

    declared = source.probe.declared_duration_s
    assert declared is not None
    assert source.duration_s == source.frames / MP3_SAMPLE_RATE
    assert source.duration_s - declared > 0.5
    assert source.duration_mismatch_s == pytest.approx(source.duration_s - declared)
    assert source.duration_mismatch is True
    assert source.decoded_path.stat().st_size == source.frames * 4
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert (record["container"], record["codec"]) == ("mp3", "mp3")
    assert (record["declared_duration_s"], record["duration_s"]) == (declared, source.duration_s)
    assert record["limits"]["duration_mismatch_s"] == source.duration_mismatch_s
    assert record["limits"]["duration_mismatch"] is True


def test_duration_mismatch_within_50_ms_is_not_flagged(
    tone: Path, run: Run, config: Config, declare_duration: Callable[[float | None], None]
) -> None:
    declare_duration(1.04)

    source = decode(tone, run, config)

    assert source.duration_mismatch_s == pytest.approx(-0.04)
    assert source.duration_mismatch is False
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert record["limits"]["duration_mismatch"] is False


def test_unknown_declared_duration_leaves_the_mismatch_null(
    tone: Path, run: Run, config: Config, declare_duration: Callable[[float | None], None]
) -> None:
    declare_duration(None)

    source = decode(tone, run, config)

    assert source.duration_mismatch_s is None
    assert source.duration_mismatch is None
    record = json.loads(source.record_path.read_text(encoding="utf-8"))
    assert record["declared_duration_s"] is None
    assert record["limits"]["duration_mismatch"] is None
