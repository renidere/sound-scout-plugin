"""`clips`: clips on the source timescale, listening copies with recorded gain, result artifacts."""

import hashlib
import io
import json
import math
import shutil
import stat
import wave
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sound_scout import clips, ingest, schema
from sound_scout.clips import ClipError, cut, listening_copy
from sound_scout.config import Config
from sound_scout.runs import Run, open_run
from tests.test_ingest import write_wav
from tests.test_schema import EXAMPLE_DOC

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)

RATE = 16_000
DURATION_S = 2.5
ONSET_S = 1.25
AMPLITUDE = 0.25
IDENTITY = {"audio": None, "model": None, "preprocessing": None, "ranges": None}
FULL_SCALE = 2**31


def burst(seconds: float = DURATION_S, onset: float = ONSET_S) -> np.ndarray:
    """Digital silence, then a 440 Hz tone from `onset` to the end; int16 at `RATE`."""
    t = np.arange(int(seconds * RATE)) / RATE
    signal = np.zeros(len(t))
    start = int(onset * RATE)
    signal[start:] = AMPLITUDE * np.sin(2 * np.pi * 440.0 * t[start:])
    return (signal * 32767).astype(np.int16)


def read_wav(path: Path) -> tuple[np.ndarray, wave._wave_params]:
    """Frames × channels float samples of a 32-bit PCM WAV and its parameters."""
    with wave.open(io.BytesIO(path.read_bytes()), "rb") as wav:
        params = wav.getparams()
        pcm = np.frombuffer(wav.readframes(params.nframes), dtype="<i4")
    return pcm.reshape(params.nframes, params.nchannels) / FULL_SCALE, params


def decoded(run: Run) -> np.ndarray:
    source = json.loads((run.path / "source.json").read_text(encoding="utf-8"))
    samples = np.fromfile(run.path / "work" / "decoded.f32", dtype="<f4")
    return samples.reshape(source["frames"], source["channels"])


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries(run: Run, name: str) -> list[str]:
    directory = run.path / name
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def peak_dbfs(samples: np.ndarray) -> float:
    return 20 * math.log10(float(np.max(np.abs(samples))))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "recordings"
    root.mkdir()
    return root


@pytest.fixture
def config(root: Path, tmp_path: Path) -> Config:
    data_root = tmp_path / "data"
    data_root.mkdir()
    return Config(allowed_roots=(root,), data_root=data_root)


@pytest.fixture
def run(config: Config) -> Run:
    return open_run("req-1", IDENTITY, config.data_root)


@pytest.fixture
def decoded_run(root: Path, run: Run, config: Config) -> Run:
    ingest.decode(write_wav(root / "burst.wav", burst(), RATE), run, config)
    return run


# --- cut --------------------------------------------------------------------------------------


def test_cut_writes_the_clip_from_the_source_signal_with_its_record(
    decoded_run: Run, config: Config
) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)

    first, last = round(0.85 * RATE), round(1.65 * RATE)
    assert clip.id == f"clip-{first}-{last}" and clip.kind == "clip" and clip.reused is False
    assert clip.path == decoded_run.path / "clips" / f"{clip.id}.wav"
    assert clip.record_path == decoded_run.path / "clips" / f"{clip.id}.json"
    assert entries(decoded_run, "clips") == [f"{clip.id}.json", f"{clip.id}.wav"]
    assert stat.S_IMODE((decoded_run.path / "clips").stat().st_mode) == 0o700
    assert stat.S_IMODE(clip.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(clip.record_path.stat().st_mode) == 0o600

    samples, params = read_wav(clip.path)
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 4, RATE)
    assert params.nframes == last - first
    assert np.array_equal(samples.astype(np.float32), decoded(decoded_run)[first:last])
    assert abs(float(np.max(np.abs(samples))) - AMPLITUDE) < 0.001
    onset_in_clip = int(np.argmax(np.abs(samples[:, 0]) > 0.01)) / RATE
    assert abs(onset_in_clip - (ONSET_S - 0.85)) < 1 / RATE

    record = json.loads(clip.record_path.read_text(encoding="utf-8"))
    assert record == clip.record
    assert record["source_start"] == clip.source_start == first / RATE
    assert record["source_end"] == clip.source_end == last / RATE
    assert (record["requested_start"], record["requested_end"]) == (1.0, 1.5)
    assert record["pad_ms"] == 150
    assert record["sha256"] == clip.sha256 == sha256_of(clip.path)
    assert record["gain_db"] == 0.0 and record["for_listening_only"] is False
    assert record["clipped_samples"] == 0
    assert record["path"] == f"clips/{clip.id}.wav" and record["run_id"] == decoded_run.run_id
    assert (record["sample_format"], record["sample_rate"], record["channels"]) == (
        "pcm_s32le",
        RATE,
        1,
    )
    assert record["frames"] == last - first and record["duration_s"] == (last - first) / RATE
    source = json.loads((decoded_run.path / "source.json").read_text(encoding="utf-8"))
    assert record["source"] == {
        "id": source["id"],
        "sha256": source["sha256"],
        "decoded_sha256": source["decoded"]["sha256"],
        "sample_rate": RATE,
        "channels": 1,
        "frames": source["frames"],
        "duration_s": DURATION_S,
    }


def test_padding_clamps_to_the_recording_boundary(decoded_run: Run, config: Config) -> None:
    head = cut(decoded_run, 0.0, 0.1, config=config)
    tail = cut(decoded_run, DURATION_S - 0.1, DURATION_S, config=config)
    unpadded = cut(decoded_run, 1.0, 1.5, pad_ms=0, config=config)

    assert head.source_start == 0.0 and head.source_end == 0.25
    assert tail.source_start == DURATION_S - 0.25 and tail.source_end == DURATION_S
    assert (unpadded.source_start, unpadded.source_end) == (1.0, 1.5)
    assert read_wav(tail.path)[1].nframes == round(0.25 * RATE)


def test_clip_samples_are_the_original_input_samples_without_gain(
    root: Path, run: Run, config: Config
) -> None:
    """Criterion 1: the WAV on disk holds the input file's own samples, no amplification."""
    original = burst()
    ingest.decode(write_wav(root / "burst.wav", original, RATE), run, config)

    clip = cut(run, 1.0, 1.5, config=config)

    samples, _ = read_wav(clip.path)
    first, last = round(clip.source_start * RATE), round(clip.source_end * RATE)
    expected = original[first:last].astype(np.float64) / 32768
    assert samples.shape == (last - first, 1)
    assert np.allclose(samples[:, 0], expected, atol=1 / 65536, rtol=0)
    assert abs(float(np.max(np.abs(samples))) - float(np.max(np.abs(expected)))) < 1 / 65536
    assert clip.record["gain_db"] == 0.0


def test_edge_clips_hold_the_first_and_last_source_frames(decoded_run: Run, config: Config) -> None:
    """Criterion 1: padding at the recording edges clamps to [0, duration], never crosses it."""
    source = decoded(decoded_run)
    pad = round(0.15 * RATE)

    head = cut(decoded_run, 0.0, 0.1, config=config)
    tail = cut(decoded_run, DURATION_S - 0.1, DURATION_S, config=config)

    head_samples, head_params = read_wav(head.path)
    tail_samples, tail_params = read_wav(tail.path)
    assert head_params.nframes == round(0.1 * RATE) + pad
    assert tail_params.nframes == round(0.1 * RATE) + pad
    assert np.array_equal(head_samples.astype(np.float32), source[: head_params.nframes])
    assert np.array_equal(tail_samples.astype(np.float32), source[-tail_params.nframes :])
    assert head.source_start == 0.0 and head.record["first_frame"] == 0
    assert tail.source_end == DURATION_S and tail.record["last_frame"] == len(source)
    assert (head.record["pad_ms"], tail.record["pad_ms"]) == (150, 150)


def test_cut_keeps_every_channel_of_a_stereo_source(root: Path, run: Run, config: Config) -> None:
    tone = burst()
    stereo = np.stack([tone, (tone // 2)], axis=1)
    ingest.decode(write_wav(root / "stereo.wav", stereo, RATE), run, config)

    clip = cut(run, 1.3, 1.4, pad_ms=0, config=config)

    samples, params = read_wav(clip.path)
    assert params.nchannels == 2 and clip.record["channels"] == 2
    first, last = round(1.3 * RATE), round(1.4 * RATE)
    assert np.array_equal(samples.astype(np.float32), decoded(run)[first:last])


def test_cut_does_not_touch_the_decoded_source(decoded_run: Run, config: Config) -> None:
    before = {
        name: sha256_of(decoded_run.path / name) for name in ("source.json", "work/decoded.f32")
    }

    cut(decoded_run, 1.0, 1.5, config=config)

    assert {name: sha256_of(decoded_run.path / name) for name in before} == before


@pytest.mark.parametrize(
    ("start", "end", "pad_ms"),
    [
        (1.5, 1.0, 150),
        (1.0, 1.0, 150),
        (-0.1, 1.0, 150),
        (2.0, DURATION_S + 0.01, 150),
        (DURATION_S, DURATION_S + 1.0, 150),
        (float("nan"), 1.0, 150),
        (1.0, float("inf"), 150),
        (True, 1.0, 150),
        (1.0, 1.5, -1),
        (1.0, 1.5, 150.0),
        (1.0, 1.0 + 1e-9, 0),
    ],
)
def test_bad_intervals_are_rejected_before_anything_is_written(
    start: object, end: object, pad_ms: object, decoded_run: Run, config: Config
) -> None:
    with pytest.raises(ClipError) as error:
        cut(decoded_run, start, end, pad_ms, config=config)

    assert error.value.code == "bad_interval"
    assert not (decoded_run.path / "clips").exists()


def test_a_run_without_a_decoded_source_is_refused(run: Run, config: Config) -> None:
    with pytest.raises(ClipError) as error:
        cut(run, 0.0, 1.0, config=config)

    assert error.value.code == "source_missing"
    assert not (run.path / "clips").exists()


def test_a_repeated_cut_returns_the_clip_already_in_the_run(
    decoded_run: Run, config: Config
) -> None:
    first = cut(decoded_run, 1.0, 1.5, config=config)
    mtime = first.path.stat().st_mtime_ns

    again = cut(decoded_run, 1.0, 1.5, config=config)

    assert again.reused is True and again.record == first.record and again.path == first.path
    assert again.path.stat().st_mtime_ns == mtime
    assert entries(decoded_run, "clips") == [f"{first.id}.json", f"{first.id}.wav"]


def test_a_clip_that_does_not_match_its_record_is_refused_not_overwritten(
    decoded_run: Run, config: Config
) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)
    tampered = clip.path.read_bytes()[:-4] + b"\0\0\0\0"
    clip.path.write_bytes(tampered)

    with pytest.raises(ClipError) as error:
        cut(decoded_run, 1.0, 1.5, config=config)

    assert error.value.code == "clip_corrupt"
    assert clip.path.read_bytes() == tampered
    with pytest.raises(ClipError) as listen_error:
        listening_copy(decoded_run, clip.id, config=config)
    assert listen_error.value.code == "clip_corrupt"
    assert not (decoded_run.path / "listen").exists()


# --- listening_copy -----------------------------------------------------------------------------


def test_listening_copy_applies_one_recorded_linear_gain(decoded_run: Run, config: Config) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)
    clip_sha = sha256_of(clip.path)

    copy = listening_copy(decoded_run, clip.id, config=config)

    assert copy.kind == "listening_copy" and copy.reused is False
    assert copy.id == clip.id.replace("clip-", "listen-", 1)
    assert copy.path == decoded_run.path / "listen" / f"{copy.id}.wav"
    assert entries(decoded_run, "listen") == [f"{copy.id}.json", f"{copy.id}.wav"]
    assert stat.S_IMODE((decoded_run.path / "listen").stat().st_mode) == 0o700
    assert stat.S_IMODE(copy.path.stat().st_mode) == 0o600
    assert sha256_of(clip.path) == clip_sha

    before, _ = read_wav(clip.path)
    after, params = read_wav(copy.path)
    assert (params.nchannels, params.sampwidth, params.framerate, params.nframes) == (
        1,
        4,
        RATE,
        before.shape[0],
    )
    assert peak_dbfs(after) < 0
    assert abs(peak_dbfs(after) - (-1.0)) < 0.01
    gain = 10 ** (copy.gain_db / 20)
    assert np.allclose(after, before * gain, atol=2 / FULL_SCALE)
    assert abs(copy.gain_db - (-1.0 - peak_dbfs(before))) < 0.01

    record = json.loads(copy.record_path.read_text(encoding="utf-8"))
    assert record == copy.record
    assert record["for_listening_only"] is True
    assert record["gain_db"] == copy.gain_db > 0
    assert abs(record["gain_linear"] - gain) < 1e-9
    assert record["target_peak_dbfs"] == -1.0
    assert abs(record["peak_dbfs_before"] - peak_dbfs(before)) < 1e-6
    assert abs(record["peak_dbfs_after"] - peak_dbfs(after)) < 1e-6
    assert record["sha256"] == copy.sha256 == sha256_of(copy.path)
    assert record["path"] == f"listen/{copy.id}.wav"
    assert (record["source_start"], record["source_end"]) == (clip.source_start, clip.source_end)
    assert record["clip"] == {"id": clip.id, "path": clip.record["path"], "sha256": clip_sha}
    assert record["source"] == clip.record["source"]


def test_listening_copy_attenuates_a_clip_already_above_the_target(
    decoded_run: Run, config: Config
) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)

    copy = listening_copy(decoded_run, clip.id, target_peak_dbfs=-20.0, config=config)

    after, _ = read_wav(copy.path)
    assert copy.gain_db < 0
    assert abs(peak_dbfs(after) - (-20.0)) < 0.01


def test_a_repeated_listening_copy_is_reused(decoded_run: Run, config: Config) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)
    first = listening_copy(decoded_run, clip.id, config=config)

    again = listening_copy(decoded_run, clip.id, config=config)

    assert again.reused is True and again.record == first.record
    assert entries(decoded_run, "listen") == [f"{first.id}.json", f"{first.id}.wav"]


def test_a_listening_copy_at_another_target_is_refused_not_reused(
    decoded_run: Run, config: Config
) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)
    first = listening_copy(decoded_run, clip.id, target_peak_dbfs=-1.0, config=config)
    first_sha = sha256_of(first.path)

    with pytest.raises(ClipError) as error:
        listening_copy(decoded_run, clip.id, target_peak_dbfs=-6.0, config=config)

    assert error.value.code == "clip_exists"
    assert "-1.0" in error.value.message and "-6.0" in error.value.message
    assert sha256_of(first.path) == first_sha
    assert entries(decoded_run, "listen") == [f"{first.id}.json", f"{first.id}.wav"]
    again = listening_copy(decoded_run, clip.id, target_peak_dbfs=-1.0, config=config)
    assert again.reused is True and again.record == first.record


def test_a_listening_copy_at_the_lowest_target_has_a_finite_level(
    decoded_run: Run, config: Config
) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)

    copy = listening_copy(decoded_run, clip.id, target_peak_dbfs=-120.0, config=config)

    after, _ = read_wav(copy.path)
    assert math.isfinite(copy.gain_db) and math.isfinite(copy.record["peak_dbfs_after"])
    assert abs(peak_dbfs(after) - (-120.0)) < 0.01


@pytest.mark.parametrize("target", [0.0, 0.5, -120.5, -200.0, float("nan"), float("-inf"), True])
def test_a_target_peak_outside_the_allowed_range_is_rejected(
    target: object, decoded_run: Run, config: Config
) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)

    with pytest.raises(ClipError) as error:
        listening_copy(decoded_run, clip.id, target, config=config)

    assert error.value.code == "bad_target"
    assert not (decoded_run.path / "listen").exists()


@pytest.mark.parametrize("clip_id", ["clip-0-1", "../source", "", "clips/x"])
def test_a_clip_not_in_the_run_is_refused(clip_id: str, decoded_run: Run, config: Config) -> None:
    with pytest.raises(ClipError) as error:
        listening_copy(decoded_run, clip_id, config=config)

    assert error.value.code == "clip_missing"


def test_a_silent_clip_has_no_listening_copy(decoded_run: Run, config: Config) -> None:
    clip = cut(decoded_run, 0.2, 0.6, config=config)
    assert float(np.max(np.abs(read_wav(clip.path)[0]))) == 0.0

    with pytest.raises(ClipError) as error:
        listening_copy(decoded_run, clip.id, config=config)

    assert error.value.code == "silent_clip"
    assert not (decoded_run.path / "listen").exists()


# --- artifacts of the result contract --------------------------------------------------------


def test_artifacts_list_clips_and_listening_copies_with_relative_paths(
    decoded_run: Run, config: Config
) -> None:
    clip = cut(decoded_run, 1.0, 1.5, config=config)
    other = cut(decoded_run, 2.0, 2.2, config=config)
    copy = listening_copy(decoded_run, clip.id, config=config)

    listed = clips.artifacts(decoded_run)

    assert listed == sorted([clip.artifact, other.artifact, copy.artifact], key=lambda a: a["id"])
    assert clip.artifact == {
        "id": clip.id,
        "kind": "clip",
        "path": f"clips/{clip.id}.wav",
        "sha256": sha256_of(clip.path),
        "source_start": clip.source_start,
        "source_end": clip.source_end,
    }
    assert copy.artifact["kind"] == "listening_copy"
    assert copy.artifact["path"] == f"listen/{copy.id}.wav"
    assert (copy.artifact["source_start"], copy.artifact["source_end"]) == (
        clip.source_start,
        clip.source_end,
    )
    serialized = json.dumps(listed) + clip.record_path.read_text() + copy.record_path.read_text()
    assert str(decoded_run.path) not in serialized
    assert str(config.data_root) not in serialized
    for artifact in listed:
        assert not Path(artifact["path"]).is_absolute() and ".." not in artifact["path"]
        assert (decoded_run.path / artifact["path"]).is_file()


def test_artifacts_pass_the_result_contract(root: Path, run: Run, config: Config) -> None:
    """The entries fit the synthetic example, whose source is 12 s long."""
    seconds = EXAMPLE_DOC["source"]["duration_s"]
    ingest.decode(write_wav(root / "long.wav", burst(seconds), RATE), run, config)
    clip = cut(run, 1.0, 1.5, config=config)
    listening_copy(run, clip.id, config=config)
    doc = deepcopy(EXAMPLE_DOC)
    doc["artifacts"] += clips.artifacts(run)

    schema.validate(doc)

    doc["artifacts"][-1]["path"] = str(run.path / doc["artifacts"][-1]["path"])
    assert any("relative to the run directory" in e for e in schema.validation_errors(doc))


def test_artifacts_are_empty_without_clips_and_refuse_a_record_without_its_wav(
    decoded_run: Run, config: Config
) -> None:
    assert clips.artifacts(decoded_run) == []
    clip = cut(decoded_run, 1.0, 1.5, config=config)
    clip.path.unlink()

    with pytest.raises(ClipError) as error:
        clips.artifacts(decoded_run)

    assert error.value.code == "clip_corrupt"
