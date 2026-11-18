"""Channel analysis (issue #17, T02): report, verdicts, mono choice and the time scale."""

import json
from pathlib import Path

import numpy as np
import pytest

from sound_scout import channels
from sound_scout.channels import ChannelError, choose_mono, report
from sound_scout.config import Config
from sound_scout.ingest import decode
from sound_scout.runs import open_run
from sound_scout.schema import validate
from tests.test_ingest import IDENTITY, SAMPLE_RATE, needs_ffmpeg, sine, write_wav
from tests.test_schema import EXAMPLE_DOC

FRAMES = 1_000


def tone(frames: int = FRAMES, amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
    return (amplitude * np.sin(np.arange(frames) * 0.2 + phase)).astype(np.float32)


def stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.column_stack([left, right]).astype(np.float32)


def test_report_measures_each_channel_without_hiding_digital_silence() -> None:
    audio = np.array([[0.5, 0], [-0.5, 0], [0, 0], [0, 0]], dtype=np.float32)

    result = report(audio)

    first, second = result["channels"]
    assert (first["channel"], second["channel"]) == (1, 2)
    assert (first["samples"], first["zero_samples"], first["zero_share"]) == (4, 2, 0.5)
    assert first["peak_linear"] == 0.5
    assert first["peak_dbfs"] == pytest.approx(-6.0206, abs=1e-4)
    assert first["rms_dbfs"] == pytest.approx(-9.0309, abs=1e-4)
    assert (second["zero_share"], second["peak_linear"], second["rms_linear"]) == (1.0, 0, 0)
    assert second["peak_dbfs"] is None and second["rms_dbfs"] is None
    assert result["active_channels"] == [1]
    assert result["correlation"] is None
    assert result["verdict"] == "one_active"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("audio", "verdict"),
    [
        (tone()[:, None], "mono"),
        (stereo(tone(), tone()), "duplicated_stereo"),
        (stereo(tone(), np.zeros(FRAMES, dtype=np.float32)), "one_active"),
        (stereo(tone(), -tone()), "cancellation"),
        (np.zeros((FRAMES, 2), dtype=np.float32), "zero"),
    ],
    ids=["mono", "duplicated_stereo", "one_active", "cancellation", "zero"],
)
def test_report_gives_dbfs_levels_zero_share_correlation_and_a_named_verdict(
    audio: np.ndarray, verdict: str
) -> None:
    result = report(audio)

    assert result["verdict"] == verdict
    assert result["verdict"] in {
        "mono",
        "duplicated_stereo",
        "one_active",
        "cancellation",
        "zero",
        "stereo",
    }
    assert "correlation" in result
    assert result["correlation"] is None or -1.0 <= result["correlation"] <= 1.0
    assert len(result["channels"]) == audio.shape[1]
    for item in result["channels"]:
        assert {"peak_dbfs", "rms_dbfs", "zero_share"} <= item.keys()
        assert 0.0 <= item["zero_share"] <= 1.0
        if item["zero_share"] < 1.0:
            assert item["rms_dbfs"] <= item["peak_dbfs"] <= 0.0
            assert item["rms_dbfs"] == pytest.approx(20 * np.log10(item["rms_linear"]))
        else:
            assert item["peak_dbfs"] is None and item["rms_dbfs"] is None


def test_active_plus_silent_channel_is_taken_sample_for_sample_without_halving() -> None:
    active = np.array([0.5, -0.25, 0.125], dtype=np.float32)
    audio = stereo(np.zeros(3, dtype=np.float32), active)

    mono, record = choose_mono(audio, report(audio))

    np.testing.assert_array_equal(mono, active)
    assert mono.dtype == np.float32
    assert record["name"] == "channel_choice"
    settings = record["settings"]
    assert (settings["method"], settings["selected_channel"]) == ("select_channel", 2)
    assert settings["verdict"] == "one_active"
    assert settings["gain_db"] == 0.0
    assert settings["arithmetic_mean_loss_db"] is None
    assert "channel 2" in settings["reason"]


def test_duplicated_stereo_is_one_channel_not_two_voices() -> None:
    signal = tone()
    audio = stereo(signal, signal)

    result = report(audio)
    mono, record = choose_mono(audio, result)

    assert result["verdict"] == "duplicated_stereo"
    assert result["correlation"] == pytest.approx(1.0)
    assert result["arithmetic_mean_loss_db"] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_array_equal(mono, signal)
    assert record["settings"]["method"] == "select_channel"
    assert record["settings"]["selected_channel"] == 1


def test_duplicated_stereo_at_different_levels_takes_the_louder_channel() -> None:
    signal = tone()
    audio = stereo(signal / 2, signal)

    result = report(audio)
    mono, record = choose_mono(audio, result)

    assert result["verdict"] == "duplicated_stereo"
    np.testing.assert_array_equal(mono, signal)
    assert record["settings"]["selected_channel"] == 2


def test_inverted_duplicate_beyond_the_cancellation_gap_takes_the_louder_channel() -> None:
    signal = tone()
    audio = stereo(signal, -0.3 * signal)

    result = report(audio)
    mono, record = choose_mono(audio, result)

    assert result["correlation"] == pytest.approx(-1.0)
    assert result["arithmetic_mean_loss_db"] < channels.CANCELLATION_LOSS_DB
    assert result["verdict"] == "duplicated_stereo"
    np.testing.assert_array_equal(mono, signal)
    assert record["settings"]["method"] == "select_channel"
    assert record["settings"]["selected_channel"] == 1
    assert "inverted polarity" in record["settings"]["reason"]


def test_mono_input_is_returned_unchanged() -> None:
    signal = tone()

    result = report(signal[:, None])
    mono, record = choose_mono(signal[:, None], result)

    assert result["verdict"] == "mono"
    assert result["correlation"] is None
    np.testing.assert_array_equal(mono, signal)
    assert record["settings"] == {
        "method": "single_channel",
        "selected_channel": 1,
        "source_channels": 1,
        "verdict": "mono",
        "reason": "the original has one channel",
        "gain_db": 0.0,
        "frames": FRAMES,
        "correlation": None,
        "arithmetic_mean_loss_db": None,
        "duplicate_correlation": 0.99,
        "cancellation_loss_db": 6.0,
    }


def test_distinct_stereo_is_averaged_without_gain() -> None:
    left, right = tone(phase=0.0), tone(phase=1.3)
    audio = stereo(left, right)

    result = report(audio)
    mono, record = choose_mono(audio, result)

    assert result["verdict"] == "stereo"
    assert -0.99 < result["correlation"] < 0.99
    np.testing.assert_allclose(mono, (left + right) / 2, atol=1e-7)
    assert record["settings"]["method"] == "arithmetic_mean"
    assert record["settings"]["selected_channel"] is None
    assert record["settings"]["gain_db"] == 0.0
    assert record["settings"]["verdict"] == channels.DISTINCT_STEREO


def test_zero_signal_fails_with_reason_zero_signal() -> None:
    audio = np.zeros((FRAMES, 2), dtype=np.float32)

    result = report(audio)
    assert result["verdict"] == "zero"
    assert result["active_channels"] == []
    assert all(item["zero_share"] == 1.0 for item in result["channels"])
    with pytest.raises(ChannelError) as raised:
        choose_mono(audio, result)

    assert raised.value.code == "zero_signal"
    assert raised.value.code in channels.REASON_CODES
    assert raised.value.message


def test_phase_cancellation_stops_the_analysis_instead_of_downmixing() -> None:
    signal = tone()
    audio = stereo(signal, -signal)

    result = report(audio)
    assert result["verdict"] == "cancellation"
    assert result["correlation"] == pytest.approx(-1.0)
    assert result["arithmetic_mean_loss_db"] is None
    assert result["arithmetic_mean_rms_dbfs"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ChannelError, match="cancel") as raised:
        choose_mono(audio, result)

    assert raised.value.code == "cancellation"


def test_partial_cancellation_beyond_six_db_is_still_cancellation() -> None:
    signal = tone()
    audio = stereo(signal, -0.8 * signal)

    result = report(audio)

    assert result["arithmetic_mean_loss_db"] > channels.CANCELLATION_LOSS_DB
    assert result["verdict"] == "cancellation"


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((0, 2), dtype=np.float32),
        np.zeros(FRAMES, dtype=np.float32),
        np.array([[np.nan, 0.0]], dtype=np.float32),
        np.array([[np.inf, 0.0]], dtype=np.float32),
    ],
    ids=["empty", "one_dimensional", "nan", "inf"],
)
def test_malformed_arrays_are_rejected_as_a_programming_error(audio: np.ndarray) -> None:
    with pytest.raises(ValueError):
        report(audio)


def test_more_than_two_channels_is_refused_instead_of_averaging_a_silent_one() -> None:
    signal = tone()
    audio = np.column_stack([signal, signal, np.zeros(FRAMES)]).astype(np.float32)

    with pytest.raises(ValueError, match="3 channels"):
        report(audio)
    with pytest.raises(ValueError, match="3 channels"):
        choose_mono(audio, {"verdict": "stereo", "active_channels": [1, 2]})


def test_choice_record_validates_in_provenance_transformations() -> None:
    audio = stereo(tone(), np.zeros(FRAMES, dtype=np.float32))
    _, record = choose_mono(audio, report(audio))
    doc = json.loads(json.dumps(EXAMPLE_DOC))
    doc["provenance"]["transformations"] = [record]

    validate(doc)
    json.dumps(record, allow_nan=False)


@needs_ffmpeg
def test_choice_keeps_the_time_scale_of_the_decoded_original(tmp_path: Path) -> None:
    root, data_root = tmp_path / "recordings", tmp_path / "data"
    root.mkdir()
    data_root.mkdir()
    config = Config(allowed_roots=(root,), data_root=data_root)
    run = open_run("req-1", IDENTITY, data_root)
    impulse_at = SAMPLE_RATE // 2
    active = np.zeros(SAMPLE_RATE, dtype=np.int16)
    active[impulse_at] = 32767
    active[impulse_at + 1 :] = sine(0.5)[: SAMPLE_RATE - impulse_at - 1]
    path = write_wav(root / "late-stereo.wav", np.column_stack([np.zeros_like(active), active]))

    source = decode(path, run, config)
    audio = channels.load(source)
    result = report(audio)
    mono, record = choose_mono(audio, result)

    assert audio.shape == (SAMPLE_RATE, 2)
    assert result["verdict"] == "one_active"
    assert record["settings"]["selected_channel"] == 2
    assert len(mono) == source.frames == SAMPLE_RATE
    assert record["settings"]["frames"] == source.frames
    assert np.flatnonzero(mono)[0] == impulse_at
    assert mono[impulse_at] == pytest.approx(32767 / 32768)
    np.testing.assert_array_equal(mono, audio[:, 1])


@needs_ffmpeg
def test_load_refuses_a_decoded_file_of_the_wrong_size(tmp_path: Path) -> None:
    root, data_root = tmp_path / "recordings", tmp_path / "data"
    root.mkdir()
    data_root.mkdir()
    config = Config(allowed_roots=(root,), data_root=data_root)
    run = open_run("req-1", IDENTITY, data_root)
    source = decode(write_wav(root / "tone.wav", sine(0.5)), run, config)
    with source.decoded_path.open("ab") as out:
        out.write(b"\0\0\0\0")

    with pytest.raises(ValueError, match="frames"):
        channels.load(source)
