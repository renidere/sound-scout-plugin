"""Acoustic measurements (issue #18, R04 layer 2): VAD spans, RMS and F0 of the working mono.

The speech-like control signal is synthesised with Praat's KlattGrid (formant synthesis of
alternating vowels at a set pitch, `sound_scout.smoke.speech_like_control`, shared with the
`acoustics` smoke run), not recorded: Silero VAD does not take burst noise or a pure
tone for speech, so a tone is tested through `acoustic_track` with given spans, and `measure`
end to end through the formant synthesis. The detector and Praat run for real; nothing is mocked.
The boundary tolerance the span tests assert (start 50 ms, end two VAD windows) and the choice
of the control are recorded in `docs/decisions/0003-vad-boundary-tolerance.md`.
"""

import json
import os
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import numpy as np
import parselmouth
import pytest

from sound_scout import acoustics, guard
from sound_scout.acoustics import (
    MEASUREMENT_NAMES,
    VadUnavailable,
    acoustic_track,
    measure,
    speech_activity,
)
from sound_scout.schema import validate
from sound_scout.smoke import place
from sound_scout.smoke import speech_like_control as speech_like
from tests.test_schema import EXAMPLE_DOC

RATE = 16_000
VAD_WINDOW_S = acoustics.VAD_WINDOW_SAMPLES / acoustics.VAD_RATE
BOUNDARY_TOLERANCE_S = 0.05
END_TOLERANCE_S = acoustics.VAD_END_LAG_WINDOWS * VAD_WINDOW_S


def sine(frequency: float, seconds: float = 2.0, rate: int = RATE, amplitude: float = 0.2):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def by_name(result: dict) -> dict[str, dict]:
    return {item["name"]: item for item in result["measurements"]}


def assert_contract(result: dict) -> None:
    """Every entry has the contract fields, no NaN anywhere, and the result document validates."""
    assert result["status"] in {"completed", "unavailable"}
    assert [item["name"] for item in result["measurements"]] == list(MEASUREMENT_NAMES)
    for item in result["measurements"]:
        assert set(item) == {"name", "value", "unit", "method", "settings", "limitations", "reason"}
        assert item["unit"] and item["method"] and item["limitations"]
        assert item["value"] is not None or item["reason"]
    json.dumps(result, allow_nan=False)
    doc = deepcopy(EXAMPLE_DOC)
    assert by_name(result)["duration_seconds"]["value"] <= doc["source"]["duration_s"]
    doc["measurements"] = result["measurements"]
    doc["findings"] = []
    validate(doc)


@pytest.mark.parametrize("frequency", [150.0, 225.0])
def test_known_tone_gives_its_frequency_and_a_digital_level(frequency: float) -> None:
    track = acoustic_track(sine(frequency), RATE, [{"start": 0.0, "end": 2.0}])

    voiced = track["f0_hz"][track["speech"] & (track["f0_hz"] > 0)]
    assert len(voiced) > 100
    assert np.percentile(voiced, 50) == pytest.approx(frequency, rel=0.03)
    assert np.percentile(voiced, 10) == pytest.approx(frequency, rel=0.03)
    assert np.percentile(voiced, 90) == pytest.approx(frequency, rel=0.03)
    levels = track["rms_dbfs"][track["speech"] & np.isfinite(track["rms_dbfs"])]
    assert np.percentile(levels, 50) == pytest.approx(20 * np.log10(0.2 / np.sqrt(2)), abs=0.2)
    assert track["speech"].all()
    assert (track["times_s"] >= 0).all() and (track["times_s"] <= 2.0).all()


def test_pitch_range_of_two_tones_in_sequence() -> None:
    audio = np.concatenate([sine(150.0), sine(225.0)])

    track = acoustic_track(audio, RATE, [{"start": 0.0, "end": 4.0}])

    voiced = track["f0_hz"][track["f0_hz"] > 0]
    semitones = 12 * np.log2(np.percentile(voiced, 90) / np.percentile(voiced, 10))
    assert semitones == pytest.approx(7.01955, abs=0.2)


def test_digital_silence_has_no_spans_and_no_fabricated_pitch_or_level() -> None:
    result = measure(np.zeros(2 * RATE, dtype=np.float32), RATE)

    assert_contract(result)
    values = by_name(result)
    assert result["status"] == "completed"
    assert values["duration_seconds"]["value"] == 2.0
    assert values["speech_spans"]["value"] == []
    assert values["f0_hz"]["value"] is None
    assert values["f0_hz"]["reason"] == "no voiced frames inside speech spans"
    assert values["rms_dbfs"]["value"] is None
    assert values["rms_dbfs"]["reason"] == "no speech frames with signal"


def test_pure_tone_is_not_speech_for_the_detector_so_f0_stays_null_with_a_reason() -> None:
    result = measure(sine(150.0), RATE)

    assert_contract(result)
    values = by_name(result)
    assert values["speech_spans"]["value"] == []
    assert values["f0_hz"]["value"] is None and values["f0_hz"]["reason"]


@pytest.mark.parametrize("frequency", [220.0, 330.0])
def test_a_1_5_s_tone_has_no_speech_so_its_level_is_null_and_says_why(frequency: float) -> None:
    """#125: the 1.5 s tones the control note test used before. Level is over speech frames only."""
    tone = np.round(sine(frequency, seconds=1.5, amplitude=0.5) * np.iinfo(np.int16).max)
    mono = (tone.astype(np.int16) / 32768).astype(np.float32)

    result = measure(mono, RATE)
    again = measure(mono.copy(), RATE)

    assert_contract(result)
    values = by_name(result)
    assert values["speech_spans"]["value"] == []
    assert values["rms_dbfs"]["settings"]["speech_frames"] == 0
    assert values["rms_dbfs"]["value"] is None
    assert values["rms_dbfs"]["reason"] == "no speech frames with signal"
    assert np.nanmedian(acoustic_track(mono, RATE, [])["rms_dbfs"]) > -10.0  # signal is there
    assert by_name(again)["rms_dbfs"] == values["rms_dbfs"]


def test_signal_shorter_than_a_tenth_of_a_second_gives_null_with_the_reason() -> None:
    result = measure(np.full(int(0.05 * RATE), 0.1, dtype=np.float32), RATE)

    assert_contract(result)
    values = by_name(result)
    assert result["status"] == "completed"
    assert values["duration_seconds"]["value"] == pytest.approx(0.05)
    assert values["speech_spans"]["value"] == []
    for name in ("rms_dbfs", "f0_hz"):
        assert values[name]["value"] is None
        assert values[name]["reason"] == acoustics.TOO_SHORT
    with pytest.raises(ValueError, match="shorter than 0.1 s"):
        acoustic_track(np.full(int(0.05 * RATE), 0.1, dtype=np.float32), RATE, [])


@pytest.mark.parametrize(("rate", "start_s"), [(RATE, 1.0), (48_000, 0.75), (44_100, 1.0)])
def test_speech_like_signal_gives_one_span_within_the_declared_tolerance_on_the_original_scale(
    rate: int, start_s: float
) -> None:
    """One utterance at three native rates: the span lands on the original time scale.

    The control is formant-synthesised vowels (`speech_like`), because the detector does not
    take noise for speech. The tolerance is the declared one: the start within 50 ms, the end
    within two detector windows (see `docs/decisions/0003` and the onset sweep below).
    """
    speech = speech_like(1.2, rate)
    audio = place(speech, start_s, 4.0, rate)

    result = measure(audio, rate)

    assert_contract(result)
    values = by_name(result)
    assert result["status"] == "completed"
    spans = values["speech_spans"]["value"]
    assert len(spans) == 1, spans
    assert spans[0]["start"] == pytest.approx(start_s, abs=BOUNDARY_TOLERANCE_S)
    assert spans[0]["end"] == pytest.approx(start_s + 1.2, abs=END_TOLERANCE_S)
    settings = values["speech_spans"]["settings"]
    assert settings["min_silence_duration_ms"] == 300
    assert settings["speech_pad_ms"] == 0
    assert settings["threshold"] == 0.5
    assert settings["resampled_for_detector"] is (rate != acoustics.VAD_RATE)
    assert values["f0_hz"]["value"]["p50"] == pytest.approx(130.0, rel=0.03)
    assert values["f0_hz"]["settings"]["pitch_floor_hz"] == 50
    assert values["f0_hz"]["settings"]["pitch_ceiling_hz"] == 600
    assert values["f0_hz"]["settings"]["time_step_s"] == 0.01
    assert values["rms_dbfs"]["settings"]["window_ms"] == 40
    assert values["rms_dbfs"]["value"]["p10"] <= values["rms_dbfs"]["value"]["p50"] <= 0.0


def test_onset_sweep_keeps_the_start_within_50_ms_and_the_end_within_two_windows() -> None:
    """The declared tolerance holds at every onset, not at placements picked to pass.

    One synthesised utterance is shifted to 143 onsets 0.5–1.5 s at a 7 ms step (more than a
    detector window at a step finer than it); the real Silero VAD runs on each placement. The
    start is held to the limitation's own bound, one detector window, not only to the 50 ms
    of the fixed placements above.
    """
    speech = speech_like(1.2, RATE)
    start_errors, end_errors = [], []

    for start_s in np.arange(0.5, 1.5 + 1e-9, 0.007):
        start_s = float(start_s)
        spans = speech_activity(place(speech, start_s, 4.0, RATE), RATE)

        assert len(spans) == 1, (start_s, spans)
        assert spans[0]["start"] == pytest.approx(start_s, abs=BOUNDARY_TOLERANCE_S), start_s
        assert spans[0]["end"] == pytest.approx(start_s + 1.2, abs=END_TOLERANCE_S), start_s
        start_errors.append(spans[0]["start"] - start_s)
        end_errors.append(spans[0]["end"] - (start_s + 1.2))

    assert len(start_errors) == 143
    assert 0 < max(np.abs(start_errors)) <= VAD_WINDOW_S < BOUNDARY_TOLERANCE_S, start_errors
    assert 0 < max(np.abs(end_errors)) <= END_TOLERANCE_S, end_errors


@pytest.mark.parametrize("tail_s", [0.05, 0.29], ids=["tail_50ms", "tail_290ms"])
def test_an_utterance_ending_inside_the_last_300_ms_ends_where_it_ends_not_at_eof(
    tail_s: float,
) -> None:
    """The detector closes a span only after 300 ms of silence; the file end is not that."""
    audio = place(speech_like(1.2, RATE), 1.0, 2.2 + tail_s, RATE)

    spans = speech_activity(audio, RATE)

    assert len(spans) == 1, spans
    assert spans[0]["end"] == pytest.approx(2.2, abs=END_TOLERANCE_S)
    assert spans[0]["end"] < 2.2 + tail_s


def test_speech_running_to_the_end_of_the_file_ends_at_the_duration() -> None:
    audio = place(speech_like(1.2, RATE), 1.0, 2.2, RATE)

    spans = speech_activity(audio, RATE)

    assert len(spans) == 1, spans
    assert spans[0]["end"] == 2.2
    assert by_name(measure(audio, RATE))["duration_seconds"]["value"] == 2.2


def test_two_utterances_with_a_pause_give_two_spans_and_the_pause_is_not_a_finding() -> None:
    speech = speech_like(0.8, RATE)
    audio = place(speech, 0.5, 4.0, RATE) + place(speech, 2.3, 4.0, RATE)

    spans = speech_activity(audio, RATE)

    assert len(spans) == 2, spans
    assert spans[0]["start"] == pytest.approx(0.5, abs=BOUNDARY_TOLERANCE_S)
    assert spans[0]["end"] == pytest.approx(1.3, abs=END_TOLERANCE_S)
    assert spans[1]["start"] == pytest.approx(2.3, abs=BOUNDARY_TOLERANCE_S)
    assert spans[1]["end"] == pytest.approx(3.1, abs=END_TOLERANCE_S)
    limitations = " ".join(by_name(measure(audio, RATE))["speech_spans"]["limitations"])
    assert "not a hesitation" in limitations


def test_limitations_name_digital_level_vad_pause_and_octave_errors() -> None:
    values = by_name(measure(place(speech_like(1.2, RATE), 1.0, 4.0, RATE), RATE))

    rms = " ".join(values["rms_dbfs"]["limitations"])
    assert "digital level" in rms and "not the loudness of the voice" in rms
    spans = " ".join(values["speech_spans"]["limitations"])
    assert "VAD pause is not a hesitation" in spans
    assert f"{acoustics.VAD_WINDOW_MS} ms windows: the start is within one window" in spans
    assert round(VAD_WINDOW_S * 1000) == acoustics.VAD_WINDOW_MS
    lag_ms = acoustics.VAD_END_LAG_WINDOWS * acoustics.VAD_WINDOW_MS
    assert f"{acoustics.VAD_END_LAG_WINDOWS} windows ({lag_ms} ms)" in spans
    assert lag_ms == round(END_TOLERANCE_S * 1000)
    f0 = " ".join(values["f0_hz"]["limitations"])
    assert "octave errors" in f0.lower()
    assert "not expressiveness" in f0


def test_vad_settings_are_the_declared_detector_parameters_and_not_changed_silently() -> None:
    """The tolerance is documented, not bought by retuning the detector (R04, issue #55)."""
    settings = by_name(measure(place(speech_like(1.2, RATE), 1.0, 4.0, RATE), RATE))[
        "speech_spans"
    ]["settings"]

    assert settings["threshold"] == 0.5
    assert settings["min_silence_duration_ms"] == 300
    assert settings["min_speech_duration_ms"] == 100
    assert settings["speech_pad_ms"] == 0
    assert settings["detector_sample_rate"] == 16_000
    assert settings["window_ms"] == 32 == acoustics.VAD_WINDOW_MS
    assert acoustics.VAD_TAIL_SAMPLES == 16_000 * 300 // 1000 + 2 * 512


def test_measures_the_given_signal_and_never_applies_gain_itself() -> None:
    """A gain copy is a separate derivative: the layer measures what it is given, unchanged.

    `numbers` here means the measurement of the original: the input stays byte-identical, the
    level is the signal's own (a 0.2 sine is -17 dBFS, not a normalised figure), `gain_db` is 0,
    and a +6 dB copy measured on its own differs by exactly that gain in level while F0 and the
    spans agree within the detector's window and Praat's precision.
    """
    speech = speech_like(1.2, RATE)
    original = place(speech, 1.0, 4.0, RATE)
    kept = original.copy()
    louder = (original * 2.0).astype(np.float32)

    base = by_name(measure(original, RATE))
    gained = by_name(measure(louder, RATE))

    np.testing.assert_array_equal(original, kept)
    assert base["rms_dbfs"]["settings"]["gain_db"] == 0.0
    tone_track = acoustic_track(sine(150.0), RATE, [{"start": 0.0, "end": 2.0}])
    assert np.nanmedian(tone_track["rms_dbfs"]) == pytest.approx(-16.99, abs=0.05)
    louder_track = acoustic_track(sine(150.0) * 2, RATE, [{"start": 0.0, "end": 2.0}])
    assert np.nanmedian(louder_track["rms_dbfs"]) - np.nanmedian(tone_track["rms_dbfs"]) == (
        pytest.approx(20 * np.log10(2), abs=1e-6)
    )
    for key in ("p10", "p50", "p90"):
        assert gained["f0_hz"]["value"][key] == pytest.approx(base["f0_hz"]["value"][key], rel=0.01)
        assert gained["rms_dbfs"]["value"][key] > base["rms_dbfs"]["value"][key]
    pairs = zip(base["speech_spans"]["value"], gained["speech_spans"]["value"], strict=True)
    for before, after in pairs:
        assert after["start"] == pytest.approx(before["start"], abs=VAD_WINDOW_S + 1e-6)
        assert after["end"] == pytest.approx(before["end"], abs=VAD_WINDOW_S + 1e-6)


def test_without_the_vad_asset_the_stage_is_unavailable_and_nothing_is_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_asset() -> str:
        raise VadUnavailable("Silero VAD asset not found in faster_whisper")

    monkeypatch.setattr(acoustics, "vad_asset_path", no_asset)

    result = measure(place(speech_like(1.2, RATE), 1.0, 4.0, RATE), RATE)

    assert_contract(result)
    assert result["status"] == "unavailable"
    assert result["reason"] == "Silero VAD asset not found in faster_whisper"
    values = by_name(result)
    assert values["duration_seconds"]["value"] == 4.0
    assert values["speech_spans"]["value"] is None
    assert values["speech_spans"]["reason"] == result["reason"]
    assert values["speech_spans"]["settings"]["asset"] is None
    for name in ("rms_dbfs", "f0_hz"):
        assert values[name]["value"] is None
        assert values[name]["reason"] == acoustics.SPANS_UNAVAILABLE
    with pytest.raises(VadUnavailable):
        speech_activity(np.zeros(RATE, dtype=np.float32), RATE)


def test_without_onnxruntime_the_stage_is_unavailable_and_settings_stay_honest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acoustics, "package_dir", lambda name: None)
    monkeypatch.setattr(acoustics, "installed_version", lambda distribution: None)

    result = measure(place(speech_like(1.2, RATE), 1.0, 4.0, RATE), RATE)

    assert_contract(result)
    assert result["status"] == "unavailable"
    assert "onnxruntime is not installed" in result["reason"]
    settings = by_name(result)["speech_spans"]["settings"]
    assert settings["onnxruntime_version"] is None
    assert settings["faster_whisper_version"] is None
    assert by_name(result)["f0_hz"]["settings"]["parselmouth_version"] is None


def test_a_detector_failure_becomes_unavailable_and_other_errors_still_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import faster_whisper.vad

    def unreadable_model():
        return faster_whisper.vad.SileroVADModel(str(tmp_path / "missing.onnx"))

    monkeypatch.setattr(faster_whisper.vad, "get_vad_model", unreadable_model)
    result = measure(place(speech_like(1.2, RATE), 1.0, 4.0, RATE), RATE)

    assert_contract(result)
    assert result["status"] == "unavailable"
    assert result["reason"].startswith("Silero VAD could not run: ")
    assert "missing.onnx" in result["reason"]

    def broken(*args, **kwargs):
        raise RuntimeError("Applying the VAD filter requires the onnxruntime package")

    monkeypatch.setattr(faster_whisper.vad, "get_speech_timestamps", broken)
    result = measure(place(speech_like(1.2, RATE), 1.0, 4.0, RATE), RATE)

    assert_contract(result)
    assert result["status"] == "unavailable"
    assert result["reason"] == (
        "Silero VAD could not run: Applying the VAD filter requires the onnxruntime package"
    )

    def bug(*args, **kwargs):
        raise ZeroDivisionError("a bug in the caller is not a detector failure")

    monkeypatch.setattr(faster_whisper.vad, "get_speech_timestamps", bug)
    with pytest.raises(ZeroDivisionError):
        measure(place(speech_like(1.2, RATE), 1.0, 4.0, RATE), RATE)


def test_vad_asset_comes_from_the_installed_faster_whisper_package() -> None:
    name = acoustics.vad_asset_path()

    assert name.startswith("silero") and name.endswith(".onnx")
    assert (
        by_name(measure(np.zeros(RATE, dtype=np.float32), RATE))["speech_spans"]["settings"][
            "asset"
        ]
        == name
    )


@pytest.mark.parametrize(
    ("mono", "rate", "message"),
    [
        (np.zeros((10, 2), dtype=np.float32), RATE, "one-dimensional"),
        (np.zeros(0, dtype=np.float32), RATE, "non-empty"),
        (np.array([0.0, np.nan], dtype=np.float32), RATE, "finite"),
        (np.zeros(10, dtype=np.float32), 0, "positive integer"),
        (np.zeros(10, dtype=np.float32), 16000.0, "positive integer"),
    ],
    ids=["stereo", "empty", "nan", "zero_rate", "float_rate"],
)
def test_bad_input_is_rejected_before_any_measurement(mono, rate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        measure(mono, rate)


def test_praat_version_is_recorded_in_the_f0_settings() -> None:
    values = by_name(measure(np.zeros(RATE, dtype=np.float32), RATE))

    assert values["f0_hz"]["settings"]["praat_version"] == parselmouth.PRAAT_VERSION
    assert values["f0_hz"]["method"] == "praat_parselmouth_ac_percentiles"


@pytest.mark.parametrize("f0", [150.0, 225.0])
@pytest.mark.parametrize("rate", [RATE, 48_000])
def test_control_pitch_of_150_and_225_hz_gives_p50_within_3_percent_through_measure(
    f0: float, rate: int
) -> None:
    """Issue #18 control through the user-visible path: real Silero VAD, real Praat, no spans given.

    A pure sine is not speech for the detector, so the control pitch is carried by the formant
    synthesis; all three percentiles come from voiced frames inside the detected span only.
    """
    result = measure(place(speech_like(1.2, rate, f0=f0), 1.0, 4.0, rate), rate)

    assert_contract(result)
    values = by_name(result)
    assert result["status"] == "completed"
    assert len(values["speech_spans"]["value"]) == 1
    f0_value = values["f0_hz"]["value"]
    assert set(f0_value) == {"p10", "p50", "p90"}
    assert f0_value["p50"] == pytest.approx(f0, rel=0.03)
    assert f0_value["p10"] <= f0_value["p50"] <= f0_value["p90"]
    assert values["f0_hz"]["settings"]["voiced_speech_frames"] > 50


ONNXRUNTIME_TELEMETRY_STORE = Path(".cache/Microsoft/DeveloperTools/.onnxruntime")
ONNXRUNTIME_SESSION_FILE = ".ses"
# onnxruntime writes no telemetry under a CI runner it recognises by these variables, which would
# make the control below fail and the checks above it pass for nothing.
ONNXRUNTIME_CI_MARKERS = (
    "CI",
    "TF_BUILD",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "CODEBUILD_BUILD_ID",
    "BUILDKITE",
    "TEAMCITY_VERSION",
    "APPVEYOR",
    "BITBUCKET_BUILD_NUMBER",
)

VAD_CHILD = """
import json, os, sys
from sound_scout import acoustics
from sound_scout.smoke import place, speech_like_control

imported_early = "onnxruntime" in sys.modules
result = acoustics.measure(place(speech_like_control(1.2, 16000), 1.0, 3.0, 16000), 16000)
spans = {item["name"]: item for item in result["measurements"]}["speech_spans"]["value"]
print(json.dumps({
    "imported_early": imported_early,
    "loaded": "onnxruntime" in sys.modules,
    "status": result["status"],
    "spans": len(spans or []),
    "telemetry_env": os.environ.get("ORT_DISABLE_TELEMETRY"),
}))
"""


def run_with_private_home(tmp_path: Path, script: str, telemetry: str | None) -> tuple[dict, list]:
    """Run `script` in a child with HOME and TMPDIR in empty temp dirs, no CI markers and the
    telemetry variable unset or set to `telemetry`; return its JSON and every path created."""
    home, temp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    temp.mkdir()
    dropped = {*guard.TELEMETRY_OFF_ENV, *ONNXRUNTIME_CI_MARKERS}
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    if telemetry is not None:
        env["ORT_DISABLE_TELEMETRY"] = telemetry
    env.update(HOME=str(home), TMPDIR=str(temp))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    created = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    return json.loads(result.stdout.strip().splitlines()[-1]), [
        path for path in created if path not in {"home", "tmp"}
    ]


@pytest.mark.parametrize("telemetry", [None, "0"])
def test_real_vad_run_writes_no_onnxruntime_telemetry(
    tmp_path: Path, telemetry: str | None
) -> None:
    """R09/R10: a real Silero VAD run switches onnxruntime telemetry off before its import,
    even when the environment had it unset or explicitly on."""
    outcome, created = run_with_private_home(tmp_path, VAD_CHILD, telemetry)

    assert outcome == {
        "imported_early": False,
        "loaded": True,
        "status": "completed",
        "spans": 1,
        "telemetry_env": "1",
    }
    assert not (tmp_path / "home" / ONNXRUNTIME_TELEMETRY_STORE).exists()
    assert not (tmp_path / "tmp" / ONNXRUNTIME_SESSION_FILE).exists()
    assert created == []


def test_plain_onnxruntime_import_writes_telemetry_the_check_above_would_see(
    tmp_path: Path,
) -> None:
    """The control for the test above: the same child, importing onnxruntime itself without
    the variable, does write the store and the session file, so their absence means something."""
    script = 'import json, onnxruntime\nprint(json.dumps({"loaded": True}))\n'
    outcome, _ = run_with_private_home(tmp_path, script, None)

    assert outcome == {"loaded": True}
    assert (tmp_path / "home" / ONNXRUNTIME_TELEMETRY_STORE).is_dir()
    assert (tmp_path / "tmp" / ONNXRUNTIME_SESSION_FILE).exists()
