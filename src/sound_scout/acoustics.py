"""Layer 2 of R04: measurements of the working mono signal — duration, speech spans, RMS, F0.

Ported from the owner's speech layer (`speech_activity`, `acoustic_track`, spec § 6) with its
proven checks kept: a known tone yields its frequency and a digital, not acoustic, level; silence
yields no fabricated pitch; every number is finite or `null` with a reason. `measure` takes the
mono signal `channels.choose_mono` produced, at its native sample rate, and measures that signal
as it is: nothing here applies gain, trims or normalises, so a louder copy for listening never
changes a number (R04, R06).

Speech spans come from Silero VAD through `faster_whisper.vad`, its ONNX asset shipped inside the
installed `faster-whisper` package (never downloaded) and run by onnxruntime on the CPU. The
detector wants 16 kHz: a signal at another rate is resampled by libswresample (PyAV) for the
detector only, and the spans are reported in seconds of the original. F0 is Praat's raw
autocorrelation through parselmouth over 50–600 Hz at a 10 ms step; RMS is a 40 ms window at
the same step. Percentiles are taken only over frames whose centre lies in a speech span, and
for F0 only over frames Praat found voiced. Without the VAD asset, without onnxruntime or when
the detector fails to load, the stage is `unavailable`: spans, RMS and F0 are `null` with that
reason, never guessed from the whole signal.

The result of `measure` is `{"status", "reason", "measurements"}`; each measurement entry has the
shape of `measurements[]` in the result contract (`name`, `value`, `unit`, `method`, `settings`,
`limitations`, `reason`). The per-frame trajectories `acoustic_track` returns are kept in memory
for the caller; writing them to the run directory is the report's concern.
"""

import os

import numpy as np
import parselmouth

from sound_scout.doctor import VAD_ASSET, installed_version, package_dir

VAD_RATE = 16_000
VAD_WINDOW_SAMPLES = 512
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 300
VAD_MIN_SPEECH_MS = 100
VAD_PAD_MS = 0
VAD_WINDOW_MS = 1000 * VAD_WINDOW_SAMPLES // VAD_RATE
# The detector closes a span up to this many windows after speech ends (docs/decisions/0003).
VAD_END_LAG_WINDOWS = 2
VAD_TAIL_SAMPLES = VAD_RATE * VAD_MIN_SILENCE_MS // 1000 + VAD_END_LAG_WINDOWS * VAD_WINDOW_SAMPLES
F0_FLOOR_HZ = 50.0
F0_CEILING_HZ = 600.0
F0_SILENCE_THRESHOLD = 0.03
F0_VOICING_THRESHOLD = 0.5
STEP_S = 0.01
RMS_WINDOW_S = 0.04
MIN_TRACK_SECONDS = 0.1
PERCENTILES = (10, 50, 90)
MEASUREMENT_NAMES = ("duration_seconds", "speech_spans", "rms_dbfs", "f0_hz")
SPANS_UNAVAILABLE = "speech spans unavailable, so speech frames cannot be selected"
TOO_SHORT = f"signal shorter than {MIN_TRACK_SECONDS} s, the track is not computed"

DBFS_LIMITATION = (
    "dBFS is a digital level relative to full scale, not the loudness of the voice: "
    "it is not dB SPL or LUFS and depends on gain, microphone distance and processing "
    "before the recording"
)
VAD_PAUSE_LIMITATION = (
    "a gap between speech spans is where the detector heard no speech; a VAD pause is not a "
    "hesitation, a filler or a defect, and its cause is unknown"
)
F0_OCTAVE_LIMITATION = "octave errors (halving or doubling of the true F0) are possible"
F0_RANGE_LIMITATION = (
    "F0 is not expressiveness: a narrow range is not monotony and a low voice is not a defect"
)

LIMITATIONS = {
    "duration_seconds": [
        "the duration of the decoded mono signal, not the duration the container declares"
    ],
    "speech_spans": [
        VAD_PAUSE_LIMITATION,
        f"boundaries are quantised to the detector's {VAD_WINDOW_MS} ms windows: the start is "
        "within one window of the speech onset and the end can lag the end of speech by up to "
        f"{VAD_END_LAG_WINDOWS} windows ({VAD_END_LAG_WINDOWS * VAD_WINDOW_MS} ms), as measured "
        "on a synthetic control, not on real speech",
        "the detector runs on a 16 kHz copy made for it alone; the spans are reported on the "
        "time scale of the original signal",
        "a speech span says only that speech was detected, not what was said or how",
    ],
    "rms_dbfs": [
        DBFS_LIMITATION,
        "percentiles over 40 ms frames inside speech spans only; the level between spans is "
        "not included",
        "computed on the working mono signal without gain; a listening copy is a separate "
        "derivative",
    ],
    "f0_hz": [
        F0_OCTAVE_LIMITATION,
        F0_RANGE_LIMITATION,
        "raw autocorrelation (Praat `To Pitch (ac)`), not the filtered variant; percentiles are "
        "over frames Praat found voiced inside speech spans only",
    ],
}


class VadUnavailable(Exception):
    """The speech detector cannot run here; `reason` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def measure(mono: np.ndarray, rate: int) -> dict:
    """Measurements of the working mono signal as `{"status", "reason", "measurements"}`."""
    _require_mono(mono, rate)
    frames = int(mono.shape[0])
    duration = frames / rate
    measurements = [
        _measurement(
            "duration_seconds",
            duration,
            "s",
            "frames / sample_rate",
            {"frames": frames, "sample_rate": rate},
        )
    ]
    try:
        spans = speech_activity(mono, rate)
    except VadUnavailable as error:
        status, reason, spans = "unavailable", error.reason, None
    else:
        status, reason = "completed", None
    measurements.append(
        _measurement(
            "speech_spans",
            spans,
            "s",
            "silero_vad",
            _vad_settings(rate, spans),
            reason=reason,
        )
    )
    measurements += _track_measurements(mono, rate, spans)
    return {"status": status, "reason": reason, "measurements": measurements}


def speech_activity(mono: np.ndarray, rate: int) -> list[dict]:
    """Speech spans `{"start", "end"}` in seconds of the original signal (Silero VAD)."""
    _require_mono(mono, rate)
    vad_asset_path()
    if package_dir("onnxruntime") is None:
        raise VadUnavailable("onnxruntime is not installed, the Silero VAD cannot run")
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_PAD_MS,
    )
    detector_input = mono if rate == VAD_RATE else _resample(mono, rate, VAD_RATE)
    # The detector closes a span only after min_silence of quiet; without this tail a span still
    # open at the end of the signal would be reported as running to the end of the file.
    detector_input = np.concatenate(
        [
            np.ascontiguousarray(detector_input, dtype=np.float32),
            np.zeros(VAD_TAIL_SAMPLES, dtype=np.float32),
        ]
    )
    try:
        timestamps = get_speech_timestamps(detector_input, options, sampling_rate=VAD_RATE)
    except Exception as error:
        if not _is_detector_failure(error):
            raise
        raise VadUnavailable(f"Silero VAD could not run: {error}") from error
    duration = mono.shape[0] / rate
    return [
        {"start": item["start"] / VAD_RATE, "end": min(item["end"] / VAD_RATE, duration)}
        for item in timestamps
    ]


def acoustic_track(mono: np.ndarray, rate: int, spans: list[dict]) -> dict:
    """Per-frame trajectories at a 10 ms step: time, speech mask, F0 (0 = unvoiced), RMS.

    `rms_dbfs` is NaN where the 40 ms window holds digital silence. Raises `ValueError` for a
    signal shorter than `MIN_TRACK_SECONDS`, which Praat cannot track at a 50 Hz floor.
    """
    _require_mono(mono, rate)
    if mono.shape[0] < rate * MIN_TRACK_SECONDS:
        raise ValueError(TOO_SHORT)
    pitch = parselmouth.Sound(mono.astype(np.float64), rate).to_pitch_ac(
        time_step=STEP_S,
        pitch_floor=F0_FLOOR_HZ,
        pitch_ceiling=F0_CEILING_HZ,
        silence_threshold=F0_SILENCE_THRESHOLD,
        voicing_threshold=F0_VOICING_THRESHOLD,
    )
    times = pitch.xs()
    speech = np.zeros(len(times), dtype=bool)
    for span in spans:
        speech |= (times >= span["start"]) & (times < span["end"])
    return {
        "times_s": times,
        "speech": speech,
        "f0_hz": pitch.selected_array["frequency"],
        "pitch_strength": pitch.selected_array["strength"],
        "rms_dbfs": _windowed_rms_dbfs(mono, rate, times),
    }


def vad_asset_path() -> str:
    """Name of the Silero ONNX asset inside the installed faster-whisper package."""
    try:
        from faster_whisper.utils import get_assets_path
    except ImportError as error:
        raise VadUnavailable(f"faster_whisper cannot be imported: {error}") from error
    assets = get_assets_path()
    try:
        names = sorted(os.listdir(assets))
    except OSError:
        names = []
    for name in names:
        if VAD_ASSET.match(name) and os.path.isfile(os.path.join(assets, name)):
            return name
    raise VadUnavailable("Silero VAD asset not found in faster_whisper")


def _is_detector_failure(error: Exception) -> bool:
    # faster_whisper signals a missing onnxruntime with RuntimeError; onnxruntime's own errors
    # (a missing or unreadable asset, an unsupported graph) share no base class but its module.
    return isinstance(error, RuntimeError | OSError) or type(error).__module__.startswith(
        "onnxruntime"
    )


def _track_measurements(mono: np.ndarray, rate: int, spans: list[dict] | None) -> list[dict]:
    rms_settings = {
        "window_ms": round(RMS_WINDOW_S * 1000),
        "step_ms": round(STEP_S * 1000),
        "percentiles": list(PERCENTILES),
        "reference": "full scale 1.0",
        "gain_db": 0.0,
        "speech_frames": None,
    }
    f0_settings = {
        "algorithm": "praat to_pitch_ac",
        "praat_version": parselmouth.PRAAT_VERSION,
        "parselmouth_version": installed_version("praat-parselmouth"),
        "time_step_s": STEP_S,
        "pitch_floor_hz": F0_FLOOR_HZ,
        "pitch_ceiling_hz": F0_CEILING_HZ,
        "silence_threshold": F0_SILENCE_THRESHOLD,
        "voicing_threshold": F0_VOICING_THRESHOLD,
        "percentiles": list(PERCENTILES),
        "voiced_speech_frames": None,
    }
    rms_value = f0_value = None
    if spans is None:
        rms_reason = f0_reason = SPANS_UNAVAILABLE
    else:
        try:
            track = acoustic_track(mono, rate, spans)
        except ValueError as error:
            rms_reason = f0_reason = str(error)
        else:
            speech = track["speech"]
            levels = track["rms_dbfs"][speech & np.isfinite(track["rms_dbfs"])]
            voiced = track["f0_hz"][speech & (track["f0_hz"] > 0)]
            rms_settings["speech_frames"] = int(speech.sum())
            f0_settings["voiced_speech_frames"] = int(len(voiced))
            rms_value, rms_reason = _percentiles(levels, "no speech frames with signal")
            f0_value, f0_reason = _percentiles(voiced, "no voiced frames inside speech spans")
    return [
        _measurement(
            "rms_dbfs",
            rms_value,
            "dBFS",
            "windowed_rms_percentiles",
            rms_settings,
            reason=rms_reason,
        ),
        _measurement(
            "f0_hz",
            f0_value,
            "Hz",
            "praat_parselmouth_ac_percentiles",
            f0_settings,
            reason=f0_reason,
        ),
    ]


def _percentiles(values: np.ndarray, empty_reason: str) -> tuple[dict | None, str | None]:
    if not len(values):
        return None, empty_reason
    points = np.percentile(values, PERCENTILES)
    return {f"p{p}": float(v) for p, v in zip(PERCENTILES, points, strict=True)}, None


def _windowed_rms_dbfs(mono: np.ndarray, rate: int, centres_s: np.ndarray) -> np.ndarray:
    half = round(rate * RMS_WINDOW_S / 2)
    centres = np.rint(centres_s * rate).astype(int)
    left = np.maximum(0, centres - half)
    right = np.minimum(mono.shape[0], centres + half)
    squares = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    power = np.maximum(0.0, (squares[right] - squares[left]) / (right - left))
    dbfs = np.full(len(power), np.nan)
    nonzero = power > 0
    dbfs[nonzero] = 10 * np.log10(power[nonzero])
    return dbfs


def _resample(mono: np.ndarray, rate: int, target: int) -> np.ndarray:
    import av

    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(mono, dtype=np.float32)[np.newaxis, :], format="flt", layout="mono"
    )
    frame.sample_rate = rate
    resampler = av.AudioResampler(format="flt", layout="mono", rate=target)
    pieces = resampler.resample(frame) + resampler.resample(None)
    return np.concatenate([piece.to_ndarray()[0] for piece in pieces])


def _vad_settings(rate: int, spans: list[dict] | None) -> dict:
    return {
        "detector": "silero_vad via faster_whisper.vad.get_speech_timestamps",
        "asset": _optional(vad_asset_path),
        "faster_whisper_version": installed_version("faster-whisper"),
        "onnxruntime_version": installed_version("onnxruntime"),
        "threshold": VAD_THRESHOLD,
        "neg_threshold": round(max(VAD_THRESHOLD - 0.15, 0.01), 6),
        "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
        "min_speech_duration_ms": VAD_MIN_SPEECH_MS,
        "speech_pad_ms": VAD_PAD_MS,
        "detector_sample_rate": VAD_RATE,
        "window_ms": VAD_WINDOW_MS,
        "resampled_for_detector": rate != VAD_RATE,
        "resampler": None
        if rate == VAD_RATE
        else f"libswresample via PyAV {installed_version('av')}",
        "speech_seconds": None if spans is None else sum(s["end"] - s["start"] for s in spans),
    }


def _optional(lookup) -> str | None:
    try:
        return lookup()
    except VadUnavailable:
        return None


def _measurement(
    name: str,
    value: object,
    unit: str,
    method: str,
    settings: dict,
    *,
    reason: str | None = None,
) -> dict:
    if value is None and not reason:
        raise ValueError(f"{name}: a null value needs a reason")
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "method": method,
        "settings": settings,
        "limitations": list(LIMITATIONS[name]),
        "reason": reason,
    }


def _require_mono(mono: np.ndarray, rate: int) -> None:
    if not isinstance(mono, np.ndarray) or mono.ndim != 1 or not mono.shape[0]:
        raise ValueError("expected a non-empty one-dimensional array of samples")
    if not np.isfinite(mono).all():
        raise ValueError("samples must be finite")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError(f"sample rate must be a positive integer, got {rate!r}")
