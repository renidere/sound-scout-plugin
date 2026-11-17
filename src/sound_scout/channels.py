"""Channel analysis of the decoded original: what each channel holds and which one to work on (T02).

Ported from the owner's speech layer (`channel_report`, `choose_mono`, spec § 6) with its checks
kept: a silent channel is never averaged into the active one, duplicated stereo is one voice
rather than two, and a zero signal or phase cancellation stops the analysis instead of yielding
invented numbers. `load` reads the `work/decoded.f32` that `ingest.decode` wrote into a
frames × channels float32 array; `report` measures every channel on that original signal (peak
and RMS in dBFS, share of zero samples, inter-channel correlation) and gives one verdict. The
version is bounded at `config.MAX_CHANNELS` (spec § 2); `budgets.max_channels` cannot exceed it
and `ingest.probe` refuses a wider file, so more channels here raise `ValueError`
rather than averaging channels the checks below do not cover:

- `zero`: every channel is digital silence;
- `mono`: one channel;
- `one_active`: one channel carries signal, the other only zeros;
- `cancellation`: the arithmetic mean is more than `CANCELLATION_LOSS_DB` below the quieter
  channel, so the channels largely cancel each other (L = -R);
- `duplicated_stereo`: both channels carry the same signal (absolute correlation at least
  `DUPLICATE_CORRELATION`), possibly at different levels or with inverted polarity;
- `stereo`: two distinct active channels.

`choose_mono` turns the report into the working mono signal and a `channel_choice`
transformation record in the shape of `provenance.transformations` of the result contract.
Nothing is resampled, trimmed or amplified: the mono signal has the frames of the original and
`gain_db` is always 0; a louder copy for listening is a separate derivative. A `zero` or
`cancellation` verdict raises `ChannelError` with `code` `zero_signal` or `cancellation`, the
reason the prepare stage reports as `failed`.
"""

import math

import numpy as np

from sound_scout.config import MAX_CHANNELS
from sound_scout.ingest import BYTES_PER_SAMPLE, Source

ZERO_SIGNAL = "zero_signal"
CANCELLATION = "cancellation"
REASON_CODES = (ZERO_SIGNAL, CANCELLATION)
TRANSFORMATION_NAME = "channel_choice"
DUPLICATE_CORRELATION = 0.99
CANCELLATION_LOSS_DB = 6.0
_SAMPLE_DTYPE = np.dtype("<f4")


class ChannelError(Exception):
    """The channels leave nothing to analyse; `code` is one of `REASON_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def load(source: Source) -> np.ndarray:
    """The decoded samples of `source` as a frames × channels float32 array, untouched."""
    channels = source.probe.channels
    expected = source.frames * channels * BYTES_PER_SAMPLE
    if source.decoded_path.stat().st_size != expected:
        raise ValueError(f"{source.decoded_path.name} does not hold {source.frames} frames")
    return np.fromfile(source.decoded_path, dtype=_SAMPLE_DTYPE).reshape(source.frames, channels)


def report(audio: np.ndarray) -> dict:
    """Per-channel levels, zero share, inter-channel correlation and the verdict; JSON-ready."""
    _require_samples(audio)
    channels = [_channel_report(index, channel) for index, channel in enumerate(audio.T)]
    active = [item for item in channels if item["zero_share"] < 1.0]
    correlation = _correlation(audio) if audio.shape[1] == 2 else None
    mean_rms = _rms(audio.mean(axis=1, dtype=np.float64))
    quietest = min((item["rms_linear"] for item in active), default=0.0)
    cancels = len(active) > 1 and mean_rms < quietest * 10 ** (-CANCELLATION_LOSS_DB / 20)
    return {
        "channels": channels,
        "active_channels": [item["channel"] for item in active],
        "correlation": correlation,
        "arithmetic_mean_rms_dbfs": _dbfs(mean_rms),
        "arithmetic_mean_loss_db": _loss_db(quietest, mean_rms) if len(active) > 1 else None,
        "verdict": _verdict(audio.shape[1], active, correlation, cancels),
    }


def choose_mono(audio: np.ndarray, channel_report: dict) -> tuple[np.ndarray, dict]:
    """The working mono signal and its `channel_choice` transformation record.

    The signal keeps the frames and sample positions of `audio`: a channel is taken as it is or
    distinct channels are averaged, never a silent one. Raises `ChannelError` on a `zero` or
    `cancellation` verdict.
    """
    _require_samples(audio)
    verdict = channel_report["verdict"]
    active = channel_report["active_channels"]
    if verdict == "zero":
        raise ChannelError(ZERO_SIGNAL, "every channel is digital silence")
    if verdict == "cancellation":
        loss = channel_report["arithmetic_mean_loss_db"]
        lost = "to digital silence" if loss is None else f"{loss:.1f} dB"
        raise ChannelError(
            CANCELLATION,
            f"the channels cancel each other, the arithmetic mean loses {lost}; "
            "automatic downmix refused",
        )
    if verdict == "mono":
        selected, method = 1, "single_channel"
        reason = "the original has one channel"
    elif verdict == "one_active":
        selected, method = active[0], "select_channel"
        reason = f"channel {selected} carries the signal, the other is digital silence"
    elif verdict == "duplicated_stereo":
        loudest = max(channel_report["channels"], key=lambda item: item["rms_linear"])
        selected, method = loudest["channel"], "select_channel"
        correlation = channel_report["correlation"]
        polarity = " with inverted polarity" if correlation < 0 else ""
        reason = (
            f"both channels carry the same signal{polarity} (correlation {correlation:.3f}), "
            f"channel {selected} has the higher RMS"
        )
    else:
        selected, method = None, "arithmetic_mean"
        reason = "distinct active channels are averaged"
    if selected is not None:
        mono = np.array(audio[:, selected - 1], dtype=_SAMPLE_DTYPE)
    else:
        mono = audio.mean(axis=1, dtype=np.float64).astype(_SAMPLE_DTYPE)
    record = {
        "name": TRANSFORMATION_NAME,
        "settings": {
            "method": method,
            "selected_channel": selected,
            "source_channels": audio.shape[1],
            "verdict": verdict,
            "reason": reason,
            "gain_db": 0.0,
            "frames": int(audio.shape[0]),
            "correlation": channel_report["correlation"],
            "arithmetic_mean_loss_db": channel_report["arithmetic_mean_loss_db"],
            "duplicate_correlation": DUPLICATE_CORRELATION,
            "cancellation_loss_db": CANCELLATION_LOSS_DB,
        },
    }
    return mono, record


def _require_samples(audio: np.ndarray) -> None:
    if audio.ndim != 2 or not audio.shape[0] or not audio.shape[1]:
        raise ValueError("expected a non-empty frames x channels array")
    if audio.shape[1] > MAX_CHANNELS:
        raise ValueError(f"{audio.shape[1]} channels, the version handles at most {MAX_CHANNELS}")
    if not np.isfinite(audio).all():
        raise ValueError("samples must be finite")


def _channel_report(index: int, channel: np.ndarray) -> dict:
    samples = len(channel)
    zero_samples = samples - int(np.count_nonzero(channel))
    peak = float(np.max(np.abs(channel)))
    rms = _rms(channel)
    return {
        "channel": index + 1,
        "samples": samples,
        "zero_samples": zero_samples,
        "zero_share": zero_samples / samples,
        "peak_linear": peak,
        "peak_dbfs": _dbfs(peak),
        "rms_linear": rms,
        "rms_dbfs": _dbfs(rms),
    }


def _correlation(audio: np.ndarray) -> float | None:
    left, right = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _loss_db(reference: float, actual: float) -> float | None:
    """How much `actual` is below `reference` in dB; None when the mean is digital silence."""
    if actual == 0 or reference == 0:
        return None
    return 20 * math.log10(reference / actual)


def _verdict(
    channel_count: int, active: list[dict], correlation: float | None, cancels: bool
) -> str:
    if not active:
        return "zero"
    if channel_count == 1:
        return "mono"
    if len(active) == 1:
        return "one_active"
    if cancels:
        return "cancellation"
    if correlation is not None and abs(correlation) >= DUPLICATE_CORRELATION:
        return "duplicated_stereo"
    return "stereo"


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _dbfs(linear: float) -> float | None:
    return 20 * math.log10(linear) if linear > 0 else None
