"""`word_prominence`: the word index of `PROSODY-RULE v1` per word of the trusted ASR segments.

What it is and what it is not (docs/decisions/0038-word-prominence-measurement.md): the confirmatory
run of 0037 checked one thing — that the index separates words with a RuPro `word_pa` pitch accent
from words without one in the same intonation phrase (RUEG 1.0, Russian monolingual narratives).
The value is the raw index, comparable only between words of one ASR segment; there is no
threshold and no 0/1. It is a measurement, never a finding, and it changes no backend status.

The functions from `normalize` to `index` are a copy of `sound_scout.eval.prosody` (rule steps
3–6 of 0019), unchanged: that file is pinned by 0034 and the product does not import the
experiment. `tests/test_prominence.py` checks that both give equal numbers on one input. A change
to the formula here or to `acoustics.py` needs a new decision on what the 0037 check still covers.

The unit of normalisation is one trusted segment of the raw transcript (`asr.trusted_segments`),
cut from the 16 kHz copy from `min(segment start, first word start)` to `max(segment end, last
word end)`, as the run of 0037 cut one IP. A word gets `value: null` with a code:
`no_letters` (nothing left after `normalize`), `no_frames` (no positive length or no track frame),
`no_voiced_frames` (no voiced frame in the segment), `segment_too_short` (the track refused the cut)
and `single_word_segment` (fewer than two words left: a z-score of one word is always 0). Word
times are clamped to the signal, as `asr.suspect_segments` clamps a segment. The word text never
enters the entry: each word is a pointer into `asr/raw.json`.
"""

import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import parselmouth

from sound_scout import acoustics, asr, schema
from sound_scout.doctor import installed_version
from sound_scout.findings import TRANSCRIPT_ARTIFACT_ID

NAME = schema.WORD_MEASUREMENT
UNIT = "z_sum"
METHOD = "prosody_rule_v1_word_index"
SAMPLE_RATE = 16_000
VOWELS = frozenset("аеиоуыэюя")
FEATURES = ("f0", "rms", "dur")
RULE = "PROSODY-RULE v1"
# `prosody.RULE_SHA256` and `prosody_confirm.PROSODY_RULE_SHA256`, checked by a test.
RULE_SHA256 = "b6ce9fcfbf24b8fa706b2470d47b9967f896fcf677fe5b9326b72bf6af28cc65"
VALIDATION = {
    "decision": "docs/decisions/0037-word-prominence-rupro.md",
    "report": "docs/eval/2026-11-28-prosody-rupro-1.md",
    "rule": "PROSODY-RUPRO-RULE v1",
    "rule_sha256": "6571cfe492e65cd4ef520068c51f0c3e9fedef5945d9aec769a5dec8c77df68c",
    "target": "pitch accent presence (RuPro word_pa) within one intonation phrase",
    "verdict": "PASS",
    "auc": 0.763,
    "ci95": [0.739, 0.788],
    "coverage": 0.831,
}
LIMITATIONS = [
    "Проверено только наличие pitch accent: на RuPro (RUEG 1.0, монолингвальные пересказы на "
    "русском, 37 говорящих) AUC 0,763, 95 % CI 0,739–0,788 (0037).",
    "Значения сравнимы только между словами одного сегмента ASR. Порога нет: значение не "
    "означает «акцент есть» или «акцента нет».",
    "Времена слов дают ASR; ошибка ASR сдвигает или теряет слово.",
    "Единица нормировки — сегмент ASR, не интонационная фраза; на RuPro проверялись IP с медианой "
    "2 слова.",
    "Сигнал — производная 16 кГц рабочего моно без усиления.",
    "Измерение не меняет статус audio backend.",
]

NO_LETTERS = "no_letters"
NO_FRAMES = "no_frames"
NO_VOICED_FRAMES = "no_voiced_frames"
SEGMENT_TOO_SHORT = "segment_too_short"
SINGLE_WORD_SEGMENT = "single_word_segment"
REASON_CODES = (NO_LETTERS, NO_FRAMES, NO_VOICED_FRAMES, SEGMENT_TOO_SHORT, SINGLE_WORD_SEGMENT)
NO_TRUSTED_WORDS = "в транскрипте нет доверенных слов"
NO_VALUES = "ни одно слово не получило значение"
VAD_UNAVAILABLE = "детектор речи недоступен"


# --- the index: a copy of sound_scout.eval.prosody, unchanged -------------------------------


def normalize(text: str) -> str:
    """Lower case, ё → е, letters and digits only: no punctuation, marks or combining accents."""
    return "".join(ch for ch in text.casefold().replace("ё", "е") if ch.isalnum())


def vowel_count(form: str) -> int:
    return max(1, sum(ch in VOWELS for ch in form))


def track(samples: np.ndarray) -> dict:
    """`acoustics.acoustic_track` of a 16 kHz cut with its speech mask from the VAD spans."""
    mono = np.asarray(samples, dtype=np.float64)
    return acoustics.acoustic_track(mono, SAMPLE_RATE, acoustics.speech_activity(mono, SAMPLE_RATE))


def frames_of(track_: dict, word: dict) -> np.ndarray:
    times = track_["times_s"]
    return (times >= word["start"]) & (times < word["end"])


def kept_words(track_: dict, words: list[dict]) -> tuple[list[dict], int]:
    """ASR words with a positive duration and at least one frame (rule step 3), and the rest."""
    kept = [w for w in words if w["end"] - w["start"] > 0 and bool(frames_of(track_, w).any())]
    return kept, len(words) - len(kept)


def raw_features(track_: dict, words: list[dict]) -> list[dict]:
    """f0 (semitones), rms (dB) and dur (log) of each word against its utterance (rule step 5)."""
    f0, rms = track_["f0_hz"], track_["rms_dbfs"]
    voiced = f0 > 0
    f0_median = float(np.median(f0[voiced])) if voiced.any() else None
    speech_levels = rms[track_["speech"] & np.isfinite(rms)]
    rms_median = float(np.median(speech_levels)) if len(speech_levels) else None
    durations = [math.log((w["end"] - w["start"]) / vowel_count(w["form"])) for w in words]
    dur_median = float(np.median(durations)) if durations else 0.0
    result = []
    for word, duration in zip(words, durations, strict=True):
        frames = frames_of(track_, word)
        word_f0 = f0[frames & voiced]
        word_rms = rms[frames & np.isfinite(rms)]
        result.append(
            {
                "f0": 12 * math.log2(float(word_f0.max()) / f0_median)
                if len(word_f0) and f0_median
                else 0.0,
                "rms": float(word_rms.max()) - rms_median
                if len(word_rms) and rms_median is not None
                else 0.0,
                "dur": duration - dur_median,
            }
        )
    return result


def zscores(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    sigma = float(array.std()) if len(array) else 0.0
    if sigma == 0:
        return [0.0] * len(array)
    return [float(v) for v in (array - array.mean()) / sigma]


def index(features: list[dict]) -> list[dict]:
    """z-scores of each feature within the utterance and their equal-weight sum (rule step 6)."""
    z = {name: zscores([f[name] for f in features]) for name in FEATURES}
    return [
        {**{f"z_{name}": z[name][i] for name in FEATURES}, "index": sum(z[n][i] for n in FEATURES)}
        for i in range(len(features))
    ]


# --- the measurement -----------------------------------------------------------------------


def measure_words(mono16k: np.ndarray, raw: dict, duration_s: float) -> dict:
    """`{"entry", "reason"}`: the `word_prominence` entry of `measurements[]`, or `None` and why.

    `mono16k` is the 16 kHz copy the transcript `raw` was made from; `duration_s` is the source
    duration the word times are clamped to."""
    trusted, _ = asr.trusted_segments(raw)
    positions = {id(segment): i for i, segment in enumerate(raw["segments"])}
    segments = [s for s in trusted if s.get("words")]
    if not segments:
        return {"entry": None, "reason": NO_TRUSTED_WORDS}
    limit = min(float(duration_s), len(mono16k) / SAMPLE_RATE)
    values: list[dict] = []
    try:
        for segment in segments:
            values += _segment_values(mono16k, segment, positions[id(segment)], limit)
    except acoustics.VadUnavailable as error:
        return {"entry": None, "reason": f"{VAD_UNAVAILABLE}: {error.reason}"}
    missing = Counter(v["reason"] for v in values if v["value"] is None)
    with_value = len(values) - sum(missing.values())
    if not with_value:
        return {"entry": None, "reason": f"{NO_VALUES}: {_codes(missing)}"}
    entry = {
        "name": NAME,
        "value": values,
        "unit": UNIT,
        "method": METHOD,
        "settings": _settings(len(segments), len(values), with_value, missing),
        "validation": dict(VALIDATION, ci95=list(VALIDATION["ci95"])),
        "limitations": list(LIMITATIONS),
        "reason": None,
    }
    return {"entry": entry, "reason": None}


def _segment_values(mono16k: np.ndarray, segment: dict, position: int, limit: float) -> list[dict]:
    words = segment["words"]
    begin = max(0.0, min(float(segment["start"]), float(words[0]["start"])))
    end = min(limit, max(float(segment["end"]), float(words[-1]["end"])))
    values = []
    shifted = []
    for j, word in enumerate(words):
        start, stop = float(word["start"]), float(word["end"])
        values.append(
            {
                "start": _clamped(start, limit),
                "end": _clamped(stop, limit),
                "transcript_ref": {
                    "artifact_ref": TRANSCRIPT_ARTIFACT_ID,
                    "pointer": f"$.segments[{position}].words[{j}]",
                },
                "value": None,
                "reason": None,
            }
        )
        form = normalize(str(word.get("word") or ""))
        if form:
            shifted.append({"form": form, "start": start - begin, "end": stop - begin, "at": j})
        else:
            values[j]["reason"] = NO_LETTERS
    cut = mono16k[max(0, round(begin * SAMPLE_RATE)) : max(0, round(end * SAMPLE_RATE))]
    try:
        track_ = track(cut)
    except ValueError:
        return _without(values, shifted, SEGMENT_TOO_SHORT)
    kept, _ = kept_words(track_, shifted)
    _without(values, [w for w in shifted if w not in kept], NO_FRAMES)
    if not (track_["f0_hz"] > 0).any():
        return _without(values, kept, NO_VOICED_FRAMES)
    if len(kept) < 2:
        return _without(values, kept, SINGLE_WORD_SEGMENT)
    for word, score in zip(kept, index(raw_features(track_, kept)), strict=True):
        values[word["at"]]["value"] = score["index"]
    return values


def _without(values: list[dict], words: list[dict], code: str) -> list[dict]:
    for word in words:
        values[word["at"]]["reason"] = code
    return values


def _clamped(time_s: float, limit: float) -> float:
    return min(max(time_s, 0.0), limit)


def _codes(missing: Counter) -> str:
    return ", ".join(f"{code} — {missing[code]}" for code in REASON_CODES if missing[code])


def _settings(segments: int, words: int, with_value: int, missing: Counter) -> dict:
    return {
        "rule": RULE,
        "rule_sha256": RULE_SHA256,
        "features": list(FEATURES),
        "weights": "equal",
        "normalization_unit": "trusted_asr_segment",
        "signal": "work/mono16k",
        "sample_rate": SAMPLE_RATE,
        "f0": {
            "algorithm": "praat to_pitch_ac",
            "praat_version": parselmouth.PRAAT_VERSION,
            "parselmouth_version": installed_version("praat-parselmouth"),
            "time_step_s": acoustics.STEP_S,
            "pitch_floor_hz": acoustics.F0_FLOOR_HZ,
            "pitch_ceiling_hz": acoustics.F0_CEILING_HZ,
            "silence_threshold": acoustics.F0_SILENCE_THRESHOLD,
            "voicing_threshold": acoustics.F0_VOICING_THRESHOLD,
        },
        "rms": {"window_ms": round(acoustics.RMS_WINDOW_S * 1000), "gain_db": 0.0},
        "speech_mask": "silero_vad on the segment cut, settings as speech_spans",
        "code_sha256": {
            "prominence.py": _sha256(Path(__file__)),
            "acoustics.py": _sha256(Path(acoustics.__file__)),
        },
        "counts": {
            "segments": segments,
            "words": words,
            "with_value": with_value,
            "without_value": {code: missing[code] for code in REASON_CODES if missing[code]},
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
