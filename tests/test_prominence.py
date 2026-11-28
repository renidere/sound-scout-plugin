"""`prominence.measure_words`: the `word_prominence` entry on synthesized audio (decision 0038).

The signal is formant-synthesised (`smoke.speech_like_control`), the words and the transcript
are written by the test: this proves the contract and that the copy of the formula gives the
numbers of `eval.prosody`, never that the index hears anything.
"""

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sound_scout import acoustics, prominence, schema
from sound_scout.eval import prosody, prosody_confirm, prosody_rupro
from sound_scout.smoke import speech_like_control

RATE = prominence.SAMPLE_RATE
REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "schema" / "examples" / "synthetic-result.json"
COPIED = ("normalize", "vowel_count", "track", "frames_of", "kept_words", "raw_features")
COPIED += ("zscores", "index")
# (text, seconds, f0 Hz, amplitude): words that differ in pitch, level and length
WORDS = (
    (" Мама", 0.30, 120.0, 0.25),
    (" мыла", 0.45, 190.0, 0.6),
    (" раму", 0.25, 125.0, 0.3),
    (" утром.", 0.40, 110.0, 0.4),
)
GAP_S = 0.05
LEAD_S = 0.2


def utterance(words=WORDS) -> tuple[np.ndarray, list[dict]]:
    """Speech-like words one after another after a lead of silence, with their ASR times."""
    pieces, spans, at = [np.zeros(int(LEAD_S * RATE), np.float32)], [], LEAD_S
    for text, seconds, f0, amplitude in words:
        voiced = speech_like_control(seconds, RATE, f0=f0) * (amplitude / 0.5)
        pieces += [voiced.astype(np.float32), np.zeros(int(GAP_S * RATE), np.float32)]
        spans.append({"word": text, "start": round(at, 3), "end": round(at + seconds, 3)})
        at += seconds + GAP_S
    pieces.append(np.zeros(int(LEAD_S * RATE), np.float32))
    return np.concatenate(pieces), spans


def segment(words: list[dict], start: float, end: float, suspect: bool = False) -> dict:
    return {
        "start": start,
        "end": end,
        "text": "".join(w["word"] for w in words) or " ",
        "suspect_hallucination": suspect,
        "words": [dict(w, probability=0.9) for w in words],
    }


def transcript(*segments: dict) -> dict:
    return {"segments": list(segments)}


@pytest.fixture(scope="module")
def spoken() -> tuple[np.ndarray, list[dict]]:
    return utterance()


def test_product_index_equals_eval_index(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    raw = transcript(segment(words, 0.0, duration))

    found = prominence.measure_words(samples, raw, duration)

    forms = [{"form": prosody.normalize(w["word"]), "marked": False} for w in words]
    expected = prosody.score_utterance(prosody.track(samples), words, forms)
    assert [w["score"]["index"] for w in expected["words"]] == [
        v["value"] for v in found["entry"]["value"]
    ]
    assert len({v["value"] for v in found["entry"]["value"]}) == len(words)


def test_the_copied_functions_are_the_eval_functions_unchanged() -> None:
    for name in COPIED:
        assert inspect.getsource(getattr(prominence, name)) == inspect.getsource(
            getattr(prosody, name)
        ), name
    assert (prominence.SAMPLE_RATE, prominence.VOWELS, prominence.FEATURES) == (
        prosody.SAMPLE_RATE,
        prosody.VOWELS,
        prosody.FEATURES,
    )
    assert prominence.RULE_SHA256 == prosody_confirm.PROSODY_RULE_SHA256 == prosody.RULE_SHA256
    assert prominence.VALIDATION["rule_sha256"] == prosody_rupro.RULE_SHA256


def test_pinned_files_keep_the_0034_hashes() -> None:
    for name, pinned in prosody_confirm.FROZEN_SHA256.items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == pinned, name


def test_the_entry_names_method_settings_validation_and_limits_and_no_word_text(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    raw = transcript(segment(words, 0.0, duration))

    entry = prominence.measure_words(samples, raw, duration)["entry"]

    assert (entry["name"], entry["unit"], entry["method"]) == (
        "word_prominence",
        "z_sum",
        "prosody_rule_v1_word_index",
    )
    assert entry["reason"] is None
    settings = entry["settings"]
    assert settings["rule"] == "PROSODY-RULE v1"
    assert settings["features"] == ["f0", "rms", "dur"]
    assert settings["normalization_unit"] == "trusted_asr_segment"
    assert settings["f0"]["pitch_floor_hz"] == acoustics.F0_FLOOR_HZ
    assert (
        settings["code_sha256"]["acoustics.py"]
        == (prosody_confirm.FROZEN_SHA256["src/sound_scout/acoustics.py"])
    )
    assert settings["counts"] == {"segments": 1, "words": 4, "with_value": 4, "without_value": {}}
    validation = entry["validation"]
    assert validation["decision"] == "docs/decisions/0037-word-prominence-rupro.md"
    assert validation["report"] == "docs/eval/2026-11-28-prosody-rupro-1.md"
    assert (validation["auc"], validation["ci95"], validation["coverage"]) == (
        0.763,
        [0.739, 0.788],
        0.831,
    )
    limits = " ".join(entry["limitations"])
    assert "только наличие pitch accent" in limits
    assert "монолингвальные пересказы на русском" in limits
    assert "Времена слов дают ASR" in limits
    for value, word in zip(entry["value"], words, strict=True):
        assert (value["start"], value["end"]) == (word["start"], word["end"])
    pointers = [v["transcript_ref"] for v in entry["value"]]
    assert pointers == [
        {"artifact_ref": "transcript-raw", "pointer": f"$.segments[0].words[{j}]"} for j in range(4)
    ]
    text = json.dumps(entry, ensure_ascii=False).casefold()
    assert not any(prosody.normalize(w["word"]) in text for w in words)
    for root in ("дикц", "просод", "фокус", "слух", "diction", "focus", "hearing"):
        assert root not in " ".join(entry["limitations"]).casefold(), root


def test_the_entry_passes_the_result_contract(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    raw = transcript(segment(words, 0.0, duration), segment([], 0.0, 0.0))
    result = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    entry = prominence.measure_words(samples, raw, duration)["entry"]
    result["measurements"] = [m for m in result["measurements"] if m["name"] != entry["name"]]
    result["measurements"].append(entry)

    assert schema.validation_errors(result) == []


def test_the_index_is_normalised_within_each_trusted_segment(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    flagged = segment([dict(words[0], word=" выдумка")], 0.0, 0.1, suspect=True)
    first, second = segment(words[:2], 0.0, 0.95), segment(words[2:], 0.95, duration)
    raw = transcript(flagged, first, second)

    entry = prominence.measure_words(samples, raw, duration)["entry"]

    pointers = [v["transcript_ref"]["pointer"] for v in entry["value"]]
    assert pointers == [
        "$.segments[1].words[0]",
        "$.segments[1].words[1]",
        "$.segments[2].words[0]",
        "$.segments[2].words[1]",
    ]
    values = [v["value"] for v in entry["value"]]
    assert values[0] == pytest.approx(-values[1]) and values[2] == pytest.approx(-values[3])
    assert entry["settings"]["counts"]["segments"] == 2


def test_raw_transcript_is_not_changed(spoken) -> None:
    samples, words = spoken
    raw = transcript(segment(words, 0.0, len(samples) / RATE))
    before = copy.deepcopy(raw)

    prominence.measure_words(samples, raw, len(samples) / RATE)

    assert raw == before


def by_pointer(found: dict) -> dict[str, dict]:
    return {v["transcript_ref"]["pointer"]: v for v in found["entry"]["value"]}


def test_a_word_without_letters_and_one_without_length_get_null_with_their_codes(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    dash = {"word": " —", "start": words[1]["end"], "end": words[2]["start"]}
    empty = {"word": " и", "start": words[3]["end"], "end": words[3]["end"]}
    raw = transcript(segment([words[0], words[1], dash, words[2], words[3], empty], 0.0, duration))

    found = prominence.measure_words(samples, raw, duration)

    values = by_pointer(found)
    assert values["$.segments[0].words[2]"] | {"transcript_ref": None} == {
        "start": dash["start"],
        "end": dash["end"],
        "transcript_ref": None,
        "value": None,
        "reason": "no_letters",
    }
    assert values["$.segments[0].words[5]"]["reason"] == "no_frames"
    assert values["$.segments[0].words[5]"]["start"] == values["$.segments[0].words[5]"]["end"]
    assert found["entry"]["settings"]["counts"]["without_value"] == {
        "no_letters": 1,
        "no_frames": 1,
    }
    expected = prominence.measure_words(
        samples, transcript(segment(words, 0.0, duration)), duration
    )
    kept = [values[f"$.segments[0].words[{j}]"]["value"] for j in (0, 1, 3, 4)]
    assert kept == [v["value"] for v in expected["entry"]["value"]]


def test_a_single_word_segment_gets_no_value_and_the_others_keep_theirs(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    raw = transcript(segment(words[:1], 0.0, 0.5), segment(words[1:], 0.5, duration))

    found = prominence.measure_words(samples, raw, duration)

    alone = by_pointer(found)["$.segments[0].words[0]"]
    assert (alone["value"], alone["reason"]) == (None, "single_word_segment")
    others = [v for v in found["entry"]["value"] if v is not alone]
    assert all(v["value"] is not None and v["reason"] is None for v in others)


def test_a_segment_without_voice_gives_no_entry_with_the_reason() -> None:
    samples = np.zeros(RATE, np.float32)
    words = [
        {"word": " тихо", "start": 0.1, "end": 0.4},
        {"word": " тут", "start": 0.5, "end": 0.8},
    ]

    found = prominence.measure_words(samples, transcript(segment(words, 0.0, 1.0)), 1.0)

    assert found["entry"] is None
    assert found["reason"] == f"{prominence.NO_VALUES}: no_voiced_frames — 2"


def test_a_segment_shorter_than_the_track_minimum_gives_segment_too_short(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    short = [{"word": " а", "start": 0.0, "end": 0.02}, {"word": " и", "start": 0.02, "end": 0.04}]
    raw = transcript(segment(short, 0.0, 0.05), segment(words, 0.05, duration))

    found = prominence.measure_words(samples, raw, duration)

    values = by_pointer(found)
    assert values["$.segments[0].words[0]"]["reason"] == "segment_too_short"
    assert values["$.segments[0].words[1]"]["reason"] == "segment_too_short"
    assert values["$.segments[1].words[0]"]["value"] is not None


def test_word_times_are_clamped_to_the_signal(spoken) -> None:
    samples, words = spoken
    duration = len(samples) / RATE
    late = dict(words[3], end=duration + 0.3)
    raw = transcript(segment([*words[:3], late], 0.0, duration))

    found = prominence.measure_words(samples, raw, duration - 0.01)

    assert found["entry"]["value"][-1]["end"] == duration - 0.01


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [segment([{"word": " выдумка", "start": 0.1, "end": 0.5}], 0.0, 0.6, suspect=True)],
        [segment([], 0.0, 0.6)],
    ],
    ids=["no segments", "only flagged", "no words"],
)
def test_no_trusted_words_gives_no_entry_with_the_reason(segments: list[dict]) -> None:
    samples = np.zeros(RATE, np.float32)

    found = prominence.measure_words(samples, transcript(*segments), 1.0)

    assert found == {"entry": None, "reason": prominence.NO_TRUSTED_WORDS}


def test_without_the_speech_detector_there_is_no_entry(
    spoken, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples, words = spoken

    def unavailable(mono: np.ndarray, rate: int) -> list[dict]:
        raise acoustics.VadUnavailable("onnxruntime is not installed")

    monkeypatch.setattr(acoustics, "speech_activity", unavailable)

    found = prominence.measure_words(samples, transcript(segment(words, 0.0, 2.0)), 2.0)

    assert found == {
        "entry": None,
        "reason": f"{prominence.VAD_UNAVAILABLE}: onnxruntime is not installed",
    }
