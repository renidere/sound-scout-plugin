"""`findings`: backend observations on the source timescale, the unlocalized list, ASR
disagreements, and the validator over the document and the run's files (issue #31)."""

import dataclasses
import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sound_scout import asr, findings, ingest, review, schema
from sound_scout._schema_messages import schema_errors
from sound_scout.clips import Clip, cut
from sound_scout.config import Config
from sound_scout.eval import manifest
from sound_scout.findings import Built, build
from sound_scout.runs import Run, open_run
from sound_scout.schema import ResultValidationError
from tests.test_clips import RATE, burst
from tests.test_ingest import write_wav
from tests.test_review import FakeBackend
from tests.test_schema import EXAMPLE_DOC

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)

IDENTITY = {"audio": None, "model": None, "preprocessing": None, "ranges": None}
SOURCE_DURATION = 2.5
MARKER = "MARKER-a1b2c3"
CLIP_START, CLIP_END = 0.85, 1.65  # cut(1.0, 1.5) padded by 150 ms
NEGATIVE = {
    "category": "pronunciation",
    "observation": "окончание второго слова смазано",
    "start_seconds": 0.2,
    "end_seconds": 0.4,
    "confidence_note": "слышно отчётливо, согласный на конце неясен",
    "impact": "слушатель может услышать другое слово",
    "correction": "повторить слово с ясным конечным согласным",
}
NEUTRAL = {
    **NEGATIVE,
    "category": "semantic_stress",
    "observation": "выделено первое слово фразы",
    "start_seconds": 0.1,
    "end_seconds": 0.3,
    "correction": None,
}
WHOLE_CLIP = {
    **NEGATIVE,
    "category": "pause_phrasing",
    "observation": "фраза разорвана паузой",
    "start_seconds": 0.0,
    "end_seconds": 0.8,
    "correction": "убрать паузу внутри фразы",
}
UNLOCALIZED = {
    "category": "completeness",
    "observation": "конец фразы звучит подвешенным",
    "confidence_note": "место точно не определено",
    "impact": "фраза кажется незаконченной",
    "correction": None,
}
RESPONSE = {
    "observations": [NEGATIVE, NEUTRAL, WHOLE_CLIP],
    "unlocalized_observations": [UNLOCALIZED],
    "uncertainties": [],
    "limitations": ["слышен только переданный фрагмент"],
}
NO_INTERFERENCE = "в рассмотренном фрагменте помех не обнаружено"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    root, data_root = tmp_path / "recordings", tmp_path / "data"
    root.mkdir()
    data_root.mkdir()
    return Config(allowed_roots=(root,), data_root=data_root)


@pytest.fixture
def run(config: Config) -> Run:
    run = open_run("req-1", IDENTITY, config.data_root)
    source = write_wav(config.allowed_roots[0] / "burst.wav", burst(), RATE)
    ingest.decode(source, run, config)
    return run


@pytest.fixture
def clip(run: Run, config: Config) -> Clip:
    return cut(run, 1.0, 1.5, config=config)


def reviewed(
    run: Run, config: Config, clip: Clip, response: object = RESPONSE, payload=None
) -> review.ReviewOutcome:
    text = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
    return review.run_review(
        FakeBackend(text=text),
        payload or review.clip_from_cut(clip),
        run,
        config,
        review.load_prompt(),
    )


def raw_segment(words: list[dict], signals: list[str] | None = None) -> dict:
    """A raw transcript segment over `words`, flagged `suspect_hallucination` when `signals`."""
    return {
        "start": words[0]["start"] if words else 0.0,
        "end": words[-1]["end"] if words else 0.0,
        "text": "x",
        "words": words,
        "hallucination_signals": list(signals or []),
        "suspect_hallucination": bool(signals),
    }


def write_raw(run: Run, words: list[dict], segments: list[dict] | None = None) -> dict:
    (run.path / "asr").mkdir()
    raw = {"kind": "transcript_raw", "segments": segments or [raw_segment(words)]}
    (run.path / "asr" / "raw.json").write_text(json.dumps(raw, ensure_ascii=False), "utf-8")
    return asr.read_raw(run)


def document(run: Run, built: Built, outcome: review.ReviewOutcome) -> dict:
    """The synthetic example carrying what `build` made, as #33 will assemble it."""
    doc = deepcopy(EXAMPLE_DOC)
    doc["run_id"] = run.run_id
    doc["source"]["duration_s"] = SOURCE_DURATION
    # its words point into the example's own transcript artifact and 12 s source
    doc["measurements"] = [m for m in doc["measurements"] if m["name"] != "word_prominence"]
    doc["findings"] = deepcopy(built.findings)
    doc["unlocalized"] = deepcopy(built.unlocalized)
    doc["artifacts"] = deepcopy(built.artifacts)
    raw = any(a["id"] == "transcript-raw" for a in built.artifacts)
    doc["transcript"] = {
        "raw": {"artifact_ref": "transcript-raw"} if raw else None,
        "corrections": deepcopy(built.corrections),
        "suspect_segments": [],
    }
    doc["coverage"] = {
        "requested": deepcopy(built.reviewed),
        "reviewed": deepcopy(built.reviewed),
        "unreviewed": [],
    }
    doc["stages"]["audio_review"] = {
        "requested": True,
        "status": "completed",
        "reason": None,
        "elapsed_ms": 10,
    }
    doc["provenance"].update(
        backend="test-double",
        input_modalities=["audio", "text"],
        audio_payload_sha256=outcome.provenance["payload"]["sha256"],
        prompt_sha256=outcome.provenance["prompt"]["sha256"],
        synthetic=built.synthetic,
    )
    doc["errors"] = deepcopy(built.errors)
    return doc


@pytest.fixture
def built(run: Run, config: Config, clip: Clip) -> tuple[Built, review.ReviewOutcome]:
    outcome = reviewed(run, config, clip)
    return build(run, [outcome], [], None, config=config), outcome


@pytest.fixture
def doc(run: Run, built) -> dict:
    return document(run, *built)


def assert_rejected(doc: dict, fragment: str, run: Run | None = None) -> list[str]:
    with pytest.raises(ResultValidationError) as excinfo:
        findings.validate(doc, run=run)
    errors = excinfo.value.errors
    assert any(fragment in error for error in errors), errors
    assert errors == findings.validation_errors(doc, run=run)
    return errors


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- build ------------------------------------------------------------------------------------


def test_build_maps_observations_onto_the_source_timescale(run: Run, clip: Clip, built) -> None:
    result, _ = built
    assert (clip.source_start, clip.source_end) == (CLIP_START, CLIP_END)
    review_dir = run.path / "review" / clip.id
    clip_ref = {"type": "clip", "ref": clip.id, "sha256": sha256_of(clip.path)}
    review_ref = {
        "type": "audio_model_observation",
        "ref": f"review-{clip.id}",
        "sha256": sha256_of(review_dir / "response.raw.txt"),
    }
    assert result.findings == [
        {
            "id": f"f-{clip.id}-1",
            "category": "pronunciation",
            "observation": NEGATIVE["observation"],
            "start": 1.05,
            "end": 1.25,
            "evidence_basis": "audio_model_observation",
            "evidence_refs": [clip_ref, review_ref],
            "confidence_note": NEGATIVE["confidence_note"],
            "impact": NEGATIVE["impact"],
            "correction": NEGATIVE["correction"],
            "polarity": "negative",
        },
        {
            "id": f"f-{clip.id}-2",
            "category": "semantic_stress",
            "observation": NEUTRAL["observation"],
            "start": 0.95,
            "end": 1.15,
            "evidence_basis": "audio_model_observation",
            "evidence_refs": [clip_ref, review_ref],
            "confidence_note": NEUTRAL["confidence_note"],
            "impact": NEUTRAL["impact"],
            "correction": None,
            "polarity": "neutral",
        },
    ]
    assert result.findings[0]["start"] != NEGATIVE["start_seconds"]
    assert result.artifacts == [
        clip.artifact,
        {
            "id": f"review-{clip.id}",
            "kind": "review_response",
            "path": f"review/{clip.id}/response.raw.txt",
            "sha256": review_ref["sha256"],
            "source_start": CLIP_START,
            "source_end": CLIP_END,
        },
        {
            "id": f"review-{clip.id}-provenance",
            "kind": "review_provenance",
            "path": f"review/{clip.id}/provenance.json",
            "sha256": sha256_of(review_dir / "provenance.json"),
            "source_start": CLIP_START,
            "source_end": CLIP_END,
        },
    ]
    assert result.reviewed == [{"start": CLIP_START, "end": CLIP_END}]
    assert result.corrections == [] and result.errors == []
    assert result.synthetic is True


def test_unlocalized_and_whole_clip_observations_stay_out_of_findings(clip: Clip, built) -> None:
    result, _ = built
    assert [(u["id"], u["reason"], u["category"]) for u in result.unlocalized] == [
        (f"u-{clip.id}-1", "whole_clip", "pause_phrasing"),
        (f"u-{clip.id}-2", "no_timecodes", "completeness"),
    ]
    assert result.unlocalized[1]["observation"] == UNLOCALIZED["observation"]
    assert all("start" not in u and "end" not in u for u in result.unlocalized)
    assert all(u["evidence_basis"] == "audio_model_observation" for u in result.unlocalized)
    observations = {f["observation"] for f in result.findings}
    assert WHOLE_CLIP["observation"] not in observations
    assert UNLOCALIZED["observation"] not in observations


def test_built_document_passes_the_validator_with_its_run(run: Run, doc: dict) -> None:
    findings.validate(doc, run=run)
    assert doc["provenance"]["synthetic"] is True and doc["provenance"]["backend"] == "test-double"
    assert findings.validate(doc) is None


def test_review_clip_from_cut_is_the_payload_the_build_checks(clip: Clip) -> None:
    payload = review.clip_from_cut(clip)
    assert payload.id == clip.id and payload.sample_rate == RATE and payload.channels == 1
    assert payload.duration_s == pytest.approx(CLIP_END - CLIP_START)
    assert np.frombuffer(payload.payload, dtype="<f4").size == round((CLIP_END - CLIP_START) * RATE)


def test_payload_of_another_clip_is_refused(run: Run, config: Config, clip: Clip) -> None:
    other = review.clip_from_cut(clip)
    reversed_pcm = np.frombuffer(other.payload, dtype="<f4")[::-1].tobytes()
    outcome = reviewed(run, config, clip, payload=review.make_clip(clip.id, reversed_pcm, RATE))
    result = build(run, [outcome], [], None, config=config)
    assert [e["code"] for e in result.errors] == ["review_payload_mismatch"]
    assert result.errors[0]["stage"] == "audio_review"
    assert result.findings == result.unlocalized == result.reviewed == result.artifacts == []


def test_invalid_response_adds_nothing_to_reviewed(run: Run, config: Config, clip: Clip) -> None:
    outcome = reviewed(run, config, clip, response="not json")
    result = build(run, [outcome], [], None, config=config)
    assert [e["code"] for e in result.errors] == ["review_response_invalid"]
    assert result.reviewed == [] and result.findings == []


def test_review_of_a_clip_not_in_the_run_is_refused(run: Run, config: Config, clip: Clip) -> None:
    stray = review.make_clip("clip-1-2", review.clip_from_cut(clip).payload, RATE)
    result = build(run, [reviewed(run, config, clip, payload=stray)], [], None, config=config)
    assert [e["code"] for e in result.errors] == ["review_clip_missing"]


def test_changed_review_record_is_refused(run: Run, config: Config, clip: Clip) -> None:
    outcome = reviewed(run, config, clip)
    outcome.raw_response_path.write_text("{}", encoding="utf-8")
    result = build(run, [outcome], [], None, config=config)
    assert [e["code"] for e in result.errors] == ["review_record_mismatch"]
    assert result.findings == []


def test_observation_past_the_clip_end_is_no_finding(run: Run, config: Config, clip) -> None:
    past_end = {**NEGATIVE, "start_seconds": 0.8001, "end_seconds": 0.8005}
    response = {**RESPONSE, "observations": [past_end, NEUTRAL]}
    result = build(run, [reviewed(run, config, clip, response)], [], None, config=config)
    assert [e["code"] for e in result.errors] == ["review_interval_empty"]
    assert result.errors[0]["stage"] == "audio_review"
    assert [f["observation"] for f in result.findings] == [NEUTRAL["observation"]]
    assert all(f["start"] < f["end"] for f in result.findings)
    assert result.reviewed == [{"start": CLIP_START, "end": CLIP_END}]


def test_test_double_and_real_backend_cannot_be_mixed(run: Run, config: Config, clip) -> None:
    outcome = reviewed(run, config, clip)
    real = replace(outcome, provenance={**outcome.provenance, "backend": {"name": "qwen2-audio"}})
    with pytest.raises(ValueError, match="cannot make one result"):
        build(run, [outcome, real], [], None, config=config)


# --- ASR vs audio model -----------------------------------------------------------------------


def test_disagreement_with_asr_keeps_both_versions_uncertain(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, [{"start": 1.1, "end": 1.2, "word": " приме", "probability": 0.9}])
    raw_path = run.path / "asr" / "raw.json"
    raw_before = sha256_of(raw_path)
    outcome = reviewed(run, config, clip)
    result = build(run, [outcome], [], transcript, config=config)

    assert result.corrections == [
        {
            "start": 1.05,
            "end": 1.25,
            "original": "приме",
            "corrected": NEGATIVE["observation"],
            "source": "audio_model_observation",
            "evidence_refs": [
                result.findings[0]["evidence_refs"][1],
                {"type": "transcript", "ref": "transcript-raw", "sha256": raw_before},
            ],
            "certainty": "uncertain",
            "note": findings.DISAGREEMENT_NOTE,
        }
    ]
    assert sha256_of(raw_path) == raw_before
    assert asr.read_raw(run) == transcript
    assert {"id": "transcript-raw", "kind": "transcript_raw"}.items() <= result.artifacts[
        -1
    ].items()
    findings.validate(document(run, result, outcome), run=run)


def test_no_disagreement_without_overlapping_words_or_transcript(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, [{"start": 2.0, "end": 2.3, "word": " слово", "probability": 0.9}])
    outcome = reviewed(run, config, clip)
    assert build(run, [outcome], [], transcript, config=config).corrections == []
    assert build(run, [outcome], [], None, config=config).corrections == []


def test_neutral_pronunciation_over_words_is_no_disagreement(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, [{"start": 1.1, "end": 1.2, "word": " приме", "probability": 0.9}])
    response = {**RESPONSE, "observations": [{**NEGATIVE, "correction": None}]}
    outcome = reviewed(run, config, clip, response)
    assert build(run, [outcome], [], transcript, config=config).corrections == []


SUSPECT_WORD = {"start": 1.1, "end": 1.2, "word": " выдумка", "probability": 0.95}


def with_raw_artifact(run: Run, doc: dict, transcript: dict) -> dict:
    """`doc` citing the run's `asr/raw.json` as its raw transcript, as the pipeline does."""
    doc["artifacts"].append(
        {
            "id": "transcript-raw",
            "kind": "transcript_raw",
            "path": "asr/raw.json",
            "sha256": sha256_of(run.path / "asr" / "raw.json"),
            "source_start": None,
            "source_end": None,
        }
    )
    doc["transcript"]["raw"] = {"artifact_ref": "transcript-raw"}
    doc["transcript"]["suspect_segments"] = asr.suspect_segments(transcript, SOURCE_DURATION)
    return doc


def test_a_flagged_segment_is_never_quoted_as_a_disagreement(
    run: Run, config: Config, clip: Clip
) -> None:
    segments = [raw_segment([SUSPECT_WORD], ["silent_span"])]
    transcript = write_raw(run, [], segments)
    raw_before = sha256_of(run.path / "asr" / "raw.json")
    outcome = reviewed(run, config, clip)

    result = build(run, [outcome], [], transcript, config=config)

    assert result.corrections == []
    assert result.findings[0]["category"] == "pronunciation", "the finding itself stays"
    assert "выдумка" not in json.dumps(dataclasses.asdict(result), ensure_ascii=False)
    assert sha256_of(run.path / "asr" / "raw.json") == raw_before
    doc = with_raw_artifact(run, document(run, result, outcome), transcript)
    assert doc["transcript"]["suspect_segments"] == [
        {"start": 1.1, "end": 1.2, "signals": ["silent_span"], "certainty": "uncertain"}
    ]
    findings.validate(doc, run=run)


WORDS = [
    {"start": 1.1, "end": 1.2, "word": " раз", "probability": 0.9},
    {"start": 1.2, "end": 1.4, "word": " два", "probability": 0.9},
]


def with_word_prominence(run: Run, doc: dict, pointers: list[str]) -> tuple[dict, int]:
    """`doc` with a `word_prominence` entry whose words point into the run's raw transcript."""
    entry = deepcopy(next(m for m in EXAMPLE_DOC["measurements"] if m["name"] == "word_prominence"))
    entry["value"] = [
        {
            "start": 1.1 + 0.1 * j,
            "end": 1.2 + 0.1 * j,
            "transcript_ref": {"artifact_ref": "transcript-raw", "pointer": pointer},
            "value": 1.0 - 2 * j,
            "reason": None,
        }
        for j, pointer in enumerate(pointers)
    ]
    doc["measurements"].append(entry)
    return doc, len(doc["measurements"]) - 1


def test_word_prominence_pointers_must_name_words_of_the_hashed_raw_transcript(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, WORDS)
    outcome = reviewed(run, config, clip)
    result = build(run, [outcome], [], transcript, config=config)
    pointers = ["$.segments[0].words[0]", "$.segments[0].words[1]"]
    doc, _ = with_word_prominence(run, document(run, result, outcome), pointers)
    findings.validate(doc, run=run)

    doc, index = with_word_prominence(
        run,
        document(run, result, outcome),
        ["$.segments[0].words[1]", "$.segments[0].words[2]", "$.segments[999].words[0]"],
    )
    errors = assert_rejected(doc, "is not a word of the raw transcript", run)
    assert errors == [
        f"$.measurements[{index}].value[{j}].transcript_ref.pointer: {pointer} is not a word "
        "of the raw transcript"
        for j, pointer in ((1, "$.segments[0].words[2]"), (2, "$.segments[999].words[0]"))
    ]


def test_word_prominence_checks_its_raw_transcript_file_even_when_nothing_else_cites_it(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, WORDS)
    response = {**RESPONSE, "observations": [{**NEGATIVE, "correction": None}]}
    outcome = reviewed(run, config, clip, response)
    result = build(run, [outcome], [], transcript, config=config)
    assert not result.corrections and "transcript-raw" not in {a["id"] for a in result.artifacts}
    doc = with_raw_artifact(run, document(run, result, outcome), transcript)
    doc["transcript"].update(raw=None, suspect_segments=[])
    doc, index = with_word_prominence(run, doc, ["$.segments[0].words[0]"])
    findings.validate(doc, run=run)

    (run.path / "asr" / "raw.json").unlink()
    [artifact] = [i for i, a in enumerate(doc["artifacts"]) if a["id"] == "transcript-raw"]
    assert assert_rejected(doc, "file is missing", run) == [
        f"$.artifacts[{artifact}]: file is missing"
    ]
    doc["artifacts"][artifact]["sha256"] = None
    assert_rejected(
        doc,
        f"$.measurements[{index}].value[0].transcript_ref.artifact_ref: 'transcript-raw' must "
        "carry a sha256",
    )


def test_a_transcript_without_the_flag_is_refused_by_build(
    run: Run, config: Config, clip: Clip
) -> None:
    segment = raw_segment([SUSPECT_WORD])
    del segment["suspect_hallucination"]
    transcript = write_raw(run, [], [segment])
    with pytest.raises(ValueError, match=r"segments\[0\]\.suspect_hallucination"):
        build(run, [reviewed(run, config, clip)], [], transcript, config=config)


def test_validate_with_the_run_requires_the_flagged_segments_of_raw_json(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, [], [raw_segment([SUSPECT_WORD], ["silent_span"])])
    outcome = reviewed(run, config, clip)
    doc = with_raw_artifact(
        run,
        document(run, build(run, [outcome], [], transcript, config=config), outcome),
        transcript,
    )
    findings.validate(doc, run=run)
    mismatch = "$.transcript.suspect_segments: does not match the suspect_hallucination segments"

    doc["transcript"]["suspect_segments"] = []
    findings.validate(doc)
    assert_rejected(doc, mismatch, run)
    doc["transcript"]["suspect_segments"] = asr.suspect_segments(transcript, SOURCE_DURATION)
    doc["transcript"]["suspect_segments"][0]["end"] = 1.3
    assert_rejected(doc, mismatch, run)

    raw_path = run.path / "asr" / "raw.json"
    raw_path.write_text(json.dumps({"segments": [{"text": " выдумка"}]}), "utf-8")
    doc["artifacts"][-1]["sha256"] = sha256_of(raw_path)
    errors = assert_rejected(
        doc, "$.transcript.raw: raw transcript: segments[0].suspect_hallucination", run
    )
    assert not any("выдумка" in error for error in errors)


# --- document rules ---------------------------------------------------------------------------


def test_finding_outside_coverage_reviewed_is_rejected(doc: dict) -> None:
    doc["coverage"]["reviewed"] = [{"start": CLIP_START, "end": 1.0}]
    doc["coverage"]["unreviewed"] = [{"start": 1.0, "end": CLIP_END}]
    doc["status"] = "partial"
    assert_rejected(doc, "$.findings[0]: outside coverage.reviewed")


def widen_coverage(doc: dict) -> None:
    doc["coverage"]["requested"] = [{"start": 0.0, "end": SOURCE_DURATION}]
    doc["coverage"]["reviewed"] = [{"start": 0.0, "end": SOURCE_DURATION}]


def test_finding_inside_reviewed_but_outside_its_review_response_is_rejected(doc: dict) -> None:
    widen_coverage(doc)
    findings.validate(doc)
    doc["findings"][0].update(start=1.7, end=1.9)
    errors = assert_rejected(doc, "$.findings[0]: outside review_response")
    assert not any("outside coverage.reviewed" in error for error in errors)


def test_review_response_without_a_source_range_does_not_qualify(doc: dict) -> None:
    for artifact in doc["artifacts"][1:3]:
        artifact.update(source_start=None, source_end=None)
    assert_rejected(doc, "$.findings[0]: review_response 'review-")
    assert_rejected(doc, "has no source range")


def correction(doc: dict, start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "original": "приме",
        "corrected": "окончание смазано",
        "source": "audio_model_observation",
        "evidence_refs": [deepcopy(doc["findings"][0]["evidence_refs"][1])],
        "certainty": "uncertain",
        "note": None,
    }


def test_audio_correction_outside_reviewed_or_its_review_response_is_rejected(doc: dict) -> None:
    doc["transcript"]["corrections"] = [correction(doc, 1.1, 1.2)]
    findings.validate(doc)
    doc["transcript"]["corrections"] = [correction(doc, 1.7, 1.9)]
    assert_rejected(doc, "$.transcript.corrections[0]: outside coverage.reviewed")
    widen_coverage(doc)
    errors = assert_rejected(doc, "$.transcript.corrections[0]: outside review_response")
    assert not any("outside coverage.reviewed" in error for error in errors)


def test_evidence_hash_must_match_the_artifact(doc: dict) -> None:
    doc["findings"][0]["evidence_refs"][1]["sha256"] = "00" * 32
    assert_rejected(doc, "$.findings[0].evidence_refs[1].sha256: does not match artifact")
    doc["findings"][0]["evidence_refs"][1]["ref"] = "review-missing"
    assert_rejected(doc, "$.findings[0].evidence_refs[1].ref: unknown artifact 'review-missing'")


def test_every_finding_cites_a_clip_that_covers_it(doc: dict) -> None:
    clip_ref = doc["findings"][0]["evidence_refs"].pop(0)
    assert_rejected(doc, "$.findings[0].evidence_refs: needs at least one reference of type 'clip'")
    doc["findings"][0]["evidence_refs"].append(clip_ref)
    widen_coverage(doc)
    doc["findings"][0].update(start=0.5, end=1.0)
    assert_rejected(doc, f"$.findings[0]: outside clip {clip_ref['ref']!r} source range")
    doc["unlocalized"][0]["evidence_refs"].pop(0)
    assert_rejected(doc, "$.unlocalized[0].evidence_refs: needs at least one reference of type")


def test_ids_are_unique_across_findings_and_unlocalized(doc: dict) -> None:
    doc["unlocalized"][0]["id"] = doc["findings"][0]["id"]
    assert_rejected(doc, f"$.unlocalized[0].id: duplicate id {doc['findings'][0]['id']!r}")


def test_review_response_needs_its_provenance_record(doc: dict) -> None:
    del doc["artifacts"][2]
    assert_rejected(doc, "has no review_provenance artifact in 'review/clip-")


def test_shifted_clip_range_must_match_the_clip_record(run: Run, doc: dict) -> None:
    shift = 0.8
    for artifact in doc["artifacts"]:
        artifact["source_start"] = round(artifact["source_start"] + shift, 6)
        artifact["source_end"] = round(artifact["source_end"] + shift, 6)
    for finding in doc["findings"]:
        finding["start"] = round(finding["start"] + shift, 6)
        finding["end"] = round(finding["end"] + shift, 6)
    for name in ("requested", "reviewed"):
        for span in doc["coverage"][name]:
            span["start"] = round(span["start"] + shift, 6)
            span["end"] = round(span["end"] + shift, 6)
    findings.validate(doc)
    assert_rejected(doc, "$.artifacts[0]: clip artifact does not match its record", run)


def test_clip_artifact_needs_its_record_in_the_run(run: Run, doc: dict) -> None:
    (run.path / "clips" / f"{doc['artifacts'][0]['id']}.json").unlink()
    assert_rejected(doc, "$.artifacts[0]: the run holds no clip record", run)


def test_artifact_path_with_a_nul_is_a_validation_error(run: Run, doc: dict) -> None:
    doc["artifacts"][1]["path"] = "review/x\0/response.raw.txt"
    assert_rejected(doc, "$.artifacts[1].path: must not contain a NUL character", run)


def test_unlocalized_observation_is_bounded_by_coverage(doc: dict) -> None:
    doc["coverage"]["reviewed"] = [{"start": CLIP_START, "end": 1.0}]
    doc["coverage"]["unreviewed"] = [{"start": 1.0, "end": CLIP_END}]
    doc["status"] = "partial"
    assert_rejected(doc, "$.unlocalized[0]: clip 'clip-")
    assert_rejected(doc, "$.unlocalized[0]: review_response 'review-")


@pytest.mark.parametrize(
    "category",
    [
        "emotion",
        "health",
        "personality",
        "confidence",
        "attractiveness",
        "уверенность",
        "Self-Confidence",
    ],
)
@pytest.mark.parametrize("field", ["findings", "unlocalized"])
def test_forbidden_inferences_are_rejected(doc: dict, field: str, category: str) -> None:
    doc[field][0]["category"] = category
    errors = assert_rejected(doc, f"$.{field}[0].category: names a forbidden inference (R04, R06)")
    assert not any(category in error for error in errors), errors


def test_category_outside_the_dictionary_or_off_its_basis_is_rejected(doc: dict) -> None:
    doc["findings"][0]["category"] = "charm"
    assert_rejected(doc, "$.findings[0].category: not in the category dictionary (R04, R06)")
    doc["findings"][0]["category"] = "tempo"
    assert_rejected(
        doc,
        "$.findings[0].category: this category cannot rest on evidence_basis "
        "'audio_model_observation', only on ['measurement'] (R04, R06)",
    )
    doc["unlocalized"][0]["category"] = "filler_words"
    assert_rejected(
        doc,
        "$.unlocalized[0].category: this category cannot rest on evidence_basis "
        "'audio_model_observation', only on ['transcript'] (R04, R06)",
    )


@pytest.mark.parametrize(
    ("category", "fragment"),
    [
        (MARKER, "not in the category dictionary (R04, R06)"),
        (f"emotion {MARKER}", "not in the category dictionary (R04, R06)"),
        (" Self-Confidence ", "names a forbidden inference (R04, R06)"),
    ],
)
@pytest.mark.parametrize("field", ["findings", "unlocalized"])
def test_category_errors_do_not_quote_the_category(
    doc: dict, field: str, category: str, fragment: str
) -> None:
    doc[field][0]["category"] = category
    assert not schema_errors(schema.SCHEMA, doc)
    errors = assert_rejected(doc, f"$.{field}[0].category: {fragment}")
    with pytest.raises(ResultValidationError) as excinfo:
        schema.validate(doc)
    for text in (*errors, str(excinfo.value)):
        assert MARKER not in text and category.strip() not in text, text


@pytest.mark.parametrize(
    ("field", "category", "allowed"),
    [("findings", "tempo", "['measurement']"), ("unlocalized", "filler_words", "['transcript']")],
)
def test_category_on_a_wrong_basis_error_does_not_quote_the_category(
    doc: dict, field: str, category: str, allowed: str
) -> None:
    doc[field][0]["category"] = category
    assert not schema_errors(schema.SCHEMA, doc)
    errors = assert_rejected(
        doc,
        f"$.{field}[0].category: this category cannot rest on evidence_basis "
        f"'audio_model_observation', only on {allowed} (R04, R06)",
    )
    with pytest.raises(ResultValidationError) as excinfo:
        schema.validate(doc)
    for text in (*errors, str(excinfo.value)):
        assert category not in text, text


def measured(doc: dict, category: str, polarity: str = "neutral") -> dict:
    clip_ref = deepcopy(doc["findings"][0]["evidence_refs"][0])
    return {
        **deepcopy(doc["findings"][0]),
        "id": "f-measured",
        "category": category,
        "evidence_basis": "measurement",
        "evidence_refs": [{"type": "measurement", "ref": "speech_rate", "sha256": None}, clip_ref],
        "correction": None,
        "polarity": polarity,
    }


def test_sound_category_cannot_rest_on_a_measurement(doc: dict) -> None:
    doc["findings"].append(measured(doc, "pitch_range"))
    findings.validate(doc)
    doc["findings"][-1]["category"] = "semantic_stress"
    assert_rejected(
        doc,
        "$.findings[2].category: this category cannot rest on evidence_basis 'measurement', "
        "only on ['audio_model_observation', 'human_verified'] (R04, R06)",
    )
    doc["findings"][-1]["evidence_refs"][0]["sha256"] = "ab" * 32
    assert_rejected(doc, "evidence_refs[0].sha256: must be null for a measurement")


def test_negative_finding_needs_a_correction(doc: dict) -> None:
    doc["findings"][0]["correction"] = None
    assert_rejected(doc, "$.findings[0].correction: a negative finding needs a short action")
    doc["findings"][0]["correction"] = "  "
    assert_rejected(doc, "$.findings[0].correction: a negative finding needs a short action")


def test_positive_conclusion_is_bounded_by_coverage_reviewed(doc: dict) -> None:
    doc["findings"][1].update(observation=NO_INTERFERENCE, polarity="positive")
    doc["findings"].append(measured(doc, "pause_length", "positive"))
    findings.validate(doc)
    doc["findings"][-1].update(start=1.5, end=2.0)
    doc["coverage"]["requested"] = [{"start": CLIP_START, "end": 2.0}]
    doc["coverage"]["unreviewed"] = [{"start": CLIP_END, "end": 2.0}]
    doc["status"] = "partial"
    errors = assert_rejected(doc, "$.findings[2]: outside coverage.reviewed")
    assert not any(error.startswith("$.findings[1]") for error in errors)


def test_positive_conclusion_without_audio_review_is_rejected(doc: dict) -> None:
    doc["findings"] = [measured(doc, "pause_length", "positive")]
    doc["findings"][0]["observation"] = NO_INTERFERENCE
    doc["unlocalized"] = []
    doc["stages"]["audio_review"].update(status="unavailable", reason="backend не выбран")
    doc["coverage"]["unreviewed"] = deepcopy(doc["coverage"]["requested"])
    doc["coverage"]["reviewed"] = []
    doc["provenance"]["backend"] = "none"
    doc["status"] = "partial"
    assert_rejected(doc, "$.findings[0]: outside coverage.reviewed")
    doc["findings"][0]["polarity"] = "neutral"
    findings.validate(doc)


def test_test_double_result_must_be_marked_synthetic(doc: dict) -> None:
    doc["provenance"]["synthetic"] = False
    assert_rejected(doc, "$.provenance.synthetic: a 'test-double' backend result must be")
    doc["provenance"]["synthetic"] = True
    doc["provenance"]["note"] = "результат"
    assert_rejected(doc, "$.provenance.synthetic")


def test_dictionary_is_one_from_backend_to_eval() -> None:
    assert review.CATEGORIES == schema.SOUND_CATEGORIES == manifest.CATEGORIES
    assert schema.SCHEMA_VERSION == "1.2"
    assert set(schema.CATEGORY_BASES) == set(
        schema.SOUND_CATEGORIES + schema.MEASUREMENT_CATEGORIES + schema.TRANSCRIPT_CATEGORIES
    )


# --- run-directory rules ----------------------------------------------------------------------


def test_missing_or_changed_evidence_file_is_rejected(run: Run, doc: dict) -> None:
    response = run.path / doc["artifacts"][1]["path"]
    original = response.read_bytes()
    response.write_bytes(original + b" ")
    assert_rejected(doc, "$.artifacts[1]: sha256 does not match the file", run)
    response.unlink()
    assert_rejected(doc, "$.artifacts[1]: file is missing", run)
    response.write_bytes(original)
    findings.validate(doc, run=run)
    clip_wav = run.path / doc["artifacts"][0]["path"]
    clip_wav.rename(clip_wav.with_suffix(".bak"))
    clip_wav.symlink_to(clip_wav.with_suffix(".bak"))
    assert_rejected(doc, "$.artifacts[0]: not a regular file", run)


def rewrite_provenance(run: Run, doc: dict, change) -> None:
    path = run.path / doc["artifacts"][2]["path"]
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    doc["artifacts"][2]["sha256"] = sha256_of(path)


def test_provenance_of_an_invalid_response_is_rejected(run: Run, doc: dict) -> None:
    rewrite_provenance(run, doc, lambda r: r["response"].update(valid=False))
    assert_rejected(doc, "$.artifacts[2]: review provenance does not record a valid response", run)


def test_provenance_of_another_run_or_clip_is_rejected(run: Run, doc: dict) -> None:
    rewrite_provenance(run, doc, lambda r: r.update(run_id="other", clip_id="clip-0-1"))
    errors = assert_rejected(doc, "$.artifacts[2]: review provenance belongs to another run", run)
    assert any("was made for clip 'clip-0-1'" in error for error in errors)


def test_test_double_review_cannot_back_a_real_backend_result(run: Run, doc: dict) -> None:
    doc["provenance"].update(backend="qwen2-audio", synthetic=False, note=None)
    assert_rejected(doc, "the review came from the test double, but provenance.backend", run)
    rewrite_provenance(run, doc, lambda r: r["backend"].update(name="other-backend"))
    assert_rejected(doc, "review backend 'other-backend' does not match provenance.backend", run)


# --- content against the stored response -----------------------------------------------------

CHANGED = "текст, которого в ответе backend нет"
MISMATCH = "does not match the review response it cites"


def assert_rejected_quietly(doc: dict, fragment: str, run: Run) -> list[str]:
    """Rejected with `fragment`, and no error repeats text of the document or the response."""
    errors = assert_rejected(doc, fragment, run)
    texts = [CHANGED, *(obs["observation"] for obs in RESPONSE["observations"])]
    texts += [UNLOCALIZED["observation"], NEGATIVE["correction"], "приме"]
    texts += [obs["category"] for obs in RESPONSE["observations"]]
    texts += [UNLOCALIZED["category"], "repetition_self_correction"]
    assert not any(text in error for error in errors for text in texts), errors
    return errors


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start", 1.0),
        ("end", 1.3),
        ("category", "repetition_self_correction"),
        ("observation", CHANGED),
        ("confidence_note", CHANGED),
        ("impact", CHANGED),
        ("correction", CHANGED),
        ("polarity", "neutral"),
    ],
)
def test_a_finding_changed_against_its_stored_response_is_rejected(
    run: Run, doc: dict, field: str, value: object
) -> None:
    findings.validate(doc, run=run)
    doc["findings"][0][field] = value
    findings.validate(doc)
    assert_rejected_quietly(doc, f"$.findings[0]: {MISMATCH}", run)


def test_a_changed_unlocalized_observation_is_rejected(run: Run, doc: dict) -> None:
    doc["unlocalized"][1]["observation"] = CHANGED
    findings.validate(doc)
    assert_rejected_quietly(doc, f"$.unlocalized[1]: {MISMATCH}", run)
    doc["unlocalized"][1]["observation"] = UNLOCALIZED["observation"]
    doc["unlocalized"][0]["reason"] = "no_timecodes"
    assert_rejected_quietly(doc, f"$.unlocalized[0]: {MISMATCH}", run)


@pytest.mark.parametrize("field", ["findings", "unlocalized"])
def test_an_extra_or_missing_entry_against_the_cited_response_is_rejected(
    run: Run, doc: dict, field: str
) -> None:
    extra = {**deepcopy(doc[field][0]), "id": "extra-1"}
    doc[field].append(extra)
    findings.validate(doc)
    assert_rejected_quietly(
        doc, f"$.{field}[2]: the review response it cites makes no such entry", run
    )
    del doc[field][1:]
    findings.validate(doc)
    assert_rejected_quietly(
        doc, f"$.{field}: 1 entries the review response at $.artifacts[1] makes are missing", run
    )


def test_a_finding_moved_off_the_audio_basis_is_missing_from_its_response(
    run: Run, doc: dict
) -> None:
    doc["findings"].pop(1)
    doc["unlocalized"].clear()
    assert_rejected_quietly(doc, "$.unlocalized: 2 entries the review response at", run)


@pytest.fixture
def disagreeing(run: Run, config: Config, clip: Clip) -> dict:
    """A document whose negative pronunciation finding disagrees with an ASR word."""
    transcript = write_raw(run, [{"start": 1.1, "end": 1.2, "word": " приме", "probability": 0.9}])
    outcome = reviewed(run, config, clip)
    doc = document(run, build(run, [outcome], [], transcript, config=config), outcome)
    assert len(doc["transcript"]["corrections"]) == 1
    return doc


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(original=CHANGED),
        lambda c: c.update(corrected=CHANGED),
        lambda c: c.update(start=1.1),
        lambda c: c.update(end=1.2),
    ],
)
def test_a_changed_audio_correction_is_rejected(run: Run, disagreeing: dict, change) -> None:
    findings.validate(disagreeing, run=run)
    change(disagreeing["transcript"]["corrections"][0])
    findings.validate(disagreeing)
    assert_rejected_quietly(disagreeing, f"$.transcript.corrections[0]: {MISMATCH}", run)


def test_an_added_or_dropped_audio_correction_is_rejected(run: Run, disagreeing: dict) -> None:
    corrections = disagreeing["transcript"]["corrections"]
    corrections.append(deepcopy(corrections[0]))
    findings.validate(disagreeing)
    assert_rejected_quietly(
        disagreeing,
        "$.transcript.corrections[1]: the review response it cites makes no such entry",
        run,
    )
    corrections.clear()
    assert_rejected_quietly(
        disagreeing, "$.transcript.corrections: 1 entries the review response at", run
    )


def drop_all_it_makes(doc: dict) -> None:
    doc["findings"].clear()
    doc["unlocalized"].clear()
    doc["transcript"]["corrections"].clear()


def test_every_entry_of_a_response_dropped_is_rejected(run: Run, disagreeing: dict) -> None:
    drop_all_it_makes(disagreeing)
    findings.validate(disagreeing)
    errors = assert_rejected_quietly(
        disagreeing, "$.findings: 2 entries the review response at $.artifacts[1] makes", run
    )
    for path in ("$.unlocalized: 2 entries", "$.transcript.corrections: 1 entries"):
        assert any(error.startswith(path) for error in errors), errors


def test_an_uncited_response_is_checked_against_its_files_and_clip(run: Run, doc: dict) -> None:
    drop_all_it_makes(doc)
    rewrite_provenance(run, doc, lambda r: r.update(run_id="another-run"))
    assert_rejected(doc, "review provenance belongs to another run", run)
    (run.path / "clips" / f"{doc['artifacts'][0]['id']}.json").unlink()
    assert_rejected(doc, "$.artifacts[0]: the run holds no clip record", run)
    (run.path / doc["artifacts"][1]["path"]).write_text("{}", encoding="utf-8")
    assert_rejected(doc, "$.artifacts[1]: sha256 does not match the file", run)


def test_a_response_that_makes_nothing_still_validates(run: Run, config: Config, clip) -> None:
    empty = {**RESPONSE, "observations": [], "unlocalized_observations": []}
    outcome = reviewed(run, config, clip, empty)
    built = build(run, [outcome], [], None, config=config)
    assert built.findings == built.unlocalized == []
    findings.validate(document(run, built, outcome), run=run)


def test_a_broken_file_is_not_read_for_the_content_check(
    run: Run, disagreeing: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes((run.path / "asr" / "raw.json").read_bytes())
    (run.path / "asr" / "raw.json").unlink()
    (run.path / "asr" / "raw.json").symlink_to(outside)
    clip_record = run.path / "clips" / f"{disagreeing['artifacts'][0]['id']}.json"
    (tmp_path / "clip.json").write_bytes(clip_record.read_bytes())
    clip_record.unlink()
    clip_record.symlink_to(tmp_path / "clip.json")
    read: list[str] = []
    original = findings._read_json
    monkeypatch.setattr(
        findings, "_read_json", lambda r, relative: read.append(relative) or original(r, relative)
    )
    assert_rejected(disagreeing, "not a regular file", run)
    assert "asr/raw.json" not in read
    assert not any(relative.startswith("clips/") for relative in read), read


def test_text_redacted_by_the_report_still_matches_its_response(run: Run, doc: dict) -> None:
    original = doc["findings"][0]["observation"]
    doc["findings"][0]["observation"] = original.replace("второго", "<user>")
    findings.validate(doc, run=run)
    doc["findings"][0]["observation"] = "<private path>"
    findings.validate(doc, run=run)
    spoiled = [
        f"<user>{original}",
        original.replace("второго", "<user> лишнее"),
        "<user><user>" + original[1:],
    ]
    for observation in spoiled:
        doc["findings"][0]["observation"] = observation
        assert_rejected(doc, f"$.findings[0]: {MISMATCH}", run)


def test_a_stored_response_the_validator_refuses_is_rejected(run: Run, doc: dict) -> None:
    review_id = doc["artifacts"][1]["id"]
    path = run.path / doc["artifacts"][1]["path"]
    path.write_text('{"observations": "x"}', encoding="utf-8")
    new_sha256 = sha256_of(path)
    rewrite_provenance(run, doc, lambda r: r["response"].update(sha256=new_sha256))
    doc["artifacts"][1]["sha256"] = new_sha256
    for claim in doc["findings"] + doc["unlocalized"]:
        for ref in claim["evidence_refs"]:
            if ref["ref"] == review_id:
                ref["sha256"] = new_sha256
    findings.validate(doc)
    assert_rejected(
        doc, "$.artifacts[1]: review response does not pass the response validator", run
    )


def test_human_verified_without_a_record_on_disk_is_rejected(run: Run, doc: dict) -> None:
    label = {
        "id": "label-1",
        "kind": "human_label",
        "path": "labels/label-1.json",
        "sha256": "ab" * 32,
        "source_start": None,
        "source_end": None,
    }
    doc["artifacts"].append(label)
    doc["findings"][1].update(
        evidence_basis="human_verified",
        evidence_refs=[
            doc["findings"][1]["evidence_refs"][0],
            {"type": "human_label", "ref": "label-1", "sha256": "ab" * 32},
        ],
    )
    findings.validate(doc)
    assert_rejected(doc, "$.artifacts[3]: file is missing", run)
    (run.path / "labels").mkdir()
    (run.path / "labels" / "label-1.json").write_text("{}", encoding="utf-8")
    assert_rejected(doc, "$.artifacts[3]: sha256 does not match the file", run)


# --- acceptance criteria of #31 not covered above ---------------------------------------------


def test_audio_model_observation_without_a_review_artifact_is_rejected(doc: dict) -> None:
    doc["findings"][0]["evidence_refs"] = doc["findings"][0]["evidence_refs"][:1]
    assert_rejected(
        doc,
        "$.findings[0].evidence_refs: evidence_basis 'audio_model_observation' needs at least one "
        "reference of type 'audio_model_observation'",
    )


@pytest.mark.parametrize("version", ["1.0", "1.1", "1.3"])
def test_schema_version_bump_is_required(doc: dict, version: str) -> None:
    findings.validate(doc)
    doc["schema_version"] = version
    assert_rejected(doc, "$.schema_version")


def test_schema_bump_requires_polarity_and_evidence_ref_sha256(doc: dict) -> None:
    del doc["findings"][0]["polarity"]
    assert_rejected(doc, "$.findings[0]: 'polarity' is a required property")
    doc["findings"][0]["polarity"] = "good"
    assert_rejected(doc, "$.findings[0].polarity")
    doc["findings"][0]["polarity"] = "negative"
    del doc["findings"][0]["evidence_refs"][0]["sha256"]
    assert_rejected(doc, "$.findings[0].evidence_refs[0]: 'sha256' is a required property")


# --- overlapping windows (issue #25, criterion 2) --------------------------------------------

WINDOW_A, WINDOW_B = (0.0, 1.5), (1.0, SOURCE_DURATION)  # overlap [1.0, 1.5]
REQUESTED = [(0.0, SOURCE_DURATION)]


def observation(start: float, end: float, **changes: object) -> dict:
    return {**NEGATIVE, "start_seconds": start, "end_seconds": end, **changes}


def windowed(
    run: Run,
    config: Config,
    a: list[dict],
    b: list[dict],
    transcript: dict | None = None,
    requested: list[tuple[float, float]] | None = REQUESTED,
) -> tuple[Built, review.ReviewOutcome]:
    """Two overlapping windows reviewed with observations on each clip's own scale."""
    outcomes = []
    for (start, end), observations in ((WINDOW_A, a), (WINDOW_B, b)):
        response = {**RESPONSE, "observations": observations, "unlocalized_observations": []}
        outcomes.append(reviewed(run, config, cut(run, start, end, 0, config=config), response))
    built = build(run, outcomes, [], transcript, config=config, requested=requested)
    return built, outcomes[0]


def boundary_errors(built: Built) -> list[dict]:
    return [e for e in built.errors if e["code"] == findings.WINDOW_BOUNDARY]


def test_the_same_observation_in_the_overlap_of_two_windows_is_one_finding(
    run: Run, config: Config
) -> None:
    built, first = windowed(run, config, [observation(1.1, 1.4)], [observation(0.15, 0.45)])

    [finding] = built.findings
    assert (finding["start"], finding["end"]) == (1.1, 1.45)
    assert [ref["type"] for ref in finding["evidence_refs"]] == [
        "clip",
        "audio_model_observation",
        "clip",
        "audio_model_observation",
    ]
    assert len({ref["ref"] for ref in finding["evidence_refs"]}) == 4
    assert built.reviewed == [{"start": 0.0, "end": SOURCE_DURATION}]
    assert boundary_errors(built) == []
    findings.validate(document(run, built, first), run=run)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ({"category": "semantic_stress"}, {}),
        ({"correction": None}, {}),
    ],
    ids=["other category", "other polarity"],
)
def test_overlap_observations_of_another_kind_stay_two(run: Run, config: Config, a, b) -> None:
    built, first = windowed(
        run, config, [observation(1.1, 1.4, **a)], [observation(0.15, 0.45, **b)]
    )
    assert len(built.findings) == 2
    findings.validate(document(run, built, first), run=run)


def test_equal_observations_outside_the_overlap_stay_two(run: Run, config: Config) -> None:
    built, first = windowed(run, config, [observation(0.2, 0.5)], [observation(1.2, 1.4)])
    assert [(f["start"], f["end"]) for f in built.findings] == [(0.2, 0.5), (2.2, 2.4)]
    assert boundary_errors(built) == []
    findings.validate(document(run, built, first), run=run)


def test_an_event_cut_by_both_windows_is_merged_across_the_overlap(
    run: Run, config: Config
) -> None:
    built, first = windowed(run, config, [observation(0.8, 1.5)], [observation(0.0, 0.8)])

    [finding] = built.findings
    assert (finding["start"], finding["end"]) == (0.8, 1.8)
    assert boundary_errors(built) == []
    findings.validate(document(run, built, first), run=run)


def test_a_finding_at_an_inner_window_edge_without_a_pair_is_marked(
    run: Run, config: Config
) -> None:
    built, _ = windowed(run, config, [observation(1.2, 1.5)], [observation(1.2, 1.5)])

    [mark] = boundary_errors(built)
    assert mark["stage"] == "audio_review"
    assert mark["message"].startswith(f"finding {built.findings[0]['id']} touches the inner edge")
    assert "at 1.5 s" in mark["message"]


def test_a_merged_finding_keeps_the_mark_of_an_edge_no_window_heard_across(
    run: Run, config: Config
) -> None:
    built, _ = windowed(
        run,
        config,
        [observation(1.1, 1.4)],
        [observation(0.1, 1.5)],
        requested=[(0.0, SOURCE_DURATION + 1.0)],
    )

    [finding] = built.findings
    assert (finding["start"], finding["end"]) == (1.1, SOURCE_DURATION)
    [mark] = boundary_errors(built)
    assert mark["message"].startswith(f"finding {finding['id']} touches the inner edge")
    assert f"at {SOURCE_DURATION} s" in mark["message"]


def windows_reviewed(
    run: Run, config: Config, observed: list[tuple[tuple[float, float], list[dict]]]
) -> tuple[Built, review.ReviewOutcome]:
    outcomes = []
    for (start, end), observations in observed:
        response = {**RESPONSE, "observations": observations, "unlocalized_observations": []}
        outcomes.append(reviewed(run, config, cut(run, start, end, 0, config=config), response))
    return build(run, outcomes, [], None, config=config, requested=REQUESTED), outcomes[0]


def test_an_event_heard_in_three_overlapping_windows_is_one_finding(
    run: Run, config: Config
) -> None:
    built, first = windows_reviewed(
        run,
        config,
        [
            ((0.0, 1.5), [observation(1.3, 1.4)]),
            ((1.0, SOURCE_DURATION), [observation(0.3, 0.4)]),
            ((1.2, SOURCE_DURATION), [observation(0.1, 0.2)]),
        ],
    )

    [finding] = built.findings
    assert (finding["start"], finding["end"]) == (1.3, 1.4)
    assert [ref["type"] for ref in finding["evidence_refs"]] == [
        "clip",
        "audio_model_observation",
    ] * 3
    assert boundary_errors(built) == []
    findings.validate(document(run, built, first), run=run)


def test_a_merge_never_takes_two_findings_of_one_window(run: Run, config: Config) -> None:
    built, first = windows_reviewed(
        run,
        config,
        [
            ((0.0, 1.5), [observation(1.05, 1.15), observation(1.3, 1.45)]),
            ((1.0, SOURCE_DURATION), [observation(0.0, 0.5)]),
        ],
    )

    assert [(f["start"], f["end"]) for f in built.findings] == [(1.0, 1.5), (1.3, 1.45)]
    assert [len(f["evidence_refs"]) for f in built.findings] == [4, 2]
    assert boundary_errors(built) == []
    findings.validate(document(run, built, first), run=run)


def test_without_the_requested_ranges_no_finding_is_marked(run: Run, config: Config) -> None:
    built, _ = windowed(
        run, config, [observation(1.2, 1.5)], [observation(1.2, 1.5)], requested=None
    )
    assert len(built.findings) == 2 and boundary_errors(built) == []


def test_a_finding_at_the_edge_of_the_requested_range_is_not_marked(
    run: Run, config: Config
) -> None:
    built, _ = windowed(run, config, [observation(0.0, 0.3)], [observation(1.2, 1.5)])
    assert len(built.findings) == 2 and boundary_errors(built) == []


def test_a_merged_disagreement_with_asr_is_one_correction_citing_both_windows(
    run: Run, config: Config
) -> None:
    transcript = write_raw(run, [{"start": 1.2, "end": 1.3, "word": " пример", "probability": 0.9}])

    built, first = windowed(
        run, config, [observation(1.1, 1.4)], [observation(0.15, 0.45)], transcript
    )

    [correction_] = built.corrections
    assert [ref["type"] for ref in correction_["evidence_refs"]] == [
        "audio_model_observation",
        "audio_model_observation",
        "transcript",
    ]
    assert (correction_["start"], correction_["end"]) == (1.1, 1.45)
    findings.validate(document(run, built, first), run=run)


def test_the_validator_rebuilds_the_merge_and_rejects_a_changed_merged_finding(
    run: Run, config: Config
) -> None:
    built, first = windowed(run, config, [observation(1.1, 1.4)], [observation(0.15, 0.45)])
    doc = document(run, built, first)
    doc["findings"][0]["end"] = 1.4
    assert_rejected(doc, "$.findings[0]: does not match the review response it cites", run)
    doc = document(run, built, first)
    doc["findings"][0]["evidence_refs"] = doc["findings"][0]["evidence_refs"][:2]
    doc["findings"][0]["end"] = 1.4
    assert_rejected(doc, "$.findings[0]: does not match the review response it cites", run)


def test_clips_with_a_gap_between_them_do_not_cover_a_finding_together(
    run: Run, config: Config
) -> None:
    built, first = windowed(run, config, [observation(0.8, 1.5)], [observation(0.0, 0.8)])
    doc = document(run, built, first)
    for artifact in doc["artifacts"]:
        if artifact["source_start"] == WINDOW_B[0]:
            artifact["source_start"] = 1.6
    errors = assert_rejected(doc, "$.findings[0]: outside clip")
    assert any("$.findings[0]: outside review_response" in error for error in errors)
