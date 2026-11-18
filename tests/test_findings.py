"""`findings`: backend observations on the source timescale, the unlocalized list, ASR
disagreements, and the validator over the document and the run's files (issue #31)."""

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sound_scout import asr, findings, ingest, review, schema
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


def write_raw(run: Run, words: list[dict]) -> dict:
    (run.path / "asr").mkdir()
    raw = {"kind": "transcript_raw", "segments": [{"text": "x", "words": words}]}
    (run.path / "asr" / "raw.json").write_text(json.dumps(raw, ensure_ascii=False), "utf-8")
    return asr.read_raw(run)


def document(run: Run, built: Built, outcome: review.ReviewOutcome) -> dict:
    """The synthetic example carrying what `build` made, as #33 will assemble it."""
    doc = deepcopy(EXAMPLE_DOC)
    doc["run_id"] = run.run_id
    doc["source"]["duration_s"] = SOURCE_DURATION
    doc["findings"] = deepcopy(built.findings)
    doc["unlocalized"] = deepcopy(built.unlocalized)
    doc["artifacts"] = deepcopy(built.artifacts)
    raw = any(a["id"] == "transcript-raw" for a in built.artifacts)
    doc["transcript"] = {
        "raw": {"artifact_ref": "transcript-raw"} if raw else None,
        "corrections": deepcopy(built.corrections),
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
    assert_rejected(doc, f"$.{field}[0].category: forbidden inference {category!r} (R04, R06)")


def test_category_outside_the_dictionary_or_off_its_basis_is_rejected(doc: dict) -> None:
    doc["findings"][0]["category"] = "charm"
    assert_rejected(doc, "$.findings[0].category: 'charm' is not in the category dictionary")
    doc["findings"][0]["category"] = "tempo"
    assert_rejected(doc, "'tempo' cannot rest on evidence_basis 'audio_model_observation'")
    doc["unlocalized"][0]["category"] = "filler_words"
    assert_rejected(doc, "$.unlocalized[0].category: 'filler_words' cannot rest on")


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
    assert_rejected(doc, "'semantic_stress' cannot rest on evidence_basis 'measurement'")
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
    assert schema.SCHEMA_VERSION == "1.1"
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


@pytest.mark.parametrize("version", ["1.0", "1.2"])
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
