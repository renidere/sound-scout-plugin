"""Result contract v1 (issue #11): the schema files, the example and the cross-field rules."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sound_scout import schema
from sound_scout._schema_messages import schema_errors
from sound_scout.schema import ResultValidationError, validate, validation_errors

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_SCHEMA = REPO_ROOT / "schema" / "result.v1.schema.json"
PACKAGED_SCHEMA = REPO_ROOT / "src" / "sound_scout" / "result.v1.schema.json"
EXAMPLE = REPO_ROOT / "schema" / "examples" / "synthetic-result.json"
TOP_LEVEL_KEYS = (
    "schema_version",
    "request_id",
    "run_id",
    "status",
    "source",
    "capabilities",
    "stages",
    "coverage",
    "provenance",
    "transcript",
    "measurements",
    "findings",
    "assessment_status",
    "artifacts",
    "limits",
    "errors",
    "cost",
    "unlocalized",
)
HEX64 = "ab" * 32
EXAMPLE_DOC = json.loads(EXAMPLE.read_text(encoding="utf-8"))
WHOLE_CLIP = "clip-0-192000"
WHOLE_CLIP_SHA = "1c" * 32
REVIEW_DIR = f"review/{WHOLE_CLIP}"


@pytest.fixture
def doc() -> dict:
    return deepcopy(EXAMPLE_DOC)


def artifact(id: str, kind: str, sha256: str | None = HEX64) -> dict:
    return {
        "id": id,
        "kind": kind,
        "path": f"{id}.json",
        "sha256": sha256,
        "source_start": None,
        "source_end": None,
    }


def ref(type: str, id: str, sha256: str | None = HEX64) -> dict:
    return {"type": type, "ref": id, "sha256": None if type == "measurement" else sha256}


def clip_ref() -> dict:
    return ref("clip", WHOLE_CLIP, WHOLE_CLIP_SHA)


CATEGORY_FOR_BASIS = {
    "measurement": "tempo",
    "transcript": "filler_words",
    "audio_model_observation": "pronunciation",
    "human_verified": "pronunciation",
}


def finding(id: str, basis: str, refs: list[dict]) -> dict:
    """A neutral finding on 1–2 s with the given refs plus the example's whole-source clip."""
    return {
        "id": id,
        "category": CATEGORY_FOR_BASIS.get(basis, "content"),
        "observation": "synthetic observation",
        "start": 1.0,
        "end": 2.0,
        "evidence_basis": basis,
        "evidence_refs": [*refs, clip_ref()],
        "confidence_note": "",
        "impact": "",
        "correction": None,
        "polarity": "neutral",
    }


def review_artifacts(
    start: float | None = 0.0, end: float | None = 12.0, name: str = "review-response"
) -> list[dict]:
    """A review response and its provenance record in one review directory, one range."""
    response = artifact(name, "review_response")
    provenance = artifact(f"{name}-provenance", "review_provenance", "cd" * 32)
    response["path"] = f"{REVIEW_DIR}/response.raw.txt"
    provenance["path"] = f"{REVIEW_DIR}/provenance.json"
    for item in (response, provenance):
        item.update(source_start=start, source_end=end)
    return [response, provenance]


def complete_review(doc: dict) -> dict:
    """The example turned into a finished run: audio review done over the whole request."""
    doc["status"] = "complete"
    doc["stages"]["audio_review"] = {
        "requested": True,
        "status": "completed",
        "reason": None,
        "elapsed_ms": 1500,
    }
    if all(a["id"] != "review-response" for a in doc["artifacts"]):
        doc["artifacts"] += review_artifacts()
    doc["provenance"].update(
        backend="test-double",
        input_modalities=["audio", "text"],
        audio_payload_sha256=HEX64,
        prompt_sha256=HEX64,
    )
    doc["coverage"]["reviewed"] = deepcopy(doc["coverage"]["requested"])
    doc["coverage"]["unreviewed"] = []
    return doc


def assert_rejected(doc: dict, fragment: str) -> list[str]:
    """`validate` raises and the error list names the violated rule by its message."""
    with pytest.raises(ResultValidationError) as excinfo:
        validate(doc)
    errors = excinfo.value.errors
    assert any(fragment in error for error in errors), errors
    assert errors == validation_errors(doc)
    return errors


def test_schema_is_valid_draft_2020_12_and_packaged_copy_is_byte_equal() -> None:
    Draft202012Validator.check_schema(schema.SCHEMA)
    assert schema.SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    root_bytes = ROOT_SCHEMA.read_bytes()
    assert root_bytes == PACKAGED_SCHEMA.read_bytes()
    assert root_bytes == schema.SCHEMA_BYTES
    assert hashlib.sha256(root_bytes).hexdigest() == schema.SCHEMA_SHA256
    assert schema.SCHEMA["properties"]["schema_version"] == {"const": schema.SCHEMA_VERSION}


def test_schema_requires_every_top_level_key_of_spec_section_4() -> None:
    root = json.loads(ROOT_SCHEMA.read_text(encoding="utf-8"))
    for key in TOP_LEVEL_KEYS:
        assert key in root["properties"], key
        assert key in root["required"], key
    assert tuple(root["required"]) == TOP_LEVEL_KEYS
    assert set(root["properties"]) == set(TOP_LEVEL_KEYS)
    assert root["properties"]["status"]["enum"] == ["complete", "partial", "blocked", "failed"]
    assert tuple(root["properties"]["stages"]["required"]) == schema.STAGES


def test_synthetic_example_is_marked_synthetic_and_passes(doc: dict) -> None:
    validate(doc)
    assert doc["provenance"]["synthetic"] is True
    assert doc["provenance"]["note"].startswith("synthetic: ")
    assert "synthetic" in doc["provenance"]["note"].lower()
    assert doc["provenance"]["schema_sha256"] == schema.SCHEMA_SHA256
    assert doc["status"] == "partial"
    assert "path" not in doc["source"]


@pytest.mark.parametrize("key", TOP_LEVEL_KEYS)
def test_missing_top_level_key_is_rejected(doc: dict, key: str) -> None:
    del doc[key]
    assert_rejected(doc, f"'{key}' is a required property")


def test_unknown_keys_are_rejected(doc: dict) -> None:
    doc["extra"] = 1
    assert_rejected(doc, "$: Additional properties")
    doc = deepcopy(EXAMPLE_DOC)
    doc["source"]["bitrate"] = 1
    assert_rejected(doc, "$.source: Additional properties")


def test_source_path_is_optional(doc: dict) -> None:
    doc["source"]["path"] = "recording.wav"
    validate(doc)


@pytest.mark.parametrize(
    ("start", "end", "fragment"),
    [
        (5.0, 5.0, "start must be less than end"),
        (6.0, 5.0, "start must be less than end"),
        (1.0, 12.5, "exceeds source duration"),
        (-0.5, 1.0, "$.findings[0].start"),
    ],
)
def test_finding_timecodes_are_ordered_and_bounded(
    doc: dict, start: float, end: float, fragment: str
) -> None:
    doc["findings"][0].update(start=start, end=end)
    assert_rejected(doc, fragment)


def test_coverage_timecodes_are_ordered_and_bounded(doc: dict) -> None:
    doc["coverage"]["requested"] = [{"start": 0.0, "end": 13.0}]
    doc["coverage"]["unreviewed"] = [{"start": 0.0, "end": 13.0}]
    assert_rejected(doc, "$.coverage.requested[0].end: exceeds source duration")
    doc = deepcopy(EXAMPLE_DOC)
    doc["coverage"]["requested"] = [{"start": 12.0, "end": 0.0}]
    assert_rejected(doc, "$.coverage.requested[0]: start must be less than end")


def test_correction_timecodes_are_ordered_and_bounded(doc: dict) -> None:
    doc["transcript"]["corrections"][0].update(start=4.0, end=3.0)
    assert_rejected(doc, "$.transcript.corrections[0]: start must be less than end")
    doc["transcript"]["corrections"][0].update(start=11.0, end=12.1)
    assert_rejected(doc, "$.transcript.corrections[0].end: exceeds source duration")


def test_artifact_source_timecodes_are_both_set_ordered_and_bounded(doc: dict) -> None:
    clip = artifact("clip-1", "clip")
    clip.update(source_start=2.0, source_end=4.0)
    doc["artifacts"].append(clip)
    validate(doc)
    clip.update(source_start=4.0, source_end=2.0)
    assert_rejected(doc, "$.artifacts[6]: start must be less than end")
    clip.update(source_start=-1.0, source_end=2.0)
    assert_rejected(doc, "$.artifacts[6].start: must be >= 0")
    clip.update(source_start=2.0, source_end=99.0)
    assert_rejected(doc, "$.artifacts[6].end: exceeds source duration")
    clip.update(source_start=2.0, source_end=None)
    assert_rejected(doc, "$.artifacts[6]: source_start and source_end must both be set")


def test_suspect_segments_carry_an_interval_and_signals_without_text(doc: dict) -> None:
    assert doc["transcript"]["suspect_segments"] == [
        {"start": 11.4, "end": 12.0, "signals": ["silent_span"], "certainty": "uncertain"}
    ]
    item = doc["transcript"]["suspect_segments"][0]
    item.update(start=12.0, end=12.0)
    validate(doc)
    cases = [
        (lambda i: i.pop("signals"), "[0]: 'signals' is a required property"),
        (lambda i: i.update(signals=["echo"]), "[0].signals[0]: must be one of"),
        (lambda i: i.update(signals=[]), "[0].signals: must have at least 1 items"),
        (lambda i: i.update(signals=["silent_span"] * 2), "[0].signals: has non-unique items"),
        (lambda i: i.update(certainty="clear"), "[0].certainty: 'uncertain' was expected"),
        (lambda i: i.update(text=RESULT_TEXT), "[0]: Additional properties are not allowed"),
        (lambda i: i.update(start=-0.1), "[0].start: must be at least 0"),
        (lambda i: i.update(start=5.0, end=4.0), "[0]: start must not exceed end"),
        (lambda i: i.update(end=12.5), "[0].end: exceeds source duration 12.0"),
    ]
    for change, fragment in cases:
        broken = deepcopy(EXAMPLE_DOC)
        change(broken["transcript"]["suspect_segments"][0])
        errors = assert_rejected(broken, "$.transcript.suspect_segments" + fragment)
        assert not any(RESULT_TEXT in e for e in errors), errors


def test_suspect_segments_are_required_and_empty_without_a_raw_transcript(doc: dict) -> None:
    del doc["transcript"]["suspect_segments"]
    assert_rejected(doc, "$.transcript: 'suspect_segments' is a required property")
    doc = deepcopy(EXAMPLE_DOC)
    doc["transcript"]["raw"] = None
    doc["findings"][1]["evidence_refs"][0] = ref("measurement", "speech_rate")
    doc["findings"][1].update(evidence_basis="measurement", category="tempo")
    assert_rejected(doc, "$.transcript.suspect_segments: must be empty when transcript.raw is null")
    doc["transcript"]["suspect_segments"] = []
    validate(doc)


def test_duplicate_finding_and_artifact_ids_are_rejected(doc: dict) -> None:
    doc["findings"].append(deepcopy(doc["findings"][0]))
    assert_rejected(doc, "$.findings[2].id: duplicate id 'f-001'")
    doc = deepcopy(EXAMPLE_DOC)
    doc["artifacts"].append(deepcopy(doc["artifacts"][0]))
    assert_rejected(doc, "$.artifacts[6].id: duplicate id 'transcript-raw'")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_infinity_are_rejected_but_null_passes(doc: dict, value: float) -> None:
    doc["measurements"][0]["value"] = value
    assert_rejected(doc, "$.measurements[0].value: NaN/Infinity is not allowed, use null")
    doc = deepcopy(EXAMPLE_DOC)
    doc["measurements"][1]["settings"]["threshold"] = [1.0, {"depth": value}]
    assert_rejected(doc, "$.measurements[1].settings: NaN/Infinity under an undeclared key")
    doc = deepcopy(EXAMPLE_DOC)
    doc["measurements"][0].update(value=None, reason="synthetic: not obtained")
    validate(doc)


def test_measurement_null_value_needs_a_reason_and_other_value_shapes_pass(doc: dict) -> None:
    doc["measurements"][0].update(value=None, reason=None)
    assert_rejected(doc, "$.measurements[0].reason: a reason is required when value is null")
    doc["measurements"][0]["reason"] = "  "
    assert_rejected(doc, "$.measurements[0].reason: a reason is required when value is null")
    doc = deepcopy(EXAMPLE_DOC)
    doc["measurements"][0]["value"] = {"p10": 100.0, "p50": 120.5, "p90": 140.0}
    doc["measurements"][1].update(value=[{"start": 0.5, "end": 3.0}], reason=None)
    validate(doc)
    doc["measurements"][0]["value"] = {}
    assert_rejected(
        doc, "$.measurements[0].value: must be a valid nullable number or array or object"
    )
    doc = deepcopy(EXAMPLE_DOC)
    doc["measurements"][1].update(value=[{"start": 3.0, "end": 3.0}], reason=None)
    assert_rejected(doc, "$.measurements[1].value[0]: start must be less than end, got 3.0 >= 3.0")
    doc["measurements"][1]["value"] = [{"start": 1.0, "end": 12.5}]
    assert_rejected(doc, "$.measurements[1].value[0].end: exceeds source duration 12.0, got 12.5")
    doc = deepcopy(EXAMPLE_DOC)
    del doc["measurements"][0]["limitations"]
    assert_rejected(doc, "$.measurements[0]: 'limitations' is a required property")


def test_evidence_basis_outside_the_enumeration_is_rejected(doc: dict) -> None:
    doc["findings"][0]["evidence_basis"] = "model_guess"
    assert_rejected(doc, "$.findings[0].evidence_basis: must be one of ['measurement',")
    root = json.loads(ROOT_SCHEMA.read_text(encoding="utf-8"))
    assert root["$defs"]["finding"]["properties"]["evidence_basis"]["enum"] == [
        "measurement",
        "transcript",
        "audio_model_observation",
        "human_verified",
    ]
    assert tuple(root["$defs"]["finding"]["properties"]["evidence_basis"]["enum"]) == (
        schema.EVIDENCE_BASES
    )


def test_stage_statuses_need_a_reason_unless_completed(doc: dict) -> None:
    doc["stages"]["asr"]["status"] = "done"
    assert_rejected(doc, "$.stages.asr.status: must be one of ['completed',")
    doc = deepcopy(EXAMPLE_DOC)
    doc["stages"]["asr"].update(status="skipped", reason=None)
    assert_rejected(doc, "$.stages.asr.reason: a reason is required when status is 'skipped'")
    doc["stages"]["asr"]["reason"] = "   "
    assert_rejected(doc, "$.stages.asr.reason")
    doc["stages"]["asr"].update(status="failed", reason="decoder crashed")
    doc["status"] = "failed"
    doc["errors"] = [{"stage": "asr", "code": "asr_failed", "message": "decoder crashed"}]
    validate(doc)
    doc = deepcopy(EXAMPLE_DOC)
    doc["stages"]["asr"]["reason"] = None
    validate(doc)


def test_stage_status_enumeration_is_exactly_the_four_spec_values() -> None:
    root = json.loads(ROOT_SCHEMA.read_text(encoding="utf-8"))
    assert root["$defs"]["stage"]["properties"]["status"]["enum"] == [
        "completed",
        "skipped",
        "unavailable",
        "failed",
    ]


@pytest.mark.parametrize("status", ["skipped", "unavailable", "failed"])
def test_every_non_completed_stage_status_requires_a_reason(doc: dict, status: str) -> None:
    doc["stages"]["acoustics"].update(status=status, reason=None)
    assert_rejected(
        doc, f"$.stages.acoustics.reason: a reason is required when status is {status!r}"
    )
    doc["stages"]["acoustics"]["reason"] = ""
    assert_rejected(doc, "$.stages.acoustics.reason: a reason is required")
    doc["stages"]["acoustics"]["reason"] = "synthetic reason"
    validate(doc)


def test_human_verified_requires_a_human_label_artifact_with_a_hash(doc: dict) -> None:
    raw_sha = doc["artifacts"][0]["sha256"]
    doc["findings"].append(
        finding("f-003", "human_verified", [ref("transcript", "transcript-raw", raw_sha)])
    )
    assert_rejected(
        doc, "$.findings[2].evidence_refs: evidence_basis 'human_verified' needs at least one "
    )
    doc["artifacts"].append(artifact("label-1", "human_label"))
    doc["findings"][2]["evidence_refs"] = [ref("human_label", "label-1"), clip_ref()]
    validate(doc)
    doc["artifacts"][-1]["kind"] = "other"
    assert_rejected(doc, "artifact 'label-1' must be of kind 'human_label'")
    doc["artifacts"][-1].update(kind="human_label", sha256=None)
    assert_rejected(doc, "evidence artifact 'label-1' must carry a sha256")
    doc["findings"][2]["evidence_refs"] = [ref("human_label", "missing"), clip_ref()]
    assert_rejected(doc, "$.findings[2].evidence_refs[0].ref: unknown artifact 'missing'")


def test_evidence_refs_point_at_known_measurements_and_artifacts(doc: dict) -> None:
    doc["findings"][0]["evidence_refs"][0] = ref("measurement", "unknown_metric")
    assert_rejected(doc, "$.findings[0].evidence_refs[0].ref: unknown measurement 'unknown_metric'")
    doc = deepcopy(EXAMPLE_DOC)
    doc["findings"][1]["evidence_refs"][0] = ref("transcript", "report-md")
    assert_rejected(doc, "artifact 'report-md' must be of kind 'transcript_raw'")
    doc = deepcopy(EXAMPLE_DOC)
    doc["findings"][1]["evidence_refs"].append(ref("artifact", "report-md"))
    assert_rejected(doc, "evidence artifact 'report-md' must carry a sha256")
    doc["artifacts"][2]["sha256"] = HEX64
    validate(doc)
    doc["findings"][1]["evidence_refs"] = []
    assert_rejected(doc, "$.findings[1].evidence_refs: must have at least 1 items")


def test_evidence_artifacts_must_carry_a_hash(doc: dict) -> None:
    doc["artifacts"][0]["sha256"] = None
    errors = assert_rejected(
        doc, "$.findings[1].evidence_refs[0].ref: evidence artifact 'transcript-raw' must carry"
    )
    assert "$.transcript.raw.artifact_ref: 'transcript-raw' must carry a sha256" in errors
    assert (
        "$.transcript.corrections[0].evidence_refs[0].ref: evidence artifact 'transcript-raw' "
        "must carry a sha256"
    ) in errors
    doc = deepcopy(EXAMPLE_DOC)
    doc = complete_review(doc)
    doc["findings"].append(
        finding(
            "f-003", "audio_model_observation", [ref("audio_model_observation", "review-response")]
        )
    )
    validate(doc)
    doc["artifacts"][-2]["sha256"] = None
    assert_rejected(doc, "evidence artifact 'review-response' must carry a sha256")
    doc["findings"][2]["evidence_refs"].append(ref("artifact", "transcript-raw"))
    doc["artifacts"][-2]["sha256"] = HEX64
    assert_rejected(doc, "$.findings[2].evidence_refs[2].sha256: does not match artifact")
    doc["findings"][2]["evidence_refs"][2]["sha256"] = doc["artifacts"][0]["sha256"]
    validate(doc)


def test_transcript_raw_must_reference_a_transcript_raw_artifact(doc: dict) -> None:
    doc["transcript"]["raw"]["artifact_ref"] = "report-md"
    assert_rejected(doc, "$.transcript.raw.artifact_ref: 'report-md' must be an artifact")
    doc["transcript"]["raw"] = None
    doc["transcript"]["suspect_segments"] = []
    doc["findings"][1]["evidence_refs"][0] = ref("measurement", "speech_rate")
    doc["findings"][1].update(evidence_basis="measurement", category="tempo")
    validate(doc)


def test_audio_model_observation_requires_a_completed_audio_review(doc: dict) -> None:
    doc["findings"].append(
        finding(
            "f-003", "audio_model_observation", [ref("audio_model_observation", "review-response")]
        )
    )
    doc["artifacts"] += review_artifacts()
    assert_rejected(doc, "$.findings[2]: audio_model_observation evidence requires a completed")
    doc = complete_review(doc)
    doc["assessment_status"]["diction"] = {"status": "assessed", "reason": None}
    validate(doc)
    doc["provenance"]["backend"] = "none"
    assert_rejected(doc, "$.provenance.backend: audio_review completed without a backend")
    doc["provenance"].update(backend="test-double", audio_payload_sha256=None)
    assert_rejected(doc, "$.provenance.audio_payload_sha256: required when audio_review")
    doc["provenance"].update(audio_payload_sha256=HEX64, prompt_sha256=None)
    assert_rejected(doc, "$.provenance.prompt_sha256: required when audio_review")
    doc["provenance"].update(prompt_sha256=HEX64, input_modalities=["text"])
    assert_rejected(doc, "$.provenance.input_modalities: audio_review completed without 'audio'")


def test_correction_source_needs_matching_evidence(doc: dict) -> None:
    correction = doc["transcript"]["corrections"][0]
    del correction["evidence_refs"]
    assert_rejected(doc, "'evidence_refs' is a required property")
    correction["evidence_refs"] = []
    assert_rejected(doc, "$.transcript.corrections[0].evidence_refs: must have at least 1 items")
    correction["evidence_refs"] = [
        ref("transcript", "transcript-raw", doc["artifacts"][0]["sha256"])
    ]
    correction["source"] = "human_verified"
    assert_rejected(
        doc,
        "$.transcript.corrections[0].evidence_refs: source 'human_verified' needs at least one "
        "reference of type 'human_label'",
    )
    doc["artifacts"].append(artifact("label-1", "human_label"))
    correction["evidence_refs"] = [ref("human_label", "label-1")]
    validate(doc)
    doc["artifacts"][-1]["sha256"] = None
    assert_rejected(
        doc,
        "$.transcript.corrections[0].evidence_refs[0].ref: evidence artifact 'label-1' must "
        "carry a sha256",
    )
    correction["evidence_refs"] = [ref("human_label", "missing")]
    assert_rejected(doc, "$.transcript.corrections[0].evidence_refs[0].ref: unknown artifact")


def test_correction_from_audio_model_requires_a_completed_audio_review(doc: dict) -> None:
    correction = doc["transcript"]["corrections"][0]
    raw_ref = deepcopy(correction["evidence_refs"][0])
    doc["artifacts"] += review_artifacts()
    correction["source"] = "audio_model_observation"
    assert_rejected(
        doc,
        "$.transcript.corrections[0].evidence_refs: source 'audio_model_observation' needs at "
        "least one reference of type 'audio_model_observation'",
    )
    correction["evidence_refs"] = [ref("audio_model_observation", "review-response")]
    errors = assert_rejected(
        doc,
        "$.transcript.corrections[0]: audio_model_observation evidence requires a completed "
        "audio_review stage",
    )
    assert errors == [
        "$.transcript.corrections[0]: audio_model_observation evidence requires a completed "
        "audio_review stage",
        "$.transcript.corrections[0]: outside coverage.reviewed",
    ]
    doc = complete_review(doc)
    validate(doc)
    correction["source"] = "transcript"
    correction["evidence_refs"] = [raw_ref, ref("audio_model_observation", "review-response")]
    validate(doc)
    doc["stages"]["audio_review"].update(status="failed", reason="backend timed out")
    doc["status"] = "partial"
    doc["coverage"]["reviewed"] = []
    doc["coverage"]["unreviewed"] = deepcopy(doc["coverage"]["requested"])
    assert_rejected(doc, "$.transcript.corrections[0]: audio_model_observation evidence requires")


def test_text_only_evidence_cannot_carry_an_audio_model_observation_ref(doc: dict) -> None:
    doc["findings"][1]["evidence_refs"].append(ref("audio_model_observation", "review-response"))
    doc["artifacts"] += review_artifacts()
    assert_rejected(doc, "$.findings[1]: audio_model_observation evidence requires a completed")


def test_complete_status_requires_every_requested_stage_completed(doc: dict) -> None:
    doc = complete_review(doc)
    validate(doc)
    doc["stages"]["asr"].update(status="skipped", reason="asr disabled by the request")
    assert_rejected(doc, "$.stages.asr.status: status 'complete' requires every requested stage")
    doc["stages"]["asr"]["requested"] = False
    validate(doc)


def test_complete_status_requires_reviewed_to_cover_requested(doc: dict) -> None:
    doc = complete_review(doc)
    doc["coverage"]["reviewed"] = [{"start": 0.0, "end": 8.0}]
    doc["coverage"]["unreviewed"] = [{"start": 8.0, "end": 12.0}]
    assert_rejected(doc, "$.coverage.reviewed: status 'complete' requires reviewed to cover")
    doc["coverage"]["reviewed"] = [{"start": 0.0, "end": 5.0}, {"start": 5.0, "end": 12.0}]
    doc["coverage"]["unreviewed"] = []
    validate(doc)
    doc["coverage"]["reviewed"] = [{"start": 0.0, "end": 5.0}, {"start": 5.0005, "end": 12.0}]
    validate(doc)


def test_unreviewed_must_equal_requested_minus_reviewed(doc: dict) -> None:
    doc["coverage"]["unreviewed"] = []
    assert_rejected(doc, "$.coverage.unreviewed: must equal requested minus reviewed")
    doc = complete_review(doc)
    doc["status"] = "partial"
    doc["coverage"]["reviewed"] = [{"start": 2.0, "end": 4.0}]
    doc["coverage"]["unreviewed"] = [{"start": 0.0, "end": 2.0}, {"start": 4.0, "end": 12.0}]
    validate(doc)
    doc["coverage"]["unreviewed"] = [{"start": 4.0, "end": 12.0}, {"start": 0.0, "end": 2.0}]
    assert_rejected(doc, "$.coverage.unreviewed[1]: ranges must be sorted and not overlap")


def test_reviewed_ranges_need_a_completed_audio_review(doc: dict) -> None:
    doc["coverage"]["reviewed"] = [{"start": 0.0, "end": 12.0}]
    doc["coverage"]["unreviewed"] = []
    assert_rejected(doc, "$.coverage.reviewed: must be empty unless audio_review is completed")
    doc = complete_review(doc)
    validate(doc)
    doc["coverage"] = {"requested": [], "reviewed": [], "unreviewed": []}
    assert_rejected(doc, "$.coverage.requested: audio_review completed without any range")
    assert_rejected(doc, "$.coverage.reviewed: audio_review completed without any range")
    doc["coverage"]["requested"] = [{"start": 0.0, "end": 12.0}]
    doc["coverage"]["unreviewed"] = [{"start": 0.0, "end": 12.0}]
    doc["status"] = "partial"
    errors = assert_rejected(doc, "$.coverage.reviewed: audio_review completed without any range")
    assert len(errors) == 1


def test_no_sound_layer_means_empty_coverage(doc: dict) -> None:
    doc["coverage"] = {"requested": [], "reviewed": [], "unreviewed": []}
    doc["stages"]["audio_review"].update(requested=False, status="skipped", reason="not requested")
    validate(doc)


def test_failed_or_blocked_status_requires_errors(doc: dict) -> None:
    doc["status"] = "failed"
    assert_rejected(doc, "$.errors: status 'failed' requires at least one error")
    doc["status"] = "blocked"
    assert_rejected(doc, "$.errors: status 'blocked' requires at least one error")
    doc["errors"] = [{"stage": None, "code": "permission_required", "message": "no approval"}]
    validate(doc)
    doc["errors"] = [{"stage": "mixing", "code": "x", "message": "m"}]
    assert_rejected(doc, "$.errors[0].stage")


def test_validated_ru_requires_both_layer_flags_and_a_backend(doc: dict) -> None:
    doc["capabilities"].update(validated_ru=True, diction_validated_ru=True)
    assert_rejected(doc, "requires diction_validated_ru and prosody_validated_ru")
    doc["capabilities"]["prosody_validated_ru"] = True
    assert_rejected(doc, "$.capabilities.validated_ru: requires a selected audio_backend")
    doc["capabilities"]["audio_backend"] = {"name": "test-double", "model": None, "revision": None}
    validate(doc)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize(
    ("place", "path"),
    [
        ((), "$"),
        (("stages",), "$.stages"),
        (("measurements", 1, "settings"), "$.measurements[1].settings"),
    ],
)
def test_non_finite_number_under_an_undeclared_key_names_its_parent(
    doc: dict, value: float, place: tuple, path: str
) -> None:
    target = doc
    for step in place:
        target = target[step]
    target[RESULT_TEXT] = {"nested": [value]}
    errors = validation_errors(doc)
    assert f"{path}: NaN/Infinity under an undeclared key is not allowed" in errors, errors
    assert not any(RESULT_TEXT in error for error in errors), errors


@pytest.mark.parametrize("kind", ["actual", "estimated"])
def test_zero_cost_error_names_allowed_providers_not_the_given_one(doc: dict, kind: str) -> None:
    doc["cost"].update(kind=kind, amount=0, provider=RESULT_TEXT)
    with pytest.raises(ResultValidationError) as excinfo:
        validate(doc)
    expected = f"$.cost.amount: zero {kind} cost is only accepted for providers ['local', 'none']"
    assert expected in excinfo.value.errors
    assert RESULT_TEXT not in str(excinfo.value)


def test_cost_rules(doc: dict) -> None:
    doc["cost"].update(kind="actual", amount=None)
    assert_rejected(doc, "$.cost.amount: required when kind is 'actual'")
    doc["cost"].update(kind="estimated", amount=None)
    assert_rejected(doc, "$.cost.amount: required when kind is 'estimated'")
    doc["cost"].update(kind="actual", amount=0, provider="gemini")
    assert_rejected(doc, "$.cost.amount: zero actual cost is only accepted for providers")
    doc["cost"].update(kind="estimated", amount=0, provider="gemini")
    assert_rejected(doc, "$.cost.amount: zero estimated cost is only accepted for providers")
    doc["cost"].update(kind="estimated", amount=0, provider="none")
    validate(doc)
    doc["cost"].update(kind="unknown", amount=0.5, provider="gemini")
    assert_rejected(doc, "$.cost.amount: must be null when kind is 'unknown'")
    doc["cost"].update(kind="estimated", amount=0.5, currency=None)
    assert_rejected(doc, "$.cost.currency: required when amount is greater than 0")
    doc["cost"].update(currency="USD")
    validate(doc)
    doc["cost"].update(currency="usd")
    assert_rejected(doc, "$.cost.currency")
    doc["cost"].update(kind="unknown", amount=None, currency=None)
    validate(doc)


def test_limits_duration_mismatch_fields_are_in_the_schema() -> None:
    limits = schema.SCHEMA["properties"]["limits"]
    assert limits["additionalProperties"] is False
    for key in ("duration_mismatch_tolerance_s", "duration_mismatch_s", "duration_mismatch"):
        assert key in limits["required"], key
    assert limits["properties"]["duration_mismatch_tolerance_s"] == {
        "type": "number",
        "exclusiveMinimum": 0,
    }


def test_limits_duration_mismatch_passes_when_flag_matches_the_measured_gap(doc: dict) -> None:
    doc["limits"].update(duration_mismatch_s=-0.5, duration_mismatch=True)
    validate(doc)
    doc["limits"].update(duration_mismatch_s=0.5, duration_mismatch=True)
    validate(doc)
    doc["limits"].update(duration_mismatch_s=0.04, duration_mismatch=False)
    validate(doc)
    doc["limits"].update(duration_mismatch_s=0.05, duration_mismatch=False)
    validate(doc)
    doc["limits"].update(duration_mismatch_s=None, duration_mismatch=None)
    validate(doc)
    doc["limits"].update(duration_mismatch_tolerance_s=1.0, duration_mismatch_s=0.5)
    doc["limits"].update(duration_mismatch=False)
    validate(doc)


def test_limits_duration_mismatch_flag_must_agree_with_the_measured_gap(doc: dict) -> None:
    doc["limits"].update(duration_mismatch_s=-0.5, duration_mismatch=False)
    assert_rejected(
        doc,
        "$.limits.duration_mismatch: must be true, |duration_mismatch_s| = 0.5 exceeds "
        "duration_mismatch_tolerance_s 0.05",
    )
    doc["limits"].update(duration_mismatch_s=0.01, duration_mismatch=True)
    assert_rejected(
        doc,
        "$.limits.duration_mismatch: must be false, |duration_mismatch_s| = 0.01 is within "
        "duration_mismatch_tolerance_s 0.05",
    )
    doc["limits"].update(duration_mismatch_s=0.5, duration_mismatch=None)
    assert_rejected(
        doc, "$.limits.duration_mismatch: must be a boolean when duration_mismatch_s is a number"
    )
    doc["limits"].update(duration_mismatch_s=None, duration_mismatch=False)
    assert_rejected(
        doc, "$.limits.duration_mismatch: must be null when duration_mismatch_s is null"
    )


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        ("duration_mismatch_tolerance_s", 0, "must be greater than 0"),
        ("duration_mismatch_tolerance_s", None, "must be of type number"),
        ("duration_mismatch_s", "0.5", "must be of type ['number', 'null']"),
        ("duration_mismatch", 1, "must be of type ['boolean', 'null']"),
    ],
)
def test_limits_duration_mismatch_shape_is_enforced_by_the_schema(
    doc: dict, key: str, value: object, fragment: str
) -> None:
    doc["limits"][key] = value
    assert_rejected(doc, f"$.limits.{key}: {fragment}")
    del doc["limits"][key]
    assert_rejected(doc, f"$.limits: '{key}' is a required property")


def test_assessment_status_rules(doc: dict) -> None:
    doc["assessment_status"]["diction"]["reason"] = None
    assert_rejected(doc, "$.assessment_status.diction.reason: a reason is required")
    doc = deepcopy(EXAMPLE_DOC)
    doc["assessment_status"]["prosody"] = {"status": "assessed", "reason": None}
    assert_rejected(doc, "$.assessment_status.prosody.status: must be not_assessed without")
    doc = complete_review(deepcopy(EXAMPLE_DOC))
    doc["assessment_status"]["prosody"] = {"status": "assessed", "reason": None}
    validate(doc)


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("/tmp/report.md", "must be relative"),
        ("C:\\runs\\report.md", "must be relative"),
        ("\\\\server\\share\\report.md", "must be relative"),
        ("../report.md", "must not contain '..'"),
        ("clips/../../report.md", "must not contain '..'"),
        ("clips\\..\\report.md", "must not contain '..'"),
        ("clips/report\0.md", "must not contain a NUL character"),
    ],
)
def test_artifact_path_must_stay_inside_the_run_directory(
    doc: dict, path: str, fragment: str
) -> None:
    doc["artifacts"][2]["path"] = path
    assert_rejected(doc, f"$.artifacts[2].path: {fragment}")


def test_clip_artifacts_need_source_timecodes(doc: dict) -> None:
    doc["artifacts"].append(artifact("clip-1", "clip"))
    assert_rejected(doc, "$.artifacts[6]: kind 'clip' requires source_start and source_end")
    doc["artifacts"][6].update(kind="listening_copy", source_start=0.0, source_end=12.0)
    validate(doc)


def test_validate_raises_with_the_same_errors_as_validation_errors(doc: dict) -> None:
    doc["findings"][0]["start"] = 11.0
    doc["findings"][0]["end"] = 10.0
    doc["findings"].append(deepcopy(doc["findings"][1]))
    with pytest.raises(ResultValidationError) as excinfo:
        validate(doc)
    assert excinfo.value.errors == validation_errors(doc)
    assert len(excinfo.value.errors) == 2
    assert str(excinfo.value).startswith("result does not match the v1 contract:\n")
    for error in excinfo.value.errors:
        assert error in str(excinfo.value)


def test_shape_errors_come_alone_without_cross_rule_crashes(doc: dict) -> None:
    doc["stages"] = []
    doc["source"]["duration_s"] = "long"
    errors = validation_errors(doc)
    assert errors == [
        "$.source.duration_s: must be of type number",
        "$.stages: must be of type object",
    ]
    assert validation_errors(None) == ["$: must be of type object"]
    assert validation_errors([]) == ["$: must be of type object"]


RESULT_TEXT = "SECRET-фраза-из-записи"


def set_path(doc: dict, path: tuple, value: object) -> None:
    target = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("findings", 0, "start"), RESULT_TEXT, "$.findings[0].start: must be of type number"),
        (("findings", 0, "observation"), [RESULT_TEXT], "$.findings[0].observation: must be of"),
        (("findings", 0, "evidence_basis"), RESULT_TEXT, "$.findings[0].evidence_basis: must be"),
        (("transcript", "corrections", 0, "certainty"), RESULT_TEXT, "certainty: must be one of"),
        (("run_id",), RESULT_TEXT * 20, "$.run_id: must be at most 256 characters long"),
        (("findings",), RESULT_TEXT, "$.findings: must be of type array"),
        (("findings",), {"observation": RESULT_TEXT}, "$.findings: must be of type array"),
        (("stages",), RESULT_TEXT, "$.stages: must be of type object"),
        (("stages",), {RESULT_TEXT: RESULT_TEXT}, "$.stages: Additional properties are not"),
        (("transcript", "raw"), RESULT_TEXT, "$.transcript.raw: must be null or object"),
        (("schema_version",), RESULT_TEXT, "$.schema_version: '1.2' was expected"),
        (("cost", "currency"), RESULT_TEXT, "$.cost.currency: must be null or string"),
    ],
)
def test_schema_errors_name_the_rule_without_echoing_document_values(
    doc: dict, path: tuple, value: object, error: str
) -> None:
    set_path(doc, path, value)
    errors = assert_rejected(doc, error)
    assert not any(RESULT_TEXT in e for e in errors), errors
    with pytest.raises(ResultValidationError) as excinfo:
        validate(doc)
    assert RESULT_TEXT not in str(excinfo.value)


def test_non_object_document_is_named_without_its_value() -> None:
    errors = validation_errors([RESULT_TEXT])
    assert errors == ["$: must be of type object"]


RESULT_SCHEMA_RULES = {
    "type": (("source", "duration_s"), "long", "$.source.duration_s: must be of type number"),
    "required": (("source",), {}, "$.source: 'id' is a required property"),
    "additionalProperties": (
        ("source", "extra"),
        1,
        "$.source: Additional properties are not allowed, allowed: [",
    ),
    "const": (("schema_version",), "2.0", "$.schema_version: '1.2' was expected"),
    "enum": (("stages", "asr", "status"), "done", "$.stages.asr.status: must be one of ["),
    "minLength": (("source", "id"), "", "$.source.id: must be at least 1 characters long"),
    "maxLength": (("run_id",), "r" * 257, "$.run_id: must be at most 256 characters long"),
    "minimum": (("source", "channels"), 0, "$.source.channels: must be at least 1"),
    "exclusiveMinimum": (
        ("limits", "duration_mismatch_tolerance_s"),
        0,
        "$.limits.duration_mismatch_tolerance_s: must be greater than 0",
    ),
    "pattern": (
        ("errors",),
        [{"stage": None, "code": "Bad-Code", "message": RESULT_TEXT}],
        "$.errors[0].code: must match pattern '^[a-z][a-z0-9_]*$'",
    ),
    "uniqueItems": (
        ("provenance", "input_modalities"),
        ["text", "text"],
        "$.provenance.input_modalities: has non-unique items, every item must be unique",
    ),
    "minItems": (
        ("findings", 0, "evidence_refs"),
        [],
        "$.findings[0].evidence_refs: must have at least 1 items",
    ),
    "maximum": (
        ("measurements", 2, "validation", "auc"),
        2,
        "$.measurements[2].validation.auc: must be at most 1",
    ),
    "maxItems": (
        ("measurements", 2, "validation", "ci95"),
        [0.1, 0.2, 0.3],
        "$.measurements[2].validation.ci95: must have at most 2 items",
    ),
    "oneOf": (
        ("measurements", 0, "value"),
        {},
        "$.measurements[0].value: must be a valid nullable number or array or object",
    ),
}
# Ключевые слова, которые сами не нарушаются: ссылки, вложенные схемы и аннотации.
STRUCTURAL_KEYWORDS = {"$schema", "$id", "$ref", "$defs", "title", "description"}
STRUCTURAL_KEYWORDS |= {"properties", "items"}


def schema_keywords(node: object) -> set[str]:
    if isinstance(node, list):
        return set().union(*(schema_keywords(item) for item in node))
    if not isinstance(node, dict):
        return set()
    found = set(node)
    for key, value in node.items():
        if key in ("properties", "$defs"):
            found |= set().union(*(schema_keywords(sub) for sub in value.values()))
        elif isinstance(value, dict | list):
            found |= schema_keywords(value)
    return found


def test_every_rule_of_the_result_schema_has_its_own_message() -> None:
    rules = schema_keywords(schema.SCHEMA) - STRUCTURAL_KEYWORDS
    assert rules == set(RESULT_SCHEMA_RULES) | {"minProperties"}


@pytest.mark.parametrize("rule", sorted(RESULT_SCHEMA_RULES))
def test_each_result_schema_rule_is_named_with_its_path(doc: dict, rule: str) -> None:
    path, value, error = RESULT_SCHEMA_RULES[rule]
    set_path(doc, path, value)
    errors = assert_rejected(doc, error)
    assert not any("violates schema rule" in e for e in errors), errors
    assert not any(RESULT_TEXT in e for e in errors), errors


def test_min_properties_reached_only_inside_one_of_has_its_own_message() -> None:
    # В схеме результата minProperties стоит внутри oneOf и наружу выходит как ошибка oneOf.
    schema_part = schema.SCHEMA["$defs"]["measurement"]["properties"]["value"]["oneOf"][2]
    assert schema_errors(schema_part, {}) == ["$: must have at least 1 properties"]


# --- word_prominence (decision 0038) -------------------------------------------------------


def word_entry(doc: dict) -> dict:
    return next(m for m in doc["measurements"] if m["name"] == "word_prominence")


def test_the_example_holds_a_word_prominence_entry_with_its_validation(doc: dict) -> None:
    entry = word_entry(doc)
    assert entry["validation"]["decision"] == "docs/decisions/0037-word-prominence-rupro.md"
    assert entry["validation"]["report"] == "docs/eval/2026-11-28-prosody-rupro-1.md"
    assert all(v["transcript_ref"]["artifact_ref"] == "transcript-raw" for v in entry["value"])
    validate(doc)


def test_a_measurement_without_validation_still_passes(doc: dict) -> None:
    for measurement in doc["measurements"]:
        measurement.pop("validation", None)
    validate(doc)


def test_a_word_value_needs_a_reason_only_when_it_is_null(doc: dict) -> None:
    word_entry(doc)["value"][2]["reason"] = None
    index = doc["measurements"].index(word_entry(doc))
    assert_rejected(doc, f"$.measurements[{index}].value[2].reason: a reason is required")
    doc = deepcopy(EXAMPLE_DOC)
    word_entry(doc)["value"][0]["reason"] = "no_frames"
    assert_rejected(doc, f"$.measurements[{index}].value[0].reason: must be null when the word")


def test_a_word_value_must_point_into_a_raw_transcript_artifact(doc: dict) -> None:
    index = doc["measurements"].index(word_entry(doc))
    word_entry(doc)["value"][0]["transcript_ref"]["artifact_ref"] = "report-md"
    assert_rejected(doc, f"$.measurements[{index}].value[0].transcript_ref.artifact_ref")
    doc = deepcopy(EXAMPLE_DOC)
    word_entry(doc)["value"][0]["transcript_ref"]["pointer"] = "$.text"
    assert_rejected(doc, f"$.measurements[{index}].value: must be a valid nullable number")


def test_an_empty_word_span_passes_only_without_a_value(doc: dict) -> None:
    index = doc["measurements"].index(word_entry(doc))
    validate(doc)  # value[2] is a word of zero length without a value
    word_entry(doc)["value"][2].update(value=0.5, reason=None)
    assert_rejected(doc, f"$.measurements[{index}].value[2]: start must be less than end")
    doc = deepcopy(EXAMPLE_DOC)
    word_entry(doc)["value"][1]["end"] = doc["source"]["duration_s"] + 1
    assert_rejected(doc, f"$.measurements[{index}].value[1].end: exceeds source duration")


def test_a_word_value_that_is_not_finite_is_rejected(doc: dict) -> None:
    index = doc["measurements"].index(word_entry(doc))
    word_entry(doc)["value"][0]["value"] = float("nan")
    assert_rejected(doc, f"$.measurements[{index}].value[0].value: NaN/Infinity is not allowed")


def test_word_prominence_holds_values_per_word_not_time_ranges(doc: dict) -> None:
    index = doc["measurements"].index(word_entry(doc))
    word_entry(doc)["value"] = [{"start": 3.2, "end": 3.55}]
    assert_rejected(doc, f"$.measurements[{index}].value: word_prominence holds a value per word")


def test_an_empty_list_is_still_an_empty_range_list_not_a_word_list(doc: dict) -> None:
    word_entry(doc)["value"] = []
    validate(doc)
