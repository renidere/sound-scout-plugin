"""Result contract v1 (issue #11): the schema files, the example and the cross-field rules."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sound_scout import schema
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
)
HEX64 = "ab" * 32
EXAMPLE_DOC = json.loads(EXAMPLE.read_text(encoding="utf-8"))


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


def finding(id: str, basis: str, refs: list[dict]) -> dict:
    return {
        "id": id,
        "category": "test",
        "observation": "synthetic observation",
        "start": 1.0,
        "end": 2.0,
        "evidence_basis": basis,
        "evidence_refs": refs,
        "confidence_note": "",
        "impact": "",
        "correction": None,
    }


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
        doc["artifacts"].append(artifact("review-response", "review_response"))
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
    assert_rejected(doc, "$.artifacts[4]: start must be less than end")
    clip.update(source_start=-1.0, source_end=2.0)
    assert_rejected(doc, "$.artifacts[4].start: must be >= 0")
    clip.update(source_start=2.0, source_end=99.0)
    assert_rejected(doc, "$.artifacts[4].end: exceeds source duration")
    clip.update(source_start=2.0, source_end=None)
    assert_rejected(doc, "$.artifacts[4]: source_start and source_end must both be set")


def test_duplicate_finding_and_artifact_ids_are_rejected(doc: dict) -> None:
    doc["findings"].append(deepcopy(doc["findings"][0]))
    assert_rejected(doc, "$.findings[2].id: duplicate id 'f-001'")
    doc = deepcopy(EXAMPLE_DOC)
    doc["artifacts"].append(deepcopy(doc["artifacts"][0]))
    assert_rejected(doc, "$.artifacts[4].id: duplicate id 'transcript-raw'")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_infinity_are_rejected_but_null_passes(doc: dict, value: float) -> None:
    doc["measurements"][0]["value"] = value
    assert_rejected(doc, "$.measurements[0].value: NaN/Infinity is not allowed, use null")
    doc = deepcopy(EXAMPLE_DOC)
    doc["measurements"][1]["settings"]["threshold"] = [1.0, {"depth": value}]
    assert_rejected(
        doc, "$.measurements[1].settings.threshold[1].depth: NaN/Infinity is not allowed"
    )
    doc = deepcopy(EXAMPLE_DOC)
    doc["measurements"][0]["value"] = None
    validate(doc)


def test_evidence_basis_outside_the_enumeration_is_rejected(doc: dict) -> None:
    doc["findings"][0]["evidence_basis"] = "model_guess"
    assert_rejected(doc, "$.findings[0].evidence_basis: 'model_guess' is not one of")
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
    assert_rejected(doc, "$.stages.asr.status: 'done' is not one of")
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
    doc["findings"].append(
        finding("f-003", "human_verified", [{"type": "transcript", "ref": "transcript-raw"}])
    )
    assert_rejected(
        doc, "$.findings[2].evidence_refs: evidence_basis 'human_verified' needs at least one "
    )
    doc["artifacts"].append(artifact("label-1", "human_label"))
    doc["findings"][2]["evidence_refs"] = [{"type": "human_label", "ref": "label-1"}]
    validate(doc)
    doc["artifacts"][-1]["kind"] = "other"
    assert_rejected(doc, "artifact 'label-1' must be of kind 'human_label'")
    doc["artifacts"][-1].update(kind="human_label", sha256=None)
    assert_rejected(doc, "evidence artifact 'label-1' must carry a sha256")
    doc["findings"][2]["evidence_refs"] = [{"type": "human_label", "ref": "missing"}]
    assert_rejected(doc, "$.findings[2].evidence_refs[0].ref: unknown artifact 'missing'")


def test_evidence_refs_point_at_known_measurements_and_artifacts(doc: dict) -> None:
    doc["findings"][0]["evidence_refs"] = [{"type": "measurement", "ref": "unknown_metric"}]
    assert_rejected(doc, "$.findings[0].evidence_refs[0].ref: unknown measurement 'unknown_metric'")
    doc = deepcopy(EXAMPLE_DOC)
    doc["findings"][1]["evidence_refs"] = [{"type": "transcript", "ref": "report-md"}]
    assert_rejected(doc, "artifact 'report-md' must be of kind 'transcript_raw'")
    doc = deepcopy(EXAMPLE_DOC)
    doc["findings"][1]["evidence_refs"].append({"type": "artifact", "ref": "report-md"})
    validate(doc)
    doc["findings"][1]["evidence_refs"] = []
    assert_rejected(doc, "$.findings[1].evidence_refs: [] should be non-empty")


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
    doc["artifacts"][0]["sha256"] = HEX64
    doc = complete_review(doc)
    doc["findings"].append(
        finding(
            "f-003",
            "audio_model_observation",
            [{"type": "audio_model_observation", "ref": "review-response"}],
        )
    )
    validate(doc)
    doc["artifacts"][-1]["sha256"] = None
    assert_rejected(doc, "evidence artifact 'review-response' must carry a sha256")
    doc["findings"][2]["evidence_refs"].append({"type": "artifact", "ref": "report-md"})
    doc["artifacts"][-1]["sha256"] = HEX64
    validate(doc)


def test_transcript_raw_must_reference_a_transcript_raw_artifact(doc: dict) -> None:
    doc["transcript"]["raw"]["artifact_ref"] = "report-md"
    assert_rejected(doc, "$.transcript.raw.artifact_ref: 'report-md' must be an artifact")
    doc["transcript"]["raw"] = None
    doc["findings"][1]["evidence_refs"] = [{"type": "measurement", "ref": "speech_rate"}]
    doc["findings"][1]["evidence_basis"] = "measurement"
    validate(doc)


def test_audio_model_observation_requires_a_completed_audio_review(doc: dict) -> None:
    doc["findings"].append(
        finding(
            "f-003",
            "audio_model_observation",
            [{"type": "audio_model_observation", "ref": "review-response"}],
        )
    )
    doc["artifacts"].append(artifact("review-response", "review_response"))
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
    assert_rejected(doc, "$.transcript.corrections[0].evidence_refs: [] should be non-empty")
    correction["evidence_refs"] = [{"type": "transcript", "ref": "transcript-raw"}]
    correction["source"] = "human_verified"
    assert_rejected(
        doc,
        "$.transcript.corrections[0].evidence_refs: source 'human_verified' needs at least one "
        "reference of type 'human_label'",
    )
    doc["artifacts"].append(artifact("label-1", "human_label"))
    correction["evidence_refs"] = [{"type": "human_label", "ref": "label-1"}]
    validate(doc)
    doc["artifacts"][-1]["sha256"] = None
    assert_rejected(
        doc,
        "$.transcript.corrections[0].evidence_refs[0].ref: evidence artifact 'label-1' must "
        "carry a sha256",
    )
    correction["evidence_refs"] = [{"type": "human_label", "ref": "missing"}]
    assert_rejected(doc, "$.transcript.corrections[0].evidence_refs[0].ref: unknown artifact")


def test_correction_from_audio_model_requires_a_completed_audio_review(doc: dict) -> None:
    correction = doc["transcript"]["corrections"][0]
    doc["artifacts"].append(artifact("review-response", "review_response"))
    correction["source"] = "audio_model_observation"
    assert_rejected(
        doc,
        "$.transcript.corrections[0].evidence_refs: source 'audio_model_observation' needs at "
        "least one reference of type 'audio_model_observation'",
    )
    correction["evidence_refs"] = [{"type": "audio_model_observation", "ref": "review-response"}]
    errors = assert_rejected(
        doc,
        "$.transcript.corrections[0]: audio_model_observation evidence requires a completed "
        "audio_review stage",
    )
    assert len(errors) == 1
    doc = complete_review(doc)
    validate(doc)
    correction["source"] = "transcript"
    correction["evidence_refs"] = [
        {"type": "transcript", "ref": "transcript-raw"},
        {"type": "audio_model_observation", "ref": "review-response"},
    ]
    validate(doc)
    doc["stages"]["audio_review"].update(status="failed", reason="backend timed out")
    doc["status"] = "partial"
    doc["coverage"]["reviewed"] = []
    doc["coverage"]["unreviewed"] = deepcopy(doc["coverage"]["requested"])
    assert_rejected(doc, "$.transcript.corrections[0]: audio_model_observation evidence requires")


def test_text_only_evidence_cannot_carry_an_audio_model_observation_ref(doc: dict) -> None:
    doc["findings"][1]["evidence_refs"].append(
        {"type": "audio_model_observation", "ref": "review-response"}
    )
    doc["artifacts"].append(artifact("review-response", "review_response"))
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
    ],
)
def test_artifact_path_must_stay_inside_the_run_directory(
    doc: dict, path: str, fragment: str
) -> None:
    doc["artifacts"][2]["path"] = path
    assert_rejected(doc, f"$.artifacts[2].path: {fragment}")


def test_clip_artifacts_need_source_timecodes(doc: dict) -> None:
    doc["artifacts"].append(artifact("clip-1", "clip"))
    assert_rejected(doc, "$.artifacts[4]: kind 'clip' requires source_start and source_end")
    doc["artifacts"][4].update(kind="listening_copy", source_start=0.0, source_end=12.0)
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
        "$.source.duration_s: 'long' is not of type 'number'",
        "$.stages: [] is not of type 'object'",
    ]
    assert validation_errors(None) == ["$: None is not of type 'object'"]
    assert validation_errors([]) == ["$: [] is not of type 'object'"]
