"""Human labels v1 (issue #27): the schema, the validator, the `human_verified` evidence record."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sound_scout import asr, review, schema
from sound_scout.eval import labels
from sound_scout.eval.labels import (
    LabelsError,
    LabelsReport,
    evidence_record,
    main,
    validate,
    validation_report,
)
from sound_scout.eval.manifest import CATEGORIES
from tests.test_schema import EXAMPLE_DOC, finding, ref

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_SCHEMA = REPO_ROOT / "schema" / "labels.v1.schema.json"
PACKAGED_SCHEMA = REPO_ROOT / "src" / "sound_scout" / "eval" / "labels.v1.schema.json"
LABELING = REPO_ROOT / "docs" / "eval" / "LABELING.md"
SOURCE_SHA256 = EXAMPLE_DOC["source"]["sha256"]


def event(category: str, start: float, end: float, **extra: object) -> dict:
    doc = {
        "category": category,
        "start": start,
        "end": end,
        "description": "synthetic: what the listener hears",
        "certainty": "clear",
        "comment": None,
    }
    doc.update(extra)
    return doc


def valid_labels() -> dict:
    """Synthetic labels of a 12 s clip: every category, one uncertain, one second opinion."""
    return {
        "schema_version": "1.0",
        "clip_id": "h-pron-1",
        "clip_sha256": SOURCE_SHA256,
        "clip_duration_s": 12.0,
        "annotator": "listener-a",
        "annotated_at": "2026-11-18T10:00:00+03:00",
        "model_outputs_seen": False,
        "events": [
            event("pronunciation", 0.5, 1.2),
            event("repetition_self_correction", 2.0, 3.5),
            event(
                "semantic_stress",
                4.0,
                4.6,
                second_opinion={"annotator": "listener-b", "agrees": True, "comment": None},
            ),
            event("pause_phrasing", 6.0, 7.5, certainty="uncertain", comment="synthetic: why"),
            event("completeness", 10.5, 12.0),
        ],
    }


@pytest.fixture
def doc() -> dict:
    return valid_labels()


def write(tmp_path: Path, doc: object, name: str = "h-pron-1.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def assert_rejected(doc: dict, fragment: str) -> LabelsReport:
    report = validation_report(doc)
    assert not report.ok
    assert any(fragment in error for error in report.errors), report.errors
    return report


def test_schema_is_valid_draft_2020_12_and_packaged_copy_is_byte_equal() -> None:
    Draft202012Validator.check_schema(labels.SCHEMA)
    root_bytes = ROOT_SCHEMA.read_bytes()
    assert root_bytes == PACKAGED_SCHEMA.read_bytes() == labels.SCHEMA_BYTES
    assert hashlib.sha256(root_bytes).hexdigest() == labels.SCHEMA_SHA256
    assert labels.SCHEMA["properties"]["schema_version"] == {"const": labels.SCHEMA_VERSION}
    assert tuple(labels.SCHEMA["$defs"]["category"]["enum"]) == CATEGORIES
    event_schema = labels.SCHEMA["$defs"]["event"]
    assert tuple(event_schema["properties"]["certainty"]["enum"]) == labels.CERTAINTIES
    assert "second_opinion" in event_schema["properties"]


def test_labels_require_annotator_date_events_and_independence_flag() -> None:
    required = labels.SCHEMA["required"]
    for key in ("annotator", "annotated_at", "events", "model_outputs_seen", "clip_duration_s"):
        assert key in required, key
    assert labels.SCHEMA["properties"]["model_outputs_seen"] == {"const": False}
    event_required = labels.SCHEMA["$defs"]["event"]["required"]
    for key in ("category", "start", "end", "description", "certainty", "comment"):
        assert key in event_required, key


def test_valid_label_file_passes(tmp_path: Path, doc: dict) -> None:
    report = validate(write(tmp_path, doc))
    assert report.ok and report.errors == []
    assert report == validation_report(doc)


def test_clip_without_events_is_valid(doc: dict) -> None:
    doc["events"] = []
    assert validation_report(doc).ok


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d.pop("annotator"), "'annotator' is a required property"),
        (lambda d: d.update(annotator="  "), "$.annotator"),
        (lambda d: d.pop("annotated_at"), "'annotated_at' is a required property"),
        (lambda d: d.update(annotated_at="yesterday"), "must be an ISO 8601 date or date-time"),
        (lambda d: d.pop("model_outputs_seen"), "'model_outputs_seen' is a required property"),
        (lambda d: d.update(model_outputs_seen=True), "$.model_outputs_seen: False was expected"),
        (lambda d: d.update(run_id="0" * 32), "$: Additional properties"),
        (lambda d: d.update(clip_sha256="abc"), "$.clip_sha256"),
        (lambda d: d["events"][0].update(category="emotion"), "$.events[0].category"),
        (lambda d: d["events"][0].update(certainty="maybe"), "$.events[0].certainty"),
        (lambda d: d["events"][0].update(description=""), "$.events[0].description"),
        (lambda d: d["events"][0].update(start=-0.1), "$.events[0].start"),
        (lambda d: d["events"][0].pop("comment"), "'comment' is a required property"),
    ],
)
def test_shape_rules(doc: dict, mutate, fragment: str) -> None:
    mutate(doc)
    assert_rejected(doc, fragment)


@pytest.mark.parametrize(
    "text",
    [
        "compare with run 0123456789abcdef0123456789abcdef",
        "checked against result.json",
        "see report.md of the analysis",
        "as the model said in response.raw.txt",
        "ASR raw.json had it",
        "data/runs/x/clips",
        "same as run_id above",
    ],
)
@pytest.mark.parametrize("field", ["description", "comment"])
def test_reference_to_a_model_run_or_report_is_rejected(doc: dict, field: str, text: str) -> None:
    doc["events"][0][field] = text
    report = assert_rejected(doc, f"$.events[0].{field}: refers to a model run or its output")
    assert not any(text in error for error in report.errors), "free text must not be echoed"


def test_reference_in_note_or_second_opinion_is_rejected(doc: dict) -> None:
    doc["note"] = "labelled after reading report.md"
    assert_rejected(doc, "$.note: refers to a model run or its output")
    doc = valid_labels()
    doc["events"][2]["second_opinion"]["comment"] = "agreed with result.json"
    assert_rejected(doc, "$.events[2].second_opinion.comment: refers to a model run")


def test_model_output_file_names_follow_the_names_the_package_writes() -> None:
    for name in (
        review.RAW_RESPONSE_FILE_NAME,
        review.PROVENANCE_FILE_NAME,
        asr.RAW_FILE_NAME,
        asr.CORRECTED_FILE_NAME,
    ):
        assert name in labels.MODEL_OUTPUT_FILE_NAMES, name


def test_hashes_and_ordinary_text_are_not_mistaken_for_run_references(doc: dict) -> None:
    doc["clip_sha256"] = "0123456789abcdef" * 4
    doc["events"][0]["description"] = "окончание «-ться» смазано, run-on фраза; results ok"
    assert validation_report(doc).ok


LABEL_TEXT = "секрет-разметки"


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda d: d.update(clip_id=f"{LABEL_TEXT} 1"), "$.clip_id"),
        (lambda d: d.update(clip_sha256=LABEL_TEXT), "$.clip_sha256"),
        (lambda d: d.update(model_outputs_seen=LABEL_TEXT), "$.model_outputs_seen"),
        (lambda d: d.update({LABEL_TEXT: "x"}), "$: Additional properties are not allowed"),
        (lambda d: d.update(note=[LABEL_TEXT]), "$.note"),
        (lambda d: d["events"].append(LABEL_TEXT), "$.events[5]"),
        (lambda d: d["events"][0].update(category=LABEL_TEXT), "$.events[0].category"),
        (lambda d: d["events"][0].update(certainty=LABEL_TEXT), "$.events[0].certainty"),
        (lambda d: d["events"][0].update(start=LABEL_TEXT), "$.events[0].start"),
        (lambda d: d["events"][2].update(second_opinion=LABEL_TEXT), "$.events[2].second_opinion"),
    ],
)
def test_schema_errors_name_the_rule_without_echoing_label_text(
    tmp_path: Path, capsys, doc: dict, mutate, path: str
) -> None:
    mutate(doc)
    report = assert_rejected(doc, path)
    assert not any(LABEL_TEXT in error for error in report.errors), report.errors
    assert main([str(write(tmp_path, doc))]) == 1
    assert LABEL_TEXT not in capsys.readouterr().out


@pytest.mark.parametrize("clip_id", ["0123456789abcdef0123456789abcdef", "result.json", "raw.json"])
def test_clip_identifiers_are_not_scanned_as_model_references(doc: dict, clip_id: str) -> None:
    doc["clip_id"] = clip_id
    assert validation_report(doc).ok
    doc["note"] = clip_id
    assert_rejected(doc, "$.note: refers to a model run or its output")


def test_events_outside_the_clip_or_inverted_are_rejected(doc: dict) -> None:
    doc["events"][4]["end"] = 12.5
    assert_rejected(doc, "$.events[4].end: exceeds clip duration 12.0, got 12.5")
    doc = valid_labels()
    doc["events"][0].update(start=1.2, end=1.2)
    assert_rejected(doc, "$.events[0]: start must be less than end")


@pytest.mark.parametrize("comment", [None, "", "   "])
def test_uncertain_without_comment_is_rejected(doc: dict, comment: str | None) -> None:
    doc["events"][3]["comment"] = comment
    assert_rejected(doc, "$.events[3].comment: an uncertain event needs a comment")


def test_second_opinion_is_for_prosody_from_another_listener_and_disagreement_is_uncertain(
    doc: dict,
) -> None:
    doc["events"][0]["second_opinion"] = {"annotator": "listener-b", "agrees": True, "comment": ""}
    assert_rejected(doc, "a second opinion is recorded for prosody categories only")
    doc = valid_labels()
    doc["events"][2]["second_opinion"]["annotator"] = " Listener-A "
    assert_rejected(doc, "$.events[2].second_opinion.annotator: must be another listener")
    doc = valid_labels()
    doc["events"][2]["second_opinion"]["agrees"] = False
    assert_rejected(doc, "$.events[2].certainty: listeners disagree")
    doc["events"][2].update(certainty="uncertain", comment="synthetic: listeners disagree")
    assert validation_report(doc).ok


def test_unreadable_or_missing_file_is_a_report_not_an_exception(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert validate(path).errors[0].startswith("$: not a UTF-8 JSON document")
    assert validate(tmp_path / "missing.json").errors[0].startswith("$: cannot read labels:")


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "1e999"])
def test_non_finite_number_is_rejected_before_the_schema(
    tmp_path: Path, doc: dict, raw: str
) -> None:
    text = json.dumps(doc).replace('"clip_duration_s": 12.0', f'"clip_duration_s": {raw}', 1)
    path = tmp_path / "labels.json"
    path.write_text(text, encoding="utf-8")
    report = validate(path)
    assert report.errors[0] == "$.clip_duration_s: NaN/Infinity is not allowed, use null"


def test_main_prints_each_file_and_exits_non_zero_on_any_invalid(tmp_path: Path, capsys) -> None:
    good = write(tmp_path, valid_labels())
    assert main([str(good)]) == 0
    assert "labels valid" in capsys.readouterr().out
    bad_doc = valid_labels()
    del bad_doc["annotator"]
    bad = write(tmp_path, bad_doc, "bad.json")
    assert main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "labels not valid" in out and "'annotator' is a required property" in out
    assert main([]) == 2


def human_verified_result(label_sha256: str = "ab" * 32) -> dict:
    result = deepcopy(EXAMPLE_DOC)
    result["findings"].append(
        finding("f-human", "human_verified", [ref("human_label", "labels-1", label_sha256)])
    )
    return result


def example_artifact(artifact_id: str) -> dict:
    return next(a for a in EXAMPLE_DOC["artifacts"] if a["id"] == artifact_id)


def test_evidence_record_from_a_valid_file_carries_its_hash_and_passes_schema_validate(
    tmp_path: Path,
) -> None:
    path = write(tmp_path, valid_labels())
    record = evidence_record(
        path,
        artifact_id="labels-1",
        artifact_path="labels/h-pron-1.json",
        source_sha256=SOURCE_SHA256,
    )
    assert record["kind"] == "human_label"
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    result = human_verified_result(record["sha256"])
    with pytest.raises(schema.ResultValidationError) as excinfo:
        schema.validate(result)
    assert any("unknown artifact 'labels-1'" in error for error in excinfo.value.errors)
    result["artifacts"].append(record)
    schema.validate(result)


def test_human_verified_without_a_human_label_record_fails_schema_validate() -> None:
    result = human_verified_result()
    raw = example_artifact("transcript-raw")
    result["findings"][-1]["evidence_refs"] = [ref("transcript", raw["id"], raw["sha256"])]
    errors = schema.validation_errors(result)
    assert any(
        "evidence_basis 'human_verified' needs at least one reference of type 'human_label'" in e
        for e in errors
    ), errors


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d.update(model_outputs_seen=True), "$.model_outputs_seen"),
        (lambda d: d.pop("annotator"), "'annotator' is a required property"),
        (lambda d: d["events"][0].update(comment="see report.md"), "refers to a model run"),
        (lambda d: d.update(clip_sha256="ab" * 32), "labels describe another clip"),
    ],
)
def test_evidence_record_is_refused_for_labels_that_fail_or_describe_another_clip(
    tmp_path: Path, mutate, fragment: str
) -> None:
    doc = valid_labels()
    mutate(doc)
    with pytest.raises(LabelsError) as excinfo:
        evidence_record(
            write(tmp_path, doc),
            artifact_id="labels-1",
            artifact_path="labels/x.json",
            source_sha256=SOURCE_SHA256,
        )
    assert any(fragment in error for error in excinfo.value.errors), excinfo.value.errors


def test_evidence_record_of_a_missing_or_broken_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LabelsError, match="cannot read labels"):
        evidence_record(
            tmp_path / "missing.json",
            artifact_id="l",
            artifact_path="l.json",
            source_sha256=SOURCE_SHA256,
        )
    broken = tmp_path / "broken.json"
    broken.write_bytes(b"\xff\xfe")
    with pytest.raises(LabelsError, match="not a UTF-8 JSON document"):
        evidence_record(
            broken, artifact_id="l", artifact_path="l.json", source_sha256=SOURCE_SHA256
        )


def test_labeling_guide_covers_listening_boundaries_ambiguity_and_independence() -> None:
    text = LABELING.read_text(encoding="utf-8")
    for heading in (
        "## Что слушать",
        "## Как ставить границы",
        "## Неоднозначное",
        "## Независимость от выходов модели",
    ):
        assert heading in text, heading
    for fragment in (
        "model_outputs_seen",
        "`uncertain`",
        "`second_opinion`",
        "исходной шкале",
        "python -m sound_scout.eval.labels",
        "PROTOCOL.md",
    ):
        assert fragment in text, fragment
    for name in CATEGORIES:
        assert f"`{name}`" in text, name
