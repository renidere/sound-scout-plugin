"""Human labels v1 of one corpus clip (T04, T06, R05): the schema, the rules, the evidence record.

The schema (`schema/labels.v1.schema.json`, packaged copy next to this module) checks the shape of
one label file; what it cannot express — events inside the clip, a comment on every `uncertain`,
a second opinion only for prosody and only from another listener, no trace of a model run or its
output — is plain code here. `validate` returns a report like the manifest validator does.
`evidence_record` is the only way this package turns labels into `human_verified` evidence: it
validates the very bytes it hashes and refuses otherwise (R05: `human_verified` is a separate,
genuine human labelling, never a model's self-description).
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator

from sound_scout.eval.manifest import PROSODY_CATEGORIES
from sound_scout.schema import _non_finite_numbers

SCHEMA_VERSION = "1.0"
SCHEMA_FILE_NAME = "labels.v1.schema.json"
SCHEMA_BYTES = files("sound_scout.eval").joinpath(SCHEMA_FILE_NAME).read_bytes()
SCHEMA: dict = json.loads(SCHEMA_BYTES)
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_BYTES).hexdigest()
CERTAINTIES = ("clear", "uncertain")
MODEL_OUTPUT_FILE_NAMES = (
    "result.json",
    "report.md",
    "response.raw.txt",
    "provenance.json",
    "raw.json",
    "corrected.json",
)
_RUN_ID = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{32}(?![0-9A-Fa-f])")
_RUN_REFERENCE = re.compile(
    r"\brun[_ -]?id\b|(?:^|[\\/\s])runs[\\/]|(?<![\w.-])(?:"
    + "|".join(re.escape(name) for name in MODEL_OUTPUT_FILE_NAMES)
    + r")\b",
    re.IGNORECASE,
)

_IDENTIFIER_PATHS = ("$.clip_id", "$.clip_sha256")


class LabelsError(Exception):
    """The label file cannot become `human_verified` evidence; `errors` lists every reason."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("labels are not valid human evidence:\n" + "\n".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class LabelsReport:
    """`errors` is empty when the label file is valid."""

    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(path: str | Path) -> LabelsReport:
    """Validate the label file at `path`; missing file or bad JSON is a report, not an exception."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return LabelsReport([f"$: cannot read labels: {exc}"])
    return _bytes_report(data)[1]


def validation_report(labels: object) -> LabelsReport:
    """Every violation as `<json path>: <message>`; the rules run only on a sound shape."""
    errors = _non_finite_numbers(labels, "$")
    errors += [
        f"{error.json_path}: {_schema_message(error)}"
        for error in sorted(Draft202012Validator(SCHEMA).iter_errors(labels), key=_json_path)
    ]
    if errors:
        return LabelsReport(errors)
    assert isinstance(labels, dict)
    return LabelsReport(
        _model_reference_errors(labels, "$") + _date_errors(labels) + _event_errors(labels)
    )


def evidence_record(
    path: str | Path, *, artifact_id: str, artifact_path: str, source_sha256: str
) -> dict:
    """The `human_label` artifact of a result, built from a label file that passed `validate`.

    The record carries the SHA-256 of the exact bytes validated here; labels of another clip
    (`clip_sha256` differs from the analysed source) are refused. `artifact_path` is the label
    file's place relative to the run directory, as the result contract requires.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise LabelsError([f"$: cannot read labels: {exc}"]) from exc
    labels, report = _bytes_report(data)
    if not report.ok:
        raise LabelsError(report.errors)
    if labels["clip_sha256"] != source_sha256:
        raise LabelsError(["$.clip_sha256: labels describe another clip than the analysed source"])
    return {
        "id": artifact_id,
        "kind": "human_label",
        "path": artifact_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_start": None,
        "source_end": None,
    }


def _bytes_report(data: bytes) -> tuple[dict, LabelsReport]:
    try:
        labels = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, LabelsReport([f"$: not a UTF-8 JSON document: {exc}"])
    return labels, validation_report(labels)


def _json_path(error) -> str:
    return error.json_path


def _schema_message(error) -> str:
    """Built from the schema rule alone: `error.message` quotes the annotator's text."""
    rule = error.validator_value
    match error.validator:
        case "required":
            return "; ".join(
                f"{name!r} is a required property" for name in rule if name not in error.instance
            )
        case "additionalProperties":
            allowed = sorted(error.schema["properties"])
            return f"Additional properties are not allowed, allowed: {allowed}"
        case "const":
            return f"{rule!r} was expected"
        case "enum":
            return f"must be one of {rule}"
        case "type":
            return f"must be of type {rule}"
        case "pattern":
            return f"must match pattern {rule!r}"
        case "minimum":
            return f"must be at least {rule}"
        case "exclusiveMinimum":
            return f"must be greater than {rule}"
        case "oneOf":
            return "must be null or a valid second opinion"
    return f"violates schema rule {error.validator!r}"


def _model_reference_errors(value: object, path: str) -> list[str]:
    """Independence (T06): no run ID, no run directory, no file a model or ASR wrote.

    The text itself is not repeated in the message: it is the annotator's free text. Corpus
    identifiers (`clip_id`, `clip_sha256`) are not free text: the schema pins their shape, and a
    clip ID may legitimately look like a hex digest or a file name.
    """
    if path in _IDENTIFIER_PATHS:
        return []
    if isinstance(value, str):
        if _RUN_ID.search(value) or _RUN_REFERENCE.search(value):
            return [
                f"{path}: refers to a model run or its output; labels are made without "
                "seeing model outputs"
            ]
        return []
    if isinstance(value, dict):
        return [e for k, v in value.items() for e in _model_reference_errors(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [e for i, v in enumerate(value) for e in _model_reference_errors(v, f"{path}[{i}]")]
    return []


def _date_errors(labels: dict) -> list[str]:
    try:
        datetime.fromisoformat(labels["annotated_at"])
    except ValueError:
        return ["$.annotated_at: must be an ISO 8601 date or date-time"]
    return []


def _event_errors(labels: dict) -> list[str]:
    duration = labels["clip_duration_s"]
    annotator = labels["annotator"].strip().casefold()
    errors = []
    for i, event in enumerate(labels["events"]):
        path = f"$.events[{i}]"
        start, end = event["start"], event["end"]
        if start >= end:
            errors.append(f"{path}: start must be less than end, got {start} >= {end}")
        if end > duration:
            errors.append(f"{path}.end: exceeds clip duration {duration}, got {end}")
        if event["certainty"] == "uncertain" and not _text(event["comment"]):
            errors.append(f"{path}.comment: an uncertain event needs a comment saying why")
        second = event.get("second_opinion")
        if second is None:
            continue
        if event["category"] not in PROSODY_CATEGORIES:
            errors.append(
                f"{path}.second_opinion: a second opinion is recorded for prosody categories "
                f"only, got {event['category']!r}"
            )
        if second["annotator"].strip().casefold() == annotator:
            errors.append(f"{path}.second_opinion.annotator: must be another listener")
        if not second["agrees"] and event["certainty"] != "uncertain":
            errors.append(
                f"{path}.certainty: listeners disagree, the event must be 'uncertain', "
                "not forced either way"
            )
    return errors


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def format_report(path: str | Path, report: LabelsReport) -> str:
    if report.ok:
        return f"{path}: labels valid"
    return "\n".join([f"{path}: labels not valid:"] + [f"  {error}" for error in report.errors])


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m sound_scout.eval.labels <labels.json>...", file=sys.stderr)
        return 2
    reports = [(path, validate(path)) for path in args]
    for path, report in reports:
        print(format_report(path, report))
    return 0 if all(report.ok for _, report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
