"""Result contract v1: the packaged JSON Schema and the validator shared by CLI and MCP.

The schema (`schema/result.v1.schema.json`, packaged copy next to this module) checks shape
and enumerations; the cross-field rules of spec § 4 — timecodes, evidence references, honest
statuses, cost — are plain code here, because a schema cannot express them (R05, R09, R11).
"""

import hashlib
import json
import math
import re
from importlib.resources import files
from pathlib import PurePosixPath, PureWindowsPath

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "1.0"
SCHEMA_FILE_NAME = "result.v1.schema.json"
SCHEMA_BYTES = files("sound_scout").joinpath(SCHEMA_FILE_NAME).read_bytes()
SCHEMA: dict = json.loads(SCHEMA_BYTES)
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_BYTES).hexdigest()
STAGES = ("prepare", "asr", "acoustics", "audio_review", "report")
EVIDENCE_BASES = ("measurement", "transcript", "audio_model_observation", "human_verified")
REF_TYPE_FOR_BASIS = {
    "measurement": "measurement",
    "transcript": "transcript",
    "audio_model_observation": "audio_model_observation",
    "human_verified": "human_label",
}
ARTIFACT_KIND_FOR_REF_TYPE = {
    "transcript": "transcript_raw",
    "audio_model_observation": "review_response",
    "human_label": "human_label",
}
TIME_EPS = 0.001
_LOCAL_PROVIDERS = ("local", "none")
_PATH_SEPARATORS = re.compile(r"[\\/]")


class ResultValidationError(Exception):
    """The result document violates the v1 contract; `errors` lists every violation."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("result does not match the v1 contract:\n" + "\n".join(errors))
        self.errors = errors


def validate(result: object) -> None:
    errors = validation_errors(result)
    if errors:
        raise ResultValidationError(errors)


def validation_errors(result: object) -> list[str]:
    """Every violation as `<json path>: <message>`; cross-field rules run only on a sound shape."""
    errors = _non_finite_numbers(result, "$")
    errors += [
        f"{error.json_path}: {error.message}"
        for error in sorted(Draft202012Validator(SCHEMA).iter_errors(result), key=_json_path)
    ]
    if errors:
        return errors
    assert isinstance(result, dict)
    return (
        _timecode_errors(result)
        + _uniqueness_errors(result)
        + _stage_errors(result)
        + _evidence_errors(result)
        + _audio_review_errors(result)
        + _completeness_errors(result)
        + _capability_errors(result)
        + _assessment_errors(result)
        + _cost_errors(result)
        + _artifact_errors(result)
    )


def _json_path(error) -> str:
    return error.json_path


def _non_finite_numbers(value: object, path: str) -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: NaN/Infinity is not allowed, use null"]
    if isinstance(value, dict):
        return [e for k, v in value.items() for e in _non_finite_numbers(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [e for i, v in enumerate(value) for e in _non_finite_numbers(v, f"{path}[{i}]")]
    return []


def _timecode_errors(result: dict) -> list[str]:
    duration = result["source"]["duration_s"]
    spans: list[tuple[str, object, object]] = []
    spans += [(f"$.findings[{i}]", f["start"], f["end"]) for i, f in enumerate(result["findings"])]
    for name, ranges in result["coverage"].items():
        spans += [(f"$.coverage.{name}[{i}]", r["start"], r["end"]) for i, r in enumerate(ranges)]
    spans += [
        (f"$.transcript.corrections[{i}]", c["start"], c["end"])
        for i, c in enumerate(result["transcript"]["corrections"])
    ]
    errors = []
    for i, artifact in enumerate(result["artifacts"]):
        start, end = artifact["source_start"], artifact["source_end"]
        if start is None and end is None:
            continue
        if start is None or end is None:
            errors.append(f"$.artifacts[{i}]: source_start and source_end must both be set")
            continue
        spans.append((f"$.artifacts[{i}]", start, end))
    for path, start, end in spans:
        if start < 0:
            errors.append(f"{path}.start: must be >= 0, got {start}")
        if start >= end:
            errors.append(f"{path}: start must be less than end, got {start} >= {end}")
        if end > duration:
            errors.append(f"{path}.end: exceeds source duration {duration}, got {end}")
    return errors


def _uniqueness_errors(result: dict) -> list[str]:
    errors = []
    for field in ("findings", "artifacts"):
        seen: set[str] = set()
        for i, item in enumerate(result[field]):
            if item["id"] in seen:
                errors.append(f"$.{field}[{i}].id: duplicate id {item['id']!r}")
            seen.add(item["id"])
    return errors


def _stage_errors(result: dict) -> list[str]:
    return [
        f"$.stages.{name}.reason: a reason is required when status is {stage['status']!r}"
        for name, stage in result["stages"].items()
        if stage["status"] != "completed" and not _text(stage["reason"])
    ]


def _evidence_claims(result: dict) -> list[tuple[str, str, str, list[dict]]]:
    """Every (path, basis field, basis, evidence_refs) claiming evidence: findings, corrections."""
    claims = [
        (f"$.findings[{i}]", "evidence_basis", f["evidence_basis"], f["evidence_refs"])
        for i, f in enumerate(result["findings"])
    ]
    claims += [
        (f"$.transcript.corrections[{i}]", "source", c["source"], c["evidence_refs"])
        for i, c in enumerate(result["transcript"]["corrections"])
    ]
    return claims


def _evidence_errors(result: dict) -> list[str]:
    measurement_names = {m["name"] for m in result["measurements"]}
    artifacts = {a["id"]: a for a in result["artifacts"]}
    errors = []
    for claim_path, field, basis, refs in _evidence_claims(result):
        expected = REF_TYPE_FOR_BASIS[basis]
        if all(ref["type"] != expected for ref in refs):
            errors.append(
                f"{claim_path}.evidence_refs: {field} {basis!r} needs at least one "
                f"reference of type {expected!r}"
            )
        for j, ref in enumerate(refs):
            path = f"{claim_path}.evidence_refs[{j}].ref"
            if ref["type"] == "measurement":
                if ref["ref"] not in measurement_names:
                    errors.append(f"{path}: unknown measurement {ref['ref']!r}")
                continue
            artifact = artifacts.get(ref["ref"])
            if artifact is None:
                errors.append(f"{path}: unknown artifact {ref['ref']!r}")
                continue
            kind = ARTIFACT_KIND_FOR_REF_TYPE.get(ref["type"])
            if kind is None:
                continue
            if artifact["kind"] != kind:
                errors.append(f"{path}: artifact {ref['ref']!r} must be of kind {kind!r}")
            if artifact["sha256"] is None:
                errors.append(f"{path}: evidence artifact {ref['ref']!r} must carry a sha256")
    raw = result["transcript"]["raw"]
    if raw is not None:
        artifact = artifacts.get(raw["artifact_ref"])
        if artifact is None or artifact["kind"] != "transcript_raw":
            errors.append(
                f"$.transcript.raw.artifact_ref: {raw['artifact_ref']!r} must be an artifact of "
                "kind 'transcript_raw'"
            )
        elif artifact["sha256"] is None:
            errors.append(
                f"$.transcript.raw.artifact_ref: {raw['artifact_ref']!r} must carry a sha256"
            )
    return errors


def _audio_review_errors(result: dict) -> list[str]:
    errors = []
    review_completed = result["stages"]["audio_review"]["status"] == "completed"
    coverage = result["coverage"]
    if not review_completed:
        errors += [
            f"{path}: audio_model_observation evidence requires a completed audio_review stage"
            for path, _, basis, refs in _evidence_claims(result)
            if basis == "audio_model_observation"
            or any(ref["type"] == "audio_model_observation" for ref in refs)
        ]
        if coverage["reviewed"]:
            errors.append("$.coverage.reviewed: must be empty unless audio_review is completed")
    provenance = result["provenance"]
    if review_completed:
        for name in ("requested", "reviewed"):
            if not coverage[name]:
                errors.append(f"$.coverage.{name}: audio_review completed without any range")
        if provenance["backend"] == "none":
            errors.append("$.provenance.backend: audio_review completed without a backend")
        for field in ("audio_payload_sha256", "prompt_sha256"):
            if provenance[field] is None:
                errors.append(f"$.provenance.{field}: required when audio_review is completed")
        if "audio" not in provenance["input_modalities"]:
            errors.append("$.provenance.input_modalities: audio_review completed without 'audio'")
    return errors


def _completeness_errors(result: dict) -> list[str]:
    errors = []
    coverage = result["coverage"]
    for name, ranges in coverage.items():
        for i in range(1, len(ranges)):
            if ranges[i]["start"] < ranges[i - 1]["end"] - TIME_EPS:
                errors.append(f"$.coverage.{name}[{i}]: ranges must be sorted and not overlap")
    uncovered = _subtract(coverage["requested"], coverage["reviewed"])
    if not _same_ranges(coverage["unreviewed"], uncovered):
        errors.append(
            "$.coverage.unreviewed: must equal requested minus reviewed, expected "
            + json.dumps([{"start": s, "end": e} for s, e in uncovered])
        )
    if result["status"] == "complete":
        errors += [
            f"$.stages.{name}.status: status 'complete' requires every requested stage to be "
            f"completed, {name} is {stage['status']!r}"
            for name, stage in result["stages"].items()
            if stage["requested"] and stage["status"] != "completed"
        ]
        if uncovered:
            errors.append(
                "$.coverage.reviewed: status 'complete' requires reviewed to cover requested"
            )
    elif result["status"] in ("failed", "blocked") and not result["errors"]:
        errors.append(f"$.errors: status {result['status']!r} requires at least one error")
    return errors


def _subtract(requested: list[dict], reviewed: list[dict]) -> list[tuple[float, float]]:
    """`requested` minus `reviewed` as (start, end) pairs, ignoring gaps up to TIME_EPS."""
    pieces = []
    for wanted in requested:
        cursor = wanted["start"]
        for seen in sorted(reviewed, key=lambda r: r["start"]):
            if seen["end"] <= cursor or seen["start"] >= wanted["end"]:
                continue
            if seen["start"] - cursor > TIME_EPS:
                pieces.append((cursor, seen["start"]))
            cursor = max(cursor, seen["end"])
        if wanted["end"] - cursor > TIME_EPS:
            pieces.append((cursor, wanted["end"]))
    return pieces


def _same_ranges(ranges: list[dict], pairs: list[tuple[float, float]]) -> bool:
    return len(ranges) == len(pairs) and all(
        abs(r["start"] - s) <= TIME_EPS and abs(r["end"] - e) <= TIME_EPS
        for r, (s, e) in zip(ranges, pairs, strict=True)
    )


def _capability_errors(result: dict) -> list[str]:
    capabilities = result["capabilities"]
    if not capabilities["validated_ru"]:
        return []
    errors = []
    if not (capabilities["diction_validated_ru"] and capabilities["prosody_validated_ru"]):
        errors.append(
            "$.capabilities.validated_ru: requires diction_validated_ru and prosody_validated_ru"
        )
    if capabilities["audio_backend"] is None:
        errors.append("$.capabilities.validated_ru: requires a selected audio_backend")
    return errors


def _assessment_errors(result: dict) -> list[str]:
    errors = []
    review_completed = result["stages"]["audio_review"]["status"] == "completed"
    for name, assessment in result["assessment_status"].items():
        path = f"$.assessment_status.{name}"
        if assessment["status"] == "not_assessed" and not _text(assessment["reason"]):
            errors.append(f"{path}.reason: a reason is required when not_assessed")
        if (
            name in ("diction", "prosody")
            and not review_completed
            and assessment["status"] != ("not_assessed")
        ):
            errors.append(f"{path}.status: must be not_assessed without a completed audio_review")
    return errors


def _cost_errors(result: dict) -> list[str]:
    cost = result["cost"]
    kind, amount = cost["kind"], cost["amount"]
    errors = []
    if kind in ("actual", "estimated") and amount is None:
        errors.append(f"$.cost.amount: required when kind is {kind!r}")
    if kind == "unknown" and amount is not None:
        errors.append("$.cost.amount: must be null when kind is 'unknown'")
    if amount is not None and amount > 0 and cost["currency"] is None:
        errors.append("$.cost.currency: required when amount is greater than 0")
    if kind in ("actual", "estimated") and amount == 0 and cost["provider"] not in _LOCAL_PROVIDERS:
        errors.append(
            f"$.cost.amount: zero {kind} cost is only accepted for providers "
            f"{list(_LOCAL_PROVIDERS)}, got provider {cost['provider']!r}"
        )
    return errors


def _artifact_errors(result: dict) -> list[str]:
    errors = []
    for i, artifact in enumerate(result["artifacts"]):
        path = artifact["path"]
        windows = PureWindowsPath(path)
        if PurePosixPath(path).is_absolute() or windows.drive or windows.root:
            errors.append(f"$.artifacts[{i}].path: must be relative to the run directory")
        elif ".." in _PATH_SEPARATORS.split(path):
            errors.append(f"$.artifacts[{i}].path: must not contain '..' segments")
        if artifact["kind"] in ("clip", "listening_copy") and (
            artifact["source_start"] is None or artifact["source_end"] is None
        ):
            errors.append(
                f"$.artifacts[{i}]: kind {artifact['kind']!r} requires source_start and source_end"
            )
    return errors


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
