"""Result contract v1: the packaged JSON Schema and the validator shared by CLI and MCP.

The schema (`schema/result.v1.schema.json`, packaged copy next to this module) checks shape
and enumerations; the cross-field rules of spec § 4 — timecodes, evidence references and their
hashes, the category dictionary, coverage of audio claims, honest statuses, cost, limits — are
plain code here, because a schema cannot express them (R04, R05, R06, R09, R11). Everything here
reads the document alone; checks against the files of a run live in `sound_scout.findings`.
"""

import hashlib
import json
import math
import re
from importlib.resources import files
from pathlib import PurePosixPath, PureWindowsPath

from sound_scout._schema_messages import schema_errors
from sound_scout.doctor import TEST_DOUBLE

SCHEMA_VERSION = "1.2"
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
    "clip": "clip",
}
# One dictionary from the backend response to the eval protocol (docs/eval/PROTOCOL.md § 2).
SOUND_CATEGORIES = (
    "pronunciation",
    "repetition_self_correction",
    "semantic_stress",
    "pause_phrasing",
    "completeness",
)
MEASUREMENT_CATEGORIES = ("tempo", "level", "pitch_range", "pause_length")
TRANSCRIPT_CATEGORIES = ("filler_words", "content")
CATEGORY_BASES = {
    **dict.fromkeys(SOUND_CATEGORIES, ("audio_model_observation", "human_verified")),
    **dict.fromkeys(MEASUREMENT_CATEGORIES, ("measurement",)),
    **dict.fromkeys(TRANSCRIPT_CATEGORIES, ("transcript",)),
}
FORBIDDEN_CATEGORIES = frozenset(
    {
        "emotion",
        "emotions",
        "mood",
        "health",
        "diagnosis",
        "personality",
        "confidence",
        "self_confidence",
        "attractiveness",
        "charisma",
        "intelligence",
        "honesty",
        "anxiety",
        "nervousness",
        "monotony",
        "эмоции",
        "здоровье",
        "диагноз",
        "личность",
        "уверенность",
        "привлекательность",
        "харизма",
        "тревожность",
        "монотонность",
    }
)
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
    errors = _non_finite_numbers(SCHEMA, result)
    errors += schema_errors(SCHEMA, result)
    if errors:
        return errors
    assert isinstance(result, dict)
    return (
        _timecode_errors(result)
        + _suspect_segment_errors(result)
        + _uniqueness_errors(result)
        + _stage_errors(result)
        + _evidence_errors(result)
        + _audio_review_errors(result)
        + _completeness_errors(result)
        + _capability_errors(result)
        + _assessment_errors(result)
        + _cost_errors(result)
        + _limits_errors(result)
        + _artifact_errors(result)
        + _measurement_errors(result)
        + _category_errors(result)
        + _polarity_errors(result)
        + _clip_errors(result)
        + _audio_coverage_errors(result)
        + _review_provenance_errors(result)
        + _synthetic_errors(result)
    )


def _non_finite_numbers(schema: dict, document: object) -> list[str]:
    """NaN/Infinity anywhere in `document`, before its schema check. A path names only keys
    `schema` declares: any other key is text of the document, so a number under it is reported
    at the object holding the key (R09, R11)."""
    return list(dict.fromkeys(_non_finite(document, "$", [schema], schema)))


def _non_finite(value: object, path: str, schemas: list, root: dict) -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: NaN/Infinity is not allowed, use null"]
    schemas = _alternatives(schemas, root)
    errors = []
    if isinstance(value, dict):
        for key, item in value.items():
            declared = [s["properties"][key] for s in schemas if key in s.get("properties", {})]
            if declared:
                errors += _non_finite(item, f"{path}.{key}", declared, root)
            elif _has_non_finite(item):
                errors.append(f"{path}: NaN/Infinity under an undeclared key is not allowed")
    elif isinstance(value, list):
        items = [s["items"] for s in schemas if isinstance(s.get("items"), dict)]
        for i, item in enumerate(value):
            errors += _non_finite(item, f"{path}[{i}]", items, root)
    return errors


def _alternatives(schemas: list, root: dict) -> list[dict]:
    """The subschemas a value may match: local `$ref`s resolved, `oneOf`/`anyOf`/`allOf` opened."""
    found = []
    for subschema in schemas:
        if not isinstance(subschema, dict):
            continue
        found.append(subschema)
        if "$ref" in subschema:
            target = root
            for part in subschema["$ref"].removeprefix("#/").split("/"):
                target = target[part]
            found += _alternatives([target], root)
        for keyword in ("oneOf", "anyOf", "allOf"):
            found += _alternatives(subschema.get(keyword, []), root)
    return found


def _has_non_finite(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_non_finite(item) for item in value)
    return False


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
    for i, measurement in enumerate(result["measurements"]):
        if isinstance(measurement["value"], list):
            spans += [
                (f"$.measurements[{i}].value[{j}]", r["start"], r["end"])
                for j, r in enumerate(measurement["value"])
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


def _suspect_segment_errors(result: dict) -> list[str]:
    """Flagged ASR segments lie inside the source; a zero-length one is kept, since Whisper
    emits such segments and hiding them is not allowed. Without a raw transcript there are
    none (docs/decisions/0005)."""
    transcript = result["transcript"]
    if transcript["raw"] is None and transcript["suspect_segments"]:
        return ["$.transcript.suspect_segments: must be empty when transcript.raw is null"]
    duration = result["source"]["duration_s"]
    errors = []
    for i, item in enumerate(transcript["suspect_segments"]):
        path = f"$.transcript.suspect_segments[{i}]"
        if item["start"] > item["end"]:
            errors.append(f"{path}: start must not exceed end, got {item['start']} > {item['end']}")
        if item["end"] > duration:
            errors.append(f"{path}.end: exceeds source duration {duration}, got {item['end']}")
    return errors


def _uniqueness_errors(result: dict) -> list[str]:
    """Artifact ids are unique; finding and unlocalized ids share one namespace."""
    errors = []
    for fields in (("findings", "unlocalized"), ("artifacts",)):
        seen: set[str] = set()
        for field in fields:
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


def evidence_claims(result: dict) -> list[tuple[str, str, str, list[dict]]]:
    """Every (path, basis field, basis, evidence_refs) claiming evidence: findings, unlocalized
    observations, corrections."""
    claims = [
        (f"$.{name}[{i}]", "evidence_basis", f["evidence_basis"], f["evidence_refs"])
        for name in ("findings", "unlocalized")
        for i, f in enumerate(result[name])
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
    for claim_path, field, basis, refs in evidence_claims(result):
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
                if ref["sha256"] is not None:
                    errors.append(
                        f"{claim_path}.evidence_refs[{j}].sha256: must be null for a measurement"
                    )
                continue
            artifact = artifacts.get(ref["ref"])
            if artifact is None:
                errors.append(f"{path}: unknown artifact {ref['ref']!r}")
                continue
            kind = ARTIFACT_KIND_FOR_REF_TYPE.get(ref["type"])
            if kind is not None and artifact["kind"] != kind:
                errors.append(f"{path}: artifact {ref['ref']!r} must be of kind {kind!r}")
            if artifact["sha256"] is None:
                errors.append(f"{path}: evidence artifact {ref['ref']!r} must carry a sha256")
            elif ref["sha256"] != artifact["sha256"]:
                errors.append(
                    f"{claim_path}.evidence_refs[{j}].sha256: does not match artifact "
                    f"{ref['ref']!r}"
                )
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
            for path, _, basis, refs in evidence_claims(result)
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
    uncovered = subtract(coverage["requested"], coverage["reviewed"])
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


def subtract(requested: list[dict], reviewed: list[dict]) -> list[tuple[float, float]]:
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
            f"{list(_LOCAL_PROVIDERS)}"
        )
    return errors


def _limits_errors(result: dict) -> list[str]:
    """`duration_mismatch` restates `|duration_mismatch_s| > tolerance`; null means undeclared."""
    limits = result["limits"]
    mismatch_s, flag = limits["duration_mismatch_s"], limits["duration_mismatch"]
    tolerance = limits["duration_mismatch_tolerance_s"]
    if mismatch_s is None:
        if flag is not None:
            return [
                "$.limits.duration_mismatch: must be null when duration_mismatch_s is null "
                f"(the container declared no duration), got {flag!r}"
            ]
        return []
    if flag is None:
        return [
            "$.limits.duration_mismatch: must be a boolean when duration_mismatch_s is a number, "
            f"got null for duration_mismatch_s {mismatch_s}"
        ]
    expected = abs(mismatch_s) > tolerance
    if flag != expected:
        relation = "exceeds" if expected else "is within"
        return [
            f"$.limits.duration_mismatch: must be {str(expected).lower()}, "
            f"|duration_mismatch_s| = {abs(mismatch_s)} {relation} "
            f"duration_mismatch_tolerance_s {tolerance}"
        ]
    return []


def _artifact_errors(result: dict) -> list[str]:
    errors = []
    for i, artifact in enumerate(result["artifacts"]):
        path = artifact["path"]
        windows = PureWindowsPath(path)
        if "\0" in path:
            errors.append(f"$.artifacts[{i}].path: must not contain a NUL character")
        elif PurePosixPath(path).is_absolute() or windows.drive or windows.root:
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


def _measurement_errors(result: dict) -> list[str]:
    """A value not obtained is null with a reason, never a bare null (spec § 4, R11)."""
    return [
        f"$.measurements[{i}].reason: a reason is required when value is null"
        for i, measurement in enumerate(result["measurements"])
        if measurement["value"] is None and not _text(measurement["reason"])
    ]


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _category_errors(result: dict) -> list[str]:
    """Categories come from one dictionary, each on its own basis; forbidden inferences never
    pass (R04: no emotions, health, personality, confidence; R06: low F0 is not monotony)."""
    errors = []
    for name in ("findings", "unlocalized"):
        for i, item in enumerate(result[name]):
            path, category = f"$.{name}[{i}].category", item["category"]
            basis = item["evidence_basis"]
            if _normalized(category) in FORBIDDEN_CATEGORIES:
                errors.append(f"{path}: names a forbidden inference (R04, R06)")
            elif category not in CATEGORY_BASES:
                errors.append(f"{path}: not in the category dictionary (R04, R06)")
            elif basis not in CATEGORY_BASES[category]:
                errors.append(
                    f"{path}: this category cannot rest on evidence_basis {basis!r}, "
                    f"only on {list(CATEGORY_BASES[category])} (R04, R06)"
                )
    return errors


def _normalized(category: str) -> str:
    return re.sub(r"[\s-]+", "_", category.strip().lower())


def _polarity_errors(result: dict) -> list[str]:
    return [
        f"$.findings[{i}].correction: a negative finding needs a short action (R05)"
        for i, finding in enumerate(result["findings"])
        if finding["polarity"] == "negative" and not _text(finding["correction"])
    ]


def _clip_errors(result: dict) -> list[str]:
    """Every finding and unlocalized observation names a playable clip (R05); a finding's clips
    cover its interval (`_not_covering`)."""
    artifacts = {a["id"]: a for a in result["artifacts"]}
    errors = []
    for name in ("findings", "unlocalized"):
        for i, item in enumerate(result[name]):
            path = f"$.{name}[{i}]"
            clip_refs = [ref["ref"] for ref in item["evidence_refs"] if ref["type"] == "clip"]
            if not clip_refs:
                errors.append(f"{path}.evidence_refs: needs at least one reference of type 'clip'")
            if name != "findings":
                continue
            clips = [artifacts[ref] for ref in clip_refs if ref in artifacts]
            for clip in _not_covering(clips, item["start"], item["end"]):
                errors.append(f"{path}: outside clip {clip['id']!r} source range")
    return errors


def _audio_coverage_errors(result: dict) -> list[str]:
    """An audio claim lies inside coverage.reviewed and inside the source ranges of the review
    responses it cites (`_not_covering`); a positive finding is bounded by coverage.reviewed too
    (R05)."""
    reviewed = result["coverage"]["reviewed"]
    artifacts = {a["id"]: a for a in result["artifacts"]}
    errors = []
    claims = [
        (f"$.findings[{i}]", f, f["evidence_basis"], f["polarity"] == "positive")
        for i, f in enumerate(result["findings"])
    ]
    claims += [
        (f"$.transcript.corrections[{i}]", c, c["source"], False)
        for i, c in enumerate(result["transcript"]["corrections"])
    ]
    for path, item, basis, positive in claims:
        audio = _is_audio_claim(basis, item["evidence_refs"])
        if (audio or positive) and subtract([item], reviewed):
            errors.append(f"{path}: outside coverage.reviewed")
        if not audio:
            continue
        responses = cited_review_responses(item["evidence_refs"], artifacts)
        outside = _not_covering(responses, item["start"], item["end"])
        for response in responses:
            if not _has_range(response):
                errors.append(f"{path}: review_response {response['id']!r} has no source range")
            elif response in outside:
                errors.append(f"{path}: outside review_response {response['id']!r} source range")
    for i, item in enumerate(result["unlocalized"]):
        path = f"$.unlocalized[{i}]"
        cited = cited_review_responses(item["evidence_refs"], artifacts)
        cited += [
            artifacts[ref["ref"]]
            for ref in item["evidence_refs"]
            if ref["type"] == "clip" and ref["ref"] in artifacts
        ]
        for artifact in cited:
            if not _has_range(artifact):
                errors.append(f"{path}: {artifact['kind']} {artifact['id']!r} has no source range")
            elif subtract([_range(artifact)], reviewed):
                errors.append(
                    f"{path}: {artifact['kind']} {artifact['id']!r} outside coverage.reviewed"
                )
    return errors


def _review_provenance_errors(result: dict) -> list[str]:
    """Each cited review response has its provenance record next to it, hashed, same range."""
    artifacts = {a["id"]: a for a in result["artifacts"]}
    cited: dict[str, dict] = {}
    for _, _, _, refs in evidence_claims(result):
        for response in cited_review_responses(refs, artifacts):
            cited[response["id"]] = response
    return [
        f"$.artifacts[{_index(result, response['id'])}]: review_response {response['id']!r} has "
        f"no review_provenance artifact in {str(PurePosixPath(response['path']).parent)!r} "
        "with a sha256 and the same source range"
        for response in cited.values()
        if review_provenance_for(result, response) is None
    ]


def review_provenance_for(result: dict, response: dict) -> dict | None:
    """The `review_provenance` artifact paired with a `review_response` artifact, if any."""
    directory = PurePosixPath(response["path"]).parent
    return next(
        (
            a
            for a in result["artifacts"]
            if a["kind"] == "review_provenance"
            and PurePosixPath(a["path"]).parent == directory
            and a["sha256"] is not None
            and _range(a) == _range(response)
        ),
        None,
    )


def _synthetic_errors(result: dict) -> list[str]:
    """A test double proves nothing about hearing: its result is marked synthetic (T03, T09)."""
    provenance = result["provenance"]
    if provenance["backend"] != TEST_DOUBLE:
        return []
    note = provenance["note"]
    if provenance["synthetic"] is True and isinstance(note, str) and note.startswith("synthetic"):
        return []
    return [
        f"$.provenance.synthetic: a {TEST_DOUBLE!r} backend result must be synthetic: true "
        "with a note starting with 'synthetic'"
    ]


def _is_audio_claim(basis: str, refs: list[dict]) -> bool:
    return basis == "audio_model_observation" or any(
        ref["type"] == "audio_model_observation" for ref in refs
    )


def cited_review_responses(refs: list[dict], artifacts: dict[str, dict]) -> list[dict]:
    return [
        artifacts[ref["ref"]]
        for ref in refs
        if ref["type"] == "audio_model_observation"
        and artifacts.get(ref["ref"], {}).get("kind") == "review_response"
    ]


def _has_range(artifact: dict) -> bool:
    return artifact["source_start"] is not None and artifact["source_end"] is not None


def _range(artifact: dict) -> dict:
    return {"start": artifact["source_start"], "end": artifact["source_end"]}


def _not_covering(cited: list[dict], start: float, end: float) -> list[dict]:
    """The cited artifacts that do not cover `[start, end]`. Overlapping windows cover a merged
    finding together: when each overlaps the interval and their union leaves no gap in it, none
    is reported; otherwise each one that does not cover it alone is."""
    if (
        cited
        and all(
            _has_range(a) and a["source_start"] < end and start < a["source_end"] for a in cited
        )
        and not subtract([{"start": start, "end": end}], [_range(a) for a in cited])
    ):
        return []
    return [a for a in cited if not _covers(a, start, end)]


def _covers(artifact: dict, start: float, end: float) -> bool:
    return (
        _has_range(artifact)
        and artifact["source_start"] - TIME_EPS <= start
        and end <= artifact["source_end"] + TIME_EPS
    )


def _index(result: dict, artifact_id: str) -> int:
    return next(i for i, a in enumerate(result["artifacts"]) if a["id"] == artifact_id)
