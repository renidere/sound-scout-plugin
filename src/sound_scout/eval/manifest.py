"""Qualification corpus manifest v1 (Gate 1, T04–T06): the schema and the rules between records.

The schema (`schema/corpus-manifest.v1.schema.json`, packaged copy next to this module) checks the
shape of one record; what a schema cannot express — one split per source group, unique hashes,
categories versus negative controls, the minimum volumes of docs/eval/PROTOCOL.md — is plain code
here. `validate` returns a report instead of raising: the evaluation runner and the owner need the
whole list of shortfalls, not the first one.
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath, PureWindowsPath

from jsonschema import Draft202012Validator

from sound_scout.schema import _non_finite_numbers

SCHEMA_VERSION = "1.0"
SCHEMA_FILE_NAME = "corpus-manifest.v1.schema.json"
SCHEMA_BYTES = files("sound_scout.eval").joinpath(SCHEMA_FILE_NAME).read_bytes()
SCHEMA: dict = json.loads(SCHEMA_BYTES)
SCHEMA_SHA256 = hashlib.sha256(SCHEMA_BYTES).hexdigest()
SPLITS = ("dev", "heldout")
DICTION_CATEGORIES = ("pronunciation", "repetition_self_correction")
PROSODY_CATEGORIES = ("semantic_stress", "pause_phrasing", "completeness")
CATEGORIES = DICTION_CATEGORIES + PROSODY_CATEGORIES
CATEGORY_GROUPS = {"diction": DICTION_CATEGORIES, "prosody": PROSODY_CATEGORIES}
PAIR_CATEGORY = "semantic_stress"
MIN_CLIPS = 20
MIN_NEGATIVE_CONTROLS = 5
MIN_HELDOUT_EVENTS_PER_GROUP = 5
MIN_HELDOUT_PAIRS = 2
_PATH_SEPARATORS = re.compile(r"[\\/]")


@dataclass(frozen=True)
class ManifestReport:
    """`errors` is empty when the manifest qualifies; `heldout_coverage` lists every category."""

    errors: list[str]
    heldout_coverage: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def unrepresented(self) -> list[str]:
        return [name for name in CATEGORIES if self.heldout_coverage[name] == 0]


def validate(path: str | Path) -> ManifestReport:
    """Validate the manifest at `path`; a missing file or bad JSON is a report, not an exception."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return ManifestReport([f"$: cannot read manifest: {exc}"], _empty_coverage())
    try:
        manifest = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ManifestReport([f"$: not a UTF-8 JSON document: {exc}"], _empty_coverage())
    return validation_report(manifest)


def validation_report(manifest: object) -> ManifestReport:
    """Every violation as `<json path>: <message>`; record rules run only on a sound shape."""
    errors = _non_finite_numbers(manifest, "$")
    errors += [
        f"{error.json_path}: {error.message}"
        for error in sorted(Draft202012Validator(SCHEMA).iter_errors(manifest), key=_json_path)
    ]
    if errors:
        return ManifestReport(errors, _empty_coverage())
    assert isinstance(manifest, dict)
    records = manifest["records"]
    errors = (
        _record_errors(records)
        + _uniqueness_errors(records)
        + _split_errors(records)
        + _pair_errors(records)
    )
    coverage = _heldout_coverage(records)
    return ManifestReport(errors + _volume_errors(records, coverage), coverage)


def _json_path(error) -> str:
    return error.json_path


def _empty_coverage() -> dict[str, int]:
    return dict.fromkeys(CATEGORIES, 0)


def _heldout_coverage(records: list[dict]) -> dict[str, int]:
    coverage = _empty_coverage()
    for record in records:
        if record["split"] == "heldout":
            for name in record["categories"]:
                coverage[name] += 1
    return coverage


def _record_errors(records: list[dict]) -> list[str]:
    errors = []
    for i, record in enumerate(records):
        path = record["path"]
        windows = PureWindowsPath(path)
        if PurePosixPath(path).is_absolute() or windows.drive or windows.root:
            errors.append(f"$.records[{i}].path: must be relative to the manifest directory")
        elif ".." in _PATH_SEPARATORS.split(path):
            errors.append(f"$.records[{i}].path: must not contain '..' segments")
        if record["negative_control"] and record["categories"]:
            errors.append(f"$.records[{i}].categories: a negative control carries no categories")
        if not record["negative_control"] and not record["categories"]:
            errors.append(
                f"$.records[{i}].categories: a record without categories must be a negative control"
            )
        if record.get("pair_id") is not None and PAIR_CATEGORY not in record["categories"]:
            errors.append(
                f"$.records[{i}].pair_id: a member of a same-words pair carries {PAIR_CATEGORY!r}"
            )
    return errors


def _uniqueness_errors(records: list[dict]) -> list[str]:
    errors = []
    for field in ("id", "sha256"):
        first_index: dict[str, int] = {}
        for i, record in enumerate(records):
            value = record[field]
            if value in first_index:
                errors.append(
                    f"$.records[{i}].{field}: duplicate of $.records[{first_index[value]}]"
                    f" ({value!r})"
                )
            else:
                first_index[value] = i
    return errors


def _split_errors(records: list[dict]) -> list[str]:
    """Variants of one source never straddle dev and held-out (T06)."""
    splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        splits[record["source_group"]].add(record["split"])
    return [
        f"$.records: source_group {group!r} is present in both dev and heldout"
        for group, seen in splits.items()
        if len(seen) > 1
    ]


def _pair_errors(records: list[dict]) -> list[str]:
    members: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("pair_id") is not None:
            members[record["pair_id"]].append(record)
    errors = []
    for pair_id, pair in members.items():
        if len(pair) < 2:
            errors.append(f"$.records: pair {pair_id!r} has a single record")
        if len({record["source_group"] for record in pair}) > 1:
            errors.append(f"$.records: pair {pair_id!r} spans several source groups")
    return errors


def _heldout_pairs(records: list[dict]) -> int:
    counts = Counter(
        record["pair_id"]
        for record in records
        if record.get("pair_id") is not None and record["split"] == "heldout"
    )
    return sum(1 for n in counts.values() if n >= 2)


def _volume_errors(records: list[dict], coverage: dict[str, int]) -> list[str]:
    """The minimum volumes of PROTOCOL.md; every shortfall is named, none is averaged away."""
    errors = []
    if len(records) < MIN_CLIPS:
        errors.append(f"$.records: {len(records)} clips, the corpus needs at least {MIN_CLIPS}")
    controls = sum(1 for record in records if record["negative_control"])
    if controls < MIN_NEGATIVE_CONTROLS:
        errors.append(
            f"$.records: {controls} negative controls, the corpus needs at least "
            f"{MIN_NEGATIVE_CONTROLS}"
        )
    for name in CATEGORIES:
        if coverage[name] == 0:
            errors.append(
                f"$.records: category {name!r} is not represented in heldout; "
                "an unrepresented category does not qualify"
            )
    for group, names in CATEGORY_GROUPS.items():
        events = sum(coverage[name] for name in names)
        if events < MIN_HELDOUT_EVENTS_PER_GROUP:
            errors.append(
                f"$.records: {events} {group} events in heldout, the protocol needs at least "
                f"{MIN_HELDOUT_EVENTS_PER_GROUP}"
            )
    pairs = _heldout_pairs(records)
    if pairs < MIN_HELDOUT_PAIRS:
        errors.append(
            f"$.records: {pairs} same-words pairs in heldout, the protocol needs at least "
            f"{MIN_HELDOUT_PAIRS}"
        )
    return errors


def format_report(report: ManifestReport) -> str:
    lines = ["held-out coverage (records per category):"]
    lines += [f"  {name}: {count}" for name, count in report.heldout_coverage.items()]
    if report.unrepresented:
        lines.append("not represented in held-out: " + ", ".join(report.unrepresented))
    lines.append("manifest qualifies" if report.ok else "manifest does not qualify:")
    lines += [f"  {error}" for error in report.errors]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m sound_scout.eval.manifest <manifest.json>", file=sys.stderr)
        return 2
    report = validate(args[0])
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
