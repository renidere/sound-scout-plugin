"""Import one returned working copy of a blind labeling package (#91).

`sound-scout eval import-labels <submission> --package <package>` checks `labels/<ID>.json` and
the optional `review.csv` of one returned blind folder against `<package>/operator-private/
alias-map.json` alone: the expected neutral IDs, `clip_sha256`, duration, the source record ID,
`source_corpus` and split. Audio, `clip-index.json` and the blind folders are never read.

A file is unsigned when `annotator` and `annotated_at` are both blank: it is copied, never
validated, never mapped and never counted as "no events"; an unsigned file with events is
refused. A signed file goes through `sound_scout.eval.labels.validate`. The import is atomic: any
refusal writes nothing and names every refused file with its reason code. On success a new
`<corpus dir>/<package name>-returns/<NNN>/` holds the byte copy of the submission
(`submission/`), the signed files with only `clip_id` replaced by the source record ID
(`mapped/<source_corpus>/<original id>.json`) and the receipt `import.json` without absolute
paths. The manifest and the split are only read, never written: a dev record stays dev.
Rules: docs/decisions/0014-label-import.md.
"""

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sound_scout import __version__
from sound_scout.eval import labels

FORMAT_VERSION = 1
RECEIPT_KIND = "label_import_receipt"
ALIAS_MAP_KIND = "private_transport_aliases_not_qualification_manifest"
ALIAS_MAP_PATH = Path("operator-private") / "alias-map.json"
RETURNS_SUFFIX = "-returns"
RECEIPT_NAME = "import.json"
LABELS_DIR = "labels"
REVIEW_CSV = "review.csv"
REVIEW_HEADER = [
    "clip_id",
    "audio_path",
    "duration_s",
    "decision",
    "model_outputs_seen",
    "annotator",
    "date",
    "note",
]
REVIEW_DECISIONS = ("", "use", "reject", "needs_second_listener")
REVIEW_SEEN = ("", "no", "false", "yes", "true")
SPLITS = ("dev", "heldout")
DIR_MODE = 0o700
FILE_MODE = 0o600
MAX_SUBMISSIONS = 999

IMPORTED = "imported"
UNSIGNED = "unsigned"
ALREADY_IMPORTED = "already_imported"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NUMBER = re.compile(r"[0-9]{3}")


class LabelImportError(Exception):
    """The whole command is refused (alias map, layout, nothing new); nothing was written."""


@dataclass(frozen=True)
class Refusal:
    """One reason a file of the submission is refused; `file` is relative to the submission."""

    file: str
    code: str
    details: tuple[str, ...] = ()


class ImportRefused(Exception):
    """At least one file of the submission is refused; nothing was written."""

    def __init__(self, refusals: list[Refusal]) -> None:
        self.refusals = refusals
        self.files = sorted({refusal.file for refusal in refusals})
        super().__init__(f"import refused: {len(self.files)} files, nothing written")


@dataclass(frozen=True)
class ImportSummary:
    """`output` is `<package name>-returns/<NNN>`, relative to the corpus directory."""

    output: str
    path: Path
    imported: int
    unsigned: int
    already_imported: int


@dataclass(frozen=True)
class Alias:
    alias: str
    source_corpus: str
    original_id: str
    sha256: str
    duration_s: float
    split: str


@dataclass
class _Checked:
    alias: Alias
    name: str
    data: bytes
    labels: dict
    signed: bool
    manifest_check: str
    status: str = ""
    refusals: list[Refusal] = field(default_factory=list)


def import_submission(submission_dir: str | Path, package_dir: str | Path) -> ImportSummary:
    package = Path(package_dir).resolve()
    submission = Path(submission_dir).resolve()
    if not package.is_dir():
        raise LabelImportError("--package is not a directory")
    if not submission.is_dir():
        raise LabelImportError("the submission is not a directory")
    if submission.is_relative_to(package):
        raise LabelImportError(
            "the submission lies inside the package: fill a copy of the blind folder, not the "
            "delivered package"
        )
    alias_map_bytes, aliases = load_alias_map(package / ALIAS_MAP_PATH)
    returns = package.parent / f"{package.name}{RETURNS_SUFFIX}"
    earlier = _earlier_imports(returns)
    label_files = _label_files(submission)

    refusals: list[Refusal] = []
    review_rows, review_bytes = _review_rows(submission, aliases, refusals)
    manifests: dict[str, dict | None] = {}
    checked: list[_Checked] = []
    for entry in label_files:
        item = _check_label_file(entry, aliases, package.parent, manifests, refusals)
        if item is None:
            continue
        _check_signed(item, entry, review_rows, earlier)
        refusals += item.refusals
        if not item.refusals:
            checked.append(item)
    if refusals:
        raise ImportRefused(refusals)
    if not any(item.status == IMPORTED for item in checked):
        raise LabelImportError("nothing to import: no new signed label file in the submission")
    return _write(returns, package.name, alias_map_bytes, checked, review_bytes)


def load_alias_map(path: Path) -> tuple[bytes, dict[str, Alias]]:
    """The neutral ID → source record table; a message names the field, never its value."""
    name = ALIAS_MAP_PATH.as_posix()
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LabelImportError(f"{name}: cannot be read ({type(exc).__name__})") from exc
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelImportError(f"{name}: not a UTF-8 JSON document") from exc
    if not isinstance(document, dict) or document.get("kind") != ALIAS_MAP_KIND:
        raise LabelImportError(f"{name}: kind must be {ALIAS_MAP_KIND!r}")
    records = document.get("records")
    if not isinstance(records, list):
        raise LabelImportError(f"{name}: records must be a list")
    aliases: dict[str, Alias] = {}
    originals: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        at = f"{name}: records[{index}]"
        alias = _alias(record, at)
        if alias.alias in aliases:
            raise LabelImportError(f"{at}.alias: repeats an earlier record")
        if (alias.source_corpus, alias.original_id) in originals:
            raise LabelImportError(f"{at}.original_record.id: repeats within its source_corpus")
        aliases[alias.alias] = alias
        originals.add((alias.source_corpus, alias.original_id))
    return data, aliases


def _alias(record: object, at: str) -> Alias:
    if not isinstance(record, dict):
        raise LabelImportError(f"{at}: must be an object")
    original = record.get("original_record")
    if not isinstance(original, dict):
        raise LabelImportError(f"{at}.original_record: must be an object")
    for key, value in (
        ("alias", record.get("alias")),
        ("source_corpus", record.get("source_corpus")),
        ("original_record.id", original.get("id")),
    ):
        if not _is_identifier(value):
            raise LabelImportError(f"{at}.{key}: must be an identifier")
    if not isinstance(original.get("sha256"), str) or not _SHA256.fullmatch(original["sha256"]):
        raise LabelImportError(f"{at}.original_record.sha256: must be 64 lowercase hex digits")
    duration = original.get("duration_s")
    if not _is_number(duration) or not math.isfinite(duration) or duration <= 0:
        raise LabelImportError(f"{at}.original_record.duration_s: must be a number above 0")
    if original.get("split") not in SPLITS:
        raise LabelImportError(f"{at}.original_record.split: must be one of {list(SPLITS)}")
    return Alias(
        alias=record["alias"],
        source_corpus=record["source_corpus"],
        original_id=original["id"],
        sha256=original["sha256"],
        duration_s=duration,
        split=original["split"],
    )


def _is_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _earlier_imports(returns: Path) -> dict[str, set[str]]:
    """alias → SHA-256 of every file an earlier submission of this package imported."""
    if returns.is_symlink() or (returns.exists() and not returns.is_dir()):
        raise LabelImportError(f"{returns.name}/ is not a plain directory")
    earlier: dict[str, set[str]] = {}
    for number in _submission_numbers(returns):
        receipt = returns / f"{number:03d}" / RECEIPT_NAME
        try:
            document = json.loads(receipt.read_text(encoding="utf-8"))
            records = [r for r in document["records"] if r["status"] == IMPORTED]
            pairs = [(r["alias"], r["submission_sha256"]) for r in records]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LabelImportError(
                f"{returns.name}/{number:03d}/{RECEIPT_NAME}: cannot be read as a receipt"
            ) from exc
        for alias, digest in pairs:
            earlier.setdefault(alias, set()).add(digest)
    return earlier


def _submission_numbers(returns: Path) -> list[int]:
    if not returns.is_dir():
        return []
    return sorted(int(entry.name) for entry in returns.iterdir() if _NUMBER.fullmatch(entry.name))


def _label_files(submission: Path) -> list[Path]:
    directory = submission / LABELS_DIR
    if directory.is_symlink() or not directory.is_dir():
        raise LabelImportError(f"the submission has no {LABELS_DIR}/ directory")
    entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    if not entries:
        raise LabelImportError(f"{LABELS_DIR}/ of the submission is empty")
    return entries


def _review_rows(
    submission: Path, aliases: dict[str, Alias], refusals: list[Refusal]
) -> tuple[dict[str, dict], bytes | None]:
    path = submission / REVIEW_CSV
    if not os.path.lexists(path):
        return {}, None
    if path.is_symlink() or not path.is_file():
        refusals.append(Refusal(REVIEW_CSV, "bad_review_csv", ("not a regular file",)))
        return {}, None
    data = path.read_bytes()
    try:
        rows = list(csv.reader(data.decode("utf-8-sig").splitlines(), strict=True))
    except (UnicodeDecodeError, csv.Error):
        refusals.append(Refusal(REVIEW_CSV, "bad_review_csv", ("not a UTF-8 CSV file",)))
        return {}, data
    if not rows or rows[0] != REVIEW_HEADER:
        refusals.append(
            Refusal(
                REVIEW_CSV, "bad_review_csv", (f"the header must be {','.join(REVIEW_HEADER)}",)
            )
        )
        return {}, data
    by_id: dict[str, dict] = {}
    first_row: dict[str, int] = {}
    for number, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        if len(row) != len(REVIEW_HEADER):
            problem = f"row {number}: has {len(row)} columns, expected {len(REVIEW_HEADER)}"
            refusals.append(Refusal(REVIEW_CSV, "bad_review_csv", (problem,)))
            continue
        values = dict(zip(REVIEW_HEADER, row, strict=True))
        clip_id = values["clip_id"]
        problems = []
        if values["decision"].strip() not in REVIEW_DECISIONS:
            problems.append(f"row {number}: decision must be one of {list(REVIEW_DECISIONS)}")
        if values["model_outputs_seen"].strip() not in REVIEW_SEEN:
            problems.append(f"row {number}: model_outputs_seen must be one of {list(REVIEW_SEEN)}")
        if problems:
            refusals.append(Refusal(REVIEW_CSV, "bad_review_csv", tuple(problems)))
        if clip_id not in aliases:
            detail = f"row {number}: clip_id is not in the alias map"
            refusals.append(Refusal(REVIEW_CSV, "unknown_id", (detail,)))
        elif clip_id in by_id:
            detail = f"row {number}: clip_id repeats row {first_row[clip_id]}"
            refusals.append(Refusal(REVIEW_CSV, "duplicate_id", (detail,)))
        else:
            by_id[clip_id] = values
            first_row[clip_id] = number
    return by_id, data


def _check_label_file(
    entry: Path,
    aliases: dict[str, Alias],
    corpus_dir: Path,
    manifests: dict[str, dict | None],
    refusals: list[Refusal],
) -> _Checked | None:
    """The checks every label file gets, signed or not; None when the file cannot be read."""
    name = f"{LABELS_DIR}/{entry.name}"
    if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
        refusals.append(Refusal(name, "unexpected_entry", ("only regular <ID>.json files",)))
        return None
    alias = aliases.get(entry.stem)
    if alias is None:
        refusals.append(Refusal(name, "unknown_id", ("the file name is not in the alias map",)))
        return None
    data = entry.read_bytes()
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        document = None
    if not isinstance(document, dict):
        refusals.append(Refusal(name, "not_json", ("not a UTF-8 JSON object",)))
        return None
    signed = not (_blank(document.get("annotator")) and _blank(document.get("annotated_at")))
    item = _Checked(alias, name, data, document, signed, "absent")
    if document.get("clip_id") != alias.alias:
        item.refusals.append(Refusal(name, "id_mismatch", ("clip_id differs from the file name",)))
    if document.get("clip_sha256") != alias.sha256:
        item.refusals.append(Refusal(name, "changed_sha256", ("clip_sha256 differs from the map",)))
    duration = document.get("clip_duration_s")
    if not _is_number(duration) or duration != alias.duration_s:
        item.refusals.append(
            Refusal(name, "changed_duration", ("clip_duration_s differs from the map",))
        )
    if not signed and document.get("events") != []:
        item.refusals.append(
            Refusal(name, "unsigned_with_events", ("events need annotator and annotated_at",))
        )
    item.manifest_check = _manifest_check(item, corpus_dir, manifests)
    return item


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _manifest_check(item: _Checked, corpus_dir: Path, manifests: dict[str, dict | None]) -> str:
    """'matched' or 'absent'; a manifest that disagrees refuses the file. Only ever read."""
    alias = item.alias
    if alias.source_corpus not in manifests:
        manifests[alias.source_corpus] = _manifest_records(corpus_dir / alias.source_corpus)
    records = manifests[alias.source_corpus]
    if records is None:
        return "absent"
    record = records.get(alias.original_id)
    problems = []
    if record is None:
        problems.append("the manifest of its source_corpus has no record with this id")
    else:
        if record.get("split") != alias.split:
            problems.append("split differs between the manifest and the map")
        if record.get("sha256") != alias.sha256:
            problems.append("sha256 differs between the manifest and the map")
    if problems:
        item.refusals.append(Refusal(item.name, "manifest_mismatch", tuple(problems)))
    return "matched"


def _manifest_records(corpus: Path) -> dict | None:
    path = corpus / "manifest.json"
    if not path.is_file():
        return None
    unreadable = LabelImportError(f"{corpus.name}/manifest.json: cannot be read as a manifest")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))["records"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise unreadable from exc
    if not isinstance(records, list):
        raise unreadable
    return {r["id"]: r for r in records if isinstance(r, dict) and "id" in r}


def _check_signed(
    item: _Checked, entry: Path, review_rows: dict[str, dict], earlier: dict[str, set[str]]
) -> None:
    if not item.signed:
        item.status = UNSIGNED
        return
    report = labels.validate(entry)
    if not report.ok:
        item.refusals.append(Refusal(item.name, "invalid_labels", tuple(report.errors)))
    row = review_rows.get(item.alias.alias)
    if row is not None:
        problems = []
        if row["decision"].strip() == "reject":
            problems.append("review.csv rejects this clip, yet the file is signed")
        if row["model_outputs_seen"].strip() in ("yes", "true"):
            problems.append("review.csv says model outputs were seen, yet the file is signed")
        if problems:
            item.refusals.append(Refusal(item.name, "review_conflict", tuple(problems)))
    digest = hashlib.sha256(item.data).hexdigest()
    seen = earlier.get(item.alias.alias, set())
    if seen and digest not in seen:
        item.refusals.append(
            Refusal(item.name, "duplicate_id", ("imported earlier with other content",))
        )
    item.status = ALREADY_IMPORTED if digest in seen else IMPORTED


def _write(
    returns: Path,
    package_name: str,
    alias_map_bytes: bytes,
    checked: list[_Checked],
    review_bytes: bytes | None,
) -> ImportSummary:
    returns.mkdir(mode=DIR_MODE, exist_ok=True)
    numbers = _submission_numbers(returns)
    number = (numbers[-1] if numbers else 0) + 1
    if number > MAX_SUBMISSIONS:
        raise LabelImportError(f"{returns.name}/ holds {MAX_SUBMISSIONS} submissions already")
    final = returns / f"{number:03d}"
    staging = Path(tempfile.mkdtemp(prefix=f".{number:03d}.tmp-", dir=returns))
    try:
        records = [_write_record(staging, item) for item in checked]
        review_sha256 = None
        if review_bytes is not None:
            review_sha256 = _write_bytes(staging / "submission" / REVIEW_CSV, review_bytes)
        receipt = {
            "format_version": FORMAT_VERSION,
            "kind": RECEIPT_KIND,
            "package": package_name,
            "alias_map_sha256": hashlib.sha256(alias_map_bytes).hexdigest(),
            "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool_version": __version__,
            "review_csv_sha256": review_sha256,
            "records": records,
        }
        text = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        _write_bytes(staging / RECEIPT_NAME, text.encode("utf-8"))
        if os.path.lexists(final):
            raise LabelImportError(f"{returns.name}/{final.name} already exists")
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    statuses = [record["status"] for record in records]
    return ImportSummary(
        output=f"{returns.name}/{final.name}",
        path=final,
        imported=statuses.count(IMPORTED),
        unsigned=statuses.count(UNSIGNED),
        already_imported=statuses.count(ALREADY_IMPORTED),
    )


def _write_record(staging: Path, item: _Checked) -> dict:
    alias = item.alias
    submission_sha256 = _write_bytes(staging / "submission" / item.name, item.data)
    mapped_path = mapped_sha256 = None
    if item.status == IMPORTED:
        mapped = {**item.labels, "clip_id": alias.original_id}
        mapped_path = f"mapped/{alias.source_corpus}/{alias.original_id}.json"
        text = json.dumps(mapped, ensure_ascii=False, indent=2) + "\n"
        mapped_sha256 = _write_bytes(staging / mapped_path, text.encode("utf-8"))
        if not labels.validate(staging / mapped_path).ok:
            raise RuntimeError("the mapped copy of a valid label file does not validate")
    return {
        "alias": alias.alias,
        "status": item.status,
        "source_corpus": alias.source_corpus,
        "original_id": alias.original_id,
        "split": alias.split,
        "submission_sha256": submission_sha256,
        "mapped_path": mapped_path,
        "mapped_sha256": mapped_sha256,
        "manifest_check": item.manifest_check,
    }


def _write_bytes(path: Path, data: bytes) -> str:
    """Write `data` and return the SHA-256 of what is on disk, which must equal the source's."""
    path.parent.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    written = hashlib.sha256(path.read_bytes()).hexdigest()
    if written != hashlib.sha256(data).hexdigest():
        raise RuntimeError("a copied file differs from its source")
    return written
