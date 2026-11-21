"""Run ID, request ID and the private run directory under the data root (R11).

`open_run` claims a request: a repeat with the same request ID and the same identity hash
returns the existing run and does no work, a repeat with a different identity hash is rejected
and the existing run stays untouched. The identity hash is SHA-256 of canonical JSON over the
R11 fields; a field not known yet is present but empty (null). The data root comes from the
operator configuration (`config.Config.data_root`), never from a tool argument.

Layout: `<data_root>/runs/<run_id>/` (0700) holds `request.json` (with the request's
`task_context`, read by `comparison`) and `manifest.json` (0600);
`<data_root>/runs/by-request/<sha256(request_id)>` names the run of a request, so a raw
request ID never becomes a path segment. Run directories are created exclusively and are never
overwritten; the index entry is published with a hard link, so it either names a run or does
not exist. Symlinked `runs`, `by-request` and run directories are rejected: the data root is
private, and a substituted directory must not pass as one.

An index entry whose run directory is gone (for example, old runs were removed but the index was
not) makes `open_run` raise `RequestRunMissing`; the entry stays and no new run is claimed, so a
repeat of that request is rejected explicitly rather than charged again (R11). `open_run` does
not repair such an entry: the operator removes the index entry together with the run directory.
"""

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sound_scout import __version__

IDENTITY_FIELDS = ("audio", "model", "preprocessing", "ranges", "prompt_schema", "task_context")
RUNS_DIR_NAME = "runs"
INDEX_DIR_NAME = "by-request"
REQUEST_FILE_NAME = "request.json"
MANIFEST_FILE_NAME = "manifest.json"
DIR_MODE = 0o700
FILE_MODE = 0o600
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class RequestMismatch(Exception):
    """The request ID is already bound to a different identity hash; nothing was changed."""


class RequestIndexCorrupt(Exception):
    """The request index entry does not name a run; the request cannot be opened safely."""


class RequestRunMissing(RequestIndexCorrupt):
    """The request index entry names a run whose directory no longer exists."""


@dataclass(frozen=True)
class Run:
    run_id: str
    path: Path
    request_id: str
    identity_hash: str
    reused: bool
    complete: bool
    created: bool


def identity_hash(identity: Mapping[str, object]) -> str:
    """SHA-256 of canonical JSON (sorted keys, compact) over `IDENTITY_FIELDS`."""
    unknown = sorted(set(identity) - set(IDENTITY_FIELDS))
    if unknown:
        raise ValueError(f"unknown identity fields: {unknown}; known: {list(IDENTITY_FIELDS)}")
    canonical = {name: identity.get(name) for name in IDENTITY_FIELDS}
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_run_id() -> str:
    return uuid.uuid4().hex


def open_run(request_id: str, identity: Mapping[str, object], data_root: Path) -> Run:
    """Return the run of `request_id`, creating it when the request is new.

    `reused` is true only for a complete run with the same identity hash: the caller then has
    a ready result and must not repeat the work. `created` is true only when this call claimed
    the request and the work is the caller's to do. A partial run (manifest `complete: false`)
    is returned with `created=False, reused=False, complete=False`: the same directory, nothing
    recreated, its outcome unknown, so the caller resumes or reports it instead of charging the
    request again. A different identity hash raises `RequestMismatch`.
    """
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string")
    digest = identity_hash(identity)
    runs_dir = data_root / RUNS_DIR_NAME
    index_dir = runs_dir / INDEX_DIR_NAME
    _ensure_private_dir(runs_dir)
    _ensure_private_dir(index_dir)
    index_path = index_dir / hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    if not index_path.exists():
        run_id = new_run_id()
        run_dir = runs_dir / run_id
        _create_private_dir(run_dir)
        _write_private_json(
            run_dir / REQUEST_FILE_NAME,
            {
                "request_id": request_id,
                "identity_hash": digest,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "package_version": __version__,
                "task_context": identity.get("task_context"),
            },
        )
        _write_private_json(run_dir / MANIFEST_FILE_NAME, {"run_id": run_id, "complete": False})
        if _publish_index_entry(index_dir, index_path, run_id):
            return Run(run_id, run_dir, request_id, digest, False, False, created=True)
        # Another caller claimed the request first; our fresh, unindexed directory goes.
        shutil.rmtree(run_dir)
    return _existing_run(runs_dir, index_path, request_id, digest)


def complete_run(run: Run) -> Run:
    """Mark the run complete in `manifest.json` and return the updated handle."""
    manifest_path = run.path / MANIFEST_FILE_NAME
    manifest = _read_json(manifest_path)
    manifest["complete"] = True
    staging = run.path / f".{MANIFEST_FILE_NAME}.{uuid.uuid4().hex}"
    _write_private_json(staging, manifest)
    os.replace(staging, manifest_path)
    return Run(
        run.run_id, run.path, run.request_id, run.identity_hash, run.reused, True, run.created
    )


def discard_run(run: Run) -> None:
    """Remove a run this call created and that never started its work, with its index entry.

    Only for `run.created and not run.complete` (else `ValueError`): `app.analyze` uses it when
    its worker refused to start on changed code (#141), so no stage ran and nothing was charged.
    An index entry that does not name the run raises `RequestIndexCorrupt` and nothing is removed.
    The entry goes first, so a failure in between leaves an unindexed directory, never an entry
    that names a missing run (`RequestRunMissing`).
    """
    if not run.created or run.complete:
        raise ValueError("only a run this call created and did not complete can be discarded")
    digest = hashlib.sha256(run.request_id.encode("utf-8")).hexdigest()
    index_path = run.path.parent / INDEX_DIR_NAME / digest
    try:
        named = index_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        named = None
    if named != run.run_id:
        raise RequestIndexCorrupt(f"index entry {index_path.name} does not name run {run.run_id}")
    os.unlink(index_path)
    shutil.rmtree(run.path)


def _publish_index_entry(index_dir: Path, index_path: Path, run_id: str) -> bool:
    """Bind the request to `run_id` atomically; false when another run already holds it."""
    staging = index_dir / f".{index_path.name}.{uuid.uuid4().hex}"
    _write_private_text(staging, run_id, sync=True)
    try:
        os.link(staging, index_path)
    except FileExistsError:
        return False
    finally:
        os.unlink(staging)
    return True


def _existing_run(runs_dir: Path, index_path: Path, request_id: str, digest: str) -> Run:
    run_id = index_path.read_text(encoding="utf-8").strip()
    if not RUN_ID_PATTERN.match(run_id):
        raise RequestIndexCorrupt(f"index entry {index_path.name} does not name a run")
    run_dir = runs_dir / run_id
    try:
        _require_private_dir(run_dir)
    except FileNotFoundError:
        raise RequestRunMissing(
            f"index entry {index_path.name} names run {run_id}, but its directory is missing"
        ) from None
    request = _read_json(run_dir / REQUEST_FILE_NAME)
    if request["identity_hash"] != digest:
        raise RequestMismatch(
            f"request {request_id!r} is already bound to run {run_id} with another identity"
        )
    complete = _read_json(run_dir / MANIFEST_FILE_NAME).get("complete") is True
    return Run(run_id, run_dir, request_id, digest, complete, complete, created=False)


def _ensure_private_dir(path: Path) -> None:
    try:
        _create_private_dir(path)
    except FileExistsError:
        _require_private_dir(path)


def _require_private_dir(path: Path) -> None:
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise FileExistsError(f"{path} is not a directory or is a symlink")


def _create_private_dir(path: Path) -> None:
    os.mkdir(path, DIR_MODE)
    os.chmod(path, DIR_MODE)


def _write_private_json(path: Path, data: dict[str, object]) -> None:
    _write_private_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_private_text(path: Path, text: str, *, sync: bool = False) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    try:
        os.fchmod(fd, FILE_MODE)
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(text)
        if sync:
            out.flush()
            os.fsync(fd)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
