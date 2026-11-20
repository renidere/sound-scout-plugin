"""Cleanup of the staging copies the Hermes plugin uploads (decision 0013 § 6, Decision in #122).

`POST /jobs` of the plugin writes each upload as `<uuid4().hex>.upload` directly in
`jobs.staging_root` and submits it; the job's `request.input.path` then names that file. The
runner calls `sweep` about once a minute (`SWEEP_SECONDS`), so nobody starts it by hand. It
deletes a staging copy when every job that names it has ended and either

- the job is `finished` and its note is `delivered`: it is never retried, the copy goes at once;
- `jobs.staging_retention_seconds` have passed since the job's `updated_at`: a `failed` or
  `cancelled` job keeps its copy until then, even after its note, because `jobs retry` needs
  it; once the copy is gone `jobs.retry` refuses the job with `not_retryable`.

An upload no job names (the plugin keeps it after `cli_timeout` or `cli_failed`) goes once its
mtime is older than the retention period. A `queued` or `running` job keeps its copy forever.

Only regular files named `<32 hex>.upload` directly in the staging root are deleted, through a
directory descriptor opened without following a symlink; a symlink, a directory, any other name
and any input from another allowed root are never touched. A job whose copy is gone gets
`staging_removed_at` and is not looked at again. The whole pass holds `jobs.lock`, and an
unreadable `job.json` stops it: a file it names could otherwise pass for an orphan.
"""

import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

from sound_scout import jobs
from sound_scout.config import Config

UPLOAD_NAME = re.compile(r"[0-9a-f]{32}\.upload")
SWEEP_SECONDS = 60
STAGING_REMOVED_AT = jobs.STAGING_REMOVED_AT


def sweep(config: Config, *, now: datetime | None = None) -> list[dict]:
    """Delete the staging copies that are due; one `{name, reason}` per deleted name, where
    `reason` is `delivered`, `retention` or `orphan`. Names only, never a path."""
    root = config.jobs.staging_root
    if root is None:
        return []
    now = now or datetime.now(UTC)
    retention = config.jobs.staging_retention_seconds
    with jobs._store(config.data_root) as (jobs_dir, _):
        try:
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError:
            return []
        try:
            named = _jobs_by_name(jobs_dir, root)
            if named is None:
                return []
            removed = []
            for name, holders in named.items():
                reason = _due(holders, now, retention)
                if reason is None:
                    continue
                if _unlink_regular(fd, name) is False:
                    continue
                for job_dir, job in holders:
                    job[STAGING_REMOVED_AT] = now.isoformat()
                    jobs._save(job_dir, job)
                removed.append({"name": name, "reason": reason})
            for name in sorted(os.listdir(fd)):
                if not UPLOAD_NAME.fullmatch(name) or name in named:
                    continue
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if now.timestamp() - info.st_mtime < retention:
                    continue
                if _unlink_regular(fd, name):
                    removed.append({"name": name, "reason": "orphan"})
            return removed
        finally:
            os.close(fd)


def _jobs_by_name(jobs_dir: Path, root: Path) -> dict[str, list[tuple[Path, dict]]] | None:
    """The jobs whose input is an upload of the staging root and still there, by file name;
    None when a `job.json` cannot be read."""
    named: dict[str, list[tuple[Path, dict]]] = {}
    for job_dir in jobs._job_dirs(jobs_dir):
        try:
            job = jobs._load(job_dir)
            path = Path(job["request"]["input"]["path"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if job.get(STAGING_REMOVED_AT) or path.parent != root:
            continue
        if UPLOAD_NAME.fullmatch(path.name):
            named.setdefault(path.name, []).append((job_dir, job))
    return named


def _due(holders: list[tuple[Path, dict]], now: datetime, retention: int) -> str | None:
    reasons = set()
    for _, job in holders:
        if job["state"] in jobs.NOT_ENDED:
            return None
        delivery = job.get("delivery")
        if (
            job["state"] == jobs.FINISHED
            and isinstance(delivery, dict)
            and delivery.get("state") == "delivered"
        ):
            reasons.add("delivered")
        elif (now - datetime.fromisoformat(job["updated_at"])).total_seconds() >= retention:
            reasons.add("retention")
        else:
            return None
    return "retention" if "retention" in reasons else "delivered"


def _unlink_regular(fd: int, name: str) -> bool | None:
    """Delete `name` in the directory `fd` when it is a regular file: True deleted, None
    already gone, False not a regular file and left alone."""
    try:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            return False
        os.unlink(name, dir_fd=fd)
    except FileNotFoundError:
        return None
    return True
