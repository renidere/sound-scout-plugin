"""The file job store of the analysis queue (decision 0013 § 3.6): submit, status, result, cancel.

CLI (`sound-scout jobs`) and MCP call these functions and nothing else: all job logic is here.

Layout: `<data_root>/jobs/<job_id>/` (0700) holds `job.json` (0600) with the request, the
resolved plan, the binding and the job state; `cancel.json` is the cancel marker of a running
job, which the runner turns into its `CancelToken`. `<data_root>/jobs/by-request/<sha256(
request_id)>` names the job of a request, published by a hard link like `runs.open_run`: the
same request returns the same job, the same `request_id` with another identity raises
`runs.RequestMismatch`. The job identity is the SHA-256 of the input, the request (layers,
task, intent), the plan and the binding; the run identity of `runs` stays as it is.

`submit` checks synchronously only what is cheap and decides admission: the request fields, the
path policy (`paths.resolve_input`, `paths.open_input`), the size against
`budgets.max_input_bytes` and the SHA-256 of the input, which the identity needs. It never runs
ffprobe or a stage: the input check is the first stage the runner runs. Admission refuses a
submit that would put more than `jobs.max_queued_jobs` jobs in `queued` (`queue_full`), or more
than `jobs.max_staging_bytes` bytes of input in the jobs not yet ended, `queued` and `running`,
counting the new one (`staging_full`). A repeat of an accepted request is answered with its job
whatever the queue holds.

The job state (`STATES`) is the queue's own: `finished` means the runner stopped working on the
job, not that the result is `complete` or that every requested feature was covered. The result
keeps its own `status` and `coverage` in `result.json` of the run each attempt names
(`attempts[]`, each `{"number", "request_id", "run_id", ...}`, appended by the runner).

Every read-modify-write of the store holds `<data_root>/jobs/jobs.lock` (`flock`, held for
milliseconds): a submit and its admission count, and a state change such as `queued` to
`cancelled`, cannot interleave with another one. `job.json` is replaced atomically.
"""

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sound_scout import __version__, ingest, paths, report, runs, schema
from sound_scout.config import Config

JOBS_DIR_NAME = "jobs"
INDEX_DIR_NAME = "by-request"
JOB_FILE_NAME = "job.json"
CANCEL_MARKER_NAME = "cancel.json"
LOCK_FILE_NAME = "jobs.lock"
QUEUED = "queued"
RUNNING = "running"
FINISHED = "finished"
CANCELLED = "cancelled"
FAILED = "failed"
STATES = (QUEUED, RUNNING, FINISHED, CANCELLED, FAILED)
NOT_ENDED = (QUEUED, RUNNING)
# `app.LAYERS`, `app.LAYER_STAGES` and `app.WORKER_STAGES`, repeated so that a submit does not
# import `app` (it loads Praat and the speech detector) and answers within a second.
LAYERS = ("content", "measurements", "sound")
LAYER_STAGES = {"content": "asr", "measurements": "acoustics", "sound": "audio_review"}
WORKER_STAGES = ("prepare", "acoustics", "asr", "audio_review")
SUBMITTER_CLI = "cli"

INVALID_REQUEST = "invalid_request"
QUEUE_FULL = "queue_full"
STAGING_FULL = "staging_full"
JOB_NOT_FOUND = "job_not_found"
NO_RESULT = "no_result"


class JobError(Exception):
    """A refused job request; `code` is the reason code CLI and MCP report. Nothing changed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# The other refusals of this module with their reason codes; `ingest.IngestError` has its own.
REFUSALS = (
    (paths.PathDenied, "path_denied"),
    (runs.RequestMismatch, "request_mismatch"),
    (runs.RequestIndexCorrupt, "request_index_corrupt"),
)


@dataclass(frozen=True)
class SubmitRequest:
    path: str
    request_id: str | None = None
    layers: tuple[str, ...] = LAYERS
    task: str | None = None
    intent: str | None = None


@dataclass(frozen=True)
class Binding:
    """Where the job came from and where its answer goes; stored as given, nothing acts on it."""

    submitter: str = SUBMITTER_CLI
    connection: str | None = None
    profile: str | None = None
    session_id: str | None = None


def submit(request: SubmitRequest, binding: Binding, config: Config) -> dict:
    """Queue one analysis and answer `{job_id, request_id, state, reused}`; see the docstring."""
    request_id = runs.new_run_id() if request.request_id is None else request.request_id
    _check_identifier("request_id", request_id)
    layers = _checked_layers(request.layers)
    for name in ("task", "intent"):
        value = getattr(request, name)
        if value is not None and not isinstance(value, str):
            raise JobError(INVALID_REQUEST, f"{name} must be a string, got {type(value).__name__}")
    _check_identifier("submitter", binding.submitter)
    for name in ("connection", "profile", "session_id"):
        if getattr(binding, name) is not None:
            _check_identifier(name, getattr(binding, name))
    resolved, size, sha256 = _hashed_input(request.path, config)
    plan = resolved_plan(layers, config)
    analysis = {"layers": sorted(layers), "task": request.task, "intent": request.intent}
    bound = dataclasses.asdict(binding)
    digest = _identity_hash(sha256, analysis, plan, bound)
    with _store(config.data_root) as (jobs_dir, index_dir):
        index_path = index_dir / hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        if index_path.exists():
            return _answer(_indexed_job(jobs_dir, index_path, request_id, digest), reused=True)
        _admit(jobs_dir, size, config)
        job_id = runs.new_run_id()
        job_dir = jobs_dir / job_id
        runs._create_private_dir(job_dir)
        now = _now()
        job = {
            "job_id": job_id,
            "request_id": request_id,
            "identity_hash": digest,
            "package_version": __version__,
            "submitted_at": now,
            "updated_at": now,
            "request": {
                "input": {"path": str(resolved), "size_bytes": size, "sha256": sha256},
                **analysis,
            },
            "plan": plan,
            "binding": bound,
            "state": QUEUED,
            "state_reason": None,
            "attempts": [],
        }
        try:
            _save(job_dir, job)
            published = runs._publish_index_entry(index_dir, index_path, job_id)
        except BaseException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        if not published:
            shutil.rmtree(job_dir)
            return _answer(_indexed_job(jobs_dir, index_path, request_id, digest), reused=True)
        return _answer(job, reused=False)


def resolved_plan(layers: tuple[str, ...], config: Config) -> dict:
    """The fixed plan of today's `app.analyze`: its stages and the operator's budgets.

    A plain dict, so modes can later narrow it; a plan never widens roots, network, limits or
    the backend of the operator configuration.
    """
    wanted = {"prepare", *(LAYER_STAGES[layer] for layer in layers)}
    return {
        "stages": [stage for stage in WORKER_STAGES if stage in wanted],
        "budgets": dataclasses.asdict(config.budgets),
        "backend": config.backend,
    }


def status(job_id: str, config: Config) -> dict:
    """The job's state without its private input path: state, times, attempts, cancel marker."""
    job_dir = _job_dir(job_id, config)
    return _view(job_dir, _load(job_dir))


def result(job_id: str, config: Config) -> dict:
    """The compact answer of the run of the latest attempt that has a result, next to the job
    state and that attempt's number.

    A newer attempt still without `result.json` (running, or ended without a result) does not
    hide the result of an earlier one; `attempt` and `run_dir` name which one answers.
    `job_state` is the queue's, `status` and `coverage` are the result's own. A job with no run
    result yet is `no_result`, whatever its state.
    """
    job_dir = _job_dir(job_id, config)
    job = _load(job_dir)
    for attempt in reversed(job["attempts"]):
        run_id = attempt.get("run_id")
        if not isinstance(run_id, str) or not runs.RUN_ID_PATTERN.fullmatch(run_id):
            continue
        run_dir = config.data_root / runs.RUNS_DIR_NAME / run_id
        result_path = run_dir / report.RESULT_FILE_NAME
        if not result_path.is_file():
            continue
        document = json.loads(result_path.read_text(encoding="utf-8"))
        run_rel = run_dir.relative_to(config.data_root).as_posix()
        head = {"job_id": job_id, "job_state": job["state"], "attempt": attempt.get("number")}
        return report.compact(document, run_dir=run_rel, head=head)
    raise JobError(NO_RESULT, f"job {job_id} is {job['state']} and has no run result yet")


def cancel(job_id: str, config: Config) -> dict:
    """Cancel a job: `queued` becomes `cancelled` at once; a `running` job gets the cancel marker
    and stays `running` until the runner stops it; an ended job is left as it is.

    The answer is the job's status with `cancel`: `cancelled`, `requested` or `already_ended`.
    """
    with _store(config.data_root):
        job_dir = _job_dir(job_id, config)
        job = _load(job_dir)
        if job["state"] == QUEUED:
            job.update(
                state=CANCELLED, state_reason="cancelled before it started", updated_at=_now()
            )
            _save(job_dir, job)
            outcome = CANCELLED
        elif job["state"] == RUNNING:
            with contextlib.suppress(FileExistsError):
                runs._write_private_json(job_dir / CANCEL_MARKER_NAME, {"requested_at": _now()})
            outcome = "requested"
        else:
            outcome = "already_ended"
        return _view(job_dir, job) | {"cancel": outcome}


def _hashed_input(path: str, config: Config) -> tuple[Path, int, str]:
    if not isinstance(path, str) or not path:
        raise JobError(INVALID_REQUEST, "path must be a non-empty string")
    resolved = paths.resolve_input(path, config)
    with paths.open_input(resolved, config) as file:
        size = os.fstat(file.fileno()).st_size
        limit = config.budgets.max_input_bytes
        if size > limit:
            raise ingest.IngestError(ingest.INPUT_TOO_LARGE, f"{size} bytes, the limit is {limit}")
        return resolved, size, ingest._sha256(file)


def _admit(jobs_dir: Path, size: int, config: Config) -> None:
    queued, staged = 0, size
    for job_dir in jobs_dir.iterdir():
        # A directory without job.json is a submit that died before its rollback: no job.
        if (
            not runs.RUN_ID_PATTERN.fullmatch(job_dir.name)
            or not (job_dir / JOB_FILE_NAME).exists()
        ):
            continue
        job = _load(job_dir)
        if job["state"] == QUEUED:
            queued += 1
        if job["state"] in NOT_ENDED:
            staged += job["request"]["input"]["size_bytes"]
    limits = config.jobs
    if queued >= limits.max_queued_jobs:
        raise JobError(
            QUEUE_FULL,
            f"{queued} jobs are queued, the limit is jobs.max_queued_jobs = "
            f"{limits.max_queued_jobs}; submit again after some of them start or end",
        )
    if staged > limits.max_staging_bytes:
        raise JobError(
            STAGING_FULL,
            f"queued and running jobs with this input of {size} bytes would hold {staged} "
            f"bytes, the limit is jobs.max_staging_bytes = {limits.max_staging_bytes}",
        )


def _indexed_job(jobs_dir: Path, index_path: Path, request_id: str, digest: str) -> dict:
    job_id = index_path.read_text(encoding="utf-8").strip()
    if not runs.RUN_ID_PATTERN.fullmatch(job_id):
        raise runs.RequestIndexCorrupt(f"index entry {index_path.name} does not name a job")
    job_dir = jobs_dir / job_id
    try:
        runs._require_private_dir(job_dir)
    except FileNotFoundError:
        raise runs.RequestIndexCorrupt(
            f"index entry {index_path.name} names job {job_id}, but its directory is missing"
        ) from None
    job = _load(job_dir)
    if job["identity_hash"] != digest:
        raise runs.RequestMismatch(
            f"request {request_id!r} is already bound to job {job_id} with another identity"
        )
    return job


def _identity_hash(sha256: str, analysis: dict, plan: dict, binding: dict) -> str:
    identity = {"audio": sha256, "analysis": analysis, "plan": plan, "binding": binding}
    text = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checked_layers(layers: object) -> tuple[str, ...]:
    if isinstance(layers, str) or not isinstance(layers, tuple | list) or not layers:
        raise JobError(
            INVALID_REQUEST, f"layers must be a non-empty list of {list(LAYERS)}, got {layers!r}"
        )
    if any(layer not in LAYERS for layer in layers) or len(set(layers)) != len(layers):
        raise JobError(
            INVALID_REQUEST, f"layers must be distinct items of {list(LAYERS)}, got {list(layers)}"
        )
    return tuple(layers)


def _check_identifier(name: str, value: object) -> None:
    """The result contract's `identifier_string` bounds, like `request_id` of `app.analyze`."""
    bounds = schema.SCHEMA["$defs"]["identifier_string"]
    if not isinstance(value, str) or not bounds["minLength"] <= len(value) <= bounds["maxLength"]:
        raise JobError(
            INVALID_REQUEST,
            f"{name} must be a string of {bounds['minLength']}..{bounds['maxLength']} "
            f"characters, got {type(value).__name__} of length "
            f"{len(value) if isinstance(value, str) else None}",
        )


def _job_dir(job_id: object, config: Config) -> Path:
    if not isinstance(job_id, str) or not runs.RUN_ID_PATTERN.fullmatch(job_id):
        raise JobError(JOB_NOT_FOUND, f"not a job id: {job_id!r}")
    jobs_dir = config.data_root / JOBS_DIR_NAME
    job_dir = jobs_dir / job_id
    try:
        runs._require_private_dir(jobs_dir)
        runs._require_private_dir(job_dir)
    except FileNotFoundError:
        raise JobError(JOB_NOT_FOUND, f"no job {job_id} under the data root") from None
    return job_dir


def _view(job_dir: Path, job: dict) -> dict:
    run_ids = [attempt.get("run_id") for attempt in job["attempts"]]
    return {
        "job_id": job["job_id"],
        "request_id": job["request_id"],
        "state": job["state"],
        "state_reason": job["state_reason"],
        "submitted_at": job["submitted_at"],
        "updated_at": job["updated_at"],
        "stages": job["plan"]["stages"],
        "attempts": len(job["attempts"]),
        "run_id": run_ids[-1] if run_ids else None,
        "cancel_requested": (job_dir / CANCEL_MARKER_NAME).exists(),
    }


def _answer(job: dict, *, reused: bool) -> dict:
    return {
        "job_id": job["job_id"],
        "request_id": job["request_id"],
        "state": job["state"],
        "reused": reused,
    }


@contextlib.contextmanager
def _store(data_root: Path) -> Iterator[tuple[Path, Path]]:
    """The private jobs and index directories, under the store lock for the whole block."""
    jobs_dir = data_root / JOBS_DIR_NAME
    index_dir = jobs_dir / INDEX_DIR_NAME
    runs._ensure_private_dir(jobs_dir)
    runs._ensure_private_dir(index_dir)
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(jobs_dir / LOCK_FILE_NAME, flags, runs.FILE_MODE)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield jobs_dir, index_dir
    finally:
        os.close(fd)


def _load(job_dir: Path) -> dict:
    return runs._read_json(job_dir / JOB_FILE_NAME)


def _save(job_dir: Path, job: dict) -> None:
    staging = job_dir / f".{JOB_FILE_NAME}.{runs.new_run_id()}"
    runs._write_private_json(staging, job)
    os.replace(staging, job_dir / JOB_FILE_NAME)


def _now() -> str:
    return datetime.now(UTC).isoformat()
