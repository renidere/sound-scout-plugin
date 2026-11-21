"""The file job store of the analysis queue (decision 0013 § 3.6): submit, status, result, cancel.

CLI (`sound-scout jobs`) and MCP call these functions and nothing else: all job logic is here.
`limits` answers the budgets and `jobs.staging_root` to the Hermes plugin, a client of the CLI.

Layout: `<data_root>/jobs/<job_id>/` (0700) holds `job.json` (0600) with the request, the
resolved plan, the binding and the job state; `cancel.json` is the cancel marker of a running
job, which the runner turns into its `CancelToken`. `<data_root>/jobs/by-request/<sha256(
request_id)>` names the job of a request, published by a hard link like `runs.open_run`: the
same request returns the same job, the same `request_id` with another identity raises
`runs.RequestMismatch`. The job identity is the SHA-256 of the input, the request (mode, layers,
task, intent), the plan and the binding; the run identity of `runs` stays as it is.

The request's `mode` (`MODES`, default `auto`) is resolved at submit by `resolved_plan` into the
layers and the plan stored in `job.json` with `capability_version` (`CAPABILITY_VERSION`, the
version of the table mode -> layers -> stages and of the plan fields): `quick` is content and
measurements, `full` all three layers, `targeted` the layers the caller named, `auto` is
`targeted` when layers are named and `full` otherwise. The plan holds the stages, the operator's
budgets and backend at submit, nothing else: no root, network, data root or model. The runner
does not resolve the plan again: `applied_plan` checks the stored plan against the configuration
of the moment and refuses (`plan_refused`) a plan with another key, another backend, another
capability version or stages of no layer set; budgets are the smaller of the plan and the
configuration, so a plan only narrows. `full` runs only the configured backend.

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

`run_runner` is the one runner: it holds `<data_root>/jobs/runner.lock` (`flock`, no waiting: a
second runner is `runner_running`) for its whole life, not `analyze.lock`, which `app.analyze`
takes itself. It takes the oldest `queued` job (by `queued_at`), moves it to `running` with a
new attempt `{"number": n, "request_id": "<job_id>-a<n>", "lease": ...}` and calls `app.analyze`
unchanged with that `request_id`, so a run left incomplete never blocks the next attempt. The
lease names the runner process: `boot_id`, `pid`, its start time in clock ticks since boot
(`/proc/<pid>/stat`, against a reused pid) and the attempt number. `Busy` from `app.analyze` (a
synchronous MCP call holds `analyze.lock`) drops that attempt, which ran nothing, and puts the
job back to `queued` with its place in the queue; the runner tries again after `poll_seconds`.
While an attempt runs, a watcher thread turns the job's `cancel.json` into the `CancelToken` of
`app.analyze`, and the job ends `cancelled`. A refusal of `app.analyze` or an unexpected error
ends the job `failed` with its reason code; the runner goes on with the next job. A run that
recorded another input SHA-256 than the one submitted (`source` of its result, `source.json`)
ends the job `failed` with `input_changed`: it analysed other audio than the job names, so its
attempt is marked `input_changed` and `result` never answers with it (R02).

`recover` takes a `running` job whose lease owner is gone (another boot, no such pid, a zombie,
or a pid with another start time): its attempt is marked `interrupted` and the job goes back to
`queued`, where the runner takes it as attempt n+1; with a cancel marker it ends `cancelled`.
`retry` puts a `failed` or `cancelled` job (an interrupted attempt that `recover` ended
`cancelled` too) back to `queued` at the end of the queue, with the same `job_id` and
`request_id`; the runner takes it as attempt n+1. Only a client calls `retry` or `recover`:
nothing retries on its own.

The runner keeps the code it loaded at its start, while each attempt's worker (`app.analyze`,
`spawn`) loads the code installed on disk at that moment: after an update the two would differ,
and the old runner would check a new worker's result with its old schema (#139). So the runner
takes `doctor.code_fingerprint` of the installed package when it starts, and `_claim_next` takes it
again, under `jobs.lock`, before it moves a job to `running`. When the two differ (or the
package cannot be read), the runner starts no attempt: the job stays `queued` untouched, no
attempt and no lease, the runner logs why on stderr, releases `runner.lock` and returns, so
`jobs run` exits 0. A runner that cannot read the package at its start logs that and returns
too, before it takes `runner.lock`. An update after the claim and before the attempt's worker
starts is caught by the worker itself (`app.CodeChanged`, #141): the attempt and its lease are
removed as for `Busy`, `state_reason` names the code change, and the runner returns the same way.
A runner started after that (`jobs start`) loads the new code and runs the job.
`runner_unit_argv` is the `systemd-run --user` command that starts the runner as the transient
unit `RUNNER_UNIT` outside the gateway's cgroup (the gateway unit has `KillMode=mixed`, so its
children die on each restart); `start_runner_unit` runs it.

`delivery` of `job.json` (`null` at submit) is the state of the job's final note to its chat,
kept by `sound_scout.delivery` under the same `jobs.lock`; `status` shows it. About once a minute
the runner deletes the staging copies that are due (`sound_scout.staging.sweep`) and marks their
jobs `staging_removed_at`.

A control attempt (Decision in #110) is a job submitted with `previous_run_id` and `criterion`,
always together. `submit` checks them with the functions of `compare`: the criterion by
`comparison.parse_criterion`, the initial run by `comparison.load_attempt` (complete, its
`result.json` valid), and refuses with the same codes (`criterion_invalid`, `run_not_found`,
`run_incomplete`, `result_invalid`). A criterion the plan cannot answer is `invalid_request`: the
verdict needs the `sound` layer, a criterion with a word the `content` layer too. The link is
stored as `request.control` and counts in the job identity; the plan does not change. After
`app.analyze` the runner calls `app.compare(previous_run_id, <run of the attempt>, criterion)`
and records its summary as `compare` of the attempt in the same write that ends the attempt, so
a `finished` control job always carries the verdict of its last attempt (or why there is none:
`refused`, `failed`, `not_run`). The full document stays `<data_root>/compare/<compare_id>.json`;
the comparison never changes the job state. Each attempt, after a retry or a recover too, is
compared again with the same initial run and criterion.
"""

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sound_scout import (
    __version__,
    comparison,
    doctor,
    ingest,
    paths,
    proc,
    report,
    runs,
    schema,
)
from sound_scout.config import CONFIG_PATH_ENV, Budgets, Config

JOBS_DIR_NAME = "jobs"
INDEX_DIR_NAME = "by-request"
JOB_FILE_NAME = "job.json"
CANCEL_MARKER_NAME = "cancel.json"
LOCK_FILE_NAME = "jobs.lock"
RUNNER_LOCK_NAME = "runner.lock"
RUNNER_UNIT = "sound-scout-jobs-runner"
# Passed on to the transient unit when set: the decoder on PATH and the ASR model cache.
RUNNER_ENV = ("PATH", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME")
SYSTEMD_RUN_TIMEOUT_SECONDS = 30.0
SYSTEMD_RUN_MAX_OUTPUT_BYTES = 64 * 1024
POLL_SECONDS = 1.0
CANCEL_POLL_SECONDS = 0.1
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
DEAD_PROCESS_STATES = ("Z", "X", "x")
QUEUED = "queued"
RUNNING = "running"
FINISHED = "finished"
CANCELLED = "cancelled"
FAILED = "failed"
STATES = (QUEUED, RUNNING, FINISHED, CANCELLED, FAILED)
NOT_ENDED = (QUEUED, RUNNING)
INTERRUPTED = "interrupted"
# `app.LAYERS`, `app.LAYER_STAGES` and `app.WORKER_STAGES`, repeated so that a submit does not
# import `app` (it loads Praat and the speech detector) and answers within a second.
LAYERS = ("content", "measurements", "sound")
LAYER_STAGES = {"content": "asr", "measurements": "acoustics", "sound": "audio_review"}
WORKER_STAGES = ("prepare", "acoustics", "asr", "audio_review")
# `app.STAGE_RECORD`, `app.CURRENT_STAGE` and `app.CHECKPOINT`, repeated for the same reason:
# `status` reads the stage records and the marker of the run: the finished and current stages.
STAGE_RECORD = "stage-{name}.json"
CURRENT_STAGE = "stage-current.json"
CHECKPOINT = "checkpoint"
SUBMITTER_CLI = "cli"

INVALID_REQUEST = "invalid_request"
QUEUE_FULL = "queue_full"
STAGING_FULL = "staging_full"
JOB_NOT_FOUND = "job_not_found"
NO_RESULT = "no_result"
RUNNER_RUNNING = "runner_running"
RUNNER_START_FAILED = "runner_start_failed"
NOT_RETRYABLE = "not_retryable"
PLAN_REFUSED = "plan_refused"
# Set by `staging.sweep` on a job whose staging copy it removed.
STAGING_REMOVED_AT = "staging_removed_at"

MODES = ("quick", "targeted", "full", "auto")
DEFAULT_MODE = "auto"
CAPABILITY_VERSION = 1
MODE_LAYERS = {"quick": ("content", "measurements"), "full": LAYERS}
PLAN_KEYS = frozenset(
    {"capability_version", "mode", "resolved_mode", "resolution", "stages", "budgets", "backend"}
)
RETRYABLE = (FAILED, CANCELLED)


class JobError(Exception):
    """A refused job request; `code` is the reason code CLI and MCP report. Nothing changed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


UNREADABLE_CODE = "cannot read the installed sound_scout package"
CHANGED_CODE = "the installed sound_scout code changed since this runner started"
CODE_CHANGED_ATTEMPT = (
    "the installed sound_scout code changed before the attempt's worker started; the job waits "
    "for a new runner (jobs start)"
)
BUSY_REASON = "analyze.lock is held by another analysis; the job waits in the queue"


class _CodeChanged(Exception):
    """The installed code is not the code the runner loaded, or cannot be read (`reason`);
    the job named stays `queued` (or ends `cancelled`, its `state`, when a cancel came first)."""

    def __init__(self, job_id: str, reason: str, state: str = "queued") -> None:
        super().__init__(job_id, reason)
        self.job_id = job_id
        self.reason = reason
        self.state = state


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
    layers: tuple[str, ...] | None = None
    task: str | None = None
    intent: str | None = None
    mode: str = DEFAULT_MODE
    previous_run_id: str | None = None
    criterion: str | None = None


@dataclass(frozen=True)
class Binding:
    """Where the job came from and where its answer goes; `profile` and `session_id` address the
    final note of `sound_scout.delivery`."""

    submitter: str = SUBMITTER_CLI
    connection: str | None = None
    profile: str | None = None
    session_id: str | None = None


def submit(request: SubmitRequest, binding: Binding, config: Config) -> dict:
    """Queue one analysis and answer `{job_id, request_id, state, reused}`; see the docstring."""
    request_id = runs.new_run_id() if request.request_id is None else request.request_id
    _check_identifier("request_id", request_id)
    layers, plan = resolved_plan(request.mode, request.layers, config)
    for name in ("task", "intent"):
        value = getattr(request, name)
        if value is not None and not isinstance(value, str):
            raise JobError(INVALID_REQUEST, f"{name} must be a string, got {type(value).__name__}")
    control = _checked_control(request, layers, config)
    _check_identifier("submitter", binding.submitter)
    for name in ("connection", "profile", "session_id"):
        if getattr(binding, name) is not None:
            _check_identifier(name, getattr(binding, name))
    resolved, size, sha256 = _hashed_input(request.path, config)
    analysis = {
        "mode": request.mode,
        "layers": sorted(layers),
        "task": request.task,
        "intent": request.intent,
        "control": control,
    }
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
            "queued_at": now,
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
            "delivery": None,
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


def _checked_control(
    request: SubmitRequest, layers: tuple[str, ...], config: Config
) -> dict | None:
    """The link of a control attempt to its initial run, checked as `compare` checks it; `None`
    for an ordinary job. See the module docstring."""
    if request.previous_run_id is None and request.criterion is None:
        return None
    if request.previous_run_id is None or request.criterion is None:
        raise JobError(INVALID_REQUEST, "previous_run_id and criterion go together")
    try:
        criterion = comparison.parse_criterion(request.criterion)
        if "sound" not in layers:
            raise JobError(
                INVALID_REQUEST,
                "a control attempt needs the sound layer: the verdict comes only from audio review",
            )
        if criterion.word is not None and "content" not in layers:
            raise JobError(
                INVALID_REQUEST,
                f"a control attempt by {criterion.kind} needs the content layer: the word is "
                "looked up in the transcript",
            )
        comparison.load_attempt(request.previous_run_id, config)
    except comparison.CompareError as error:
        raise JobError(error.code, error.message) from None
    return {"previous_run_id": request.previous_run_id, "criterion": criterion.raw}


def resolved_plan(mode: object, layers: object, config: Config) -> tuple[tuple[str, ...], dict]:
    """The layers and the plan of a request's mode; see the module docstring. `invalid_request`
    for an unknown mode, layers with `quick` or `full`, or `targeted` without layers."""
    if not isinstance(mode, str) or mode not in MODES:
        raise JobError(INVALID_REQUEST, f"mode must be one of {list(MODES)}, got {mode!r}")
    if mode in MODE_LAYERS and layers is not None:
        raise JobError(INVALID_REQUEST, f"layers only with targeted or auto, not with {mode}")
    if mode == "targeted" and layers is None:
        raise JobError(INVALID_REQUEST, "mode targeted needs the layers to analyze")
    resolved, resolution = mode, None
    if mode == "auto":
        resolved = "full" if layers is None else "targeted"
        resolution = "auto: no layers named" if layers is None else "auto: layers named"
    chosen = MODE_LAYERS[resolved] if resolved in MODE_LAYERS else _checked_layers(layers)
    plan = {
        "capability_version": CAPABILITY_VERSION,
        "mode": mode,
        "resolved_mode": resolved,
        "resolution": resolution,
        "stages": _stages(chosen),
        "budgets": dataclasses.asdict(config.budgets),
        "backend": config.backend,
    }
    return tuple(layer for layer in LAYERS if layer in chosen), plan


def applied_plan(plan: object, config: Config) -> tuple[tuple[str, ...], Config]:
    """The layers of a stored plan and the configuration its attempt runs with: `config` with
    each budget the smaller of the plan's and the configuration's. Roots, network, data root,
    ASR and backend always come from `config`. `plan_refused` for a plan that would widen it or
    that this version does not know; see the module docstring."""
    if not isinstance(plan, dict):
        raise JobError(PLAN_REFUSED, f"the plan is not an object: {type(plan).__name__}")
    version = plan.get("capability_version")
    if type(version) is not int or version != CAPABILITY_VERSION:
        raise JobError(
            PLAN_REFUSED,
            f"plan of capability version {version!r}, this version runs {CAPABILITY_VERSION}; "
            "submit the recording again",
        )
    extra, missing = sorted(set(plan) - PLAN_KEYS), sorted(PLAN_KEYS - set(plan))
    if extra or missing:
        raise JobError(PLAN_REFUSED, f"plan keys: unknown {extra}, missing {missing}")
    if plan["mode"] not in MODES or plan["resolved_mode"] not in ("targeted", *MODE_LAYERS):
        raise JobError(
            PLAN_REFUSED, f"plan mode {plan['mode']!r} resolved as {plan['resolved_mode']!r}"
        )
    stages = plan["stages"]
    layers = tuple(
        layer for layer in LAYERS if isinstance(stages, list) and LAYER_STAGES[layer] in stages
    )
    expected = MODE_LAYERS.get(plan["resolved_mode"], layers)
    if not layers or stages != _stages(layers) or layers != expected:
        raise JobError(
            PLAN_REFUSED,
            f"plan stages {stages!r} are not the stages of its {plan['resolved_mode']} layers",
        )
    if plan["backend"] != config.backend:
        raise JobError(
            PLAN_REFUSED,
            f"plan backend {plan['backend']!r} is not the configured backend {config.backend!r}",
        )
    budgets, fields = plan["budgets"], [field.name for field in dataclasses.fields(Budgets)]
    if not isinstance(budgets, dict) or sorted(budgets) != sorted(fields):
        raise JobError(PLAN_REFUSED, f"plan budgets must have exactly the fields {fields}")
    for name, value in budgets.items():
        if type(value) is not int or value < 1:
            raise JobError(PLAN_REFUSED, f"plan budget {name} must be an integer >= 1")
    narrowed = {name: min(budgets[name], getattr(config.budgets, name)) for name in fields}
    return layers, dataclasses.replace(config, budgets=Budgets(**narrowed))


def _stages(layers: tuple[str, ...]) -> list[str]:
    wanted = {"prepare", *(LAYER_STAGES[layer] for layer in layers)}
    return [stage for stage in WORKER_STAGES if stage in wanted]


def status(job_id: str, config: Config) -> dict:
    """The job's state without its private input path: state, times, attempts, cancel marker,
    `current_stage` and `finished_stages` of the latest attempt's run.

    `current_stage` is `{stage, started_at}` of the run's `work/stage-current.json` only while
    the job is `running` and that stage has no final record yet; otherwise `None` (between two
    stages, before the first one, or once the job ended: the marker stays in the run and names
    the last stage started, not one that runs)."""
    job_dir = _job_dir(job_id, config)
    return _view(job_dir, _load(job_dir), config)


def limits(config: Config) -> dict:
    """The operator's budgets and queue limits a client checks before it uploads, and the
    staging root the Hermes plugin writes uploads to (`None`: the plugin takes none)."""
    staging_root = config.jobs.staging_root
    return {
        "max_duration_seconds": config.budgets.max_duration_seconds,
        "max_input_bytes": config.budgets.max_input_bytes,
        "max_queued_jobs": config.jobs.max_queued_jobs,
        "max_staging_bytes": config.jobs.max_staging_bytes,
        "staging_root": None if staging_root is None else str(staging_root),
    }


def result(job_id: str, config: Config) -> dict:
    """The compact answer of the run of the latest attempt that has a result, next to the job
    state and that attempt's number.

    A newer attempt still without `result.json` (running, or ended without a result) does not
    hide the result of an earlier one; `attempt` and `run_dir` name which one answers. A run that
    analysed other audio than the submitted SHA-256 never answers: an `input_changed` attempt, or
    one whose `result.json` records another `source` (a recovered attempt is not checked by the
    runner).
    `job_state` is the queue's, `status` and `coverage` are the result's own. A job with no run
    result yet is `no_result`, whatever its state.
    """
    job_dir = _job_dir(job_id, config)
    job = _load(job_dir)
    for attempt in reversed(job["attempts"]):
        if attempt.get("input_changed"):
            continue
        run_id = attempt.get("run_id")
        if not isinstance(run_id, str) or not runs.RUN_ID_PATTERN.fullmatch(run_id):
            continue
        run_dir = config.data_root / runs.RUNS_DIR_NAME / run_id
        result_path = run_dir / report.RESULT_FILE_NAME
        if not result_path.is_file():
            continue
        document = json.loads(result_path.read_text(encoding="utf-8"))
        if document.get("source", {}).get("sha256") != job["request"]["input"]["sha256"]:
            continue
        run_rel = run_dir.relative_to(config.data_root).as_posix()
        head = {"job_id": job_id, "job_state": job["state"], "attempt": attempt.get("number")}
        if job["request"].get("control"):
            head["compare"] = attempt.get("compare")
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
        return _view(job_dir, job, config) | {"cancel": outcome}


def retry(job_id: str, config: Config) -> dict:
    """Queue a `failed` or `cancelled` job again, at the end of the queue, through admission;
    the runner takes it as its next attempt `<job_id>-a<n+1>`. The job keeps its `job_id`,
    `request_id`, request, plan, binding and delivery; `retries` keeps how it ended before.

    The answer is the job's status with `retry: queued` and `next_attempt`. `not_retryable` for
    a job in another state or one whose staging copy `staging.sweep` removed (it has no input
    any more), `queue_full` or `staging_full` like a submit."""
    with _store(config.data_root) as (jobs_dir, _):
        job_dir = _job_dir(job_id, config)
        job = _load(job_dir)
        if job["state"] not in RETRYABLE:
            hint = (
                "; if its runner is gone, run jobs recover first" if job["state"] == RUNNING else ""
            )
            raise JobError(
                NOT_RETRYABLE,
                f"job {job_id} is {job['state']}; only a failed or cancelled job is retried{hint}",
            )
        if job.get(STAGING_REMOVED_AT):
            raise JobError(
                NOT_RETRYABLE,
                f"job {job_id} has no input: its staging copy was removed at "
                f"{job[STAGING_REMOVED_AT]} after the retention period; submit the recording again",
            )
        _admit(jobs_dir, job["request"]["input"]["size_bytes"], config)
        (job_dir / CANCEL_MARKER_NAME).unlink(missing_ok=True)
        number = len(job["attempts"]) + 1
        next_attempt = f"{job_id}-a{number}"
        now = _now()
        job.setdefault("retries", []).append(
            {
                "requested_at": now,
                "after_state": job["state"],
                "after_reason": job["state_reason"],
                "next_attempt": next_attempt,
            }
        )
        job.update(
            state=QUEUED,
            state_reason=f"retry requested: attempt {number} ({next_attempt})",
            queued_at=now,
            updated_at=now,
        )
        _save(job_dir, job)
        return _view(job_dir, job, config) | {"retry": QUEUED, "next_attempt": next_attempt}


def run_runner(
    config: Config, *, exit_when_idle: bool = False, poll_seconds: float = POLL_SECONDS
) -> None:
    """Run queued jobs one at a time, oldest first, until stopped; see the module docstring.

    With `exit_when_idle` it returns once no job is `queued`. Raises `JobError`
    (`runner_running`) when another runner holds `runner.lock`.
    """
    from sound_scout import app, staging

    try:
        loaded_code = doctor.code_fingerprint()
    except OSError as error:
        _log(f"{UNREADABLE_CODE} ({type(error).__name__}); exiting without starting an attempt")
        return
    swept_at = None
    with _runner_lock(config.data_root):
        while True:
            if swept_at is None or time.monotonic() - swept_at >= staging.SWEEP_SECONDS:
                swept_at = time.monotonic()
                try:
                    for item in staging.sweep(config):
                        _log(f"staging {item['name']} removed: {item['reason']}")
                except (OSError, ValueError) as error:
                    _log(f"staging sweep failed: {type(error).__name__}")
            try:
                claimed = _claim_next(config, loaded_code)
                if claimed is None:
                    if exit_when_idle:
                        return
                    time.sleep(poll_seconds)
                    continue
                job, attempt = claimed
                outcome = _run_attempt(app, job, attempt, config)
            except _CodeChanged as changed:
                ended = (
                    f"job {changed.job_id} cancelled"
                    if changed.state == CANCELLED
                    else f"job {changed.job_id} stays queued"
                )
                tail = (
                    "start a new runner (jobs start) to run it" if changed.state == QUEUED else ""
                )
                _log(f"{changed.reason}; {ended}; exiting" + (f": {tail}" if tail else ""))
                return
            _log(f"job {job['job_id']} attempt {attempt['number']}: {outcome}")
            if outcome == QUEUED:
                time.sleep(poll_seconds)


def recover(config: Config) -> list[dict]:
    """Mark the attempts of `running` jobs whose lease owner is gone `interrupted` and queue the
    job again (or end it `cancelled` when a cancel was requested); one entry per recovered job:
    `{job_id, interrupted_attempt, state, next_attempt}`. A live lease is left alone."""
    recovered = []
    with _store(config.data_root) as (jobs_dir, _):
        for job_dir in _job_dirs(jobs_dir):
            job = _load(job_dir)
            if job["state"] != RUNNING or not job["attempts"]:
                continue
            attempt = job["attempts"][-1]
            if _lease_alive(attempt.get("lease")):
                continue
            number = attempt["number"]
            attempt.update(
                outcome=INTERRUPTED,
                ended_at=_now(),
                run_id=_run_id_of(attempt["request_id"], config),
            )
            if (job_dir / CANCEL_MARKER_NAME).exists():
                state = CANCELLED
                reason = f"attempt {number} was interrupted (its runner is gone) after a cancel"
                next_attempt = None
            else:
                state = QUEUED
                next_attempt = f"{job['job_id']}-a{number + 1}"
                reason = (
                    f"attempt {number} was interrupted (its runner is gone); jobs recover queued "
                    f"attempt {number + 1} ({next_attempt})"
                )
            job.update(state=state, state_reason=reason, updated_at=_now())
            _save(job_dir, job)
            recovered.append(
                {
                    "job_id": job["job_id"],
                    "interrupted_attempt": number,
                    "state": state,
                    "next_attempt": next_attempt,
                }
            )
    return recovered


def runner_unit_argv(
    config_path: Path, environ: Mapping[str, str], *, python: str = sys.executable
) -> list[str]:
    """The `systemd-run --user` command that starts the runner as the transient unit `RUNNER_UNIT`.

    The unit gets the absolute configuration file and the `RUNNER_ENV` variables set in
    `environ`, nothing else of the caller's environment; `--collect` frees the fixed name after
    the runner ends, whatever its exit code.
    """
    argv = [
        "systemd-run",
        "--user",
        f"--unit={RUNNER_UNIT}",
        "--collect",
        "--description=Sound Scout jobs runner",
        f"--setenv={CONFIG_PATH_ENV}={Path(config_path).absolute()}",
    ]
    argv += [f"--setenv={name}={environ[name]}" for name in RUNNER_ENV if name in environ]
    return [*argv, python, "-m", "sound_scout", "jobs", "run"]


def start_runner_unit(config_path: Path) -> dict:
    """Start the runner through `systemd-run --user`; `{unit, started}` or `runner_start_failed`
    with what systemd-run said (for example, the unit is already running)."""
    argv = runner_unit_argv(config_path, os.environ)
    try:
        outcome = proc.run(
            argv,
            timeout=SYSTEMD_RUN_TIMEOUT_SECONDS,
            max_output_bytes=SYSTEMD_RUN_MAX_OUTPUT_BYTES,
        )
    except OSError as error:
        raise JobError(RUNNER_START_FAILED, f"cannot run systemd-run: {error.strerror}") from None
    if outcome.timed_out or outcome.returncode != 0:
        said = " ".join(outcome.stderr.decode("utf-8", "replace").split())
        raise JobError(
            RUNNER_START_FAILED,
            f"systemd-run exited {outcome.returncode}"
            + (" (timed out)" if outcome.timed_out else "")
            + (f": {said}" if said else ""),
        )
    return {"unit": RUNNER_UNIT, "started": True}


def _claim_next(config: Config, loaded_code: str) -> tuple[dict, dict] | None:
    """The oldest `queued` job, now `running` with a new attempt and its lease; None if none.
    Raises `_CodeChanged` and changes nothing when the installed code is not `loaded_code`."""
    with _store(config.data_root) as (jobs_dir, _):
        queued = [job for job in map(_load, _job_dirs(jobs_dir)) if job["state"] == QUEUED]
        if not queued:
            return None
        job = min(
            queued,
            key=lambda item: (item.get("queued_at") or item["submitted_at"], item["job_id"]),
        )
        try:
            installed_code = doctor.code_fingerprint()
        except OSError as error:
            raise _CodeChanged(job["job_id"], UNREADABLE_CODE) from error
        if installed_code != loaded_code:
            raise _CodeChanged(job["job_id"], CHANGED_CODE)
        number = len(job["attempts"]) + 1
        attempt = {
            "number": number,
            "request_id": f"{job['job_id']}-a{number}",
            "run_id": None,
            "started_at": _now(),
            "ended_at": None,
            "outcome": None,
            "input_changed": False,
            "lease": _lease(number),
        }
        job["attempts"].append(attempt)
        job.update(state=RUNNING, state_reason=None, updated_at=_now())
        _save(jobs_dir / job["job_id"], job)
        return job, attempt


def _run_attempt(app, job: dict, attempt: dict, config: Config) -> str:
    """Run one attempt through `app.analyze` and record how it ended; the new job state.
    Raises `_CodeChanged` (state `queued`, or `cancelled` after a cancel) once the attempt is
    recorded, when the worker refused changed code (#141)."""
    job_dir = config.data_root / JOBS_DIR_NAME / job["job_id"]
    request = job["request"]
    token = app.CancelToken()
    done = threading.Event()
    watcher = threading.Thread(target=_watch_cancel, args=(job_dir, token, done), daemon=True)
    watcher.start()
    run_id, state, reason, changed, compared = None, FINISHED, None, False, None
    try:
        layers, effective = applied_plan(job["plan"], config)
        analysis = app.analyze(
            app.AnalyzeRequest(
                request["input"]["path"],
                request_id=attempt["request_id"],
                layers=layers,
                task=request["task"],
                intent=request["intent"],
            ),
            effective,
            cancel=token,
        )
        run_id = analysis.run.run_id
        changed = _recorded_sha256s(analysis) != {request["input"]["sha256"]}
        if changed:
            message = "the run read another file than the one submitted (its SHA-256 differs)"
            state, reason = FAILED, f"{ingest.INPUT_CHANGED}: {_private(message, request, config)}"
        elif token.cancelled and not analysis.run.complete:
            state, reason = CANCELLED, "cancelled while running; the run keeps what was done"
        compared = _compared(app, job, run_id, state, config)
    except app.Busy:
        state = QUEUED
    except app.CodeChanged:
        state, reason = QUEUED, CODE_CHANGED_ATTEMPT
    except JobError as error:
        state, reason = FAILED, f"{error.code}: {error.message}"
    except ingest.IngestError as error:
        state, reason = FAILED, f"{error.code}: {_private(error.message, request, config)}"
    except tuple(kind for kind, _ in app.REFUSALS) as error:
        code = next(code for kind, code in app.REFUSALS if isinstance(error, kind))
        state, reason = FAILED, f"{code}: {_private(str(error), request, config)}"
    except Exception as error:
        # Only the type: the message of an unexpected error may carry a path or a transcript.
        state, reason = FAILED, f"internal_error: {type(error).__name__}"
    finally:
        done.set()
        watcher.join()
    if compared is None and state == FAILED and request.get("control"):
        compared = {"state": "not_run", "reason": FAILED}
    ended = _end_attempt(
        job_dir, attempt, state, reason, run_id, config, input_changed=changed, compare=compared
    )
    if reason == CODE_CHANGED_ATTEMPT:
        raise _CodeChanged(job["job_id"], CHANGED_CODE, ended)
    return ended


def _compared(app, job: dict, run_id: str, state: str, config: Config) -> dict | None:
    """The summary of the control attempt's comparison with its initial run, `None` for an
    ordinary job; see the module docstring. The `message` of the reasons stays in the full
    document: it may quote a stage reason."""
    control = job["request"].get("control")
    if not control:
        return None
    if state != FINISHED:
        return {"state": "not_run", "reason": state}
    try:
        compared = app.compare(control["previous_run_id"], run_id, control["criterion"], config)
    except comparison.CompareError as error:
        return {"state": "refused", "code": error.code, "message": error.message}
    except Exception as error:
        return {"state": "failed", "error": f"internal_error: {type(error).__name__}"}
    document = compared.result
    backend = document["conditions_diff"]["backend"]
    return {
        "state": "completed",
        "compare_id": document["compare_id"],
        "path": compared.path.relative_to(config.data_root).as_posix(),
        "criterion": document["criterion"]["raw"],
        "verdict": document["verdict"],
        "reasons": [
            {"code": item["code"], "attempt": item["attempt"]} for item in document["reasons"]
        ],
        "audio_review": {
            label: document["attempts"][label]["audio_review"]["status"] for label in ("a", "b")
        },
        "backend": {label: (backend[label] or {}).get("backend") for label in ("a", "b")},
    }


def _recorded_sha256s(analysis) -> set[object]:
    """Every input SHA-256 the run recorded: its result's `source` and, once the input was
    decoded, `source.json`, which a worker stopped before its stage record still leaves."""
    recorded = {analysis.result.get("source", {}).get("sha256")}
    source_path = analysis.run.path / ingest.SOURCE_FILE_NAME
    if source_path.is_file():
        recorded.add(json.loads(source_path.read_text(encoding="utf-8")).get("sha256"))
    return recorded


def _private(message: str, request: dict, config: Config) -> str:
    """The reason as `status` shows it: the input, the roots and the data root by placeholders."""
    return paths.without_paths(message, request["input"]["path"], config)


def _end_attempt(
    job_dir: Path,
    attempt: dict,
    state: str,
    reason: str | None,
    run_id: str | None,
    config: Config,
    *,
    input_changed: bool = False,
    compare: dict | None = None,
) -> str:
    """Record how the attempt ended, with the summary of its comparison when it has one, and
    answer the job's new state.

    Under `jobs.lock`, so a `cancel` cannot land between the cancel-marker check of a `Busy`
    attempt and its requeue. A `QUEUED` attempt keeps the given reason, `BUSY_REASON` without one.
    """
    with _store(config.data_root):
        job = _load(job_dir)
        current = job["attempts"][-1] if job["attempts"] else None
        if job["state"] != RUNNING or current is None or current["number"] != attempt["number"]:
            return job["state"]  # `jobs recover` took this attempt from a runner it saw as gone.
        if state == QUEUED and (job_dir / CANCEL_MARKER_NAME).exists():
            state = CANCELLED
            if reason is None:
                reason = "cancelled while waiting for analyze.lock"
            else:
                reason = "cancelled before its attempt started"
        if state == QUEUED:
            job["attempts"].pop()
            reason = reason or BUSY_REASON
        else:
            current.update(
                run_id=run_id or _run_id_of(current["request_id"], config),
                ended_at=_now(),
                outcome=state,
                input_changed=input_changed,
            )
            if compare is not None:
                current["compare"] = compare
        job.update(state=state, state_reason=reason, updated_at=_now())
        _save(job_dir, job)
        return state


def _watch_cancel(job_dir: Path, token, done: threading.Event) -> None:
    while not done.wait(CANCEL_POLL_SECONDS):
        if (job_dir / CANCEL_MARKER_NAME).exists():
            token.cancel()
            return


@contextlib.contextmanager
def _runner_lock(data_root: Path) -> Iterator[None]:
    jobs_dir = data_root / JOBS_DIR_NAME
    runs._ensure_private_dir(jobs_dir)
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(jobs_dir / RUNNER_LOCK_NAME, flags, runs.FILE_MODE)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise JobError(
                RUNNER_RUNNING, "another jobs runner holds runner.lock on this data root"
            ) from None
        yield
    finally:
        os.close(fd)


def _lease(number: int) -> dict:
    pid = os.getpid()
    return {
        "boot_id": _boot_id(),
        "pid": pid,
        "process_start": _process_start(pid),
        "attempt": number,
        "started_at": _now(),
    }


def _lease_alive(lease: object) -> bool:
    """Whether the lease's runner still runs: this boot, its pid, the same process start."""
    if not isinstance(lease, dict) or not isinstance(lease.get("pid"), int):
        return False
    if lease.get("boot_id") != _boot_id():
        return False
    started = lease.get("process_start")
    return started is not None and _process_start(lease["pid"]) == started


def _boot_id() -> str | None:
    try:
        return BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except OSError:
        return None


def _process_start(pid: int) -> int | None:
    """Field 22 of `/proc/<pid>/stat`, the start in clock ticks since boot; None without it or
    when the process is a zombie or dead (it exited, only its parent has not reaped it yet)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = stat.rpartition(")")[2].split()
    if fields[0] in DEAD_PROCESS_STATES:
        return None
    return int(fields[19])


def _run_id_of(request_id: str, config: Config) -> str | None:
    index = config.data_root / runs.RUNS_DIR_NAME / runs.INDEX_DIR_NAME
    entry = index / hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    try:
        run_id = entry.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return run_id if runs.RUN_ID_PATTERN.fullmatch(run_id) else None


def _job_dirs(jobs_dir: Path) -> list[Path]:
    """The job directories; one without job.json is a submit that died before its rollback."""
    return [
        job_dir
        for job_dir in jobs_dir.iterdir()
        if runs.RUN_ID_PATTERN.fullmatch(job_dir.name) and (job_dir / JOB_FILE_NAME).exists()
    ]


def _log(line: str) -> None:
    print(f"sound-scout jobs run: {line}", file=sys.stderr, flush=True)


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
    for job_dir in _job_dirs(jobs_dir):
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


def _view(job_dir: Path, job: dict, config: Config) -> dict:
    run_ids = [attempt.get("run_id") for attempt in job["attempts"]]
    work_dir = _work_dir(job, config)
    finished = _finished_stages(job, work_dir)
    return {
        "job_id": job["job_id"],
        "request_id": job["request_id"],
        "state": job["state"],
        "state_reason": job["state_reason"],
        "submitted_at": job["submitted_at"],
        "updated_at": job["updated_at"],
        "stages": job["plan"]["stages"],
        "mode": job["plan"].get("mode"),
        "resolved_mode": job["plan"].get("resolved_mode"),
        "attempts": len(job["attempts"]),
        "retries": len(job.get("retries", [])),
        "run_id": run_ids[-1] if run_ids else None,
        "cancel_requested": (job_dir / CANCEL_MARKER_NAME).exists(),
        "current_stage": _current_stage(job, work_dir, finished),
        "finished_stages": finished,
        "delivery": job.get("delivery"),
        "control": job["request"].get("control"),
        "compare": job["attempts"][-1].get("compare") if job["attempts"] else None,
    }


def _work_dir(job: dict, config: Config) -> Path | None:
    """`work/` of the latest attempt's run, `None` before the first attempt or its run."""
    if not job["attempts"]:
        return None
    attempt = job["attempts"][-1]
    run_id = attempt.get("run_id") or _run_id_of(attempt["request_id"], config)
    if not isinstance(run_id, str) or not runs.RUN_ID_PATTERN.fullmatch(run_id):
        return None
    return config.data_root / runs.RUNS_DIR_NAME / run_id / ingest.WORK_DIR_NAME


def _current_stage(job: dict, work_dir: Path | None, finished: list[dict]) -> dict | None:
    if job["state"] != RUNNING or work_dir is None:
        return None
    try:
        marker = json.loads((work_dir / CURRENT_STAGE).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(marker, dict) or not isinstance(marker.get("started_at"), str):
        return None
    stage = marker.get("stage")
    if stage not in job["plan"]["stages"] or any(item["stage"] == stage for item in finished):
        return None
    return {"stage": stage, "started_at": marker["started_at"]}


def _finished_stages(job: dict, work_dir: Path | None) -> list[dict]:
    """`{stage, status}` of each planned stage whose final record the latest attempt's run holds,
    in plan order; a checkpoint record or an unreadable one does not count. No `reason`: it may
    carry a path or text of the recording."""
    if work_dir is None:
        return []
    finished = []
    for stage in job["plan"]["stages"]:
        try:
            record = json.loads((work_dir / STAGE_RECORD.format(name=stage)).read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get(CHECKPOINT):
            continue
        entry = record.get("entry")
        if isinstance(entry, dict) and isinstance(entry.get("status"), str):
            finished.append({"stage": stage, "status": entry["status"]})
    return finished


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
