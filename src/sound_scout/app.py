"""The one analysis entry point for CLI and MCP: `analyze` (Gate 2, R11, T08, spec § 2).

`analyze` takes the host-wide lock `<data_root>/analyze.lock` (`flock`, no waiting: a second
analysis gets `Busy`, there is no queue), checks the input in the calling process (`ingest.probe`
and the SHA-256 of the file) and claims the request with `runs.open_run`: a repeat of a complete
run with the same identity returns its `result.json` without running anything, a different
identity raises `runs.RequestMismatch`, and a run that never completed raises `RunIncomplete`
rather than charging the request again.

The stages run in a child process (`multiprocessing`, `spawn`), the worker: prepare (ingest,
channels, the 16 kHz copy for ASR), acoustics, asr, audio_review (the clip, one backend call,
findings). After each stage the worker writes `work/stage-<name>.json` atomically: the contract
entry, errors, its steps with their times and its part of the result. The caller watches
`budgets.worker_deadline_seconds` and the `CancelToken`; on either it sends SIGTERM and, after
`TERMINATE_GRACE_SECONDS`, SIGKILL. SIGTERM raises `WorkerStopped` in the worker, which no stage
boundary catches, so `proc.run` kills the group of a running decoder on the way out. The handler
also notes the stop first, and every step and record write checks that note: a stop swallowed on
its way out (a lazy C-extension import reports it as `ImportError`) still ends the worker before
it can record the interrupted stage as anything. The worker and every decoder are also bound to
their parent with `proc.die_with_parent`, so neither outlives a killed parent. Stdout of the
worker is pointed at stderr: stdout belongs to the MCP protocol.

The result is always assembled in the calling process from the stage records, one function for a
complete and a partial run alike: the first requested stage without a record is the interrupted
one (`deadline_exceeded`, `cancelled`, `worker_crashed` in `errors[]`), the requested stages after
it are `skipped`. `report.write` checks and writes `result.json` and `report.md`. The run is
marked complete except after a cancel: a repeat of a cancelled request is refused, never rerun.
No stage is retried and the backend is called at most once (R11).

`compare` reads two finished runs and compares them by one criterion (`sound_scout.comparison`).

`stages` keeps the five keys of the contract: the channel choice is part of `prepare` and the
findings part of `audio_review`; their own times are in the steps of the stage records.
"""

import contextlib
import dataclasses
import fcntl
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.process import BaseProcess
from pathlib import Path

import numpy as np

from sound_scout import (
    __version__,
    acoustics,
    asr,
    channels,
    clips,
    comparison,
    doctor,
    findings,
    guard,
    ingest,
    paths,
    prepare,
    proc,
    report,
    review,
    runs,
    schema,
)
from sound_scout.config import Config

LAYERS = ("content", "measurements", "sound")
LAYER_STAGES = {"content": "asr", "measurements": "acoustics", "sound": "audio_review"}
WORKER_STAGES = ("prepare", "acoustics", "asr", "audio_review")
POLL_SECONDS = 0.05
TERMINATE_GRACE_SECONDS = 0.5
LOCK_FILE_NAME = "analyze.lock"
LOCK_FILE_MODE = 0o600
STAGE_RECORD = "stage-{name}.json"
CLIP_PAD_MS = 0
DEADLINE_EXCEEDED = "deadline_exceeded"
CANCELLED = "cancelled"
WORKER_CRASHED = "worker_crashed"
INTERNAL_ERROR = "internal_error"
REVIEW_CALL_FAILED = "review_call_failed"
ASR_FAILED = "asr_failed"
NO_BACKEND_REASON = "backend = none: audio backend не выбран (Gate 0)"
NOT_REQUESTED_REASON = "слой не запрошен"
CANCELLED_REASON = "отменён"
SYNTHETIC_NOTE = "synthetic: test double, не доказательство способности слышать"
HALLUCINATION_LIMITATION = "ASR: есть сегменты suspect_hallucination"
QUALIFICATION_FLAGS = ("validated_ru", "diction_validated_ru", "prosody_validated_ru")
EXPECTED_ERRORS = (
    ingest.IngestError,
    channels.ChannelError,
    prepare.PrepareError,
    asr.AsrError,
    clips.ClipError,
    review.AudioPayloadMissing,
    paths.PathDenied,
)
_CODELESS_ERRORS = {
    review.AudioPayloadMissing: "audio_payload_missing",
    paths.PathDenied: "path_denied",
}


class InvalidRequest(ValueError):
    """The request itself is malformed (layers, request ID); nothing was opened or run."""


class Busy(Exception):
    """Another analysis holds the lock of this data root; no run was created."""


class RunIncomplete(Exception):
    """The request ID names a run that did not complete; nothing was run, use a new request ID."""


class WorkerStopped(BaseException):
    """Raised in the worker by SIGTERM; not an `Exception`, so no stage boundary swallows it."""


# The refusals `analyze` raises before any stage runs, with the reason code CLI and MCP report.
REFUSALS = (
    (InvalidRequest, "invalid_request"),
    (paths.PathDenied, "path_denied"),
    (Busy, "busy"),
    (RunIncomplete, "run_incomplete"),
    (runs.RequestMismatch, "request_mismatch"),
    (runs.RequestRunMissing, "request_run_missing"),
    (runs.RequestIndexCorrupt, "request_index_corrupt"),
)


_stop_signal: int | None = None


@dataclass(frozen=True)
class AnalyzeRequest:
    path: str
    request_id: str | None = None
    layers: tuple[str, ...] = LAYERS
    task: str | None = None
    intent: str | None = None


@dataclass(frozen=True)
class Analysis:
    result: dict
    run: runs.Run
    reused: bool


class CancelToken:
    """Set from any thread; `analyze` stops its worker within `TERMINATE_GRACE_SECONDS` + a poll."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class _Interruption:
    code: str
    reason: str


@dataclass(frozen=True)
class _Preflight:
    probe: ingest.Probe
    audio_sha256: str
    description: review.BackendDescription | None


def analyze(
    request: AnalyzeRequest,
    config: Config,
    *,
    backend: review.AudioBackend | None = None,
    cancel: CancelToken | None = None,
) -> Analysis:
    """Analyse one recording; see the module docstring.

    Raises `InvalidRequest` for a bad request, `Busy`, `paths.PathDenied` and `ingest.IngestError`
    for an input refused before any run exists, `runs.RequestMismatch` (and the other `runs`
    index errors) and `RunIncomplete` for a request ID that cannot be run again.
    """
    layers = _checked_layers(request.layers)
    request_id = runs.new_run_id() if request.request_id is None else request.request_id
    _check_request_id(request_id)
    request = dataclasses.replace(request, request_id=request_id, layers=layers)
    with _analysis_lock(config.data_root):
        preflight = _Preflight(
            probe=ingest.probe(request.path, config),
            audio_sha256=ingest.input_sha256(request.path, config),
            description=None if backend is None else backend.describe(),
        )
        run = runs.open_run(request_id, _identity(request, config, preflight), config.data_root)
        if run.reused:
            result = json.loads((run.path / report.RESULT_FILE_NAME).read_text(encoding="utf-8"))
            return Analysis(result, run, True)
        if not run.created:
            raise RunIncomplete(
                f"request {request_id!r} names run {run.run_id}, which did not complete; "
                "it is not run again, use a new request_id"
            )
        interruption = _execute(request, config, run, backend, preflight.audio_sha256, cancel)
        document = _assemble(run, request, config, _read_records(run), interruption, preflight)
        written = report.write(run, document, config=config)
        if interruption is None or interruption.code != CANCELLED:
            run = runs.complete_run(run)
        return Analysis(written, run, False)


def compare(run_a: str, run_b: str, criterion: str, config: Config) -> comparison.Comparison:
    """Attempt `run_a` (initial) against `run_b` (control) by one criterion (R07).

    Both are `run_id`s of complete `analyze` runs under the data root; the result is written to
    `<data_root>/compare/<compare_id>.json`. Nothing is run and no lock is taken: compare only
    reads finished runs. Raises `comparison.CompareError` with its `code`, writing nothing.
    """
    return comparison.compare(run_a, run_b, criterion, config)


def _checked_layers(layers: object) -> tuple[str, ...]:
    if isinstance(layers, str) or not isinstance(layers, tuple | list) or not layers:
        raise InvalidRequest(f"layers must be a non-empty list of {list(LAYERS)}, got {layers!r}")
    unknown = [layer for layer in layers if layer not in LAYERS]
    if unknown or len(set(layers)) != len(layers):
        raise InvalidRequest(f"layers must be distinct items of {list(LAYERS)}, got {list(layers)}")
    return tuple(layers)


def _check_request_id(request_id: object) -> None:
    """The result contract's `identifier_string`, checked before anything is run."""
    bounds = schema.SCHEMA["$defs"]["identifier_string"]
    if not isinstance(request_id, str) or not (
        bounds["minLength"] <= len(request_id) <= bounds["maxLength"]
    ):
        raise InvalidRequest(
            f"request_id must be a string of {bounds['minLength']}..{bounds['maxLength']} "
            f"characters, got {type(request_id).__name__} of length "
            f"{len(request_id) if isinstance(request_id, str) else None}"
        )


@contextlib.contextmanager
def _analysis_lock(data_root: Path) -> Iterator[None]:
    """One analysis per data root, across processes; the kernel drops the lock with its holder."""
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(data_root / LOCK_FILE_NAME, flags, LOCK_FILE_MODE)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy("another analysis is running on this data root") from None
        yield
    finally:
        os.close(fd)


def _identity(request: AnalyzeRequest, config: Config, preflight: _Preflight) -> dict:
    description = preflight.description
    snapshot, missing = doctor.snapshot(config.asr.model, config.asr.revision)
    if description is None:
        backend = "none"
    else:
        backend = _backend_name(description) | {"settings": dict(description.settings)}
    return {
        "audio": preflight.audio_sha256,
        "model": {
            "asr": {
                "model": config.asr.model,
                "revision": config.asr.revision,
                "resolved_revision": None if missing else snapshot["revision"],
            },
            "backend": backend,
        },
        "preprocessing": {
            "channels": channels.TRANSFORMATION_NAME,
            "resample": prepare.RESAMPLER_SETTINGS,
            "clip_pad_ms": CLIP_PAD_MS,
        },
        "ranges": None,
        "prompt_schema": {
            "prompt_sha256": review.load_prompt().sha256,
            "schema_version": schema.SCHEMA_VERSION,
        },
        "task_context": {
            "layers": sorted(request.layers),
            "task": request.task,
            "intent": request.intent,
        },
    }


def _backend_name(description: review.BackendDescription) -> dict:
    return {"name": description.name, "model": description.model, "revision": description.revision}


# --- the worker and its supervision ----------------------------------------------------------


def _execute(
    request: AnalyzeRequest,
    config: Config,
    run: runs.Run,
    backend: review.AudioBackend | None,
    audio_sha256: str,
    cancel: CancelToken | None,
) -> _Interruption | None:
    """Run the worker to its end, the deadline or the cancel; None when it finished cleanly."""
    if cancel is not None and cancel.cancelled:
        return _Interruption(CANCELLED, CANCELLED_REASON)
    deadline_seconds = config.budgets.worker_deadline_seconds
    deadline = time.monotonic() + deadline_seconds
    process = _start_worker(request, config, run, backend, audio_sha256)
    try:
        while process.exitcode is None:
            process.join(POLL_SECONDS)
            if process.exitcode is not None:
                break
            if cancel is not None and cancel.cancelled:
                return _Interruption(CANCELLED, CANCELLED_REASON)
            if time.monotonic() >= deadline:
                return _Interruption(DEADLINE_EXCEEDED, f"прерван: deadline {deadline_seconds} с")
    finally:
        _stop(process)
    if process.exitcode != 0:
        return _Interruption(WORKER_CRASHED, f"worker завершился с кодом {process.exitcode}")
    return None


def _start_worker(
    request: AnalyzeRequest,
    config: Config,
    run: runs.Run,
    backend: review.AudioBackend | None,
    audio_sha256: str,
) -> BaseProcess:
    # spawn, not fork: forking a threaded process (the MCP server) is unsafe.
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_work,
        args=(request, config, run, backend, audio_sha256, os.getpid()),
        name="sound-scout-worker",
        daemon=False,
    )
    process.start()
    return process


def _stop(process: BaseProcess) -> None:
    if process.exitcode is None:
        process.terminate()
        process.join(TERMINATE_GRACE_SECONDS)
    if process.exitcode is None:
        process.kill()
        process.join()


def _work(
    request: AnalyzeRequest,
    config: Config,
    run: runs.Run,
    backend: review.AudioBackend | None,
    audio_sha256: str,
    parent_pid: int,
) -> None:
    """The worker process: every stage in order, a record after each."""
    proc.die_with_parent(parent_pid)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    signal.signal(signal.SIGTERM, _raise_stopped)
    guard.install_network_guard(config)
    try:
        _run_stages(request, config, run, backend, audio_sha256)
    except WorkerStopped:
        raise SystemExit(128 + signal.SIGTERM) from None


def _raise_stopped(signum: int, frame: object) -> None:
    global _stop_signal
    _stop_signal = signum
    raise WorkerStopped(f"signal {signum}")


def _raise_if_stopped() -> None:
    """Raise again a stop some code below swallowed, so it never becomes a stage status."""
    if _stop_signal is not None:
        raise WorkerStopped(f"signal {_stop_signal}")


@dataclass(frozen=True)
class _Prepared:
    source: ingest.Source
    mono: np.ndarray
    mono16k: prepare.Mono16k | None


class _Stage:
    """The record of one stage as it runs: timed steps, errors and its part of the result."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.started = time.monotonic()
        self.steps: list[dict] = []
        self.errors: list[dict] = []
        self.fragment: dict = {}

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        _raise_if_stopped()
        started_at, started = _now(), time.monotonic()
        try:
            yield
        finally:
            self.steps.append(
                {
                    "name": name,
                    "started_at": started_at,
                    "finished_at": _now(),
                    "elapsed_ms": _elapsed_ms(started),
                }
            )

    def error(self, code: str, message: str) -> None:
        self.errors.append({"stage": self.name, "code": code, "message": message})

    def done(self, status: str, reason: str | None) -> dict:
        return {
            "stage": self.name,
            "entry": {
                "requested": True,
                "status": status,
                "reason": reason,
                "elapsed_ms": _elapsed_ms(self.started),
            },
            "errors": self.errors,
            "steps": self.steps,
            "fragment": self.fragment,
        }

    def failed(self, error: Exception) -> dict:
        if isinstance(error, EXPECTED_ERRORS):
            code = getattr(error, "code", None) or next(
                name for kind, name in _CODELESS_ERRORS.items() if isinstance(error, kind)
            )
            message = str(error) or type(error).__name__
        else:
            code, message = INTERNAL_ERROR, f"{type(error).__name__}: {error}"
            # the message may hold private paths; it stays in the run's stage record only
            print(f"stage {self.name}: {code}: {type(error).__name__}", file=sys.stderr)
        self.error(code, message)
        return self.done("failed", message)


def _run_stages(
    request: AnalyzeRequest,
    config: Config,
    run: runs.Run,
    backend: review.AudioBackend | None,
    audio_sha256: str,
) -> None:
    requested = {LAYER_STAGES[layer] for layer in request.layers}
    prepared, record = _prepare_stage(request.path, run, config, audio_sha256, "asr" in requested)
    _write_record(run, config, record)
    no_signal = f"нет подготовленного сигнала: prepare {record['entry']['status']}"
    asr_completed = False
    for name in WORKER_STAGES[1:]:
        if name not in requested:
            continue
        if prepared is None:
            record = _Stage(name).done("skipped", no_signal)
        elif name == "acoustics":
            record = _acoustics_stage(prepared)
        elif name == "asr":
            record = _asr_stage(prepared, run, config)
            asr_completed = record["entry"]["status"] == "completed"
        else:
            record = _review_stage(prepared, request, run, config, backend, asr_completed)
        _write_record(run, config, record)


def _prepare_stage(
    path: str, run: runs.Run, config: Config, audio_sha256: str, for_asr: bool
) -> tuple[_Prepared | None, dict]:
    stage = _Stage("prepare")
    try:
        with stage.step("ingest"):
            source = ingest.decode(path, run, config)
        stage.fragment.update(
            source={
                "id": source.id,
                "sha256": source.sha256,
                "duration_s": source.duration_s,
                "channels": source.probe.channels,
            },
            duration_mismatch_s=source.duration_mismatch_s,
            duration_mismatch=source.duration_mismatch,
        )
        if source.sha256 != audio_sha256:
            raise ingest.IngestError(
                ingest.INPUT_CHANGED, "the input is not the file checked before the run"
            )
        with stage.step("channels"):
            audio = channels.load(source)
            mono, choice = channels.choose_mono(audio, channels.report(audio))
        stage.fragment["transformations"] = [choice]
        mono16k = None
        if for_asr:
            with stage.step("mono16k"):
                mono16k = prepare.mono16k(mono, source.probe.sample_rate, run, config)
            stage.fragment["transformations"].append(mono16k.transformation)
    except Exception as error:
        return None, stage.failed(error)
    return _Prepared(source, mono, mono16k), stage.done("completed", None)


def _acoustics_stage(prepared: _Prepared) -> dict:
    stage = _Stage("acoustics")
    try:
        with stage.step("measure"):
            measured = acoustics.measure(prepared.mono, prepared.source.probe.sample_rate)
    except Exception as error:
        return stage.failed(error)
    stage.fragment["measurements"] = measured["measurements"]
    return stage.done(measured["status"], measured["reason"])


def _asr_stage(prepared: _Prepared, run: runs.Run, config: Config) -> dict:
    stage = _Stage("asr")
    assert prepared.mono16k is not None
    try:
        with stage.step("transcribe"):
            transcription = asr.transcribe(prepared.mono16k.samples, run, config)
    except Exception as error:
        return stage.failed(error)
    if transcription.status == "completed":
        stage.fragment["artifact"] = {
            "id": findings.TRANSCRIPT_ARTIFACT_ID,
            "kind": "transcript_raw",
            "path": f"{asr.ASR_DIR_NAME}/{asr.RAW_FILE_NAME}",
            "sha256": transcription.raw_sha256,
            "source_start": None,
            "source_end": None,
        }
        stage.fragment["flags"] = list(transcription.flags)
        stage.fragment["suspect_segments"] = asr.suspect_segments(
            transcription.raw, prepared.source.duration_s
        )
    elif transcription.status == "failed":
        stage.error(ASR_FAILED, transcription.reason or ASR_FAILED)
    return stage.done(transcription.status, transcription.reason)


def _review_stage(
    prepared: _Prepared,
    request: AnalyzeRequest,
    run: runs.Run,
    config: Config,
    backend: review.AudioBackend | None,
    asr_completed: bool,
) -> dict:
    stage = _Stage("audio_review")
    if backend is None:
        return stage.done("unavailable", NO_BACKEND_REASON)
    try:
        with stage.step("clip"):
            cut = clips.cut(run, 0.0, prepared.source.duration_s, CLIP_PAD_MS, config=config)
            clip = review.clip_from_cut(cut)
        prompt = review.load_prompt()
        with stage.step("review"):
            try:
                outcome = review.run_review(
                    backend, clip, run, config, prompt, task=request.task, intent=request.intent
                )
            except EXPECTED_ERRORS:
                raise
            except Exception as error:  # the call itself failed: recorded, never repeated (R11)
                message = f"{type(error).__name__}: {error}"
                stage.error(REVIEW_CALL_FAILED, message)
                return stage.done("failed", message)
        if outcome.response is not None:
            stage.fragment.update(
                model_limitations=list(outcome.response.limitations),
                model_uncertainties=list(outcome.response.uncertainties),
            )
        stage.fragment.update(
            backend=outcome.provenance["backend"],
            payload_sha256=outcome.provenance["payload"]["sha256"],
            prompt_sha256=prompt.sha256,
        )
        transcript = asr.read_raw(run) if asr_completed else None
        with stage.step("findings"):
            built = findings.build(run, [outcome], None, transcript, config=config)
    except Exception as error:
        return stage.failed(error)
    stage.fragment["built"] = dataclasses.asdict(built)
    stage.errors += built.errors
    if built.reviewed:
        return stage.done("completed", None)
    first = built.errors[0] if built.errors else {"code": INTERNAL_ERROR, "message": "no range"}
    return stage.done("failed", f"{first['code']}: {first['message']}")


def _write_record(run: runs.Run, config: Config, record: dict) -> None:
    """`work/stage-<name>.json`, replaced atomically so a killed worker leaves no torn record."""
    _raise_if_stopped()
    work_dir = paths.resolve_output(run.path, ingest.WORK_DIR_NAME, config)
    try:
        os.mkdir(work_dir, ingest.DIR_MODE)
    except FileExistsError:
        pass
    else:
        os.chmod(work_dir, ingest.DIR_MODE)
    name = STAGE_RECORD.format(name=record["stage"])
    staging = f".{name}.{uuid.uuid4().hex}"
    text = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with paths.open_output(work_dir, staging, config) as out:
        out.write(text.encode("utf-8"))
    os.replace(work_dir / staging, work_dir / name)


def _read_records(run: runs.Run) -> dict[str, dict]:
    records = {}
    for name in WORKER_STAGES:
        path = run.path / ingest.WORK_DIR_NAME / STAGE_RECORD.format(name=name)
        with contextlib.suppress(FileNotFoundError):
            records[name] = json.loads(path.read_text(encoding="utf-8"))
    return records


# --- the result document ---------------------------------------------------------------------


def _assemble(
    run: runs.Run,
    request: AnalyzeRequest,
    config: Config,
    records: dict[str, dict],
    interruption: _Interruption | None,
    preflight: _Preflight,
) -> dict:
    """The result document from the stage records: the only place a result is put together."""
    started = time.monotonic()
    stages, errors = _stages(request.layers, records, interruption)
    fragments = {name: record["fragment"] for name, record in records.items()}
    prepared = fragments.get("prepare", {})
    review_fragment = fragments.get("audio_review", {})
    completed = {name: stage["status"] == "completed" for name, stage in stages.items()}
    built = review_fragment.get("built") if completed["audio_review"] else None
    sha256 = preflight.audio_sha256
    source = prepared.get("source") or {
        "id": f"src-{sha256[:16]}",
        "sha256": sha256,
        "duration_s": preflight.probe.declared_duration_s or 0.0,
        "channels": preflight.probe.channels,
    }
    requested = []
    if "sound" in request.layers and completed["prepare"]:
        requested = [{"start": 0.0, "end": source["duration_s"]}]
    reviewed = built["reviewed"] if built else []
    unreviewed = [{"start": s, "end": e} for s, e in schema.subtract(requested, reviewed)]
    artifacts = _merged_artifacts(
        ([fragments["asr"]["artifact"]] if completed["asr"] else [])
        + (review_fragment["built"]["artifacts"] if "built" in review_fragment else [])
    )
    description = preflight.description
    synthetic = description is not None and description.name == doctor.TEST_DOUBLE
    real_backend = description is not None and not synthetic
    if real_backend:
        flags = doctor.qualification_flags(config.data_root)
    else:
        flags = dict.fromkeys(QUALIFICATION_FLAGS, False)
    limitations = [
        f"слой {layer} не выполнен: {stages[LAYER_STAGES[layer]]['reason']}"
        for layer in request.layers
        if not completed[LAYER_STAGES[layer]]
    ]
    if asr.SUSPECT_HALLUCINATION in fragments.get("asr", {}).get("flags", []):
        limitations.append(HALLUCINATION_LIMITATION)
    for prefix, key in (
        (review.MODEL_LIMITATION, "model_limitations"),
        (review.MODEL_UNCERTAINTY, "model_uncertainties"),
    ):
        limitations += [prefix + text for text in review_fragment.get(key, [])]
    if not completed["prepare"]:
        status = "failed"
    elif all(completed[name] for name, stage in stages.items() if stage["requested"]) and (
        not unreviewed
    ):
        status = "complete"
    else:
        status = "partial"
    budgets = config.budgets
    document = {
        "schema_version": schema.SCHEMA_VERSION,
        "request_id": request.request_id,
        "run_id": run.run_id,
        "status": status,
        "source": source,
        "capabilities": {
            "layers": {
                layer: {
                    "available": completed[stage_name],
                    "reason": stages[stage_name]["reason"],
                }
                for layer, stage_name in LAYER_STAGES.items()
            },
            "audio_backend": None if description is None else _backend_name(description),
            **{name: flags[name] for name in QUALIFICATION_FLAGS},
        },
        "stages": {},
        "coverage": {"requested": requested, "reviewed": reviewed, "unreviewed": unreviewed},
        "provenance": {
            "versions": {"sound_scout": __version__},
            "backend": "none" if description is None else description.name,
            "model": None if description is None else description.model,
            "revision": None if description is None else description.revision,
            "input_modalities": [] if description is None else list(description.input_modalities),
            "audio_payload_sha256": review_fragment.get("payload_sha256"),
            "prompt_sha256": review_fragment.get("prompt_sha256"),
            "schema_sha256": schema.SCHEMA_SHA256,
            "transformations": prepared.get("transformations", []),
            "limitations": limitations,
            "synthetic": synthetic,
            "note": SYNTHETIC_NOTE if synthetic else None,
        },
        "transcript": {
            "raw": {"artifact_ref": findings.TRANSCRIPT_ARTIFACT_ID} if completed["asr"] else None,
            "corrections": built["corrections"] if built else [],
            "suspect_segments": fragments["asr"]["suspect_segments"] if completed["asr"] else [],
        },
        "measurements": fragments.get("acoustics", {}).get("measurements", []),
        "findings": built["findings"] if built else [],
        "assessment_status": {
            "diction": _assessment(stages["audio_review"]),
            "prosody": _assessment(stages["audio_review"]),
            "content": _assessment(stages["asr"]),
        },
        "artifacts": artifacts,
        "limits": {
            "max_duration_seconds": budgets.max_duration_seconds,
            "max_input_bytes": budgets.max_input_bytes,
            "max_channels": budgets.max_channels,
            "worker_deadline_seconds": budgets.worker_deadline_seconds,
            "duration_mismatch_tolerance_s": ingest.DURATION_MISMATCH_TOLERANCE_S,
            "duration_mismatch_s": prepared.get("duration_mismatch_s"),
            "duration_mismatch": prepared.get("duration_mismatch"),
        },
        "errors": errors,
        "cost": report.cost(),
        "unlocalized": built["unlocalized"] if built else [],
    }
    stages["report"] = {
        "requested": True,
        "status": "completed",
        "reason": None,
        "elapsed_ms": _elapsed_ms(started),
    }
    document["stages"] = {name: stages[name] for name in schema.STAGES}
    return document


def _stages(
    layers: tuple[str, ...], records: dict[str, dict], interruption: _Interruption | None
) -> tuple[dict[str, dict], list[dict]]:
    """The contract entry of each worker stage and the errors; the cut-off one is interrupted."""
    requested = {"prepare": True} | {
        stage: layer in layers for layer, stage in LAYER_STAGES.items()
    }
    stages: dict[str, dict] = {}
    errors: list[dict] = []
    cut_off: str | None = None
    for name in WORKER_STAGES:
        record = records.get(name)
        if not requested[name]:
            stages[name] = _entry(False, "skipped", NOT_REQUESTED_REASON)
        elif record is not None:
            stages[name] = record["entry"]
            errors += record["errors"]
        elif cut_off is None:
            if interruption is None:
                interruption = _Interruption(INTERNAL_ERROR, "worker ended without a stage record")
            cut_off = interruption.reason
            stages[name] = _entry(True, "failed", cut_off)
            errors.append({"stage": name, "code": interruption.code, "message": cut_off})
        else:
            stages[name] = _entry(True, "skipped", cut_off)
    return stages, errors


def _entry(requested: bool, status: str, reason: str) -> dict:
    return {"requested": requested, "status": status, "reason": reason, "elapsed_ms": None}


def _assessment(stage: dict) -> dict:
    if stage["status"] == "completed":
        return {"status": "assessed", "reason": None}
    return {"status": "not_assessed", "reason": stage["reason"]}


def _merged_artifacts(items: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for artifact in items:
        known = merged.setdefault(artifact["id"], artifact)
        if known != artifact:
            raise ValueError(f"two different artifacts share the id {artifact['id']!r}")
    return list(merged.values())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
