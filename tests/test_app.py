"""`app.analyze`: stages and their statuses, partial results, deadline, cancel, cache, the lock.

Audio is synthesized in the temp dir. `FakeBackend` is the test double: it proves the pipeline
contract and the `synthetic` mark, never that the tool hears anything. The `slow` test runs the
real cached ASR model on the development recording named by `SOUND_SCOUT_DEV_AUDIO` and is
skipped, with the reason, without it.
"""

import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sound_scout import app, doctor, findings, proc, review, runs
from sound_scout.app import AnalyzeRequest, Busy, CancelToken, RunIncomplete, analyze
from sound_scout.config import Asr, Budgets, Config
from sound_scout.review import BackendDescription, RawReviewResponse, ReviewRequest
from tests.conftest import WavFactory
from tests.test_proc import process_gone
from tests.test_review import FakeBackend

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)

CHILD = (
    "import os, sys, time\n"
    "path = sys.argv[1]\n"
    "open(path + '.tmp', 'w').write(str(os.getpid()))\n"
    "os.replace(path + '.tmp', path)\n"
    "time.sleep(600)\n"
)
ALL_LAYERS = ("content", "measurements", "sound")
LISTENED = ("measurements", "sound")


def write_pid(path: Path, pid: int) -> None:
    path.with_suffix(".tmp").write_text(str(pid))
    os.replace(path.with_suffix(".tmp"), path)


@dataclass
class SlowFakeBackend(FakeBackend):
    """Records the worker's pid, then waits on a child that never ends: a hung backend call."""

    pid_dir: str = ""

    def review(self, request: ReviewRequest) -> RawReviewResponse:
        write_pid(Path(self.pid_dir) / "worker", os.getpid())
        child = str(Path(self.pid_dir) / "child")
        proc.run([sys.executable, "-c", CHILD, child], timeout=900, max_output_bytes=1024)
        return super().review(request)


@dataclass
class StopSwallowingBackend(SlowFakeBackend):
    """Swallows the worker's stop inside the hung call and answers as if nothing happened."""

    def review(self, request: ReviewRequest) -> RawReviewResponse:
        try:
            return super().review(request)
        except BaseException:
            return FakeBackend.review(self, request)


@dataclass
class RaisingBackend(FakeBackend):
    calls_file: str = ""

    def review(self, request: ReviewRequest) -> RawReviewResponse:
        with open(self.calls_file, "a") as calls:
            calls.write("call\n")
        raise RuntimeError("backend is down")


def wait_for(path: Path, seconds: float = 20.0) -> Path:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} did not appear in {seconds} s"
        time.sleep(0.02)
    return path


def pid_of(path: Path) -> int:
    return int(wait_for(path).read_text())


@pytest.fixture
def config(tmp_path: Path) -> Config:
    root, data_root = tmp_path / "recordings", tmp_path / "data"
    root.mkdir()
    data_root.mkdir()
    return Config(allowed_roots=(root,), data_root=data_root, asr=Asr(model="absent/model"))


@pytest.fixture
def wav(make_wav: WavFactory, config: Config) -> Path:
    return make_wav("recordings/tone.wav", seconds=1.0)


@pytest.fixture
def pid_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "pids"
    directory.mkdir()
    return directory


def stage_record(run: runs.Run, name: str) -> dict:
    return json.loads((run.path / "work" / f"stage-{name}.json").read_text(encoding="utf-8"))


def manifest(run: runs.Run) -> dict:
    return json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))


def codes(result: dict) -> set[tuple[str | None, str]]:
    return {(error["stage"], error["code"]) for error in result["errors"]}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- stages ----------------------------------------------------------------------------------


def test_stages_run_in_order_and_an_unavailable_one_does_not_stop_the_rest(
    config: Config, wav: Path
) -> None:
    analysis = analyze(AnalyzeRequest(str(wav), layers=ALL_LAYERS), config)

    result, run = analysis.result, analysis.run
    stages = result["stages"]
    assert list(stages) == ["prepare", "asr", "acoustics", "audio_review", "report"]
    assert {name: stage["status"] for name, stage in stages.items()} == {
        "prepare": "completed",
        "asr": "unavailable",
        "acoustics": "completed",
        "audio_review": "unavailable",
        "report": "completed",
    }
    assert all(isinstance(stage["elapsed_ms"], int) for stage in stages.values())
    assert stages["asr"]["reason"] and stages["audio_review"]["reason"]
    assert result["status"] == "partial"
    prepare_steps = [step["name"] for step in stage_record(run, "prepare")["steps"]]
    assert prepare_steps == ["ingest", "channels", "mono16k"]
    review_steps = stage_record(run, "audio_review")["steps"]
    assert review_steps == []
    started = [
        stage_record(run, name)["steps"][0]["started_at"] for name in ("prepare", "acoustics")
    ]
    assert started == sorted(started)
    assert [m["name"] for m in result["measurements"]][0] == "duration_seconds"
    assert result["provenance"]["transformations"][0]["name"] == "channel_choice"
    assert analysis.reused is False and manifest(run)["complete"] is True
    findings.validate(result, run=run)


def test_backend_none_gives_partial_and_diction_prosody_not_assessed(
    config: Config, wav: Path
) -> None:
    config = dataclasses.replace(config, backend="none")

    analysis = analyze(AnalyzeRequest(str(wav), layers=ALL_LAYERS), config)

    result = analysis.result
    assert result["status"] == "partial"
    for name in ("diction", "prosody"):
        assert result["assessment_status"][name]["status"] == "not_assessed"
        assert "backend = none" in result["assessment_status"][name]["reason"]
    duration = result["source"]["duration_s"]
    assert result["coverage"]["unreviewed"] == [{"start": 0.0, "end": duration}]
    assert result["coverage"]["reviewed"] == []
    assert result["provenance"]["backend"] == "none"
    assert result["capabilities"]["audio_backend"] is None
    assert result["capabilities"]["layers"]["sound"]["available"] is False
    findings.validate(result, run=analysis.run)


def test_a_layer_not_requested_is_skipped_with_its_reason(config: Config, wav: Path) -> None:
    result = analyze(AnalyzeRequest(str(wav), layers=("measurements",)), config).result

    assert result["status"] == "complete"
    for name in ("asr", "audio_review"):
        assert result["stages"][name] == {
            "requested": False,
            "status": "skipped",
            "reason": app.NOT_REQUESTED_REASON,
            "elapsed_ms": None,
        }
    assert result["coverage"] == {"requested": [], "reviewed": [], "unreviewed": []}


def test_a_failed_prepare_skips_the_requested_stages_and_fails_the_result(
    config: Config, make_wav: WavFactory
) -> None:
    silent = make_wav("recordings/silent.wav", seconds=0.5, frequency=0.0)

    result = analyze(AnalyzeRequest(str(silent), layers=ALL_LAYERS), config).result

    assert result["status"] == "failed"
    assert result["stages"]["prepare"]["status"] == "failed"
    assert ("prepare", "zero_signal") in codes(result)
    for name in ("asr", "acoustics", "audio_review"):
        assert result["stages"][name]["status"] == "skipped"
        assert "prepare failed" in result["stages"][name]["reason"]


@pytest.mark.parametrize("layers", [(), ("sound", "sound"), ("emotion",), "sound"])
def test_a_bad_layer_list_is_refused_before_anything_runs(
    config: Config, wav: Path, layers: object
) -> None:
    with pytest.raises(app.InvalidRequest, match="layers"):
        analyze(AnalyzeRequest(str(wav), layers=layers), config)  # type: ignore[arg-type]
    assert not (config.data_root / "runs").exists()


def test_an_input_outside_the_roots_is_refused_before_a_run_exists(
    config: Config, make_wav: WavFactory
) -> None:
    outside = make_wav("outside.wav")

    with pytest.raises(app.paths.PathDenied):
        analyze(AnalyzeRequest(str(outside)), config)
    assert not (config.data_root / "runs").exists()


@pytest.mark.parametrize("request_id", ["", "r" * 257, 7])
def test_a_bad_request_id_is_refused_before_anything_runs(
    config: Config, wav: Path, request_id: object
) -> None:
    with pytest.raises(app.InvalidRequest, match="request_id"):
        analyze(AnalyzeRequest(str(wav), request_id=request_id), config)  # type: ignore[arg-type]
    assert not (config.data_root / "runs").exists()


def test_a_request_id_at_the_contract_bounds_passes_the_check() -> None:
    app._check_request_id("r")
    app._check_request_id("r" * 256)


def test_an_unexpected_stage_error_logs_one_line_without_its_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = app._Stage("acoustics").failed(RuntimeError("cannot read /home/someone/secret.wav"))

    err = capsys.readouterr().err
    assert err == "stage acoustics: internal_error: RuntimeError\n"
    assert record["errors"][0]["message"] == "RuntimeError: cannot read /home/someone/secret.wav"
    assert record["entry"]["status"] == "failed"


# --- deadline and cancel ---------------------------------------------------------------------


def test_deadline_stops_the_worker_and_its_child_and_keeps_what_is_ready(
    config: Config, wav: Path, pid_dir: Path
) -> None:
    config = dataclasses.replace(config, budgets=Budgets(worker_deadline_seconds=5))
    started = time.monotonic()

    analysis = analyze(
        AnalyzeRequest(str(wav), layers=LISTENED),
        config,
        backend=SlowFakeBackend(pid_dir=str(pid_dir)),
    )

    assert time.monotonic() - started < 5 + 2
    result, run = analysis.result, analysis.run
    assert result["status"] == "partial"
    assert ("audio_review", "deadline_exceeded") in codes(result)
    assert result["stages"]["audio_review"]["status"] == "failed"
    assert result["stages"]["acoustics"]["status"] == "completed"
    assert process_gone(pid_of(pid_dir / "worker"), wait_seconds=1.0)
    assert process_gone(pid_of(pid_dir / "child"), wait_seconds=1.0)
    for name in (
        "source.json",
        "work/decoded.f32",
        "work/stage-prepare.json",
        "work/stage-acoustics.json",
        "result.json",
        "report.md",
    ):
        assert (run.path / name).is_file(), name
    assert list((run.path / "clips").glob("clip-*.wav"))
    assert result["provenance"]["synthetic"] is True
    assert manifest(run)["complete"] is True
    findings.validate(result, run=run)


def test_cancel_inside_a_subprocess_wait_frees_the_worker_within_two_seconds(
    config: Config, wav: Path, pid_dir: Path
) -> None:
    token = CancelToken()
    cancelled_at: list[float] = []

    def cancel_when_the_child_runs() -> None:
        wait_for(pid_dir / "child")
        cancelled_at.append(time.monotonic())
        token.cancel()

    canceller = threading.Thread(target=cancel_when_the_child_runs)
    canceller.start()
    analysis = analyze(
        AnalyzeRequest(str(wav), layers=LISTENED),
        config,
        backend=SlowFakeBackend(pid_dir=str(pid_dir)),
        cancel=token,
    )
    returned_at = time.monotonic()
    canceller.join()

    assert returned_at - cancelled_at[0] <= 2.0
    assert process_gone(pid_of(pid_dir / "worker"), wait_seconds=1.0)
    assert process_gone(pid_of(pid_dir / "child"), wait_seconds=1.0)
    assert ("audio_review", "cancelled") in codes(analysis.result)
    assert analysis.result["status"] == "partial"
    assert manifest(analysis.run)["complete"] is False


def test_cancel_between_stages_frees_the_worker_within_two_seconds(
    config: Config, wav: Path, pid_dir: Path
) -> None:
    token = CancelToken()
    cancelled_at: list[float] = []

    def cancel_after_prepare() -> None:
        deadline = time.monotonic() + 20
        while not list(config.data_root.glob("runs/*/work/stage-prepare.json")):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        cancelled_at.append(time.monotonic())
        token.cancel()

    canceller = threading.Thread(target=cancel_after_prepare)
    canceller.start()
    analysis = analyze(
        AnalyzeRequest(str(wav), layers=LISTENED),
        config,
        backend=SlowFakeBackend(pid_dir=str(pid_dir)),
        cancel=token,
    )
    returned_at = time.monotonic()
    canceller.join()

    assert returned_at - cancelled_at[0] <= 2.0
    stages = analysis.result["stages"]
    assert stages["prepare"]["status"] == "completed"
    order = ("acoustics", "audio_review")
    statuses = [stages[name]["status"] for name in order]
    cut = statuses.index("failed")
    assert all(status == "completed" for status in statuses[:cut])
    assert all(status == "skipped" for status in statuses[cut + 1 :])
    assert (order[cut], "cancelled") in codes(analysis.result)
    if (pid_dir / "worker").exists():
        assert process_gone(pid_of(pid_dir / "worker"), wait_seconds=1.0)
    assert manifest(analysis.run)["complete"] is False


@pytest.mark.parametrize(
    ("code", "complete"), [(app.CANCELLED, False), (app.DEADLINE_EXCEEDED, True)]
)
def test_a_stop_swallowed_inside_a_stage_writes_no_record_and_the_stage_is_interrupted(
    config: Config, wav: Path, monkeypatch: pytest.MonkeyPatch, code: str, complete: bool
) -> None:
    def measure_swallowing_the_stop(mono: object, rate: int) -> dict:
        # A lazy C-extension import reports the stop as ImportError; acoustics calls it unavailable.
        try:
            try:
                app._raise_stopped(signal.SIGTERM, None)
            except app.WorkerStopped as stop:
                raise ImportError("initialization failed") from stop
        except ImportError as error:
            return {"status": "unavailable", "reason": str(error), "measurements": []}

    def execute_in_process(request: AnalyzeRequest, config: Config, run: runs.Run, *rest: object):
        try:
            app._run_stages(request, config, run, None, rest[1])
        except app.WorkerStopped:
            return app._Interruption(code, "прерван")
        raise AssertionError("the swallowed stop let the worker write the stage record")

    monkeypatch.setattr(app, "_stop_signal", None)
    monkeypatch.setattr(app.acoustics, "measure", measure_swallowing_the_stop)
    monkeypatch.setattr(app, "_execute", execute_in_process)

    analysis = analyze(AnalyzeRequest(str(wav), layers=("measurements",)), config)

    result, run = analysis.result, analysis.run
    assert stage_record(run, "prepare")["entry"]["status"] == "completed"
    assert not (run.path / "work" / "stage-acoustics.json").exists()
    assert result["stages"]["acoustics"]["status"] == "failed"
    assert ("acoustics", code) in codes(result)
    assert result["status"] == "partial"
    assert manifest(run)["complete"] is complete


@pytest.mark.parametrize("code", [app.CANCELLED, app.DEADLINE_EXCEEDED])
def test_a_stop_swallowed_in_a_step_keeps_the_next_step_of_the_stage_from_starting(
    config: Config, wav: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    real_cut, real_run_review = app.clips.cut, app.review.run_review
    review_calls: list[str] = []

    def cut_swallowing_the_stop(*args: object, **kwargs: object) -> object:
        clip = real_cut(*args, **kwargs)
        # A lazy C-extension import reports the stop as ImportError, and the caller shrugs it off.
        try:
            try:
                app._raise_stopped(signal.SIGTERM, None)
            except app.WorkerStopped as stop:
                raise ImportError("initialization failed") from stop
        except ImportError:
            pass
        return clip

    def counting_run_review(*args: object, **kwargs: object) -> object:
        review_calls.append("review")
        return real_run_review(*args, **kwargs)

    def execute_in_process(request: AnalyzeRequest, config: Config, run: runs.Run, *rest: object):
        try:
            app._run_stages(request, config, run, rest[0], rest[1])
        except app.WorkerStopped:
            return app._Interruption(code, "прерван")
        raise AssertionError("the swallowed stop let the stage run to its record")

    monkeypatch.setattr(app, "_stop_signal", None)
    monkeypatch.setattr(app.clips, "cut", cut_swallowing_the_stop)
    monkeypatch.setattr(app.review, "run_review", counting_run_review)
    monkeypatch.setattr(app, "_execute", execute_in_process)

    analysis = analyze(AnalyzeRequest(str(wav), layers=LISTENED), config, backend=FakeBackend())

    result, run = analysis.result, analysis.run
    assert review_calls == []
    assert stage_record(run, "acoustics")["entry"]["status"] == "completed"
    assert not (run.path / "work" / "stage-audio_review.json").exists()
    assert result["stages"]["audio_review"]["status"] == "failed"
    assert ("audio_review", code) in codes(result)
    assert result["status"] == "partial"


def test_a_backend_that_swallows_the_cancel_leaves_no_completed_review(
    config: Config, wav: Path, pid_dir: Path
) -> None:
    token = CancelToken()
    cancelled_at: list[float] = []

    def cancel_when_the_child_runs() -> None:
        wait_for(pid_dir / "child")
        cancelled_at.append(time.monotonic())
        token.cancel()

    canceller = threading.Thread(target=cancel_when_the_child_runs)
    canceller.start()
    analysis = analyze(
        AnalyzeRequest(str(wav), layers=LISTENED),
        config,
        backend=StopSwallowingBackend(pid_dir=str(pid_dir)),
        cancel=token,
    )
    returned_at = time.monotonic()
    canceller.join()

    assert returned_at - cancelled_at[0] <= 2.0
    assert process_gone(pid_of(pid_dir / "worker"), wait_seconds=1.0)
    assert not (analysis.run.path / "work" / "stage-audio_review.json").exists()
    assert analysis.result["stages"]["audio_review"]["status"] == "failed"
    assert ("audio_review", "cancelled") in codes(analysis.result)
    assert analysis.result["findings"] == []
    assert manifest(analysis.run)["complete"] is False


def test_a_token_cancelled_before_the_start_runs_no_worker(
    config: Config, wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_worker(*args: object) -> None:
        raise AssertionError("the worker must not start")

    monkeypatch.setattr(app, "_start_worker", no_worker)
    token = CancelToken()
    token.cancel()

    analysis = analyze(AnalyzeRequest(str(wav), layers=ALL_LAYERS), config, cancel=token)

    result = analysis.result
    assert result["status"] == "failed"
    assert ("prepare", "cancelled") in codes(result)
    assert result["stages"]["prepare"]["status"] == "failed"
    assert all(result["stages"][name]["status"] == "skipped" for name in ("asr", "acoustics"))
    assert result["source"]["sha256"] == sha256_of(wav)
    assert not (analysis.run.path / "source.json").exists()
    assert manifest(analysis.run)["complete"] is False


# --- request ID, cache and the lock ----------------------------------------------------------


def test_a_repeat_returns_the_cached_result_and_a_changed_file_is_a_mismatch(
    config: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/tone.wav")
    request = AnalyzeRequest(str(wav), request_id="req-cache", layers=("measurements",))
    first = analyze(request, config)

    def no_worker(*args: object) -> None:
        raise AssertionError("a cached result runs no stage")

    monkeypatch.setattr(app, "_start_worker", no_worker)
    again = analyze(request, config)

    assert again.reused is True and again.run.run_id == first.run.run_id
    assert again.result == first.result
    result_file = first.run.path / "result.json"
    before = sha256_of(result_file)
    make_wav("recordings/tone.wav", frequency=440.0)
    with pytest.raises(runs.RequestMismatch):
        analyze(request, config)
    assert sha256_of(result_file) == before


@dataclass
class TunedFakeBackend(FakeBackend):
    temperature: float = 0.0

    def describe(self) -> BackendDescription:
        description = super().describe()
        return dataclasses.replace(description, settings={"temperature": self.temperature})


def test_the_cache_identity_holds_the_resolved_asr_revision_and_the_backend_settings(
    config: Config, wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = AnalyzeRequest(str(wav), request_id="req-identity", layers=("measurements",))
    model = {"repo": config.asr.model, "files": {}}
    monkeypatch.setattr(app.doctor, "snapshot", lambda *_: (model | {"revision": "aaa"}, None))
    analyze(request, config, backend=TunedFakeBackend())

    assert analyze(request, config, backend=TunedFakeBackend()).reused is True
    with pytest.raises(runs.RequestMismatch):
        analyze(request, config, backend=TunedFakeBackend(temperature=0.7))
    monkeypatch.setattr(app.doctor, "snapshot", lambda *_: (model | {"revision": "bbb"}, None))
    with pytest.raises(runs.RequestMismatch):
        analyze(request, config, backend=TunedFakeBackend())


def test_an_asr_model_missing_from_the_cache_has_no_resolved_revision(config: Config) -> None:
    preflight = app._Preflight(probe=None, audio_sha256="0" * 64, description=None)  # type: ignore[arg-type]
    request = AnalyzeRequest("x", request_id="r", layers=("content",))

    identity = app._identity(request, config, preflight)

    assert identity["model"]["asr"]["resolved_revision"] is None
    assert identity["model"]["backend"] == "none"


def test_a_repeat_of_a_cancelled_request_is_refused_not_rerun(
    config: Config, wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = CancelToken()
    token.cancel()
    request = AnalyzeRequest(str(wav), request_id="req-cancel", layers=("measurements",))
    analyze(request, config, cancel=token)

    def no_worker(*args: object) -> None:
        raise AssertionError("an incomplete run is not run again")

    monkeypatch.setattr(app, "_start_worker", no_worker)
    with pytest.raises(RunIncomplete):
        analyze(request, config)


def test_a_second_analysis_in_the_same_process_is_busy_without_a_queue(
    config: Config, wav: Path, pid_dir: Path
) -> None:
    token = CancelToken()
    outcome: list[object] = []

    def first() -> None:
        request = AnalyzeRequest(str(wav), request_id="req-a", layers=LISTENED)
        outcome.append(
            analyze(request, config, backend=SlowFakeBackend(pid_dir=str(pid_dir)), cancel=token)
        )

    thread = threading.Thread(target=first)
    thread.start()
    try:
        wait_for(pid_dir / "child")
        started = time.monotonic()
        with pytest.raises(Busy):
            analyze(AnalyzeRequest(str(wav), request_id="req-b"), config)
        assert time.monotonic() - started < 1.0
    finally:
        token.cancel()
        thread.join()
    index = config.data_root / "runs" / "by-request"
    assert not (index / hashlib.sha256(b"req-b").hexdigest()).exists()
    assert (index / hashlib.sha256(b"req-a").hexdigest()).exists()
    assert outcome


def test_an_analysis_in_another_process_makes_this_one_busy(config: Config, wav: Path) -> None:
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "time.sleep(600)\n",
            str(config.data_root / app.LOCK_FILE_NAME),
        ],
        stdout=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == b"locked\n"
        with pytest.raises(Busy):
            analyze(AnalyzeRequest(str(wav)), config)
    finally:
        holder.kill()
        holder.wait()
    assert not (config.data_root / "runs").exists()
    assert analyze(AnalyzeRequest(str(wav), layers=("measurements",)), config).result


def test_the_lock_is_released_after_an_error(config: Config, make_wav: WavFactory) -> None:
    with pytest.raises(app.paths.PathDenied):
        analyze(AnalyzeRequest(str(make_wav("outside.wav"))), config)
    fd = os.open(config.data_root / app.LOCK_FILE_NAME, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


# --- the audio backend -----------------------------------------------------------------------


def test_synthetic_smoke_with_the_test_double_is_complete_and_marked_synthetic(
    config: Config, wav: Path
) -> None:
    analysis = analyze(AnalyzeRequest(str(wav), layers=LISTENED), config, backend=FakeBackend())

    result, run = analysis.result, analysis.run
    assert result["status"] == "complete", result["stages"]
    assert result["stages"]["audio_review"]["status"] == "completed"
    assert result["provenance"]["synthetic"] is True
    assert result["provenance"]["note"].startswith("synthetic")
    assert result["provenance"]["backend"] == doctor.TEST_DOUBLE
    assert result["capabilities"]["validated_ru"] is False
    assert result["assessment_status"]["diction"]["status"] == "assessed"
    assert result["findings"] and result["unlocalized"]
    assert result["coverage"]["unreviewed"] == []
    assert result["provenance"]["limitations"] == [
        review.MODEL_LIMITATION + "only the passed fragment was heard",
        review.MODEL_UNCERTAINTY + "the first word may be a name",
    ]
    markdown = (run.path / "report.md").read_text(encoding="utf-8").splitlines()
    marker = next(i for i, line in enumerate(markdown) if "Синтетический прогон" in line)
    assert marker < markdown.index("## Этапы")
    steps = [step["name"] for step in stage_record(run, "audio_review")["steps"]]
    assert steps == ["review", "findings"]
    findings.validate(result, run=run)


FAKE_WHISPER = """
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list
    temperature: float = 0.0


@dataclass
class Info:
    language: str = "ru"
    language_probability: float = 1.0


class WhisperModel:
    \"\"\"Test double: a confident phrase over whatever it is given, as the real model does
    on digital silence; proves the pipeline contract, never that anything was heard.\"\"\"

    def __init__(self, path, **settings):
        pass

    def transcribe(self, audio, **parameters):
        words = [Word(0.3, 0.95, " выдумка", 0.93)]
        return iter([Segment(0, 0, 0.0, 0.95, " выдумка", [1], -0.1, 0.9, 0.01, words)]), Info()
"""


def fake_asr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Config, Path]:
    """The spawned worker imports `faster_whisper` from `sys.path`, so a test double package
    placed first stands in for the model; the snapshot is a fake one in a fake HF cache.
    Returns the config and a recording of 1 s of digital silence, then 0.5 s of a tone."""
    fake = tmp_path / "fake-packages" / "faster_whisper"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(FAKE_WHISPER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(fake.parent))
    snapshot = tmp_path / "hf" / "models--o--n" / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents=True)
    for name in doctor.WEIGHT_FILES:
        (snapshot / name).write_bytes(b"not a model")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    root, data_root = tmp_path / "recordings", tmp_path / "data"
    root.mkdir()
    data_root.mkdir()
    config = Config(allowed_roots=(root,), data_root=data_root, asr=Asr(model="o/n"))
    t = np.arange(8_000) / 16_000
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16)
    wav = root / "silence-then-tone.wav"
    with wave.open(str(wav), "wb") as out:  # 1 s of digital silence, then 0.5 s of a tone
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16_000)
        out.writeframes(np.concatenate([np.zeros(16_000, np.int16), tone]).tobytes())
    return config, wav


def test_a_confident_phrase_over_silence_reaches_the_result_only_as_an_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, wav = fake_asr(tmp_path, monkeypatch)

    analysis = analyze(
        AnalyzeRequest(str(wav), layers=("content", "sound")), config, backend=FakeBackend()
    )

    result, run = analysis.result, analysis.run
    assert result["stages"]["asr"]["status"] == "completed", result["stages"]
    assert result["stages"]["audio_review"]["status"] == "completed", result["stages"]
    assert result["transcript"]["suspect_segments"] == [
        {"start": 0.0, "end": 0.95, "signals": ["silent_span"], "certainty": "uncertain"}
    ]
    assert app.HALLUCINATION_LIMITATION in result["provenance"]["limitations"]
    assert result["findings"][0]["category"] == "pronunciation"
    assert result["transcript"]["corrections"] == []
    assert "выдумка" in (run.path / "asr" / "raw.json").read_text(encoding="utf-8")
    for name in ("result.json", "report.md"):
        assert "выдумка" not in (run.path / name).read_text(encoding="utf-8"), name
    findings.validate(result, run=run)


def test_an_invalid_backend_response_fails_the_review_once_and_gives_partial(
    config: Config, wav: Path
) -> None:
    analysis = analyze(
        AnalyzeRequest(str(wav), layers=LISTENED), config, backend=FakeBackend(text="not json")
    )

    result, run = analysis.result, analysis.run
    assert result["stages"]["audio_review"]["status"] == "failed"
    assert ("audio_review", findings.RESPONSE_INVALID) in codes(result)
    assert result["findings"] == [] and result["status"] == "partial"
    assert result["assessment_status"]["diction"]["status"] == "not_assessed"
    assert len(list((run.path / "review").glob("*/provenance.json"))) == 1
    findings.validate(result, run=run)


def test_a_failing_backend_call_is_recorded_and_not_repeated(
    config: Config, wav: Path, tmp_path: Path
) -> None:
    calls = tmp_path / "calls"
    backend = RaisingBackend(calls_file=str(calls))

    analysis = analyze(AnalyzeRequest(str(wav), layers=LISTENED), config, backend=backend)

    result = analysis.result
    assert result["stages"]["audio_review"]["status"] == "failed"
    assert ("audio_review", app.REVIEW_CALL_FAILED) in codes(result)
    assert calls.read_text() == "call\n"
    assert result["provenance"]["synthetic"] is True
    provenance = list((analysis.run.path / "review").glob("*/provenance.json"))
    assert len(provenance) == 1
    assert json.loads(provenance[0].read_text())["response"]["valid"] is False


@dataclass
class RecordingBackend(FakeBackend):
    """The test double with a window limit; it logs every request it gets (the worker is another
    process) and answers invalid or fails at the call it is told to."""

    log: str = ""
    max_audio_seconds: float | None = None
    invalid_at: int | None = None
    fail_at: int | None = None

    def describe(self) -> BackendDescription:
        description = super().describe()
        if self.max_audio_seconds is None:
            return description
        limits = {review.MAX_AUDIO_SECONDS_LIMIT: self.max_audio_seconds}
        return dataclasses.replace(description, limits=limits)

    def review(self, request: ReviewRequest) -> RawReviewResponse:
        fields = {f.name: getattr(request, f.name) for f in dataclasses.fields(request)}
        fields["payload"] = hashlib.sha256(request.payload).hexdigest()
        fields["response_fields"] = list(request.response_fields)
        with open(self.log, "a", encoding="utf-8") as out:
            out.write(json.dumps(fields, ensure_ascii=False) + "\n")
        call = len(self.requests)
        self.requests.append(request)
        if call == self.fail_at:
            raise RuntimeError("backend is down")
        if call == self.invalid_at:
            return RawReviewResponse("not json")
        return RawReviewResponse(self.text, self.request_id)


def logged(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def long_wav(make_wav: WavFactory, config: Config) -> Path:
    return make_wav("recordings/long.wav", seconds=11.0)


WINDOWS = [(0.0, 5.0), (3.0, 8.0), (6.0, 11.0)]


def test_a_long_recording_is_reviewed_window_by_window_and_covered_in_full(
    config: Config, long_wav: Path, tmp_path: Path
) -> None:
    log = tmp_path / "requests.jsonl"
    backend = RecordingBackend(log=str(log), max_audio_seconds=5.0)

    analysis = analyze(AnalyzeRequest(str(long_wav), layers=LISTENED), config, backend=backend)

    result, run = analysis.result, analysis.run
    requests = logged(log)
    assert result["status"] == "complete", result["errors"]
    assert result["coverage"]["reviewed"] == [{"start": 0.0, "end": 11.0}]
    assert result["coverage"]["unreviewed"] == []
    clips = [a for a in result["artifacts"] if a["kind"] == "clip"]
    assert [(c["source_start"], c["source_end"]) for c in clips] == WINDOWS
    assert [r["clip_id"] for r in requests] == [c["id"] for c in clips]
    assert all(r["duration_s"] <= 5.0 for r in requests)
    assert result["provenance"]["audio_payload_sha256"] == requests[0]["payload"]
    report_md = (run.path / "report.md").read_text(encoding="utf-8")
    assert (
        "SHA-256 audio payload первого окна из 3 (payload каждого окна — в артефактах "
        f"review_provenance): {requests[0]['payload']};"
    ) in report_md
    assert len(result["findings"]) == 3 and len(result["unlocalized"]) == 3
    assert (
        result["provenance"]["limitations"].count(
            review.MODEL_LIMITATION + "only the passed fragment was heard"
        )
        == 3
    )
    findings.validate(result, run=run)


def test_an_invalid_window_leaves_its_uncovered_part_unreviewed_and_the_result_partial(
    config: Config, long_wav: Path, tmp_path: Path
) -> None:
    log = tmp_path / "requests.jsonl"
    backend = RecordingBackend(log=str(log), max_audio_seconds=5.0, invalid_at=1)

    analysis = analyze(AnalyzeRequest(str(long_wav), layers=LISTENED), config, backend=backend)

    result = analysis.result
    assert len(logged(log)) == 3
    assert result["stages"]["audio_review"]["status"] == "completed"
    assert result["status"] == "partial"
    assert result["coverage"]["unreviewed"] == [{"start": 5.0, "end": 6.0}]
    assert ("audio_review", findings.RESPONSE_INVALID) in codes(result)
    findings.validate(result, run=analysis.run)


def test_a_failed_window_call_stops_the_review_and_leaves_the_rest_unreviewed(
    config: Config, long_wav: Path, tmp_path: Path
) -> None:
    log = tmp_path / "requests.jsonl"
    backend = RecordingBackend(log=str(log), max_audio_seconds=5.0, fail_at=1)

    analysis = analyze(AnalyzeRequest(str(long_wav), layers=LISTENED), config, backend=backend)

    result = analysis.result
    assert len(logged(log)) == 2
    assert result["stages"]["audio_review"]["status"] == "completed"
    assert ("audio_review", app.REVIEW_CALL_FAILED) in codes(result)
    assert result["coverage"]["unreviewed"] == [{"start": 5.0, "end": 11.0}]
    assert result["status"] == "partial"
    findings.validate(result, run=analysis.run)


def test_a_failed_first_window_call_fails_the_review_without_calling_the_rest(
    config: Config, long_wav: Path, tmp_path: Path
) -> None:
    log = tmp_path / "requests.jsonl"
    backend = RecordingBackend(log=str(log), max_audio_seconds=5.0, fail_at=0)

    analysis = analyze(AnalyzeRequest(str(long_wav), layers=LISTENED), config, backend=backend)

    result = analysis.result
    assert len(logged(log)) == 1
    assert result["stages"]["audio_review"]["status"] == "failed"
    assert ("audio_review", app.REVIEW_CALL_FAILED) in codes(result)
    assert result["coverage"]["unreviewed"] == [{"start": 0.0, "end": 11.0}]


@dataclass
class WindowHangingBackend(RecordingBackend):
    """Answers the windows before `hang_at`, then hangs on a child like `SlowFakeBackend`."""

    hang_at: int = 1
    pid_dir: str = ""

    def review(self, request: ReviewRequest) -> RawReviewResponse:
        if len(self.requests) == self.hang_at:
            write_pid(Path(self.pid_dir) / "worker", os.getpid())
            child = str(Path(self.pid_dir) / "child")
            proc.run([sys.executable, "-c", CHILD, child], timeout=900, max_output_bytes=1024)
        return super().review(request)


@pytest.mark.parametrize("hang_at", [1, 0], ids=["after a window", "on the first window"])
def test_an_interrupted_review_keeps_the_windows_it_finished(
    config: Config, long_wav: Path, tmp_path: Path, pid_dir: Path, hang_at: int
) -> None:
    log = tmp_path / "requests.jsonl"
    backend = WindowHangingBackend(
        log=str(log), max_audio_seconds=5.0, hang_at=hang_at, pid_dir=str(pid_dir)
    )
    token = CancelToken()

    def cancel_when_the_child_runs() -> None:
        wait_for(pid_dir / "child")
        token.cancel()

    canceller = threading.Thread(target=cancel_when_the_child_runs)
    canceller.start()
    analysis = analyze(
        AnalyzeRequest(str(long_wav), layers=("sound",)), config, backend=backend, cancel=token
    )
    canceller.join()

    result, run = analysis.result, analysis.run
    stage = result["stages"]["audio_review"]
    assert len(logged(log)) == hang_at  # the hung call never logs
    assert ("audio_review", "cancelled") in codes(result)
    assert result["status"] == "partial"
    assert stage["reason"] == app.CANCELLED_REASON
    if hang_at:
        assert stage["status"] == "completed"
        assert result["coverage"]["reviewed"] == [{"start": 0.0, "end": 5.0}]
        assert result["coverage"]["unreviewed"] == [{"start": 5.0, "end": 11.0}]
        assert len(result["findings"]) == 1
        assert result["provenance"]["audio_payload_sha256"] == logged(log)[0]["payload"]
        assert stage_record(run, "audio_review")[app.CHECKPOINT] is True
    else:
        assert stage["status"] == "failed"
        assert result["coverage"]["reviewed"] == [] and result["findings"] == []
        assert not (run.path / "work" / "stage-audio_review.json").exists()
    findings.validate(result, run=run)


def test_a_backend_limit_that_is_not_a_length_fails_the_review_before_any_call(
    config: Config, long_wav: Path, tmp_path: Path
) -> None:
    log = tmp_path / "requests.jsonl"
    backend = RecordingBackend(log=str(log), max_audio_seconds=3.0)

    analysis = analyze(AnalyzeRequest(str(long_wav), layers=LISTENED), config, backend=backend)

    assert logged(log) == []
    assert analysis.result["stages"]["audio_review"]["status"] == "failed"
    assert "max_audio_seconds" in analysis.result["stages"]["audio_review"]["reason"]


def test_the_first_review_pass_is_blind_to_the_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, wav = fake_asr(tmp_path, monkeypatch)
    log = tmp_path / "requests.jsonl"
    backend = RecordingBackend(log=str(log))

    analysis = analyze(
        AnalyzeRequest(str(wav), layers=("content", "sound"), task="прочитать", intent="ясно"),
        config,
        backend=backend,
    )

    result, run = analysis.result, analysis.run
    assert result["stages"]["asr"]["status"] == "completed", result["stages"]
    assert "выдумка" in (run.path / "asr" / "raw.json").read_text(encoding="utf-8")
    [request] = logged(log)
    assert set(request) == {
        "clip_id",
        "payload",
        "sample_rate",
        "channels",
        "sample_format",
        "duration_s",
        "prompt",
        "prompt_sha256",
        "task",
        "intent",
        "response_fields",
    }
    assert request["prompt"] == review.load_prompt().text
    assert (request["task"], request["intent"]) == ("прочитать", "ясно")
    assert "выдумка" not in json.dumps(request, ensure_ascii=False)
    trusted = backend_request(request).trusted_fields
    assert set(trusted) == {"clip_id", "duration_s", "task", "intent"}
    report_md = (run.path / "report.md").read_text(encoding="utf-8")
    assert f"SHA-256 audio payload: {request['payload']};" in report_md
    findings.validate(result, run=run)


def backend_request(logged_request: dict) -> ReviewRequest:
    return ReviewRequest(**{**logged_request, "payload": b"", "response_fields": ()})


# --- stdout and orphans ----------------------------------------------------------------------


def test_analyze_writes_nothing_to_stdout(config: Config, wav: Path) -> None:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from sound_scout.app import AnalyzeRequest, analyze\n"
        "from sound_scout.config import Asr, Config\n"
        "config = Config(allowed_roots=(Path(sys.argv[1]),), data_root=Path(sys.argv[2]),\n"
        "                asr=Asr(model='absent/model'))\n"
        "analyze(AnalyzeRequest(sys.argv[3]), config)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(wav.parent), str(config.data_root), str(wav)],
        capture_output=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout == b""
    assert list(config.data_root.glob("runs/*/result.json"))


# --- the development recording ---------------------------------------------------------------


@pytest.mark.slow
def test_the_dev_recording_runs_real_asr_with_backend_none(tmp_path: Path) -> None:
    dev_audio = os.environ.get("SOUND_SCOUT_DEV_AUDIO")
    if not dev_audio:
        pytest.skip("SOUND_SCOUT_DEV_AUDIO is not set")
    recording = Path(dev_audio).resolve()
    if not recording.is_file():
        pytest.skip("SOUND_SCOUT_DEV_AUDIO does not name a file")
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = Config(allowed_roots=(recording.parent,), data_root=data_root, backend="none")
    _, reason = doctor.snapshot(config.asr.model, config.asr.revision)
    if reason is not None:
        pytest.skip(f"the ASR model is not in the local cache: {reason}")
    try:
        analysis = analyze(AnalyzeRequest(str(recording), layers=ALL_LAYERS), config)
        result = analysis.result
        stages = {name: stage["status"] for name, stage in result["stages"].items()}
        summary = (stages, result["status"], sorted(codes(result)))
        assert stages["prepare"] == "completed", summary
        assert stages["asr"] == "completed", summary
        assert stages["acoustics"] == "completed", summary
        assert stages["audio_review"] == "unavailable", summary
        assert result["status"] == "partial", summary
        assert result["assessment_status"]["content"]["status"] == "assessed", summary
        for name in ("diction", "prosody"):
            assert result["assessment_status"][name]["status"] == "not_assessed", summary
        errors = findings.validation_errors(result, run=analysis.run)
        assert errors == [], len(errors)
    finally:
        shutil.rmtree(data_root)


# --- telemetry -------------------------------------------------------------------------------

ANALYZE_CHILD = """
import json, os, sys
from pathlib import Path
from sound_scout.app import AnalyzeRequest, analyze
from sound_scout.config import Config, Network

base = Path({base!r})
config = Config(
    allowed_roots=(base / "recordings",),
    data_root=base / "data",
    network=Network(allow={allow}),
)
result = analyze(AnalyzeRequest({wav!r}, layers=("measurements",)), config).result
print(json.dumps({{
    "acoustics": result["stages"]["acoustics"]["status"],
    "parent_loaded": "onnxruntime" in sys.modules,
    "parent_env": os.environ.get("ORT_DISABLE_TELEMETRY"),
}}))
"""


@pytest.mark.parametrize("allow", [False, True])
def test_analyze_worker_runs_the_real_vad_without_onnxruntime_telemetry(
    tmp_path: Path, make_wav: WavFactory, allow: bool
) -> None:
    """R09: the analyze worker, not the inherited environment, switches onnxruntime telemetry
    off before the speech detector loads it, whatever `network.allow` says."""
    (tmp_path / "recordings").mkdir()
    (tmp_path / "data").mkdir()
    wav = make_wav("recordings/tone.wav", seconds=1.0)
    home, temp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    temp.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "ORT_DISABLE_TELEMETRY"}
    env.update(HOME=str(home), TMPDIR=str(temp))
    script = ANALYZE_CHILD.format(base=str(tmp_path), wav=str(wav), allow=allow)

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"acoustics": "completed", "parent_loaded": False, "parent_env": None}
    assert not (home / ".cache/Microsoft/DeveloperTools/.onnxruntime").exists()
    assert not (temp / ".ses").exists()
    assert sorted(home.rglob("*")) == [] and sorted(temp.rglob("*")) == []
