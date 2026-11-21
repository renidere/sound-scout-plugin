"""The jobs runner (#98): FIFO, `runner.lock`, requeue on `Busy`, cancel marker, lease, recover;
the plan the runner applies and the explicit retry (#107).

The analysis substitutes below stand in for `app.analyze` where the test is about the queue, the
locks and the cancel contract, not about what the tool hears; the tests marked `needs_ffmpeg`
run the real `app.analyze` on synthesized audio.
"""

import dataclasses
import fcntl
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sound_scout import app, config, ingest, jobs, report, runs
from sound_scout.config import Config
from sound_scout.jobs import Binding, JobError, SubmitRequest
from tests.conftest import WavFactory
from tests.test_app import SlowFakeBackend, codes, pid_of, wait_for
from tests.test_jobs import ABSENT_ASR_MODEL, baseline_run, job_file, needs_ffmpeg, set_job
from tests.test_proc import process_gone

POLL = 0.02
BUSY_POLL = 2.0


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    (tmp_path / "recordings").mkdir()
    (tmp_path / "data").mkdir()
    return Config(
        allowed_roots=(tmp_path / "recordings",),
        data_root=tmp_path / "data",
        asr=config.Asr(model=ABSENT_ASR_MODEL),
    )


def queue(cfg: Config, wav: Path, **fields: object) -> str:
    return jobs.submit(SubmitRequest(str(wav), **fields), Binding(), cfg)["job_id"]


def substitute(
    calls: list[str], *, until_cancelled: bool = False, fail_first: bool = False
) -> Callable[..., app.Analysis]:
    """Stands in for `app.analyze`: claims a real run for the request ID and returns it with the
    `source` of the file it was given, as the real result records it."""

    def analyze(
        request: app.AnalyzeRequest,
        config: Config,
        *,
        backend: object = None,
        cancel: app.CancelToken | None = None,
    ) -> app.Analysis:
        calls.append(request.request_id)
        if fail_first and len(calls) == 1:
            raise RuntimeError(f"unexpected failure reading {request.path}")
        run = runs.open_run(request.request_id, {}, config.data_root)
        result = {"source": {"sha256": ingest.input_sha256(request.path, config)}}
        if until_cancelled:
            deadline = time.monotonic() + 10
            while not cancel.cancelled:
                assert time.monotonic() < deadline, "the cancel marker never reached the token"
                time.sleep(POLL)
            return app.Analysis(result, run, False)
        return app.Analysis(result, runs.complete_run(run), False)

    return analyze


def runner_thread(cfg: Config, poll_seconds: float = POLL) -> threading.Thread:
    thread = threading.Thread(
        target=jobs.run_runner,
        args=(cfg,),
        kwargs={"exit_when_idle": True, "poll_seconds": poll_seconds},
    )
    thread.start()
    return thread


def wait_until(condition: Callable[[], bool], seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(POLL)


def test_the_runner_takes_queued_jobs_oldest_first_each_with_its_attempt_request_id(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/a.wav")
    job_ids = [queue(cfg, wav, request_id=f"r-{n}") for n in range(3)]
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{job_id}-a1" for job_id in job_ids]
    for job_id in job_ids:
        job = job_file(cfg, job_id)
        assert job["state"] == "finished" and job["state_reason"] is None
        [attempt] = job["attempts"]
        assert attempt["number"] == 1 and attempt["request_id"] == f"{job_id}-a1"
        assert attempt["outcome"] == "finished" and attempt["ended_at"]
        assert attempt["run_id"] == jobs._run_id_of(f"{job_id}-a1", cfg)
        lease = attempt["lease"]
        assert lease["pid"] == os.getpid() and lease["attempt"] == 1
        assert lease["boot_id"] == jobs._boot_id() and lease["process_start"] is not None
        assert jobs.status(job_id, cfg)["run_id"] == attempt["run_id"]


def test_a_second_runner_is_refused_by_runner_lock_not_analyze_lock(cfg: Config) -> None:
    jobs_dir = cfg.data_root / "jobs"
    jobs_dir.mkdir(mode=0o700)
    fd = os.open(jobs_dir / "runner.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(JobError) as refused:
            jobs.run_runner(cfg, exit_when_idle=True)
        assert refused.value.code == "runner_running"
    finally:
        os.close(fd)
    assert not (cfg.data_root / "analyze.lock").exists()
    jobs.run_runner(cfg, exit_when_idle=True)


@needs_ffmpeg
def test_busy_puts_the_job_back_in_its_place_in_the_queue(
    cfg: Config, make_wav: WavFactory
) -> None:
    first = queue(cfg, make_wav("recordings/a.wav", seconds=1.5))
    second = queue(cfg, make_wav("recordings/b.wav", seconds=1.5, frequency=330.0))

    with app._analysis_lock(cfg.data_root):
        # The runner waits BUSY_POLL after a requeue, so its next claim cannot land between
        # the reads below and show the job `running` for a moment.
        runner = runner_thread(cfg, poll_seconds=BUSY_POLL)
        wait_until(lambda: (job_file(cfg, first)["state_reason"] or "").startswith("analyze.lock"))
        time.sleep(10 * POLL)
        for job_id in (first, second):
            job = job_file(cfg, job_id)
            assert job["state"] == "queued" and job["attempts"] == []
        assert jobs.status(first, cfg)["state"] == "queued"
    runner.join(timeout=120)

    assert not runner.is_alive()
    started = []
    for job_id in (first, second):
        job = job_file(cfg, job_id)
        assert job["state"] == "finished", job["state_reason"]
        [attempt] = job["attempts"]
        assert attempt["request_id"] == f"{job_id}-a1"
        started.append(attempt["started_at"])
        assert jobs.result(job_id, cfg)["status"] == "partial"
    assert started == sorted(started)


def test_a_cancel_marker_stops_the_running_attempt_through_the_cancel_token(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"), request_id="r-1")
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls, until_cancelled=True))
    runner = runner_thread(cfg)
    wait_until(lambda: calls == [f"{job_id}-a1"])
    assert jobs.status(job_id, cfg)["state"] == "running"

    assert jobs.cancel(job_id, cfg)["cancel"] == "requested"
    requested = time.monotonic()
    runner.join(timeout=10)
    stopped = time.monotonic()

    assert not runner.is_alive()
    assert stopped - requested < 2.0
    job = job_file(cfg, job_id)
    assert job["state"] == "cancelled"
    assert job["state_reason"].startswith("cancelled while running")
    assert job["attempts"][0]["outcome"] == "cancelled"
    assert job["attempts"][0]["run_id"] == jobs._run_id_of(f"{job_id}-a1", cfg)
    assert jobs.status(job_id, cfg)["cancel_requested"] is True


@pytest.mark.parametrize("cancel_in_the_window", [False, True], ids=["busy", "busy-then-cancel"])
def test_a_busy_attempt_is_requeued_unless_a_cancel_landed_before_its_end(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, cancel_in_the_window: bool
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    store, armed = jobs._store, []

    def store_with_a_cancel_first(data_root: Path):
        if armed:
            armed.clear()
            jobs.cancel(job_id, cfg)
        return store(data_root)

    def busy(request: app.AnalyzeRequest, config: Config, **_: object) -> app.Analysis:
        if cancel_in_the_window:
            armed.append(True)
        raise app.Busy("analyze.lock is held")

    monkeypatch.setattr(jobs, "_store", store_with_a_cancel_first)
    monkeypatch.setattr(app, "analyze", busy)
    job, attempt = jobs._claim_next(cfg, jobs.code_fingerprint())

    state = jobs._run_attempt(app, job, attempt, cfg)

    job = job_file(cfg, job_id)
    assert job["state"] == state
    if not cancel_in_the_window:
        assert state == "queued" and job["attempts"] == []
        assert job["state_reason"].startswith("analyze.lock is held")
        return
    assert state == "cancelled"
    assert job["state_reason"] == "cancelled while waiting for analyze.lock"
    [ended] = job["attempts"]
    assert ended["outcome"] == "cancelled" and ended["ended_at"]


@needs_ffmpeg
def test_a_cancel_marker_stops_a_running_real_analyze_through_the_cancel_token(
    cfg: Config, make_wav: WavFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    job_id = queue(cfg, make_wav("recordings/a.wav"), layers=("measurements", "sound"))
    real_analyze = app.analyze

    def with_hung_backend(request: app.AnalyzeRequest, config: Config, **kwargs: object):
        backend = SlowFakeBackend(pid_dir=str(pid_dir))
        return real_analyze(request, config, backend=backend, cancel=kwargs["cancel"])

    monkeypatch.setattr(app, "analyze", with_hung_backend)
    runner = runner_thread(cfg)
    try:
        wait_for(pid_dir / "child", seconds=60)
        assert jobs.status(job_id, cfg)["state"] == "running"

        assert jobs.cancel(job_id, cfg)["cancel"] == "requested"
        requested = time.monotonic()
        runner.join(timeout=30)
        stopped = time.monotonic()
    finally:
        (cfg.data_root / "jobs" / job_id / "cancel.json").touch()
        runner.join(timeout=60)

    assert not runner.is_alive()
    assert stopped - requested < 3.0
    assert process_gone(pid_of(pid_dir / "worker"), wait_seconds=1.0)
    assert process_gone(pid_of(pid_dir / "child"), wait_seconds=1.0)
    job = job_file(cfg, job_id)
    assert job["state"] == "cancelled"
    assert job["attempts"][0]["outcome"] == "cancelled"
    answer = jobs.result(job_id, cfg)
    assert answer["job_state"] == "cancelled" and answer["status"] == "partial"
    run_dir = cfg.data_root / answer["run_dir"]
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert ("audio_review", "cancelled") in codes(result)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False


@needs_ffmpeg
def test_jobs_status_merges_state_current_stage_and_finished_stages_of_a_real_running_job(
    cfg: Config, make_wav: WavFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    config_path = tmp_path / "sound-scout.toml"
    config_path.write_text(
        f"allowed_roots = [{json.dumps(str(cfg.allowed_roots[0]))}]\n"
        f"data_root = {json.dumps(str(cfg.data_root))}\n"
        f"[asr]\nmodel = {json.dumps(ABSENT_ASR_MODEL)}\n",
        encoding="utf-8",
    )
    job_id = queue(cfg, make_wav("recordings/a.wav"), layers=("measurements", "sound"))
    real_analyze = app.analyze

    def with_hung_backend(request: app.AnalyzeRequest, config: Config, **kwargs: object):
        backend = SlowFakeBackend(pid_dir=str(pid_dir))
        return real_analyze(request, config, backend=backend, cancel=kwargs["cancel"])

    monkeypatch.setattr(app, "analyze", with_hung_backend)
    runner = runner_thread(cfg)
    try:
        wait_for(pid_dir / "child", seconds=60)
        shown = run_jobs_cli(config_path, "status", job_id, "--json")
        plain = run_jobs_cli(config_path, "status", job_id).stdout.splitlines()
        jobs.cancel(job_id, cfg)
        runner.join(timeout=30)
    finally:
        (cfg.data_root / "jobs" / job_id / "cancel.json").touch()
        runner.join(timeout=60)

    assert shown.returncode == 0, shown.stderr
    running = json.loads(shown.stdout)
    assert running["state"] == "running"
    assert running["current_stage"]["stage"] == "audio_review"
    assert running["finished_stages"] == [
        {"stage": "prepare", "status": "completed"},
        {"stage": "acoustics", "status": "completed"},
    ]
    assert "state: running" in plain
    assert any(line.startswith("current_stage: ") and '"audio_review"' in line for line in plain)
    ended = jobs.status(job_id, cfg)
    assert ended["state"] == "cancelled" and ended["current_stage"] is None
    assert ended["finished_stages"] == running["finished_stages"]


def replace_with_other_audio(wav: Path, make_wav: WavFactory) -> None:
    """Put other synthesized audio at `wav`'s path, as an editor saving over it would."""
    other = make_wav("other.wav", seconds=1.5, frequency=440.0)
    os.replace(other, wav)


def assert_failed_input_changed(cfg: Config, job_id: str, wav: Path) -> None:
    job = job_file(cfg, job_id)
    assert job["state"] == "failed"
    [attempt] = job["attempts"]
    assert attempt["outcome"] == "failed" and attempt["input_changed"] is True
    assert attempt["run_id"] is not None
    reason = jobs.status(job_id, cfg)["state_reason"]
    assert reason.startswith("input_changed: ")
    for private in (str(wav), wav.name, str(cfg.allowed_roots[0]), str(cfg.data_root)):
        assert private not in reason
    with pytest.raises(JobError) as refused:
        jobs.result(job_id, cfg)
    assert refused.value.code == "no_result"


@needs_ffmpeg
def test_an_input_replaced_after_submit_fails_the_job_input_changed_without_its_result(
    cfg: Config, make_wav: WavFactory, tmp_path: Path
) -> None:
    wav = make_wav("recordings/a.wav", seconds=1.5)
    job_id = queue(cfg, wav)
    submitted = job_file(cfg, job_id)["request"]["input"]["sha256"]
    replace_with_other_audio(wav, make_wav)
    assert ingest.input_sha256(wav, cfg) != submitted

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert_failed_input_changed(cfg, job_id, wav)
    run_dir = cfg.data_root / runs.RUNS_DIR_NAME / job_file(cfg, job_id)["attempts"][0]["run_id"]
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["source"]["sha256"] != submitted


@needs_ffmpeg
def test_an_input_replaced_after_analyze_checked_it_fails_the_job_input_changed(
    cfg: Config, make_wav: WavFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/a.wav", seconds=1.5)
    job_id = queue(cfg, wav)
    submitted = job_file(cfg, job_id)["request"]["input"]["sha256"]
    real_input_sha256 = ingest.input_sha256

    def replaced_after_the_check(path: str, config: Config) -> str:
        checked = real_input_sha256(path, config)
        replace_with_other_audio(wav, make_wav)
        return checked

    monkeypatch.setattr(ingest, "input_sha256", replaced_after_the_check)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert_failed_input_changed(cfg, job_id, wav)
    run_dir = cfg.data_root / runs.RUNS_DIR_NAME / job_file(cfg, job_id)["attempts"][0]["run_id"]
    source = json.loads((run_dir / ingest.SOURCE_FILE_NAME).read_text(encoding="utf-8"))
    assert source["sha256"] != submitted


def test_a_source_json_of_another_input_fails_the_job_input_changed(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = queue(cfg, wav)
    analyze = substitute([])

    def decoded_another_input(request: app.AnalyzeRequest, config: Config, **kwargs: object):
        analysis = analyze(request, config, **kwargs)
        assert (
            analysis.result["source"]["sha256"]
            == job_file(cfg, job_id)["request"]["input"]["sha256"]
        )
        source = {"sha256": "0" * 64}
        (analysis.run.path / ingest.SOURCE_FILE_NAME).write_text(json.dumps(source), "utf-8")
        return analysis

    monkeypatch.setattr(app, "analyze", decoded_another_input)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert_failed_input_changed(cfg, job_id, wav)


@needs_ffmpeg
@pytest.mark.parametrize("replaced", [True, False], ids=["replaced", "unchanged"])
def test_result_answers_a_recovered_attempt_only_when_its_run_read_the_submitted_input(
    cfg: Config, make_wav: WavFactory, replaced: bool
) -> None:
    wav = make_wav("recordings/a.wav", seconds=1.5)
    job_id = queue(cfg, wav)
    request = job_file(cfg, job_id)["request"]
    if replaced:
        replace_with_other_audio(wav, make_wav)
    app.analyze(
        app.AnalyzeRequest(
            str(wav),
            request_id=f"{job_id}-a1",
            layers=tuple(request["layers"]),
            task=request["task"],
            intent=request["intent"],
        ),
        cfg,
    )
    lease = jobs._lease(1) | {"boot_id": "another-boot"}
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": None, "lease": lease}
    set_job(cfg, job_id, state="running", attempts=[attempt])
    [recovered] = jobs.recover(cfg)
    assert recovered["interrupted_attempt"] == 1

    if replaced:
        with pytest.raises(JobError) as refused:
            jobs.result(job_id, cfg)
        assert refused.value.code == "no_result"
    else:
        answer = jobs.result(job_id, cfg)
        assert answer["attempt"] == 1 and answer["job_state"] == "queued"


@needs_ffmpeg
def test_an_unchanged_input_ends_the_job_finished_with_its_result(
    cfg: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav", seconds=1.5)
    job_id = queue(cfg, wav)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    job = job_file(cfg, job_id)
    assert job["state"] == "finished" and job["state_reason"] is None
    assert job["attempts"][0]["input_changed"] is False
    answer = jobs.result(job_id, cfg)
    assert answer["job_state"] == "finished" and answer["attempt"] == 1
    run_dir = cfg.data_root / answer["run_dir"]
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["source"]["sha256"] == job["request"]["input"]["sha256"]


@needs_ffmpeg
def test_an_input_changed_attempt_does_not_hide_the_result_of_an_earlier_attempt(
    cfg: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav", seconds=1.5)
    job_id = queue(cfg, wav)
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)
    first = jobs.result(job_id, cfg)
    replace_with_other_audio(wav, make_wav)
    set_job(cfg, job_id, state="queued")

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    job = job_file(cfg, job_id)
    assert job["state"] == "failed" and job["state_reason"].startswith("input_changed: ")
    assert [(a["outcome"], a["input_changed"]) for a in job["attempts"]] == [
        ("finished", False),
        ("failed", True),
    ]
    answer = jobs.result(job_id, cfg)
    assert answer["attempt"] == 1 and answer["run_dir"] == first["run_dir"]
    assert answer["job_state"] == "failed"


def test_a_refused_or_crashed_attempt_fails_its_job_and_the_runner_goes_on(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/a.wav")
    crashed = queue(cfg, wav, request_id="r-1")
    gone = queue(cfg, make_wav("recordings/gone.wav", frequency=330.0), request_id="r-2")
    (cfg.allowed_roots[0] / "gone.wav").unlink()
    calls: list[str] = []
    real_analyze = app.analyze

    def crash_then_real(request: app.AnalyzeRequest, config: Config, **kwargs: object):
        if not calls:
            return substitute(calls, fail_first=True)(request, config, **kwargs)
        calls.append(request.request_id)
        return real_analyze(request, config, **kwargs)

    monkeypatch.setattr(app, "analyze", crash_then_real)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{crashed}-a1", f"{gone}-a1"]
    first = job_file(cfg, crashed)
    assert first["state"] == "failed"
    assert first["state_reason"] == "internal_error: RuntimeError"
    assert first["attempts"][0]["outcome"] == "failed"
    second = job_file(cfg, gone)
    assert second["state"] == "failed" and second["attempts"][0]["outcome"] == "failed"
    assert second["state_reason"].split(":")[0] in ("path_denied", "input_not_found")
    assert str(cfg.allowed_roots[0]) not in second["state_reason"]


def other_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.mark.parametrize(
    ("change", "recovered"),
    [
        ({}, False),
        ({"boot_id": "another-boot"}, True),
        ({"pid": "dead"}, True),
        ({"process_start": -1}, True),
    ],
    ids=["alive", "another-boot", "pid-gone", "pid-reused"],
)
def test_recover_takes_only_attempts_whose_lease_owner_is_gone(
    cfg: Config, make_wav: WavFactory, change: dict, recovered: bool
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    lease = jobs._lease(1) | change
    if lease["pid"] == "dead":
        lease["pid"] = other_pid()
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": None, "lease": lease}
    set_job(cfg, job_id, state="running", attempts=[attempt])

    answer = jobs.recover(cfg)

    job = job_file(cfg, job_id)
    if not recovered:
        assert answer == [] and job["state"] == "running" and "outcome" not in job["attempts"][0]
        return
    assert answer == [
        {
            "job_id": job_id,
            "interrupted_attempt": 1,
            "state": "queued",
            "next_attempt": f"{job_id}-a2",
        }
    ]
    assert job["state"] == "queued" and f"{job_id}-a2" in job["state_reason"]
    assert job["attempts"][0]["outcome"] == "interrupted"
    assert jobs.recover(cfg) == []


def test_recover_takes_an_attempt_whose_lease_owner_is_a_zombie(
    cfg: Config, make_wav: WavFactory
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        lease = jobs._lease(1) | {"pid": child.pid, "process_start": None}
        wait_until(lambda: jobs._process_start(child.pid) is not None)
        lease["process_start"] = jobs._process_start(child.pid)
        attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": None, "lease": lease}
        set_job(cfg, job_id, state="running", attempts=[attempt])
        assert jobs.recover(cfg) == []

        child.stdin.close()
        stat = Path(f"/proc/{child.pid}/stat")
        wait_until(lambda: stat.read_text().rpartition(")")[2].split()[0] == "Z")
        assert jobs._process_start(child.pid) is None

        [answer] = jobs.recover(cfg)
    finally:
        child.wait()

    assert answer["state"] == "queued" and answer["next_attempt"] == f"{job_id}-a2"
    assert job_file(cfg, job_id)["attempts"][0]["outcome"] == "interrupted"


def test_recover_ends_an_interrupted_job_cancelled_when_a_cancel_was_requested(
    cfg: Config, make_wav: WavFactory
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    lease = jobs._lease(1) | {"boot_id": "another-boot"}
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": None, "lease": lease}
    set_job(cfg, job_id, state="running", attempts=[attempt])
    jobs.cancel(job_id, cfg)

    [answer] = jobs.recover(cfg)

    assert answer["state"] == "cancelled" and answer["next_attempt"] is None
    assert job_file(cfg, job_id)["state"] == "cancelled"


RUNNER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from sound_scout import app, jobs
    from sound_scout.config import Config

    def blocked(request, config, *, backend=None, cancel=None):
        Path(sys.argv[3]).write_text(request.request_id, encoding="utf-8")
        time.sleep(600)

    app.analyze = blocked
    cfg = Config(allowed_roots=(Path(sys.argv[1]),), data_root=Path(sys.argv[2]))
    jobs.run_runner(cfg, poll_seconds=0.02)
    """
)


def test_recover_after_a_killed_runner_queues_a_visible_new_attempt(
    cfg: Config, make_wav: WavFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    script, started = tmp_path / "runner.py", tmp_path / "started"
    script.write_text(RUNNER_SCRIPT, encoding="utf-8")
    runner = subprocess.Popen(
        [sys.executable, str(script), str(cfg.allowed_roots[0]), str(cfg.data_root), str(started)]
    )
    try:
        wait_until(started.exists)
        assert started.read_text(encoding="utf-8") == f"{job_id}-a1"
        job = job_file(cfg, job_id)
        assert job["state"] == "running" and job["attempts"][0]["lease"]["pid"] == runner.pid
        assert jobs.recover(cfg) == []
    finally:
        runner.kill()
        runner.wait()
    assert job_file(cfg, job_id)["state"] == "running"

    config_path = tmp_path / "sound-scout.toml"
    config_path.write_text(
        f"allowed_roots = [{json.dumps(str(cfg.allowed_roots[0]))}]\n"
        f"data_root = {json.dumps(str(cfg.data_root))}\n",
        encoding="utf-8",
    )
    recovered = run_jobs_cli(config_path, "recover", "--json")
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout) == [
        {
            "job_id": job_id,
            "interrupted_attempt": 1,
            "state": "queued",
            "next_attempt": f"{job_id}-a2",
        }
    ]
    assert jobs.status(job_id, cfg)["state"] == "queued"
    assert job_file(cfg, job_id)["attempts"][0]["outcome"] == "interrupted"

    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{job_id}-a2"]
    job = job_file(cfg, job_id)
    assert job["state"] == "finished"
    assert [(a["number"], a["outcome"]) for a in job["attempts"]] == [
        (1, "interrupted"),
        (2, "finished"),
    ]
    assert run_jobs_cli(config_path, "recover").stdout == ""


def run_jobs_cli(config_path: Path, *args: str, path: str | None = None):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env[config.CONFIG_PATH_ENV] = str(config_path)
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        [sys.executable, "-m", "sound_scout", "jobs", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@needs_ffmpeg
def test_jobs_run_through_the_console_runs_a_queued_job_with_the_real_analyze(
    tmp_path: Path, make_wav: WavFactory
) -> None:
    for name in ("recordings", "data"):
        (tmp_path / name).mkdir()
    config_path = tmp_path / "sound-scout.toml"
    config_path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'recordings'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n"
        f"[asr]\nmodel = {json.dumps(ABSENT_ASR_MODEL)}\n",
        encoding="utf-8",
    )
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    submitted = run_jobs_cli(config_path, "submit", str(wav), "--json")
    job_id = json.loads(submitted.stdout)["job_id"]

    ran = run_jobs_cli(config_path, "run", "--exit-when-idle")

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == ""
    assert f"job {job_id} attempt 1: finished" in ran.stderr
    status = json.loads(run_jobs_cli(config_path, "status", job_id, "--json").stdout)
    assert status["state"] == "finished" and status["attempts"] == 1
    result = json.loads(run_jobs_cli(config_path, "result", job_id, "--json").stdout)
    assert result["job_state"] == "finished" and result["status"] == "partial"
    run_request = json.loads(
        (tmp_path / "data" / result["run_dir"] / "request.json").read_text(encoding="utf-8")
    )
    assert run_request["request_id"] == f"{job_id}-a1"


def test_runner_unit_argv_is_a_fixed_user_unit_with_the_config_and_no_other_environment(
    tmp_path: Path,
) -> None:
    environ = {"PATH": "/usr/bin:/bin", "HF_HOME": "/models/hf", "SECRET_TOKEN": "x"}

    argv = jobs.runner_unit_argv(tmp_path / "sound-scout.toml", environ, python="/venv/python")

    assert argv == [
        "systemd-run",
        "--user",
        "--unit=sound-scout-jobs-runner",
        "--collect",
        "--description=Sound Scout jobs runner",
        f"--setenv=SOUND_SCOUT_CONFIG={tmp_path / 'sound-scout.toml'}",
        "--setenv=PATH=/usr/bin:/bin",
        "--setenv=HF_HOME=/models/hf",
        "/venv/python",
        "-m",
        "sound_scout",
        "jobs",
        "run",
    ]
    assert Path(jobs.runner_unit_argv(Path("rel.toml"), {})[5].split("=", 2)[2]).is_absolute()


FAKE_SYSTEMD_RUN = """#!/bin/sh
printf '%s\\n' "$@" > "{record}"
{tail}
"""


@pytest.mark.parametrize("fails", [False, True], ids=["started", "refused"])
def test_jobs_start_runs_systemd_run_and_reports_its_refusal(tmp_path: Path, fails: bool) -> None:
    bin_dir, record = tmp_path / "bin", tmp_path / "argv"
    bin_dir.mkdir()
    tail = "echo 'Failed to start transient service unit: Unit already exists.' >&2; exit 1"
    fake = bin_dir / "systemd-run"
    fake.write_text(FAKE_SYSTEMD_RUN.format(record=record, tail=tail if fails else "exit 0"))
    fake.chmod(0o700)
    for name in ("recordings", "data"):
        (tmp_path / name).mkdir()
    config_path = tmp_path / "sound-scout.toml"
    config_path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'recordings'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n",
        encoding="utf-8",
    )

    started = run_jobs_cli(
        config_path, "start", "--json", path=f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    )

    passed = record.read_text(encoding="utf-8").splitlines()
    assert passed[:2] == ["--user", "--unit=sound-scout-jobs-runner"]
    assert f"--setenv=SOUND_SCOUT_CONFIG={config_path}" in passed
    assert passed[-5:] == [sys.executable, "-m", "sound_scout", "jobs", "run"]
    if fails:
        assert started.returncode == 1 and started.stdout == ""
        assert started.stderr.startswith("sound-scout jobs: runner_start_failed: ")
        assert "Unit already exists" in started.stderr
    else:
        assert started.returncode == 0, started.stderr
        assert json.loads(started.stdout) == {"unit": "sound-scout-jobs-runner", "started": True}


# --- plan modes and retry (#107) -------------------------------------------------------------


def recorder(seen: list[tuple[app.AnalyzeRequest, Config, object]]) -> Callable[..., app.Analysis]:
    """`substitute`, keeping the request, the configuration and the backend it was called with."""
    analyze = substitute([])

    def record(request: app.AnalyzeRequest, config: Config, **options: object) -> app.Analysis:
        seen.append((request, config, options.get("backend")))
        return analyze(request, config, **options)

    return record


@pytest.mark.parametrize(
    "change",
    [
        {"allowed_roots": ["/"]},
        {"network": {"allow": True}},
        {"backend": "llama"},
        {"capability_version": 2},
        {"stages": ["prepare", "asr", "audio_review", "acoustics"]},
        {"budgets": {"worker_deadline_seconds": 10_000}},
    ],
    ids=["roots", "network", "backend", "version", "stages", "budget-fields"],
)
def test_a_widened_plan_in_job_json_fails_the_job_plan_refused_before_analyze(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, change: dict
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    set_job(cfg, job_id, plan=job_file(cfg, job_id)["plan"] | change)
    seen: list = []
    monkeypatch.setattr(app, "analyze", recorder(seen))

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert seen == []
    job = job_file(cfg, job_id)
    assert job["state"] == "failed" and job["state_reason"].startswith("plan_refused: ")
    assert [attempt["outcome"] for attempt in job["attempts"]] == ["failed"]


def test_the_runner_applies_the_plan_with_budgets_narrowed_by_the_configuration_of_the_moment(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/a.wav")
    full = queue(cfg, wav, request_id="r-full")
    quick = queue(cfg, wav, request_id="r-quick", mode="quick")
    budgets = job_file(cfg, full)["plan"]["budgets"] | {"max_duration_seconds": 10_000}
    set_job(cfg, full, plan=job_file(cfg, full)["plan"] | {"budgets": budgets})
    narrowed = dataclasses.replace(
        cfg, budgets=dataclasses.replace(cfg.budgets, worker_deadline_seconds=30)
    )
    seen: list = []
    monkeypatch.setattr(app, "analyze", recorder(seen))

    jobs.run_runner(narrowed, exit_when_idle=True, poll_seconds=POLL)

    [(full_request, full_config, full_backend), (quick_request, _, quick_backend)] = seen
    assert full_request.layers == app.LAYERS
    assert quick_request.layers == ("content", "measurements")
    assert full_config.budgets.worker_deadline_seconds == 30
    assert full_config.budgets.max_duration_seconds == cfg.budgets.max_duration_seconds
    assert full_config.allowed_roots == cfg.allowed_roots and full_config.backend == "none"
    assert full_backend is None and quick_backend is None
    assert {job_file(cfg, job_id)["state"] for job_id in (full, quick)} == {"finished"}


def test_an_input_outside_the_roots_narrowed_after_submit_fails_the_job_path_denied(
    cfg: Config, make_wav: WavFactory, tmp_path: Path
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    (tmp_path / "other").mkdir()
    narrowed = dataclasses.replace(cfg, allowed_roots=(tmp_path / "other",))

    jobs.run_runner(narrowed, exit_when_idle=True, poll_seconds=POLL)

    job = job_file(cfg, job_id)
    assert job["state"] == "failed" and job["state_reason"].startswith("path_denied: ")


@needs_ffmpeg
def test_full_with_backend_none_marks_audio_review_unavailable_and_quick_skips_it(
    cfg: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav", seconds=1.5)
    full = queue(cfg, wav, request_id="r-full", mode="full")
    quick = queue(cfg, wav, request_id="r-quick", mode="quick")

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    def audio_review(job_id: str) -> dict:
        run_id = job_file(cfg, job_id)["attempts"][-1]["run_id"]
        path = cfg.data_root / runs.RUNS_DIR_NAME / run_id / report.RESULT_FILE_NAME
        return json.loads(path.read_text(encoding="utf-8"))["stages"]["audio_review"]

    assert audio_review(full)["status"] == "unavailable"
    assert audio_review(full)["reason"] == app.NO_BACKEND_REASON
    assert audio_review(quick)["status"] == "skipped"
    assert audio_review(quick)["reason"] == app.NOT_REQUESTED_REASON


def test_retry_of_a_failed_job_queues_its_next_attempt_with_the_same_ids(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"), request_id="r-1")
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls, fail_first=True))
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)
    delivery = {"state": "delivered", "tag": f"sound-scout-job:{job_id}"}
    set_job(cfg, job_id, delivery=delivery)
    before = job_file(cfg, job_id)

    answer = jobs.retry(job_id, cfg)

    assert answer["state"] == "queued" and answer["retry"] == "queued"
    assert answer["next_attempt"] == f"{job_id}-a2" and answer["retries"] == 1
    assert answer["job_id"] == job_id and answer["request_id"] == "r-1"
    job = job_file(cfg, job_id)
    assert job["retries"] == [
        {
            "requested_at": job["queued_at"],
            "after_state": "failed",
            "after_reason": "internal_error: RuntimeError",
            "next_attempt": f"{job_id}-a2",
        }
    ]
    assert job["state_reason"] == f"retry requested: attempt 2 ({job_id}-a2)"
    for kept in ("request_id", "request", "plan", "binding", "delivery", "submitted_at"):
        assert job[kept] == before[kept], kept
    assert job["queued_at"] > job["submitted_at"]

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{job_id}-a1", f"{job_id}-a2"]
    job = job_file(cfg, job_id)
    assert job["state"] == "finished" and job["delivery"] == delivery
    assert [(a["number"], a["request_id"], a["outcome"]) for a in job["attempts"]] == [
        (1, f"{job_id}-a1", "failed"),
        (2, f"{job_id}-a2", "finished"),
    ]
    assert jobs.status(job_id, cfg)["attempts"] == 2


def test_retry_of_a_job_cancelled_before_it_started_is_its_first_attempt(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    jobs.cancel(job_id, cfg)

    assert jobs.retry(job_id, cfg)["next_attempt"] == f"{job_id}-a1"
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{job_id}-a1"]
    assert job_file(cfg, job_id)["state"] == "finished"


@pytest.mark.parametrize("interrupted", [False, True], ids=["cancelled", "interrupted"])
def test_retry_of_a_cancelled_job_drops_its_cancel_marker_so_the_new_attempt_runs(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    lease = jobs._lease(1) | ({"boot_id": "another-boot"} if interrupted else {})
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": None, "lease": lease}
    set_job(cfg, job_id, state="running", attempts=[attempt])
    jobs.cancel(job_id, cfg)
    if interrupted:
        [recovered] = jobs.recover(cfg)
        assert recovered["state"] == "cancelled"
        assert job_file(cfg, job_id)["attempts"][0]["outcome"] == "interrupted"
    else:
        set_job(cfg, job_id, state="cancelled", attempts=[attempt | {"outcome": "cancelled"}])
    assert jobs.status(job_id, cfg)["cancel_requested"] is True

    answer = jobs.retry(job_id, cfg)

    assert answer["next_attempt"] == f"{job_id}-a2" and answer["cancel_requested"] is False
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)
    assert calls == [f"{job_id}-a2"]
    job = job_file(cfg, job_id)
    assert job["state"] == "finished"
    assert [a["outcome"] for a in job["attempts"]] == [
        "interrupted" if interrupted else "cancelled",
        "finished",
    ]


@pytest.mark.parametrize("state", ["queued", "running", "finished"])
def test_retry_of_a_job_that_is_queued_running_or_finished_is_not_retryable(
    cfg: Config, make_wav: WavFactory, state: str
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    set_job(cfg, job_id, state=state)
    path = cfg.data_root / "jobs" / job_id / "job.json"
    before = path.read_bytes()

    with pytest.raises(JobError) as refused:
        jobs.retry(job_id, cfg)

    assert refused.value.code == "not_retryable" and f"is {state}" in refused.value.message
    assert ("jobs recover" in refused.value.message) is (state == "running")
    assert path.read_bytes() == before


def test_retry_of_an_unknown_job_is_job_not_found(cfg: Config) -> None:
    with pytest.raises(JobError) as refused:
        jobs.retry("f" * 32, cfg)

    assert refused.value.code == "job_not_found"


@pytest.mark.parametrize("limit", ["queue", "staging"])
def test_retry_goes_through_admission_like_a_submit(
    cfg: Config, make_wav: WavFactory, limit: str
) -> None:
    first, second = make_wav("recordings/a.wav"), make_wav("recordings/b.wav", frequency=330.0)
    failed = queue(cfg, first, request_id="r-1")
    queue(cfg, second, request_id="r-2")
    set_job(cfg, failed, state="failed", state_reason="internal_error: RuntimeError")
    path = cfg.data_root / "jobs" / failed / "job.json"
    before = path.read_bytes()
    size = first.stat().st_size + second.stat().st_size
    tight = {"max_queued_jobs": 1} if limit == "queue" else {"max_staging_bytes": size - 1}

    with pytest.raises(JobError) as refused:
        jobs.retry(failed, dataclasses.replace(cfg, jobs=config.Jobs(**tight)))

    assert refused.value.code == f"{limit}_full"
    assert path.read_bytes() == before
    assert jobs.retry(failed, cfg)["state"] == "queued"


def test_a_retried_job_waits_behind_a_job_submitted_after_it(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/a.wav")
    first = queue(cfg, wav, request_id="r-1")
    later = queue(cfg, wav, request_id="r-2")
    set_job(cfg, first, state="failed", state_reason="internal_error: RuntimeError")
    jobs.retry(first, cfg)
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{later}-a1", f"{first}-a1"]


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_the_runner_never_retries_an_ended_job_on_its_own(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    set_job(cfg, job_id, state=state)
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)
    assert jobs.recover(cfg) == []

    assert calls == []
    job = job_file(cfg, job_id)
    assert job["state"] == state and job["attempts"] == [] and "retries" not in job


# --- the staging sweep of the runner (#122) --------------------------------------------------


def staging_cfg(cfg: Config) -> Config:
    root = cfg.data_root.parent / "staging"
    root.mkdir()
    return Config(
        allowed_roots=(*cfg.allowed_roots, root),
        data_root=cfg.data_root,
        asr=cfg.asr,
        jobs=config.Jobs(staging_root=root),
    )


def test_the_runner_removes_a_delivered_staging_copy_without_a_command(
    cfg: Config, make_wav: WavFactory
) -> None:
    cfg = staging_cfg(cfg)
    path = cfg.jobs.staging_root / f"{'a' * 32}.upload"
    make_wav("recordings/a.wav").rename(path)
    job_id = queue(cfg, path)
    delivered = {"state": "delivered", "tag": "t", "confirmed_by": "send"}
    set_job(cfg, job_id, state="finished", delivery=delivered)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert not path.exists()
    assert job_file(cfg, job_id)["staging_removed_at"]


def test_a_failing_sweep_does_not_stop_the_runner(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    from sound_scout import staging

    def failing_sweep(config: Config) -> list[dict]:
        raise OSError("disk gone")

    monkeypatch.setattr(staging, "sweep", failing_sweep)
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))
    job_id = queue(cfg, make_wav("recordings/a.wav"))

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{job_id}-a1"]
    assert job_file(cfg, job_id)["state"] == "finished"
    assert "staging sweep failed: OSError" in capfd.readouterr().err


def test_criterion_3_jobs_run_cleans_staging_by_the_retention_from_sound_scout_toml(
    tmp_path: Path, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("recordings", "staging", "data"):
        (tmp_path / name).mkdir()
    roots = [str(tmp_path / "recordings"), str(tmp_path / "staging")]
    config_path = tmp_path / "sound-scout.toml"
    config_path.write_text(
        f"allowed_roots = {json.dumps(roots)}\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n"
        f"[asr]\nmodel = {json.dumps(ABSENT_ASR_MODEL)}\n"
        f"[jobs]\nstaging_root = {json.dumps(roots[1])}\nstaging_retention_seconds = 600\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(config_path))
    cfg = config.load()
    assert cfg.jobs.staging_retention_seconds == 600

    def staged(name: str) -> tuple[Path, str]:
        path = tmp_path / "staging" / f"{name * 32}.upload"
        make_wav(f"recordings/{name}.wav").rename(path)
        return path, queue(cfg, path)

    old = (datetime.now(UTC) - timedelta(seconds=601)).isoformat()
    recent = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    expired, expired_id = staged("a")
    set_job(cfg, expired_id, state="failed", updated_at=old)
    young, young_id = staged("b")
    set_job(cfg, young_id, state="failed", updated_at=recent)

    ran = run_jobs_cli(config_path, "run", "--exit-when-idle")

    assert ran.returncode == 0, ran.stderr
    assert f"staging {expired.name} removed: retention" in ran.stderr
    assert not expired.exists() and job_file(cfg, expired_id)["staging_removed_at"]
    assert young.exists() and "staging_removed_at" not in job_file(cfg, young_id)


def test_retry_of_a_job_whose_staging_copy_was_removed_is_refused_with_a_clear_error(
    cfg: Config, make_wav: WavFactory
) -> None:
    from sound_scout import staging

    cfg = staging_cfg(cfg)
    path = cfg.jobs.staging_root / f"{'c' * 32}.upload"
    make_wav("recordings/c.wav").rename(path)
    job_id = queue(cfg, path)
    old = (
        datetime.now(UTC) - timedelta(seconds=cfg.jobs.staging_retention_seconds + 1)
    ).isoformat()
    set_job(
        cfg, job_id, state="failed", state_reason="internal_error: RuntimeError", updated_at=old
    )
    assert [item["reason"] for item in staging.sweep(cfg)] == ["retention"]
    job_json = cfg.data_root / "jobs" / job_id / "job.json"
    before = job_json.read_bytes()

    with pytest.raises(JobError) as refused:
        jobs.retry(job_id, cfg)

    assert refused.value.code == "not_retryable"
    assert "staging copy was removed" in refused.value.message
    assert "submit the recording again" in refused.value.message
    assert job_json.read_bytes() == before
    assert not path.exists()


# --- a control attempt: compared with its initial run before the job ends (#110) -------------
# Without an audio backend the real verdict is always `inconclusive`: these tests prove the order
# and the records of the runner, never that the tool hears an improvement (R01, R03).


def control_job(cfg: Config, make_wav: WavFactory, **fields: object) -> tuple[str, str]:
    initial = baseline_run(cfg, make_wav("recordings/initial.wav", seconds=1.5))
    wav = make_wav("recordings/control.wav", seconds=1.5, frequency=330.0)
    link = {"previous_run_id": initial, "criterion": "completeness_of_phrase"}
    return queue(cfg, wav, request_id="r-control", **link | fields), initial


@needs_ffmpeg
def test_a_control_job_without_a_backend_ends_finished_with_an_inconclusive_comparison(
    cfg: Config, make_wav: WavFactory
) -> None:
    job_id, initial = control_job(cfg, make_wav)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    job = job_file(cfg, job_id)
    assert job["state"] == "finished"
    [attempt] = job["attempts"]
    summary = attempt["compare"]
    assert summary["state"] == "completed" and summary["verdict"] == "inconclusive"
    assert summary["criterion"] == "completeness_of_phrase"
    assert {"code": "audio_review_missing", "attempt": "a"} in summary["reasons"]
    assert {"code": "audio_review_missing", "attempt": "b"} in summary["reasons"]
    assert summary["backend"] == {"a": "none", "b": "none"}
    assert summary["audio_review"] == {"a": "unavailable", "b": "unavailable"}
    assert summary["path"] == f"compare/{summary['compare_id']}.json"
    document = json.loads((cfg.data_root / summary["path"]).read_text(encoding="utf-8"))
    assert document["verdict"] == summary["verdict"]
    assert [item["code"] for item in document["reasons"]] == [
        item["code"] for item in summary["reasons"]
    ]
    assert document["attempts"]["a"]["run_id"] == initial
    assert document["attempts"]["b"]["run_id"] == attempt["run_id"]
    assert all(set(item) == {"code", "attempt"} for item in summary["reasons"])
    assert jobs.status(job_id, cfg)["compare"] == summary
    answer = jobs.result(job_id, cfg)
    assert answer["attempt"] == 1 and answer["compare"] == summary
    assert len(json.dumps(answer, ensure_ascii=False).encode("utf-8")) <= 4096


@needs_ffmpeg
def test_the_comparison_runs_before_the_job_is_finished_and_lands_in_the_same_write(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, initial = control_job(cfg, make_wav)
    real_compare = app.compare
    seen: list[dict] = []

    def watched(run_a: str, run_b: str, criterion: str, config: Config):
        seen.append(job_file(cfg, job_id))
        assert (run_a, criterion) == (initial, "completeness_of_phrase")
        return real_compare(run_a, run_b, criterion, config)

    monkeypatch.setattr(app, "compare", watched)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    [during] = seen
    assert during["state"] == "running" and "compare" not in during["attempts"][-1]
    job = job_file(cfg, job_id)
    assert job["state"] == "finished" and job["attempts"][-1]["compare"]["state"] == "completed"


@needs_ffmpeg
def test_an_initial_run_removed_after_submit_is_a_refused_comparison_of_a_finished_job(
    cfg: Config, make_wav: WavFactory
) -> None:
    job_id, initial = control_job(cfg, make_wav)
    shutil.rmtree(cfg.data_root / "runs" / initial)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    job = job_file(cfg, job_id)
    assert job["state"] == "finished"
    summary = job["attempts"][-1]["compare"]
    assert summary["state"] == "refused" and summary["code"] == "run_not_found"
    assert not (cfg.data_root / "compare").exists()


@needs_ffmpeg
def test_an_unexpected_comparison_error_is_failed_by_its_type_and_the_job_stays_finished(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _ = control_job(cfg, make_wav)

    def broken(*args: object) -> None:
        raise ValueError("a message that may carry a path")

    monkeypatch.setattr(app, "compare", broken)

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    job = job_file(cfg, job_id)
    assert job["state"] == "finished"
    assert job["attempts"][-1]["compare"] == {
        "state": "failed",
        "error": "internal_error: ValueError",
    }


def no_compare(calls: list[object]) -> Callable[..., None]:
    def compare(*args: object) -> None:
        calls.append(args)
        raise AssertionError("app.compare was called")

    return compare


@needs_ffmpeg
def test_a_cancelled_control_attempt_is_not_compared(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _ = control_job(cfg, make_wav)
    calls: list[str] = []
    compared: list[object] = []
    monkeypatch.setattr(app, "analyze", substitute(calls, until_cancelled=True))
    monkeypatch.setattr(app, "compare", no_compare(compared))
    runner = runner_thread(cfg)
    wait_until(lambda: calls == [f"{job_id}-a1"])

    jobs.cancel(job_id, cfg)
    runner.join(timeout=10)

    job = job_file(cfg, job_id)
    assert job["state"] == "cancelled" and compared == []
    assert job["attempts"][-1]["compare"] == {"state": "not_run", "reason": "cancelled"}


def test_an_ordinary_job_is_never_compared(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = queue(cfg, make_wav("recordings/a.wav"), request_id="r-1")
    compared: list[object] = []
    monkeypatch.setattr(app, "analyze", substitute([]))
    monkeypatch.setattr(app, "compare", no_compare(compared))

    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    job = job_file(cfg, job_id)
    assert job["state"] == "finished" and compared == []
    assert "compare" not in job["attempts"][-1]
    assert jobs.status(job_id, cfg)["compare"] is None
    assert jobs.status(job_id, cfg)["control"] is None


@needs_ffmpeg
def test_each_attempt_of_a_retried_control_job_has_its_own_comparison(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _ = control_job(cfg, make_wav)
    real_analyze = app.analyze
    calls: list[str] = []

    def crash_then_real(request: app.AnalyzeRequest, config: Config, **kwargs: object):
        calls.append(request.request_id)
        if len(calls) == 1:
            raise RuntimeError("unexpected failure")
        return real_analyze(request, config, **kwargs)

    monkeypatch.setattr(app, "analyze", crash_then_real)
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)
    assert job_file(cfg, job_id)["state"] == "failed"

    jobs.retry(job_id, cfg)
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    first, second = job_file(cfg, job_id)["attempts"]
    assert first["compare"] == {"state": "not_run", "reason": "failed"}
    assert second["compare"]["state"] == "completed"
    assert second["compare"]["verdict"] == "inconclusive"
    answer = jobs.result(job_id, cfg)
    assert answer["attempt"] == 2 and answer["compare"] == second["compare"]


# --- the runner never starts an attempt on code older than the installed code (#139) ----------


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of the installed package that `code_fingerprint` reads instead of the real one."""
    package = tmp_path / "installed" / "sound_scout"
    shutil.copytree(jobs.PACKAGE_DIR, package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(jobs, "PACKAGE_DIR", package)
    return package


def runner_lock_held(cfg: Config) -> bool:
    lock = cfg.data_root / "jobs" / "runner.lock"
    if not lock.exists():
        return False
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def edit_module(package: Path) -> None:
    with (package / "asr.py").open("a", encoding="utf-8") as module:
        module.write("\n# updated\n")


def edit_schema(package: Path) -> None:
    schema = package / "result.v1.schema.json"
    schema.write_text(schema.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def make_unreadable(package: Path) -> None:
    (package / "report.py").chmod(0)


def test_code_fingerprint_covers_every_packaged_file_but_not_pycache(installed: Path) -> None:
    import sound_scout

    real = jobs.code_fingerprint(Path(sound_scout.__file__).resolve().parent)
    assert jobs.code_fingerprint() == real
    (installed / "__pycache__").mkdir()
    (installed / "__pycache__" / "asr.cpython-313.pyc").write_bytes(b"compiled")
    assert jobs.code_fingerprint() == real
    edit_schema(installed)
    edited = jobs.code_fingerprint()
    assert edited != real
    (installed / "eval" / "new.py").write_text("", encoding="utf-8")
    assert jobs.code_fingerprint() not in (real, edited)


@pytest.mark.parametrize(
    "change", [edit_module, edit_schema, make_unreadable], ids=["module", "schema", "unreadable"]
)
def test_criteria_1_4_a_runner_waiting_on_older_code_starts_no_attempt_and_exits(
    cfg: Config,
    make_wav: WavFactory,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    installed: Path,
    change: Callable[[Path], None],
) -> None:
    if change is make_unreadable and os.geteuid() == 0:
        pytest.skip("root reads a file without permissions")
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))
    runner = threading.Thread(
        target=jobs.run_runner, args=(cfg,), kwargs={"poll_seconds": POLL}, daemon=True
    )
    runner.start()
    wait_until(lambda: runner_lock_held(cfg))

    change(installed)
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    queued = job_file(cfg, job_id)
    runner.join(timeout=30)

    assert not runner.is_alive(), "the runner did not exit on the changed code"
    assert calls == []
    job = job_file(cfg, job_id)
    assert job == queued and job["state"] == "queued" and job["attempts"] == []
    assert not runner_lock_held(cfg)
    err = capfd.readouterr().err
    reason = jobs.UNREADABLE_CODE if change is make_unreadable else jobs.CHANGED_CODE
    assert f"{reason}; job {job_id} stays queued" in err
    assert (jobs.CHANGED_CODE in err) is (change is not make_unreadable)


def test_a_runner_that_cannot_read_the_package_at_its_start_exits_and_leaves_jobs_queued(
    cfg: Config,
    make_wav: WavFactory,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    installed: Path,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads a file without permissions")
    calls: list[str] = []
    monkeypatch.setattr(app, "analyze", substitute(calls))
    job_id = queue(cfg, make_wav("recordings/a.wav"))
    queued = job_file(cfg, job_id)
    make_unreadable(installed)

    jobs.run_runner(cfg, poll_seconds=POLL)

    assert calls == [] and job_file(cfg, job_id) == queued
    assert not runner_lock_held(cfg)
    err = capfd.readouterr().err
    assert f"{jobs.UNREADABLE_CODE} (PermissionError)" in err
    assert jobs.CHANGED_CODE not in err

    (installed / "report.py").chmod(0o644)
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{job_id}-a1"] and job_file(cfg, job_id)["state"] == "finished"


def test_criteria_2_3_an_update_between_attempts_leaves_the_next_job_to_a_new_runner(
    cfg: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, installed: Path
) -> None:
    wav = make_wav("recordings/a.wav")
    first, second = (queue(cfg, wav, request_id=f"r-{n}") for n in range(2))
    calls: list[str] = []
    analyze = substitute(calls)

    def updated_while_running(request: app.AnalyzeRequest, config: Config, **kwargs: object):
        edit_module(installed)
        return analyze(request, config, **kwargs)

    monkeypatch.setattr(app, "analyze", updated_while_running)
    runner = threading.Thread(
        target=jobs.run_runner, args=(cfg,), kwargs={"poll_seconds": POLL}, daemon=True
    )
    runner.start()
    runner.join(timeout=30)

    assert not runner.is_alive(), "the runner did not exit on the changed code"
    assert calls == [f"{first}-a1"]
    assert job_file(cfg, first)["state"] == "finished"
    assert job_file(cfg, second)["state"] == "queued"
    assert job_file(cfg, second)["attempts"] == []

    monkeypatch.setattr(app, "analyze", analyze)
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)

    assert calls == [f"{first}-a1", f"{second}-a1"]
    [attempt] = job_file(cfg, second)["attempts"]
    assert attempt["outcome"] == "finished" and job_file(cfg, second)["state"] == "finished"


UPDATED_LOG = """

_log_of_the_old_code = _log


def _log(line: str) -> None:
    _log_of_the_old_code(f"[updated code] {line}")
"""


@needs_ffmpeg
def test_criterion_3_a_new_jobs_run_process_runs_the_waiting_job_on_the_updated_code(
    tmp_path: Path, make_wav: WavFactory
) -> None:
    site = tmp_path / "installed"
    shutil.copytree(
        jobs.PACKAGE_DIR, site / "sound_scout", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("recordings", "data"):
        (tmp_path / name).mkdir()
    config_path = tmp_path / "sound-scout.toml"
    config_path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'recordings'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n"
        f"[asr]\nmodel = {json.dumps(ABSENT_ASR_MODEL)}\n",
        encoding="utf-8",
    )
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env.update({config.CONFIG_PATH_ENV: str(config_path), "PYTHONPATH": str(site)})

    def cli(*args: str, **options: object):
        return subprocess.run(
            [sys.executable, "-m", "sound_scout", "jobs", *args],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            **options,
        )

    where = subprocess.run(
        [sys.executable, "-c", "import sound_scout; print(sound_scout.__file__)"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert Path(where.stdout.strip()).parent == site / "sound_scout"
    old_runner = subprocess.Popen(
        [sys.executable, "-m", "sound_scout", "jobs", "run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        cfg = Config(allowed_roots=(tmp_path / "recordings",), data_root=tmp_path / "data")
        wait_until(lambda: runner_lock_held(cfg))
        with (site / "sound_scout" / "jobs.py").open("a", encoding="utf-8") as module:
            module.write(UPDATED_LOG)
        wav = make_wav("recordings/tone.wav", seconds=1.5)
        job_id = json.loads(cli("submit", str(wav), "--json").stdout)["job_id"]
        old_out, old_err = old_runner.communicate(timeout=60)
    finally:
        if old_runner.poll() is None:
            old_runner.kill()
            old_runner.communicate()

    assert old_runner.returncode == 0, old_err
    assert f"job {job_id} stays queued" in old_err and "[updated code]" not in old_err
    waiting = json.loads(cli("status", job_id, "--json").stdout)
    assert waiting["state"] == "queued" and waiting["attempts"] == 0

    new_runner = cli("run", "--exit-when-idle", timeout=300)

    assert new_runner.returncode == 0, new_runner.stderr
    assert f"[updated code] job {job_id} attempt 1: finished" in new_runner.stderr
    done = json.loads(cli("status", job_id, "--json").stdout)
    assert done["state"] == "finished" and done["attempts"] == 1
