"""The jobs runner (#98): FIFO, `runner.lock`, requeue on `Busy`, cancel marker, lease, recover.

The analysis substitutes below stand in for `app.analyze` where the test is about the queue, the
locks and the cancel contract, not about what the tool hears; the tests marked `needs_ffmpeg`
run the real `app.analyze` on synthesized audio.
"""

import fcntl
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from sound_scout import app, config, jobs, runs
from sound_scout.config import Config
from sound_scout.jobs import Binding, JobError, SubmitRequest
from tests.conftest import WavFactory
from tests.test_app import SlowFakeBackend, codes, pid_of, wait_for
from tests.test_jobs import ABSENT_ASR_MODEL, job_file, needs_ffmpeg, set_job
from tests.test_proc import process_gone

POLL = 0.02


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
    """Stands in for `app.analyze`: claims a real run for the request ID and returns it."""

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
        if until_cancelled:
            deadline = time.monotonic() + 10
            while not cancel.cancelled:
                assert time.monotonic() < deadline, "the cancel marker never reached the token"
                time.sleep(POLL)
            return app.Analysis({}, run, False)
        return app.Analysis({}, runs.complete_run(run), False)

    return analyze


def runner_thread(cfg: Config) -> threading.Thread:
    thread = threading.Thread(
        target=jobs.run_runner, args=(cfg,), kwargs={"exit_when_idle": True, "poll_seconds": POLL}
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
        runner = runner_thread(cfg)
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
    job, attempt = jobs._claim_next(cfg)

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
