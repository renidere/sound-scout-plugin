import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sound_scout import app, config, ingest, jobs, paths, runs
from sound_scout.config import Budgets, Config, Jobs
from sound_scout.jobs import Binding, JobError, SubmitRequest
from tests.conftest import WavFactory

ABSENT_ASR_MODEL = "absent/model"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)


@pytest.fixture
def setup(tmp_path: Path) -> Config:
    (tmp_path / "recordings").mkdir()
    (tmp_path / "data").mkdir()
    return Config(allowed_roots=(tmp_path / "recordings",), data_root=tmp_path / "data")


def with_jobs(cfg: Config, **limits: int) -> Config:
    return dataclasses.replace(cfg, jobs=Jobs(**limits))


def job_file(cfg: Config, job_id: str) -> dict:
    return json.loads((cfg.data_root / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))


def set_job(cfg: Config, job_id: str, **fields: object) -> None:
    """Stand in for the runner (#98): it changes the state and appends attempts."""
    path = cfg.data_root / "jobs" / job_id / "job.json"
    job = json.loads(path.read_text(encoding="utf-8")) | fields
    path.write_text(json.dumps(job), encoding="utf-8")


def test_submit_writes_job_json_and_answers_within_a_second(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/voice.wav", seconds=30.0)
    binding = Binding(submitter="desktop", connection="local", profile="default", session_id="s-1")

    started = time.perf_counter()
    answer = jobs.submit(SubmitRequest(str(wav), request_id="r-1", task="фраза"), binding, setup)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert answer == {
        "job_id": answer["job_id"],
        "request_id": "r-1",
        "state": "queued",
        "reused": False,
    }
    job_dir = setup.data_root / "jobs" / answer["job_id"]
    assert oct(job_dir.stat().st_mode & 0o777) == "0o700"
    assert oct((job_dir / "job.json").stat().st_mode & 0o777) == "0o600"
    job = job_file(setup, answer["job_id"])
    assert job["state"] == "queued" and job["state_reason"] is None
    assert job["request"] == {
        "input": {
            "path": str(wav.resolve()),
            "size_bytes": wav.stat().st_size,
            "sha256": ingest.input_sha256(wav, setup),
        },
        "layers": sorted(app.LAYERS),
        "task": "фраза",
        "intent": None,
    }
    assert job["plan"] == {
        "stages": list(app.WORKER_STAGES),
        "budgets": dataclasses.asdict(Budgets()),
        "backend": "none",
    }
    assert job["binding"] == {
        "submitter": "desktop",
        "connection": "local",
        "profile": "default",
        "session_id": "s-1",
    }
    assert job["attempts"] == []
    assert "status" not in job and "coverage" not in job


def test_submit_does_not_check_the_input_the_runner_does(
    setup: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_audio = setup.allowed_roots[0] / "notes.wav"
    not_audio.write_text("это не аудио", encoding="utf-8")

    def no_decoder(*args: object, **kwargs: object) -> None:
        raise AssertionError("submit ran a decoder")

    monkeypatch.setattr(ingest, "probe", no_decoder)
    monkeypatch.setattr(ingest, "run", no_decoder)

    answer = jobs.submit(SubmitRequest(str(not_audio)), Binding(), setup)

    assert answer["state"] == "queued"
    assert job_file(setup, answer["job_id"])["binding"]["submitter"] == "cli"


def test_the_plan_names_only_the_stages_of_the_requested_layers(setup: Config) -> None:
    assert jobs.resolved_plan(("content",), setup)["stages"] == ["prepare", "asr"]
    assert jobs.resolved_plan(("sound", "measurements"), setup)["stages"] == [
        "prepare",
        "acoustics",
        "audio_review",
    ]


def test_the_same_request_returns_the_same_job_and_another_identity_is_a_mismatch(
    setup: Config, make_wav: WavFactory
) -> None:
    first, other = make_wav("recordings/a.wav"), make_wav("recordings/b.wav", frequency=330.0)
    answer = jobs.submit(SubmitRequest(str(first), request_id="r-1"), Binding(), setup)

    again = jobs.submit(SubmitRequest(str(first), request_id="r-1"), Binding(), setup)
    assert again == answer | {"reused": True}

    for request, binding in (
        (SubmitRequest(str(other), request_id="r-1"), Binding()),
        (SubmitRequest(str(first), request_id="r-1", intent="вопрос"), Binding()),
        (SubmitRequest(str(first), request_id="r-1"), Binding(session_id="another-chat")),
    ):
        with pytest.raises(runs.RequestMismatch, match="r-1"):
            jobs.submit(request, binding, setup)
    assert len(list((setup.data_root / "jobs").glob("*/job.json"))) == 1


def test_concurrent_submits_of_one_request_make_one_job(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    answers: list[dict] = []

    def submit() -> None:
        answers.append(jobs.submit(SubmitRequest(str(wav), request_id="r-1"), Binding(), setup))

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({answer["job_id"] for answer in answers}) == 1
    assert sorted(answer["reused"] for answer in answers) == [False, True, True, True]


def test_a_request_index_entry_without_its_job_is_refused(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    answer = jobs.submit(SubmitRequest(str(wav), request_id="r-1"), Binding(), setup)
    shutil.rmtree(setup.data_root / "jobs" / answer["job_id"])

    with pytest.raises(runs.RequestIndexCorrupt, match="directory is missing"):
        jobs.submit(SubmitRequest(str(wav), request_id="r-1"), Binding(), setup)


def test_status_reports_the_job_state_without_the_input_path(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = jobs.submit(SubmitRequest(str(wav), layers=("content",)), Binding(), setup)["job_id"]

    found = jobs.status(job_id, setup)

    assert found["job_id"] == job_id and found["state"] == "queued"
    assert found["stages"] == ["prepare", "asr"]
    assert found["attempts"] == 0 and found["run_id"] is None
    assert found["cancel_requested"] is False
    assert str(setup.allowed_roots[0]) not in json.dumps(found)


def test_cancel_moves_a_queued_job_to_cancelled_at_once(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = jobs.submit(SubmitRequest(str(wav)), Binding(), setup)["job_id"]

    cancelled = jobs.cancel(job_id, setup)

    assert cancelled["cancel"] == "cancelled" and cancelled["state"] == "cancelled"
    assert jobs.status(job_id, setup)["state"] == "cancelled"
    assert job_file(setup, job_id)["state"] == "cancelled"
    assert not (setup.data_root / "jobs" / job_id / "cancel.json").exists()
    assert jobs.cancel(job_id, setup)["cancel"] == "already_ended"


def test_cancel_of_a_running_job_leaves_the_marker_for_the_runner(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = jobs.submit(SubmitRequest(str(wav)), Binding(), setup)["job_id"]
    set_job(setup, job_id, state="running")

    first = jobs.cancel(job_id, setup)
    second = jobs.cancel(job_id, setup)

    assert first["cancel"] == second["cancel"] == "requested"
    assert first["state"] == "running" and first["cancel_requested"] is True
    marker = json.loads(
        (setup.data_root / "jobs" / job_id / "cancel.json").read_text(encoding="utf-8")
    )
    assert set(marker) == {"requested_at"}
    assert job_file(setup, job_id)["state"] == "running"


def test_admission_refuses_a_submit_above_max_queued_jobs(
    setup: Config, make_wav: WavFactory
) -> None:
    cfg = with_jobs(setup, max_queued_jobs=2)
    wavs = [make_wav(f"recordings/{n}.wav", frequency=200.0 + n) for n in range(4)]
    first = jobs.submit(SubmitRequest(str(wavs[0]), request_id="r-0"), Binding(), cfg)
    jobs.submit(SubmitRequest(str(wavs[1]), request_id="r-1"), Binding(), cfg)

    with pytest.raises(JobError) as refused:
        jobs.submit(SubmitRequest(str(wavs[2]), request_id="r-2"), Binding(), cfg)
    assert refused.value.code == "queue_full"
    assert "2 jobs are queued" in refused.value.message
    assert "jobs.max_queued_jobs = 2" in refused.value.message
    assert len(list((cfg.data_root / "jobs").glob("*/job.json"))) == 2

    assert jobs.submit(SubmitRequest(str(wavs[0]), request_id="r-0"), Binding(), cfg)["reused"]
    set_job(cfg, first["job_id"], state="running")
    assert jobs.submit(SubmitRequest(str(wavs[2]), request_id="r-2"), Binding(), cfg)["job_id"]


def test_admission_refuses_a_submit_above_max_staging_bytes(
    setup: Config, make_wav: WavFactory
) -> None:
    small, large = make_wav("recordings/small.wav"), make_wav("recordings/large.wav", seconds=3)
    size = small.stat().st_size
    cfg = with_jobs(setup, max_staging_bytes=size + large.stat().st_size - 1)
    job_id = jobs.submit(SubmitRequest(str(small)), Binding(), cfg)["job_id"]
    set_job(cfg, job_id, state="running")

    with pytest.raises(JobError) as refused:
        jobs.submit(SubmitRequest(str(large)), Binding(), cfg)
    assert refused.value.code == "staging_full"
    assert f"jobs.max_staging_bytes = {cfg.jobs.max_staging_bytes}" in refused.value.message

    set_job(cfg, job_id, state="finished")
    assert jobs.submit(SubmitRequest(str(large)), Binding(), cfg)["state"] == "queued"


def test_submit_refuses_a_bad_request_before_writing_anything(
    setup: Config, make_wav: WavFactory, tmp_path: Path
) -> None:
    wav = make_wav("recordings/a.wav")
    outside = make_wav("outside.wav")
    small = dataclasses.replace(setup, budgets=Budgets(max_input_bytes=wav.stat().st_size - 1))

    with pytest.raises(paths.PathDenied):
        jobs.submit(SubmitRequest(str(outside)), Binding(), setup)
    with pytest.raises(ingest.IngestError) as too_large:
        jobs.submit(SubmitRequest(str(wav)), Binding(), small)
    assert too_large.value.code == ingest.INPUT_TOO_LARGE
    for request, binding in (
        (SubmitRequest(str(wav), layers=("speech",)), Binding()),
        (SubmitRequest(str(wav), layers=("content", "content")), Binding()),
        (SubmitRequest(str(wav), request_id=""), Binding()),
        (SubmitRequest(str(wav)), Binding(submitter="")),
        (SubmitRequest(str(wav)), Binding(session_id="x" * 257)),
    ):
        with pytest.raises(JobError) as refused:
            jobs.submit(request, binding, setup)
        assert refused.value.code == "invalid_request"
    assert not (setup.data_root / "jobs").exists() or not list(
        (setup.data_root / "jobs").glob("*/job.json")
    )


@pytest.mark.parametrize("call", [jobs.status, jobs.result, jobs.cancel])
def test_an_unknown_job_id_is_job_not_found(setup: Config, call) -> None:
    for job_id in ("0" * 32, "../runs", ""):
        with pytest.raises(JobError) as refused:
            call(job_id, setup)
        assert refused.value.code == "job_not_found"


def test_result_of_a_job_without_a_run_is_no_result(setup: Config, make_wav: WavFactory) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    jobs.cancel(job_id, setup)

    with pytest.raises(JobError) as refused:
        jobs.result(job_id, setup)
    assert refused.value.code == "no_result"
    assert "cancelled" in refused.value.message


def test_a_submit_that_fails_after_creating_its_directory_leaves_no_job_behind(
    setup: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = make_wav("recordings/a.wav")

    def fail(*args: object) -> bool:
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(runs, "_publish_index_entry", fail)
        with pytest.raises(OSError, match="disk full"):
            jobs.submit(SubmitRequest(str(wav), request_id="r-1"), Binding(), setup)
    jobs_dir = setup.data_root / "jobs"
    assert [d for d in jobs_dir.iterdir() if runs.RUN_ID_PATTERN.fullmatch(d.name)] == []

    answer = jobs.submit(SubmitRequest(str(wav), request_id="r-1"), Binding(), setup)
    assert answer["state"] == "queued" and answer["reused"] is False


def test_admission_skips_a_job_directory_without_job_json(
    setup: Config, make_wav: WavFactory
) -> None:
    first = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)
    orphan = setup.data_root / "jobs" / runs.new_run_id()
    orphan.mkdir(mode=0o700)

    second = jobs.submit(SubmitRequest(str(make_wav("recordings/b.wav"))), Binding(), setup)

    assert second["state"] == "queued" and second["job_id"] != first["job_id"]
    assert not (orphan / "job.json").exists()


@needs_ffmpeg
def test_job_state_is_separate_from_result_status_and_coverage(
    setup: Config, make_wav: WavFactory
) -> None:
    cfg = dataclasses.replace(setup, asr=config.Asr(model=ABSENT_ASR_MODEL))
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    job_id = jobs.submit(SubmitRequest(str(wav)), Binding(), cfg)["job_id"]
    analysis = app.analyze(app.AnalyzeRequest(str(wav), request_id=f"{job_id}-a1"), cfg)
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": analysis.run.run_id}
    set_job(cfg, job_id, state="finished", attempts=[attempt])

    answer = jobs.result(job_id, cfg)

    assert answer["job_state"] == "finished"
    assert answer["status"] == analysis.result["status"] == "partial"
    assert answer["run_dir"] == f"runs/{analysis.run.run_id}"
    assert answer["coverage"]["reviewed_s"] == 0
    assert jobs.status(job_id, cfg)["run_id"] == analysis.run.run_id
    assert "status" not in jobs.status(job_id, cfg)


@needs_ffmpeg
def test_result_answers_with_the_latest_attempt_that_has_a_result(
    setup: Config, make_wav: WavFactory
) -> None:
    cfg = dataclasses.replace(setup, asr=config.Asr(model=ABSENT_ASR_MODEL))
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    job_id = jobs.submit(SubmitRequest(str(wav)), Binding(), cfg)["job_id"]
    analysis = app.analyze(app.AnalyzeRequest(str(wav), request_id=f"{job_id}-a1"), cfg)
    first = {"number": 1, "request_id": f"{job_id}-a1", "run_id": analysis.run.run_id}
    second = {"number": 2, "request_id": f"{job_id}-a2", "run_id": runs.new_run_id()}
    set_job(cfg, job_id, state="running", attempts=[first, second])

    answer = jobs.result(job_id, cfg)

    assert list(answer)[:4] == ["job_id", "job_state", "attempt", "run_dir"]
    assert answer["job_state"] == "running" and answer["attempt"] == 1
    assert answer["run_dir"] == f"runs/{analysis.run.run_id}"
    assert len(json.dumps(answer, ensure_ascii=False).encode("utf-8")) <= 4096


def test_the_stage_names_of_jobs_are_those_of_the_application_layer() -> None:
    assert jobs.LAYERS == app.LAYERS
    assert jobs.LAYER_STAGES == app.LAYER_STAGES
    assert jobs.WORKER_STAGES == app.WORKER_STAGES
    assert jobs.STAGE_RECORD == app.STAGE_RECORD
    assert jobs.CHECKPOINT == app.CHECKPOINT


def test_limits_answer_the_budgets_the_queue_limits_and_the_staging_root(setup: Config) -> None:
    staging = setup.allowed_roots[0]
    cfg = dataclasses.replace(
        setup,
        budgets=Budgets(max_duration_seconds=60, max_input_bytes=4096),
        jobs=Jobs(max_queued_jobs=2, max_staging_bytes=8192, staging_root=staging),
    )

    assert jobs.limits(cfg) == {
        "max_duration_seconds": 60,
        "max_input_bytes": 4096,
        "max_queued_jobs": 2,
        "max_staging_bytes": 8192,
        "staging_root": str(staging),
    }
    assert jobs.limits(setup)["staging_root"] is None


def write_stage_record(cfg: Config, run_id: str, stage: str, record: object) -> None:
    work = cfg.data_root / runs.RUNS_DIR_NAME / run_id / ingest.WORK_DIR_NAME
    work.mkdir(parents=True, exist_ok=True)
    text = record if isinstance(record, str) else json.dumps(record)
    (work / jobs.STAGE_RECORD.format(name=stage)).write_text(text, encoding="utf-8")


def stage_record(stage: str, status: str, **more: object) -> dict:
    return {"stage": stage, "entry": {"status": status, "reason": "/private/path"}, **more}


def test_finished_stages_are_empty_before_the_first_attempt(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]

    assert jobs.status(job_id, setup)["finished_stages"] == []


def test_finished_stages_of_a_running_attempt_come_from_its_run_through_the_index(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    run_id, request_id = runs.new_run_id(), f"{job_id}-a1"
    index = setup.data_root / runs.RUNS_DIR_NAME / runs.INDEX_DIR_NAME
    index.mkdir(parents=True)
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    (index / digest).write_text(run_id, encoding="utf-8")
    write_stage_record(setup, run_id, "asr", stage_record("asr", "failed"))
    write_stage_record(setup, run_id, "prepare", stage_record("prepare", "completed"))
    write_stage_record(setup, run_id, "acoustics", "{torn")
    write_stage_record(
        setup, run_id, "audio_review", stage_record("audio_review", "completed", checkpoint=True)
    )
    attempt = {"number": 1, "request_id": request_id, "run_id": None}
    set_job(setup, job_id, state="running", attempts=[attempt])

    found = jobs.status(job_id, setup)

    assert found["finished_stages"] == [
        {"stage": "prepare", "status": "completed"},
        {"stage": "asr", "status": "failed"},
    ]
    assert "/private/path" not in json.dumps(found)


def test_finished_stages_follow_the_latest_attempt_and_its_plan(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = jobs.submit(SubmitRequest(str(wav), layers=("content",)), Binding(), setup)["job_id"]
    old, new = runs.new_run_id(), runs.new_run_id()
    write_stage_record(setup, old, "prepare", stage_record("prepare", "completed"))
    write_stage_record(setup, new, "acoustics", stage_record("acoustics", "completed"))
    write_stage_record(setup, new, "prepare", stage_record("prepare", "failed"))
    attempts = [
        {"number": 1, "request_id": f"{job_id}-a1", "run_id": old},
        {"number": 2, "request_id": f"{job_id}-a2", "run_id": new},
    ]
    set_job(setup, job_id, state="finished", attempts=attempts)

    assert jobs.status(job_id, setup)["finished_stages"] == [
        {"stage": "prepare", "status": "failed"}
    ]


# --- the real console entry point ------------------------------------------------------------


def run_cli(config_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env[config.CONFIG_PATH_ENV] = str(config_path)
    return subprocess.run(
        [sys.executable, "-m", "sound_scout", "jobs", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    for name in ("recordings", "data"):
        (tmp_path / name).mkdir()
    path = tmp_path / "sound-scout.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'recordings'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n"
        "[jobs]\nmax_queued_jobs = 1\n",
        encoding="utf-8",
    )
    return path


def test_jobs_through_the_console_entry(config_file: Path, make_wav: WavFactory) -> None:
    wav = make_wav("recordings/voice.wav", seconds=2.0)
    other = make_wav("recordings/other.wav", frequency=330.0)

    submitted = run_cli(
        config_file, "submit", str(wav), "--request-id", "r-1", "--session-id", "s-1", "--json"
    )
    assert submitted.returncode == 0, submitted.stderr
    answer = json.loads(submitted.stdout)
    job_id = answer["job_id"]
    assert answer["state"] == "queued" and answer["reused"] is False
    job = json.loads(
        (config_file.parent / "data" / "jobs" / job_id / "job.json").read_text(encoding="utf-8")
    )
    assert job["binding"] == {
        "submitter": "cli",
        "connection": None,
        "profile": None,
        "session_id": "s-1",
    }

    full = run_cli(config_file, "submit", str(other), "--request-id", "r-2")
    assert full.returncode == 1
    assert full.stdout == ""
    assert full.stderr.startswith("sound-scout jobs: queue_full: ")
    assert full.stderr.count("\n") == 1

    mismatch = run_cli(config_file, "submit", str(other), "--request-id", "r-1")
    assert mismatch.returncode == 1
    assert mismatch.stderr.startswith("sound-scout jobs: request_mismatch: ")

    status = run_cli(config_file, "status", job_id)
    assert status.returncode == 0, status.stderr
    assert (
        json.loads(run_cli(config_file, "status", job_id, "--json").stdout)["finished_stages"] == []
    )
    assert f"job_id: {job_id}" in status.stdout.splitlines()
    assert "state: queued" in status.stdout.splitlines()

    no_result = run_cli(config_file, "result", job_id)
    assert no_result.returncode == 1
    assert no_result.stderr.startswith("sound-scout jobs: no_result: ")

    cancelled = run_cli(config_file, "cancel", job_id, "--json")
    assert cancelled.returncode == 0, cancelled.stderr
    assert json.loads(cancelled.stdout)["state"] == "cancelled"
    assert json.loads(run_cli(config_file, "status", job_id, "--json").stdout)["state"] == (
        "cancelled"
    )

    unknown = run_cli(config_file, "status", "f" * 32)
    assert unknown.returncode == 1
    assert unknown.stderr.startswith("sound-scout jobs: job_not_found: ")
    assert str(config_file.parent) not in unknown.stderr + full.stderr


def test_jobs_limits_through_the_console(config_file: Path) -> None:
    staging = config_file.parent / "recordings"
    config_file.write_text(
        config_file.read_text(encoding="utf-8")
        + f"staging_root = {json.dumps(str(staging))}\n[budgets]\nmax_input_bytes = 2048\n",
        encoding="utf-8",
    )

    as_json = run_cli(config_file, "limits", "--json")
    as_text = run_cli(config_file, "limits")

    assert as_json.returncode == 0, as_json.stderr
    assert json.loads(as_json.stdout) == {
        "max_duration_seconds": Budgets().max_duration_seconds,
        "max_input_bytes": 2048,
        "max_queued_jobs": 1,
        "max_staging_bytes": Jobs().max_staging_bytes,
        "staging_root": str(staging),
    }
    assert as_text.returncode == 0, as_text.stderr
    assert "max_input_bytes: 2048" in as_text.stdout.splitlines()
    assert f"staging_root: {staging}" in as_text.stdout.splitlines()


def test_jobs_limits_with_a_broken_configuration_is_config_error(config_file: Path) -> None:
    config_file.write_text(
        config_file.read_text(encoding="utf-8") + 'staging_root = "/nowhere/staging"\n',
        encoding="utf-8",
    )

    refused = run_cli(config_file, "limits", "--json")

    assert refused.returncode == 1
    assert refused.stdout == ""
    assert refused.stderr.startswith("sound-scout jobs: config_error: ")


def test_jobs_submit_through_the_console_writes_job_json_and_answers_within_a_second(
    config_file: Path, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/voice.wav", seconds=30.0)

    started = time.perf_counter()
    submitted = run_cli(config_file, "submit", str(wav))
    elapsed = time.perf_counter() - started

    assert submitted.returncode == 0, submitted.stderr
    assert elapsed < 1.0
    lines = submitted.stdout.splitlines()
    job_id = lines[0].removeprefix("job_id: ")
    assert lines[0] == f"job_id: {job_id}" and "state: queued" in lines
    job_json = config_file.parent / "data" / "jobs" / job_id / "job.json"
    assert json.loads(job_json.read_text(encoding="utf-8"))["state"] == "queued"


def test_jobs_submit_outside_the_roots_is_one_line_path_denied(
    config_file: Path, make_wav: WavFactory
) -> None:
    outside = make_wav("outside.wav")

    refused = run_cli(config_file, "submit", str(outside))

    assert refused.returncode == 1
    assert refused.stderr.startswith("sound-scout jobs: path_denied: ")
    assert not list((config_file.parent / "data").glob("jobs/*/job.json"))


@needs_ffmpeg
def test_jobs_result_prints_the_job_state_next_to_the_result_summary(
    tmp_path: Path, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from sound_scout.cli import main

    for name in ("recordings", "data"):
        (tmp_path / name).mkdir()
    path = tmp_path / "sound-scout.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'recordings'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n"
        f"[network]\nallow = true\n[asr]\nmodel = {json.dumps(ABSENT_ASR_MODEL)}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    cfg = config.load()
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    assert main(["jobs", "submit", str(wav), "--json"]) == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    analysis = app.analyze(app.AnalyzeRequest(str(wav), request_id=f"{job_id}-a1"), cfg)
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": analysis.run.run_id}
    set_job(cfg, job_id, state="finished", attempts=[attempt])

    assert main(["jobs", "result", job_id]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        f"job_id: {job_id}",
        "job_state: finished",
        "attempt: 1",
        "status: partial",
    ]
    assert f"run_dir: runs/{analysis.run.run_id}" in lines
