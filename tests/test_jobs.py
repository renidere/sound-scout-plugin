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
        "mode": "auto",
        "layers": sorted(app.LAYERS),
        "task": "фраза",
        "intent": None,
        "control": None,
    }
    assert job["plan"] == {
        "capability_version": 1,
        "mode": "auto",
        "resolved_mode": "full",
        "resolution": "auto: no layers named",
        "stages": list(app.WORKER_STAGES),
        "budgets": dataclasses.asdict(Budgets()),
        "backend": "none",
    }
    assert job["queued_at"] == job["submitted_at"]
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
    assert jobs.resolved_plan("targeted", ("content",), setup)[1]["stages"] == ["prepare", "asr"]
    layers, plan = jobs.resolved_plan("auto", ("sound", "measurements"), setup)
    assert layers == ("measurements", "sound")
    assert plan["stages"] == ["prepare", "acoustics", "audio_review"]


@pytest.mark.parametrize(
    ("mode", "named", "layers", "resolved", "resolution", "stages"),
    [
        (
            "quick",
            None,
            ("content", "measurements"),
            "quick",
            None,
            ["prepare", "asr", "acoustics"],
        ),
        ("full", None, app.LAYERS, "full", None, list(app.WORKER_STAGES)),
        ("targeted", ("sound",), ("sound",), "targeted", None, ["prepare", "audio_review"]),
        (
            "auto",
            ("content",),
            ("content",),
            "targeted",
            "auto: layers named",
            ["prepare", "asr"],
        ),
        ("auto", None, app.LAYERS, "full", "auto: no layers named", list(app.WORKER_STAGES)),
    ],
    ids=["quick", "full", "targeted", "auto-layers", "auto-none"],
)
def test_the_resolver_turns_each_mode_into_layers_and_a_versioned_plan(
    setup: Config,
    mode: str,
    named: tuple | None,
    layers: tuple,
    resolved: str,
    resolution: str | None,
    stages: list,
) -> None:
    got, plan = jobs.resolved_plan(mode, named, setup)

    assert got == layers
    assert set(plan) == jobs.PLAN_KEYS
    assert plan == {
        "capability_version": jobs.CAPABILITY_VERSION,
        "mode": mode,
        "resolved_mode": resolved,
        "resolution": resolution,
        "stages": stages,
        "budgets": dataclasses.asdict(setup.budgets),
        "backend": setup.backend,
    }
    assert jobs.CAPABILITY_VERSION == 1
    assert jobs.applied_plan(plan, setup) == (layers, setup)


@pytest.mark.parametrize(
    ("mode", "layers", "message"),
    [
        ("fast", None, "mode must be one of"),
        (None, None, "mode must be one of"),
        ("quick", ("content",), "layers only with targeted or auto"),
        ("full", ("sound",), "layers only with targeted or auto"),
        ("targeted", None, "mode targeted needs the layers"),
        ("targeted", (), "layers must be a non-empty list"),
        ("auto", ("voice",), "layers must be distinct items"),
    ],
)
def test_a_bad_mode_request_is_invalid_request_and_writes_nothing(
    setup: Config, make_wav: WavFactory, mode: object, layers: object, message: str
) -> None:
    wav = make_wav("recordings/voice.wav")

    with pytest.raises(JobError) as refused:
        jobs.submit(SubmitRequest(str(wav), mode=mode, layers=layers), Binding(), setup)

    assert refused.value.code == "invalid_request" and message in refused.value.message
    assert list(setup.data_root.iterdir()) == []


def stored_plan(cfg: Config, **change: object) -> dict:
    return jobs.resolved_plan("full", None, cfg)[1] | change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"allowed_roots": ["/"]}, "unknown ['allowed_roots']"),
        ({"roots": ["/"]}, "unknown ['roots']"),
        ({"network": {"allow": True}}, "unknown ['network']"),
        ({"data_root": "/tmp"}, "unknown ['data_root']"),
        ({"model": "owner/name"}, "unknown ['model']"),
        ({"backend": "llama"}, "plan backend 'llama' is not the configured backend 'none'"),
        ({"capability_version": 0}, "plan of capability version 0"),
        ({"capability_version": 2}, "plan of capability version 2"),
        ({"capability_version": True}, "plan of capability version True"),
        ({"stages": ["prepare", "asr", "acoustics"]}, "are not the stages"),
        ({"stages": ["asr"]}, "are not the stages"),
        ({"stages": ["prepare"]}, "are not the stages"),
        ({"stages": ["prepare", "asr", "llm"]}, "are not the stages"),
        ({"stages": "prepare,asr"}, "are not the stages"),
        ({"stages": ["prepare", "asr"]}, "are not the stages of its full layers"),
        ({"mode": "everything"}, "plan mode 'everything'"),
        ({"resolved_mode": "auto"}, "resolved as 'auto'"),
    ],
    ids=lambda value: json.dumps(value) if isinstance(value, dict) else None,
)
def test_applied_plan_refuses_a_plan_that_would_widen_the_configuration(
    setup: Config, change: dict, message: str
) -> None:
    with pytest.raises(JobError) as refused:
        jobs.applied_plan(stored_plan(setup, **change), setup)

    assert refused.value.code == "plan_refused" and message in refused.value.message


@pytest.mark.parametrize(
    "budgets",
    [
        {"max_duration_seconds": 180},
        {**dataclasses.asdict(Budgets()), "max_network_calls": 1},
        {**dataclasses.asdict(Budgets()), "worker_deadline_seconds": True},
        {**dataclasses.asdict(Budgets()), "worker_deadline_seconds": 0},
        {**dataclasses.asdict(Budgets()), "worker_deadline_seconds": 1.5},
        {**dataclasses.asdict(Budgets()), "worker_deadline_seconds": "180"},
        [180],
    ],
    ids=["missing", "unknown", "bool", "zero", "float", "string", "list"],
)
def test_applied_plan_refuses_budgets_it_cannot_narrow_by(setup: Config, budgets: object) -> None:
    with pytest.raises(JobError) as refused:
        jobs.applied_plan(stored_plan(setup, budgets=budgets), setup)

    assert refused.value.code == "plan_refused" and "plan budget" in refused.value.message


def test_the_stages_run_asr_before_acoustics_and_a_plan_of_the_old_order_still_runs(
    setup: Config,
) -> None:
    assert jobs.WORKER_STAGES == ("prepare", "asr", "acoustics", "audio_review")
    plan = jobs.resolved_plan("quick", None, setup)[1]
    assert plan["stages"] == ["prepare", "asr", "acoustics"]
    stored = plan | {"stages": ["prepare", "acoustics", "asr"]}  # queued before decision 0038

    assert jobs.applied_plan(stored, setup) == (("content", "measurements"), setup)
    with pytest.raises(JobError, match="are not the stages"):
        jobs.applied_plan(plan | {"stages": ["prepare", "asr", "asr"]}, setup)


def test_a_plan_of_the_old_format_without_its_version_is_refused(setup: Config) -> None:
    old = {
        "stages": list(app.WORKER_STAGES),
        "budgets": dataclasses.asdict(Budgets()),
        "backend": "none",
    }

    with pytest.raises(JobError) as refused:
        jobs.applied_plan(old, setup)

    assert refused.value.code == "plan_refused"
    assert "capability version None" in refused.value.message
    assert "submit the recording again" in refused.value.message
    with pytest.raises(JobError, match="plan_refused"):
        jobs.applied_plan("full", setup)


def test_budgets_are_the_smaller_of_the_plan_and_the_configuration(setup: Config) -> None:
    plan = stored_plan(
        setup,
        budgets=dataclasses.asdict(Budgets())
        | {"worker_deadline_seconds": 10_000, "max_channels": 1},
    )
    narrowed = dataclasses.replace(
        setup, budgets=dataclasses.replace(setup.budgets, max_duration_seconds=30)
    )

    layers, effective = jobs.applied_plan(plan, narrowed)

    assert layers == app.LAYERS
    assert effective.budgets == dataclasses.replace(
        Budgets(), max_channels=1, max_duration_seconds=30
    )
    assert effective.budgets.worker_deadline_seconds == Budgets().worker_deadline_seconds
    assert dataclasses.replace(effective, budgets=narrowed.budgets) == narrowed


def test_the_jobs_module_never_builds_a_model_or_passes_a_backend() -> None:
    import ast

    tree = ast.parse(Path(jobs.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {f"{node.module}.{alias.name}" for alias in node.names} | {node.module}
    forbidden = (
        "review",
        "smoke",
        "eval",
        "llama",
        "transformers",
        "ctranslate2",
        "faster_whisper",
    )
    assert not [name for name in imported if any(part in forbidden for part in name.split("."))]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert "backend" not in {keyword.arg for keyword in node.keywords}


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


def test_a_request_submitted_before_the_stage_order_changed_reuses_its_job(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    answer = jobs.submit(SubmitRequest(str(wav), request_id="r-1", mode="quick"), Binding(), setup)
    job = json.loads((setup.data_root / "jobs" / answer["job_id"] / "job.json").read_text())
    assert job["plan"]["stages"] == ["prepare", "asr", "acoustics"]
    legacy = {  # the identity as `submit` hashed it before decision 0038
        "audio": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "analysis": {
            "mode": "quick",
            "layers": ["content", "measurements"],
            "task": None,
            "intent": None,
            "control": None,
        },
        "plan": job["plan"] | {"stages": ["prepare", "acoustics", "asr"]},
        "binding": dataclasses.asdict(Binding()),
    }
    text = json.dumps(legacy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert job["identity_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()

    again = jobs.submit(SubmitRequest(str(wav), request_id="r-1", mode="quick"), Binding(), setup)
    assert again == answer | {"reused": True}
    with pytest.raises(runs.RequestMismatch, match="r-1"):
        jobs.submit(SubmitRequest(str(wav), request_id="r-1", mode="full"), Binding(), setup)


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


@pytest.mark.parametrize("call", [jobs.status, jobs.result, jobs.report, jobs.cancel])
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


@needs_ffmpeg
def test_report_answers_report_md_of_the_attempt_that_result_answers_with(
    setup: Config, make_wav: WavFactory
) -> None:
    cfg = dataclasses.replace(setup, asr=config.Asr(model=ABSENT_ASR_MODEL))
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    job_id = jobs.submit(SubmitRequest(str(wav)), Binding(), cfg)["job_id"]
    analysis = app.analyze(app.AnalyzeRequest(str(wav), request_id=f"{job_id}-a1"), cfg)
    first = {"number": 1, "request_id": f"{job_id}-a1", "run_id": analysis.run.run_id}
    second = {"number": 2, "request_id": f"{job_id}-a2", "run_id": runs.new_run_id()}
    set_job(cfg, job_id, state="failed", attempts=[first, second])

    answer = jobs.report(job_id, cfg)

    report_md = analysis.run.path / "report.md"
    assert answer == {
        "job_id": job_id,
        "job_state": "failed",
        "attempt": jobs.result(job_id, cfg)["attempt"],
        "run_id": analysis.run.run_id,
        "run_dir": f"runs/{analysis.run.run_id}",
        "markdown": report_md.read_text(encoding="utf-8"),
        "size_bytes": report_md.stat().st_size,
        "truncated": False,
    }
    assert answer["attempt"] == 1
    assert str(setup.allowed_roots[0]) not in json.dumps(answer)


def fake_run(
    cfg: Config, job_id: str, *, report_md: bytes | None, sha256: str | None = None
) -> str:
    """A run directory with `result.json` of the job's input and, when given, `report.md`."""
    run_id = runs.new_run_id()
    run_dir = cfg.data_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    source = sha256 or job_file(cfg, job_id)["request"]["input"]["sha256"]
    (run_dir / "result.json").write_text(json.dumps({"source": {"sha256": source}}))
    if report_md is not None:
        (run_dir / "report.md").write_bytes(report_md)
    return run_id


def test_report_skips_an_input_changed_attempt_and_one_of_other_audio(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    good = fake_run(setup, job_id, report_md="# Отчёт 1\n".encode())
    changed = fake_run(setup, job_id, report_md=b"# changed\n")
    other = fake_run(setup, job_id, report_md=b"# other\n", sha256="0" * 64)
    attempts = [
        {"number": 1, "request_id": f"{job_id}-a1", "run_id": good},
        {"number": 2, "request_id": f"{job_id}-a2", "run_id": changed, "input_changed": True},
        {"number": 3, "request_id": f"{job_id}-a3", "run_id": other},
    ]
    set_job(setup, job_id, state="failed", attempts=attempts)

    answer = jobs.report(job_id, setup)

    assert answer["attempt"] == 1 and answer["markdown"] == "# Отчёт 1\n"
    set_job(setup, job_id, attempts=attempts[1:])
    with pytest.raises(JobError) as refused:
        jobs.report(job_id, setup)
    assert refused.value.code == "no_result"


def test_report_of_a_run_without_report_md_or_of_an_unknown_job_is_refused(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    with pytest.raises(JobError) as refused:
        jobs.report(job_id, setup)
    assert refused.value.code == "no_result"
    run_id = fake_run(setup, job_id, report_md=None)
    set_job(setup, job_id, attempts=[{"number": 1, "request_id": f"{job_id}-a1", "run_id": run_id}])

    with pytest.raises(JobError) as refused:
        jobs.report(job_id, setup)
    assert refused.value.code == "no_result"
    for unknown in ("0" * 32, "../runs"):
        with pytest.raises(JobError) as refused:
            jobs.report(unknown, setup)
        assert refused.value.code == "job_not_found"


def test_a_report_above_the_cap_is_cut_at_a_line_break(setup: Config, make_wav: WavFactory) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    line = "Строка отчёта с кириллицей и словом «ошибка».\n".encode()
    text = line * (300 * 1024 // len(line) + 1)
    run_id = fake_run(setup, job_id, report_md=text)
    set_job(setup, job_id, attempts=[{"number": 1, "request_id": f"{job_id}-a1", "run_id": run_id}])

    answer = jobs.report(job_id, setup)

    cut = answer["markdown"].encode("utf-8")
    assert answer["truncated"] is True and answer["size_bytes"] == len(text)
    assert len(cut) <= jobs.REPORT_VIEW_MAX_BYTES == 256 * 1024
    assert cut.endswith(b"\n") and text.startswith(cut)
    assert len(cut) > jobs.REPORT_VIEW_MAX_BYTES - len(line)


def test_a_report_above_the_cap_without_a_line_break_is_cut_at_a_character(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    text = "ж".encode() * (jobs.REPORT_VIEW_MAX_BYTES // 2 + 10)
    run_id = fake_run(setup, job_id, report_md=b"x" + text)
    set_job(setup, job_id, attempts=[{"number": 1, "request_id": f"{job_id}-a1", "run_id": run_id}])

    answer = jobs.report(job_id, setup)

    cut = answer["markdown"].encode("utf-8")
    assert answer["truncated"] is True and "\ufffd" not in answer["markdown"]
    assert len(cut) == jobs.REPORT_VIEW_MAX_BYTES - 1 and (b"x" + text).startswith(cut)


def test_jobs_report_through_the_cli_prints_the_markdown_or_its_json(
    setup: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
) -> None:
    from sound_scout.cli import main

    path = tmp_path / "sound-scout.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(setup.allowed_roots[0]))}]\n"
        f"data_root = {json.dumps(str(setup.data_root))}\n"
        "[network]\nallow = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    assert main(["jobs", "report", job_id]) == 1
    assert capsys.readouterr().err.startswith("sound-scout jobs: no_result: ")
    small = fake_run(setup, job_id, report_md="# Отчёт\nтекст\n".encode())
    big = fake_run(setup, job_id, report_md=b"x\n" * (jobs.REPORT_VIEW_MAX_BYTES // 2 + 10))
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": small}
    set_job(setup, job_id, state="finished", attempts=[attempt])

    assert main(["jobs", "report", job_id]) == 0
    printed = capsys.readouterr()
    assert printed.out == "# Отчёт\nтекст\n" and printed.err == ""
    assert main(["jobs", "report", job_id, "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["markdown"] == "# Отчёт\nтекст\n" and answer["truncated"] is False

    set_job(setup, job_id, attempts=[attempt | {"run_id": big}])
    assert main(["jobs", "report", job_id]) == 0
    printed = capsys.readouterr()
    assert len(printed.out) == jobs.REPORT_VIEW_MAX_BYTES
    assert printed.err.startswith("sound-scout jobs: report.md is ")
    assert printed.err.rstrip().endswith(
        f"the whole file is runs/{big}/report.md under the data root"
    )


def test_status_shows_when_the_staging_copy_was_removed(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = jobs.submit(SubmitRequest(str(make_wav("recordings/a.wav"))), Binding(), setup)[
        "job_id"
    ]
    assert jobs.status(job_id, setup)["staging_removed_at"] is None
    set_job(setup, job_id, state="failed", staging_removed_at="2026-11-23T10:00:00Z")

    assert jobs.status(job_id, setup)["staging_removed_at"] == "2026-11-23T10:00:00Z"


def test_the_stage_names_of_jobs_are_those_of_the_application_layer() -> None:
    assert jobs.LAYERS == app.LAYERS
    assert jobs.LAYER_STAGES == app.LAYER_STAGES
    assert jobs.WORKER_STAGES == app.WORKER_STAGES
    assert jobs.STAGE_RECORD == app.STAGE_RECORD
    assert jobs.CURRENT_STAGE == app.CURRENT_STAGE
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


def write_current_stage(cfg: Config, run_id: str, marker: object) -> None:
    work = cfg.data_root / runs.RUNS_DIR_NAME / run_id / ingest.WORK_DIR_NAME
    work.mkdir(parents=True, exist_ok=True)
    text = marker if isinstance(marker, str) else json.dumps(marker)
    (work / jobs.CURRENT_STAGE).write_text(text, encoding="utf-8")


def running_job(cfg: Config, wav: Path) -> tuple[str, str]:
    """A submitted job the runner moved to `running` with attempt 1 and its run id."""
    job_id = jobs.submit(SubmitRequest(str(wav)), Binding(), cfg)["job_id"]
    run_id = runs.new_run_id()
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": run_id}
    set_job(cfg, job_id, state="running", attempts=[attempt])
    return job_id, run_id


def test_status_follows_each_stage_transition_of_a_running_job(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id, run_id = running_job(setup, make_wav("recordings/a.wav"))
    assert jobs.status(job_id, setup)["current_stage"] is None

    finished: list[dict] = []
    for i, stage in enumerate(jobs.WORKER_STAGES):
        started_at = f"2026-11-20T10:00:0{i}+00:00"
        write_current_stage(setup, run_id, {"stage": stage, "started_at": started_at})
        found = jobs.status(job_id, setup)
        assert found["state"] == "running"
        assert found["current_stage"] == {"stage": stage, "started_at": started_at}
        assert found["finished_stages"] == finished

        if stage == "audio_review":
            write_stage_record(setup, run_id, stage, stage_record(stage, "failed", checkpoint=True))
            assert jobs.status(job_id, setup)["current_stage"] == found["current_stage"]
        write_stage_record(setup, run_id, stage, stage_record(stage, "completed"))
        finished.append({"stage": stage, "status": "completed"})
        found = jobs.status(job_id, setup)
        assert found["current_stage"] is None, "between stages nothing runs"
        assert found["finished_stages"] == finished


@pytest.mark.parametrize("state", ["queued", "finished", "cancelled", "failed"])
def test_a_job_that_is_not_running_has_no_current_stage(
    setup: Config, make_wav: WavFactory, state: str
) -> None:
    job_id, run_id = running_job(setup, make_wav("recordings/a.wav"))
    write_stage_record(setup, run_id, "prepare", stage_record("prepare", "completed"))
    write_current_stage(setup, run_id, {"stage": "acoustics", "started_at": "2026-11-20T10:00Z"})
    set_job(setup, job_id, state=state)

    found = jobs.status(job_id, setup)

    assert found["state"] == state
    assert found["current_stage"] is None
    assert found["finished_stages"] == [{"stage": "prepare", "status": "completed"}]


@pytest.mark.parametrize(
    "marker",
    [
        "{torn",
        ["acoustics"],
        {"stage": "acoustics"},
        {"stage": "report", "started_at": "2026-11-20T10:00Z"},
        {"stage": "asr", "started_at": "2026-11-20T10:00Z"},
    ],
    ids=["torn", "not_an_object", "no_start_time", "not_a_worker_stage", "not_in_the_plan"],
)
def test_an_unusable_marker_names_no_current_stage(
    setup: Config, make_wav: WavFactory, marker: object
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = jobs.submit(SubmitRequest(str(wav), layers=("measurements",)), Binding(), setup)[
        "job_id"
    ]
    run_id = runs.new_run_id()
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": run_id}
    set_job(setup, job_id, state="running", attempts=[attempt])
    write_current_stage(setup, run_id, marker)

    assert jobs.status(job_id, setup)["current_stage"] is None


def test_a_running_attempt_without_its_marker_yet_has_no_current_stage(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id, _ = running_job(setup, make_wav("recordings/a.wav"))

    found = jobs.status(job_id, setup)

    assert found["state"] == "running"
    assert found["current_stage"] is None and found["finished_stages"] == []


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


@pytest.mark.slow
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

    data = config_file.parent / "data"
    running = Config(allowed_roots=(config_file.parent / "recordings",), data_root=data)
    run_id = runs.new_run_id()
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": run_id}
    set_job(running, job_id, state="running", attempts=[attempt])
    write_stage_record(running, run_id, "prepare", stage_record("prepare", "completed"))
    write_current_stage(running, run_id, {"stage": "asr", "started_at": "2026-11-20T10:00Z"})
    merged = json.loads(run_cli(config_file, "status", job_id, "--json").stdout)
    assert merged["state"] == "running"
    assert merged["current_stage"] == {"stage": "asr", "started_at": "2026-11-20T10:00Z"}
    assert merged["finished_stages"] == [{"stage": "prepare", "status": "completed"}]
    plain = run_cli(config_file, "status", job_id).stdout.splitlines()
    assert 'current_stage: {"stage": "asr", "started_at": "2026-11-20T10:00Z"}' in plain
    set_job(running, job_id, state="queued", attempts=[])

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


@pytest.mark.slow
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


@pytest.mark.slow
def test_jobs_submit_mode_and_retry_through_the_console(
    config_file: Path, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/voice.wav")
    data = config_file.parent / "data"

    bad = run_cli(config_file, "submit", str(wav), "--mode", "fast")
    assert bad.returncode == 1 and bad.stdout == ""
    assert bad.stderr.startswith("sound-scout jobs: invalid_request: mode must be one of")
    layered = run_cli(config_file, "submit", str(wav), "--mode", "quick", "--layers", "sound")
    assert layered.returncode == 1
    assert layered.stderr.startswith("sound-scout jobs: invalid_request: layers only with")
    assert not (data / "jobs").exists() or not jobs._job_dirs(data / "jobs")

    submitted = run_cli(config_file, "submit", str(wav), "--mode", "quick", "--json")
    assert submitted.returncode == 0, submitted.stderr
    job_id = json.loads(submitted.stdout)["job_id"]
    plan = json.loads((data / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))["plan"]
    assert plan["mode"] == plan["resolved_mode"] == "quick"
    assert plan["stages"] == ["prepare", "asr", "acoustics"]
    status = json.loads(run_cli(config_file, "status", job_id, "--json").stdout)
    assert status["mode"] == "quick" and status["retries"] == 0

    queued = run_cli(config_file, "retry", job_id)
    assert queued.returncode == 1 and queued.stdout == ""
    assert queued.stderr.startswith("sound-scout jobs: not_retryable: ")

    assert run_cli(config_file, "cancel", job_id).returncode == 0
    retried = run_cli(config_file, "retry", job_id)
    assert retried.returncode == 0, retried.stderr
    lines = retried.stdout.splitlines()
    assert "state: queued" in lines and "retry: queued" in lines
    assert f"next_attempt: {job_id}-a1" in lines and "retries: 1" in lines

    assert run_cli(config_file, "cancel", job_id).returncode == 0
    again = run_cli(config_file, "retry", job_id, "--json")
    assert again.returncode == 0, again.stderr
    answer = json.loads(again.stdout)
    assert answer["retry"] == "queued" and answer["retries"] == 2
    assert answer["job_id"] == job_id and answer["next_attempt"] == f"{job_id}-a1"


# --- a control attempt: the link to the initial run and its criterion (#110) -----------------


def baseline_run(cfg: Config, wav: Path, request_id: str = "initial") -> str:
    """A real initial attempt: `app.analyze` with `backend = none` and an absent ASR model."""
    return app.analyze(app.AnalyzeRequest(str(wav), request_id=request_id), cfg).run.run_id


def asr_absent(cfg: Config) -> Config:
    return dataclasses.replace(cfg, asr=config.Asr(model=ABSENT_ASR_MODEL))


def no_job_written(cfg: Config) -> bool:
    jobs_dir = cfg.data_root / "jobs"
    return not jobs_dir.exists() or not jobs._job_dirs(jobs_dir)


@needs_ffmpeg
def test_a_control_job_stores_its_link_with_the_normalized_criterion(
    setup: Config, make_wav: WavFactory
) -> None:
    cfg = asr_absent(setup)
    initial = baseline_run(cfg, make_wav("recordings/initial.wav", seconds=1.5))
    wav = make_wav("recordings/control.wav", seconds=1.5, frequency=330.0)
    request = SubmitRequest(
        str(wav), request_id="r-c", previous_run_id=initial, criterion=" emphasis_on: Слово "
    )

    answer = jobs.submit(request, Binding(), cfg)

    control = {"previous_run_id": initial, "criterion": "emphasis_on:Слово"}
    job = job_file(cfg, answer["job_id"])
    assert job["request"]["control"] == control
    assert job["plan"] == jobs.resolved_plan("auto", None, cfg)[1]
    view = jobs.status(answer["job_id"], cfg)
    assert view["control"] == control and view["compare"] is None
    assert jobs.submit(request, Binding(), cfg) == answer | {"reused": True}
    other = dataclasses.replace(request, criterion="completeness_of_phrase")
    with pytest.raises(runs.RequestMismatch):
        jobs.submit(other, Binding(), cfg)
    plain = jobs.submit(SubmitRequest(str(wav), request_id="r-plain"), Binding(), cfg)
    assert job_file(cfg, plain["job_id"])["request"]["control"] is None
    assert jobs.status(plain["job_id"], cfg)["control"] is None


@needs_ffmpeg
def test_a_control_job_compare_would_refuse_is_refused_at_submit_and_writes_nothing(
    setup: Config, make_wav: WavFactory
) -> None:
    cfg = asr_absent(setup)
    initial = baseline_run(cfg, make_wav("recordings/initial.wav", seconds=1.5))
    wav = str(make_wav("recordings/control.wav", seconds=1.5, frequency=330.0))
    incomplete = runs.open_run("left-incomplete", {}, cfg.data_root).run_id
    linked = cfg.data_root / "runs" / ("e" * 32)
    linked.symlink_to(cfg.data_root / "runs" / initial)

    def refusal(**fields: object) -> str:
        with pytest.raises(JobError) as refused:
            jobs.submit(SubmitRequest(wav, **fields), Binding(), cfg)
        assert no_job_written(cfg)
        return refused.value.code

    for criterion in (
        "louder",
        "emphasis_on:два слова",
        "completeness_of_phrase:x",
        "emphasis_on:…",
    ):
        assert refusal(previous_run_id=initial, criterion=criterion) == "criterion_invalid"
    assert refusal(previous_run_id=initial) == "invalid_request"
    assert refusal(criterion="completeness_of_phrase") == "invalid_request"
    for run_id in ("not-a-run", "f" * 32, linked.name):
        assert refusal(previous_run_id=run_id, criterion="completeness_of_phrase") == (
            "run_not_found"
        )
    assert refusal(previous_run_id=incomplete, criterion="completeness_of_phrase") == (
        "run_incomplete"
    )
    assert refusal(mode="quick", previous_run_id=initial, criterion="completeness_of_phrase") == (
        "invalid_request"
    )
    sound_only = {"mode": "targeted", "layers": ("sound",), "previous_run_id": initial}
    assert refusal(**sound_only, criterion="emphasis_on:слово") == "invalid_request"

    accepted = jobs.submit(
        SubmitRequest(wav, **sound_only, criterion="completeness_of_phrase"), Binding(), cfg
    )
    assert job_file(cfg, accepted["job_id"])["request"]["control"]["criterion"] == (
        "completeness_of_phrase"
    )
    (cfg.data_root / "runs" / initial / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(JobError) as refused:
        jobs.submit(
            SubmitRequest(wav, previous_run_id=initial, criterion="completeness_of_phrase"),
            Binding(),
            cfg,
        )
    assert refused.value.code == "result_invalid"
    assert [path.name for path in jobs._job_dirs(cfg.data_root / "jobs")] == [accepted["job_id"]]


@pytest.mark.slow
@needs_ffmpeg
def test_jobs_submit_of_a_control_attempt_through_the_console(
    config_file: Path, make_wav: WavFactory
) -> None:
    data = config_file.parent / "data"
    cfg = Config(
        allowed_roots=(config_file.parent / "recordings",),
        data_root=data,
        asr=config.Asr(model=ABSENT_ASR_MODEL),
    )
    initial = baseline_run(cfg, make_wav("recordings/initial.wav", seconds=1.5))
    wav = str(make_wav("recordings/control.wav", seconds=1.5, frequency=330.0))
    link = ("--previous-run-id", initial, "--criterion", "completeness_of_phrase")

    bad = run_cli(config_file, "submit", wav, "--previous-run-id", initial, "--criterion", "bad")
    assert bad.returncode == 1 and bad.stdout == ""
    assert bad.stderr.startswith("sound-scout jobs: criterion_invalid: ")
    assert no_job_written(cfg)

    text = run_cli(config_file, "submit", wav, *link, "--request-id", "r-text")
    assert text.returncode == 0, text.stderr
    assert "state: queued" in text.stdout.splitlines()
    run_cli(config_file, "cancel", text.stdout.split("job_id: ", 1)[1].split()[0])
    submitted = run_cli(config_file, "submit", wav, *link, "--request-id", "r-json", "--json")
    assert submitted.returncode == 0, submitted.stderr
    job_id = json.loads(submitted.stdout)["job_id"]
    status = json.loads(run_cli(config_file, "status", job_id, "--json").stdout)
    assert status["control"] == {"previous_run_id": initial, "criterion": "completeness_of_phrase"}

    ran = run_cli(config_file, "run", "--exit-when-idle")
    assert ran.returncode == 0, ran.stderr
    shown = run_cli(config_file, "result", job_id)
    assert shown.returncode == 0, shown.stderr
    compare_line = next(line for line in shown.stdout.splitlines() if line.startswith("compare: "))
    summary = json.loads(compare_line.removeprefix("compare: "))
    assert summary["state"] == "completed" and summary["verdict"] == "inconclusive"
