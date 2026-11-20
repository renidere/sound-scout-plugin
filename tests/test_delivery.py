"""Delivery state of a job and the text of its final note (`sound_scout.delivery`, #102)."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from sound_scout import app, config, delivery, jobs, mcp_server
from sound_scout.config import Config
from sound_scout.jobs import Binding, JobError, SubmitRequest
from tests.conftest import WavFactory

ABSENT_ASR_MODEL = "absent/model"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)
ALL_DONE = ("готово", "всё выполнено", "все выполнено", "всё хорошо")


@pytest.fixture
def setup(tmp_path: Path) -> Config:
    (tmp_path / "recordings").mkdir()
    (tmp_path / "data").mkdir()
    return Config(
        allowed_roots=(tmp_path / "recordings",),
        data_root=tmp_path / "data",
        asr=config.Asr(model=ABSENT_ASR_MODEL),
    )


def job_file(cfg: Config, job_id: str) -> dict:
    return json.loads((cfg.data_root / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))


def set_job(cfg: Config, job_id: str, **fields: object) -> None:
    """Stand in for the runner: it changes the state and appends attempts."""
    path = cfg.data_root / "jobs" / job_id / "job.json"
    path.write_text(json.dumps(job_file(cfg, job_id) | fields), encoding="utf-8")


def ended_job(
    cfg: Config,
    wav: Path,
    *,
    session_id: str | None = "chat-1",
    profile: str | None = None,
    state: str = "finished",
    reason: str | None = None,
    layers: tuple[str, ...] = jobs.LAYERS,
) -> str:
    binding = Binding(submitter="desktop", profile=profile, session_id=session_id)
    job_id = jobs.submit(SubmitRequest(str(wav), layers=layers), binding, cfg)["job_id"]
    set_job(cfg, job_id, state=state, state_reason=reason)
    return job_id


def analyzed_job(cfg: Config, wav: Path, **kwargs: object) -> str:
    """A job whose attempt 1 is a real `app.analyze` run: the ASR model is absent, so `partial`."""
    job_id = ended_job(cfg, wav, **kwargs)
    layers = tuple(job_file(cfg, job_id)["request"]["layers"])
    analysis = app.analyze(
        app.AnalyzeRequest(str(wav), request_id=f"{job_id}-a1", layers=layers), cfg
    )
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": analysis.run.run_id}
    set_job(cfg, job_id, attempts=[attempt])
    return job_id


def test_submit_writes_delivery_null_and_status_shows_it(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="queued")

    assert job_file(setup, job_id)["delivery"] is None
    assert jobs.status(job_id, setup)["delivery"] is None
    answer, is_error = mcp_server.call("speech_job_status", {"job_id": job_id}, setup)
    assert is_error is False and answer["delivery"] is None


def test_the_delivery_field_is_not_part_of_the_job_identity(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    first = jobs.submit(SubmitRequest(str(wav), request_id="r-1"), Binding(), setup)
    set_job(setup, first["job_id"], delivery={"state": "delivered"})

    again = jobs.submit(SubmitRequest(str(wav), request_id="r-1"), Binding(), setup)

    assert again == first | {"reused": True}


def test_next_gives_only_ended_jobs_of_the_profile_with_a_chat(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    for state in ("queued", "running"):
        ended_job(setup, wav, state=state)
    ended_job(setup, wav, session_id=None)
    ended_job(setup, wav, profile="work")
    legacy = ended_job(setup, wav)
    legacy_job = job_file(setup, legacy)
    del legacy_job["delivery"]
    (setup.data_root / "jobs" / legacy / "job.json").write_text(json.dumps(legacy_job))

    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}

    oldest = ended_job(setup, wav, state="cancelled", reason="cancelled before it started")
    newer = ended_job(setup, wav, profile="default", state="failed", reason="internal_error: X")

    first = delivery.next_delivery("default", "loop-1", setup)
    second = delivery.next_delivery("default", "loop-1", setup)

    assert first["action"] == "send" and first["job_id"] == oldest
    assert first["session_id"] == "chat-1" and first["tag"] == f"sound-scout-job:{oldest}"
    assert second["job_id"] == newer
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    work = delivery.next_delivery("work", "loop-1", setup)
    assert work["action"] == "send"
    assert job_file(setup, work["job_id"])["binding"]["profile"] == "work"


def test_the_sending_marker_is_written_before_next_answers(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))

    answer = delivery.next_delivery("default", "loop-1", setup)

    marker = job_file(setup, job_id)["delivery"]
    assert answer["action"] == "send"
    assert marker["state"] == "sending" and marker["loop_id"] == "loop-1"
    assert marker["tag"] == answer["tag"] and marker["claimed_at"]
    assert job_file(setup, job_id)["state"] == "finished"
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    assert delivery.next_delivery("default", "loop-2", setup) == {
        "action": "check",
        "job_id": job_id,
        "session_id": "chat-1",
        "tag": answer["tag"],
    }
    assert job_file(setup, job_id)["delivery"] == marker


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("delivered", {"state": "delivered", "confirmed_by": "send"}),
        ("delivered_by_history", {"state": "delivered", "confirmed_by": "history"}),
        ("unconfirmed", {"state": "delivery_unconfirmed", "error": "timed out"}),
        ("failed", {"state": "delivery_failed", "error": "timed out"}),
    ],
)
def test_end_records_how_the_send_ended(
    setup: Config, make_wav: WavFactory, outcome: str, expected: dict
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    delivery.next_delivery("default", "loop-1", setup)

    status = delivery.end_delivery(job_id, outcome, "timed out", setup)

    recorded = job_file(setup, job_id)["delivery"]
    assert {key: recorded[key] for key in expected} == expected
    assert recorded["tag"] == f"sound-scout-job:{job_id}"
    assert recorded["claimed_at"] and recorded["ended_at"]
    assert status["delivery"] == recorded and status["state"] == "finished"
    assert delivery.next_delivery("default", "loop-2", setup) == {"action": "none"}


def test_end_error_is_short_and_without_paths(setup: Config, make_wav: WavFactory) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = ended_job(setup, wav)
    delivery.next_delivery("default", "loop-1", setup)

    delivery.end_delivery(job_id, "failed", f"{setup.data_root} {'x' * 500}", setup)

    error = job_file(setup, job_id)["delivery"]["error"]
    assert len(error) == 300 and str(setup.data_root) not in error


def test_released_puts_the_job_back_to_wait(setup: Config, make_wav: WavFactory) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    delivery.next_delivery("default", "loop-1", setup)

    delivery.end_delivery(job_id, "released", "no gateway in this process owns it", setup)

    assert job_file(setup, job_id)["delivery"] is None
    assert delivery.next_delivery("default", "loop-1", setup)["job_id"] == job_id


def test_end_of_a_job_not_sending_is_refused_and_changes_nothing(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    path = setup.data_root / "jobs" / job_id / "job.json"
    before = path.read_bytes()

    with pytest.raises(JobError) as refused:
        delivery.end_delivery(job_id, "delivered", None, setup)
    assert refused.value.code == "delivery_not_sending"
    assert path.read_bytes() == before

    delivery.next_delivery("default", "loop-1", setup)
    with pytest.raises(JobError) as unknown:
        delivery.end_delivery(job_id, "resend", None, setup)
    assert unknown.value.code == "invalid_request"
    delivery.end_delivery(job_id, "delivered", None, setup)
    ended = path.read_bytes()
    with pytest.raises(JobError) as twice:
        delivery.end_delivery(job_id, "failed", "late", setup)
    assert twice.value.code == "delivery_not_sending"
    assert path.read_bytes() == ended


def test_next_refuses_an_empty_loop_id(setup: Config) -> None:
    with pytest.raises(JobError) as refused:
        delivery.next_delivery("default", "", setup)
    assert refused.value.code == "invalid_request"


@needs_ffmpeg
def test_note_of_a_partial_result_lists_each_requested_layer_with_its_status(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    job_id = analyzed_job(setup, wav)
    result = jobs.result(job_id, setup)

    text = delivery.next_delivery("default", "loop-1", setup)["text"]

    lines = text.splitlines()
    assert result["status"] == "partial"
    assert lines[0] == (
        "Sound Scout: анализ записи закончен — `partial`: частичный результат: выполнены не все "
        "запрошенные слои или не весь охват."
    )
    assert "- содержание (ASR): `unavailable` — недоступно" in lines
    for layer, name in (("measurements", "измерения (акустика)"), ("sound", "звучание")):
        status = result["stages"][jobs.LAYER_STAGES[layer]]["status"]
        assert any(line.startswith(f"- {name}") and f"`{status}`" in line for line in lines)
    assert any(line.startswith("Не оценено: ") for line in lines)
    assert f"Отчёт: {result['run_dir']}/report.md" in lines
    assert lines[-1] == f"sound-scout-job:{job_id}"
    assert not any(phrase in text.lower() for phrase in ALL_DONE)
    for private in (str(wav), str(setup.data_root), str(setup.allowed_roots[0])):
        assert private not in text


@needs_ffmpeg
def test_note_leaves_out_the_layers_not_requested(setup: Config, make_wav: WavFactory) -> None:
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    analyzed_job(setup, wav, layers=("content",), state="cancelled")

    text = delivery.next_delivery("default", "loop-1", setup)["text"]

    assert "Задание отменено." in text.splitlines()
    assert "содержание (ASR)" in text
    assert "измерения" not in text and "звучание" not in text


@pytest.mark.parametrize(
    ("state", "reason", "head"),
    [
        (
            "cancelled",
            "cancelled before it started",
            "Sound Scout: анализ записи не дал результата — задание `cancelled`.",
        ),
        (
            "failed",
            "input_unreadable: /home/someone/voice.wav is broken",
            "Sound Scout: анализ записи не дал результата — задание `failed`, код "
            "`input_unreadable`.",
        ),
    ],
)
def test_note_without_a_result_names_the_state_and_its_code(
    setup: Config, make_wav: WavFactory, state: str, reason: str, head: str
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state=state, reason=reason)

    lines = delivery.next_delivery("default", "loop-1", setup)["text"].splitlines()

    assert lines == [
        head,
        "- содержание (ASR): не выполнено",
        "- измерения (акустика): не выполнено",
        "- звучание (audio review): не выполнено",
        f"sound-scout-job:{job_id}",
    ]


def test_an_unreadable_result_reads_as_no_result_and_does_not_block_later_jobs(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    broken = ended_job(setup, wav)
    run_id = "a" * 32
    run_dir = setup.data_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text('{"source": {"sha2', encoding="utf-8")
    set_job(setup, broken, attempts=[{"number": 1, "request_id": "r", "run_id": run_id}])
    later = ended_job(setup, wav, session_id="chat-2")

    first = delivery.next_delivery("default", "loop-1", setup)
    second = delivery.next_delivery("default", "loop-1", setup)

    assert first["job_id"] == broken and second["job_id"] == later
    assert first["text"].splitlines()[0] == (
        "Sound Scout: анализ записи не дал результата — задание `finished`."
    )
    assert job_file(setup, broken)["delivery"]["state"] == "sending"


# --- the real console entry point ------------------------------------------------------------


def run_cli(config_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ) | {config.CONFIG_PATH_ENV: str(config_path)}
    return subprocess.run(
        [sys.executable, "-m", "sound_scout", "jobs", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture
def config_file(setup: Config) -> Path:
    path = setup.data_root.parent / "sound-scout.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(setup.allowed_roots[0]))}]\n"
        f"data_root = {json.dumps(str(setup.data_root))}\n",
        encoding="utf-8",
    )
    return path


def test_deliver_next_and_end_through_the_console_entry(
    setup: Config, config_file: Path, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))

    none = run_cli(config_file, "deliver-next", "--profile=work", "--loop=l-1")
    assert none.returncode == 0, none.stderr
    assert none.stdout.strip() == "action: none"

    sent = run_cli(config_file, "deliver-next", "--profile=default", "--loop=l-1", "--json")
    assert sent.returncode == 0, sent.stderr
    answer = json.loads(sent.stdout)
    assert answer["action"] == "send" and answer["job_id"] == job_id
    assert answer["text"].endswith(f"sound-scout-job:{job_id}")

    failed = run_cli(
        config_file, "deliver-end", "--outcome=failed", "--error=-chat deleted", job_id
    )
    assert failed.returncode == 0, failed.stderr
    assert "delivery_failed" in failed.stdout

    status = json.loads(run_cli(config_file, "status", job_id, "--json").stdout)
    assert status["delivery"]["state"] == "delivery_failed"
    assert status["delivery"]["error"] == "-chat deleted"

    again = run_cli(config_file, "deliver-end", "--outcome=delivered", job_id, "--json")
    assert again.returncode == 1 and again.stdout == ""
    assert again.stderr.startswith("sound-scout jobs: delivery_not_sending: ")
    assert again.stderr.count("\n") == 1

    bad = run_cli(config_file, "deliver-next", "--profile=default", "--loop=")
    assert bad.returncode == 1
    assert bad.stderr.startswith("sound-scout jobs: invalid_request: ")


# --- redelivery on the operator's request (#122) ---------------------------------------------


def unconfirmed_job(cfg: Config, wav: Path, **kwargs: object) -> str:
    job_id = ended_job(cfg, wav, **kwargs)
    delivery.next_delivery("default", "loop-1", cfg)
    delivery.end_delivery(job_id, "unconfirmed", "timed out", cfg)
    return job_id


def test_redeliver_takes_a_delivery_unconfirmed_job_and_records_the_request(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = unconfirmed_job(setup, make_wav("recordings/a.wav"))
    before = job_file(setup, job_id)

    status = delivery.request_redelivery(job_id, setup)

    after = job_file(setup, job_id)
    [request] = after["redeliveries"]
    assert request["previous"] == before["delivery"] and request["requested_at"]
    assert after["delivery"] == {
        "state": "redelivery_requested",
        "tag": f"sound-scout-job:{job_id}",
        "requested_at": request["requested_at"],
    }
    assert after["updated_at"] == before["updated_at"] and after["state"] == "finished"
    assert status["delivery"] == after["delivery"]


@pytest.mark.parametrize(
    "current",
    [None, "absent", "sending", "delivered", "delivery_failed", "redelivery_requested"],
)
def test_redeliver_refuses_every_other_delivery_state_and_leaves_job_json_as_it_is(
    setup: Config, make_wav: WavFactory, current: str | None
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    path = setup.data_root / "jobs" / job_id / "job.json"
    job = job_file(setup, job_id)
    if current == "absent":
        del job["delivery"]
    elif current is not None:
        job["delivery"] = {"state": current, "tag": delivery.tag(job_id), "requested_at": "t"}
    path.write_text(json.dumps(job), encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(JobError) as refused:
        delivery.request_redelivery(job_id, setup)

    assert refused.value.code == "delivery_not_unconfirmed"
    assert "only delivery_unconfirmed" in refused.value.message
    assert path.read_bytes() == before


def test_redeliver_of_an_unknown_job_is_job_not_found(setup: Config) -> None:
    with pytest.raises(JobError) as refused:
        delivery.request_redelivery("f" * 32, setup)
    assert refused.value.code == "job_not_found"


def test_next_answers_redeliver_with_the_marker_written_first(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = unconfirmed_job(setup, make_wav("recordings/a.wav"), session_id="chat-9")
    requested = delivery.request_redelivery(job_id, setup)["delivery"]["requested_at"]

    answer = delivery.next_delivery("default", "loop-2", setup)

    assert answer == {
        "action": "redeliver",
        "job_id": job_id,
        "session_id": "chat-9",
        "tag": f"sound-scout-job:{job_id}",
        "text": delivery.note_text(job_file(setup, job_id), setup),
    }
    marker = job_file(setup, job_id)["delivery"]
    assert marker["state"] == "sending" and marker["loop_id"] == "loop-2"
    assert marker["redelivery"] is True and marker["requested_at"] == requested
    assert delivery.next_delivery("default", "loop-2", setup) == {"action": "none"}
    check = delivery.next_delivery("default", "loop-3", setup)
    assert check["action"] == "check" and check["job_id"] == job_id


@pytest.mark.parametrize("outcome", ["delivered", "delivered_by_history", "unconfirmed", "failed"])
def test_every_outcome_of_a_redelivery_carries_redelivery(
    setup: Config, make_wav: WavFactory, outcome: str
) -> None:
    job_id = unconfirmed_job(setup, make_wav("recordings/a.wav"))
    delivery.request_redelivery(job_id, setup)
    delivery.next_delivery("default", "loop-1", setup)

    delivery.end_delivery(job_id, outcome, "timed out", setup)

    recorded = job_file(setup, job_id)["delivery"]
    assert recorded["redelivery"] is True and recorded["ended_at"]
    assert recorded["state"] == delivery.OUTCOMES[outcome][0]


def test_a_released_redelivery_waits_as_redelivery_requested_not_as_a_plain_send(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = unconfirmed_job(setup, make_wav("recordings/a.wav"))
    requested = delivery.request_redelivery(job_id, setup)["delivery"]
    delivery.next_delivery("default", "loop-1", setup)

    delivery.end_delivery(job_id, "released", "no gateway in this process owns it", setup)

    assert job_file(setup, job_id)["delivery"] == requested
    assert delivery.next_delivery("default", "loop-1", setup)["action"] == "redeliver"


def test_two_redelivery_requests_are_recorded_each_with_its_time(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = unconfirmed_job(setup, make_wav("recordings/a.wav"))
    for _ in range(2):
        delivery.request_redelivery(job_id, setup)
        delivery.next_delivery("default", "loop-1", setup)
        delivery.end_delivery(job_id, "unconfirmed", "history unread", setup)

    first, second = job_file(setup, job_id)["redeliveries"]
    assert first["requested_at"] < second["requested_at"]
    assert first["previous"]["state"] == second["previous"]["state"] == "delivery_unconfirmed"
    assert "redelivery" not in first["previous"] and second["previous"]["redelivery"] is True


def test_jobs_redeliver_through_the_console_entry(
    setup: Config, config_file: Path, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = unconfirmed_job(setup, wav)
    other = unconfirmed_job(setup, wav)

    as_text = run_cli(config_file, "redeliver", job_id)
    assert as_text.returncode == 0, as_text.stderr
    assert "redelivery_requested" in as_text.stdout
    as_json = run_cli(config_file, "redeliver", "--json", other)
    assert as_json.returncode == 0, as_json.stderr
    assert json.loads(as_json.stdout)["delivery"]["state"] == "redelivery_requested"

    path = setup.data_root / "jobs" / job_id / "job.json"
    before = path.read_bytes()
    again = run_cli(config_file, "redeliver", job_id)
    assert again.returncode == 1 and again.stdout == ""
    assert again.stderr.startswith("sound-scout jobs: delivery_not_unconfirmed: ")
    assert path.read_bytes() == before
    unknown = run_cli(config_file, "redeliver", "f" * 32)
    assert unknown.returncode == 1
    assert unknown.stderr.startswith("sound-scout jobs: job_not_found: ")


# --- acceptance criteria of #122 through the real console entry ------------------------------


@pytest.mark.parametrize(
    "current",
    [None, "absent", "sending", "delivered", "delivery_failed", "redelivery_requested"],
)
def test_criterion_2_jobs_redeliver_refuses_every_other_delivery_state_and_keeps_job_json(
    setup: Config, config_file: Path, make_wav: WavFactory, current: str | None
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    path = setup.data_root / "jobs" / job_id / "job.json"
    job = job_file(setup, job_id)
    if current == "absent":
        del job["delivery"]
    elif current is not None:
        job["delivery"] = {"state": current, "tag": delivery.tag(job_id), "requested_at": "t"}
    path.write_text(json.dumps(job), encoding="utf-8")
    before = path.read_bytes()

    refused = run_cli(config_file, "redeliver", job_id)

    assert refused.returncode == 1 and refused.stdout == ""
    assert refused.stderr.startswith("sound-scout jobs: delivery_not_unconfirmed: ")
    assert "only delivery_unconfirmed" in refused.stderr
    assert path.read_bytes() == before


def test_criterion_2_each_accepted_jobs_redeliver_is_recorded_in_job_json_with_its_time(
    setup: Config, config_file: Path, make_wav: WavFactory
) -> None:
    job_id = unconfirmed_job(setup, make_wav("recordings/a.wav"))
    for _ in range(2):
        accepted = run_cli(config_file, "redeliver", "--json", job_id)
        assert accepted.returncode == 0, accepted.stderr
        assert json.loads(accepted.stdout)["delivery"]["state"] == "redelivery_requested"
        delivery.next_delivery("default", "loop-1", setup)
        delivery.end_delivery(job_id, "unconfirmed", "timed out", setup)

    first, second = job_file(setup, job_id)["redeliveries"]
    for request in (first, second):
        assert datetime.fromisoformat(request["requested_at"]).tzinfo is not None
    assert first["requested_at"] < second["requested_at"]
