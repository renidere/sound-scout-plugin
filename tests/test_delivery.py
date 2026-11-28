"""Delivery state of a job and the text of its final note (`sound_scout.delivery`, #102)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from sound_scout import app, config, delivery, jobs, mcp_server, report
from sound_scout.config import Config
from sound_scout.jobs import Binding, JobError, SubmitRequest
from sound_scout.smoke import speech_like_control
from tests.conftest import WavFactory

ABSENT_ASR_MODEL = "absent/model"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)
ALL_DONE = ("готово", "всё выполнено", "все выполнено", "всё хорошо")


@pytest.fixture(autouse=True)
def turn_wait_over(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests of the note state machine run after the turn wait (decision 0030); the tests of
    the wait itself set `TURN_WAIT_SECONDS` back."""
    monkeypatch.setattr(delivery, "TURN_WAIT_SECONDS", 0)


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
    asr_reason = result["stages"]["asr"]["reason"]
    assert asr_reason
    assert f"- содержание (ASR): `unavailable` — недоступно; причина: {asr_reason}" in lines
    for layer, name in (("measurements", "измерения (акустика)"), ("sound", "звучание")):
        status = result["stages"][jobs.LAYER_STAGES[layer]]["status"]
        assert any(line.startswith(f"- {name}") and f"`{status}`" in line for line in lines)
    assert any(line.startswith("Не оценено: ") for line in lines)
    assert f"Отчёт: {result['run_dir']}/report.md" in lines
    assert lines[-1] == f"sound-scout-job:{job_id}"
    assert not any(phrase in text.lower() for phrase in ALL_DONE)
    for private in (str(wav), str(setup.data_root), str(setup.allowed_roots[0])):
        assert private not in text
    assert any(t.startswith("word_prominence") for t in result["limitations"])
    assert "word_prominence" not in text and "pitch accent" not in text  # no measurements (0038)


def set_stages(cfg: Config, job_id: str, **stages: tuple[str, str]) -> None:
    """Rewrite the stages of the job's `result.json` as an interrupted worker leaves them."""
    run_id = job_file(cfg, job_id)["attempts"][0]["run_id"]
    path = cfg.data_root / "runs" / run_id / "result.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for stage, (status, reason) in stages.items():
        document["stages"][stage] |= {"status": status, "reason": reason}
    path.write_text(json.dumps(document), encoding="utf-8")


@needs_ffmpeg
def test_note_of_a_job_cancelled_while_asr_runs_says_cancelled_on_each_layer(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = analyzed_job(setup, make_wav("recordings/tone.wav", seconds=1.5), state="cancelled")
    cancelled = app.CANCELLED_REASON
    set_stages(
        setup,
        job_id,
        asr=("failed", cancelled),
        acoustics=("skipped", cancelled),
        audio_review=("skipped", cancelled),
    )

    lines = delivery.next_delivery("default", "loop-1", setup)["text"].splitlines()

    assert "Задание отменено." in lines
    assert "- содержание (ASR): `failed` — ошибка; причина: отменён" in lines
    assert "- измерения (акустика): `skipped` — пропущено; причина: отменён" in lines
    assert "- звучание (audio review): `skipped` — пропущено; причина: отменён" in lines


@needs_ffmpeg
def test_note_of_the_stages_app_cuts_at_a_cancel_during_asr_says_cancelled_on_each_layer(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = analyzed_job(setup, make_wav("recordings/tone.wav", seconds=1.5), state="cancelled")
    run_id = job_file(setup, job_id)["attempts"][0]["run_id"]
    run_dir = setup.data_root / "runs" / run_id
    prepare = json.loads((run_dir / "work" / "stage-prepare.json").read_text(encoding="utf-8"))
    cut, _ = app._stages(
        app.LAYERS,
        {"prepare": prepare},
        app._Interruption(app.CANCELLED, app.CANCELLED_REASON),
    )
    assert {name: (cut[name]["status"], cut[name]["reason"]) for name in app.WORKER_STAGES[1:]} == {
        "asr": ("failed", "отменён"),
        "acoustics": ("skipped", "отменён"),
        "audio_review": ("skipped", "отменён"),
    }
    set_stages(setup, job_id, **{name: (cut[name]["status"], cut[name]["reason"]) for name in cut})

    lines = delivery.next_delivery("default", "loop-1", setup)["text"].splitlines()

    assert lines[1:5] == [
        "Задание отменено.",
        "- содержание (ASR): `failed` — ошибка; причина: отменён",
        "- измерения (акустика): `skipped` — пропущено; причина: отменён",
        "- звучание (audio review): `skipped` — пропущено; причина: отменён",
    ]


@needs_ffmpeg
def test_note_shows_a_failed_stage_reason_without_private_values_and_keeps_completed_lines(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    job_id = analyzed_job(setup, wav)
    error = f"ffmpeg exited with 1 on {wav}\nin {setup.data_root}/runs"
    set_stages(
        setup,
        job_id,
        asr=("failed", error),
        acoustics=("completed", "measured on the working mono"),
    )

    text = delivery.next_delivery("default", "loop-1", setup)["text"]

    lines = text.splitlines()
    [asr_line] = [line for line in lines if line.startswith("- содержание (ASR): ")]
    assert asr_line == (
        f"- содержание (ASR): `failed` — ошибка; причина: ffmpeg exited with 1 on "
        f"{report.PRIVATE_PATH} in {report.PRIVATE_PATH}"
    )
    assert "- измерения (акустика): `completed` — выполнено" in lines
    for private in (str(wav), str(setup.data_root), str(setup.allowed_roots[0])):
        assert private not in text


@needs_ffmpeg
@pytest.mark.parametrize(
    "private_path",
    ("/mnt/usb/Voice Memos/call.m4a", "~/Voice Memos/call.m4a", r"C:\Voice Memos\call.m4a"),
)
def test_note_masks_the_entire_path_with_spaces(
    setup: Config, make_wav: WavFactory, private_path: str
) -> None:
    job_id = analyzed_job(setup, make_wav("recordings/tone.wav", seconds=1.5))
    set_stages(setup, job_id, asr=("failed", f"ffprobe failed: {private_path}"))

    text = delivery.note_text(job_file(setup, job_id), setup)

    assert (
        f"- содержание (ASR): `failed` — ошибка; причина: ffprobe failed: {report.PRIVATE_PATH}"
    ) in text.splitlines()
    assert "Voice" not in text and "Memos" not in text and "call.m4a" not in text


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


@pytest.mark.slow
def test_deliver_next_and_end_through_the_console_entry(
    setup: Config, config_file: Path, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))

    none = run_cli(config_file, "deliver-next", "--profile=work", "--loop=l-1")
    assert none.returncode == 0, none.stderr
    assert none.stdout.strip() == "action: none"

    sent = run_cli(
        config_file,
        "deliver-next",
        "--profile=default",
        "--loop=l-1",
        "--idle-chat=chat-1",
        "--json",
    )
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


def claim(cfg: Config, job_id: str, loop_id: str = "loop-1") -> dict:
    """`next_delivery` after the loop found the job's chat without a turn."""
    session_id = job_file(cfg, job_id)["binding"]["session_id"]
    return delivery.next_delivery("default", loop_id, cfg, idle_chats=(session_id,))


def unconfirmed_job(cfg: Config, wav: Path, **kwargs: object) -> str:
    job_id = ended_job(cfg, wav, **kwargs)
    claim(cfg, job_id)
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
        "attempt": 0,
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


@pytest.mark.slow
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


# --- one note per attempt after a retry (#187) ------------------------------------------------


def run_attempt(cfg: Config, job_id: str, state: str = "finished") -> None:
    """Stand in for the runner: one more attempt that starts now and ends `state`."""
    job = job_file(cfg, job_id)
    number = len(job["attempts"]) + 1
    attempt = {"number": number, "request_id": f"{job_id}-a{number}", "run_id": None}
    attempt["started_at"] = jobs._now()
    set_job(cfg, job_id, attempts=[*job["attempts"], attempt], state=state, state_reason=None)


def delivered(cfg: Config, job_id: str) -> dict:
    """The loop takes the next note of `job_id` and its send ends `delivered`."""
    answer = delivery.next_delivery("default", "loop-1", cfg)
    assert answer["action"] == "send" and answer["job_id"] == job_id
    delivery.end_delivery(job_id, "delivered", None, cfg)
    return answer


@pytest.mark.parametrize("ended", ["cancelled", "failed"])
def test_a_retried_attempt_gets_one_note_of_its_own_with_its_own_tag(
    setup: Config, make_wav: WavFactory, ended: str
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, ended)
    first = delivered(setup, job_id)
    noted = job_file(setup, job_id)["delivery"]

    jobs.retry(job_id, setup)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    run_attempt(setup, job_id)
    second = delivery.next_delivery("default", "loop-1", setup)

    assert first["tag"] == f"sound-scout-job:{job_id}" and noted["attempt"] == 1
    assert noted["state"] == "delivered"
    assert second["action"] == "send" and second["session_id"] == "chat-1"
    assert second["tag"] == f"sound-scout-job:{job_id}-a2" != first["tag"]
    assert second["text"].splitlines()[-1] == second["tag"]
    assert first["tag"] not in second["text"].splitlines()
    job = job_file(setup, job_id)
    assert job["deliveries"] == [noted]
    assert job["delivery"]["state"] == "sending" and job["delivery"]["attempt"] == 2
    delivery.end_delivery(job_id, "delivered", None, setup)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    assert delivery.next_delivery("default", "loop-2", setup) == {"action": "none"}
    final = jobs.status(job_id, setup)["delivery"]
    assert final["state"] == "delivered" and final["tag"] == second["tag"]
    assert final["attempt"] == 2


def test_each_later_retry_gets_its_own_tag_and_keeps_every_earlier_note(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "failed")
    tags = [delivered(setup, job_id)["tag"]]
    for _ in range(2):
        jobs.retry(job_id, setup)
        run_attempt(setup, job_id, "failed")
        tags.append(delivered(setup, job_id)["tag"])

    job = job_file(setup, job_id)
    assert tags == [f"sound-scout-job:{job_id}", f"sound-scout-job:{job_id}-a2"] + [
        f"sound-scout-job:{job_id}-a3"
    ]
    assert [item["tag"] for item in job["deliveries"]] == tags[:2]
    assert job["delivery"]["tag"] == tags[2]


def test_a_job_without_a_retry_gets_one_note_with_the_job_tag_after_a_recovered_attempt(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "running")
    run_attempt(setup, job_id)

    answer = delivered(setup, job_id)

    assert answer["tag"] == f"sound-scout-job:{job_id}"
    assert job_file(setup, job_id)["delivery"]["attempt"] == 2
    assert "deliveries" not in job_file(setup, job_id)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}


def test_a_retry_before_the_first_note_gives_one_note_of_the_last_attempt(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "cancelled")
    jobs.retry(job_id, setup)
    run_attempt(setup, job_id)

    answer = delivered(setup, job_id)

    assert answer["tag"] == f"sound-scout-job:{job_id}"
    assert job_file(setup, job_id)["delivery"]["attempt"] == 2
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}


def test_a_retry_cancelled_in_the_queue_runs_no_attempt_and_gets_no_note(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="cancelled")
    first = delivered(setup, job_id)
    assert job_file(setup, job_id)["delivery"]["attempt"] == 0

    jobs.retry(job_id, setup)
    jobs.cancel(job_id, setup)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    jobs.retry(job_id, setup)
    run_attempt(setup, job_id)
    second = delivered(setup, job_id)

    assert first["tag"] == f"sound-scout-job:{job_id}"
    assert second["tag"] == f"sound-scout-job:{job_id}-a1"
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}


def test_a_stopped_loop_of_a_retried_attempt_leaves_a_check_with_the_attempt_tag(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "cancelled")
    delivered(setup, job_id)
    jobs.retry(job_id, setup)
    run_attempt(setup, job_id)
    sent = delivery.next_delivery("default", "loop-1", setup)

    check = delivery.next_delivery("default", "loop-2", setup)

    assert check == {
        "action": "check",
        "job_id": job_id,
        "session_id": "chat-1",
        "tag": f"sound-scout-job:{job_id}-a2",
    }
    assert sent["tag"] == check["tag"]
    delivery.end_delivery(job_id, "delivered_by_history", None, setup)
    recorded = job_file(setup, job_id)["delivery"]
    assert recorded["tag"] == check["tag"] and recorded["attempt"] == 2
    assert delivery.next_delivery("default", "loop-2", setup) == {"action": "none"}


def test_a_redelivery_of_a_retried_attempt_keeps_its_tag_and_attempt(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "failed")
    delivered(setup, job_id)
    jobs.retry(job_id, setup)
    run_attempt(setup, job_id)
    delivery.next_delivery("default", "loop-1", setup)
    delivery.end_delivery(job_id, "unconfirmed", "timed out", setup)

    requested = delivery.request_redelivery(job_id, setup)["delivery"]
    answer = delivery.next_delivery("default", "loop-2", setup)

    tag = f"sound-scout-job:{job_id}-a2"
    assert requested["tag"] == tag and requested["attempt"] == 2
    assert answer["action"] == "redeliver" and answer["tag"] == tag
    assert answer["text"].splitlines()[-1] == tag
    assert len(job_file(setup, job_id)["deliveries"]) == 1
    delivery.end_delivery(job_id, "released", "no gateway in this process owns it", setup)
    waiting = job_file(setup, job_id)["delivery"]
    assert waiting["state"] == "redelivery_requested" and waiting["attempt"] == 2
    assert delivery.next_delivery("default", "loop-2", setup)["tag"] == tag


def test_a_released_note_of_a_retried_attempt_is_claimed_again_with_the_same_tag(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "failed")
    delivered(setup, job_id)
    jobs.retry(job_id, setup)
    run_attempt(setup, job_id)
    first = delivery.next_delivery("default", "loop-1", setup)
    delivery.end_delivery(job_id, "released", "no gateway in this process owns it", setup)

    again = delivery.next_delivery("default", "loop-1", setup)

    assert again["tag"] == first["tag"] == f"sound-scout-job:{job_id}-a2"
    assert len(job_file(setup, job_id)["deliveries"]) == 1


def test_a_note_recorded_before_187_counts_the_attempts_that_started_before_it(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "failed")
    delivered(setup, job_id)
    old = job_file(setup, job_id)["delivery"]
    del old["attempt"]
    set_job(setup, job_id, delivery=old)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}

    jobs.retry(job_id, setup)
    run_attempt(setup, job_id)
    answer = delivery.next_delivery("default", "loop-1", setup)

    assert answer["action"] == "send" and answer["tag"] == f"sound-scout-job:{job_id}-a2"
    assert job_file(setup, job_id)["deliveries"] == [old | {"attempt": 1}]


def test_a_note_recorded_before_187_of_a_job_retried_before_the_update_gets_no_late_note(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "failed")
    delivered(setup, job_id)
    old = job_file(setup, job_id)["delivery"]
    del old["attempt"]
    set_job(setup, job_id, delivery=old)
    run_attempt(setup, job_id, "failed")

    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    assert job_file(setup, job_id)["delivery"] == old
    assert "deliveries" not in job_file(setup, job_id)


def test_a_redelivery_request_superseded_by_a_later_attempt_gives_way_to_its_note(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "failed")
    delivery.next_delivery("default", "loop-1", setup)
    delivery.end_delivery(job_id, "unconfirmed", "timed out", setup)
    requested = delivery.request_redelivery(job_id, setup)["delivery"]
    jobs.retry(job_id, setup)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    run_attempt(setup, job_id)

    answer = delivery.next_delivery("default", "loop-1", setup)

    assert answer["action"] == "send" and answer["tag"] == f"sound-scout-job:{job_id}-a2"
    job = job_file(setup, job_id)
    assert job["deliveries"] == [requested]
    assert requested["state"] == "redelivery_requested" and requested["attempt"] == 1
    delivery.end_delivery(job_id, "delivered", None, setup)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}


def test_the_note_of_a_retried_attempt_never_reports_the_result_of_an_earlier_one(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = analyzed_job(setup, wav, state="cancelled")
    first = delivered(setup, job_id)
    jobs.retry(job_id, setup)
    run_attempt(setup, job_id, "failed")
    set_job(setup, job_id, state_reason="input_error: the input is gone")

    answer = delivery.next_delivery("default", "loop-1", setup)

    assert first["text"].startswith("Sound Scout: анализ записи закончен")
    assert jobs.result(job_id, setup)["attempt"] == 1
    lines = answer["text"].splitlines()
    assert lines[0] == (
        "Sound Scout: анализ записи не дал результата — задание `failed`, код `input_error`."
    )
    assert not any(line.startswith("Отчёт:") for line in lines)
    assert lines[-1] == f"sound-scout-job:{job_id}-a2"


@pytest.mark.slow
def test_the_note_of_a_retried_attempt_with_its_own_result_reports_that_result(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = analyzed_job(setup, wav, state="cancelled")
    delivered(setup, job_id)
    jobs.retry(job_id, setup)
    layers = tuple(job_file(setup, job_id)["request"]["layers"])
    analysis = app.analyze(
        app.AnalyzeRequest(str(wav), request_id=f"{job_id}-a2", layers=layers), setup
    )
    attempts = job_file(setup, job_id)["attempts"]
    attempt = {"number": 2, "request_id": f"{job_id}-a2", "run_id": analysis.run.run_id}
    set_job(setup, job_id, attempts=[*attempts, attempt], state="finished", state_reason=None)

    answer = delivery.next_delivery("default", "loop-1", setup)

    lines = answer["text"].splitlines()
    assert lines[0].startswith("Sound Scout: анализ записи закончен — `partial`")
    assert f"Отчёт: runs/{analysis.run.run_id}/report.md" in lines
    assert lines[-1] == f"sound-scout-job:{job_id}-a2"


@pytest.mark.slow
def test_a_retried_attempt_gets_its_note_through_the_console_entry(
    setup: Config, make_wav: WavFactory, config_file: Path
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "cancelled")
    delivered(setup, job_id)
    retried = run_cli(config_file, "retry", job_id, "--json")
    assert retried.returncode == 0, retried.stderr
    run_attempt(setup, job_id)

    sent = run_cli(
        config_file, "deliver-next", "--profile=default", "--loop=l", "--idle-chat=chat-1", "--json"
    )
    ended = run_cli(config_file, "deliver-end", "--outcome=delivered", job_id, "--json")
    again = run_cli(config_file, "deliver-next", "--profile=default", "--loop=l", "--json")

    assert json.loads(sent.stdout)["tag"] == f"sound-scout-job:{job_id}-a2"
    assert ended.returncode == 0, ended.stderr
    assert json.loads(ended.stdout)["delivery"]["state"] == "delivered"
    assert json.loads(again.stdout) == {"action": "none"}


# --- the comparison block of a control job's note (#110) -------------------------------------


def speech_wav(cfg: Config, name: str, f0: float) -> Path:
    """A 1.5 s formant-synthesised 16-bit WAV the Silero VAD hears as one speech span."""
    samples = np.round(speech_like_control(1.5, 16_000, f0=f0) * np.iinfo(np.int16).max)
    path = cfg.allowed_roots[0] / name
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16_000)
        out.writeframes(samples.astype(np.int16).tobytes())
    return path


def control_job(cfg: Config, initial_wav: Path, wav: Path) -> str:
    """A control job the real runner analyzed and compared; `backend = none`."""
    initial = app.analyze(app.AnalyzeRequest(str(initial_wav), request_id="initial"), cfg)
    request = SubmitRequest(
        str(wav), previous_run_id=initial.run.run_id, criterion="completeness_of_phrase"
    )
    job_id = jobs.submit(request, Binding(session_id="chat-1"), cfg)["job_id"]
    jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=0.02)
    return job_id


@pytest.mark.slow
@needs_ffmpeg
def test_note_of_a_control_job_without_a_backend_says_inconclusive_and_why(
    setup: Config,
) -> None:
    initial_wav = speech_wav(setup, "initial.wav", 130.0)
    job_id = control_job(setup, initial_wav, speech_wav(setup, "control.wav", 180.0))
    summary = job_file(setup, job_id)["attempts"][-1]["compare"]
    document = json.loads((setup.data_root / summary["path"]).read_text(encoding="utf-8"))

    text = delivery.next_delivery("default", "loop-1", setup)["text"]

    lines = text.splitlines()
    assert lines[-1] == f"sound-scout-job:{job_id}"
    assert lines[-2] == f"Сравнение: {summary['path']}"
    verdict = next(line for line in lines if line.startswith("Сравнение с исходной попыткой"))
    assert verdict == (
        "Сравнение с исходной попыткой по критерию `completeness_of_phrase`: `inconclusive` — "
        "сравнение недостоверно."
    )
    reasons = next(line for line in lines if line.startswith("Причины: "))
    for side in ("A", "B"):
        assert (
            f"попытка {side}: audio review `unavailable` — audio backend не выбран "
            "(`backend = none`)"
        ) in reasons
    level = document["conditions_diff"]["level_dbfs"]
    assert isinstance(level["a"], float) and isinstance(level["b"], float), level
    assert "уровень речи не измерен" not in text
    assert [(item["code"], item["attempt"]) for item in document["reasons"]] == [
        ("audio_review_missing", "a"),
        ("audio_review_missing", "b"),
    ]
    for item in document["reasons"]:  # its message quotes the stage reason; the note never does
        assert item["message"] not in text
    assert not any(phrase in text.lower() for phrase in ALL_DONE)
    assert str(setup.data_root) not in text


FRESH_ANALYSIS = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from sound_scout import app, config
    from sound_scout.config import Config

    model, roots, data_root, *wavs = sys.argv[1:]
    cfg = Config(
        allowed_roots=(Path(roots),), data_root=Path(data_root), asr=config.Asr(model=model)
    )
    levels = {}
    for wav in wavs:
        result = app.analyze(app.AnalyzeRequest(wav, request_id=Path(wav).stem), cfg).result
        levels[Path(wav).name] = {m["name"]: m for m in result["measurements"]}
    print(json.dumps(levels))
    """
)


def level_of(measurements: dict) -> tuple:
    rms, spans = measurements["rms_dbfs"], measurements["speech_spans"]
    return rms["value"], rms["reason"], rms["settings"]["speech_frames"], spans["value"]


def analyzed_alone(cfg: Config, wavs: list[Path]) -> dict:
    """The same analyses in a fresh interpreter with a minimal environment: no other test ran."""
    data_root = Path(tempfile.mkdtemp(dir=cfg.data_root.parent))
    env = {name: os.environ[name] for name in ("PATH", "HOME", "TMPDIR") if name in os.environ}
    out = subprocess.run(
        [sys.executable, "-I", "-c", FRESH_ANALYSIS, ABSENT_ASR_MODEL]
        + [str(cfg.allowed_roots[0]), str(data_root)]
        + [str(wav) for wav in wavs],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout
    return {name: level_of(item) for name, item in json.loads(out).items()}


@pytest.mark.slow
@needs_ffmpeg
@pytest.mark.parametrize("recording", ["tone", "speech_like"])
def test_the_level_of_each_attempt_is_the_same_in_the_suite_and_alone(
    setup: Config, make_wav: WavFactory, recording: str
) -> None:
    """#125: the former (tone) and current (speech_like) control note inputs give one level each."""
    if recording == "tone":
        wavs = [
            make_wav("recordings/initial.wav", seconds=1.5),
            make_wav("recordings/control.wav", seconds=1.5, frequency=330.0),
        ]
    else:
        wavs = [speech_wav(setup, "initial.wav", 130.0), speech_wav(setup, "control.wav", 180.0)]
    in_suite = {
        wav.name: level_of(
            {
                m["name"]: m
                for m in app.analyze(app.AnalyzeRequest(str(wav)), setup).result["measurements"]
            }
        )
        for wav in wavs
    }

    assert analyzed_alone(setup, wavs) == in_suite
    for value, reason, speech_frames, spans in in_suite.values():
        if recording == "tone":  # a pure tone is not speech for the VAD: null with its reason
            assert (value, reason, speech_frames, spans) == (
                None,
                "no speech frames with signal",
                0,
                [],
            )
        else:
            assert reason is None and spans and speech_frames > 0
            assert -30.0 < value["p50"] < -10.0


@pytest.mark.parametrize(
    ("summary", "line"),
    [
        (
            {"state": "completed", "verdict": "improved", "reasons": []},
            " по критерию `emphasis_on:слово`: `improved` — улучшение по этому критерию; один "
            "контроль не доказывает устойчивый навык.",
        ),
        (
            {"state": "completed", "verdict": "not_improved", "reasons": []},
            " по критерию `emphasis_on:слово`: `not_improved` — улучшения по этому критерию нет.",
        ),
        (
            {"state": "refused", "code": "run_not_found", "message": "run x is not here"},
            " не выполнено: код `run_not_found`.",
        ),
        (
            {"state": "failed", "error": "internal_error: ValueError"},
            " не выполнено: внутренняя ошибка.",
        ),
        (
            {"state": "not_run", "reason": "cancelled"},
            " не выполнено: контрольный анализ не закончен (`cancelled`).",
        ),
        (None, " не выполнено: контрольный анализ не закончен (`failed`)."),
    ],
    ids=["improved", "not_improved", "refused", "failed", "not_run", "no_summary"],
)
def test_note_labels_each_comparison_summary_written_into_job_json(
    setup: Config, make_wav: WavFactory, summary: dict | None, line: str
) -> None:
    """The text contract of each label on a summary written by hand, not a capability (R01)."""
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="failed", reason="x: y")
    control = {"previous_run_id": "a" * 32, "criterion": "emphasis_on:слово"}
    compare_id = "c" * 32
    if summary is not None and summary["state"] == "completed":
        summary |= {"criterion": "emphasis_on:слово", "path": f"compare/{compare_id}.json"}
    attempt = {"number": 1, "request_id": f"{job_id}-a1", "run_id": None}
    if summary is not None:
        attempt["compare"] = summary
    request = job_file(setup, job_id)["request"] | {"control": control}
    set_job(setup, job_id, request=request, attempts=[attempt])

    lines = delivery.next_delivery("default", "loop-1", setup)["text"].splitlines()

    assert lines[-1] == f"sound-scout-job:{job_id}"
    assert f"Сравнение с исходной попыткой{line}" in lines
    assert not any(item.startswith("Причины") for item in lines)
    assert "run x is not here" not in "\n".join(lines)
    if summary is not None and summary["state"] == "completed":
        assert lines[-2] == f"Сравнение: compare/{compare_id}.json"


def test_every_reason_code_of_compare_v1_has_a_label() -> None:
    from sound_scout import comparison

    codes = comparison.SCHEMA["$defs"]["reason"]["properties"]["code"]["enum"]
    assert sorted(delivery.COMPARE_REASON_LABELS) == sorted(codes)


def test_note_of_an_ordinary_job_has_no_comparison(setup: Config, make_wav: WavFactory) -> None:
    ended_job(setup, make_wav("recordings/a.wav"), state="failed", reason="x: y")

    text = delivery.next_delivery("default", "loop-1", setup)["text"]

    assert "Сравнение" not in text


# --- no note for an attempt the chat saw, none before the turn check (#190, decision 0030) ---


def ended_attempt(cfg: Config, wav: Path, *, session_id: str = "chat-1", ended_at=None) -> str:
    """A bound job whose attempt 1 ended `finished` at `ended_at` (now by default)."""
    job_id = ended_job(cfg, wav, session_id=session_id, state="running")
    run_attempt(cfg, job_id)
    job = job_file(cfg, job_id)
    job["attempts"][-1]["ended_at"] = ended_at or jobs._now()
    set_job(cfg, job_id, attempts=job["attempts"])
    return job_id


@pytest.fixture
def turn_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(delivery, "TURN_WAIT_SECONDS", 600)


def test_mark_seen_writes_seen_at_of_the_last_attempt_once_and_keeps_updated_at(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_attempt(setup, make_wav("recordings/a.wav"))
    before = job_file(setup, job_id)

    status = delivery.mark_seen(job_id, "chat-1", 1, setup)
    marked = job_file(setup, job_id)
    again = delivery.mark_seen(job_id, "chat-1", 1, setup)

    assert (
        marked["attempts"][-1]["seen_at"] and status["seen_at"] == marked["attempts"][-1]["seen_at"]
    )
    assert marked["updated_at"] == before["updated_at"] and marked["delivery"] is None
    assert job_file(setup, job_id) == marked and again["seen_at"] == status["seen_at"]


@pytest.mark.parametrize("case", ["other_chat", "unbound", "queued", "running", "earlier"])
def test_mark_seen_refuses_not_seen_and_leaves_job_json_as_it_is(
    setup: Config, make_wav: WavFactory, case: str
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = ended_attempt(setup, wav)
    attempt = 1
    if case == "unbound":
        job_id = ended_job(setup, wav, session_id=None, state="running")
        run_attempt(setup, job_id)
    elif case in ("queued", "running"):
        set_job(setup, job_id, state=case)
    elif case == "earlier":
        run_attempt(setup, job_id, "failed")
    before = job_file(setup, job_id)

    with pytest.raises(JobError) as refused:
        delivery.mark_seen(job_id, "chat-2" if case == "other_chat" else "chat-1", attempt, setup)

    assert refused.value.code == "not_seen"
    assert job_file(setup, job_id) == before


@pytest.mark.parametrize("attempt", [0, -1, True, "1", None])
def test_mark_seen_refuses_a_malformed_attempt(
    setup: Config, make_wav: WavFactory, attempt: object
) -> None:
    job_id = ended_attempt(setup, make_wav("recordings/a.wav"))
    with pytest.raises(JobError) as refused:
        delivery.mark_seen(job_id, "chat-1", attempt, setup)
    assert refused.value.code == "invalid_request"
    with pytest.raises(JobError) as missing:
        delivery.mark_seen("b" * 32, "chat-1", 1, setup)
    assert missing.value.code == "job_not_found"


def test_a_seen_attempt_ends_seen_in_chat_and_the_next_job_comes_in_the_same_call(
    setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    seen = ended_attempt(setup, wav)
    other = ended_attempt(setup, wav, session_id="chat-2")
    delivery.mark_seen(seen, "chat-1", 1, setup)

    answer = delivery.next_delivery("default", "loop-1", setup)

    assert answer["action"] == "send" and answer["job_id"] == other
    marker = job_file(setup, seen)["delivery"]
    assert marker["state"] == "seen_in_chat" and marker["attempt"] == 1
    assert marker["tag"] == f"sound-scout-job:{seen}"
    assert marker["seen_at"] == job_file(setup, seen)["attempts"][0]["seen_at"]
    assert marker["ended_at"]
    delivery.end_delivery(other, "delivered", None, setup)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}
    with pytest.raises(JobError) as refused:
        delivery.request_redelivery(seen, setup)
    assert refused.value.code == "delivery_not_unconfirmed"


def test_an_unseen_attempt_is_probed_and_job_json_stays_byte_for_byte(
    setup: Config, make_wav: WavFactory, turn_wait: None
) -> None:
    job_id = ended_attempt(setup, make_wav("recordings/a.wav"))
    path = setup.data_root / "jobs" / job_id / "job.json"
    before = path.read_bytes()

    answer = delivery.next_delivery("default", "loop-1", setup)

    assert answer == {"action": "probe", "job_id": job_id, "session_id": "chat-1"}
    assert path.read_bytes() == before


def test_an_idle_chat_gets_its_note_and_a_busy_chat_is_passed_over(
    setup: Config, make_wav: WavFactory, turn_wait: None
) -> None:
    wav = make_wav("recordings/a.wav")
    busy = ended_attempt(setup, wav, session_id="chat-x")
    idle = ended_attempt(setup, wav, session_id="chat-y")

    assert delivery.next_delivery("default", "l", setup, busy_chats=("chat-x",)) == {
        "action": "probe",
        "job_id": idle,
        "session_id": "chat-y",
    }
    sent = delivery.next_delivery(
        "default", "l", setup, idle_chats=("chat-y",), busy_chats=("chat-x",)
    )
    assert sent["action"] == "send" and sent["job_id"] == idle
    assert job_file(setup, busy)["delivery"] is None
    assert delivery.next_delivery("default", "l", setup, busy_chats=("chat-x",)) == {
        "action": "none"
    }
    assert delivery.next_delivery("default", "l", setup, idle_chats=("chat-x",))["job_id"] == busy


def test_after_the_turn_wait_the_note_goes_without_a_probe(
    setup: Config, make_wav: WavFactory, turn_wait: None
) -> None:
    job_id = ended_attempt(
        setup, make_wav("recordings/a.wav"), ended_at="2026-11-25T08:00:00+00:00"
    )

    answer = delivery.next_delivery("default", "loop-1", setup, busy_chats=("chat-1",))

    assert answer["action"] == "send" and answer["job_id"] == job_id
    assert job_file(setup, job_id)["delivery"]["state"] == "sending"


def test_a_redelivery_request_waits_for_the_turn_check_too(
    setup: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = unconfirmed_job(setup, make_wav("recordings/a.wav"))
    delivery.request_redelivery(job_id, setup)
    monkeypatch.setattr(delivery, "TURN_WAIT_SECONDS", 600)

    assert delivery.next_delivery("default", "loop-1", setup)["action"] == "probe"
    answer = delivery.next_delivery("default", "loop-1", setup, idle_chats=("chat-1",))
    assert answer["action"] == "redeliver"


def test_a_seen_mark_does_not_cancel_an_explicit_redelivery_request(
    setup: Config, make_wav: WavFactory, turn_wait: None
) -> None:
    job_id = ended_attempt(setup, make_wav("recordings/a.wav"))
    claim(setup, job_id)
    delivery.end_delivery(job_id, "unconfirmed", "timed out", setup)
    delivery.request_redelivery(job_id, setup)
    delivery.mark_seen(job_id, "chat-1", 1, setup)

    answer = delivery.next_delivery("default", "loop-1", setup, idle_chats=("chat-1",))

    assert answer["action"] == "redeliver" and answer["job_id"] == job_id
    job = job_file(setup, job_id)
    assert job["attempts"][-1]["seen_at"] and job["delivery"]["state"] == "sending"
    assert job["delivery"]["redelivery"] is True
    assert "seen_in_chat" not in json.dumps(job)


def test_after_a_retry_of_a_seen_attempt_the_new_attempt_gets_its_own_note(
    setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), state="running")
    run_attempt(setup, job_id, "failed")
    delivery.mark_seen(job_id, "chat-1", 1, setup)
    assert delivery.next_delivery("default", "loop-1", setup) == {"action": "none"}

    jobs.retry(job_id, setup)
    run_attempt(setup, job_id)
    second = delivery.next_delivery("default", "loop-1", setup)

    assert second["action"] == "send" and second["tag"] == f"sound-scout-job:{job_id}-a2"
    job = job_file(setup, job_id)
    assert [item["state"] for item in job["deliveries"]] == ["seen_in_chat"]
    assert job["delivery"]["attempt"] == 2


@pytest.mark.slow
def test_jobs_seen_and_deliver_next_chats_through_the_console_entry(
    setup: Config, make_wav: WavFactory, config_file: Path
) -> None:
    job_id = ended_attempt(setup, make_wav("recordings/a.wav"))

    shown = run_cli(config_file, "seen", "--session-id=chat-1", "--attempt=1", job_id)
    assert shown.returncode == 0, shown.stderr
    assert "seen_at: " in shown.stdout
    as_json = run_cli(config_file, "seen", "--session-id=chat-1", "--attempt=1", job_id, "--json")
    assert (
        json.loads(as_json.stdout)["seen_at"] == job_file(setup, job_id)["attempts"][0]["seen_at"]
    )
    for args in (["--session-id=chat-2", "--attempt=1"], ["--session-id=chat-1", "--attempt=2"]):
        refused = run_cli(config_file, "seen", *args, job_id)
        assert refused.returncode == 1 and refused.stdout == ""
        assert refused.stderr.startswith("sound-scout jobs: not_seen: ")
    for attempt in ("x", "0", "-1"):
        bad = run_cli(config_file, "seen", "--session-id=chat-1", f"--attempt={attempt}", job_id)
        assert bad.returncode == 1 and bad.stderr.startswith("sound-scout jobs: invalid_request: ")

    none = run_cli(config_file, "deliver-next", "--profile=default", "--loop=l", "--json")
    assert json.loads(none.stdout) == {"action": "none"}
    assert job_file(setup, job_id)["delivery"]["state"] == "seen_in_chat"
    other = ended_attempt(setup, make_wav("recordings/b.wav"), session_id="chat-2")
    probe = run_cli(config_file, "deliver-next", "--profile=default", "--loop=l", "--json")
    assert json.loads(probe.stdout)["action"] == "probe"
    busy = run_cli(
        config_file, "deliver-next", "--profile=default", "--loop=l", "--busy-chat=chat-2", "--json"
    )
    assert json.loads(busy.stdout) == {"action": "none"}
    sent = run_cli(
        config_file, "deliver-next", "--profile=default", "--loop=l", "--idle-chat=chat-2", "--json"
    )
    assert json.loads(sent.stdout)["job_id"] == other
    bad = run_cli(config_file, "deliver-next", "--profile=default", "--loop=l", "--idle-chat=")
    assert bad.returncode == 1 and bad.stderr.startswith("sound-scout jobs: invalid_request: ")
