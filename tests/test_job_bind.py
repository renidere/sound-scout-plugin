"""`jobs.bind` and `sound-scout jobs bind`: the chat binding of an MCP job (decision 0025, #140)."""

import json
from pathlib import Path

import pytest

from sound_scout import jobs, mcp_server
from sound_scout.config import Config
from sound_scout.jobs import Binding, JobError, SubmitRequest
from tests.conftest import WavFactory
from tests.test_jobs import config_file, job_file, run_cli, set_job, setup  # noqa: F401


def mcp_job(cfg: Config, wav: Path, request_id: str = "r-1") -> str:
    answer, is_error = mcp_server.call(
        "speech_job_submit", {"path": str(wav), "request_id": request_id}, cfg
    )
    assert is_error is False, answer
    return answer["job_id"]


def test_bind_writes_the_chat_and_bound_at_and_keeps_the_identity(
    setup: Config,  # noqa: F811
    make_wav: WavFactory,
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = mcp_job(setup, wav)
    before = job_file(setup, job_id)

    answer = jobs.bind(job_id, "chat-a", "default", None, setup)

    after = job_file(setup, job_id)
    assert after["binding"] == {
        "submitter": "mcp",
        "connection": None,
        "profile": "default",
        "session_id": "chat-a",
    }
    assert isinstance(after["bound_at"], str)
    assert after["identity_hash"] == before["identity_hash"]
    assert after["state"] == before["state"] and after["delivery"] is None
    assert answer == jobs.status(job_id, setup)
    assert answer["binding"]["session_id"] == "chat-a" and answer["bound_at"] == after["bound_at"]
    again, is_error = mcp_server.call(
        "speech_job_submit", {"path": str(wav), "request_id": "r-1"}, setup
    )
    assert is_error is False and again == {
        "job_id": job_id,
        "request_id": "r-1",
        "state": "queued",
        "reused": True,
    }


def test_an_ended_job_is_bound_too(setup: Config, make_wav: WavFactory) -> None:  # noqa: F811
    job_id = mcp_job(setup, make_wav("recordings/a.wav"))
    set_job(setup, job_id, state="finished")

    assert jobs.bind(job_id, "chat-a", None, None, setup)["binding"]["session_id"] == "chat-a"


def test_a_second_bind_is_not_bindable_and_changes_nothing(
    setup: Config,  # noqa: F811
    make_wav: WavFactory,
) -> None:
    job_id = mcp_job(setup, make_wav("recordings/a.wav"))
    jobs.bind(job_id, "chat-a", "default", None, setup)
    bound = job_file(setup, job_id)

    with pytest.raises(JobError) as refused:
        jobs.bind(job_id, "chat-b", "default", None, setup)

    assert refused.value.code == jobs.NOT_BINDABLE == "not_bindable"
    assert job_file(setup, job_id) == bound


@pytest.mark.parametrize("submitter", ["cli", "hermes-plugin", "desktop"])
def test_a_job_of_another_submitter_is_not_bindable(
    submitter: str,
    setup: Config,  # noqa: F811
    make_wav: WavFactory,
) -> None:
    wav = make_wav("recordings/a.wav")
    job_id = jobs.submit(SubmitRequest(str(wav)), Binding(submitter=submitter), setup)["job_id"]
    before = job_file(setup, job_id)

    with pytest.raises(JobError) as refused:
        jobs.bind(job_id, "chat-a", None, None, setup)

    assert refused.value.code == "not_bindable"
    assert job_file(setup, job_id) == before


@pytest.mark.parametrize("delivery", [{"state": "delivered"}, "missing"])
def test_a_job_with_a_delivery_state_or_without_the_field_is_not_bindable(
    delivery: object,
    setup: Config,  # noqa: F811
    make_wav: WavFactory,
) -> None:
    job_id = mcp_job(setup, make_wav("recordings/a.wav"))
    path = setup.data_root / "jobs" / job_id / "job.json"
    job = job_file(setup, job_id)
    if delivery == "missing":
        del job["delivery"]
    else:
        job["delivery"] = delivery
    path.write_text(json.dumps(job), encoding="utf-8")

    with pytest.raises(JobError) as refused:
        jobs.bind(job_id, "chat-a", None, None, setup)

    assert refused.value.code == "not_bindable"
    assert job_file(setup, job_id) == job


@pytest.mark.parametrize(
    ("session_id", "profile", "connection"),
    [("", None, None), ("x" * 257, None, None), (7, None, None), ("chat-a", "", None)],
)
def test_a_malformed_field_is_invalid_request(
    session_id: object,
    profile: object,
    connection: object,
    setup: Config,  # noqa: F811
    make_wav: WavFactory,
) -> None:
    job_id = mcp_job(setup, make_wav("recordings/a.wav"))
    before = job_file(setup, job_id)

    with pytest.raises(JobError) as refused:
        jobs.bind(job_id, session_id, profile, connection, setup)

    assert refused.value.code == "invalid_request"
    assert job_file(setup, job_id) == before


def test_an_unknown_job_is_job_not_found(setup: Config) -> None:  # noqa: F811
    with pytest.raises(JobError) as refused:
        jobs.bind("f" * 32, "chat-a", None, None, setup)

    assert refused.value.code == "job_not_found"


@pytest.mark.slow
def test_jobs_bind_through_the_console(config_file: Path, make_wav: WavFactory) -> None:  # noqa: F811
    wav = make_wav("recordings/a.wav")
    data = config_file.parent / "data"
    cfg = Config(allowed_roots=(config_file.parent / "recordings",), data_root=data)
    job_id = mcp_job(cfg, wav)

    as_text = run_cli(config_file, "bind", "--session-id=-chat-a", "--profile=default", job_id)
    assert as_text.returncode == 0, as_text.stderr
    assert f"job_id: {job_id}" in as_text.stdout.splitlines()
    assert job_file(cfg, job_id)["binding"]["session_id"] == "-chat-a"

    again = run_cli(config_file, "bind", "--session-id=chat-b", job_id, "--json")
    assert again.returncode == 1 and again.stdout == ""
    assert again.stderr.startswith("sound-scout jobs: not_bindable: ")

    other_id = jobs.submit(SubmitRequest(str(wav), request_id="r-2"), Binding(), cfg)["job_id"]
    set_job(cfg, other_id, state="cancelled")
    refused = run_cli(config_file, "bind", "--session-id=chat-b", other_id)
    assert refused.returncode == 1 and refused.stderr.startswith("sound-scout jobs: not_bindable: ")

    set_job(cfg, job_id, binding={"submitter": "mcp"} | dict.fromkeys(jobs.BINDING_FIELDS))
    as_json = run_cli(config_file, "bind", "--session-id=chat-c", job_id, "--json")
    assert as_json.returncode == 0, as_json.stderr
    assert json.loads(as_json.stdout)["binding"]["session_id"] == "chat-c"

    unknown = run_cli(config_file, "bind", "--session-id=chat-a", "f" * 32)
    assert unknown.returncode == 1 and unknown.stderr.startswith("sound-scout jobs: job_not_found:")
