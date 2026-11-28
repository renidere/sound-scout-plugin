"""The `seen` `post_tool_call` hook of the Hermes plugin (`hermes/plugin/sound-scout/seen.py`,
decision 0030, #190), without Hermes.

The same contract test as `tests/test_hermes_binding.py`: `hermes_call` stands in for Hermes
v0.23.3 (`caafe5389a`, read only). The MCP server, `sound-scout jobs seen`, `deliver-next` and
`deliver-end` are the real ones over a temporary data root; `send`, `history_has` and the turn
check are fakes. No live Hermes and no real chat are touched.
"""

import ast
import json
import logging
import sys
from pathlib import Path

import pytest

from sound_scout import jobs, mcp_server
from sound_scout.config import Config
from tests import test_hermes_binding, test_hermes_delivery
from tests.conftest import WavFactory
from tests.test_delivery import analyzed_job, ended_job, job_file, run_attempt
from tests.test_hermes_binding import hermes_call, hermes_result, local_turn
from tests.test_hermes_delivery import Chats, needs_ffmpeg, ticks

MODULE = test_hermes_binding.PACKAGE / "seen.py"
RESULT_TOOL = "mcp__speech_review__speech_job_result"
plugin = test_hermes_binding.plugin
setup = test_hermes_delivery.setup


def read_result_in_turn(plugin, cfg: Config, job_id: str, turn: dict) -> tuple[dict, bool]:
    """The agent's `speech_job_result` in a turn: the real MCP call, then the hook."""
    args = {"job_id": job_id}
    answer, is_error = mcp_server.call("speech_job_result", args, cfg)
    result, status = hermes_result(answer, is_error)
    hermes_call(
        plugin.seen.post_tool_call,
        turn=turn,
        tool_name=RESULT_TOOL,
        args=args,
        result=result,
        status=status,
    )
    return answer, is_error


def lines(chats: Chats) -> list[tuple[str, str]]:
    return [(chat, text.splitlines()[-1]) for chat, text in chats.sent]


# --- criteria 1 and 2: a result read in the bound chat's turn gets no note -------------------


@pytest.mark.slow
@needs_ffmpeg
def test_criterion_1_a_result_read_in_the_bound_chat_marks_it_seen_and_no_note_is_sent(
    plugin, setup: Config, make_wav: WavFactory, caplog: pytest.LogCaptureFixture
) -> None:
    job_id = analyzed_job(setup, make_wav("recordings/tone.wav", seconds=1.5), session_id="chat-a")
    caplog.set_level(logging.INFO)

    answer, is_error = read_result_in_turn(plugin, setup, job_id, local_turn("chat-a"))

    assert is_error is False and answer["attempt"] == 1
    assert job_file(setup, job_id)["attempts"][0]["seen_at"]
    assert "sound-scout seen: seen" in caplog.text and "chat-a" not in caplog.text
    chats = Chats(setup)
    assert ticks(plugin.delivery, chats, 2) == ["idle", "idle"]
    assert chats.sent == []
    assert jobs.status(job_id, setup)["delivery"]["state"] == "seen_in_chat"


# --- criteria 4 and 5: another chat or a failed call leaves the attempt unseen ---------------


@pytest.mark.slow
@needs_ffmpeg
def test_criterion_5_a_result_read_in_another_chat_marks_nothing_and_the_note_goes(
    plugin, setup: Config, make_wav: WavFactory, caplog: pytest.LogCaptureFixture
) -> None:
    job_id = analyzed_job(setup, make_wav("recordings/tone.wav", seconds=1.5), session_id="chat-a")
    caplog.set_level(logging.INFO)

    read_result_in_turn(plugin, setup, job_id, local_turn("chat-b"))

    assert "seen_at" not in job_file(setup, job_id)["attempts"][0]
    assert "sound-scout seen: not_seen" in caplog.text
    chats = Chats(setup)
    assert ticks(plugin.delivery, chats, 2) == ["delivered", "idle"]
    assert lines(chats) == [("chat-a", f"sound-scout-job:{job_id}")]


@pytest.mark.slow
def test_criterion_5_a_failed_result_call_marks_nothing_and_the_note_goes(
    plugin, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-a", state="running")
    run_attempt(setup, job_id, "failed")

    answer, is_error = read_result_in_turn(plugin, setup, job_id, local_turn("chat-a"))

    assert is_error is True and answer["error"]["code"] == "no_result"
    assert "seen_at" not in job_file(setup, job_id)["attempts"][0]
    chats = Chats(setup)
    assert ticks(plugin.delivery, chats, 2) == ["delivered", "idle"]
    assert lines(chats) == [("chat-a", f"sound-scout-job:{job_id}")]


# --- the hook alone, with a recording CLI ----------------------------------------------------


JOB_ID = "a" * 32


def result_of(**fields) -> str:
    answer = {"job_id": JOB_ID, "job_state": "finished", "attempt": 2, "status": "partial"}
    return json.dumps({"result": json.dumps(answer | fields)})


def outcome(plugin, *, tool_name=RESULT_TOOL, result=None, status="ok", env=None, answer=None):
    """The outcome of `mark_after_result` and the argv lists the CLI got."""
    calls = []
    env = local_turn("chat-a") if env is None else env

    def cli(argv):
        calls.append(argv)
        return {"job_id": JOB_ID} if answer is None else answer

    said = plugin.seen.mark_after_result(
        tool_name,
        result_of() if result is None else result,
        status,
        session_env=lambda name: env.get(name, ""),
        cli=cli,
    )
    return said, calls


@pytest.mark.parametrize("tool_name", [RESULT_TOOL, "mcp__x__speech_job_result"])
@pytest.mark.parametrize("state", ["finished", "failed", "cancelled"])
def test_a_result_of_an_ended_job_of_any_server_name_is_marked(
    plugin, tool_name: str, state: str
) -> None:
    said, calls = outcome(plugin, tool_name=tool_name, result=result_of(job_state=state))

    assert said == "seen"
    assert calls == [["seen", "--session-id=chat-a", "--attempt=2", JOB_ID]]


@pytest.mark.parametrize(
    "tool_name",
    [
        "mcp__speech_review__speech_job_status",
        "mcp__speech_review__speech_job_submit",
        "speech_job_result",
        None,
    ],
)
def test_another_tool_is_left_alone(plugin, tool_name) -> None:
    assert outcome(plugin, tool_name=tool_name) == (None, [])


@pytest.mark.parametrize("status", ["error", "blocked", ""])
def test_a_result_call_that_did_not_succeed_is_left_alone(plugin, status: str) -> None:
    assert outcome(plugin, status=status) == (None, [])


@pytest.mark.parametrize(
    "env",
    [
        {"HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "chat-a"},
        {"HERMES_SESSION_CHAT_ID": "chat-a"},
        {"HERMES_SESSION_PLATFORM": "local", "HERMES_SESSION_CHAT_ID": "c" * 129},
        {"HERMES_SESSION_PLATFORM": "local", "HERMES_SESSION_CHAT_ID": "chat\na"},
    ],
)
def test_no_valid_local_chat_calls_no_cli(plugin, env: dict) -> None:
    assert outcome(plugin, env=env) == ("no_chat", [])


@pytest.mark.parametrize(
    "result",
    [
        result_of(job_state="queued"),
        result_of(job_state="running"),
        result_of(job_state=None),
        result_of(job_id="A" * 32),
        result_of(job_id="../" + "a" * 29),
        result_of(attempt=0),
        result_of(attempt=True),
        result_of(attempt="2"),
        result_of(attempt=None),
        "not json",
        json.dumps({"result": "not json"}),
        json.dumps({"result": {"job_id": JOB_ID}}),
        json.dumps({"result": json.dumps(["job_id"])}),
        None,
    ],
)
def test_a_result_of_a_job_not_ended_or_malformed_calls_no_cli(plugin, result) -> None:
    calls = []
    said = plugin.seen.mark_after_result(
        RESULT_TOOL, result, "ok", session_env=local_turn("chat-a").get, cli=calls.append
    )
    assert (said, calls) == ("not_terminal", [])


def test_a_refusal_and_a_silent_cli_are_outcomes(plugin) -> None:
    assert outcome(plugin, answer={"error": "not_seen"})[0] == "not_seen"
    calls = []
    said = plugin.seen.mark_after_result(
        RESULT_TOOL,
        result_of(),
        "ok",
        session_env=local_turn("chat-a").get,
        cli=lambda argv: calls.append(argv),
    )
    assert said == "cli_failed" and len(calls) == 1


def test_the_models_arguments_never_reach_the_mark(plugin, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(plugin.delivery, "run_cli", lambda argv: calls.append(argv) or {})
    model_args = {"job_id": "b" * 32, "session_id": "chat-b", "attempt": 9}

    for turn in (local_turn("chat-a"), {}):
        hermes_call(
            plugin.seen.post_tool_call,
            turn=turn,
            tool_name=RESULT_TOOL,
            args=model_args,
            result=result_of(),
            status="ok",
        )

    assert calls == [["seen", "--session-id=chat-a", "--attempt=2", JOB_ID]]
    assert "chat-b" not in json.dumps(calls) and "b" * 32 not in json.dumps(calls)


def test_an_exception_never_leaves_the_hook_and_logs_no_chat(
    plugin, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("chat-a secret")

    monkeypatch.setattr(plugin.delivery, "run_cli", boom)
    caplog.set_level(logging.INFO)

    hermes_call(
        plugin.seen.post_tool_call,
        turn=local_turn("chat-a"),
        tool_name=RESULT_TOOL,
        args={},
        result=result_of(),
        status="ok",
    )

    assert "sound-scout seen: RuntimeError" in caplog.text
    assert "chat-a" not in caplog.text and JOB_ID not in caplog.text


def test_the_hook_imports_only_the_stdlib_and_its_package() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 11))
    top_level, relative = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level:
            relative |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module.split(".")[0])
    assert top_level <= set(sys.stdlib_module_names)
    assert relative == {"binding", "delivery"}
    assert "sound_scout" not in source
    assert "args" not in {
        node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript | ast.Attribute) and isinstance(node.value, ast.Name)
    }


# --- criterion 6: the decision record --------------------------------------------------------


def test_criterion_6_the_decision_record_names_the_rule_the_signal_and_the_limit() -> None:
    record = Path(__file__).resolve().parent.parent / "docs/decisions/0030-seen-attempt-no-note.md"
    text = " ".join(record.read_text(encoding="utf-8").split())

    assert "seen_in_chat" in text and "`sound-scout jobs seen --session-id=<чат>" in text
    assert "`TURN_WAIT_SECONDS` = 600" in text
    assert "admissions.list" in text and "logical_owner" in text
    assert "## Остаточный риск" in record.read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    return " ".join(text.split(start, 1)[1].split(end, 1)[0].split())


def test_criterion_6_decisions_0013_0018_and_a01_a14_carry_the_rule_and_the_owner_step() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    delivery_section = _between(
        (docs / "decisions/0013-voice-note-integration.md").read_text(encoding="utf-8"),
        "### 3.7. Доставка в чат",
        "### 3.8.",
    )
    retry_section = _between(
        (docs / "decisions/0018-job-modes-and-retry.md").read_text(encoding="utf-8"),
        "### Повтор и доставка",
        "\n## ",
    )
    acceptance = (docs / "acceptance/A01-A14.md").read_text(encoding="utf-8")
    a09 = next(line for line in acceptance.splitlines() if line.startswith("| A09 |"))
    o10 = next(line for line in acceptance.splitlines() if line.startswith("- **O10.**"))

    for section in (delivery_section, retry_section):
        assert "#190" in section and "0030" in section
        assert "`speech_job_result`" in section and "`seen_in_chat`" in section
        assert "600 с" in section
    assert "`sound-scout jobs seen`" in delivery_section
    assert "`-a<n>`" in retry_section
    assert "O10" in a09 and "0030" in a09
    assert "0.4.0" in o10 and "0.5.6" in o10 and "`sound-scout-job:<job_id>-a2`" in o10
    assert "Заметки Sound Scout об этом задании в X быть не должно" in o10
