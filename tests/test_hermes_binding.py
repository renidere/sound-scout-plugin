"""The `post_tool_call` hook of the Hermes plugin (`hermes/plugin/sound-scout/binding.py`, decision
0025, #140), without Hermes.

A contract test: `hermes_call` stands in for Hermes v0.23.3 (`caafe5389a`, read only). It wraps the
MCP answer as `tools/mcp_tool_handlers.py` does, derives `status` as
`model_tools._tool_result_observer_fields` does, and calls the hook with the keywords of
`model_tools._emit_post_tool_call_hook` in `contextvars.copy_context()` on a worker thread, as
`hermes_cli/plugins_dispatch.py` does. The session context is a stand-in
`gateway.session_context` over `ContextVar`s. The MCP server, `sound-scout jobs bind`,
`deliver-next` and `deliver-end` are the real ones over a temporary data root; the runner is
stood in for by setting the job state, and `send`/`history_has` are fakes. No live Hermes and no
real chat are touched.
"""

import ast
import contextvars
import importlib
import importlib.util
import json
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

from sound_scout import config, jobs, mcp_server
from sound_scout.config import Config
from tests.conftest import WavFactory
from tests.test_delivery import set_job
from tests.test_hermes_delivery import Chats, ticks

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "hermes" / "plugin" / "sound-scout"
MODULE = PACKAGE / "binding.py"
SUBMIT = "mcp__speech_review__speech_job_submit"
SESSION_VARS = ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_ID")
TURN = {name: contextvars.ContextVar(name, default=None) for name in SESSION_VARS}


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch):
    """The plugin package as Hermes imports it, with stand-ins for the Hermes modules it reads."""
    spec = importlib.util.spec_from_file_location(
        "sound_scout_plugin_binding",
        PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)],
    )
    package = importlib.util.module_from_spec(spec)
    for name in ("binding", "delivery"):
        monkeypatch.delitem(sys.modules, f"{spec.name}.{name}", raising=False)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)

    def get_session_env(name: str, default: str = "") -> str:
        value = TURN[name].get() if name in TURN else None
        return default if value is None else value

    gateway = types.ModuleType("gateway")
    session_context = types.ModuleType("gateway.session_context")
    session_context.get_session_env = get_session_env
    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: Path("hermes-home")
    hermes_constants.profile_name_for_home = lambda home: None
    monkeypatch.setitem(sys.modules, "gateway", gateway)
    monkeypatch.setitem(sys.modules, "gateway.session_context", session_context)
    monkeypatch.setitem(sys.modules, "hermes_constants", hermes_constants)
    return package


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A data root, the real `sound-scout` of this venv and its configuration for `run_cli`."""
    for name in ("recordings", "data"):
        (tmp_path / name).mkdir()
    path = tmp_path / "sound-scout.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'recordings'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOUND_SCOUT_BIN", str(Path(sys.executable).parent / "sound-scout"))
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    return config.load()


def hermes_result(answer: dict, is_error: bool) -> tuple[str, str]:
    """What Hermes hands the hook for one MCP answer: the result string and its status."""
    text = mcp_server.result(answer, is_error).content[0].text
    if is_error:
        return json.dumps({"error": text}), "error"
    return json.dumps({"result": text}, ensure_ascii=False), "ok"


def hermes_call(hook, *, turn: dict, tool_name: str, args: dict, result: str, status: str) -> None:
    """One `post_tool_call` as Hermes makes it, inside the turn's session context."""

    def in_turn() -> None:
        for name, value in turn.items():
            TURN[name].set(value)
        context = contextvars.copy_context()
        worker = threading.Thread(
            target=context.run,
            args=(hook,),
            kwargs={
                "tool_name": tool_name,
                "args": args,
                "result": result,
                "task_id": "",
                "session_id": turn.get("HERMES_SESSION_ID", ""),
                "tool_call_id": "call-1",
                "turn_id": "",
                "api_request_id": "",
                "conversation_id": "",
                "duration_ms": 5,
                "status": status,
                "error_type": None if status == "ok" else "tool_error",
                "error_message": None,
                "middleware_trace": [],
                "telemetry_schema_version": "hermes.observer.v1",
            },
        )
        worker.start()
        worker.join(30)

    contextvars.Context().run(in_turn)


def local_turn(chat_id: str) -> dict:
    return {
        "HERMES_SESSION_PLATFORM": "local",
        "HERMES_SESSION_CHAT_ID": chat_id,
        "HERMES_SESSION_ID": f"segment-of-{chat_id}",
    }


def submit_in_turn(plugin, cfg: Config, wav: Path, turn: dict, request_id: str) -> str:
    """The agent's `speech_job_submit` in a turn: the real MCP call, then the hook."""
    args = {"path": str(wav), "request_id": request_id}
    answer, is_error = mcp_server.call("speech_job_submit", args, cfg)
    result, status = hermes_result(answer, is_error)
    hermes_call(
        plugin.binding.post_tool_call,
        turn=turn,
        tool_name=SUBMIT,
        args=args,
        result=result,
        status=status,
    )
    return answer["job_id"]


def finish(cfg: Config, job_id: str) -> None:
    set_job(cfg, job_id, state="finished")


def sent_to(chats: Chats) -> list[tuple[str, str]]:
    return [(chat, text.splitlines()[-1]) for chat, text in chats.sent]


# --- criterion 2: one note to the chat of the turn, none to another ---------------------------


def test_criterion_2_an_mcp_job_from_chat_a_delivers_one_note_to_a_and_none_to_b(
    plugin, setup: Config, make_wav: WavFactory, caplog: pytest.LogCaptureFixture
) -> None:
    wav = make_wav("recordings/a.wav")
    caplog.set_level(logging.INFO)

    job_a = submit_in_turn(plugin, setup, wav, local_turn("chat-a"), "r-a")
    finish(setup, job_a)
    chats = Chats(setup)
    assert ticks(plugin.delivery, chats) == ["delivered", "idle", "idle"]

    assert sent_to(chats) == [("chat-a", f"sound-scout-job:{job_a}")]
    status = jobs.status(job_a, setup)
    assert status["binding"] == {
        "submitter": "mcp",
        "connection": None,
        "profile": "default",
        "session_id": "chat-a",
    }
    assert status["delivery"]["state"] == "delivered"
    assert "sound-scout binding: bound" in caplog.text
    assert "chat-a" not in caplog.text and str(wav) not in caplog.text

    job_b = submit_in_turn(plugin, setup, wav, local_turn("chat-b"), "r-b")
    finish(setup, job_b)
    ticks(plugin.delivery, chats)

    assert sent_to(chats) == [
        ("chat-a", f"sound-scout-job:{job_a}"),
        ("chat-b", f"sound-scout-job:{job_b}"),
    ]


def test_a_turn_outside_a_local_chat_leaves_the_job_unbound_and_sends_nothing(
    plugin, setup: Config, make_wav: WavFactory
) -> None:
    turn = {"HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "chat-a"}
    job_id = submit_in_turn(plugin, setup, make_wav("recordings/a.wav"), turn, "r-1")
    finish(setup, job_id)
    chats = Chats(setup)

    assert ticks(plugin.delivery, chats) == ["idle", "idle", "idle"]
    assert chats.sent == [] and jobs.status(job_id, setup)["binding"]["session_id"] is None


def test_a_job_that_ends_before_its_bind_waits_and_then_gets_one_note(
    plugin, setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    args = {"path": str(wav), "request_id": "r-1"}
    answer, is_error = mcp_server.call("speech_job_submit", args, setup)
    finish(setup, answer["job_id"])
    chats = Chats(setup)

    assert ticks(plugin.delivery, chats, 2) == ["idle", "idle"]
    assert chats.sent == []

    result, status = hermes_result(answer, is_error)
    hermes_call(
        plugin.binding.post_tool_call,
        turn=local_turn("chat-a"),
        tool_name=SUBMIT,
        args=args,
        result=result,
        status=status,
    )
    assert ticks(plugin.delivery, chats) == ["delivered", "idle", "idle"]
    assert sent_to(chats) == [("chat-a", f"sound-scout-job:{answer['job_id']}")]


def test_without_the_hook_an_mcp_job_is_never_delivered(
    plugin, setup: Config, make_wav: WavFactory
) -> None:
    answer, _ = mcp_server.call(
        "speech_job_submit", {"path": str(make_wav("recordings/a.wav")), "request_id": "r-1"}, setup
    )
    finish(setup, answer["job_id"])
    chats = Chats(setup)

    assert ticks(plugin.delivery, chats, 4) == ["idle"] * 4
    assert chats.sent == []
    assert jobs.status(answer["job_id"], setup)["binding"]["session_id"] is None


def test_a_second_hook_call_from_chat_b_is_not_bindable_and_the_note_goes_to_a_only(
    plugin, setup: Config, make_wav: WavFactory, caplog: pytest.LogCaptureFixture
) -> None:
    wav = make_wav("recordings/a.wav")
    args = {"path": str(wav), "request_id": "r-1"}
    answer, is_error = mcp_server.call("speech_job_submit", args, setup)
    result, status = hermes_result(answer, is_error)
    caplog.set_level(logging.INFO)
    for chat in ("chat-a", "chat-b"):
        hermes_call(
            plugin.binding.post_tool_call,
            turn=local_turn(chat),
            tool_name=SUBMIT,
            args=args,
            result=result,
            status=status,
        )
    finish(setup, answer["job_id"])
    chats = Chats(setup)
    ticks(plugin.delivery, chats)

    assert "sound-scout binding: not_bindable" in caplog.text
    assert sent_to(chats) == [("chat-a", f"sound-scout-job:{answer['job_id']}")]


def test_a_refused_submit_reaches_the_hook_as_an_error_and_binds_nothing(
    plugin, setup: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = {"path": str(make_wav("recordings/a.wav")), "request_id": "r-1", "session_id": "chat-b"}
    answer, is_error = mcp_server.call("speech_job_submit", args, setup)
    result, status = hermes_result(answer, is_error)
    calls = []
    monkeypatch.setattr(plugin.delivery, "run_cli", lambda argv: calls.append(argv))

    hermes_call(
        plugin.binding.post_tool_call,
        turn=local_turn("chat-a"),
        tool_name=SUBMIT,
        args=args,
        result=result,
        status=status,
    )

    assert is_error is True and status == "error" and calls == []
    assert not (setup.data_root / "jobs").exists()


# --- the hook alone, with a recording CLI ----------------------------------------------------


JOB_ID = "a" * 32
RESULT = json.dumps({"result": json.dumps({"job_id": JOB_ID, "state": "queued"})})


def outcome(plugin, *, tool_name=SUBMIT, result=RESULT, status="ok", env=None, answer=None):
    """The outcome of `bind_after_submit` and the argv lists the CLI got."""
    calls = []
    env = local_turn("chat-a") if env is None else env

    def cli(argv):
        calls.append(argv)
        return {"job_id": JOB_ID} if answer is None else answer

    said = plugin.binding.bind_after_submit(
        tool_name,
        result,
        status,
        session_env=lambda name: env.get(name, ""),
        profile=lambda: "default",
        cli=cli,
    )
    return said, calls


@pytest.mark.parametrize("tool_name", [SUBMIT, "mcp__x__speech_job_submit"])
def test_a_submit_of_any_server_name_is_bound(plugin, tool_name: str) -> None:
    said, calls = outcome(plugin, tool_name=tool_name)

    assert said == "bound"
    assert calls == [["bind", "--session-id=chat-a", "--profile=default", JOB_ID]]


@pytest.mark.parametrize(
    "tool_name",
    ["mcp__speech_review__speech_job_status", "speech_job_submit", "terminal", None],
)
def test_another_tool_is_left_alone(plugin, tool_name) -> None:
    assert outcome(plugin, tool_name=tool_name) == (None, [])


@pytest.mark.parametrize("status", ["error", "blocked", ""])
def test_a_submit_that_did_not_succeed_is_left_alone(plugin, status: str) -> None:
    assert outcome(plugin, status=status) == (None, [])


@pytest.mark.parametrize(
    "env",
    [
        {"HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "chat-a"},
        {"HERMES_SESSION_CHAT_ID": "chat-a"},
        {"HERMES_SESSION_PLATFORM": "local", "HERMES_SESSION_CHAT_ID": ""},
        {"HERMES_SESSION_PLATFORM": "local", "HERMES_SESSION_CHAT_ID": "c" * 129},
        {"HERMES_SESSION_PLATFORM": "local", "HERMES_SESSION_CHAT_ID": "chat\na"},
        {"HERMES_SESSION_PLATFORM": "local", "HERMES_SESSION_CHAT_ID": "chat\x00a"},
    ],
)
def test_no_valid_local_chat_calls_no_cli(plugin, env: dict) -> None:
    assert outcome(plugin, env=env) == ("no_chat", [])


def test_a_chat_id_of_128_characters_is_bound(plugin) -> None:
    env = {"HERMES_SESSION_PLATFORM": "local", "HERMES_SESSION_CHAT_ID": "c" * 128}
    said, calls = outcome(plugin, env=env)
    assert said == "bound" and calls[0][1] == "--session-id=" + "c" * 128


@pytest.mark.parametrize(
    "result",
    [
        "not json",
        json.dumps({"job_id": JOB_ID}),
        json.dumps({"result": {"job_id": JOB_ID}}),
        json.dumps({"result": "not json"}),
        json.dumps({"result": json.dumps({"job_id": "A" * 32})}),
        json.dumps({"result": json.dumps({"job_id": "../" + "a" * 29})}),
        json.dumps({"result": json.dumps(["job_id"])}),
        None,
    ],
)
def test_a_result_without_a_job_id_calls_no_cli(plugin, result) -> None:
    assert outcome(plugin, result=result) == ("no_job_id", [])


def test_a_refusal_and_a_silent_cli_are_outcomes(plugin) -> None:
    assert outcome(plugin, answer={"error": "not_bindable"})[0] == "not_bindable"

    calls = []
    said = plugin.binding.bind_after_submit(
        SUBMIT,
        RESULT,
        "ok",
        session_env=local_turn("chat-a").get,
        profile=lambda: "default",
        cli=lambda argv: calls.append(argv),
    )
    assert said == "cli_failed" and len(calls) == 1


def test_the_models_arguments_never_reach_the_binding(plugin, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(plugin.delivery, "run_cli", lambda argv: calls.append(argv) or {})
    model_args = {"path": "a.wav", "request_id": "r-1", "session_id": "chat-b", "profile": "other"}

    hermes_call(
        plugin.binding.post_tool_call,
        turn=local_turn("chat-a"),
        tool_name=SUBMIT,
        args=model_args,
        result=RESULT,
        status="ok",
    )
    hermes_call(
        plugin.binding.post_tool_call,
        turn={},
        tool_name=SUBMIT,
        args=model_args,
        result=RESULT,
        status="ok",
    )

    assert calls == [["bind", "--session-id=chat-a", "--profile=default", JOB_ID]]
    assert "chat-b" not in json.dumps(calls) and "other" not in json.dumps(calls)


def test_the_hook_reads_the_chat_id_not_its_session_id_argument(plugin, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(plugin.delivery, "run_cli", lambda argv: calls.append(argv) or {})

    plugin.binding.post_tool_call(
        tool_name=SUBMIT, args={}, result=RESULT, status="ok", session_id="segment-1"
    )

    assert calls == []  # no session context: the hook's session_id argument binds nothing


@pytest.mark.parametrize("broken", ["session_env", "run_cli", "profile"])
def test_an_exception_never_leaves_the_hook_and_logs_no_chat(
    plugin, monkeypatch, caplog: pytest.LogCaptureFixture, broken: str
) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("chat-a secret")

    if broken == "session_env":
        monkeypatch.setattr(plugin.binding, "hermes_session_env", boom)
    elif broken == "run_cli":
        monkeypatch.setattr(plugin.delivery, "run_cli", boom)
    else:
        monkeypatch.setattr(plugin.delivery, "hermes_profile", boom)
    caplog.set_level(logging.INFO)

    hermes_call(
        plugin.binding.post_tool_call,
        turn=local_turn("chat-a"),
        tool_name=SUBMIT,
        args={"session_id": "chat-b"},
        result=RESULT,
        status="ok",
    )

    assert "sound-scout binding: RuntimeError" in caplog.text
    assert "chat-a" not in caplog.text and "chat-b" not in caplog.text and JOB_ID not in caplog.text


def test_register_adds_one_post_tool_call_hook_and_starts_the_delivery_loop(
    plugin, monkeypatch
) -> None:
    started = []
    monkeypatch.setattr(plugin.delivery, "start", lambda: started.append(True))

    class Ctx:
        def __init__(self) -> None:
            self.hooks = []

        def register_hook(self, name, callback, *, hold_stream=False):
            self.hooks.append((name, callback))

    ctx = Ctx()
    assert plugin.register(ctx) is None
    assert ctx.hooks == [("post_tool_call", plugin.binding.post_tool_call)]
    assert started == [True]


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
    assert relative == {"delivery"}
    assert "sound_scout" not in source
    assert "args" not in {
        node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript | ast.Attribute) and isinstance(node.value, ast.Name)
    }


def test_the_hermes_edge_reads_the_session_context(plugin) -> None:
    def in_turn() -> str:
        TURN["HERMES_SESSION_CHAT_ID"].set("chat-a")
        return plugin.binding.hermes_session_env("HERMES_SESSION_CHAT_ID")

    assert contextvars.Context().run(in_turn) == "chat-a"
    assert (
        importlib.import_module("gateway.session_context") is sys.modules["gateway.session_context"]
    )


# --- criterion 1: the decision record ---------------------------------------------------------


def test_criterion_1_the_decision_record_gives_mcp_jobs_a_binding_from_the_turn_not_the_model() -> (
    None
):
    record = REPO / "docs" / "decisions" / "0025-mcp-chat-binding.md"
    text = " ".join(record.read_text(encoding="utf-8").split())

    assert "MCP-задания получают привязку к чату" in text
    assert "Её источник — контекст сессии хода Hermes, прочитанный плагином `sound-scout`" in text
    assert "Аргументы инструмента и текст модели привязку не дают никогда" in text
    assert '`HERMES_SESSION_CHAT_ID` при `HERMES_SESSION_PLATFORM == "local"`' in text
    assert "## Источник: contextvar хода, `chat_id`, а не `session_id`" in record.read_text(
        encoding="utf-8"
    )
