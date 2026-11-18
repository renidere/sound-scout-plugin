"""`sound-scout serve`: the MCP stdio adapter over `app`, `doctor` and `report` (R12, R08, T09a)."""

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import anyio
import pytest
from mcp import Client

from sound_scout import app, config, mcp_server, report
from tests.conftest import WavFactory
from tests.test_cli import ABSENT_ASR_MODEL, needs_ffmpeg, offline_env, write_config

REPO = Path(__file__).resolve().parent.parent
TOOL_NAMES = ["speech_capabilities", "speech_analyze", "speech_compare"]
BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{120,}")
RESPONSE_SECONDS = 120
OPERATOR_ARGUMENTS = [
    ("allowed_roots", ["/tmp"]),
    ("allowed_root", "/tmp"),
    ("data_root", "/tmp"),
    ("model", "owner/name"),
    ("asr", {"model": "owner/name"}),
    ("revision", "abcdef0"),
    ("backend", "none"),
    ("budgets", {"worker_deadline_seconds": 1}),
    ("worker_deadline_seconds", 1),
    ("network", {"allow": True}),
    ("config", "/tmp/sound-scout.toml"),
    ("approved", True),
]
VALID_ARGUMENTS = {
    "speech_capabilities": {},
    "speech_analyze": {"path": "recordings/tone.wav", "request_id": "r-1"},
    "speech_compare": {"run_a": "a" * 32, "run_b": "b" * 32, "criterion": "completeness_of_phrase"},
}


@pytest.fixture
def loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> config.Config:
    """`network.allow = true` (no audit hook in this session) and an ASR model not installed."""
    path = write_config(tmp_path, allow_network=True, asr_model=ABSENT_ASR_MODEL)
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    return config.load()


class StdioSession:
    """`sound-scout serve` in a subprocess, spoken to line by line; keeps every stdout line."""

    def __init__(self, env: dict[str, str]) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "sound_scout", "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self.lines: list[str] = []
        self._incoming: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self._incoming.put(line)
        self._incoming.put(None)

    def _send(self, message: dict) -> None:
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        self._send(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        )
        while True:
            line = self._incoming.get(timeout=RESPONSE_SECONDS)
            assert line is not None, "the server closed stdout before answering"
            self.lines.append(line)
            message = json.loads(line)
            if message.get("id") == self._next_id:
                return message

    def initialize(self) -> dict:
        answer = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        )
        self.notify("notifications/initialized")
        return answer

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})["result"]

    def close(self) -> tuple[int, str]:
        self.process.stdin.close()
        code = self.process.wait(timeout=RESPONSE_SECONDS)
        while (line := self._incoming.get(timeout=RESPONSE_SECONDS)) is not None:
            self.lines.append(line)
        return code, self.process.stderr.read()


def answer_of(tool_result: dict) -> dict:
    [block] = tool_result["content"]
    assert block["type"] == "text"
    return json.loads(block["text"])


def text_of(tool_result: dict) -> str:
    return tool_result["content"][0]["text"]


def without_run_identity(result: dict) -> dict:
    """The result with what differs between two runs of one input removed: IDs and times."""
    document = json.loads(json.dumps(result))
    for key in ("run_id", "request_id", "run_dir"):
        document.pop(key, None)
    for stage in document.get("stages", {}).values():
        stage.pop("elapsed_ms", None)
    return document


# --- tools: names, schemas, annotations ------------------------------------------------------


def test_the_tools_are_the_three_of_r12_with_closed_schemas() -> None:
    assert [tool.name for tool in mcp_server.TOOLS] == TOOL_NAMES
    for tool in mcp_server.TOOLS:
        assert tool.description
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema["additionalProperties"] is False
    analyze = mcp_server.TOOLS[1].input_schema
    assert list(analyze["properties"]) == ["path", "request_id", "layers", "task"]
    assert analyze["required"] == ["path", "request_id"]
    assert analyze["properties"]["layers"]["items"]["enum"] == list(app.LAYERS)
    compare = mcp_server.TOOLS[2].input_schema
    assert compare["required"] == ["run_a", "run_b", "criterion"]


def test_annotations_follow_the_side_effects() -> None:
    hints = {tool.name: tool.annotations for tool in mcp_server.TOOLS}
    assert hints["speech_capabilities"].read_only_hint is True
    assert hints["speech_analyze"].read_only_hint is False
    assert hints["speech_compare"].read_only_hint is False
    assert all(hint.open_world_hint is False for hint in hints.values())
    assert all(hint.destructive_hint is False for hint in hints.values())


def test_docs_list_every_tool_exactly_as_the_code_declares_it() -> None:
    text = (REPO / "docs" / "mcp.md").read_text(encoding="utf-8")
    blocks = dict(re.findall(r"### `(\w+)`\n\n```json\n(.*?)\n```", text, flags=re.S))
    assert list(blocks) == TOOL_NAMES
    for tool in mcp_server.TOOLS:
        declared = tool.model_dump(by_alias=True, exclude_none=True, mode="json")
        assert json.loads(blocks[tool.name]) == declared


# --- call: the argument validator and the refusals -------------------------------------------


@pytest.mark.parametrize(("argument", "value"), OPERATOR_ARGUMENTS)
@pytest.mark.parametrize("tool", TOOL_NAMES)
def test_no_argument_overrides_the_operator_configuration(
    tool: str, argument: str, value: object, loaded: config.Config
) -> None:
    answer, is_error = mcp_server.call(tool, {**VALID_ARGUMENTS[tool], argument: value}, loaded)

    assert is_error is True
    assert answer["error"]["code"] == "invalid_arguments"
    assert answer["error"]["message"].startswith("Additional properties are not allowed, allowed: ")
    assert repr(value) not in answer["error"]["message"]
    assert list(loaded.data_root.iterdir()) == []


@pytest.mark.parametrize(
    ("arguments", "where"),
    [
        ({"request_id": "r-1"}, "'path' is a required property"),
        ({"path": "a.wav"}, "'request_id' is a required property"),
        ({"path": 7, "request_id": "r-1"}, "path: must be of type string"),
        ({"path": "a.wav", "request_id": ""}, "request_id: "),
        ({"path": "a.wav", "request_id": "r" * 257}, "request_id: "),
        ({"path": "a.wav", "request_id": "r-1", "layers": []}, "layers: "),
        ({"path": "a.wav", "request_id": "r-1", "layers": ["voice"]}, "layers.0: "),
        ({"path": "a.wav", "request_id": "r-1", "layers": ["sound", "sound"]}, "layers: "),
        ({"path": "a.wav", "request_id": "r-1", "layers": "sound"}, "layers: "),
        ({"path": "a.wav", "request_id": "r-1", "task": ["x"]}, "task: "),
    ],
)
def test_malformed_analyze_arguments_are_refused_before_anything_runs(
    arguments: dict, where: str, loaded: config.Config
) -> None:
    answer, is_error = mcp_server.call("speech_analyze", arguments, loaded)

    assert is_error is True
    assert answer["error"]["code"] == "invalid_arguments"
    assert where in answer["error"]["message"]
    assert list(loaded.data_root.iterdir()) == []


def test_an_argument_error_does_not_quote_the_value(loaded: config.Config) -> None:
    secret = "текст транскрипта " * 20
    answer, is_error = mcp_server.call(
        "speech_analyze", {"path": "a.wav", "request_id": "r-1", "task": [secret]}, loaded
    )

    assert is_error is True
    assert answer["error"] == {
        "code": "invalid_arguments",
        "message": "task: must be of type string",
    }


def test_an_unexpected_error_logs_its_type_only(
    loaded: config.Config, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(loaded: object) -> dict:
        raise RuntimeError("транскрипт: секретная фраза")

    monkeypatch.setattr(mcp_server.doctor, "report", broken)
    with caplog.at_level(logging.INFO, logger="sound_scout.mcp"):
        answer, is_error = mcp_server.call("speech_capabilities", {}, loaded)

    assert is_error is True
    assert answer["error"] == {"code": "internal_error", "message": "RuntimeError"}
    assert [record.exc_info for record in caplog.records] == [None]
    assert "speech_capabilities failed: RuntimeError in " in caplog.text
    assert "секретная фраза" not in caplog.text


def test_a_refused_path_is_logged_without_the_path(
    loaded: config.Config, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    outside = tmp_path / "elsewhere" / "voice.wav"
    with caplog.at_level(logging.INFO, logger="sound_scout.mcp"):
        answer, is_error = mcp_server.call(
            "speech_analyze", {"path": str(outside), "request_id": "r-1"}, loaded
        )

    assert is_error is True
    assert answer["error"]["code"] == "path_denied"
    assert str(outside) in answer["error"]["message"]
    assert "speech_analyze refused: path_denied: outside allowed roots in " in caplog.text
    assert str(outside) not in caplog.text and "elsewhere" not in caplog.text


def test_compare_without_a_criterion_is_refused(loaded: config.Config) -> None:
    answer, is_error = mcp_server.call("speech_compare", {"run_a": "a", "run_b": "b"}, loaded)
    assert is_error is True
    assert answer["error"] == {
        "code": "invalid_arguments",
        "message": "'criterion' is a required property",
    }


def test_an_unknown_tool_is_refused(loaded: config.Config) -> None:
    answer, is_error = mcp_server.call("speech_transcribe", {}, loaded)
    assert is_error is True
    assert answer["error"]["code"] == "unknown_tool"


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\someone\voice.wav",
        "C:/Users/someone/voice.wav",
        "https://example.com/voice.wav",
        "file:///tmp/voice.wav",
        "concat:a.wav|b.wav",
    ],
)
def test_a_windows_path_or_a_url_is_refused_with_a_clear_message(
    path: str, loaded: config.Config
) -> None:
    answer, is_error = mcp_server.call(
        "speech_analyze", {"path": path, "request_id": "r-1"}, loaded
    )

    assert is_error is True
    assert answer["error"]["code"] == "path_denied"
    assert answer["error"]["message"].startswith("not a file path: ")
    assert "Windows path, a URL" in answer["error"]["hint"]
    assert not (loaded.data_root / "runs").exists()


def test_a_path_outside_the_allowed_roots_is_refused(
    loaded: config.Config, make_wav: WavFactory
) -> None:
    outside = make_wav("elsewhere.wav")
    answer, is_error = mcp_server.call(
        "speech_analyze", {"path": str(outside), "request_id": "r-1"}, loaded
    )
    assert is_error is True
    assert answer["error"]["code"] == "path_denied"
    assert answer["error"]["message"].startswith("outside allowed roots: ")


@needs_ffmpeg
def test_compare_answers_with_the_written_comparison_and_its_path(
    loaded: config.Config, make_wav: WavFactory
) -> None:
    run_ids = []
    for name, frequency in (("a.wav", 220.0), ("b.wav", 330.0)):
        wav = make_wav(f"recordings/{name}", frequency=frequency)
        analyzed, is_error = mcp_server.call(
            "speech_analyze", {"path": str(wav), "request_id": name}, loaded
        )
        assert is_error is False
        run_ids.append(analyzed["run_id"])

    answer, is_error = mcp_server.call(
        "speech_compare",
        {"run_a": run_ids[0], "run_b": run_ids[1], "criterion": "completeness_of_phrase"},
        loaded,
    )

    assert is_error is False
    path = answer.pop("path")
    assert path.startswith("compare/")
    assert answer == json.loads((loaded.data_root / path).read_text(encoding="utf-8"))
    assert [answer["attempts"][label]["run_id"] for label in ("a", "b")] == run_ids


def test_an_unknown_run_is_a_compare_refusal(loaded: config.Config) -> None:
    answer, is_error = mcp_server.call(
        "speech_compare",
        {"run_a": "0" * 32, "run_b": "1" * 32, "criterion": "completeness_of_phrase"},
        loaded,
    )
    assert is_error is True
    assert answer["error"]["code"] == "run_not_found"
    assert not (loaded.data_root / "compare").exists()


def test_an_unexpected_error_is_internal_error_with_its_type_only(
    loaded: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("private text from /home/someone/recording.wav")

    monkeypatch.setattr(app, "analyze", broken)

    answer, is_error = mcp_server.call(
        "speech_analyze", {"path": "a.wav", "request_id": "r-1"}, loaded
    )
    assert is_error is True
    assert answer == {"error": {"code": "internal_error", "message": "RuntimeError"}}


# --- the server in-process: refusals are results, calls are serialized ------------------------


def test_the_server_has_no_telemetry_middleware(loaded: config.Config) -> None:
    assert mcp_server.build_server(loaded).middleware == []


def test_a_refusal_is_a_tool_result_not_a_protocol_error(loaded: config.Config) -> None:
    async def scenario() -> object:
        async with Client(mcp_server.build_server(loaded)) as client:
            listed = await client.list_tools()
            refused = await client.call_tool(
                "speech_analyze", {"path": r"C:\Users\someone\voice.wav", "request_id": "r-1"}
            )
            extra = await client.call_tool("speech_capabilities", {"data_root": "/tmp"})
            return listed, refused, extra

    listed, refused, extra = anyio.run(scenario)

    assert [tool.name for tool in listed.tools] == TOOL_NAMES
    assert refused.is_error is True
    assert json.loads(refused.content[0].text)["error"]["code"] == "path_denied"
    assert extra.is_error is True
    assert json.loads(extra.content[0].text)["error"]["code"] == "invalid_arguments"


def test_tool_calls_run_one_at_a_time(
    loaded: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    running, overlaps, lock = [0], [], threading.Lock()

    def slow(name: str, arguments: dict | None, loaded: object, cancel: object) -> tuple:
        with lock:
            running[0] += 1
            overlaps.append(running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return {"status": "ok"}, False

    monkeypatch.setattr(mcp_server, "call", slow)

    async def scenario() -> None:
        async with (
            Client(mcp_server.build_server(loaded)) as client,
            anyio.create_task_group() as group,
        ):
            for _ in range(3):
                group.start_soon(client.call_tool, "speech_capabilities", {})

    anyio.run(scenario)

    assert overlaps == [1, 1, 1]


def test_a_cancelled_call_cancels_the_worker_and_holds_the_lock_until_it_stops(
    loaded: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens, events = [], []
    analyzing = threading.Event()

    def blocking(request: object, loaded: object, cancel: app.CancelToken) -> object:
        tokens.append(cancel)
        analyzing.set()
        while not cancel.cancelled:
            time.sleep(0.01)
        time.sleep(0.3)
        events.append("worker stopped")
        raise app.Busy("stopped")

    def capabilities(loaded: object) -> dict:
        events.append("next call started")
        return {"status": "ok"}

    monkeypatch.setattr(mcp_server.app, "analyze", blocking)
    monkeypatch.setattr(mcp_server.doctor, "report", capabilities)

    async def scenario() -> object:
        async with Client(mcp_server.build_server(loaded)) as client:
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as group:
                    group.start_soon(
                        client.call_tool, "speech_analyze", {"path": "a.wav", "request_id": "r-1"}
                    )
                    await anyio.to_thread.run_sync(analyzing.wait)
                    scope.cancel()
            with anyio.fail_after(10):
                return await client.call_tool("speech_capabilities", {})

    following = anyio.run(scenario)

    assert following.is_error is not True
    assert len(tokens) == 1 and tokens[0].cancelled
    assert events == ["worker stopped", "next call started"]


def test_a_call_cancelled_before_its_thread_starts_does_not_wait_for_it(
    loaded: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(mcp_server, "call", lambda *arguments: calls.append(arguments))
    outcome = []

    async def scenario() -> None:
        with anyio.CancelScope() as scope:
            scope.cancel()
            await mcp_server._call_in_thread("speech_capabilities", {}, loaded)
        outcome.append(scope.cancelled_caught)

    runner = threading.Thread(target=anyio.run, args=(scenario,), daemon=True)
    runner.start()
    runner.join(10)

    assert not runner.is_alive()
    assert outcome == [True]
    assert calls == []


def test_log_lines_mask_the_data_root_the_allowed_roots_and_the_home_directory(
    loaded: config.Config,
) -> None:
    formatter = mcp_server.MaskingFormatter(mcp_server.private_paths(loaded))
    home = Path.home()
    try:
        raise ValueError(f"cannot read {loaded.data_root}/runs/x")
    except ValueError:
        record = logging.LogRecord(
            "sound_scout.mcp",
            logging.ERROR,
            __file__,
            1,
            "refused %s and %s",
            (f"{loaded.allowed_roots[0]}/a.wav", f"{home}/notes.txt"),
            sys.exc_info(),
        )
    text = formatter.format(record)

    assert "refused <allowed root>/a.wav and ~/notes.txt" in text
    assert "cannot read <data root>/runs/x" in text
    assert str(loaded.data_root) not in text and str(home) + "/" not in text


# --- the server over stdio, against the CLI (T09a) --------------------------------------------


@pytest.fixture
def stdio_env(tmp_path: Path) -> dict[str, str]:
    """The guard on (`network.allow = false`), backend none, and HOME at the temp directory."""
    path = write_config(tmp_path, allow_network=False, asr_model=ABSENT_ASR_MODEL)
    path.write_text('backend = "none"\n' + path.read_text(encoding="utf-8"), encoding="utf-8")
    env = offline_env(path)
    env["HOME"] = str(tmp_path)
    return env


@needs_ffmpeg
def test_stdout_is_only_json_rpc_and_logs_go_to_stderr_with_user_paths_masked(
    stdio_env: dict[str, str], tmp_path: Path, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/tone.wav")
    outside = make_wav("elsewhere.wav")
    session = StdioSession(stdio_env)

    initialized = session.initialize()["result"]
    listed = session.request("tools/list")["result"]
    capabilities = session.call_tool("speech_capabilities", {})
    analyzed = session.call_tool("speech_analyze", {"path": str(wav), "request_id": "r-1"})
    refused = session.call_tool("speech_analyze", {"path": str(outside), "request_id": "r-2"})
    code, stderr = session.close()

    assert code == 0, stderr
    assert initialized["serverInfo"]["name"] == "sound-scout"
    assert "sampling" not in initialized["capabilities"]
    assert [tool["name"] for tool in listed["tools"]] == TOOL_NAMES
    assert capabilities.get("isError") is not True
    assert answer_of(capabilities)["network_guard"]["active"] is True
    assert answer_of(analyzed)["status"] == "partial"
    assert refused["isError"] is True
    for line in session.lines:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0"
        assert "id" in message and ("result" in message or "error" in message)
    assert len(session.lines) == 5

    assert "sound_scout.mcp: speech_analyze: partial in " in stderr
    assert "speech_analyze refused: path_denied: outside allowed roots in " in stderr
    assert "elsewhere.wav" not in stderr
    assert str(tmp_path) not in stderr


@needs_ffmpeg
def test_cli_and_mcp_give_the_same_result_and_the_same_compact_answer(
    stdio_env: dict[str, str], tmp_path: Path, make_wav: WavFactory
) -> None:
    """T09a: one synthetic WAV through `analyze --json` and through `speech_analyze`."""
    wav = make_wav("recordings/tone.wav", seconds=1.5)
    data = tmp_path / "data"
    cli = subprocess.run(
        [sys.executable, "-m", "sound_scout", "analyze", str(wav), "--json", "--request-id", "c"],
        capture_output=True,
        text=True,
        check=False,
        env=stdio_env,
    )
    assert cli.returncode == 3, cli.stderr
    session = StdioSession(stdio_env)
    session.initialize()
    analyzed = session.call_tool("speech_analyze", {"path": str(wav), "request_id": "m"})
    code, stderr = session.close()
    assert code == 0, stderr

    by_cli, by_mcp = json.loads(cli.stdout), answer_of(analyzed)
    result_cli = json.loads((data / by_cli["run_dir"] / "result.json").read_text("utf-8"))
    result_mcp = json.loads((data / by_mcp["run_dir"] / "result.json").read_text("utf-8"))
    assert result_mcp["request_id"] == "m" and result_cli["request_id"] == "c"
    assert result_mcp["run_id"] != result_cli["run_id"]
    assert without_run_identity(result_mcp) == without_run_identity(result_cli)
    assert without_run_identity(by_mcp) == without_run_identity(by_cli)
    assert by_mcp == report.compact(result_mcp, run_dir=f"runs/{result_mcp['run_id']}")
    assert sorted(os.listdir(data / by_mcp["run_dir"])) == sorted(
        os.listdir(data / by_cli["run_dir"])
    )

    text = text_of(analyzed)
    assert len(text.encode("utf-8")) <= report.COMPACT_MAX_BYTES
    assert not BASE64_RUN.search(text)
    assert str(tmp_path) not in text
