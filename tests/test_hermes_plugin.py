"""Spike B package of the Hermes plugin (#96, decision 0015), loaded without Hermes.

A stub `fastapi` records the routes; the `_deliver`/`_run` seams (or stub `gateway`/
`hermes_constants` modules) stand in for the gateway and systemd. No real `systemd-run` runs.
"""

import ast
import asyncio
import importlib.util
import json
import re
import subprocess
import sys
import threading
import types
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "hermes" / "plugin" / "sound-scout"
API = PACKAGE / "dashboard" / "plugin_api.py"
PLUGIN_JS = PACKAGE / "desktop" / "plugin.js"
DECISION = REPO / "docs" / "decisions" / "0015-spike-b-backend-package.md"
ROUTES = {
    ("GET", "/health"),
    ("POST", "/spike/deliver"),
    ("GET", "/spike/deliver/{token}"),
    ("POST", "/spike/runner"),
    ("GET", "/spike/runner"),
}


class StubRouter:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], object] = {}

    def _route(self, method: str, path: str):
        def record(handler):
            self.routes[(method, path)] = handler
            return handler

        return record

    def get(self, path: str):
        return self._route("GET", path)

    def post(self, path: str):
        return self._route("POST", path)


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", _module("fastapi", APIRouter=StubRouter))
    spec = importlib.util.spec_from_file_location("sound_scout_spike_b_plugin_api", API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    _join_delivery_threads(module)


def _join_delivery_threads(api) -> None:
    for thread in threading.enumerate():
        if thread.name == api.DELIVER_THREAD:
            thread.join(timeout=5)
            assert not thread.is_alive()


def call(api, method: str, path: str, *args):
    return asyncio.run(api.router.routes[(method, path)](*args))


def delivery(api, token: str) -> dict:
    return call(api, "GET", "/spike/deliver/{token}", token)


def without_hermes(monkeypatch) -> None:
    for name in ("hermes_constants", "gateway", "gateway.session_cron"):
        monkeypatch.setitem(sys.modules, name, None)  # `import` of a None entry raises ImportError


def with_hermes(monkeypatch, *, owner, delivered: list) -> None:
    def deliver_to_local_chat(home, chat_id, text):
        delivered.append((home, chat_id, text))
        return None

    session_cron = _module(
        "gateway.session_cron",
        owner_for_home=lambda home: owner,
        deliver_to_local_chat=deliver_to_local_chat,
    )
    monkeypatch.setitem(sys.modules, "gateway", _module("gateway", session_cron=session_cron))
    monkeypatch.setitem(sys.modules, "gateway.session_cron", session_cron)
    monkeypatch.setitem(
        sys.modules, "hermes_constants", _module("hermes_constants", get_hermes_home=lambda: "HOME")
    )


def test_router_has_exactly_the_spike_routes(api) -> None:
    assert set(api.router.routes) == ROUTES


@pytest.mark.parametrize("owner", [(object(), object()), None])
def test_health_reports_whether_this_process_owns_the_sessions(api, monkeypatch, owner) -> None:
    with_hermes(monkeypatch, owner=owner, delivered=[])
    answer = call(api, "GET", "/health")
    assert answer["session_owner"] is (owner is not None)
    assert answer["ok"] is True
    assert answer["plugin"] == "sound-scout"
    assert answer["stage"] == "spike-b"
    assert answer["version"] == "0.0.1"
    assert isinstance(answer["pid"], int)


def test_health_without_hermes_has_no_session_owner(api, monkeypatch) -> None:
    without_hermes(monkeypatch)
    assert call(api, "GET", "/health")["session_owner"] is None


def test_deliver_calls_the_gateway_once_from_a_worker_thread(api, monkeypatch) -> None:
    calls = []
    release = threading.Event()

    def deliver(home, chat_id, text):
        calls.append((home, chat_id, text, threading.current_thread()))
        release.wait(5)
        return None

    monkeypatch.setattr(api, "_home", lambda: "HOME")
    monkeypatch.setattr(api, "_deliver", deliver)
    answer = call(api, "POST", "/spike/deliver", {"chat_id": "chat-1"})
    assert answer["ok"] is True
    assert answer["delay_s"] == 0
    token = answer["token"]
    assert delivery(api, token) == {
        "ok": True,
        "state": "pending",
        "error": None,
        "chat_id": "chat-1",
    }

    release.set()
    _join_delivery_threads(api)
    assert delivery(api, token) == {
        "ok": True,
        "state": "delivered",
        "error": None,
        "chat_id": "chat-1",
    }
    assert len(calls) == 1
    home, chat_id, text, thread = calls[0]
    assert (home, chat_id) == ("HOME", "chat-1")
    assert thread is not threading.current_thread()
    assert thread.name == api.DELIVER_THREAD
    first, directive = text.split("\n\n")
    assert token in first
    assert directive == f'::{api.DIRECTIVE}{{job="spike-{token}"}}'


def test_deliver_reaches_the_gateway_function_of_hermes(api, monkeypatch) -> None:
    delivered = []
    with_hermes(monkeypatch, owner=None, delivered=delivered)
    token = call(api, "POST", "/spike/deliver", {"chat_id": "chat-2", "delay_s": 0})["token"]
    _join_delivery_threads(api)
    assert delivered == [("HOME", "chat-2", api.note_text(token))]
    assert delivery(api, token)["state"] == "delivered"


def test_deliver_records_the_gateway_error(api, monkeypatch) -> None:
    calls = []

    def deliver(home, chat_id, text):
        calls.append(chat_id)
        return f"local chat {chat_id}: no gateway in this process owns its session"

    monkeypatch.setattr(api, "_deliver", deliver)
    token = call(api, "POST", "/spike/deliver", {"chat_id": "chat-3"})["token"]
    _join_delivery_threads(api)
    record = delivery(api, token)
    assert record["state"] == "failed"
    assert record["error"] == "local chat chat-3: no gateway in this process owns its session"
    assert calls == ["chat-3"]


def test_deliver_records_an_exception_without_retrying(api, monkeypatch) -> None:
    calls = []

    def deliver(home, chat_id, text):
        calls.append(chat_id)
        raise RuntimeError("owner loop stalled")

    monkeypatch.setattr(api, "_deliver", deliver)
    token = call(api, "POST", "/spike/deliver", {"chat_id": "chat-4"})["token"]
    _join_delivery_threads(api)
    record = delivery(api, token)
    assert record["state"] == "failed"
    assert record["error"] == repr(RuntimeError("owner loop stalled"))
    assert calls == ["chat-4"]


def test_deliver_without_hermes_fails_with_a_reason(api, monkeypatch) -> None:
    without_hermes(monkeypatch)
    token = call(api, "POST", "/spike/deliver", {"chat_id": "chat-5"})["token"]
    _join_delivery_threads(api)
    record = delivery(api, token)
    assert record["state"] == "failed"
    assert "deliver_to_local_chat unavailable" in record["error"]


def test_deliver_waits_the_delay_in_the_worker(api, monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        api, "time", types.SimpleNamespace(sleep=lambda s: events.append(("sleep", s)))
    )
    monkeypatch.setattr(api, "_deliver", lambda home, chat_id, text: events.append(("deliver",)))
    answer = call(api, "POST", "/spike/deliver", {"chat_id": "c" * 256, "delay_s": 120})
    assert answer["ok"] is True
    assert answer["delay_s"] == 120
    _join_delivery_threads(api)
    assert events == [("sleep", 120), ("deliver",)]


@pytest.mark.parametrize("chat_id", ["", "c" * 257, "chat\n1", None, 5])
def test_deliver_rejects_a_bad_chat_id(api, monkeypatch, chat_id) -> None:
    monkeypatch.setattr(api, "_deliver", lambda *args: pytest.fail("must not deliver"))
    answer = call(api, "POST", "/spike/deliver", {"chat_id": chat_id})
    assert answer["ok"] is False
    assert answer["error"] == "bad_chat_id"
    assert api._deliveries == {}


@pytest.mark.parametrize("delay_s", [-1, 121, True, "5", 1.5, None])
def test_deliver_rejects_a_bad_delay(api, monkeypatch, delay_s) -> None:
    monkeypatch.setattr(api, "_deliver", lambda *args: pytest.fail("must not deliver"))
    answer = call(api, "POST", "/spike/deliver", {"chat_id": "chat-6", "delay_s": delay_s})
    assert answer["ok"] is False
    assert answer["error"] == "bad_delay"
    assert api._deliveries == {}


def test_unknown_delivery_token_is_refused(api) -> None:
    answer = delivery(api, uuid.uuid4().hex)
    assert answer["ok"] is False
    assert answer["error"] == "unknown_token"


def fake_run(calls: list, *, returncode=0, stdout="", stderr=""):
    def run(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(list(argv), returncode, stdout, stderr)

    return run


def test_runner_start_runs_the_fixed_systemd_run_argv(api, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(api, "_run", fake_run(calls))
    answer = call(api, "POST", "/spike/runner")
    assert calls == [api.START_ARGV]
    assert api.START_ARGV[:3] == ("systemd-run", "--user", "--unit=sound-scout-spike-b")
    assert answer == {"ok": True, "unit": api.UNIT, "returncode": 0, "stderr": ""}


def test_runner_start_reports_a_refusal_of_systemd(api, monkeypatch) -> None:
    stderr = "x" * 3000 + "Failed to start transient service unit: Unit already exists."
    monkeypatch.setattr(api, "_run", fake_run([], returncode=1, stderr=stderr))
    answer = call(api, "POST", "/spike/runner")
    assert answer["ok"] is False
    assert answer["returncode"] == 1
    assert answer["stderr"] == stderr[-api.STDERR_TAIL :]


@pytest.mark.parametrize(
    ("method", "exc", "error"),
    [
        ("POST", FileNotFoundError("systemd-run"), "systemd_run_missing"),
        ("POST", subprocess.TimeoutExpired("systemd-run", 15), "timeout"),
        ("POST", PermissionError("denied"), "run_failed"),
        ("GET", FileNotFoundError("systemctl"), "systemd_run_missing"),
        ("GET", subprocess.TimeoutExpired("systemctl", 15), "timeout"),
    ],
)
def test_runner_routes_turn_a_failed_run_into_a_refusal(api, monkeypatch, method, exc, error):
    def run(argv):
        raise exc

    monkeypatch.setattr(api, "_run", run)
    answer = call(api, method, "/spike/runner")
    assert answer["ok"] is False
    assert answer["error"] == error
    assert answer["detail"]


def test_runner_status_parses_the_unit_properties(api, monkeypatch) -> None:
    calls = []
    stdout = (
        "LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=4242\n"
        "ControlGroup=/user.slice/app.slice/sound-scout-spike-b.service\n\nno separator\n"
    )
    monkeypatch.setattr(api, "_run", fake_run(calls, stdout=stdout))
    answer = call(api, "GET", "/spike/runner")
    assert calls == [api.STATUS_ARGV]
    assert api.STATUS_ARGV[:4] == ("systemctl", "--user", "show", api.UNIT)
    assert answer["ok"] is True
    assert answer["properties"] == {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": "4242",
        "ControlGroup": "/user.slice/app.slice/sound-scout-spike-b.service",
    }


def test_runner_status_reports_a_failed_systemctl(api, monkeypatch) -> None:
    monkeypatch.setattr(api, "_run", fake_run([], returncode=1, stderr="Failed to connect to bus"))
    answer = call(api, "GET", "/spike/runner")
    assert answer["ok"] is False
    assert answer["stderr"] == "Failed to connect to bus"
    assert answer["properties"] == {}


@pytest.mark.parametrize("runtime_dir", [None, "RUNTIME"])
def test_run_passes_argv_without_shell_and_a_runtime_dir(api, monkeypatch, runtime_dir) -> None:
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    if runtime_dir is None:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    else:
        monkeypatch.setenv("XDG_RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(api.subprocess, "run", run)
    api._run(api.START_ARGV)
    assert seen["argv"] == list(api.START_ARGV)
    assert seen["timeout"] == api.RUN_TIMEOUT_S
    assert "shell" not in seen
    expected = runtime_dir or f"/run/user/{api.os.getuid()}"
    assert seen["env"]["XDG_RUNTIME_DIR"] == expected


def test_plugin_api_imports_only_the_stdlib_and_fastapi_and_parses_as_python_311() -> None:
    tree = ast.parse(API.read_text(encoding="utf-8"), feature_version=(3, 11))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module.split(".")[0])
    assert top_level - set(sys.stdlib_module_names) == {"fastapi"}
    assert "sound_scout" not in API.read_text(encoding="utf-8").replace("sound-scout", "")


def _rest_calls(source: str) -> set[tuple[str, str]]:
    pattern = re.compile(
        r"ctx\.rest\(\s*[`'](?P<path>[^`']+)[`'](?:\s*,\s*\{\s*method:\s*'(?P<m>\w+)')?"
    )
    return {
        (match["m"] or "GET", re.sub(r"\$\{\w+\}", "{token}", match["path"]))
        for match in pattern.finditer(source)
    }


def test_desktop_plugin_calls_every_route_and_claims_the_directive(api) -> None:
    source = PLUGIN_JS.read_text(encoding="utf-8")
    assert "id: 'sound-scout'" in source
    imports = set(re.findall(r"^import .* from '([^']+)'", source, re.MULTILINE))
    assert imports == {"@hermes/plugin-sdk", "react/jsx-runtime"}
    assert source.count("import ") == 2
    assert _rest_calls(source) == set(api.router.routes)
    assert f"const DIRECTIVE = '{api.DIRECTIVE}'" in source
    assert "name: DIRECTIVE" in source
    job_pattern = re.search(r"const SPIKE_JOB = /(.+)/\n", source)[1]
    note_job = re.search(r'job="([^"]+)"', api.note_text(uuid.uuid4().hex))[1]
    assert re.fullmatch(job_pattern, note_job)
    assert not re.fullmatch(job_pattern, "spike-../../etc")


def test_package_manifests_name_the_plugin_and_its_api() -> None:
    manifest = json.loads((PACKAGE / "dashboard" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "sound-scout"
    assert manifest["api"] == API.name
    assert manifest["tab"]["hidden"] is True
    plugin_yaml = (PACKAGE / "plugin.yaml").read_text(encoding="utf-8")
    assert re.search(r"^name: sound-scout$", plugin_yaml, re.MULTILINE)
    assert not re.search(r"^kind:", plugin_yaml, re.MULTILINE)


def test_agent_half_registers_nothing() -> None:
    spec = importlib.util.spec_from_file_location(
        "sound_scout_spike_b_agent", PACKAGE / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.register(object()) is None


def test_decision_records_the_routes_the_unit_and_the_argv(api) -> None:
    text = DECISION.read_text(encoding="utf-8")
    assert "Статус:** инструкция готова; результаты ожидают прогона владельцем" in text
    assert " ".join(api.START_ARGV) in text
    assert api.UNIT in text
    for method, path in api.router.routes:
        assert f"{method} /api/plugins/{api.PLUGIN}{path}" in text
    for section in ("## Инструкция владельцу", "## Результаты", "## Удаление"):
        assert section in text.splitlines()
    for command in (
        "hermes config get plugins.isolation",
        "hermes plugins enable sound-scout",
        "hermes plugins remove sound-scout",
        "systemctl --user restart hermes-autonomous.service",
    ):
        assert command in text
    assert "/home/" not in text
    assert str(Path.home()) not in text
    assert not re.search(r"\b[A-Za-z]:\\", text)


def _section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1 : end])


def test_criterion_2_instruction_enables_in_process_and_removes_the_package() -> None:
    text = DECISION.read_text(encoding="utf-8")
    instruction = _section(text, "## Инструкция владельцу")
    assert "hermes config get plugins.isolation" in instruction
    assert "`in_process`" in instruction
    assert "`host`" in instruction and "остановиться" in instruction
    install = instruction.index("cp -r <checkout sound-scout-plugin>/hermes/plugin/sound-scout")
    enable = instruction.index("hermes plugins enable sound-scout")
    restart = instruction.index("systemctl --user restart hermes-autonomous.service")
    assert install < enable < restart
    removal = _section(text, "## Удаление")
    remove = removal.index("hermes plugins remove sound-scout")
    assert remove < removal.index("systemctl --user restart hermes-autonomous.service")
    assert "hermes plugins list" in removal
    assert "desktop-plugins\\sound-scout" in removal


def test_criterion_6_decision_has_a_results_table_pending_the_owner() -> None:
    text = DECISION.read_text(encoding="utf-8")
    assert "результаты ожидают прогона владельцем" in text.splitlines()[2]
    rows = [
        line.split("|")[1:-1]
        for line in _section(text, "## Результаты").splitlines()
        if line.startswith("| ") and not line.startswith("| критерий")
    ]
    by_criterion = {cells[0].strip().split(".")[0]: cells[2].strip() for cells in rows}
    assert {"1", "2", "3", "4а", "4б", "4в", "5а", "5б", "5в", "6"} <= set(by_criterion)
    for criterion in ("3", "4а", "4б", "4в", "5а", "5б", "5в", "6"):
        assert by_criterion[criterion] == "ожидает прогона владельцем"
