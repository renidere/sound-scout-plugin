"""The network guard is installed in a child interpreter: an audit hook cannot be removed and
must not poison the test session; only the flag and the `allow = true` path run in-process."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sound_scout import guard
from sound_scout.config import Config, Network

PRELUDE = """
import json, os, socket, sys
from pathlib import Path
from sound_scout import guard
from sound_scout.config import Config, Network

def config(allow):
    return Config(allowed_roots=(Path("/"),), data_root=Path("/"), network=Network(allow=allow))
"""


def run_in_child(body: str, extra_env: dict[str, str] | None = None) -> dict:
    """Run the script in a fresh interpreter and return the JSON it prints on its last line."""
    script = PRELUDE + textwrap.dedent(body)
    managed = guard.OFFLINE_ENV.keys() | guard.TELEMETRY_OFF_ENV.keys()
    env = {key: value for key, value in os.environ.items() if key not in managed}
    env.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def attempt_connection() -> str:
    return """
    def attempt():
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=1)
        except guard.NetworkDenied as error:
            return "denied: " + str(error)
        except OSError as error:
            return "os error: " + str(error)
        return "connected"
    """


def test_guard_denies_connections_and_sets_offline_env() -> None:
    outcome = run_in_child(
        attempt_connection()
        + """
    before = network_guard_before = guard.network_guard_active()
    installed = guard.install_network_guard(config(allow=False))
    print(json.dumps({
        "before": before,
        "installed": installed,
        "active": guard.network_guard_active(),
        "attempt": attempt(),
        "env": {name: os.environ.get(name) for name in guard.OFFLINE_ENV},
    }))
    """
    )
    assert outcome["before"] is False
    assert outcome["installed"] is True
    assert outcome["active"] is True
    assert outcome["attempt"].startswith("denied: network access is disabled")
    assert outcome["env"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("inherited", [None, "0"])
def test_guard_switches_telemetry_off_whatever_network_allow_says(
    allow: bool, inherited: str | None
) -> None:
    outcome = run_in_child(
        f"""
    before = os.environ.get("ORT_DISABLE_TELEMETRY")
    installed = guard.install_network_guard(config(allow={allow}))
    print(json.dumps({{
        "before": before,
        "installed": installed,
        "telemetry": os.environ.get("ORT_DISABLE_TELEMETRY"),
    }}))
    """,
        {} if inherited is None else {"ORT_DISABLE_TELEMETRY": inherited},
    )
    assert outcome == {"before": inherited, "installed": not allow, "telemetry": "1"}


@pytest.mark.parametrize(
    "operation",
    [
        'socket.socket().connect(("127.0.0.1", 9))',
        'socket.socket().connect_ex(("127.0.0.1", 9))',
        'socket.getaddrinfo("localhost", 80)',
        'socket.gethostbyname("localhost")',
        'socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("127.0.0.1", 9))',
        'socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendmsg([b"x"], [], 0, ("127.0.0.1", 9))',
        'socket.socket(socket.AF_INET).bind(("127.0.0.1", 0))',
        'socket.socket(socket.AF_INET6).bind(("::1", 0))',
    ],
)
def test_guard_denies_each_network_operation(operation: str) -> None:
    outcome = run_in_child(
        f"""
    guard.install_network_guard(config(allow=False))
    try:
        {operation}
        result = "allowed"
    except guard.NetworkDenied:
        result = "denied"
    except OSError as error:
        result = "os error: " + str(error)
    print(json.dumps({{"result": result}}))
    """
    )
    assert outcome["result"] == "denied"


def test_guard_leaves_unix_socket_bind_allowed(tmp_path: Path) -> None:
    outcome = run_in_child(
        f"""
    guard.install_network_guard(config(allow=False))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind({str(tmp_path / "listener")!r})
        sock.listen(1)
    print(json.dumps({{"bound": True}}))
    """
    )
    assert outcome["bound"] is True


def test_guard_is_installed_once_for_repeated_calls() -> None:
    outcome = run_in_child(
        attempt_connection()
        + """
    hooks = []
    real_addaudithook = sys.addaudithook
    sys.addaudithook = lambda hook: (hooks.append(hook), real_addaudithook(hook))
    results = [guard.install_network_guard(config(allow=False)) for _ in range(3)]
    print(json.dumps({"hooks": len(hooks), "results": results, "attempt": attempt()}))
    """
    )
    assert outcome["hooks"] == 1
    assert outcome["results"] == [True, True, True]
    assert outcome["attempt"].startswith("denied")


def test_without_guard_the_same_connection_fails_as_a_plain_os_error() -> None:
    report = '\n    print(json.dumps({"attempt": attempt()}))\n'
    outcome = run_in_child(attempt_connection() + report)
    assert outcome["attempt"].startswith("os error")


def test_allowed_network_installs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in guard.OFFLINE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    allowed = Config(allowed_roots=(Path("/"),), data_root=Path("/"), network=Network(allow=True))
    assert guard.install_network_guard(allowed) is False
    assert guard.network_guard_active() is False
    for name in guard.OFFLINE_ENV:
        assert name not in os.environ
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"


def test_guard_flag_is_off_in_this_test_session() -> None:
    assert guard.network_guard_active() is False
