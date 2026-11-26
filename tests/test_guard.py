"""An audit hook cannot be removed, so once a test installs the guard the hook stays in the pytest
process; it denies only while the guard flag is on, and the autouse fixture
`reset_process_globals` switches the flag off after each test. Tests of the denying hook itself
run in a child interpreter so they see a fresh process."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from sound_scout import guard
from sound_scout.config import Config, Network
from tests import conftest

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


def test_guard_leaves_unix_socket_bind_allowed(short_socket_dir: Path) -> None:
    outcome = run_in_child(
        f"""
    guard.install_network_guard(config(allow=False))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind({str(short_socket_dir / "listener")!r})
        sock.listen(1)
    print(json.dumps({{"bound": True}}))
    """
    )
    assert outcome["bound"] is True


LONG_TMPDIR_CHARS = 100
UNIX_SOCKET_TESTS = (
    "tests/test_guard.py::test_guard_leaves_unix_socket_bind_allowed",
    "tests/test_gpu_host.py::"
    "test_the_hook_denies_an_internet_bind_and_logs_it_but_allows_unix_and_loopback",
    "tests/test_gpu_host.py::test_a_guarded_server_starts_workers_that_stay_guarded",
)
UNIX_SOCKET_TEST_CASES = 6


@pytest.mark.slow
def test_unix_socket_tests_pass_with_a_long_tmpdir(tmp_path: Path) -> None:
    # With TMPDIR this long, a socket in `tmp_path` passes the 107-byte limit of Linux (#196).
    long_tmpdir = tmp_path / ("d" * max(1, LONG_TMPDIR_CHARS - len(str(tmp_path)) - 1))
    long_tmpdir.mkdir()
    assert len(str(long_tmpdir)) >= LONG_TMPDIR_CHARS
    env = {k: v for k, v in os.environ.items() if k not in conftest.TEMP_DIR_VARIABLES}
    env.pop("PYTEST_ADDOPTS", None)
    env["TMPDIR"] = str(long_tmpdir)
    socket_parent = Path(
        tempfile.mkdtemp(prefix=conftest.SHORT_SOCKET_PREFIX, dir=conftest.SHORT_SOCKET_PARENT)
    )
    env[conftest.SHORT_SOCKET_PARENT_VARIABLE] = str(socket_parent)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *UNIX_SOCKET_TESTS],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        left = list(socket_parent.iterdir())
    finally:
        shutil.rmtree(socket_parent, ignore_errors=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert f"{UNIX_SOCKET_TEST_CASES} passed" in result.stdout
    assert not left


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


def test_a_denial_the_caller_swallows_stays_on_record() -> None:
    outcome = run_in_child(
        attempt_connection()
        + """
    guard.install_network_guard(config(allow=False))
    before = guard.denied_events()
    attempt()
    socket.socket(socket.AF_UNIX).close()
    print(json.dumps({"before": list(before), "after": list(guard.denied_events())}))
    """
    )
    assert outcome == {"before": [], "after": ["socket.getaddrinfo"]}


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


def test_after_install_the_process_stays_guarded_until_it_exits(tmp_path: Path) -> None:
    """Production path (#172): the CLI installs the guard; later CLI calls, an allowing install
    and the rest of the process never switch it off; a real loopback connect stays denied."""
    recordings, data = tmp_path / "recordings", tmp_path / "data"
    recordings.mkdir()
    data.mkdir()
    settings = tmp_path / "sound-scout.toml"
    settings.write_text(
        f"allowed_roots = [{json.dumps(str(recordings))}]\n"
        f"data_root = {json.dumps(str(data))}\n[network]\nallow = false\n",
        encoding="utf-8",
    )
    outcome = run_in_child(
        """
    import contextlib, io
    from sound_scout import cli

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    address = listener.getsockname()

    def connect():
        try:
            with socket.socket() as sock:
                sock.settimeout(5)
                sock.connect(address)
        except guard.NetworkDenied:
            return "denied"
        return "connected"

    attempts = [connect()]
    with contextlib.redirect_stdout(io.StringIO()):
        codes = [cli.main(["jobs", "limits", "--json"])]
        attempts.append(connect())
        codes.append(cli.main(["jobs", "limits", "--json"]))
    attempts.append(connect())
    allowing = guard.install_network_guard(config(allow=True))
    attempts.append(connect())
    print(json.dumps({
        "codes": codes, "attempts": attempts, "allowing": allowing,
        "active": guard.network_guard_active(), "denied": list(guard.denied_events()),
    }))
    """,
        {"SOUND_SCOUT_CONFIG": str(settings)},
    )
    assert outcome["codes"] == [0, 0]
    assert outcome["attempts"] == ["connected", "denied", "denied", "denied"]
    assert outcome["allowing"] is False
    assert outcome["active"] is True
    assert outcome["denied"] == ["socket.connect"] * 3
