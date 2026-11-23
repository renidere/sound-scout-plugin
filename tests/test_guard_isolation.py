"""The autouse fixture `reset_process_globals` of `tests/conftest.py` keeps the network guard,
which a CLI test installs in the pytest process, out of the next test (#172). A child `pytest`
runs two tests in a fixed order: A installs the guard through the CLI; B connects on loopback
and finds the guard's environment variables as the session started (`HF_HUB_OFFLINE` unset,
`ORT_DISABLE_TELEMETRY=0`), and the SIGHUP handler, `tempfile.tempdir` and the current directory
that A changed back as they were."""

import json
import os
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sound_scout import config, guard
from sound_scout.config import Config, Network

REPO = Path(__file__).resolve().parent.parent

ORDERED_TESTS = """
import os
import signal
import socket
import tempfile

from sound_scout import cli, guard

GUARD_ENV = [*guard.OFFLINE_ENV, *guard.TELEMETRY_OFF_ENV]
STARTING_ENV = {name: os.environ.get(name) for name in GUARD_ENV}
STARTING = (signal.getsignal(signal.SIGHUP), tempfile.tempdir, os.getcwd())


def test_a_installs_the_guard_through_the_cli(capsys, tmp_path):
    signal.signal(signal.SIGHUP, lambda *_: None)
    tempfile.tempdir = str(tmp_path)
    os.chdir(tmp_path)
    assert cli.main(["jobs", "limits", "--json"]) == 0
    assert guard.network_guard_active()
    try:
        socket.getaddrinfo("localhost", 80)
    except guard.NetworkDenied:
        pass
    assert guard.denied_events() == ("socket.getaddrinfo",)


def test_b_opens_a_loopback_socket():
    assert not guard.network_guard_active() and guard.denied_events() == ()
    assert {name: os.environ.get(name) for name in GUARD_ENV} == STARTING_ENV
    assert (signal.getsignal(signal.SIGHUP), tempfile.tempdir, os.getcwd()) == STARTING
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with socket.create_connection(listener.getsockname(), timeout=5):
            pass
"""


def run_ordered(tmp_path: Path, *plugin: str) -> subprocess.CompletedProcess[str]:
    recordings, data = tmp_path / "recordings", tmp_path / "data"
    recordings.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    settings = tmp_path / "sound-scout.toml"
    settings.write_text(
        f"allowed_roots = [{json.dumps(str(recordings))}]\n"
        f"data_root = {json.dumps(str(data))}\n[network]\nallow = false\n",
        encoding="utf-8",
    )
    module = tmp_path / "test_ordered.py"
    module.write_text(textwrap.dedent(ORDERED_TESTS), encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in guard.OFFLINE_ENV and key != "PYTEST_ADDOPTS"
    }
    env.update({config.CONFIG_PATH_ENV: str(settings), "ORT_DISABLE_TELEMETRY": "0"})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", "pyproject.toml"]
        + [*plugin, "--rootdir", str(tmp_path), str(module)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_a_guard_the_cli_installs_does_not_reach_the_next_test(tmp_path: Path) -> None:
    result = run_ordered(tmp_path, "-p", "tests.conftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_without_the_fixture_the_guard_leaks_into_the_next_test(tmp_path: Path) -> None:
    result = run_ordered(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 passed" in result.stdout
    assert re.search(
        r"^FAILED \S*test_ordered\.py::test_b_opens_a_loopback_socket", result.stdout, re.M
    )


def denying_config() -> Config:
    return Config(allowed_roots=(Path("/"),), data_root=Path("/"), network=Network(allow=False))


@pytest.mark.parametrize("phase", ["installed", "reset"])
def test_the_hook_denies_only_while_the_guard_is_on(phase: str) -> None:
    assert guard.install_network_guard(denying_config()) is True
    if phase == "reset":
        guard.reset_for_tests()
        socket.getaddrinfo("127.0.0.1", 80)
        assert not guard.network_guard_active() and guard.denied_events() == ()
    else:
        with pytest.raises(guard.NetworkDenied):
            socket.getaddrinfo("127.0.0.1", 80)
        assert guard.network_guard_active() and guard.denied_events() == ("socket.getaddrinfo",)


def test_the_guard_comes_back_on_a_second_install_after_a_reset() -> None:
    guard.install_network_guard(denying_config())
    guard.reset_for_tests()
    assert guard.install_network_guard(denying_config()) is True
    with pytest.raises(guard.NetworkDenied):
        socket.getaddrinfo("127.0.0.1", 80)


def test_no_product_code_switches_the_guard_off() -> None:
    callers = [
        path.relative_to(REPO).as_posix()
        for path in (REPO / "src").rglob("*.py")
        if "reset_for_tests(" in path.read_text(encoding="utf-8")
    ]
    assert callers == ["src/sound_scout/guard.py"]


PARTIAL_RESET = """
import os

import pytest

from sound_scout import guard


@pytest.fixture(autouse=True)
def reset_part():
    env = {name: os.environ.get(name) for name in [*guard.OFFLINE_ENV, *guard.TELEMETRY_OFF_ENV]}
    yield
    guard.reset_for_tests()
    if RESET_ENV:
        for name, value in env.items():
            os.environ.pop(name, None) if value is None else os.environ.update({name: value})
"""


@pytest.mark.parametrize(
    ("reset_env", "leak"),
    [(False, "STARTING_ENV"), (True, "STARTING")],
    ids=["guard_env_leaks", "signal_tempdir_cwd_leak"],
)
def test_each_reset_of_the_fixture_is_needed(tmp_path: Path, reset_env: bool, leak: str) -> None:
    """Without the env reset B sees the guard's variables; with it but without the rest, B sees
    A's SIGHUP handler, `tempfile.tempdir` or current directory: the checks of B are not void."""
    (tmp_path / "conftest.py").write_text(
        f"RESET_ENV = {reset_env}\n" + textwrap.dedent(PARTIAL_RESET), encoding="utf-8"
    )
    result = run_ordered(tmp_path)
    assert "1 failed, 1 passed" in result.stdout, result.stdout + result.stderr
    failure = re.search(r"^>\s+assert (.+)$", result.stdout, re.M)
    assert failure and failure[1].rstrip().endswith(f"== {leak}"), result.stdout
