"""#137: scripts/session_owner_check.py reports the session owner state on synthetic input."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "session_owner_check.py"

spec = importlib.util.spec_from_file_location("session_owner_check", SCRIPT)
check_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_script)

SENTINEL = "tok-3f9a1c-SENTINEL"
PEER = "192.0.2.77"
LAUNCHER = "import os, sys\nsys.path.insert(0, '/home/someone/hermes/hermes-agent')\n"
GATEWAY = ["/opt/py/bin/python3", "-I", "-c", LAUNCHER, "gateway", "run"]
SSH_SERVE = [
    "/opt/py/bin/python3",
    "-I",
    "-c",
    LAUNCHER,
    "serve",
    "--isolated",
    "--host",
    "127.0.0.1",
    "--port",
    "0",
    "--ssh-session-token-file",
    f"/home/someone/hermes/desktop-ssh/{SENTINEL}",
]
MCP_HELPER = ["/home/someone/.venv/bin/python", "/home/someone/.venv/bin/sound-scout", "serve"]
UV_HELPER = ["/usr/bin/uv", "run", "--offline", "sound-scout", "serve"]
BTIME = 1_700_000_000
TICKS = 123_456
BORN = BTIME + TICKS / os.sysconf("SC_CLK_TCK")


def units(serve_state: str = "inactive", gateway_pid: int = 100) -> str:
    return (
        f"Id=hermes-gateway.service\nActiveState=active\nUnitFileState=enabled\n"
        f"MainPID={gateway_pid}\n\n"
        f"Id=hermes-autonomous.service\nActiveState={serve_state}\nUnitFileState=disabled\n"
        f"MainPID={300 if serve_state == 'active' else 0}\n"
    )


def listen(*rows: tuple[str, int]) -> str:
    return "".join(
        f'LISTEN 0      2048   {addr}   0.0.0.0:*    users:(("hermes",pid={pid},fd=21))\n'
        for addr, pid in rows
    )


def established(*rows: tuple[int, int]) -> str:
    return "".join(
        f'0      0      10.0.0.5:{port}   {PEER}:5{port}   users:(("hermes",pid={pid},fd=30))\n'
        for port, pid in rows
    )


@pytest.mark.parametrize(
    ("argv", "kind"),
    [
        (GATEWAY, "gateway"),
        (SSH_SERVE, "serve"),
        (["/home/someone/.local/bin/hermes", "--profile", "x", "serve", "--isolated"], "serve"),
        (["hermes", "dashboard", "--port", "9119"], "dashboard"),
        (["/usr/bin/python3", "-m", "hermes_cli.main", "chat"], "chat"),
        (["/usr/bin/python3", "-m", "tui_gateway.entry"], "tui_gateway"),
        (["hermes", "gateway", "attach", "--json"], "gateway attach"),
        (["hermes", "gateway"], "gateway"),
        (["hermes", "-p", "work", "gateway", "run", "--replace"], "gateway"),
        (["hermes", "-m", "x", "serve"], "serve"),
        (["hermes", "--model=x", "--provider", "y", "dashboard"], "dashboard"),
        (["hermes", "-c", "serve", "chat"], "chat"),
        (["hermes", "-z", SENTINEL], "client"),
        (["hermes", "--oneshot", SENTINEL, "--toolsets", SENTINEL], "client"),
        (["hermes", SENTINEL], "client"),
        (["hermes", "gateway", SENTINEL], "client"),
        (["/usr/bin/python3", "/opt/hermes-agent/hermes_cli/main.py", "serve"], "serve"),
        (
            ["/usr/bin/python3", "-I", "/opt/hermes-agent/hermes_cli/main.py", "gateway", "run"],
            "gateway",
        ),
        (["hermes"], "client"),
        (MCP_HELPER, None),
        (UV_HELPER, None),
        (["/usr/bin/python3", "-c", "print(1)", "serve"], None),
        ([], None),
    ],
)
def test_classify_tells_the_gateway_from_writers_and_non_hermes(argv, kind) -> None:
    assert check_script.classify(argv) == kind


@pytest.mark.parametrize(
    "argv",
    [["hermes", "-z", SENTINEL], ["hermes", SENTINEL, "serve"], ["hermes", "gateway", SENTINEL]],
)
def test_classify_prints_no_argv_text(argv) -> None:
    kind = check_script.classify(argv)
    assert SENTINEL not in kind
    assert kind == "client" or kind in check_script.SUBCOMMANDS


def run_check(**over):
    args = {
        "units_text": units(),
        "listen_text": listen(("0.0.0.0:9119", 100)),
        "established_text": established((9119, 100), (9119, 100)),
        "ledger": [
            {"pid": 999, "purpose": "serve", "argv": SENTINEL, "create_time": BORN, "boot_id": "b"}
        ],
        "procs": {100: "gateway"},
        "alive": {100, 999},
        "births": {999: BORN},
        "boot_id": "a",
        "port": 9119,
    } | over
    return check_script.check(*args.values())


def test_clean_state_is_ok() -> None:
    report = run_check()
    assert report["verdict"] == "ok"
    assert "established: :9119 PID 100: 2" in report["lines"]
    assert "ledger serve PID 999: stale" in report["lines"]
    assert report["lines"][-1].startswith("verdict: ok — gateway PID 100 is the only writer")


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"procs": {100: "gateway", 200: "serve"}}, "live Hermes serve process 200"),
        ({"procs": {100: "gateway", 200: "dashboard"}}, "live Hermes dashboard process 200"),
        ({"units_text": units("active")}, "hermes-autonomous.service is active"),
        (
            {
                "procs": {100: "gateway", 200: "client"},
                "listen_text": listen(("0.0.0.0:9119", 100), ("127.0.0.1:37575", 200)),
            },
            "Hermes listener :37575 held by PID 200",
        ),
        (
            {
                "alive": {100, 200},
                "births": {200: BORN + 1.5},
                "ledger": [{"pid": 200, "purpose": "serve", "create_time": BORN, "boot_id": "a"}],
            },
            "spawn ledger: live serve PID 200",
        ),
        (
            {
                "alive": {100, 200},
                "births": {200: BORN},
                "ledger": [{"pid": 200, "purpose": "dashboard", "create_time": BORN}],
            },
            "spawn ledger: live dashboard PID 200",
        ),
    ],
)
def test_second_writer_is_reported(over, reason) -> None:
    report = run_check(**over)
    assert report["verdict"] == "second_writer"
    assert reason in report["lines"][-1]


@pytest.mark.parametrize(
    ("over", "verdict"),
    [
        ({"procs": {}}, "no_owner"),
        ({"units_text": units(gateway_pid=555)}, "owner_outside_unit"),
        ({"listen_text": listen(("0.0.0.0:9120", 100))}, "listener_missing"),
        ({"units_text": None}, "unreadable"),
        ({"ledger": None}, "unreadable"),
    ],
)
def test_missing_owner_or_input_is_not_ok(over, verdict) -> None:
    assert run_check(**over)["verdict"] == verdict


@pytest.mark.parametrize(
    ("entry", "births", "state"),
    [
        (
            {"pid": 999, "purpose": "serve", "create_time": BORN, "boot_id": "a"},
            {999: BORN + 60},
            "stale",
        ),
        (
            {"pid": 999, "purpose": "serve", "create_time": BORN, "boot_id": "a"},
            {999: None},
            "unverified",
        ),
        ({"pid": 999, "purpose": "serve", "boot_id": "a"}, {999: BORN}, "unverified"),
        ({"pid": 998, "purpose": "serve", "create_time": BORN}, {}, "stale"),
        ({"pid": 999, "purpose": SENTINEL, "create_time": BORN}, {999: BORN}, "live"),
        ({"pid": 100, "purpose": "serve", "create_time": BORN}, {100: BORN}, "live"),
    ],
)
def test_a_ledger_entry_without_a_matching_live_writer_is_not_a_writer(
    entry, births, state
) -> None:
    report = run_check(ledger=[entry], births=births)
    label = "other" if entry["purpose"] == SENTINEL else "serve"
    assert f"ledger {label} PID {entry['pid']}: {state}" in report["lines"]
    assert report["verdict"] == "ok"
    assert SENTINEL not in "\n".join(report["lines"])


def fake_host(tmp_path: Path, procs: dict[int, list[str]], serve_state: str) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    others = [("127.0.0.1:37575", pid) for pid in procs if pid != 100]
    listeners = listen(("0.0.0.0:9119", 100), *others)
    outputs = {
        "systemctl": units(serve_state)
        + f"ExecStart={{ path=/x ; argv[]=/x serve --token {SENTINEL} }}\n",
        "ss": "",
    }
    (bin_dir / "systemctl").write_text(f"#!/bin/sh\ncat <<'EOF'\n{outputs['systemctl']}EOF\n")
    (bin_dir / "ss").write_text(
        "#!/bin/sh\n"
        "case \"$*\" in *established*) cat <<'EOF'\n"
        f"{established((9119, 100))}EOF\n;; *) cat <<'EOF'\n{listeners}EOF\n;; esac\n"
    )
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-1\n")
    (proc / "stat").write_text(f"cpu  1 2 3 4\nbtime {BTIME}\nprocesses 9\n")
    stat_tail = " ".join(["S", *["0"] * 18, str(TICKS), "0", "0"])
    for pid, argv in procs.items():
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (proc / str(pid) / "stat").write_text(f"{pid} (py (x) 3) {stat_tail}\n")
    home = tmp_path / "home"
    home.mkdir()
    ledger = [
        {
            "pid": pid,
            "purpose": "serve" if check_script.classify(argv) == "serve" else "mcp-helper",
            "argv": " ".join(argv),
            "create_time": BORN,
            "boot_id": "boot-1",
        }
        for pid, argv in procs.items()
        if pid != 100
    ]
    (home / "spawn-ledger.json").write_text(json.dumps(ledger))
    return {
        "argv": [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--proc-root",
            str(proc),
            "--hermes-home",
            str(home),
        ],
        "env": {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
    }


def run_cli(host: dict) -> subprocess.CompletedProcess:
    return subprocess.run(host["argv"], env=host["env"], capture_output=True, text=True, timeout=30)


def test_cli_exits_zero_on_a_clean_host(tmp_path: Path) -> None:
    done = run_cli(fake_host(tmp_path, {100: GATEWAY, 101: MCP_HELPER}, "inactive"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines()[-1].startswith("verdict: ok")
    assert "processes: 100=gateway\n" in done.stdout
    assert "ledger other PID 101: live\n" in done.stdout


def test_cli_exits_nonzero_on_a_live_ssh_backend_and_prints_no_secret(tmp_path: Path) -> None:
    done = run_cli(fake_host(tmp_path, {100: GATEWAY, 200: SSH_SERVE}, "active"))
    assert done.returncode == 1, done.stdout + done.stderr
    last = done.stdout.splitlines()[-1]
    assert last.startswith("verdict: second_writer")
    for reason in (
        "live Hermes serve process 200",
        "hermes-autonomous.service is active",
        "Hermes listener :37575 held by PID 200",
        "spawn ledger: live serve PID 200",
    ):
        assert reason in last
    for leak in (SENTINEL, PEER, "/home/", "ssh-session-token-file", "ExecStart"):
        assert leak not in done.stdout + done.stderr


def test_cli_exits_two_when_the_tools_are_missing(tmp_path: Path) -> None:
    host = fake_host(tmp_path, {100: GATEWAY}, "inactive")
    host["env"]["PATH"] = str(tmp_path / "empty")
    done = run_cli(host)
    assert done.returncode == 2
    assert done.stdout.splitlines()[-1].startswith("verdict: unreadable")


def test_access_doc_lists_every_path_with_the_gateway_as_owner() -> None:
    doc = (REPO / "docs" / "hermes-access.md").read_text(encoding="utf-8")
    table = doc.split("## Поддерживаемые пути", 1)[1].split("\n\n", 2)[1]
    owner = "`hermes gateway run` (`hermes-gateway.service`)"
    for path in (
        "Windows Desktop по LAN",
        "Windows Desktop по Tailscale",
        "`hermes` CLI на станции",
        "TUI на станции",
        "Мессенджеры",
    ):
        row = next(line for line in table.splitlines() if line.startswith(f"| {path} |"))
        cells = row.split("|")
        assert cells[3].strip() == owner and "да" in cells[4]
    assert "Ни один путь не запускает второй `serve` и Desktop SSH backend." in doc
    cli = doc.split("### `hermes` CLI на станции", 1)[1].split("\n### ", 1)[0]
    assert "`GatewayAgent`" in cli and "`hermes_cli/cli_agent_setup_mixin.py:641`" in cli
    assert "`--safe-mode`" in cli and "Этот маршрут не второй писатель." in cli
    unsupported = doc.split("## Не поддерживается", 1)[1].split("\n## ", 1)[0]
    ssh = next(line for line in unsupported.splitlines() if line.startswith("| Desktop SSH mode"))
    assert "второй писатель истории чатов" in ssh
    assert "`0.0.0.0:9119`" in doc and "/home/" not in doc
    claude_md = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "`docs/hermes-access.md`, `scripts/session_owner_check.py`" in claude_md


def test_criterion_4_check_runs_only_read_commands_and_changes_no_file(tmp_path: Path) -> None:
    host = fake_host(tmp_path, {100: GATEWAY, 200: SSH_SERVE}, "active")
    calls = tmp_path / "calls.log"
    for tool in (tmp_path / "bin").iterdir():
        shebang, body = tool.read_text().split("\n", 1)
        tool.write_text(f'{shebang}\necho "$(basename "$0") $*" >> {calls}\n{body}')
    watched = [tmp_path / "proc", tmp_path / "home"]

    def snapshot() -> dict:
        return {
            p: (p.read_bytes(), p.stat().st_mtime_ns)
            for root in watched
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    before = snapshot()
    done = run_cli(host)
    assert done.returncode == 1, done.stdout + done.stderr
    assert snapshot() == before
    assert sorted(calls.read_text().splitlines()) == [
        "ss -ltnpH",
        "ss -tnpH state established",
        "systemctl --user show -p Id,ActiveState,UnitFileState,MainPID "
        "hermes-gateway.service hermes-autonomous.service",
    ]


def test_criterion_4_script_has_no_network_and_no_file_write() -> None:
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])
    }
    assert imported <= {"argparse", "json", "os", "re", "subprocess", "sys", "pathlib"}
    forbidden = {"open", "write_text", "write_bytes", "unlink", "mkdir", "touch", "rename"}
    forbidden |= {"replace", "chmod", "rmdir", "kill", "Popen", "system", "socket"}
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not called & forbidden
