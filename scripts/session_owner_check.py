"""#137: read-only check that the Hermes gateway is the only writer of the chat history.

    uv run python -I scripts/session_owner_check.py [--port 9119]

Stdlib only; it changes nothing. It reads `systemctl --user show` of the Hermes units, the TCP
listeners and established connections from `ss`, `<HERMES_HOME>/spawn-ledger.json` and the argv of
the live processes under `/proc`. A second writer is a live Hermes `serve` or `dashboard` (Desktop
SSH backend, `hermes-autonomous.service`), a standalone `tui_gateway.entry`, a Hermes listener or
connection held by a process other than the gateway, or a `serve`/`dashboard` ledger entry whose
process is still alive (docs/hermes-access.md). It prints states, PIDs, ports, counts and fixed
command labels only: never an argv, an environment, a path or a peer address.

Exit 0 — verdict `ok`; 1 — `second_writer`; 2 — `no_owner`, `owner_outside_unit`, `listener_missing`
or `unreadable`.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GATEWAY_UNIT = "hermes-gateway.service"
SERVE_UNIT = "hermes-autonomous.service"
UNIT_FIELDS = ("Id", "ActiveState", "UnitFileState", "MainPID")
WRITER_KINDS = {"serve", "dashboard", "tui_gateway"}
WRITER_PURPOSES = {"serve", "dashboard"}
TIMEOUT = 10
PYTHON_VALUE_OPTIONS = {"-X", "-W", "-Q"}
# Top-level value flags of Hermes v0.23.3: `_VALUE_FLAGS_FALLBACK`, `_OPTIONAL_VALUE_FLAGS_FALLBACK`
# and `PRE_ARGPARSE_INHERITED_FLAGS` in `hermes_cli/_parser.py`, skipped like its `command_argv`.
HERMES_VALUE_OPTIONS = {
    *("-z", "--oneshot", "-m", "--model", "--provider", "--reasoning", "-t", "--toolsets"),
    *("-r", "--resume", "-s", "--skills", "--usage-file", "--in", "--cpu-profile"),
    *("-c", "--continue", "-p", "--profile"),
}
# `_BUILTIN_SUBCOMMANDS` (`hermes_cli/main.py`) and the gateway subparsers: the only printed labels.
SUBCOMMANDS = {
    *("acp", "approvals", "auth", "backup", "bundles", "checkpoints", "codex-runtime"),
    *("completion", "clipboard", "computer-use", "connectors", "config", "console", "cron"),
    *("curator", "dashboard", "serve", "debug", "doctor", "dump", "egress", "fallback"),
    *("gateway", "hooks", "import", "insights", "gui", "desktop", "kanban", "login", "logout"),
    *("logs", "lsp", "mcp", "memory", "migrate", "moa", "journey", "memory-graph", "learning"),
    *("model", "monitoring", "pairing", "pause", "peer", "pets", "plugins", "portal", "profile"),
    *("project", "proxy", "prompt-size", "resume", "send", "sessions", "setup", "skin", "skills"),
    *("slack", "status", "tools", "uninstall", "update", "usage", "vault", "webhook", "whatsapp"),
    *("whatsapp-cloud", "worktree", "chat", "secrets", "security", "browser", "verify", "help"),
}
GATEWAY_SUBCOMMANDS = {
    *("ensure", "attach", "tickets", "start", "stop", "restart", "status", "install"),
    *("uninstall", "list", "setup", "migrate-legacy", "migrate", "enroll"),
}
CLOCK_SLACK = 2.0
PID = re.compile(r"pid=(\d+)")
EXIT = {"ok": 0, "second_writer": 1}


def run(argv: list[str]) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _hermes_args(argv: list[str]) -> list[str] | None:
    """Arguments after the Hermes program, or None when the process is not Hermes."""
    if not argv:
        return None
    name = Path(argv[0]).name
    if name == "hermes":
        return argv[1:]
    if not name.startswith("python"):
        return None
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "-c":
            return argv[i + 2 :] if i + 1 < len(argv) and "hermes" in argv[i + 1] else None
        if token == "-m":
            module = argv[i + 1] if i + 1 < len(argv) else ""
            if module == "tui_gateway.entry":
                return ["tui_gateway"]
            return argv[i + 2 :] if module.split(".")[0] in {"hermes", "hermes_cli"} else None
        if token in PYTHON_VALUE_OPTIONS:
            i += 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        if token.endswith("tui_gateway/entry.py"):
            return ["tui_gateway"]
        is_hermes = Path(token).name == "hermes" or token.endswith("hermes_cli/main.py")
        return argv[i + 1 :] if is_hermes else None
    return None


def _command(args: list[str]) -> list[str]:
    """The subcommand and its arguments, past the top-level flags and their values."""
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            return args[i + 1 :]
        if not token.startswith("-"):
            return args[i:]
        i += 2 if "=" not in token and token in HERMES_VALUE_OPTIONS else 1
    return []


def classify(argv: list[str]) -> str | None:
    """A fixed label of a Hermes process (`gateway` for the gateway itself), else None."""
    args = _hermes_args(argv)
    if args is None:
        return None
    if args == ["tui_gateway"]:
        return "tui_gateway"
    command = _command(args)
    if not command or command[0] not in SUBCOMMANDS:
        return "client"
    if command[0] != "gateway":
        return command[0]
    sub = next(iter(_command(command[1:])), None)
    if sub in {None, "run"}:  # bare `hermes gateway` runs the gateway too
        return "gateway"
    return f"gateway {sub}" if sub in GATEWAY_SUBCOMMANDS else "client"


def processes(proc_root: Path) -> tuple[dict[int, str], set[int]]:
    """Kinds of the live Hermes processes and the PIDs of all live processes."""
    found, alive = {}, set()
    for entry in proc_root.iterdir() if proc_root.is_dir() else ():
        if not entry.name.isdigit():
            continue
        alive.add(int(entry.name))
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        kind = classify([part.decode("utf-8", "replace") for part in raw.split(b"\0") if part])
        if kind is not None:
            found[int(entry.name)] = kind
    return found, alive


def parse_units(text: str) -> dict[str, dict[str, str]]:
    units, current = {}, {}
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if current.get("Id"):
                units[current["Id"]] = current
            current = {}
            continue
        key, _, value = line.partition("=")
        if key in UNIT_FIELDS:
            current[key] = value.strip()
    return units


def _port(address: str) -> int | None:
    tail = address.rpartition(":")[2]
    return int(tail) if tail.isdigit() else None


def parse_sockets(text: str) -> list[tuple[int, int | None, list[int]]]:
    """(local port, peer port, owner PIDs) of each `ss -tnpH` line."""
    rows = []
    for line in text.splitlines():
        addresses = [t for t in line.split() if ":" in t and not t.startswith("users:")]
        if len(addresses) < 2 or (local := _port(addresses[0])) is None:
            continue
        rows.append((local, _port(addresses[1]), [int(p) for p in PID.findall(line)]))
    return rows


def read_ledger(path: Path) -> list[dict] | None:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        data = data.get("entries", [])
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else None


def birth(proc_root: Path, pid: int) -> float | None:
    """Epoch start of `pid` from `/proc/<pid>/stat` and the boot time in `/proc/stat`."""
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()
        btime = next(
            int(line.split()[1])
            for line in (proc_root / "stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        return btime + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def ledger_state(
    entry: dict, alive: set[int], births: dict[int, float | None], boot_id: str
) -> str:
    """`live` when the entry's process still runs, `stale` when it is gone or another process."""
    pid, recorded = entry.get("pid"), entry.get("create_time")
    if not isinstance(pid, int) or pid not in alive:
        return "stale"
    if entry.get("boot_id") and boot_id and entry["boot_id"] != boot_id:
        return "stale"
    started = births.get(pid)
    if type(recorded) not in (int, float) or started is None:
        return "unverified"
    return "live" if abs(started - recorded) <= CLOCK_SLACK else "stale"


def check(
    units_text, listen_text, established_text, ledger, procs, alive, births, boot_id, port
) -> dict:
    lines, writers, problems = [], [], []
    if units_text is None:
        problems.append(("unreadable", "systemctl --user show failed"))
        units = {}
    else:
        units = parse_units(units_text)
    for name in (GATEWAY_UNIT, SERVE_UNIT):
        unit = units.get(name, {})
        state = f"{unit.get('ActiveState', '?')}/{unit.get('UnitFileState', '?')}"
        lines.append(f"unit {name}: {state} MainPID={unit.get('MainPID', '?')}")
    serve_unit = units.get(SERVE_UNIT, {})
    if serve_unit.get("ActiveState") in {"active", "activating", "reloading"}:
        writers.append(f"{SERVE_UNIT} is {serve_unit['ActiveState']}")

    gateway = units.get(GATEWAY_UNIT, {})
    main_pid = int(gateway.get("MainPID", "0") or 0)
    # `gateway.lock` admits one gateway per Hermes home: each live one owns, none is a writer.
    owners = {pid for pid, kind in procs.items() if kind == "gateway"}
    if not owners:
        problems.append(("no_owner", "no live `hermes gateway run`"))
    elif gateway.get("ActiveState") != "active" or main_pid not in owners:
        problems.append(("owner_outside_unit", f"the gateway does not run in {GATEWAY_UNIT}"))
    for pid, kind in sorted(procs.items()):
        if kind in WRITER_KINDS:
            writers.append(f"live Hermes {kind} process {pid}")
    lines.append(
        "processes: "
        + (", ".join(f"{pid}={kind}" for pid, kind in sorted(procs.items())) or "no Hermes process")
    )

    if listen_text is None or established_text is None:
        problems.append(("unreadable", "ss failed"))
        listen_text, established_text = listen_text or "", established_text or ""
    hermes_ports = {}
    for local, _, pids in parse_sockets(listen_text):
        for pid in pids:
            if pid in procs:
                hermes_ports.setdefault(local, set()).add(pid)
    for local, pids in sorted(hermes_ports.items()):
        lines.append(f"listener :{local}: PID {', '.join(map(str, sorted(pids)))}")
        for pid in sorted(pids - owners):
            writers.append(f"Hermes listener :{local} held by PID {pid}, not the gateway")
    if not owners & hermes_ports.get(port, set()):
        problems.append(("listener_missing", f"the gateway does not listen on :{port}"))
    connections = {}
    for local, _, pids in parse_sockets(established_text):
        if local in hermes_ports:
            for pid in pids or [0]:
                connections[(local, pid)] = connections.get((local, pid), 0) + 1
                if pid and pid not in owners:
                    writers.append(f"connection on :{local} held by PID {pid}, not the gateway")
    lines.append(
        "established: "
        + (
            ", ".join(f":{p} PID {pid}: {n}" for (p, pid), n in sorted(connections.items()))
            or "none on Hermes listeners"
        )
    )

    if ledger is None:
        problems.append(("unreadable", "spawn-ledger.json is not a JSON list"))
        ledger = []
    for entry in ledger:
        pid, purpose = entry.get("pid"), entry.get("purpose")
        label = purpose if purpose in WRITER_PURPOSES else "other"
        state = ledger_state(entry, alive, births, boot_id)
        lines.append(f"ledger {label} PID {pid if isinstance(pid, int) else '?'}: {state}")
        if state == "live" and label != "other" and pid not in owners:
            writers.append(f"spawn ledger: live {label} PID {pid}")
    if not ledger:
        lines.append("ledger: no entries")

    if writers:
        verdict, reasons = "second_writer", sorted(set(writers))
    elif problems:
        verdict, reasons = problems[0][0], [why for _, why in problems]
    else:
        verdict, reasons = "ok", [f"gateway PID {main_pid} is the only writer, listens on :{port}"]
    lines.append(f"verdict: {verdict} — {'; '.join(reasons)}")
    return {"verdict": verdict, "lines": lines}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=9119, help="the gateway Webapp port")
    parser.add_argument("--hermes-home", type=Path, default=None)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    args = parser.parse_args(argv)
    home = args.hermes_home or Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    try:
        boot_id = (args.proc_root / "sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = ""
    fields = ",".join(UNIT_FIELDS)
    ledger = read_ledger(home / "spawn-ledger.json")
    procs, alive = processes(args.proc_root)
    births = {
        e["pid"]: birth(args.proc_root, e["pid"])
        for e in ledger or []
        if isinstance(e.get("pid"), int) and e["pid"] in alive
    }
    report = check(
        run(["systemctl", "--user", "show", "-p", fields, GATEWAY_UNIT, SERVE_UNIT]),
        run(["ss", "-ltnpH"]),
        run(["ss", "-tnpH", "state", "established"]),
        ledger,
        procs,
        alive,
        births,
        boot_id,
        args.port,
    )
    print("\n".join(report["lines"]))
    return EXIT.get(report["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
