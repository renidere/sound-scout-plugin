"""Host side of a GPU reference run (#147): `serve`, `stop`, `state`, `evidence`, `selftest`,
`preflight`, `weights`, `pack`. Stdlib only (`preflight` reads the image's own torch), Python
3.10+, never imports `sound_scout`: the station copies this file and `guard/sitecustomize.py` to
`/workspace/ss/` and checks their SHA-256 there before use.

`serve` starts the vLLM server in a new session with an allowlisted environment, offline
variables and the audit-hook guard on `PYTHONPATH`. Every `POLL_S` it reads the socket inodes of
every process of that session and `/proc/net/{tcp,tcp6,udp,udp6}`: an internet socket with a
non-loopback end kills the whole session with SIGKILL and writes a `violation` line. It also
writes the RSS of the session and the GPU memory to the stats file and stops the server at the
deadline or over the RSS limit. Native sockets of gloo and NCCL are bound to `lo` (#160): a
listener on the bridge address would be a violation. This is a guard against a library that
reaches out by accident, not a sandbox against the host's administrator.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

GUARD_DIR = Path(__file__).resolve().parent / "guard"
ENV_KEEP = ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH")
ENV_PREFIXES = ("CUDA_", "NVIDIA_")
INIT_ENVIRON = Path("/proc/1/environ")
ENV_FIXED = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_HOST_IP": "127.0.0.1",
    "GLOO_SOCKET_IFNAME": "lo",
    "NCCL_SOCKET_IFNAME": "lo",
}
NET_TABLES = ("tcp", "tcp6", "udp", "udp6")
TCP_LISTEN = "0A"
POLL_S = 0.5
STATS_S = 5.0
GRACE_S = 10.0
SUMS_FILE = "SHA256SUMS"
CHUNK = 1 << 20
RSS_FIELDS = {
    "VmRSS:": "rss_bytes",
    "RssAnon:": "rss_anon_bytes",
    "RssFile:": "rss_file_bytes",
    "RssShmem:": "rss_shmem_bytes",
}
EVIDENCE_FILES = ("server.log", "hostrun.log", "guard.log", "server-stats.jsonl")
EVIDENCE_MAX_BYTES = 65536
CMDLINE_MAX = 512


# --- environment and process tree ----------------------------------------------------------


def guarded_env(environ, guard_dir: Path, guard_log: Path | None) -> dict:
    """Only `ENV_KEEP` and the `ENV_PREFIXES` variables survive, plus the guard and offline
    variables; tokens a provider puts into the container environment are dropped."""
    env = {
        name: value
        for name, value in environ.items()
        if name in ENV_KEEP or name.startswith(ENV_PREFIXES)
    }
    env.update(ENV_FIXED)
    env["PYTHONPATH"] = str(guard_dir)
    if guard_log is not None:
        env["SS_GUARD_LOG"] = str(guard_log)
    return env


def session_pids(session: int) -> list[int]:
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            text = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        fields = text.rsplit(")", 1)[1].split()
        if int(fields[3]) == session:
            pids.append(int(entry))
    return pids


def socket_inodes(pids) -> set[int]:
    inodes = set()
    for pid in pids:
        directory = f"/proc/{pid}/fd"
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            try:
                link = os.readlink(f"{directory}/{name}")
            except OSError:
                continue
            if link.startswith("socket:["):
                inodes.add(int(link[8:-1]))
    return inodes


def _address(text: str) -> tuple[str, int]:
    host, port = text.split(":")
    raw = bytes.fromhex(host)
    if len(raw) == 4:
        address = ipaddress.IPv4Address(raw[::-1])
    else:
        address = ipaddress.IPv6Address(b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4)))
    return str(address), int(port, 16)


def parse_net(text: str, table: str) -> list[dict]:
    """Rows of one `/proc/net/<table>` text: inode, local and remote (host, port), state."""
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 10:
            continue
        rows.append(
            {
                "table": table,
                "local": _address(parts[1]),
                "remote": _address(parts[2]),
                "state": parts[3],
                "inode": int(parts[9]),
            }
        )
    return rows


def _loopback(host: str) -> bool:
    address = ipaddress.ip_address(host)
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def violations(rows, inodes) -> list[dict]:
    """Rows of the tree's sockets with a non-loopback local end, or a remote end that is neither
    loopback nor unconnected; a listener off loopback is a non-loopback local end."""
    found = []
    for row in rows:
        if row["inode"] not in inodes:
            continue
        remote_host, remote_port = row["remote"]
        unconnected = ipaddress.ip_address(remote_host).is_unspecified and remote_port == 0
        if not _loopback(row["local"][0]) or not (unconnected or _loopback(remote_host)):
            found.append(row)
    return found


def read_net(root: Path = Path("/proc/net")) -> list[dict]:
    rows = []
    for table in NET_TABLES:
        with contextlib.suppress(OSError):
            rows += parse_net((root / table).read_text(), table)
    return rows


def tree_memory(pids) -> dict:
    """Sums over the session of VmRSS (the limit) and its RssAnon, RssFile, RssShmem parts."""
    totals = dict.fromkeys(RSS_FIELDS.values(), 0)
    for pid in pids:
        with contextlib.suppress(OSError, ValueError):
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in RSS_FIELDS:
                    totals[RSS_FIELDS[parts[0]]] += int(parts[1]) * 1024
    return totals


def gpu_memory_mib() -> int | None:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        return sum(int(line) for line in done.stdout.decode().split())
    except ValueError:
        return None


def _write_line(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps(record, sort_keys=True) + "\n")


def _kill_session(child: subprocess.Popen, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, sig)
    for pid in session_pids(child.pid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)


def monitor(child, *, deadline: float, rss_limit: int, stats: Path, stop: threading.Event):
    """Watch the session of `child` until it exits or a limit ends it; the reason or None."""
    next_stats = 0.0
    while child.poll() is None:
        pids = session_pids(child.pid)
        reason = None
        bad = violations(read_net(), socket_inodes(pids))
        if bad:
            _kill_session(child, signal.SIGKILL)
            _write_line(stats, {"event": "violation", "t": time.time(), "sockets": bad})
            child.wait()
            return "violation"
        memory = tree_memory(pids)
        rss = memory["rss_bytes"]
        if time.monotonic() >= next_stats:
            next_stats = time.monotonic() + STATS_S
            record = {"event": "stats", "t": time.time(), **memory}
            record["gpu_memory_used_mib"] = gpu_memory_mib()
            _write_line(stats, record)
        if rss > rss_limit:
            reason = "rss_limit"
        elif time.time() >= deadline:
            reason = "deadline"
        elif stop.is_set():
            reason = "stopped"
        if reason is not None:
            _kill_session(child, signal.SIGTERM)
            try:
                child.wait(GRACE_S)
            except subprocess.TimeoutExpired:
                _kill_session(child, signal.SIGKILL)
                child.wait()
            return reason
        stop.wait(POLL_S)
    return None


# --- commands ------------------------------------------------------------------------------


def serve(args) -> int:
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    env = guarded_env(os.environ, Path(args.guard_dir), args.guard_log)
    with open(args.log, "ab") as log:
        child = subprocess.Popen(
            args.argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if args.pid_file:
        Path(args.pid_file).write_text(str(os.getpid()))
    _write_line(args.stats, {"event": "started", "t": time.time(), "pid": child.pid})
    reason = monitor(
        child, deadline=args.deadline, rss_limit=args.rss_limit, stats=args.stats, stop=stop
    )
    _kill_session(child, signal.SIGKILL)
    _write_line(
        args.stats,
        {"event": "exit", "t": time.time(), "reason": reason, "returncode": child.returncode},
    )
    return 0 if reason in (None, "stopped") else 1


def _session_survivors(stats: Path | None) -> list[int]:
    """Live processes of the server session `serve` recorded in its `started` line."""
    started = [r for r in _records(stats) if r.get("event") == "started"] if stats else []
    session = started[-1].get("pid") if started else None
    if type(session) is not int:
        return []
    return [pid for pid in session_pids(session) if _alive(pid)]


def stop_command(args) -> int:
    """SIGTERM to `serve`; 0 once it and every process of the server session are gone. A
    survivor of a `serve` that is already gone is killed (nothing else would end it)."""
    try:
        pid = int(Path(args.pid_file).read_text())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        pid = None
    ends = time.monotonic() + args.timeout
    while True:
        if pid is None or not _alive(pid):
            survivors = _session_survivors(args.stats)
            if not survivors:
                return 0
            for survivor in survivors:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(survivor, signal.SIGKILL)
        if time.monotonic() >= ends:
            return 1
        time.sleep(0.2)


def _alive(pid: int) -> bool:
    """A process that exited is gone even while its parent has not reaped it (a zombie)."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return text.rsplit(")", 1)[1].split()[0] != "Z"


def _records(path: Path) -> list[dict]:
    """The lines of a stats file; a torn line being written is skipped."""
    records = []
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return records
    for line in lines:
        with contextlib.suppress(ValueError):
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def serve_state(stats: Path, pid_file: Path) -> dict:
    """Whether `serve` started, is alive, and how it ended; the pid is read before the stats, so
    a `serve` gone between the two reads has already written its `exit` line."""
    try:
        alive = _alive(int(Path(pid_file).read_text()))
    except (OSError, ValueError):
        alive = False
    records = _records(stats)
    exits = [r for r in records if r.get("event") == "exit"]
    violations_ = [r for r in records if r.get("event") == "violation"]
    return {
        "kind": "serve_state",
        "started": any(r.get("event") == "started" for r in records),
        "serve_alive": alive,
        "exit": {key: exits[-1].get(key) for key in ("reason", "returncode", "t")}
        if exits
        else None,
        "violation": violations_[-1].get("sockets") if violations_ else None,
    }


def state_command(args) -> int:
    print(json.dumps(serve_state(args.stats, args.pid_file), sort_keys=True))
    return 0


def tail(path: Path, max_bytes: int) -> dict:
    """The last at most `max_bytes` of a file, from the start of a line."""
    try:
        with open(path, "rb") as source:
            size = os.fstat(source.fileno()).st_size
            source.seek(max(0, size - max_bytes))
            data = source.read(max_bytes)
    except FileNotFoundError:
        return {"missing": True}
    if size > max_bytes:
        newline = data.find(b"\n")
        data = data[newline + 1 :] if newline >= 0 else b""
    return {"size": size, "tail": data.decode("utf-8", errors="replace")}


def processes(proc: Path = Path("/proc")) -> list[dict]:
    """Every process this namespace sees, read from `/proc` without `ps`."""
    found = []
    for entry in sorted(os.listdir(proc), key=lambda name: (len(name), name)):
        if not entry.isdigit():
            continue
        try:
            stat_ = (proc / entry / "stat").read_text()
            raw = (proc / entry / "cmdline").read_bytes()
        except OSError:
            continue
        fields = stat_.rsplit(")", 1)[1].split()
        found.append(
            {
                "pid": int(entry),
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "sid": int(fields[3]),
                "state": fields[0],
                "rss_bytes": tree_memory([int(entry)])["rss_bytes"],
                "cmdline": raw.rstrip(b"\0")
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")[:CMDLINE_MAX],
            }
        )
    return found


def evidence(args) -> int:
    directory = Path(args.dir)
    files = {name: tail(directory / name, args.max_bytes) for name in EVIDENCE_FILES}
    print(
        json.dumps(
            {"kind": "server_evidence", "files": files, "processes": processes()}, sort_keys=True
        )
    )
    return 0


GUARDED_PRELUDE = """
import sys
import sitecustomize
if not getattr(sitecustomize, "GUARD_ACTIVE", False):
    print("unguarded"); sys.exit(3)
"""
LOOPBACK_CHILD = """
import socket
server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1)
socket.create_connection(server.getsockname(), timeout=5).close(); print("allowed")
"""
OUTSIDE_CHILD = """
import socket
try:
    socket.socket().connect(("192.0.2.1", 9))
except PermissionError:
    print("denied")
else:
    print("allowed")
"""
LOOKUP_CHILD = """
import socket
try:
    socket.getaddrinfo("huggingface.co", 443)
except PermissionError:
    print("denied")
else:
    print("allowed")
"""
SPAWN_CHILD = """
import multiprocessing, socket, sys

def probe(queue):
    import sitecustomize
    if not getattr(sitecustomize, "GUARD_ACTIVE", False):
        queue.put("unguarded"); return
    try:
        socket.socket().connect(("192.0.2.1", 9))
    except PermissionError:
        queue.put("denied")
    else:
        queue.put("allowed")

if __name__ == "__main__":
    context = multiprocessing.get_context(sys.argv[1])
    queue = context.Queue()
    process = context.Process(target=probe, args=(queue,))
    process.start(); print(queue.get(timeout=60)); process.join(30)
"""
NATIVE_CHILD = """
import ctypes, socket, struct, sys, time
host, port = sys.argv[1], int(sys.argv[2])
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
address = struct.pack("=H", socket.AF_INET) + struct.pack("!H", port) + socket.inet_aton(host)
buffer = ctypes.create_string_buffer(address + bytes(8), 16)
print("connect", libc.connect(fd, buffer, 16), flush=True)
time.sleep(30)
"""


def own_address(fib_trie: Path = Path("/proc/net/fib_trie")) -> str | None:
    """A local non-loopback IPv4 address of this network namespace, or None."""
    try:
        lines = fib_trie.read_text().splitlines()
    except OSError:
        return None
    for previous, line in zip(lines, lines[1:], strict=False):
        if line.strip() == "/32 host LOCAL":
            host = previous.strip().removeprefix("|-- ").strip()
            with contextlib.suppress(ValueError):
                if not ipaddress.ip_address(host).is_loopback:
                    return host
    return None


def _guarded_child(code: str, env: dict, workdir: Path, *extra: str) -> str:
    script = workdir / f"child-{abs(hash(code))}.py"
    script.write_text(GUARDED_PRELUDE + code)
    done = subprocess.run(
        [sys.executable, str(script), *extra],
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
        cwd=workdir,
    )
    output = done.stdout.decode(errors="replace").strip().splitlines()
    return output[-1] if output else f"exit {done.returncode}"


def _native_check(env: dict, workdir: Path) -> dict:
    host = own_address()
    if host is None:
        return {"status": "unavailable", "reason": "no non-loopback address"}
    listener = socket.socket()
    listener.bind((host, 0))
    listener.listen(1)
    listener.settimeout(10)
    script = workdir / "native-child.py"
    script.write_text(GUARDED_PRELUDE + NATIVE_CHILD)
    port = str(listener.getsockname()[1])
    child = subprocess.Popen(
        [sys.executable, str(script), host, port],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    accepted = None
    with contextlib.suppress(OSError):
        accepted, _ = listener.accept()
    stats = workdir / "native-stats.jsonl"
    reason = monitor(
        child,
        deadline=time.time() + 20,
        rss_limit=1 << 40,
        stats=stats,
        stop=threading.Event(),
    )
    for sock in (accepted, listener):
        if sock is not None:
            sock.close()
    caught = reason == "violation"
    return {"status": "passed" if caught else "failed", "reason": reason}


def selftest(args) -> int:
    with tempfile.TemporaryDirectory() as scratch:
        workdir = Path(scratch)
        env = guarded_env(os.environ, Path(args.guard_dir), workdir / "guard.log")
        checks = {
            "loopback_connect": _guarded_child(LOOPBACK_CHILD, env, workdir) == "allowed",
            "outside_connect_denied": _guarded_child(OUTSIDE_CHILD, env, workdir) == "denied",
            "outside_lookup_denied": _guarded_child(LOOKUP_CHILD, env, workdir) == "denied",
        }
        for method in ("spawn", "fork", "forkserver"):
            denied = _guarded_child(SPAWN_CHILD, env, workdir, method) == "denied"
            checks[f"{method}_child_denied"] = denied
        native = _native_check(env, workdir)
        try:
            netns = subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=10)
            netns_available = netns.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            netns_available = False
    result = {
        "kind": "guard_selftest",
        "checks": checks,
        "native_detector": native,
        "netns_available": netns_available,
        "ok": all(checks.values()) and native["status"] != "failed",
    }
    text = json.dumps(result, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if result["ok"] else 1


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_cuda() -> str | None:
    """`torch.version.cuda`: the CUDA the torch wheel was built for, whatever the session env."""
    try:
        from torch import version
    except Exception:
        return None
    return version.cuda if isinstance(version.cuda, str) else None


def _image_env(name: str, environ: Path = INIT_ENVIRON) -> str | None:
    """A variable of the container's first process, which starts with the image `ENV`."""
    try:
        entries = environ.read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = name.encode() + b"="
    for entry in entries:
        if entry.startswith(prefix):
            return entry[len(prefix) :].decode("utf-8", "replace")
    return None


def preflight(args) -> int:
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "vllm": _version("vllm"),
                "torch": _version("torch"),
                "transformers": _version("transformers"),
                "cuda": _torch_cuda(),
                "cuda_env": os.environ.get("CUDA_VERSION"),
                "cuda_image": _image_env("CUDA_VERSION"),
                "disk_free_bytes": shutil.disk_usage(args.workspace).free,
            },
            sort_keys=True,
        )
    )
    return 0


def _digest(path: Path, kind: str, size: int) -> str:
    digest = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git_oid":
        digest.update(f"blob {size}\0".encode())
    with open(path, "rb") as source:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def weights_receipt(manifest: dict, model: str, directory: Path) -> dict:
    """Size and hash of every manifest file: SHA-256 for LFS files, git blob SHA-1 otherwise."""
    spec = manifest[model]
    files, missing, mismatched = {}, [], []
    expected = set()
    for entry in spec["files"]:
        path = directory / entry["path"]
        expected.add(entry["path"])
        kind = "sha256" if entry.get("lfs_sha256") else "git_oid"
        want = entry["lfs_sha256"] if kind == "sha256" else entry["git_oid"]
        if not path.is_file() or path.is_symlink():
            missing.append(entry["path"])
            continue
        size = path.stat().st_size
        files[entry["path"]] = {"size": size, kind: _digest(path, kind, size)}
        if files[entry["path"]] != {"size": entry["size"], kind: want}:
            mismatched.append(entry["path"])
    extra = sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
        and path.relative_to(directory).parts[0] != ".cache"
        and str(path.relative_to(directory)) not in expected
    )
    return {
        "kind": "weights_receipt",
        "model": model,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "files": files,
        "missing": missing,
        "mismatched": mismatched,
        "extra": extra,
        "ok": not (missing or mismatched or extra),
    }


def weights(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    receipt = weights_receipt(manifest, args.model, Path(args.dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("ok", "missing", "mismatched", "extra")}))
    return 0 if receipt["ok"] else 1


def sha256_file(path: Path) -> str:
    return _digest(path, "sha256", 0)


def pack(args) -> int:
    """`SHA256SUMS` of every regular file under the root (no symlinks), then one tar of both."""
    root, out = Path(args.root), Path(args.out)
    sums = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file() or relative == SUMS_FILE:
            continue
        if path.resolve() == out.resolve():
            continue
        sums.append((relative, sha256_file(path)))
    (root / SUMS_FILE).write_text("".join(f"{digest}  {name}\n" for name, digest in sums))
    with tarfile.open(out, "w") as archive:
        for name, _ in sums:
            archive.add(root / name, arcname=name, recursive=False)
        archive.add(root / SUMS_FILE, arcname=SUMS_FILE, recursive=False)
    print(json.dumps({"archive_sha256": sha256_file(out), "files": len(sums)}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hostrun.py")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_ = commands.add_parser("serve")
    serve_.add_argument("--deadline", type=float, required=True)
    serve_.add_argument("--rss-limit", type=int, required=True)
    serve_.add_argument("--stats", type=Path, required=True)
    serve_.add_argument("--log", type=Path, required=True)
    serve_.add_argument("--guard-log", type=Path)
    serve_.add_argument("--pid-file", type=Path)
    serve_.add_argument("--guard-dir", default=str(GUARD_DIR))
    serve_.add_argument("argv", nargs=argparse.REMAINDER)
    stop_ = commands.add_parser("stop")
    stop_.add_argument("--pid-file", required=True)
    stop_.add_argument("--timeout", type=float, default=60.0)
    stop_.add_argument("--stats", type=Path)
    state_ = commands.add_parser("state")
    state_.add_argument("--stats", type=Path, required=True)
    state_.add_argument("--pid-file", type=Path, required=True)
    evidence_ = commands.add_parser("evidence")
    evidence_.add_argument("--dir", required=True)
    evidence_.add_argument("--max-bytes", type=int, default=EVIDENCE_MAX_BYTES)
    selftest_ = commands.add_parser("selftest")
    selftest_.add_argument("--out")
    selftest_.add_argument("--guard-dir", default=str(GUARD_DIR))
    preflight_ = commands.add_parser("preflight")
    preflight_.add_argument("--workspace", required=True)
    weights_ = commands.add_parser("weights")
    for name in ("--manifest", "--model", "--dir", "--out"):
        weights_.add_argument(name, required=True)
    pack_ = commands.add_parser("pack")
    pack_.add_argument("--root", required=True)
    pack_.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.argv[:1] == ["--"]:
            args.argv = args.argv[1:]
        if not args.argv:
            parser.error("serve needs the server argv after --")
    handlers = {
        "serve": serve,
        "stop": stop_command,
        "state": state_command,
        "evidence": evidence,
        "selftest": selftest,
        "preflight": preflight,
        "weights": weights,
        "pack": pack,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
