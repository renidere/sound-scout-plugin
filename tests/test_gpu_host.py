"""Host files of the GPU stack (#147): the audit-hook guard, the `/proc` socket detector, the
allowlisted environment, the weights receipt, `pack` and `serve` limits.

The guard runs here in real child processes on this station; nothing reaches beyond loopback
(each probe first checks that the hook is loaded, and the hook denies before the system call).
This proves the guard and the host commands, not the image they will run in.
"""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from sound_scout.eval import vast
from sound_scout.eval.gpu_host import hostrun

HOST_DIR = vast.HOST_FILES_DIR


def _guarded_python(tmp_path: Path, code: str) -> str:
    env = hostrun.guarded_env(os.environ, HOST_DIR, tmp_path / "guard.log")
    script = tmp_path / "probe.py"
    script.write_text(hostrun.GUARDED_PRELUDE + code)
    done = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, timeout=120, cwd=tmp_path
    )
    return done.stdout.decode().strip().splitlines()[-1]


def test_selftest_allows_loopback_and_denies_outside_lookup_children_and_native(tmp_path) -> None:
    out = tmp_path / "selftest.json"
    done = subprocess.run(
        [sys.executable, str(HOST_DIR / "hostrun.py"), "selftest", "--guard-dir", str(HOST_DIR)]
        + ["--out", str(out)],
        capture_output=True,
        timeout=300,
    )
    result = json.loads(out.read_text())
    assert done.returncode == 0, result
    assert result["checks"] == {
        "loopback_connect": True,
        "outside_connect_denied": True,
        "outside_lookup_denied": True,
        "spawn_child_denied": True,
        "fork_child_denied": True,
        "forkserver_child_denied": True,
    }
    assert result["native_detector"]["status"] in ("passed", "unavailable")
    assert isinstance(result["netns_available"], bool)


def test_the_hook_denies_an_internet_bind_and_logs_it_but_allows_unix_and_loopback(
    tmp_path, short_socket_dir
) -> None:
    code = """
import socket
socket.socket().bind(("127.0.0.1", 0))
socket.socket(socket.AF_UNIX).bind("{unix}")
try:
    socket.socket().bind(("0.0.0.0", 0))
except PermissionError:
    print("denied")
"""
    code = code.replace("{unix}", str(short_socket_dir / "s"))
    assert _guarded_python(tmp_path, code) == "denied"
    line = json.loads((tmp_path / "guard.log").read_text().splitlines()[-1])
    assert line["event"] == "socket.bind" and line["target"] == "'0.0.0.0'"


def test_a_probe_without_the_guard_stops_before_any_socket(tmp_path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(hostrun.GUARDED_PRELUDE + hostrun.OUTSIDE_CHILD)
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(tmp_path)}
    (tmp_path / "sitecustomize.py").write_text("GUARD_ACTIVE = False\n")
    done = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, timeout=60)
    assert done.returncode == 3 and done.stdout.decode().strip() == "unguarded"


TCP = """  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  inode
   0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 101 1
   1: 0100007F:1F40 0100007F:C350 01 00000000:00000000 00:00000000 00000000  1000        0 102 1
   2: 0A00000A:D431 08080808:01BB 01 00000000:00000000 00:00000000 00000000  1000        0 103 1
   3: 00000000:2328 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 104 1
   4: 0A00000A:D432 08080808:01BB 01 00000000:00000000 00:00000000 00000000  1000        0 999 1
"""
TCP6 = """  sl  local_address                         remote_address                        st
   0: 0000000000000000FFFF00000100007F:1F40 0000000000000000FFFF00000100007F:C351 01 0 0 0 0 0 105 1
   1: 00000000000000000000000001000000:1F41 00000000000000000000000000000000:0000 0A 0 0 0 0 0 106 1
   2: 00000000000000000000000000000000:1F42 00000000000000000000000000000000:0000 0A 0 0 0 0 0 107 1
"""


def test_detector_flags_non_loopback_ends_of_the_tree_only() -> None:
    rows = hostrun.parse_net(TCP, "tcp") + hostrun.parse_net(TCP6, "tcp6")
    assert rows[2]["remote"] == ("8.8.8.8", 443) and rows[2]["local"] == ("10.0.0.10", 54321)
    found = hostrun.violations(rows, {101, 102, 103, 104, 105, 106, 107})
    assert sorted(row["inode"] for row in found) == [103, 104, 107]
    assert hostrun.violations(rows, {101, 102, 105, 106}) == []


def test_own_address_reads_a_local_non_loopback_entry(tmp_path) -> None:
    trie = tmp_path / "fib_trie"
    trie.write_text(
        "Main:\n  +-- 0.0.0.0/0 3 0 5\n     |-- 127.0.0.1\n        /32 host LOCAL\n"
        "     |-- 192.168.7.20\n        /32 host LOCAL\n"
    )
    assert hostrun.own_address(trie) == "192.168.7.20"
    trie.write_text("Main:\n     |-- 127.0.0.1\n        /32 host LOCAL\n")
    assert hostrun.own_address(trie) is None


def test_guarded_env_keeps_the_allowlist_and_drops_tokens(tmp_path) -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "HF_TOKEN": "hf_x",
        "VAST_CONTAINERLABEL": "C.1",
        "CONTAINER_API_KEY": "secret",
        "PYTHONPATH": "/elsewhere",
    }
    env = hostrun.guarded_env(environ, tmp_path / "guard", tmp_path / "g.log")
    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        **hostrun.ENV_FIXED,
        "PYTHONPATH": str(tmp_path / "guard"),
        "SS_GUARD_LOG": str(tmp_path / "g.log"),
    }


def _manifest(tmp_path: Path) -> tuple[dict, Path]:
    models = vast.vast_fake.synthetic_models()
    directory = tmp_path / "m"
    spec = models["qwen25omni"]
    vast.vast_fake.hf_download(
        ["download", spec["repo"], "--revision", spec["revision"], "--local-dir", str(directory)]
    )
    return models, directory


def test_weights_receipt_matches_the_manifest_and_the_expected_files(tmp_path) -> None:
    models, directory = _manifest(tmp_path)
    (directory / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
    (directory / ".cache" / "huggingface" / "x.metadata").write_text("hf bookkeeping")
    receipt = hostrun.weights_receipt(models, "qwen25omni", directory)
    assert receipt["ok"] is True
    assert receipt["files"] == vast.expected_files(models, "qwen25omni")


def test_weights_receipt_names_a_changed_missing_or_extra_file(tmp_path) -> None:
    models, directory = _manifest(tmp_path)
    (directory / "config.json").write_bytes(b"changed")
    (directory / "model-00001-of-00001.safetensors").unlink()
    (directory / "spk_dict.pt").write_bytes(b"pickle")
    receipt = hostrun.weights_receipt(models, "qwen25omni", directory)
    assert receipt["ok"] is False
    assert receipt["mismatched"] == ["config.json"]
    assert receipt["missing"] == ["model-00001-of-00001.safetensors"]
    assert receipt["extra"] == ["spk_dict.pt"]


def test_pack_writes_sums_of_regular_files_and_the_archive_passes_the_station_check(
    tmp_path, capsys
) -> None:
    root = tmp_path / "out"
    (root / "qwen25omni").mkdir(parents=True)
    (root / "guard-selftest.json").write_text("{}")
    (root / "qwen25omni" / "server.log").write_text("log")
    (root / "link").symlink_to(root / "guard-selftest.json")
    archive = tmp_path / "results.tar"
    assert hostrun.main(["pack", "--root", str(root), "--out", str(archive)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed.pop("disk_free_bytes") > 0
    assert printed == {"archive_sha256": vast._sha256(archive), "files": 2}
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == [
            "SHA256SUMS",
            "guard-selftest.json",
            "qwen25omni/server.log",
        ]
    assert vast.verify_archive(archive, tmp_path / "host") == 2
    assert not (root / "SHA256SUMS").exists()


def test_pack_of_a_log_another_process_keeps_appending_to_still_matches_its_sums(
    tmp_path,
) -> None:
    """A sync packs `out/` while the live server writes its logs (#149): each file is read
    once, so the archived bytes are the hashed bytes, and the station check passes."""
    root = tmp_path / "out"
    root.mkdir()
    log = root / "server.log"
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "with open(sys.argv[1], 'a') as f:\n"
            "    for _ in range(40000):\n"
            "        f.write('x' * 255 + '\\n'); f.flush(); time.sleep(0.0002)\n",
            str(log),
        ]
    )
    try:
        time.sleep(0.3)
        for number in range(4):
            archive = tmp_path / f"sync-{number}.tar"
            argv = [sys.executable, str(HOST_DIR / "hostrun.py"), "pack", "--root", str(root)]
            done = subprocess.run([*argv, "--out", str(archive)], capture_output=True, timeout=60)
            assert done.returncode == 0, done.stderr
            assert vast.verify_archive(archive, tmp_path / f"host-{number}") == 1
        assert writer.poll() is None, "the writer must still run while the packs are made"
    finally:
        writer.kill()
        writer.wait()


def test_size_counts_partial_downloads_and_reads_the_exit_code(tmp_path, capsys) -> None:
    model = tmp_path / "models" / "m"
    (model / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (model / "config.json").write_bytes(b"x" * 10)
    (model / ".cache" / "huggingface" / "download" / "a.incomplete").write_bytes(b"y" * 5)
    rc = tmp_path / "download.rc"
    assert hostrun.main(["size", "--dir", str(model), "--rc-file", str(rc)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert (first["bytes"], first["files"], first["rc"]) == (15, 2, None)
    assert first["disk_free_bytes"] > 0
    rc.write_text("3\n")
    hostrun.main(["size", "--dir", str(tmp_path / "models" / "absent"), "--rc-file", str(rc)])
    second = json.loads(capsys.readouterr().out)
    assert (second["bytes"], second["rc"]) == (0, 3)


def _serve(tmp_path: Path, *extra: str, child: str) -> tuple[int, list[dict]]:
    stats = tmp_path / "stats.jsonl"
    argv = [sys.executable, str(HOST_DIR / "hostrun.py"), "serve", "--guard-dir", str(HOST_DIR)]
    argv += ["--stats", str(stats), "--log", str(tmp_path / "server.log"), *extra]
    argv += ["--", sys.executable, "-c", child]
    done = subprocess.run(argv, capture_output=True, timeout=120)
    return done.returncode, [json.loads(line) for line in stats.read_text().splitlines()]


def test_serve_kills_the_server_at_the_rss_limit(tmp_path) -> None:
    code, lines = _serve(
        tmp_path,
        "--deadline",
        str(time.time() + 60),
        "--rss-limit",
        "1000",
        child="import time; time.sleep(60)",
    )
    assert code == 1 and lines[-1]["event"] == "exit" and lines[-1]["reason"] == "rss_limit"
    peak = lines[-1]["peak"]
    assert peak["rss_bytes"] > 1000 and set(peak) == set(hostrun.RSS_FIELDS.values())


@pytest.mark.slow
def test_serve_kills_the_server_at_the_deadline(tmp_path) -> None:
    code, lines = _serve(
        tmp_path,
        "--deadline",
        str(time.time() + 1),
        "--rss-limit",
        str(1 << 40),
        child="import time; time.sleep(60)",
    )
    assert code == 1 and lines[-1]["reason"] == "deadline"


def test_serve_runs_the_server_under_the_guard(tmp_path) -> None:
    child = hostrun.GUARDED_PRELUDE + (
        "import socket\n"
        "try:\n    socket.getaddrinfo('huggingface.co', 443)\n"
        "except PermissionError:\n    print('denied', flush=True)\n"
    )
    code, lines = _serve(
        tmp_path, "--deadline", str(time.time() + 60), "--rss-limit", str(1 << 40), child=child
    )
    assert code == 0 and lines[-1]["reason"] is None
    assert (tmp_path / "server.log").read_text().strip() == "denied"


@pytest.mark.parametrize("name", ["hostrun.py", "sitecustomize.py"])
def test_host_files_import_nothing_of_sound_scout(name) -> None:
    text = (HOST_DIR / name).read_text()
    assert "import sound_scout" not in text and "from sound_scout" not in text


def test_image_env_reads_the_first_process_environment_not_the_session(tmp_path) -> None:
    environ = tmp_path / "environ"
    environ.write_bytes(b"PATH=/usr/bin\0CUDA_VERSION=13.0.2\0NVIDIA_VISIBLE_DEVICES=all\0")
    assert hostrun._image_env("CUDA_VERSION", environ) == "13.0.2"
    assert hostrun._image_env("CUDA_HOME", environ) is None
    assert hostrun._image_env("CUDA_VERSION", tmp_path / "missing") is None


def test_preflight_takes_cuda_from_torch_and_only_records_the_session_variable(tmp_path) -> None:
    site = tmp_path / "site"
    (site / "torch").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text("")
    (site / "torch" / "version.py").write_text("cuda = '13.0'\n")
    env = {**os.environ, "PYTHONPATH": str(site), "CUDA_VERSION": "13.0.3"}
    done = subprocess.run(
        [sys.executable, str(HOST_DIR / "hostrun.py"), "preflight", "--workspace", str(tmp_path)],
        capture_output=True,
        env=env,
        check=True,
    )
    found = json.loads(done.stdout)
    assert found["cuda"] == "13.0" and found["cuda_env"] == "13.0.3"
    assert found["disk_free_bytes"] > 0
    (site / "torch" / "version.py").write_text("raise ImportError('no torch')\n")
    done = subprocess.run(
        [sys.executable, str(HOST_DIR / "hostrun.py"), "preflight", "--workspace", str(tmp_path)],
        capture_output=True,
        env=env,
        check=True,
    )
    assert json.loads(done.stdout)["cuda"] is None


def test_guarded_env_binds_gloo_and_nccl_to_loopback(tmp_path) -> None:
    env = hostrun.guarded_env(
        {"PATH": "/usr/bin", "GLOO_SOCKET_IFNAME": "eth0"}, tmp_path / "guard", None
    )
    assert env["GLOO_SOCKET_IFNAME"] == "lo" and env["NCCL_SOCKET_IFNAME"] == "lo"
    assert env["VLLM_HOST_IP"] == "127.0.0.1"


# A server in the shape of `vllm serve`: it starts one worker by the method in argv, waits for the
# worker's report over loopback TCP and answers `/health` 200 with it only then.
WORKER_SERVER = """
import json, multiprocessing, os, socket, subprocess, sys, threading, time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def probe(port, unix_path):
    import sitecustomize
    results = {"guarded": getattr(sitecustomize, "GUARD_ACTIVE", False)}
    with socket.socket(socket.AF_UNIX) as unix:
        unix.connect(unix_path); unix.sendall(b"ping"); results["unix"] = unix.recv(4) == b"pong"
    for name, call in (
        ("outside_connect", lambda: socket.socket().connect(("192.0.2.1", 9))),
        ("outside_lookup", lambda: socket.getaddrinfo("huggingface.co", 443)),
        ("any_bind", lambda: socket.socket().bind(("", 0))),
    ):
        try:
            call()
        except PermissionError:
            results[name + "_denied"] = True
        else:
            results[name + "_denied"] = False
    report = json.dumps({"pid": os.getpid(), "results": results}).encode()
    urllib.request.urlopen(f"http://127.0.0.1:{port}/report", report, timeout=30).read()
    while True:
        time.sleep(60)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        probe(int(sys.argv[2]), sys.argv[3]); sys.exit(0)
    method, workdir, socket_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    reports = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            reports.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

        def do_GET(self):
            body = json.dumps(reports[0]).encode() if reports else b""
            self.send_response(200 if reports else 503)
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    unix_path = os.path.join(socket_dir, "worker.sock")
    unix = socket.socket(socket.AF_UNIX); unix.bind(unix_path); unix.listen(1)
    if method == "exec":
        worker = subprocess.Popen([sys.executable, __file__, "worker", str(port), unix_path])
    else:
        context = multiprocessing.get_context(method)
        context.Process(target=probe, args=(port, unix_path)).start()

    def answer():
        connection, _ = unix.accept()
        if connection.recv(4) == b"ping":
            connection.sendall(b"pong")
        connection.close()

    threading.Thread(target=answer, daemon=True).start()
    with open(os.path.join(workdir, "port"), "w") as out:
        out.write(str(port))
    server.serve_forever()
"""


def _start_serve(tmp_path: Path, argv: list[str]) -> subprocess.Popen:
    command = [sys.executable, str(HOST_DIR / "hostrun.py"), "serve", "--guard-dir", str(HOST_DIR)]
    command += ["--deadline", str(time.time() + 120), "--rss-limit", str(1 << 40)]
    command += ["--stats", str(tmp_path / "stats.jsonl"), "--log", str(tmp_path / "server.log")]
    command += ["--guard-log", str(tmp_path / "guard.log"), "--pid-file", str(tmp_path / "pid")]
    return subprocess.Popen([*command, "--", *argv], cwd=tmp_path)


def _wait_for(condition, seconds: float = 60.0):
    ends = time.monotonic() + seconds
    while time.monotonic() < ends:
        value = condition()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def _stats(tmp_path: Path) -> list[dict]:
    return hostrun._records(tmp_path / "stats.jsonl")


def _health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as answer:
            return json.loads(answer.read())
    except OSError:
        return None


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver", "exec"])
def test_a_guarded_server_starts_workers_that_stay_guarded(
    tmp_path, short_socket_dir, method
) -> None:
    script = tmp_path / "server.py"
    script.write_text(hostrun.GUARDED_PRELUDE + WORKER_SERVER)
    argv = [sys.executable, str(script), method, str(tmp_path), str(short_socket_dir)]
    serve = _start_serve(tmp_path, argv)
    try:
        port = int(
            _wait_for(lambda: (tmp_path / "port").is_file() and (tmp_path / "port").read_text())
        )
        report = _wait_for(lambda: _health(port))
        assert report["results"] == {
            "guarded": True,
            "unix": True,
            "outside_connect_denied": True,
            "outside_lookup_denied": True,
            "any_bind_denied": True,
        }
        server_pid = next(r["pid"] for r in _stats(tmp_path) if r["event"] == "started")
        worker = report["pid"]
        assert worker != server_pid and hostrun._alive(worker)
        assert worker in hostrun.session_pids(server_pid)
        denials = [json.loads(line) for line in (tmp_path / "guard.log").read_text().splitlines()]
        assert {d["event"] for d in denials if d["pid"] == worker} == {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.bind",
        }
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        subprocess.run([*stop, str(tmp_path / "pid"), "--timeout", "60"], check=True, timeout=90)
        assert serve.wait(30) == 0
        last = _stats(tmp_path)[-1]
        assert last["event"] == "exit" and last["reason"] == "stopped"
        assert not hostrun._alive(worker) and hostrun.session_pids(server_pid) == []
    finally:
        if serve.poll() is None:
            serve.send_signal(signal.SIGTERM)
            serve.wait(30)


SESSION_SERVER = """
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
time.sleep(120)
"""


def test_stop_ends_only_when_no_process_of_the_server_session_survives(tmp_path) -> None:
    script = tmp_path / "server.py"
    script.write_text(SESSION_SERVER)
    serve = _start_serve(tmp_path, [sys.executable, str(script)])
    server_pid = None
    try:
        server_pid = _wait_for(
            lambda: next((r["pid"] for r in _stats(tmp_path) if r["event"] == "started"), None)
        )
        _wait_for(
            lambda: len([p for p in hostrun.session_pids(server_pid) if hostrun._alive(p)]) == 2
        )
        serve.kill()
        serve.wait(30)
        assert [p for p in hostrun.session_pids(server_pid) if hostrun._alive(p)]
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        stop += [str(tmp_path / "pid"), "--stats", str(tmp_path / "stats.jsonl")]
        done = subprocess.run([*stop, "--timeout", "30"], timeout=60)
        assert done.returncode == 0
        assert not [p for p in hostrun.session_pids(server_pid) if hostrun._alive(p)]
    finally:
        if serve.poll() is None:
            serve.kill()
            serve.wait(30)
        if server_pid is not None:
            for pid in hostrun.session_pids(server_pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


def test_stop_fails_while_a_process_of_the_server_session_survives(tmp_path, monkeypatch) -> None:
    stats = tmp_path / "stats.jsonl"
    started = {"event": "started", "t": 0, "pid": 4242, "start": 7, "ticks": 8}
    stats.write_text(json.dumps(started) + "\n")
    monkeypatch.setattr(hostrun, "session_pids", lambda session: [4242] if session == 4242 else [])
    monkeypatch.setattr(hostrun, "_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(hostrun, "start_ticks", lambda pid: 7 if pid == 4242 else None)
    killed = []
    monkeypatch.setattr(hostrun.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    args = ["stop", "--pid-file", str(tmp_path / "absent.pid"), "--stats", str(stats)]
    assert hostrun.main([*args, "--timeout", "0.5"]) == 1
    assert killed and set(killed) == {(4242, signal.SIGKILL)}
    assert hostrun.main(["stop", "--pid-file", str(tmp_path / "absent.pid"), "--timeout", "0"]) == 0


def _fake_gone_leader(monkeypatch, members) -> list:
    """A recorded session 4242 whose leader is gone; `members()` gives its live (pid, start) at
    each pass of `stop`. The SIGKILLs `stop` sends are recorded and land nowhere."""
    monkeypatch.setattr(hostrun, "session_pids", lambda session: [p for p, _ in members()])
    monkeypatch.setattr(hostrun, "_alive", lambda pid: True)
    monkeypatch.setattr(hostrun, "start_ticks", lambda pid: dict(members()).get(pid))
    killed = []
    monkeypatch.setattr(hostrun.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    return killed


@pytest.mark.parametrize(
    ("lines", "start", "proved"),
    [
        ([{"event": "stats", "ticks": 9}], 8, True),
        ([{"event": "stats", "ticks": 9}, {"event": "exit", "ticks": 20}], 15, True),
        ([{"event": "stats", "ticks": 9}], 9, False),
        ([{"event": "stats", "ticks": 9}], 15, False),
    ],
    ids=["before a stats tick", "before the exit tick", "at the last tick", "after the last tick"],
)
def test_stop_kills_a_survivor_of_a_gone_leader_only_with_proof_and_never_reports_success(
    tmp_path, monkeypatch, lines, start, proved
) -> None:
    stats = tmp_path / "stats.jsonl"
    started = {"event": "started", "t": 0, "pid": 4242, "start": 7, "ticks": 8}
    stats.write_text("".join(json.dumps(line) + "\n" for line in [started, *lines]))
    killed = _fake_gone_leader(monkeypatch, lambda: [(5000, start)])
    args = ["stop", "--pid-file", str(tmp_path / "absent.pid"), "--stats", str(stats)]
    assert hostrun.main([*args, "--timeout", "0.3"]) == 1
    assert set(killed) == ({(5000, signal.SIGKILL)} if proved else set())


def test_stop_keeps_killing_a_survivor_it_proved_once_after_the_proof_is_gone(
    tmp_path, monkeypatch
) -> None:
    stats = tmp_path / "stats.jsonl"
    started = {"event": "started", "t": 0, "pid": 4242, "start": 7, "ticks": 8}
    stats.write_text(json.dumps(started) + "\n")
    passes = iter([[(5000, 7), (5001, 15)], [(5001, 15)], [(5001, 15)], []])
    current: list = []

    def members():
        return current

    killed = _fake_gone_leader(monkeypatch, members)
    real_survivors = hostrun._session_survivors

    def next_pass(stats_path, accepted):
        current[:] = next(passes)
        return real_survivors(stats_path, accepted)

    monkeypatch.setattr(hostrun, "_session_survivors", next_pass)
    args = ["stop", "--pid-file", str(tmp_path / "absent.pid"), "--stats", str(stats)]
    assert hostrun.main([*args, "--timeout", "30"]) == 0
    assert killed == [(5000, signal.SIGKILL), *[(5001, signal.SIGKILL)] * 3]


LEADER_THAT_LEAVES = """
import os, subprocess, sys, time
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
while not os.path.exists(sys.argv[1]):
    time.sleep(0.05)
"""


@pytest.mark.slow
def test_stop_kills_the_survivors_of_a_server_session_whose_leader_is_gone(tmp_path) -> None:
    script, leave = tmp_path / "server.py", tmp_path / "leave"
    script.write_text(LEADER_THAT_LEAVES)
    serve = _start_serve(tmp_path, [sys.executable, str(script), str(leave)])
    server_pid = None
    try:
        server_pid = _wait_for(
            lambda: next((r["pid"] for r in _stats(tmp_path) if r["event"] == "started"), None)
        )
        (worker,) = _wait_for(
            lambda: [p for p in hostrun.session_pids(server_pid) if p != server_pid]
        )
        worker_start = hostrun.start_ticks(worker)
        _wait_for(
            lambda: any(r.get("ticks", 0) > worker_start for r in _stats(tmp_path)), seconds=30
        )
        serve.kill()
        serve.wait(30)
        leave.touch()
        _wait_for(lambda: hostrun.start_ticks(server_pid) is None, seconds=30)
        assert hostrun._alive(worker)
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        stop += [str(tmp_path / "pid"), "--stats", str(tmp_path / "stats.jsonl")]
        assert subprocess.run([*stop, "--timeout", "30"], timeout=60).returncode == 0
        assert not hostrun._alive(worker)
    finally:
        leave.touch()
        if serve.poll() is None:
            serve.kill()
            serve.wait(30)
        if server_pid is not None:
            for pid in hostrun.session_pids(server_pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


# A leader that starts a worker and leaves once `serve` is gone: no line of `serve` comes after
# the worker started, so nothing proves the worker is of the recorded session.
LEADER_THAT_FORKS_AFTER_SERVE = """
import os, subprocess, sys, time
go, ready = sys.argv[1], sys.argv[2]
while not os.path.exists(go):
    time.sleep(0.05)
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(ready, "w") as out:
    out.write(str(worker.pid))
"""


@pytest.mark.slow
def test_stop_fails_without_a_signal_on_a_survivor_it_cannot_prove_of_the_recorded_session(
    tmp_path,
) -> None:
    script, go, ready = tmp_path / "server.py", tmp_path / "go", tmp_path / "ready"
    script.write_text(LEADER_THAT_FORKS_AFTER_SERVE)
    serve = _start_serve(tmp_path, [sys.executable, str(script), str(go), str(ready)])
    server_pid = None
    try:
        server_pid = _wait_for(
            lambda: next((r["pid"] for r in _stats(tmp_path) if r["event"] == "started"), None)
        )
        _wait_for(lambda: [r for r in _stats(tmp_path) if r["event"] == "stats"])
        serve.kill()
        serve.wait(30)
        go.touch()
        worker = int(_wait_for(lambda: ready.is_file() and ready.read_text()))
        _wait_for(lambda: hostrun.start_ticks(server_pid) is None, seconds=30)
        worker_start = hostrun.start_ticks(worker)
        assert worker_start > max(r["ticks"] for r in _stats(tmp_path))
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        stop += [str(tmp_path / "pid"), "--stats", str(tmp_path / "stats.jsonl")]
        assert subprocess.run([*stop, "--timeout", "1"], timeout=60).returncode == 1
        assert hostrun._alive(worker) and hostrun.start_ticks(worker) == worker_start
    finally:
        go.touch()
        if serve.poll() is None:
            serve.kill()
            serve.wait(30)
        if server_pid is not None:
            for pid in hostrun.session_pids(server_pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


# A vLLM-like leader that dies first on SIGTERM, with two workers started after a stats line:
# one in the leader's process group, one in a process group of its own that ignores SIGTERM.
LEADER_WITH_LATE_WORKERS = """
import os, subprocess, sys, time
go, ready = sys.argv[1], sys.argv[2]
while not os.path.exists(go):
    time.sleep(0.05)
deaf = "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
deaf += "open(sys.argv[1], 'w').close(); time.sleep(120)"
own_group = subprocess.Popen([sys.executable, "-c", deaf, ready + ".deaf"], process_group=0)
same_group = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
while not os.path.exists(ready + ".deaf"):
    time.sleep(0.05)
with open(ready, "w") as out:
    out.write(f"{own_group.pid} {same_group.pid}")
time.sleep(120)
"""


def test_stop_still_kills_every_live_process_of_the_recorded_server_session(tmp_path) -> None:
    script, go, ready = tmp_path / "server.py", tmp_path / "go", tmp_path / "ready"
    script.write_text(LEADER_WITH_LATE_WORKERS)
    serve = _start_serve(tmp_path, [sys.executable, str(script), str(go), str(ready)])
    server_pid = None
    try:
        server_pid = _wait_for(
            lambda: next((r["pid"] for r in _stats(tmp_path) if r["event"] == "started"), None)
        )
        _wait_for(lambda: [r for r in _stats(tmp_path) if r["event"] == "stats"])
        go.touch()
        workers = [int(p) for p in _wait_for(lambda: ready.is_file() and ready.read_text()).split()]
        assert all(pid in hostrun.session_pids(server_pid) for pid in workers)
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        stop += [str(tmp_path / "pid"), "--stats", str(tmp_path / "stats.jsonl")]
        assert subprocess.run([*stop, "--timeout", "30"], timeout=60).returncode == 0
        assert serve.wait(30) == 0
        assert not [p for p in hostrun.session_pids(server_pid) if hostrun._alive(p)]
        assert not any(hostrun._alive(pid) for pid in workers)
    finally:
        if serve.poll() is None:
            serve.kill()
            serve.wait(30)
        if server_pid is not None:
            for pid in hostrun.session_pids(server_pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


def _start_ticks(pid: int) -> int | None:
    """Field 22 of `/proc/<pid>/stat`, read here and not by `hostrun`."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except OSError:
        return None


# A new session that took the recorded session id: its leader stays, or it forks and leaves the
# way a daemon does. The recorded leader started a tick before it; `serve` last saw that leader
# no later than the new session started. With the leader gone nothing proves the survivor is of
# the new session either: it gets no signal, and `stop` fails.
DAEMON = """
import os, time
if os.fork() == 0:
    time.sleep(60)
os._exit(0)
"""


@pytest.mark.parametrize("leader", ["alive", pytest.param("gone", marks=pytest.mark.slow)])
def test_stop_leaves_a_new_session_that_took_the_recorded_session_id(tmp_path, leader) -> None:
    code = "import time; time.sleep(60)" if leader == "alive" else DAEMON
    first = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    session = first.pid
    try:
        if leader == "gone":
            first.wait(30)
        (stranger,) = _wait_for(
            lambda: [p for p in hostrun.session_pids(session) if hostrun._alive(p)]
        )
        start = _start_ticks(stranger)
        stats = tmp_path / "stats.jsonl"
        started = {"event": "started", "t": 0, "pid": session, "start": start - 1}
        stats.write_text(json.dumps({**started, "ticks": start}) + "\n")
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        stop += [str(tmp_path / "absent.pid"), "--stats", str(stats), "--timeout", "1"]
        assert subprocess.run(stop, timeout=60).returncode == (0 if leader == "alive" else 1)
        assert hostrun._alive(stranger) and _start_ticks(stranger) == start
    finally:
        for pid in hostrun.session_pids(session):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        if first.poll() is None:
            first.wait(30)


@pytest.mark.parametrize("recorded", ["pid", "pid and an earlier start"])
def test_stop_sends_no_sigterm_to_a_process_that_took_the_pid_of_serve(tmp_path, recorded) -> None:
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        start = _wait_for(lambda: _start_ticks(stranger.pid))
        pid_file = tmp_path / "serve.pid"
        pid_file.write_text(f"{stranger.pid} {start - 1}" if " " in recorded else str(stranger.pid))
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        assert subprocess.run([*stop, str(pid_file), "--timeout", "5"], timeout=60).returncode == 0
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait(30)


def test_serve_records_its_own_and_the_leader_start_for_stop(tmp_path) -> None:
    serve = _start_serve(tmp_path, [sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        started = _wait_for(
            lambda: next((r for r in _stats(tmp_path) if r["event"] == "started"), None)
        )
        assert started["start"] == hostrun.start_ticks(started["pid"]) <= started["ticks"]
        pid_file = tmp_path / "pid"
        _wait_for(lambda: pid_file.is_file() and pid_file.read_text())
        assert pid_file.read_text() == f"{serve.pid} {hostrun.start_ticks(serve.pid)}"
        assert hostrun.serve_state(tmp_path / "stats.jsonl", pid_file)["serve_alive"] is True
        stats = _wait_for(lambda: [r for r in _stats(tmp_path) if r["event"] == "stats"])
        assert stats[0]["ticks"] >= started["ticks"]
        serve.send_signal(signal.SIGTERM)
        assert serve.wait(30) == 0
        assert hostrun.serve_state(tmp_path / "stats.jsonl", pid_file)["serve_alive"] is False
        records = _stats(tmp_path)
        assert records[-1]["event"] == "exit"
        assert records[-1]["ticks"] >= max(r["ticks"] for r in records[:-1])
    finally:
        if serve.poll() is None:
            serve.send_signal(signal.SIGTERM)
            serve.wait(30)


# `serve` whose first wait of the poll loop holds until a signal comes: in `Condition.wait` (the
# lock of the `Event` held) or in `time.sleep`, whichever the loop waits in; the file `argv[2]`
# says the wait holds. The SIGTERM of `stop` is the only one, so it lands inside that wait.
SIGNAL_IN_WAIT = """
import os, signal, sys, threading, time
sys.path.insert(0, sys.argv[1])
import hostrun

in_wait, received = sys.argv[2], []
install, condition_wait, sleep = signal.signal, threading.Condition.wait, time.sleep


def recording_signal(sig, handler):
    def record(*args):
        received.append(sig)
        handler(*args)

    return install(sig, record)


def hold_for_signal():
    if not os.path.exists(in_wait):
        open(in_wait, "w").close()
        ends = time.monotonic() + 60
        while not received and time.monotonic() < ends:
            sleep(0.01)


def held_condition_wait(self, timeout=None):
    hold_for_signal()
    return condition_wait(self, timeout)


def held_sleep(seconds):
    hold_for_signal()
    sleep(seconds)


signal.signal = recording_signal
threading.Condition.wait, time.sleep = held_condition_wait, held_sleep
hostrun.gpu_memory_mib = lambda: None  # an nvidia-smi run would hold in its own wait
sys.exit(hostrun.main(sys.argv[3:]))
"""


def test_serve_stops_on_a_sigterm_that_lands_inside_its_poll_wait(tmp_path) -> None:
    """#166: the handler set a `threading.Event`, which takes the lock the poll loop's
    `Event.wait` holds, so a SIGTERM there deadlocked `serve` and `stop` returned 1."""
    wrapper = tmp_path / "serve.py"
    wrapper.write_text(SIGNAL_IN_WAIT)
    in_wait = tmp_path / "in-wait"
    command = [sys.executable, str(wrapper), str(HOST_DIR), str(in_wait), "serve"]
    command += ["--guard-dir", str(HOST_DIR)]
    command += ["--deadline", str(time.time() + 120), "--rss-limit", str(1 << 40)]
    command += ["--stats", str(tmp_path / "stats.jsonl"), "--log", str(tmp_path / "server.log")]
    command += ["--pid-file", str(tmp_path / "pid"), "--"]
    serve = subprocess.Popen([*command, sys.executable, "-c", "import time; time.sleep(120)"])
    server_pid = None
    try:
        server_pid = _wait_for(
            lambda: next((r["pid"] for r in _stats(tmp_path) if r["event"] == "started"), None)
        )
        _wait_for(lambda: (tmp_path / "pid").is_file() and in_wait.is_file())
        stop = [sys.executable, str(HOST_DIR / "hostrun.py"), "stop", "--pid-file"]
        stop += [str(tmp_path / "pid"), "--stats", str(tmp_path / "stats.jsonl")]
        assert subprocess.run([*stop, "--timeout", "20"], timeout=60).returncode == 0
        assert serve.wait(30) == 0
        last = _stats(tmp_path)[-1]
        assert last["event"] == "exit" and last["reason"] == "stopped"
        assert not [p for p in hostrun.session_pids(server_pid) if hostrun._alive(p)]
    finally:
        if serve.poll() is None:
            serve.kill()
            serve.wait(30)
        if server_pid is not None:
            for pid in hostrun.session_pids(server_pid):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


NATIVE_SERVER = """
import multiprocessing, sys, time


def native(host, port):
    import ctypes, socket, struct
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    address = struct.pack("=H", socket.AF_INET) + struct.pack("!H", port) + socket.inet_aton(host)
    libc.connect(fd, ctypes.create_string_buffer(address + bytes(8), 16), 16)
    time.sleep(60)


if __name__ == "__main__":
    multiprocessing.get_context("fork").Process(
        target=native, args=(sys.argv[1], int(sys.argv[2]))
    ).start()
    time.sleep(60)
"""


def test_a_worker_native_socket_off_loopback_kills_the_whole_session(tmp_path) -> None:
    host = hostrun.own_address()
    if host is None:
        pytest.skip("this network namespace has no non-loopback IPv4 address")
    script = tmp_path / "server.py"
    script.write_text(hostrun.GUARDED_PRELUDE + NATIVE_SERVER)
    with socket.socket() as listener:
        listener.bind((host, 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        serve = _start_serve(tmp_path, [sys.executable, str(script), host, str(port)])
        try:
            assert serve.wait(60) == 1
        finally:
            if serve.poll() is None:
                serve.kill()
                serve.wait(30)
    records = _stats(tmp_path)
    server_pid = next(r["pid"] for r in records if r["event"] == "started")
    assert records[-1]["event"] == "exit" and records[-1]["reason"] == "violation"
    (violation,) = [r for r in records if r["event"] == "violation"]
    assert any(s["local"][0] == host for s in violation["sockets"])
    assert hostrun.session_pids(server_pid) == []


def _state(tmp_path: Path, capsys) -> dict:
    stats, pid_file = tmp_path / "stats.jsonl", tmp_path / "serve.pid"
    assert hostrun.main(["state", "--stats", str(stats), "--pid-file", str(pid_file)]) == 0
    return json.loads(capsys.readouterr().out)


def test_state_reports_alive_exit_and_violation(tmp_path, capsys) -> None:
    assert _state(tmp_path, capsys) == {
        "kind": "serve_state",
        "started": False,
        "serve_alive": False,
        "exit": None,
        "violation": None,
    }
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "serve.pid").write_text(f"{child.pid} {hostrun.start_ticks(child.pid)}")
        stats = tmp_path / "stats.jsonl"
        stats.write_text(json.dumps({"event": "started", "t": 1.0, "pid": 5}) + "\n")
        with open(stats, "a") as out:
            out.write('{"event": "stats", "t": 2.0, "rss_by')
        state = _state(tmp_path, capsys)
        assert state["started"] and state["serve_alive"] and state["exit"] is None
    finally:
        child.kill()
        child.wait()
    sockets = [{"table": "tcp", "local": ["10.0.0.2", 4000], "remote": ["0.0.0.0", 0]}]
    stats.write_text(
        "".join(
            json.dumps(line) + "\n"
            for line in (
                {"event": "started", "t": 1.0, "pid": 5},
                {"event": "violation", "t": 3.0, "sockets": sockets},
                {"event": "exit", "t": 3.5, "reason": "violation", "returncode": -9},
            )
        )
    )
    state = _state(tmp_path, capsys)
    assert state["serve_alive"] is False and state["violation"] == sockets
    assert state["exit"] == {"reason": "violation", "returncode": -9, "t": 3.5}


def test_state_reports_alive_only_for_the_recorded_start_ticks(tmp_path, capsys) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        start = hostrun.start_ticks(child.pid)
        pid_file = tmp_path / "serve.pid"
        for text in (
            f"{child.pid} {start + 1}",
            f"{child.pid}",
            f"{child.pid} later",
            f"{child.pid} {start} {start}",
        ):
            pid_file.write_text(text)
            assert _state(tmp_path, capsys)["serve_alive"] is False, text
        pid_file.write_text(f"{child.pid} {start}")
        assert _state(tmp_path, capsys)["serve_alive"] is True
    finally:
        child.kill()
        child.wait()


def test_evidence_tails_are_bounded_line_cut_and_list_processes(tmp_path, capsys) -> None:
    log = b"".join(b"line %03d of the server\n" % i for i in range(100)) + "конец\n".encode()
    (tmp_path / "server.log").write_bytes(log)
    (tmp_path / "hostrun.log").write_bytes(b"short\n")
    (tmp_path / "server-stats.jsonl").write_bytes(b"x" * 300)
    marker = "evidence-marker-" + str(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", marker])
    try:
        assert hostrun.main(["evidence", "--dir", str(tmp_path), "--max-bytes", "100"]) == 0
        found = json.loads(capsys.readouterr().out)
    finally:
        child.kill()
        child.wait()
    files = found["files"]
    assert found["kind"] == "server_evidence"
    assert files["guard.log"] == {"missing": True}
    assert files["hostrun.log"] == {"size": 6, "tail": "short\n"}
    server = files["server.log"]
    assert server["size"] == len(log) and server["tail"].endswith("конец\n")
    assert len(server["tail"].encode()) <= 100 and server["tail"].startswith("line ")
    assert log.decode().endswith(server["tail"])
    assert files["server-stats.jsonl"] == {"size": 300, "tail": ""}
    (entry,) = [p for p in found["processes"] if p["pid"] == child.pid]
    assert entry["ppid"] == os.getpid() and marker in entry["cmdline"]
    assert entry["state"] in "RSD" and entry["rss_bytes"] > 0
    assert all(len(p["cmdline"]) <= hostrun.CMDLINE_MAX for p in found["processes"])


@pytest.mark.slow
def test_stats_record_the_rss_breakdown_and_the_limit_still_uses_vmrss(
    tmp_path, monkeypatch
) -> None:
    code, lines = _serve(
        tmp_path,
        "--deadline",
        str(time.time() + 1),
        "--rss-limit",
        str(1 << 40),
        child="import time; time.sleep(60)",
    )
    stats = [line for line in lines if line["event"] == "stats"]
    assert stats and stats[0]["rss_bytes"] > 0
    parts = ("rss_anon_bytes", "rss_file_bytes", "rss_shmem_bytes")
    assert sum(stats[0][part] for part in parts) == stats[0]["rss_bytes"]

    def run(memory: dict) -> str | None:
        monkeypatch.setattr(hostrun, "tree_memory", lambda pids: memory)
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        return hostrun.monitor(
            child,
            deadline=time.time() + 1,
            rss_limit=1000,
            stats=tmp_path / "monitor.jsonl",
            stop=threading.Event(),
        )

    big = 10**9
    assert (
        run({"rss_bytes": 10, "rss_anon_bytes": big, "rss_file_bytes": big, "rss_shmem_bytes": big})
        == "deadline"
    )
    assert (
        run({"rss_bytes": big, "rss_anon_bytes": 0, "rss_file_bytes": 0, "rss_shmem_bytes": 0})
        == "rss_limit"
    )
