"""Acceptance criteria of #147 proven end to end: a canary key through the whole dry run, the
watchdog in a process of its own after the agent is killed, the archive hash check of the
download and the host guard on real sockets.

The provider and the host are the fakes of `vast_fake` on 127.0.0.1, which the issue names as the
proof medium; the key reader, the journal, the watchdog, the guard and the hash checks are real.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from sound_scout import config
from sound_scout.eval import vast, vast_fake
from sound_scout.eval.gpu_host import hostrun

CANARY = "CANARY-" + "q9Zk" * 8


def _config(tmp_path: Path) -> config.Config:
    for name in ("in", "data"):
        (tmp_path / name).mkdir()
    return config.Config(
        allowed_roots=((tmp_path / "in").resolve(),), data_root=(tmp_path / "data").resolve()
    )


@pytest.mark.slow
def test_criterion_2_a_canary_key_reaches_no_output_argv_environment_payload_or_file(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    monkeypatch.setattr(vast.secrets, "token_urlsafe", lambda n=None: CANARY)
    config_ = _config(tmp_path)
    monkeypatch.setattr(vast.config, "load", lambda: config_)
    started = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):
        def __init__(self, args, *rest, **kwargs):
            started.append((args, kwargs.get("env")))
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    payloads = []

    class RecordingFake(vast_fake.FakeVast):
        def stop(self):
            payloads.extend(inst.get("payload") for inst in self.instances.values())
            super().stop()

    monkeypatch.setattr(vast.vast_fake, "FakeVast", RecordingFake)
    out = tmp_path / "data" / "dry"

    assert vast.main(["dry-run", "--out", str(out)]) == 0

    assert (out / "fake-api-key").read_text().strip() == CANARY
    printed = capfd.readouterr()
    assert CANARY not in printed.out and CANARY not in printed.err
    assert json.loads(printed.out.splitlines()[-1])["closed"]["event"] == "closed"
    assert started, "the dry run starts the host commands as processes"
    for args, env in started:
        assert CANARY not in json.dumps([str(a) for a in args]), args
        assert CANARY not in json.dumps(dict(env or os.environ)), args
    assert len(payloads) == 1 and payloads[0]["env"] == {}
    assert CANARY not in json.dumps(payloads[0])
    assert payloads[0]["onstart"].startswith(vast.ONSTART.split("{", 1)[0])
    checked = 0
    for path in out.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name != "fake-api-key":
            assert CANARY.encode() not in path.read_bytes(), path
            checked += 1
    host_root = out / "host-root"
    assert (host_root / "ss" / "hostrun.py").is_file() and checked > 20
    assert any(p.name == "journal.jsonl" for p in out.rglob("journal.jsonl"))


AGENT = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    from sound_scout.eval import vast

    gpu, api_base, key_path, deadline, mode = sys.argv[1:6]
    key = vast.read_api_key(Path(key_path))
    attempt_id = vast.new_attempt_id()
    journal = vast.Journal.create(Path(gpu), attempt_id, key)
    journal.append(
        "opened", attempt_id=attempt_id, label=vast.LABEL_PREFIX + attempt_id,
        deadline_unix=float(deadline), api_base=api_base, key_path=key_path,
        initial_credit=10.0, early_stop_usd=2.5,
    )
    journal.append("armed", units=["u-watch", "u-cutoff"], argv=[], dry_run=True)
    package = {
        "offer_id": 4242, "envelope": {"dph_total": {"lte": 1.533}}, "ceiling_usd": 4.0,
        "early_stop_usd": 2.5, "deadline_minutes": 150, "disk_gb": 150, "in_gb": 102.95,
        "out_gb": 1.0,
    }
    (journal.dir / vast.HEARTBEAT_FILE).write_text(str(time.time()))
    try:
        vast.create_instance(vast.Api(api_base, key, timeout=0.5), journal, package, None)
    except vast.CreateOutcomeUnknown:
        pass
    print(journal.path, flush=True)
    time.sleep(600)
    """
)


def _watchdog(journal: Path, command: str) -> subprocess.CompletedProcess:
    """The exact command a timer starts, from its argv builder, in a process of its own."""
    if command == "watch":
        argv = vast.watch_timer_argv(journal, "A", python=sys.executable)
    else:
        argv = vast.cutoff_timer_argv(journal, "A", time.time(), python=sys.executable)
    return subprocess.run(argv[argv.index("--") + 1 :], capture_output=True, timeout=120)


@pytest.mark.slow
@pytest.mark.parametrize("create_mode", ["ok", "drop"], ids=["create-answered", "create-lost"])
def test_criterion_3_watchdog_in_its_own_process_destroys_by_id_after_the_agent_is_killed(
    tmp_path: Path, create_mode
) -> None:
    key_path = tmp_path / "vast_api_key"
    key_path.write_text(CANARY + "\n")
    key_path.chmod(0o600)
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700)
    fake = vast_fake.FakeVast(CANARY, create_mode=create_mode).start()
    agent = None
    try:
        deadline = time.time() + 25
        agent = subprocess.Popen(
            [sys.executable, "-c", AGENT, str(gpu), fake.url, str(key_path), str(deadline)]
            + [create_mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        journal_path = Path(agent.stdout.readline().decode().strip())
        os.kill(agent.pid, signal.SIGKILL)
        agent.wait(10)
        assert fake.count("PUT") == 1 and fake.count("DELETE") == 0
        events = [e["event"] for e in vast.Journal(journal_path).replay()]
        expected = "create_response" if create_mode == "ok" else "create_outcome_unknown"
        assert events[-1] == expected

        first = _watchdog(journal_path, "watch")
        if create_mode == "ok":
            assert time.time() < deadline, "the first tick must run before the deadline"
            assert json.loads(first.stdout) == {"outcome": "ok"}
            assert fake.count("DELETE") == 0
            while time.time() < deadline:
                time.sleep(0.5)
            second = _watchdog(journal_path, "watch")
            assert json.loads(second.stdout) == {"outcome": "closed"}, second.stderr
            outputs = [first, second]
        else:
            assert json.loads(first.stdout) == {"outcome": "closed"}, first.stderr
            outputs = [first]
        deletes = [e["path"] for e in fake.log if e["method"] == "DELETE"]
        assert deletes == [f"/api/v0/instances/{vast_fake.FIRST_INSTANCE_ID}/"]
        assert not any(fake._visible(item) for item in fake.instances.values())
        replay = vast.Journal(journal_path).replay()
        closed = replay[-1]
        assert closed["event"] == "closed"
        assert ("deadline" if create_mode == "ok" else "create_outcome_unknown") in closed["reason"]
        assert any(
            e["event"] == "absence_verified" and e["instance_id"] == vast_fake.FIRST_INSTANCE_ID
            for e in replay
        )
        if create_mode == "drop":
            assert "adopted_by_label" in [e["event"] for e in replay]

        later = _watchdog(journal_path, "cutoff")
        assert json.loads(later.stdout) == {"outcome": "closed"}
        assert fake.count("PUT") == 1 and fake.count("DELETE") == 1
        for done in [*outputs, later]:
            assert CANARY.encode() not in done.stdout + done.stderr
        assert CANARY not in journal_path.read_text()
    finally:
        if agent is not None and agent.poll() is None:
            agent.kill()
        fake.stop()


def _pack_host(tmp_path: Path, corrupt: bool):
    root = tmp_path / "host-root"
    (root / "ss").mkdir(parents=True)
    (root / "out" / "qwen25omni").mkdir(parents=True)
    shutil.copyfile(vast.HOST_FILES_DIR / "hostrun.py", root / "ss" / "hostrun.py")
    (root / "out" / "guard-selftest.json").write_text('{"ok": true}')
    (root / "out" / "qwen25omni" / "run.json").write_text('{"runtime": "vllm"}')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)

    class Host(vast.LocalHost):
        def get(self, remote: str, local: Path, timeout: float = 1800.0) -> None:
            super().get(remote, local, timeout)
            if corrupt:
                data = bytearray(local.read_bytes())
                data[-1] ^= 0xFF
                local.write_bytes(bytes(data))

    return Host(root, {"PATH": f"{bin_dir}:/usr/bin:/bin"})


@pytest.mark.parametrize("corrupt", [False, True], ids=["intact", "one-byte-changed"])
def test_criterion_4_download_checks_the_archive_sha256_against_the_host(
    tmp_path: Path, corrupt
) -> None:
    host = _pack_host(tmp_path, corrupt)
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700)
    journal = vast.Journal.create(gpu, "A", None)
    if corrupt:
        with pytest.raises(vast.VastError, match="SHA-256 differs"):
            vast.download(host, journal.dir, journal, 60)
        assert "download_verified" not in [e["event"] for e in journal.replay()]
        assert not (journal.dir / vast.HOST_DIR).exists()
    else:
        result = vast.download(host, journal.dir, journal, 60)
        assert result["files"] == 2 and len(result["archive_sha256"]) == 64
        verified = journal.replay()[-1]
        assert verified["event"] == "download_verified"
        assert verified["archive_sha256"] == vast._sha256(journal.dir / "results.tar")
        assert (journal.dir / vast.HOST_DIR / "qwen25omni" / "run.json").is_file()


def test_criterion_1_guard_blocks_an_outside_connect_and_allows_a_loopback_listener(
    tmp_path: Path,
) -> None:
    env = hostrun.guarded_env(os.environ, vast.HOST_FILES_DIR, tmp_path / "guard.log")
    code = hostrun.GUARDED_PRELUDE + textwrap.dedent(
        """
        import socket
        server = socket.socket(); server.bind(("127.0.0.1", 0)); server.listen(1)
        client = socket.create_connection(server.getsockname(), timeout=5)
        peer, _ = server.accept(); client.sendall(b"ping"); assert peer.recv(4) == b"ping"
        print("loopback ok")
        for target in (("192.0.2.1", 9), ("198.51.100.7", 53)):
            try:
                socket.socket().connect(target)
            except PermissionError:
                print("denied", target[0])
            else:
                print("allowed", target[0])
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, timeout=60, cwd=tmp_path
    )
    assert done.stdout.decode().split("\n")[:3] == [
        "loopback ok",
        "denied 192.0.2.1",
        "denied 198.51.100.7",
    ], done.stderr
    log = [json.loads(line) for line in (tmp_path / "guard.log").read_text().splitlines()]
    assert {entry["event"] for entry in log} >= {"socket.connect"}
    assert any("192.0.2.1" in entry["target"] for entry in log)
