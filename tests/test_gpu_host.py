"""Host files of the GPU stack (#147): the audit-hook guard, the `/proc` socket detector, the
allowlisted environment, the weights receipt, `pack` and `serve` limits.

The guard runs here in real child processes on this station; nothing reaches beyond loopback
(each probe first checks that the hook is loaded, and the hook denies before the system call).
This proves the guard and the host commands, not the image they will run in.
"""

import json
import os
import subprocess
import sys
import tarfile
import time
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


def test_selftest_allows_loopback_and_denies_outside_lookup_spawn_and_native(tmp_path) -> None:
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
    }
    assert result["native_detector"]["status"] in ("passed", "unavailable")
    assert isinstance(result["netns_available"], bool)


def test_the_hook_denies_an_internet_bind_and_logs_it_but_allows_unix_and_loopback(
    tmp_path,
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
    assert _guarded_python(tmp_path, code.replace("{unix}", str(tmp_path / "s"))) == "denied"
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
    (directory / ".cache" / "huggingface").mkdir(parents=True)
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
    assert printed == {"archive_sha256": vast._sha256(archive), "files": 2}
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == [
            "SHA256SUMS",
            "guard-selftest.json",
            "qwen25omni/server.log",
        ]
    assert vast.verify_archive(archive, tmp_path / "host") == 2


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
