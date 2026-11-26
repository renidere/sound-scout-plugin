import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sound_scout import proc
from sound_scout.proc import RunResult, ffmpeg_argv, ffprobe_argv, run

PYTHON = sys.executable
ONE_MIB = 1024 * 1024


def process_gone(pid: int, wait_seconds: float = 5.0) -> bool:
    """True once the process no longer exists or is only a zombie awaiting its reaper."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            return True
        if state == "Z":
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def test_runs_argv_and_returns_structured_result() -> None:
    result = run(
        [PYTHON, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"],
        timeout=30,
        max_output_bytes=ONE_MIB,
    )
    assert result == RunResult(
        returncode=3,
        stdout=b"out\n",
        stderr=b"err\n",
        timed_out=False,
        truncated=False,
        duration_seconds=result.duration_seconds,
    )
    assert 0 < result.duration_seconds < 30


@pytest.mark.parametrize("argv", ["sleep 1", b"sleep 1", ["sleep", 1], [], ("sleep", None), 5])
def test_rejects_anything_but_a_list_of_str(argv: object) -> None:
    with pytest.raises(TypeError, match="argv must be"):
        run(argv, timeout=1, max_output_bytes=ONE_MIB)  # type: ignore[arg-type]


def test_no_shell_so_metacharacters_are_plain_arguments(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    result = run(["echo", f"x; touch {marker} && echo $HOME"], timeout=10, max_output_bytes=ONE_MIB)
    assert result.returncode == 0
    assert result.stdout == f"x; touch {marker} && echo $HOME\n".encode()
    assert not marker.exists()


def test_stdin_is_closed_for_the_child() -> None:
    result = run(
        [PYTHON, "-c", "import sys; print(len(sys.stdin.read()))"], timeout=10, max_output_bytes=64
    )
    assert result.stdout == b"0\n"


@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_timeout(value: float) -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        run(["true"], timeout=value, max_output_bytes=ONE_MIB)


def test_timeout_kills_the_process_and_reports_it() -> None:
    started = time.monotonic()
    result = run(["sleep", "30"], timeout=0.5, max_output_bytes=ONE_MIB)
    assert result.timed_out is True
    assert result.returncode != 0
    assert time.monotonic() - started < 5
    assert 0.5 <= result.duration_seconds < 5


def test_timeout_leaves_no_sleep_process_and_no_group_behind() -> None:
    # `exec` makes the printed pid the pid of `sleep` itself, the leader of the new group.
    result = run(["sh", "-c", "echo $$; exec sleep 30"], timeout=0.5, max_output_bytes=ONE_MIB)
    assert result.timed_out is True
    sleep_pid = int(result.stdout)
    assert process_gone(sleep_pid), "the sleep process survived the timeout"
    with pytest.raises(ProcessLookupError):
        os.killpg(sleep_pid, 0)


def test_timeout_kills_the_whole_process_group() -> None:
    script = "echo $$; sleep 30 & echo $!; wait"
    result = run(["sh", "-c", script], timeout=0.5, max_output_bytes=ONE_MIB)
    assert result.timed_out is True
    shell_pid, sleep_pid = (int(line) for line in result.stdout.split())
    assert process_gone(shell_pid), "the shell survived the timeout"
    assert process_gone(sleep_pid), "the background child of the shell survived the timeout"


def test_interrupted_run_kills_the_whole_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs: list[proc._Output] = []
    read_until = proc._Output.read_until

    def read_then_interrupt(self: proc._Output, deadline: float) -> bool:
        read_until(self, time.monotonic() + 0.5)  # collect the pids the shell prints
        outputs.append(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(proc._Output, "read_until", read_then_interrupt)
    script = "echo $$; sleep 30 & echo $!; wait"
    with pytest.raises(KeyboardInterrupt):
        run(["sh", "-c", script], timeout=30, max_output_bytes=ONE_MIB)
    shell_pid, sleep_pid = (int(line) for line in bytes(outputs[0].stdout).split())
    assert process_gone(shell_pid), "the shell survived the interruption"
    assert process_gone(sleep_pid), "the background child of the shell survived the interruption"


def test_output_before_the_timeout_is_kept() -> None:
    result = run(
        ["sh", "-c", "echo partial; echo warn >&2; sleep 30"], timeout=0.5, max_output_bytes=ONE_MIB
    )
    assert result.timed_out is True
    assert result.stdout == b"partial\n"
    assert result.stderr == b"warn\n"


def test_output_is_capped_per_stream_and_flagged() -> None:
    script = "import sys; sys.stdout.write('a' * 300000); sys.stderr.write('b' * 10)"
    result = run([PYTHON, "-c", script], timeout=30, max_output_bytes=1000)
    assert result.truncated is True
    assert result.timed_out is False
    assert result.returncode == 0
    assert result.stdout == b"a" * 1000
    assert result.stderr == b"b" * 10


def test_output_exactly_at_the_cap_is_not_truncated() -> None:
    result = run([PYTHON, "-c", "print('x' * 9)"], timeout=30, max_output_bytes=10)
    assert result.truncated is False
    assert result.stdout == b"x" * 9 + b"\n"


def test_rejects_negative_output_cap() -> None:
    with pytest.raises(ValueError, match="max_output_bytes"):
        run(["true"], timeout=1, max_output_bytes=-1)


def test_start_copies_both_streams_to_the_log_up_to_the_cap_without_blocking(
    tmp_path: Path,
) -> None:
    script = "import sys; sys.stderr.write('b' * 10); sys.stderr.flush(); print('a' * 300000)"
    log_path = tmp_path / "server.log"
    with open(log_path, "wb") as log:
        child, copier = proc.start([PYTHON, "-c", script], log=log, max_log_bytes=1000)
        assert child.wait(timeout=30) == 0
        copier.join(timeout=30)
        assert not copier.is_alive()
    assert log_path.read_bytes() == b"b" * 10 + b"a" * 990


def test_start_puts_the_child_in_its_own_session_without_stdin(tmp_path: Path) -> None:
    script = "import os, sys; print(os.getsid(0) == os.getpid(), sys.stdin.read() == '')"
    log_path = tmp_path / "server.log"
    with open(log_path, "wb") as log:
        child, copier = proc.start([PYTHON, "-c", script], log=log, max_log_bytes=ONE_MIB)
        child.wait(timeout=30)
        copier.join(timeout=30)
    assert log_path.read_text() == "True True\n"


@pytest.mark.parametrize(
    ("argv", "cap", "error"), [("sleep 1", 10, TypeError), (["true"], -1, ValueError)]
)
def test_start_rejects_a_command_string_and_a_negative_log_cap(
    tmp_path: Path, argv: object, cap: int, error: type
) -> None:
    with open(tmp_path / "server.log", "wb") as log, pytest.raises(error):
        proc.start(argv, log=log, max_log_bytes=cap)  # type: ignore[arg-type]


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is applied on Linux only")
def test_memory_limit_applies_to_the_child() -> None:
    script = "import resource, json; print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))"
    result = run(
        [PYTHON, "-c", script], timeout=30, max_output_bytes=ONE_MIB, memory_bytes=512 * ONE_MIB
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [512 * ONE_MIB, 512 * ONE_MIB]
    unlimited = run([PYTHON, "-c", script], timeout=30, max_output_bytes=ONE_MIB)
    assert json.loads(unlimited.stdout) != [512 * ONE_MIB, 512 * ONE_MIB]


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is applied on Linux only")
def test_memory_limit_stops_an_allocation_above_it() -> None:
    script = "x = bytearray(400 * 1024 * 1024); print('allocated')"
    result = run(
        [PYTHON, "-c", script], timeout=30, max_output_bytes=ONE_MIB, memory_bytes=256 * ONE_MIB
    )
    assert result.returncode != 0
    assert b"MemoryError" in result.stderr
    assert result.stdout == b""


def test_memory_limit_is_skipped_on_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc.sys, "platform", "darwin")
    assert proc._memory_limit(ONE_MIB) is None
    result = run(["true"], timeout=10, max_output_bytes=16, memory_bytes=ONE_MIB)
    assert result.returncode == 0


def test_rejects_non_positive_memory_limit() -> None:
    with pytest.raises(ValueError, match="memory_bytes must be positive"):
        run(["true"], timeout=1, max_output_bytes=16, memory_bytes=0)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG exists on Linux only")
def test_a_child_dies_with_its_killed_parent(tmp_path: Path) -> None:
    pid_file = tmp_path / "child"
    child = (
        "import os, sys, time\n"
        "open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid()))\n"
        "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
        "time.sleep(600)\n"
    )
    parent_script = (
        "import sys\n"
        "from sound_scout import proc\n"
        "proc.run([sys.executable, '-c', sys.argv[1], sys.argv[2]], timeout=600,"
        " max_output_bytes=1024)\n"
    )
    parent = subprocess.Popen([PYTHON, "-c", parent_script, child, str(pid_file)])
    try:
        deadline = time.monotonic() + 30
        while not pid_file.exists():
            assert time.monotonic() < deadline, "the child did not start"
            time.sleep(0.02)
        child_pid = int(pid_file.read_text())
        parent.kill()
        parent.wait()
        assert process_gone(child_pid, wait_seconds=2.0)
    finally:
        parent.kill()
        parent.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG exists on Linux only")
def test_a_started_child_gets_the_chosen_signal_when_its_parent_is_killed(tmp_path: Path) -> None:
    """SIGTERM lets the child stop what it owns before it exits (#198)."""
    ready, caught = tmp_path / "ready", tmp_path / "caught"
    child = (
        "import os, signal, sys, time\n"
        "def caught(s, f):\n"
        "    open(sys.argv[2] + '.tmp', 'w').write(str(s))\n"
        "    os.replace(sys.argv[2] + '.tmp', sys.argv[2])\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, caught)\n"
        "open(sys.argv[1], 'w').close()\n"
        "time.sleep(600)\n"
    )
    parent_script = (
        "import signal, sys, time\n"
        "from sound_scout import proc\n"
        "proc.start([sys.executable, '-c', *sys.argv[1:]], log=open('/dev/null', 'wb'),"
        " max_log_bytes=0, parent_death_signal=signal.SIGTERM)\n"
        "time.sleep(600)\n"
    )
    parent = subprocess.Popen([PYTHON, "-c", parent_script, child, str(ready), str(caught)])
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert time.monotonic() < deadline, "the child did not start"
            time.sleep(0.02)
        parent.kill()
        parent.wait()
        deadline = time.monotonic() + 10
        while not caught.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert caught.read_text() == str(int(signal.SIGTERM))
    finally:
        parent.kill()
        parent.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG exists on Linux only")
def test_a_started_child_does_not_inherit_an_ignored_parent_death_signal(tmp_path: Path) -> None:
    """An ignored signal stays ignored across exec; the child setup resets it (#198)."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # `reset_process_globals` restores it
    log = tmp_path / "status"
    with log.open("wb") as handle:
        child, copier = proc.start(
            ["cat", "/proc/self/status"],
            log=handle,
            max_log_bytes=1 << 16,
            parent_death_signal=signal.SIGTERM,
        )
        child.wait(10)
        copier.join(10)
    ignored = re.search(r"^SigIgn:\s*([0-9a-f]+)$", log.read_text(), re.M)
    assert ignored is not None
    assert not int(ignored.group(1), 16) & (1 << (signal.SIGTERM - 1))


def test_die_with_parent_is_a_no_op_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "_LIBC", None)
    assert proc.die_with_parent(os.getpid() + 1) is None


def test_decoder_argv_always_restrict_protocols_and_stdin() -> None:
    assert ffmpeg_argv("-i", "in.wav", "-f", "null", "-") == [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        "in.wav",
        "-f",
        "null",
        "-",
    ]
    assert ffprobe_argv("-show_streams", "in.wav") == [
        "ffprobe",
        "-hide_banner",
        "-protocol_whitelist",
        "file,pipe",
        "-show_streams",
        "in.wav",
    ]


def test_ffmpeg_argv_restricts_protocols_for_every_input() -> None:
    argv = ffmpeg_argv("-i", "a.wav", "-ss", "1", "-i", "b.wav", "-f", "null", "-")
    inputs = [index for index, arg in enumerate(argv) if arg == "-i"]
    assert len(inputs) == 2
    for index in inputs:
        assert argv[index - 2 : index] == ["-protocol_whitelist", "file,pipe"]


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)
NON_FILE_SOURCES = ["http://127.0.0.1:9/x.wav", "concat:a.wav|b.wav"]


@needs_ffmpeg
def test_ffprobe_reads_a_local_wav(wav_file: Path) -> None:
    result = run(
        ffprobe_argv("-v", "error", "-print_format", "json", "-show_streams", str(wav_file)),
        timeout=30,
        max_output_bytes=ONE_MIB,
    )
    assert result.returncode == 0, result.stderr
    stream = json.loads(result.stdout)["streams"][0]
    assert stream["codec_type"] == "audio"
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1


@needs_ffmpeg
@pytest.mark.parametrize("source", NON_FILE_SOURCES)
def test_ffprobe_refuses_non_file_protocols_itself(source: str) -> None:
    result = run(
        ffprobe_argv("-v", "error", "-show_streams", source), timeout=30, max_output_bytes=ONE_MIB
    )
    assert result.returncode != 0
    assert b"not on whitelist" in result.stderr
    assert b"Connection" not in result.stderr


@needs_ffmpeg
@pytest.mark.parametrize("source", NON_FILE_SOURCES)
def test_ffmpeg_refuses_non_file_protocols_itself(source: str) -> None:
    result = run(
        ffmpeg_argv("-v", "error", "-i", source, "-f", "null", "-"),
        timeout=30,
        max_output_bytes=ONE_MIB,
    )
    assert result.returncode != 0
    assert b"not on whitelist" in result.stderr
    assert b"Connection" not in result.stderr


@needs_ffmpeg
def test_ffmpeg_refuses_a_non_file_second_input(wav_file: Path) -> None:
    result = run(
        ffmpeg_argv(
            "-v", "error", "-i", str(wav_file), "-i", NON_FILE_SOURCES[0], "-f", "null", "-"
        ),
        timeout=30,
        max_output_bytes=ONE_MIB,
    )
    assert result.returncode != 0
    assert b"not on whitelist" in result.stderr
    assert b"Connection" not in result.stderr


@needs_ffmpeg
def test_ffmpeg_decodes_a_local_wav(wav_file: Path) -> None:
    result = run(
        ffmpeg_argv("-v", "error", "-i", str(wav_file), "-f", "s16le", "-"),
        timeout=30,
        max_output_bytes=4 * ONE_MIB,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout) == 16_000 * 2
    assert result.truncated is False
