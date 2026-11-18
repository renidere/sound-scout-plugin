"""Bounded child processes: argv only, a process group, wall time, output and memory caps (R10).

`run` and `start` are the only ways the package starts a program. `run` takes a list of
arguments (never a command string, never a shell), starts the child in a new session so that the
whole process group can be killed when the wall time runs out or the caller is interrupted (R11:
no orphaned worker), keeps at most `max_output_bytes` of each of stdout and stderr, and on Linux
caps the child's address space with `RLIMIT_AS` when `memory_bytes` is given. Every child also gets
`die_with_parent` before `exec`: on Linux it is killed when the process that started it dies, even
by SIGKILL, so a killed analysis worker leaves no decoder behind (R11).

`start` is for the one long-lived child, the Gate 0 `llama-server` (`eval.gate0`): the same argv
check, new session and `die_with_parent`, but no wall time, because its caller stops it; its
stdout and stderr go to a log file kept to `max_log_bytes`.

`ffmpeg_argv` and `ffprobe_argv` build the decoder command lines: the protocol whitelist
`file,pipe` makes the decoder itself refuse `http://`, `concat:` and every other non-file
source, independently of the path checks done before. ffmpeg applies the whitelist to the
input that follows it, so `ffmpeg_argv` repeats it before every `-i`; ffprobe takes one
input. ffprobe never reads stdin and does not know `-nostdin`; `run` opens stdin from
`/dev/null` for every child anyway.
"""

import contextlib
import ctypes
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import BinaryIO

PROTOCOL_WHITELIST = "file,pipe"
KILL_GRACE_SECONDS = 5.0
_READ_CHUNK = 65536
_PR_SET_PDEATHSIG = 1
# Resolved once in the parent: a dlopen in the forked child of a threaded process can deadlock.
_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    truncated: bool
    duration_seconds: float


def run(
    argv: Sequence[str],
    *,
    timeout: float,
    max_output_bytes: int,
    memory_bytes: int | None = None,
) -> RunResult:
    """Run `argv` without a shell and return its bounded outcome.

    On timeout the child's whole process group is killed with SIGKILL and `timed_out` is set;
    the output collected until then is returned. `truncated` is set when either stream
    exceeded `max_output_bytes`; the excess is drained and dropped. `memory_bytes` sets
    `RLIMIT_AS` for the child on Linux and is ignored elsewhere.
    """
    argv = _checked_argv(argv)
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    if max_output_bytes < 0:
        raise ValueError(f"max_output_bytes must not be negative, got {max_output_bytes!r}")
    if memory_bytes is not None and memory_bytes <= 0:
        raise ValueError(f"memory_bytes must be positive, got {memory_bytes!r}")
    started = time.monotonic()
    child = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=_child_setup(memory_bytes),
    )
    assert child.stdout is not None and child.stderr is not None
    output = _Output(child.stdout, child.stderr, max_output_bytes)
    timed_out = False
    try:
        try:
            if not output.read_until(started + timeout):
                raise subprocess.TimeoutExpired(argv, timeout)
            child.wait(timeout=max(0.0, started + timeout - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(child)
            # The pipes close when the group is dead; a grandchild that left the group may
            # keep them open, so reading stops after a grace period either way.
            output.read_until(time.monotonic() + KILL_GRACE_SECONDS)
            child.wait()
    finally:
        # Any other exception (Ctrl+C, MemoryError, OSError) must not leave the group behind.
        if child.returncode is None:
            _kill_group(child)
            child.wait()
        child.stdout.close()
        child.stderr.close()
    return RunResult(
        returncode=child.returncode,
        stdout=bytes(output.stdout),
        stderr=bytes(output.stderr),
        timed_out=timed_out,
        truncated=output.truncated,
        duration_seconds=time.monotonic() - started,
    )


def start(
    argv: Sequence[str], *, log: BinaryIO, max_log_bytes: int
) -> tuple[subprocess.Popen, threading.Thread]:
    """Start a long-lived child the caller stops itself; return it and the thread copying its log.

    stdout and stderr go to `log`, at most `max_log_bytes`; the excess is drained and dropped,
    so the child never blocks on a full pipe. The thread ends when the child closes its output:
    join it after stopping the child and before closing `log`.
    """
    argv = _checked_argv(argv)
    if max_log_bytes < 0:
        raise ValueError(f"max_log_bytes must not be negative, got {max_log_bytes!r}")
    child = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        preexec_fn=_child_setup(None),
    )
    copier = threading.Thread(
        target=_copy_capped, args=(child.stdout, log, max_log_bytes), daemon=True
    )
    copier.start()
    return child, copier


def _copy_capped(source, target: BinaryIO, limit: int) -> None:
    room = limit
    with source:
        while chunk := os.read(source.fileno(), _READ_CHUNK):
            if room > 0:
                target.write(chunk[:room])
                target.flush()
                room -= min(room, len(chunk))


def ffmpeg_argv(*args: str) -> list[str]:
    argv = ["ffmpeg", "-nostdin", "-hide_banner"]
    for arg in args:
        if arg == "-i":
            argv += ["-protocol_whitelist", PROTOCOL_WHITELIST]
        argv.append(arg)
    return argv


def ffprobe_argv(*args: str) -> list[str]:
    return ["ffprobe", "-hide_banner", "-protocol_whitelist", PROTOCOL_WHITELIST, *args]


def _checked_argv(argv: object) -> list[str]:
    if isinstance(argv, str | bytes) or not isinstance(argv, Sequence):
        raise TypeError("argv must be a list of arguments, not a command string")
    if not argv or not all(isinstance(item, str) for item in argv):
        raise TypeError("argv must be a non-empty list of str")
    return list(argv)


def die_with_parent(parent_pid: int | None = None) -> None:
    """Have the kernel SIGKILL this process when its parent dies (Linux; elsewhere a no-op).

    With `parent_pid`, a parent that died before the request took effect is noticed too: the
    process kills itself, as the signal would have.
    """
    if _LIBC is None:
        return
    if _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if parent_pid is not None and os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _child_setup(memory_bytes: int | None) -> Callable[[], None]:
    parent_pid = os.getpid()
    limit = _memory_limit(memory_bytes)

    def setup() -> None:
        die_with_parent(parent_pid)
        if limit is not None:
            limit()

    return setup


def _memory_limit(memory_bytes: int | None) -> Callable[[], None] | None:
    if memory_bytes is None or sys.platform != "linux":
        return None
    import resource

    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    return limit


class _Output:
    """Both pipes of a child, each kept to `limit` bytes; the excess is drained and dropped."""

    def __init__(self, stdout, stderr, limit: int) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.truncated = False
        self._limit = limit
        self._open = {stdout.fileno(): self.stdout, stderr.fileno(): self.stderr}

    def read_until(self, deadline: float) -> bool:
        """Read until both pipes hit EOF (True) or the deadline passes (False)."""
        with selectors.DefaultSelector() as selector:
            for fd in self._open:
                selector.register(fd, selectors.EVENT_READ)
            while self._open:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, _READ_CHUNK)
                    if not chunk:
                        selector.unregister(key.fd)
                        del self._open[key.fd]
                        continue
                    buffer = self._open[key.fd]
                    room = self._limit - len(buffer)
                    if len(chunk) > room:
                        self.truncated = True
                    buffer += chunk[: max(0, room)]
        return True


def _kill_group(child: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
