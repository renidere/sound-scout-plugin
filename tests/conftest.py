import functools
import os
import re
import shutil
import signal
import subprocess
import tempfile
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from sound_scout import guard
from sound_scout.config import root_denied_name

WavFactory = Callable[..., Path]
Encoder = Callable[..., Path]

TEMP_DIR_VARIABLES = ("TMPDIR", "TEMP", "TMP")


def temp_dir_bars_tmp_path(temp_dir: Path) -> bool:
    """Whether the root rule refuses a `tmp_path` below `temp_dir`, not `temp_dir` itself.

    A Hermes media dir such as `.hermes/cache/audio` is an allowed root, but pytest puts
    `tmp_path` in a directory below it, and there `.hermes` is denied.
    """
    return root_denied_name(temp_dir / "pytest-of-probe") is not None


def use_a_temp_dir_outside_denied_names() -> None:
    """Move TMPDIR to the system temp dir when the inherited one has a denied name (issue #89).

    Hermes runs commands with TMPDIR under `~/.hermes`. Tests use `tmp_path` as an allowed root,
    and the root rule of #66 refuses a root there. Child processes inherit the new TMPDIR.
    """
    if not temp_dir_bars_tmp_path(Path(tempfile.gettempdir()).resolve()):
        return
    for variable in TEMP_DIR_VARIABLES:
        os.environ.pop(variable, None)
    tempfile.tempdir = None
    fallback = Path(tempfile.gettempdir()).resolve()
    if temp_dir_bars_tmp_path(fallback):
        raise pytest.UsageError(f"no temp dir outside a denied name, last tried: {fallback}")
    os.environ["TMPDIR"] = str(fallback)
    tempfile.tempdir = str(fallback)


def pytest_configure(config: pytest.Config) -> None:
    use_a_temp_dir_outside_denied_names()


GUARD_ENV = (*guard.OFFLINE_ENV, *guard.TELEMETRY_OFF_ENV)


def signal_handlers() -> dict[signal.Signals, object]:
    """The handler of every signal Python can read one of (`None`: set outside Python)."""
    handlers = {}
    for number in signal.valid_signals():
        try:
            handlers[number] = signal.getsignal(number)
        except (OSError, ValueError):
            continue
    return handlers


@pytest.fixture(autouse=True)
def reset_process_globals() -> Iterator[None]:
    """Undo the process-wide state product code changes in a test, so it cannot reach the next
    one (#172): the network guard, its denials and its environment variables, signal handlers,
    `tempfile.tempdir` and the current directory. The audit hook of the guard stays in the
    process: it denies nothing while the guard is off."""
    env = {name: os.environ.get(name) for name in GUARD_ENV}
    handlers, temp_dir, cwd = signal_handlers(), tempfile.tempdir, os.getcwd()
    yield
    guard.reset_for_tests()
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    for number, before in handlers.items():
        if before is not None and signal.getsignal(number) is not before:
            signal.signal(number, before)
    tempfile.tempdir = temp_dir
    os.chdir(cwd)


SHORT_SOCKET_PARENT = "/tmp"
SHORT_SOCKET_PARENT_VARIABLE = "SOUND_SCOUT_TEST_SOCKET_PARENT"
SHORT_SOCKET_PREFIX = "sst-"


@pytest.fixture
def short_socket_dir() -> Iterator[Path]:
    """A directory with a short path for `AF_UNIX` sockets (#196). Linux takes at most 107 bytes
    for a socket path, and `tmp_path` under a long TMPDIR is longer. The test removes it after."""
    parent = os.environ.get(SHORT_SOCKET_PARENT_VARIABLE, SHORT_SOCKET_PARENT)
    path = Path(tempfile.mkdtemp(prefix=SHORT_SOCKET_PREFIX, dir=parent))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def node() -> str:
    """The `node` binary for the Desktop plugin suites in `tests/js/`; CI fails without it."""
    path = shutil.which("node")
    if path is None:
        if os.environ.get("CI"):
            pytest.fail("node is not installed: CI must run the tests/js suites, not skip them")
        pytest.skip("node is not installed")
    return path


LOOP_HEADER = re.compile(r"^ *for \(const .+? of (?P<rest>\[.*)$")
TEST_CALL = re.compile(r"^ *test\(")


def _array_length(text: str) -> int:
    """Elements of the array literal at the start of `text`: top-level commas, strings skipped."""
    depth, count, quote, empty, escaped = 0, 0, None, True, False
    for char in text:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
        elif char in "[({":
            depth += 1
            if depth == 1:
                continue
        elif char in "])}":
            depth -= 1
            if depth == 0:
                return count + (0 if empty else 1)
        elif char == "," and depth == 1:
            count, empty = count + 1, True
            continue
        if not char.isspace():
            empty = False
    raise ValueError("unclosed array literal in a for loop header")


def declared_node_tests(source: str) -> int:
    """Tests a `node --test` suite declares: each top-level `test(` call once, each one inside
    `for (const … of [literal])` loops (top-level or nested) once per loop element. Test bodies
    end at the `})` line of the call's indent and are skipped. Any other `test(` line raises
    `ValueError`: the count would not match what node runs."""
    lines = source.splitlines()
    loops: list[tuple[int, int]] = []
    total, test_body = 0, None
    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        if test_body is not None:
            if indent == test_body and line.lstrip().startswith("})"):
                test_body = None
            continue
        inside = indent > loops[-1][0] if loops else indent == 0
        multiplier = loops[-1][1] if loops else 1
        if (header := LOOP_HEADER.match(line)) and inside:
            rest = "\n".join([header["rest"], *lines[index + 1 :]])
            loops.append((indent, multiplier * _array_length(rest)))
        elif loops and indent == loops[-1][0] and line.strip() == "}":
            loops.pop()
        elif TEST_CALL.match(line):
            if not inside:
                raise ValueError(f"line {index + 1}: test() call outside a counted loop")
            total += multiplier
            test_body = None if line.rstrip().endswith(")") else indent
    return total


def assert_node_suite_passes(node: str, suite: Path, cwd: Path) -> int:
    """Run `suite` with `node --test`: every declared test passes, none is skipped or todo."""
    result = subprocess.run(
        [node, "--experimental-vm-modules", "--test", str(suite)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
    assert result.returncode == 0, tail
    declared = declared_node_tests(suite.read_text(encoding="utf-8"))
    assert declared > 0, f"{suite.name} declares no test() call"
    passed = re.search(r"^[#ℹ] pass (\d+)$", result.stdout, re.MULTILINE)
    assert passed and int(passed[1]) == declared, f"declared {declared}\n{tail}"
    for status in ("skipped|skip", "todo"):
        assert re.search(rf"^[#ℹ] (?:{status}) 0$", result.stdout, re.MULTILINE), tail
    return declared


@functools.cache
def ffmpeg_encoders() -> frozenset[str]:
    """The encoder names the installed ffmpeg lists; empty without ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return frozenset()
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    ).stdout
    _, _, entries = listing.partition(" ------\n")
    return frozenset(line.split()[1] for line in entries.splitlines() if line.strip())


def require_encoders(*names: str) -> None:
    """Skip the test, naming the encoder, when ffmpeg lacks one of `names`."""
    missing = [name for name in names if name not in ffmpeg_encoders()]
    if missing:
        pytest.skip(f"ffmpeg encoder not available: {', '.join(missing)}")


@pytest.fixture
def make_wav(tmp_path: Path) -> WavFactory:
    """Synthesize a 16-bit PCM mono WAV (a sine tone) in the temp dir and return its path."""

    def _make_wav(
        name: str = "tone.wav",
        seconds: float = 1.0,
        sample_rate: int = 16_000,
        frequency: float = 220.0,
    ) -> Path:
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * np.iinfo(np.int16).max).astype(
            np.int16
        )
        path = tmp_path / name
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            out.writeframes(samples.tobytes())
        return path

    return _make_wav


@pytest.fixture
def wav_file(make_wav: WavFactory) -> Path:
    return make_wav()


@pytest.fixture
def encode() -> Encoder:
    """Transcode a WAV with ffmpeg: `encode(source, target, "libopus", *more ffmpeg args)`.

    `args` go between the input and the audio encoder, so a second `-i`, `-map` or `-c:v` fit
    there. A missing encoder skips the test naming it; extra encoders named in `args` are the
    test's own `require_encoders` call.
    """

    def _encode(source: Path, target: Path, encoder: str, *args: str) -> Path:
        require_encoders(encoder)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), *args, "-c:a", encoder, str(target)],
            check=True,
            capture_output=True,
        )
        return target

    return _encode
