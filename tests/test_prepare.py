"""`prepare.mono16k`: the 16 kHz derivative for ASR, its record, idempotency and failure paths."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from sound_scout import acoustics, channels, ingest, prepare, proc, schema
from sound_scout.config import MIB, Budgets, Config
from sound_scout.prepare import PrepareError, mono16k
from sound_scout.runs import Run, open_run
from tests.test_ingest import write_wav
from tests.test_proc import process_gone

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)

SR = prepare.SAMPLE_RATE
IDENTITY = {"audio": None, "model": None, "preprocessing": None, "ranges": None}
DURATION_S = 2.5
ONSET_S = 1.25
TRANSFORMATION_SCHEMA = schema.SCHEMA["properties"]["provenance"]["properties"]["transformations"][
    "items"
]


def burst(rate: int, seconds: float = DURATION_S, onset: float = ONSET_S) -> np.ndarray:
    """Digital silence, then a 440 Hz tone from `onset` to the end; float32 at `rate`."""
    t = np.arange(int(seconds * rate)) / rate
    signal = np.zeros(len(t), dtype=np.float32)
    start = int(onset * rate)
    signal[start:] = (0.5 * np.sin(2 * np.pi * 440.0 * t[start:])).astype(np.float32)
    return signal


def onset_of(samples: np.ndarray, rate: int) -> float:
    return int(np.argmax(np.abs(samples) > 0.05)) / rate


def work_entries(run: Run) -> list[str]:
    work = run.path / "work"
    return sorted(p.name for p in work.iterdir()) if work.exists() else []


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "recordings"
    root.mkdir()
    return root


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    data_root.mkdir()
    return data_root


@pytest.fixture
def config(root: Path, data_root: Path) -> Config:
    return Config(allowed_roots=(root,), data_root=data_root)


@pytest.fixture
def run(data_root: Path) -> Run:
    return open_run("req-1", IDENTITY, data_root)


def install_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> Path:
    """A shell script named `ffmpeg` first on PATH that runs `script` for every call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "ffmpeg"
    exe.write_text(f"#!/bin/sh\n{script}\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return exe


@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """A fake `ffmpeg` on PATH that answers `-version` and runs `script` for anything else."""

    def _install(script: str) -> Path:
        return install_ffmpeg(
            tmp_path,
            monkeypatch,
            f'if [ "$1" = "-version" ]; then echo "ffmpeg version 0.0-fake"; exit 0; fi\n{script}',
        )

    return _install


def never_run(*args, **kwargs):
    raise AssertionError("ffmpeg must not run for a derivative that is already in the run")


# --- the derivative and its record ------------------------------------------------------------


@pytest.mark.parametrize("rate", [48_000, 44_100])
def test_mono16k_writes_the_derivative_and_its_record(rate: int, run: Run, config: Config) -> None:
    mono = burst(rate)

    result = mono16k(mono, rate, run, config)

    assert result.reused is False
    assert result.path == run.path / "work" / "mono16k.f32"
    assert result.record_path == run.path / "work" / "mono16k.json"
    assert work_entries(run) == ["mono16k.f32", "mono16k.json"]
    assert stat.S_IMODE((run.path / "work").stat().st_mode) == 0o700
    assert stat.S_IMODE(result.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.record_path.stat().st_mode) == 0o600
    assert result.samples.dtype == np.float32 and result.samples.ndim == 1
    assert result.samples.flags.writeable
    assert np.array_equal(np.fromfile(result.path, dtype="<f4"), result.samples)

    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record == result.record
    assert record["kind"] == "mono16k" and record["run_id"] == run.run_id
    assert record["path"] == "work/mono16k.f32" and record["format"] == "f32le"
    assert (record["sample_rate"], record["channels"]) == (SR, 1)
    assert record["frames"] == len(result.samples) == int(DURATION_S * SR)
    assert record["duration_s"] == len(result.samples) / SR
    assert record["sha256"] == sha256_of(result.path) == result.sha256
    assert record["source"] == {
        "sample_rate": rate,
        "channels": 1,
        "frames": len(mono),
        "duration_s": len(mono) / rate,
        "sha256": hashlib.sha256(mono.tobytes()).hexdigest(),
    }
    settings = record["transformation"]["settings"]
    assert record["transformation"]["name"] == "resample_16k_mono"
    assert settings["source_sample_rate"] == rate and settings["target_sample_rate"] == SR
    assert settings["source_frames"] == len(mono) and settings["frames"] == record["frames"]
    assert settings["resampler"] == "swr"
    assert settings["resampler_settings"] == prepare.RESAMPLER_SETTINGS
    assert settings["filter"].startswith("aresample=osr=16000:resampler=swr:")
    assert all(f"{k}={v}" in settings["filter"] for k, v in prepare.RESAMPLER_SETTINGS.items())
    assert settings["sha256"] == record["sha256"] and settings["gain_db"] == 0.0
    assert settings["duration_tolerance_s"] == 0.001
    assert abs(settings["duration_mismatch_s"]) < 0.001
    first_line = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    assert first_line == f"ffmpeg version {settings['ffmpeg_version']}" or first_line.startswith(
        f"ffmpeg version {settings['ffmpeg_version']} "
    )
    assert record["limits"] == {
        "max_duration_seconds": config.budgets.max_duration_seconds,
        "decoded_samples_limit": config.budgets.max_duration_seconds * SR,
        "max_subprocess_seconds": config.budgets.max_subprocess_seconds,
        "max_subprocess_memory_mib": config.budgets.max_subprocess_memory_mib,
    }


def test_transformation_has_the_shape_of_provenance_transformations(
    run: Run, config: Config
) -> None:
    result = mono16k(burst(48_000), 48_000, run, config)

    Draft202012Validator(TRANSFORMATION_SCHEMA).validate(result.transformation)
    json.dumps(result.transformation, allow_nan=False)


@pytest.mark.parametrize("rate", [48_000, 44_100, 22_050, 16_000])
def test_duration_and_event_offset_match_the_source_within_a_frame(
    rate: int, run: Run, config: Config
) -> None:
    mono = burst(rate)

    result = mono16k(mono, rate, run, config)

    assert abs(len(result.samples) / SR - len(mono) / rate) < 0.001
    assert abs(onset_of(result.samples, SR) - ONSET_S) < 0.001
    assert abs(float(np.max(np.abs(result.samples))) - 0.5) < 0.01


def test_the_source_is_not_changed_and_measurements_stay_on_the_original(
    root: Path, run: Run, config: Config
) -> None:
    """The channel report and acoustics work on the decoded original; ASR gets the derivative."""
    rate = 48_000
    write_wav(root / "burst.wav", (burst(rate) * 32767).astype(np.int16), rate)
    source = ingest.decode(root / "burst.wav", run, config)
    audio = channels.load(source)
    report = channels.report(audio)
    mono, choice = channels.choose_mono(audio, report)
    before = (mono.copy(), sha256_of(source.decoded_path), source.record_path.read_bytes())

    derivative = mono16k(mono, source.probe.sample_rate, run, config)
    measured = acoustics.measure(mono, source.probe.sample_rate)

    assert np.array_equal(mono, before[0])
    assert (sha256_of(source.decoded_path), source.record_path.read_bytes()) == before[1:]
    assert channels.report(channels.load(source)) == report
    assert choice["settings"]["frames"] == len(mono) == source.frames
    assert measured["status"] in {"completed", "unavailable"}, measured["reason"]
    duration = next(m for m in measured["measurements"] if m["name"] == "duration_seconds")
    assert duration["value"] == source.duration_s
    assert len(derivative.samples) == int(source.duration_s * SR) != source.frames
    assert work_entries(run) == ["decoded.f32", "mono16k.f32", "mono16k.json"]
    assert derivative.record["source"]["sha256"] == hashlib.sha256(mono.tobytes()).hexdigest()


def test_ffmpeg_runs_through_proc_run_with_argv_whitelist_and_budgets(
    run: Run, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 1: the resample is a `proc.run` call (argv list, no shell) with the protocol
    whitelist before the input and the subprocess budgets of the configuration."""
    calls: list[tuple[list[str], dict]] = []

    def spy(argv, **kwargs):
        calls.append((argv, kwargs))
        return proc.run(argv, **kwargs)

    monkeypatch.setattr(prepare, "run", spy)
    budgets = Budgets(
        max_duration_seconds=7, max_subprocess_seconds=11, max_subprocess_memory_mib=333
    )
    tight = Config(config.allowed_roots, config.data_root, budgets)

    result = mono16k(burst(48_000), 48_000, run, tight)

    assert result.reused is False and len(calls) == 2
    for argv, kwargs in calls:
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
        assert argv[0] == "ffmpeg"
        assert kwargs["timeout"] == 11 and kwargs["memory_bytes"] == 333 * MIB
    argv, kwargs = calls[1]
    assert kwargs["max_output_bytes"] == 7 * SR * 4
    assert "-nostdin" in argv
    inputs = [i for i, a in enumerate(argv) if a == "-i"]
    assert len(inputs) == 1
    assert argv[inputs[0] - 2 : inputs[0]] == ["-protocol_whitelist", "file,pipe"]
    source = Path(argv[inputs[0] + 1])
    assert source.parent == run.path / "work" and not source.exists()
    assert argv[-2:] == ["-f", "-"] or argv[-3:] == ["-f", "f32le", "-"]
    assert argv[argv.index("-af") + 1] == prepare.resample_filter()
    assert not any(c in a for a in argv for c in "|;&$`")


def test_mono16k_resamples_with_the_network_guard_installed(
    run: Run, config: Config, tmp_path: Path
) -> None:
    """Criterion 1: the process-wide network ban (R09) is irreversible, so a child interpreter
    installs it and builds the derivative; the record it writes is checked here."""
    script = f"""
import json, os
import numpy as np
from pathlib import Path
from sound_scout import guard, prepare
from sound_scout.config import Config
from sound_scout.runs import Run
config = Config((Path(os.environ["T_ROOT"]),), Path(os.environ["T_DATA"]))
assert guard.install_network_guard(config) is True
rate = 44_100
t = np.arange(int({DURATION_S} * rate)) / rate
mono = np.where(t >= {ONSET_S}, 0.5 * np.sin(2 * np.pi * 440.0 * t), 0.0).astype(np.float32)
run = Run(os.environ["T_RUN_ID"], Path(os.environ["T_RUN"]), "req", "hash", False, False, True)
result = prepare.mono16k(mono, rate, run, config)
out = {{"frames": len(result.samples), "reused": result.reused}}
out["guard"] = guard.network_guard_active()
print(json.dumps(out))
"""
    env = {
        **os.environ,
        "T_ROOT": str(config.allowed_roots[0]),
        "T_DATA": str(config.data_root),
        "T_RUN_ID": run.run_id,
        "T_RUN": str(run.path),
    }
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=env,
    )

    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out == {"frames": int(DURATION_S * SR), "reused": False, "guard": True}
    assert work_entries(run) == ["mono16k.f32", "mono16k.json"]
    derivative = np.fromfile(run.path / "work" / "mono16k.f32", dtype="<f4")
    record = json.loads((run.path / "work" / "mono16k.json").read_text(encoding="utf-8"))
    assert record["sha256"] == sha256_of(run.path / "work" / "mono16k.f32")
    assert abs(onset_of(derivative, SR) - ONSET_S) < 0.001


# --- idempotency ------------------------------------------------------------------------------


def test_a_repeated_call_returns_the_written_derivative_without_ffmpeg(
    run: Run, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    mono = burst(44_100)
    first = mono16k(mono, 44_100, run, config)
    stamps = (first.path.stat().st_mtime_ns, first.record_path.stat().st_mtime_ns)
    monkeypatch.setattr(prepare, "run", never_run)

    second = mono16k(mono, 44_100, run, config)

    assert second.reused is True and first.reused is False
    assert np.array_equal(second.samples, first.samples)
    assert second.record == first.record
    assert (second.path, second.record_path) == (first.path, first.record_path)
    assert (first.path.stat().st_mtime_ns, first.record_path.stat().st_mtime_ns) == stamps


def test_another_mono_for_the_same_run_is_refused_not_overwritten(run: Run, config: Config) -> None:
    first = mono16k(burst(48_000), 48_000, run, config)
    before = (sha256_of(first.path), first.record_path.read_bytes())

    with pytest.raises(PrepareError, match="built from another mono signal") as info:
        mono16k(burst(48_000, onset=0.5), 48_000, run, config)

    assert info.value.code == "source_mismatch"
    assert (sha256_of(first.path), first.record_path.read_bytes()) == before


def test_the_same_mono_at_another_rate_is_refused_not_reused(
    run: Run, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    mono = burst(48_000)
    first = mono16k(mono, 48_000, run, config)
    before = (sha256_of(first.path), first.record_path.read_bytes())
    monkeypatch.setattr(prepare, "run", never_run)

    with pytest.raises(PrepareError, match="another mono signal .*at 48000 Hz") as info:
        mono16k(mono, 44_100, run, config)

    assert info.value.code == "source_mismatch"
    assert (sha256_of(first.path), first.record_path.read_bytes()) == before


def test_a_derivative_that_does_not_match_its_record_is_refused(run: Run, config: Config) -> None:
    mono = burst(48_000)
    first = mono16k(mono, 48_000, run, config)
    first.path.write_bytes(first.path.read_bytes()[:-4])

    with pytest.raises(PrepareError, match="does not match its record") as info:
        mono16k(mono, 48_000, run, config)

    assert info.value.code == "derivative_corrupt"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file without permission bits")
def test_an_unreadable_derivative_is_refused_as_corrupt(run: Run, config: Config) -> None:
    mono = burst(48_000)
    first = mono16k(mono, 48_000, run, config)
    first.path.chmod(0)

    with pytest.raises(PrepareError, match="work/mono16k.f32 is not readable") as info:
        mono16k(mono, 48_000, run, config)

    assert info.value.code == "derivative_corrupt"


def test_a_derivative_without_its_record_is_refused(run: Run, config: Config) -> None:
    mono = burst(48_000)
    first = mono16k(mono, 48_000, run, config)
    first.record_path.unlink()

    with pytest.raises(PrepareError, match="work/mono16k.json is missing") as info:
        mono16k(mono, 48_000, run, config)

    assert info.value.code == "derivative_corrupt"
    assert work_entries(run) == ["mono16k.f32"]


# --- failure paths leave no partial derivative --------------------------------------------------


@pytest.mark.parametrize(
    ("mono", "rate", "message"),
    [
        (np.full(100, np.nan, dtype=np.float32), 48_000, "finite"),
        (np.zeros(100, dtype=np.float64), 48_000, "float32"),
        (np.zeros((100, 2), dtype=np.float32), 48_000, "one-dimensional"),
        (np.zeros(0, dtype=np.float32), 48_000, "empty"),
        (np.zeros(100, dtype=np.float32), 0, "positive integer"),
        (np.zeros(100, dtype=np.float32), 48_000.0, "positive integer"),
    ],
)
def test_bad_audio_is_refused_before_anything_runs(
    mono: np.ndarray, rate: object, message: str, run: Run, config: Config, monkeypatch
) -> None:
    monkeypatch.setattr(prepare, "run", never_run)

    with pytest.raises(PrepareError, match=message) as info:
        mono16k(mono, rate, run, config)

    assert info.value.code == "bad_audio"
    assert work_entries(run) == []


def test_ffmpeg_failure_is_reported_with_its_reason_and_leaves_nothing(
    run: Run, config: Config, fake_ffmpeg: Callable[[str], Path]
) -> None:
    fake_ffmpeg(
        'head -c 4000 /dev/zero\necho "Invalid data found when processing input" >&2\nexit 1'
    )

    with pytest.raises(PrepareError, match="ffmpeg failed: Invalid data found") as info:
        mono16k(burst(48_000), 48_000, run, config)

    assert info.value.code == "resample_failed"
    assert work_entries(run) == []
    assert sorted(p.name for p in run.path.iterdir()) == ["manifest.json", "request.json"]


def test_a_signal_over_the_duration_budget_is_refused_not_cut(run: Run, config: Config) -> None:
    tight = Config(config.allowed_roots, config.data_root, Budgets(max_duration_seconds=1))

    with pytest.raises(PrepareError, match=f"more than {SR} samples resampled") as info:
        mono16k(burst(48_000), 48_000, run, tight)

    assert info.value.code == "decoded_samples_exceeded"
    assert work_entries(run) == []


def test_a_zero_duration_budget_is_an_explicit_error_and_leaves_work_as_it_was(
    run: Run, config: Config
) -> None:
    """Criterion 4: a zero budget refuses the resample with its reason; a `work/` that already
    holds the decoded original keeps only that, no partial derivative and no temp source file."""
    (run.path / "work").mkdir(0o700)
    (run.path / "work" / "decoded.f32").write_bytes(b"\0" * 16)
    zero = Config(config.allowed_roots, config.data_root, Budgets(max_duration_seconds=0))

    with pytest.raises(PrepareError, match="more than 0 samples resampled") as info:
        mono16k(burst(48_000), 48_000, run, zero)

    assert info.value.code == "decoded_samples_exceeded"
    assert "0 s at 16000 Hz" in str(info.value)
    assert work_entries(run) == ["decoded.f32"]
    assert (run.path / "work" / "decoded.f32").read_bytes() == b"\0" * 16


def test_a_timeout_is_reported_with_the_budget_and_the_child_is_gone(
    run: Run, config: Config, fake_ffmpeg: Callable[[str], Path], tmp_path: Path
) -> None:
    pid_file = tmp_path / "pid"
    fake_ffmpeg(f"echo $$ > {pid_file}\nexec sleep 30")
    slow = Config(config.allowed_roots, config.data_root, Budgets(max_subprocess_seconds=1))

    with pytest.raises(PrepareError, match="ffmpeg exceeded 1 s") as info:
        mono16k(burst(48_000), 48_000, run, slow)

    assert info.value.code == "subprocess_timeout"
    assert process_gone(int(pid_file.read_text()))
    assert work_entries(run) == []


def test_a_version_probe_that_hangs_is_a_timeout(
    run: Run, config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "pid"
    install_ffmpeg(tmp_path, monkeypatch, f"echo $$ > {pid_file}\nexec sleep 30")
    slow = Config(config.allowed_roots, config.data_root, Budgets(max_subprocess_seconds=1))

    with pytest.raises(PrepareError, match="ffmpeg exceeded 1 s") as info:
        mono16k(burst(48_000), 48_000, run, slow)

    assert info.value.code == "subprocess_timeout"
    assert process_gone(int(pid_file.read_text()))
    assert work_entries(run) == []


def test_a_version_probe_without_a_version_line_is_decoder_unavailable(
    run: Run, config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_ffmpeg(tmp_path, monkeypatch, "echo not ffmpeg; exit 0")

    with pytest.raises(PrepareError, match="unrecognized output") as info:
        mono16k(burst(48_000), 48_000, run, config)

    assert info.value.code == "decoder_unavailable"
    assert work_entries(run) == []


def test_a_wrong_length_from_ffmpeg_is_a_duration_mismatch(
    run: Run, config: Config, fake_ffmpeg: Callable[[str], Path]
) -> None:
    fake_ffmpeg(f"head -c {4 * SR} /dev/zero")

    with pytest.raises(PrepareError, match=f"produced {SR} frames") as info:
        mono16k(burst(48_000), 48_000, run, config)

    assert info.value.code == "duration_mismatch"
    assert work_entries(run) == []


def test_missing_ffmpeg_is_unavailable(
    run: Run, config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(PrepareError, match="ffmpeg is not installed") as info:
        mono16k(burst(48_000), 48_000, run, config)

    assert info.value.code == "decoder_unavailable"
    assert work_entries(run) == []


def test_a_failed_record_write_removes_the_derivative(
    run: Run, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    (run.path / "work").mkdir(0o700)
    real_open_output = prepare.paths.open_output

    def failing(run_dir: Path, name: str, config: Config):
        if name == "mono16k.json":
            raise OSError("no space left on device")
        return real_open_output(run_dir, name, config)

    monkeypatch.setattr(prepare.paths, "open_output", failing)

    with pytest.raises(OSError, match="no space left"):
        mono16k(burst(48_000), 48_000, run, config)

    assert work_entries(run) == []
    assert (run.path / "work").is_dir()


def test_derivatives_go_only_into_a_run_dir_under_the_data_root(
    tmp_path: Path, config: Config
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    stray = Run("0" * 32, outside, "req", "hash", False, False, True)

    with pytest.raises(prepare.paths.PathDenied, match="outside data root"):
        mono16k(burst(48_000), 48_000, stray, config)

    assert list(outside.iterdir()) == []
