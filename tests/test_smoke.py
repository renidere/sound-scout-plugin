"""Smoke runs of `decoder` and `acoustics` (issue #46): the real ffmpeg, Silero VAD and Praat.

The success tests run the components for real on synthetic signals and are the capability
proof behind `ready`. The refusal tests substitute a measurement result, a broken input or a
sleeping decoder only to show that a failed run neither writes nor touches a receipt; they
prove the refusal path, not a capability.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from sound_scout import acoustics, doctor, guard, ingest, smoke
from sound_scout.acoustics import VadUnavailable
from sound_scout.config import MIB, Budgets, Config
from sound_scout.smoke import SmokeFailed

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)
EARLIER = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    recordings, data = tmp_path / "recordings", tmp_path / "data"
    recordings.mkdir()
    data.mkdir()
    return Config(allowed_roots=(recordings.resolve(),), data_root=data.resolve())


@pytest.fixture
def acoustics_present(config: Config) -> None:
    state = doctor.component_status("acoustics", config, config.data_root)
    if state["status"] == "unavailable":
        pytest.skip(f"acoustics is unavailable here: {state['reason']}")


def receipt_path(config: Config, component: str) -> Path:
    return config.data_root / doctor.RECEIPTS_DIR_NAME / f"{component}.json"


def status(config: Config, component: str) -> dict:
    return doctor.report(config)["components"][component]


def assert_only_receipts_in_data_root(config: Config) -> None:
    assert sorted(os.listdir(config.data_root)) == [doctor.RECEIPTS_DIR_NAME]


def assert_private(config: Config, receipt: Path) -> None:
    text = receipt.read_text(encoding="utf-8")
    for forbidden in (str(config.data_root), str(Path.home()), smoke.SMOKE_DIR_PREFIX):
        assert forbidden not in text
    assert doctor.TEST_DOUBLE not in text


def stale_receipt(
    component: str, config: Config, monkeypatch: pytest.MonkeyPatch
) -> tuple[bytes, str]:
    """A receipt of an earlier smoke run that the current fingerprint no longer matches."""
    earlier = {"test": "earlier run", "input": "synthetic"}
    if component == "decoder":
        other_budgets = Budgets(max_subprocess_seconds=config.budgets.max_subprocess_seconds + 1)
        path = doctor.write_receipt(
            "decoder", replace(config, budgets=other_budgets), earlier, now=EARLIER
        )
    else:
        real_version = doctor.installed_version
        with monkeypatch.context() as patch:
            patch.setattr(doctor, "installed_version", lambda name: f"{real_version(name)}-earlier")
            path = doctor.write_receipt("acoustics", config, earlier, now=EARLIER)
    before = status(config, component)
    assert before["status"] == "unvalidated" and "stale" in before["reason"]
    return path.read_bytes(), before["last_smoke_at"]


def assert_kept(config: Config, component: str, saved: tuple[bytes, str]) -> None:
    content, last_smoke_at = saved
    assert receipt_path(config, component).read_bytes() == content
    after = status(config, component)
    assert after["status"] == "unvalidated"
    assert after["last_smoke_at"] == last_smoke_at == EARLIER.isoformat()
    assert_only_receipts_in_data_root(config)


@needs_ffmpeg
def test_decoder_smoke_writes_the_receipt_and_doctor_reports_ready(config: Config) -> None:
    assert status(config, "decoder")["status"] == "unvalidated"

    path = smoke.decoder(config)

    assert path == receipt_path(config, "decoder")
    assert status(config, "decoder")["status"] == "ready"
    recorded = json.loads(path.read_text(encoding="utf-8"))["smoke"]
    assert recorded["test"] == "sound_scout.smoke.decoder"
    assert recorded["input"] == "synthetic"
    assert recorded["signal"] == {
        "format": "wav_pcm_s16le",
        "sample_rate": 48_000,
        "channels": 1,
        "seconds": 1.0,
        "frequency_hz": 440.0,
        "amplitude": 0.5,
    }
    measured = recorded["measured"]
    assert measured["frames"] == 48_000
    assert measured["duration_s"] == 1.0
    assert (measured["container"], measured["codec"]) == ("wav", "pcm_s16le")
    assert measured["max_abs_error"] <= smoke.DECODER_MAX_ABS_ERROR
    assert_private(config, path)
    assert_only_receipts_in_data_root(config)


def test_acoustics_smoke_writes_the_receipt_and_doctor_reports_ready(
    config: Config, acoustics_present: None
) -> None:
    path = smoke.acoustics(config)

    assert path == receipt_path(config, "acoustics")
    assert status(config, "acoustics")["status"] == "ready"
    recorded = json.loads(path.read_text(encoding="utf-8"))["smoke"]
    assert recorded["test"] == "sound_scout.smoke.acoustics"
    assert recorded["input"] == "synthetic"
    assert recorded["signal"]["control"] == "praat_klattgrid_formant_synthesis"
    assert recorded["tolerance"] == {"f0_rel": 0.03, "span_start_s": 0.05, "span_end_s": 0.064}
    measured = recorded["measured"]
    assert measured["f0_hz"]["p50"] == pytest.approx(150.0, rel=0.03)
    assert measured["span"]["start"] == pytest.approx(1.0, abs=0.05)
    assert measured["span"]["end"] == pytest.approx(2.2, abs=0.064)
    assert measured["voiced_speech_frames"] > 50
    assert measured["silence_spans"] == 0
    assert_private(config, path)
    assert_only_receipts_in_data_root(config)


@needs_ffmpeg
def test_decoder_smoke_timeout_keeps_the_old_receipt(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, budgets=Budgets(max_subprocess_seconds=1))
    saved = stale_receipt("decoder", config, monkeypatch)
    sleepers = tmp_path / "bin"
    sleepers.mkdir()
    for tool in ("ffmpeg", "ffprobe"):
        script = sleepers / tool
        script.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
        script.chmod(0o755)
    real_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", f"{sleepers}{os.pathsep}{real_path}")

    with pytest.raises(SmokeFailed) as failure:
        smoke.decoder(config)

    monkeypatch.setenv("PATH", real_path)
    assert failure.value.component == "decoder"
    assert failure.value.reason.startswith(f"{ingest.SUBPROCESS_TIMEOUT}: ")
    assert str(failure.value).startswith("decoder smoke failed: subprocess_timeout")
    assert_kept(config, "decoder", saved)


@needs_ffmpeg
def test_decoder_smoke_on_a_broken_input_keeps_the_old_receipt(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = stale_receipt("decoder", config, monkeypatch)
    real_write = smoke._write_wav

    def truncated(path: Path, samples: np.ndarray, rate: int) -> None:
        real_write(path, samples, rate)
        with open(path, "r+b") as file:
            file.truncate(os.path.getsize(path) // 2)

    monkeypatch.setattr(smoke, "_write_wav", truncated)

    with pytest.raises(SmokeFailed) as failure:
        smoke.decoder(config)

    assert failure.value.reason.startswith(f"{ingest.TRUNCATED}: ")
    assert_kept(config, "decoder", saved)


@needs_ffmpeg
def test_decoder_smoke_passes_with_a_data_root_below_a_denied_name(tmp_path: Path) -> None:
    """A data root like `~/.hermes/sound-scout`: the generated input is no user recording."""
    recordings, data = tmp_path / "recordings", tmp_path / ".hermes" / "data"
    recordings.mkdir()
    data.mkdir(parents=True)
    config = Config(allowed_roots=(recordings.resolve(),), data_root=data.resolve())

    assert smoke.decoder(config) == receipt_path(config, "decoder")

    assert status(config, "decoder")["status"] == "ready"
    assert_only_receipts_in_data_root(config)


def shifted_sample(audio: np.ndarray) -> np.ndarray:
    audio[1000, 0] += 1e-3
    return audio


@needs_ffmpeg
@pytest.mark.parametrize(
    ("alter", "expected"),
    [
        (shifted_sample, "decoded samples differ from the written ones by 0.000999"),
        (lambda audio: audio[:-1], "decoded array frames is 47999"),
    ],
    ids=["shifted_sample", "truncated_array"],
)
def test_decoder_smoke_refuses_samples_that_differ_from_the_written_ones(
    alter, expected: str, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = stale_receipt("decoder", config, monkeypatch)
    real_load = smoke.channels.load
    monkeypatch.setattr(smoke.channels, "load", lambda source: alter(real_load(source)))

    with pytest.raises(SmokeFailed) as failure:
        smoke.decoder(config)

    assert failure.value.component == "decoder"
    assert failure.value.reason.startswith(expected)
    assert_kept(config, "decoder", saved)


def altered_measure(change):
    """The real `measure`, its result then changed by `change(signal, values_by_name)`."""
    real = smoke.measure

    def wrapper(signal: np.ndarray, rate: int) -> dict:
        result = real(signal, rate)
        change(signal, {item["name"]: item for item in result["measurements"]})
        return result

    return wrapper


def test_acoustics_smoke_refuses_f0_outside_the_tolerance(
    config: Config, acoustics_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = stale_receipt("acoustics", config, monkeypatch)

    def pitch_off(signal, values):
        if values["f0_hz"]["value"] is not None:
            values["f0_hz"]["value"].update(p10=160.0, p50=150.0 * 1.1, p90=170.0)

    monkeypatch.setattr(smoke, "measure", altered_measure(pitch_off))

    with pytest.raises(SmokeFailed) as failure:
        smoke.acoustics(config)

    assert failure.value.component == "acoustics"
    reason = failure.value.reason
    assert "f0" in reason and "165.0" in reason and "150.0" in reason
    assert_kept(config, "acoustics", saved)


def test_acoustics_smoke_refuses_spans_on_silence(
    config: Config, acoustics_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = stale_receipt("acoustics", config, monkeypatch)

    def span_in_silence(signal, values):
        if not signal.any():
            values["speech_spans"]["value"] = [{"start": 0.5, "end": 1.0}]

    monkeypatch.setattr(smoke, "measure", altered_measure(span_in_silence))

    with pytest.raises(SmokeFailed) as failure:
        smoke.acoustics(config)

    assert "silence gave speech spans" in failure.value.reason
    assert_kept(config, "acoustics", saved)


@pytest.mark.parametrize("missing", ["vad_asset", "onnxruntime"])
def test_acoustics_smoke_refuses_an_unavailable_stage(
    missing: str, config: Config, acoustics_present: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detector failing at run time, or a package doctor already finds missing."""
    saved = stale_receipt("acoustics", config, monkeypatch)
    with monkeypatch.context() as patch:
        if missing == "vad_asset":

            def no_asset() -> str:
                raise VadUnavailable("Silero VAD asset not found in faster_whisper")

            patch.setattr(acoustics, "vad_asset_path", no_asset)
            expected = "stage is unavailable: Silero VAD asset not found in faster_whisper"
        else:
            real_version = doctor.installed_version
            patch.setattr(
                doctor,
                "installed_version",
                lambda name: None if name == "onnxruntime" else real_version(name),
            )
            expected = "package onnxruntime is not installed"

        with pytest.raises(SmokeFailed) as failure:
            smoke.acoustics(config)

    assert failure.value.reason == expected
    assert_kept(config, "acoustics", saved)


def test_smoke_without_a_data_root_fails_with_the_reason(config: Config) -> None:
    gone = replace(config, data_root=config.data_root / "missing")

    outcomes = smoke.run(gone, ("decoder",))

    assert isinstance(outcomes["decoder"], SmokeFailed)
    assert outcomes["decoder"].reason == "the data root is not an existing directory"
    assert not gone.data_root.exists()


def test_run_goes_on_after_a_failure_and_rejects_unknown_components(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(loaded: Config) -> Path:
        raise SmokeFailed("decoder", "broken on purpose")

    monkeypatch.setattr(smoke, "decoder", failing)
    monkeypatch.setattr(smoke, "acoustics", lambda loaded: Path("ran"))

    outcomes = smoke.run(config)

    assert list(outcomes) == ["decoder", "acoustics"]
    assert outcomes["decoder"].reason == "broken on purpose"
    assert outcomes["acoustics"] == Path("ran")
    with pytest.raises(ValueError, match="asr"):
        smoke.run(config, ("asr",))


@needs_ffmpeg
def test_decoder_smoke_runs_the_decoder_under_the_operator_budgets(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        config, budgets=Budgets(max_subprocess_seconds=17, max_subprocess_memory_mib=1536)
    )
    calls = []
    real_run = ingest.run

    def spy(argv, **limits):
        calls.append((argv[0], limits))
        return real_run(argv, **limits)

    monkeypatch.setattr(ingest, "run", spy)

    smoke.decoder(config)

    assert [program for program, _ in calls] == ["ffprobe", "ffmpeg"]
    for _, limits in calls:
        assert limits["timeout"] == 17
        assert limits["memory_bytes"] == 1536 * MIB
    assert status(config, "decoder")["status"] == "ready"


CHILD = """
import json, os, sys
from pathlib import Path
from sound_scout import guard, smoke
from sound_scout.config import Config

base = Path({base!r})
config = Config(allowed_roots=(base / "recordings",), data_root=base / "data")

def snapshot():
    seen = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if Path(dirpath, name) != base / "data")
        for name in [""] + sorted(filenames):
            path = os.path.join(dirpath, name) if name else dirpath
            info = os.lstat(path)
            seen.append((os.path.relpath(path, base), info.st_size, info.st_mtime_ns))
    return seen

sockets, spawned = [], []
def audit(event, args):
    if event.startswith("socket."):
        sockets.append(event)
    elif event == "subprocess.Popen":
        spawned.append(os.path.basename(str(args[1][0])))

assert guard.install_network_guard(config) is True
sys.addaudithook(audit)
before = snapshot()
outcomes = smoke.run(config)
after = snapshot()
print(json.dumps({{
    "changed": sorted({{entry[0] for entry in set(before) ^ set(after)}}),
    "outcomes": {{name: str(value) for name, value in outcomes.items()}},
    "paths": [isinstance(value, Path) for value in outcomes.values()],
    "sockets": sockets,
    "spawned": spawned,
    "data_root": sorted(os.listdir(base / "data")),
}}))
"""


@needs_ffmpeg
def test_smoke_writes_nothing_outside_the_data_root_under_the_guard(
    config: Config, tmp_path: Path, acoustics_present: None
) -> None:
    home, temp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    temp.mkdir()
    managed = guard.OFFLINE_ENV.keys() | guard.TELEMETRY_OFF_ENV.keys()
    env = {key: value for key, value in os.environ.items() if key not in managed}
    env.update(HOME=str(home), TMPDIR=str(temp))
    script = CHILD.format(base=str(tmp_path.resolve()))

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["paths"] == [True, True], outcome["outcomes"]
    assert outcome["changed"] == []
    assert outcome["sockets"] == []
    assert outcome["spawned"] and set(outcome["spawned"]) <= {"ffmpeg", "ffprobe"}
    assert outcome["data_root"] == [doctor.RECEIPTS_DIR_NAME]
    assert sorted(os.listdir(config.data_root / doctor.RECEIPTS_DIR_NAME)) == [
        "acoustics.json",
        "decoder.json",
    ]
