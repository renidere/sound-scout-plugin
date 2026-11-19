"""Acceptance checks of issue #41: scripts/profile.py measures the real `analyze` (T10, R11)."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "profile.py"

spec = importlib.util.spec_from_file_location("profile_script", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.mark.parametrize(
    ("duration", "budget"), [(5.0, 60), (30.0, 60), (30.1, 180), (180.0, 180), (180.5, None)]
)
def test_budget_follows_r11(duration: float, budget: int | None) -> None:
    assert profile.budget_for(duration) == budget


def test_synthetic_speech_repeats_the_formant_control_every_three_seconds() -> None:
    rate = 16_000
    audio = profile.synthetic_speech(7.0, rate)
    assert audio.dtype == np.float32 and len(audio) == 7 * rate

    def peak(start: float, end: float) -> float:
        return float(np.max(np.abs(audio[int(start * rate) : int(end * rate)])))

    assert peak(0.1, 1.1) > 0.1 and peak(3.1, 4.1) > 0.1 and peak(6.1, 7.0) < 1e-6
    assert peak(1.25, 2.95) == 0.0 and peak(4.25, 5.95) == 0.0


def test_reasons_lose_absolute_paths_but_keep_relative_ones() -> None:
    reason = "cannot read /data/private dir/x.wav (work/stage-asr.json)"
    assert profile.redacted(reason) == "cannot read <path> dir/x.wav (work/stage-asr.json)"
    assert profile.redacted(None) is None


def write_record(run_dir: Path, name: str, status: str, elapsed: int, steps=(), reason=None):
    record = {
        "stage": name,
        "entry": {"requested": True, "status": status, "reason": reason, "elapsed_ms": elapsed},
        "errors": [],
        "steps": [{"name": step, "elapsed_ms": ms} for step, ms in steps],
        "fragment": {},
    }
    (run_dir / "work" / f"stage-{name}.json").write_text(json.dumps(record))


def test_stage_times_read_the_run_records(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "asr").mkdir()
    write_record(tmp_path, "prepare", "completed", 140, [("ingest", 80), ("mono16k", 60)])
    write_record(tmp_path, "acoustics", "completed", 90, [("measure", 90)])
    write_record(tmp_path, "asr", "completed", 5000, [("transcribe", 5000)])
    write_record(tmp_path, "audio_review", "unavailable", 1, reason="backend = none: /home/x")
    timing = {"timing": {"model_load_ms": 1200, "inference_ms": 3700}, "text": "secret"}
    (tmp_path / "asr" / "raw.json").write_text(json.dumps(timing))
    stages = profile.stage_times(tmp_path, {})
    assert stages == {
        "prepare": {
            "status": "completed",
            "elapsed_ms": 140,
            "steps": {"ingest": 80, "mono16k": 60},
        },
        "acoustics": {"status": "completed", "elapsed_ms": 90, "steps": {"measure": 90}},
        "asr": {
            "status": "completed",
            "elapsed_ms": 5000,
            "model_load_ms": 1200,
            "inference_ms": 3700,
        },
        "audio_review": {
            "status": "unavailable",
            "elapsed_ms": None,
            "model_load_ms": None,
            "inference_ms": None,
            "reason": "backend = none: <path>",
        },
    }


def test_stage_times_without_records_keep_the_answer_status(tmp_path: Path) -> None:
    stages = profile.stage_times(tmp_path, {"asr": {"status": "failed"}})
    assert stages["asr"] == {"status": "failed", "elapsed_ms": None, "reason": "нет записи этапа"}
    assert stages["prepare"]["status"] is None


def test_durations_up_to_the_longest_r11_target_are_accepted() -> None:
    assert profile.parse_durations("30,180") == (30.0, 180.0)


def test_profile_config_lifts_the_worker_and_subprocess_deadlines(tmp_path: Path) -> None:
    text = profile.config_text([tmp_path], tmp_path / "data", profile.config.Asr())
    budgets = tomllib.loads(text)["budgets"]
    assert budgets["worker_deadline_seconds"] == profile.PROFILE_DEADLINE_SECONDS
    assert budgets["max_subprocess_seconds"] == profile.PROFILE_DEADLINE_SECONDS


def test_output_inside_the_repository_is_refused(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(REPO_ROOT / "profile.json")],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "--output must be outside the repository" in done.stderr
    assert not (REPO_ROOT / "profile.json").exists()
    with pytest.raises(profile.ProfileError, match="no such directory"):
        profile.checked_output(str(tmp_path / "missing" / "profile.json"))


@pytest.mark.parametrize("text", ["", "30,abc", "0", "-5", "180.5", "4000"])
def test_bad_durations_are_refused(text: str) -> None:
    with pytest.raises(Exception, match="durations|expected seconds"):
        profile.parse_durations(text)


def run_profile(tmp_path: Path, dev_audio: Path | None) -> tuple[subprocess.CompletedProcess, str]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    operator = tmp_path / "operator"
    (operator / "in").mkdir(parents=True)
    (operator / "data").mkdir()
    # a model that is not cached: asr is `unavailable` at once, no model is loaded
    (operator / "sound-scout.toml").write_text(
        f'allowed_roots = ["{operator / "in"}"]\ndata_root = "{operator / "data"}"\n'
        '[asr]\nmodel = "nobody/not-cached"\n'
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {**os.environ, "SOUND_SCOUT_CONFIG": str(operator / "sound-scout.toml")}
    env["TMPDIR"] = str(scratch)
    env.pop("SOUND_SCOUT_DEV_AUDIO", None)
    if dev_audio is not None:
        env["SOUND_SCOUT_DEV_AUDIO"] = str(dev_audio)
    output = tmp_path / "out" / "profile.json"
    output.parent.mkdir()
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--durations", "3"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert list(scratch.iterdir()) == [], "the temporary data root is removed"
    return done, output.read_text(encoding="utf-8") if output.exists() else ""


def test_profile_measures_cold_and_warm_runs_without_private_data(tmp_path: Path) -> None:
    private = tmp_path / "private-recordings"
    private.mkdir()
    recording = private / "secret-name.wav"
    profile.write_wav(recording, profile.synthetic_speech(4.0), profile.SYNTHETIC_RATE)
    done, text = run_profile(tmp_path, recording)
    assert done.returncode == 0, done.stderr
    digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    for private_value in ("secret-name", "private-recordings", digest):
        assert private_value not in text + done.stdout
    assert str(tmp_path) not in text
    result = json.loads(text)
    assert result["profile_version"] == 1
    assert result["config"]["backend"] == "none"
    assert result["config"]["asr_model"] == "nobody/not-cached"
    assert result["config"]["worker_deadline_seconds"] == profile.PROFILE_DEADLINE_SECONDS
    assert result["budget"]["targets"] == [
        {"max_input_s": 30, "max_processing_s": 60},
        {"max_input_s": 180, "max_processing_s": 180},
    ]
    assert "Gate 0 NOT READY" in result["budget"]["scope"]
    assert set(result["host"]) == {
        "os",
        "cpu_model",
        "logical_cpus",
        "memory_total_mib",
        "python",
        "uv",
        "ffmpeg",
    }
    synthetic, development = result["inputs"]
    assert (synthetic["label"], synthetic["duration_s"], synthetic["budget_s"]) == (
        "synthetic 3 s",
        3.0,
        60,
    )
    assert (development["label"], development["kind"]) == ("development recording", "development")
    assert development["duration_s"] == 4.0
    for item in (synthetic, development):
        assert item["status"] == "measured"
        assert [run["phase"] for run in item["runs"]] == ["cold", "warm"]
        for run in item["runs"]:
            assert (run["exit_code"], run["result_status"]) == (3, "partial")
            assert run["wall_ms"] > 0 and run["peak_rss_mib"] > 0
            assert run["within_budget"] is (run["wall_ms"] <= 60_000)
            stages = run["stages"]
            assert stages["prepare"]["status"] == "completed"
            assert set(stages["prepare"]["steps"]) == {"ingest", "channels", "mono16k"}
            assert stages["acoustics"]["steps"]["measure"] >= 0
            assert stages["asr"]["status"] == "unavailable"
            assert stages["asr"]["model_load_ms"] is None
            assert stages["audio_review"] == {
                "status": "unavailable",
                "elapsed_ms": None,
                "model_load_ms": None,
                "inference_ms": None,
                "reason": stages["audio_review"]["reason"],
            }
            assert stages["audio_review"]["reason"].startswith("backend = none")
    assert "synthetic 3 s: measured" in done.stdout


def test_profile_without_the_development_recording_records_it_as_skipped(
    tmp_path: Path,
) -> None:
    done, text = run_profile(tmp_path, None)
    assert done.returncode == 0, done.stderr
    development = json.loads(text)["inputs"][-1]
    assert development == {
        "label": "development recording",
        "kind": "development",
        "status": "skipped",
        "reason": "SOUND_SCOUT_DEV_AUDIO не задана",
    }
