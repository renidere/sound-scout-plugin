"""Acceptance checks of issue #41: scripts/clean-install-check.sh and its expectations.

The script itself is run here against a stand-in `uv` that records how it was called: the real
clean install runs the whole suite, this one included, so it is run by hand and in CI.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
CHECK = SCRIPTS / "clean-install-check.sh"
EXPECT = SCRIPTS / "clean_install_expect.py"

spec = importlib.util.spec_from_file_location("clean_install_expect", EXPECT)
expect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expect)

DOCTOR = {
    "components": {
        "decoder": {"status": "unvalidated"},
        "asr": {"status": "unavailable"},
        "acoustics": {"status": "unvalidated"},
        "audio_review": {"status": "permission_required"},
    },
    "network_guard": {"active": True},
}


def analyze_answer(asr: str = "unavailable", audio_review: str = "unavailable") -> dict:
    stages = {"prepare": "completed", "acoustics": "completed", "asr": asr}
    stages["audio_review"] = audio_review
    return {"status": "partial", "stages": {k: {"status": v} for k, v in stages.items()}}


@pytest.mark.parametrize("asr", ["completed", "unavailable"])
def test_expect_accepts_a_partial_result_with_asr_completed_or_unavailable(asr: str) -> None:
    assert expect.check(DOCTOR, analyze_answer(asr), 3) == []


@pytest.mark.parametrize(
    ("doctor", "analyze", "code", "failure"),
    [
        (DOCTOR, analyze_answer("failed"), 3, 'analyze.stages.asr: expected "completed" or '),
        (DOCTOR, analyze_answer(audio_review="completed"), 3, "analyze.stages.audio_review"),
        (DOCTOR, analyze_answer(), 0, "analyze exit code: expected 3, got 0"),
        (DOCTOR, {**analyze_answer(), "status": "complete"}, 3, "analyze.status"),
        (DOCTOR, None, 3, "analyze.stages.prepare: expected"),
        (
            {**DOCTOR, "network_guard": {"active": False}},
            analyze_answer(),
            3,
            "doctor.network_guard.active: expected true, got false",
        ),
        (
            {"components": {**DOCTOR["components"], "audio_review": {"status": "ready"}}},
            analyze_answer(),
            3,
            "doctor.components.audio_review.status",
        ),
        (
            {**DOCTOR, "components": {"decoder": {"status": "ready"}}},
            analyze_answer(),
            3,
            "doctor.components.asr: expected present, got missing",
        ),
    ],
)
def test_expect_names_each_deviation(doctor, analyze, code: int, failure: str) -> None:
    failures = expect.check(doctor, analyze, code)
    assert any(line.startswith(f"fail: {failure}") for line in failures), failures


def test_expect_cli_prints_ok_details_and_fail_lines(tmp_path: Path) -> None:
    doctor, analyze = tmp_path / "doctor.json", tmp_path / "analyze.json"
    doctor.write_text(json.dumps(DOCTOR))
    analyze.write_text(json.dumps(analyze_answer("completed")))

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", str(EXPECT), *args], capture_output=True, text=True
        )

    assert run(str(doctor), str(analyze), "3").stdout == "ok\n"
    assert run("--detail", "doctor", str(doctor)).stdout == (
        "decoder=unvalidated asr=unavailable acoustics=unvalidated "
        "audio_review=permission_required\n"
    )
    assert run("--detail", "analyze", str(analyze)).stdout == (
        "status=partial prepare=completed acoustics=completed asr=completed "
        "audio_review=unavailable\n"
    )
    failed = run(str(doctor), str(tmp_path / "missing.json"), "1")
    assert failed.returncode == 1
    assert 'fail: analyze.status: expected "partial", got null' in failed.stdout
    assert run(str(doctor)).returncode == 2


FAKE_UV = """#!/bin/bash
echo "$* | cfg=${SOUND_SCOUT_CONFIG-unset} dev=${SOUND_SCOUT_DEV_AUDIO-unset} \
hf=${HF_HUB_OFFLINE-unset} net=${SOUND_SCOUT_NO_NETWORK-unset}" >> "$CALLS"
case "$1 $2 $3" in
  "--version  ") echo "uv 0.0.0" ;;
  "sync --locked --all-groups") ;;
  "run pytest "*)
    if [ -n "$FAIL_PYTEST" ]; then echo "collected in $(pwd)"; echo "== 1 failed =="; exit 1; fi
    echo "======= 3 passed, 1 skipped in 0.10s =======" ;;
  "run sound-scout doctor") cat "$FAKE_DOCTOR" ;;
  "run sound-scout analyze")
    [ -s "$4" ] || { echo "sound-scout analyze: input_error: missing" >&2; exit 1; }
    cat "$SOUND_SCOUT_CONFIG" > "$CALLS.cfg"; cat "$FAKE_ANALYZE"; exit 3 ;;
  "run python -I") shift 2; exec "$PYTHON" "$@" ;;
  *) echo "unexpected uv call: $*" >&2; exit 64 ;;
esac
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A committed repository holding the two scripts, as the clean install sees it."""
    repo = tmp_path / "source"
    (repo / "scripts").mkdir(parents=True)
    for script in (CHECK, EXPECT):
        shutil.copy(script, repo / "scripts" / script.name)
    git = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com"]
    git += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def run_check(
    checkout: Path, tmp_path: Path, analyze: dict, *args: str, **env: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    (tmp_path / "doctor.json").write_text(json.dumps(DOCTOR))
    (tmp_path / "analyze.json").write_text(json.dumps(analyze))
    calls = tmp_path / "calls"
    done = subprocess.run(
        ["bash", str(checkout / "scripts" / CHECK.name), *args],
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CALLS": str(calls),
            "PYTHON": sys.executable,
            "FAKE_DOCTOR": str(tmp_path / "doctor.json"),
            "FAKE_ANALYZE": str(tmp_path / "analyze.json"),
            "SOUND_SCOUT_CONFIG": "/operator/sound-scout.toml",
            "SOUND_SCOUT_DEV_AUDIO": "/operator/recording.wav",
            **env,
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    return done, calls.read_text().splitlines() if calls.exists() else []


def summary_steps(stdout: str) -> list[tuple[str, str]]:
    return [
        (fields[0].strip(), fields[1].strip())
        for fields in (line.split("|") for line in stdout.splitlines() if "|" in line)
    ]


def test_check_prints_a_summary_and_passes_on_the_expected_statuses(
    checkout: Path, tmp_path: Path
) -> None:
    done, calls = run_check(checkout, tmp_path, analyze_answer("unavailable"))
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    commit = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=checkout, capture_output=True, text=True
    ).stdout.strip()
    assert lines[0] == f"clean-install-check: {commit}"
    assert summary_steps(done.stdout) == [
        (step, "ok") for step in ("clone", "uv sync", "pytest", "doctor", "analyze", "expect")
    ]
    assert lines[3].endswith("| 3 passed, 1 skipped in 0.10s")
    assert lines[4].endswith("audio_review=permission_required")
    assert lines[5].endswith(
        "| exit=3 status=partial prepare=completed acoustics=completed asr=unavailable "
        "audio_review=unavailable"
    )
    assert lines[6].endswith("| ok")
    assert lines[-1] == "result: ok"
    pytest_call = next(call for call in calls if call.startswith("run pytest"))
    assert pytest_call == "run pytest | cfg=unset dev=unset hf=1 net=1"
    for name in ("doctor", "analyze"):
        call = next(call for call in calls if call.startswith(f"run sound-scout {name}"))
        assert "dev=unset hf=1 net=1" in call
        assert "cfg=/operator" not in call and "sound-scout.toml" in call
    config = (tmp_path / "calls.cfg").read_text()
    assert 'backend = "none"' in config


def test_check_fails_when_audio_review_is_not_unavailable(checkout: Path, tmp_path: Path) -> None:
    done, _ = run_check(checkout, tmp_path, analyze_answer("completed", "completed"))
    assert done.returncode == 1
    assert summary_steps(done.stdout)[-1] == ("expect", "fail")
    assert "fail: analyze.stages.audio_review" in done.stdout
    assert done.stdout.splitlines()[-1] == "result: fail (expect)"


def test_check_stops_at_the_first_failed_step_and_hides_the_temp_dir(
    checkout: Path, tmp_path: Path
) -> None:
    done, calls = run_check(checkout, tmp_path, analyze_answer(), FAIL_PYTEST="1")
    assert done.returncode == 1
    assert summary_steps(done.stdout)[-1] == ("pytest", "fail")
    assert done.stdout.splitlines()[-1] == "result: fail (pytest)"
    assert not any(call.startswith("run sound-scout") for call in calls)
    assert "collected in <tmp>/repo" in done.stderr
    assert "/tmp" not in done.stderr.replace("<tmp>", "")


def test_check_passes_its_args_to_the_pytest_step_only(checkout: Path, tmp_path: Path) -> None:
    """#167: CI runs the clean install without the rehearsal, which has a job of its own."""
    done, calls = run_check(
        checkout, tmp_path, analyze_answer("unavailable"), "-m", "not rehearsal"
    )
    assert done.returncode == 0, done.stdout + done.stderr
    pytest_call = next(call for call in calls if call.startswith("run pytest"))
    assert pytest_call == "run pytest -m not rehearsal | cfg=unset dev=unset hf=1 net=1"
    assert sum("rehearsal" in call for call in calls) == 1
