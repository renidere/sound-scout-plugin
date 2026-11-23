"""Acceptance checks of issues #65 and #70: the ffmpeg install step in CI has bounded time
and survives a slow apt mirror by retrying."""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from sound_scout.eval import rehearsal

REPO_ROOT = Path(__file__).resolve().parent.parent
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def step_block(text: str, name: str) -> str:
    """The first step named `name`, from its `- ` line (which may carry an anchor) to the next."""
    start = text.rindex("\n      - ", 0, text.index(f"name: {name}\n")) + 1
    end = text.find("\n      - ", start)
    return text[start:] if end == -1 else text[start:end]


def timeout_minutes(block: str) -> int:
    match = re.search(r"^\s*timeout-minutes:\s*(\d+)\s*$", block, re.MULTILINE)
    assert match, "timeout-minutes is missing"
    return int(match.group(1))


def shell_int(block: str, name: str) -> int:
    match = re.search(rf"^\s*{name}=(\d+)\s*$", block, re.MULTILINE)
    assert match, f"{name} is missing"
    return int(match.group(1))


def test_install_ffmpeg_step_has_its_own_timeout_below_the_job_timeout() -> None:
    text = CI.read_text(encoding="utf-8")
    job_timeout = timeout_minutes(text.split("steps:", 1)[0])
    step_timeout = timeout_minutes(step_block(text, "Install ffmpeg"))
    assert step_timeout < job_timeout


def test_install_ffmpeg_step_retries_the_download_within_its_timeout() -> None:
    step = step_block(CI.read_text(encoding="utf-8"), "Install ffmpeg")
    attempts = shell_int(step, "attempts")
    attempt_minutes = shell_int(step, "attempt_minutes")
    kill_after = re.search(r"--kill-after=(\d+)s", step)
    assert kill_after, "kill-after is missing"
    kill_after_minutes = int(kill_after.group(1)) / 60
    assert 2 <= attempts <= 5
    # Худший успешный путь: все попытки, кроме последней, висят до timeout и kill-after,
    # последняя укладывается в свой предел, затем установка из кэша apt.
    install_margin_minutes = 2
    worst_success = (
        (attempts - 1) * (attempt_minutes + kill_after_minutes)
        + attempt_minutes
        + install_margin_minutes
    )
    assert worst_success <= timeout_minutes(step)
    assert attempts * (attempt_minutes + kill_after_minutes) < timeout_minutes(step)
    assert 'for attempt in $(seq "$attempts")' in step
    assert 'sudo timeout --kill-after=30s "${attempt_minutes}m"' in step
    assert re.search(r"install -y --download-only ffmpeg", step)
    assert re.search(r'apt-get "\$\{apt_opts\[@\]\}" install -y --no-download ffmpeg', step)


def test_install_ffmpeg_step_retries_apt_a_bounded_number_of_times() -> None:
    step = step_block(CI.read_text(encoding="utf-8"), "Install ffmpeg")
    retries = re.search(r"Acquire::Retries=(\d+)", step)
    assert retries and 1 <= int(retries.group(1)) <= 5
    request_timeout = re.search(r"Acquire::http::Timeout=(\d+)", step)
    assert request_timeout and int(request_timeout.group(1)) <= 60
    assert 'apt-get "$@" update && apt-get "$@" install' in step
    assert '"${apt_opts[@]}"' in step


def test_install_ffmpeg_step_fails_the_job_when_apt_gives_up() -> None:
    step = step_block(CI.read_text(encoding="utf-8"), "Install ffmpeg")
    assert "continue-on-error" not in step
    assert "|| true" not in step
    assert "set +e" not in step
    exhausted = r"echo \"::error::ffmpeg packages were not downloaded in .*\n\s*exit 1\n"
    assert re.search(exhausted, step)


def run_script(text: str, name: str) -> str:
    """The literal `run: |` block of a step, dedented as YAML reads it."""
    lines = step_block(text, name).splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "run: |") + 1
    body = lines[start:]
    indent = min(len(line) - len(line.lstrip()) for line in body if line.strip())
    return "\n".join(line[indent:] for line in body) + "\n"


FAKE_SUDO = '#!/bin/sh\nexec "$@"\n'

# Записывает пределы, заданные шагом, и исполняет настоящий timeout с теми же сигналами,
# но в секундах вместо минут.
FAKE_TIMEOUT = """#!/bin/bash
echo "timeout $1 $2" >> "$CALLS"
shift 2
exec {real} --kill-after=1s 1s "$@"
"""

# Попытка скачивания номер n ведёт себя по MODES[n]: hang — медленное зеркало, которое
# игнорирует SIGTERM; fail — ошибка apt; ok — пакеты скачаны.
FAKE_APT = """#!/bin/bash
echo "apt-get $*" >> "$CALLS"
case " $* " in
  *" --download-only "*)
    n=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$COUNT"
    read -ra modes <<< "$MODES"
    case "${modes[n-1]:-ok}" in
      hang) trap '' TERM; exec sleep 60 ;;
      fail) exit 100 ;;
    esac ;;
esac
exit 0
"""


def run_install_step(
    tmp_path: Path, modes: list[str]
) -> tuple[subprocess.CompletedProcess, list[str]]:
    script = tmp_path / "install-ffmpeg.sh"
    script.write_text(run_script(CI.read_text(encoding="utf-8"), "Install ffmpeg"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fakes = {
        "sudo": FAKE_SUDO,
        "timeout": FAKE_TIMEOUT.format(real=shutil.which("timeout")),
        "apt-get": FAKE_APT,
    }
    for name, body in fakes.items():
        fake = bin_dir / name
        fake.write_text(body)
        fake.chmod(0o755)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CALLS": str(calls),
        "COUNT": str(tmp_path / "count"),
        "MODES": " ".join(modes),
    }
    # GitHub Actions runs a step without `shell:` as `bash -e {0}`.
    done = subprocess.run(
        ["bash", "-e", str(script)], env=env, capture_output=True, text=True, timeout=60
    )
    return done, calls.read_text().splitlines()


@pytest.mark.slow
@pytest.mark.parametrize("modes", [["hang"], ["fail"], ["hang", "fail"], ["fail", "hang"]])
def test_install_ffmpeg_step_survives_one_or_two_slow_or_failing_downloads(
    tmp_path: Path, modes: list[str]
) -> None:
    done, calls = run_install_step(tmp_path, modes)
    assert done.returncode == 0, done.stdout + done.stderr
    downloads = [c for c in calls if "--download-only" in c]
    assert len(downloads) == len(modes) + 1
    assert all("Acquire::Retries=3" in c for c in downloads)
    assert [c for c in calls if c.startswith("timeout")] == ["timeout --kill-after=30s 5m"] * (
        len(modes) + 1
    )
    assert done.stdout.count("::warning::ffmpeg download attempt") == len(modes)
    assert "::error::" not in done.stdout
    assert calls[-1].startswith("apt-get ") and calls[-1].endswith(
        "install -y --no-download ffmpeg"
    )
    assert sum("install -y --no-download ffmpeg" in c for c in calls) == 1


@pytest.mark.slow
@pytest.mark.parametrize("modes", [["hang", "hang", "hang"], ["fail", "hang", "fail"]])
def test_install_ffmpeg_step_fails_with_a_reason_after_three_bad_downloads(
    tmp_path: Path, modes: list[str]
) -> None:
    started = time.monotonic()
    done, calls = run_install_step(tmp_path, modes)
    elapsed = time.monotonic() - started
    assert done.returncode == 1
    assert (
        "::error::ffmpeg packages were not downloaded in 3 attempts of 5 min each "
        "(slow or failing apt mirror)"
    ) in done.stdout
    assert sum("--download-only" in c for c in calls) == 3
    assert not any("--no-download" in c for c in calls)
    # Висящее зеркало, глухое к SIGTERM, обрывает timeout: 1 с + 1 с kill-after на попытку.
    assert elapsed < 3 * 2 + 5


# Issue #41: the clean install and the history scan run as jobs of their own.


def workflow_jobs() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]


def named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_clean_install_job_reuses_the_ffmpeg_step_of_checks_and_runs_the_script() -> None:
    jobs = workflow_jobs()
    job = jobs["clean-install"]
    for name in ("Detect the package", "Install ffmpeg"):
        assert named_step(job, name) == named_step(jobs["checks"], name)
    text = CI.read_text(encoding="utf-8")
    assert text.count("name: Install ffmpeg\n") == 1, "the step is aliased, not copied"
    uses = [step["uses"] for step in job["steps"] if "uses" in step]
    checks_uses = [step["uses"] for step in jobs["checks"]["steps"] if "uses" in step]
    assert uses == checks_uses
    # the suite of checks, without the #149 rehearsal (the job rehearsal has it, #167)
    assert job["steps"][-1]["run"] == 'bash scripts/clean-install-check.sh -m "not rehearsal"'
    assert job["timeout-minutes"] == jobs["checks"]["timeout-minutes"] == 45


def test_history_scan_job_checks_out_the_whole_history_and_runs_the_scanner() -> None:
    job = workflow_jobs()["history-scan"]
    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0
    assert job["steps"][-1]["run"] == "bash scripts/scan-history.sh"
    assert all("continue-on-error" not in step for step in job["steps"])


def test_criterion_4_ci_runs_the_full_suite_split_between_checks_and_rehearsal() -> None:
    """#164: the fast set is `-m "not slow"` on the station; CI still runs every test.
    #167: checks runs all but the rehearsal, the job rehearsal runs the rest (the shards are
    checked in tests/test_eval_rehearsal.py)."""
    jobs = workflow_jobs()
    assert named_step(jobs["checks"], "Test")["run"] == 'uv run pytest -m "not rehearsal"'
    assert named_step(jobs["rehearsal"], "Test")["run"] == "uv run pytest -m rehearsal"
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    addopts = re.search(r'^addopts\s*=\s*"([^"]*)"', pyproject, re.MULTILINE)
    assert addopts and "-m" not in addopts.group(1).split()


def test_the_rehearsal_job_names_the_scenario_count_and_stays_within_its_limit() -> None:
    """#167: the job comment gives the scenario count of the code; checks keeps its 45 min."""
    text = CI.read_text(encoding="utf-8")
    job = text[text.index("\n  rehearsal:\n") :]
    comment = job[: job.index("\n    timeout-minutes:")]
    assert f"сценариев: {len(rehearsal.SCENARIOS)}, unit: {len(rehearsal.UNITS)}." in comment
    assert "Предел 20 мин" in comment
    jobs = workflow_jobs()
    assert jobs["rehearsal"]["timeout-minutes"] == 20
    assert jobs["checks"]["timeout-minutes"] == 45
    steps = jobs["rehearsal"]["steps"]
    assert [s.get("uses", "").split("@")[0] for s in steps if "uses" in s] == [
        "actions/checkout",
        "astral-sh/setup-uv",
    ]
    assert [s["name"] for s in steps if "name" in s] == ["Sync", "Test"]
    # the rehearsal never calls ffmpeg: no ffmpeg step, the same locked environment as checks
    assert named_step(jobs["rehearsal"], "Sync")["run"] == named_step(jobs["checks"], "Sync")["run"]
    test = named_step(jobs["rehearsal"], "Test")
    assert test["env"]["HF_HUB_OFFLINE"] == test["env"]["SOUND_SCOUT_NO_NETWORK"] == "1"
