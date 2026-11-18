"""Acceptance checks of issue #65: the ffmpeg install step in CI has bounded time."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def step_block(text: str, name: str) -> str:
    start = text.index(f"- name: {name}\n")
    end = text.find("\n      - ", start)
    return text[start:] if end == -1 else text[start:end]


def timeout_minutes(block: str) -> int:
    match = re.search(r"^\s*timeout-minutes:\s*(\d+)\s*$", block, re.MULTILINE)
    assert match, "timeout-minutes is missing"
    return int(match.group(1))


def test_install_ffmpeg_step_has_its_own_timeout_below_the_job_timeout() -> None:
    text = CI.read_text(encoding="utf-8")
    job_timeout = timeout_minutes(text.split("steps:", 1)[0])
    step_timeout = timeout_minutes(step_block(text, "Install ffmpeg"))
    assert step_timeout <= 5
    assert step_timeout < job_timeout


def test_install_ffmpeg_step_retries_apt_a_bounded_number_of_times() -> None:
    step = step_block(CI.read_text(encoding="utf-8"), "Install ffmpeg")
    retries = re.search(r"Acquire::Retries=(\d+)", step)
    assert retries and 1 <= int(retries.group(1)) <= 5
    request_timeout = re.search(r"Acquire::http::Timeout=(\d+)", step)
    assert request_timeout and int(request_timeout.group(1)) <= 60
    assert re.search(r'apt-get "\$\{apt_opts\[@\]\}" update', step)
    assert re.search(r'apt-get "\$\{apt_opts\[@\]\}" install -y ffmpeg', step)


def test_install_ffmpeg_step_fails_the_job_when_apt_gives_up() -> None:
    step = step_block(CI.read_text(encoding="utf-8"), "Install ffmpeg")
    assert "continue-on-error" not in step
    assert "|| true" not in step
    assert "set +e" not in step
