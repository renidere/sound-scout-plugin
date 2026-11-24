"""Acceptance checks of issues #65, #70 and #183: CI installs ffmpeg through one path with bounded
time — the .deb set from the Actions cache with no apt mirror, or on a cache miss a download that
survives a slow apt mirror by retrying."""

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
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "install-ffmpeg"
ACTION = ACTION_DIR / "action.yml"
INSTALL = ACTION_DIR / "install.sh"
DEPENDENCIES = REPO_ROOT / "docs" / "dependencies.md"


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


def shell_function(name: str) -> str:
    text = INSTALL.read_text(encoding="utf-8")
    start = text.index(f"\n{name}() {{\n") + 1
    return text[start : text.index("\n}\n", start) + 2]


def action_steps() -> list[dict]:
    yaml = pytest.importorskip("yaml")
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    return action["runs"]["steps"]


def test_install_ffmpeg_step_has_its_own_timeout_below_the_job_timeout() -> None:
    text = CI.read_text(encoding="utf-8")
    job_timeout = timeout_minutes(text.split("steps:", 1)[0])
    step_timeout = timeout_minutes(step_block(text, "Install ffmpeg"))
    assert step_timeout < job_timeout


def test_install_ffmpeg_step_retries_the_download_within_its_timeout() -> None:
    step_timeout = timeout_minutes(step_block(CI.read_text(encoding="utf-8"), "Install ffmpeg"))
    restore = next(s for s in action_steps() if s.get("id") == "restore")
    restore_minutes = int(restore["env"]["SEGMENT_DOWNLOAD_TIMEOUT_MINS"])
    download = shell_function("download_and_install")
    attempts = shell_int(download, "attempts")
    attempt_minutes = shell_int(download, "attempt_minutes")
    kill_after = re.search(r"--kill-after=(\d+)s", download)
    assert kill_after, "kill-after is missing"
    kill_after_minutes = int(kill_after.group(1)) / 60
    assert 2 <= attempts <= 5
    # Худший успешный путь: застрявшее восстановление кэша до своего предела, все попытки, кроме
    # последней, висят до timeout и kill-after, последняя укладывается в свой предел, затем
    # установка из кэша apt и сохранение набора в кэш Actions.
    install_margin_minutes = 2
    save_margin_minutes = 2
    worst_success = (
        restore_minutes
        + (attempts - 1) * (attempt_minutes + kill_after_minutes)
        + attempt_minutes
        + install_margin_minutes
        + save_margin_minutes
    )
    assert worst_success <= step_timeout
    # ::error:: после всех попыток успевает в лог до предела шага.
    assert restore_minutes + attempts * (attempt_minutes + kill_after_minutes) < step_timeout
    assert 'for attempt in $(seq "$attempts")' in download
    assert 'sudo timeout --kill-after=30s "${attempt_minutes}m"' in download
    assert re.search(r"install -y --download-only ffmpeg", download)
    assert re.search(r'apt-get "\$\{apt_opts\[@\]\}" install -y --no-download ffmpeg', download)


def test_install_ffmpeg_step_retries_apt_a_bounded_number_of_times() -> None:
    download = shell_function("download_and_install")
    retries = re.search(r"Acquire::Retries=(\d+)", download)
    assert retries and 1 <= int(retries.group(1)) <= 5
    request_timeout = re.search(r"Acquire::http::Timeout=(\d+)", download)
    assert request_timeout and int(request_timeout.group(1)) <= 60
    assert 'apt-get "$@" update && apt-get "$@" install' in download
    assert '"${apt_opts[@]}"' in download


def test_install_ffmpeg_step_fails_the_job_when_apt_gives_up() -> None:
    step = step_block(CI.read_text(encoding="utf-8"), "Install ffmpeg")
    assert "continue-on-error" not in step
    assert all("continue-on-error" not in s for s in action_steps())
    for name in ("download_and_install", "install_from_cache"):
        body = shell_function(name)
        assert "|| true" not in body
        assert "set +e" not in body
    assert "\nset -euo pipefail\n" in INSTALL.read_text(encoding="utf-8")
    exhausted = r"echo \"::error::ffmpeg packages were not downloaded in .*\n\s*exit 1\n"
    assert re.search(exhausted, shell_function("download_and_install"))


# Issue #183: one install path for checks and clean-install, the cache of the .deb set.


def test_checks_and_clean_install_use_the_one_install_action() -> None:
    jobs = workflow_jobs()
    for job in ("checks", "clean-install"):
        step = named_step(jobs[job], "Install ffmpeg")
        assert step["uses"] == "./.github/actions/install-ffmpeg"
        assert "run" not in step
    assert not any("install-ffmpeg" in str(step) for step in jobs["rehearsal"]["steps"])
    text = CI.read_text(encoding="utf-8")
    assert "apt-get" not in text, "apt runs only inside the action"


def test_the_action_restores_installs_and_saves_the_deb_set_under_one_key() -> None:
    steps = action_steps()
    assert [s.get("id") or s["name"] for s in steps] == [
        "key",
        "restore",
        "Install the ffmpeg packages",
        "Save the ffmpeg packages",
    ]
    key, restore, install, save = steps
    assert key["run"] == 'bash "$GITHUB_ACTION_PATH/install.sh" key'
    assert install["run"] == 'bash "$GITHUB_ACTION_PATH/install.sh" install'
    assert restore["uses"].split("@")[0] == "actions/cache/restore"
    assert save["uses"].split("@")[0] == "actions/cache/save"
    assert restore["uses"].split("@")[1] == save["uses"].split("@")[1]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", restore["uses"].split("@")[1]), "pinned to a tag"
    debs = "${{ runner.temp }}/ffmpeg-debs"
    assert restore["with"] == save["with"] == {"path": debs, "key": "${{ steps.key.outputs.key }}"}
    assert key["env"]["FFMPEG_DEBS"] == install["env"]["FFMPEG_DEBS"] == debs
    assert install["env"]["CACHE_HIT"] == "${{ steps.restore.outputs.cache-hit }}"
    assert "if" not in install
    assert save["if"] == "steps.restore.outputs.cache-hit != 'true'"


def test_dependencies_doc_records_the_cache_action_and_the_ffmpeg_of_ci() -> None:
    steps = action_steps()
    ci_paragraph = next(
        block
        for block in DEPENDENCIES.read_text(encoding="utf-8").split("\n\n")
        if block.startswith("CI (`.github/workflows/ci.yml`)")
    )
    for step in steps:
        if "uses" in step:
            assert f"`{step['uses']}`" in ci_paragraph
    assert "`ffmpeg` 7:8.0.1-3ubuntu2" in ci_paragraph
    assert "`ubuntu-26.04`" in ci_paragraph


FAKE_SUDO = '#!/bin/sh\nexec "$@"\n'

# Записывает пределы, заданные шагом, и исполняет настоящий timeout с теми же сигналами,
# но в секундах вместо минут.
FAKE_TIMEOUT = """#!/bin/bash
echo "timeout $1 $2" >> "$CALLS"
shift 2
exec {real} --kill-after=1s 1s "$@"
"""

# Попытка скачивания номер n ведёт себя по MODES[n]: hang — медленное зеркало, которое
# игнорирует SIGTERM; fail — ошибка apt; stale — скачаны пакеты $STALE_DEBS, затем ошибка apt;
# ok — пакеты $DEBS скачаны в $APT_ARCHIVES.
FAKE_APT = """#!/bin/bash
echo "apt-get $*" >> "$CALLS"
case " $* " in
  *" clean "*) rm -f "$APT_ARCHIVES"/*.deb ;;
  *" --download-only "*)
    n=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$COUNT"
    read -ra modes <<< "$MODES"
    case "${modes[n-1]:-ok}" in
      hang) trap '' TERM; exec sleep 60 ;;
      fail) exit 100 ;;
      stale) for deb in $STALE_DEBS; do echo deb > "$APT_ARCHIVES/$deb"; done; exit 100 ;;
    esac
    for deb in $DEBS; do echo deb > "$APT_ARCHIVES/$deb"; done ;;
esac
exit 0
"""

FAKE_DPKG = """#!/bin/bash
case "$1" in
  --print-architecture) echo amd64 ;;
  -i) echo "dpkg $*" >> "$CALLS"; exit "${DPKG_EXIT:-0}" ;;
  *) exit 2 ;;
esac
"""

# Package и Version берутся из имени файла apt `<package>_<version>_<arch>.deb`.
FAKE_DPKG_DEB = r"""#!/bin/bash
name=$(basename "${@: -1}" .deb)
version=${name#*_}
printf '%s %s' "${name%%_*}" "${version%_*}" | sed 's/%3a/:/g'
"""

# Установленные версии — пары `<package>=<version>` из $INSTALLED.
FAKE_DPKG_QUERY = """#!/bin/bash
for pair in $INSTALLED; do
  if [ "${pair%%=*}" = "${@: -1}" ]; then echo "installed ${pair#*=}"; exit 0; fi
done
echo "dpkg-query: no packages found matching ${@: -1}" >&2
exit 1
"""

FAKE_FFMPEG = '#!/bin/sh\necho "ffmpeg version 8.0.1-3ubuntu2 Copyright"\necho "built with gcc"\n'

DEB_SET = ["ffmpeg_7%3a8.0.1-3ubuntu2_amd64.deb", "libavcodec62_7%3a8.0.1-3ubuntu2_amd64.deb"]
INSTALLED = "ffmpeg=7:8.0.1-3ubuntu2 libavcodec62=7:8.0.1-3ubuntu2"


def run_install(
    tmp_path: Path,
    command: str = "install",
    modes: list[str] = (),
    cache_hit: str = "",
    cached: list[str] = (),
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, list[str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fakes = {
        "sudo": FAKE_SUDO,
        "timeout": FAKE_TIMEOUT.format(real=shutil.which("timeout")),
        "apt-get": FAKE_APT,
        "dpkg": FAKE_DPKG,
        "dpkg-deb": FAKE_DPKG_DEB,
        "dpkg-query": FAKE_DPKG_QUERY,
        "ffmpeg": FAKE_FFMPEG,
    }
    for name, body in fakes.items():
        fake = bin_dir / name
        fake.write_text(body)
        fake.chmod(0o755)
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "left-from-the-image.deb").write_text("old")
    debs = tmp_path / "ffmpeg-debs"
    if cached:
        debs.mkdir()
        for name in cached:
            (debs / name).write_text("deb")
    calls = tmp_path / "calls"
    calls.touch()
    environ = {
        key: value
        for key, value in os.environ.items()
        if key not in ("ImageOS", "ImageVersion", "GITHUB_RUN_ID", "CACHE_HIT")
    }
    environ.update(
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        CALLS=str(calls),
        COUNT=str(tmp_path / "count"),
        MODES=" ".join(modes),
        DEBS=" ".join(DEB_SET),
        STALE_DEBS="",
        INSTALLED=INSTALLED,
        APT_ARCHIVES=str(archives),
        FFMPEG_DEBS=str(debs),
        CACHE_HIT=cache_hit,
        GITHUB_OUTPUT=str(tmp_path / "github-output"),
    )
    environ.update(env or {})
    # The composite step runs `bash "$GITHUB_ACTION_PATH/install.sh" <command>` with `shell: bash`.
    done = subprocess.run(
        ["bash", "-eo", "pipefail", str(INSTALL), command],
        env=environ,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return done, calls.read_text().splitlines(), debs


def test_a_cache_hit_installs_the_cached_set_with_no_apt_and_no_mirror(tmp_path: Path) -> None:
    done, calls, _ = run_install(tmp_path, cache_hit="true", cached=DEB_SET)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not any(c.startswith(("apt-get", "timeout")) for c in calls), calls
    assert calls == [
        "dpkg -i " + " ".join(str(tmp_path / "ffmpeg-debs" / name) for name in sorted(DEB_SET))
    ]
    assert (
        "ffmpeg package cache hit: dpkg -i of 2 cached .deb files "
        "(ffmpeg_7%3a8.0.1-3ubuntu2_amd64.deb); no apt-get update, "
        "nothing downloaded from the apt mirror"
    ) in done.stdout
    assert "ffmpeg version 8.0.1-3ubuntu2 Copyright\n" in done.stdout
    assert re.search(r"^ffmpeg installed in \d+ s \(cache hit\)$", done.stdout, re.MULTILINE)
    assert "::error::" not in done.stdout and "::warning::" not in done.stdout


@pytest.mark.parametrize(
    ("cached", "dpkg_exit", "reason"),
    [
        ([], "0", "the ffmpeg package cache has no single ffmpeg_*.deb (0 files)"),
        (DEB_SET[1:], "0", "the ffmpeg package cache has no single ffmpeg_*.deb (1 files)"),
        (DEB_SET, "1", "the cached ffmpeg packages did not install with dpkg -i"),
    ],
)
def test_a_bad_cache_hit_fails_with_a_reason_and_never_falls_back_to_the_mirror(
    tmp_path: Path, cached: list[str], dpkg_exit: str, reason: str
) -> None:
    done, calls, _ = run_install(
        tmp_path, cache_hit="true", cached=cached, env={"DPKG_EXIT": dpkg_exit}
    )
    assert done.returncode == 1
    assert f"::error::{reason}" in done.stdout
    assert "gh cache delete" in done.stdout
    assert not any(c.startswith(("apt-get", "timeout")) for c in calls), calls
    assert "ffmpeg installed in" not in done.stdout


@pytest.mark.parametrize("cache_hit", ["", "false"])
def test_a_cache_miss_downloads_installs_and_hands_the_downloaded_set_to_the_cache(
    tmp_path: Path, cache_hit: str
) -> None:
    done, calls, debs = run_install(tmp_path, cache_hit=cache_hit)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ffmpeg package cache miss: download from the apt mirror with bounded retry" in (
        done.stdout
    )
    assert calls[0].startswith("apt-get ") and calls[0].endswith(" clean")
    assert sum("--download-only" in c for c in calls) == 1
    assert calls[-1].endswith("install -y --no-download ffmpeg")
    assert not any(c.startswith("dpkg") for c in calls)
    # only what this install downloaded, not a .deb that was in the apt cache before it
    assert sorted(p.name for p in debs.iterdir()) == sorted(DEB_SET)
    assert "2 downloaded .deb files go to the ffmpeg package cache" in done.stdout
    assert re.search(r"^ffmpeg installed in \d+ s \(cache miss\)$", done.stdout, re.MULTILINE)


def test_a_cache_miss_leaves_superseded_archives_of_a_failed_attempt_out_of_the_cache(
    tmp_path: Path,
) -> None:
    stale = ["ffmpeg_7%3a8.0.1-3ubuntu1_amd64.deb", "libswscale9_7%3a8.0.1-3ubuntu1_amd64.deb"]
    done, calls, debs = run_install(tmp_path, modes=["stale"], env={"STALE_DEBS": " ".join(stale)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert sum("--download-only" in c for c in calls) == 2
    assert sorted(p.name for p in debs.iterdir()) == sorted(DEB_SET)
    assert (
        "2 downloaded .deb files are not of the installed package set and stay out of the "
        "ffmpeg package cache"
    ) in done.stdout
    assert "2 downloaded .deb files go to the ffmpeg package cache" in done.stdout
    assert "::error::" not in done.stdout


@pytest.mark.parametrize(
    ("downloaded", "files"),
    [([], 0), (DEB_SET[1:], 1)],
    ids=["no-deb", "no-ffmpeg-deb"],
)
def test_a_cache_miss_saves_no_set_without_a_single_ffmpeg_deb(
    tmp_path: Path, downloaded: list[str], files: int
) -> None:
    done, calls, debs = run_install(tmp_path, env={"DEBS": " ".join(downloaded)})
    assert done.returncode == 0, done.stdout + done.stderr
    assert calls[-1].endswith("install -y --no-download ffmpeg")
    assert not debs.exists(), "the save step gets no path to store"
    assert (
        f"::warning::the downloaded set has no single ffmpeg_*.deb ({files} files): "
        "nothing goes to the ffmpeg package cache"
    ) in done.stdout
    assert "go to the ffmpeg package cache" not in done.stdout
    assert re.search(r"^ffmpeg installed in \d+ s \(cache miss\)$", done.stdout, re.MULTILINE)


def test_the_cache_key_names_the_runner_image_arch_cpu_level_and_the_script(
    tmp_path: Path,
) -> None:
    image = {"ImageOS": "ubuntu26", "ImageVersion": "20261116.202.1", "GITHUB_RUN_ID": "7"}
    (tmp_path / "a").mkdir()
    done, _, _ = run_install(tmp_path / "a", "key", env=image)
    assert done.returncode == 0, done.stdout + done.stderr
    output = (tmp_path / "a" / "github-output").read_text()
    match = re.fullmatch(
        r"key=ffmpeg-debs-ubuntu26-20261116\.202\.1-amd64-(x86-64-v\d|cpu-unknown)-([0-9a-f]{16})\n",
        output,
    )
    assert match, output
    script_sha = subprocess.run(
        ["sha256sum", str(INSTALL)], capture_output=True, text=True, check=True
    ).stdout[:16]
    assert match.group(2) == script_sha
    assert "::warning::" not in done.stdout

    (tmp_path / "b").mkdir()
    done, _, _ = run_install(tmp_path / "b", "key", env={"GITHUB_RUN_ID": "7"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "::warning::ImageOS or ImageVersion is not set" in done.stdout
    assert (
        (tmp_path / "b" / "github-output")
        .read_text()
        .startswith("key=ffmpeg-debs-unknown-image-run-7-amd64-")
    )


@pytest.mark.slow
@pytest.mark.parametrize("modes", [["hang"], ["fail"], ["hang", "fail"], ["fail", "hang"]])
def test_install_ffmpeg_step_survives_one_or_two_slow_or_failing_downloads(
    tmp_path: Path, modes: list[str]
) -> None:
    done, calls, debs = run_install(tmp_path, modes=modes)
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
    assert sorted(p.name for p in debs.iterdir()) == sorted(DEB_SET)


@pytest.mark.slow
@pytest.mark.parametrize("modes", [["hang", "hang", "hang"], ["fail", "hang", "fail"]])
def test_install_ffmpeg_step_fails_with_a_reason_after_three_bad_downloads(
    tmp_path: Path, modes: list[str]
) -> None:
    started = time.monotonic()
    done, calls, debs = run_install(tmp_path, modes=modes)
    elapsed = time.monotonic() - started
    assert done.returncode == 1
    assert (
        "::error::ffmpeg packages were not downloaded in 3 attempts of 5 min each "
        "(slow or failing apt mirror)"
    ) in done.stdout
    assert sum("--download-only" in c for c in calls) == 3
    assert not any("--no-download" in c for c in calls)
    assert not debs.exists(), "nothing goes to the cache"
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
