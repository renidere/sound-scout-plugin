import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sound_scout.config import root_denied_name
from tests.conftest import TEMP_DIR_VARIABLES, use_a_temp_dir_outside_denied_names

REPO_ROOT = Path(__file__).resolve().parents[1]


def inherit_temp_dir(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    """Make `temp_dir` the TMPDIR the session inherited; monkeypatch restores the real one."""
    for variable in TEMP_DIR_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TMPDIR", str(temp_dir))
    monkeypatch.setattr(tempfile, "tempdir", None)


def test_tmp_path_and_the_temp_dir_are_outside_every_denied_name(tmp_path: Path) -> None:
    assert root_denied_name(tmp_path.resolve()) is None
    assert root_denied_name(Path(tempfile.gettempdir()).resolve()) is None


@pytest.mark.parametrize(
    "below_hermes",
    [("cache", "scratch"), ("cache", "audio")],
    ids=["scratch", "media-dir"],
)
def test_a_temp_dir_under_hermes_is_replaced_by_the_system_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, below_hermes: tuple[str, ...]
) -> None:
    scratch = tmp_path.joinpath(".hermes", *below_hermes)
    scratch.mkdir(parents=True)
    inherit_temp_dir(monkeypatch, scratch)

    use_a_temp_dir_outside_denied_names()

    replaced = Path(os.environ["TMPDIR"])
    assert replaced != scratch
    assert root_denied_name(replaced) is None
    assert tempfile.gettempdir() == str(replaced)


def test_a_usual_temp_dir_is_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    inherit_temp_dir(monkeypatch, scratch)

    use_a_temp_dir_outside_denied_names()

    assert os.environ["TMPDIR"] == str(scratch)
    assert tempfile.gettempdir() == str(scratch)


def test_no_temp_dir_outside_a_denied_name_stops_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch = tmp_path / ".hermes" / "cache" / "scratch"
    scratch.mkdir(parents=True)
    inherit_temp_dir(monkeypatch, scratch)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(scratch))

    with pytest.raises(pytest.UsageError, match="no temp dir outside a denied name"):
        use_a_temp_dir_outside_denied_names()


def test_a_session_with_tmpdir_under_hermes_passes_the_tests_that_use_tmp_path_as_a_root(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / ".hermes" / "cache" / "scratch"
    scratch.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in TEMP_DIR_VARIABLES}
    env.pop("PYTEST_ADDOPTS", None)
    env["TMPDIR"] = str(scratch)
    temp_dir_test = (
        "tests/test_temp_dir.py::test_tmp_path_and_the_temp_dir_are_outside_every_denied_name"
    )
    root_test = "tests/test_findings.py::test_build_maps_observations_onto_the_source_timescale"

    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider"]
        + [temp_dir_test, root_test],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert child.returncode == 0, child.stdout[-2000:] + child.stderr[-2000:]
    assert f"PASSED {temp_dir_test}" in child.stdout
    # test_findings skips on purpose without FFmpeg; with it, the tmp_path root must pass.
    if shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None:
        assert f"PASSED {root_test}" in child.stdout
    assert not any(scratch.iterdir())


def test_the_session_keeps_only_the_tmp_path_of_failed_tests_for_one_session(
    request: pytest.FixtureRequest,
) -> None:
    assert request.config.getini("tmp_path_retention_policy") == "failed"
    assert int(request.config.getini("tmp_path_retention_count")) == 1


RETENTION_TESTS = """
import os

import pytest


def record(request, tmp_path, name):
    (tmp_path / "evidence.txt").write_text(name)
    policy = request.config.getini("tmp_path_retention_policy")
    count = request.config.getini("tmp_path_retention_count")
    with open(os.environ["RECORD"], "a") as record_file:
        record_file.write(f"{name} {policy} {count} {tmp_path}\\n")


def test_pass(request, tmp_path):
    record(request, tmp_path, "pass")


@pytest.mark.skipif(os.environ.get("FAIL") != "1", reason="a run that passes")
def test_fail(request, tmp_path):
    record(request, tmp_path, "fail")
    assert False
"""

# The pytest defaults: what a child run gets when the repo's retention settings are removed.
DEFAULT_RETENTION = ["-o", "tmp_path_retention_policy=all", "-o", "tmp_path_retention_count=3"]


def run_child(
    tmp_path: Path, temp_root: Path, *, fail: bool, options: list[str] | None = None
) -> dict[str, tuple[str, str, Path]]:
    """Run the retention tests in a child pytest on `temp_root` with the repo's pyproject.toml."""
    temp_root.mkdir(exist_ok=True)
    test_file = tmp_path / "test_retention.py"
    test_file.write_text(RETENTION_TESTS)
    record = tmp_path / "record.txt"
    record.write_text("")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env |= {
        "PYTEST_DEBUG_TEMPROOT": str(temp_root),
        "RECORD": str(record),
        "FAIL": "1" if fail else "0",
    }
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + ["-c", str(REPO_ROOT / "pyproject.toml"), *(options or []), str(test_file)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == (1 if fail else 0), child.stdout[-2000:] + child.stderr
    recorded = {}
    for line in record.read_text().splitlines():
        name, policy, count, path = line.split(" ", 3)
        recorded[name] = (policy, count, Path(path))
    return recorded


def sessions(temp_root: Path) -> list[Path]:
    return sorted(temp_root.glob("pytest-of-*/pytest-[0-9]*"))


def passed_test_dirs(temp_root: Path) -> list[Path]:
    return [
        evidence.parent
        for evidence in temp_root.glob("pytest-of-*/pytest-[0-9]*/*/evidence.txt")
        if not evidence.parent.is_symlink() and evidence.read_text() == "pass"
    ]


@pytest.mark.slow
def test_criterion_2_three_passing_runs_keep_the_dirs_of_at_most_one_passed_run(
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "temproot"
    for _ in range(3):
        recorded = run_child(tmp_path, temp_root, fail=False)
        policy, count, passed = recorded["pass"]
        assert (policy, count) == ("failed", "1")
        assert not passed.exists()
    assert len(sessions(temp_root)) <= 1
    assert passed_test_dirs(temp_root) == []

    # Control: without the repo settings, the same three runs keep three passed runs.
    control_root = tmp_path / "control-temproot"
    for _ in range(3):
        recorded = run_child(tmp_path, control_root, fail=False, options=DEFAULT_RETENTION)
        assert recorded["pass"][:2] == ("all", "3")
    assert len(sessions(control_root)) == 3
    assert len(passed_test_dirs(control_root)) == 3


@pytest.mark.slow
def test_criterion_3_a_failed_test_keeps_its_tmp_path_files_readable_after_the_session(
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "temproot"

    first = run_child(tmp_path, temp_root, fail=True)
    assert first["fail"][:2] == ("failed", "1")
    assert not first["pass"][2].exists()
    assert (first["fail"][2] / "evidence.txt").read_text() == "fail"
    assert sessions(temp_root) == [first["fail"][2].parent]

    second = run_child(tmp_path, temp_root, fail=True)
    assert (second["fail"][2] / "evidence.txt").read_text() == "fail"
    assert sessions(temp_root) == [second["fail"][2].parent]

    third = run_child(tmp_path, temp_root, fail=False)
    assert not third["pass"][2].parent.exists()
    # The prune runs after the passed session is removed, so the last failed one stays readable.
    assert sessions(temp_root) == [second["fail"][2].parent]
    assert (second["fail"][2] / "evidence.txt").read_text() == "fail"
