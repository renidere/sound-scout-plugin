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
