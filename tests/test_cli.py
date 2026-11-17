import subprocess
import sys
from importlib.metadata import version

import pytest

from sound_scout.cli import SUBCOMMANDS, main

EXPECTED_SUBCOMMANDS = ["doctor", "analyze", "compare", "serve"]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sound_scout", *args], capture_output=True, text=True, check=False
    )


def test_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"sound-scout {version('sound-scout')}"


def test_help_lists_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert list(SUBCOMMANDS) == EXPECTED_SUBCOMMANDS
    for name in EXPECTED_SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize("command", EXPECTED_SUBCOMMANDS)
def test_subcommand_exits_2_not_implemented(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command]) == 2
    captured = capsys.readouterr()
    assert "not implemented" in captured.err
    assert captured.out == ""


def test_missing_subcommand_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_unknown_subcommand_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["transcribe"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_console_script_module_entry() -> None:
    assert run_cli("--version").stdout.strip() == f"sound-scout {version('sound-scout')}"
    result = run_cli("doctor")
    assert result.returncode == 2
    assert "not implemented" in result.stderr
