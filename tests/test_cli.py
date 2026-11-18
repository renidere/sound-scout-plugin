import json
import os
import shutil
import subprocess
import sys
import textwrap
from importlib.metadata import version
from pathlib import Path

import pytest

from sound_scout import config, doctor, guard
from sound_scout.cli import SUBCOMMANDS, main

EXPECTED_SUBCOMMANDS = ["doctor", "analyze", "compare", "serve", "eval", "smoke"]
NOT_IMPLEMENTED_SUBCOMMANDS = ["analyze", "compare", "serve"]
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sound_scout", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def write_config(tmp_path: Path, *, allow_network: bool) -> Path:
    recordings, data = tmp_path / "recordings", tmp_path / "data"
    recordings.mkdir()
    data.mkdir()
    path = tmp_path / "sound-scout.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(recordings))}]\n"
        f"data_root = {json.dumps(str(data))}\n"
        f"[network]\nallow = {json.dumps(allow_network)}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config_without_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`network.allow = true`, so an in-process `doctor` installs no audit hook in this session."""
    path = write_config(tmp_path, allow_network=True)
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    return path


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


@pytest.mark.parametrize("command", NOT_IMPLEMENTED_SUBCOMMANDS)
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


def test_doctor_prints_a_table_and_exits_0(
    config_without_guard: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    header = [cell.strip() for cell in lines[0].split(" | ")]
    assert header == ["component", "status", "last smoke", "fingerprint", "reason"]
    report = doctor.report(config.load())
    for line, (name, component) in zip(lines[1:5], report["components"].items(), strict=True):
        cells = [cell.strip() for cell in line.split(" | ")]
        assert cells[0] == name
        assert cells[1] == component["status"]
        assert cells[2] == "-"
        assert cells[3] == component["fingerprint"]["hash"][:12]
        assert cells[4] == component["reason"]
    assert lines[5:] == [
        "validated_ru: false",
        "diction_validated_ru: false",
        "prosody_validated_ru: false",
        "network_guard: inactive (network.allow = true)",
    ]


def test_doctor_json_matches_report(
    config_without_guard: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    expected = doctor.report(config.load())
    assert printed["components"] == expected["components"]
    assert printed["sound_scout"] == version("sound-scout")
    assert printed["network_guard"] == {
        "active": False,
        "network_allow": True,
        "offline_env": expected["network_guard"]["offline_env"],
    }
    for flag in ("validated_ru", "diction_validated_ru", "prosody_validated_ru"):
        assert printed[flag] is False


@pytest.mark.parametrize("status", doctor.STATUSES)
def test_doctor_exits_0_whatever_the_statuses(
    status: str,
    config_without_guard: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Diagnostics, not a check: `ready` and every other status alike end with exit code 0."""
    real = doctor.report(config.load())
    smoke_at = "2026-11-17T12:00:00+00:00" if status == "ready" else None
    reason = None if status == "ready" else f"reason for {status}"
    for component in real["components"].values():
        component.update(status=status, reason=reason, last_smoke_at=smoke_at)
    monkeypatch.setattr(doctor, "report", lambda loaded: real)

    assert main(["doctor"]) == 0
    table = capsys.readouterr().out.splitlines()
    assert [line.split(" | ")[1].strip() for line in table[1:5]] == [status] * 4
    assert table[1].split(" | ")[2].strip() == (smoke_at or "-")
    assert table[1].split(" | ")[4].strip() == (reason or "-")

    assert main(["doctor", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert {c["status"] for c in printed["components"].values()} == {status}


def test_doctor_without_configuration_is_one_line_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(tmp_path / "missing.toml"))
    assert main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("config error: ")
    assert captured.err.count("\n") == 1
    assert "missing.toml" in captured.err


def test_doctor_with_invalid_configuration_names_the_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path, allow_network=True)
    path.write_text(path.read_text() + "[asr]\nmodel = 'x'\n")
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    assert main(["doctor"]) == 1
    assert "asr.model" in capsys.readouterr().err


def test_console_script_module_entry(tmp_path: Path) -> None:
    assert run_cli("--version").stdout.strip() == f"sound-scout {version('sound-scout')}"
    path = write_config(tmp_path, allow_network=False)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env[config.CONFIG_PATH_ENV] = str(path)
    result = run_cli("doctor", "--json", env=env)
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    assert set(printed["components"]) == set(doctor.COMPONENTS)
    assert printed["network_guard"] == {
        "active": True,
        "network_allow": False,
        "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    }
    assert not (tmp_path / "data" / "receipts").exists()
    assert run_cli("analyze").returncode == 2


def offline_env(path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in guard.OFFLINE_ENV}
    env[config.CONFIG_PATH_ENV] = str(path)
    return env


def tree_state(root: Path) -> list[tuple[str, int, int]]:
    """Every path under `root` with its size and mtime, to show that a command wrote nothing."""
    return sorted(
        (str(path.relative_to(root)), path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in [root, *root.rglob("*")]
    )


def require_acoustics() -> None:
    loaded = config.load()
    state = doctor.component_status("acoustics", loaded, loaded.data_root)
    if state["status"] == "unavailable":
        pytest.skip(f"acoustics is unavailable here: {state['reason']}")


@needs_ffmpeg
def test_smoke_writes_both_receipts_and_doctor_then_reports_them_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path, allow_network=False)
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    require_acoustics()

    result = run_cli("smoke", env=offline_env(path))

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "decoder: ok (receipts/decoder.json)",
        "acoustics: ok (receipts/acoustics.json)",
    ]
    before_doctor = tree_state(tmp_path / "data")
    report = run_cli("doctor", "--json", env=offline_env(path))
    assert report.returncode == 0, report.stderr
    components = json.loads(report.stdout)["components"]
    assert components["decoder"]["status"] == "ready"
    assert components["acoustics"]["status"] == "ready"
    assert sorted(os.listdir(tmp_path / "data")) == [doctor.RECEIPTS_DIR_NAME]
    assert tree_state(tmp_path / "data") == before_doctor


@needs_ffmpeg
def test_smoke_decoder_writes_only_the_decoder_receipt(
    config_without_guard: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["smoke", "decoder"]) == 0
    assert capsys.readouterr().out == "decoder: ok (receipts/decoder.json)\n"
    receipts = config_without_guard.parent / "data" / doctor.RECEIPTS_DIR_NAME
    assert os.listdir(receipts) == ["decoder.json"]


def test_smoke_without_configuration_is_one_line_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(tmp_path / "missing.toml"))
    assert main(["smoke"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("config error: ")
    assert captured.err.count("\n") == 1


@pytest.mark.parametrize("component", ["asr", "audio_review"])
def test_smoke_of_another_component_is_a_usage_error(
    component: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["smoke", component])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_failed_smoke_exits_1_and_names_the_reason(
    config_without_guard: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "bin"
    broken.mkdir()
    for tool in ("ffmpeg", "ffprobe"):
        script = broken / tool
        script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{broken}{os.pathsep}{os.environ['PATH']}")

    assert main(["smoke", "decoder"]) == 1

    assert capsys.readouterr().out.startswith("decoder: failed: corrupted: ffprobe failed")
    assert not (tmp_path / "data" / doctor.RECEIPTS_DIR_NAME).exists()


@needs_ffmpeg
def test_smoke_installs_the_network_guard(tmp_path: Path) -> None:
    path = write_config(tmp_path, allow_network=False)
    script = textwrap.dedent(
        """
        import json
        from sound_scout import cli, guard
        before = guard.network_guard_active()
        code = cli.main(["smoke", "decoder"])
        print(json.dumps({"before": before, "code": code, "active": guard.network_guard_active()}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=offline_env(path),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"before": False, "code": 0, "active": True}
