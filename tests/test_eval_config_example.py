"""The eval config example of the prosody runs (#201) loads with the product config loader."""

import json
import re
import tomllib
from pathlib import Path

import pytest

from sound_scout import config

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "docs" / "examples" / "eval-config.example.toml"
GUIDE_PATH = REPO_ROOT / "docs" / "eval" / "PROSODY-CONFIG.md"
INPUT_PLACEHOLDER = "/path/to/empty-input-dir"
DATA_PLACEHOLDER = "/path/to/sound-scout-plugin/data"


def filled_example(tmp_path: Path, data_root: Path) -> Path:
    inputs = tmp_path / "in"
    inputs.mkdir()
    text = EXAMPLE_PATH.read_text(encoding="utf-8")
    text = text.replace(json.dumps(INPUT_PLACEHOLDER), json.dumps(str(inputs)))
    text = text.replace(json.dumps(DATA_PLACEHOLDER), json.dumps(str(data_root)))
    path = tmp_path / "sound-scout.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_example_holds_only_placeholder_paths() -> None:
    data = tomllib.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    assert data["allowed_roots"] == [INPUT_PLACEHOLDER]
    assert data["data_root"] == DATA_PLACEHOLDER
    assert Path(data["data_root"]).name == "data"
    assert data["network"] == {"allow": False}


def test_filled_example_loads_with_the_config_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "repo" / "data"
    data_root.mkdir(parents=True)
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(filled_example(tmp_path, data_root)))
    loaded = config.load()
    assert loaded.data_root == data_root.resolve()
    assert loaded.allowed_roots == ((tmp_path / "in").resolve(),)
    assert loaded.network.allow is False
    assert loaded.backend == "none"


def test_example_as_is_is_refused_by_the_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(EXAMPLE_PATH))
    with pytest.raises(config.ConfigError, match="not an existing directory"):
        config.load()


def test_guide_links_only_existing_repo_paths() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    linked = re.findall(r"`((?:docs|src|tests|schema|scripts)/[^`\s]+)`", guide)
    assert EXAMPLE_PATH.relative_to(REPO_ROOT).as_posix() in linked
    assert [path for path in linked if not (REPO_ROOT / path).exists()] == []


def test_guide_names_the_variable_the_loader_reads() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    exported = set(re.findall(r"^export (\w+)=", guide, flags=re.MULTILINE))
    named = set(re.findall(r"`(SOUND_SCOUT_\w+)`", guide))
    assert exported == {config.CONFIG_PATH_ENV}
    assert named <= {config.CONFIG_PATH_ENV}
