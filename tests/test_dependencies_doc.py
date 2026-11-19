"""Acceptance check of issue #41: docs/dependencies.md §1 has not drifted from uv.lock."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPENDENCIES = REPO_ROOT / "docs" / "dependencies.md"
ROW = re.compile(r"^\| `([A-Za-z0-9_.-]+)`[^|]*\| ([^|]+?) \|", re.MULTILINE)


def documented_packages(text: str) -> dict[str, str]:
    section = text.split("## 1. ", 1)[1].split("\n## ", 1)[0]
    return {name.lower(): version for name, version in ROW.findall(section)}


def locked_packages() -> dict[str, str]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {package["name"]: package["version"] for package in lock["package"]}


def direct_dependencies() -> set[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = project["project"]["dependencies"] + [
        item for group in project["dependency-groups"].values() for item in group
    ]
    return {re.match(r"[A-Za-z0-9_.-]+", item).group(0).lower() for item in requirements}


def test_section_1_lists_every_direct_dependency() -> None:
    documented = documented_packages(DEPENDENCIES.read_text(encoding="utf-8"))
    assert direct_dependencies() <= set(documented)


def test_section_1_versions_are_the_locked_versions() -> None:
    documented = documented_packages(DEPENDENCIES.read_text(encoding="utf-8"))
    locked = locked_packages()
    assert documented, "no package rows found in section 1"
    drift = {
        name: (version, locked.get(name))
        for name, version in documented.items()
        if locked.get(name) != version
    }
    assert drift == {}, "docs/dependencies.md §1 (documented, locked)"


def test_a_drifted_row_is_caught() -> None:
    text = "## 1. Python-пакеты\n\n| `numpy` | 0.0.1 | x | BSD |\n\n## 2. Other\n| `uv` | 9 |\n"
    assert documented_packages(text) == {"numpy": "0.0.1"}
    assert locked_packages()["numpy"] != "0.0.1"
