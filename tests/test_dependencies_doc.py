"""Acceptance check of issue #41: docs/dependencies.md §1 has not drifted from uv.lock."""

import re
import tomllib
from pathlib import Path

from sound_scout.eval.corpus import MS_SNSD_COMMIT, SOURCES

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


DATA_ROW = re.compile(
    r"^\|[^|]+\|[^|]+\| `([^`]+)` \|[^|]+\|[^|]+\| (\d+) \| `([^`]+)` \| `([0-9a-f]{64})` \|$",
    re.MULTILINE,
)


def documented_sources(text: str) -> dict[str, tuple[int, str, str]]:
    section = text.split("## 5. ", 1)[1].split("\n## ", 1)[0]
    return {
        name: (int(size), checksum, sha) for name, size, checksum, sha in DATA_ROW.findall(section)
    }


def test_section_5_pins_every_corpus_source_like_the_sources_table() -> None:
    documented = documented_sources(DEPENDENCIES.read_text(encoding="utf-8"))
    pinned = {s.name: (s.size, s.checksum, s.sha256) for s in SOURCES}
    assert all(sha256 for _, _, sha256 in pinned.values()), "a SOURCES row has no SHA-256"
    assert documented == pinned


def test_a_drifted_source_row_is_caught() -> None:
    source = SOURCES[0]
    row = (
        f"| RINCO | v0.1 | `{source.name}` | {source.url} | CC BY 4.0 | {source.size + 1} | "
        f"`{source.checksum}` | `{source.sha256}` |"
    )
    documented = documented_sources(f"## 5. Данные\n\n{row}\n\n## 6. Other\n")
    assert documented[source.name] != (source.size, source.checksum, source.sha256)


OPEN_LICENSES = ("CC BY 4.0", "CC BY-SA 3.0", "CC0")
NO_REGISTRATION_URLS = {
    "rinco": "https://zenodo.org/records/8119302/files/",
    "demand": "https://zenodo.org/records/1227121/files/",
    "ms-snsd": f"https://raw.githubusercontent.com/microsoft/MS-SNSD/{MS_SNSD_COMMIT}/",
}


def test_criterion_1_only_open_licensed_files_are_pinned_with_source_version_and_license() -> None:
    """#78 criterion 1: RINCO, 2-3 DEMAND rooms, MS-SNSD Babble/NeighborSpeaking, open licenses."""
    by_dataset: dict[str, list[str]] = {}
    for source in SOURCES:
        by_dataset.setdefault(source.dataset, []).append(source.name)
        assert source.license.startswith(OPEN_LICENSES), source.name
        assert source.url.startswith(NO_REGISTRATION_URLS[source.dataset]), source.name
    assert set(by_dataset) == {"rinco", "demand", "ms-snsd"}
    assert "RINCO_Russian_Intonation_Corpus-v01.zip" in by_dataset["rinco"]
    assert 2 <= len(by_dataset["demand"]) <= 3
    assert all(re.fullmatch(r"[A-Z]+_16k\.zip", name) for name in by_dataset["demand"])
    assert {name.split("_", 1)[0] for name in by_dataset["ms-snsd"]} == {
        "Babble",
        "NeighborSpeaking",
    }

    section = DEPENDENCIES.read_text(encoding="utf-8").split("## 5. ", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 8 and cells[2].startswith("`"):
            rows[cells[2].strip("`")] = (cells[1], cells[3], cells[4])
    assert rows == {s.name: (s.version, s.url, s.license) for s in SOURCES}
