"""Acceptance checks of issue #41: docs/environment.md as its reader sees it (T10, R11)."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT = REPO_ROOT / "docs" / "environment.md"
HOME_PLACEHOLDERS = {"someone", "user", "username", "runner", "example", "<user>"}


def sections() -> dict[str, str]:
    text = ENVIRONMENT.read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    return {part.split("\n", 1)[0]: part for part in parts}


def section(prefix: str) -> str:
    return next(body for title, body in sections().items() if title.startswith(prefix))


def test_environment_doc_has_the_six_sections() -> None:
    titles = list(sections())
    assert len(titles) == 6, titles
    for number, words in enumerate(
        (
            "Закреплённая среда",
            "Receipts моделей",
            "чистой установки",
            "Нет приватного",
            "Аппаратная конфигурация",
            "UX-бюджет R11",
        ),
        start=1,
    ):
        assert titles[number - 1].startswith(f"{number}. ") and words in titles[number - 1]


def test_pinned_environment_names_python_uv_ffmpeg_and_points_to_the_lock() -> None:
    pinned = section("1. ")
    assert "Python 3.13" in pinned
    assert "`uv`" in pinned and "FFmpeg" in pinned
    assert "uv.lock" in pinned and "docs/dependencies.md" in pinned
    assert "uv export --locked" in pinned


def test_model_receipts_section_names_offline_commands_and_the_hash_source() -> None:
    receipts = section("2. ")
    assert "docs/dependencies.md" in receipts and "SHA-256" in receipts
    assert "sound-scout doctor --json" in receipts
    assert "asr/model-receipt.json" in receipts
    assert "sha256sum" in receipts


def test_hardware_section_describes_the_station_without_personal_data() -> None:
    hardware = section("5. ")
    for row in ("CPU", "логических ядер", "RAM", "ОС и ядро", "Python", "uv", "FFmpeg", "GPU"):
        assert re.search(rf"^\| {re.escape(row)} \| \S", hardware, re.MULTILINE), row
    assert "CPU-only" in hardware


def test_profile_table_covers_30_s_180_s_and_the_development_recording_against_r11() -> None:
    budget = section("6. ")
    rows = [line for line in budget.splitlines() if line.startswith("| ") and "|---" not in line]
    header, data = rows[0], rows[1:]
    for column in ("wall", "ASR load", "ASR inference", "peak RSS", "бюджет", "в бюджете"):
        assert column in header
    for label, limit in (
        ("synthetic 30 с", "60"),
        ("synthetic 180 с", "180"),
        ("development recording", "60"),
    ):
        phases = {cells[1] for cells in (row.split(" | ") for row in data) if label in cells[0]}
        assert phases == {"cold", "warm"}, label
        assert all(f"| {limit} |" in row for row in data if label in row), label
    assert "audio_review не измерен" in budget
    assert "не установлено" in budget


def test_environment_doc_holds_no_personal_home_path() -> None:
    text = ENVIRONMENT.read_text(encoding="utf-8")
    users = set(re.findall(r"/home/([^/\s`]+)/", text))
    assert users <= HOME_PLACEHOLDERS, "a personal /home/<user>/ path in docs/environment.md"
