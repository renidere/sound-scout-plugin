"""Decision 0022 names the next Gate 0 candidate, its hypothesis and the owner's numbers (#143)."""

import re
from pathlib import Path

from sound_scout.eval import gate0

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs/decisions/0022-gate0-next-candidate.md"
REVISION = "89b785438c8901d4635e42f50480ba5985a1bbf1"
RUNNER_UP_REVISION = "501d13dba70e3c02b1b12c78fd9e6b08ad413960"
COLUMNS = 9


def text() -> str:
    return DOC.read_text(encoding="utf-8")


def flat(value: str) -> str:
    return " ".join(value.split())


def section(heading: str) -> str:
    document = text()
    level = heading.split(" ", 1)[0]
    start = document.index(f"\n{heading}\n")
    rest = document[start + len(heading) + 2 :]
    ends = [m.start() for m in re.finditer(rf"^#{{1,{len(level)}}} ", rest, re.MULTILINE)]
    return rest[: ends[0]] if ends else rest


def table_rows(block: str) -> list[list[str]]:
    rows = [line for line in block.splitlines() if line.startswith("|")]
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
    return [row for row in cells[2:] if row]


def bytes_of(cell: str) -> int:
    return int(cell.replace(" ", "").replace(" ", ""))


def test_comparison_covers_five_untested_candidates_with_every_field() -> None:
    rows = table_rows(section("## 2. Сравнение кандидатов"))
    assert len(rows) >= 5
    for row in rows:
        assert len(row) == COLUMNS, row
        assert all(row), row
    names = " ".join(row[0] for row in rows)
    for tested in ("Gemma 4 E4B", "Borealis", "GigaChat", "MOSS"):
        assert tested not in names
    verdicts = [row[-1] for row in rows]
    assert sum("**выбран**" in v for v in verdicts) == 1
    assert sum("**запасной**" in v for v in verdicts) == 1


def test_choice_meets_p3_v1_and_the_rule_stays_unchanged() -> None:
    facts = flat(section("### 3.1. Факты (первичные источники, 2026-11-21)"))
    assert f"ревизия `{REVISION}`" in facts
    assert "кадр представления ≈ 40 мс. Это 25 позиций на секунду звука" in facts
    assert "750 ≥ 20 × длительность" in facts
    assert "Маршрут llama.cpp отличается от эталона" in facts
    assert "`GATE0-RULE v1` из `0007` §3 без изменений" in text()
    assert "Qwen3-Omni" in section("### 3.3. Запасной кандидат и отклонённые")
    assert f"ревизия `{RUNNER_UP_REVISION}`" in flat(
        section("### 3.3. Запасной кандидат и отклонённые")
    )


def test_hypothesis_is_testable_and_states_no_result_as_fact() -> None:
    hypothesis = flat(section("### 3.2. Гипотеза H-Q25O"))
    assert "проходит C (12 из 12" in hypothesis
    assert "не меньше 2 из 3 синтетических пар" in hypothesis
    assert "**Опровержение.**" in hypothesis
    assert "Агент исход не предсказывает" in hypothesis
    assert "Клипы, вопросы" not in hypothesis  # #154: V2 and V3 are not the files of 0007
    assert "Клипы — набор 2026-11-19, тот же, что у `0010` и `0011`" in hypothesis
    assert "`V2` и `V3` в нём — не те файлы, что в `0007` (`0010` §2.1, #86)" in hypothesis
    for outcome in ("P1–P3 нарушено", "C не прошёл", "C прошёл, S не прошёл", "C и S прошли"):
        assert f"| {outcome} |" in section("### 3.2. Гипотеза H-Q25O")
    assert not re.search(r"(?i)пройд[её]т gate 0|получит ready|будет ready", text())


def test_download_names_revision_files_and_their_exact_total() -> None:
    download = section("## 4. Загрузка")
    assert f"resolve/{REVISION}/" in download
    rows = table_rows(download)
    assert [row[0] for row in rows] == [
        "`Qwen2.5-Omni-7B-Q8_0.gguf`",
        "`mmproj-Qwen2.5-Omni-7B-f16.gguf`",
    ]
    for row in rows:
        assert re.fullmatch(r"`[0-9a-f]{64}`", row[2]), row
    total = sum(bytes_of(row[1]) for row in rows)
    assert total == 10_742_180_512
    assert "**Всего: 10 742 180 512 байт ≈ 10,74 GB.**" in download


def test_memory_ceiling_and_context_match_the_llama_runtime() -> None:
    ceiling = flat(section("## 5. Потолок памяти"))
    assert gate0.RSS_LIMIT_BYTES == 18 * 10**9
    assert "**18 GB: RSS сервера ≤ 18e9 байт**" in ceiling
    assert "Это оценка, не измерение" in ceiling
    assert gate0.CONTEXT_TOKENS == 4096
    assert "`-c 4096`" in section("## 7. План runtime")


def test_runtime_plan_preregisters_the_clip_set_of_2026_11_19() -> None:
    """#155: step 3 names the set of `0010` §2.1, not `0007` §2.1, as the clip source."""
    plan = flat(section("## 7. План runtime"))
    step = plan.split("3. Предрегистрация:", 1)[1].split("4. Проверки до запросов", 1)[0]
    assert "клипы по `0007` §2.1" not in step
    assert "`0007` §2.1" not in step
    assert "клипы набора 2026-11-19 (`0010` §2.1;" in step
    assert "`V2` и `V3` в нём — не те файлы, что в `0007`, #86)" in step


def test_wall_time_per_clip_is_an_estimate_from_measured_runs() -> None:
    timing = flat(section("## 6. Время на клип Gate 0"))
    assert timing.startswith("Оценка, не измерение.")
    assert "**На один клип Gate 0** (K1, K2, P, A): ≈ 4–7 мин." in timing
    assert "`0007` §4" in timing


def test_owner_ask_names_model_revision_licence_size_and_ceiling() -> None:
    ask = flat(section("## 9. Что нужно от владельца"))
    for item in (
        "**Модель:** Qwen2.5-Omni-7B",
        f"**Ревизия:** `{REVISION}`",
        "**Лицензия:** Apache-2.0",
        "`qwen-research`",
        "**Загрузка:** 2 файла, 10 742 180 512 байт ≈ **10,74 GB**",
        "**Потолок памяти:** **18 GB**",
        "**Риск:** русская речь в первичных источниках не заявлена",
    ):
        assert item in ask, item


def test_weights_are_recorded_only_under_the_owner_approval_and_links_resolve() -> None:
    dependencies = (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    sections = [part for part in dependencies.split("\n#### ") if REVISION in part]
    assert len(sections) == 1  # #145, after owner decision 7065525535 on #90
    assert sections[0].startswith("Qwen2.5-Omni-7B") and "issuecomment-7065525535" in sections[0]
    assert "веса не загружались, inference не было" in text()
    for name in re.findall(r"`(00\d\d)`", text()):
        assert len(list((REPO / "docs/decisions").glob(f"{name}-*.md"))) == 1, name
    for comment in ("90#issuecomment-7064906952", "90#issuecomment-7024154195"):
        assert f"https://github.com/renidere/sound-scout-plugin/issues/{comment}" in text()
    assert not re.search(r"/(home|Users)/|[A-Za-z]:\\", text())
