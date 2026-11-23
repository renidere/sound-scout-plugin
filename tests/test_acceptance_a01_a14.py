"""Lint of docs/acceptance/A01-A14.md, the acceptance table of the unified plugin (#112)."""

import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_DOC = REPO_ROOT / "docs" / "acceptance" / "A01-A14.md"
ACCEPTANCE = REPO_ROOT / "docs" / "ACCEPTANCE.md"
IDS = tuple(f"A{n:02d}" for n in range(1, 15))
COLUMNS = ["ID", "критерий", "действие", "evidence", "результат"]
STATUS = re.compile(
    r"^`(пройдено|частично|не пройдено|не проверено|ждёт владельца: O\d(?:, O\d)*|ждёт #\d+)`"
)
REPO_URL = "https://github.com/renidere/sound-scout-plugin"
EVIDENCE_LINK = re.compile(
    re.escape(REPO_URL) + r"/(?:issues/\d+#issuecomment-\d+|commit/[0-9a-f]{7,40}|actions/runs/\d+)"
)
DATE = re.compile(r"\b2026-\d\d-\d\d\b")
STAND = re.compile(r"\bS\d")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PASSED_ALL = "A01–A14 пройдены"
SUMMARY = ("пройдено", "частично", "не проверено", "ждёт владельца")
SUMMARY_COUNT = re.compile(
    r"(?<![\wё])(?<!не )(не пройдено|пройдено|частично|не проверено|ждёт владельца|ждёт задачи)"
    r" (\d+)(?!\d)"
)


def table(text: str) -> list[list[str]]:
    """The rows of the table whose header is COLUMNS, header excluded."""
    lines = text.splitlines()
    header = "| " + " | ".join(COLUMNS) + " |"
    start = lines.index(header)
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("| "):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split(" | ")])
    return rows


def status(row: list[str]) -> str:
    match = STATUS.match(row[4])
    return match.group(1) if match else ""


def problems(rows: list[list[str]]) -> list[str]:
    """Everything wrong with the table: ids, columns, status words, the linked evidence behind
    `пройдено` (with a date and a stand) and behind the passed part of `частично`."""
    found = []
    if [row[0] for row in rows] != list(IDS):
        found.append(f"ids are {[row[0] for row in rows]}, not A01…A14 in order")
    for row in rows:
        if len(row) != len(COLUMNS):
            found.append(f"{row[0]}: {len(row)} cells")
            continue
        if not status(row):
            found.append(f"{row[0]}: the result does not start with a status word")
        evidence = row[3]
        if status(row) == "пройдено" and not (
            EVIDENCE_LINK.search(evidence) and DATE.search(evidence) and STAND.search(evidence)
        ):
            found.append(f"{row[0]}: `пройдено` without a linked, dated evidence on a stand")
        if status(row) == "частично" and not EVIDENCE_LINK.search(evidence):
            found.append(f"{row[0]}: `частично` without a linked evidence for its passed part")
    return found


def test_the_table_has_a01_to_a14_each_with_a_status_word_and_evidence_behind_a_pass() -> None:
    assert problems(table(TABLE_DOC.read_text(encoding="utf-8"))) == []


def test_the_checker_refuses_a_pass_or_a_partial_without_evidence_and_a_table_of_13_rows() -> None:
    rows = table(TABLE_DOC.read_text(encoding="utf-8"))
    tests_only = [*rows[0][:3], "контракт: `tests/test_jobs.py`", "`пройдено` — по тестам"]
    assert any("A01: `пройдено`" in p for p in problems([tests_only, *rows[1:]]))
    linked = [*rows[0][:3], f"S2 — {REPO_URL}/actions/runs/1, 2026-11-23", "`пройдено`"]
    assert problems([linked, *rows[1:]]) == []
    no_stand = [*rows[0][:3], f"{REPO_URL}/actions/runs/1, 2026-11-23", "`пройдено`"]
    assert any("A01: `пройдено`" in p for p in problems([no_stand, *rows[1:]]))
    unlinked = [*rows[0][:3], "S1 — П1 2026-11-23", "`частично` — часть пройдена"]
    assert any("A01: `частично`" in p for p in problems([unlinked, *rows[1:]]))
    assert any("not A01…A14" in p for p in problems(rows[:13]))
    unknown = [*rows[1][:4], "`ок` — всё хорошо"]
    assert any("A02: the result" in p for p in problems([rows[0], unknown, *rows[2:]]))


def test_the_stands_name_hermes_desktop_a_full_sha256_and_a_commit() -> None:
    text = TABLE_DOC.read_text(encoding="utf-8")
    stands = text.split("## Стенды", 1)[1].split("\n## ", 1)[0]
    assert "caafe5389a" in stands
    assert "Desktop 0.23.3" in stands
    assert re.search(r"\b[0-9a-f]{64}\b", stands)
    assert re.search(r"`main` `[0-9a-f]{7,40}`", stands)


def test_the_owner_steps_o1_to_o9_and_every_waited_step_are_written_out() -> None:
    text = TABLE_DOC.read_text(encoding="utf-8")
    steps = text.split("## Шаги владельца", 1)[1].split("\n## ", 1)[0]
    for n in range(1, 10):
        assert f"- **O{n}.**" in steps, n
    waited = {n for row in table(text) for n in re.findall(r"\b[OО](\d+)\b", row[4])}
    assert waited
    for n in waited:
        assert f"- **O{n}.**" in steps, n


def test_the_doc_holds_no_personal_path_and_no_ip_address() -> None:
    text = TABLE_DOC.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "C:\\Users" not in text
    assert not IPV4.search(text)


def category(row: list[str]) -> str:
    word = status(row)
    if word.startswith("ждёт владельца"):
        return "ждёт владельца"
    return "ждёт задачи" if word.startswith("ждёт #") else word


def summary_problems(rows: list[list[str]], section: str) -> list[str]:
    """What is wrong with the status count of the A01–A14 section of docs/ACCEPTANCE.md."""
    counts = Counter(category(row) for row in rows)
    written: dict[str, int] = {}
    for match in SUMMARY_COUNT.finditer(section):
        written.setdefault(match.group(1), int(match.group(2)))
    found = [f"{word} is not counted" for word in SUMMARY if word not in written]
    for word in sorted(set(counts) | set(written)):
        if written.get(word, 0) != counts.get(word, 0):
            found.append(f"{word}: written {written.get(word, 0)}, table {counts.get(word, 0)}")
    if sum(written.values()) != len(IDS):
        found.append(f"the count sums to {sum(written.values())}, not {len(IDS)}")
    return found


def acceptance_section() -> str:
    text = " ".join(ACCEPTANCE.read_text(encoding="utf-8").split())
    return text.split("## Приёмка единого плагина (A01–A14)", 1)[1].split(" ## ", 1)[0]


def test_acceptance_links_the_table_and_counts_its_statuses() -> None:
    rows = table(TABLE_DOC.read_text(encoding="utf-8"))
    section = acceptance_section()
    assert "`docs/acceptance/A01-A14.md`" in section
    assert summary_problems(rows, section) == []
    passed = sum(category(row) == "пройдено" for row in rows)
    text = ACCEPTANCE.read_text(encoding="utf-8")
    assert PASSED_ALL not in " ".join(text.split()) or passed == len(IDS)


def test_acceptance_holds_no_personal_path_and_no_ip_address() -> None:
    section = acceptance_section()
    assert "/home/" not in section
    assert "C:\\Users" not in section
    assert not IPV4.search(section)


def test_the_count_checker_refuses_a_wrong_or_prefix_count_and_an_uncounted_status() -> None:
    base = table(TABLE_DOC.read_text(encoding="utf-8"))[0][:4]

    def rows(*results: str) -> list[list[str]]:
        return [[*base, result] for result in results]

    current = rows(*["`частично` — x"] * 10, *["`ждёт владельца: O1` — x"] * 3, "`не проверено`")
    good = "пройдено 0, частично 10, не проверено 1, ждёт владельца 3."
    assert summary_problems(current, good) == []
    assert summary_problems(current, good.replace("частично 10", "частично 1"))
    one_partial = rows("`частично` — x", *["`ждёт владельца: O1` — x"] * 12, "`не проверено`")
    prefix = "пройдено 0, частично 10, не проверено 1, ждёт владельца 12."
    assert any("частично: written 10, table 1" in p for p in summary_problems(one_partial, prefix))
    failed = [*rows("`не пройдено` — x"), *current[1:]]
    assert any("не пройдено: written 0, table 1" in p for p in summary_problems(failed, good))
    waiting = [*rows("`ждёт #140` — x"), *current[1:]]
    assert any("ждёт задачи" in p for p in summary_problems(waiting, good))
    fixed = "пройдено 0, частично 9, не проверено 1, ждёт владельца 3, ждёт задачи 1."
    assert summary_problems(waiting, fixed) == []
    no_owner = good.replace(", ждёт владельца 3", "")
    assert "ждёт владельца is not counted" in summary_problems(current, no_owner)
