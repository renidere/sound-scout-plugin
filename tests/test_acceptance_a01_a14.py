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


def test_a11_names_the_chat_path_and_no_longer_waits_for_an_owner_decision() -> None:
    text = TABLE_DOC.read_text(encoding="utf-8")
    a11 = next(row for row in table(text) if row[0] == "A11")
    for word in (
        "«Отправить и разобрать»",
        "`sound-scout-job:<job_id>`",
        "после обеих заметок пользователь просит",
        "«Обычная отправка»",
        "`speech_job_result`",
        "`speech_compare`",
        "шаг 7",
        "«Задание контрольной попытки уже есть»",
    ):
        assert word in a11[2], word
    assert "plugin.js" not in a11[3]
    assert "`hermes/skills/speech-review/SKILL.md`" in a11[3]
    skill = (REPO_ROOT / "hermes" / "skills" / "speech-review" / "SKILL.md").read_text("utf-8")
    step = skill.split("7. **Сравнение попыток.**", 1)[1]
    assert "**Задание контрольной попытки уже есть.**" in step
    assert "нет evidence на живой системе" in a11[3]
    assert status(a11) == "не проверено"
    assert f"{REPO_URL}/issues/90" in a11[4]
    assert f"{REPO_URL}/issues/175#issuecomment-7125074558" in a11[4]
    assert "нужно решение владельца" not in a11[4]
    assert not any("Нужен владелец" in line and "A11" in line for line in text.splitlines())


OWNER_RUN = f"{REPO_URL}/issues/112#issuecomment-7125530016"


def test_the_owner_run_of_24_november_is_stand_s3_and_backs_every_pass() -> None:
    text = TABLE_DOC.read_text(encoding="utf-8")
    stands = text.split("## Стенды", 1)[1].split("\n## ", 1)[0]
    s3 = stands.split("- **S3**", 1)[1].split("\n- **", 1)[0]
    assert "cb0e3979ffeb9760f4928d113fae0b72fbb5e11ffd411418b145a2a2980f406e" in s3
    assert "`main` `b1f2b27`" in s3
    assert OWNER_RUN in s3
    rows = {row[0]: row for row in table(text)}
    assert {rid for rid, row in rows.items() if status(row) == "пройдено"} == set()
    for rid in ("A01", "A12"):
        assert OWNER_RUN in rows[rid][3], rid
        assert status(rows[rid]) == "частично", rid
        assert "S3" in rows[rid][4] and "2026-11-24" in rows[rid][4], rid
    assert "ждёт владельца: O9" in rows["A01"][4]
    assert "Версии Hermes и Desktop стенда S3" in rows["A01"][4]
    assert "ждёт владельца: O6" in rows["A12"][4]
    assert "«замечаний нет»" in rows["A12"][3]
    assert "Tab, Enter и Space" not in rows["A12"][3]


def test_the_rows_never_run_live_stay_open_and_name_what_is_missing() -> None:
    rows = {row[0]: row for row in table(TABLE_DOC.read_text(encoding="utf-8"))}
    assert status(rows["A08"]) == "ждёт владельца: O6"
    assert "`не проверено на живой системе`" in rows["A08"][4]
    for rid in ("A06", "A10", "A13"):
        assert status(rows[rid]) == "частично", rid
        assert "ждёт владельца: O6" in rows[rid][4], rid
    assert "ждёт владельца: O9" in rows["A02"][4]
    assert "`не проверено на живой системе`" in rows["A03"][4]
    assert "`не проверено на живой системе`" in rows["A05"][4]
    a07 = rows["A07"]
    assert status(a07) == "частично"
    assert f"{REPO_URL}/issues/140#issuecomment-7125714049" in a07[3]
    assert "ждёт #140" not in a07[4] and "issues/140" not in a07[4]
    assert "«разрыв MCP» — путь задания MCP из #140" not in a07[2]
    assert "«разрыв MCP» — `не проверено на живой системе`" in a07[4]
    mcp_break = "test_a_queued_job_runs_and_is_delivered_once_after_the_server_stops"
    assert f"«разрыв MCP» — контракт, не приёмка: `{mcp_break}`" in a07[3]
    mcp_tests = (REPO_ROOT / "tests" / "test_mcp_server.py").read_text(encoding="utf-8")
    assert f"def {mcp_break}(" in mcp_tests
    assert "контрактного теста именно обрыва нет" not in a07[4]


def test_o6_is_rewritten_to_cancel_the_second_job_while_the_first_runs() -> None:
    text = TABLE_DOC.read_text(encoding="utf-8")
    steps = text.split("## Шаги владельца", 1)[1].split("\n## ", 1)[0]
    assert OWNER_RUN in steps
    o6 = steps.split("- **O6.**", 1)[1].split("\n- **", 1)[0]
    for words in (
        "Не выполнен 2026-11-24",
        "запиши около 3 минут, запиши вторую запись, отмени вторую, пока идёт первая, "
        "затем повтори",
        "«Отменить разбор», пока первое ещё идёт",
        "«Повторить разбор»",
        "«Открыть отчёт»",
        "нажимай клавишей Tab, затем Enter или Space",
        "Кнопка «Отмена» рекордера отменяет запись",
        # #187: the retried attempt's result comes to the chat as a note of its own
        "В X должны прийти три заметки",
        "об отмене (метка `sound-scout-job:<job_id>`)",
        "об итоге повтора отдельной заметкой (метка `sound-scout-job:<job_id>-a1`",
    ):
        assert words in o6, words
    for n in (1, 3, 4, 5, 7, 8):
        assert f"- **O{n}.** Выполнено 2026-11-24" in steps, n
    assert "- **O2.** Проверено владельцем ранее" in steps
    o9 = steps.split("- **O9.**", 1)[1].split("\n", 1)[0]
    for words in (
        "одно подключение или несколько",
        "`Sound Scout: health`",
        "версии Hermes и Desktop",
    ):
        assert words in o9, words


def test_acceptance_dates_the_owner_run_with_its_stand_versions() -> None:
    section = acceptance_section()
    assert "на 2026-11-24" in section
    assert "стенд S3: `main` `b1f2b27`, `plugin.js` `cb0e3979…`" in section
    assert OWNER_RUN in section
    assert "пройдено 0, частично 12" in section
    assert "O2 (проверен владельцем ранее, стенд не назван)" in section
    assert "Не выполнен шаг O6" in section


def test_o10_and_o11_install_0_4_0_and_check_the_player_in_the_chat_and_in_the_plugin() -> None:
    """#216 criterion 5: the owner's Windows steps for the recording player."""
    steps = TABLE_DOC.read_text(encoding="utf-8").split("## Шаги владельца", 1)[1]
    o10 = steps.split("- **O10.**", 1)[1].split("\n- **", 1)[0]
    o11 = steps.split("- **O11.**", 1)[1].split("\n", 1)[0]
    assert "`sound-scout` 0.4.0" in o10 and "`version=0.4.0 pid=…`" in o10
    assert "0.3.0" not in o10 + o11
    for words in (
        "`sound-scout` 0.4.0",
        "0039",
        "Под ответом — блок с именем `sound-scout-….wav` и плеером",
        "⋮ плеера → «Скачать»",
        "`certutil -hashfile <файл> SHA256`",
        "Вернись к первой и включи её: звучит первая запись",
        "Закрой и снова открой Desktop",
        "На карточке задания нажми «Прослушать», затем «Пуск»",
        "время растёт и не сбрасывается, пока карточка обновляется",
        "«Пауза», «−5 с», «+5 с», «В начало»",
        "Если плеера под ответом нет, запиши, что видно",
        "Если плеер плагина не играет, запиши текст ошибки у плеера",
    ):
        assert words in o11, words
