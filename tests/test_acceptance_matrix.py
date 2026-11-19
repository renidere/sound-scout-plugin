"""Lint of docs/ACCEPTANCE.md, the Gate 3 matrix of issue #42 (R01–R12, T01–T10)."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCEPTANCE = REPO_ROOT / "docs" / "ACCEPTANCE.md"
ITEMS = tuple(f"R{n:02d}" for n in range(1, 13)) + tuple(f"T{n:02d}" for n in range(1, 11))
STATUSES = ("проверено", "partial", "blocked", "not_applicable")
TEST_ID = re.compile(r"`(tests/test_\w+\.py::[\w:\[\]|.-]+)`")
ISSUE_LINK = re.compile(r"\[#\d+\]\(https://github\.com/renidere/sound-scout-plugin/issues/\d+\)")
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
SECTIONS = (
    "Матрица",
    "Статусы",
    "Evaluation note",
    "Что передать владельцу",
    "Риски и незавершённое",
    "Ресурсный профиль",
)
STATUS_LAYERS = (
    "Код",
    "Audio capability",
    "Русская квалификация",
    "CLI/MCP (T09a)",
    "Hermes (T09b)",
)


def sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    return {part.split("\n", 1)[0].strip(): part for part in parts}


def table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split(" | ")])
    return rows[1:]


def collected_ids() -> set[str]:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    ids = {line for line in done.stdout.splitlines() if "::" in line}
    return ids | {node.split("[", 1)[0] for node in ids}


def matrix_problems(text: str, collected: set[str]) -> list[str]:
    """Every problem of the matrix section; an empty list means it passes."""
    matrix = sections(text).get("Матрица")
    if matrix is None:
        return ["no section «Матрица»"]
    problems = []
    seen: dict[str, int] = {}
    for cells in table_rows(matrix):
        item = cells[0]
        if item not in ITEMS:
            problems.append(f"unknown item {item!r}")
            continue
        seen[item] = seen.get(item, 0) + 1
        if len(cells) != 5:
            problems.append(f"{item}: {len(cells)} cells instead of 5")
            continue
        status, _, tests, evidence = cells[1:]
        if status not in STATUSES:
            problems.append(f"{item}: status {status!r} is not one of {STATUSES}")
        cited = TEST_ID.findall(tests)
        if status in ("проверено", "partial") and not cited:
            problems.append(f"{item}: no test cited")
        for test in cited:
            if test not in collected:
                problems.append(f"{item}: test {test} is not collected by pytest")
        if status in ("partial", "blocked") and not (
            "blocked:" in evidence and ISSUE_LINK.search(evidence)
        ):
            problems.append(f"{item}: {status} without «blocked:» naming its blocker issue")
        if status == "not_applicable" and "not_applicable" not in evidence:
            problems.append(f"{item}: not_applicable without its reason")
    problems += [f"{item}: missing" for item in ITEMS if item not in seen]
    problems += [f"{item}: {count} rows" for item, count in seen.items() if count > 1]
    return problems


@pytest.fixture(scope="module")
def text() -> str:
    return ACCEPTANCE.read_text(encoding="utf-8")


def test_acceptance_matrix_lists_each_item_once_with_collected_tests(text: str) -> None:
    assert matrix_problems(text, collected_ids()) == []


def test_matrix_tests_column_names_tests_only_by_their_pytest_node_id(text: str) -> None:
    for cells in table_rows(sections(text)["Матрица"]):
        leftover = TEST_ID.sub("", cells[3]).replace(",", "").strip()
        assert leftover in ("", "—"), (cells[0], leftover)


def test_acceptance_doc_has_every_section(text: str) -> None:
    titles = list(sections(text))
    for title in SECTIONS:
        assert any(found.startswith(title) for found in titles), title


def test_statuses_name_five_layers_each_with_a_date_and_evidence(text: str) -> None:
    rows = {cells[0]: cells for cells in table_rows(sections(text)["Статусы"])}
    for layer in STATUS_LAYERS:
        row = next(cells for name, cells in rows.items() if name.startswith(layer))
        status, date, evidence = row[1:]
        assert status, layer
        assert DATE.fullmatch(date), layer
        assert "`" in evidence or ISSUE_LINK.search(evidence), layer
    assert all(row[1] != "complete" for row in rows.values())


def test_statuses_follow_gate0_and_the_open_blockers(text: str) -> None:
    rows = {cells[0]: cells for cells in table_rows(sections(text)["Статусы"])}
    by_layer = {
        layer: next(cells for name, cells in rows.items() if name.startswith(layer))
        for layer in STATUS_LAYERS
    }
    assert by_layer["Audio capability"][1] == "NOT READY"
    assert by_layer["Русская квалификация"][1] == "NOT QUALIFIED"
    assert by_layer["Hermes (T09b)"][1] == "проверено"
    assert "`c70d7c00363e4c6d8427b0101730505b`" in by_layer["Hermes (T09b)"][3]
    assert "0007-gate0-result.md" in sections(text)["Статусы"]


def test_evaluation_note_says_qualification_was_not_done_and_why(text: str) -> None:
    note = sections(text)["Evaluation note"]
    assert "Квалификация не проводилась" in note
    for word in ("held-out", "human labels", "precision", "recall", "abstention", "Failures"):
        assert word in note, word
    assert "issues/29" in note


def test_resource_profile_points_to_the_environment_doc(text: str) -> None:
    assert "docs/environment.md" in sections(text)["Ресурсный профиль"]


SYNTHETIC_HEADER = (
    "## Матрица\n\n| пункт | статус | код | тесты | evidence или blocked |\n|---|---|---|---|---|\n"
)


def synthetic(rows: dict[str, str]) -> str:
    return SYNTHETIC_HEADER + "".join(rows.values())


def good_rows() -> dict[str, str]:
    return {
        item: f"| {item} | проверено | `src/x.py` | `tests/test_a.py::test_b` | doc |\n"
        for item in ITEMS
    }


def test_lint_accepts_a_complete_synthetic_matrix() -> None:
    rows = good_rows()
    rows["R01"] = "| R01 | проверено | `src/x.py` | `tests/test_a.py::test_c` | doc |\n"
    collected = {"tests/test_a.py::test_b", "tests/test_a.py::test_c"}
    assert matrix_problems(synthetic(rows), collected) == []


def test_lint_reports_a_missing_and_a_duplicated_item() -> None:
    rows = good_rows()
    del rows["T10"]
    rows["dup"] = rows["R03"]
    problems = matrix_problems(synthetic(rows), {"tests/test_a.py::test_b"})
    assert problems == ["T10: missing", "R03: 2 rows"]


def test_lint_reports_a_test_pytest_does_not_collect() -> None:
    rows = good_rows()
    rows["T02"] = "| T02 | проверено | `src/x.py` | `tests/test_a.py::test_gone` | doc |\n"
    problems = matrix_problems(synthetic(rows), {"tests/test_a.py::test_b"})
    assert problems == ["T02: test tests/test_a.py::test_gone is not collected by pytest"]


def test_lint_reports_a_bad_status_and_a_blocked_row_without_its_blocker() -> None:
    rows = good_rows()
    rows["R01"] = "| R01 | ready | `src/x.py` | `tests/test_a.py::test_b` | doc |\n"
    rows["R02"] = "| R02 | partial | `src/x.py` | `tests/test_a.py::test_b` | ждёт владельца |\n"
    rows["R03"] = (
        "| R03 | partial | `src/x.py` | — | "
        "blocked: [#68](https://github.com/renidere/sound-scout-plugin/issues/68) |\n"
    )
    rows["R04"] = "| R04 | not_applicable | — | — | облако |\n"
    problems = matrix_problems(synthetic(rows), {"tests/test_a.py::test_b"})
    assert problems == [
        f"R01: status 'ready' is not one of {STATUSES}",
        "R02: partial without «blocked:» naming its blocker issue",
        "R03: no test cited",
        "R04: not_applicable without its reason",
    ]


def test_lint_reports_a_document_without_the_matrix() -> None:
    assert matrix_problems("# Приёмка\n\n## Статусы\n", set()) == ["no section «Матрица»"]


ISSUE_40 = "[#40](https://github.com/renidere/sound-scout-plugin/issues/40)"
T09B_EVIDENCE = (
    "Hermes v0.23.1 (`3b4215785882`)",
    "`2d37c3158ab94ed7894ccaa69624a592`",
    "`3e85aec94e234233ae5068b0ee39f74b`",
    "`compare/dee942d962c84a78b6b3602e75e73f6b.json`",
)


def status_row(text: str, layer: str) -> list[str]:
    rows = table_rows(sections(text)["Статусы"])
    return next(cells for cells in rows if cells[0].startswith(layer))


def matrix_row(text: str, item: str) -> list[str]:
    return next(cells for cells in table_rows(sections(text)["Матрица"]) if cells[0] == item)


def test_t09b_status_shows_the_issue_40_run_ids_hermes_version_and_date(text: str) -> None:
    _, status, date, evidence = status_row(text, "Hermes (T09b)")
    assert status == "проверено"
    assert date == "2026-11-19"
    for fact in T09B_EVIDENCE:
        assert fact in evidence, fact
    assert f"{ISSUE_40}, закрыта" in evidence
    assert "2026-11-19" in matrix_row(text, "T09")[4]


def test_acceptance_doc_copies_no_private_absolute_path(text: str) -> None:
    for marker in ("/home/", "/Users/", "~/", ".hermes/"):
        assert marker not in text, marker


def test_r12_and_t09_rows_do_not_refer_to_issue_40_as_open(text: str) -> None:
    for item in ("R12", "T09"):
        evidence = matrix_row(text, item)[4]
        assert ISSUE_40 not in evidence.partition("blocked:")[2], item
        closed = re.findall(re.escape(ISSUE_40) + r",? закрыта", evidence)
        assert evidence.count(ISSUE_40) == len(closed) >= 1, item


ISSUE_82 = "[#82](https://github.com/renidere/sound-scout-plugin/issues/82)"
EPIC_7 = "sound-scout-plugin/issues/7)"
DESKTOP_RUN = "`c70d7c00363e4c6d8427b0101730505b`"
DESKTOP_CALL = "issues/82#issuecomment-7006876118"
OWNER_DECISION = "issues/82#issuecomment-7006985134"


def hermes_risk_item(text: str) -> str:
    risks = next(part for title, part in sections(text).items() if title.startswith("Риски"))
    items = re.split(r"^- ", risks, flags=re.MULTILINE)
    return next(item for item in items if item.startswith("**Hermes"))


def test_r08_cites_the_issue_40_evidence_instead_of_an_unchecked_hermes_session(
    text: str,
) -> None:
    status, evidence = matrix_row(text, "R08")[1], matrix_row(text, "R08")[4]
    assert status == "проверено"
    assert "не проверено — это T09b" not in evidence
    assert "2026-11-19" in evidence
    assert f"{ISSUE_40}, закрыта" in evidence
    for fact in T09B_EVIDENCE:
        assert fact in evidence, fact
    if "мессенджер" in evidence:
        assert ISSUE_82 in evidence


def desktop_places(text: str) -> dict[str, str]:
    return {
        "R12": matrix_row(text, "R12")[4],
        "T09": matrix_row(text, "T09")[4],
        "Hermes (T09b)": status_row(text, "Hermes (T09b)")[3],
        "Hermes risk item": hermes_risk_item(text),
    }


def test_the_desktop_native_call_is_the_t09b_evidence_and_no_messenger_gap_blocks(
    text: str,
) -> None:
    for place, body in desktop_places(text).items():
        assert DESKTOP_RUN in body, place
        assert DESKTOP_CALL in body, place
        assert OWNER_DECISION in body, place
        blocked = body.partition("blocked:")[2]
        assert "мессенджер" not in blocked, place
        assert ISSUE_82 not in blocked, place
        assert EPIC_7 not in body, place
    assert "не проверен" not in text.split("## Матрица", 1)[0]
    assert DESKTOP_RUN in text.split("## Матрица", 1)[0]


def test_desktop_rows_keep_the_missing_audio_backend_limit(text: str) -> None:
    for place, body in desktop_places(text).items():
        assert "`diction` и `prosody` — `not_assessed`" in body, place
    assert matrix_row(text, "R12")[1] == "проверено"
    t09 = matrix_row(text, "T09")
    assert t09[1] == "partial"
    assert "issues/68" in t09[4].partition("blocked:")[2]


def test_r08_cites_the_desktop_call_and_names_nothing_unchecked(text: str) -> None:
    evidence = matrix_row(text, "R08")[4]
    assert "не провер" not in evidence
    assert DESKTOP_RUN in evidence
    assert DESKTOP_CALL in evidence
    assert "в настоящей сессии Hermes 2026-11-19" in evidence


def test_criterion1_the_four_places_name_the_messenger_only_as_not_required(text: str) -> None:
    for place, body in desktop_places(text).items():
        mentions = [s for s in re.split(r"(?<=[.;:])\s+", body) if "мессенджер" in s]
        assert mentions, place
        for sentence in mentions:
            assert sentence.startswith("вызов через мессенджер не требуется"), (place, sentence)


def test_criterion1_the_desktop_call_claims_no_speech_compare_and_no_hearing(text: str) -> None:
    for place, body in desktop_places(text).items():
        assert re.search(
            r"`speech_compare`[^.]*Desktop[^.]*не вызывался|"
            r"`speech_compare` в этой сессии не вызывался",
            body,
        ), place
        assert "`partial`" in body, place
    for item in ("R12", "T09"):
        assert "`audio_review` — `unavailable` (`backend = none`)" in matrix_row(text, item)[4]
    assert "validated_ru` — true" not in text
