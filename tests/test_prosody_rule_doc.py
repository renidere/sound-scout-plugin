"""Decision 0019: the PROSODY-RULE v1 block is fixed before any run, like GATE0-RULE v1 in 0007.

The hash is taken the way the document says (and `gate0.rule_sha256` does): the text between the
markers, stripped of surrounding whitespace, as UTF-8.
"""

import hashlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs/decisions/0019-next-audio-experiment.md"
BEGIN = "<!-- prosody-rule:begin -->"
END = "<!-- prosody-rule:end -->"
# SHA-256 of the PROSODY-RULE v1 block as written for #119, before any run
PROSODY_RULE_SHA256 = "b6ce9fcfbf24b8fa706b2470d47b9967f896fcf677fe5b9326b72bf6af28cc65"


def rule_sha256(document: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("the document must hold one prosody-rule block")
    block = document[document.index(BEGIN) + len(BEGIN) : document.index(END)].strip()
    if not block.startswith("```text\nPROSODY-RULE v1"):
        raise ValueError("the block does not start with PROSODY-RULE v1")
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def test_prosody_rule_block_hash_is_the_one_recorded_before_the_run() -> None:
    document = DOC.read_text(encoding="utf-8")
    assert rule_sha256(document) == PROSODY_RULE_SHA256
    assert f"SHA-256 блока: `{PROSODY_RULE_SHA256}`" in document


def test_prosody_rule_hash_changes_when_the_block_is_edited() -> None:
    document = DOC.read_text(encoding="utf-8")
    edited = document.replace("AUC >= 0,75", "AUC >= 0,70", 1)
    assert edited != document
    assert rule_sha256(edited) != PROSODY_RULE_SHA256


@pytest.mark.parametrize(
    "broken",
    [
        lambda d: d.replace(BEGIN, ""),
        lambda d: d + BEGIN + END,
        lambda d: d.replace("PROSODY-RULE v1 (", "PROSODY-RULE v2 ("),
    ],
)
def test_prosody_rule_block_refuses_missing_doubled_or_other_rule(broken) -> None:
    with pytest.raises(ValueError):
        rule_sha256(broken(DOC.read_text(encoding="utf-8")))


def test_proposal_waits_for_the_owner_and_holds_no_personal_paths() -> None:
    document = DOC.read_text(encoding="utf-8")
    assert "**предложение; ждёт решения владельца.** Не одобрено" in document
    assert "**нужен владелец:** одобрить или отклонить предложение `prosody-measure-1`" in document
    assert not re.search(r"/(home|Users)/|~/|[A-Za-z]:\\", document)


REVISION_HEADING = "## Редакция 2 (2026-11-21): что уточнено"
OWNER_DECISION = "https://github.com/renidere/sound-scout-plugin/issues/119#issuecomment-7051943525"
# claims the owner rejected in OWNER_DECISION; only the revision section may quote them
REJECTED_CLAIMS = (
    "Общее у всех четырёх",
    "Все четыре проверенные аудиомодели не различили",
    "ожидаемый исход — тот же 0/3",
    "повтор провала",
    "Runtime-fidelity лечит контракт A",
    "обосновывает остановку трат",
    "становится планкой",
    "не стоит денег",
    "вариант владельца при любом исходе",
)


def without_revision_section(document: str) -> str:
    start = document.index(REVISION_HEADING)
    end = document.index("\n## ", start + len(REVISION_HEADING))
    return document[:start] + document[end:]


def test_revision_2_names_the_owner_decision_and_keeps_the_proposal_auxiliary() -> None:
    document = DOC.read_text(encoding="utf-8")
    assert REVISION_HEADING in document
    assert OWNER_DECISION in document
    flat = " ".join(document.split())
    assert "**неодобренное вспомогательное предложение**" in flat
    assert "не снимает блокировку #90" in flat
    assert "честный ограниченный режим" in flat
    assert "редакция 2 его не разрешает" in flat


def test_rejected_claims_appear_only_where_the_revision_names_them() -> None:
    document = DOC.read_text(encoding="utf-8")
    body = " ".join(without_revision_section(document).split())
    for claim in REJECTED_CLAIMS:
        assert claim not in body, claim
    revision = " ".join(document[document.index(REVISION_HEADING) :].split())
    assert "повтор провала" in revision


def section(document: str, heading: str) -> str:
    start = document.index(heading) + len(heading)
    end = re.search(r"\n#{2,3} ", document[start:])
    return " ".join(document[start : start + end.start() if end else None].split())


def test_four_results_are_split_by_validity_not_listed_as_equal_qualifications() -> None:
    document = DOC.read_text(encoding="utf-8")
    formal = section(document, "### 1.1. Формальные прогоны Gate 0")
    other = section(document, "### 1.2. Недействительный и диагностический прогоны")
    assert "(не квалификация способности)" in document
    for candidate in ("Gemma 4 E4B-it, `0007`", "Borealis-5b-it, `0010`"):
        assert candidate in formal and candidate not in other
    for candidate in ("GigaChat3.1-Audio-10B-A1.8B, `0011`", "MOSS-Audio-8B-Thinking-ru"):
        assert candidate in other and candidate not in formal
    assert "NOT READY по P3" in other and "прогон недействителен" in other
    assert "не формальный Gate 0" in other and "маршрут DeepStack полностью не доказан" in other
    assert formal.count("NOT READY по C") == 2


def test_proposal_keeps_the_original_goal_and_the_backend_search() -> None:
    flat = " ".join(DOC.read_text(encoding="utf-8").split())
    assert "Оно не заменяет поиск пригодного audio backend и не отменяет исходную цель" in flat
    assert "не выполнение полной цели продукта и не замена поиску backend" in flat


def test_facts_limits_hypotheses_and_the_open_backend_question_have_sections() -> None:
    document = DOC.read_text(encoding="utf-8")
    for heading in ("### 1.3. Факты", "### 1.4. Ограничения", "### 1.5. Гипотезы"):
        assert section(document, heading)
    question = section(document, "### 4.1. Цель и границы вывода")
    assert "**Вопрос эксперимента:**" in question and "**На что он не отвечает:**" in question
    backend = section(document, "### 4.8. Что остаётся нерешённым для audio backend (#90)")
    assert "Кандидата с вердиктом READY нет" in backend
    assert "Выбирать модель или техническую реализацию владельцу не нужно" in backend
