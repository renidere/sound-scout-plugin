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
