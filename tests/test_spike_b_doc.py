"""Decision 0015 steps 4 and 5 follow stage A of 0017 (#129)."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs/decisions/0015-spike-b-backend-package.md"
AUTHORITY = REPO / "docs/decisions/0017-session-authority-owner.md"
STAGE_A_LINK = "(0017-session-authority-owner.md#этап-a-2026-11-20)"


def step(heading: str) -> str:
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = lines.index(heading)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("#"))
    return " ".join(" ".join(lines[start + 1 : end]).split())


def test_step_4_expects_session_owner_per_port_after_stage_a() -> None:
    health = step("### 4. Критерий 3: `GET /health` через `ctx.rest` с Windows")
    assert "которую выбирает владелец" not in health
    assert "только после смены топологии" not in health
    assert "С 2026-11-20 топология выбрана, этап A выполнен" in health
    assert STAGE_A_LINK in health
    assert "`:9119` (`hermes serve`) — уведомление `ok=true pid=<N> session_owner=false`" in health
    assert "`:9120` (gateway `hermes-gateway.service`) — уведомление" in health
    assert "`ok=true pid=<N> session_owner=true`" in health


def test_step_5_keeps_the_9119_failure_and_names_the_stage_b_condition() -> None:
    delivery = step("### 5. Критерий 4: доставка заметки ровно один раз")
    assert "на `:9119` идёт в `hermes serve`" in delivery
    assert "no gateway in this process owns its session" in delivery
    assert "`state=failed`" in delivery
    assert "После этапа A (2026-11-20) это верно" in delivery
    assert "Условие этапа B 0017" in delivery
    assert "остановить и отключить `hermes-autonomous.service`" in delivery
    assert "Windows-клиент работает только через gateway" in delivery
    assert "«Остаток: клиентская проверка»" in delivery
    assert "Шаги 5.1–5.4 вызывают маршруты Spike B `/spike/*`" in delivery
    assert "до этапа B" in delivery
    assert "после смены топологии" not in delivery


def github_slug(heading: str) -> str:
    text = heading.lstrip("#").strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def test_stage_a_link_resolves_and_stage_b_condition_matches_0017() -> None:
    authority = AUTHORITY.read_text(encoding="utf-8")
    headings = [line for line in authority.splitlines() if line.startswith("#")]
    anchor = STAGE_A_LINK.split("#", 1)[1].rstrip(")")
    assert anchor in [github_slug(h) for h in headings]
    assert "## Этап A (2026-11-20)" in headings
    stage_b = " ".join(authority.split("### Остаток: этап B", 1)[1].split("\n## ", 1)[0].split())
    assert "остановить и отключить `hermes-autonomous.service`" in stage_b
    assert "Windows-клиент работает только через gateway" in stage_b
    assert "«Остаток: этап B»" in step("### 5. Критерий 4: доставка заметки ровно один раз")
