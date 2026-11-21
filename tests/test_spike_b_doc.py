"""Decision 0015 steps 4 and 5 follow stages A and B of 0017 (#129, #136)."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs/decisions/0015-spike-b-backend-package.md"
AUTHORITY = REPO / "docs/decisions/0017-session-authority-owner.md"
STAGE_A_LINK = "(0017-session-authority-owner.md#этап-a-2026-11-20)"
STAGE_B_LINK = "(0017-session-authority-owner.md#этап-b-2026-11-21)"


def step(heading: str) -> str:
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return " ".join(" ".join(lines[start + 1 : end]).split())


def test_step_4_expects_the_gateway_on_9119_after_stage_b() -> None:
    health = step("### 4. Критерий 3: `GET /health` через `ctx.rest` с Windows")
    assert "которую выбирает владелец" not in health
    assert "только после смены топологии" not in health
    assert STAGE_A_LINK in health and STAGE_B_LINK in health
    assert (
        "`:9119` (gateway `hermes-gateway.service`, с этапа B 2026-11-21) — единственный вход "
        "Desktop, уведомление `ok=true pid=<N> session_owner=true`" in health
    )
    assert "`systemctl --user show -p MainPID hermes-gateway.service`" in health
    assert "`hermes-autonomous.service` с этапа B остановлен и отключён" in health
    current, history = health.split("История.", 1)
    assert "9120" not in current
    assert "`:9119` (`hermes serve`)" not in health
    assert "`session_owner=false`" not in current
    assert "До этапа B `:9119` слушал `hermes serve`" in history
    assert "а gateway — временный порт `:9120`; 2026-11-21 Webapp gateway перенесён" in history


def test_9120_appears_in_one_history_note_only() -> None:
    document = DOC.read_text(encoding="utf-8")
    assert document.count("9120") == 1
    note = next(p for p in document.split("\n\n") if "9120" in p)
    assert note.lstrip().startswith("История. С 2026-11-20")
    assert "временный порт `:9120`; 2026-11-21" in " ".join(note.split())


def test_step_5_dates_the_9119_failure_and_records_stage_b() -> None:
    delivery = step("### 5. Критерий 4: доставка заметки ровно один раз")
    history = "История. До этапа B (по 2026-11-21) доставка через `ctx.rest` на `:9119` шла"
    assert history in delivery
    assert "no gateway in this process owns its session" in delivery
    assert "`state=failed`" in delivery
    assert "С 2026-11-21 (0017, «Этап B (2026-11-21)») `:9119` слушает gateway" in delivery
    assert "Условие этапа B 0017 («Остаток: этап B») выполнено" in delivery
    assert "не выполнено" not in delivery
    assert "остановить и отключить `hermes-autonomous.service`" in delivery
    assert "Windows-клиент работает только через gateway" in delivery
    assert "Шаги 5.1–5.4 вызывают маршруты Spike B `/spike/*`" in delivery
    assert "до этапа B" in delivery
    assert "заменила Spike B в gateway и убрала `/spike/*`" in delivery
    assert "9120" not in delivery
    assert "после смены топологии" not in delivery


def github_slug(heading: str) -> str:
    text = heading.lstrip("#").strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def test_stage_links_resolve_and_stage_b_condition_matches_0017() -> None:
    authority = AUTHORITY.read_text(encoding="utf-8")
    headings = [line for line in authority.splitlines() if line.startswith("#")]
    for link in (STAGE_A_LINK, STAGE_B_LINK):
        anchor = link.split("#", 1)[1].rstrip(")")
        assert anchor in [github_slug(h) for h in headings]
    assert "## Этап A (2026-11-20)" in headings
    assert "## Этап B (2026-11-21)" in headings
    stage_b = " ".join(authority.split("### Остаток: этап B", 1)[1].split("\n## ", 1)[0].split())
    assert "остановить и отключить `hermes-autonomous.service`" in stage_b
    assert "Windows-клиент работает только через gateway" in stage_b
    assert "«Остаток: этап B»" in step("### 5. Критерий 4: доставка заметки ровно один раз")


GATEWAY_USE = re.compile(
    r"systemctl --user (?:restart|start|show)[^\n`]*?(hermes-[\w-]+\.service)"
    r"|`ControlGroup`[^`\n]*?вне `(hermes-[\w-]+\.service)`"
)


def gateway_units(text: str) -> set[str]:
    current = "\n\n".join(p for p in text.split("\n\n") if not p.lstrip().startswith("История."))
    return {a or b for a, b in GATEWAY_USE.findall(current)}


def test_current_steps_restart_and_check_hermes_gateway_service_only() -> None:
    document = DOC.read_text(encoding="utf-8")
    assert gateway_units(document) == {"hermes-gateway.service"}
    for heading in (
        "### 1. Проверка перед установкой (Linux, backend)",
        "### 2. Установка (Linux, backend)",
        "### 6. Критерий 5: unit вне cgroup gateway переживает перезапуск",
        "## Удаление",
    ):
        assert "hermes-gateway.service" in step(heading)
    assert "`ControlGroup` вне `hermes-gateway.service`" in document
    history = next(p for p in document.split("\n\n") if p.startswith("История. До этапа B (по"))
    assert "называли gateway unit `hermes-autonomous.service`" in " ".join(history.split())


def test_gateway_units_catches_a_restart_of_hermes_autonomous_service() -> None:
    stale = "```bash\nsystemctl --user restart hermes-autonomous.service\n```"
    assert gateway_units(stale) == {"hermes-autonomous.service"}
    check = "`ControlGroup` должен лежать вне `hermes-autonomous.service`"
    assert gateway_units(check) == {"hermes-autonomous.service"}
    dated = "История. До этапа B: systemctl --user restart hermes-autonomous.service"
    assert gateway_units(dated) == set()
