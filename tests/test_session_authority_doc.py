"""Decision 0017 records stage A of #120 and its remainder without personal data."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs/decisions/0017-session-authority-owner.md"
ISSUE = "https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-"


def section(heading: str) -> str:
    document = DOC.read_text(encoding="utf-8")
    level = heading.split(" ", 1)[0]
    start = document.index(f"\n{heading}\n")
    rest = document[start + len(heading) + 2 :]
    ends = [m.start() for m in re.finditer(rf"^#{{1,{len(level)}}} ", rest, re.MULTILINE)]
    return rest[: ends[0]] if ends else rest


def test_stage_a_records_topology_restart_health_and_rollback() -> None:
    choice = section("### Выбор владельца")
    for comment in ("7039818435", "7051997437"):
        assert f"{ISSUE}{comment}" in choice
    assert "Владелец выбрал (б)" in choice
    assert "этап A → проверка → Windows-клиент → этап B" in choice

    reported = section("### Сообщалось выполненным")
    assert f"{ISSUE}7040152960" in reported
    assert "2026-11-20 ~16:00 MSK" in reported
    assert "12:58:49" not in reported and "12:58:53" not in reported
    assert "`MainPID` 1234703 → 2297970" in reported
    assert "с LAN без входа — 401" in reported
    assert "`pid=2297970`" in reported and "`session_owner: true`" in reported

    rollback = section("### Откат этапа A")
    for step in (
        "stage-a-20261120T125829Z",
        "Удалить drop-in",
        "`systemctl --user daemon-reload`",
        "Перезапустить `hermes-gateway.service`",
        "отдельное разрешение владельца",
    ):
        assert step in rollback


def test_stage_a_separates_reported_checked_and_unchecked() -> None:
    assert "Это отчёт исполнителя, а не сверка текущего состояния" in section(
        "### Сообщалось выполненным"
    )
    checked = section("### Проверено сейчас")
    assert "Только чтение, 2026-11-20 22:30 UTC" in checked
    assert "2026-11-20 12:58:49 UTC\n  (`ActiveEnterTimestamp`)" in checked
    assert "смонтировал `/api/plugins/sound-scout/` при старте" in checked
    unchecked = " ".join(section("### Ещё не проверено").split())
    assert "`session_owner` сейчас" in unchecked
    assert (
        "Один health с `session_owner: true` не доказывает сквозную работу Windows-клиента, "
        "доставку заметки и перенос старых чатов" in unchecked
    )


def test_remainder_names_client_step_agent_check_and_stage_b() -> None:
    client = section("### Остаток: клиентская проверка")
    assert "Шаг владельца" in client
    assert "Затем агент сверяет, только чтение" in client
    assert "`session_admissions`" in client

    stage_b = " ".join(section("### Остаток: этап B").split())
    assert "Каждое действие и окно владелец одобряет отдельно" in stage_b
    for item in (
        "`hermes-autonomous.service`",
        "Убрать SSH-подключение Desktop: его backend — третий писатель",
        "`hermes sessions adopt`",
        "Риски:",
        "Откат этапа B:",
    ):
        assert item in stage_b
    assert "Удалить только drop-in, созданные на этапе B" in stage_b
    assert "`30-lan-auth.conf` этапа A остаётся" in stage_b


def test_old_readbacks_are_dated_and_doc_holds_no_private_data() -> None:
    document = DOC.read_text(encoding="utf-8")
    processes = section("## Процессы Hermes на общем `~/.hermes`")
    assert "`127.0.0.1:43189` (частный listener" in processes
    assert "Таблица — состояние на 2026-11-19. После этапа A (2026-11-20)" in processes
    assert "**История.** Текст ниже — варианты на 2026-11-20 до выбора" in document
    assert "Список на 2026-11-20 до этапа A" in section("## Что не установлено")

    assert not re.search(r"/(home|Users)/|[A-Za-z]:\\", document)
    assert not re.search(r"\b(?:192\.168|10)\.\d{1,3}\.\d{1,3}\b", document)
    assert not re.search(r"(?i)password\s*[:=]|bearer\s+[a-z0-9._-]{16,}", document)


def test_client_check_2026_11_21_separates_report_check_and_open_items() -> None:
    client = " ".join(section("### Клиентская проверка (2026-11-21)").split())
    assert f"{ISSUE}7059648596" in client
    assert "Это сообщение владельца, а не сверка" in client
    assert "Только чтение, 2026-11-21 08:04–08:15 UTC" in client
    for fact in (
        "на `:9120` — 0 соединений",
        "Строк в `session_admissions` — 0",
        "`20261121_110129_a8cd31`",
        "`version: 0.0.1`, 5 файлов = `dcb530e`",
    ):
        assert fact in client
    assert "Вывод: сообщение владельца проверкой не подтверждено" in client
    assert "Это вывод по адресам потоков, а не запись процесса о себе" in client
    assert "Вывод: за 08:01:30–08:01:48 второго писателя у этого чата не видно" in client
    assert "Исключительное владение чатом одним процессом не проверено" in client
    assert "к процессу и listener'у не отнесены" in client
    open_items = client.split("**Ещё не проверено:**", 1)[1]
    for item in ("Доставка заметки", "Перенос старых чатов", "Версия Windows Desktop"):
        assert item in open_items


def test_step_b1_names_services_order_risks_and_rollback() -> None:
    step = " ".join(section("### Шаг B1 (предложение 2026-11-21)").split())
    assert "`systemctl --user stop hermes-autonomous.service`" in step
    assert "без `disable`" in step
    assert "Агент процесс не останавливает" in step
    assert "`stop` unit'а его не завершает" in step
    assert "Для B1 достаточно условия 4" in step
    assert "`hermes-gateway.service`: не трогается" in step
    for part in (
        "Порядок:",
        "Риски:",
        "Откат B1: `systemctl --user start hermes-autonomous.service`",
    ):
        assert part in step
    assert "Откат входит в то же разрешение и окно" in step
    assert "`hermes sessions adopt` и #104 в B1 не входят" in step
