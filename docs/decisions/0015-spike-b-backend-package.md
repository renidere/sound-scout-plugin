# 0015 — Spike B: пакет-заглушка на backend (ctx.rest, доставка заметки, запуск runner)

- **Статус:** результаты записаны (2026-11-21): критерии 1–3 и 5 выполнены, 4а выполнен через
  production-доставку; 4б и 4в не доказаны —
  [#105](https://github.com/renidere/sound-scout-plugin/issues/105) и
  [#108](https://github.com/renidere/sound-scout-plugin/issues/108). Инструкцию зонда после этапа B
  0017 не выполняют.
- **Дата:** 2026-11-20.
- **Решение:** [Decision в #96](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7023401575).
- **Задача:** [#96](https://github.com/renidere/sound-scout-plugin/issues/96); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92).
- **Основание:** `docs/decisions/0013-voice-note-integration.md` § 3.2 (пакет плагина), § 3.6
  (runner вне cgroup gateway), § 3.7 (доставка в чат), § 5 шаг 3 (Spike B); Hermes Agent v0.23.1
  (`3b42157858`, без локального патча).
- **Код:** `hermes/plugin/sound-scout/` (`plugin.yaml`, `__init__.py`, `dashboard/manifest.json`,
  `dashboard/plugin_api.py`, `desktop/plugin.js`); тесты — `tests/test_hermes_plugin.py`.

## Вопрос

Backend-половина плагина должна жить в существующем gateway. До сих пор никто не проверил три
вещи, на которых стоит решение 0013:

1. маршрут `plugin_api.py` отвечает через `ctx.rest` штатного Windows-клиента;
2. `gateway.session_cron.deliver_to_local_chat`, вызванная из рабочего потока, кладёт ровно одну
   заметку в открытый и в закрытый чат, а директива `::sound-scout-report` в заметке рисуется (или
   нет) карточкой;
3. `systemd-run --user` из процесса gateway создаёт unit вне cgroup gateway, и этот unit
   переживает перезапуск gateway.

Пакет — одноразовый зонд: после прогона он удаляется, #101 заменяет `plugin_api.py` настоящими
маршрутами без `/spike/*`.

> **После #101.** На `main` маршрутов `/spike/*` и команд палитры `Sound Scout spike: …` больше
> нет: `plugin_api.py` отдаёт `GET /health`, `POST /jobs`, `GET /jobs/{id}`, а `plugin.js` — команду
> палитры `health` и директиву `sound-scout-report` (рекордер и команды заданий появятся позже).
> Весь текст ниже — раскладка пакета, маршруты и разделы 1–7, включая установку в разделе 2, —
> описывает пакет-зонд; для его прогона пакет берётся из коммита `dcb530e`, в отдельном worktree:
> `git worktree add <путь> dcb530e`, затем шаги ниже из `<путь>/hermes/plugin/sound-scout/`.

## Устройство пакета

Раскладка unified package Hermes (документация «Desktop Plugin SDK», раздел «One package, both
SDKs»):

| файл | зачем |
|---|---|
| `plugin.yaml` | `name: sound-scout`; без него `hermes plugins enable/remove` не находят пакет |
| `__init__.py` | пустой `register(ctx)`: агентской части пока нет |
| `dashboard/manifest.json` | `name: sound-scout`, `api: plugin_api.py`, вкладка скрыта (`tab.hidden`) |
| `dashboard/plugin_api.py` | маршруты под `/api/plugins/sound-scout/`; только stdlib и FastAPI, Python ≥ 3.11 |
| `desktop/plugin.js` | команды палитры, которые зовут `ctx.rest`, и директива `sound-scout-report` |

При `plugins.isolation: in_process` (значение по умолчанию) gateway монтирует `plugin_api.py` в
своём процессе при первом запросе к дашборду, и только если `sound-scout` есть в
`plugins.enabled`. Webapp gateway прогревает дашборд при старте, поэтому маршруты монтируются сразу
(0017, «Живые данные», «После этапа A»). Веб-дашборд напишет в консоль браузера, что у плагина нет `dist/index.js`, и
продолжит работу.

## Маршруты

Все ответы — HTTP 200; отказ — `{"ok": false, "error": "<код>", "detail": "<строка>"}`.

| маршрут | что делает |
|---|---|
| `GET /api/plugins/sound-scout/health` | `ok`, `plugin`, `stage`, `version`, `pid` процесса, `session_owner` (владеет ли процесс сессиями: `true`/`false`, `null` без Hermes) |
| `POST /api/plugins/sound-scout/spike/deliver` | тело `{"chat_id", "delay_s"}`; проверка (`bad_chat_id`, `bad_delay`), ответ с `token` сразу; поток `sound-scout-spike-deliver` ждёт `delay_s` и вызывает `deliver_to_local_chat` **один раз** |
| `GET /api/plugins/sound-scout/spike/deliver/{token}` | `pending`, `delivered` или `failed` с дословной ошибкой; неизвестный токен — `unknown_token`; таблица живёт в памяти процесса |
| `POST /api/plugins/sound-scout/spike/runner` | `systemd-run --user --unit=sound-scout-spike-b --description=Sound Scout spike B probe --collect --quiet -- /usr/bin/sleep 900` вне event loop; ответ `ok`, `unit`, `returncode`, хвост `stderr` |
| `GET /api/plugins/sound-scout/spike/runner` | `systemctl --user show sound-scout-spike-b.service` со свойствами `LoadState`, `ActiveState`, `SubState`, `MainPID`, `ControlGroup`, `ActiveEnterTimestamp` |

Текст заметки собирает сервер, два абзаца:

```text
Sound Scout · spike B: тестовая заметка <token>, доставлена из рабочего потока.

::sound-scout-report{job="spike-<token>"}
```

Отказы subprocess: `systemd_run_missing` (программа не найдена), `timeout` (дольше 15 с),
`run_failed` (другая ошибка ОС). Ненулевой код возврата — `ok: false` с `returncode` и `stderr`.

### Что защищает тестовые маршруты

- Hermes не импортирует `plugin_api.py`, пока владелец не включит `sound-scout`.
- Запросы проходят штатную авторизацию листенера; `ctx.rest` достаёт только свой namespace.
- Свободных параметров нет: текст заметки, имя unit и argv фиксированы; от клиента приходят только
  `chat_id` (непустая строка до 256 символов без управляющих) и `delay_s` (целое 0…120).
- Повторный старт живого unit отклоняет сам systemd: одновременно работает не больше одного
  `sleep`.
- Пакет удаляется после прогона (раздел «Удаление»).

## Инструкция владельцу

Все команды выполняет владелец. Агент ничего из этого не запускает.

**После этапа B (2026-11-21) инструкцию не выполнять.** В `~/.hermes/plugins/sound-scout` gateway
с 2026-11-21 стоит production-копия [#104](https://github.com/renidere/sound-scout-plugin/issues/104).
Шаг 2 записал бы пакет-зонд в тот же каталог, а раздел «Удаление» удалил бы её (`hermes plugins remove
sound-scout`). До этапа B зонд прогнан только на Linux: шаги 1, 2 и 6 (2026-11-20,
[7024288870](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7024288870));
Windows-шаги 3–5 не выполнялись. Доставку через #104 проверяет
[#114](https://github.com/renidere/sound-scout-plugin/issues/114). Новый прогон зонда — только по
отдельному решению владельца. Шаги ниже — запись инструкции зонда.

Gateway с этапа B (2026-11-21) — `hermes-gateway.service` (`hermes gateway run`, Webapp на
`:9119`); `hermes-autonomous.service` остановлен и отключён
([0017, «Этап B (2026-11-21)»](0017-session-authority-owner.md#этап-b-2026-11-21)).

История. До этапа B (по 2026-11-21) шаги 1, 2 и 6, строка 5а таблицы «Результаты» и раздел
«Удаление» называли gateway unit `hermes-autonomous.service`. Строки 5а–5в таблицы «Результаты»
записывают прогон шага 6 на `hermes-autonomous.service`.

### 1. Проверка перед установкой (Linux, backend)

```bash
cd <checkout sound-scout-plugin>
git rev-parse --short HEAD
hermes config get plugins.isolation
systemctl --user show -p MainPID,ActiveState hermes-gateway.service
```

- Коммит checkout записать в таблицу результатов.
- `plugins.isolation` должен быть пустым или `in_process`. Если там `host` — **остановиться**:
  переключение на `in_process` переносит в процесс gateway все пользовательские плагины, это
  отдельное решение владельца.

### 2. Установка (Linux, backend)

```bash
cp -r <checkout sound-scout-plugin>/hermes/plugin/sound-scout ~/.hermes/plugins/sound-scout
hermes plugins enable sound-scout
systemctl --user restart hermes-gateway.service
```

`hermes plugins enable` может сам попросить работающий gateway перечитать плагины; перезапуск всё
равно нужен, чтобы прогон начался с чистого процесса. Gateway с Webapp монтирует маршруты при
старте: строка монтирования появилась после перезапуска gateway после этапа A и на этапе B (0017).
После перезапуска найти строку монтирования в файлах логов Hermes: процесс `hermes gateway run` (`hermes-gateway.service`) пишет её в
`~/.hermes/logs/agent.log`, процесс `hermes serve` (до этапа B — `hermes-autonomous.service`) — в
`~/.hermes/logs/gui.log`. В журнал unit она не попадает: logger `hermes_cli.web_server` относится к компоненту `gui`, и Hermes пишет
его только в файлы логов (`hermes_logging.py:322`, `:402–407`; см. 0017, «Живые данные»).
`agent.log` — общий файл без фильтра компонента: в него пишет каждый процесс Hermes, в том числе
`hermes serve` (`hermes_logging.py:403`). Поэтому монтирование в `hermes serve` проверяется по
`gui.log`. Строка в `agent.log` относится к gateway, только если её время позже перезапуска
`hermes-gateway.service` и в `gui.log` нет строки с тем же временем.

```bash
grep "Mounted plugin API routes: /api/plugins/sound-scout/" ~/.hermes/logs/gui.log | tail -n 1
grep "Mounted plugin API routes: /api/plugins/sound-scout/" ~/.hermes/logs/agent.log | tail -n 1
```

Время строки должно быть позже перезапуска её процесса.

Остальные процессы Hermes с общей конфигурацией подхватят пакет при своём следующем перезапуске.
Их маршрут доставки ответит `local chat <id>: no gateway in this process owns its session` — это
ожидаемо.

### 3. Desktop-половина (Windows)

1. Скопировать `hermes/plugin/sound-scout/desktop/plugin.js` из checkout в каталог
   `desktop-plugins\sound-scout\plugin.js` клиента (по документации SDK — `$HERMES_HOME/desktop-plugins/<id>/`,
   по умолчанию `~/.hermes/desktop-plugins/sound-scout/`; точный путь на Windows фиксирует Spike A,
   [#95](https://github.com/renidere/sound-scout-plugin/issues/95)). При удалённом backend копия
   из `plugins/` на Windows сама не появляется.
2. **Capabilities → Plugins**: убедиться, что «Sound Scout (spike B)» включён. Скопированный
   вручную плагин без маркера установки и без `defaultEnabled: false` включён по умолчанию
   (`pluginActive(id, defaultEnabled ?? true)` в загрузчике Desktop). При необходимости выполнить
   в палитре «Reload desktop plugins».
3. В палитре (⌘K / Ctrl+K) должны появиться пять команд `Sound Scout spike: …`.

### 4. Критерий 3: `GET /health` через `ctx.rest` с Windows

1. Палитра → `Sound Scout spike: health`. Ожидание зависит от процесса, на который `ctx.rest` ведёт
   клиент ([0017, «Этап B (2026-11-21)»](0017-session-authority-owner.md#этап-b-2026-11-21)):
   `:9119` (gateway `hermes-gateway.service`, с этапа B 2026-11-21) — единственный вход Desktop,
   уведомление `ok=true pid=<N> session_owner=true`.

   История. С 2026-11-20 топология выбрана, этап A выполнен
   ([0017, «Этап A (2026-11-20)»](0017-session-authority-owner.md#этап-a-2026-11-20)). До этапа B
   `:9119` слушал `hermes serve` (уведомление `session_owner=false`), а gateway — временный порт
   `:9120`; 2026-11-21 Webapp gateway перенесён на `:9119`. С 2026-11-21 в gateway стоит
   production-копия [#104](https://github.com/renidere/sound-scout-plugin/issues/104) без
   `/spike/*`. Команда `Sound Scout spike: health` есть только в пакете-зонде, а инструкцию зонда
   после этапа B не выполняют (раздел «Инструкция владельцу»).
2. На Linux: `systemctl --user show -p MainPID hermes-gateway.service` — `MainPID` должен совпасть с
   `pid`. `hermes-autonomous.service` с этапа B остановлен и отключён.
   `session_owner=false` значит, что этот процесс не владеет чатами Desktop
   (#94, пункт 5; владелец — `hermes-gateway.service`, см. 0017).
3. Ошибку `ctx.rest` записать дословно.

### 5. Критерий 4: доставка заметки ровно один раз

История. До этапа B (по 2026-11-21) доставка через `ctx.rest` на `:9119` шла в `hermes serve` и
возвращала `local chat <id>: no gateway in this process owns its session` (`check last delivery` →
`state=failed`); см. 0017.

С 2026-11-21 (0017, «Этап B (2026-11-21)») `:9119` слушает gateway, и `ctx.rest` клиента доходит до
него. Условие этапа B 0017 («Остаток: этап B») выполнено: остановить и отключить
`hermes-autonomous.service`, и Windows-клиент работает только через gateway. Шаги 5.1–5.4 вызывают
маршруты Spike B `/spike/*`, поэтому их нужно было выполнить до этапа B. Они не выполнены; таблица
«Результаты» записывает 4а по production-доставке. На этапе B production-копия [#104](https://github.com/renidere/sound-scout-plugin/issues/104)
заменила Spike B в gateway и убрала `/spike/*`. Доставку через #104 проверяет
[#114](https://github.com/renidere/sound-scout-plugin/issues/114): 2026-11-21 заметка задания дошла
в чат владельца ровно один раз.
Шаги ниже описывают ожидание, когда клиент работает через gateway.

1. **Открытый чат X.** Открыть чат X, палитра → `Sound Scout spike: deliver note to this chat in 30 s`,
   не уходить из чата. Через ~30 с в X должна появиться ровно одна заметка с токеном из
   уведомления. Палитра → `Sound Scout spike: check last delivery` → `state=delivered`.
2. **Закрытый чат Y.** Открыть чат Y, та же команда, затем в течение 30 с закрыть чат Y
   (вкладку или окно). Подождать минуту, открыть Y: ровно одна заметка с новым токеном; в X ничего
   нового. `Sound Scout spike: check last delivery` → `state=delivered`.
3. Для обеих заметок записать, нарисовалась ли директива карточкой «Sound Scout: карточка отчёта
   spike-…» или осталась строкой `::sound-scout-report{job="spike-…"}`.
4. Если `check last delivery` показывает `failed`, записать ошибку дословно. Повторять вслепую
   нельзя: повтор — это вторая заметка. Таблица токенов живёт в памяти процесса и после
   перезапуска gateway пуста (`unknown_token`).

### 6. Критерий 5: unit вне cgroup gateway переживает перезапуск

Выполнено 2026-11-20 на `hermes-autonomous.service`, тогда процесс с пакетом; результаты — строки
5а–5в.

1. Палитра → `Sound Scout spike: start runner unit` → `unit=sound-scout-spike-b.service returncode=0`.
2. На Linux:

   ```bash
   systemctl --user show sound-scout-spike-b.service -p ActiveState,MainPID,ControlGroup
   ```

   `ControlGroup` должен лежать вне `hermes-gateway.service` (ожидается окончание
   `app.slice/sound-scout-spike-b.service`).
3. Перезапустить gateway и посмотреть снова (или палитра → `Sound Scout spike: runner unit status`):

   ```bash
   systemctl --user restart hermes-gateway.service
   systemctl --user show sound-scout-spike-b.service -p ActiveState,MainPID,ControlGroup
   ```

   Ожидается тот же `MainPID` и `ActiveState=active`.
4. Палитра → `Sound Scout spike: start runner unit` ещё раз, пока unit жив: ожидается отказ systemd
   (`returncode` не 0, в `stderr` — что unit уже существует).
5. Остановить unit:

   ```bash
   systemctl --user stop sound-scout-spike-b.service
   ```

### 7. Передача результатов

Таблицу ниже (пункты 4–6, ошибки дословно, без содержимого чатов и личных путей) владелец
отправляет комментарием в [#96](https://github.com/renidere/sound-scout-plugin/issues/96),
например:

```bash
gh issue comment 96 --repo renidere/sound-scout-plugin --body-file <файл с результатами>
```

После этого агент переносит результаты в таблицу этого документа и меняет его статус (критерий 6).

## Результаты

Коммит checkout: `3287539` (Linux-часть, 2026-11-20), Hermes Agent v0.23.1 `3b4215785882`.

| критерий | что наблюдать | результат | доказательство |
|---|---|---|---|
| 1. пакет и `GET /health` без Hermes | `tests/test_hermes_plugin.py` | выполнено в репозитории | [#96 7024144812](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7024144812) |
| 2. инструкция включения и удаления | этот документ, разделы «Инструкция владельцу» и «Удаление» | выполнено в репозитории | [#96 7024144812](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7024144812); установка production-пакета при `in_process` по умолчанию — [#104 7060137735](https://github.com/renidere/sound-scout-plugin/issues/104#issuecomment-7060137735), [#104 7062974709](https://github.com/renidere/sound-scout-plugin/issues/104#issuecomment-7062974709) |
| 3. `GET /health` через `ctx.rest` с Windows | `ok`, `pid` = `MainPID` gateway, `session_owner` | выполнено: production health 0.1.0, `session_owner=true`, `pid` = `MainPID` | палитра Desktop 0.23.3 → `Sound Scout: health`, `pid` = `MainPID` `hermes-gateway.service`: [#120 7062836106](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7062836106), [#120 7062968106](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7062968106) |
| 4а. заметка в открытый чат | ровно одна заметка, `state=delivered` | выполнено по смыслу: production-доставка, ровно одно сообщение с меткой; состояние чата не записано | поток `sound-scout-delivery` в процессе gateway — [#104 7060137735](https://github.com/renidere/sound-scout-plugin/issues/104#issuecomment-7060137735); метка задания ровно в одном сообщении Desktop-чата владельца — [#114 7060137985](https://github.com/renidere/sound-scout-plugin/issues/114#issuecomment-7060137985); вторая сессия, Hermes v0.23.3 — [#114 7063057772](https://github.com/renidere/sound-scout-plugin/issues/114#issuecomment-7063057772) |
| 4б. заметка в закрытый чат | ровно одна заметка после открытия, в другом чате ничего | не доказано: проверяет #105 | Desktop-чат не закрывали; проверяет [#105](https://github.com/renidere/sound-scout-plugin/issues/105) |
| 4в. директива `::sound-scout-report` | карточка или строка текста, для обеих заметок | не доказано: заметка без директивы, запасной путь 0013 § 3.7; карточка — #108 | Decision в #102 [#102 7026672015](https://github.com/renidere/sound-scout-plugin/issues/102#issuecomment-7026672015); карточка — [#108](https://github.com/renidere/sound-scout-plugin/issues/108) |
| 5а. unit вне cgroup gateway | `ControlGroup` вне `hermes-gateway.service` | выполнено (зонд, Linux): `ControlGroup=…/app.slice/sound-scout-spike-b.service` | [#96 7024288870](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7024288870) |
| 5б. unit переживает перезапуск gateway | тот же `MainPID`, `ActiveState=active` | выполнено (зонд, Linux): `MainPID` 725759 до и после `restart hermes-autonomous.service`, `ActiveState=active`; production: runner `sound-scout-jobs-runner.service` пережил два перезапуска `hermes-gateway.service` | [#96 7024288870](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7024288870); production — [#114 7063057772](https://github.com/renidere/sound-scout-plugin/issues/114#issuecomment-7063057772) |
| 5в. повторный старт живого unit | отказ systemd | выполнено (зонд, Linux): `returncode=1`, `Unit sound-scout-spike-b.service was already loaded`; unit остановлен | [#96 7024288870](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7024288870) |
| 6. запись результатов | эта таблица заполнена | выполнено: эта таблица | [#96](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7065924531) |

Что заменило зонд. Маршруты `/spike/*` убраны: production-копия 0.1.0 заменила зонд в gateway
([#128 7060138271](https://github.com/renidere/sound-scout-plugin/issues/128#issuecomment-7060138271)). Критерии 3, 4а и 5 подтверждены production-путём #104, #114 и #120.
Повторный прогон зонда запрещён: он вернул бы тестовые маршруты в живой gateway.

Если директива не рисуется (заметки cron хранятся как `internal_notification`), действует запасной
путь 0013 § 3.7: простой текст в заметке и карточка через опрос `GET /jobs/<id>`.

## Удаление

Linux:

```bash
systemctl --user stop sound-scout-spike-b.service
hermes plugins remove sound-scout
systemctl --user restart hermes-gateway.service
hermes plugins list
```

Первая команда нужна, только если unit ещё жив (иначе systemd скажет, что его нет). `hermes
plugins remove` удаляет каталог `~/.hermes/plugins/sound-scout` и сам убирает имя из
`plugins.enabled`. В выводе `hermes plugins list` не должно быть `sound-scout`.

Windows: удалить каталог `desktop-plugins\sound-scout` клиента и выполнить «Reload desktop plugins»
(или перезапустить Desktop). Команд `Sound Scout spike: …` в палитре быть не должно.

Факт. Linux-копию зонда заменил пакет 0.1.0 2026-11-21
([#104](https://github.com/renidere/sound-scout-plugin/issues/104),
[#128](https://github.com/renidere/sound-scout-plugin/issues/128)). Unit
`sound-scout-spike-b.service` остановлен 2026-11-20. Desktop-половину зонда на Windows не ставили:
в клиенте стоит production `plugin.js`
([#120 7062836106](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7062836106)).
