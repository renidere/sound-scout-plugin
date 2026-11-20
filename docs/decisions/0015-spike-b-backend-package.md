# 0015 — Spike B: пакет-заглушка на backend (ctx.rest, доставка заметки, запуск runner)

- **Статус:** инструкция готова; результаты ожидают прогона владельцем.
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
`plugins.enabled`. Веб-дашборд напишет в консоль браузера, что у плагина нет `dist/index.js`, и
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

### 1. Проверка перед установкой (Linux, backend)

```bash
cd <checkout sound-scout-plugin>
git rev-parse --short HEAD
hermes config get plugins.isolation
systemctl --user show -p MainPID,ActiveState hermes-autonomous.service
```

- Коммит checkout записать в таблицу результатов.
- `plugins.isolation` должен быть пустым или `in_process`. Если там `host` — **остановиться**:
  переключение на `in_process` переносит в процесс gateway все пользовательские плагины, это
  отдельное решение владельца.

### 2. Установка (Linux, backend)

```bash
cp -r <checkout sound-scout-plugin>/hermes/plugin/sound-scout ~/.hermes/plugins/sound-scout
hermes plugins enable sound-scout
systemctl --user restart hermes-autonomous.service
```

`hermes plugins enable` может сам попросить работающий gateway перечитать плагины; перезапуск всё
равно нужен, чтобы прогон начался с чистого процесса. Маршруты монтируются при первом запросе к
дашборду (например, команда «health» из шага 4). После него проверить журнал:

```bash
journalctl --user -u hermes-autonomous.service --since "15 min ago" | grep "Mounted plugin API routes: /api/plugins/sound-scout/"
```

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

1. Палитра → `Sound Scout spike: health`. В текущей топологии `ctx.rest` идёт на `:9119`
   (`hermes serve`), и ожидается уведомление `ok=true pid=<N> session_owner=false`; см. 0017.
   `session_owner=true` ожидается только после смены топологии, которую выбирает владелец (0017).
2. На Linux: `systemctl --user show -p MainPID hermes-autonomous.service` — `MainPID` должен
   совпасть с `pid`. `session_owner=false` значит, что этот процесс не владеет чатами Desktop
   (#94, пункт 5; владелец — `hermes-gateway.service`, см. 0017).
3. Ошибку `ctx.rest` записать дословно.

### 5. Критерий 4: доставка заметки ровно один раз

В текущей топологии доставка через `ctx.rest` идёт в `hermes serve` и возвращает `local chat <id>:
no gateway in this process owns its session` (`check last delivery` → `state=failed`); см. 0017.
Шаги ниже описывают ожидание после смены топологии.

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

1. Палитра → `Sound Scout spike: start runner unit` → `unit=sound-scout-spike-b.service returncode=0`.
2. На Linux:

   ```bash
   systemctl --user show sound-scout-spike-b.service -p ActiveState,MainPID,ControlGroup
   ```

   `ControlGroup` должен лежать вне `hermes-autonomous.service` (ожидается окончание
   `app.slice/sound-scout-spike-b.service`).
3. Перезапустить gateway и посмотреть снова (или палитра → `Sound Scout spike: runner unit status`):

   ```bash
   systemctl --user restart hermes-autonomous.service
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

Коммит checkout: ожидает прогона владельцем.

| критерий | что наблюдать | результат |
|---|---|---|
| 1. пакет и `GET /health` без Hermes | `tests/test_hermes_plugin.py` | выполнено в репозитории |
| 2. инструкция включения и удаления | этот документ, разделы «Инструкция владельцу» и «Удаление» | выполнено в репозитории |
| 3. `GET /health` через `ctx.rest` с Windows | `ok`, `pid` = `MainPID` gateway, `session_owner` | ожидает прогона владельцем |
| 4а. заметка в открытый чат | ровно одна заметка, `state=delivered` | ожидает прогона владельцем |
| 4б. заметка в закрытый чат | ровно одна заметка после открытия, в другом чате ничего | ожидает прогона владельцем |
| 4в. директива `::sound-scout-report` | карточка или строка текста, для обеих заметок | ожидает прогона владельцем |
| 5а. unit вне cgroup gateway | `ControlGroup` вне `hermes-autonomous.service` | ожидает прогона владельцем |
| 5б. unit переживает перезапуск gateway | тот же `MainPID`, `ActiveState=active` | ожидает прогона владельцем |
| 5в. повторный старт живого unit | отказ systemd | ожидает прогона владельцем |
| 6. запись результатов | эта таблица заполнена | ожидает прогона владельцем |

Если директива не рисуется (заметки cron хранятся как `internal_notification`), действует запасной
путь 0013 § 3.7: простой текст в заметке и карточка через опрос `GET /jobs/<id>`.

## Удаление

Linux:

```bash
systemctl --user stop sound-scout-spike-b.service
hermes plugins remove sound-scout
systemctl --user restart hermes-autonomous.service
hermes plugins list
```

Первая команда нужна, только если unit ещё жив (иначе systemd скажет, что его нет). `hermes
plugins remove` удаляет каталог `~/.hermes/plugins/sound-scout` и сам убирает имя из
`plugins.enabled`. В выводе `hermes plugins list` не должно быть `sound-scout`.

Windows: удалить каталог `desktop-plugins\sound-scout` клиента и выполнить «Reload desktop plugins»
(или перезапустить Desktop). Команд `Sound Scout spike: …` в палитре быть не должно.
