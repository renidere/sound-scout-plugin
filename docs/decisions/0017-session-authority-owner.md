# 0017 — Какой процесс Hermes владеет session authority Desktop

- **Статус:** факт установлен. Топология выбрана: вариант (б), уточнённый в план по этапам
  (2026-11-20). Этап A выполнен 2026-11-20. Этап B выполнен 2026-11-21: gateway слушает `:9119`,
  `hermes-autonomous.service` остановлен и отключён, три старых чата, которые назвал владелец,
  перенесены в authority gateway. Подробности — разделы «Этап A (2026-11-20)» и «Этап B (2026-11-21)».
- **Дата:** 2026-11-20; этап A и сверка — 2026-11-20; этап B — 2026-11-21.
- **Решение:** [Decision в #116](https://github.com/renidere/sound-scout-plugin/issues/116#issuecomment-7025500309);
  остаток после этапа A — [Decision в #120](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7052302513).
- **Задача:** [#116](https://github.com/renidere/sound-scout-plugin/issues/116); выбор топологии и
  перезапуск — [#120](https://github.com/renidere/sound-scout-plugin/issues/120); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92).
- **Основание:** `docs/decisions/0013-voice-note-integration.md` § 3.7 (доставка в чат) и § 9 п. 5
  (процесс-владелец чатов Desktop); `docs/decisions/0015-spike-b-backend-package.md` (маршрут
  `GET /health`, поле `session_owner`); решение владельца в
  [#94](https://github.com/renidere/sound-scout-plugin/issues/94), п. 5 (владельца устанавливает
  агент); [комментарий 7024288870 к #96](https://github.com/renidere/sound-scout-plugin/issues/96#issuecomment-7024288870)
  (`session_owner=false` в `hermes-autonomous.service`).
- **Hermes:** Hermes Agent v0.23.1, ревизия `3b42157858` (`~/.hermes/hermes-agent`). В `gateway/`,
  `hermes_cli/`, `tui_gateway/` локальных правок нет; незакоммиченные изменения есть только в
  `apps/desktop`, `website/docs`, `tests/tools`, `tests/tui_gateway`. Ссылки `файл:строка` ниже даны
  на эту ревизию.
- **Как получено:** только чтение. Первое наблюдение — 2026-11-19 23:42–23:46 UTC, повторная сверка —
  2026-11-19 23:49–23:50 UTC. Ничего не перезапускалось и не менялось. Этап A (изменение и
  перезапуск gateway 2026-11-20) выполнил оркестратор по разрешению владельца в #120; сверка после
  него — только чтение, 2026-11-20 22:30 UTC. Этап B (2026-11-21) тоже выполнил оркестратор по
  разрешению владельца в #120; этот документ записывает его отчёты.

## Вопрос

Spike B (0015) показал `session_owner=false` в процессе `hermes-autonomous.service`, а
`deliver_to_local_chat` работает только в процессе — владельце session authority. Задача #116
отвечает на три вопроса:

1. какой процесс и какой профиль владеют session authority Desktop на станции (PID, путь вызова
   `bind_owner`, путь `state.db`);
2. когда authority привязывается: при старте, при первом подключении клиента или при первом
   открытом чате;
3. может ли процесс `hermes-autonomous.service` владеть сессиями, при каком условии, и если не может —
   какой процесс может и что это значит для #102.

## Ответ

Наблюдение 2026-11-19, до этапов A и B (топология того времени; этап B её заменил). Session
authority профиля `default` (`~/.hermes`, `~/.hermes/state.db`) принадлежала процессу
`hermes gateway run` из `hermes-gateway.service` (PID 1234703). `hermes-autonomous.service` запускал
`hermes serve --isolated --host 0.0.0.0 --port 9119` (PID 726109): это был backend, к которому
подключался Windows Desktop. Hermes создаёт authority и вызывает `bind_owner` только внутри
`hermes gateway run`: для профиля запуска (`default`) — при старте процесса, до подключения клиентов
и до открытия чатов; для дополнительного профиля — при hot-serve в уже работающем gateway.
`hermes serve` authority не создаёт никогда. Поэтому `session_owner=false` в PID 726109 было
постоянным свойством той топологии, а не ленивой привязкой. Desktop-чаты станции тогда шли мимо
session authority (legacy-путь `tui_gateway`).

Посылка 0013 § 9 п. 5 («процесс-владелец чатов Desktop — `hermes-autonomous.service`») неверна.

С 2026-11-20 (этап A) authority держит тот же unit, процесс другой: `hermes-gateway.service`, PID
2297970. С этапа A до этапа B gateway слушал отдельный порт (Webapp со входом), `hermes serve` на
`:9119` работал, и Windows Desktop подключался к нему.

**Порт (история).** На этапе A (2026-11-20) Webapp gateway слушал временный порт `:9120`, ниже —
«listener этапа A». 2026-11-21 (этап B) Webapp перенесён на `:9119`, listener этапа A закрыт.

С 2026-11-21 (этап B) Webapp gateway слушает `0.0.0.0:9119` — единственный вход Desktop.
`hermes-autonomous.service` остановлен и отключён, Desktop SSH backend завершён. Windows Desktop
входит в gateway на `:9119`; у чатов Desktop один процесс-владелец — `hermes gateway run`. См. раздел
«Этап B (2026-11-21)».

## Процессы Hermes на общем `~/.hermes`

Наблюдение — 2026-11-19 23:42–23:50 UTC.

| | `hermes-autonomous.service` | `hermes-gateway.service` | Desktop SSH backend |
|---|---|---|---|
| PID | 726109 (`MainPID`) | 1234703 (`MainPID`) | 4084248 |
| cmdline | `hermes serve --isolated --host 0.0.0.0 --port 9119` | `hermes gateway run` | `serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file …` |
| старт (UTC) | 2026-11-19 22:41:17 | 2026-11-19 08:34:28 | 2026-11-19 20:13:37 |
| `HERMES_HOME` | `~/.hermes` | `~/.hermes` | `~/.hermes` (по ledger) |
| cgroup | `user@1000.service/app.slice/hermes-autonomous.service` | `user@1000.service/app.slice/hermes-gateway.service` | не наблюдался: процесс завершён |
| listener | `0.0.0.0:9119` (Desktop, дашборд, маршруты плагинов) | `127.0.0.1:43189` (частный listener для native-клиентов по ticket'ам control socket) | `127.0.0.1:42421` (по ledger) |
| `~/.hermes/gateway.lock` | нет | **FLOCK WRITE** | — |
| запись хоста | `~/.local/state/hermes/gateway-locks/host-serve.json`: `role=serve`, `profiles=["default"]`, `0.0.0.0:9119`; FLOCK на `host-serve.lock` | `host-gateway.json`: `role=gateway`, `profiles=["default"]`; FLOCK на `host-gateway.lock` | — |
| `~/.hermes/gateway_state.json` | — | `pid=1234703`, `kind=hermes-gateway`, `served_profiles=["default"]`, `code_sha=3b421578…`, `active_agents=0`, `active_work=null` | — |
| журнал | `~/.hermes/logs/gui.log`: `Mounted plugin API routes: /api/plugins/sound-scout/` в 22:40:13 и 22:41:19 UTC | `~/.hermes/logs/agent.log`: в 08:34:39 UTC смонтированы plugin API `subscription-usage`, `hermes-achievements`, `kanban` (прогрев дашборда после старта gateway; процесс определён по времени, в `agent.log` нет PID); строки для `sound-scout` нет | — |

Таблица — состояние на 2026-11-19. После этапа A (2026-11-20) изменилось: gateway — PID 2297970 с
12:58:49 UTC, listener этапа A (Webapp со входом); TCP `127.0.0.1:43189` больше не слушает,
control socket gateway — `~/.hermes/gateway.sock` (сверка 2026-11-20 22:30 UTC). Desktop SSH
backend теперь другой: PID 434482, `127.0.0.1:37575`, с 2026-11-20 06:56 UTC. Подробности — раздел
«Этап A (2026-11-20)». После этапа B (2026-11-21): gateway слушает `0.0.0.0:9119`,
`hermes-autonomous.service` и Desktop SSH backend не работают — раздел «Этап B (2026-11-21)».

Третий процесс — `serve --isolated`, который Desktop запускает через SSH. Он есть в
`~/.hermes/spawn-ledger.json` (`port=42421`, `spawner_pid=None`, то есть отсоединён от SSH) и в
таблице `gateway_heartbeats` (последний heartbeat 2026-11-19 23:02:42 UTC). С 23:42 UTC его нет в
`ps`, поэтому его cgroup не наблюдался. Session authority он тоже не создаёт: это `serve`.

Повторить наблюдение (только чтение):

- `systemctl --user show -p MainPID,ExecStart,ControlGroup,ActiveEnterTimestamp hermes-autonomous.service hermes-gateway.service`;
- `tr '\0' ' ' < /proc/<pid>/cmdline`, `cat /proc/<pid>/cgroup`, `HERMES_HOME` из `/proc/<pid>/environ`;
- `lslocks -o PID,TYPE,MODE,PATH | grep -E 'gateway.lock|gateway-locks'`;
- `ss -ltnp | grep 9119`, соединения — `ss -tnp state established | grep ':9119'`
  (до этапа A gateway слушал `:43189`);
- `~/.hermes/gateway_state.json` (`pid`, `kind`, `served_profiles`, `code_sha`, `active_agents`,
  `active_work`), `~/.local/state/hermes/gateway-locks/host-*.json` (`role`, `pid`, `profiles`,
  `host`, `port`).

## Где привязывается authority

Факт по исходникам ревизии `3b42157858`.

1. `gateway/run_bootstrap.py:731` `start_gateway` → `:805` `_start_gateway_reserve_profiles` (flock
   `~/.hermes/gateway.lock`: один gateway на home) → `:814` pid file → `:834`
   `await initialize_gateway_runtime(runner)`. Так стартует только `hermes gateway run`.
2. `gateway/run_runtime.py:177` `initialize_gateway_runtime` требует, чтобы процесс владел home
   (`:184` `process_ownership.owns(home)`) → `:38` `_build_profile_authority` → `:54`
   `initialize_session_authority(...)` с `state.db` этого home (`:50` проверяет, что родитель
   `state.db` совпадает с home).
3. `gateway/session_authority.py:1444` `initialize_session_authority`: `begin_runtime_epoch` (строка
   таблицы `runtime_epoch`), затем `:1475–1476` `bind_owner(authority)`.
4. `gateway/session_cron.py:22–24` `bind_owner` пишет в память процесса:
   `_owners[Path(db_path).resolve().parent] = (os.getpid(), authority, loop)`. Ни файла, ни строки
   в БД нет: привязка существует только внутри одного процесса. `:34–37` `owner_for_home(home)`
   возвращает owner, только если ключ совпадает с home, `pid` равен текущему и loop не закрыт.
   `:63–70` `deliver_to_local_chat` без owner возвращает
   `local chat <id>: no gateway in this process owns its session`.
5. Отвязка: `gateway/run_runtime.py:393` `_retire_profile_authority` → `:431` `unbind_owner`
   (остановка gateway или вывод профиля из обслуживания).
6. Hot-serve: дополнительный профиль, который работающий gateway начинает обслуживать, получает
   authority через `gateway/run_profile_reconcile.py:432` `serve_profile_runtime` →
   `gateway/run_runtime.py:471` `_build_profile_authority` (`register=False`) → те же шаги 2–4.
   Отвязка — `unserve_profile_runtime` (`run_runtime.py:529`). Это тоже происходит только в
   `hermes gateway run`: `GatewayProfileReconcileMixin` — часть `GatewayRunner` (`gateway/run.py`).
   Других мест, где создаётся authority, вне тестов и evals нет: `_build_profile_authority`
   вызывается только из `run_runtime.py:198` (старт, через `run_bootstrap.py:834`) и `:471`
   (hot-serve). `app.state.session_authority` присваивается только в `gateway/run_api.py:97` (и
   обнуляется в `:116`), то есть в дашборде, который хостит сам gateway. `hermes serve` привязку не
   вызывает ни на одном из этих путей.
7. Плагин: `hermes/plugin/sound-scout/dashboard/plugin_api.py:52–68` берёт `get_hermes_home()`
   (`~/.hermes`) и возвращает `owner_for_home(home) is not None` в том процессе, где смонтированы
   маршруты.

**Чем определяется владелец:** flock `~/.hermes/gateway.lock` (через `process_ownership`) и
`state.db` этого home. Ключ `_owners` — `Path(state.db).resolve().parent`, то есть `~/.hermes`.
Профиль — `default`, БД — `~/.hermes/state.db`. В таблице `runtime_epoch` одна строка:
`epoch=12`, `instance_id=5a90d77a…`.

### Почему у `hermes serve` authority нет

`hermes serve` (`hermes_cli/web_server.py`) поле `app.state.session_authority` не заполняет: в
этом файле его нет. Поэтому `tui_gateway/ws.py:377–385` `_authority_connection` возвращает `None`,
и каждый запрос Desktop уходит в legacy `server.dispatch` (`tui_gateway/ws.py:400`), где ход
выполняется внутри процесса serve.

Документация Hermes говорит то же самое:

- `website/docs/developer-guide/session-storage.md:7–9`: «The profile's gateway is the **only writer
  of canonical sessions**: `hermes chat`, the TUI, Desktop (local), the API server, ACP, cron, Bot
  Chat and Kanban workers are clients that admit input to its session authority over `/api/ws`».
- `website/docs/getting-started/nix-setup.md:680`: «A Hermes Desktop **remote** connection and the web
  dashboard connect to a *different* process, which is `hermes serve` or `hermes dashboard`. (A
  Desktop running on the same host attaches to the gateway itself via `hermes gateway ensure`.)»

## Когда

При старте процесса `hermes gateway run`: docstring `gateway/session_authority.py:1445` — «Call
after exclusive profile ownership, before connecting adapters/API». Дополнительный профиль
привязывается позже, при hot-serve, но тоже внутри gateway (п. 6 выше). Подключение клиента и
открытие чата на привязку не влияют. Гипотеза из комментария 7024288870 к #96 («привязывается лениво, при
открытии чата Desktop») опровергнута кодом: в `hermes serve` этого кода нет вовсе.

## Живые данные

**Наблюдалось** (только чтение):

- 2026-11-19 23:49:55 UTC, `GET http://127.0.0.1:9119/api/plugins/sound-scout/health` локально с
  `Authorization: Bearer` штатного session token дашборда: `ok=true`, `plugin=sound-scout`,
  `stage=spike-b`, `version=0.0.1`, **`pid=726109`**, **`session_owner=false`**.
  `systemctl --user show -p MainPID hermes-autonomous.service` → `726109`. Первое чтение в 23:44:30
  UTC дало те же значения.
- 23:45 и 23:49 UTC: соединений на `:9119` и `:43189` нет. `~/.hermes/state/dashboard_clients.heartbeat`
  последний раз обновлён в 23:02:54 UTC.
- `~/.hermes/state.db` (`mode=ro`): за двое суток 8 сессий с `source=desktop`, **ни у одной** нет
  строк в `session_admissions`; 4 сессии `oneshot` — все с admissions. Desktop-чаты станции
  выполнял legacy-путь serve, а не session authority.
- Маршруты `sound-scout` смонтированы в PID 726109 (`gui.log`). В PID 1234703 они не смонтированы:
  дашборд gateway прогрелся при старте (`agent.log`, 08:34:39 UTC; `gateway/run_bootstrap.py:929–934`
  → `gateway/run_api.py:301` `warm_gateway_dashboard`), а пакет `sound-scout` установлен позже, в
  22:40 UTC. Строку `Mounted plugin API routes` в `gateway.log` искать бесполезно: logger
  `hermes_cli.web_server` относится к компоненту `gui` (`hermes_logging.py:322`, `:405–406`), и
  процесс gateway пишет её только в `agent.log`.

**Выведено из кода, не наблюдалось:** `owner_for_home(~/.hermes)` внутри PID 1234703 возвращает
owner, то есть `session_owner=true`. Основание — цепочка раздела «Где привязывается authority»,
flock `gateway.lock` у PID 1234703, `host-gateway.json`, `gateway_state.json` и `~/.hermes/state.db`
(`runtime_epoch`). **Пробел:** `owner_for_home` внутри PID 1234703 вживую не читался. Маршрут health
в этом процессе сейчас не смонтирован: plugin API в gateway монтируется по требованию, при первом
запросе к `/api/plugins/sound-scout/` на `:43189` (`hermes_cli/web_server_dashboard.py:1088`
`_try_hot_mount_plugin_api`; сам дашборд gateway без прогрева тоже монтируется на первом запросе,
`gateway/run_api.py:1–13`). Такой запрос — изменение живого процесса, поэтому в этой задаче он не
выполнялся. Живое чтение возможно аутентифицированным `GET /api/plugins/sound-scout/health` на
`127.0.0.1:43189` по native ticket'у control socket, если владелец это разрешит.

**После этапа A (2026-11-20):** gateway перезапущен (PID 2297970) и монтирует маршруты
`sound-scout` при старте (`agent.log`, 12:58:53 UTC): Webapp прогревается сразу. Оркестратор
сообщил живое чтение health с входом на listener этапа A: `pid=2297970`, `session_owner: true`. Пробел для
PID 1234703 не закрыт, а потерял смысл: этого процесса больше нет, и прочитать в нём `owner_for_home`
уже нельзя. Новый gateway, по отчёту, owner видит. Повторно health не читался. См. раздел «Этап A (2026-11-20)».

## Выводы по критериям #116

1. **Процесс и профиль.** Факт: владелец — `hermes gateway run`, `hermes-gateway.service`, PID 1234703,
   профиль `default`, путь `start_gateway → initialize_gateway_runtime → _build_profile_authority →
   initialize_session_authority → bind_owner`, БД `~/.hermes/state.db`. Не наблюдалось:
   `owner_for_home` внутри PID 1234703.
2. **Когда.** Факт по коду: при старте gateway, до подключения клиентов. Живые данные:
   `session_owner=false` в PID 726109 в 23:49:55 UTC, больше чем через 15 часов после привязки в
   PID 1234703 (старт в 08:34:28 UTC); это согласуется с кодом. Повтор после открытия чата Desktop
   ничего не изменит.
3. **Может ли `hermes-autonomous.service` владеть сессиями.** В текущем виде (`serve --isolated`) —
   **нет, ни при каком состоянии клиентов**. Условие: процесс этого unit'а сам должен быть
   `hermes gateway run` для `~/.hermes`, то есть держать `gateway.lock`. Тогда `hermes-gateway.service`
   нужно остановить: второй gateway на тот же home не стартует. Сейчас сессиями владеет PID 1234703
   (`hermes-gateway.service`).
4. **Перезапуски.** В этой задаче ничего не перезапускалось и не менялось. Порядок для
   перезапуска, который владелец разрешит отдельно, и откат — в разделе «Перезапуск и откат».

## Варианты топологии для владельца

**История.** Текст ниже — варианты на 2026-11-20 до выбора. Владелец выбрал (б). Исследование
исходников уточнило его в план по этапам: gateway остаётся в `hermes-gateway.service` и получает
Webapp-listener, как в (а); `hermes-autonomous.service` отключается позже, на этапе B. Действующий
план — раздел «Этап A (2026-11-20)».

Ни один не выбран. Оба требуют перезапуска gateway и меняют либо адрес подключения
Windows-клиента, либо unit'ы. Решает владелец.

**(а) Desktop → Webapp listener `hermes-gateway.service`.** В `~/.hermes/config.yaml` включается
`webapp.enabled: true`, `webapp.host: 0.0.0.0`, `webapp.port` (порт Webapp); gateway читает
это только при старте (`website/docs/user-guide/configuration.md:3304–3321`). Для `0.0.0.0`
документация требует dashboard sign-in provider; native-клиенты, по ней же, sign-in не проходят.
Windows-клиент переключается на новый адрес. `hermes-autonomous.service` остаётся для дашборда или
отключается — отдельный вопрос. **Гипотеза, не проверено:** что штатный Windows-клиент работает как
удалённый клиент этого listener'а и попадает в authority.

**(б) `hermes-autonomous.service` → `hermes gateway run`.** Unit этого сервиса запускает
`hermes gateway run` с listener'ом на `0.0.0.0`, `hermes-gateway.service` останавливается и
отключается (один gateway на home). Конфигурация у обоих unit'ов общая (`~/.hermes/config.yaml`),
меняются unit-файлы и секция `webapp` (listener на `0.0.0.0`; при `webapp.port: 9119` адрес
Windows-клиента остаётся прежним после остановки serve). **Гипотеза, не проверено:** то же, что в (а).

**Ожидаемый `health` после любого варианта:** `GET …/api/plugins/sound-scout/health` на listener'е
процесса-владельца даёт `pid` = `MainPID` этого gateway и `session_owner=true`; для (а) маршруты
плагина должны быть смонтированы в gateway (после первого запроса — строка `Mounted plugin API
routes: /api/plugins/sound-scout/` в `agent.log`, не в `gateway.log`).

## Перезапуск и откат

Порядок записан до этапа A. С этапа A проверка нагрузки смотрит на listener Webapp вместо `:43189`, и в
неё входит Desktop SSH backend (п. 4 проверки в разделе «Этап A (2026-11-20)»). Drop-in у
`hermes-gateway.service` теперь есть: его тоже копируют.

Перезапуск gateway с активными чатами без проверки нагрузки запрещён. Порядок:

1. **Проверка нагрузки.** В `~/.hermes/gateway_state.json` `active_agents=0` и `active_work=null`; на
   `:9119` и `:43189` нет установленных соединений (`ss -tnp state established`); в `gui.log` с
   момента старта текущего процесса serve (`ActiveEnterTimestamp` `hermes-autonomous.service`) у
   **каждого** `tui prompt accepted` есть парный `tui turn finished` с теми же `ui_session` и
   `agent_session_id` — по каждому чату, а не только у последнего принятого хода. Если хоть одно не
   выполнено — ждать, не перезапускать.
2. **Копии для отката.** Скопировать `~/.hermes/config.yaml` и затронутые unit-файлы
   (`~/.config/systemd/user/hermes-autonomous.service`, `hermes-gateway.service`; drop-in нет) рядом с отметкой времени UTC в имени; записать
   текущие `MainPID` и `ActiveState` обоих unit'ов.
3. **Изменение и перезапуск** выбранного варианта.
4. **Проверка:** `health` по ожиданию из предыдущего раздела; `systemctl --user show -p
   MainPID,ActiveState` обоих unit'ов.
5. **Откат:** вернуть копии `config.yaml` и unit-файлов, `systemctl --user daemon-reload`,
   перезапустить те же unit'ы (и включить обратно отключённый), проверить `MainPID`, `ActiveState`
   и `health` — снова `pid=<MainPID hermes-autonomous.service>`, `session_owner=false`.

Побочный эффект: systemd завершает дочерние процессы gateway (MCP), идущие ходы прерываются
(комментарий 7024288870 к #96).

## Этап A (2026-11-20)

### Выбор владельца

Владелец выбрал (б). Исследование исходников Hermes v0.23.1 (`3b42157858`) уточнило его в план по
этапам: этап A → проверка → Windows-клиент → этап B
([комментарий 7039818435](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7039818435)).
Причины:

- Desktop «Remote gateway» работает с listener'ом gateway. Условие: Webapp на `0.0.0.0` с
  провайдером входа (`gateway/run_runtime.py:586-608`, `gateway/run_api.py:472-493`).
- `serve` не передаёт чаты в authority gateway (`hermes_cli/web_routers/chat_ws.py:651-706`,
  `tui_gateway/ws.py:400`). Доставка в чат, созданный через `serve`, невозможна.
- Gateway остаётся в `hermes-gateway.service`: `hermes update` и restart знают только unit'ы
  `hermes-gateway*`. Unit-файл Hermes пересоздаёт, поэтому дополнения — через drop-in.
- Риск: Webapp даёт вошедшему пользователю терминал и файлы хоста
  (`hermes_cli/web_server_surface.py:61-76`). Соединение по LAN — plain HTTP.

Топология этапа A: один gateway `hermes-gateway.service` владеет authority и слушает LAN на
listener этапа A. `hermes-autonomous.service` (`serve` на `:9119`) работает рядом до этапа B. Этап B
(2026-11-21) перенёс listener gateway на `:9119`.

Владелец задал рамки следующего шага: остаток установить, этап A не повторять, новых перезапусков
нет ([комментарий 7051997437](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7051997437)).
Остаток записан в [Decision в #120](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7052302513).

### Сообщалось выполненным

Источник — отчёт оркестратора
([комментарий 7040152960](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7040152960)),
2026-11-20 ~16:00 MSK. Это отчёт исполнителя, а не сверка текущего состояния. Здесь только факты
этого отчёта; точное время старта и монтирование маршрутов — в «Проверено сейчас».

- Нагрузка до перезапуска: `active_agents=0`, `active_work=null`; на `:43189` соединений нет.
  Проверку 3 (`gui.log`) отчёт не записал.
- Копии: `~/.hermes/backups/stage-a-20261120T125829Z/` — `config.yaml`, оба unit-файла, `pids.txt`
  (gateway 1234703, autonomous 726109). Drop-in до этапа A не было.
- Drop-in `~/.config/systemd/user/hermes-gateway.service.d/30-lan-auth.conf`: `EnvironmentFile` —
  файлы dashboard-auth. Содержимое файлов здесь не записано.
- `~/.hermes/config.yaml`: `webapp: {enabled: true, host: 0.0.0.0, port: <порт этапа A>}`.
- `systemctl --user daemon-reload`, затем мягкий `hermes gateway restart`, около 16:00 MSK.
  `MainPID` 1234703 → 2297970 (gateway из `pids.txt` → PID после перезапуска).
- Чтение после перезапуска:
  - listener этапа A на `0.0.0.0` слушает процесс gateway;
  - `/api/status`: `auth_required: true`, провайдеры `basic` и `session-token`;
  - `GET /api/plugins/sound-scout/health` с LAN без входа — 401;
  - тот же GET с входом — `pid=2297970` (= `MainPID` `hermes-gateway.service`),
    `session_owner: true`.
- `serve` на `:9119` (PID 726109) не перезапускался, его health — 200. Платформ мессенджеров в
  gateway нет (`configured: 0`).
- `hermes webapp --status`: бандл Webapp не собран. Native Desktop он не нужен.
- В gateway работает копия Spike B (`dcb530e`).

### Проверено сейчас

Только чтение, 2026-11-20 22:30 UTC
([Decision в #120](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7052302513)).
Health и запросы с входом не выполнялись.

- `hermes-gateway.service`: active, enabled, `MainPID=2297970`, с 2026-11-20 12:58:49 UTC
  (`ActiveEnterTimestamp`), `hermes gateway run`. Drop-in `30-lan-auth.conf` на месте; env-файлы не читались.
- `hermes-autonomous.service`: active, enabled, `MainPID=726109`, с 2026-11-19 22:41:17 UTC,
  `hermes serve --isolated --host 0.0.0.0 --port 9119`, без drop-in.
- `~/.hermes/gateway_state.json`: `pid=2297970`, `running`, `active_agents=0`, `active_work=null`,
  платформ мессенджеров нет, `code_version=0.23.1`.
- `~/.hermes/config.yaml`: `webapp` — `enabled: true`, `host: 0.0.0.0`, `port` — порт этапа A.
- Listener'ы: listener этапа A — PID 2297970; `0.0.0.0:9119` — PID 726109. TCP `127.0.0.1:43189`
  больше не слушает. Control socket gateway — `~/.hermes/gateway.sock`.
- Установленные соединения: на listener этапа A — 0. На `:9119` — 5, все с одного LAN-хоста владельца, на
  оба адреса станции.
- Третий backend — Desktop SSH backend: PID 434482,
  `serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file …`, `127.0.0.1:37575`, с
  2026-11-20 06:56 UTC. SSH-сессия — с того же LAN-хоста. Это ещё один писатель в `state.db`.
- `~/.hermes/state.db` (`mode=ro`): после этапа A один новый чат `source=desktop` (21:54 UTC). Строк в
  `session_admissions` у него нет: чат прошёл через `serve`, мимо authority gateway.
- `dashboard-auth.log`: после этапа A входа на gateway с Windows-хоста не видно. Порт лог не пишет;
  картина та же, что до этапа A.
- `agent.log`: gateway смонтировал `/api/plugins/sound-scout/` при старте (12:58:53 UTC).
- Копия плагина `~/.hermes/plugins/sound-scout`: `version: 0.0.1`, все 5 файлов = `dcb530e`
  (Spike B, #96). `delivery.py` нет, маршруты `/spike/deliver` и `/spike/runner` есть.
  Production-копия — [#104](https://github.com/renidere/sound-scout-plugin/issues/104).
- Hermes Agent v0.23.1 (2026.9.24), stable, ревизия `3b4215785882`.
- Каталог копий `~/.hermes/backups/stage-a-20261120T125829Z/`: `config.yaml`, оба unit-файла,
  `pids.txt`.
- Windows-хост: управлять им со станции нельзя. Записи в `~/.ssh/config` и `known_hosts` нет;
  порты 22, 5985, 3389 закрыты или фильтруются; средств удалённого управления нет.
- Пропущено: GET `/api/status` (handler читает health gateway и может записать install id), любые
  запросы под `/api/plugins/`, любые запросы с входом.

### Ещё не проверено

- `session_owner` сейчас. Последнее чтение — отчёт об этапе A.
- Версия штатного Windows Desktop: записи нет ни на станции, ни в репозитории.
- Что Windows Desktop подключается к listener этапа A как Remote gateway и проходит вход.
- Что чат, созданный через listener этапа A, получает admissions в authority gateway.
- Доставка заметки задания (#102, #114). Нужна production-копия плагина (#104).
- Перенос старых чатов (`hermes sessions adopt`) и риск двух писателей у чатов, созданных через
  `serve` на `:9119` или через SSH backend.
- Проверка 3 (`gui.log`) перед этапом A: отчёт её не записал.
- Результаты Windows-зонда 0016: все строки — «ожидает прогона владельцем».

Один health с `session_owner: true` не доказывает сквозную работу Windows-клиента, доставку заметки
и перенос старых чатов.

### Клиентская проверка (2026-11-21)

**Сообщил владелец**
([комментарий 7059648596](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7059648596),
08:02 UTC): Windows Desktop подключён к Remote gateway на listener этапа A (Wi-Fi адрес станции),
контрольный чат — `20261121_110129_a8cd31`. Это сообщение владельца, а не сверка.

**Проверено сейчас.** Только чтение, 2026-11-21 08:04–08:15 UTC. `state.db` открыт с `mode=ro`.
Сообщения, запросы с входом, запросы под `/api/plugins/`, перезапуски и `adopt` не выполнялись.

- Unit'ы не менялись: `hermes-gateway.service` — active, `MainPID=2297970`;
  `hermes-autonomous.service` — active, `MainPID=726109`. `gateway_state.json`: `pid=2297970`,
  `running`, `active_agents=0`, `active_work=null`.
- Listener'ы те же: listener этапа A — PID 2297970, `0.0.0.0:9119` — PID 726109.
- `ss -tnp state established`: на listener этапа A — 0 соединений. На `:9119` — 4, все с Windows-хоста, у
  PID 726109.
- SSH backend сменился. PID 434482 нет. Новый PID 2599942
  (`serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file …`) работает с 07:02 UTC,
  у него 2 loopback-соединения. С Windows-хоста открыты 3 SSH-сессии.
- `dashboard-auth.log` после 2026-11-20 22:30 UTC: `ws_ticket_minted` (`basic`) в 07:01:53,
  07:01:54 и 07:14:20 UTC; `refresh_success` (`basic`) в 07:41:55 UTC; 800 отказов `no_cookie` на
  `/api/auth/ws-ticket`. Тикеты 07:01:54 и 07:14:20 совпадают по времени с `ws accepted` от
  Windows-хоста в `gui.log`. Эти два соединения сейчас установлены на `:9119` у PID 726109.
- Чат `20261121_110129_a8cd31` в `state.db`: `source=desktop`, `profile_name=default`, начат в
  08:01:30 UTC. 8 сообщений за 18 с (08:01:30–08:01:48), один ход. Строк в `session_admissions` — 0,
  `session_turn_leases` пуст.
- Ход шёл по пути `tui_gateway.server` (`tui prompt accepted` → `tui turn finished`,
  `ui_session=cf782595`) и записан в `gui.log`. `gui.log` открыт только у PID 726109 и 2599942, у
  gateway 2297970 его нет.
- Потоки клиента OpenAI этого хода (`Thread-3531`, `Thread-3534` в `agent.log`) имеют адреса стеков
  внутри карты памяти PID 726109 (`/proc/<pid>/maps`). В картах PID 2297970 и 2599942 этих адресов
  нет. Вывод: ход вёл `serve` на `:9119`. Это вывод по адресам потоков, а не запись процесса о себе.
- Вывод: за 08:01:30–08:01:48 второго писателя у этого чата не видно — один принятый и один
  законченный ход, сообщения подряд, admissions нет. Исключительное владение чатом одним процессом
  не проверено.
- Проверка 3 с 2026-11-19 22:41 UTC (старт PID 726109): 52 `tui prompt accepted`, у каждого есть
  парный `tui turn finished`.
- `bot_relay roster` в `gui.log` каждую минуту называет два подключения Desktop: по Wi-Fi адресу
  станции с портом 9119 и SSH-подключение `workstation`. Подключения к listener этапа A в нём нет.
- Копия плагина `~/.hermes/plugins/sound-scout`: `version: 0.0.1`, 5 файлов = `dcb530e`.
  `origin/main` (`3d20b19`) — `version: 0.1.0`.

Вывод: сообщение владельца проверкой не подтверждено. Контрольный чат прошёл через `serve` на
`:9119`, мимо authority gateway. Сейчас на listener этапа A установленных соединений нет. Тикет 07:01:53,
`refresh_success` 07:41:55 и отказы `no_cookie` к процессу и listener'у не отнесены: порт лог не
пишет. Гипотезы: (1) Remote gateway на listener этапа A в Desktop добавлен, но активным осталось
подключение `:9119`, и чат создан в нём; (2) попытка входа на listener этапа A не удалась, и клиент остался на `:9119`.

**Ещё не проверено:**

- Работа Windows Desktop через listener этапа A: вход, соединение у gateway `MainPID`, чат со строками в
  `session_admissions`.
- Версия Windows Desktop: в сообщении владельца её нет, логи станции её не пишут.
- `session_owner` сейчас: health не читался.
- Доставка заметки в контрольный чат: не разрешена; в чат `serve` она невозможна по устройству; нужны
  #104 и #114.
- Перенос старых чатов (`hermes sessions adopt`) и риск двух писателей у чатов `serve` и SSH
  backend.

### Откат этапа A

1. Проверить нагрузку (раздел «Проверка перед любым перезапуском» ниже). При невыполненном условии —
   ждать.
2. Вернуть копии из `~/.hermes/backups/stage-a-20261120T125829Z/`: `config.yaml` и unit-файлы.
3. Удалить drop-in `~/.config/systemd/user/hermes-gateway.service.d/30-lan-auth.conf`.
4. `systemctl --user daemon-reload`.
5. Перезапустить `hermes-gateway.service`.
6. Проверить `systemctl --user show -p MainPID,ActiveState hermes-gateway.service`: новый `MainPID`,
   `active`. Проверить, что listener этапа A не слушает, а `:9119` у `hermes-autonomous.service` —
   слушает. Health на `:9119` снова даёт `session_owner=false`.

`hermes-autonomous.service` этап A не менял, его откат этапа A не трогает. Откат — тоже перезапуск:
нужно отдельное разрешение владельца.

### Проверка перед любым перезапуском

Действует с этапа A. Все условия должны выполняться:

1. `~/.hermes/gateway_state.json`: `active_agents=0`, `active_work=null`.
2. `ss -tnp state established`: нет соединений на `:9119`. С этапа B (2026-11-21) это listener
   gateway; с этапа A до этапа B условие включало и listener этапа A.
3. `gui.log`: с момента старта каждого работающего backend у каждого `tui prompt accepted` есть
   парный `tui turn finished` с теми же `ui_session` и `agent_session_id`.
4. Desktop SSH backend (`serve … --ssh-session-token-file`) не ведёт активный чат.

### Остаток: клиентская проверка

**История (2026-11-21, до этапа B).** Шаги ниже ведут клиент на listener этапа A. Этап B перенёс Webapp на
`:9119` и записал вход владельца и чат с admissions — раздел «Этап B (2026-11-21)».

Шаг владельца (агент не может управлять Windows-хостом):

1. В штатном Windows Desktop добавить Remote gateway на listener этапа A — проводной
   или Wi-Fi адрес станции. Клиент сейчас ходит на оба адреса `:9119`.
2. Войти провайдером `basic` с теми же учётными данными, что для `:9119`: оба unit'а читают одни
   env-файлы dashboard-auth. Доступен и `session-token`. Пароли и токены в задачи не писать.
3. Создать один контрольный чат и отправить одно сообщение.
4. Сообщить версию Windows Desktop с экрана About. Иначе агент берёт её из следов входа.

Затем агент сверяет, только чтение:

- `ss -tnp state established`: соединение Windows-хоста на listener этапа A у gateway `MainPID` (2297970,
  если gateway не перезапускался);
- `dashboard-auth.log`: вход с Windows-хоста после шага владельца;
- `~/.hermes/state.db`: новый чат `source=desktop` со строками в `session_admissions`;
- для этого чата не появился новый `serve` или SSH backend;
- версия Windows Desktop и провайдер входа.

Один контрольный чат доказывает только подключение, вход и попадание нового чата в authority
gateway. Он не доказывает доставку заметки (нужны #104 и #114), перенос старых чатов и отсутствие
двух писателей у чатов, созданных через `serve` `:9119` или через SSH backend. Health с входом на
listener этапа A агент читает только по отдельному разрешению владельца.

### Остаток: этап B

**История (план до 2026-11-21).** Этап B выполнен 2026-11-21; перенос — только трёх чатов,
которые назвал владелец. Раздел «Этап B (2026-11-21)».

Каждое действие и окно владелец одобряет отдельно. Разрешение на этап A этап B не покрывает.

- Сервисы: остановить и отключить `hermes-autonomous.service`. Windows-клиент работает только через
  gateway.
- Убрать SSH-подключение Desktop: его backend — третий писатель.
- По желанию: перенести Webapp на `:9119`. Это правка `~/.hermes/config.yaml` и перезапуск
  gateway, только после остановки `serve`.
- Старые чаты: `hermes sessions adopt` только для чатов, названных владельцем, и только когда их не
  держит другой процесс.
- Плагин: production-копия [#104](https://github.com/renidere/sound-scout-plugin/issues/104) вместо
  Spike B. Она уберёт маршруты `/spike/*` и добавит цикл доставки. Нужен перезапуск gateway.
- Предусловия: четыре условия проверки выше; нет соединений на `:9119` и listener этапа A; SSH backend не
  работает. На 2026-11-20 22:30 UTC не выполнено: на `:9119` 5 соединений. На 2026-11-21 08:05 UTC
  не выполнено: на `:9119` 4 соединения, SSH backend PID 2599942 работает, клиент на listener этапа A не
  подтверждён.
- Перед этапом B — новые копии `config.yaml`, unit-файлов и drop-in; записать `MainPID` и
  `ActiveState` обоих unit'ов.
- Риски: два писателя одной истории; Webapp даёт вошедшему терминал и файлы хоста; plain HTTP по
  LAN; у Windows-клиента меняется адрес.
- Откат этапа B: вернуть копии `config.yaml`, unit-файлов и drop-in, снятые перед этапом B.
  Удалить только drop-in, созданные на этапе B; `30-lan-auth.conf` этапа A остаётся (его удаляет
  только откат этапа A). Затем `daemon-reload`, перезапустить `hermes-gateway.service`, включить и
  запустить `hermes-autonomous.service`. Сверить `MainPID`, `ActiveState` и listener'ы.
- После этапа B: health даёт `pid` = gateway `MainPID` и `session_owner=true`; этап B
  записывается в это решение. Живая доставка остаётся в #114.

### Шаг B1 (предложение 2026-11-21)

**История (предложение до этапа B).** Ход этапа B 2026-11-21 и его итог — раздел
«Этап B (2026-11-21)».

Клиентская проверка 2026-11-21 не прошла. Поэтому этап B начинается с обратимого шага: остановить
`serve` на `:9119`, чтобы клиент работал только через gateway. Отключение unit'а, перенос
listener'а, `hermes sessions adopt` и #104 в B1 не входят.

Что затрагивает:

- `hermes-autonomous.service` (PID 726109, `:9119`): `stop`, без `disable`.
- Desktop SSH backend (PID 2599942): владелец закрывает SSH-подключение `workstation` в Desktop.
  Агент процесс не останавливает. Если процесс остаётся, агент записывает это; его остановка в B1 не
  входит. Его cgroup на 2026-11-21 — scope SSH-сессии
  (`user.slice/user-1000.slice/session-<n>.scope`), а не `hermes-autonomous.service`: `stop` unit'а
  его не завершает. Перед шагом 4 агент перечитывает `/proc/<pid>/cgroup`.
- Предусловие этапа B «SSH backend не работает» относится к шагам после B1 (`disable`, перенос
  listener'а, `adopt`, #104). Для B1 достаточно условия 4 проверки перед перезапуском: SSH backend не
  ведёт активный чат. B1 этот процесс не трогает.
- `hermes-gateway.service`: не трогается, `MainPID` прежний.
- Сессии: чаты, открытые через `:9119` и SSH, в том числе `20261121_110129_a8cd31`, закрываются у
  клиента. Строки `state.db` не меняются.

Порядок:

1. Владелец в окне закрывает Windows Desktop или отключает в нём подключения `:9119` и SSH.
2. Агент читает четыре условия проверки перед перезапуском и записывает их вывод. Одно условие
   ложно — стоп, ждать.
3. Агент копирует оба unit-файла и drop-in в новый каталог `~/.hermes/backups/stage-b1-<UTC>/` и
   записывает `MainPID` и `ActiveState` обоих unit'ов.
4. Агент выполняет `systemctl --user stop hermes-autonomous.service`. Сверка: `inactive`, `:9119`
   не слушает, gateway `MainPID=2297970`.
5. Владелец открывает Desktop, выбирает Remote gateway на listener этапа A, входит `basic`, создаёт новый
   контрольный чат с одним сообщением и сообщает ID чата и версию Desktop.
6. Агент сверяет только чтением: соединение на listener этапа A у gateway `MainPID`, тикет в
   `dashboard-auth.log`, новый чат со строками в `session_admissions`, нового SSH backend нет.

Риски: клиент теряет `:9119`; если вход на listener этапа A не пройдёт, клиент без Hermes до отката; чат,
продолженный в другом процессе, получает второго писателя; Webapp даёт вошедшему терминал и файлы
хоста; plain HTTP по LAN. Unit остаётся `enabled`: после перезагрузки станции `serve` на `:9119`
снова запустится.

Откат B1: `systemctl --user start hermes-autonomous.service`; сверить `active`, `:9119` слушает
новый `MainPID`; клиент возвращается к подключению `:9119`. Откат входит в то же разрешение и
окно.

## Этап B (2026-11-21)

Источник — отчёты оркестратора: в #120
([комментарий 7060133566](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7060133566), 11:28–11:38 MSK;
[комментарий 7060203442](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7060203442), 11:52 MSK;
[комментарий 7060273214](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7060273214), 12:00 MSK), в #104
([комментарий 7060137735](https://github.com/renidere/sound-scout-plugin/issues/104#issuecomment-7060137735)) и в #114
([комментарий 7060137985](https://github.com/renidere/sound-scout-plugin/issues/114#issuecomment-7060137985)). Это отчёты исполнителя; сверки в них —
только чтение.

Разрешение владельца — «разрешаю»
([комментарий 7059680390](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7059680390)), затем «давай» на перенос Webapp gateway
на `:9119`. Владелец был у Windows-клиента.

### Ход

1. 11:28 MSK — `systemctl --user stop hermes-autonomous.service`. Клиент не смог войти на
   listener этапа A: в `dashboard-auth.log` — `ws-ticket` `no_cookie`, входа на него не было. 11:31 — откат:
   `start`, `:9119` снова у `serve`.
2. 11:32 — копии в `~/.hermes/backups/stage-b-port9119-20261121T083255Z/`. В `~/.hermes/config.yaml`
   `webapp.port` → `9119`. Затем `stop hermes-autonomous.service` и
   `restart hermes-gateway.service`. Gateway слушает `0.0.0.0:9119`.
3. 11:33:57 MSK (08:33:57 UTC) — `login_success` (`basic`) с Windows-хоста. Новый чат владельца
   `local-d483e1ff…a0de914`: строка `session_admissions` seq 96, `owner_epoch` 14, `completed`.
   Соединения на `:9119` — у gateway.
4. `systemctl --user disable hermes-autonomous.service`: unit `inactive`, `disabled`.
5. 11:37 — production-копия плагина [#104](https://github.com/renidere/sound-scout-plugin/issues/104) 0.1.0 заменила Spike B 0.0.1 (`dcb530e`), затем
   перезапуск gateway. Копии до правки — `~/.hermes/backups/deploy-104-20261121T083704Z/`.
   `mcp_servers.speech-review.tools.include` — 9 инструментов; `agent.log`: `registered 9 tool(s)`.
   После `restart hermes-gateway.service` — `Mounted plugin API routes: /api/plugins/sound-scout/`
   (отчёт #104 не называет файл лога); поток `sound-scout-delivery` работает в процессе gateway.
   Маршрутов `/spike/*` нет.
6. 11:38 — проверка доставки [#114](https://github.com/renidere/sound-scout-plugin/issues/114): задание из CLI, `jobs status` — `finished`, доставка
   `delivered`, `confirmed_by: send`. Метка задания в `state.db` (`mode=ro`) есть ровно в одном
   сообщении — в чате `local-d483e1ff…a0de914`. Владелец подтвердил, что видит заметку.
7. К 11:52 (время отчёта) владелец удалил в Desktop лишние подключения и профили, в том числе
   SSH-подключение `workstation`. `gui.log` 11:51:47 — SSH-соединения `127.0.0.1` закрыты.
8. 12:00 — перенос трёх старых чатов, которые назвал владелец: `20261102_120521_cf6e0f`,
   `20261102_121441_41b4af`, `20261117_002327_f834d5`. `hermes sessions adopt` переносит только все
   чаты сразу (1215). Поэтому агент перенёс ровно эти три тем же кодом Hermes:
   `hermes_cli.sessions_adopt.adoptable_sessions` → фильтр по трём ID → `adopt_sessions`
   (`session.resume` и `session.detach` по нативному соединению gateway). До переноса — копия
   `state.db` через SQLite backup API, `integrity_check` = `ok`
   (`~/.hermes/backups/adopt-3-20261121T085802Z/`), `active_agents=0`. Итог: `adopted 3 skipped {}`;
   ни один из трёх больше не числится среди переносимых; `hermes sessions adopt --dry-run` — 1212
   (было 1215).

### Состояние после этапа B

Сверка только чтением, 2026-11-21 11:38 и 11:52 MSK.

- `hermes-gateway.service`: active, `0.0.0.0:9119`. Listener этапа A закрыт.
- `hermes-autonomous.service`: inactive, disabled.
- Desktop SSH backend больше нет: процесса PID 2599942 нет, `pgrep -af ssh-session-token-file` пуст.
  Запись в `~/.hermes/spawn-ledger.json` осталась, она устарела. `gui.log` 11:51:47 MSK:
  SSH-соединения `127.0.0.1` закрыты (`client_disconnect`).
- На `:9119` установлены только соединения Windows-хоста у gateway (PID 3578226).
- Hermes Agent v0.23.1, ревизия `3b4215785882`.

Итог: у чатов Desktop один процесс-владелец — `hermes gateway run`. Windows Desktop входит в
gateway на `:9119`. Новый чат владельца получил admissions в authority gateway. Три названных
владельцем старых чата перенесены в authority gateway.

### Не сделано и не проверено

- Остальные 1212 старых чатов не переносились: владелец назвал только три. Перенос другого чата —
  только по выбору владельца.
- Health с входом после этапа B не читался: агент не входил и пароль не читал (#104). Вывод, не
  чтение: цикл доставки делает тик только при `owner_for_home`, поэтому `delivered` в #114 говорит,
  что gateway видит owner.
- Четыре условия проверки перед перезапуском отчёты этапа B не записали.
- Версия Windows Desktop в отчётах этапа B не записана.
- Readback `plugins.isolation`: ключ в `config.yaml` не задан (#104).
- Остаток #114: постановка задания через MCP `speech_job_submit` из сессии Hermes, перезапуск
  gateway во время задания, `delivery_unconfirmed` вживую.

### Откат этапа B

1. Вернуть `~/.hermes/config.yaml` из `~/.hermes/backups/stage-b-port9119-20261121T083255Z/`.
2. `systemctl --user restart hermes-gateway.service`.
3. `systemctl --user enable --now hermes-autonomous.service`. Клиент снова входит на `:9119` в
   `serve`.

Откат #104: вернуть каталог плагина, `config.yaml`, `sound-scout.toml` и скил из
`~/.hermes/backups/deploy-104-20261121T083704Z/`, удалить drop-in `40-sound-scout.conf`, затем
`systemctl --user daemon-reload` и `restart hermes-gateway.service`.

Оба отката сразу: сначала откат #104, затем откат этапа B. Обе копии содержат `config.yaml`.
Вывод по времени копий, не чтение. Копия этапа B (11:32) старше развёртывания #104 (11:37): её
`config.yaml` отменяет и правку #104 `mcp_servers.speech-review.tools.include`. Копия #104 моложе
смены порта: её `config.yaml` снова ставит `webapp.port: 9119`, поэтому в обратном порядке порт
не вернётся.

Откат переноса трёх чатов: остановить gateway, вернуть `state.db` из
`~/.hermes/backups/adopt-3-20261121T085802Z/`, запустить gateway. Записи после 12:00 MSK потеряются.

## Последствия для #102

- Критерий 1 #102 («цикл доставки только в процессе — владельце authority») означает
  `hermes gateway run` (сейчас — `hermes-gateway.service`), а не `hermes-autonomous.service`.
  Маршруты `plugin_api.py` смонтированы в serve, поэтому цикл не может жить там. Цикл запускается из
  части плагина, которую загружает gateway (`__init__.py`, `register(ctx)`), или из дашборда,
  который хостит сам gateway, и только при `owner_for_home(get_hermes_home()) is not None`. Без
  owner задание остаётся в ожидании доставки: не получает ошибку и не уходит в другой процесс.
- Код #102 и тесты с поддельной функцией доставки (критерий 5 #102) можно писать уже сейчас.
- **Не решено (гипотеза о риске):** Desktop-чаты, созданные через serve, не принадлежат authority (у
  них нет admissions). `deliver_to_local_chat` из gateway восстановит такой чат в своей authority
  (`restore_local_session`), а serve может в это время держать тот же чат в памяти: два писателя
  одной истории. Увидит ли Desktop заметку ровно один раз, не проверено. Поэтому реальную доставку в
  #102 нельзя принять, пока владелец не выберет топологию или не проведёт живую проверку Spike B
  (шаг 5 0015) с вызовом из gateway.
- Ожидание шага 4 в 0015 (`session_owner=true` через `ctx.rest`) в текущей топологии не выполнится:
  `ctx.rest` Desktop идёт на `:9119`, то есть в serve.
- **После этапа A (2026-11-20):** топология выбрана, gateway слушает listener этапа A. Windows-клиент пока
  подключён к `:9119`, поэтому оба пункта выше остаются в силе. Они снимаются, когда клиент
  работает через gateway и в нём стоит production-копия (#104). Живая доставка — #114.
  **2026-11-21:** контрольный чат владельца тоже прошёл через `serve` на `:9119` (раздел
  «Клиентская проверка (2026-11-21)»); оба пункта в силе.
  **2026-11-21, этап B:** `ctx.rest` и чаты Desktop идут в gateway на `:9119`, `serve` остановлен,
  в gateway стоит production-копия #104. Заметка задания дошла в чат владельца (#114). Три
  названных владельцем старых чата перенесены; остальные 1212 старых чатов не переносились. Раздел
  «Этап B (2026-11-21)».

- **2026-11-22:** заметке о задании из MCP нужен тот же процесс-владелец: хук `post_tool_call`
  плагина `sound-scout` в gateway привязывает задание к локальному чату хода — решение 0025
  (`docs/decisions/0025-mcp-chat-binding.md`).

## Что не установлено

Список на 2026-11-20 до этапа A. Пометки «с этапа A» — изменения после него.

- `owner_for_home` внутри PID 1234703 вживую: маршрут health там не смонтирован, а первый запрос к
  `:43189` смонтировал бы его — это изменение живого процесса. Факт владения опирается на трассу по
  исходникам, flock `gateway.lock` и `state.db`; живое чтение ждёт разрешения владельца.
  **С этапа A:** PID 1234703 больше нет. Для нового gateway (PID 2297970) оркестратор сообщил health
  с входом на listener этапа A: `session_owner: true`. Сейчас health не перечитывался.
- Связь `instance_id` в `runtime_epoch` с PID: дескриптор runtime не записан в файл.
- cgroup SSH-backend'а Desktop (PID 4084248): процесс завершился до наблюдения. **С этапа A:**
  работает другой SSH backend, PID 434482; его cgroup тоже не записан. **2026-11-21:** PID 434482
  нет, работает PID 2599942 (с 07:02 UTC); его cgroup — scope SSH-сессии, не unit Hermes.
- Работает ли штатный Windows-клиент с Webapp listener'ом gateway (вариант (а)). **С этапа A:**
  по-прежнему не установлено, теперь это клиентская проверка этапа A; на listener этапа A соединений нет.
  **2026-11-21 (этап B):** установлено для listener'а на `:9119`: вход `basic` с Windows-хоста,
  новый чат со строкой `session_admissions` seq 96.
- **С этапа A:** версия штатного Windows Desktop; доставка заметки в чат (#104, #114); перенос
  старых чатов и риск двух писателей. Полный список — «Этап A (2026-11-20)», «Ещё не проверено».
  **2026-11-21 (этап B):** заметка доставлена (#114); три названных владельцем старых чата
  перенесены. Версия Windows Desktop по-прежнему не установлена — «Этап B (2026-11-21)», «Не сделано
  и не проверено».
