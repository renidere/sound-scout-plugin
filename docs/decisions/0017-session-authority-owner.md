# 0017 — Какой процесс Hermes владеет session authority Desktop

- **Статус:** факт установлен; выбор топологии ждёт владельца.
- **Дата:** 2026-11-20.
- **Решение:** [Decision в #116](https://github.com/renidere/sound-scout-plugin/issues/116#issuecomment-7025500309).
- **Задача:** [#116](https://github.com/renidere/sound-scout-plugin/issues/116); эпик
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
  2026-11-19 23:49–23:50 UTC. Ничего не перезапускалось и не менялось.

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

Session authority профиля `default` (`~/.hermes`, `~/.hermes/state.db`) принадлежит процессу
`hermes gateway run` из `hermes-gateway.service` (PID 1234703). `hermes-autonomous.service` запускает
`hermes serve --isolated --host 0.0.0.0 --port 9119` (PID 726109): это backend, к которому
подключается Windows Desktop. Hermes создаёт authority и вызывает `bind_owner` только внутри
`hermes gateway run`: для профиля запуска (`default`) — при старте процесса, до подключения клиентов
и до открытия чатов; для дополнительного профиля — при hot-serve в уже работающем gateway.
`hermes serve` authority не создаёт никогда. Поэтому `session_owner=false` в PID 726109 — постоянное свойство текущей
топологии, а не ленивая привязка. Desktop-чаты станции идут мимо session authority (legacy-путь
`tui_gateway`).

Посылка 0013 § 9 п. 5 («процесс-владелец чатов Desktop — `hermes-autonomous.service`») неверна.

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

Третий процесс — `serve --isolated`, который Desktop запускает через SSH. Он есть в
`~/.hermes/spawn-ledger.json` (`port=42421`, `spawner_pid=None`, то есть отсоединён от SSH) и в
таблице `gateway_heartbeats` (последний heartbeat 2026-11-19 23:02:42 UTC). С 23:42 UTC его нет в
`ps`, поэтому его cgroup не наблюдался. Session authority он тоже не создаёт: это `serve`.

Повторить наблюдение (только чтение):

- `systemctl --user show -p MainPID,ExecStart,ControlGroup,ActiveEnterTimestamp hermes-autonomous.service hermes-gateway.service`;
- `tr '\0' ' ' < /proc/<pid>/cmdline`, `cat /proc/<pid>/cgroup`, `HERMES_HOME` из `/proc/<pid>/environ`;
- `lslocks -o PID,TYPE,MODE,PATH | grep -E 'gateway.lock|gateway-locks'`;
- `ss -ltnp | grep -E '9119|43189'`, соединения — `ss -tnp state established | grep -E ':9119|:43189'`;
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

Ни один не выбран. Оба требуют перезапуска gateway и меняют либо адрес подключения
Windows-клиента, либо unit'ы. Решает владелец.

**(а) Desktop → Webapp listener `hermes-gateway.service`.** В `~/.hermes/config.yaml` включается
`webapp.enabled: true`, `webapp.host: 0.0.0.0`, `webapp.port` (по умолчанию 9120); gateway читает
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

## Что не установлено

- `owner_for_home` внутри PID 1234703 вживую: маршрут health там не смонтирован, а первый запрос к
  `:43189` смонтировал бы его — это изменение живого процесса. Факт владения опирается на трассу по
  исходникам, flock `gateway.lock` и `state.db`; живое чтение ждёт разрешения владельца.
- Связь `instance_id` в `runtime_epoch` с PID: дескриптор runtime не записан в файл.
- cgroup SSH-backend'а Desktop (PID 4084248): процесс завершился до наблюдения.
- Работает ли штатный Windows-клиент с Webapp listener'ом gateway (вариант (а)).
