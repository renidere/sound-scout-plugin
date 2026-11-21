# Доступ к Hermes: один владелец чатов для всех клиентов

- **Задача:** [#137](https://github.com/renidere/sound-scout-plugin/issues/137); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92).
- **Основание:** `docs/decisions/0017-session-authority-owner.md`, раздел «Этап B (2026-11-21)»;
  отчёты в #120: [7060133566](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7060133566),
  [7060203442](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7060203442),
  [7060273214](https://github.com/renidere/sound-scout-plugin/issues/120#issuecomment-7060273214).
- **Hermes на станции:** Hermes Agent v0.23.3, ревизия `caafe5389a` (`~/.hermes/hermes-agent`,
  обновлён автоматически 2026-11-21). Ссылки `файл:строка` ниже даны на эту ревизию. Локальные
  правки в ней есть только в `apps/desktop`, `website/docs`, `tests/`.
- **Как получено:** только чтение, 2026-11-21 23:00–23:15 UTC. Ничего не перезапускалось и не
  менялось. Сообщения, вход и запросы к gateway не выполнялись.

## Владелец чатов

Профиль `default` (`~/.hermes`, `~/.hermes/state.db`). Чатами владеет один процесс:
`hermes gateway run` из `hermes-gateway.service`. Только он создаёт session authority
(0017, «Где привязывается authority»). Документация Hermes называет gateway единственным
писателем истории (`website/docs/developer-guide/session-storage.md:7–12`).

Сверка 2026-11-21 23:11 UTC:

- `hermes-gateway.service` — `active`, `enabled`, `MainPID=1390761`, старт 2026-11-21 13:06:01 UTC;
- `hermes-autonomous.service` — `inactive`, `disabled`;
- `0.0.0.0:9119` слушает только PID 1390761. Это Webapp gateway (`webapp.host`/`webapp.port` в
  `~/.hermes/config.yaml`). Других TCP-listener'ов у процессов Hermes нет. Control socket —
  `~/.hermes/gateway.sock`;
- живых `hermes serve`, `hermes dashboard` и `tui_gateway.entry` нет;
- `~/.hermes/gateway_state.json`: `pid=1390761`, `running`, `code_version=0.23.3`,
  `active_agents=0`, платформ мессенджеров нет.

## Поддерживаемые пути

| путь | как подключается | владелец чата | доходит до gateway |
|---|---|---|---|
| Windows Desktop по LAN | Remote gateway `http://<LAN-адрес станции>:9119`, вход `basic` | `hermes gateway run` (`hermes-gateway.service`) | да, проверено на этапе B |
| Windows Desktop по Tailscale | Remote gateway `http://<Tailscale-адрес станции>:9119`, тот же вход | `hermes gateway run` (`hermes-gateway.service`) | по устройству да; вживую не проверено, Tailscale не установлен |
| `hermes` CLI на станции | `hermes chat` → `hermes gateway ensure` → локальный `/api/ws` gateway | `hermes gateway run` (`hermes-gateway.service`) | да, по коду |
| TUI на станции | `hermes --tui` → attach к gateway → локальный `/api/ws` | `hermes gateway run` (`hermes-gateway.service`) | да, по коду |
| Мессенджеры | адаптер платформы внутри процесса gateway | `hermes gateway run` (`hermes-gateway.service`) | да, по коду; платформ сейчас нет |

Ни один путь не запускает второй `serve` и Desktop SSH backend.

### Windows Desktop по LAN

Desktop подключается к Webapp listener'у gateway на `0.0.0.0:9119`. Gateway монтирует этот
listener со своей session authority (`gateway/run_api.py:99`). Сокет чата получает authority
(`hermes_cli/web_routers/chat_ws.py:665`), запросы идут в неё, а не в legacy `server.dispatch`
(`tui_gateway/ws.py:378–401`).

Факты:

- этап B, 2026-11-21 08:33:57 UTC: `login_success` (`basic`) с Windows-хоста, новый чат
  владельца получил строку `session_admissions` (0017);
- `dashboard-auth.log`: последний `login_success` с Windows-хоста — 2026-11-21 12:55:53 UTC,
  последний `ws_ticket_minted` — 16:45:46 UTC. Это после перезапуска gateway в 13:06 UTC;
- в 23:11 UTC установленных соединений на `:9119` нет: Desktop сейчас не подключён.

Версия Windows Desktop 0.23.3 — сообщение владельца. Со станции её не видно: логи её не пишут.

Риски (0017): plain HTTP по LAN; вошедший в Webapp получает терминал и файлы хоста.

### Windows Desktop по Tailscale

Тот же listener. `0.0.0.0` слушает все интерфейсы станции, в том числе будущий `tailscale0`.
Новый listener, правка `webapp.host` и перезапуск gateway не нужны.

На 2026-11-21 23:11 UTC Tailscale не установлен: нет программ `tailscale` и `tailscaled`, нет
интерфейса `tailscale0`. Вход по Tailscale не проверен.

Проверка после установки Tailscale (шаг владельца — установка и вход в Desktop; агент сеть не
меняет):

1. Владелец ставит Tailscale на станцию и на Windows-хост.
2. В Desktop владелец добавляет Remote gateway `http://<Tailscale-адрес станции>:9119`.
3. Владелец входит `basic` и создаёт один чат с одним сообщением. Пароль в задачу не пишется.
4. Агент сверяет только чтением:
   - `ss -tnp state established`: соединение с Tailscale-адреса на `:9119` у `MainPID`
     `hermes-gateway.service`;
   - `dashboard-auth.log`: `login_success` с Tailscale-адреса;
   - `~/.hermes/state.db` (`mode=ro`): новый чат со строкой в `session_admissions`;
   - `scripts/session_owner_check.py` даёт `verdict: ok`.
5. Агент пишет итог комментарием в #137 без адресов и паролей.

### `hermes` CLI на станции

`cmd_chat` выбирает один из двух маршрутов (`hermes_cli/main.py:1491–1498`). Оба доходят до
gateway профиля, Hermes v0.23.3 (`caafe5389a`).

Обычный маршрут — интерактивный `hermes chat` и `-q`/`-Q`. Запускается классический
`HermesCLI` (`cli_main`, `hermes_cli/main.py:1494–1495`, `:1580`). Агент для него —
`GatewayAgent` под именем `AIAgent` (`hermes_cli/cli_agent_setup_mixin.py:641`). По docstring
модуля CLI-процесс не строит агента, не запускает инструменты и не пишет `state.db`, ход идёт в
gateway (`hermes_cli/gateway_agent.py:1–6`). Хранилище сессий CLI — `GatewaySessionStore`: каждая
запись — verb gateway, `state.db` только читается (`hermes_cli/gateway_session_store.py:1–14`).
Соединение — `GatewayConnection` через `connect_gateway`
(`hermes_cli/gateway_connection.py:264–265`).

Обходной маршрут выбирают только аргументы: `--format stream-json`, `--safe-mode` или
`--ignore-user-config` без `--tui` (`hermes_cli/main.py:1496–1498`,
`hermes_cli/main_tui_launch.py:22–27`). Переменные `HERMES_SAFE_MODE` и
`HERMES_IGNORE_USER_CONFIG` сами маршрут не меняют: без флагов идёт обычный маршрут `HermesCLI`.
Обходной маршрут идёт в `launch_gateway_chat` (`hermes_cli/gateway_chat_startup.py:33`) и тонкий
клиент `hermes_cli/gateway_chat.py`: «through the canonical gateway, never AIAgent» (`:1`). Клиент
подключается тем же `connect_gateway` (`:172`) и создаёт сессию `session.create` на том же gateway
(`:223`). Флаги запуска уходят в эту сессию как её политика (`:207`). В этом маршруте переменные
окружения тоже попадают в политику (`:211`) и в проверку `bypass_launch` (`:69–73`). Второй
gateway и локальный агент не стартуют. Этот маршрут не второй писатель. Клиент отклоняет флаги
`--image`, `-w`, `--run-budget`, `--verbose`, `--compact` до подключения
(`hermes_cli/gateway_chat.py:17–21`, `:92`). Новая сессия с safe mode или ignore-user-config
(флагом или переменной) без `--model` получает отказ: профиль не читается (`:106–109`). Возобновление
сессии и `--format stream-json` без этих режимов `--model` не требуют.

Общая часть обоих маршрутов: `connect_gateway` ищет gateway профиля или запускает его через
`ensure_gateway_runtime` (`hermes_cli/gateway_client.py:330–361`) и подключается к локальному
`/api/ws` по одноразовому ticket'у.

Если gateway не работает, `ensure` запускает установленный systemd-unit, то есть
`hermes-gateway.service` (`hermes_cli/gateway_runtime.py:524–525`,
`hermes_cli/gateway_runtime_service.py:114–124`; `website/docs/user-guide/desktop.md:634`).
Второй gateway на тот же home не стартует: его держит flock `~/.hermes/gateway.lock` (0017).

Вживую не проверялось: `hermes chat` создаёт сессию в живом `state.db` через gateway.

### TUI на станции

`hermes --tui` запускает `ui-tui`. Клиент подключается через `localGatewayBootstrap.ts`
(`ui-tui/src/gatewayClient.ts:17–20`, `ui-tui/src/localGatewayBootstrap.ts:40`) к gateway
профиля по локальному `/api/ws` (`website/docs/user-guide/tui.md:296–309`).

Если gateway не работает, TUI запускает его через тот же `ensure_gateway_runtime` с
`idle_exit=True` (`ui-tui/scripts/gateway_bootstrap.py:53`,
`hermes_cli/gateway_runtime_attach.py:106`). Документация говорит, что такой gateway завершается
после простоя (`tui.md:298`). По коду это верно только без установленного unit'а: при
`~/.config/systemd/user/hermes-gateway.service` `ensure` запускает этот unit, а не отдельного
демона (`hermes_cli/gateway_runtime.py:524–529`, `hermes_cli/gateway_runtime_service.py:114–124`).
На станции unit установлен, поэтому TUI поднимает `hermes-gateway.service`. Второго писателя нет
и без unit'а: тот же flock `gateway.lock`. Gateway вне unit'а `scripts/session_owner_check.py`
отметит как `owner_outside_unit`.

`HERMES_TUI_GATEWAY_URL` вручную не задаётся: это внутренняя связь дашборда (`tui.md:300–302`).

Вживую не проверялось: TUI создаёт сессию в живом `state.db` через gateway.

### Мессенджеры

Адаптеры платформ (`gateway/platforms/`) работают внутри `hermes gateway run`. Описание unit'а —
«Hermes Agent Gateway - Messaging Platform Integration». Сейчас платформ нет
(`gateway_state.json`). Подключение платформы — правка `~/.hermes/config.yaml` и перезапуск
gateway: нужно отдельное разрешение владельца с окном.

## Не поддерживается

| путь | почему |
|---|---|
| Desktop SSH mode (подключение по SSH, например `workstation`) | Desktop запускает на станции `serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file …` (`apps/desktop/electron/remote-lifecycle.ts:1142`, `:1161`). У `serve` нет session authority, ходы идут в legacy `server.dispatch` (`tui_gateway/ws.py:401`): этот backend — второй писатель истории чатов. |
| `hermes serve`, в том числе `hermes-autonomous.service` | тот же legacy-путь: второй писатель (0017, «Почему у `hermes serve` authority нет»). |
| `hermes dashboard` и его вкладка Chat | дашборд запускает встроенный TUI на своём `tui_gateway` (`tui.md:300`), не на gateway. |

Контракт `hermes gateway attach` / `hermes gateway tickets` для SSH-клиентов в Hermes есть
(`website/docs/developer-guide/gateway-ssh-clients.md`). Но Desktop этой ревизии в SSH mode его не
использует и запускает `serve` (`remote-lifecycle.ts:1122–1161`). Поэтому SSH mode не
поддерживается, пока Desktop не перейдёт на этот контракт и проверка не покажет admissions у чата.

Устаревшая запись: `~/.hermes/spawn-ledger.json` на 2026-11-21 не содержит записей `serve` и
`dashboard`. Записи `mcp-helper` (MCP-сервер Sound Scout `sound-scout serve`), `browser-harness-daemon`
и `code-kernel` — дочерние процессы gateway, а не backend'ы.

## Проверка владельца

```sh
uv run python -I scripts/session_owner_check.py [--port 9119]
```

Скрипт только читает: `systemctl --user show` unit'ов `hermes-gateway.service` и
`hermes-autonomous.service`, `ss -ltnpH`, `ss -tnpH state established`,
`<HERMES_HOME>/spawn-ledger.json`, argv и время старта процессов из `/proc`. Он печатает состояния,
PID, порты, счётчики и фиксированные метки команд Hermes (`gateway`, `gateway <подкоманда>`,
встроенная подкоманда или `client`). Argv, окружение, пути и адреса клиентов он не печатает.

Запись ledger `serve` или `dashboard` живая, когда её PID жив, `boot_id` совпадает с текущей
загрузкой, а `create_time` совпадает со стартом процесса из `/proc/<pid>/stat` (допуск 2 с, как
в Hermes). Такая запись — отдельный признак второго писателя, даже если argv процесса не
распознан. Запись без `create_time` скрипт помечает `unverified` и писателем не считает. PID,
занятый другим процессом, даёт `stale`.

Последняя строка — вердикт:

| вердикт | код выхода | значение |
|---|---|---|
| `ok` | 0 | работает только gateway из `hermes-gateway.service`, он слушает `--port` |
| `second_writer` | 1 | работает `hermes serve`, `hermes dashboard` или `tui_gateway.entry`; активен `hermes-autonomous.service`; listener или соединение у процесса Hermes, который не gateway; живая запись ledger `serve`/`dashboard` у процесса, который не gateway |
| `no_owner` | 2 | нет живого `hermes gateway run` |
| `owner_outside_unit` | 2 | gateway работает, но не в `hermes-gateway.service` |
| `listener_missing` | 2 | gateway не слушает `--port` |
| `unreadable` | 2 | `systemctl` или `ss` не ответили, или ledger — не список JSON |

Прогон 2026-11-21 23:11 UTC на станции: `verdict: ok — gateway PID 1390761 is the only writer,
listens on :9119`, код 0.

Тесты — `tests/test_session_owner_check.py` (синтетические вывод `systemctl`/`ss`, `/proc` и
ledger; живой хост не нужен).
