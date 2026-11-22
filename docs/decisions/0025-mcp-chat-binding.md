# 0025 — Привязка MCP-задания к чату Hermes

- **Статус:** принято, реализовано в коде и тестах. Живая проверка в Hermes — шаг владельца
  (раздел «Что не проверено вживую»).
- **Дата:** 2026-11-22.
- **Решение:** [Decision в #140](https://github.com/renidere/sound-scout-plugin/issues/140#issuecomment-7071237568)
  и [уточнение](https://github.com/renidere/sound-scout-plugin/issues/140#issuecomment-7071271050).
  Уточнение заменяет вариант `pre_tool_call` первого Decision; где они расходятся, действует
  уточнение.
- **Задача:** [#140](https://github.com/renidere/sound-scout-plugin/issues/140); найдена в
  [#114](https://github.com/renidere/sound-scout-plugin/issues/114); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92).
- **Основание:** `docs/decisions/0013-voice-note-integration.md` § 3.6 (задания) и § 3.7
  (доставка); Decision о доставке в [#102](https://github.com/renidere/sound-scout-plugin/issues/102);
  `docs/decisions/0017-session-authority-owner.md` (процесс-владелец чатов); R09–R11
  `docs/spec/REQUIREMENTS.md`.
- **Hermes:** Hermes Agent v0.23.3, ревизия `caafe5389a` (`~/.hermes/hermes-agent`), только чтение.
  Ссылки `файл:строка` ниже даны на эту ревизию.

## Решение

MCP-задания получают привязку к чату. Её источник — контекст сессии хода Hermes, прочитанный
плагином `sound-scout` в процессе Hermes после вызова. Аргументы инструмента и текст модели
привязку не дают никогда.

- У `speech_job_submit` нет полей `connection`, `profile`, `session_id`. Закрытая схема отказывает
  любому лишнему аргументу: код `invalid_arguments` (как у любого нарушения схемы MCP-сервера),
  `job.json` не создаётся. Задание из MCP всегда создаётся без привязки (`submitter = "mcp"`,
  остальные поля `null`).
- Плагин `sound-scout` регистрирует хук `post_tool_call` (`hermes/plugin/sound-scout/binding.py`).
  Для `mcp__<сервер>__speech_job_submit` со `status == "ok"` хук берёт `job_id` из результата
  вызова, а id чата — из `HERMES_SESSION_CHAT_ID` при `HERMES_SESSION_PLATFORM == "local"`. Затем он
  вызывает по argv `sound-scout jobs bind --session-id=<chat_id> --profile=<профиль> <job_id>`.
  `args` модели хук не читает.
- `jobs bind` (`jobs.bind`) привязывает только задание `submitter == "mcp"` без привязки и с
  `delivery: null`, один раз. Иначе — отказ `not_bindable`. Он пишет `binding.session_id`,
  `binding.profile`, `binding.connection` и `bound_at`. `identity_hash` не меняется: повтор
  `speech_job_submit` с тем же `request_id` и запросом по-прежнему даёт `reused: true`.
- MCP-инструмента для привязки нет.
- Без плагина в процессе Hermes или вне локального чата привязки нет, и заметка не приходит.
  Агент читает результат через `speech_job_status` и `speech_job_result`.

## Что Hermes даёт MCP-серверу

У stdio MCP-сервера нет доверенного источника id сессии:

1. **Окружение stdio-процесса.** `tools/mcp_tool_config.py:199–244` `_build_safe_env`: ключи
   `_SAFE_ENV_KEYS` (`:147`), `XDG_*`, секреты внешнего источника, `HERMES_KANBAN_*`, `HERMES_HOME`
   (`:234`) и `env` из конфигурации. Переменных сессии нет. Подстановки `${VAR}` (`:571–594`)
   вычисляются один раз, при загрузке конфигурации (`:766`).
2. **Один процесс на все сессии.** Соединения хранятся в `_servers` (`tools/mcp_tool.py:511`) по
   имени сервера или по паре (профиль, имя) (`tools/mcp_tool_scope.py:1–15`, `:26–31`). Ключа по
   сессии нет: gateway обслуживает все чаты профиля одним процессом `speech-review`.
3. **`initialize`.** Только `clientInfo` (`tools/mcp_tool_transport.py:191–195`), roots не
   объявляются.
4. **`tools/call`.** `call_tool(tool_name, arguments=args, **extra)`
   (`tools/mcp_tool_handlers.py:623`); в `extra` только `progress_callback` (`:620–622`). Id сессии
   в `_meta` нет.
5. **`identity_header`.** Только HTTP/SSE, значение `static` или `profile`
   (`tools/mcp_tool_errors.py:251–275`); для stdio игнорируется (`tools/mcp_tool_transport.py:379–381`).
6. **`ctx.register_session_env`** (`hermes_cli/plugins_session_env.py:1–7`) доходит только до
   дочерних процессов терминала и `execute_code`, не до MCP-процессов.

## Что Hermes даёт хуку после вызова

- `post_tool_call` срабатывает после каждого вызова с аргументами `tool_name`, `args`, `result`,
  `status`, `error_type`, `session_id` и другими id (`model_tools.py:700–724`; документация
  `website/docs/user-guide/features/hooks.md:456`, `:633–650`). Возврат хука игнорируется.
- `status` вычисляется из результата: JSON с ключом `error` даёт `"error"`
  (`model_tools.py:682–697`). Отказ Sound Scout (`isError`) Hermes превращает в `tool_error`
  (`tools/mcp_tool_handlers.py:801–802`). Поэтому привязка ставится только при `status == "ok"`.
- Результат MCP-вызова без `structuredContent` и `_meta` приходит строкой
  `json.dumps({"result": <текст>})` (`tools/mcp_tool_handlers.py:803–816`). Sound Scout отвечает
  одним `TextContent` с JSON (`mcp_server.result`). `job_id` в нём пишет `jobs.submit`, не модель;
  хук принимает только `job_id` из 32 hex-символов.
- Хук вызывается с реальным именем инструмента и при вызове через мост Tool Search `tool_call`
  (`model_tools.py:942–963`).
- `post_tool_call` — ограниченный по времени хук (`hermes_cli/plugins_dispatch.py:42–43`). Он
  исполняется в `contextvars.copy_context()` хода (`:388`). Таймаут по умолчанию 30 с; при таймауте
  хук пропускается и не блокирует (`hooks.md:396`). Ошибка или таймаут значат только, что
  привязки нет; задание остаётся.
- Имя инструмента — `mcp__<сервер>__<инструмент>` (`tools/mcp_tool_schema.py:151–185`), поэтому
  подходит любое имя сервера в `mcp_servers`. Имя длиннее 64 символов Hermes укорачивает хешем;
  такое имя не совпадёт, и привязки не будет.

## Источник: contextvar хода, `chat_id`, а не `session_id`

- Контекст сессии хранится в contextvar на ход, а не в `os.environ`. Параллельные чаты одного
  процесса его не перезаписывают (`gateway/session_context.py:34–47`, `:159–201`,
  `get_session_env` `:230–235`). Gateway кладёт туда `platform` и `chat_id` источника хода
  (`gateway/run.py:3888–3918`). У локального чата `chat_id` — id этого чата
  (`gateway/session_local_recovery.py:44`).
- Аргумент хука `session_id` не используется: это физический сегмент агента, после компакции он
  меняется. `chat_id` локального чата стабилен, и именно его принимает `deliver_to_local_chat`
  (`gateway/session_cron.py:65–80`), через которую шлёт цикл доставки (#102).
- Прецедент: cron берёт место доставки из `HERMES_SESSION_PLATFORM`/`HERMES_SESSION_CHAT_ID`
  внутри вызова инструмента (`cron/job_args.py:59–100`) и для `local` вызывает
  `deliver_to_local_chat` (`cron/scheduler_delivery.py:1854–1858`).
- Хук проверяет id чата: 1–128 символов, только печатаемые. Иначе привязки нет.
- Профиль — `delivery.hermes_profile()`, та же функция, по которой цикл доставки выбирает задания
  своего профиля. `connection` не передаётся: у MCP-хода нет `connectionId` Desktop.

## Варианты

1. **Без привязки.** Документировать чтение через `speech_job_status`/`speech_job_result`.
   #114 просил заметку в чат, поставивший задание, а доверенный источник есть. Отклонён.
2. **Переменная окружения MCP-процесса** (`env` в `mcp_servers` или `${HERMES_SESSION_ID}`).
   Процесс один на все сессии, значение вычисляется один раз: заметка ушла бы в чужой чат.
   Отклонён.
3. **`_meta`, `initialize`, roots, `identity_header`.** Для stdio Hermes их не даёт; пришлось бы
   менять Hermes. Отклонён.
4. **`pre_tool_call` с `modify`** (первый Decision). Без загруженного плагина значения модели
   прошли бы в задание. Нарушает правило «никогда из текста модели». Отклонён уточнением.
5. **`pre_tool_call` с HMAC-доказательством.** Ключ 0600, его выдача оператором, криптопроверка на
   сервере, поле доказательства в схеме. Больше кода и шагов владельца при том же источнике
   доверия. Отклонён.
6. **Схема без полей привязки, `post_tool_call` и `jobs bind`.** MCP-путь не принимает привязку
   вообще, её пишет только процесс Hermes через CLI. **Выбран.**

## Гонка: задание закончилось до `bind`

Хук работает после ответа `speech_job_submit`, а runner может закончить задание раньше. Такое
задание ждёт: `delivery.next_delivery` выдаёт только задания с `session_id`. `jobs bind` разрешён
в любом состоянии задания; после него следующий тик цикла доставки шлёт одну заметку. Тест —
`tests/test_hermes_binding.py::test_a_job_that_ends_before_its_bind_waits_and_then_gets_one_note`.

## Вне угрозы: CLI

Агент с терминалом может сам запустить `sound-scout jobs submit --session-id …` или
`sound-scout jobs bind`. CLI — вход оператора, как и сегодня `jobs submit --session-id`. Правило
этого решения касается MCP-пути: аргумент инструмента никогда не становится привязкой.

## Остаточный риск

- Если плагин `sound-scout` не загружен в процессе хода, привязки нет и заметка не приходит.
  Значение модели при этом в задание не попадает: схема его не принимает.
- `get_session_env` при не установленном contextvar читает `os.environ` процесса Hermes. В
  процессе gateway переменных сессии в окружении нет; иначе это настройка оператора, а не текст
  модели.
- Привязка ставится только для локальных чатов (`platform == "local"`): цикл доставки шлёт только
  через `deliver_to_local_chat` (#102). Ход из Telegram и других платформ ставит задание без
  привязки.
- `sound-scout jobs bind` — отдельный процесс; хук ждёт его до 30 с в потоке хука Hermes.

## Проверки

Тесты без живого Hermes, аудио синтезируется:

- `tests/test_mcp_server.py`: `session_id`, `profile` или `connection` в `speech_job_submit` —
  отказ `invalid_arguments`, каталога заданий нет; задание из MCP без привязки; инструмента
  привязки нет.
- `tests/test_job_bind.py`: `jobs.bind` и `sound-scout jobs bind` — успех, `identity_hash` и
  `reused`, второй `bind`, другой `submitter`, `delivery` задан или отсутствует, недопустимые поля,
  неизвестное задание.
- `tests/test_hermes_binding.py`: хук и его отказы; сквозные сценарии с настоящими MCP-сервером и
  CLI (`jobs bind`, `deliver-next`, `deliver-end`): одна заметка в чат A и ни одной в B, вторая
  заметка в B от хода в B, ход вне локального чата, гонка, без хука, повторный хук из B. Вызов
  Hermes в этих тестах подставлен (`hermes_call`): это контрактный тест по прочитанному коду
  Hermes, а не проверка живого Hermes. Runner подставлен сменой состояния задания, `send` — подделка.

## Что не проверено вживую

Живая проверка — шаг владельца. Нужно разрешение:

1. обновить checkout, из которого работают `mcp_servers.speech-review` и `SOUND_SCOUT_BIN`, до
   коммита этой задачи;
2. скопировать `hermes/plugin/sound-scout/` в `~/.hermes/plugins/sound-scout` и
   `hermes/skills/speech-review/` в `~/.hermes/skills/media/speech-review`;
3. перезапустить `hermes-gateway.service`: это загрузит хук, и gateway заново запустит MCP-сервер
   с новой схемой. Перед этим — проверка из 0017, раздел «Проверка перед любым перезапуском»,
   копии старых файлов и план отката (0017, «Перезапуск и откат»);
4. проверить: `speech_job_submit` из локального чата Hermes даёт одну заметку в этом чате и ни
   одной в другом; `speech_job_submit` с `session_id` даёт отказ `invalid_arguments`.
