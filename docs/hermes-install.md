# Подключение Sound Scout к Hermes

Инструкция для владельца: как установить Sound Scout на хосте backend и зарегистрировать MCP
stdio сервер `sound-scout serve` в Hermes под именем `speech-review` (R12, T09b). Сам сервер и его
инструменты описаны в `docs/mcp.md`.

**Подключение к действующему Hermes — шаг владельца по отдельному разрешению** (issue #40).
Агенты эти команды не выполняют: всё, что ниже меняет конфигурацию Hermes или подключается к нему
(`hermes mcp add`, `hermes config set`, `hermes mcp test`, новая сессия), запускает владелец.

## Источники

Инструкция сверена 2026-11-19 с установленным Hermes Agent **v0.23.1** (`hermes --version`:
`2026.9.24`, коммит `3b42157858`) — по документации и исходникам в его каталоге установки
(`~/.hermes/hermes-agent`), только чтение:

- **H1** — руководство MCP, `website/docs/user-guide/features/mcp.md`
  (https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp);
- **H2** — справочник конфигурации MCP, `website/docs/reference/mcp-config-reference.md`
  (https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference);
- справка CLI той же версии: `hermes mcp add --help`, `hermes mcp configure --help`,
  `hermes config set --help`, `hermes config get --help`;
- исходники Hermes там, где документация молчит или расходится с кодом (ссылки ниже).

Цитаты:

- H2, таблица ключей: «`timeout` | number | both | Tool call timeout in seconds (default: `300`)»,
  «`connect_timeout` | number | both | Initial connection timeout in seconds (default: `60`)»,
  «`supports_parallel_tool_calls` | bool | both | Allow tools from this server to run
  concurrently».
- H1, «MCP Sampling Support»: «Sampling is **enabled by default** for all MCP servers (when the MCP
  SDK supports it)»; отключение — `sampling: enabled: false`.
- H1, «Common keys»: «`cwd` | string | Working directory for the stdio server process. Default: the
  session working directory when one is pinned (ACP/gateway sessions, `terminal.cwd`), else the
  Hermes process directory».
- H2, «`include`»: «If `include` is set, only those server-native MCP tools are registered.»
- H1, «Stdio env filtering»: «Only explicitly configured `env` plus a safe baseline are passed
  through.»
- `hermes mcp add --help`: «`--args ...` Arguments for stdio command; must be the last option».

Расхождения, найденные при сверке:

- H1 называет зарегистрированные инструменты `mcp_<server>_<tool>`, а H2 и код
  (`tools/mcp_tool_schema.py`, `MCP_TOOL_NAME_PREFIX = "mcp__"`) — `mcp__<server>__<tool>`; дефис
  в имени сервера заменяется на `_`. В этой версии инструменты называются
  `mcp__speech_review__speech_capabilities`, `mcp__speech_review__speech_analyze`,
  `mcp__speech_review__speech_compare`.
- `hermes mcp add` в v0.23.1 знает только `--command`, `--args`, `--env`, `--connect-timeout`
  (и HTTP/preset-флаги): `cwd`, `timeout`, `sampling`, `supports_parallel_tool_calls` и allowlist
  задаются потом через `hermes config set mcp_servers.<имя>.…`.
- `--args` забирает всё до конца строки, поэтому `--connect-timeout` и `--env` стоят **до**
  `--args`: в порядке `--args run sound-scout serve --connect-timeout 60` таймаут стал бы
  аргументом сервера.
- `hermes mcp configure` при выборе всех инструментов **удаляет** `tools` из записи
  (`hermes_cli/mcp_config.py`, `cmd_mcp_configure`), то есть явный allowlist пропадает. Для
  allowlist используйте `hermes config set`, а не `configure`.

## Установка на хосте backend

Команды выполняются на той машине, где работает Hermes gateway и куда он сохраняет вложения, от
того же пользователя ОС: каталог `~/.hermes/cache/audio` создаётся с правами 0700.

1. Клонировать репозиторий и поставить окружение из lock-файла:

   ```sh
   git clone https://github.com/renidere/sound-scout-plugin.git
   cd sound-scout-plugin
   uv sync --frozen
   ```

   Системные программы, модели и их ревизии — `docs/dependencies.md`. Модель ASR должна уже лежать
   в локальном кэше Hugging Face: Sound Scout ничего не скачивает (R09).

2. Создать приватный data root (вне `~/.hermes` и вне allowed roots) и файл конфигурации из
   примера:

   ```sh
   mkdir -m 700 <абсолютный путь к data root>
   cp sound-scout.example.toml sound-scout.toml
   ```

   В `sound-scout.toml` (он в `.gitignore`, в репозиторий не попадает):

   ```toml
   allowed_roots = ["<домашний каталог>/.hermes/cache/audio"]
   data_root = "<абсолютный путь к data root>"
   ```

   Правило корня (`src/sound_scout/config.py`, решение владельца A, issue #66):

   - путь только абсолютный и существующий: `~` в `sound-scout.toml` не раскрывается, пишите
     домашний каталог полностью;
   - из каталогов Hermes корнем может быть только медиакэш: `<домашний каталог>/.hermes/cache/audio`,
     legacy `<домашний каталог>/.hermes/audio_cache` или те же каталоги профиля
     `<домашний каталог>/.hermes/profiles/<профиль>/cache/audio` и
     `<домашний каталог>/.hermes/profiles/<профиль>/audio_cache`. Legacy `audio_cache` Hermes
     использует, только если этот каталог не пуст (`hermes_constants.py`, `get_hermes_dir`);
   - в пути ровно один компонент `.hermes`, и часть пути от `.hermes` вниз записана так же, как
     настоящий путь: symlink выше `.hermes` допустим, symlink, скрывающий `.hermes` под другим
     именем, отклоняется;
   - сам `~/.hermes`, `attachments`, `sessions`, `cache` и другие его подкаталоги корнем не
     бывают; ниже корня запрещённые имена (`.env*`, `.ssh`, `sound-scout.toml`, плейлисты и др.)
     отклоняются в любом месте пути;
   - data root и allowed root не вложены друг в друга.

   Бюджеты (`[budgets]`) оставьте по умолчанию или задайте явно; от них считается `timeout` ниже.

3. Проверить установку:

   ```sh
   SOUND_SCOUT_CONFIG=<абсолютный путь к sound-scout.toml> uv run --offline sound-scout doctor
   ```

   `doctor` ничего не скачивает и при любых статусах компонентов завершается с кодом 0: это
   диагноз, а не проверка. Ошибка конфигурации — отказ: одна строка
   `sound-scout doctor: config_error: …` в stderr и код 1. Без smoke-receipts компоненты
   `decoder`, `asr`, `acoustics` показаны `unvalidated`. `uv run --offline sound-scout smoke` пишет
   receipts только `decoder` и `acoustics` (на синтетическом сигнале); `asr` остаётся
   `unvalidated`, пока не выполнен `uv run --offline sound-scout smoke asr` с переменной
   `SOUND_SCOUT_DEV_AUDIO=<абсолютный путь к записи>`. Пока audio backend не выбран (`backend = "none"`, Gate 0),
   `audio_review` — `permission_required`, а анализ отвечает `partial` без слоя звучания.

4. Узнать абсолютный путь к `uv` (`command -v uv`) — он нужен в `--command`: Hermes передаёт серверу
   только безопасный базовый набор переменных окружения, и `PATH` сессии может отличаться.

## Ограничения вложений Hermes

Hermes не передаёт файлы по MCP: `speech_analyze` получает только строку абсолютного пути к файлу
на этом хосте. Мессенджеры Hermes сохраняют входящее аудио в `~/.hermes/cache/audio`
(`audio_<12 hex>.<ext>`), и модель видит этот путь. Ограничения (проверены по исходникам v0.23.1):

1. **Голосовое сообщение при `stt.enabled: true`** (умолчание Hermes) доходит до модели только
   расшифровкой, без пути к файлу (`gateway/run_inbound_transcription.py`), и в `speech_analyze`
   его передать нельзя. Запись нужно присылать **аудиофайлом или документом**: тогда модель
   получает «It is saved at: <путь>»
   (`gateway/run_inbound.py`, `_prepend_inbound_media_file_notes`).
2. **Срок 24 часа**: gateway удаляет медиакэш старше суток
   (`agent/provider_media.py`, `MEDIA_CACHE_MAX_AGE_HOURS = 24`). Анализ — в этот срок; после
   приёма входа run-каталог в data root самодостаточен.
3. **Telegram — до 20 МБ**: Bot API скачивает вложения только до 20 МБ
   (`plugins/platforms/telegram/adapter.py`, `_max_doc_bytes`). Несжатый WAV 44,1 кГц/16 бит/стерео
   превышает предел примерно через 2 минуты — присылайте ogg/mp3/flac.
4. **Нужен `terminal.backend: local`**: при docker/ssh и других удалённых backend модель видит путь,
   переведённый в контейнер или на удалённый хост (`tools/credential_files.py`,
   `to_agent_visible_cache_path`), MCP-сервер обратно его не переводит, и вызов получает
   `path_denied`. Проверить: `hermes config get terminal.backend`.
5. **Однопользовательский gateway**: `cache/audio` — общий пул всех платформ и собеседников (туда же
   пишут TTS и аудиоответы других MCP-серверов). Схема рассчитана на gateway одного пользователя.

Для CLI и Desktop без мессенджера записи можно класть в отдельный каталог и добавить его в
`allowed_roots` (вариант F из комментария владельца к issue #39).

## Регистрация в Hermes

Выполняет владелец, из любого каталога:

```sh
hermes mcp add speech-review \
  --command <абсолютный путь к uv> \
  --connect-timeout 60 \
  --env SOUND_SCOUT_CONFIG=<абсолютный путь к sound-scout.toml> \
  --args run --directory <абсолютный путь к checkout sound-scout-plugin> --offline sound-scout serve
```

Параметры:

- `speech-review` — имя сервера, ключ `mcp_servers.speech-review`; инструменты регистрируются как
  `mcp__speech_review__<инструмент>`.
- `--command <абсолютный путь к uv>` — исполняемый файл stdio-сервера; абсолютный путь не зависит
  от `PATH` процесса Hermes.
- `--connect-timeout 60` — сколько секунд Hermes ждёт запуска сервера, MCP `initialize` и списка
  инструментов. Сервер стартует за секунды (модели при старте не загружаются); 60 с — явное
  значение вместо умолчания.
- `--env SOUND_SCOUT_CONFIG=…` — путь к конфигурации оператора. Без него `sound-scout` ищет
  `sound-scout.toml` в текущем каталоге процесса; `uv run --directory <checkout>` переходит в
  checkout до запуска сервера, так что текущим каталогом будет checkout, каким бы ни был `cwd`
  сессии Hermes. Явный `SOUND_SCOUT_CONFIG` не зависит от того, где лежит toml-файл, и его можно
  держать вне checkout. Если модель ASR лежит не в `~/.cache/huggingface/hub`, добавьте сюда же
  `HF_HUB_CACHE=<абсолютный путь>`: остальные переменные окружения Hermes серверу не передаёт.
- `--args` — последний параметр, всё после него уходит серверу:
  - `run --directory <checkout>` — `uv` находит проект по каталогу, а не по текущему: вне
    checkout `uv run sound-scout` падает с `Failed to spawn: sound-scout`. Поэтому отдельный `cwd`
    не нужен ни для пробного подключения `hermes mcp add`, ни для работы;
  - `--offline` — `uv` не ходит в сеть, даже если окружение разошлось с lock-файлом (R09);
  - `sound-scout serve` — MCP stdio сервер.

`hermes mcp add` сразу подключается к серверу, показывает три инструмента и спрашивает
`Enable all 3 tools? [Y/n/select]`. Ответ `Y` сохраняет запись без `tools.include`; allowlist
задаётся следующим шагом.

## Настройка записи

Значения, которых нет у `hermes mcp add`, задаются явно — на умолчания не полагаемся (R12):

```sh
hermes config set mcp_servers.speech-review.tools.include '[speech_capabilities, speech_analyze, speech_compare]'
hermes config set mcp_servers.speech-review.sampling.enabled false
hermes config set mcp_servers.speech-review.supports_parallel_tool_calls false
hermes config set mcp_servers.speech-review.timeout 300
```

- `tools.include` — явный allowlist из трёх инструментов R12. `hermes config set` разбирает
  `[…]` как YAML-список. `hermes mcp configure` для этого не используйте: при выборе всех
  инструментов он удаляет `tools` из записи.
- `sampling.enabled false` — Hermes включает sampling по умолчанию; Sound Scout его не запрашивает и
  не объявляет, а отключение на стороне клиента убирает этот канал совсем (R09).
- `supports_parallel_tool_calls false` — сервер выполняет вызовы строго по одному (lock сервера,
  lock data root), параллельные вызовы только ждали бы друг друга или получали `busy`.
- `timeout 300` — таймаут одного вызова инструмента в секундах. Он должен оставлять запас над
  временем `speech_analyze`, иначе Hermes отменит вызов раньше, чем сервер вернёт частичный
  результат с `deadline_exceeded`. Худший случай вызова — проверка входа через ffprobe до запуска
  worker (`budgets.max_subprocess_seconds`, по умолчанию 60 с), затем сам worker
  (`budgets.worker_deadline_seconds`, по умолчанию 180 с, `src/sound_scout/app.py`), остановка
  worker (0,5 с) и сборка результата. Отсюда
  `timeout ≥ worker_deadline_seconds + max_subprocess_seconds + 60 с запаса` = 180 + 60 + 60 = 300.
  Если в `[budgets]` значения другие — пересчитайте и задайте timeout заново.

Запущенный gateway изменения существующей записи сам не применяет: его фоновая сверка с
`config.yaml` (`gateway/run_profile_reconcile.py`, `_mcp_config_reconciler` →
`tools/mcp_tool_discovery.py`, `reconcile_mcp_servers_with_config`) только подключает новые
серверы и отключает удалённые. Сервер `speech-review` она может подключить сразу после
`hermes mcp add`, ещё со старыми значениями, и `tools.include`, `sampling`, `timeout`,
`supports_parallel_tool_calls` останутся прежними. Поэтому после `hermes config set` в gateway
выполните `/reload-mcp` или перезапустите gateway; в `hermes chat` достаточно новой сессии или
`/reload-mcp` (H1, «Reloading»).

## Итоговая запись

Ожидаемая запись — `docs/examples/hermes-mcp-entry.example.yaml`: `command`, `args`,
`env.SOUND_SCOUT_CONFIG`, `connect_timeout: 60.0` (`--connect-timeout` сохраняется числом с
плавающей точкой), `timeout: 300`, `supports_parallel_tool_calls: false`, `sampling.enabled: false`,
`tools.include` из трёх инструментов, `enabled: true`. Пути в угловых скобках у владельца —
настоящие абсолютные пути хоста.

## Readback

После настройки и до заявления «подключено» владелец проверяет три вещи.

1. **Exact entry** — прочитать запись так, как её сохранил Hermes, и сверить с примером поле за
   полем:

   ```sh
   hermes config get mcp_servers.speech-review --json
   hermes mcp list
   ```

   Расхождение (нет `tools.include`, `sampling.enabled` не `false`, другой `timeout`) исправляется
   командами из раздела «Настройка записи», не правкой файла.

2. **Discovery в нужном runtime** — сначала, после того как запись настроена, обязательно
   применить её: в gateway — `/reload-mcp` или перезапуск gateway (новой сессии gateway мало:
   подключение сервера общее для процесса и держит старые значения, см. «Настройка записи»); в
   `hermes chat` — новая сессия или `/reload-mcp`. Затем — в **новой** сессии того runtime, где
   будут вызовы (gateway мессенджера или `hermes chat`). В `hermes chat` команда `/tools list` показывает ровно
   `mcp__speech_review__speech_capabilities`, `mcp__speech_review__speech_analyze`,
   `mcp__speech_review__speech_compare`; в gateway `/tools` недоступна (CLI-only по справочнику
   slash-команд), там discovery подтверждает строка `MCP: registered … tool(s) from … server(s)` в
   `agent.log`, где `speech-review` не назван среди несостоявшихся (`failed`), и реальный вызов
   из пункта 3. `hermes mcp test speech-review` только проверяет, что сервер
   запускается и отвечает, — это не discovery в runtime.

3. **Реальный вызов из новой сессии** — попросить Hermes (не вызывать SDK самому):
   - вызвать `speech_capabilities` — ответ совпадает с
     `uv run --offline sound-scout doctor --json` на том же хосте;
   - прислать короткую запись **аудиофайлом или документом** и попросить проанализировать её через
     `speech_analyze` — ответ содержит `run_id`, `status` и `run_dir` (путь run-каталога
     относительно data root), а в data root появился `<run_dir>/result.json`;
   - для двух таких run — `speech_compare` с критерием, ответ содержит `path` к
     `compare/<compare_id>.json`.

   Записать дату, версию Hermes (`hermes --version`), имя сессии/runtime и `run_id` — это
   доказательство T09b. Содержимое записи и транскрипт в отчёт не копируются.

## Что считается подключённым

Прямой вызов сервера через MCP SDK (свой клиент, `mcp.Client`, тесты `tests/test_mcp_server.py`)
или `hermes mcp test` доказывает, что сервер работает (T09a), но **не равен native tool exposure**:
он не показывает, что модель Hermes видит инструменты в своей сессии и может их вызвать (T09b,
R12). Интеграция считается подключённой, только когда выполнены все пункты:

- [ ] разрешение владельца на подключение к действующему профилю получено (issue #40);
- [ ] запись создана через `hermes mcp add` и `hermes config set`, без ручного редактирования
      `mcp_servers`;
- [ ] readback exact entry совпадает с `docs/examples/hermes-mcp-entry.example.yaml`: allowlist из
      трёх инструментов, `sampling.enabled: false`, `supports_parallel_tool_calls: false`, явные
      `connect_timeout` и `timeout`, `timeout` ≥ worker deadline + `max_subprocess_seconds` + запас;
- [ ] `terminal.backend` — `local`;
- [ ] discovery в новой сессии нужного runtime показывает ровно три инструмента
      `mcp__speech_review__…`;
- [ ] реальный вызов `speech_capabilities` и `speech_analyze` моделью Hermes из новой сессии вернул
      ответ, и в data root есть соответствующий `<run_dir>/result.json`;
- [ ] результат записан с датой, версией Hermes и `run_id`, без содержимого записи.

Пока хотя бы один пункт не выполнен, интеграция не законченная: T09a может быть успешным, T09b —
нет (раздел 5 `docs/spec/REQUIREMENTS.md`, T09).
