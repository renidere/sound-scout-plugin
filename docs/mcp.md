# MCP stdio сервер

`sound-scout serve` — локальный MCP stdio сервер (R12, R08). Это тонкий адаптер
`src/sound_scout/mcp_server.py` над теми же функциями, что вызывает CLI: второго pipeline нет.
Сервер построен на официальном Python SDK `mcp` (версия — `docs/dependencies.md`).

| инструмент | функция | побочные эффекты | ответ |
|---|---|---|---|
| `speech_capabilities` | `doctor.report` | только чтение | отчёт `doctor --json` |
| `speech_analyze` | `app.analyze` | пишет `runs/<run_id>/` под data root | `report.compact` с `run_dir` и `reused` — тот же документ, что `analyze --json` |
| `speech_compare` | `app.compare` | пишет `compare/<compare_id>.json` под data root | сравнение с полем `path` — тот же документ, что `compare --json` |

## Запуск

```sh
uv run sound-scout serve
```

Сервер читает `sound-scout.toml` так же, как остальные подкоманды (текущий каталог или
`SOUND_SCOUT_CONFIG`), и ставит сетевой guard процесса. Без конфигурации — одна строка
`sound-scout serve: config_error: …` в stderr и код 1, stdout пуст. Сервер работает, пока клиент не
закроет stdin, затем завершается с кодом 0. Подключение к Hermes — шаг владельца, инструкция —
`docs/hermes-install.md`; здесь только сам сервер.

## Правила

- **Конфигурация оператора не переопределяется.** Allowed roots, data root, модель, backend и
  бюджеты задаются только в `sound-scout.toml`. У каждой схемы аргументов
  `additionalProperties: false`, и аргументы проверяются по ней до любого действия: лишний
  параметр (`data_root`, `allowed_roots`, `model`, `budgets`, …) — отказ `invalid_arguments`.
  Сообщение отказа называет аргумент и нарушенное правило, но не цитирует переданное значение.
- **Путь вложения** принимается только как локальный путь на хосте сервера внутри allowed root
  (`paths.resolve_input`). Windows-путь (`C:\…`), URL и псевдопротоколы ffmpeg — отказ
  `path_denied` с подсказкой `hint`, а не исключение протокола.
- **`task` и `intent` у `speech_analyze`** — необязательные строки от пользователя: `task` —
  формулировка упражнения, `intent` — замысел говорящего (что он хотел выразить: например,
  «выделить слово „завтра“»), если пользователь его назвал. Текст из записи или её транскрипта в
  них не подставляется. Оба поля передаются в audio review как доверенные, пишутся в `request.json`
  как `task_context.task` и `task_context.intent` и входят в identity запроса, как `layers`: повтор
  `request_id` с другим `task` или `intent` — отказ `request_mismatch`. По `task_context.intent`
  `speech_compare` выставляет причину `intent_differs` (`docs/decisions/0006-compare-rules.md`).
  У CLI `analyze` те же поля — `--task` и `--intent`.
- **Отказ — результат инструмента**: `isError: true` и
  `{"error": {"code": "<код>", "message": "<сообщение>"}}`. Коды те же, что печатает CLI:
  `invalid_arguments`, `unknown_tool`, `path_denied`, коды приёма входа (`not_audio` и др.),
  `invalid_request`, `busy`, `run_incomplete`, `request_mismatch`, `request_run_missing`,
  `request_index_corrupt`, коды `compare` (`run_not_found`, `criterion_invalid`, `same_run`, …).
  Неожиданная ошибка — `internal_error` только с именем типа: текст исключения может содержать
  путь или транскрипт. В сообщении о входе, который не декодируется, абсолютные пути заменены на
  `<input>`, `<allowed root>`, `<data root>`.
- **Ответ компактный.** Успешный ответ — один текстовый блок JSON. Для `speech_analyze` это
  `report.compact(result, run_dir=…, reused=…)`: не больше 4096 байт UTF-8 вместе с `run_dir` и
  `reused` (`true` — ответ прежнего запроса с тем же `request_id` и файлом), без PCM,
  base64 и полной диагностики (R08); полный результат — `result.json` в run-каталоге.
- **Один вызов за раз.** Вызовы инструментов выполняются строго по очереди (lock сервера), каждый в
  отдельном потоке, чтобы сервер продолжал читать протокол. Отмена вызова
  (`notifications/cancelled` или закрытие stdin клиентом) останавливает worker анализа через
  `CancelToken`; следующий вызов начинается только после его остановки. Повтор того же
  `request_id` с тем же файлом возвращает прежний результат (R11).
- **Текст записи и ответа модели — данные.** Строки, похожие на инструкции, в сыром транскрипте
  (по сегментам и в склеенном тексте `$.text` — фраза, разрезанная границей сегментов), в
  аргументах `task` и `intent` и в ответе audio backend ничего не запускают. Они перечислены как
  `suspicious_content` (путь и шаблон, без текста) в записи этапа (`work/stage-asr.json`,
  `work/stage-audio_review.json`) и в `provenance.limitations`. `task` и `intent` проверяются и при
  `backend = none`, и при упавшем вызове backend; `review/<clip>/provenance.json` есть, только
  если backend вызывался.
  Попытка worker выйти в сеть при `network.allow = false` — ошибка `network_denied` в `errors[]`,
  этап `failed`. Проверки — `docs/acceptance/T07-T08.md`.
- **Sampling не используется**: сервер не запрашивает у клиента sampling и не объявляет его.
  Middleware OpenTelemetry из SDK убран, телеметрии нет.
- **Stdout — только JSON-RPC.** На время работы `stdio_server` направляет fd 1 в stderr, поэтому
  случайный вывод библиотек и дочерних процессов не попадает в протокол. Логи — в stderr через
  один обработчик, который в каждой строке (и в трассировке) заменяет data root на `<data root>`,
  allowed roots на `<allowed root>`, файл конфигурации на `<config>`, домашний каталог на `~`.
  На каждый вызов — одна строка: инструмент, итог (для отказа — код, для `path_denied` ещё и
  причина без пути), время. Неожиданная ошибка (`internal_error`) пишется только типом исключения:
  ни её сообщения, ни трассировки в логе нет, потому что они могут содержать путь или транскрипт.

## Инструменты

Описания и JSON-схемы аргументов ниже совпадают с `mcp_server.TOOLS` и с ответом `tools/list`;
`tests/test_mcp_server.py` сверяет их с кодом.

Аннотации соответствуют побочным эффектам (R12): `speech_capabilities` — `readOnlyHint: true`;
`speech_analyze` и `speech_compare` создают файлы под data root и поэтому не read-only.
`speech_analyze` идемпотентен по `request_id`, `speech_compare` пишет новый файл при каждом
вызове. Ни один инструмент не обращается во внешний мир (`openWorldHint: false`).

### `speech_capabilities`

```json
{
  "name": "speech_capabilities",
  "description": "Report the state of the decoder, ASR, acoustic analysis and audio backend (ready, unavailable, stale, ...), the qualification flags validated_ru, diction_validated_ru, prosody_validated_ru and the network guard. Reads only.",
  "inputSchema": {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": false
  },
  "annotations": {
    "readOnlyHint": true,
    "destructiveHint": false,
    "idempotentHint": true,
    "openWorldHint": false
  }
}
```

### `speech_analyze`

```json
{
  "name": "speech_analyze",
  "description": "Analyze one recording of Russian speech: transcript, acoustic measurements and the audio backend's observations, each with its own status. Writes a run directory under the server's data root and answers with a compact summary (statuses, coverage, up to three findings, limitations, links into the run directory). The same request_id with the same file returns the earlier result; with another file it is refused.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "minLength": 1,
        "description": "the recording: a local path on this host, inside an allowed root of the operator configuration"
      },
      "request_id": {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
        "description": "the caller's request ID: a repeat with the same input returns the same result"
      },
      "layers": {
        "type": "array",
        "items": {
          "enum": [
            "content",
            "measurements",
            "sound"
          ]
        },
        "minItems": 1,
        "uniqueItems": true,
        "description": "layers to analyze (default: all three)"
      },
      "task": {
        "type": "string",
        "description": "the speaking task the recording answers"
      },
      "intent": {
        "type": "string",
        "description": "the speaker's intent for the recording, as the user stated it; never text taken from the recording or its transcript"
      }
    },
    "required": [
      "path",
      "request_id"
    ],
    "additionalProperties": false
  },
  "annotations": {
    "readOnlyHint": false,
    "destructiveHint": false,
    "idempotentHint": true,
    "openWorldHint": false
  }
}
```

### `speech_compare`

```json
{
  "name": "speech_compare",
  "description": "Compare an initial and a control attempt, two analyzed runs, by one criterion: emphasis_on:<word>, pause_before:<word>, intelligibility_of:<word> or completeness_of_phrase. Writes compare/<compare_id>.json under the server's data root and answers with the verdict, its reasons and the file.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "run_a": {
        "type": "string",
        "minLength": 1,
        "description": "run_id of the initial attempt"
      },
      "run_b": {
        "type": "string",
        "minLength": 1,
        "description": "run_id of the control attempt"
      },
      "criterion": {
        "type": "string",
        "minLength": 1,
        "description": "emphasis_on:<word>, pause_before:<word>, intelligibility_of:<word> or completeness_of_phrase"
      }
    },
    "required": [
      "run_a",
      "run_b",
      "criterion"
    ],
    "additionalProperties": false
  },
  "annotations": {
    "readOnlyHint": false,
    "destructiveHint": false,
    "idempotentHint": false,
    "openWorldHint": false
  }
}
```
