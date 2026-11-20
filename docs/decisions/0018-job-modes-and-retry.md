# 0018 — Режимы плана задания и явный повтор

- **Статус:** принято, реализовано.
- **Дата:** 2026-11-20.
- **Решение:** [Decision в #107](https://github.com/renidere/sound-scout-plugin/issues/107#issuecomment-7028158834).
- **Задача:** [#107](https://github.com/renidere/sound-scout-plugin/issues/107); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92).
- **Основание:** `docs/decisions/0013-voice-note-integration.md` § 3.6 (план может только сужать
  бюджеты оператора, не расширяет корни, сеть, лимиты или backend; `jobs recover` ставит видимую
  новую попытку без скрытых повторов), § 3.1 (CLI, MCP и плагин — равноправные входы);
  `docs/spec/REQUIREMENTS.md` R09, R10 (границы и лимиты задаёт оператор).
- **Код:** `src/sound_scout/jobs.py` (`resolved_plan`, `applied_plan`, `retry`), входы —
  `src/sound_scout/cli.py` (`jobs submit --mode`, `jobs retry`), `src/sound_scout/mcp_server.py`
  (`speech_job_submit.mode`, `speech_job_retry`), `hermes/plugin/sound-scout/dashboard/plugin_api.py`
  (query `mode` у `POST /jobs`).

## Режимы

Режим — поле запроса `mode`, по умолчанию `auto`. Одна семантика у всех входов: CLI
`jobs submit --mode`, MCP `speech_job_submit.mode`, маршрут плагина `POST /jobs?mode=` (плагин
передаёт значение в CLI как `--mode=<значение>`, проверяет его CLI). Синхронные `speech_analyze` и
`sound-scout analyze` режимов не получают и остаются на `layers` (совместимость и eval, 0013 § 3.1).

| режим | слои | стадии |
|---|---|---|
| `quick` | `content`, `measurements` | `prepare`, `acoustics`, `asr` |
| `targeted` | названные вызывающим `layers` | `prepare` и стадии этих слоёв |
| `full` | все три | `prepare`, `acoustics`, `asr`, `audio_review` |
| `auto` | `targeted`, если `layers` названы, иначе `full` | как у выбранного режима |

- `layers` вместе с `quick` или `full`, `targeted` без `layers`, неизвестный режим — отказ
  `invalid_request` до записи задания.
- `auto` читает только запрос: названы ли слои. Правило и причина выбора пишутся в план
  (`resolved_mode`, `resolution`: `auto: layers named` / `auto: no layers named`). От
  `config.backend` `auto` не зависит: при `backend = none` честнее `audio_review: unavailable`
  («backend = none»), чем «слой не запрошен», когда пользователь от звучания не отказывался.
  Длительность входа `auto` не читает: `submit` не запускает ffprobe.
- `targeted` — слои, а не выборочные диапазоны для audio backend: единственный backend сейчас
  `none`, выборочное прослушивание проверялось бы только на подмене. Если оно появится, это будет
  `capability_version = 2`.
- `full` запускает только backend из конфигурации оператора; сейчас `KNOWN_BACKENDS = ("none",)`,
  поэтому `audio_review` — `unavailable` с причиной `backend = none`. В `jobs.py` нет пути, который
  строит модель или передаёт `backend` в `app.analyze` (проверяет AST-тест).

## План в `job.json`

`job.json["plan"]` содержит ровно поля `capability_version`, `mode`, `resolved_mode`,
`resolution`, `stages`, `budgets`, `backend`. `capability_version = 1` — версия таблицы «режим →
слои → стадии» и набора полей плана. Корней, сети, data root и модели в плане нет. Режим бюджеты
не меняет: в плане — бюджеты оператора на момент `submit`. В `job.json["request"]` пишутся
запрошенный `mode` и разрешённые `layers`; оба входят в идентичность задания.

Runner план не пересчитывает: перед `app.analyze` `applied_plan(plan, config)` сверяет сохранённый
план с конфигурацией на момент запуска и отказывает (`plan_refused`, задание `failed` с этой
причиной, `app.analyze` не вызывается):

1. `capability_version` не `1` (в том числе задание, поставленное до этой версии, без поля):
   «submit the recording again». Совместимости со старым форматом нет, как у `delivery` в #102.
2. лишний или недостающий ключ (`allowed_roots`, `roots`, `network`, `data_root`, `model`, …);
3. `mode` или `resolved_mode` вне таблицы;
4. `stages` — не стадии какого-либо набора слоёв в порядке `WORKER_STAGES` или не стадии слоёв
   `quick`/`full`;
5. `backend` не равен `config.backend`;
6. `budgets` — не ровно поля `config.Budgets` или значение не целое ≥ 1 (`bool` не принимается).

Действующий бюджет — `min(план, конфигурация)` по каждому полю: оператор сузил конфигурацию после
`submit` — работает новое значение; расширить её план не может. Корни, сеть, data root, ASR и
backend всегда берутся из конфигурации. Вход `app.analyze` проверяет по текущим `allowed_roots`,
поэтому сужение корней после `submit` даёт `failed` с `path_denied`.

## Повтор

`jobs.retry(job_id)` (CLI `jobs retry`, MCP `speech_job_retry`) — только по явному вызову клиента;
сам runner и `recover` ничего не повторяют.

- Принимает задание в `failed` или `cancelled`. Другое состояние — `not_retryable`, ничего не
  записано; у `running` в сообщении подсказка: если runner пропал, сначала `jobs recover`.
- Под `jobs.lock`: admission как у `submit` (`queue_full`, `staging_full`), снимается `cancel.json`
  (иначе наблюдатель сразу отменил бы новую попытку), в `retries[]` записывается
  `{requested_at, after_state, after_reason, next_attempt}`, задание — `queued` с новым
  `queued_at`, то есть в конце очереди. `job_id`, `request_id`, `request`, `plan`, `binding`,
  `delivery` и индекс не меняются: новых id повтор не придумывает.
- Следующую попытку `<job_id>-a<n+1>` создаёт существующий `_claim_next`. Очередь упорядочена по
  `queued_at` (у старых заданий без поля — `submitted_at`); `Busy` и `recover` `queued_at` не
  меняют и место в очереди не отнимают.
- `status` показывает `mode`, `resolved_mode` и число `retries`.

### Прерванные задания

Отдельного терминального состояния `interrupted` нет: `interrupted` — исход попытки, а не
состояние задания (`STATES` не меняются, на них опираются плагин, доставка и skill). Прерванная
попытка появляется только через `jobs recover`, и задание после него оказывается в одном из
состояний:

- `queued` — `recover` по явной команде оператора уже поставил видимую попытку n+1 (0013 § 3.6);
  повтор не нужен, `retry` ответит `not_retryable`;
- `cancelled` — у прерванной попытки был маркер отмены; `retry` принимает такое задание и
  ставит попытку n+1 (тест `test_retry_of_a_cancelled_job_drops_its_cancel_marker_so_the_new_attempt_runs[interrupted]`);
- если поставленная `recover` попытка затем закончилась `failed` или `cancelled`, `retry`
  принимает задание как обычное.

Задание, чья попытка ещё числится `running` после гибели runner, повторить нельзя, пока `jobs
recover` не пометит её `interrupted`: иначе две попытки одного задания могли бы идти одновременно.

### Повтор и доставка

`delivery` при повторе не меняется. По #102 итоговая заметка уходит не больше одного раза на
задание, а её тег сделан по `job_id`: вторая заметка с тем же тегом сломала бы проверку по истории.
Если итог ещё не доставлен, придёт итог последней попытки; если уже доставлен, новый итог виден
через `jobs status/result`. Кнопка и маршрут повтора в плагине — #108.
