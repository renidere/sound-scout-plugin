# 0020 — Контрольная попытка в очереди заданий

- **Статус:** принято, реализовано.
- **Дата:** 2026-11-20.
- **Решение:** [Decision в #110](https://github.com/renidere/sound-scout-plugin/issues/110#issuecomment-7041147825).
- **Задача:** [#110](https://github.com/renidere/sound-scout-plugin/issues/110); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92).
- **Основание:** `docs/spec/REQUIREMENTS.md` R07 (сравнение по заранее указанному критерию), T10;
  `docs/decisions/0006-compare-rules.md` (грамматика критерия, правило вердикта, коды причин);
  `docs/decisions/0013-voice-note-integration.md` § 3.1 (CLI, MCP и плагин — равноправные входы),
  § 3.6 (очередь), § 3.7 (итоговая заметка), таблица § 4 (связь контрольной попытки с заданием).
- **Код:** `src/sound_scout/jobs.py` (`SubmitRequest.previous_run_id`/`criterion`,
  `_checked_control`, `_compared`, `_end_attempt`), `src/sound_scout/delivery.py` (`note_text`,
  `COMPARE_REASON_LABELS`), входы — `src/sound_scout/cli.py` (`jobs submit --previous-run-id
  --criterion`), `src/sound_scout/mcp_server.py` (`speech_job_submit.previous_run_id`/`criterion`),
  `hermes/plugin/sound-scout/dashboard/plugin_api.py` (query `previous_run_id`/`criterion` у
  `POST /jobs`).

## Связь с исходной попыткой

Два поля запроса задания: `previous_run_id` (`run_id` исходной попытки) и `criterion`. Они
передаются только вместе. Одна семантика у всех входов: `jobs.SubmitRequest`, CLI
`jobs submit --previous-run-id --criterion`, MCP `speech_job_submit`, маршрут плагина
`POST /jobs?previous_run_id=&criterion=` (плагин передаёт каждое значение в CLI одним элементом
argv; проверяет CLI). Второго парсера нет.

Связь хранится в `job.json["request"]["control"]` как
`{"previous_run_id", "criterion"}`, критерий — в нормализованной форме (`raw` из
`comparison.parse_criterion`, как в `compare.v1`). У обычного задания `control` — `null`. Связь
входит в identity задания: тот же `request_id` с другим критерием — `request_mismatch`. План и
`capability_version` не меняются: сравнение — не стадия `app.analyze`.

## Проверки при submit

`jobs.submit` проверяет связь до хеширования входа и до записи в store; любой отказ оставляет store
нетронутым.

| условие | код |
|---|---|
| передано одно поле из двух | `invalid_request` |
| критерий вне грамматики `compare` (`comparison.parse_criterion`) | `criterion_invalid` |
| в плане нет слоя `sound` (вердикт берётся только из audio review; `quick` всегда отказ) | `invalid_request` |
| критерий со словом, в плане нет слоя `content` (слово ищется в транскрипте) | `invalid_request` |
| исходного run нет или это не настоящий каталог (`comparison.load_attempt`) | `run_not_found` |
| исходный run не завершён (`manifest.complete`) или без `result.json` | `run_incomplete` |
| `result.json` исходного run не проходит `findings.validate` | `result_invalid` |

`CompareError` превращается в `JobError` с тем же кодом. Отсутствие audio backend — не отказ: это
условие конфигурации, о нём скажет вердикт.

## Сравнение

Runner в `_run_attempt` после `app.analyze` вызывает
`app.compare(previous_run_id, <run попытки>, criterion, config)` — до перехода задания в
`finished`. Краткая сводка пишется в запись попытки `attempts[n]["compare"]` тем же `_save`, что
и исход попытки. Поэтому цикл доставки (#102) не видит законченное контрольное задание без
сводки. Полный документ — `<data_root>/compare/<compare_id>.json`.

| `compare.state` | когда | поля |
|---|---|---|
| `completed` | сравнение выполнено | `compare_id`, `path`, `criterion`, `verdict`, `reasons` (`code`, `attempt`), `audio_review` и `backend` попыток `a`/`b` |
| `refused` | `CompareError` (например, исходный run удалён после submit) | `code`, `message` |
| `failed` | любое другое исключение | `error`: `internal_error: <тип>` |
| `not_run` | попытка не закончилась `finished` (отмена, `input_changed`, отказ анализа) | `reason`: состояние попытки |

`message` причин в сводку не копируется: в нём бывает `reason` стадии. Состояние задания от
сравнения не меняется. `jobs status` показывает `control` и `compare` последней попытки,
`jobs result` — `compare` той попытки, что отвечает. Повтор (`retry`) и `recover` нового кода не
требуют: каждая законченная попытка сравнивается заново с тем же исходным run.

## Заметка

`delivery.note_text` контрольного задания добавляет после строк слоёв блок сравнения из сводки
попытки, которая отвечает:

- `completed`: «Сравнение с исходной попыткой по критерию `<criterion>`: `<verdict>` — <подпись>.»;
  для `inconclusive` следом «Причины: …» по `COMPARE_REASON_LABELS` (все 16 кодов `compare.v1`);
  для `audio_review_missing` при `backend = none` — «audio backend не выбран (`backend = none`)»;
  последняя строка блока — «Сравнение: compare/<id>.json»;
- `refused` — код; `failed` — «внутренняя ошибка»; `not_run` или нет сводки — «контрольный анализ
  не закончен».

В заметку не попадают `message` причин, окна, слова транскрипта и evidence. Метка задания
остаётся последней строкой.

## Чего это решение не доказывает

Без audio backend настоящий вердикт всегда `inconclusive`. Тесты с подписями `improved` и
`not_improved` проверяют текст заметки на сводке, записанной вручную, а не способность инструмента
(R01, R03). Реальный `improved`/`not_improved` на двух живых записях (T10) ждёт выбора backend
([#90](https://github.com/renidere/sound-scout-plugin/issues/90)); живая заметка в настоящем чате —
проверка владельца, как у #102 ([#120](https://github.com/renidere/sound-scout-plugin/issues/120)).
