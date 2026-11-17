# 0002 — Имена полей `limits` в `source.json` и в контракте результата

- **Статус:** принято.
- **Дата:** 2026-11-17.
- **Задача:** [#47](https://github.com/renidere/sound-scout-plugin/issues/47); источник расхождения —
  [#16](https://github.com/renidere/sound-scout-plugin/issues/16) (`ingest.py`, `source.json`);
  потребитель — [#32](https://github.com/renidere/sound-scout-plugin/issues/32) (раздел `limits`
  отчёта).
- **Основание:** `docs/spec/REQUIREMENTS.md` раздел 2 (заявленные параметры и фактическая
  длительность декодированных данных), раздел 4 (`limits`: фактическое/оценочное/unknown), R02,
  R10, R11.

## Решение

1. Одна величина — одно имя. Поля, общие для `source.json` (пишет `ingest.decode`) и для
   `result.json` (`limits` в `schema/result.v1.schema.json`), называются одинаково:
   `max_duration_seconds`, `max_input_bytes`, `max_channels`, `duration_mismatch_tolerance_s`,
   `duration_mismatch_s`, `duration_mismatch`. Три последних добавлены в контракт этой задачей
   как обязательные; `null` в `duration_mismatch_s` и `duration_mismatch` означает, что
   контейнер длительности не заявил. Согласованность флага с измеренным расхождением проверяет
   `schema._limits_errors`: `duration_mismatch` обязан равняться
   `|duration_mismatch_s| > duration_mismatch_tolerance_s`.
2. `max_subprocess_seconds` и `worker_deadline_seconds` **не переименовываются**: это разные
   величины из `config.Budgets`, обе уже задокументированы в `sound-scout.example.toml`.
   - `max_subprocess_seconds` — wall time **одного** дочернего вызова `ffmpeg`/`ffprobe`
     (`proc.run`, по истечении убивается группа процессов). Это лимит стадии intake, поэтому он
     записан в `source.json`.
   - `worker_deadline_seconds` — срок **всего** worker анализа (prepare → report). Это лимит
     запуска, поэтому он записан в `result.json`.
   Один сеанс декодирования не знает срока worker, а результат не обязан перечислять бюджет
   каждого дочернего процесса; у величин разные области действия и разные значения по умолчанию
   (60 с и 180 с).
3. `decoded_samples_limit` (производная от `max_duration_seconds`, частоты и числа каналов
   конкретного потока) остаётся только в `source.json`: это предел стадии intake для данного
   формата, а не бюджет запуска.

Разницу множеств имён фиксирует тест
`test_source_json_limits_share_their_names_with_the_result_contract` в `tests/test_ingest.py`:
только в `source.json` — `decoded_samples_limit`, `max_subprocess_seconds`; только в результате —
`worker_deadline_seconds`. Любое новое поле либо получает одно имя в обоих документах, либо
дополняет этот список с объяснением здесь.

## Отвергнуто

- Переименовать `max_subprocess_seconds` → `worker_deadline_seconds` в `ingest.py`: назвало бы
  60-секундный бюджет одного ffmpeg сроком worker — ложная запись в `source.json`.
- Добавить `max_subprocess_seconds` и `decoded_samples_limit` в `limits` результата: #47 этого не
  просит; если раздел `limits` отчёта (#32) потребует их, это отдельное расширение контракта.
