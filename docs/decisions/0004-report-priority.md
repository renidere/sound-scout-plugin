# 0004 — Отчёт и краткий ответ: правило приоритета findings, стоимость, приватные значения

- **Статус:** принято.
- **Дата:** 2026-11-18.
- **Задача:** [#32](https://github.com/renidere/sound-scout-plugin/issues/32); эпик
  [#5](https://github.com/renidere/sound-scout-plugin/issues/5); строится на `findings.build` /
  `findings.validate` из [#31](https://github.com/renidere/sound-scout-plugin/issues/31).
- **Основание:** `docs/spec/REQUIREMENTS.md` раздел 1 (до трёх приоритетных наблюдений, полный
  список в отчёте), R04, R05, R08, R09, R10, R11, раздел 4 (контракт результата).
- **Код:** `src/sound_scout/report.py` (`prioritized`, `compact`, `write`, `render`, `cost`).

## Решение

1. **Правило приоритета** (`report.prioritized`). Сначала влияние на понятность, затем уверенность,
   затем время. Оба критерия берутся из структуры finding, а не из текста модели: словесная
   уверенность не превращается в число (R05).
   - Влияние, ключ 1 — полярность: `negative` раньше `neutral`, `neutral` раньше `positive`.
     Нейтральное и положительное наблюдения не мешают пониманию.
   - Влияние, ключ 2 — категория, от прямого вреда пониманию к косвенному: `pronunciation`
     (неразборчивое слово или окончание), `completeness` (незавершённая фраза),
     `repetition_self_correction`, `pause_phrasing`, `semantic_stress`; затем то, что видно только
     по измерениям и транскрипту: `tempo`, `pause_length`, `level`, `pitch_range`,
     `filler_words`, `content`. Категория вне списка идёт последней.
   - Уверенность — основание: `human_verified` (настоящая человеческая разметка) раньше
     `audio_model_observation`, затем `measurement`, затем `transcript`.
   - Затем ранний `start`, затем `id`, чтобы порядок был детерминированным.
2. **Краткий ответ** (`report.compact`) — не больше 4096 байт UTF-8 JSON: `status`, `stages`
   (статус и причина), `coverage` (до четырёх диапазонов каждого вида, суммы в секундах, признак
   усечения), `assessment_status`, до трёх findings по правилу п. 1 с относительными путями клипов,
   числа `findings_total` и `unlocalized_total`, `limitations` (ограничения provenance и причины
   `not_assessed`, у synthetic-прогона — пометка первой строкой), коды `errors`, `cost`, ссылки
   `report.md`, `result.json` и клипы относительно run-каталога. Текстовые поля укорачиваются,
   пока ответ не уложится в предел. Полный список findings остаётся в `report.md` и
   `result.json` (раздел 1).
3. **Полный отчёт** (`report.write(run, result, *, config)`). Документ проверяется схемой,
   `run_id` сверяется с run, после удаления приватных значений добавляются артефакты
   `result-json` и `report-md` (`sha256: null`: файл не может содержать собственный хеш), затем
   `findings.validate(result, run=run)` проверяет документ и файлы-доказательства. `result.json` и
   `report.md` создаются эксклюзивно через `paths.open_output`; повторная запись в тот же run —
   ошибка. `report.md` строит одна функция `render` без шаблонизатора.
4. **Формулировки.** «Аудиомодель отмечает» — только у наблюдений с основанием
   `audio_model_observation` при `stages.audio_review.status == completed`. `not_assessed`
   всегда выводится с причиной и пояснением, что это не «всё хорошо». Статус выводится как есть
   (`partial` не превращается в `complete`, это уже проверяет контракт).
5. **Стоимость** (`report.cost`). Локальный прогон: `actual`, `0`, provider `local`, пометка
   «local». Внешний provider без данных: `unknown`, `amount: null`. С суммой нужна валюта ISO 4217.
   Нулевую стоимость внешнего provider отклоняет контракт (`schema._cost_errors`).
6. **Приватные значения** (R09, R10). Из записываемого документа и краткого ответа убирается
   `source.path`. Во всех строках заменяются на `<private path>` data root, run-каталог, allowed
   roots, путь конфигурации оператора и домашний каталог, а также любой другой абсолютный путь
   POSIX, `~/…` и `X:\…`; имя пользователя целым словом заменяется на `<user>`. Содержимое
   `sound-scout.toml` в контракте поля не имеет; пути из него попадают под замену.

## Рассмотренные варианты

- Отказ записывать отчёт при найденном пути вместо замены — отклонено: ошибка ffmpeg с путём в
  сообщении лишала бы пользователя всего результата, а замена сохраняет смысл сообщения.
- `report.write(run, result)` без `config` — отклонено: запись в run-каталог в проекте идёт только
  через `paths.open_output`, которому нужен data root (R10). Как и `findings.build`, `config`
  передаётся keyword-аргументом.
- Приоритет по тексту `impact`/`confidence_note` — отклонено: это угадывание по словам модели
  (R05).

## Границы

Сборку документа, вызов `write`/`compact` из `analyze` и ответ CLI/MCP делает
[#33](https://github.com/renidere/sound-scout-plugin/issues/33).
