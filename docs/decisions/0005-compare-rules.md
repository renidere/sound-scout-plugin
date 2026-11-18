# 0005 — Сравнение исходной и контрольной попыток: критерии, окна, правило вердикта

- **Статус:** принято.
- **Дата:** 2026-11-18.
- **Задача:** [#34](https://github.com/renidere/sound-scout-plugin/issues/34); CLI `compare` —
  [#35](https://github.com/renidere/sound-scout-plugin/issues/35), MCP `speech_compare` —
  [#36](https://github.com/renidere/sound-scout-plugin/issues/36).
- **Основание:** `docs/spec/REQUIREMENTS.md` R07, R04, R05, R10, T10; категории —
  `docs/spec/AUDIO_REVIEW_PROMPT.md`.
- **Код:** `src/sound_scout/comparison.py`, `app.compare`; схема `schema/compare.v1.schema.json`
  (упакованная копия — `src/sound_scout/compare.v1.schema.json`).

## Решение

1. **Попытки** задаются только `run_id` готовых run-каталогов `<data_root>/runs/<run_id>/` (A —
   исходная, B — контрольная). Путей от вызывающего нет (R10). Run должен быть настоящим каталогом
   (не symlink), `manifest.json.complete` — `true`, `result.json` проходит
   `findings.validate(document, run=…)`. Отказ — `CompareError` с кодом `criterion_invalid`,
   `run_not_found`, `run_incomplete`, `same_run` или `result_invalid`; файл сравнения не пишется,
   текст документа в сообщение не попадает.
2. **Критерии** — фиксированный список, каждый привязан к одной категории audio review:
   `emphasis_on:<слово>` → `semantic_stress`, `pause_before:<слово>` → `pause_phrasing`,
   `intelligibility_of:<слово>` → `pronunciation`, `completeness_of_phrase` → `completeness`.
   Слово одно, 1..64 символа без пробелов; сопоставление после `casefold`, `ё→е` и отбрасывания
   пунктуации по краям.
3. **Окна.** Слово ищется в сыром транскрипте `asr/raw.json` (сегменты `suspect_hallucination`
   пропускаются), а не в тексте модели. `emphasis_on`, `intelligibility_of` — интервалы всех
   вхождений; `pause_before` — от конца предыдущего слова (или 0.0) до конца слова;
   `completeness_of_phrase` — `coverage.requested` целиком. Audio review по критерию есть, если
   `stages.audio_review` — `completed` и каждое окно внутри `coverage.reviewed` (± `TIME_EPS`).
4. **Evidence** каждой попытки: findings категории критерия, пересекающие окна (для
   `completeness_of_phrase` — все), unlocalized этой категории, для `pause_before` — интервалы
   `speech_spans` с основанием `measurement`. Измерения в вердикт не входят (R04, R06).
5. **Правило вердикта.** Сначала собираются все причины недостоверности; есть хоть одна —
   `inconclusive`, причины перечисляются все сразу: `audio_review_missing`,
   `criterion_not_reviewed`, `word_not_found`, `level_differs` (|Δ p50 `rms_dbfs`| строго больше
   6 dB), `level_unknown`, `channel_differs`/`channel_unknown`, `format_differs`/`format_unknown`
   (`container`, `codec`, `sample_rate` из `source.json`), `task_differs`/`task_unknown`,
   `intent_differs`, `backend_differs`, `same_recording`, `unlocalized_observation` (в B есть unlocalized
   категории критерия с correction — только для критериев со словом), `no_baseline_issue`.
   Без причин: `not_improved`, если у B есть negative finding по критерию (для
   `completeness_of_phrase` — также unlocalized с correction), иначе `improved`. Длительность,
   число речевых интервалов, паузы и уровень в правило не входят, поэтому `improved` не следует из
   более короткой записи или меньшего числа пауз.
6. **`no_baseline_issue`.** Если в A, прослушанной по критерию, нет negative finding (и для
   `completeness_of_phrase` — unlocalized с correction), показывать улучшение не на чем: это
   `inconclusive`, а не `improved`. Для A без audio review причина не добавляется — там уже стоит
   `audio_review_missing`, и «нет замечания» в непрослушанной записи ничего не значит.
7. **Gain и задача.** Gain и микрофон записи инструменту не известны: `conditions_diff.gain`
   всегда `known: false`, само по себе это не причина (иначе недостоверным было бы любое
   сравнение), но ограничение есть в `limitations`, а косвенно gain виден по различию уровня.
   Задача попытки — `task_context.task` из `request.json` (его пишет `runs.open_run`); пробелы
   нормализуются. Разные задачи или задача только у одной — `task_differs`; run без поля
   `task_context` (старый) — `task_unknown`; задача не указана ни для одной — не причина, а
   ограничение «совпадение задания не проверено». Замысел говорящего `task_context.intent` (его
   же получает audio review) сравнивается так же и показывается в `conditions_diff.intent`:
   разный замысел или замысел только у одной попытки — `intent_differs`; не указан ни для одной —
   не причина.
8. **Результат** (`compare.v1`): `criterion`, `verdict`, `reasons[]`, `attempts.a|b` (run,
   `result_sha256`, статус audio review, окна, evidence), `conditions_diff` (`duration_s`,
   `pauses` — «в вердикт не входит», `level_dbfs`, `channel`, `format`, `gain`, `task`,
   `intent`, `backend`), `limitations` (критерий — не общая оценка; оговорка о паузах и новом
   вопросе; один контроль не доказывает навык; отсутствие замечания — не доказательство; gain
   неизвестен; условно — пометка synthetic, задача не указана, несколько вхождений слова,
   заявленные аудиомоделью ограничения и неопределённости каждой попытки с пометкой
   «попытка A/B» — из `provenance.limitations`; на вердикт они не влияют), `synthetic`.
   Документ проверяется схемой до записи и пишется в `<data_root>/compare/<compare_id>.json`
   (`compare_id = uuid4().hex`, каталог 0700, файл 0600, эксклюзивно). Analyze-lock не берётся:
   compare ничего не запускает.

## Рассмотренные варианты

- Ссылка на попытку путём или `request_id` — отклонено: путь требует ещё одной проверки границ
  (R10), `request_id` — индекса, а `run_id` у Hermes уже есть в кратком ответе.
- Задача из `review/<clip>/provenance.json` или новое поле `result.json` — отклонено: первое есть
  только при выполненном review, второе меняет контракт v1.
- Детерминированный id сравнения с повторным использованием файла — отклонено: гонка «создать или
  прочитать»; воспроизводимость дают `result_sha256` обеих попыток.

## Ограничения

Тесты (`tests/test_comparison.py`) строят попытки настоящим `app.analyze` над синтезированным
звуком, но с test double для audio review и ASR: они доказывают правила сравнения и контракт, а не
то, что инструмент слышит. При `backend = none` (Gate 0) сравнение реальных записей всегда
`inconclusive` с `audio_review_missing`. Проверка T10 на двух реальных записях — после подключения
реального backend ([#25](https://github.com/renidere/sound-scout-plugin/issues/25)) в приёмке
Gate 3 ([#42](https://github.com/renidere/sound-scout-plugin/issues/42)).
