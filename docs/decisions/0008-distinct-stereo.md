# 0008 — Различное стерео: усреднение каналов как ограничивающее условие

- **Статус:** принято.
- **Дата:** 2026-11-18.
- **Решение владельца:** вариант 2,
  [комментарий в #48](https://github.com/renidere/sound-scout-plugin/issues/48#issuecomment-6988781270).
- **Задача:** [#48](https://github.com/renidere/sound-scout-plugin/issues/48); найдено при работе над
  [#17](https://github.com/renidere/sound-scout-plugin/issues/17).
- **Основание:** `docs/spec/REQUIREMENTS.md` R02 (выбор канала фиксируется; несколько говорящих —
  неподдержанное или ограничивающее условие; без скрытых изменений сигнала), T02.
- **Код:** `src/sound_scout/channels.py` (`report`, `choose_mono`, `DISTINCT_STEREO`,
  `DISTINCT_STEREO_LIMITATION`), `src/sound_scout/app.py` (`_assemble`, `provenance.limitations`).

## Вопрос

Как трактовать два различных активных канала: корреляция по модулю ниже `DUPLICATE_CORRELATION`,
гашения нет. Варианты: (1) молча усреднять; (2) усреднять, но называть это ограничением в
результате; (3) отказ с `ChannelError`, этап `prepare` — `failed`.

## Решение

Вариант 2.

1. Вердикт `stereo` остаётся. Каналы усредняются без gain (`arithmetic_mean`, `gain_db: 0`),
   кадры и позиции отсчётов исходника сохраняются; запись `channel_choice` в
   `provenance.transformations` остаётся с `verdict: "stereo"`.
2. В `provenance.limitations` результата добавляется `channels.DISTINCT_STEREO_LIMITATION`:
   каналы различаются и усреднены, возможны несколько микрофонов или несколько говорящих. Через
   `provenance.limitations` строка попадает в `report.md` и в компактный ответ. В сравнении
   попыток (`speech_compare`) различие каналов видно только через `conditions_diff.channel`.
3. Перечень вердиктов один — шесть: `zero`, `mono`, `one_active`, `cancellation`,
   `duplicated_stereo`, `stereo`. Он совпадает в docstring модуля `channels`, в тесте
   принадлежности вердикта набору (`tests/test_channels.py`) и в критерии #17.

## Причины

- Порог `DUPLICATE_CORRELATION = 0.99` строгий: стерео одного говорящего с разнесённых микрофонов
  часто ниже него. Отказ (вариант 3) отбрасывал бы обычные записи.
- Молчаливое усреднение (вариант 1) скрыло бы случай нескольких говорящих, который R02 требует
  считать ограничивающим условием.
- Число говорящих по-прежнему не определяется: ограничение говорит «возможны», а не «обнаружено»
  (R02 запрещает заявлять надёжное распознавание количества людей без проверки).

## Проверки

- `tests/test_channels.py::test_distinct_stereo_is_averaged_without_gain` — два тона с разной фазой
  дают `stereo`, моно равно среднему каналов, `gain_db` равен 0.
- `tests/test_app.py::test_distinct_stereo_is_averaged_and_named_in_the_limitations` — сквозной
  `analyze` на синтезированном стерео WAV: для различных каналов ограничение есть в
  `provenance.limitations` и в `report.md`, для дублированного стерео и гашения его нет.
