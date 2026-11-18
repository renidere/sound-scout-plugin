# Sound Scout

Инструмент анализа звучания русской речи для агента Hermes: из голосового файла — раздельные ASR,
акустические измерения и наблюдения аудиомодели о реальном звучании, с проверяемыми таймкодами,
клипами, JSON- и Markdown-отчётом и одним контрольным заданием. Python CLI (`doctor`, `analyze`,
`compare`, `serve`) и локальный MCP stdio (`speech_capabilities`, `speech_analyze`,
`speech_compare`) поверх одного application layer.

Статус: требования зафиксированы (`docs/spec/`), реализация ведётся через задачи на доске
https://github.com/users/renidere/projects/1 автономным агентом (`docs/workflow.md`). Audio backend
не выбран, русская квалификация не пройдена, подключение к Hermes не выполнено — пока это так,
инструмент не считается готовым (Gate 0–3 в `docs/spec/REQUIREMENTS.md`).

## Документы

- `docs/spec/REQUIREMENTS.md` — назначение, архитектура, R01–R12, контракт результата, Gate 0–3,
  приёмка T01–T10.
- `docs/spec/AGENT_PROMPT.md` — поручение агенту-разработчику и границы полномочий.
- `docs/spec/AUDIO_REVIEW_PROMPT.md` — черновик инструкции audio backend и правила использования
  результата в Hermes.
- `docs/eval/PROTOCOL.md` — протокол квалификации слоя звучания (Gate 1): категории событий,
  корпус и манифест, dev/held-out, сопоставление интервалов, пороги, abstention, контроли.
- `docs/workflow.md` — как устроена автономная работа через задачи.
- `CLAUDE.md` — правила работы в репозитории для агентов.

## Приватность

Записи пользователя, расшифровки, личные отчёты, ключи и локальные пути в репозиторий не входят.
Development-данные и существующий акустический слой описаны в `local/HANDOFF.md` вне git. Тесты
синтезируют аудио сами.

## Установка и запуск

Нужны Python 3.13, [`uv`](https://docs.astral.sh/uv/) и системный FFmpeg (`ffmpeg` в `PATH`:
`apt install ffmpeg`, `brew install ffmpeg`). Установка из корня репозитория:

```sh
uv sync            # создаёт .venv и ставит пакет с dev-инструментами по uv.lock
uv run sound-scout --version
uv run sound-scout --help
```

`uv run sound-scout doctor` печатает таблицу состояния компонентов (`decoder`, `asr`,
`acoustics`, `audio_review`: `ready | unavailable | unvalidated | permission_required` с причиной,
отпечатком и временем последнего smoke-теста), `--json` — тот же отчёт в JSON; код возврата 0 при
любых статусах, это диагностика, а не проверка. `ready` требует smoke-receipt
(`<data_root>/receipts/<компонент>.json`), совпадающий с текущим отпечатком компонента; `doctor`
только читает его, ничего не пишет, не скачивает и не запускает inference.

Receipt для `decoder`, `acoustics` и `asr` оператор получает командой `smoke`, затем проверяет
`doctor`:

```sh
uv run sound-scout smoke            # decoder и acoustics; или `smoke decoder`, `smoke acoustics`
SOUND_SCOUT_DEV_AUDIO=<development-запись> uv run sound-scout smoke asr
uv run sound-scout doctor
```

`smoke` читает ту же конфигурацию, что и `doctor`, ставит сетевой guard и по-настоящему
прогоняет компоненты на синтетическом сигнале: настоящий FFmpeg декодирует WAV PCM 48 кГц длиной
1 с (точное число отсчётов, совпадение с записанным сигналом), настоящие Silero VAD и Praat
измеряют формантный контроль с F0 150 Гц (p50 в пределах ±3 %, одна речевая область) и цифровую
тишину (ни одной речевой области). Записи пользователя и сеть не нужны; сама команда пишет только
временный `<data_root>/.smoke-*`, который удаляется, и `receipts/`. Телеметрия onnxruntime
(через него работает детектор речи) выключается до его импорта (`ORT_DISABLE_TELEMETRY=1`) при
любом `network.allow`, поэтому вне data root ничего не пишется. Receipt пишется только при успехе всех проверок; при неудаче команда печатает `<компонент>: failed: <причина>`, завершается с кодом 1, а
прежний receipt остаётся нетронутым. Повторяйте `smoke`, когда `doctor` пишет, что receipt
устарел (`stale`): после обновления FFmpeg или пакетов (`praat-parselmouth`, `onnxruntime`,
`faster-whisper`, `sound-scout`) и после смены `budgets`.

`smoke asr` запускается только по имени: синтетический сигнал не доказывает, что ASR слышит речь,
поэтому команда читает development-запись с речью из `SOUND_SCOUT_DEV_AUDIO` (путь внутри
`allowed_roots`, как у `analyze`) и распознаёт её настоящей моделью из конфигурации — только уже
установленной в локальном кэше Hugging Face, offline (`local_files_only`), без загрузки. Запись
проходит тот же путь, что в `analyze` (FFmpeg, выбор канала, 16 кГц, лимиты размера,
длительности и числа каналов, бюджеты FFmpeg), но в самом процессе команды, без worker'а:
`budgets.worker_deadline_seconds` на неё не действует. Сырой транскрипт пишется во временный `<data_root>/.smoke-*` и удаляется вместе с ним. Receipt
`receipts/asr.json` пишет `asr.write_smoke_receipt` через `doctor.write_receipt` и только для
завершённого распознавания настоящей `WhisperModel` хотя бы с одним словом и без пометки
`suspect_hallucination`; в нём счётчики, вероятности, время и `input` вида
`"development recording, 10.6 s"` — ни текста транскрипта, ни путей. Отказ — переменная не
задана, модели нет, запись вне `allowed_roots`, в записи нет речи, receipt отклонён — печатает
`asr: failed: <причина>`, даёт код 1 и не трогает прежний receipt. `doctor` в обоих случаях только
читает receipt. Подкоманды `analyze`, `compare`, `serve` пока завершаются с кодом 2 и сообщением
`not implemented`. Локальная конфигурация —
`sound-scout.toml` (копия `sound-scout.example.toml`, в git не входит): `doctor` читает её через
`sound_scout.config.load()` из текущего каталога или по пути в `SOUND_SCOUT_CONFIG`; остальные
подкоманды и MCP будут читать её так же. Обязательны
`allowed_roots` и `data_root`, остальное — безопасные умолчания из примера. Без файла — явная
ошибка (код 1), не умолчания для путей.

Анализ выполняет `sound_scout.app.analyze` — общая точка входа CLI и MCP. На один data root
одновременно идёт один анализ: lock `<data_root>/analyze.lock`, второй запрос сразу получает
`Busy`, очереди нет. Этапы выполняет дочерний процесс-worker; его срок —
`budgets.worker_deadline_seconds`, по истечении worker и его дочерние процессы останавливаются, а
результат собирается из готовых этапов (`work/stage-<этап>.json`) как `partial`, а если не
завершился даже `prepare` — как `failed`. Отмена через
`CancelToken` освобождает worker не дольше чем за 2 с.

Для pre-commit-хуков установите [lefthook](https://github.com/evilmartians/lefthook) и выполните
`lefthook install`: перед коммитом по staged-файлам пройдут `ruff check` и `ruff format --check`.

## Run-каталог

Каждый запрос получает приватный каталог `<data_root>/runs/<run_id>/` (`request.json`,
`manifest.json`); оригинал остаётся read-only, все производные лежат только здесь:

- `source.json` и `work/decoded.f32` — SHA-256, метаданные и лимиты оригинала; декодированный
  поток как есть (float32, исходная частота и каналы), без ресемплинга и нормализации. Отчёт о
  каналах и акустические измерения считаются по этому сигналу.
- `work/mono16k.f32` и `work/mono16k.json` — производная для ASR: рабочее моно, ресемплированное
  системным ffmpeg до 16 кГц (`prepare.mono16k`), и запись преобразования: исходная и целевая
  частота, ресемплер и его параметры, число отсчётов до и после, SHA-256 производной, версия
  ffmpeg. Длительность совпадает с оригиналом с точностью до 1 мс, поэтому таймкоды транскрипта
  остаются на исходной шкале; измерения по производной не делаются. Повторный вызов для того же
  run возвращает уже записанную производную.
- `asr/raw.json` и `asr/model-receipt.json` — сырой транскрипт и запись о модели;
  `transcript/corrected.json` — исправления отдельным слоем (см. ниже).

Доверие к тексту ASR решает правило `suspect_hallucination`, а не вероятности слов: на цифровой
тишине модель отвечает уверенной выдуманной фразой. Доверенные и помеченные сегменты выбирает одна
функция `asr.trusted_segments(raw)`; помеченные попадают в `result.json` и `report.md` только как
интервалы `transcript.suspect_segments` с признаками и `certainty: "uncertain"`, без текста
(`docs/decisions/0005-asr-silence-hallucination.md`).

## Проверки

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest             # -m "not slow" пропускает тесты с реальными моделями
```

Тесты с маркером `slow` используют настоящую модель ASR из локального кэша Hugging Face
(`~/.cache/huggingface/hub`, без загрузки: нет модели — тест пропускается с причиной) и, для
проверки на реальной речи, путь к development-записи из `SOUND_SCOUT_DEV_AUDIO` (не задан —
пропуск). Эти тесты собирают конфигурацию сами и пишут smoke-receipt `asr`
(`asr.write_smoke_receipt`) только в свой временный `data_root`, даже при заданном
`SOUND_SCOUT_CONFIG`; в настоящий `data_root` его пишет `sound-scout smoke asr`. В
`asr/raw.json` записан класс модели, которая реально работала, и receipt для подменённой модели
отклоняется; контрактные тесты с заглушкой способность ASR не доказывают. Сырой транскрипт лежит в run-каталоге как `asr/raw.json` рядом с
`asr/model-receipt.json` (repo, revision, хеши файлов модели, версии пакетов) и после записи не
меняется; исправления — отдельный артефакт `transcript/corrected.json` со ссылкой на raw.

Те же проверки запускает CI (`.github/workflows/ci.yml`) на каждый push в `main`.
