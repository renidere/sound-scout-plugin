# Sound Scout

Sound Scout разбирает голосовой файл с русской речью. Результат — три раздельных слоя: транскрипт
ASR, акустические измерения и наблюдения аудиомодели. У каждого слоя свой статус. Таймкоды и клипы
лежат на шкале исходника. Итог — `result.json` и `report.md`. Пакет на Python даёт CLI и MCP stdio
сервер поверх одного application layer (`src/sound_scout/app.py`). Для Hermes есть плагин.

## Что работает сейчас

Состояние `main` на 2026-11-25.

- Инструмент не считается готовым: русская квалификация (Gate 1) не пройдена
  ([#29](https://github.com/renidere/sound-scout-plugin/issues/29),
  [#30](https://github.com/renidere/sound-scout-plugin/issues/30)).
- Gate 0–3 — `docs/spec/REQUIREMENTS.md`. Матрица приёмки — `docs/ACCEPTANCE.md`.
- Audio backend не подключён (`backend = none`). Этап `audio_review` поэтому `unavailable`, итог
  анализа со слоем `sound` — `partial`.

| Часть | Что делает | Код и решения |
|---|---|---|
| CLI | `doctor`, `smoke`, `analyze`, `compare`, `jobs`, `serve`, `eval` | `src/sound_scout/cli.py` |
| Приём входа | WAV, MP3, OGG/Opus, M4A/AAC через FFmpeg; лимиты R02; SHA-256 | `src/sound_scout/ingest.py` |
| Слой `content` | ASR faster-whisper offline; правила `suspect_hallucination` и `no_vad_speech` | `src/sound_scout/asr.py`, `docs/decisions/0005-asr-silence-hallucination.md`, `docs/decisions/0021-asr-vad-hallucination.md` |
| Слой `measurements` | речевые интервалы Silero VAD, F0 Praat, RMS; выбор канала | `src/sound_scout/acoustics.py`, `src/sound_scout/channels.py` |
| Слой `sound` | адаптер audio backend, окна, валидатор ответа; без backend — `unavailable` | `src/sound_scout/review.py`, `src/sound_scout/findings.py` |
| Отчёт | `result.json`, `report.md`, compact-ответ до 4 KiB | `src/sound_scout/report.py`, `docs/decisions/0004-report-priority.md` |
| Сравнение попыток | вердикт по критерию, `compare/<id>.json` | `src/sound_scout/comparison.py`, `docs/decisions/0006-compare-rules.md` |
| MCP stdio | 9 инструментов, список — в разделе ниже | `src/sound_scout/mcp_server.py`, `docs/mcp.md` |
| Очередь заданий | `submit`, `status`, `result`, `report`, `cancel`, `retry`; один runner; режимы `quick`, `targeted`, `full`, `auto`; контрольная попытка | `src/sound_scout/jobs.py`, `docs/decisions/0018-job-modes-and-retry.md`, `docs/decisions/0020-control-attempt-jobs.md` |
| Плагин Hermes | рекордер Desktop, карточка задания, привязка задания к чату, заметка с итогом в чат | `hermes/plugin/sound-scout/`, `docs/decisions/0013-voice-note-integration.md`, `docs/decisions/0025-mcp-chat-binding.md`, `docs/decisions/0029-cancel-stops-chat-turn.md` |
| Skill Hermes | когда вызывать инструменты и как читать результат | `hermes/skills/speech-review/SKILL.md` |

Проверки на станции разработки:

- MCP сервер `speech-review` подключён к Hermes 2026-11-19
  ([#40](https://github.com/renidere/sound-scout-plugin/issues/40), закрыта). Вызовы из
  `hermes -z` и из Hermes Desktop вернули `partial`: слоя `sound` нет. Подключение на другом хосте —
  шаг владельца (`docs/hermes-install.md`).
- Пакет плагина развёрнут в gateway Hermes 2026-11-21
  ([#104](https://github.com/renidere/sound-scout-plugin/issues/104), закрыта).
- Linux-путь прошёл: задание, runner, отчёт, заметка в чате
  ([#114](https://github.com/renidere/sound-scout-plugin/issues/114), закрыта).
- Рекордер прошёл минимальный путь на Windows-клиенте Desktop. Заметка дошла и в закрытый Desktop
  ([#105](https://github.com/renidere/sound-scout-plugin/issues/105), закрыта).
- Эпик Windows-клиента открыт ([#115](https://github.com/renidere/sound-scout-plugin/issues/115)).
  Приёмка A01–A14 — `docs/acceptance/A01-A14.md`.

## Интеграция с другими харнесами

| Вход | Что даёт | Кто может вызвать |
|---|---|---|
| MCP stdio `sound-scout serve` | 9 инструментов поверх `app`, `jobs` и `delivery` | любой MCP-клиент со stdio; проверен только Hermes |
| CLI `sound-scout` | те же функции, ответ `--json`, коды возврата | скрипты, CI, любой агент с shell |
| Очередь `jobs` | асинхронный анализ одним runner | CLI и MCP; плагин Hermes — через CLI |
| Плагин Hermes | рекордер, карточка, привязка к чату, заметка в чат | только Hermes |

Инструменты MCP:

- синхронные: `speech_capabilities`, `speech_analyze`, `speech_compare`;
- задания: `speech_job_submit`, `speech_job_status`, `speech_job_result`, `speech_job_cancel`,
  `speech_job_retry`, `speech_job_redeliver`.

Подключение другого MCP-хоста (Claude Code, Codex CLI, Cursor и др.) не проверено. Запись сервера
строится так же, как для Hermes (`docs/examples/hermes-mcp-entry.example.yaml`):

- команда `uv run --directory <checkout> --offline sound-scout serve`;
- `env.SOUND_SCOUT_CONFIG` — путь к `sound-scout.toml`; `env.PATH` — каталоги с системным FFmpeg;
- таймаут вызова клиента — не меньше 300 с при бюджетах по умолчанию: `speech_analyze` ждёт весь
  анализ.

Правила для любого клиента (`docs/mcp.md`):

- Файл лежит на хосте сервера внутри `allowed_roots`. URL и Windows-путь получают отказ
  `path_denied`.
- Сервер пишет только в `data_root`.
- `allowed_roots`, `data_root`, модель, backend и бюджеты задаёт оператор в `sound-scout.toml`.
  Такой аргумент инструмента получает отказ `invalid_arguments`.
- Stdout сервера несёт только JSON-RPC. Логи идут в stderr, пути в них замаскированы.
- `speech_analyze` держит lock анализа: второй анализ получает отказ `busy`.
- `speech_job_submit` сразу отвечает `job_id`. Анализ выполняет runner: `sound-scout jobs start`
  (`systemd-run --user`) или `sound-scout jobs run`. Без runner задание остаётся `queued`.

Асинхронный итог вне Hermes:

- Заметка в чат приходит только через плагин Hermes. Задание из другого клиента остаётся без
  привязки.
- Клиент опрашивает `speech_job_status`, пока состояние не станет `finished`, `failed` или
  `cancelled`, затем читает `speech_job_result`. `finished` не означает `complete`.
- Цикл доставки плагина вызывает CLI `jobs bind`, `jobs deliver-next` и `jobs deliver-end`. Свою
  доставку другой харнес может построить на этих командах. Этот путь вне Hermes не проверен.

CLI для скриптов и CI:

- `analyze --json` печатает compact-ответ. Код возврата: 0 `complete`, 3 `partial`, 4 `blocked`,
  1 `failed`.
- `doctor --json` печатает состояние компонентов и ничего не запускает.
- `jobs … --json` печатает тот же документ, что инструменты MCP.
- Отказ — одна строка в stderr, код 1. Stdout несёт только ответ.

Только для Hermes:

- рекордер и карточка задания на Desktop Plugin SDK —
  `hermes/plugin/sound-scout/desktop/plugin.js`;
- хук `post_tool_call` привязывает задание к чату хода — `hermes/plugin/sound-scout/binding.py`;
- цикл доставки заметки в процессе gateway — `hermes/plugin/sound-scout/delivery.py`;
- маршруты `/api/plugins/sound-scout/` (`GET /health`, `POST /jobs`, `GET /jobs/{job_id}`, отчёт,
  отмена, повтор) — `hermes/plugin/sound-scout/dashboard/plugin_api.py`;
- skill `speech-review` — `hermes/skills/speech-review/`.

## Зависимости

Версии, лицензии и SHA-256 — `docs/dependencies.md`. Python-пакеты закреплены в `uv.lock`.

| Часть | Зачем |
|---|---|
| Python 3.13, `uv` | среда и lock |
| FFmpeg (`ffmpeg`, `ffprobe`) из `PATH` | декодирование, ресемплинг, клипы |
| `faster-whisper`, `ctranslate2`, `av` | ASR на CPU в int8 |
| `onnxruntime` | Silero VAD (ONNX-файл из пакета `faster-whisper`), телеметрия выключена |
| `praat-parselmouth` | F0 |
| `numpy` | сигналы и измерения |
| `jsonschema` | контракт результата, манифест корпуса, разметка |
| `mcp` | MCP stdio сервер |
| extra `gate0-borealis`, `gate0-gigachat` | только прогоны Gate 0 этих моделей (`torch`, `transformers`) |
| extra `corpus-tts` | только синтез клипов корпуса (`piper-tts`, `onnx`) |
| llama.cpp `llama-server` b12359 | только прогоны Gate 0 моделей GGUF |
| Node.js ≥ 20 | только тесты плагина Desktop |

## Модели на стенде

Станция разработки работает только на CPU. Модели загружаются из локальных файлов. Сеть закрыта.

| Роль | Модель | Источник |
|---|---|---|
| ASR | Whisper large-v3-turbo в CTranslate2, `mobiuslabsgmbh/faster-whisper-large-v3-turbo` @ `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`, CPU int8, язык `ru` | `docs/dependencies.md` § 4 |
| VAD | Silero VAD, ONNX-файл из пакета `faster-whisper` 1.2.1 | `docs/dependencies.md` § 4 |
| Audio backend | нет (`backend = none`) | `sound-scout.example.toml` |

Прогоны Gate 0 кандидатов audio backend. Ни один кандидат не получил READY, поэтому `backend = none`.

| Кандидат | Runtime | Вердикт | Решение |
|---|---|---|---|
| Gemma 4 E4B-it, GGUF Q8_0 | llama.cpp, только локально; кандидат владельца (`docs/decisions/0001-audio-backend.md`) | Gate 0 не прошла — вердикт NOT READY | `docs/decisions/0007-gate0-result.md` |
| Borealis-5b-it | `torch` CPU | NOT READY | `docs/decisions/0010-gate0-borealis.md` |
| GigaChat3.1-Audio-10B-A1.8B | `torch` CPU, потолок 26 GB | NOT READY: не выполнено предусловие P3 (6,25 аудиопозиции в секунду при пороге 20); о слухе модели вердикт ничего не говорит | `docs/decisions/0011-gate0-gigachat-audio.md` |
| Qwen3-Omni-30B-A3B-Instruct, GGUF Q4_K_M | llama.cpp, `GATE0-RULE v2` | NOT READY: 2 из 12 ответов контролей пустые или обрезанные | `docs/decisions/0023-gate0-qwen3-omni.md` |
| Qwen2.5-Omni-7B, GGUF Q8_0 | llama.cpp, `GATE0-RULE v1`, диагностический прогон | NOT READY: прошли 7 из 12 пунктов контролей | `docs/decisions/0024-gate0-qwen25-omni.md` |

MOSS-Audio-8B-Thinking-ru проверен диагностическим прогоном, это не формальный Gate 0: READY не
получен ([#90](https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7024154195)).
Замену владелец выбирал в [#68](https://github.com/renidere/sound-scout-plugin/issues/68)
(закрыта). Следующий ограниченный эксперимент предлагает агент, утверждает владелец
([#119](https://github.com/renidere/sound-scout-plugin/issues/119)). Подключение backend с READY —
[#90](https://github.com/renidere/sound-scout-plugin/issues/90).

GPU-эталон 2026-11-23 ([#149](https://github.com/renidere/sound-scout-plugin/issues/149),
`docs/decisions/0028-gpu-reference-gate0.md`, «Результат»): Vast.ai, 1 × A100 80GB, vLLM 0.34.0,
веса bf16. Стенд разработки эти модели в bf16 не запускает.

| Модель | Правило | Вердикт |
|---|---|---|
| Qwen2.5-Omni-7B, bf16 | `GATE0-RULE v1` | NOT READY: не прошли 4 из 12 пунктов контролей, `P(S0)` и `A` на трёх контролях |
| Qwen3-Omni-30B-A3B-Instruct, bf16 | `GATE0-RULE v2` | NOT READY: не прошёл 1 из 12, `P(T1)` |

## Чем можно заменить

Кандидаты ниже не проверены. Замена требует решения владельца и новой записи в
`docs/dependencies.md` с ревизией и SHA-256. Кандидату audio backend нужен ещё прогон Gate 0.

ASR:

| Кандидат | Почему | Лицензия | Источник |
|---|---|---|---|
| GigaAM-v3 (`ai-sage/GigaAM-v3`) | Conformer 220–240M для русского; карточка даёт средний WER 8,4 у RNNT против 25,1 у Whisper; нужен собственный код модели | MIT | https://huggingface.co/ai-sage/GigaAM-v3 |
| T-one (`t-tech/T-one`) | потоковый CTC-Conformer 71M для русского, в первую очередь телефония | Apache-2.0 | https://huggingface.co/t-tech/T-one |
| Parakeet-TDT-0.6B-v3 (`nvidia/parakeet-tdt-0.6b-v3`) | 25 европейских языков, в том числе русский; таймкоды слов; runtime NeMo | CC-BY-4.0 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| Canary-1B-v2 (`nvidia/canary-1b-v2`) | 25 европейских языков, в том числе русский; таймкоды слов; runtime NeMo | CC-BY-4.0 | https://huggingface.co/nvidia/canary-1b-v2 |

Audio backend (не входили в Gate 0):

| Кандидат | Почему | Лицензия | Источник |
|---|---|---|---|
| Gemma 4 12B (`google/gemma-4-12B-it`) | запасной кандидат `docs/decisions/0022-gate0-next-candidate.md`; аудио до 30 с; по `0022` русский ASR есть в техотчёте; GGUF есть, аудио 12B нет в аудиотестах llama.cpp b12359 | Apache-2.0 | https://huggingface.co/google/gemma-4-12B-it |
| Qolda-AVL-5B-OptimizedTokenizer (`issai/Qolda-AVL-5B-OptimizedTokenizer`) | аудио на входе, языки казахский, русский, английский; энкодер Whisper-large-v3-turbo; по `0022` 12,5 аудиотокена в секунду — P3 `GATE0-RULE v1` не выполняется; GGUF нет | Apache-2.0 | https://huggingface.co/issai/Qolda-AVL-5B-OptimizedTokenizer |

Решение `0022` отклонило остальные модели своего обзора: нет русского, некоммерческая лицензия или
не выполнено P3.

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
читает receipt. Подкоманда `serve` запускает MCP stdio сервер с инструментами
`speech_capabilities`, `speech_analyze`, `speech_compare`, `speech_job_submit`,
`speech_job_status`, `speech_job_result`, `speech_job_cancel`, `speech_job_retry`,
`speech_job_redeliver` (описание и схемы — `docs/mcp.md`).
Локальная конфигурация —
`sound-scout.toml` (копия `sound-scout.example.toml`, в git не входит): `doctor`, `smoke`, `analyze` и
`compare` читают её через `sound_scout.config.load()` из текущего каталога или по пути в `SOUND_SCOUT_CONFIG`;
`serve` читает её так же. Обязательны
`allowed_roots` и `data_root`, остальное — безопасные умолчания из примера. Без файла — явная
ошибка (код 1), не умолчания для путей. Корень с запрещённым именем в пути (`.env*`, `.ssh`,
`.hermes`, `sound-scout.toml`, плейлисты) отклоняется при загрузке, кроме медиакаталога Hermes:
`~/.hermes/cache/audio`, legacy `~/.hermes/audio_cache` и те же каталоги профиля
`~/.hermes/profiles/<имя>/…` — с ровно одним `.hermes` в пути; часть пути от `.hermes` и ниже
должна совпадать с реальным путём: symlink выше `.hermes` допустим, а symlink, скрывающий
`.hermes` под другим именем (например `media` -> `.hermes`), отклоняется. `~/.hermes/attachments` и другие каталоги Hermes корнем не разрешены; ниже корня
запрещённые имена отклоняются по-прежнему. Ограничения входа через медиакаталог Hermes:
голосовое сообщение при `stt.enabled: true` доходит до модели только расшифровкой, без пути, —
запись присылайте аудиофайлом или документом; Hermes удаляет медиакэш через 24 часа; Telegram Bot
API скачивает вложения до 20 МБ; нужен `terminal.backend: local`; `cache/audio` — общий пул всех
собеседников, схема рассчитана на однопользовательский gateway. Ни один флаг CLI не задаёт allowed roots, data root,
модель, backend или бюджеты: это настройки оператора (R12), такой флаг отклоняется как неизвестный.

```sh
uv run sound-scout analyze <файл> [--request-id ID] [--layers content,measurements,sound] [--task ТЕКСТ] [--intent ТЕКСТ] [--json]
uv run sound-scout compare <run_id A> <run_id B> --criterion emphasis_on:<слово> [--json]
uv run sound-scout jobs submit <файл> [--request-id ID] [--mode quick|targeted|full|auto] [--layers …] [--task ТЕКСТ] [--intent ТЕКСТ] [--connection …] [--profile …] [--session-id …] [--submitter cli] [--json]
uv run sound-scout jobs status|result|cancel|retry <job_id> [--json]
uv run sound-scout jobs run [--exit-when-idle]
uv run sound-scout jobs start|recover [--json]
```

`jobs` — файловая очередь заданий (`sound_scout.jobs`, решение 0013 § 3.6): `submit` за доли секунды пишет `<data_root>/jobs/<job_id>/job.json` и печатает `job_id`; проверка входа (ffprobe) — первая стадия runner, не `submit`. Повтор `--request-id` с тем же запросом возвращает то же задание, с другим — `request_mismatch`. Сверх `jobs.max_queued_jobs` или `jobs.max_staging_bytes` — отказ `queue_full` или `staging_full`. `cancel` переводит задание из `queued` в `cancelled` сразу, у `running` оставляет маркер отмены. Состояние задания (`queued/running/finished/cancelled/failed`) отдельно от `status` результата: `result` печатает compact-ответ run последней попытки, у которой есть результат, с полями `job_state` и `attempt` (номер этой попытки); новая попытка без результата не скрывает результат прежней. Задания исполняет один runner: `jobs run` — в текущем процессе (для тестов и хостов без systemd; `--exit-when-idle` — выйти, когда очередь пуста), `jobs start` — через `systemd-run --user` как transient unit `sound-scout-jobs-runner` вне cgroup вызывающего (у unit gateway `KillMode=mixed`, его дочерние процессы гибнут при перезапуске). Runner единственный по `<data_root>/jobs/runner.lock` (второй — `runner_running`), берёт задания по порядку постановки в очередь (`queued_at`: `submit` или `retry`) и вызывает `app.analyze` с `request_id` попытки `<job_id>-a<n>`; `analyze.lock` берёт сам `app.analyze`, и при `busy` (идёт синхронный анализ) задание возвращается в очередь на своё место. Маркер отмены у `running` runner переводит в `CancelToken`, задание заканчивается `cancelled`. Отказ анализа — `failed` с кодом причины. Попытка хранит lease (boot_id, pid, время старта процесса, номер); после остановленного runner (перезагрузка, `kill`, `systemctl --user stop`) задание остаётся `running`, пока `jobs recover` не отметит попытку `interrupted` и не вернёт задание в очередь — следующая попытка видна как `<job_id>-a<n+1>`; сами повторы не происходят. `submit --mode` — режим плана: `quick` (слои `content` и `measurements`), `targeted` (названные `--layers`), `full` (все три слоя, audio backend только из конфигурации), `auto` (по умолчанию: `targeted` с `--layers`, иначе `full`); план с `capability_version` пишется в `job.json`, runner перед запуском сверяет его с текущей конфигурацией и отказывает (`plan_refused`) плану, который расширил бы корни, сеть или backend; бюджеты плана шире конфигурации не отклоняются, а сужаются до значений конфигурации. `retry` по явной команде ставит задание из `failed` или `cancelled` обратно в конец очереди с теми же `job_id` и `request_id`, следующая попытка — `<job_id>-a<n+1>`; другое состояние — `not_retryable` (решение `docs/decisions/0018-job-modes-and-retry.md`).

`--task` — формулировка упражнения, `--intent` — замысел говорящего (что он хотел выразить),
если пользователь его назвал; текст из записи или транскрипта в них не подставляется. Оба поля —
то же, что аргументы `task` и `intent` у `speech_analyze`: они входят в identity запроса
(`request_mismatch` при повторе `--request-id` с другими значениями), а по `intent` `compare`
выставляет причину `intent_differs`.

`analyze` вызывает `app.analyze` и печатает итог по compact-ответу (`--json` — сам compact-ответ
с полями `run_dir` и `reused`); run-каталог печатается относительно data root (`runs/<run_id>`). Код возврата
— по `status` результата: 0 `complete`, 3 `partial`, 4 `blocked`, 1 `failed`. `compare` вызывает
`app.compare` (критерии — `docs/decisions/0006-compare-rules.md`) и печатает вердикт, все причины
и файл `compare/<id>.json` (`--json` — сравнение целиком с полем `path`), код 0 при любом вердикте.
Отказ — ошибка конфигурации, путь вне allowed roots, вход не аудио, занятый lock, чужой
`request_id`, неизвестный run — одна строка в stderr `sound-scout <команда>: <код причины>:
<сообщение>`, код 1, без трассировки; `--verbose` добавляет в stderr путь конфигурации, полный путь
run-каталога и трассировку. Stdout содержит только ответ.

Анализ выполняет `sound_scout.app.analyze` — общая точка входа CLI и MCP. На один data root
одновременно идёт один анализ: lock `<data_root>/analyze.lock`, второй запрос сразу получает
`Busy` без ожидания (очередь заданий — `jobs`, выше). Этапы выполняет дочерний процесс-worker; его срок —
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
uv run pytest             # -m "not slow" — быстрый набор (~3 мин): без реальных моделей, долгого синтеза, сухого стека GPU и тестов от 0,8 с
```

Чистая установка, проверка истории на приватные данные и ресурсный профиль — `docs/environment.md`:
`scripts/clean-install-check.sh`, `scripts/scan-history.sh`, `scripts/profile.py`.

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
