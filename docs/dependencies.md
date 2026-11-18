# Зависимости

Полный перечень того, от чего зависит Sound Scout: Python-пакеты, системные программы, runtime
модели и файлы весов. Правило проекта: **любая новая или обновлённая зависимость вносится сюда в
том же коммите**, с версией, источником, лицензией и контрольной суммой. Для весов и бинарей
указывается ровно тот файл, который используется, а не «последняя версия». Абсолютные пути
конкретной станции сюда не пишутся; ниже — пути по умолчанию относительно домашнего каталога
оператора.

Обновлено: 2026-11-18.

## 1. Python-пакеты

Источник истины — `pyproject.toml` и `uv.lock` (`uv sync --locked`). Здесь — прямые зависимости и
версии в lock на дату обновления.
SHA-256 каждого файла пакета — в `uv.lock`. `mcp` тянет `pydantic`, `starlette`, `uvicorn`, `sse-starlette`, `pyjwt`, `opentelemetry-api` (без SDK и экспортёра); stdio сервер использует только `mcp.server.stdio`, middleware OpenTelemetry у него убран.

| Пакет | Версия в `uv.lock` | Зачем | Лицензия |
|---|---|---|---|
| `faster-whisper` | 1.2.1 | локальный ASR; поставляет ONNX-файл Silero VAD | MIT |
| `ctranslate2` (через faster-whisper) | 4.9.0 | движок inference Whisper, CPU/int8 | MIT |
| `av` | 19.0.2 | зависимость faster-whisper | BSD-3-Clause |
| `onnxruntime` | 1.32.0 | Silero VAD; телеметрия выключается `ORT_DISABLE_TELEMETRY=1` (`guard.disable_telemetry`) | MIT |
| `praat-parselmouth` | 0.4.7 | F0 (Praat), формантный контроль `smoke` | GPL-3.0 |
| `numpy` | 2.5.5 | сигналы и измерения | BSD-3-Clause |
| `jsonschema` | 4.26.0 | контракт результата, манифест корпуса, разметка | MIT |
| `mcp` | 2.3.0 | MCP stdio сервер `sound-scout serve` (`docs/mcp.md`); последний стабильный релиз на 2026-11-18; wheel `mcp-2.3.0-py3-none-any.whl` SHA-256 `dd0c44c089d16453e8ae31a3877a0054d7a2314caaa81f5e0541b9b1734b2377` | MIT |
| `mcp-types` (через mcp) | 2.3.0 | типы протокола MCP; wheel SHA-256 `968efdbdaedfab06adae40d378a34395f1090c5921d4be3c9cde283aaf76d91d` | MIT |
| `pytest` (dev) | 9.2.0 | тесты | MIT |
| `ruff` (dev) | 0.17.5 | lint и format | MIT |

## 2. Системные программы

| Программа | Версия на станции разработки | Зачем | Как ставится |
|---|---|---|---|
| FFmpeg (`ffmpeg`, `ffprobe`) | 8.0.1 | декодирование входа, ресемплинг, клипы | пакет дистрибутива (`apt install ffmpeg`) |
| `uv` | 0.13.1 | среда и lock | https://docs.astral.sh/uv/ |
| llama.cpp | v1.0.0, сборка `b12359` (commit `b1b770d90`) | runtime audio backend (раздел 3) | архив релиза, раздел 3.2 |

CI (`.github/workflows/ci.yml`): `actions/checkout@v7.0.1`, `astral-sh/setup-uv@v10.3.3`, FFmpeg из
apt runner'а.

## 3. Audio-understanding backend

Решение владельца от 2026-11-18 (`docs/spec/REQUIREMENTS.md` раздел 9,
`docs/decisions/0001-audio-backend.md`): **Gemma 4 E4B-it** в GGUF через llama.cpp, только локально,
только CPU на станции разработки.

### 3.1. Веса

Репозиторий `ggml-org/gemma-4-E4B-it-GGUF` на Hugging Face, ревизия
`8903cb408c7df39cde2ed15fac397e6d7be92951` (lastModified 2026-11-12). Лицензия Apache-2.0,
не gated. Каталог по умолчанию:
`~/.cache/sound-scout/models/ggml-org--gemma-4-E4B-it-GGUF/<ревизия>/`.

| Файл | Байт | SHA-256 | Назначение |
|---|---|---|---|
| `gemma-4-E4B-it-Q8_0.gguf` | 8 031 237 344 | `49008e7cc7edadf154c8748904a89ee86eb6905d220aac568f6b7b92ca6aa0c6` | языковая модель, Q8_0 |
| `mmproj-gemma-4-E4B-it-BF16.gguf` | 991 552 288 | `fc1c6a1b566041bf41268b388bf59f0431ab86b3f4a7e34e996c0b4772509e37` | аудио/визуальный проектор (энкодер USM-Conformer), BF16 |

Загрузка — только по прямому URL `https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/<ревизия>/<файл>`
с проверкой SHA-256 после загрузки. Флаги llama.cpp `-hf`, `-mu`, `-mmu` (загрузка из сети при
запуске) не используются: модель всегда подаётся локальными `-m` и `--mmproj`.

Не используются: `gemma-4-E4B-it-BF16.gguf` (15 GB, вне бюджета), `Q4_0` и `mmproj-…-Q8_0`
(запасной облегчённый вариант, решение владельца), `mtp-*` (speculative decoding, не нужен).

### 3.2. Runtime

llama.cpp v1.0.0 — последний стабильный релиз на 2026-11-18; его бинарная сборка `b12359`
(`nightly-tag.txt` релиза v1.0.0). Архив `llama-b12359-bin-ubuntu-x64.tar.gz` (CPU, x64),
23 311 084 байт, SHA-256
`0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e`. Каталог по умолчанию:
`~/.local/opt/llama.cpp/b12359/`. Лицензия MIT.

Из сборки используется только `llama-server` (`src/sound_scout/eval/gate0.py`; запуск и результат —
`docs/decisions/0007-gate0-result.md`). `llama-mtmd-cli` не используется: шаблон чата Gemma 4 он
без `--jinja` отклоняет, а с `--jinja` шаблон включает режим рассуждений (`<|think|>`). Сервер
запускается с `--jinja` и локальными `-m`/`--mmproj`; аудио подаётся в `POST /v1/chat/completions`
частью сообщения `input_audio` (WAV в base64, `"format": "wav"`). В запросе передаётся
`"chat_template_kwargs": {"enable_thinking": false}` — просьба подавить режим рассуждений, а не
гарантия: в Gate 0 на transcript-only запросах варианта A сервер всё равно вернул
`reasoning_content` (`docs/decisions/0007-gate0-result.md`, разделы 4 и 6).
Мультимодальность `llama-server` помечена в документации llama.cpp как experimental.

Сервер слушает только `127.0.0.1` на свободном порту, один анализ за раз (`--parallel 1`).

### 3.3. Ограничения модели, влияющие на код

- Один аудиовход — не длиннее 30 с (Gemma audio guide); вход 16 kHz mono. Длинная запись режется
  на клипы ≤ 30 с по речевым интервалам, offsets пересчитываются на исходную шкалу (R03).
- Таймкодов модель не выдаёт: локализация строится по клипу и его границам, наблюдение без
  локализации остаётся `unlocalized` (R05).
- Русский: в техотчёте есть русский ASR (FLEURS ru WER 0.068) и перевод CoVoST ru→en; свидетельств
  оценки произношения и просодии нет — это проверяет Gate 0 spike.

## 4. Модели, уже лежащие на станции разработки

| Модель | Источник и ревизия | Где | Использование |
|---|---|---|---|
| Whisper large-v3-turbo (CTranslate2) | `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, ревизия `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`; `model.bin` SHA-256 `e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da` | кэш Hugging Face (`~/.cache/huggingface/hub`) | ASR, `local_files_only=True` |
| Silero VAD (ONNX) | файл из пакета `faster-whisper` 1.2.1 | внутри пакета | речевые интервалы |

Скачивание любых других весов или ревизий требует решения владельца (R09).
