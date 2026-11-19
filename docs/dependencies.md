# Зависимости

Полный перечень того, от чего зависит Sound Scout: Python-пакеты, системные программы, runtime
модели, файлы весов и открытые данные квалификационного корпуса. Правило проекта: **любая новая или обновлённая зависимость вносится сюда в
том же коммите**, с версией, источником, лицензией и контрольной суммой. Для весов и бинарей
указывается ровно тот файл, который используется, а не «последняя версия». Абсолютные пути
конкретной станции сюда не пишутся; ниже — пути по умолчанию относительно домашнего каталога
оператора.

Обновлено: 2026-11-19.

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

`ffmpeg` и `ffprobe` ищутся через `PATH` процесса (`shutil.which` в `src/sound_scout/doctor.py`,
имена без пути в argv `src/sound_scout/proc.py`): первый найденный в `PATH` и есть используемая
версия. Под Hermes MCP-сервер получает `PATH` Hermes, где первым стоит его собственный FFmpeg
(`~/.hermes/tools/…`), поэтому запись `speech-review` задаёт `env.PATH` из системных каталогов
(`docs/hermes-install.md`, «Регистрация в Hermes»).

CI (`.github/workflows/ci.yml`): `actions/checkout@v7.0.1`, `astral-sh/setup-uv@v10.3.3`, FFmpeg из
apt runner'а.

## 3. Audio-understanding backend

Решение владельца от 2026-11-18 (`docs/spec/REQUIREMENTS.md` раздел 9,
`docs/decisions/0001-audio-backend.md`): **Gemma 4 E4B-it** в GGUF через llama.cpp, только локально,
только CPU на станции разработки — кандидат для Gate 0.

Gate 0 для неё закончился вердиктом **NOT READY** (`docs/decisions/0007-gate0-result.md` §5):
backend не упакован, `backend` остаётся `none`, `doctor` для `audio_review` —
`permission_required`; замену выбирает владелец
([#68](https://github.com/renidere/sound-scout-plugin/issues/68)). Записи о весах и runtime ниже
остаются: веса лежат на станции разработки, по ним работает `sound_scout.eval.gate0`.

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
  оценки произношения и просодии нет; Gate 0 spike дал NOT READY
  (`docs/decisions/0007-gate0-result.md` §5).

## 4. Модели, уже лежащие на станции разработки

| Модель | Источник и ревизия | Где | Использование |
|---|---|---|---|
| Whisper large-v3-turbo (CTranslate2) | `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, ревизия `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`; `model.bin` SHA-256 `e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da` | кэш Hugging Face (`~/.cache/huggingface/hub`) | ASR, `local_files_only=True` |
| Silero VAD (ONNX) | файл из пакета `faster-whisper` 1.2.1 | внутри пакета | речевые интервалы |

Скачивание любых других весов или ревизий требует решения владельца (R09).

## 5. Данные квалификационного корпуса

Открытые наборы для пилотного корпуса `pilot-1` (Gate 1, T06). Загрузку разрешил владелец
2026-11-19 ([#29](https://github.com/renidere/sound-scout-plugin/issues/29),
[#78](https://github.com/renidere/sound-scout-plugin/issues/78)) ровно для файлов ниже; другие
файлы этих наборов не скачиваются. Данные лежат в `data/corpus/sources/<набор>/` вне git. Их
скачивает `scripts/fetch_corpus_sources.py` по таблице `sound_scout.eval.corpus.SOURCES`:
размер, сумма издателя и SHA-256 проверяются по потоку и ещё раз по файлу, перечитанному с диска;
`verify` перехэширует файлы на месте. Расхождение таблицы `SOURCES` и этого раздела ловит
`tests/test_dependencies_doc.py`. Сумма издателя: у Zenodo — MD5, у MS-SNSD — git blob SHA-1 на
закреплённом коммите. Лицензия DEMAND расходится: текст записи Zenodo — CC BY-SA 3.0, поле
лицензии — CC BY 4.0; принята более строгая CC BY-SA 3.0. Шумы MS-SNSD — Freesound CC0 по README
набора (код MIT); источник отдельного файла README не называет. Как из них собран корпус —
`docs/decisions/0009-corpus-pilot-1.md`.

| Набор | Версия / ревизия | Файл | Источник (URL) | Лицензия | Байт | Сумма издателя | SHA-256 |
|---|---|---|---|---|---|---|---|
| RINCO (Русский интонационный корпус) | v0.1, DOI 10.5281/zenodo.8119302 | `RINCO_Russian_Intonation_Corpus-v01.zip` | https://zenodo.org/records/8119302/files/RINCO_Russian_Intonation_Corpus-v01.zip?download=1 | CC BY 4.0 | 291339598 | `md5:63ff0a4366e82cfaac21a220dba5f27b` | `243688e816bfdf9adbfd65ef5c6104acccd0742ec9bb78b00edb2a299a7148f1` |
| RINCO (Русский интонационный корпус) | v0.1, DOI 10.5281/zenodo.8119302 | `help-markup-ru.pdf` | https://zenodo.org/records/8119302/files/help-markup-ru.pdf?download=1 | CC BY 4.0 | 101972 | `md5:ba07834ca3e32193198688eb1644e479` | `78ebf0c95050f86abe4aeb9daa142f70dfb1a8f42b6d5357078f37af1e094218` |
| RINCO (Русский интонационный корпус) | v0.1, DOI 10.5281/zenodo.8119302 | `rubrikator-2011.pdf` | https://zenodo.org/records/8119302/files/rubrikator-2011.pdf?download=1 | CC BY 4.0 | 217274 | `md5:7cf8332d1b7ebf46cce2ef5501f3bee5` | `58eb4b0a4d953f2298486fa3d32ba052d56dd17132e87043419399262fa2e189` |
| DEMAND | 1.0, DOI 10.5281/zenodo.1227121 | `DLIVING_16k.zip` | https://zenodo.org/records/1227121/files/DLIVING_16k.zip?download=1 | CC BY-SA 3.0 (текст записи; поле Zenodo — CC BY 4.0, берётся более строгая) | 80170627 | `md5:46741384d9e434a0bd8b3ec1830b6052` | `2b1726fe06e41551ce2397f2aaf3e4fb692c912d81914b04708df0bbb5252338` |
| DEMAND | 1.0, DOI 10.5281/zenodo.1227121 | `OOFFICE_16k.zip` | https://zenodo.org/records/1227121/files/OOFFICE_16k.zip?download=1 | CC BY-SA 3.0 (текст записи; поле Zenodo — CC BY 4.0, берётся более строгая) | 88995191 | `md5:7b61cc2d182d5a654cb9c3101ddd4041` | `7570c952f621990d0d71b8398107d9a78066254f883a5569abec9c233ffda358` |
| DEMAND | 1.0, DOI 10.5281/zenodo.1227121 | `DKITCHEN_16k.zip` | https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip?download=1 | CC BY-SA 3.0 (текст записи; поле Zenodo — CC BY 4.0, берётся более строгая) | 110501049 | `md5:7ffbf52d7f4699f96927846103dc8788` | `9d68e46d2709847c59580770e2f8aa160607ebff7475788174aa1080332cc845` |
| MS-SNSD | commit fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742, noise_test/Babble_2.wav | `Babble_2.wav` | https://raw.githubusercontent.com/microsoft/MS-SNSD/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742/noise_test/Babble_2.wav | CC0 (шумы Freesound по README MS-SNSD; код MIT) | 981122 | `git-blob-sha1:3c426baec1c75b4866f88e256b4e55165ae24ccb` | `b7f9872b1e4aef0949672d577cdd00132273c2b53efb4a23add549ee0cbbad90` |
| MS-SNSD | commit fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742, noise_test/Babble_6.wav | `Babble_6.wav` | https://raw.githubusercontent.com/microsoft/MS-SNSD/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742/noise_test/Babble_6.wav | CC0 (шумы Freesound по README MS-SNSD; код MIT) | 686764 | `git-blob-sha1:8efd80a6a452657dc7b8097d1a0c59560d195fdf` | `c355c815d3cc34e0826fb1ba72fab8c3d221d7be4b25e629932b8a7e083d4af5` |
| MS-SNSD | commit fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742, noise_train/NeighborSpeaking_2.wav | `NeighborSpeaking_2.wav` | https://raw.githubusercontent.com/microsoft/MS-SNSD/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742/noise_train/NeighborSpeaking_2.wav | CC0 (шумы Freesound по README MS-SNSD; код MIT) | 706052 | `git-blob-sha1:aba9083a05dc576ed56feb5dfa89bd6f67586a31` | `dff6281d248beeff5458b4fcb71387c19f62982eecafd00b78e3a0a6a8c6dc52` |
| MS-SNSD | commit fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742, noise_train/NeighborSpeaking_4.wav | `NeighborSpeaking_4.wav` | https://raw.githubusercontent.com/microsoft/MS-SNSD/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742/noise_train/NeighborSpeaking_4.wav | CC0 (шумы Freesound по README MS-SNSD; код MIT) | 760012 | `git-blob-sha1:f8009b35aff8a2bcbd6a069852a8e773e60c966d` | `8b179254d5b8c039563cf5206531dbcc0bb24ef7bf077456c503d17d4d083d47` |
