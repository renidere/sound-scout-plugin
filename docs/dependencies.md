# Зависимости

Полный перечень того, от чего зависит Sound Scout: Python-пакеты, системные программы, runtime
модели, файлы весов и открытые данные квалификационного корпуса. Правило проекта: **любая новая или обновлённая зависимость вносится сюда в
том же коммите**, с версией, источником, лицензией и контрольной суммой. Для весов и бинарей
указывается ровно тот файл, который используется, а не «последняя версия». Абсолютные пути
конкретной станции сюда не пишутся; ниже — пути по умолчанию относительно домашнего каталога
оператора.

Обновлено: 2026-11-22.

## 1. Python-пакеты

Источник истины — `pyproject.toml` и `uv.lock` (`uv sync --locked`). Здесь — прямые зависимости и
версии в lock на дату обновления.
SHA-256 каждого файла пакета — в `uv.lock`. Extra `gate0-borealis` (`uv sync --locked --extra gate0-borealis`) нужен только для Gate 0 spike Borealis-5b-it (#68) и по умолчанию не ставится, CI его не тянет; кроме трёх пакетов ниже он приносит `sympy`, `networkx`, `jinja2`, `setuptools`, `regex`, `typer`/`rich` (через `transformers`), а `tokenizers` и `huggingface-hub` уже были в lock через faster-whisper. Extra `gate0-gigachat` (`uv sync --locked --extra gate0-gigachat`) нужен только для Gate 0 spike GigaChat3.1-Audio-10B-A1.8B (#85), тоже не ставится по умолчанию и в CI; он объявлен в `[tool.uv] conflicts` с `gate0-borealis` (разные `torch` и `transformers`), поэтому в lock у `torch` и `transformers` по две версии. Кроме четырёх пакетов ниже он приносит `transformers` 4.57.6 со своими зависимостями: `tokenizers` 0.22.2 (wheel `cp39-abi3-manylinux_2_17_x86_64` SHA-256 `369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67`) и `huggingface-hub` 0.36.2 (SHA-256 `48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270`) вместо версий faster-whisper, `requests`, `urllib3`, `certifi`, `charset-normalizer`, `setuptools`, `sympy`, `networkx`, `jinja2`, `regex`. Пины `==`: `torchaudio` 2.11.0 — последний CPU-wheel для cp313 в `https://download.pytorch.org/whl/cpu`, и он требует ровно `torch` 2.11.0; `transformers` 4.57.6 — версия, на которой upstream проверил код модели (карточка), а в `transformers` 5 слои MoE `DeepseekV3` хранят экспертов слитыми тензорами, и ключи checkpoint не совпадут с каркасом (`docs/decisions/0011-gate0-gigachat-audio.md` §1). `mcp` тянет `pydantic`, `starlette`, `uvicorn`, `sse-starlette`, `pyjwt`, `opentelemetry-api` (без SDK и экспортёра); stdio сервер использует только `mcp.server.stdio`, middleware OpenTelemetry у него убран.

Пакет плагина Hermes (`hermes/plugin/sound-scout/dashboard/plugin_api.py`, #101) исполняется в процессе gateway и берёт FastAPI и Starlette из окружения Hermes (на станции разработки 0.133.1 и 1.9.0); python-multipart ему не нужен с #168 (`POST /jobs` принимает JSON `data_url`). Тесты плагина берут Starlette 1.10.0 из lock транзитивно через `mcp`, новых пакетов в lock нет.

Extra `corpus-tts` (`uv sync --locked --extra corpus-tts`) нужен только для синтеза клипов
корпуса (`python -m sound_scout.eval.synthetic`, #81) и по умолчанию не ставится, CI его не тянет;
`protobuf` и `typing-extensions` для `onnx` уже были в lock.

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
| `torch` (extra `gate0-borealis`) | 2.15.0+cpu | runtime Borealis-5b-it для Gate 0 spike #68 (раздел 3.1); CPU-сборка из индекса `https://download.pytorch.org/whl/cpu`, wheel `torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl` SHA-256 `0f6f69701b2d38ddc412364dfb92e2e13e74960d33fa3573b4930ced626e81c7` | BSD-3-Clause и др. (`license_expression` PyPI: Apache-2.0, BSD-2/3-Clause, MIT, BSL-1.0, ISC, Zlib) |
| `transformers` (extra `gate0-borealis`) | 5.23.0 | Whisper-энкодер, Qwen3 и токенизатор для Borealis-5b-it; wheel SHA-256 `018e82fd07490ad5bb58414f65e31576033dc7f36bed10c574843f69b90991bf` | Apache-2.0 |
| `safetensors` (extra `gate0-borealis`) | 0.8.0 | потоковое чтение `model.safetensors` Borealis-5b-it; wheel (`cp310-abi3-manylinux_2_17_x86_64`) SHA-256 `fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774` | Apache-2.0 |
| `torch` (extra `gate0-gigachat`) | 2.11.0+cpu | runtime GigaChat3.1-Audio-10B-A1.8B для Gate 0 spike #85 (раздел 3.1); CPU-сборка из индекса `https://download.pytorch.org/whl/cpu`, wheel `torch-2.11.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl` SHA-256 `45025d7752dbc6b4c784c03afaee9c5f19730ce084b2e43fc9a2fe1677d9ff86`; для macOS в lock — `torch` 2.11.0 того же индекса | BSD-3-Clause и др. (как `torch` выше) |
| `torchaudio` (extra `gate0-gigachat`) | 2.11.0+cpu | log-mel (`MelSpectrogram`) процессора GigaChat3.1-Audio; CPU-сборка из того же индекса, wheel `torchaudio-2.11.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl` SHA-256 `3c0175d0ed054bf0dc3b154a744b1a127c94291b3f3b7bdd0639b4b238c89445`; для macOS в lock — `torchaudio` 2.11.0 | BSD-2-Clause |
| `transformers` (extra `gate0-gigachat`) | 4.57.6 | `DeepseekV3ForCausalLM`, токенизатор и шаблон GigaChat3.1-Audio; wheel SHA-256 `4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550` | Apache-2.0 |
| `safetensors` (extra `gate0-gigachat`) | 0.8.0 | потоковое чтение шардов декодера GigaChat3.1-Audio; тот же wheel, что выше | Apache-2.0 |
| `piper-tts` (extra `corpus-tts`) | 1.8.0 | TTS для синтетических клипов с дефектами дикции корпуса `pilot-1` (#81, раздел 6), только CPU (`use_cuda=False`); espeak-ng встроен в wheel; последний стабильный релиз на 2026-11-19; wheel `piper_tts-1.8.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl` SHA-256 `25b4d3f31ff70c8fa7151908e00aaa5650cbdf16bca8fcf21299f3941b89a7d3` | GPL-3.0-or-later |
| `onnx` (extra `corpus-tts`) | 1.23.2 | extra `alignment` у piper-tts: патч модели в памяти, чтобы Piper отдал выравнивание (отсчёты на фонему); файл модели не меняется; последний стабильный релиз на 2026-11-19; wheel `onnx-1.23.2-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` SHA-256 `7abf381d278f31ac62487fddedc9dd42da842dce94d5d43536836ee3efdf4a2b` | Apache-2.0 |
| `ml-dtypes` (через onnx) | 0.7.0 | типы данных onnx; wheel (`cp313-manylinux_2_27_x86_64`) SHA-256 `5fb56999e1a03f47af5583af4089e00d04b68f87d4fadd659a5927ecd7687c3d` | Apache-2.0 |
| `pathvalidate` (через piper-tts) | 3.3.1 | проверка имён файлов в piper-tts; wheel `pathvalidate-3.3.1-py3-none-any.whl` SHA-256 `5263baab691f8e1af96092fa5137ee17df5bdfbd6cff1fcac4d6ef4bc2e1735f` | MIT |
| `pytest` (dev) | 9.2.0 | тесты | MIT |
| `ruff` (dev) | 0.17.5 | lint и format | MIT |

## 2. Системные программы

| Программа | Версия на станции разработки | Зачем | Как ставится |
|---|---|---|---|
| FFmpeg (`ffmpeg`, `ffprobe`) | 8.0.1 | декодирование входа, ресемплинг, клипы | пакет дистрибутива (`apt install ffmpeg`); в CI — абзац CI ниже |
| `uv` | 0.13.1 | среда и lock | https://docs.astral.sh/uv/ |
| llama.cpp | v1.0.0, сборка `b12359` (commit `b1b770d90`) | runtime audio backend (раздел 3) | архив релиза, раздел 3.2 |
| Node.js | 24.21.0 (минимум 20: `node:test`, `vm.SourceTextModule` под `--experimental-vm-modules`) | только тесты зонда Spike A и рекордера Desktop (`tests/js/probe_a.test.mjs`, `tests/js/recorder.test.mjs` на общем загрузчике `tests/js/load_plugin.mjs`; запуск из `tests/test_hermes_probe_a.py` и `tests/test_hermes_plugin.py`); npm-пакетов нет | https://nodejs.org/; в CI — из образа `ubuntu-latest` |

`ffmpeg` и `ffprobe` ищутся через `PATH` процесса (`shutil.which` в `src/sound_scout/doctor.py`,
имена без пути в argv `src/sound_scout/proc.py`): первый найденный в `PATH` и есть используемая
версия. Под Hermes MCP-сервер получает `PATH` Hermes, где первым стоит его собственный FFmpeg
(`~/.hermes/tools/…`), поэтому запись `speech-review` задаёт `env.PATH` из системных каталогов
(`docs/hermes-install.md`, «Регистрация в Hermes»).

espeak-ng отдельно не ставится: фонемизатор и его данные встроены в wheel `piper-tts` (extra
`corpus-tts`, раздел 1), системный пакет не нужен.

CI (`.github/workflows/ci.yml`): `actions/checkout@v7.0.1`, `astral-sh/setup-uv@v10.3.3`,
`actions/cache/restore@v6.1.0` и `actions/cache/save@v6.1.0` (последний стабильный релиз
`actions/cache` на 2026-11-24, тег на commit `55cc8345863c7cc4c66a329aec7e433d2d1c52a9`, MIT),
Node.js — предустановленный в образе runner'а (без `actions/setup-node`). Без Node тесты зонда и
рекордера в CI падают, а не пропускаются (фикстура `node` в `tests/conftest.py`). FFmpeg в CI —
пакет дистрибутива Ubuntu из архива `archive.ubuntu.com` (`resolute/universe`): на образе
`ubuntu-26.04` версии 20261116.202.1 это `ffmpeg` 7:8.0.1-3ubuntu2 и 89 пакетов .deb (69 MB) вместе
с зависимостями. Jobs `checks` и `clean-install` ставят его одним локальным действием
`.github/actions/install-ffmpeg` (#183): набор .deb лежит в кэше Actions под ключом из `ImageOS`,
архитектуры, уровня CPU, `ImageVersion` и SHA-256 `install.sh`; при попадании набор ставит
`dpkg -i` без `apt-get` и без зеркала apt. Образ runner'а без своего набора берёт по `restore-keys`
набор другого образа той же ОС, архитектуры и уровня CPU (#184): `dpkg -i` ставит его без зеркала,
не заменяя пакеты образа, затем ffmpeg кодирует тестовый сигнал, и набор сохраняется под ключом
этого образа. Негодный набор снимается `dpkg --purge`, его .deb идут в кэш apt, и с зеркала
докачивается остальное. Без набора всё скачивается с зеркала с ограниченными повторами (#65, #70),
и в кэш идут только .deb установленных версий; набор без единственного `ffmpeg_*.deb` не
сохраняется (предупреждение в логе).

## 3. Audio-understanding backend

Решение владельца от 2026-11-18 (`docs/spec/REQUIREMENTS.md` раздел 9,
`docs/decisions/0001-audio-backend.md`): **Gemma 4 E4B-it** в GGUF через llama.cpp, только локально,
только CPU на станции разработки — кандидат для Gate 0.

Gate 0 для неё закончился вердиктом **NOT READY** (`docs/decisions/0007-gate0-result.md` §5):
backend не упакован, `backend` остаётся `none`, `doctor` для `audio_review` —
`permission_required`; замену владелец выбирал в
[#68](https://github.com/renidere/sound-scout-plugin/issues/68) (закрыта), следующий ограниченный
эксперимент — [#119](https://github.com/renidere/sound-scout-plugin/issues/119). Записи о весах и
runtime ниже остаются: веса лежат на станции разработки, по ним работает `sound_scout.eval.gate0`.

Запасной кандидат **Borealis-5b-it** (`docs/spec/REQUIREMENTS.md` раздел 9; решение владельца в
[#68](https://github.com/renidere/sound-scout-plugin/issues/68) (закрыта)) проверен тем же протоколом и тоже получил **NOT READY**
(`docs/decisions/0010-gate0-borealis.md` §5): ничего не упаковано, `backend` остаётся `none`.
Его веса удалены со станции разработки 2026-11-19 (техническая заметка
[#85](https://github.com/renidere/sound-scout-plugin/issues/85), чтобы освободить место); их
SHA-256 и runtime остаются записанными ниже (3.1, 3.2).

Кандидат вне бюджета памяти **GigaChat3.1-Audio-10B-A1.8B** (разрешение владельца на
[#85](https://github.com/renidere/sound-scout-plugin/issues/85), потолок 26 GB) проверен тем же
протоколом и правилом и тоже получил **NOT READY** (`docs/decisions/0011-gate0-gigachat-audio.md`
§5): не выполнено предусловие P3 правила (≥ 20 аудиотокенов на секунду клипа, у энкодера модели —
6,25 позиции в секунду); о слухе модели этот вердикт ничего не говорит. Ничего не упаковано,
`backend` остаётся `none`. Веса (23,7 GB) лежат на станции разработки (3.1), удалить их — решение
владельца.

Кандидат **Qwen3-Omni-30B-A3B-Instruct** (GGUF Q4_K_M и mmproj Q8_0; решение владельца
[7065525535](https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7065525535),
правило `GATE0-RULE v2`, потолок 26 GB) прошёл предусловия P1–P3 и получил **NOT READY**
(`docs/decisions/0023-gate0-qwen3-omni.md` §5): контроли C — 10 из 12, оба непрошедших пункта —
пустой или обрезанный ответ закрытого вопроса, выдуманной речи нет. Ничего не упаковано, `backend`
остаётся `none`. Веса (19,88 GB) лежат на станции разработки (3.1), удалить их — решение владельца.

Кандидат **Qwen2.5-Omni-7B** (GGUF Q8_0 и mmproj f16; тот же комментарий владельца, п. 2,
диагностический прогон по `GATE0-RULE v1`, потолок 18 GB) прошёл предусловия P1–P3 и получил
**NOT READY** (`docs/decisions/0024-gate0-qwen25-omni.md` §5): контроли C — прошли 7 из 12; `да` в
вопросе P на тишине и шуме, выдуманное наблюдение о произношении в варианте A на всех трёх
контролях. Ничего не упаковано,
`backend` остаётся `none`. Веса (10,74 GB) лежат на станции разработки (3.1), удалить их —
решение владельца.

Запасной кандидат **Gemma 4 12B-it** (GGUF Q8_0 и mmproj BF16; решение владельца
[7173265163](https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7173265163),
правило `GATE0-RULE v1`, потолок 18 GB) прошёл предусловия P1–P3 и получил **NOT READY**
(`docs/decisions/0031-gate0-gemma4-12b.md` §5): контроли C — прошли 7 из 12; `речь` на тоне в K1
и K2, вариант A на трёх контролях не отправлен. Сторож RSS (17,5e9) остановил прогон на запросе 32
из 42; вердикт от остановки не зависит. Ничего не упаковано, `backend` остаётся `none`. Веса
(12,84 GB) лежат на станции разработки (3.1), удалить их — решение владельца.

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

#### Qwen3-Omni-30B-A3B-Instruct — кандидат Gate 0 #144 (решение владельца 7065525535)

Решение владельца — [комментарий 7065525535](https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7065525535)
(2026-11-21): формальный Gate 0 по `GATE0-RULE v2`, загрузка GGUF Q4_K_M и mmproj Q8_0 с
ревизией, закреплённой при загрузке, потолок памяти 26 GB. Репозиторий
`ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF` на Hugging Face, ревизия
`6e35a28f4a19b18730f8949b0c579c6429649ab8` (lastModified 2026-04-13), не gated. Исходная модель
`Qwen/Qwen3-Omni-30B-A3B-Instruct`, ревизия `26291f793822fb6be9555850f06dfe95f2d7e695`. Лицензия
Apache-2.0 по карточке исходной модели (`license_name: apache-2.0`); у карточки GGUF поля лицензии
нет. Каталог по умолчанию:
`~/.cache/sound-scout/models/ggml-org--Qwen3-Omni-30B-A3B-Instruct-GGUF/<ревизия>/`. Итог —
`docs/decisions/0023-gate0-qwen3-omni.md`.

| Файл | Байт | SHA-256 | Назначение |
|---|---|---|---|
| `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf` | 18 557 053 952 | `d9e2876556e7873e02c0359f832432ee2d67ab7dd0cee3efe0f77fd7a1f4dd85` | языковая модель (thinker), Q4_K_M |
| `mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf` | 1 325 020 128 | `1104376db833f1e89c84834144ac3863340c2cd1ddaeddb39cb0247fb5c20c8d` | аудио/визуальный проектор (энкодер AuT, тип `qwen3a`), Q8_0 |

Загрузка — только по прямому URL `https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/<ревизия>/<файл>`
(`curl --proto =https --proto-redir =https`) в файл `.part`; после загрузки сверяются размер и
SHA-256 на диске с LFS sha256 из `GET /api/models/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/tree/<ревизия>`.
`gate0 run --runtime qwen3omni` пересчитывает SHA-256 перед стартом сервера (предусловие P2) по
константе `QWEN3OMNI_WEIGHTS`, равной этой таблице. Не используются: `Q8_0` (32,5 GB) и `bf16`
LLM, `mmproj-…-bf16.gguf`.

#### Qwen2.5-Omni-7B — диагностический Gate 0 #145 (решение владельца 7065525535)

Решение владельца — [комментарий 7065525535](https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7065525535)
(2026-11-21), п. 2: прогон по `GATE0-RULE v1` для понимания слабых мест, загрузка Q8_0 и mmproj
f16 закреплённой ревизии, потолок памяти 18 GB. Репозиторий `ggml-org/Qwen2.5-Omni-7B-GGUF` на
Hugging Face, ревизия `89b785438c8901d4635e42f50480ba5985a1bbf1` (lastModified 2025-05-26), не
gated. Исходная модель `Qwen/Qwen2.5-Omni-7B`, ревизия `ae9e1690543ffd5c0221dc27f79834d0294cba00`.
Лицензия: текст `LICENSE` исходной модели — Apache-2.0; карточка GGUF — `license_name:
qwen-research`; расхождение принято владельцем для этого локального прогона. Каталог по
умолчанию: `~/.cache/sound-scout/models/ggml-org--Qwen2.5-Omni-7B-GGUF/<ревизия>/`. Итог —
`docs/decisions/0024-gate0-qwen25-omni.md`.

| Файл | Байт | SHA-256 | Назначение |
|---|---|---|---|
| `Qwen2.5-Omni-7B-Q8_0.gguf` | 8 098 524 032 | `2cb12eb661bc8db5536463dd2dadd4a25c8bd42a2346bc1cefddd2e2245f641a` | языковая модель (thinker), Q8_0 |
| `mmproj-Qwen2.5-Omni-7B-f16.gguf` | 2 643 656 480 | `7ed062b35f15f3c4cacdac72d5291db2f908f178ba4fb19e210b555f233e069d` | аудио/визуальный проектор (энкодер Qwen2-Audio, тип `qwen2.5o`), f16 |

Загрузка — только по прямому URL `https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/<ревизия>/<файл>`
(`curl --proto =https`) в файл `.part`; после загрузки размер и SHA-256 на диске сверены с LFS
sha256 из API Hugging Face. `gate0 run --runtime qwen25omni` пересчитывает SHA-256 перед стартом
сервера (предусловие P2) по константе `QWEN25OMNI_WEIGHTS`, равной этой таблице. Не используются:
`Q4_K_M`, `f16` LLM (15,24 GB), `mmproj-…-Q8_0`.

#### Gemma 4 12B-it — Gate 0 #193 (решение владельца 7173265163)

Решение владельца — [комментарий 7173265163](https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7173265163)
(2026-11-25): загрузка двух файлов закреплённой ревизии и локальный прогон Gate 0 на CPU по
`GATE0-RULE v1`. Репозиторий `ggml-org/gemma-4-12B-it-GGUF` на Hugging Face, ревизия
`501d13dba70e3c02b1b12c78fd9e6b08ad413960` (lastModified 2026-11-12), не gated. Исходная модель
`google/gemma-4-12B-it`, ревизия `43c9d014b231474fa7212dbb48a762566c2b255d`. Лицензия Apache-2.0
(карточка GGUF `license: apache-2.0` и карточка исходной модели). Каталог по умолчанию:
`~/.cache/sound-scout/models/ggml-org--gemma-4-12B-it-GGUF/<ревизия>/`. Решение —
`docs/decisions/0031-gate0-gemma4-12b.md`.

| Файл | Байт | SHA-256 | Назначение |
|---|---|---|---|
| `gemma-4-12B-it-Q8_0.gguf` | 12 669 641 920 | `3730903965b3e57bcad0a387110322ce66409559c356a1ae1b61701024e5ce60` | языковая модель, Q8_0 |
| `mmproj-gemma-4-12B-it-BF16.gguf` | 175 115 648 | `b9d4500e269ea3ad8a83b1fc55f553ed8c6b9ed35309b86c2fe9c4d8b47f9440` | аудио/визуальный проектор (тип аудио `gemma4ua`), BF16 |

Всего 12 844 757 568 байт. Загрузка — только по прямому URL
`https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/<ревизия>/<файл>`
(`curl --proto =https --proto-redir =https`) в файл `.part`; после загрузки размер и SHA-256 на
диске сверяются с LFS sha256 из `GET /api/models/ggml-org/gemma-4-12B-it-GGUF/tree/<ревизия>`, и
только затем `.part` переименовывается. `gate0 run --runtime gemma12b` пересчитывает SHA-256 перед
стартом сервера (предусловие P2) по константе `GEMMA12B_WEIGHTS`, равной этой таблице. Не
используются: `BF16` LLM (23,8 GB), `Q4_0`, `mmproj-…-Q8_0`, `mtp-*`.

#### Qolda-AVL-5B — Gate 0 #194 (решение владельца 7173265163)

Решение владельца — [комментарий 7173265163](https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7173265163)
(2026-11-25): загрузка весов `issai/Qolda-AVL-5B` и закреплённого кода runtime, один локальный
прогон Gate 0 на CPU по `GATE0-RULE v1`. Репозиторий `issai/Qolda-AVL-5B-GGUF` на Hugging Face,
ревизия `e2c55b47e6dfcc224718e466691aeaa49ec56bd9` (lastModified 2026-08-19), не gated. Исходная
модель — `issai/Qolda-AVL-5B` @ `e09b9c23faee8429b4d503636176edfc4ce061a6` (не скачивается).
Лицензия Apache-2.0 (карточка GGUF `license: apache-2.0`). Каталог по умолчанию:
`~/.cache/sound-scout/models/issai--Qolda-AVL-5B-GGUF/<ревизия>/`, пути файлов — как в
репозитории. Решение — `docs/decisions/0032-gate0-qolda-avl-5b.md`.

| Файл | Байт | SHA-256 | Назначение |
|---|---|---|---|
| `Q8_0/Qolda-AVL-5B-Q8_0.gguf` | 4 280 406 368 | `613406fad8ab6969cffd9991826f52b6889b4bd38fd6ae493d946b6a7ab67083` | языковая модель (архитектура `qwen3vl`), Q8_0 |
| `mmproj/mmproj-Qolda-AVL-5B-F16.gguf` | 2 193 746 656 | `041aa58d1d39d2e0c3f4555c7dc8bf2e64798fb9328567c10fb7a9bf4bc96116` | аудио- и визуальный проектор (тип аудио `qwen3avl`), F16 |

Всего 6 474 153 024 байт. Загрузка — только по прямому URL
`https://huggingface.co/issai/Qolda-AVL-5B-GGUF/resolve/<ревизия>/<файл>`
(`curl --proto =https --proto-redir =https`) в файл `.part`; после загрузки размер и SHA-256 на
диске сверяются с LFS sha256 из `GET /api/models/issai/Qolda-AVL-5B-GGUF/tree/<ревизия>`, и только
затем `.part` переименовывается. `gate0 run --runtime qolda` пересчитывает SHA-256 перед стартом
сервера (предусловие P2) по константе `QOLDA_WEIGHTS`, равной этой таблице. Не используются:
`BF16`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `mmproj-…-vision-only-F16.gguf`, `imatrix`.

#### GigaChat3.1-Audio-10B-A1.8B — кандидат Gate 0 #85 (разрешение владельца 2026-11-19)

Разрешение владельца — тело [#85](https://github.com/renidere/sound-scout-plugin/issues/85)
(2026-11-19): кандидат вне бюджета памяти `docs/spec/REQUIREMENTS.md` §9 (≈ 22,5 GB в bf16),
загрузка файлов одной закреплённой ревизии на станцию разработки, прогон offline на CPU под
жёстким потолком 26 GB. Репозиторий `ai-sage/GigaChat3.1-Audio-10B-A1.8B` на Hugging Face,
ревизия `bf73d03a43bdf5118f5a4dbdc24ba6f56ac31cfb`. Лицензия MIT (карточка модели), не gated.
Каталог по умолчанию:
`~/.cache/sound-scout/models/ai-sage--GigaChat3.1-Audio-10B-A1.8B/<ревизия>/`. Итог spike —
`docs/decisions/0011-gate0-gigachat-audio.md`.

Хэшируемый набор — 14 файлов (23 723 198 064 байт, из них шесть файлов весов — 23 711 959 176);
эталон — LFS sha256 или git blob oid из
`GET https://huggingface.co/api/models/ai-sage/GigaChat3.1-Audio-10B-A1.8B/tree/<ревизия>?recursive=1`
(ответ лежит рядом с весами как `.hf-tree.json`):

| Файл | Байт | SHA-256 | Эталон HF API |
|---|---|---|---|
| `config.json` | 4 826 | `2a83735f1a6499f159b78aea6d1f88f773236a0ff2002be73610cd6676671f54` | git blob `392051491cdd7bfded5e0ba42be198fbd8531437` |
| `chat_template.jinja` | 28 973 | `12f52c126f74ffd3c3c38cc843bec48906b37b353b227c8c040a9e8e5bfc4a79` | git blob `b848e79669731649d6b6484387de5bd1462e6e11` |
| `modeling_gigachat_audio.py` | 61 057 | `ec96f0f9052b88616e427c9b75ed1cff6df98ed43176cdc11655ed6633b72e61` | git blob `b400917e009f3811aecc9e5a1795f1738f106b81` |
| `audio_encoder.safetensors` | 2 366 253 232 | `95b6fe65562c5b87d48b518f97a912dfb8860adf51783af39e5d13a0ce021965` | LFS sha256 (тот же) |
| `decoder/config.json` | 1 270 | `1bba4d4a6a9d6d3465f68166e7b3551383094cd8f993f7f964d5245deb14c2ea` | git blob `7e133ee0496f6c1ad6c39c6204e92598ed609fab` |
| `decoder/model.safetensors.index.json` | 458 679 | `8491cb6db08c983f6b47fc65c90d3906686bda59517474ea8b8e20d95bceb7f6` | git blob `5eff8858729525e46e316e6da142ce4c4acf5c14` |
| `decoder/model-00001-of-00005.safetensors` | 4 999 793 776 | `5a98fcd9ce4f563eac0bd3aa33587a90e546ffa0b585813a30f0b4ad52b5e489` | LFS sha256 (тот же) |
| `decoder/model-00002-of-00005.safetensors` | 4 996 400 960 | `9186d95fd53aa261847e2b4e707e87f03df0332269c2a27782120cafa0fdc206` | LFS sha256 (тот же) |
| `decoder/model-00003-of-00005.safetensors` | 4 996 401 664 | `2c0a0dda91744d0eb9358455660f568d0b4483bdecc3ce5b5e7441f039211f4e` | LFS sha256 (тот же) |
| `decoder/model-00004-of-00005.safetensors` | 4 996 401 664 | `af3668891d8be7bafb3bd7caa47c473139a39b78698840af92ed0db6d65a47c1` | LFS sha256 (тот же) |
| `decoder/model-00005-of-00005.safetensors` | 1 356 707 880 | `5e3f313efe2c60c1cefa2420e910792bd05d68b7ccb87cbf03495b49aad6175c` | LFS sha256 (тот же) |
| `decoder/tokenizer.json` | 10 680 799 | `9b12bf74d46dff311b65556c116167cb33f0e94cf2a4e2b7f82feeeae8eef596` | LFS sha256 (тот же) |
| `decoder/tokenizer_config.json` | 3 008 | `de7f61f4bd0b8241cf627a9002c3f0aa850ff16ca392f1ff548a24701f176994` | git blob `eee5447eca9f017a59b707e9160df5841d8db90f` |
| `decoder/special_tokens_map.json` | 276 | `5d4a6b397eb900972536128341670e9a499823c21291ffc3bf7f29f7a69dae0a` | git blob `d91fcde70f007f758e57bf674b33e7c820bdb514` |

Правило загрузки: файлы весов — официальным клиентом `huggingface_hub` (`hf_hub_download`,
`token=False`, ревизия закреплена, `local_dir` — каталог выше, без второй копии в общем кэше HF,
транспорт Xet выключен `HF_HUB_DISABLE_XET=1`, докачка неполного файла, повтор при обрыве); сразу
после загрузки сверяются размер и LFS sha256, только потом файл становится read-only. При
расхождении файл качается ещё один раз, второе расхождение — стоп. У малых файлов сверяется
`git hash-object --no-filters` с git blob oid. Перед smoke SHA-256 всех 14 файлов сверяются ещё
раз (`sha256sum -c`), а `gate0 run --runtime gigachat` пересчитывает их перед стартом сервера
(предусловие P2) по константе `GIGACHAT_WEIGHTS`, равной этой таблице.
`modeling_gigachat_audio.py` исполняется (энкодер, адаптер, процессор и `generate` модели),
поэтому входит в хэшируемый набор; сервер исполняет его только после сверки SHA-256 с константой
`MODELING_SHA256` и именно проверенные байты. Не читаются: `README.md`, `.gitattributes`,
`vllm_gigachat_audio.py`, `tokenizer.json`, `tokenizer_config.json` и `special_tokens_map.json`
верхнего уровня (по oid — копии файлов `decoder/`; токенизатор берётся из `decoder/`).

#### Borealis-5b-it — кандидат Gate 0 spike #68 (решение владельца 7003859372)

Решение владельца — [комментарий 7003859372](https://github.com/renidere/sound-scout-plugin/issues/68#issuecomment-7003859372)
(2026-11-19): запасной кандидат `docs/spec/REQUIREMENTS.md` §9, загрузка весов на станцию
разработки разрешена, ревизия — head репозитория на момент загрузки. Репозиторий
`Vikhrmodels/Borealis-5b-it` на Hugging Face, ревизия `66e8899f74a7cfa8931dbdbb8a907534b927ddb4`
(head на 2026-11-19, lastModified 2025-12-19). Лицензия Apache-2.0, не gated. Каталог по
умолчанию: `~/.cache/sound-scout/models/Vikhrmodels--Borealis-5b-it/<ревизия>/`. Итог spike —
`docs/decisions/0010-gate0-borealis.md`.

Загружены ровно 9 файлов (10 149 393 702 байт); эталон — LFS sha256 или git blob oid из
`GET https://huggingface.co/api/models/Vikhrmodels/Borealis-5b-it/tree/<ревизия>?recursive=1`
(ответ лежит рядом с весами как `.hf-tree.json`):

| Файл | Байт | SHA-256 | Эталон HF API |
|---|---|---|---|
| `model.safetensors` | 10 133 495 512 | `ceaafa32e56e93bc43eb4807c226ef08942718a3b37747f2f9183f7d5d063b11` | LFS sha256 (тот же) |
| `tokenizer.json` | 11 423 035 | `fcb4fb9e735fe694250e8eb740459e5a7d515387a7497ae78af2177fa3fc8a21` | LFS sha256 (тот же) |
| `config.json` | 1 083 | `c5708c8e3f02672adccee2e9399f5c6e26d2fbec5bc3e9cd4964755af9a6ed01` | git blob `0410b9706bf1fa3de8830cc21ff49e068e590776` |
| `tokenizer_config.json` | 5 808 | `5712797be9e1c0d5cde3fe07400b3aff5ae4b895d2f785fdaa33c1e4e010ed45` | git blob `28b71d84daf75258197d80be453d4c4d64c049f4` |
| `special_tokens_map.json` | 2 546 | `54c79929443cd1849332ad1d811acab82dd00a876b74b2e520d40b1ccae48ea1` | git blob `b200b5ea3fa3bb994448f8042ef732d101b2d55a` |
| `added_tokens.json` | 762 | `3bed5ae134cddb1f871e4143159dbc898fa2f1e63da4850d4fbfe136c5ef46c4` | git blob `739eb403f448db705e6412b5c1e8df4c4d187487` |
| `vocab.json` | 2 776 833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | git blob `4783fe10ac3adce15ac8f358ef5462739852c569` |
| `merges.txt` | 1 671 853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` | git blob `31349551d90c7606f325fe0f11bbb8bd5fa0d7c7` |
| `modeling_borealis.py` | 16 270 | `802073ea6547cc122e92ed3912a209dfedd13268143f15d7fe1eb07d739c0a02` | git blob `98f1f9b561ee9259f98435ac44a6c821ccdaf5ad` |

Правило загрузки: каждый файл — `curl -fL --proto =https` по прямому URL
`https://huggingface.co/Vikhrmodels/Borealis-5b-it/resolve/<ревизия>/<файл>` в `<файл>.part`;
сразу после загрузки проверяются размер и LFS sha256 (`sha256sum`) или git blob oid
(`git hash-object --no-filters`), только потом файл переименовывается и становится read-only.
При расхождении файл качается ещё один раз, второе расхождение — стоп. Перед загрузкой модели
SHA-256 сверяются ещё раз (`sha256sum -c`), а `gate0 run --runtime borealis` пересчитывает их
перед стартом сервера (предусловие P2) по константе `BOREALIS_WEIGHTS`, равной этой таблице.
`modeling_borealis.py` исполняется (сервер импортирует его по пути), поэтому входит в
хэшируемый набор. Не загружаются `processing_borealis.py`, `push_model.py`,
`Borealis_Demo.ipynb`, `README.md`, `generation_config.json`, `vllm_borealis/*`; из
`Qwen/Qwen3-4B` и `openai/whisper-large-v3` не загружается ничего — нужные числа их конфигов
записаны константами в `src/sound_scout/eval/gate0_borealis.py`.

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

Тот же `llama-server` b12359 запускает Qwen3-Omni-30B-A3B-Instruct (`gate0 run --runtime
qwen3omni`, #144): тот же argv с файлами этой модели и флагом `--load-mode none` (без mmap),
тип проектора `qwen3a`. Запуск возможен
только внутри cgroup `systemd-run --user --scope -p MemoryMax=26000000000 -p MemorySwapMax=0 -- …`
(иначе отказ до попытки) и когда у хоста `MemAvailable` ≥ 22e9 байт; сторож RSS сервера —
25,5e9 байт. Подробности и измерения — `docs/decisions/0023-gate0-qwen3-omni.md`.

Так же запускается Qwen2.5-Omni-7B (`gate0 run --runtime qwen25omni`, #145): тот же argv с
файлами этой модели и `--load-mode none`, тип проектора `qwen2.5o`. Запуск возможен только внутри
cgroup `systemd-run --user --scope -p MemoryMax=18000000000 -p MemorySwapMax=0 -- …` (иначе отказ
до попытки) и когда у хоста `MemAvailable` ≥ 14e9 байт; сторож RSS сервера — 17,5e9 байт.
Подробности и измерения — `docs/decisions/0024-gate0-qwen25-omni.md`.

Так же запускается Gemma 4 12B-it (`gate0 run --runtime gemma12b`, #193): тот же argv с файлами
этой модели и `--load-mode none`, тип аудиопроектора `gemma4ua`. Запуск возможен только внутри
cgroup `systemd-run --user --scope -p MemoryMax=18000000000 -p MemorySwapMax=0 -- …` (иначе отказ
до попытки) и когда у хоста `MemAvailable` ≥ 18e9 байт; сторож RSS сервера — 17,5e9 байт.
Подробности и измерения — `docs/decisions/0031-gate0-gemma4-12b.md`.

#### Runtime Qolda-AVL-5B (Gate 0 #194)

Stock llama.cpp b12359 аудио этой модели не запускает, а патч авторов к b12359 не применяется
(`docs/decisions/0032-gate0-qolda-avl-5b.md` §1). Runtime `qolda` — `llama-server`, собранный на
станции из llama.cpp `ea63b4d` с патчем `qwen3avl` авторов; каталог
`~/.local/opt/llama.cpp/ea63b4d-qwen3avl/` (исходники — `src/`, бинарник — `llama-server`).

| Что | Закрепление | Лицензия |
|---|---|---|
| llama.cpp | коммит `ea63b4d32ea1b66bdbe369be7f9443f6c00f8b31` (2026-07-30), дерево `817854c32aa6667e3a4a74b9d6be41bab472c85c`; `git archive --format=tar` коммита — 165 949 440 байт, SHA-256 `2a437542bd3727502a4dfca62fc9200b4714844f455e274a99ad8d604c57fc5b` | MIT |
| `llama.cpp-patch/qwen3avl-support.patch` из `issai/Qolda-AVL-5B-GGUF` @ `e2c55b47e6dfcc224718e466691aeaa49ec56bd9` | 23 627 байт, git blob `a0b01da76b4042109d10cf863c33fb01063bd235`, SHA-256 `bb27de1efd5b546d74b834395570f49d5e017db872f0586a771edf59d0f89bf3`; 8 файлов, +228/−8 | Apache-2.0 |
| `cmake` 4.4.5 (только сборка) | `cmake-4.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` с PyPI, 30 246 479 байт, SHA-256 `85625a9b6d5e88ff99c7eb79123d85570844e77ac076366cf75c609cc0b1b579`; каталог `~/.cache/sound-scout/build-tools/cmake-4.4.5/`, запуск `uv tool run --offline --from <wheel> cmake` | BSD-3-Clause (CMake), Apache-2.0 (обёртка PyPI) |
| собранный `llama-server` | 14 820 096 байт, SHA-256 `ddf8a43459d68da28a403fcd1469c8d31982feb52f106122e1a2acf1d0ee6809`, `--version`: `version: 1 (ea63b4d)`, `built with GNU 15.2.0 for Linux x86_64` | MIT |

Загрузка: исходники — `git fetch --depth 1 https://github.com/ggml-org/llama.cpp <коммит>` в
пустой каталог и `git checkout FETCH_HEAD` (`git rev-parse HEAD` равен коммиту); патч и wheel — по
прямому URL в `.part`, сверка размера, SHA-256 (у патча и git blob), затем переименование. Патч —
`git apply --check`, затем `git apply`. Сборка — системный `g++` 15.2.0 (раздел 2), без сети:
`cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF
-DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF
-DLLAMA_BUILD_EXAMPLES=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON` и
`cmake --build build --target llama-server -j 8`; в `build/` нет `_deps`, в журналах сборки нет
`file(DOWNLOAD`, `FetchContent` и `ExternalProject`. Без OpenSSL у сервера нет HTTPS-клиента.
Бинарник скопирован в каталог выше и сделан read-only. `gate0 run --runtime qolda` до старта
сервера сверяет его SHA-256 с константой `QOLDA_SERVER_SHA256` и строку `--version` с коммитом
`ea63b4d`; расхождение — отказ `precondition_failed`. Argv — как у `gemma12b`, плюс `-c 8192`,
`--cache-ram 0` и `--reasoning-format deepseek`. Запуск возможен только внутри cgroup
`systemd-run --user --scope -p MemoryMax=18000000000 -p MemorySwapMax=0 -- …` (иначе отказ до
попытки) и когда у хоста `MemAvailable` ≥ 14e9 байт; сторож RSS сервера — 17,5e9 байт.

#### Runtime Borealis-5b-it (Gate 0 spike #68)

GGUF и llama.cpp для этой архитектуры нет. Runtime — `torch` CPU + `transformers` из extra
`gate0-borealis` (раздел 1) и локальный HTTP-сервер `python -m sound_scout.eval.gate0_borealis
--model-dir <веса> --host 127.0.0.1 --port <свободный> --threads <N>`, который отвечает на те же
маршруты, что `llama-server` у `gate0` (`/health`, `/apply-template`, `/tokenize`,
`/v1/chat/completions`). Штатный `from_pretrained` модели не используется: он загружает конфиги
и токенизатор из чужих репозиториев и строит fp32-скелет (пик ≈ 29 GB). Каркас строится на
`meta` из констант, тензоры F16 читаются из `model.safetensors` по одному и приводятся к bf16;
`trust_remote_code` не используется. До импорта `transformers` выставляются `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`; после bind сокета ставится сетевой guard
процесса; без него (`[network] allow = true`) сервер не стартует. Запуск `gate0 run --runtime borealis` возможен только внутри cgroup с потолком памяти:
`systemd-run --user --scope -p MemoryMax=20000000000 -p MemorySwapMax=0 -- …` (иначе отказ до
попытки). Подробности и измерения — `docs/decisions/0010-gate0-borealis.md`.

#### Runtime GigaChat3.1-Audio-10B-A1.8B (Gate 0 spike #85)

GGUF с аудиопроектором для этой модели нет, vLLM-путь карточки рассчитан на GPU. Runtime —
`torch` и `torchaudio` CPU + `transformers` 4.57.6 из extra `gate0-gigachat` (раздел 1) и
локальный HTTP-сервер `python -m sound_scout.eval.gate0_gigachat --model-dir <веса> --host
127.0.0.1 --port <свободный> --threads <N>` с теми же маршрутами, что у `gate0_borealis`.
Штатный `from_pretrained` модели не используется: ему нужен `trust_remote_code`, а пик его загрузки
≈ 26 GB. Энкодер и адаптер строятся классами модели и приводятся к bf16 первыми, декодер —
`DeepseekV3ForCausalLM` из `transformers` на `meta`, тензоры читаются по одному; `trust_remote_code`
не используется. Сеть: те же `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_HUB_DISABLE_TELEMETRY=1`, guard процесса после bind; если за загрузку guard запретил connect,
резолв или отправку (не `socket.bind`), сервер не стартует (код 3). Запуск `gate0 run --runtime gigachat` возможен только внутри cgroup
`systemd-run --user --scope -p MemoryMax=26000000000 -p MemorySwapMax=0 -- …` (иначе отказ до
попытки) и когда у хоста `MemAvailable` ≥ 22,5e9 байт; сторож RSS сервера — 25,5e9 байт.
Подробности и измерения — `docs/decisions/0011-gate0-gigachat-audio.md`.

### 3.3. Ограничения модели, влияющие на код

- Один аудиовход — не длиннее 30 с (Gemma audio guide); вход 16 kHz mono. Длинная запись режется
  на клипы ≤ 30 с по речевым интервалам, offsets пересчитываются на исходную шкалу (R03).
- Таймкодов модель не выдаёт: локализация строится по клипу и его границам, наблюдение без
  локализации остаётся `unlocalized` (R05).
- Русский: в техотчёте есть русский ASR (FLEURS ru WER 0.068) и перевод CoVoST ru→en; свидетельств
  оценки произношения и просодии нет; Gate 0 spike дал NOT READY
  (`docs/decisions/0007-gate0-result.md` §5).

Borealis-5b-it (spike #68):

- Аудио всегда дополняется или обрезается до 30 с (`prepare_audio` модели, `padding="max_length"`):
  в языковую модель всегда идут 375 аудиопозиций, независимо от длины клипа.
- Таймкодов нет; JSON-режима нет — `response_format` не применяется, ответ проверяет только
  `review.validate_response`.
- Один аудиовход на запрос; вход 16 kHz mono.

GigaChat3.1-Audio-10B-A1.8B (spike #85):

- Шаг энкодера с адаптером — 160 мс: 6,25 аудиопозиции в секунду клипа (63 позиции на 10 с).
  В аудиоблок модель сама вписывает метки времени `HH:MM:SS` (каждые 60 с) и длительность, а
  шаблон — блок файла `<|file|>{"id": …, "name": …, "type": "audio"}<|/file|>`.
- JSON-режима нет: `response_format` не применяется, ответ проверяет только
  `review.validate_response`. Шаблон без системного сообщения вставляет штатный developer system
  модели (≈ 1 000 токенов).
- В `config.json` модели есть `n_fft 320` и `center false`, но код модели их не читает: он берёт
  `n_fft = win_length = 400` и `center=True`; это штатное поведение, сервер его не меняет.
- Один аудиовход на запрос; вход 16 kHz mono.

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
`docs/decisions/0009-corpus-pilot-1.md`. Исключение для #149 (решение владельца
[7085422087](https://github.com/renidere/sound-scout-plugin/issues/149#issuecomment-7085422087)):
реплики RINCO без файлов `pilot-1` идут к моделям на арендованном инстансе этой задачи —
`docs/decisions/0028-gpu-reference-gate0.md`, «Дополнение 2026-11-23», Д.7.

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

## 6. Голос TTS и данные NC для синтетики и контроля ASR (#81)

Файлы для двух дополнений черновика `pilot-1`: синтетических клипов с известными ошибками
ударения и искажениями слов (`sound_scout.eval.synthetic`) и контроля «речь с ошибками ASR» из
Open STT (`sound_scout.eval.openstt`). Загрузку ровно этих файлов разрешил владелец 2026-11-19
([#81](https://github.com/renidere/sound-scout-plugin/issues/81)); лицензии NC и синтетику для
квалификационного корпуса владелец принял в
[#29](https://github.com/renidere/sound-scout-plugin/issues/29). Использование — только локальная
квалификация; файлы лежат в `data/corpus/sources/<набор>/` вне git. Голос Piper — не audio
backend и в `analyze` не входит: он только создаёт синтетические клипы. Их скачивает тот же
`scripts/fetch_corpus_sources.py` по таблице `sound_scout.eval.corpus.EXTRA_SOURCES` (размер, сумма
издателя и SHA-256 — по потоку и по диску), расхождение таблицы и этого раздела ловит
`tests/test_dependencies_doc.py`. Отдельный раздел нужен потому, что раздел 5 обязан совпадать с
`SOURCES` (#78). Сумма издателя у `.onnx` — LFS oid Hugging Face (SHA-256 содержимого), у
`.onnx.json` и `MODEL_CARD` — git blob SHA-1 на закреплённой ревизии, у Open STT — MD5 Zenodo
(совпадает с `md5sum.lst` репозитория open_stt). Лицензия Open STT расходится: README open_stt —
CC BY-NC 4.0, поле Zenodo — CC BY 4.0; принята более строгая CC BY-NC 4.0. Голос `denis` дообучен
от базовой модели `rdh` (CC0) и по `MODEL_CARD` требует Piper ≥ 1.5. Манифест
`asr_calls_2_val.csv` не скачивается: пары wav/txt берутся из самого архива. Правила —
`docs/decisions/0012-synthetic-diction-and-openstt.md`. Исключение для #149 (решение владельца
7085422087): 20 новых фраз голоса `denis` вне `pilot-1` (`synthetic.GPU_EXT_SENTENCES`) идут к
моделям на арендованном инстансе этой задачи — `0028`, «Дополнение 2026-11-23», Д.7.

| Набор | Версия / ревизия | Файл | Источник (URL) | Лицензия | Байт | Сумма издателя | SHA-256 |
|---|---|---|---|---|---|---|---|
| Piper, голос `ru_RU-denis-medium` | rhasspy/piper-voices @ 04ca88f5685ca5253c701af3687e155b113bd512, ru/ru_RU/denis/medium | `ru_RU-denis-medium.onnx` | https://huggingface.co/rhasspy/piper-voices/resolve/04ca88f5685ca5253c701af3687e155b113bd512/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx | MIT (репозиторий голосов rhasspy/piper-voices), датасет голоса CC0 | 63221984 | `sha256:879bc365cb14d1380e3f5ed6044162ccf54c99f6161871569b7f830aec3ce841` | `879bc365cb14d1380e3f5ed6044162ccf54c99f6161871569b7f830aec3ce841` |
| Piper, голос `ru_RU-denis-medium` | rhasspy/piper-voices @ 04ca88f5685ca5253c701af3687e155b113bd512, ru/ru_RU/denis/medium | `ru_RU-denis-medium.onnx.json` | https://huggingface.co/rhasspy/piper-voices/resolve/04ca88f5685ca5253c701af3687e155b113bd512/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json | MIT (репозиторий голосов rhasspy/piper-voices), датасет голоса CC0 | 6196 | `git-blob-sha1:59da8d82e1416c9b53095278419c410ad2ed324b` | `5891b94580ae739a8cdc5d17591db6800333f85164d13599aca893883d99b9eb` |
| Piper, голос `ru_RU-denis-medium` | rhasspy/piper-voices @ 04ca88f5685ca5253c701af3687e155b113bd512, ru/ru_RU/denis/medium | `MODEL_CARD` | https://huggingface.co/rhasspy/piper-voices/resolve/04ca88f5685ca5253c701af3687e155b113bd512/ru/ru_RU/denis/medium/MODEL_CARD | MIT (репозиторий голосов rhasspy/piper-voices), датасет голоса CC0 | 418 | `git-blob-sha1:a97e89d1da2fcbced2e4a42ac49a670a1c19c8a8` | `917110d5056810771fd3daea2644bbadf1341816b7631f2043b33be245903432` |
| Open STT `asr_calls_2_val` | v1.0, DOI 10.5281/zenodo.4899885 | `asr_calls_2_val.tar.gz` | https://zenodo.org/records/4899885/files/asr_calls_2_val.tar.gz?download=1 | CC BY-NC 4.0 (README open_stt; поле Zenodo — CC BY 4.0, берётся более строгая) | 805251737 | `md5:5146eef0aa64af41e17c5ec18a9cab5c` | `aea64611133129e8c9b5d829953f5bfb9cc2d1bd95b9e429fa8e00e2c83f6461` |

## 7. GPU-эталон (Vast.ai)

Стек эталонных прогонов на арендованном GPU (#147, `docs/decisions/0026-gpu-stack.md`):
`src/sound_scout/eval/vast.py`, файлы хоста `src/sound_scout/eval/gpu_host/`. Здесь закреплены
образ, версии runtime и веса двух кандидатов; тест `tests/test_eval_vast.py` сверяет раздел с
константами кода `IMAGE`, `RUNTIME` и `MODELS`. **Закреплено, на станцию не загружается; загрузка
на арендованный хост — только по одобрению пакета (#148, #149).** Этот раздел появился раньше
загрузки, хотя PROPOSAL § 5 п. 5 ставил запись в коммит загрузки: критерий 1 #147 требует
закрепления сейчас. Новых Python-пакетов стек не добавляет: только stdlib и `numpy` из раздела 1.

### 7.1. Образ и runtime

| Что | Значение | Источник | Лицензия |
|---|---|---|---|
| Образ | `vllm/vllm-openai:v0.34.0` | Docker Hub, тег релиза vLLM `v0.34.0` (2026-11-16) | Apache-2.0 (vLLM) |
| Индекс (multi-arch) | `sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1` | реестр Docker Hub, чтение 2026-11-21 | — |
| Манифест amd64 | `sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484` | там же | — |
| Размер в сжатом виде | 10 036 518 499 байт | там же | — |
| Ссылка в create | `vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1` | Vast принял ссылку по digest: #148, попытки 1, 3, 5 (2026-11-22) | — |
| vLLM | `0.34.0` | Dockerfile тега | Apache-2.0 |
| torch | `2.14.0` | Dockerfile тега | BSD-3-Clause |
| CUDA (torch) | `13.0` | `torch.version.cuda` колеса из индекса `cu130` (история образа: `--index-url …/whl/cu130`); preflight сверяет это значение (#158); драйвер хоста ≥ 580, compute capability ≥ 8.0 | NVIDIA EULA |
| CUDA (образ) | `13.0.2` | `ENV CUDA_VERSION=13.0.2` в config blob `sha256:0372147e2b3daa1baa5c6aa9b90ae5de2f926e25742650f91fa24dbca9cc645a` манифеста amd64, чтение 2026-11-22; preflight только записывает | NVIDIA EULA |
| Python образа | `3.12` | Dockerfile тега (Ubuntu 24.04) | PSF-2.0 |
| transformers | `>=5.16.1,<5.23.0`; в образе — `5.22.2` | `requirements/common.txt` тега; `5.22.2` — preflight на хосте, #148, попытки 3 и 5 (2026-11-22); preflight сверяет диапазон | Apache-2.0 |

Запуск — `vast.vllm_argv`: локальный каталог весов, `--dtype bfloat16`, `--max-model-len 8192`,
`--max-num-seqs 1`, `--seed 0`, `--enforce-eager`, `--generation-config vllm`, одно аудио на
запрос, `--no-enable-prefix-caching` и `--mm-processor-cache-gb 0` (prefix cache и кэш
мультимодального процессора выключены, #161), `127.0.0.1:8000`, без `--trust-remote-code`
(загружается только thinker: у Qwen3-Omni веса talker и code2wav лежат в тех же шардах,
загрузчик vLLM их пропускает). Имена флагов — `vllm/engine/arg_utils.py` тега `v0.34.0`
(`--enable-prefix-caching` как `BooleanOptionalAction`; `--mm-processor-cache-gb`, `0` выключает
кэш — `vllm/config/multimodal.py`). Поле llama.cpp `cache_prompt` vLLM не читает:
`gate0.sampling("vllm")` его не отправляет; остальные поля `SAMPLING` (`temperature`, `seed`,
`chat_template_kwargs`) есть в `ChatCompletionRequest`
(`vllm/entrypoints/openai/chat_completion/protocol.py` тега `v0.34.0`).

### 7.2. Веса bf16

Ревизии и хеши — из публичного API Hugging Face (чтение 2026-11-21), локально не считались.
Сумма издателя у LFS-файлов — SHA-256 содержимого, у остальных — git blob SHA-1 на ревизии
(как у GigaChat в разделе 3.1). `spk_dict.pt` Qwen2.5-Omni (pickle, нужен только talker) не
загружается (`hf download --exclude spk_dict.pt`). Сверку на хосте делает `hostrun.py weights`,
квитанцию — Gate 0 клиент до первого запроса.

#### Qwen2.5-Omni-7B — `Qwen/Qwen2.5-Omni-7B` @ `ae9e1690543ffd5c0221dc27f79834d0294cba00`

Лицензия: Apache-2.0 (файл `LICENSE`). Файлов: 19, байт: 22 379 037 779.

| Файл | Байт | Сумма издателя |
|---|---|---|
| `.gitattributes` | 1570 | `git-blob-sha1:52373fe24473b1aa44333d318f578ae6bf04b49b` |
| `LICENSE` | 11343 | `git-blob-sha1:9b1dd49441cd594560125cc7d79e886d0816cd8f` |
| `README.md` | 36838 | `git-blob-sha1:9eb4a8e90deb0e208a9db6fdf1ea2e4c6dbf518f` |
| `added_tokens.json` | 579 | `git-blob-sha1:4562256351969d3b68f81fd5623ae42a9c7fbb25` |
| `chat_template.json` | 1313 | `git-blob-sha1:e696060ab7ab6e20f60e4e09b18e5da4c1830000` |
| `config.json` | 13186 | `git-blob-sha1:633e1b2b157826cce16b51dee904750f02a7ac9c` |
| `generation_config.json` | 74 | `git-blob-sha1:1c1ac9d737684b833c80a438e30fc3ba53e89bd4` |
| `merges.txt` | 1671853 | `git-blob-sha1:31349551d90c7606f325fe0f11bbb8bd5fa0d7c7` |
| `model-00001-of-00005.safetensors` | 4985055504 | `sha256:5edb02fd7c98803239468375cc9dc1bff492865c2aa086b78f348597021d6cbc` |
| `model-00002-of-00005.safetensors` | 4991496800 | `sha256:7c99b55c6e5bc63fd4b19d4dc23cdc3ddac4b0101bb3c0958cc2b5d05c2bbafe` |
| `model-00003-of-00005.safetensors` | 4991496904 | `sha256:ad00c3ac296300db905934ed213c4077ff49b85d30c0099270c814e2c77ec812` |
| `model-00004-of-00005.safetensors` | 4969489824 | `sha256:152bc7d81441eaba22547d8d96c03d32dd592ce0e0e1d0e449347a4b23a532d3` |
| `model-00005-of-00005.safetensors` | 2425322160 | `sha256:b8b18276481ba8cdf4fe2c98ac4c7a2da6e0d1c8a51850d162a391760cb2b81e` |
| `model.safetensors.index.json` | 233160 | `git-blob-sha1:e8728f00dd29744bb14d7ed4f374c4f05c75b6c5` |
| `preprocessor_config.json` | 667 | `git-blob-sha1:63c79f1c43bce173ab13d94096dea54b3688436f` |
| `special_tokens_map.json` | 832 | `git-blob-sha1:8c2a71eaac0f526df324364d953503def113b7a1` |
| `tokenizer.json` | 11421870 | `sha256:8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99` |
| `tokenizer_config.json` | 6469 | `git-blob-sha1:22c6d3668a32e4fe6dc09331bff96e7cc3a2290f` |
| `vocab.json` | 2776833 | `git-blob-sha1:4783fe10ac3adce15ac8f358ef5462739852c569` |

#### Qwen3-Omni-30B-A3B-Instruct — `Qwen/Qwen3-Omni-30B-A3B-Instruct` @ `26291f793822fb6be9555850f06dfe95f2d7e695`

Лицензия: Apache-2.0 (`license_name` карточки; файла `LICENSE` нет). Файлов: 25, байт: 70 530 595 574.

| Файл | Байт | Сумма издателя |
|---|---|---|
| `.gitattributes` | 1519 | `git-blob-sha1:a6344aac8c09253b3b630fb776ae94478aa0275b` |
| `README.md` | 90295 | `git-blob-sha1:64fde95f19d9f979785606ece4e484d896cd7738` |
| `chat_template.json` | 6772 | `git-blob-sha1:339ecf8a9f1ac161a9b59f833c85e6d92146b78c` |
| `config.json` | 13671 | `git-blob-sha1:18c714a7fa2d8881e8e1441ec3759b6338c9c5ea` |
| `generation_config.json` | 146 | `git-blob-sha1:59f9fe8f7db6b35dc244fc7eb52743b31533bbb5` |
| `merges.txt` | 1671839 | `git-blob-sha1:20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0` |
| `model-00001-of-00015.safetensors` | 4997899632 | `sha256:f95d142b727fbd0698f0ed9478e222d4f29467df855e21862e46bd1401049065` |
| `model-00002-of-00015.safetensors` | 4997754216 | `sha256:320c1f31f7ddc4efd2501891590eca2c481ce4eca3db33d70ebbb637dae85764` |
| `model-00003-of-00015.safetensors` | 4997754216 | `sha256:306f06b05ef032f1fa51eb62e72c3432324e7a084fa0d15ed17ff856adffcf33` |
| `model-00004-of-00015.safetensors` | 4997755648 | `sha256:4cb551f433bfb4d83a5843e433e74688d71a96eeb1b62217b37d719026d559b5` |
| `model-00005-of-00015.safetensors` | 4997755792 | `sha256:8aec5e8cc0e0b179a81f690af31a1149238f4b1d18ac60e3f690020fdf7d915f` |
| `model-00006-of-00015.safetensors` | 4997755792 | `sha256:8ff40a3d9e9c59a35a10257c1c5b216f472e6a102a4d6059bd31183eee03cb4c` |
| `model-00007-of-00015.safetensors` | 4997755792 | `sha256:f6d84cc1d47178850f9ba1fe9c01f45ee63d9a8b9975f92821ee2b4a0387860b` |
| `model-00008-of-00015.safetensors` | 4997755792 | `sha256:9f369ccdc9d2d2a55df794005b23cdbce44494a72e4a0069a150ae471cfea720` |
| `model-00009-of-00015.safetensors` | 4997755792 | `sha256:980f754b21c4b20f36e02d184b792f5625f7c9d24b8a5c9da94287cf3cfde11b` |
| `model-00010-of-00015.safetensors` | 4997755792 | `sha256:5c5a798376321d74df2eb3f235e1c8c68669f09b0c753b5698a16e681204d808` |
| `model-00011-of-00015.safetensors` | 4997755792 | `sha256:6f7c051bd391c89279b43e7342ff4a243ad2ba2144f4aa62233fd2b02d72fce0` |
| `model-00012-of-00015.safetensors` | 4997755792 | `sha256:a8b09a9367cd31ebf928e87b9eaf002492379d9d21d352b480854c2fbe6d9978` |
| `model-00013-of-00015.safetensors` | 4999771808 | `sha256:d9b7441c3a33e066d109cd6338e74ed680de9aa407d4df66eba5cb3c15c6d738` |
| `model-00014-of-00015.safetensors` | 4996618552 | `sha256:73f09f187ff7b469a8937b39b8ac772e539e0d61311526c345558c5e6a6eb09a` |
| `model-00015-of-00015.safetensors` | 553698794 | `sha256:68880d566a50643807a4e73f5227c4f468bb31532925ed5fae78a51bdd39492a` |
| `model.safetensors.index.json` | 2727350 | `git-blob-sha1:037226d18d29071d0ef1ca6b2a85c43be5ce61aa` |
| `preprocessor_config.json` | 603 | `git-blob-sha1:7df5b45379044a680218cb2b7ed4d01a6633f537` |
| `tokenizer_config.json` | 7344 | `git-blob-sha1:6ff9fd60cc623bb54bbd603cbd418c97a11528d7` |
| `vocab.json` | 2776833 | `git-blob-sha1:4783fe10ac3adce15ac8f358ef5462739852c569` |
