# Зависимости

Полный перечень того, от чего зависит Sound Scout: Python-пакеты, системные программы, runtime
модели, файлы весов и открытые данные квалификационного корпуса. Правило проекта: **любая новая или обновлённая зависимость вносится сюда в
том же коммите**, с версией, источником, лицензией и контрольной суммой. Для весов и бинарей
указывается ровно тот файл, который используется, а не «последняя версия». Абсолютные пути
конкретной станции сюда не пишутся; ниже — пути по умолчанию относительно домашнего каталога
оператора.

Обновлено: 2026-11-20.

## 1. Python-пакеты

Источник истины — `pyproject.toml` и `uv.lock` (`uv sync --locked`). Здесь — прямые зависимости и
версии в lock на дату обновления.
SHA-256 каждого файла пакета — в `uv.lock`. Extra `gate0-borealis` (`uv sync --locked --extra gate0-borealis`) нужен только для Gate 0 spike Borealis-5b-it (#68) и по умолчанию не ставится, CI его не тянет; кроме трёх пакетов ниже он приносит `sympy`, `networkx`, `jinja2`, `setuptools`, `regex`, `typer`/`rich` (через `transformers`), а `tokenizers` и `huggingface-hub` уже были в lock через faster-whisper. Extra `gate0-gigachat` (`uv sync --locked --extra gate0-gigachat`) нужен только для Gate 0 spike GigaChat3.1-Audio-10B-A1.8B (#85), тоже не ставится по умолчанию и в CI; он объявлен в `[tool.uv] conflicts` с `gate0-borealis` (разные `torch` и `transformers`), поэтому в lock у `torch` и `transformers` по две версии. Кроме четырёх пакетов ниже он приносит `transformers` 4.57.6 со своими зависимостями: `tokenizers` 0.22.2 (wheel `cp39-abi3-manylinux_2_17_x86_64` SHA-256 `369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67`) и `huggingface-hub` 0.36.2 (SHA-256 `48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270`) вместо версий faster-whisper, `requests`, `urllib3`, `certifi`, `charset-normalizer`, `setuptools`, `sympy`, `networkx`, `jinja2`, `regex`. Пины `==`: `torchaudio` 2.11.0 — последний CPU-wheel для cp313 в `https://download.pytorch.org/whl/cpu`, и он требует ровно `torch` 2.11.0; `transformers` 4.57.6 — версия, на которой upstream проверил код модели (карточка), а в `transformers` 5 слои MoE `DeepseekV3` хранят экспертов слитыми тензорами, и ключи checkpoint не совпадут с каркасом (`docs/decisions/0011-gate0-gigachat-audio.md` §1). `mcp` тянет `pydantic`, `starlette`, `uvicorn`, `sse-starlette`, `pyjwt`, `opentelemetry-api` (без SDK и экспортёра); stdio сервер использует только `mcp.server.stdio`, middleware OpenTelemetry у него убран.

Пакет плагина Hermes (`hermes/plugin/sound-scout/dashboard/plugin_api.py`, #101) исполняется в процессе gateway и берёт FastAPI, Starlette и python-multipart из окружения Hermes (на станции разработки 0.133.1, 1.9.0 и 0.0.32); тесты плагина берут Starlette 1.10.0 и python-multipart 0.0.34 из lock транзитивно через `mcp`, новых пакетов в lock нет.

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
| FFmpeg (`ffmpeg`, `ffprobe`) | 8.0.1 | декодирование входа, ресемплинг, клипы | пакет дистрибутива (`apt install ffmpeg`) |
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

CI (`.github/workflows/ci.yml`): `actions/checkout@v7.0.1`, `astral-sh/setup-uv@v10.3.3`, FFmpeg из
apt runner'а, Node.js — предустановленный в образе runner'а (без `actions/setup-node`). Без Node
тесты зонда и рекордера в CI падают, а не пропускаются (фикстура `node` в `tests/conftest.py`).

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
`docs/decisions/0012-synthetic-diction-and-openstt.md`.

| Набор | Версия / ревизия | Файл | Источник (URL) | Лицензия | Байт | Сумма издателя | SHA-256 |
|---|---|---|---|---|---|---|---|
| Piper, голос `ru_RU-denis-medium` | rhasspy/piper-voices @ 04ca88f5685ca5253c701af3687e155b113bd512, ru/ru_RU/denis/medium | `ru_RU-denis-medium.onnx` | https://huggingface.co/rhasspy/piper-voices/resolve/04ca88f5685ca5253c701af3687e155b113bd512/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx | MIT (репозиторий голосов rhasspy/piper-voices), датасет голоса CC0 | 63221984 | `sha256:879bc365cb14d1380e3f5ed6044162ccf54c99f6161871569b7f830aec3ce841` | `879bc365cb14d1380e3f5ed6044162ccf54c99f6161871569b7f830aec3ce841` |
| Piper, голос `ru_RU-denis-medium` | rhasspy/piper-voices @ 04ca88f5685ca5253c701af3687e155b113bd512, ru/ru_RU/denis/medium | `ru_RU-denis-medium.onnx.json` | https://huggingface.co/rhasspy/piper-voices/resolve/04ca88f5685ca5253c701af3687e155b113bd512/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json | MIT (репозиторий голосов rhasspy/piper-voices), датасет голоса CC0 | 6196 | `git-blob-sha1:59da8d82e1416c9b53095278419c410ad2ed324b` | `5891b94580ae739a8cdc5d17591db6800333f85164d13599aca893883d99b9eb` |
| Piper, голос `ru_RU-denis-medium` | rhasspy/piper-voices @ 04ca88f5685ca5253c701af3687e155b113bd512, ru/ru_RU/denis/medium | `MODEL_CARD` | https://huggingface.co/rhasspy/piper-voices/resolve/04ca88f5685ca5253c701af3687e155b113bd512/ru/ru_RU/denis/medium/MODEL_CARD | MIT (репозиторий голосов rhasspy/piper-voices), датасет голоса CC0 | 418 | `git-blob-sha1:a97e89d1da2fcbced2e4a42ac49a670a1c19c8a8` | `917110d5056810771fd3daea2644bbadf1341816b7631f2043b33be245903432` |
| Open STT `asr_calls_2_val` | v1.0, DOI 10.5281/zenodo.4899885 | `asr_calls_2_val.tar.gz` | https://zenodo.org/records/4899885/files/asr_calls_2_val.tar.gz?download=1 | CC BY-NC 4.0 (README open_stt; поле Zenodo — CC BY 4.0, берётся более строгая) | 805251737 | `md5:5146eef0aa64af41e17c5ec18a9cab5c` | `aea64611133129e8c9b5d829953f5bfb9cc2d1bd95b9e429fa8e00e2c83f6461` |
