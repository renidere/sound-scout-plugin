# 0001 — Audio-understanding backend: два кандидата для Gate 0

- **Статус:** решение владельца принято 2026-11-18 — раздел 0. Разделы 1–7 — исходное сравнение от 2026-11-17, сохранено как история; его рекомендация заменена.
- **Дата:** 2026-11-17.
- **Задача:** [#22](https://github.com/renidere/sound-scout-plugin/issues/22), эпик
  [#3](https://github.com/renidere/sound-scout-plugin/issues/3).
- **Основание:** `docs/spec/REQUIREMENTS.md` раздел 5 (Gate 0), R09; `docs/spec/AGENT_PROMPT.md`
  B.5–B.7 и «Полномочия и запреты».
- **Оговорка Gate 0:** лучший backend не фиксируется без согласованного пилота. Этот документ
  сравнивает кандидатов по первичным источникам и называет один путь; способность модели слышать
  русскую речь доказывает только spike (#3) с реальным аудиовходом.
- **Что сделано и не сделано:** только чтение источников через fetch-инструменты. Ничего не
  установлено, не скачано, ни один model API не вызывался (раздел 6).

Все источники прочитаны 2026-11-17; где факт перепроверен в ходе этой задачи, это указано.
Обозначения источников вида `[Q-card]` раскрыты в разделе 7.

## 0. Решение владельца (2026-11-18)

Владелец задал бюджет ресурсов: не больше 20 GB оперативной памяти и 12–16 GB видеопамяти, открытые
и лёгкие модели; первая версия работает на станции разработки (CPU, без дискретного GPU).
Qwen3-Omni-30B-A3B (≈ 19,9 GB) и облачный Gemini 3.8 Flash в этот бюджет и предпочтения не
входят. Повторное сравнение по первичным источникам (2026-11-18, полный отчёт со ссылками хранится
у владельца вне git) дало новую пару для Gate 0:

| | Gemma 4 E4B-it — **выбрана** | Borealis-5b-it — запасной |
|---|---|---|
| Лицензия, веса | Apache-2.0, открытые, не gated | Apache-2.0, открытые |
| Аудиовход | сырой сигнал 16 kHz mono, энкодер USM-Conformer, **до 30 с на вход** | энкодер Whisper-large-v3 + Qwen3-4B, вход после 30 с отрезается |
| Русский | FLEURS ru WER 0.068, CoVoST ru→en (Tech Report); в карточке не назван отдельно | «Languages: Russian, English» в карточке |
| Произношение и просодия | свидетельств нет | свидетельств нет (только WER/CER) |
| Память | GGUF Q8_0 + mmproj BF16 ≈ 9,0 GB, CPU-путь через llama.cpp | ≈ 10,1 GB весов; штатный загрузчик ≈ 30 GB пиково, GGUF нет |
| Runtime | llama.cpp mtmd (`--audio`), без remote code | transformers + `trust_remote_code`, тянет конфиги с Hub |

Выбор Gemma 4 E4B-it: укладывается в бюджет с запасом, имеет официальный CPU-путь в llama.cpp и
одинаковый путь на будущем GPU, не требует исполнения стороннего кода; чистый transcript-only
baseline — та же модель с текстом вместо аудио. Borealis остаётся запасным кандидатом, если spike
даст NOT READY: у неё русский заявлен прямо, но больше практических препятствий и непроверенный
лицензионный риск обучающих данных (инструкции, сгенерированные Gemini, фрагменты аудиокниг).

GigaChat (предложение владельца): единственная открытая аудиомодель Сбера —
GigaChat3.1-Audio-10B-A1.8B (MIT по метаданным карточки) — по русским бенчмаркам сильнее всех
найденных, но в bf16 занимает ≈ 22,5 GB, квантованной аудиоверсии и GGUF с аудиопроектором нет.
Вернуть её можно при станции с GPU больше 16 GB или при согласии на неофициальную 8-bit
квантизацию. GigaAM — ASR и классификатор эмоций, не аудио-LLM.

Не проверены в повторном сравнении: Gemma 4 12B Unified, MiniCPM-o, Step-Audio 2 mini,
MiDashengLM-7B, MiMo-Audio-7B; скорость любой модели на этом CPU.

Свидетельств оценки русского произношения и просодии в первичных источниках нет ни у одной
лёгкой модели, включая выбранную. Способность проверяет spike (#23): пары записей с одинаковыми
словами и разным звучанием, transcript-only baseline, отрицательные контроли (тишина, шум,
пустой вход), trace реального аудиовхода, детерминированное декодирование.

Разрешение владельца от 2026-11-18: загрузить `gemma-4-E4B-it-Q8_0.gguf` и
`mmproj-gemma-4-E4B-it-BF16.gguf` из `ggml-org/gemma-4-E4B-it-GGUF` и установить llama.cpp v1.0.0,
запускать на станции разработки. Точные ревизии, размеры и SHA-256 — `docs/dependencies.md`;
требование — `docs/spec/REQUIREMENTS.md` раздел 9.

## 1. Сравнительная таблица

| Критерий | Qwen3-Omni-30B-A3B-Instruct (локально, GGUF Q4_K_M, llama.cpp) | Gemini 3.8 Flash (облако, Gemini API) |
|---|---|---|
| Аудиовход | Да, bytes/PCM: `llama-server` `/completion` с `multimodal_data` (base64); мультимодальность `llama-server` целиком помечена «experimental» (OAI chat, `/completion`, embeddings); модель и проектор через `--mmproj` или `-hf …:Q4_K_M` [L-server], [L-mm]. Энкодер AuT ресемплирует в 16 kHz, 12.5 токена/с, до 40 мин аудио на запрос [Q-paper] | Да, нативно: inline base64 при запросе ≤ 20 МБ, иначе Files API; WAV/MP3/AIFF/AAC/OGG/FLAC/MPEG/M4A/L16/Opus/ALAW/MULAW/WebM; до 9.5 ч; 32 токена/с (1 мин = 1920 токенов); «Downsampled to 16 Kbps», многоканальное сводится в один канал [G-audio] |
| Русский | Заявлен: 19 языков речевого входа, `ru` в Table 3 техотчёта [Q-paper]; список языков в README [Q-gh]. Тег карточки HF — только `en` (перепроверено 2026-11-17 через HF API) | Нет свидетельств в первичных источниках: ни страница аудио [G-audio], ни страница модели [G-model] языки аудиовхода не перечисляют |
| Дикция/просодия | Нет прямых свидетельств. Есть MMAU 77.5, MMSU 69.0, VoiceBench (аудио-рассуждение, не оценка произношения) [Q-card]; эмоция/просодия — только в качественных примерах Captioner, §9.2 [Q-paper]; MELD/VocalSound не приводятся | Нет прямых свидетельств. Заявлены «Emotion detection in speech and music» и «Analyzing specific segments with timestamps», бенчмарков нет [G-audio] |
| Лицензия/доступ | Apache-2.0: HF-метаданные карточки `license: other`, `license_name: apache-2.0`; файла `LICENSE` нет ни в листинге карточки, ни в листинге GGUF-репозитория, у GGUF-репозитория лицензионных метаданных нет вовсе (перепроверено 2026-11-17 через HF API) [Q-card], [Q-gguf]. Открытый доступ, не gated | Условия Gemini API, платный тариф (Paid Services) [G-terms]; ключ API; доступ только после согласования по R09 |
| Ресурсы | Q4_K_M 18 557 053 952 Б (≈ 17.3 GiB) + mmproj Q8_0 1 325 020 128 Б (≈ 1.2 GiB) ≈ 18.5 GiB RAM при mmap, плюс KV-кэш [Q-gguf]. Остаток на хосте ≈ 10 GiB под ОС, FFmpeg, faster-whisper. Q8_0 (32.5 ГБ) и bf16 (61.1 ГБ) в 30 GiB не входят. BF16 в transformers: 78.9–144.8 ГБ GPU (Instruct, BF16, видеовход 15–120 с) [Q-card] — неприменимо. 3B активных параметров ⇒ скорость порядка dense-3B на CPU; численно ни одним первичным источником не подтверждена — измерить в spike (#3) | Локальных ресурсов нет; задержка не документирована. Вход до 1 048 576 токенов [G-model] |
| Стоимость | 0 ₽ за вызов; разовая загрузка ≈ 19.9 ГБ | $0.75 / 1M входных токенов до 2026-12-31, $1.50 с 2027-01-01 (аудио без отдельного тарифа); вывод $3.75 → $7.50 за 1M [G-price]. ⇒ ≈ $0.0014/мин аудио (с 2027 ≈ $0.0029/мин) плюс вывод. Free tier есть, но «content used to improve our products» — для записей пользователя исключён [G-terms] |
| Приватность | Offline, `local_files_only`, без сетевых вызовов; записи не покидают хост | Платный тариф: не используется для обучения; «Google logs prompts and responses for a limited period of time» для контроля злоупотреблений, срок не назван; хранение «in any country in which Google or its agents maintain facilities»; DPA, Google как processor [G-terms] |

### 1.1. Не вошли в таблицу

Отклонены на предварительном отборе (по источникам, прочитанным 2026-11-17, см. раздел 7, кроме
Audio Flamingo 3 и SALMONN — их первичные источники не читались):

- Qwen2.5-Omni-7B — Apache-2.0, GGUF Q4_K_M 4.68 ГБ, есть MELD 0.570 и VocalSound 0.939, но русский
  в карточке и бенчмарках не упомянут (en/zh/yue/fr/de). Запасной локальный вариант, если владелец
  не согласует ≈ 20 ГБ.
- Qwen2-Audio-7B-Instruct — карточка только `en`; llama.cpp GGUF не даёт («very poor result»).
- Voxtral-Mini-3B-2507 — 8 языков без русского. Phi-4-multimodal-instruct — аудио без русского,
  рекомендуемый максимум 40 с. Ultravox v0.5 — русский только как covost2 ru→en (перевод).
  Kimi-Audio-7B-Instruct — en/zh.
- Audio Flamingo 3, SALMONN — не читались: по известным условиям (некоммерческая лицензия /
  research-код) и без заявленного русского не проходят предварительный отбор.
- OpenAI gpt-audio-1.5 — аудиовход $32 / 1M токенов (в ≈ 40 раз дороже Gemini за токен), лимиты
  длины и список языков не документированы, abuse-логи 30 дней.
- Gemini 2.5 Flash/Pro — deprecated, с 2026-12-01 запросы роутятся на 3.5 Flash-Lite / 3.8 Flash
  [G-models].

## 2. По каждому кандидату

### 2.1. Qwen3-Omni-30B-A3B-Instruct (GGUF Q4_K_M, llama.cpp)

**Как передаётся аудио.** Файл декодируется локально (FFmpeg) и передаётся в `llama-server`:
эндпоинт `/completion` принимает `multimodal_data` — массив base64-строк, подставляемых в prompt
на место маркеров; OAI-совместимый `/v1/chat/completions` принимает аудио как `input_audio`. В
README сервера мультимодальность помечена «experimental» целиком — OAI chat, `/completion` и
embeddings [L-server]. Проектор
загружается флагом `--mmproj` либо автоматически через `-hf <repo>:Q4_K_M` [L-mm]. Внутри модели
энкодер AuT (≈ 0.6B параметров) работает на 16 kHz и выдаёт 12.5 токена в секунду; заявленный
максимум — 40 минут аудио на запрос [Q-paper]. Это слуховой вход (сырой сигнал), не транскрипт.

**Свидетельства слухового анализа русской речи.** Русский прямо заявлен как один из 19 языков
речевого входа (Table 3 техотчёта [Q-paper]; README [Q-gh]). Это свидетельство речевого понимания
на русском, не оценки произношения: бенчмарки карточки — MMAU 77.5 (Gemini-2.5-Pro 77.4,
GPT-4o-Audio 62.5), MMSU 69.0, VoiceBench — измеряют аудио-рассуждение [Q-card]; эмоция и
просодия показаны только в качественных примерах Captioner (§9.2 [Q-paper]). Паралингвистических
бенчмарков (MELD, VocalSound) в источниках нет. Итог: оценка русской дикции/просодии — нет
свидетельств; решает spike (#3).

**Известные ограничения.**

- Вариант Thinking «may even introduce a higher propensity for hallucinations» на перцептивных
  задачах [Q-paper] — брать Instruct.
- GGUF содержит только Thinker (текстовый выход); Talker (речевой выход) не нужен и не входит.
  Особенности vLLM/transformers (`disable_talker()` ≈ −10 ГБ, BF16-память Instruct 78.9–144.8 ГБ
  GPU при видеовходе 15–120 с) к GGUF-пути не относятся [Q-gh], [Q-card].
- Мультимодальность в `llama-server` целиком помечена «experimental» — OAI chat, `/completion` с
  `multimodal_data`, embeddings [L-server]; поведение маршрута проверяет spike (#3).
- Скорость на CPU первичными источниками не подтверждена; активных параметров 3B на токен, но
  обработка 1 минуты аудио — это ≈ 750 входных токенов плюс prefill энкодера.
- Карточка HF тегирована только `en`; лицензия указана в метаданных (`license_name: apache-2.0`),
  файла `LICENSE` в репозиториях нет — при согласовании загрузки владелец принимает лицензию по
  карточке [Q-card].

### 2.2. Gemini 3.8 Flash (Gemini API)

**Как передаётся аудио.** `generateContent` с аудио inline (base64) при общем размере запроса
≤ 20 МБ, иначе загрузка через Files API и ссылка на файл; форматы WAV, MP3, AIFF, AAC, OGG, FLAC,
MPEG, M4A, L16, Opus, ALAW, MULAW, WebM; до 9.5 ч аудио на запрос; 32 токена в секунду (1 мин =
1920 токенов); сигнал «Downsampled to 16 Kbps», многоканальное аудио сводится в один канал;
поддерживаются ссылки на отрезки по таймкодам `MM:SS` [G-audio]. Модель `gemini-3.8-flash`
(stable, обновлена в сентябре 2026): входы Text/Image/Video/Audio/PDF, выход текст, до 1 048 576
входных токенов [G-model].

**Свидетельства слухового анализа русской речи.** Нет. Страница аудио заявляет «Emotion
detection in speech and music» и анализ сегментов по таймкодам, но без бенчмарков и без перечня
языков аудиовхода [G-audio]; страница модели языки аудио не перечисляет [G-model]. Поддержка
русского ни заявлена, ни опровергнута — «нет свидетельств в первичных источниках».

**Известные ограничения.**

- Линейка 2.5 deprecated (роутинг с 2026-12-01) — в сравнение не входит [G-models]; Live- и
  TTS-модели для задачи не подходят (другая модальность выхода).
- Задержка не документирована; даунсэмплинг до 16 Kbps и моно — потеря информации до анализа.
- Цена входа удваивается с 2027-01-01 [G-price].
- Приватность: логи «limited period» без срока, страна хранения не фиксируется [G-terms].
- По R09 внешний вызов требует доверенного host-approval с проверяемой привязкой к
  provider/model, хешам записей, числу запросов, бюджету и сроку; такого механизма в репозитории
  и у установленного Hermes нет ⇒ вызов `blocked` до отдельного решения владельца.

## 3. Целевой хост и оценка ресурсов

Факты из `local/HANDOFF.md` (вне git; проверено на станции 2026-11-17): 16 CPU, 30 GiB RAM,
встроенная Intel-графика, **без дискретного GPU и CUDA**, FFmpeg 8.0.1, Python 3.13 через `uv`.
В кэше Hugging Face лежат только три модели faster-whisper (large-v3-turbo, base, small),
≈ 2.2 ГБ; каталогов llama.cpp/ollama/torch/modelscope нет; `llama-server`, `llama-cli`, `ollama`,
`vllm` не установлены (раздел 6).

Локальный кандидат на этом хосте:

| Вариант | Веса (источник: [Q-gguf]) | Вывод для 30 GiB |
|---|---|---|
| Q4_K_M + mmproj Q8_0 | 18 557 053 952 + 1 325 020 128 Б ≈ 18.5 GiB | укладывается: остаток ≈ 11.5 GiB минус KV-кэш и контекст ⇒ ≈ 10 GiB под ОС, FFmpeg, faster-whisper |
| Q8_0 + mmproj Q8_0 | 32 484 494 336 + 1 325 020 128 = 33 809 514 464 Б ≈ 31.5 GiB | не входит |
| bf16 (GGUF или transformers) | 61 096 856 576 Б ≈ 56.9 GiB; transformers BF16 (Instruct, видеовход 15–120 с) 78.9–144.8 ГБ GPU [Q-card] | не входит |

Оценка памяти при mmap: веса читаются с диска по мере обращения, но для вывода модель эффективно
резидентна целиком — считать 18.5 GiB занятыми. KV-кэш для клипа 30–180 с (≈ 400–2300 аудио-
токенов плюс инструкция) — порядка сотен МБ; это оценка без источника.

Ожидаемая задержка: источника нет. **Гипотеза:** при 3B активных параметров на 16 CPU — минуты на
клип 30–60 с (prefill энкодера плюс генерация ответа в сотни токенов). UX-бюджет R11 (клип до
30 с — до 60 с обработки; до 180 с — до 180 с) этим не доказан и может не выполняться; измерить
холодную и тёплую задержку, model load и inference отдельно — в spike (#3).

## 4. Рекомендация

Один путь: **локальный Qwen3-Omni-30B-A3B-Instruct, GGUF Q4_K_M + mmproj Q8_0, через llama.cpp**
(`llama-server` на `127.0.0.1` и свободном порту, один анализ за раз, лимиты wall time и памяти
по R10). Почему:

1. Спека допускает локальный backend как полноценный MVP (R09); облачный — только явный opt-in
   после согласования провайдера, модели, записей, бюджета, хранения и при доверенном
   host-approval, которого нет ⇒ внешний вызов `blocked`.
2. Среди локальных audio-language моделей только Qwen3-Omni документирует русский как язык
   речевого понимания; Gate 0 требует именно этого, а не multilingual ASR. Меньший Qwen2.5-Omni-7B
   при прочих равных был бы предпочтительнее, но по критерию «русский» он — нет свидетельств.
3. Он укладывается в 30 GiB с запасом ≈ 10 GiB; единственная неизвестная — задержка на CPU.

Gemini 3.8 Flash — документированная альтернатива только после NOT READY локального spike и
отдельного решения владельца; не fallback и не автоматический переход.

Что проверит spike (#3): реальный аудиовход (trace с подачей bytes/PCM, T03); русский материал с
одинаковыми словами и разным звучанием (T04); transcript-only negative control; provenance по
R03; холодную/тёплую задержку и RAM по R11. Если модель только пересказывает текст или не
различает целевые явления — NOT READY, backend останавливается.

## 5. Нужны разрешения владельца

- **нужен владелец:** скачать `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF` —
  `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf` (18 557 053 952 Б ≈ 18.6 ГБ) и
  `mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf` (1 325 020 128 Б ≈ 1.33 ГБ); лицензия
  Apache-2.0 по метаданным карточки базовой модели (файла `LICENSE` в репозиториях нет); куда —
  в кэш Hugging Face или в отдельный каталог llama.cpp (решает владелец).
- **нужен владелец:** установить runtime llama.cpp — бинарь `llama-server` из релиза llama.cpp
  (при желании и `llama-mtmd-cli` для ручной проверки); версия — последний стабильный релиз на
  момент установки. `llama-cpp-python` не рассматривается: описанный маршрут `/completion` +
  `multimodal_data` — это `llama-server`, а не его Python-обёртка.
- **нужен владелец:** лимиты spike — число записей, их длительность, wall time, RAM.
- **при выборе Gemini (альтернатива):** провайдер Google Gemini API, модель `gemini-3.8-flash`,
  платный тариф; конкретные записи/диапазоны; число запросов; бюджет (оценка ≈ $0.0014/мин
  входа + вывод по $3.75/1M, с 2027 — вдвое); условия хранения («limited period», страна не
  фиксируется); доверенный host-approval по R09 — пока его нет, вызов `blocked`.
- **ждёт #3:** сам spike с inference.

## 6. `doctor` и кэш моделей до/после

Команды `doctor` на `origin/main` ещё нет (нет `pyproject.toml` и пакета); заменена ручной
инвентаризацией. Команды — из Decision к #22, плюс листинг `snapshots` (у
`faster-whisper-large-v3-turbo` в кэше нет каталога `refs/`, поэтому строка `cat …/refs/main`
штатно даёт ошибку, а ревизия видна по `snapshots`). Запускались в начале и в конце работы; вывод
вставлен дословно (домашний каталог заменён на `~`).

```
date -u +%Y-%m-%dT%H:%M:%SZ
ls -1 ~/.cache/huggingface/hub
du -sh ~/.cache/huggingface/hub ~/.cache/huggingface/xet
cat ~/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/refs/main
ls -d ~/.cache/llama.cpp ~/.ollama ~/.cache/torch ~/.cache/modelscope 2>&1
command -v llama-server llama-cli ollama vllm 2>&1 || echo "none installed"
uv tool list
ls -1 ~/.cache/huggingface/hub/models--*/snapshots
```

До начала работы:

```
2026-11-17T09:48:25Z
CACHEDIR.TAG
models--Systran--faster-whisper-base
models--Systran--faster-whisper-small
models--mobiuslabsgmbh--faster-whisper-large-v3-turbo
2.2G	~/.cache/huggingface/hub
108K	~/.cache/huggingface/xet
cat: ~/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/refs/main: No such file or directory
ls: cannot access '~/.cache/llama.cpp': No such file or directory
ls: cannot access '~/.ollama': No such file or directory
ls: cannot access '~/.cache/torch': No such file or directory
ls: cannot access '~/.cache/modelscope': No such file or directory
none installed
graphifyy v0.9.100
- graphify
- graphify-mcp
~/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots:
ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66

~/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots:
536b0662742c02347bc0e980a01041f333bce120

~/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots:
0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf
```

После завершения работы:

```
2026-11-17T09:50:48Z
CACHEDIR.TAG
models--Systran--faster-whisper-base
models--Systran--faster-whisper-small
models--mobiuslabsgmbh--faster-whisper-large-v3-turbo
2.2G	~/.cache/huggingface/hub
108K	~/.cache/huggingface/xet
cat: ~/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/refs/main: No such file or directory
ls: cannot access '~/.cache/llama.cpp': No such file or directory
ls: cannot access '~/.ollama': No such file or directory
ls: cannot access '~/.cache/torch': No such file or directory
ls: cannot access '~/.cache/modelscope': No such file or directory
none installed
graphifyy v0.9.100
- graphify
- graphify-mcp
~/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots:
ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66

~/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots:
536b0662742c02347bc0e980a01041f333bce120

~/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots:
0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf
```

Отличие двух копий — только строка `date`; остальное совпадает построчно
(`sha256sum` обеих копий без первой строки: `156e934a0a37677965ed7980cbbf89d7d4d7f0028be220caf28e4897e22df93a`).

## 7. Источники

Все прочитаны 2026-11-17 при подготовке Decision к #22; пометка «перепроверено» — прочитано
повторно в ходе этой задачи, 2026-11-17.

| Метка | Источник | Что взято |
|---|---|---|
| [Q-card] | https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct | карточка: языки, таблица памяти BF16, бенчмарки MMAU/MMSU/VoiceBench; лицензия — перепроверено через `https://huggingface.co/api/models/Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| [Q-gh] | https://github.com/QwenLM/Qwen3-Omni | README: 19 языков речевого входа (ru), `disable_talker()` |
| [Q-paper] | https://arxiv.org/html/2509.17765 | техотчёт: AuT 0.6B / 16 kHz / 12.5 Hz, 40 мин, Table 3, §9.2, ограничения Thinking |
| [Q-gguf] | https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/tree/main | размеры GGUF; байты и отсутствие лицензионных метаданных — перепроверено через `https://huggingface.co/api/models/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF?blobs=true` |
| [L-mm] | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md | поддержка аудио-моделей, `--mmproj`, `-hf …:Q4_K_M`; Qwen2-Audio без GGUF |
| [L-server] | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | `/completion` с `multimodal_data`, мультимодальность целиком «experimental» |
| [G-audio] | https://aistudio.google.com/docs/audio (редирект с ai.google.dev/gemini-api/docs/audio) | форматы, 9.5 ч, 32 ток/с, 20 МБ, таймкоды, 16 Kbps/моно, «emotion detection» |
| [G-price] | https://aistudio.google.com/docs/pricing | тарифы 3.8 Flash; 3.5 Flash-Lite ($0.30), 2.5 Flash ($1.00); free vs paid |
| [G-terms] | https://aistudio.google.com/docs/terms | Unpaid/Paid Services, логирование, хранение, DPA |
| [G-models] | https://aistudio.google.com/docs/models | список моделей, deprecation 2.5 и роутинг с 2026-12-01 |
| [G-model] | https://aistudio.google.com/docs/models/gemini-3.8-flash | модальности, лимит входа 1 048 576 токенов, дата обновления |

Отклонённые кандидаты (раздел 1.1), прочитаны 2026-11-17:
https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct, https://arxiv.org/abs/2407.10759,
https://huggingface.co/Qwen/Qwen2.5-Omni-7B,
https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/tree/main,
https://huggingface.co/mistralai/Voxtral-Mini-3B-2507,
https://huggingface.co/microsoft/Phi-4-multimodal-instruct,
https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_1-8b,
https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct,
https://developers.openai.com/api/docs/guides/audio-chat-completions,
https://developers.openai.com/api/docs/pricing, https://developers.openai.com/api/docs/guides/your-data,
https://aistudio.google.com/docs/models/gemini-3.5-flash-lite.

[Q-card]: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
[Q-gh]: https://github.com/QwenLM/Qwen3-Omni
[Q-paper]: https://arxiv.org/html/2509.17765
[Q-gguf]: https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/tree/main
[L-mm]: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
[L-server]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
[G-audio]: https://aistudio.google.com/docs/audio
[G-price]: https://aistudio.google.com/docs/pricing
[G-terms]: https://aistudio.google.com/docs/terms
[G-models]: https://aistudio.google.com/docs/models
[G-model]: https://aistudio.google.com/docs/models/gemini-3.8-flash
