# 0026. Стек GPU-эталона: образ, развёртывание, сторож, guard хоста

Статус: принято (Decision в [#147](https://github.com/renidere/sound-scout-plugin/issues/147#issuecomment-7071797977),
эпик [#146](https://github.com/renidere/sound-scout-plugin/issues/146)). Дата: 2026-11-22.

Аренды не было, расходов не было, к Vast.ai не было ни одного вызова. Всё ниже проверено на
поддельном API и поддельном vLLM на 127.0.0.1. Первый настоящий запуск — пилот
[#148](https://github.com/renidere/sound-scout-plugin/issues/148) по одобрению пакета (#149).

## 1. Что сделано

- `src/sound_scout/eval/vast.py` — один модуль на stdlib `urllib`, запуск
  `python -m sound_scout.eval.vast deploy|watch|cutoff|destroy|status|dry-run`. Он не входит в
  CLI `sound-scout` и в MCP. Внешний хост один — `API_BASE = https://console.vast.ai`. Анализ
  записей остаётся offline (R09).
- `src/sound_scout/eval/gpu_host/sitecustomize.py` и `hostrun.py` — stdlib, Python 3.10+, без
  импорта `sound_scout`. На хост они идут через `scp` в `/workspace/ss/` (guard — в
  `/workspace/ss/guard/`), их SHA-256 читается с хоста и сверяется до первого использования.
- `src/sound_scout/eval/vast_fake.py` — поддельный API Vast.ai и поддельный vLLM для тестов и
  dry run.
- `gate0.py`: runtime `vllm` и `prepare --controls-only`.
- Закрепление образа и весов — `docs/dependencies.md` раздел 7; время — `docs/gpu-timing.md`;
  пример пакета — `docs/examples/gpu-package.example.json`.

## 2. Образ и runtime

`IMAGE` в коде: `vllm/vllm-openai:v0.34.0`, индекс `sha256:20fdb089…1ce1`, amd64
`sha256:cf5b2fdd…c484`, 10 036 518 499 байт. В create идёт `IMAGE_REF` — тег вместе с digest
индекса. `RUNTIME`: vLLM 0.34.0, torch 2.14.0, CUDA 13.0 (`torch.version.cuda`), Python 3.12,
transformers `>=5.16.1,<5.23.0`. Preflight сверяет vLLM, torch и CUDA точно, transformers — по
диапазону; Python записывается, но не сверяется.

`MODELS` — ревизии и файлы двух моделей из публичного API HF; `spk_dict.pt` исключён.

**CUDA без окружения SSH (#158).** Попытка 1 пилота #148 остановилась на preflight:
`runtime differs from the pin in ['cuda']`. Тогда preflight брал `CUDA_VERSION` из окружения
SSH-сессии. SSH-сессия не обязана наследовать `ENV` образа. Пин `13.0.3` тоже не совпадал с
образом. Найденные значения в журнал не попали, поэтому причина не доказана. Метаданные образа из
реестра Docker Hub (2026-11-22, без загрузки слоёв):

| что | значение |
|---|---|
| индекс тега `v0.34.0` | `sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1` |
| манифест linux/amd64 | `sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484` |
| config blob amd64 | `sha256:0372147e2b3daa1baa5c6aa9b90ae5de2f926e25742650f91fa24dbca9cc645a`, 46 307 байт |
| `ENV` | `CUDA_VERSION=13.0.2`, `NV_CUDA_CUDART_VERSION=13.0.96-1`, `NVIDIA_REQUIRE_CUDA=cuda>=13.0 …`, `VLLM_IMAGE_TAG=vllm/vllm-openai:v0.34.0` |
| labels | `ai.vllm.image.tag=vllm/vllm-openai:v0.34.0`, `org.opencontainers.image.revision=62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` |
| установка torch (история) | `--index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION \| cut -d. -f1,2 \| tr -d '.')`, то есть `cu130` |

Решение. Preflight сверяет `torch.version.cuda` с пином `13.0` в формате этого источника. Torch
отдаёт CUDA своей сборки, окружение сессии на это не влияет. `CUDA_VERSION` образа (`13.0.2`)
закреплён digest образа и записан в `IMAGE`. На хосте preflight только записывает его из
окружения первого процесса контейнера (`cuda_image`, `/proc/1/environ`) и из сессии (`cuda_env`).
Все найденные значения — vllm, torch, cuda, transformers, драйвер — попадают в `detail` шага
`preflight` до сравнения. Отказ оставляет их в журнале.

**Отличие от пакета #148.** #158 меняет пин CUDA в `RUNTIME` с `13.0.3` на `13.0`. Предрегистрация
`0027` § 3 по-прежнему называет `13.0.3`; запись не меняется. Поэтому следующая попытка #148 — новый
пакет. Ей нужно новое разрешение владельца.

**Честная граница CPU smoke.** Образ на станцию не загружается (решение владельца). Dry run
показывает работу Gate 0 клиента, guard и HTTP-контракта vLLM на CPU станции. Он не проверяет
образ, vLLM, модель и слух. Утверждения «образ запускает клиент» здесь нет: первый запуск образа —
preflight и пилот X1/X2 в #148.

## 3. Ключ и HTTP

- `read_api_key` открывает `~/.config/vastai/vast_api_key` с `O_NOFOLLOW` и проверяет `fstat`:
  обычный файл, владелец — этот пользователь, нет прав группы и других, не больше 512 байт, один
  ASCII-токен без пробелов длиной 16–256. Текст ошибки не содержит ключа.
- Ключ не идёт в argv, окружение, URL, журнал, лог, исключения и на хост. Таймеры получают путь
  журнала; в `opened` записан путь к файлу ключа, не ключ. Каждый процесс читает ключ сам.
- `Api`: `build_opener(ProxyHandler({}), NoRedirect())`, `Authorization` через
  `add_unredirected_header`, 3xx — ошибка `Redirected`. База — только `API_BASE` или http на
  loopback (поддельный API). Повторов нет; повтор — следующий тик сторожа, только чтения и destroy.
- `redact` закрывает ключ в теле ошибок провайдера и в каждой записи журнала. `check_no_secret`
  проверяет payload create и каждый файл до `scp`: ключ и имена `VAST_API_KEY`, `HF_TOKEN`,
  `Authorization` — отказ.

## 4. Журнал, create, сторож

- Журнал `<data_root>/gpu/<attempt>/journal.jsonl` (0600 в каталоге 0700): append-only JSON
  Lines, `os.write` одной строки, затем `fsync`. Порванная последняя строка пропускается и
  удаляется перед следующей записью; битая строка в середине — отказ. Новая попытка запрещена,
  пока есть попытка без `closed`.
- Create — один PUT после `create_requested` с `fsync`. Предусловия: `armed`, оба таймера
  активны, ручной тик сторожа дал `idle`, на аккаунте нет инстансов, свежий offer в конверте,
  оценка по PROPOSAL § 7 не выше потолка. Любой исход, кроме `success` и целого `new_contract`, —
  `create_outcome_unknown`; второй PUT в попытке невозможен.
- Таймеры `systemd-run --user`: `…-watch` раз в 60 с и `…-cutoff` по `OnCalendar` в UTC на
  дедлайн и на дедлайн + 15 мин (конец поиска потерянного create). Ключа и `--setenv` в argv нет.
  `arm` проверяет `Linger=yes` и отказывает, если интерпретатор лежит в `.work-task/`.
- Тик (`watch_tick`): свои id — из `create_response` и по уникальной метке попытки. Destroy всех
  своих id: дедлайн, неизвестный исход create, больше одного инстанса с меткой, пульс агента
  старше 300 с, `destroy_requested`, расход ≥ раннего порога. 200 или 404 на DELETE — затем
  проверка отсутствия по списку с меткой и по id; потом `absence_verified`, `closed` и остановка
  таймеров. Чужая метка не трогается. Потерянный create без инстанса ищется до дедлайн + 15 мин.

## 5. Сетевой guard на хосте

- `sitecustomize.py` ставит `sys.addaudithook` во всех Python-процессах дерева vLLM (дети через
  `spawn` получают его по `PYTHONPATH`). Разрешены `AF_UNIX` и loopback (`127.0.0.0/8`, `::1`);
  остальные connect, bind, sendto, sendmsg и поиск имён — `PermissionError` до системного вызова.
  Каждый отказ — строка в `$SS_GUARD_LOG`.
- `hostrun.py serve`: окружение по allowlist плюс offline-переменные, новая сессия, каждые 0,5 с —
  сокеты всех процессов сессии по `/proc/<pid>/fd` и `/proc/net/{tcp,tcp6,udp,udp6}`. Сокет с
  не-loopback концом (и listener не на loopback) — `SIGKILL` сессии и запись `violation`. Там же
  RSS дерева и память GPU раз в 5 с; RSS сверх лимита и дедлайн — остановка.
- `hostrun.py selftest`: loopback проходит; `192.0.2.1:9`, `getaddrinfo("huggingface.co")` и
  ребёнок `spawn` — отказ хука; native connect через `ctypes` к своему не-loopback адресу —
  детектор убивает процесс; `unshare -n true` — `netns_available`. Каждая проба сначала проверяет,
  что хук загружен, и без него ничего не отправляет.
- **Граница.** Хук закрывает Python-код (HF hub, телеметрия, `get_ip` vLLM). Native-сокеты
  (libzmq, NCCL, gloo) детектор видит только постфактум, с шагом опроса. Это защита от случайного
  выхода в сеть, а не песочница против администратора хоста. Риск доступа хоста к данным владелец
  принимал в #149.

## 6. Шаги deploy и dry run

`deploy --package P --approval URL`: `opened`, `arm`, `create`, `ready` (`running` и порт
`22/tcp`; `exited`/`unknown`/`offline` — destroy; затем SSH-ключ попытки на инстанс и опрос
`ssh … true` до ответа, не дольше `ssh_timeout_s`; create не повторяется), `preflight`,
`weights` (`hf download` по ревизии, сверка `hostrun.py weights`, квитанция на станцию), для
каждой модели `smoke` (сервер под guard, туннель, пилот K1 на X1/X2) и `run`
(`gate0.run_spike(runtime="vllm")`), `download` (SHA-256 архива и каждого файла по `SHA256SUMS`),
`destroy`. Любая ошибка ведёт в destroy. Пути `decision` пакета — относительно корня репозитория:
`deploy` запускается из него.

`dry-run` идёт против поддельного API на 127.0.0.1: каталог вместо хоста, поддельные `vllm`, `hf`
и `nvidia-smi`, поддельные метаданные пакетов runtime, синтетические веса, настоящий
`hostrun.py serve` и guard, настоящий Gate 0 клиент на `--controls-only`. Таймеры не взводятся:
argv записывается в `armed` с `dry_run: true`, системные команды в dry run запрещены. Вывод —
`timeline.json` и `fake-api.jsonl`. Прогон 2026-11-22: 11 шагов, все `ok`, один PUT, один DELETE,
отсутствие по id и метке, 3,9 с.

## 7. Решения агента внутри Decision

- SHA-256 файлов хоста сравнивается со станцией по выводу `sha256sum <файлы>`, а не
  `sha256sum -c`: та же проверка без путей в файле сумм.
- Квитанция весов забирается сразу после шага `weights`: `run` идёт раньше `download`. Та же
  квитанция потом входит в архив и сверяется по `SHA256SUMS`.
- Второй срок cutoff (дедлайн + 15 мин) — второй `--on-calendar` того же таймера. Два
  `OnCalendar` у transient-таймера проверены на станции (systemd 259) таймером с `/bin/true` на
  2099 год; он остановлен и удалён.
- Блокировку держит агент только на время записи в журнал и create, а не весь deploy: иначе тики
  сторожа пропускались бы весь прогон.
- X1 и X2 — 1 кГц при −20 dBFS: различается только длительность.
- Чтение offer — `POST /api/v0/bundles/` с фильтром `{"ask_contract_id": {"eq": N}}`: по `id`
  реальный API отдаёт пустой `offers`. Список инстансов — `GET /api/v1/instances/` со слешем на
  каждой странице: без слеша реальный API отвечает 301, а стек отказывает в редиректе
  (`Redirected`). Обе формы проверены чтением реального API в #148; поддельный API отвечает на
  неверные формы так же (#156). Ответ `show` принимается и объектом, и списком в `instances`;
  404 или пустой `instances` — инстанса нет. Формы `PUT /asks` и ответа `DELETE` без аренды не
  проверены.
- Детектор считает listener на `127.0.0.0/8` и `::1` допустимым, как и хук.
- Если у хоста нет не-loopback адреса, native-проба `unavailable`, а не провал.

## 8. Что не измерено

Время шагов, pull образа, приём digest, `ssh_direct` с образом vLLM, скорость HF, VRAM и RSS,
`netns_available`, время destroy, лаг биллинга, действие `kill -TERM 1` на тарификацию, поведение
vLLM с `cache_prompt`, `model` и `/tokenize` — всё это измеряет только #148
(`docs/gpu-timing.md`, раздел «Меряет только платный пилот»).
