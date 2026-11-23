# Среда, чистая установка и ресурсный профиль

Закреплённая среда Sound Scout, проверка чистой установки, проверка истории на приватные данные и
ресурсный профиль `analyze` против UX-бюджета R11 (`docs/spec/REQUIREMENTS.md` R11, T10;
`docs/spec/AGENT_PROMPT.md` D.16; задача [#41](https://github.com/renidere/sound-scout-plugin/issues/41)).
Версии, лицензии и контрольные суммы здесь не повторяются: их источник — `uv.lock` и
`docs/dependencies.md`.

## 1. Закреплённая среда

- **Python 3.13** — `.python-version` и `requires-python = ">=3.13,<3.14"` в `pyproject.toml`.
- **`uv` и FFmpeg** — версии на станции разработки и способ установки: `docs/dependencies.md` §2.
  В CI FFmpeg ставится из apt runner'а, `uv` — `astral-sh/setup-uv` той версии, что в §2.
- **Python-пакеты** — источник истины `uv.lock`; установка только `uv sync --locked`. Прямые
  зависимости с версиями и лицензиями — `docs/dependencies.md` §1; что §1 не разошёлся с
  `uv.lock`, проверяет `tests/test_dependencies_doc.py`. Полный закреплённый список со всеми
  транзитивными пакетами выводит `uv export --locked --no-hashes`.

## 2. Receipts моделей и как их получить без сети

Repo, ревизии и SHA-256 моделей — `docs/dependencies.md` §3 (Gemma 4 E4B-it GGUF и llama.cpp) и
§4 (Whisper large-v3-turbo, Silero VAD). Всё ниже работает offline и ничего не скачивает:

- `uv run sound-scout doctor --json` — состояние компонентов и отпечатки по локальным файлам
  (модель ASR берётся только из кэша, `local_files_only`).
- `uv run sound-scout smoke` — smoke-прогоны `decoder` и `acoustics` на синтетике;
  `SOUND_SCOUT_DEV_AUDIO=<запись> uv run sound-scout smoke asr` — ASR на development-записи.
  Smoke-receipts пишутся в `<data_root>/receipts/`, их читает `doctor`.
- Каждый run `analyze` с выполненным ASR хранит `asr/model-receipt.json`: модель, ревизию снапшота
  и отпечаток файлов, которыми получен транскрипт.
- Файлы GGUF сверяются вручную: `sha256sum` каждого файла против таблицы §3.1.

## 3. Проверка чистой установки

```sh
bash scripts/clean-install-check.sh
```

Скрипт клонирует **закоммиченный** HEAD во временный каталог (незакоммиченные правки в проверку
не попадают), ставит среду `uv sync --locked --all-groups`, гоняет `uv run pytest`, затем
`sound-scout doctor --json` и `sound-scout analyze` синтетического WAV (5 с, 48 кГц) с временной
конфигурацией `backend = "none"`. Конфигурация оператора и development-запись не участвуют
(`SOUND_SCOUT_CONFIG` и `SOUND_SCOUT_DEV_AUDIO` сняты), сеть закрыта (`HF_HUB_OFFLINE=1`).
Временный каталог удаляется при любом выходе.

Сводка: первая строка — `clean-install-check: <commit>`, затем строка на шаг
`<шаг> | ok|fail | <секунды> | <деталь>`:

| шаг | деталь |
|---|---|
| `clone` | дата коммита |
| `uv sync` | версия `uv` |
| `pytest` | итог pytest (`N passed, M skipped`) |
| `doctor` | `компонент=статус` для `decoder`, `asr`, `acoustics`, `audio_review` |
| `analyze` | код выхода и статусы результата и этапов |
| `expect` | `ok` или строки `fail: <поле>: expected …, got …` (`scripts/clean_install_expect.py`) |

Последняя строка — `result: ok` (код 0) или `result: fail (<шаг>)` (код 1); на первом `fail`
скрипт останавливается, полный вывод шага уходит в stderr с временным каталогом в виде `<tmp>`.

Ожидается: `doctor` показывает все четыре компонента, `audio_review = permission_required`,
сетевой guard активен; `analyze` возвращает `partial` с кодом 3, `prepare` и `acoustics` —
`completed`, `audio_review` — `unavailable` (`backend = none`). `asr` допускается `completed` или
`unavailable`: в CI модели нет, и там ожидается `unavailable`; на станции с моделью в кэше —
`completed`. `failed` — всегда ошибка. В CI это job `clean-install`: он вызывает скрипт с
аргументами `-m "not rehearsal"`, и скрипт передаёт их шагу `pytest`. Репетицию #149 на том же
коммите проверяет job `rehearsal` по шардам (#167).

## 4. Нет приватного в git

```sh
bash scripts/scan-history.sh [<rev>]
```

Проверяет всю историю ревизии (по умолчанию `HEAD`), включая файлы, удалённые позже:

- каждый когда-либо добавленный путь (в том числе переименованием): медиафайлы по расширению без
  учёта регистра, `.env`/`.env.*` (кроме `*.example`), `sound-scout.toml`, `data/`, `runs/`,
  `local/`, `.work-task/`, `.revmux/`, `.venv/`, `*.pem`, `*.p12`, `*.key`;
- каждая добавленная строка: паттерны ключей — та же строка `patterns`, что в
  `.claude/skills/work-task/scripts/scan-staged.sh` (расхождение ловит
  `tests/test_scan_history.py`), и любой путь `/home/<user>/`, кроме заглушек из переменной
  `home_allowlist` в начале скрипта (`someone`, `user`, `username`, `runner`, `example`).

Вывод — `ok` (код 0) или строки `BLOCK: <commit> <файл>[:<строка>] <причина>` (код 1); найденное
значение не печатается. В CI это job `history-scan` с полной историей (`fetch-depth: 0`).
Находка в уже запушенной истории — красный CI и решение владельца (ротация ключа, переписывание
истории); агент историю не переписывает, задача уходит в Need a Person. Тестовые «секреты» и
пути в тестах сканера собираются конкатенацией во время выполнения и в git не попадают.

## 5. Аппаратная конфигурация станции разработки

Из блока `host` профиля (раздел 6):

| параметр | значение |
|---|---|
| CPU | Intel Core i5-14400 |
| логических ядер | 16 |
| RAM | 31 294 MiB |
| ОС и ядро | Linux 7.0.0-43-generic |
| Python | 3.13.16 |
| uv | 0.13.1 |
| FFmpeg | 8.0.1-3ubuntu2 |
| GPU | не используется: CPU-only (`docs/decisions/0001-audio-backend.md`) |

## 6. Ресурсный профиль и UX-бюджет R11

```sh
SOUND_SCOUT_DEV_AUDIO=<запись> uv run python scripts/profile.py --output <вне репозитория>/profile.json
```

`scripts/profile.py` запускает настоящий `sound-scout analyze` дочерним процессом, по два раза
подряд на каждый вход: `cold` (первый прогон входа в сессии) и `warm` (повтор). Входы —
синтетика 30 и 180 с (формантный контроль `smoke.speech_like_control`: 1,2 с гласных каждые 3 с)
и, если задана `SOUND_SCOUT_DEV_AUDIO`, development-запись. Конфигурация и data root временные
(0700, удаляются после замера), `backend = "none"`, `[asr]` — из конфигурации оператора, deadline
worker'а поднят до 900 с, чтобы измерить фактическое время, а не получить `deadline_exceeded`.
`wall` — от запуска `analyze` до его завершения, включая старт Python; пиковый RSS — `ru_maxrss`
самого большого процесса дерева из `os.wait4`; времена этапов — из `work/stage-*.json` и
`asr/raw.json` run'а. В `profile.json` нет пути, имени файла, SHA-256, run id и текста записи.

Ограничения замера: page cache не сбрасывается (нужен root), поэтому `cold` — первый прогон в
сессии, а не холодный диск. Каждый `analyze` запускает новый worker и загружает модель ASR заново:
тёплой модели в процессе нет ни в CLI, ни в MCP, и `warm` отличается от `cold` только кэшем ОС.
Во время замера на машине шла другая работа (load average 2,74 перед стартом), поэтому разброс
cold/warm в пару секунд — шум, а не эффект прогрева.

Замер 2026-11-19, commit `6cb03224507f` (код профиля — рабочее дерево задачи #41 поверх него),
модель ASR `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, `backend = none`. Времена в секундах.

| вход | фаза | wall | prepare | acoustics | ASR load | ASR inference | peak RSS, MiB | бюджет | в бюджете |
|---|---|---|---|---|---|---|---|---|---|
| synthetic 30 с | cold | 11,4 | 0,18 | 0,19 | 2,3 | 7,3 | 2170 | 60 | да |
| synthetic 30 с | warm | 9,5 | 0,18 | 0,21 | 2,9 | 4,7 | 2171 | 60 | да |
| synthetic 180 с | cold | 35,0 | 0,46 | 0,85 | 1,5 | 30,6 | 2653 | 180 | да |
| synthetic 180 с | warm | 33,4 | 0,63 | 0,76 | 1,4 | 29,1 | 2653 | 180 | да |
| development recording, 10,6 с | cold | 8,8 | 0,17 | 0,12 | 1,6 | 5,4 | 2123 | 60 | да |
| development recording, 10,6 с | warm | 11,0 | 0,17 | 0,12 | 1,7 | 7,3 | 2123 | 60 | да |

Все шесть прогонов завершились `partial` (код 3): `prepare`, `acoustics` и `asr` — `completed`,
`audio_review` — `unavailable` (`backend = none`).

Вывод:

- Бюджет R11 проверен **только для пути без звучания** (decode, каналы, акустика, ASR): на этой
  станции он выдержан с запасом — 30 с входа за ≤ 11,4 с, 180 с за ≤ 35,0 с, пик памяти ≈ 2,6 GiB.
- **audio_review не измерен.** Backend слухового анализа не подключён (`backend = none`): Gate 0
  дал NOT READY (`docs/decisions/0007-gate0-result.md`). Поэтому соответствие R11 для полного
  сценария со слоем звучания **не установлено**, и измерения выше нельзя выдавать за него.
- Превышений бюджета в измеренном пути нет. Если после выбора backend полный сценарий выйдет за
  бюджет, это ограничение: по R11 иной режим согласует владелец до заявления готовности.
