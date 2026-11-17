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

Подкоманды `doctor`, `analyze`, `compare`, `serve` пока завершаются с кодом 2 и сообщением
`not implemented`. Локальная конфигурация — `sound-scout.toml` (копия `sound-scout.example.toml`,
в git не входит).

Для pre-commit-хуков установите [lefthook](https://github.com/evilmartians/lefthook) и выполните
`lefthook install`: перед коммитом по staged-файлам пройдут `ruff check` и `ruff format --check`.

## Проверки

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest             # -m "not slow" пропускает тесты с реальными моделями
```

Те же проверки запускает CI (`.github/workflows/ci.yml`) на каждый push в `main`.
