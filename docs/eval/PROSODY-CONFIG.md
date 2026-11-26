# Конфигурация прогонов просодии

Документ называет конфигурацию для прогонов просодии (#201):

- серии 1–3 `prosody-measure-1`: `python -m sound_scout.eval.prosody prepare|sanity|run
  [--series N]` (решение 0019);
- подтверждающий прогон `prosody-confirm-1`: `python -m sound_scout.eval.prosody_confirm
  <команда>` (решение 0034).

## Что нужно

- `data_root` — абсолютный путь к каталогу `data/` основного checkout репозитория. Там лежат
  источники корпуса `corpus/sources/`, `corpus/pilot-1/`, клипы Gate 0 и каталоги серий
  `prosody-measure-1*/` и `prosody-confirm-1/`.
- Дерево задачи `.work-task/<n>` своего `data/` не имеет. Его `data_root` — тоже `data/`
  основного checkout.
- `allowed_roots` загрузчик требует всегда. Команды просодии вход через него не читают. Подходит
  любой существующий пустой каталог вне `data_root`.
- `[network] allow = false`. Сеть закрыта, это часть Q1.

Пример — `docs/examples/eval-config.example.toml`.

## Как задать

1. Скопируйте пример вне дерева, например в `~/.cache/sst/<имя>/sound-scout.toml`.
2. Замените оба пути на абсолютные существующие каталоги.
3. Укажите файл в переменной окружения и запустите команду из корня дерева:

```bash
export SOUND_SCOUT_CONFIG=~/.cache/sst/<имя>/sound-scout.toml
uv run python -m sound_scout.eval.prosody_confirm check-fresh
```

Команды берут путь к решению (`--decision`) относительно текущего каталога. Поэтому каталог
запуска — корень дерева.

## Что будет с конфигурацией оператора

Без `SOUND_SCOUT_CONFIG` загрузчик читает `sound-scout.toml` из текущего каталога. В дереве
задачи это копия конфигурации оператора. Её `data_root` — data root оператора, там нет
источников корпуса. Тогда Q1 отказывает с `missing`. В #199 `prosody_confirm prepare` с такой
конфигурацией записал INVALID `result.json` в data root оператора. Перед прогоном проверьте
`SOUND_SCOUT_CONFIG`.

`check-fresh` с неверной конфигурацией тоже не безвреден. Он создаёт
`<data_root>/prosody-confirm-1/`. При повторе клипа или сбое их файлов он пишет INVALID
`result.json`. После этого `prepare`, `import`, `sanity` и `run` отказывают. Поэтому проверяйте
`SOUND_SCOUT_CONFIG` перед любой командой `prosody_confirm`.
