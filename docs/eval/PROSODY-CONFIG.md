# Конфигурация прогонов просодии

Документ называет конфигурацию для прогонов просодии (#201):

- серии 1–3 `prosody-measure-1`: `python -m sound_scout.eval.prosody prepare|sanity|run
  [--series N]` (решение 0019);
- подтверждающий прогон `prosody-confirm-1`: `python -m sound_scout.eval.prosody_confirm
  <команда>` (решение 0034). `sanity` и `run` идут только из замороженного дерева
  `local/frozen/prosody-confirm-1` (решение 0037 § 4.1);
- подтверждающий прогон `prosody-rupro-1`: `python -m sound_scout.eval.prosody_rupro <команда>`
  (решение 0037). Источники RuPro лежат в `data/prosody-rupro-1/sources/`. `prepare`, `sanity` и
  `run` идут только из замороженного дерева `local/frozen/prosody-rupro-1`.

## Что нужно

- `data_root` — абсолютный путь к каталогу `data/` основного checkout репозитория. Там лежат
  источники корпуса `corpus/sources/`, `corpus/pilot-1/`, клипы Gate 0 и каталоги серий
  `prosody-measure-1*/` и `prosody-confirm-1/`.
- Дерево задачи `.work-task/<n>` своего `data/` не имеет. Его `data_root` — тоже `data/`
  основного checkout.
- `allowed_roots` загрузчик требует всегда. Команды просодии вход через него не читают. Подходит
  любой существующий пустой каталог вне `data_root`.
- `[network] allow = false`. Сеть закрыта, это часть Q1. Исключение — `prosody_rupro fetch`: ему
  нужна своя конфигурация с `[network] allow = true`, без неё команда отказывает.

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
источников корпуса `corpus/sources/`.

Каждая команда `prosody_confirm` и `prosody_rupro` проверяет `<data_root>/corpus/sources/` сразу
после загрузки конфигурации (#201). Если каталога нет или он пуст, команда ничего не пишет и
выходит с кодом 2. Ошибка называет путь к файлу конфигурации, каталог источников и
`SOUND_SCOUT_CONFIG`.

Проверка есть в коде основного дерева с #201. Замороженные деревья
`local/frozen/prosody-confirm-1` и `local/frozen/prosody-rupro-1` содержат более старый код без
неё. Перед командой из замороженного дерева проверьте `SOUND_SCOUT_CONFIG` вручную.

После #201 `prosody_confirm.py` основного дерева не совпадает с пином кода `0034` § 7 (SHA-256
`8009636aad669e7c172f902e6edf6e34e679539b1793e71e9aa6087572693675`). Основной прогон требует
пройденную санитарную попытку с тем же SHA-256 модуля, а обе попытки уже сделаны. Поэтому
`sanity` из основного дерева на настоящем `data_root` пошла бы в ветку INVALID. Каталог
`data/prosody-confirm-1/` закрыт на запись (`0037` § 4.1), и команда выходит с кодом 2 без
записи. `sanity` и `run` `prosody-confirm-1` запускайте только из
`local/frozen/prosody-confirm-1`.

Команды серий 1–3 `python -m sound_scout.eval.prosody` этой проверки не имеют: `prosody.py`
закреплён решением 0034. С неверной конфигурацией `prepare` пишет INVALID `result.json` в
`<data_root>/prosody-measure-1*/`. Так было в #199 с `prosody_confirm prepare` до проверки.
Вердикты серий 1–3 уже записаны, повтора нет. Перед командой `prosody` проверьте
`SOUND_SCOUT_CONFIG`.

`check-fresh` на верном `data_root` по-прежнему пишет INVALID `result.json`, если находит
повтор клипа или сбой файлов клипов. Это правило решений 0034 и 0037, а не ошибка конфигурации.
