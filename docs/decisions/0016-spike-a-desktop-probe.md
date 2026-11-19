# 0016 — Spike A: зонд штатного Desktop (owner сессии, file.attach, настоящий ли ACK)

- **Статус:** инструкция готова; результаты ожидают прогона владельцем.
- **Дата:** 2026-11-20.
- **Решение:** [Decision в #95](https://github.com/renidere/sound-scout-plugin/issues/95#issuecomment-7024314864).
- **Задача:** [#95](https://github.com/renidere/sound-scout-plugin/issues/95); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92). По порядку этапов из 0013 § 5
  (решение владельца 20 ноября 2026) прогон относится к этапу Windows-клиента
  ([#115](https://github.com/renidere/sound-scout-plugin/issues/115)), который стартует после
  [#114](https://github.com/renidere/sound-scout-plugin/issues/114).
- **Основание:** `docs/decisions/0013-voice-note-integration.md` § 3.3 (рекордер), § 3.4
  (отправка), § 5 шаг 2 (Spike A); `docs/decisions/0015-spike-b-backend-package.md` (Spike B,
  отдельный пакет `sound-scout`); Hermes Agent `3b42157858` (закоммиченный SDK, без локального
  патча `retainSession`); recorder Voice Note v0.4.2 (`ced69ea`, MIT).
- **Код:** `hermes/probe/sound-scout-probe-a/plugin.js` (id `sound-scout-probe-a`),
  `scripts/probe_a_check_attach.py`; тесты — `tests/js/probe_a.test.mjs`,
  `tests/js/load_plugin.mjs`, `tests/test_hermes_probe_a.py`.

## Вопрос

Рекордер 0013 § 3.3 должен работать только на штатных вызовах Desktop Plugin SDK. Никто ещё не
выполнял эти вызовы из плагина на Windows-клиенте владельца против удалённого gateway станции.
Spike A отвечает на четыре вопроса:

1. какие `focusedStoredSessionId`, `focusedSessionOwner` и `profileRoutes` видит плагин у старого,
   переименованного и нового чата, и где Desktop ищет `desktop-plugins`;
2. доходит ли синтетический WAV 30 с через `retainProfile` → `session.resume` → `file.attach` байт
   в байт (сверка SHA-256 на станции);
3. настоящий ли ACK у `prompt.submit` с `submission_id` и у `composer.submit`, когда тестовый чат
   на экране и когда он скрыт;
4. что из этого выбирает #103 для «Отправить».

Тесты репозитория запускают зонд на поддельном SDK. Они доказывают только контракт зонда (порядок
вызовов, release, вердикты, байты WAV, обезличивание), а не поведение Hermes. Ответы на вопросы дают
только прогон владельца и журнал.

## Устройство зонда

Один ESM-файл без JSX и без backend-половины. Импорт — только `@hermes/plugin-sdk` (`host`,
`PALETTE_AREA`). `host.retainSession` не используется: его нет в закоммиченном SDK.

`register(ctx)` пишет запись `loaded` (что есть в SDK: `typeof` у `profileRoutes`,
`retainProfile`, `requestProfile`, `composer.submit`, `retainSession`, `ctx.files`,
`ctx.os.writeClipboard`, `hermesDesktop.desktopPluginsRoot`; есть ли `crypto.subtle`), подписывается
на `focusedStoredSessionId` и `focusedSessionOwner` (запись `focus_change` при каждом изменении,
отписка через `ctx.onDispose`) и регистрирует восемь команд палитры. Сам `register` не вызывает
`requestProfile`, `retainProfile`, `composer.*` и сеть.

| команда | что делает | пишет в чат или на станцию |
|---|---|---|
| `Probe A: snapshot — old chat` / `— renamed chat` / `— new chat` | запись `snapshot`: `stored`, `runtime`, `owner`, `profile`, `active_profile`, `connection`, `gateway`, `routes`, `owner_matches`, `plugins_root` | нет |
| `Probe A: mark this chat as test target` | `target = {stored, owner}` из текущего фокуса; без открытого чата — отказ `no_chat`, во время пишущей команды — `busy` | нет |
| `Probe A: attach synthetic 30 s WAV to target` | `retainProfile` → `session.resume` (`omit_messages: true`) → `file.attach` (таймаут 120 с) → `release` в `finally` | файл `sound-scout-probe-a-<token>.wav` в `attachments/` станции, без сообщения |
| `Probe A: prompt.submit to target` | `prompt.submit {session_id, text, submission_id}`, повтор того же `submission_id`, `prompt.receipt`, проверка чтением | одно сообщение и один ход агента в тестовом чате |
| `Probe A: composer.submit to target` | `host.composer.submit(stored, text)`, проверка чтением | одно сообщение (если ACK настоящий) и один ход агента |
| `Probe A: copy log` | журнал в буфер обмена | нет |

Пишущие команды без отмеченной цели отказывают (`no_target`) и ничего не вызывают; пока идёт
предыдущая пишущая команда, следующая отказывает (`busy`). Методы RPC зонда — только
`session.resume`, `file.attach`, `prompt.submit`, `prompt.receipt`, все через
`host.requestProfile`.

**Маршрут** (attach и оба submit): owner из `target`; если он `null` и цель на экране — owner
перечитывается из `focusedSessionOwner` (`owner_fallback: "reread"`); маршрут — единственный из
`profileRoutes()` с тем же `connectionId` и `profile`; owner всё ещё `null` при одном маршруте —
этот маршрут (`owner_fallback: "single_route"`); иначе отказ `ambiguous_route` с числом совпадений
`matches`, без `retainProfile`.

**Проверка чтением** (оба submit): каждые 2 с в течение 30 с `session.resume {session_id: stored}`
без `omit_messages`; в `messages` считаются сообщения пользователя с маркером
`sound-scout-probe-a <token>`. Цикл идёт до конца окна, чтобы увидеть поздний дубль. В журнал
попадают только `occurrences`, `present_after_ms`, `message_count` и счётчики чтений
(`verify_reads`, `verify_valid_reads` — ответы с массивом `messages`, `verify_read_errors`), не
тексты. Маркера нет, а чтение хоть раз упало или ни одно не вернуло `messages` — проверка не
состоялась: вердикт обоих submit `inconclusive`, а не `false_ack` или `refused`.

**Вердикт `prompt.submit`:** `duplicate` — маркер больше одного раза; `refused` — ошибка RPC и
маркера нет; `false_refusal` — ошибка RPC, а сообщение есть; `real` — есть `admission_id`, повтор
вернул ту же admission и маркер ровно один; `false_ack` — есть `admission_id`, маркера нет;
`inconclusive` — всё остальное (например, повтор вернул другую admission при одном маркере).

**Вердикт `composer.submit`:** `duplicate` — маркер больше одного раза; `real` — ACK `true` и маркер
один; `false_ack` — ACK `true`, маркера нет; `false_refusal` — ACK `false`, маркер есть; `refused`
— ACK `false`, маркера нет.

Текст сообщения: `sound-scout-probe-a <token> (<метод>, visible|hidden): тестовое сообщение зонда
Sound Scout, ничего не делай, ответь одним словом «ок».` Ссылки `@file:` в нём нет: ход агента не
запускает анализ записи (связка attach + submit — #103/#105).

### Журнал

JSON Lines, запись `{seq, at, event, …}`. После каждой записи весь журнал пишется в
`ctx.files.writeText('data', 'probe-a-log.jsonl', …)` (папка данных плагина); ошибка записи
фиксируется один раз записью `log_write_failed` и зонд не останавливает. События: `loaded`,
`focus_change`, `snapshot`, `target_marked`, `attach`, `submit`, `refused`, `error`.

Обезличивание делается при записи:

- пути: `C:\Users\<имя>` → `%USERPROFILE%`, `/home/<имя>` и `/root` → `~`;
- каждый `connectionId` → псевдоним `conn-1`, `conn-2`… (таблица соответствия только в памяти);
- тексты ошибок обрезаются до 300 символов, пути и `connectionId` в них заменяются так же;
- названия чатов, тексты сообщений и истории, токены доступа в журнал не попадают; попадают id
  сессий, профили, булевы значения, счётчики, коды ошибок и длительности шагов в мс.

## Эталонный WAV

48 кГц, стерео, PCM16, 30 с: 1 440 000 кадров, **5 760 044 байт**. Отсчёты — целые треугольники
`tri(n, P, A)` (`t = floor((n mod P)·4A/P)`; `t < A` → `t`; `t < 3A` → `2A − t`; иначе `t − 4A`):
левый канал `tri(n, 100, 8000)`, правый `tri(n, 150, 6000)`. Зонд переводит их во float
(`v / 32768` для отрицательных, `v / 32767` для остальных) и кодирует функцией `encodeWav` из
Voice Note v0.4.2 без изменений, чанками по 4096 кадров; `scripts/probe_a_check_attach.py` строит
те же байты напрямую из целых. Совпадение байт проверяет
`tests/test_hermes_probe_a.py::test_probe_wav_is_byte_identical_to_the_backend_generator`.

SHA-256 эталона:

```text
4da599053f59cba6d689ff8320ca4441e2b79d96ad6a6aa1fd4a15d036ac2431
```

## Инструкция владельцу

Все шаги выполняет владелец. Агент ничего из этого не запускает: не ставит зонд, не пишет в чаты
и не трогает конфигурацию Hermes.

### 1. Подготовка (Windows)

1. Штатный Hermes Desktop, подключённый к удалённому gateway станции. Записать версию Desktop
   (About).
2. Создать отдельный тестовый чат, например «probe-a test». Зонд пишет только в чат, отмеченный
   командой `Probe A: mark this chat as test target`. Каждый из четырёх submit-тестов — одно
   сообщение и один ход агента в этом чате.

### 2. Установка (Windows)

1. Скопировать `hermes/probe/sound-scout-probe-a/plugin.js` из checkout в
   `%USERPROFILE%\.hermes\desktop-plugins\sound-scout-probe-a\plugin.js`; если задан
   `HERMES_HOME` — в `%HERMES_HOME%\desktop-plugins\sound-scout-probe-a\plugin.js`.
2. Capabilities → Plugins: включить «Sound Scout probe A», затем в палитре (Ctrl+K) — «Reload
   desktop plugins».
3. В палитре должно появиться восемь команд `Probe A: …`.
4. Если плагин не появился, записать, куда положен файл: это тоже результат о каталоге
   `desktop-plugins`.

### 3. Критерий 2: owner, маршруты и каталог

1. Открыть чат, который существовал до запуска Desktop → `Probe A: snapshot — old chat`.
2. Переименовать этот чат → `Probe A: snapshot — renamed chat`.
3. «Новый чат» (черновик) → `Probe A: snapshot — new chat`.
4. Отправить в новый чат первое сообщение вручную → ещё раз `Probe A: snapshot — new chat`.

Между снимками в журнале будут записи `focus_change`; в каждом снимке — `plugins_root`.

### 4. Критерий 3: attach и SHA-256

1. Открыть тестовый чат → `Probe A: mark this chat as test target` →
   `Probe A: attach synthetic 30 s WAV to target` (цель на экране).
2. Переключиться на другой чат → снова `Probe A: attach synthetic 30 s WAV to target` (цель скрыта:
   сценарий «записал в X, переключился на Y»).
3. Для каждой записи `attach` журнала выполнить на станции из checkout:

   ```bash
   uv run python -I scripts/probe_a_check_attach.py <path из журнала> --sha256 <sha256 из журнала>
   ```

   `~` в `path` раскрывает shell станции. Скрипт только читает файл и печатает одну JSON-строку
   (`ok`, `bytes`, `sha256`, `matches_log`, `matches_synthetic`, `wav`); путь и содержимое файла он
   не печатает. Сохранить обе строки.

### 5. Критерий 4: настоящий ли ACK

1. Тестовый чат на экране → `Probe A: prompt.submit to target`, дождаться уведомления с вердиктом
   (до ~40 с) → `Probe A: composer.submit to target`, дождаться уведомления.
2. Переключиться на другой чат (тестовый не висит плиткой) → `Probe A: prompt.submit to target` →
   `Probe A: composer.submit to target`.
3. Открыть тестовый чат и глазами сверить число сообщений с маркерами `sound-scout-probe-a …` с
   журналом.

«Скрытый чат» здесь — тестовый чат, которого нет на экране. Флаг скрытой сессии зонд не трогает.

### 6. Передача

`Probe A: copy log` → вставить журнал в комментарий к
[#95](https://github.com/renidere/sound-scout-plugin/issues/95) вместе с JSON-строками скрипта,
версией Desktop и наблюдениями из шага 5.3. Репозиторий публичный: перед отправкой глазами
проверить, что в тексте нет имени пользователя и названий чатов.

### 7. Удаление

- Windows: удалить папку `desktop-plugins\sound-scout-probe-a` и выполнить «Reload desktop
  plugins»; команд `Probe A: …` в палитре быть не должно.
- Станция: удалить файлы `~/.hermes/attachments/sound-scout-probe-a-*.wav` по `path` из журнала.
- Тестовый чат — по желанию.

## Результаты

Версия Desktop и коммит checkout: ожидает прогона владельцем.

| пункт | что наблюдать | результат |
|---|---|---|
| установка, каталог `desktop-plugins` | восемь команд в палитре, `plugins_root` в снимке | ожидает прогона владельцем |
| 2а. старый чат | `stored`, `owner`, `routes`, `owner_matches` | ожидает прогона владельцем |
| 2б. переименованный чат | тот же `stored`, `owner` не `null`, `owner_matches = 1` | ожидает прогона владельцем |
| 2в. новый чат до и после первого сообщения | `stored`, `owner`, `owner_matches` | ожидает прогона владельцем |
| 3а. attach, цель на экране | запись `attach`, `matches_log && matches_synthetic` | ожидает прогона владельцем |
| 3б. attach, цель скрыта | запись `attach`, `matches_log && matches_synthetic` | ожидает прогона владельцем |
| 4а. `prompt.submit`, цель на экране | вердикт, `replay_same_admission`, `occurrences` | ожидает прогона владельцем |
| 4б. `prompt.submit`, цель скрыта | вердикт, `replay_same_admission`, `occurrences` | ожидает прогона владельцем |
| 4в. `composer.submit`, цель на экране | вердикт, `ack`, `occurrences` | ожидает прогона владельцем |
| 4г. `composer.submit`, цель скрыта | вердикт, `ack`, `occurrences` | ожидает прогона владельцем |
| 5. выбор пути для #103 | по правилу анализа ниже | ожидает прогона владельцем |

## Правило анализа

Таблицу заполняет агент по журналу и строкам скрипта из комментария владельца в #95, затем меняет
статус этого документа.

- **ACK настоящий**, если вердикт `real` и для цели на экране, и для скрытой.
- **Путь для #103:** `prompt.submit` с `submission_id`, если он `real` в обоих случаях. Иначе
  `composer.submit` с проверкой чтением, если он `real` хотя бы для цели на экране; тогда при
  скрытом чате recorder обязан отказывать. Если не подходит ни один — решает владелец.
- **Owner:**
  - `null` у старого или переименованного чата → в 0013 § 3.3 действует правило перечитывания при
    отправке;
  - смена `stored` при переименовании → id не стабилен, нужен пересмотр 0013;
  - `owner_matches ≠ 1` → `ambiguous_route` в рекордере.
- **Attach:** `matches_log && matches_synthetic` для обеих записей `attach`. Иначе путь
  `file.attach` для рекордера не подтверждён.
- `loaded.sdk.retainSession` на штатном клиенте ожидается `undefined`; другое значение значит, что
  клиент не штатный, и прогон не отвечает на вопрос 0013.
