# 0016 — Spike A: зонд штатного Desktop (owner сессии, file.attach, настоящий ли ACK)

- **Статус:** результаты прогона 2026-11-21; анализ журнала записан (шаг 7). Открыты случай
  перенесённого старого чата (2а) и `prompt.submit` при скрытой цели (4б): вердикт зонда и повтор
  `submission_id` не записаны.
- **Дата:** 2026-11-20; прогон владельца и анализ агента — 2026-11-21.
- **Решение:** [Decision в #95](https://github.com/renidere/sound-scout-plugin/issues/95#issuecomment-7024314864).
- **Пересмотр под этап B 0017:** [Decision в #95](https://github.com/renidere/sound-scout-plugin/issues/95#issuecomment-7064941502)
  (2026-11-21): код зонда не меняется, инструкция владельцу переписана под топологию gateway.
- **Задача:** [#95](https://github.com/renidere/sound-scout-plugin/issues/95); эпик
  [#115](https://github.com/renidere/sound-scout-plugin/issues/115) (этап Windows-клиента по
  порядку этапов из 0013 § 5, решение владельца 20 ноября 2026).
- **Основание:** `docs/decisions/0013-voice-note-integration.md` § 3.3 (рекордер), § 3.4
  (отправка), § 5 шаг 2 (Spike A); `docs/decisions/0015-spike-b-backend-package.md` (Spike B,
  отдельный пакет `sound-scout`); `docs/decisions/0017-session-authority-owner.md` (этап B:
  владелец чатов Desktop — gateway); recorder Voice Note v0.4.2 (`ced69ea`, MIT).
- **Hermes:** Hermes Agent v0.23.3, ревизия `caafe5389a`. Зонд написан по `3b42157858` (v0.23.1).
  Вызовы SDK и параметры RPC зонда сверены с v0.23.3: изменений нет. `host.retainSession` в
  закоммиченном SDK по-прежнему нет.
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

## Топология прогона

- Windows Desktop 0.23.3 входит напрямую в `hermes gateway run` станции, Webapp на `:9119`
  (этап B 0017). Desktop SSH backend нет, `hermes-autonomous.service` остановлен.
- `file.attach` пишет файл в `attachments/` профиля gateway на станции. Там его читает
  `scripts/probe_a_check_attach.py`.
- Каталог `desktop-plugins` и журнал зонда — локальные каталоги Windows, не станции.
- Обмен файлами с владельцем — сетевая шара `H:\SoundScout\` (на станции
  `/mnt/HybridHeap/SoundScout/`). Журнал возвращается на шару, а не в публичный issue.

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
`ctx.files.writeText('data', 'probe-a-log.jsonl', …)` (папка данных плагина; по коду v0.23.3 —
`%APPDATA%\Hermes\plugin-files\data\sound-scout-probe-a\probe-a-log.jsonl`, на Windows не
проверено); ошибка записи
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

Все шаги выполняет владелец, кроме шага 7 (агент). Агент не ставит зонд, не пишет в чаты и не
трогает конфигурацию Hermes.

### 1. Подготовка (Windows)

1. Записать версию Desktop (About).
2. Тестовый чат создаётся в шаге 3.2 как новый черновик. Только в него зонд пишет. Каждый из
   четырёх submit-тестов — одно сообщение и один ход агента в этом чате.

### 2. Установка (Windows)

1. Capabilities → Plugins → «Open Desktop plugins folder». В открытой папке уже лежит
   `sound-scout`. По коду v0.23.3 это `<HERMES_HOME>\desktop-plugins`. `HERMES_HOME` берётся из
   окружения или реестра; без него — `%LOCALAPPDATA%\hermes`, а `%USERPROFILE%\.hermes` — только
   если `%LOCALAPPDATA%\hermes` нет, а `%USERPROFILE%\.hermes` есть.
2. Скопировать туда папку `H:\SoundScout\desktop-plugins\sound-scout-probe-a\` (внутри один
   `plugin.js`). Это `hermes/probe/sound-scout-probe-a/plugin.js` с `main`.
3. Сверка в PowerShell: `Get-FileHash <папка>\sound-scout-probe-a\plugin.js` должен дать SHA-256
   зонда:

   ```text
   b054e3b9d139d718ebc3ea28da3ac5ae3a817b124b4b7de044a89c11c03ab74f
   ```

4. Палитра (Ctrl+K) → `Reload desktop plugins`. Если «Sound Scout probe A» выключен в
   Capabilities → Plugins, включить и повторить Reload.
5. В палитре должно появиться восемь команд `Probe A: …`. Если их нет, записать это в
   `notes.txt` (шаг 6): это тоже результат о каталоге `desktop-plugins`.

### 3. Критерий 2: owner, маршруты и каталог

Снимки ничего не пишут.

1. Открыть один из трёх старых чатов, перенесённых на этапе B → `Probe A: snapshot — old chat`.
2. Создать новый чат: появится черновик без названия. Это тестовый чат. У черновика Desktop 0.23.3
   названия нет; оно появится после первого сообщения или в шаге 3.4. Черновик на экране →
   `Probe A: snapshot — new chat`.
3. Отправить в черновик вручную первое сообщение (например «привет»), дождаться ответа →
   `Probe A: snapshot — new chat`.
4. Переименовать тестовый чат (например в «probe-a test») → `Probe A: snapshot — renamed chat`.

Открытие старого чата в Desktop само вызывает `session.resume`; зонд в снимке RPC не делает. Между
снимками в журнале будут записи `focus_change`; в каждом снимке — `plugins_root`.

Старые чаты, не перенесённые на этапе B, не открывать. В v0.23.3 `session.resume` такого чата от
Desktop переносит его под gateway (`gateway/session_local_migration.py::adopt_legacy_session`, тот
же путь, что у `hermes sessions adopt`). Это изменение живой системы, а зонд его вызывать не должен.
После открытия чат уже перенесён, и снимок не показал бы неперенесённый случай. Поэтому этот случай
Spike A не измеряет.

### 4. Критерий 3: attach

1. Тестовый чат на экране → `Probe A: mark this chat as test target` →
   `Probe A: attach synthetic 30 s WAV to target`.
2. Открыть один из трёх перенесённых чатов (тестовый не висит плиткой) →
   `Probe A: attach synthetic 30 s WAV to target`
   (цель скрыта: сценарий «записал в X, переключился на Y»).
3. Сообщений в чат attach не пишет. Сверку SHA-256 делает агент (шаг 7).

### 5. Критерий 4: настоящий ли ACK

1. Тестовый чат на экране → `Probe A: prompt.submit to target`, дождаться уведомления с вердиктом
   (до ~40 с) → `Probe A: composer.submit to target`, дождаться уведомления.
2. Открыть один из трёх перенесённых чатов (тестовый не висит плиткой) →
   `Probe A: prompt.submit to target` → `Probe A: composer.submit to target`.
3. Открыть тестовый чат и сосчитать сообщения с маркерами `sound-scout-probe-a …`.

«Скрытый чат» здесь — тестовый чат, которого нет на экране. Флаг скрытой сессии зонд не трогает.

### 6. Возврат на шару

1. Скопировать `%APPDATA%\Hermes\plugin-files\data\sound-scout-probe-a\probe-a-log.jsonl` в
   `H:\SoundScout\probe-a\probe-a-log.jsonl`.
2. Всегда, даже если файл был: `Probe A: copy log` → Блокнот → вставить → «Сохранить как»
   `H:\SoundScout\probe-a\probe-a-log.clipboard.jsonl` (UTF-8). Файл мог отстать или оборваться;
   буфер — второй источник.
3. Рядом положить `H:\SoundScout\probe-a\notes.txt`: версия Desktop; число маркеров из шага 5.3;
   был ли файл журнала в шаге 6.1 (это результат о папке данных) и его путь.
4. Написать в [#95](https://github.com/renidere/sound-scout-plugin/issues/95) одну строку: «журнал
   на шаре». Журнал в issue не вставлять: репозиторий публичный.

### 7. Агент

1. Прочитать `/mnt/HybridHeap/SoundScout/probe-a/`. Журналом считать ту из копий
   `probe-a-log.jsonl` и `probe-a-log.clipboard.jsonl`, в которой больше записей; при равенстве —
   `probe-a-log.jsonl`. Расхождение копий записать в «Результаты».
2. Для каждой записи `attach` выполнить на станции из checkout (`~` в `path` раскрыть):

   ```bash
   uv run python -I scripts/probe_a_check_attach.py <path из журнала> --sha256 <sha256 из журнала>
   ```

   Скрипт только читает файл и печатает одну JSON-строку (`ok`, `bytes`, `sha256`, `matches_log`,
   `matches_synthetic`, `wav`); путь и содержимое файла он не печатает.
3. По `stored` из снимков сверить через `state.db` (`mode=ro`), был ли чат перенесён и есть ли у
   тестового чата строки `session_admissions` после каждого submit.
4. Заполнить «Результаты», сменить статус документа, отчитаться в #95 без содержимого журнала сверх
   полей вердиктов.

### 8. Удаление

- Windows: удалить папку `sound-scout-probe-a` и выполнить `Reload desktop plugins`; команд
  `Probe A: …` в палитре быть не должно.
- Шара: агент удаляет `desktop-plugins/sound-scout-probe-a/` и `probe-a/` после заполнения этого
  документа. Удалено 2026-11-22 ([#153](https://github.com/renidere/sound-scout-plugin/issues/153)):
  новый прогон зонда не запланирован. Перед удалением обе папки упакованы в
  `archive/153-spike-a-share/spike-a-share.tar.gz` в data root станции, вне git. SHA-256 архива —
  `d9196cdadfab5c8e3fd34f8edc5e9cf750fb7e61777a20fb03525b26fde2fbef`, SHA-256 трёх файлов —
  `files.sha256` рядом. Журнал `probe-a-log.clipboard.jsonl` и `notes.txt` — источники
  «Результатов» — есть только в этом архиве. Для нового прогона `plugin.js` берётся из git, а
  пустая `probe-a/` создаётся заново.
- Станция: файлы `attachments/sound-scout-probe-a-*.wav` лежат в профиле Hermes. Агент удаляет их
  только по явному «удаляй» владельца в #95 и по точным именам из журнала. Без этого их удаляет
  владелец.
- Тестовый чат — по желанию.

## Результаты

**Источники.** Все значения ниже взяты из этих источников. Тексты сообщений, названия чатов и
сырые `connectionId` сюда не попадают; id сессий сокращены.

- Desktop: Windows Desktop 0.23.3 (`notes.txt`, About). Зонд: `plugin.js` из `37d5470`, до
  `78f1a7e` он не менялся. SHA-256 копии на шаре совпадает с `b054e3b9…` (агент сверил на
  станции). Результат `Get-FileHash` на Windows в `notes.txt` не записан.
- Журнал: одна копия, `probe-a-log.clipboard.jsonl`, 27 записей (seq 1–27, 16:45:46–16:58:29
  UTC). Файл `probe-a-log.jsonl` с Windows не копировался (`notes.txt`). Сравнить копии нельзя.
  Запись в папку данных плагина на Windows не подтверждена. Последний submit (`prompt.submit`,
  цель скрыта) закончился после снятия копии, его записи `submit` в журнале нет.
- `scripts/probe_a_check_attach.py` на станции для обоих файлов `attach` (квитанция оркестратора).
- `state.db` Hermes, открыт `mode=ro` (данные оркестратора): маркеры зонда и `session_admissions`
  тестового чата.
- Чаты в журнале: тестовый чат `local-d483e1ff…` и второй чат `local-e1827bbe…`. Других
  `stored`, кроме `null`, в журнале нет. Владелец сначала отметил целью второй чат (seq 11), затем
  тестовый (seq 13). Цель всех пишущих команд — `local-d483e1ff…`. «Цель скрыта» везде значит:
  на экране второй чат.
- Маршруты: во всех четырёх снимках два маршрута профиля `default`. `conn-2` — `mode: local`,
  `primary: null`. `conn-1` — `mode: remote`, `primary: true`. Owner во всех записях —
  `conn-1`/`default`. `owner_fallback` во всех пишущих записях — `null`.

| пункт | что наблюдать | результат |
|---|---|---|
| установка, каталог `desktop-plugins` | восемь команд в палитре, `plugins_root` в снимке | Зонд загрузился (seq 1 `loaded`). Владелец выполнил все восемь команд (seq 4–27, `copy log` — по `notes.txt`). `plugins_root` во всех снимках — `%USERPROFILE%\AppData\Local\hermes\desktop-plugins`: это ветка `%LOCALAPPDATA%\hermes` без `HERMES_HOME`. `loaded.sdk.retainSession = undefined`: клиент штатный. Что `sound-scout` лежит в той же папке, `notes.txt` отдельно не подтверждает. |
| 2а. старый перенесённый чат | `stored`, `owner`, `routes`, `owner_matches` | Измерено частично. Снимок `old` (seq 6) сделан в тестовом чате `local-d483e1ff…`. Этот чат создан 2026-11-21 на gateway и на этапе B не переносился (`notes.txt`, данные оркестратора). В нём `stored = runtime`, owner `conn-1`/`default`, два маршрута, `owner_matches = 1`. Ни один из трёх перенесённых чатов в журнале не встречается: случай перенесённого чата не измерен. |
| 2б. переименованный чат | тот же `stored`, `owner` не `null`, `owner_matches = 1` | `stored` до и после переименования один: `local-e1827bbe…` (seq 9 и seq 10). `runtime = stored`, owner `conn-1`/`default`, `owner_matches = 1`. Id при переименовании не меняется. Само переименование журнал не показывает: названия чатов не пишутся. |
| 2в. новый чат до и после первого сообщения | `stored`, `owner`, `owner_matches` | Черновик (seq 4): `stored = null`, `runtime = null`, owner уже `conn-1`/`default`, `owner_matches = 1`. После первого сообщения (seq 8, seq 9): `stored = runtime = local-e1827bbe…`, owner тот же, `owner_matches = 1`. |
| 3а. attach, цель на экране | запись `attach`, `matches_log && matches_synthetic` | seq 19: `visible: true`, `attached` и `uploaded` — `true`, 5 760 044 байт, SHA-256 равен эталону. Шаги: `retainProfile` 0 мс, `session.resume` 145 мс, кодирование 42 мс, `file.attach` 263 мс. Скрипт на станции: `ok`, `matches_log`, `matches_synthetic`, WAV 30 с, 48 кГц, стерео, PCM16. |
| 3б. attach, цель скрыта | запись `attach`, `matches_log && matches_synthetic` | seq 17: `visible: false`, `attached` и `uploaded` — `true`, 5 760 044 байт, SHA-256 равен эталону. Шаги: `retainProfile` 0 мс, `session.resume` 157 мс, кодирование 45 мс, `file.attach` 244 мс. Скрипт на станции: `ok`, `matches_log`, `matches_synthetic`, WAV 30 с, 48 кГц, стерео, PCM16. |
| 4а. `prompt.submit`, цель на экране | вердикт, `replay_same_admission`, `occurrences` | seq 22: вердикт `real`. ACK несёт `admission_id`, статус `queued`. Повтор того же `submission_id` вернул ту же admission (`replay_same_admission: true`). `occurrences = 1` через 2405 мс; чтений 15, все с `messages`, ошибок 0. В `state.db` маркер `d5e7ea2a99a4` есть один раз. |
| 4б. `prompt.submit`, цель скрыта | вердикт, `replay_same_admission`, `occurrences` | Измерено частично. Записи `submit` в копии журнала нет: копию сняли раньше. `state.db`: маркер `0c715d42d1a5` есть в тестовом чате один раз (16:59:51 UTC). История подтверждает одну доставку. ACK на стороне клиента (`admission_id`, нет ошибки RPC), вердикт зонда и `replay_same_admission` не записаны. Две более ранние попытки (seq 21, seq 25) получили отказ `busy`: предыдущая команда ещё шла в окне проверки. |
| 4в. `composer.submit`, цель на экране | вердикт, `ack`, `occurrences` | seq 26: вердикт `real`, `ack: true`, `occurrences = 1` через 2210 мс; чтений 15, все с `messages`, ошибок 0. В `state.db` маркер `526e9a078d3e` есть один раз. |
| 4г. `composer.submit`, цель скрыта | вердикт, `ack`, `occurrences` | seq 27: вердикт `refused`, `ack: false`, `occurrences = 0`. Чтений 15, все с `messages`, ошибок 0; `message_count` 13 → 13. Ничего не записано: отказ fail-closed. Маркера `4a83b1f54e2a` среди маркеров тестового чата в `state.db` нет. |
| 5. выбор пути для #103 | по правилу анализа ниже | Отправка — `prompt.submit` с `submission_id`: `real` на экране (seq 22), при скрытой цели сообщение дошло один раз (`state.db`), ACK клиента не записан: для скрытой цели выбор предварительный. `composer.submit` работает только для чата на экране. Attach — `retainProfile` → `session.resume` → `file.attach`. Подробно — «Выводы для #103». |

`session_admissions` тестового чата: по чтению оркестратора (`state.db`, `mode=ro`) admissions
117–119 завершены. Какая строка какому маркеру соответствует, здесь не установлено. О повторе
`submission_id` при скрытой цели число строк ничего не говорит: повтор не измерен.

## Выводы для #103

Правило анализа требует вердикт `real` у `prompt.submit` и для цели на экране, и для скрытой. Для
скрытой цели записи зонда нет. История `state.db` подтверждает одну доставку: маркер ровно один.
ACK на стороне клиента (`admission_id`, нет ошибки RPC) и повтор `submission_id` не наблюдались.
Поэтому выбор `prompt.submit` для скрытой цели предварительный.

- **Отправка:** рекордер отправляет через `prompt.submit` с `submission_id` (через
  `host.requestProfile`). Для чата на экране вердикт `real` (seq 22). Для скрытого чата сообщение
  дошло один раз (история), ACK клиента не записан: выбор для этого случая предварительный. Другого
  пути для скрытого чата нет: `composer.submit` там fail-closed.
- **`composer.submit` — не путь доставки.** Он работает только для чата на экране. Для скрытого
  чата ACK `false`, и ничего не записано (fail-closed).
- **Attach:** `retainProfile` → `session.resume` (`omit_messages: true`) → `file.attach` работает
  для цели на экране и для скрытой. Байты на станции совпали с эталоном (SHA-256). Каждый шаг
  короче 0,3 с.
- **Маршрут:** у этого Desktop два маршрута профиля `default`, локальный и удалённый. Выбор
  только по `profile` дал бы два совпадения. Рекордер выбирает маршрут по `connectionId` и
  `profile`, как зонд: тогда `owner_matches = 1`.
- **Owner:** owner не `null` ни в одной записи, и у черновика тоже. Перечитывание owner
  (`owner_fallback`) в прогоне не понадобилось. Правило перечитывания из 0013 § 3.3 остаётся:
  перенесённый старый чат не измерен.
- **Id чата:** при переименовании `stored` не меняется. У черновика до первого сообщения
  `stored = null`: такой чат целью записи быть не может.
- **Каталог:** на этом клиенте `desktop-plugins` — `%LOCALAPPDATA%\hermes\desktop-plugins`.

Открыто:

- 2а для перенесённого старого чата не измерен. Чат, не перенесённый на этапе B, Spike A не
  измеряет по построению (шаг 3).
- 4б: ACK для скрытой цели не записан. Записи `submit` (`admission_id`, ошибка RPC, вердикт) и
  повтора `submission_id` при скрытой цели в копии журнала нет. Закрывает это повтор 4б владельцем
  с записью `submit` в журнал.
- Запись журнала в папку данных плагина на Windows не проверена: файл не копировался.

**Файлы на станции.** Два синтетических WAV остаются в `attachments/` профиля gateway:
`sound-scout-probe-a-52602a7ecd1a.wav` (цель скрыта) и `sound-scout-probe-a-19dcd34a54b2.wav`
(цель на экране). Владелец не сказал «удаляй». Агент удалит их только после этого слова в #95 и
только по этим именам.

## Правило анализа

Таблицу заполняет агент по журналу и `notes.txt` с шары и строкам скрипта (шаг 7), затем меняет
статус этого документа.

- **ACK настоящий**, если вердикт `real` и для цели на экране, и для скрытой.
- **Путь для #103:** `prompt.submit` с `submission_id`, если он `real` в обоих случаях. Иначе
  `composer.submit` с проверкой чтением, если он `real` хотя бы для цели на экране; тогда при
  скрытом чате recorder обязан отказывать. Если не подходит ни один — решает владелец.
- **Owner:**
  - `null` у старого или переименованного чата → в 0013 § 3.3 действует правило перечитывания при
    отправке;
  - смена `stored` при переименовании → id не стабилен, нужен пересмотр 0013;
  - `owner_matches ≠ 1` → `ambiguous_route` в рекордере;
  - неперенесённый старый чат не измеряется (шаг 3): открытие переносит его под gateway.
- **Каталог:** `plugins_root` должен совпасть с папкой, куда владелец положил `sound-scout`.
- **`session_admissions`:** вердикт `real` у `prompt.submit` должен иметь строку в
  `session_admissions` тестового чата.
- **Attach:** `matches_log && matches_synthetic` для обеих записей `attach`. Иначе путь
  `file.attach` для рекордера не подтверждён.
- `loaded.sdk.retainSession` на штатном клиенте ожидается `undefined`; другое значение значит, что
  клиент не штатный, и прогон не отвечает на вопрос 0013.
