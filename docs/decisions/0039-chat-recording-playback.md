# 0039 — Отправленная запись играет в чате

- **Статус:** принято, реализовано в коде и тестах. Плеер в чате проверил владелец на Windows
  2026-11-28 с плагином 0.4.0
  ([комментарий в #216](https://github.com/renidere/sound-scout-plugin/issues/216#issuecomment-7242200161)).
  Плеер плагина удалён в 0.5.0 (#217, «Правка 2026-11-28 (#217)»). Шаги владельца O10 и O11
  (`docs/acceptance/A01-A14.md`) проверяют плагин 0.5.0 и только плеер в чате.
- **Дата:** 2026-11-28.
- **Решение владельца:** 2026-11-28, место записи — чат, а не плеер плагина; спецификация —
  [Decision в #216](https://github.com/renidere/sound-scout-plugin/issues/216#issuecomment-7239970887).
- **Задача:** [#216](https://github.com/renidere/sound-scout-plugin/issues/216), удаление плеера
  плагина — [#217](https://github.com/renidere/sound-scout-plugin/issues/217); связано —
  [#214](https://github.com/renidere/sound-scout-plugin/issues/214) (плеер плагина,
  [Decision](https://github.com/renidere/sound-scout-plugin/issues/214#issuecomment-7238679289)),
  #108 (простой текст заметки), #188 (остановка хода). Доставка — 0013 § 3.7; привязка чата — 0025;
  отмена хода — 0029; заметка попытки — 0030.
- **Hermes:** Hermes Agent и Desktop v0.23.3, ревизия `caafe5389ae585ef77387a962aee05b1bcf17029`
  (`caafe5389a`, https://github.com/NousResearch/hermes-agent). Код прочитан только чтением, файлы
  из `HEAD` checkout станции. Пути ниже — от корня репозитория Hermes. Desktop на Windows считается
  0.23.3 по #120; версию на стенде S3 владелец ещё не записал (O9).

## Почему

Владелец слушает каждую отправленную запись у её сообщения в чате. Некоторые упражнения
возвращаются к ранней записи через несколько других. #214 добавил «Прослушать» в рекордере и на
карточке. На Windows 2026-11-28 плеер 0.3.0 не дал звука, кнопка пуска сбрасывалась. Владелец
сказал: место записи — чат.

## Какие пути чата могут играть WAV

| путь | что рисует Desktop | играет WAV |
|---|---|---|
| (а) вложение `file.attach` в сообщении пользователя | неактивный чип `@file:` | нет |
| (б) файл `send_document`/`send_voice` | ничего: у `Local` нет отправки | нет |
| заметка доставки | пузырь пользователя без медиа | нет |
| (в) области Desktop Plugin SDK | нет области сообщения пользователя | нет |
| строка `MEDIA:<путь>.wav` в ответе ассистента | встроенный `<audio controls>` | да |

1. **(а) `file.attach`.** `gateway/session_ancillary.py:180` отдаёт `ref_text = '@file:<ref_path>'`.
   `apps/desktop/src/lib/chat-messages/hydration.ts` уносит ведущий блок `@file:` в
   `attachmentRefs`. `thread/user-message.tsx` рисует их через `DirectiveContent`. В
   `directive-text.tsx` у вида `file` нет записи в `DIRECTIVE_ACTIONS`, поэтому `DirectiveChip` —
   неактивный `<span>`. Текст сообщения пользователя проходит `UserMessageText`: только код и чипы,
   без Markdown и медиа.
2. **(б) `send_document`.** `LocalSessionAdapter` (`gateway/session_local.py`) не переопределяет
   `send_document` и `send_voice`. Базовый метод (`gateway/platforms/base.py:3065`,
   `_send_media_fallback_notice`) пишет в лог `send_document fallback: native file send
   unavailable` и шлёт предупреждение через `send`. `LocalSessionAdapter.send` ничего не делает.
   Карточку `report.md` с «Скачать» и «Открыть предпросмотр» рисует сам Desktop по строке `MEDIA:` в
   ответе агента (п. 5).
3. **Заметка доставки.** `deliver_to_local_chat` → `session_completion_notes.note_if_unviewed` →
   `append_completion_note` пишет строку с ролью `user` и `display_kind = internal_notification`
   (`persistence/sqlite/conversation/_store/detached_results.py`). Этого вида нет в
   `NOTICE_DISPLAY_KINDS` (`hydration.ts`): заметка — пузырь пользователя без медиа. Решение #108
   (простой текст) остаётся верным.
4. **(в) SDK.** `apps/desktop/src/sdk/index.ts` экспортирует `COMPOSER_AREAS`, `PALETTE_AREA`,
   `ROUTES_AREA`, `SIDEBAR_*`, `SESSION_ROW_AREAS`, `CHAT_EMPTY_AREA`, `TRANSCRIPT_DIRECTIVE_AREA` и
   другие. Области сообщения пользователя нет. `TRANSCRIPT_DIRECTIVE_AREA`
   (`lib/transcript-directives.ts`) рисует `::name{…}` только в Markdown ассистента:
   `remarkTranscriptDirectives` подключён только в `components/assistant-ui/markdown-text.tsx`.
   Фраза 0013 § 3.7 «директивы нет» поэтому неточна (правка 2026-11-28 в 0013). Область
   `TOOL_MEDIA_RENDERER_AREA` (`lib/tool-media-renderers.ts`) SDK не экспортирует, и она заменяет
   плеер только для файла инструмента самого плагина; запись создаёт `file.attach`.
5. **Ответ ассистента.** `renderMediaTags` (`lib/chat-messages/parts.ts`) превращает строку
   `MEDIA:<путь>` ответа в ссылку `#media:`. `MarkdownLink` (`markdown-text.tsx`) рисует аудио как
   `MediaPlaybackAttachment`: `<audio controls preload="metadata">` с именем файла. WAV —
   `audio/wav` (`lib/media.ts`, `MEDIA_BY_EXT`). У удалённого gateway источник звука —
   `hermes-media://remote/<путь>` (`resolveMediaPlaybackSrc`, `mediaGatewayStreamUrl`); Electron
   проксирует его на `/api/files/stream` с Range (`electron/media-protocol.ts`,
   `hermes_cli/web_routers/files.py::stream_managed_file`), `.wav` есть в
   `_STREAMABLE_MEDIA_EXTENSIONS`. ⋮ → «Скачать» встроенного плеера обрабатывает `will-download`
   (`electron/main.ts:6712`), имя файла задаёт `mediaProtocolDownloadName`.

Итог: в чате Desktop 0.23.3 WAV играет только из сообщения ассистента.

## Выбор

Agent-половина плагина получает два хука (`hermes/plugin/sound-scout/recording_media.py`).

- `pre_llm_call` (`agent/turn_context.py` `_collect_pre_llm_call_context`) получает `turn_id` и
  `user_message` — сохраняемый текст хода с ведущим блоком `@file:`. Хук ищет ссылки
  `@file:…attachments/sound-scout-<12 hex>.wav` и запоминает путь
  `<get_hermes_home()>/attachments/<имя>`. Путь — обычный файл от 1 байта, не symlink. Каталог из
  ссылки не используется, только имя по шаблону. Контекст в ход хук не добавляет.
- `transform_llm_output` (`agent/turn_finalizer.py` `apply_llm_output_transform`) срабатывает один
  раз за ход до первой записи строки ассистента. Хук добавляет в конец ответа этого хода строку
  `MEDIA:<путь>`. В истории лежит ответ с этой строкой, поэтому плеер стоит под сообщением записи,
  остаётся после следующих сообщений и после перезапуска Desktop. Итог хода приходит в
  `message.complete` с `response_transformed`, Desktop заменяет им текст потока
  (`use-message-stream/index.ts:1009–1034`). Хук без `hold_stream`: поток не задерживается.

Модель не участвует. Ход без ссылки на запись, другой файл, ответ с отдельной строкой
`MEDIA:<путь>`, пустой ответ — без изменений. Путь в тексте ответа строку не заменяет: хук её
добавит. Ошибка хука оставляет ответ как есть. Лог называет только исход.

Два уточнения к спецификации Decision:

- Ключ ожидающей записи — только `turn_id`, без `session_id`. Сжатие контекста внутри хода меняет
  `agent.session_id` между хуками (`agent/conversation_compression.py`), а `turn_id` остаётся. В
  `turn_id` есть uuid, поэтому ходы разных сессий не совпадают.
- Хук запоминает запись только в ходе локального чата Desktop (`HERMES_SESSION_PLATFORM` =
  `local`, как у `binding.py`). В мессенджере gateway отправил бы файл по строке `MEDIA:` в чат
  мессенджера: это передача записи наружу (R09).

В #216 плагин ещё держал свой плеер «Прослушать» в рекордере и на карточке (#214, правка #216).
На Windows он звука не дал. Плеер удалён в 0.5.0 (#217): запись слушается только в чате.

## Отклонённые варианты

1. Вложение пользователя (а): неактивный чип, хука SDK нет.
2. `send_document`/`send_voice` (б) из плагина: у `Local` ничего не делает.
3. Строка `MEDIA:` в заметке доставки: пузырь пользователя без медиа; заметка есть только у
   «Отправить и разобрать».
4. Инструкция skill `speech-review`, чтобы модель писала `MEDIA:` сама: не детерминировано.
5. Директива `::sound-scout-recording{name}` и свой плеер в ответе: нужен тот же хук ответа, а
   свой плеер уже сбоил.

## Ограничения

- Ход без текста ответа (остановка «Отменить разбор» по #188, сбой хода) строки `MEDIA:` не
  получает. У такой записи плеера нет нигде: плеера плагина больше нет. Владелец принял это
  2026-11-28 (#217).
- Запись должна лежать в `<hermes home>/attachments/` (`attachments.storage` по умолчанию). При
  `attachments.storage: workspace` строки нет.
- Путь Hermes home с пробелом или кавычкой строку не получает: такой путь сбил бы правило
  `MEDIA_LINE_RE` Desktop.
- В логе gateway остаётся строка `send_voice fallback … for <путь>`: это поведение Hermes для
  `MEDIA:` у `Local`, как у `report.md`.
- Модель видит строку `MEDIA:` в истории следующих ходов.
- Записи, отправленные до установки 0.4.0, плеера в чате не получат: прежние ответы не меняются.
- Плагин не читает `attachments/`: файл играет и скачивается только через встроенный плеер Desktop.

## Проверка

- `tests/test_hermes_recording_media.py`: хуки с ключевыми словами вызова Hermes v0.23.3 в
  контексте хода на рабочем потоке; строка `MEDIA:<home>/attachments/<имя>` у хода записи и её
  отсутствие у других ходов, файлов и пустого ответа; без дубля; ограничение `PENDING_MAX`; ошибки
  хука; регистрация без `hold_stream`; лог без путей и текста.
- Живая проверка владельца на Windows 2026-11-28 (плагин 0.4.0, Desktop): под ответом на запись
  стоит встроенный плеер, звук слышен, плеер остаётся после следующих сообщений и перезапуска
  Desktop, ⋮ → «Скачать» сохраняет файл
  ([комментарий в #216](https://github.com/renidere/sound-scout-plugin/issues/216#issuecomment-7242200161)).
  Повтор с 0.5.0 — шаг O11.

## Правка 2026-11-28 (#217)

Решение владельца 2026-11-28
([проверка в #216](https://github.com/renidere/sound-scout-plugin/issues/216#issuecomment-7242200161),
[#217](https://github.com/renidere/sound-scout-plugin/issues/217)): плеер плагина удаляется, запись
слушается только в чате. Причина: плеер в чате на Windows играет, плеер плагина звука не даёт.

- `desktop/plugin.js`: нет кнопки «Прослушать», диалога плеера, `new Audio()`, «Скачать» и имени
  записи в состоянии рекордера и задания.
- `dashboard/plugin_api.py`: нет маршрута `GET /recordings/{name}`; `attachments/` плагин не читает
  (0013 § 9 п. 2 снова без исключения).
- Хуки `recording_media.py` не меняются.
- Версия пакета — 0.5.0.

## Не доказано до O11

- Версия Desktop на стенде равна 0.23.3 (O9).
- Встроенный плеер играет и по LAN, и по Tailscale; перемотка по Range работает.
- Строка `MEDIA:` появляется сразу по `message.complete`, а не только после перезагрузки истории.
- Desktop с `plugin.js` 0.5.0 не показывает «Прослушать», плеер в чате работает как с 0.4.0.

## Развёртывание

- Нужен владелец: копия `hermes/plugin/sound-scout/` 0.5.0 в `~/.hermes/plugins/sound-scout/` и
  перезапуск `hermes-gateway.service` по 0017 § 4. Без перезапуска хуки и `VERSION` не загружаются.
- Нужен владелец: копия `desktop/plugin.js` на шару, с шары — в
  `%LOCALAPPDATA%\hermes\desktop-plugins\sound-scout\`, перезапуск Desktop.
- Runner `sound-scout-jobs-runner.service` перезапускать не нужно: пакет `sound_scout` не меняется.
  Skill `speech-review` не меняется.
