# 0030 — Нет заметки, если ход чата уже прочитал итог попытки

- **Статус:** принято, реализовано в коде и тестах. Живая проверка — шаг владельца O10
  (`docs/acceptance/A01-A14.md`) после установки плагина `sound-scout` 0.2.0 и skill
  `speech-review` 0.5.6.
- **Дата:** 2026-11-25.
- **Решение владельца:** 2026-11-25, заметка попытки уходит в чат, только если итог в чате никто
  не видел; спецификация — [Decision в #190](https://github.com/renidere/sound-scout-plugin/issues/190#issuecomment-7158572434).
- **Задача:** [#190](https://github.com/renidere/sound-scout-plugin/issues/190); связано —
  [#187](https://github.com/renidere/sound-scout-plugin/issues/187) (заметка на каждую попытку),
  эпик [#92](https://github.com/renidere/sound-scout-plugin/issues/92). Привязка чата — 0025;
  доставка — 0013 § 3.7; повтор — 0018 «Повтор и доставка».
- **Hermes:** Hermes Agent v0.23.3, ревизия `caafe5389a`. Код прочитан только чтением. Пути ниже —
  от корня установки Hermes, строки — по этой ревизии.

## Почему

Цикл доставки шлёт заметку на каждую закончившуюся попытку (#187). В чатах владельца агент
читает итог в том же ходе (`speech_job_status`, `speech_job_result`) и отвечает им. Заметка
повторяет ответ. Кроме того, компакция истории Hermes сливает заметку со следующим сообщением
пользователя, и порядок сообщений в чате ломается. Меньше заметок — меньше слияний. Само слияние —
поведение Hermes; его не чиним и Hermes не меняем.

## Правило

- Успешный вызов `speech_job_result` в ходе привязанного чата для закончившегося задания
  (`finished`, `failed`, `cancelled`) помечает последнюю попытку увиденной. Метку ставит второй
  хук `post_tool_call` модуля `hermes/plugin/sound-scout/seen.py` через `sound-scout jobs seen
  --session-id=<чат> --attempt=<n> <job_id>`. Чат хук берёт из контекста хода
  (`HERMES_SESSION_PLATFORM == "local"`, `HERMES_SESSION_CHAT_ID`), как `binding.py` (0025), а не
  из аргументов модели.
- Метка — поле `seen_at` попытки `job.json["attempts"][n-1]` (`delivery.mark_seen`). Отказ
  `not_seen`, без изменений: чат не тот, что в `binding.session_id`, задание без привязки, задание
  не закончилось, или `attempt` — не последняя попытка (ход прочитал итог прежней попытки: заметка
  последней нужна). Повторная метка ничего не меняет. `updated_at` не меняется: от него считает
  срок staging. MCP-инструмента для метки нет: её пишет только процесс Hermes.
- Для увиденной попытки `next_delivery` пишет состояние доставки `seen_in_chat` (`tag`, `attempt`,
  `seen_at`, `ended_at`) и `send` не отдаёт. `tag` — тот же, что получила бы заметка: нумерация
  `-a<n>` следующих попыток не сдвигается. После `jobs retry` запись `seen_in_chat` уходит в
  `deliveries`, и новая попытка получает свою заметку. `request_redelivery` её отклоняет: он
  принимает только `delivery_unconfirmed`.
- Заметка не уходит, пока в привязанном чате идёт или стоит в очереди ход, который ещё может
  прочитать итог. Через `TURN_WAIT_SECONDS` = 600 с после конца попытки заметка уходит без
  проверки.
- Попытка, итог которой никто не прочитал, получает одну заметку, как раньше: агент не ждал
  итога, или владелец нажал «Повторить разбор» в карточке и хода нет.
- Метка «увидено» к `redelivery_requested` не применяется: это явная просьба оператора. Срок
  ожидания применяется, от `requested_at`.
- Текст заметки не меняется.

## Сигнал хода: очередь admission чата

Сигнал хода — не хук, а очередь admission чата в session authority. Цикл доставки уже работает в
процессе-владельце и читает её из своего потока.

- `gateway/session_authority.py:349–356` `_handle`: `execution_state` — `running`, если в
  pending-строках admission чата есть `started`, и `unknown`, если есть `unknown`. По нему Hermes
  отдаёт `running` в `session.resume` (`gateway/session_controls.py:759`) и в `session.info`
  (`gateway/session_authority.py:493`). Тот же флаг читает карточка при отмене (0029).
- `persistence/sqlite/authority/_admission_store.py:121–127` `list_session_admissions`: pending —
  статусы `queued`, `started`, `unknown`. Обёртка — `persistence/sqlite/authority/admissions.py:175–178`
  `admissions.list(session_id=…, pending_only=True)`.
- `_admission_store.py:158–178`: голова FIFO ждёт, пока в чате есть `started` или `unknown`.
  Заметка `deliver_to_local_chat` — тоже admission (`gateway/session_cron.py:43–62`,
  `admit_route_wake`). Значит, заметка пишется в историю только после всего, что стоит в очереди
  перед ней. Принятую заметку отозвать нельзя, поэтому решение принимается до `send`.
- `gateway/session_authority.py:277–284` `logical_owner`: идентичность FIFO — корень линии
  компакции. Прецедент того же вопроса «есть ли что-то в FIFO чата» —
  `gateway/run_turn_persistence.py:484`:
  `bool(authority_persistence(authority).admissions.list(session_id=owner))`.
- `gateway/session_cron.py:36–40` `owner_for_home` даёт `(authority, loop)` только в
  процессе-владельце. `gateway/session_cron.py:65–80`: чужой поток выполняет работу на loop
  владельца через `asyncio.run_coroutine_threadsafe(…).result(timeout=…)`.

Плагин делает так же: `hermes_chat_busy(chat_id)` выполняет на loop владельца
`bool(authority_persistence(authority).admissions.list(session_id=authority.logical_owner(chat_id)))`
с пределом `PROBE_TIMEOUT_SECONDS` = 10 с. Без владельца — `RuntimeError`. Ошибка проверки
считается «занято» и ограничена тем же сроком.

Чтобы не писать маркер `sending` до проверки, `deliver-next` получает действие `probe` без записи
и аргументы `--idle-chat=<id>`/`--busy-chat=<id>` (повторяемые). Тик цикла до
`PROBES_PER_TICK` = 16 раз: `deliver-next` с накопленными множествами; на `probe` — проверка чата
и новый вызов. Свободный чат — заметка уходит; занятый — задание пропускается, цикл ищет дальше.
Занятый чат X не задерживает заметку свободного чата Y. Тик, который кончился без отправки при
занятом чате, возвращает `waiting_turn`.

## Хуки хода — почему не они

- `pre_llm_call` — один раз в начале хода (`agent/turn_context.py:912–972`, вызов `:1362`).
  `post_llm_call` — один раз после цикла инструментов (`agent/turn_finalizer.py:479–517`).
  `on_session_end` — в конце каждого хода, с `completed`/`failed`/`interrupted`
  (`agent/turn_finalizer.py:912–926`).
- Все три — хуки с ограничением по времени: при таймауте Hermes вызов пропускает
  (`hermes_cli/plugins_dispatch.py:42–45`). `finalize_turn` вызывается из `_finalize_from_state`
  (`agent/conversation_loop.py:1582–1587`), не в `finally`: исключение из цикла хода
  `on_session_end` не даёт. Аргументы хуков — `session_id` агента (физический сегмент), а не id
  чата.

## Варианты

1. **Сигнал хода — хуки `pre_llm_call` + `on_session_end`**, состояние «чат → ход идёт» в памяти
   плагина. Пропущенный хук оставляет ложное состояние; перезагрузка плагинов создаёт новый модуль,
   а поток доставки остаётся со старым; очередь `queued` перед заметкой хуки не видят. Отклонён.
2. **Только `post_llm_call`.** Начала хода не даёт. Отклонён.
3. **Очередь admission чата из потока доставки.** Та же правда, по которой Hermes сам ставит
   заметку в очередь и показывает `running`. Состояние в БД authority переживает перезагрузку
   плагина и перезапуск gateway; своего состояния у плагина нет. Минус — непубличный API
   (`authority.persistence.admissions`, `logical_owner`); плагин и так читает `_physical_target`
   и `authority.db.get_messages`. **Выбран.**
4. **Захват маркера, проверка, `released`.** Старейшее задание в занятом чате блокирует заметки
   других чатов; запись `job.json` каждые 10 с; падение между захватом и освобождением даёт
   осиротевший маркер и ложный `delivery_unconfirmed`. Отклонён.
5. **Только метка «увидено», без ожидания.** Ход читает итог примерно через минуту после конца
   задания, а цикл успевает поставить заметку раньше; admission отменить нельзя. Отклонён.

## Срок: 600 с от конца попытки

- В чатах владельца ход читает итог примерно через 1 мин после конца задания. 600 с — запас в 10
  раз.
- После срока заметка, поставленная при идущем ходе, всё равно ждёт в FIFO за ним
  (`_admission_store.py:158–178`). Цена срабатывания срока — только возможный дубль, не заметка
  посреди хода.
- Срок нужен для зависших `unknown`-строк, ошибки чтения очереди и хода, который итог так и не
  читает. Таймаут бездействия хода Hermes — 1800 с (`gateway/run_turn.py:1283–1291`,
  `HERMES_AGENT_TIMEOUT`); для этих случаев он слишком долгий.
- Начало срока — `ended_at` последней попытки; у задания без попыток (отменено в очереди) и у
  попытки без `ended_at` — `updated_at` задания; у `redelivery_requested` — `requested_at`. Время,
  которое нельзя прочитать, считается прошедшим: заметка не застревает.

## Другие пути

- Повтор из карточки, хода нет: очередь чата пуста → заметка уходит сразу, как сейчас.
- Ход идёт, но итог не читает (агент не ждал, голосовое без просьбы): заметка уходит после конца
  хода или после срока.
- `failed` без результата: `speech_job_result` отвечает отказом `no_result` (`status` не `ok`) →
  метки нет → заметка уходит.
- Ход прочитал итог прежней попытки (у последней нет результата) → `not_seen` → заметка последней
  попытки уходит.

## Остаточный риск

- Id чата хода не совпал с `binding.session_id` (например, физический сегмент после компакции
  вместо корня) → метки нет → одна заметка, как без этого решения.
- Хук пропущен по таймауту Hermes или CLI не ответил → метки нет → одна заметка.
- Хук ставит метку, когда `speech_job_result` вернул итог, а не когда ответ дошёл до чата. Ход
  прочитал итог и оборвался до ответа (сбой модели, отмена, падение gateway) → метка есть →
  заметки нет, и итога попытки в чате нет. Итог по-прежнему читается: `speech_job_status`
  показывает `seen_at`, а `speech_job_result` или карточка задания отдают результат. `jobs
  redeliver` состояние `seen_in_chat` не принимает: повторная доставка по 0018 и #122 остаётся
  только для `delivery_unconfirmed`. Расширять её — отдельное решение владельца.
- Ход читает итог позже 600 с после конца попытки → заметка уже ушла → дубль ответа, как до этого
  решения.

## Что не проверено вживую

- Чтение очереди admission в живом gateway: тесты подменяют `chat_busy`
  (`tests/test_hermes_delivery.py`) и модули Hermes (`test_the_turn_check_reads_the_pending_admissions_of_the_lineage_root_on_the_owner_loop`).
- Хук `seen` в живом ходе Hermes: тест `tests/test_hermes_seen.py` повторяет вызов хука по
  `model_tools._emit_post_tool_call_hook`, как тесты 0025.
- Проверяет владелец шагом O10 после установки: копия `hermes/plugin/sound-scout/` в
  `~/.hermes/plugins/sound-scout` и `hermes/skills/speech-review/` в
  `~/.hermes/skills/media/speech-review`, перезапуск `hermes-gateway.service` (порядок 0017 § 4).

## Тесты

`tests/test_delivery.py` (раздел «no note for an attempt the chat saw»), `tests/test_hermes_seen.py`,
`tests/test_hermes_delivery.py` (раздел «acceptance criteria of #190»),
`tests/test_hermes_binding.py::test_register_adds_the_binding_and_seen_hooks_and_starts_the_delivery_loop`.
