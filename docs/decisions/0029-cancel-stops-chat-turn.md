# 0029 — «Отменить разбор» останавливает ход Hermes в чате задания

- **Статус:** принято, реализовано в коде и тестах. Живая проверка на Windows-клиенте — шаг
  владельца O6 (`docs/acceptance/A01-A14.md`).
- **Дата:** 2026-11-24.
- **Решение владельца:** [Decision в #188](https://github.com/renidere/sound-scout-plugin/issues/188#issuecomment-7133616629):
  «Отменить разбор» также останавливает активный ход Hermes этого чата.
- **Задача:** [#188](https://github.com/renidere/sound-scout-plugin/issues/188); найдена в O6
  [#112](https://github.com/renidere/sound-scout-plugin/issues/112). Только штатные вызовы SDK —
  [#103](https://github.com/renidere/sound-scout-plugin/issues/103).
- **Hermes:** Hermes Agent v0.23.3, ревизия `caafe5389a`; Windows Desktop 0.23.3 собран из того же
  дерева `apps/desktop`. Код прочитан только чтением. Пути ниже — от корня установки Hermes,
  строки — по ревизии `caafe5389a`.

## Какой вызов SDK останавливает ход

Отдельного метода SDK вида `host.stopTurn` нет. Ход останавливает RPC gateway
`session.interrupt`. Плагину он доступен через штатный `host.requestProfile`, тот же путь, что у
`session.resume`, `file.attach` и `prompt.submit` в рекордере (решения 0013 и 0016).

- `apps/desktop/src/sdk/index.ts:1384` — `host.requestProfile(route, method, params, timeoutMs,
  options)`. Это JSON-RPC gateway по маршруту `profileRoutes` без списка разрешённых методов.
  Реализация — `apps/desktop/src/sdk/profile-request.ts:25` (`requestPluginProfile` →
  `requestGatewayForAgent`).
- `apps/desktop/src/app/session/hooks/use-prompt-actions/index.ts:753` (`cancelRun`) — кнопка
  «Стоп» самого Desktop. Она шлёт `session.interrupt` с `session_id` runtime-сессии чата (строка 831).
- `apps/desktop/src/plugins/hermes-bots/group-rounds.ts:606` — штатный плагин Desktop
  `hermes-bots` останавливает ход участника тем же `session.interrupt` через свой маршрут.
- `apps/desktop/src/api/client.ts:52` и `:532` — `session.interrupt` входит в `ATTACH_REQUIRED`:
  клиент сам подключает сессию к сокету до вызова.
  `apps/desktop/src/api/canonical-protocol.ts:420` — клиент берёт `execution_generation` из
  параметров или из своего кэша сессии.
- `gateway/session_controls.py:973` и `gateway/session_authority.py:972` — обработчик в
  `hermes gateway run`. Поколение не совпало — отказ `stale_generation`. Ход идёт
  (`execution_state == 'running'`) — жёсткая остановка агента (`request_hard_interrupt`). Хода нет —
  ответ без ошибки и без остановки.
- `gateway/session_controls.py:759` — ответ `session.resume` несёт `running` и
  `execution_generation` сессии. `persistence/sqlite/authority/_admission_store.py:176` — поколение
  растёт на каждом новом ходе.

## Как работает кнопка

1. `POST /jobs/{job_id}/cancel` через `ctx.rest`, как раньше. Отказ показывается на карточке как
   раньше; ход чата тогда не останавливается.
2. После принятой отмены — маршрут задания `profileRoutes` по `connectionId` + `profile` задания →
   `retainProfile` → `session.resume` `{session_id: <чат задания>, omit_messages: true}`.
   Чат задания — тот, в который ушло голосовое (`job.sessionId`); карточка видна только в нём.
   Сфокусированный чат и другие чаты не участвуют.
3. `running !== true` — хода нет: больше вызовов нет, ошибки нет.
4. `running === true` — `session.interrupt` `{session_id: <runtime-сессия из resume>,
   execution_generation: <из resume>}`. Поколение из ответа `resume` закрепляет остановку за ходом,
   который шёл в момент нажатия: новый ход после него получит отказ `stale_generation`, а не
   остановку.
5. Сбой любого шага остановки — задание остаётся отменённым. Карточка показывает «Разбор отменён,
   но ход чата не остановлен: <текст ошибки Desktop>» (`desktopErrorText`, без IPC-обёртки).
   Сокет маршрута освобождается в `finally`.
6. Пока отмена шла, карточку могло занять новое задание. Его чат другой — ход чата отменённого
   задания всё равно останавливается, ошибка остановки на чужую карточку не попадает. Его чат тот
   же — остановки нет: идущий ход может быть ходом нового задания, который пользователь только что
   запросил.

## Границы

- Останавливается только идущий ход. Сообщения чата в очереди Hermes (`pending`) остаются: их
  снимает `prompt.cancel`, владелец этого не решал.
- Фоновые процессы хода (`terminal(background=true)`) и делегирования `session.interrupt` не
  завершает (комментарий в `use-prompt-actions/index.ts` у `cancelRun`).
- Отмена задания идёт первой: остановка хода не может помешать отмене разбора.

## Проверки

- `tests/js/recorder.test.mjs` на поддельном SDK: остановка хода чата задания и ни одного вызова
  для другого чата; чат без хода — только отмена задания, без ошибки; сбой `session.interrupt`,
  `session.resume` или `retainProfile` — задание отменено, на карточке текст ошибки Desktop; отказ
  отмены задания — ход не трогается; новое задание на карточке во время отмены — ход
  останавливается только при другом чате.
- Живая проверка — шаг владельца O6: после «Отменить разбор» ход в чате X останавливается без
  кнопки «Стоп».
