# Предупреждение о прежнем направлении `hermes-voice-note` (черновик)

**Статус: черновик. Публикация ждёт одобрения владельца ([#117](https://github.com/renidere/sound-scout-plugin/issues/117), третий критерий). В `renidere/hermes-voice-note` ничего не опубликовано.**

- **Задача:** [#117](https://github.com/renidere/sound-scout-plugin/issues/117); эпик
  [#92](https://github.com/renidere/sound-scout-plugin/issues/92).
- **Основание:** `docs/decisions/0013-voice-note-integration.md` § 2 (факты), § 3.2 (пакет плагина),
  § 3.3 (рекордер), § 9 п. 4; решение владельца в
  [#94, п. 4](https://github.com/renidere/sound-scout-plugin/issues/94#issuecomment-7024493488):
  обратимая точная пометка, repository, releases и artifacts не удаляются, новый рекордер эту задачу
  не ждёт.
- **Сверено** 2026-11-20 только чтением: README `renidere/hermes-voice-note` (ветка `main`) и описания
  выпусков `v0.4.1`, `v0.4.2`.

## Что помечается

| место | что там сейчас | блок |
|---|---|---|
| `README.md` ветки `main` | установка Windows ZIP со встроенным SDK, `Start-Voice-Note.cmd`, SDK patch для Webapp | «Блок для README» — в начало файла |
| выпуск `v0.4.1` | portable-сборка Hermes Light `3b42157` со своим SDK | «Блок для выпусков» — в начало описания |
| выпуск `v0.4.2` | тот же EXE со встроенным плагином и выделенным профилем | тот же блок |

README и описания выпусков написаны по-русски, поэтому блоки тоже на русском. Код плагина,
остальные файлы, теги и файлы выпусков не меняются. Блок обёрнут маркерами
`<!-- sound-scout-direction-warning:start -->` и `<!-- sound-scout-direction-warning:end -->`;
по ним его можно точно найти и снять.

Требования версий (0013 § 2):

- v0.2.0–v0.3.2 — только штатные вызовы SDK (`profileRoutes`, `retainProfile`, `requestProfile`,
  `file.attach`, `composer.submit`);
- v0.4.0 — обязательный `host.retainSession`, которого нет в штатном SDK: в штатном клиенте плагин
  не работает;
- v0.4.1–v0.4.2 — тот же `plugin.js`, SDK приходит в неподписанном Windows EXE со своими
  `HERMES_HOME` и `userData`, то есть вторым клиентом и вторым профилем.

Целевое требование: плагин `sound-scout` работает в штатном Hermes Desktop только на штатных вызовах
Desktop Plugin SDK; отдельный EXE, форк Hermes, второй профиль и второй клиент не нужны и не
допускаются (0013 § 3.2, § 3.3). Опора — закоммиченный SDK Hermes Agent v0.23.1 (`3b4215785882`)
без патча `retainSession` (0016).

## Согласованность с документами #111

[#111](https://github.com/renidere/sound-scout-plugin/issues/111) правит `README.md`,
`docs/hermes-install.md`, `docs/ACCEPTANCE.md` и `docs/spec/README.md`. Блоки не дают второй
инструкции к ним:

- они не содержат команд установки и подключения: подключение к Hermes и включение backend-части
  (`plugins.enabled`, перезапуск gateway) остаются шагом владельца по `docs/hermes-install.md` и
  0013 § 3.2;
- они не называют новый рекордер готовым: ход работ — эпик #92;
- они не говорят о T09b, A01–A14 и статусе ТЗ, поэтому не расходятся с `docs/ACCEPTANCE.md` и
  `docs/spec/README.md`;
- направление в блоках совпадает с 0013 и строкой «Направление» в `CLAUDE.md`.

## Блок для README

```markdown readme-warning
<!-- sound-scout-direction-warning:start -->
> [!WARNING]
> **Это направление больше не целевое.** Отдельный Windows EXE (Hermes Light, `Start-Voice-Note.cmd`) и второй профиль Hermes со своими `HERMES_HOME` и `userData` — не целевой путь. Плагин v0.4.0–v0.4.2 требует `host.retainSession`, которого нет в штатном Hermes Desktop Plugin SDK, поэтому в штатном клиенте он не работает.
>
> Целевое направление — один плагин `sound-scout` в репозитории [renidere/sound-scout-plugin](https://github.com/renidere/sound-scout-plugin). Он работает в штатном Hermes Desktop только на штатных вызовах Desktop Plugin SDK (`profileRoutes`, `retainProfile`, `session.resume`, `file.attach`), без `host.retainSession`, отдельного EXE, форка Hermes, второго профиля и второго клиента. Рекордер переносится туда из v0.4.2. Обоснование — [решение 0013](https://github.com/renidere/sound-scout-plugin/blob/main/docs/decisions/0013-voice-note-integration.md), ход работ — [эпик #92](https://github.com/renidere/sound-scout-plugin/issues/92); новый плагин ещё в разработке.
>
> Этот репозиторий, его выпуски и их файлы остаются доступны, ничего не удаляется. Текст ниже описывает прежнее направление и сохранён без изменений.
<!-- sound-scout-direction-warning:end -->
```

## Блок для выпусков v0.4.1 и v0.4.2

```markdown release-warning
<!-- sound-scout-direction-warning:start -->
> [!WARNING]
> **Этот выпуск относится к прежнему, нецелевому направлению.** Его Windows-комплект — отдельный неофициальный неподписанный EXE со своими `HERMES_HOME` и `userData`, то есть второй клиент и второй профиль Hermes. Это не целевой путь. Плагин этого выпуска требует `host.retainSession`, которого нет в штатном Hermes Desktop Plugin SDK.
>
> Целевое направление — один плагин `sound-scout` в репозитории [renidere/sound-scout-plugin](https://github.com/renidere/sound-scout-plugin). Он работает в штатном Hermes Desktop только на штатных вызовах Desktop Plugin SDK, без отдельного EXE, форка Hermes, второго профиля и второго клиента. Обоснование — [решение 0013](https://github.com/renidere/sound-scout-plugin/blob/main/docs/decisions/0013-voice-note-integration.md), ход работ — [эпик #92](https://github.com/renidere/sound-scout-plugin/issues/92); новый плагин ещё в разработке.
>
> Выпуск и все его файлы остаются доступны, ничего не удаляется. Текст ниже — исходное описание выпуска без изменений.
<!-- sound-scout-direction-warning:end -->
```

## Публикация (только после одобрения владельца)

Команды выполняет агент из корня checkout `sound-scout-plugin` на коммите с этим файлом. Нужен `gh`
с правом записи в `renidere/hermes-voice-note`. Вставку и снятие делает
`scripts/voice_note_warning.py` (только stdlib): блок берётся из этого файла, текст с целым блоком не
меняется. Повторный запуск ничего не вставляет второй раз. Текст с маркером без целого блока
скрипт не трогает и завершается с кодом 2: такой текст правят вручную. Копии описаний выпусков до правки
сохраняются в `local/voice-note-warning/` (вне git).

### README

```sh
set -euo pipefail
SS="$PWD"
WORK="$(mktemp -d)"
git clone --quiet https://github.com/renidere/hermes-voice-note "$WORK/hermes-voice-note"
cd "$WORK/hermes-voice-note"
if python3 -I "$SS/scripts/voice_note_warning.py" has README.md; then
  echo "README: the warning is already there, nothing to do"
else
  python3 -I "$SS/scripts/voice_note_warning.py" insert readme README.md README.md
  git add README.md
  git commit --quiet -m "docs: mark the separate Windows client direction as not the target"
  git push --quiet origin HEAD:main
  echo "README commit: $(git rev-parse HEAD)"
fi
cd "$SS"
```

### Выпуски

```sh
set -euo pipefail
REPO=renidere/hermes-voice-note
BK="$PWD/local/voice-note-warning"
mkdir -p "$BK"
for TAG in v0.4.1 v0.4.2; do
  gh release view "$TAG" -R "$REPO" --json body > "$BK/$TAG.current.json"
  if python3 -I scripts/voice_note_warning.py has "$BK/$TAG.current.json"; then
    echo "$TAG: the warning is already there, nothing to do"
    continue
  fi
  cp "$BK/$TAG.current.json" "$BK/$TAG.before.json"
  python3 -I scripts/voice_note_warning.py insert release "$BK/$TAG.before.json" "$BK/$TAG.after.md"
  gh release edit "$TAG" -R "$REPO" --notes-file "$BK/$TAG.after.md"
  echo "$TAG: published, backup sha256 $(sha256sum "$BK/$TAG.before.json" | cut -d' ' -f1)"
done
```

После публикации агент пишет в #117: SHA коммита README, суммы SHA-256 копий `*.before.json` и
команды отката ниже.

## Откат

### README

Снять блок по маркерам (работает и после других правок README):

```sh
set -euo pipefail
SS="$PWD"
WORK="$(mktemp -d)"
git clone --quiet https://github.com/renidere/hermes-voice-note "$WORK/hermes-voice-note"
cd "$WORK/hermes-voice-note"
python3 -I "$SS/scripts/voice_note_warning.py" strip README.md README.md
if git diff --quiet README.md; then
  echo "README: no warning, nothing to do"
else
  git add README.md
  git commit --quiet -m "docs: remove the Sound Scout direction warning"
  git push --quiet origin HEAD:main
fi
cd "$SS"
```

Другой путь, пока коммит предупреждения не перекрыт другими правками README: `git revert --no-edit
<README commit>` в свежем клоне, затем `git push origin HEAD:main`.

### Выпуски

Снять блок по маркерам:

```sh
set -euo pipefail
REPO=renidere/hermes-voice-note
BK="$PWD/local/voice-note-warning"
mkdir -p "$BK"
for TAG in v0.4.1 v0.4.2; do
  gh release view "$TAG" -R "$REPO" --json body > "$BK/$TAG.revert.json"
  python3 -I scripts/voice_note_warning.py strip "$BK/$TAG.revert.json" "$BK/$TAG.reverted.md"
  if ! python3 -I scripts/voice_note_warning.py has "$BK/$TAG.revert.json"; then
    echo "$TAG: no warning, nothing to do"
    continue
  fi
  gh release edit "$TAG" -R "$REPO" --notes-file "$BK/$TAG.reverted.md"
done
```

Точное восстановление из копии, если описание после публикации не менялось (`strip` текста без блока
отдаёт его без изменений):

```sh
TAG=v0.4.1  # или v0.4.2
python3 -I scripts/voice_note_warning.py strip "local/voice-note-warning/$TAG.before.json" "local/voice-note-warning/$TAG.before.md"
gh release edit "$TAG" -R renidere/hermes-voice-note --notes-file "local/voice-note-warning/$TAG.before.md"
```
