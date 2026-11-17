---
name: new-task
description: Files a new task for Sound Scout as a GitHub issue of renidere/sound-scout-plugin in the Backlog column of the board github.com/users/renidere/projects/1, written in Russian the way the owner writes tasks — context, references to the spec (R01–R12, T01–T10), acceptance criteria an agent can prove, dependencies. First looks across the whole board for a task that already covers it and says where that one stands instead of filing a duplicate. Asks the person only for what neither their words nor the code and the spec answer. Use whenever someone in the sound-scout-plugin checkout describes something to build, fix or investigate — "заведи задачу", "запиши в бэклог", "file a task", "add this to the board", "нужно сделать …", or the same in any language. Also used by work-task (in agent mode, no questions) to file the problems its agents run into along the way.
---

# File a new task

Two kinds of callers:

- **A person** (the default): the owner, or someone working with them, says what should be built,
  fixed or looked into. You turn that into a task the autonomous agent (`work-task`) can carry
  out alone, asking only for what you can't find out yourself.
- **An agent** — the args say `agent mode` (work-task's Filer step). There is nobody to ask; see
  **Agent mode** at the end.

`<skill dir>` is the directory of this file. The board script is shared with work-task:
`<skill dir>/../work-task/scripts/board.sh`.

## Where things are

- The repo: `$WS`, what `bash <skill dir>/../work-task/scripts/workspace.sh` prints. Read
  `CLAUDE.md`, `docs/spec/REQUIREMENTS.md` (the R/T items) and the code quietly to understand what
  exists and what is planned — that is how you ask good questions and write precise criteria.
- The tasks: issues in `renidere/sound-scout-plugin`, every one of them on the board. The epics
  (label `epic`, with sub-issues) group them by stage; `bash <scripts>/board.sh tasks` lists
  every card.
- `gh` must be logged in (`gh auth status`). If it isn't, tell the person to run
  `! gh auth login` and stop.

## Talking to the person

- **Their language**, in chat. The issue itself is in Russian (every task on this board is), with
  code identifiers, commands and spec IDs as they are.
- **Offer, don't interrogate.** You've read the spec and the code, so you can usually guess the
  answer; let them pick it with AskUserQuestion, concrete options built from what the project
  really does. Ask only what neither their message nor the repo answers, up to four questions in
  one round, and aim for one round.
- Don't explain how you work. They want the task filed.

## 1. Understand the request

Decide what it is:

- **Bug** (label `bug`) — the tool does something it isn't meant to: an error, a wrong status, a
  report that claims more than the run did, a test that passes for the wrong reason.
- **Feature** (label `enhancement`) — something new, or a change to behaviour that works as
  built.
- **Spike** (label `spike`) — a question to answer or a capability to check before building; its
  result is a document under `docs/decisions/` or `docs/eval/`, not product code.

**Area**: one or more `area:*` labels — `core`, `audio-io`, `acoustics`, `asr`, `audio-backend`,
`eval`, `reports`, `cli`, `mcp`, `hermes`, `security`, `infra`, `docs`. **Priority**: one of
`priority:blocker` (the project can't move without it), `critical` (important, no way around),
`major` (in the way, with a workaround), `minor`, `trivial`. **Permission**: `permission-required`
when any part needs what the spec reserves to the owner — a model download, external inference,
recordings, spending, connecting Hermes — and `need-a-person` when the whole task is a person's
step (then the card goes to Need a Person, not Backlog).

### Already on the board?

Before asking anything, look for a task that already covers this — open or done:

- `bash <scripts>/board.sh tasks > <mktemp file>` lists every card: status, open or closed, link
  and title. Read the open ones in full and `grep -i` the closed ones with the request's key
  words and their synonyms.
- Search the bodies too: `gh search issues "<words>" --repo renidere/sound-scout-plugin --limit 20 --json url,title,state`,
  twice with different key words.
- Read every candidate with `bash <scripts>/board.sh issue <url>` before calling it a match.

A match: tell them — its title, the link, and where it stands (Backlog, In Progress, Need a
Person, or closed and how it ended) — then ask: add their details there as a comment, leave it,
or file theirs as different. A partial match: file only the rest and name the related task in the
body.

## 2. Fill the gaps

What a finished task needs:

- **Context** — why, for whom, what the spec says (the R/T items it serves; quote the limit or
  rule when one applies).
- **Acceptance criteria** — two to six, each a single fact a pytest test or a real command can
  prove: "`analyze` refuses a file over 100 MiB with an explicit error, nothing is written", not
  "input validation works". A bug's first criterion is the expected behaviour, and one more says
  it stays fixed in the situation it broke in. A spike's criteria name the document, the questions
  it must answer and the sources it must cite.
- **Dependencies** — tasks that must be closed first, as full issue URLs on a `Зависит от` line.
- **Permissions** — what the owner must grant or do, when anything.
- **Technical notes** — pointers for the agent: modules, the spec sections, what to reuse (the
  existing speech layer is described in `local/HANDOFF.md`, outside git — refer to it by that
  name, never by an absolute personal path).

## 3. Write it

```
<Контекст: два-четыре предложения — зачем, что говорит ТЗ (R…/T…), что уже есть.>

## Критерии приёмки

- [ ] …

## Зависит от

- https://github.com/renidere/sound-scout-plugin/issues/<n>  (или «ничего»)

## Разрешения

<что нужно от владельца, или раздел опущен>

## Технические заметки

<модули, разделы ТЗ, что переиспользовать; без абсолютных личных путей>
```

**Title**: the outcome in under 80 characters, Russian, code identifiers as they are:
"`analyze`: отказ при превышении 180 с / 100 MiB без скрытого усечения".

## 4. Show it, then file it

Show the draft: title, labels and body as they will be filed. Ask with AskUserQuestion:
"Завести" / "Изменить". Change what they ask and show it again. Then:

1. Take the second look at the board from **Already on the board?** — a teammate or an agent may
   have filed the same thing meanwhile.
2. Write the body to a temp file (`mktemp`) and
   `gh issue create --repo renidere/sound-scout-plugin --title "<title>" --body-file <file> --label <a> --label <b> …`.
3. Put it on the board: `bash <scripts>/board.sh add <issue url>` (Backlog; `need-a-person` for a
   person's step). If that fails, retry once; still failing, give them the link and say the task
   exists but isn't on the board yet.
4. Under an epic: `bash <scripts>/board.sh add-subissue <epic url> <issue url>`. Blocked by other
   open tasks: `bash <scripts>/board.sh blocked-by <issue url> <blocker url>...` — the queue
   passes over an issue with an open "blocked by" link.

Tell them in one or two sentences: the link and that it is in Backlog.

## Agent mode

The args say `agent mode`, give the issue the problems were found on (`found by <url>`) and list
the problems as work-task's agents reported them. Nobody answers questions: decide everything
yourself from the code on `origin/main` (`git -C <$WS> fetch origin main`, then
`git -C <$WS> show origin/main:<path>`). Don't change any code.

1. **Check each problem** against the code, read only. Drop it, with a reason, if it's wrong,
   already fixed, only a style nit, or about this machine (a missing tool, a login, a config
   value) rather than the code. Merge entries that are the same problem.
2. **Look for an open issue that already covers it**:
   `gh issue list --repo renidere/sound-scout-plugin --state open --search "<key words>" --limit 20`,
   twice with different key words, plus `--search "work-task-found-by in:body"`. A match: don't
   file, report it.

   An issue that carries `<!-- work-task-found-by: <this found-by url> -->` is not someone else's:
   an earlier attempt of this same Filer created it and may have failed before putting it on the
   board. Look for these without `--search`, whose index lags behind a just-created issue:
   `gh issue list --repo renidere/sound-scout-plugin --state open --limit 1000 --json url,title,body --jq '.[] | select(.body | contains("<!-- work-task-found-by: <found-by url> -->")) | "\(.url)\t\(.title)"'`.
   One that covers a problem is not filed again: if `bash <scripts>/board.sh item <url>` succeeds
   and prints nothing, or a Status of `-`, run `board.sh add <url>`; report it under FILED either
   way, never EXISTS, so work-task carries it.
3. **File the rest**, one issue each: bug, feature or spike, priority by what the problem costs
   (a red CI on main stops every task: `priority:blocker`), area labels, title and body as in
   **3**, from the code instead of a person. Then add, at the very top and the very bottom:

   ```
   <!-- work-task-found-by: <found-by url> -->
   Найдено при работе над <found-by url>.

   <the body>

   ## Технические заметки

   <где: файлы со строками, как ссылки на main (https://github.com/renidere/sound-scout-plugin/blob/<main sha>/<path>#L<n>);
   как проявилось: упавшая проверка, тест или замечание ревью, ошибка в несколько строк>
   ```

   Never paste values from `.env` files, `sound-scout.toml` or secrets, never an absolute
   personal path, never a recording's content.
4. **Put each new issue on the board** in Backlog: `bash <scripts>/board.sh add <url>`. When the
   found-by issue has a parent epic and the problem belongs to the same area, make the new issue
   its sub-issue too (`board.sh add-subissue`).

Reply with exactly this, for the caller to log:

```
FILED: <one line each: new issue url — title>, or none
EXISTS: <one line each: problem → existing issue url>, or none
DROPPED: <one line each: problem — why>, or none
RESULT: done | failed — <reason>
```
