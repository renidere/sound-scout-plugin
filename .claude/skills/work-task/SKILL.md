---
name: work-task
description: Takes one Sound Scout task — a GitHub issue URL or number of renidere/sound-scout-plugin from the board github.com/users/renidere/projects/1 — and carries it to the end without asking anything. An Executor checks whether the task is still relevant and implements what is missing in a git worktree of its own; a Verifier proves every acceptance criterion with a pytest test or a real command; revmux reviews the change; a Lander applies the findings, pushes to main and waits for green CI; then the issue is commented, closed and moved to Done. Work that needs a permission or a step only the owner can take ends in Need a Person with what is needed named. Other problems the agents run into are filed as new issues and worked the same way. Use whenever someone in the sound-scout-plugin checkout gives an issue link or number and wants it done — "do this task <url>", "take #12", "work on this issue", "сделай задачу 12", or the same in any language — and whenever work-backlog hands over a task. For a task that has no issue yet, use new-task first.
---

# Work one Sound Scout task

Input: an issue URL (or `#<number>` / `<number>`, which means
`https://github.com/renidere/sound-scout-plugin/issues/<number>`). The args may also say
`found by <url>, depth <n>`: this issue was filed by an earlier work-task for a problem it ran into
(see **7. Found problems**). No such words means depth 0. `recovery of a hung task` means an
earlier attempt died mid-way and this one picks the task up. `unattended backlog run` means
work-backlog is the caller (**Gate** step 4). `found problems: file only` means the problems the
agents find are filed (step 5) but not worked here.

Work without asking. work-backlog calls this skill unattended, and a person who calls it directly
wants a finished task, not a conversation. Never ask questions, never wait for confirmation, never
enter plan mode: an open question goes to a Decision agent, a permission that only the owner can
give ends the task in Need a Person (**Held**), a breakage goes through **Failures**.

Wherever this skill says **end**, this work-task is over and control goes back to whoever invoked
it. It never ends the turn while a caller is waiting — nobody is at the keyboard to start it
again — except to wait for an agent it launched, whose result wakes it.

You orchestrate; subagents read code, write it, test it, review it and land it. Run **one subagent
at a time** — launch it, wait for its result, then go on: they share the task's tree, and each step
builds on the previous one. Keep your own context small: don't read code yourself, and pass on
what agents return rather than re-deriving it.

## The flow

```
Start → Gate + claim → Scope [Decision] → Prepare → 1 Executor → 2 Verifier → 3 revmux → 4 Lander → 5 File found → 6 Done → 7 Found problems
```

| step | who | what |
|---|---|---|
| Decision | agent, only when needed | checks whether the task is still relevant; settles open questions, posts a `Decision:` comment with the spec |
| Prepare | you | a git worktree on the task's own branch cut from `origin/main` |
| 1 Executor | agent | checks whether the task is still relevant; implements what is missing; runs the checks |
| 2 Verifier | agent | a named pytest test or a real command per acceptance criterion, run in the tree |
| 3 revmux | agent | reviews the change and the tests together |
| 4 Lander | agent | applies the findings, commits on the branch, rebases onto main and pushes it there, waits for green CI |
| 5 File found | agent, only when problems were found | files each other problem the agents ran into as a new issue on the board |
| 6 Done | you | comments, closes the issue, moves the card to Done — or, for work that waits for the owner, leaves it open in Need a Person (**Held**); for criteria that wait for another open issue, links it as a blocker and ends `blocked` (**Partly done**) |
| 7 Found problems | you | the newly filed issues go through this same flow, one after another, down to depth 2 |

The brief for each agent is in `references/agent-briefs.md` — read it before the first agent.

## Ground rules

- **A tree of its own.** The task works only in its own git worktree (**Prepare**), on a branch
  `task/<number>` cut from `origin/main`. The main checkout in `$WS` belongs to a person: never
  edit, stash, check out, pull or reset it — only `git fetch` and `scripts/worktree.sh` touch it.
  So several tasks run side by side in one checkout, and none of them sees another's files.
- **Only onto `main`.** The branch lands on main as a fast-forward: rebased onto `origin/main`,
  pushed with `git push origin HEAD:main`. Never force-push, never rewrite pushed history, never
  push another branch. Nothing deploys: main is the shared truth, CI (`.github/workflows/ci.yml`)
  runs on every push to it, and a red run is fixed forward.
- **The spec is binding.** `docs/spec/REQUIREMENTS.md`, `docs/spec/AGENT_PROMPT.md` and
  `docs/spec/AUDIO_REVIEW_PROMPT.md` are the product's requirements; `CLAUDE.md` is how to work in
  the repo. The permission boundaries there hold for every agent: no model downloads, no external
  inference, no uploads of recordings, no spending, no change to the running Hermes, no
  `approved=true` invented — a task that needs one of those does its other parts and ends
  **Held** with the permission named. Recordings are read-only development data and never enter
  git.
- **This machine is checked by running.** Whether a check or a command can run here is learned by
  running it. A failure about this machine quotes the command that failed and its output.
- **Commit only the task's own files**, by explicit path, after `scripts/scan-staged.sh` says `ok`.
  Never `.env*`, `sound-scout.toml`, `data/`, `runs/`, `local/`, `.revmux/`, recordings or
  reports; never `--no-verify` — the repo's hooks are part of the checks.
- **One agent per task.** `scripts/claim.sh` puts a claim comment on the issue; only the agent
  holding the claim touches the issue, its card or its code. Refresh the claim before every step,
  and release it whatever the outcome. See **Gate** and **The claim while working**.
- **Temp files** (comment bodies, revmux output) go in a temp directory outside the repo
  (`mktemp -d`), never inside one.
- **Comments on the issue are in Russian** — the Decision, the outcome, the found-problem issues:
  the board is read by the owner. Code, commit subjects, test names and the agents' replies to
  you are in English.
- **Log as you go** to **the log**: `$WS/backlog-run.md` (create it if needed). The log is the
  memory that survives a context compaction or a crash — after a compaction, re-read this file
  and this task's lines in the log (the ones with its URL) to see where you are. Other tasks may
  be writing to it at the same moment, so add each line with one shell append —
  `printf '%s\n' '<line>' >> "<the log>"` — and never rewrite the file. Every step line names its
  task:

  ```
  - <YYYY-MM-DD HH:MM> | <issue url> | started | item=<item-id> | claim=<claim-id>
    - <HH:MM> #<n> <step>: <one line — e.g. "decision <comment url>", "prepared", "executor: changed", "verifier: 4 pass, 1 no-test", "review: 3 findings", "pushed <sha7> CI green">
    - <HH:MM> #<n> found: <one entry of an agent's FOUND line, as given>
    - <HH:MM> #<n> filed: <new issue url> | exists: <issue url> | dropped: <one line why>
  - <YYYY-MM-DD HH:MM> | <issue url> | done|failed|skipped|blocked|stop|held | <sha7|-> | <reason if not done>
  ```

## Start

`$WS` is the main checkout of sound-scout-plugin — what `bash <skill dir>/scripts/workspace.sh`
prints: found from the current directory, a folder inside it, or a task tree under `.work-task/`.
`<skill dir>` is the directory of this file — always give agents its absolute path.

1. `bash <skill dir>/scripts/preflight.sh` — the machine only: the checkout, GitHub access, the
   board, `uv`, `ffmpeg`. `STOP:` lines mean only a person can fix it: log the task as `stop`,
   print the first STOP line and end with the result line (see **6**) carrying `stop`. Touch
   nothing else.
2. Find three things for every agent:
   - the revmux **profile**: `default` (revmux's own default — no `--profile` flag) when
     `codex login status` succeeds, `claude-only` otherwise;
   - the **platform**: `uname -s`;
   - the **slot**: `agent <k> of <N>` from the args — a queue run works N tasks side by side —
     or `1 of 1` without. The agents use it to take their share of this machine.

   In a queue run this skill runs in a task agent: a background subagent nobody watches. An agent
   you launch reports back on its own: ending your turn while it runs is how you wait for it, and
   its result wakes you. Long commands run in the foreground, in blocks, as the briefs' shared
   block says.
3. If the log already has a `started` line for this issue and no final line, an earlier attempt
   was interrupted — or is still running in another session; from here the two look the same.
   Don't release its claim and don't touch its changes: go on to the **Gate**, whose claim
   decides. A live claim means `taken` and you skip; a claim nobody refreshed for
   `CLAIM_STALE_MINUTES` gives way to yours, and **Prepare** then saves the dead attempt's
   half-done changes on a `leftover/` branch. Anything it pushed is on main, and the relevance
   check will see it.

## Gate

1. Read the issue: `bash <skill dir>/scripts/board.sh issue <url>` (title, state, labels, body,
   open blockers, comments — `gh issue view --comments` drops the body when not on a terminal).
   If it has a parent epic, read the epic the same way and the headlines of decisions already
   made under it: `board.sh decisions <parent-url>`. Get the card and its status:
   `board.sh item <url>` (empty: the issue isn't on the board — work it anyway and skip the card
   moves).
2. Already closed: log `skipped | already closed`, end. Nothing has been touched yet.
3. **Claim it**, before changing anything — card, comments or code:
   `bash <skill dir>/scripts/claim.sh acquire <url>`.
   - `acquired <claim-id>`: the task is yours. Keep the claim id; it goes in the `started` line.
   - `taken <claim-id> <minutes>`: another agent is on it. Touch nothing — no card move, no
     comment, no release. Log `skipped | claimed by <claim-id>`, end.
   - `orphan <claim-id> <comment-id|->` (exit 5): the claim failed and its comment could not be
     removed, so it may still be on the issue, blocking every later attempt as `taken`. Run
     `claim.sh release <url> <claim-id>` once, log `orphan <claim-id> | released` or
     `orphan <claim-id> | release failed: <error>`, then log `failed | could not claim: orphan`
     and end without touching anything else.
   - Any other exit: the network or `gh` failed and nobody knows who holds the task. Try once
     more; still failing, log `failed | could not claim: <error>`, end without touching anything.
4. **Look again**, now that you hold the claim: read the issue state and `board.sh item <url>`
   once more. The other agent may have finished between your first read and your claim.
   - Closed now: release, log `skipped | done by another agent`, end.
   - Called from a queue run (`unattended backlog run` in the args) and the card is not in
     Backlog: another run took it and finished or failed it — unless the args say this is a
     **recovery** of a hung task, when In Progress is exactly what you expect. Release, log
     `skipped | handled by another run`, end. A person calling this skill directly may mean to
     retry an In Progress or Need a Person task; for them the card's status doesn't matter.
5. **Blocked**: an open "blocked by" link, or a `Зависит от` / `Depends on` line naming a task
   that is still open. Building on a missing foundation either redoes the other task or fails.
   It'll be picked again when the blockers are done; until then:
   - Put the card back with `board.sh status <item> backlog` if it isn't there — **before** the
     comment, so a run that dies between the two leaves no In Progress card behind a `blocked`
     outcome.
   - **The blockers** are every open one of both kinds, each as its full issue URL.
   - `board.sh last-outcome <url> --body` prints the newest outcome and its comment. When it is
     `blocked` and the issues it names are exactly these blockers, an earlier run already said
     so: post nothing. Otherwise post the outcome comment (`gh issue comment <url> --body-file <file>`):

     ```
     <!-- work-task-outcome: blocked -->
     **Заблокировано.** Задача ждёт закрытия задач ниже; агент возьмёт её снова, когда они будут закрыты.

     - https://github.com/renidere/sound-scout-plugin/issues/<n>
     ```

     One line per blocker, the URL alone. Nothing else in the comment names an issue — not this
     task, no `#<n>`: an unattended caller reads every issue an outcome comment names as a
     blocker. A comment that still fails after one retry doesn't change the result: log the step
     line `comment failed: <error>` and go on.
   - Remove yourself as assignee if the log says `assigned=yes`, release the claim, log
     `blocked | waits for <url>`, end.
6. **Take it on the board**: `gh issue edit <url> --add-assignee @me` and
   `board.sh status <item> in-progress`, then log `started` with the claim id and `assigned=yes`
   if you weren't an assignee already. The assignee is for people; the claim is the lock.

## The claim while working

The claim lasts as long as you refresh it: one not refreshed for `CLAIM_STALE_MINUTES` (360 by
default) counts as abandoned, and the next agent that tries to claim the issue removes it and takes
over. So:

- **Before every step** — Decision, Prepare, Executor, Verifier, Review, Lander, File found,
  Done — run `claim.sh heartbeat <url> <claim-id>`.
- **`lost`** means another agent decided this one was dead and took the task. Stop at once: don't
  commit, push, comment or move the card, and leave the tree alone: the agent that took over works
  in the same one, and its **Prepare** already saved what was in it. Log `failed | lost the claim`
  and end with the result line `failed`. Pushes made before stay.
- A heartbeat that errors (network) is not `lost`: retry it before the step; if it keeps failing,
  treat it as a failed step.
- **Release on every way out** once you hold the claim — done, held, failed, skipped, blocked,
  stop: `claim.sh release <url> <claim-id>` is the last thing before the result line. The only
  claim you never release is one you lost while working.
- **The assignee** stays after done, held and failed. A blocked task goes back to Backlog
  unowned: if the log says `assigned=yes`, remove yourself with
  `gh issue edit <url> --remove-assignee @me`. Never remove anyone else.

## Scope and Decision

If the task is clear and fully specified — its body has acceptance criteria an engineer can build
from, and no open choice — write a short **scope note** yourself (the criteria, the modules you
expect it to touch) and go to **Prepare**. Otherwise start the **Decision agent**: a task titled
"Decision: …" or labelled `spike`, an investigation, a choice between options, or a task whose
criteria leave a design question open. When in doubt, decide: a decision on record is cheap, a
wrong guess costs a whole implementation.

Like the Executor, it first checks whether the task is still relevant — against `origin/main`,
since the task has no tree yet and the main checkout may be behind or hold someone's work — so
that no decision is designed and posted for work that is already there or no longer wanted:

- **already done** — no `Decision:` comment. Write the scope note yourself and go to **Prepare**:
  the Executor confirms it, and the Verifier proves it.
- **obsolete** — no `Decision:` comment. Go to **6** and close the issue with the reasoning, no
  commits.
- **needed** — it decides what is still missing.

It posts a `Decision:` comment on the issue (in Russian) with the options, the choice, the reasons
and a concrete spec, and returns whether code is needed. No code needed: go to **6** with no
commits. A `spike` task often ends there: its deliverable is the decision document the Decision
agent or the Executor writes under `docs/decisions/`, which still lands through steps **1**–**4**.

**Not doable here** — the work lives entirely outside this repo or needs only a person: file
found problems (step 5), post an outcome comment `skipped` in the shape of **6**
(`**Здесь не выполнимо.**` with the reason and a `**Нужен человек:**` line), move the card to
Need a Person (`board.sh status <item> need-a-person`) — no run picks Need a Person — log
`skipped`, release the claim, print the result line `skipped` and go on to **7**.

**Needs a permission** — the task, or part of it, needs what the spec reserves to the owner
(`permission-required` label, or the Decision finds it): a model download, external inference,
uploading recordings, spending, connecting to the running Hermes, approving a corpus. Nobody here
can grant it. The parts that need no permission are built and landed as usual; the task then ends
**Held** (under **6**), its card in Need a Person, with the permission named exactly. Never fake
it: no mocked inference presented as real, no `approved=true` written by an agent.

`skipped` is for work no run of this skill can ever do here. Work that waits for another issue
still open is never `skipped`: it is `blocked` — at the Gate when the issue says so up front, as
**Partly done** when an agent finds it on the way. Work that waits for the owner is `held`.

Log it as `scope: <one line>` once it is settled, before **Prepare**.

## Prepare

The task works in a **tree** of its own: a git worktree at `$WS/.work-task/<number>`, on a branch
`task/<number>` cut from `origin/main`. The main checkout stays exactly as a person left it.

1. `bash <skill dir>/scripts/worktree.sh add <number>`. It fetches `origin/main` and prints:
   - `ready <path>` — the tree, clean on fresh `origin/main`, with the main checkout's
     gitignored local config (`sound-scout.toml`, `.env*`) copied in. It has no virtualenv yet:
     the first agent that needs one runs `uv sync` there.
   - `saved <branch>` — an earlier attempt's uncommitted changes in that tree, now committed on
     that `leftover/` branch (never `git stash`). Log it.
   - `moved <path>` — a folder that sat in the tree's place but was no worktree, moved aside.
     Log it.
   - `ahead <path> <n>: <commits>` — the branch holds an earlier attempt's commits that never
     reached a remote: see **Leftovers** below.
2. Log `prepared`. From here on, every change in that tree is this task's.

`FAIL` lines before the Executor has started: nothing of the task's code exists yet and the task
isn't at fault — put it back: `board.sh status <item> backlog`, remove yourself as assignee if the
log says `assigned=yes`, remove the tree, release the claim, log `stop | <reason>`, print the
result line `stop` and end. The `Decision:` comment, if any, stays and is the spec next time.

**The tree goes when the task does.** Whatever way it ends while you hold its claim — done, held,
failed, skipped, blocked, stop — `worktree.sh remove <number>` comes just before the claim is
released, and prints `none` when the key has no tree. Without the claim the tree may be another
session's, live: an ending before it was acquired and a lost claim never touch it. It saves
whatever is uncommitted on a `leftover/` branch and deletes the branch once every commit on it is
on a remote; a branch still holding unpushed commits stays: `kept task/<n>: <k> unpushed`. Log the
`saved`, `moved` and `kept` lines.

### Leftovers

An `ahead` tree holds commits an earlier attempt at this task made and never pushed. Look them up
in the log:

- Commits the log records as `unpushed <sha>` (a Lander whose push failed) were reviewed
  already — a **Repair agent** lands them from that tree: rebase onto `origin/main`, resolve
  conflicts keeping both sides' intent, run the checks, push to main, wait for green CI.
- Anything else never got that far — don't push it. Keep it on a branch of its own and put the
  tree back on origin: `git -C <path> branch rescue/<n>-<YYYYMMDD-HHMM> && git -C <path> reset --hard origin/main`.
  Log the branch name, so the final summary tells a person where it is.
- If the Repair agent can't land them either, fall back to the rescue branch the same way.

Either way the tree ends clean on `origin/main`, as a `ready` one.

## 1. Executor

The Executor edits the task's tree only. It first checks whether the task is still relevant: each
acceptance criterion against the code on main — sibling tasks often land part of the work. Then
it implements what is missing, following `CLAUDE.md` and the spec, with unit tests for the new
behaviour, runs the checks, and doesn't commit. It returns a verdict per criterion (already met /
implemented / waits for another issue / needs the owner / can't be met here) and one of:

- **needed** — it made changes: go on.
- **already done** — nothing to change: go on anyway; the Verifier's tests are the proof, and they
  get committed.
- **obsolete** — the criteria no longer make sense (superseded, the feature removed): go to **6**
  and close the issue with the reasoning, no commits.

**Criteria that wait for another task.** A criterion that needs what another issue still open
builds can't be built here without redoing that issue. The Executor reports it
`waits for <issue url>` and builds the rest. The criteria that can be built go through **2** to
**5** as usual, and the task ends **Partly done** (under **6**), `blocked` on those issues. Every
criterion waiting, with nothing built: skip **2** to **4** and end the same way.

**Criteria that need the owner.** A criterion that needs a permission or a step only the owner
can take (**Scope and Decision**) is reported `needs the owner: <what exactly>`; the Executor
builds everything else, honestly marked (`permission_required`, `unvalidated`, `blocked` in the
code's own statuses, never a fake success). The task ends **Held**.

## 2. Verifier

It proves every acceptance criterion except those waiting for another issue or the owner: for
each, a pytest test named after it in the tree's `tests/` — written where the Executor's tests
don't already cover it, kept where they do — or, for a criterion about a command's behaviour, the
real command run in the tree with its output. It runs the tests it names and the repo's check,
and classifies each criterion:

- **pass** — a named test or command proves it.
- **no test** — not provable by a test (a document, a decision); it names what proves it instead
  (the file, the command receipt).
- **fail** — the implementation is wrong or incomplete. Give the failing tests to a new Executor
  once, then run the Verifier again. Still failing: **Failures**.

A test that passes because it asserts nothing, or because a mock stands in for the thing the
criterion is about (a real audio inference, a real decode), is `fail` — the spec is explicit that
mocked providers prove contracts, not capability.

## 3. revmux

One Review agent. It runs one revmux round on the uncommitted changes of the tree with the profile
from **Start** — the round sees the implementation and the tests together — and returns the
findings as they are, unjudged. Its RESULT:

- `reviewed`, or `partial` — some sources degraded, the others reported: reviewed, as it is. The
  Lander gets the findings.
- `could not run — <first error line>` — the round produced no review. Retry once with a new
  Review agent, given the error; twice: the change is landed unreviewed, and the comment and the
  log say "not reviewed".
- `missing` (revmux or a CLI its profile runs isn't installed) goes straight to **Failures**,
  without a retry.

## 4. Lander

Before it starts, read main's newest finished CI run:
`gh run list --repo renidere/sound-scout-plugin --workflow ci.yml --branch main --status completed -L 5 --json conclusion,url,headSha`,
the first that isn't `cancelled`. A red one that isn't this task's own (the issue is about that
failure, or it found this task) is fixed first: run the Filer (**5**) with the one entry
`main CI is red before this task's push — <run url>`, log `fixed main first: <run url> → <issue url>`,
and work that issue now with the Skill tool (`work-task`, args
`<its issue url> — found by <this issue url>, depth <this depth + 1>, found problems: file only`),
heartbeating this task's claim before each of its steps. Then go on, whatever its result.

One Lander. It weighs each finding against the code, fixes the valid ones and skips the rest with
a one-line reason, reruns the check, then: commit on the task branch (conventional subject, ending
in `(#<number>)`, no closing keywords), rebase onto `origin/main` — a rebase that brought in main
commits gets lint and the tests again, since another task's code may call what this one changed —
push it to main (`git push origin HEAD:main`), and wait for that push's CI run to finish green,
found by the pushed commit's full sha from `git rev-parse HEAD` (a short sha returns an empty list).
Another task may land meanwhile: a push refused because main moved goes back to the rebase. Red
CI it fixes forward once in a new commit; red again, or a push it can't land, comes back as a
failure. A red that main's run before the push had already, in the same job and test, isn't this
task's: the Lander leaves it and reports it (`RED-BEFORE`); the task's commits are on main, the
red goes through the Filer and the task ends **Partly done** on that issue.

It returns the SHA, the CI result and run URL, findings fixed and skipped, the final verdict per
criterion, and anything left unpushed.

## Other problems along the way

Agents keep running into things that are wrong but aren't this task: a bug next to the code they
change, a check that fails for an unrelated reason, a test that passes for the wrong reason, a
valid review finding outside the task's scope, a line in the spec the code contradicts. Each
agent's reply ends with a `FOUND:` line for them. They get the same treatment as the task itself —
an issue on the board, then this whole flow — but never inside this task.

- After every agent, copy each FOUND entry to the log as a `found:` line, word for word.
- Don't hand them to the next agent of this task, and don't let it fix them. The one exception:
  if a found problem is what stops this task, the task fails through **Failures**, and its failure
  comment names the problem's new issue.
- Problems of this machine — a missing tool, an expired login, a value missing from
  `sound-scout.toml` — are not found problems: they're failures or STOPs.

## 5. File found problems

On every way out that posts an outcome comment — done, held, skipped from Decision, failed,
blocked as **Partly done**; not the Gate's blocked — and only if the log has `found:` lines for
this task: before writing that comment, start the **Filer agent** with those lines. On a lost
claim there's no comment and no filing.

The Filer files them with the **new-task** skill (`<skill dir>/../new-task/SKILL.md`) in its agent
mode: it checks each problem against the code on main (read only), drops the ones that turn out
wrong or already fixed, merges duplicates, and looks for an open issue that already covers each
one. The rest become issues written the way every task on the board is — in Russian, with
acceptance criteria — plus a `work-task-found-by` marker and technical notes for the next
Executor. Each new issue goes on the board in Backlog. It doesn't change code.

Log one `filed:`, `exists:` or `dropped:` line per problem. The outcome comment's Found line links
the filed and the existing ones. A Filer that fails twice doesn't change the task's result: log
`filing failed: <error>`, the Found line lists the problems in a few words each as `not filed`,
and step 7 has nothing to work.

## 6. Done

A task with criteria that wait for another open issue isn't done, whatever landed: finish as
**Partly done**. One with criteria that need the owner isn't done either: finish as **Held**.

- Comment on the issue (`gh issue comment <url> --body-file <file>`). This is an **outcome
  comment**: its first line is `<!-- work-task-outcome: done -->` (`held`, `failed`, `skipped`
  and `blocked` for the other kinds). The backlog agent reads that marker to tell a task left on
  purpose from one whose agent died mid-way, so every final comment carries it — and nothing
  else does. The comment is read by the owner scanning the board, in Russian: it says what
  happened, links the evidence, and stops. Use this shape and nothing more:

  ```
  <!-- work-task-outcome: done -->
  **Готово.** <одно-два предложения: что теперь работает, с точки зрения пользователя инструмента — или почему менять ничего не пришлось>

  - <критерий, коротко> — `tests/<file>.py::<test name>`
  - <критерий> — без теста: <почему>; подтверждено `<файл или команда>`
  - <критерий> — не доказано: <почему>

  **Коммит:** [<sha7>](https://github.com/renidere/sound-scout-plugin/commit/<sha>) ([CI](<run url>))
  **Ревью:** <profile>, <n> замечаний: <n> исправлено, <n> пропущено (<почему, коротко>)
  **Агенты решили:** <по строке на выбор, который владелец захотел бы знать>
  **Нужен человек:** <ручной шаг>
  **Решение:** <comment url>
  **Найдено попутно:** #<n>, #<n> (новые) · #<n> (уже было)
  ```

  - A line whose answer would be "none" is left out. The first line and the criteria are always
    there. The commit line is too, except where nothing was committed.
  - One line per item. No paragraphs, no explaining how a test works.
  - A recovery run says so in the first line: `**Готово** (подхвачено после прерванной попытки).`
- `gh issue edit <url> --add-assignee @me`, then `gh issue close <url>` and
  `board.sh status <item> done`.
- Parent epic: `board.sh subissues <parent-url>`; if every sub-issue is now closed, close the epic
  and move its card (`board.sh item <parent-url>`) to Done.
- Remove the tree (`worktree.sh remove <number>`), release the claim, log the final line, and
  print the **result line** — the caller logs and counts it:

  ```
  work-task: done|failed|skipped|blocked|stop|held | <issue url> | <sha7|-> | <reason or ->
  ```

  Every way a task ends prints its result line in your own reply text, alone on a line of its
  own, as plain text: never in a code block, a bullet, bold or a sentence, and never echoed by a
  command. It is this task's, not the end of the turn: go on to **7**. Whatever ends the turn,
  its last message ends with the result line of every task this run handled.

### Held

Criteria that need the owner (**Scope and Decision**, **1. Executor**). What could be built has
landed on main; the rest waits for a person, so the issue stays open and the task ends `held`:

- `board.sh status <item> need-a-person` **before** the comment: no run picks Need a Person.
- The outcome comment, in the shape of **6** with its own first line:

  ```
  <!-- work-task-outcome: held -->
  **Ждёт владельца.** <одно-два предложения: что уже сделано и что станет возможным после шага владельца>

  - <критерий> — `tests/<file>.py::<test name>`
  - <критерий> — ждёт владельца: <что именно нужно — разрешение, данные, команда>

  **Коммит:** …
  **Ревью:** …
  **Нужен человек:** <точно, что сделать: какое разрешение дать, какой файл положить куда, какую команду выполнить>; затем вернуть карточку в Backlog — следующий запуск достроит остальное
  **Решение:** <comment url>
  **Найдено попутно:** …
  ```

- No close, no epic check. Remove the tree, release the claim, log `held | <sha7|-> | <the step>`,
  print the result line `held`, then go on to **7**.

The run after needs nothing of its own: once the owner has done their part and moved the card
back to Backlog, the queue offers the task again, the relevance check finds the landed parts and
builds the rest.

### Partly done

The Executor reported criteria as `waits for <issue url>`, or the Lander a `RED-BEFORE`: some of
the task can't be done until other issues, still open, are. What could be done has landed; the
rest waits. The task ends `blocked`, like the Gate's blocked step, so the queue takes it up again
once those issues are closed. The **blockers** are the issues the waiting criteria name, each as
its full issue URL.

- **Link them**: `board.sh blocked-by <url> <blocker url> ...` adds a GitHub "blocked by" link
  from this issue to each blocker that doesn't have one yet. Still failing after one retry: log
  `blocked-by failed: <error>` and go on — the outcome comment names the blockers too.
- `board.sh status <item> backlog` **before** the comment.
- The Found line goes in a comment of its own, posted just before the outcome comment and without
  an outcome marker: `**Найдено попутно:** …`. An unattended caller reads every issue a `blocked`
  outcome comment names as a blocker, so the outcome names no issue but the blockers — no `#<n>`
  anywhere in it. This task's own Decision comment is the one other link to an issue page.
- The outcome comment:

  ```
  <!-- work-task-outcome: blocked -->
  **Сделано частично**, ждёт https://github.com/renidere/sound-scout-plugin/issues/<n>. <одно предложение: что уже работает>

  - <критерий> — `tests/<file>.py::<test name>`
  - <критерий> — ждёт https://github.com/renidere/sound-scout-plugin/issues/<n>

  **Коммит:** …
  **Ревью:** …
  **Решение:** <comment url>
  ```

  Nothing landed (every criterion waits): the first line is `**Заблокировано.** Задача ждёт <url>;
  до него строить нечего.`, then the criteria lines, Decision and nothing else.
- Remove yourself as assignee if the log says `assigned=yes`, remove the tree, release the claim,
  log `blocked | <sha7|-> | waits for <blocker urls>`, print the result line
  `work-task: blocked | <url> | <sha7|-> | waits for <blocker urls>`, then go on to **7**.

## 7. Found problems

With `found problems: file only` in the args, end, the result line last in the reply: the filed
issues wait on the board, and the queue picks them up like any other task.

Otherwise, every issue the Filer opened for this task (its `filed:` lines; not `exists:`, and not
one a `fixed main first:` line names, worked before the Lander) now goes through this same flow,
right away, one at a time, in the order filed: the Skill tool with `work-task` and args
`<new issue url> — found by <this issue url>, depth <this depth + 1>`, followed by whatever else
this task's args said about the caller (`unattended backlog run, agent <k> of <N>`). Each
invocation runs from **Start** and ends with its own result lines, and its own step 7 for what it
finds in turn.

- **Found issues are worked down to depth 2.** An issue at depth 2 still files what it finds
  (step 5), but doesn't work it here. Without the limit, a chain of "found while fixing the
  found" never ends.
- **Before each issue**, check whether to go on. End step 7 at once when a `stop` result line was
  printed since this step 7 began, or the log shows the **third failure in a row** (final lines
  only; skips and blocks neither break nor extend the streak). Three in a row usually means the
  environment is broken.
- After a compaction: an issue with a `filed:` line for this task and no final line of its own is
  not finished. One with a `started` line later than its `filed:` line is still running — carry
  it on, never invoke it again. The rest are still to be worked.

With nothing filed, all of them worked, or step 7 ended early: end.

## Failures

Retry a failed step once, with a new agent that is given the error. If it fails again:

- A commit already pushed and green stays as it is; say so in the comment and the log.
- File found problems (step 5), if any.
- Move the card **before** the comment. Read `board.sh last-outcome <url>` first: when the newest
  outcome is `failed` too — the attempt before this one failed as well — to Need a Person
  (`board.sh status <item> need-a-person`), where a person looks into it and no run picks it.
  Otherwise back to Backlog (`board.sh status <item> backlog`): the queue offers it again.
- Comment the blocker on the issue — an outcome comment starting with
  `<!-- work-task-outcome: failed -->`, as short as at Done: `**Не удалось:** <шаг и что сломалось,
  одно предложение>`, the error in a code block of at most five lines — the command that failed
  and its output — `**Пробовали:** <одна строка>`, and the commit and found lines of **6** when
  they apply.
- Remove the tree with `worktree.sh remove <number>` — what isn't pushed stays on a `leftover/`
  branch, for a person looking into the failure — release the claim, log `failed`.
- Result line `failed`, then step 7: a found problem is no less real because this task failed.

A commit that can't be pushed (a conflict the Lander couldn't resolve, a push refused three
times) stays on the task branch: removing the tree keeps a branch with unpushed commits (`kept`).
Log it as `unpushed <sha>: <why>` and return `failed`: the next attempt at this task finds it in
**Prepare** (`ahead`) and recognises it from that log line as reviewed work, which a Repair agent
lands.
