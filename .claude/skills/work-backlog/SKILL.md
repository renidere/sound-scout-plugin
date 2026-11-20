---
name: work-backlog
description: The Sound Scout backlog agent — runs unattended for hours and works through the Backlog column of the board github.com/users/renidere/projects/1, oldest first, with as many tasks at once as it is told (one by default), each handed to the work-task skill in a background agent of its own (its own git worktree cut from origin/main, Executor, Verifier, revmux review, push to main with green CI, close and Done), then logging and summarising the run in backlog-run.md. Use whenever someone in the sound-scout-plugin checkout asks to start the backlog agent, run or clear the backlog, work through the board — "start the backlog agent", "работай бэклог", "разбери бэклог", "run the backlog with 2 agents", or the same in any language — even without naming the skill. For one specific issue use work-task.
---

# Work the Sound Scout backlog

You run unattended: nobody is at the computer, so nobody can answer a question, approve a plan or
click a button. Never ask questions, never wait for confirmation, never enter plan mode. Keep going
until no task qualifies.

Each task is carried out by the **work-task** skill (`<skill dir>/../work-task/SKILL.md`) in a
**task agent**: a subagent of this session, started in the background. work-task orchestrates the
task's agents (Decision, Executor, Verifier, revmux, Lander, Filer) in a git worktree of its own,
on a branch cut from `origin/main`; it lands the branch on main, comments, closes the issue and
moves the card. Your part: start task agents, keep count, notice when the run should stop, and
write the summary.

**How many task agents**: the number the args give — "2", "2 agents", "два параллельно", in any
language; none given: 1. Each task agent takes one task. Task agents never meet in the files:
every task has its own tree, and the main checkout is never touched. Two tasks meet only on main,
where the later Lander rebases onto the earlier one's push; the claim on each issue keeps two
task agents off the same task. This machine has no discrete GPU and the audio stages are
CPU-heavy: more than 2 agents at once is rarely useful.

**The log** is `$WS/backlog-run.md` — your memory: after a compaction, re-read this file and the
current run's section of the log. The task agents write to it too, so add your lines the way they
do — one shell append per line (`printf '%s\n' '<line>' >> "<the log>"`), never by rewriting the
file. Two runs in one checkout would share it: start one at a time.

## Start

`<scripts>` is the absolute path of `<skill dir>/../work-task/scripts`. `$WS` is the main
checkout — what `bash <scripts>/workspace.sh` prints.

1. `bash <scripts>/preflight.sh` — the machine only. `STOP:` lines: print the first one as your
   one-line explanation and stop — every task would fail the same way.
2. Append a run header to the log (create the file if needed):
   `## Run <YYYY-MM-DD HH:MM> — profile <revmux profile> — <N> agents` — the profile `default`
   when `codex login status` succeeds, `claude-only` otherwise.
3. **Leftover trees**: `bash <scripts>/worktree.sh list` prints, per key with a tree left in
   `$WS/.work-task/`, the key and the issue URL. A tree whose issue is closed finished its work
   and stopped before cleaning up: `worktree.sh remove <key>`, and log what it printed (`saved`
   leftover branches, `moved` folders and `kept` branches go in the summary). Leave every other
   one: its work may be running in another session, and a task's next attempt picks its tree up.
4. **Hung tasks** — tasks an earlier run left In Progress without finishing. Their cards are In
   Progress, so the queue never offers them again; left alone they'd hang there for ever. Deal
   with them before the queue.

## Hung tasks

`bash <scripts>/board.sh in-progress` lists every task in In Progress with a verdict, as TSV
`item-id  issue-url  verdict  claim-id  claim-minutes  title`:

| verdict | what it means | what to do |
|---|---|---|
| `closed` | the issue is closed but the card wasn't moved | `board.sh status <item> done`; if a claim is listed, `claim.sh release <url> <claim-id>`; log `recovered \| closed already, card moved to Done` |
| `abandoned` | only stale claims and no outcome after them — the agent died mid-task | work it again (below) |
| `working` | a live claim | leave it — a crashed session and one still running look the same; once nobody refreshes the claim for `CLAIM_STALE_MINUTES` it turns `abandoned` |
| `left-failed` | the agent finished with a failure comment but the card stayed | `board.sh status <item> backlog`; if a claim is listed, release it |
| `left-skipped`, `left-held` | the agent finished and left it for a person, but the card stayed | `board.sh status <item> need-a-person`; if a claim is listed, release it |
| `unknown` | no claim, no outcome — a person moved the card, or a `blocked` last word | leave it: it may be someone's work |

**Work it again** = start a task agent for it (see **Task agents**) with the args
`<issue-url> — recovery of a hung task, unattended backlog run`. work-task claims it (a stale claim
gives way), saves any half-done changes the dead attempt left in its tree, and starts over;
whatever the dead attempt already pushed is on main, and the Executor's relevance check finds it.

Hung tasks go first: they fill the free slots before the queue does. Log one line per hung task
handled, and keep the counts for the summary.

## Task agents

A task agent runs in a slot numbered 1 to N; a slot is free when it has no `dispatched` line in
this run without an `agent <k> returned` line after it. To start one for a task, take the lowest
free slot `k`, log

```
- <YYYY-MM-DD HH:MM> | <issue url> | dispatched | agent <k>
```

and launch it with the Agent tool: a general-purpose subagent, description `#<number>`, in the
background, with the prompt

```
Use the work-task skill (Skill tool) with args: <issue url> — <the task's args>, agent <k> of <N>
Run it from Start to its end, its step 7 included, without asking anything, and return its result lines.
```

Its notification wakes you when it returns: log `- <HH:MM> agent <k> returned`. If its task has no
final line after its `dispatched` line, the agent stopped mid-task — usually an early report while
one of its step agents still ran, sometimes a usage limit. **Resume it at once** with SendMessage
(its transcript, claim and tree are intact): tell it to continue where it stopped, run its step
agents in the foreground, and stay until the task's final step. Its slot stays taken meanwhile. Log
`- <HH:MM> agent <k> resumed: <one line>`. Only when the resume fails (the agent is gone, or a usage
limit stops it again) log
`- <YYYY-MM-DD HH:MM> | <issue url> | failed | - | agent <k> ended without a result: <its last words, one line>`
and leave the issue as it is: its claim goes stale, and a later run recovers it as `abandoned`.

The task agents live in this session: closing it ends them mid-task, and the next run recovers
their tasks as hung ones.

## The loop

1. **Fill the free slots**, unless the run is stopping (step 3): hung tasks to work again first,
   then the queue. `bash <scripts>/board.sh queue` lists every qualifying task, oldest first, as
   TSV `item-id  issue-url  parent-url|-  title`: open issues in Backlog without sub-issues (those
   are epics) and without an open "blocked by" link. Take the first one that has no task line
   (`| <url> |`) in this run's section of the log — a task this run already dispatched, handled,
   blocked or failed isn't retried until the next run. Start a task agent for it with the args
   `<issue-url> — unattended backlog run`. Read the queue afresh for every free slot: the board
   moves while task agents run.
2. **Wait** for a task agent to return — its notification wakes you; ending your turn while task
   agents run is how you wait. None running and none to start: go to **Finish**. Never end your
   turn with no task agent running before Finish.
3. **On each return**, read the final lines the log gained since the last one you acted on, and
   act on them as one task each:
   - `done`, `held`, `skipped`, `blocked`: nothing to do (a `held` task waits in Need a Person for
     the owner's step; a `skipped` may mean another agent holds the task — the lock doing its
     job).
   - `failed`: nothing to do — unless the log now shows the **third failure in a row** (count from
     the log, so the count survives a compaction; skips and blocks don't break or extend the
     streak). Three in a row usually means the environment is broken — an expired login, a full
     disk, a tool gone — and every further task would fail too: the run is **stopping**.
   - `stop`: the run is **stopping** — work-task hit something only a person can fix.

   Stopping: start no more task agents, and let the running ones finish their tasks. Then back
   to 1. Between tasks you need nothing of a finished task's details; don't carry them.

## Finish

Append a summary to the log under the run header:

- hung tasks found at the start: recovered (with their results), being worked elsewhere, and
  left for a person, with issue links;
- tasks done, held, failed, skipped and blocked, with issue links — and which of them were
  problems found along the way (`found by` in their args), with the task that found them;
- task agents that ended without a result, with their task;
- issues filed for found problems and left on the board;
- decisions made, with links to the `Decision:` comments;
- leftover branches (`saved`), folders moved aside (`moved`), rescue branches and task branches
  kept with unpushed commits, so a person can find their work;
- what waits for the owner (`held` and Need a Person), per issue, with the step named;
- why the run ended: no tasks left, 3 failures in a row, a stop (with its reason), or a preflight
  STOP.

Then print a short message in Russian, in plain words: how many tasks were done, how many failed
or were skipped or are waiting on other tasks or on the owner, how many hung tasks were picked up
again, how many new tasks were filed, what the owner has to do by hand, and that the details are
in `backlog-run.md`. Then stop.
