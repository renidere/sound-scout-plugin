"""Delivery loop of the Sound Scout plugin: one final note per ended attempt of a job to its
bound chat, unless a turn of that chat read the result (decision 0013 § 3.7, Decision in #102; a
retried attempt's own note, Decision in #187; the seen mark and the turn wait, decision 0030).
Each note has its own tag: `sound-scout-job:<job_id>` for the first one,
`sound-scout-job:<job_id>-a<n>` after a retry; the history check matches the whole tag.

Runs on the Hermes interpreter (Python >= 3.11) and imports only the stdlib; Hermes modules are
imported inside the edge functions. Sound Scout keeps the delivery state: the loop calls
`sound-scout jobs deliver-next` (the next note, its `sending` marker already written) and
`sound-scout jobs deliver-end` (how the send ended). `register(ctx)` starts the loop in every
Hermes process that loads the package, but only the process that owns the session authority
(`owner_for_home`, decision 0017: `hermes gateway run`) does anything: elsewhere each tick ends
`idle_no_owner` before any call.

Hermes does not deduplicate notes, so there is no blind repeat: one `send` per marker, the
address is always the `session_id` that `deliver-next` names. A timeout or an exception of
`send` (the note may still arrive: the coroutine goes on after the timeout) is settled by the job
tag in the chat history: found is `delivered_by_history`, not found is `unconfirmed`. Any other
error of the chat (unknown or deleted chat, admission refused, no route) is `failed`. "No gateway
in this process owns its session" returns before sending and is `released`: the job waits for
the owner.

`redeliver` (the operator's `sound-scout jobs redeliver`, Decision in #122) first looks for the
job tag in the whole history of the chat: found is `delivered_by_history` without a send; not
found is one `send`, settled as above; a history that cannot be read, or one longer than
`HISTORY_SCAN` messages without the tag, is `unconfirmed` without a send, because the tag's
absence is not proven. The history is read with `include_ancestors=True`: after a compression the
first note may sit in a parent segment of the lineage; and with `include_compacted=True`: the
note may sit in a row compaction archived (`active = 0, compacted = 1`). Hermes v0.23.1
`get_messages` takes `latest` together with `include_ancestors` and `include_compacted` (only
`after_id` is refused with either) and returns the newest `limit` messages of the merged lineage
in chronological order; read-only check of `hermes_state_messages.py:1385`.

The tag counts only as a line of its own outside a tool result: the note ends with it, while the
job status the agent reads shows it inside JSON. Hermes writes the note as a `user` row
(`hermes_state_detached_results._append_detached_result`), so user and assistant rows are read; a
user or the agent quoting the note is taken as delivered, which misses a redelivery rather than
sending a second note.

No note for an attempt the chat saw, and none in the middle of a turn (decision 0030, Decision
in #190). A note not yet due comes from `deliver-next` as `probe`: the loop asks `chat_busy`
whether the chat has a pending admission in the session authority and calls `deliver-next` again
with the chat as `--idle-chat` (send now) or `--busy-chat` (the note waits; look further), up to
`PROBES_PER_TICK` times a tick. An error of the check counts as busy; Sound Scout sends the note
anyway `TURN_WAIT_SECONDS` after the attempt ended. An attempt the `seen` hook marked ends
`seen_in_chat` inside `deliver-next`, without a send. The turn signal is the admission queue of
the chat, not a hook: a hook Hermes skips on its timeout or a turn that raises would leave a
wrong state, and a queued turn ahead of the note shows in the queue only. Hermes v0.23.3
(`caafe5389a`), read only: `gateway/session_authority.py:349-356` (`execution_state` from the
pending admissions), `:277-284` (`logical_owner`, the FIFO identity),
`persistence/sqlite/authority/admissions.py:175-178` (`admissions.list`, pending only),
`gateway/run_turn_persistence.py:484` (the same "is anything in the chat FIFO" check),
`gateway/session_cron.py:65-80` (work on the owner loop from another thread).

The Hermes names follow the Decision in #102, read from Hermes v0.23.1 (`3b42157858`):
`hermes_constants.get_hermes_home`, `hermes_constants.profile_name_for_home`,
`gateway.session_cron.owner_for_home` (it returns `(authority, loop)` or None),
`gateway.session_cron.deliver_to_local_chat`, `_physical_target(authority, chat_id)` and
`authority.db.get_messages`.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import uuid

THREAD_NAME = "sound-scout-delivery"
POLL_SECONDS = 10
CLI_TIMEOUT_SECONDS = 30
CLI_MAX_OUTPUT_BYTES = 64 * 1024
HISTORY_SCAN = 2000
PROBES_PER_TICK = 16
PROBE_TIMEOUT_SECONDS = 10
BIN_ENV = "SOUND_SCOUT_BIN"
CONFIG_ENV = "SOUND_SCOUT_CONFIG"
PASSED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
NO_OWNER = "no gateway in this process owns its session"
TIMED_OUT = "delivery not confirmed: TimeoutError"
ORPHAN_ERROR = "marker left by a stopped delivery loop; tag not found in chat history"
REDELIVERY_UNREAD = "redelivery not sent: chat history could not be read in full"
# Refusals of deliver-end after which the outcome can never be recorded: drop it.
FINAL_REFUSALS = ("delivery_not_sending", "job_not_found")

_log = logging.getLogger("sound-scout.delivery")
_not_configured_logged = False


def run_cli(args: list[str]) -> dict | None:
    """`sound-scout jobs <args> --json`: its answer, `{"error": <code>}` for a refusal, or None
    when it did not answer (not configured, timed out, crashed, unreadable output)."""
    global _not_configured_logged
    binary, config = os.environ.get(BIN_ENV), os.environ.get(CONFIG_ENV)
    if not binary or not config:
        if not _not_configured_logged:
            _log.warning("sound-scout delivery: not_configured (%s, %s)", BIN_ENV, CONFIG_ENV)
            _not_configured_logged = True
        return None
    env = {name: os.environ[name] for name in PASSED_ENV if name in os.environ}
    env[CONFIG_ENV] = config
    try:
        done = subprocess.run(
            [binary, "jobs", *args, "--json"],
            capture_output=True,
            timeout=CLI_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        _log.warning("sound-scout delivery: cli %s: %s", args[0], type(exc).__name__)
        return None
    if done.returncode != 0:
        said = done.stderr[:CLI_MAX_OUTPUT_BYTES].decode("utf-8", "replace")
        parts = said.split(": ", 2)
        code = parts[1] if len(parts) == 3 else f"exit {done.returncode}"
        _log.warning("sound-scout delivery: cli %s refused: %s", args[0], code)
        return {"error": code}
    if len(done.stdout) > CLI_MAX_OUTPUT_BYTES:
        return None
    try:
        answer = json.loads(done.stdout)
    except ValueError:
        return None
    return answer if isinstance(answer, dict) else None


def tick(*, owner, profile, loop_id: str, cli, send, history_has, chat_busy, state: dict) -> str:
    """One step of the loop; every dependency is given. Returns what it did."""
    if not owner():
        return "idle_no_owner"
    if "pending" in state:
        job_id, outcome, error = state["pending"]
        if not _record(cli, job_id, outcome, error):
            return "end_pending"
        del state["pending"]
        return "end_recorded"
    idle: set[str] = set()
    busy: set[str] = set()
    for _ in range(PROBES_PER_TICK):
        args = ["deliver-next", f"--profile={profile()}", f"--loop={loop_id}"]
        args += [f"--idle-chat={chat}" for chat in sorted(idle)]
        args += [f"--busy-chat={chat}" for chat in sorted(busy)]
        answer = cli(args)
        if answer is None or "error" in answer:
            return "cli_failed"
        if answer["action"] != "probe":
            break
        chat_id = answer["session_id"]
        (busy if _chat_busy(chat_busy, chat_id) else idle).add(chat_id)
    else:
        return "waiting_turn"
    if answer["action"] not in ("check", "send", "redeliver"):
        return "waiting_turn" if busy else "idle"
    job_id = answer["job_id"]
    # The marker is written: whatever happens now, the outcome reaches deliver-end or pending.
    try:
        outcome, error = _settle(answer, send, history_has)
    except Exception as exc:
        outcome, error = "unconfirmed", f"delivery loop raised {type(exc).__name__}"
    _log.info("sound-scout delivery: job %s: %s", job_id, outcome)
    try:
        recorded = _record(cli, job_id, outcome, error)
    except Exception as exc:
        _log.warning("sound-scout delivery: deliver-end raised %s", type(exc).__name__)
        recorded = False
    if not recorded:
        state["pending"] = (job_id, outcome, error)
    return outcome


def _chat_busy(chat_busy, chat_id: str) -> bool:
    """Whether a turn runs or waits in the chat; a check that fails counts as busy."""
    try:
        return bool(chat_busy(chat_id))
    except Exception as exc:
        _log.warning("sound-scout delivery: turn check failed: %s", type(exc).__name__)
        return True


def _settle(answer: dict, send, history_has) -> tuple[str, str | None]:
    chat_id, tag = answer["session_id"], answer["tag"]
    if answer["action"] == "check":
        if _history_has(history_has, chat_id, tag):
            return "delivered_by_history", None
        return "unconfirmed", ORPHAN_ERROR
    if answer["action"] == "redeliver":
        seen = _history_state(history_has, chat_id, tag)
        if seen is True:
            return "delivered_by_history", None
        if seen is None:
            return "unconfirmed", REDELIVERY_UNREAD
    return _send_once(send, history_has, chat_id, answer["text"], tag)


def _send_once(send, history_has, chat_id: str, text: str, tag: str) -> tuple[str, str | None]:
    try:
        said = send(chat_id, text)
    except Exception as exc:
        said = f"send raised {type(exc).__name__}"
        unknown = True
    else:
        if said is None:
            return "delivered", None
        unknown = TIMED_OUT in said
    if NO_OWNER in said:
        return "released", None
    if not unknown:
        return "failed", said
    if _history_has(history_has, chat_id, tag):
        return "delivered_by_history", None
    return "unconfirmed", f"{said}; tag not found in chat history"


def _history_has(history_has, chat_id: str, tag: str) -> bool:
    return _history_state(history_has, chat_id, tag) is True


def _history_state(history_has, chat_id: str, tag: str) -> bool | None:
    """Whether the tag is in the chat history; None when the history could not be read."""
    try:
        return bool(history_has(chat_id, tag))
    except Exception as exc:
        _log.warning("sound-scout delivery: history check failed: %s", type(exc).__name__)
        return None


def _record(cli, job_id: str, outcome: str, error: str | None) -> bool:
    """Whether `deliver-end` took the outcome (or can never take it)."""
    args = ["deliver-end", f"--outcome={outcome}"]
    if error:
        args.append("--error=" + error.replace("\0", ""))
    answer = cli([*args, job_id])
    if answer is None:
        return False
    return "error" not in answer or answer["error"] in FINAL_REFUSALS


def _home():
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def hermes_owner() -> bool:
    try:
        from gateway.session_cron import owner_for_home
    except ImportError:
        return False
    return owner_for_home(_home()) is not None


def hermes_profile() -> str:
    from hermes_constants import get_hermes_home, profile_name_for_home

    return profile_name_for_home(get_hermes_home()) or "default"


def hermes_send(chat_id: str, text: str) -> str | None:
    from gateway.session_cron import deliver_to_local_chat

    return deliver_to_local_chat(_home(), chat_id, text)


def hermes_chat_busy(chat_id: str) -> bool:
    """Whether the chat has a pending admission (`queued`, `started` or `unknown`): a turn runs or
    waits ahead of any note admitted now. Read on the owner loop, as `deliver_to_local_chat`
    works; see the module docstring for the Hermes lines.

    Raises when it cannot answer: `RuntimeError` without the owner process, `TimeoutError` after
    `PROBE_TIMEOUT_SECONDS`.
    """
    from gateway.session_cron import owner_for_home
    from gateway.session_persistence import authority_persistence

    owner = owner_for_home(_home())
    if owner is None:
        raise RuntimeError("no gateway in this process owns the session authority")
    authority, loop = owner

    async def pending() -> bool:
        admissions = authority_persistence(authority).admissions
        return bool(admissions.list(session_id=authority.logical_owner(chat_id)))

    future = asyncio.run_coroutine_threadsafe(pending(), loop)
    try:
        return future.result(timeout=PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        raise


def hermes_history_has(chat_id: str, tag: str) -> bool:
    """Whether the tag is in the newest `HISTORY_SCAN` messages of the chat's whole lineage.

    Raises when it cannot answer: `RuntimeError` without the owner process, `LookupError` when
    the scan read `HISTORY_SCAN` messages without the tag (older ones were not read).
    """
    from gateway.session_cron import owner_for_home

    owner = owner_for_home(_home())
    if owner is None:
        raise RuntimeError("no gateway in this process owns the session authority")
    authority = owner[0]
    try:
        from gateway.session_viewer_close import _physical_target
    except ImportError:
        target = chat_id
    else:
        target = _physical_target(authority, chat_id)
    messages = authority.db.get_messages(
        target, latest=True, limit=HISTORY_SCAN, include_ancestors=True, include_compacted=True
    )
    if any(_is_note(message, tag) for message in messages):
        return True
    if len(messages) >= HISTORY_SCAN:
        raise LookupError("history longer than the scan")
    return False


def _is_note(message, tag: str) -> bool:
    """Whether the message carries the note's tag line. A tool result (the job status shows the
    tag inside JSON) is never a note; content that is not text is matched by the whole tag, so a
    note is never missed. Whole means exact: `sound-scout-job:<job_id>` of a job's first note
    never matches the tag `sound-scout-job:<job_id>-a<n>` of a later attempt's note."""
    if _field(message, "role") == "tool":
        return False
    content = _field(message, "content")
    if not isinstance(content, str):
        return re.search(rf"{re.escape(tag)}(?![\w-])", str(content or "")) is not None
    return any(line.strip() == tag for line in content.splitlines())


def _field(message, name: str):
    return message.get(name) if isinstance(message, dict) else getattr(message, name, None)


def start(
    *,
    send=hermes_send,
    history_has=hermes_history_has,
    chat_busy=hermes_chat_busy,
    owner=hermes_owner,
    profile=hermes_profile,
    cli=run_cli,
    stop: threading.Event | None = None,
    poll_seconds: float = POLL_SECONDS,
) -> threading.Thread | None:
    """Start the loop thread once per process; None without Hermes or when it already runs.

    The name check survives a second import of the package (`discover_plugins(force=True)`).
    """
    try:
        import hermes_constants  # noqa: F401
    except ImportError:
        return None
    if any(thread.name == THREAD_NAME for thread in threading.enumerate()):
        return None
    loop = {
        "owner": owner,
        "profile": profile,
        "loop_id": uuid.uuid4().hex,
        "cli": cli,
        "send": send,
        "history_has": history_has,
        "chat_busy": chat_busy,
        "state": {},
    }
    thread = threading.Thread(
        target=_run,
        args=(loop, stop or threading.Event(), poll_seconds),
        name=THREAD_NAME,
        daemon=True,
    )
    thread.start()
    return thread


def _run(loop: dict, stop: threading.Event, poll_seconds: float) -> None:
    while not stop.is_set():
        try:
            tick(**loop)
        except Exception as exc:
            _log.warning("sound-scout delivery: tick failed: %s", type(exc).__name__)
        stop.wait(poll_seconds)
