"""Chat binding of MCP jobs (decision 0025, Decision in #140): the `post_tool_call` hook.

`speech_job_submit` takes no chat argument, so an MCP job is created unbound. After the call the
hook reads the job's `job_id` from the tool result, which `jobs.submit` wrote, and the chat from
the Hermes session context of the turn (`HERMES_SESSION_PLATFORM == "local"`,
`HERMES_SESSION_CHAT_ID`), and runs `sound-scout jobs bind` by argv. It never reads `args`: what the
model put into the call never becomes the binding. The chat id, not the hook's `session_id`
argument: that one names the agent's physical segment and changes after a compression, while the
local chat id is stable and is what `deliver_to_local_chat` takes.

Hermes v0.23.3 (`caafe5389a`), read only: `model_tools._emit_post_tool_call_hook` calls the hook
with `tool_name`, `args`, `result`, `status` and the call ids as keywords after every call, its
return ignored; `status` is `"error"` for a result with an `error` key, and a refused MCP call
(`isError`) is a `tool_error`. An MCP result without `structuredContent` and `_meta` reaches the
hook as `json.dumps({"result": <text>})` (`tools/mcp_tool_handlers.py`). The hook runs in
`contextvars.copy_context()` of the turn on a bounded worker (30 s by default; a timeout skips
it), so `gateway.session_context.get_session_env` sees the turn's chat. The tool name is
`mcp__<server>__speech_job_submit` (`tools/mcp_tool_schema.py`); a name Hermes shortens by a hash
(longer than 64 characters) does not match.

Any failure only leaves the job unbound: its result is then read with `speech_job_status` and
`speech_job_result`. A log line names the outcome only, never the chat, the arguments or the
result.
"""

import json
import logging
import re

from . import delivery

MCP_PREFIX = "mcp__"
SUBMIT_SUFFIX = "__speech_job_submit"
LOCAL_PLATFORM = "local"
CHAT_ID_MAX_CHARS = 128
_JOB_ID = re.compile(r"[0-9a-f]{32}")

_log = logging.getLogger("sound-scout.binding")


def is_submit(tool_name: object) -> bool:
    return (
        isinstance(tool_name, str)
        and tool_name.startswith(MCP_PREFIX)
        and tool_name.endswith(SUBMIT_SUFFIX)
    )


def chat_of(session_env) -> str | None:
    """The local chat of the turn, or None: another platform, no chat, a malformed id."""
    if session_env("HERMES_SESSION_PLATFORM") != LOCAL_PLATFORM:
        return None
    chat_id = session_env("HERMES_SESSION_CHAT_ID")
    if not isinstance(chat_id, str) or not 1 <= len(chat_id) <= CHAT_ID_MAX_CHARS:
        return None
    return chat_id if chat_id.isprintable() else None


def answer_of(result: object) -> dict | None:
    """The Sound Scout answer of an MCP tool result as Hermes hands it to the hook, or None."""
    try:
        wrapped = json.loads(result) if isinstance(result, str) else None
        text = wrapped.get("result") if isinstance(wrapped, dict) else None
        answer = json.loads(text) if isinstance(text, str) else None
    except ValueError:
        return None
    return answer if isinstance(answer, dict) else None


def is_job_id(value: object) -> bool:
    return isinstance(value, str) and _JOB_ID.fullmatch(value) is not None


def job_id_of(result: object) -> str | None:
    """The `job_id` of a `speech_job_submit` answer as Hermes hands it to the hook, or None."""
    job_id = (answer_of(result) or {}).get("job_id")
    return job_id if is_job_id(job_id) else None


def bind_after_submit(
    tool_name: object, result: object, status: object, *, session_env, profile, cli
) -> str | None:
    """Bind the job a successful submit queued to the turn's local chat; the outcome, or None
    for any other call. Every dependency is given."""
    if not is_submit(tool_name) or status != "ok":
        return None
    chat_id = chat_of(session_env)
    if chat_id is None:
        return "no_chat"
    job_id = job_id_of(result)
    if job_id is None:
        return "no_job_id"
    # `--opt=value`: a chat id that starts with `-` is never read as an option.
    answer = cli(["bind", f"--session-id={chat_id}", f"--profile={profile()}", job_id])
    if answer is None:
        return "cli_failed"
    return answer.get("error", "bound")


def post_tool_call(tool_name="", args=None, result=None, status="", **_) -> None:
    """The Hermes hook; `args` (the model's arguments) is never read."""
    try:
        outcome = bind_after_submit(
            tool_name,
            result,
            status,
            session_env=hermes_session_env,
            profile=delivery.hermes_profile,
            cli=delivery.run_cli,
        )
    except Exception as exc:
        outcome = type(exc).__name__
    if outcome is not None:
        _log.info("sound-scout binding: %s", outcome)


def hermes_session_env(name: str) -> str:
    from gateway.session_context import get_session_env

    return get_session_env(name)
