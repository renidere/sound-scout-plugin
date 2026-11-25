"""Seen mark of a job attempt (decision 0030, Decision in #190): the second `post_tool_call` hook.

When a turn of the bound chat reads the result of an ended attempt through `speech_job_result`,
the agent answers with it in that turn, and a note of the same attempt would repeat the answer.
After a successful call the hook reads `job_id`, `job_state` and `attempt` from the tool result
(the compact answer of `jobs.result`) and the chat from the Hermes session context of the turn,
as `binding.py` does (decision 0025), and runs `sound-scout jobs seen` by argv. It never reads
`args`. Sound Scout checks that the chat is the job's bound chat and that the attempt is the job's
last one; the delivery loop then ends that attempt's delivery `seen_in_chat` without a send.

Hermes v0.23.3 (`caafe5389a`), read only: the hook gets the keywords of `binding.py`; `status` is
`"ok"` only for a successful call, and a refusal of Sound Scout (`no_result` among them) is a
`tool_error`. The tool name is `mcp__<server>__speech_job_result`.

Any failure (no chat, a refusal, no answer of the CLI, a hook Hermes skipped on its timeout)
leaves the attempt unseen: the chat gets its one note, as without this hook. A log line names
the outcome only, never the chat, the arguments or the result.
"""

import logging

from . import binding, delivery

RESULT_SUFFIX = "__speech_job_result"
TERMINAL_STATES = ("finished", "failed", "cancelled")

_log = logging.getLogger("sound-scout.seen")


def is_result(tool_name: object) -> bool:
    return (
        isinstance(tool_name, str)
        and tool_name.startswith(binding.MCP_PREFIX)
        and tool_name.endswith(RESULT_SUFFIX)
    )


def attempt_of(result: object) -> tuple[str, int] | None:
    """`(job_id, attempt)` of a `speech_job_result` answer of an ended job as Hermes hands it to
    the hook, or None."""
    answer = binding.answer_of(result)
    if answer is None or answer.get("job_state") not in TERMINAL_STATES:
        return None
    job_id, attempt = answer.get("job_id"), answer.get("attempt")
    if not binding.is_job_id(job_id):
        return None
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        return None
    return job_id, attempt


def mark_after_result(tool_name: object, result: object, status: object, *, session_env, cli):
    """Mark the attempt a successful `speech_job_result` read as seen in the turn's local chat;
    the outcome, or None for any other call. Every dependency is given."""
    if not is_result(tool_name) or status != "ok":
        return None
    chat_id = binding.chat_of(session_env)
    if chat_id is None:
        return "no_chat"
    found = attempt_of(result)
    if found is None:
        return "not_terminal"
    job_id, attempt = found
    # `--opt=value`: a chat id that starts with `-` is never read as an option.
    answer = cli(["seen", f"--session-id={chat_id}", f"--attempt={attempt}", job_id])
    if answer is None:
        return "cli_failed"
    return answer.get("error", "seen")


def post_tool_call(tool_name="", args=None, result=None, status="", **_) -> None:
    """The Hermes hook; `args` (the model's arguments) is never read."""
    try:
        outcome = mark_after_result(
            tool_name,
            result,
            status,
            session_env=binding.hermes_session_env,
            cli=delivery.run_cli,
        )
    except Exception as exc:
        outcome = type(exc).__name__
    if outcome is not None:
        _log.info("sound-scout seen: %s", outcome)
