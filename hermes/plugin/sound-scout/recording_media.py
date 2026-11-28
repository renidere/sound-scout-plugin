"""The sent recording in the chat (decision 0039, #216): the `pre_llm_call` and
`transform_llm_output` hooks.

In the chat of Hermes Desktop 0.23.3 a WAV plays only from an assistant message: Desktop renders a
`MEDIA:<absolute path>.wav` line of the reply as its stock `<audio controls>` player
(`apps/desktop/src/lib/chat-messages/parts.ts` `renderMediaTags`,
`components/assistant-ui/markdown-text.tsx` `MediaPlaybackAttachment`). The `file.attach`
attachment of the user message is an inactive `@file:` chip, the delivery note is a user row, and
`send_document` of the `Local` platform sends nothing. So `pre_llm_call` remembers the recording
`sound-scout-<12 hex>.wav` of the `@file:` reference in the user message of the turn, and
`transform_llm_output` appends one `MEDIA:<hermes home>/attachments/<name>` line to the reply of
that turn. The reply is stored with the line, so the player stays under the recording message after
later messages and after a Desktop restart. The model takes no part.

Hermes v0.23.3 (`caafe5389a`, https://github.com/NousResearch/hermes-agent), read only:
`agent/turn_context.py` `_collect_pre_llm_call_context` calls `pre_llm_call` with `session_id`,
`turn_id`, `user_message` (the stored text of the turn, its leading `@file:` block included) and
more keywords; `agent/turn_finalizer.py` `apply_llm_output_transform` calls `transform_llm_output`
once per turn with `response_text`, `session_id`, `turn_id`, `model` and `platform` before the first
write of the assistant row, never for an empty reply, and the first non-empty string wins. The key
is `turn_id` alone: it holds a uuid, and a compression inside the turn changes `agent.session_id`
between the two hooks (`agent/conversation_compression.py`). Only a turn of a local (Desktop) chat
is remembered: on a messenger the gateway would send the file there (R09).

The name comes from the reference, the directory never does: the path is always
`<get_hermes_home()>/attachments/<name>`, a regular file of 1 byte or more, not a symlink. Any
failure leaves the reply unchanged. A log line names the outcome only, never a path, the chat or a
text.
"""

import collections
import logging
import os
import re
import stat
import threading

from . import binding

RECORDING_REF = re.compile(
    r"@file:[\"'`]?(?:\S*/)?attachments/(sound-scout-[0-9a-f]{12}\.wav)[\"'`]?(?=\s|$)"
)
ATTACHMENTS_DIR = "attachments"
PENDING_MAX = 32
# Characters that would end or break the `MEDIA:` line rule of Desktop (`MEDIA_LINE_RE`).
_UNSAFE_PATH = re.compile(r"[\s`\"']")

_pending: collections.OrderedDict[str, list[str]] = collections.OrderedDict()
_lock = threading.Lock()
_log = logging.getLogger("sound-scout.recording_media")


def _text(user_message: object) -> str:
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):
        return "\n".join(
            part["text"]
            for part in user_message
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _recording_path(name: str) -> str | None:
    """`<hermes home>/attachments/<name>` when it is a non-empty regular file, else None."""
    from hermes_constants import get_hermes_home

    path = os.path.join(os.fspath(get_hermes_home()), ATTACHMENTS_DIR, name)
    if not os.path.isabs(path) or _UNSAFE_PATH.search(path):
        return None
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return path if stat.S_ISREG(info.st_mode) and info.st_size > 0 else None


def remember(turn_id: object, user_message: object, *, platform: object) -> str | None:
    """Remember the recordings of a turn's user message; the outcome, or None for a turn
    without a recording reference."""
    if not isinstance(turn_id, str) or not turn_id or platform != binding.LOCAL_PLATFORM:
        return None
    names = list(dict.fromkeys(m.group(1) for m in RECORDING_REF.finditer(_text(user_message))))
    if not names:
        return None
    paths = [path for path in map(_recording_path, names) if path is not None]
    if not paths:
        return "no recording"
    with _lock:
        _pending[turn_id] = paths
        _pending.move_to_end(turn_id)
        while len(_pending) > PENDING_MAX:
            _pending.popitem(last=False)
    return "recording remembered"


def media_reply(turn_id: object, response_text: object) -> tuple[str | None, str | None]:
    """`(new reply, outcome)` for the reply of a turn; `(None, None)` for a turn without a
    remembered recording."""
    with _lock:
        paths = _pending.pop(turn_id, None) if isinstance(turn_id, str) else None
    if paths is None:
        return None, None
    if not isinstance(response_text, str) or not response_text.strip():
        return None, "empty reply"
    present = {line.strip() for line in response_text.splitlines()}
    missing = [path for path in paths if f"MEDIA:{path}" not in present]
    if not missing:
        return None, "media line present"
    lines = "\n".join(f"MEDIA:{path}" for path in missing)
    return f"{response_text.rstrip()}\n\n{lines}", "media line added"


def pre_llm_call(*, session_id="", turn_id="", user_message=None, **_) -> None:
    """The Hermes hook; it never adds context to the turn."""
    try:
        outcome = remember(
            turn_id,
            user_message,
            platform=binding.hermes_session_env("HERMES_SESSION_PLATFORM"),
        )
    except Exception as exc:
        outcome = type(exc).__name__
    if outcome is not None:
        _log.info("sound-scout recording media: %s", outcome)
    return None


def transform_llm_output(*, response_text="", session_id="", turn_id="", **_) -> str | None:
    """The Hermes hook: the reply with the `MEDIA:` line of the turn's recording, or None."""
    try:
        reply, outcome = media_reply(turn_id, response_text)
    except Exception as exc:
        reply, outcome = None, type(exc).__name__
    if outcome is not None:
        _log.info("sound-scout recording media: %s", outcome)
    return reply
