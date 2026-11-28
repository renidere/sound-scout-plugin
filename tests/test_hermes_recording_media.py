"""The `pre_llm_call` and `transform_llm_output` hooks of the Hermes plugin
(`hermes/plugin/sound-scout/recording_media.py`, decision 0039, #216), without Hermes.

A contract test: `in_turn` stands in for Hermes v0.23.3 (`caafe5389a`, read only). It calls
`pre_llm_call` with the keywords of `agent/turn_context.py` `_collect_pre_llm_call_context` and
`transform_llm_output` with the keywords of `agent/turn_finalizer.py` `apply_llm_output_transform`,
each in `contextvars.copy_context()` of the turn on a worker thread, as
`hermes_cli/plugins_dispatch.py` does, and applies the first non-empty string as the reply. The
Hermes home is a temporary directory with a synthesized WAV. No live Hermes is touched.
"""

import ast
import contextvars
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

from tests import test_hermes_binding
from tests.conftest import WavFactory
from tests.test_hermes_binding import TURN, local_turn

MODULE = test_hermes_binding.PACKAGE / "recording_media.py"
NAME = "sound-scout-0123456789ab.wav"
OTHER = "sound-scout-ba9876543210.wav"
TURN_ID = "segment-of-chat-a:task-1:0a1b2c3d"
REPLY = "Голосовое получил. Что разобрать?"
plugin = test_hermes_binding.plugin


@pytest.fixture
def home(plugin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Hermes home with `attachments/` as `file.attach` creates it."""
    found = tmp_path / "hermes-home"
    (found / "attachments").mkdir(parents=True)
    module = types.ModuleType("hermes_constants")
    module.get_hermes_home = lambda: found
    monkeypatch.setitem(sys.modules, "hermes_constants", module)
    return found


@pytest.fixture
def wav(home: Path, make_wav: WavFactory) -> Path:
    path = home / "attachments" / NAME
    path.write_bytes(make_wav("tone.wav", seconds=0.2).read_bytes())
    return path


def in_turn(callback, turn: dict, kwargs: dict):
    """One hook call as Hermes makes it, inside the turn's session context; its return."""
    returned = []

    def run() -> None:
        for name, value in turn.items():
            TURN[name].set(value)
        context = contextvars.copy_context()
        worker = threading.Thread(target=lambda: returned.append(context.run(callback, **kwargs)))
        worker.start()
        worker.join(30)

    contextvars.Context().run(run)
    return returned[0]


def pre_llm_call(
    plugin, user_message, *, turn_id=TURN_ID, session_id="segment-of-chat-a", turn=None
):
    return in_turn(
        plugin.recording_media.pre_llm_call,
        local_turn("chat-a") if turn is None else turn,
        {
            "session_id": session_id,
            "task_id": "task-1",
            "turn_id": turn_id,
            "user_message": user_message,
            "conversation_history": [],
            "is_first_turn": True,
            "model": "model",
            "platform": "tui",
            "parent_session_id": "",
            "sender_id": "",
        },
    )


def transform(plugin, response_text, *, turn_id=TURN_ID, session_id="segment-of-chat-a"):
    return in_turn(
        plugin.recording_media.transform_llm_output,
        local_turn("chat-a"),
        {
            "response_text": response_text,
            "session_id": session_id,
            "model": "model",
            "platform": "tui",
            "turn_id": turn_id,
        },
    )


def reply_of_turn(plugin, user_message, response_text=REPLY, **kwargs) -> str:
    """The stored reply of one turn: Hermes keeps the text when the hook returns None."""
    assert pre_llm_call(plugin, user_message, **kwargs) is None
    rewrite = transform(plugin, response_text, **kwargs)
    return rewrite if isinstance(rewrite, str) and rewrite else response_text


# --- criterion 2: the reply of the recording turn carries the player line ----------------------


@pytest.mark.parametrize(
    "ref",
    [
        f"@file:attachments/{NAME}",
        '@file:"{home}/attachments/' + NAME + '"',
        "@file:`{home}/attachments/" + NAME + "`",
    ],
)
def test_criterion_2_a_recording_turn_gets_one_media_line_of_its_attachment(
    plugin, home: Path, wav: Path, ref: str
) -> None:
    message = f"{ref.format(home=home)}\n\nразбери"

    reply = reply_of_turn(plugin, message)

    assert reply == f"{REPLY}\n\nMEDIA:{wav}"
    assert reply.splitlines()[-1] == f"MEDIA:{home}/attachments/{NAME}"
    assert transform(plugin, REPLY) is None, "the turn is forgotten after its reply"


def test_criterion_2_two_recordings_of_one_turn_get_a_line_each_in_order(
    plugin, home: Path, wav: Path
) -> None:
    other = home / "attachments" / OTHER
    other.write_bytes(wav.read_bytes())
    message = f"@file:attachments/{OTHER} @file:attachments/{NAME} @file:attachments/{OTHER}"

    reply = reply_of_turn(plugin, message, "ответ  \n")

    assert reply == f"ответ\n\nMEDIA:{other}\nMEDIA:{wav}"


def test_criterion_2_a_message_of_text_parts_is_read(plugin, wav: Path) -> None:
    message = [
        {"type": "text", "text": f"@file:attachments/{NAME}"},
        {"type": "image_url", "image_url": {"url": "data:"}},
        {"type": "text", "text": "разбери"},
    ]

    assert reply_of_turn(plugin, message) == f"{REPLY}\n\nMEDIA:{wav}"


def test_criterion_2_a_compression_inside_the_turn_keeps_the_line(plugin, wav: Path) -> None:
    """A compression changes `agent.session_id` between the hooks; `turn_id` stays."""
    assert pre_llm_call(plugin, f"@file:attachments/{NAME}", session_id="old-segment") is None

    assert transform(plugin, REPLY, session_id="new-segment") == f"{REPLY}\n\nMEDIA:{wav}"


@pytest.mark.parametrize(
    "case",
    [
        "no reference",
        "name without @file:",
        "probe file",
        "suffix -2",
        "upper-case hex",
        "11 hex",
        "13 hex",
        "outside attachments",
        "no file",
        "symlink",
        "directory",
        "empty file",
        "other turn",
        "no turn id",
        "messenger turn",
        "no platform",
    ],
)
def test_criterion_2_other_turns_and_other_files_are_left_unchanged(
    plugin, home: Path, wav: Path, case: str
) -> None:
    attachments = home / "attachments"
    message, turn, turn_id, reply_turn = f"@file:attachments/{NAME}", None, TURN_ID, TURN_ID
    if case == "no reference":
        message = "просто текст"
    elif case == "name without @file:":
        message = f"послушай attachments/{NAME}"
    elif case == "probe file":
        name = "sound-scout-probe-a-0123456789ab.wav"
        (attachments / name).write_bytes(wav.read_bytes())
        message = f"@file:attachments/{name}"
    elif case == "suffix -2":
        name = "sound-scout-0123456789ab-2.wav"
        (attachments / name).write_bytes(wav.read_bytes())
        message = f"@file:attachments/{name}"
    elif case == "upper-case hex":
        message = f"@file:attachments/{NAME.replace('ab', 'AB')}"
    elif case == "11 hex":
        message = "@file:attachments/sound-scout-0123456789a.wav"
    elif case == "13 hex":
        message = "@file:attachments/sound-scout-0123456789abc.wav"
    elif case == "outside attachments":
        message = f"@file:recordings/{NAME}"
    elif case == "no file":
        message = f"@file:attachments/{OTHER}"
    elif case == "symlink":
        (attachments / OTHER).symlink_to(wav)
        message = f"@file:attachments/{OTHER}"
    elif case == "directory":
        (attachments / OTHER).mkdir()
        message = f"@file:attachments/{OTHER}"
    elif case == "empty file":
        (attachments / OTHER).write_bytes(b"")
        message = f"@file:attachments/{OTHER}"
    elif case == "other turn":
        reply_turn = "segment-of-chat-a:task-2:ffffffff"
    elif case == "no turn id":
        turn_id = reply_turn = ""
    elif case == "messenger turn":
        turn = {**local_turn("chat-a"), "HERMES_SESSION_PLATFORM": "telegram"}
    elif case == "no platform":
        turn = {}

    assert pre_llm_call(plugin, message, turn_id=turn_id, turn=turn) is None
    assert transform(plugin, REPLY, turn_id=reply_turn) is None


def test_criterion_2_the_directory_of_the_reference_is_never_used(
    plugin, home: Path, wav: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere" / "attachments"
    elsewhere.mkdir(parents=True)
    (elsewhere / OTHER).write_bytes(wav.read_bytes())

    assert pre_llm_call(plugin, f"@file:{elsewhere}/{OTHER}") is None
    assert transform(plugin, REPLY) is None
    reply = reply_of_turn(plugin, f"@file:{elsewhere}/{NAME}")
    assert reply == f"{REPLY}\n\nMEDIA:{wav}"


def test_criterion_2_a_home_path_that_breaks_the_media_line_gets_none(
    plugin, tmp_path: Path, make_wav: WavFactory, monkeypatch
) -> None:
    home = tmp_path / "hermes home"
    (home / "attachments").mkdir(parents=True)
    (home / "attachments" / NAME).write_bytes(make_wav("tone.wav", seconds=0.2).read_bytes())
    module = types.ModuleType("hermes_constants")
    module.get_hermes_home = lambda: home
    monkeypatch.setitem(sys.modules, "hermes_constants", module)

    assert reply_of_turn(plugin, f"@file:attachments/{NAME}") == REPLY


def test_criterion_2_a_reply_that_already_names_the_file_is_unchanged(plugin, wav: Path) -> None:
    named = f"Вот запись:\nMEDIA:{wav}"

    assert pre_llm_call(plugin, f"@file:attachments/{NAME}") is None
    assert transform(plugin, named) is None
    assert reply_of_turn(plugin, f"@file:attachments/{NAME}", named) == named
    assert named.count("MEDIA:") == 1


@pytest.mark.parametrize("mention", ["Файл {wav} получен.", "См. `MEDIA:{wav}` выше."])
def test_criterion_2_a_reply_that_mentions_the_path_in_text_still_gets_the_line(
    plugin, wav: Path, mention: str
) -> None:
    text = mention.format(wav=wav)

    assert pre_llm_call(plugin, f"@file:attachments/{NAME}") is None
    assert transform(plugin, text) == f"{text}\n\nMEDIA:{wav}"


@pytest.mark.parametrize("response_text", ["", "   \n", None, 42])
def test_criterion_2_an_empty_reply_gets_no_media_line(plugin, wav: Path, response_text) -> None:
    assert pre_llm_call(plugin, f"@file:attachments/{NAME}") is None

    assert transform(plugin, response_text) is None
    assert transform(plugin, REPLY) is None, "the turn is forgotten"


def test_criterion_2_pending_turns_are_bounded(plugin, wav: Path) -> None:
    limit = plugin.recording_media.PENDING_MAX
    assert limit == 32
    turns = [f"segment:task:{n:08x}" for n in range(limit + 1)]
    for turn_id in turns:
        assert pre_llm_call(plugin, f"@file:attachments/{NAME}", turn_id=turn_id) is None

    assert len(plugin.recording_media._pending) == limit
    assert transform(plugin, REPLY, turn_id=turns[0]) is None
    assert transform(plugin, REPLY, turn_id=turns[1]) == f"{REPLY}\n\nMEDIA:{wav}"
    assert transform(plugin, REPLY, turn_id=turns[-1]) == f"{REPLY}\n\nMEDIA:{wav}"


def test_hook_errors_never_break_the_reply(plugin, wav: Path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="sound-scout.recording_media")

    def broken(*args, **kwargs):
        raise RuntimeError(str(wav))

    monkeypatch.setattr(plugin.recording_media, "_recording_path", broken)
    assert pre_llm_call(plugin, f"@file:attachments/{NAME}") is None
    monkeypatch.setattr(plugin.recording_media, "media_reply", broken)
    assert transform(plugin, REPLY) is None
    assert caplog.text.count("sound-scout recording media: RuntimeError") == 2
    assert str(wav) not in caplog.text


# --- the package -----------------------------------------------------------------------------


def test_register_adds_the_recording_hooks_without_holding_the_stream(plugin, monkeypatch) -> None:
    monkeypatch.setattr(plugin.delivery, "start", lambda: None)

    class Ctx:
        def __init__(self) -> None:
            self.hooks = []

        def register_hook(self, name, callback, **options):
            self.hooks.append((name, callback, options))

    ctx = Ctx()
    plugin.register(ctx)
    assert ctx.hooks == [
        ("post_tool_call", plugin.binding.post_tool_call, {}),
        ("post_tool_call", plugin.seen.post_tool_call, {}),
        ("pre_llm_call", plugin.recording_media.pre_llm_call, {}),
        ("transform_llm_output", plugin.recording_media.transform_llm_output, {}),
    ]


def test_recording_media_logs_no_path_and_no_text(
    plugin, home: Path, wav: Path, tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.DEBUG)
    message = f"@file:attachments/{NAME}\n\nсекретная фраза"

    reply_of_turn(plugin, message, "ответ ассистента")
    reply_of_turn(plugin, f"@file:attachments/{OTHER} фраза-два")

    outcomes = [
        record.getMessage()
        for record in caplog.records
        if record.name == "sound-scout.recording_media"
    ]
    assert outcomes == [
        "sound-scout recording media: recording remembered",
        "sound-scout recording media: media line added",
        "sound-scout recording media: no recording",
    ]
    for secret in (str(tmp_path), NAME, OTHER, "секретная", "ответ ассистента", "chat-a"):
        assert secret not in caplog.text


def test_the_hook_imports_only_the_stdlib_and_its_package() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 11))
    top_level, relative = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level:
            relative |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module.split(".")[0])
    assert top_level <= set(sys.stdlib_module_names)
    assert relative == {"binding"}
    assert "sound_scout" not in source


# --- criterion 1: the decision record --------------------------------------------------------


def test_criterion_1_the_decision_names_the_paths_the_sources_and_the_version() -> None:
    record = (
        test_hermes_binding.REPO / "docs/decisions/0039-chat-recording-playback.md"
    ).read_text(encoding="utf-8")
    for words in (
        "`file.attach`",
        "`send_document`",
        "`TRANSCRIPT_DIRECTIVE_AREA`",
        "`COMPOSER_AREAS`",
        "v0.23.3",
        "caafe5389a",
        "https://github.com/NousResearch/hermes-agent",
        "`MEDIA:",
        "`pre_llm_call`",
        "`transform_llm_output`",
        "O11",
    ):
        assert words in record, words


def test_217_the_decision_records_the_removed_plugin_player_and_its_limitation() -> None:
    record = (
        test_hermes_binding.REPO / "docs/decisions/0039-chat-recording-playback.md"
    ).read_text(encoding="utf-8")
    change = record.split("## Правка 2026-11-28 (#217)", 1)[1].split("\n## ", 1)[0]
    for words in (
        "issues/216#issuecomment-7242200161",
        "плеер плагина удаляется, запись\nслушается только в чате",
        "нет маршрута `GET /recordings/{name}`",
        "Хуки `recording_media.py` не меняются",
        "0.5.0",
    ):
        assert words in change, words
    limits = record.split("## Ограничения", 1)[1].split("\n## ", 1)[0]
    assert "У такой записи плеера нет нигде" in limits and "Владелец принял это" in limits
    assert "Для такой записи остаётся «Прослушать»" not in record
