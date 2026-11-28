"""Delivery loop of the Hermes plugin (`hermes/plugin/sound-scout/delivery.py`, #102), without
Hermes.

`cli` is the real `sound-scout` of this venv over a temporary data root; `send` and
`history_has` are fakes that record their calls. No real chat is touched.
"""

import ast
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from sound_scout import app, config, jobs
from sound_scout.config import Config
from tests.conftest import WavFactory
from tests.test_delivery import (
    analyzed_job,
    claim,
    ended_job,
    job_file,
    run_attempt,
    unconfirmed_job,
)
from tests.test_jobs_runner import POLL, runner_thread, substitute, wait_until

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "hermes" / "plugin" / "sound-scout" / "delivery.py"
TIMEOUT = "local chat {}: delivery not confirmed: TimeoutError()"
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)


@pytest.fixture
def loop():
    spec = importlib.util.spec_from_file_location("sound_scout_plugin_delivery", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    for name in ("recordings", "data"):
        (tmp_path / name).mkdir()
    path = tmp_path / "sound-scout.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'recordings'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n"
        '[asr]\nmodel = "absent/model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SOUND_SCOUT_BIN", str(Path(sys.executable).parent / "sound-scout"))
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    return config.load()


class Chats:
    """The fake send and history: what was sent where, what each send answers."""

    def __init__(self, cfg: Config, *answers, history: bool = False) -> None:
        self.cfg = cfg
        self.answers = list(answers)
        self.history = history
        self.sent: list[tuple[str, str]] = []
        self.checked: list[tuple[str, str]] = []
        self.markers: list[dict] = []

    def send(self, chat_id: str, text: str):
        self.sent.append((chat_id, text))
        job_id = text.splitlines()[-1].removeprefix("sound-scout-job:").partition("-a")[0]
        self.markers.append(job_file(self.cfg, job_id)["delivery"])
        answer = self.answers.pop(0) if self.answers else None
        if isinstance(answer, BaseException):
            raise answer
        return answer.format(chat_id) if isinstance(answer, str) else answer

    def history_has(self, chat_id: str, tag: str) -> bool:
        self.checked.append((chat_id, tag))
        if isinstance(self.history, BaseException):
            raise self.history
        return self.history


def ticks(
    loop,
    chats: Chats,
    n: int = 3,
    *,
    owner=True,
    cli=None,
    state=None,
    loop_id="loop-1",
    chat_busy=None,
):
    """`n` ticks; by default every chat has no turn (`chat_busy` answers False)."""
    state = {} if state is None else state
    return [
        loop.tick(
            owner=lambda: owner,
            profile=lambda: "default",
            loop_id=loop_id,
            cli=cli or loop.run_cli,
            send=chats.send,
            history_has=chats.history_has,
            chat_busy=chat_busy or (lambda chat_id: False),
            state=state,
        )
        for _ in range(n)
    ]


def delivery_of(cfg: Config, job_id: str) -> dict:
    return jobs.status(job_id, cfg)["delivery"]


# --- criterion 1: only in the owner process, one note per job to its chat --------------------


def test_without_the_owner_a_tick_calls_neither_the_cli_nor_send(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup)
    calls = []

    outcomes = ticks(loop, chats, owner=False, cli=lambda args: calls.append(args))

    assert outcomes == ["idle_no_owner"] * 3
    assert calls == [] and chats.sent == [] and chats.checked == []
    assert delivery_of(setup, job_id) is None


@pytest.mark.slow
def test_with_the_owner_one_job_gets_exactly_one_note_in_its_chat(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-7")
    chats = Chats(setup)

    assert ticks(loop, chats) == ["delivered", "idle", "idle"]

    assert [chat for chat, _ in chats.sent] == ["chat-7"]
    assert delivery_of(setup, job_id)["state"] == "delivered"
    assert delivery_of(setup, job_id)["confirmed_by"] == "send"


@pytest.mark.slow
def test_two_jobs_of_two_chats_get_one_note_each_in_their_own_chat(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    first = ended_job(setup, wav, session_id="chat-a")
    second = ended_job(setup, wav, session_id="chat-b")
    chats = Chats(setup)

    ticks(loop, chats)

    assert [(chat, text.splitlines()[-1]) for chat, text in chats.sent] == [
        ("chat-a", f"sound-scout-job:{first}"),
        ("chat-b", f"sound-scout-job:{second}"),
    ]


# --- criterion 2: marker first, history on a timeout, no repeat ------------------------------


def test_the_marker_is_in_job_json_while_send_runs(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    ended_job(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup)

    ticks(loop, chats, 1)

    [marker] = chats.markers
    assert marker["state"] == "sending" and marker["loop_id"] == "loop-1"


@pytest.mark.slow
@pytest.mark.parametrize("failure", [TIMEOUT, RuntimeError("gateway gone")])
def test_an_unknown_send_without_the_tag_in_history_is_unconfirmed_and_never_resent(
    loop, setup: Config, make_wav: WavFactory, failure
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-1")
    chats = Chats(setup, failure, history=False)

    assert ticks(loop, chats, 4) == ["unconfirmed", "idle", "idle", "idle"]

    assert len(chats.sent) == 1
    assert chats.checked == [("chat-1", f"sound-scout-job:{job_id}")]
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == "delivery_unconfirmed"
    assert "tag not found in chat history" in recorded["error"]


@pytest.mark.slow
@pytest.mark.parametrize("failure", [TIMEOUT, RuntimeError("gateway gone")])
def test_an_unknown_send_with_the_tag_in_history_is_delivered_by_history(
    loop, setup: Config, make_wav: WavFactory, failure
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup, failure, history=True)

    assert ticks(loop, chats) == ["delivered_by_history", "idle", "idle"]

    assert len(chats.sent) == 1
    assert delivery_of(setup, job_id)["state"] == "delivered"
    assert delivery_of(setup, job_id)["confirmed_by"] == "history"


def test_a_failing_history_check_counts_as_tag_not_found(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup, TIMEOUT, history=OSError("db locked"))

    ticks(loop, chats, 2)

    assert len(chats.sent) == 1
    assert delivery_of(setup, job_id)["state"] == "delivery_unconfirmed"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("history", "state"), [(True, "delivered"), (False, "delivery_unconfirmed")]
)
def test_a_marker_of_a_stopped_loop_is_checked_in_history_never_sent(
    loop, setup: Config, make_wav: WavFactory, history: bool, state: str
) -> None:

    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-1")
    claim(setup, job_id, "dead-loop")
    chats = Chats(setup, history=history)

    ticks(loop, chats, 3, loop_id="live-loop")

    assert chats.sent == []
    assert chats.checked == [("chat-1", f"sound-scout-job:{job_id}")]
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == state
    if not history:
        assert recorded["error"] == loop.ORPHAN_ERROR


def test_a_failed_deliver_end_is_recorded_on_the_next_tick_without_a_second_send(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup)
    ends = []

    def flaky_cli(args):
        if args[0] == "deliver-end":
            ends.append(args)
            if len(ends) == 1:
                return None
        return loop.run_cli(args)

    state = {}
    assert ticks(loop, chats, 1, cli=flaky_cli, state=state) == ["delivered"]
    assert delivery_of(setup, job_id)["state"] == "sending"
    assert state["pending"] == (job_id, "delivered", None)

    assert ticks(loop, chats, 2, cli=flaky_cli, state=state) == ["end_recorded", "idle"]

    assert len(chats.sent) == 1 and len(ends) == 2
    assert delivery_of(setup, job_id)["state"] == "delivered"


def test_an_exception_after_deliver_next_still_ends_the_job_without_a_second_send(
    loop, setup: Config, make_wav: WavFactory, monkeypatch
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup)
    real_record = loop._record
    calls = []

    def raising_record(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("chat 42 text")
        return real_record(*args)

    monkeypatch.setattr(loop, "_record", raising_record)
    state = {}
    assert ticks(loop, chats, 1, state=state) == ["delivered"]
    assert delivery_of(setup, job_id)["state"] == "sending"
    assert state["pending"] == (job_id, "delivered", None)

    assert ticks(loop, chats, 2, state=state) == ["end_recorded", "idle"]
    assert len(chats.sent) == 1
    assert delivery_of(setup, job_id)["state"] == "delivered"


def test_an_exception_while_settling_the_send_is_unconfirmed(
    loop, setup: Config, make_wav: WavFactory, monkeypatch
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup)

    def raising_send_once(*args):
        raise LookupError("chat 42 text")

    monkeypatch.setattr(loop, "_send_once", raising_send_once)
    assert ticks(loop, chats, 2) == ["unconfirmed", "idle"]
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == "delivery_unconfirmed"
    assert recorded["error"] == "delivery loop raised LookupError"


def test_run_cli_with_a_nul_in_its_arguments_did_not_answer(loop, setup: Config) -> None:
    assert loop.run_cli(["deliver-end", "--error=a\0b", "f" * 32]) is None


# --- criterion 3: the note lists each requested layer with its status ------------------------


@pytest.mark.slow
@needs_ffmpeg
def test_the_sent_note_lists_the_layers_and_does_not_read_as_all_done(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = analyzed_job(setup, make_wav("recordings/tone.wav", seconds=1.5))
    chats = Chats(setup)

    ticks(loop, chats, 1)

    [(_, text)] = chats.sent
    result = jobs.result(job_id, setup)
    lines = text.splitlines()
    assert "`partial`" in lines[0]
    for layer, name in (
        ("content", "содержание (ASR)"),
        ("measurements", "измерения (акустика)"),
        ("sound", "звучание (audio review)"),
    ):
        status = result["stages"][jobs.LAYER_STAGES[layer]]["status"]
        assert status in ("completed", "unavailable", "skipped")
        assert any(line.startswith(f"- {name}: `{status}` — ") for line in lines)
    assert {"completed", "unavailable"} <= {
        result["stages"][stage]["status"] for stage in jobs.LAYER_STAGES.values()
    }
    assert "готово" not in text.lower() and "всё выполнено" not in text.lower()


@pytest.mark.slow
@needs_ffmpeg
def test_criterion_3_note_names_completed_unavailable_and_skipped_layers_never_all_done(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = analyzed_job(setup, make_wav("recordings/tone.wav", seconds=1.5))
    run_id = job_file(setup, job_id)["attempts"][0]["run_id"]
    result_path = setup.data_root / "runs" / run_id / "result.json"
    document = json.loads(result_path.read_text(encoding="utf-8"))
    # As a worker cut off before the review leaves it: the requested layer is `skipped`.
    document["stages"]["audio_review"] |= {"status": "skipped", "reason": "worker deadline"}
    result_path.write_text(json.dumps(document), encoding="utf-8")
    chats = Chats(setup)

    assert ticks(loop, chats, 2) == ["delivered", "idle"]

    [(_, text)] = chats.sent
    lines = text.splitlines()
    asr_reason = jobs.result(job_id, setup)["stages"]["asr"]["reason"]
    assert f"- содержание (ASR): `unavailable` — недоступно; причина: {asr_reason}" in lines
    assert "- измерения (акустика): `completed` — выполнено" in lines
    assert "- звучание (audio review): `skipped` — пропущено; причина: worker deadline" in lines
    assert "`complete`" not in lines[0]
    for phrase in ("готово", "всё выполнено", "все выполнено", "всё хорошо", "all done"):
        assert phrase not in text.lower()


# --- criterion 4: a deleted or unavailable chat is a delivery error, never another chat -------


@pytest.mark.slow
@pytest.mark.parametrize(
    "said",
    [
        "local chat {}: delivery not confirmed: KeyError('{}')",
        "delivery refused: no route to local chat {}",
    ],
)
def test_a_chat_error_is_delivery_failed_in_the_job_and_nothing_else_is_sent(
    loop, setup: Config, make_wav: WavFactory, said: str
) -> None:
    wav = make_wav("recordings/a.wav")
    gone = ended_job(setup, wav, session_id="deleted-chat")
    other = ended_job(setup, wav, session_id="other-chat")
    chats = Chats(setup, said.replace("{}", "{0}"))

    assert ticks(loop, chats) == ["failed", "delivered", "idle"]

    assert [chat for chat, _ in chats.sent] == ["deleted-chat", "other-chat"]
    assert "sound-scout-job:" + gone in chats.sent[0][1]
    assert "sound-scout-job:" + other in chats.sent[1][1]
    recorded = delivery_of(setup, gone)
    assert recorded["state"] == "delivery_failed"
    assert recorded["error"] == said.replace("{}", "deleted-chat")
    assert chats.checked == []


@pytest.mark.slow
def test_no_owner_in_the_send_releases_the_job_and_the_next_tick_delivers_once(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-1")
    chats = Chats(setup, "local chat {}: no gateway in this process owns its session")

    assert ticks(loop, chats) == ["released", "delivered", "idle"]

    assert [chat for chat, _ in chats.sent] == ["chat-1", "chat-1"]
    assert delivery_of(setup, job_id)["state"] == "delivered"


# --- the loop thread, the CLI client and the Hermes edge -------------------------------------


def stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def test_start_runs_one_loop_thread_per_process(loop, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "hermes_constants", stub("hermes_constants"))
    stop = threading.Event()
    owner_checks = threading.Event()

    def owner() -> bool:
        owner_checks.set()
        return False

    first = loop.start(owner=owner, stop=stop, poll_seconds=0.01)
    try:
        assert first is not None and first.name == loop.THREAD_NAME and first.daemon
        assert owner_checks.wait(5)
        assert loop.start(owner=owner, stop=stop) is None
        again = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location("sound_scout_plugin_delivery_2", MODULE)
        )
        again.__spec__.loader.exec_module(again)
        assert again.start(owner=owner, stop=stop) is None
        assert [t.name for t in threading.enumerate()].count(loop.THREAD_NAME) == 1
    finally:
        stop.set()
        first.join(5)
    assert not first.is_alive()


def test_start_without_hermes_starts_nothing(loop, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    assert loop.start(owner=lambda: True) is None
    assert loop.THREAD_NAME not in [t.name for t in threading.enumerate()]


def test_an_exception_in_a_tick_does_not_stop_the_loop(loop, caplog) -> None:
    stop = threading.Event()
    calls = []

    def owner() -> bool:
        calls.append(1)
        if len(calls) >= 3:
            stop.set()
        raise RuntimeError("chat 42 text")

    with caplog.at_level(logging.WARNING, logger="sound-scout.delivery"):
        loop._run(
            {
                "owner": owner,
                "profile": None,
                "loop_id": "l",
                "cli": None,
                "send": None,
                "history_has": None,
                "chat_busy": None,
                "state": {},
            },
            stop,
            0,
        )

    assert len(calls) == 3
    assert "tick failed: RuntimeError" in caplog.text and "chat 42" not in caplog.text


def test_run_cli_without_its_variables_logs_not_configured_once(loop, monkeypatch, caplog) -> None:
    monkeypatch.delenv("SOUND_SCOUT_BIN", raising=False)
    with caplog.at_level(logging.WARNING, logger="sound-scout.delivery"):
        assert loop.run_cli(["deliver-next"]) is None
        assert loop.run_cli(["deliver-next"]) is None
    assert caplog.text.count("not_configured") == 1


def test_run_cli_answers_a_refusal_with_its_code(loop, setup: Config) -> None:
    answer = loop.run_cli(["deliver-end", "--outcome=delivered", "f" * 32])
    assert answer == {"error": "job_not_found"}
    assert loop.run_cli(["deliver-next", "--profile=default", "--loop=l"]) == {"action": "none"}


def test_run_cli_passes_only_the_allowed_environment_without_shell(
    loop, monkeypatch, setup: Config
) -> None:
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs, argv=argv)
        return types.SimpleNamespace(returncode=0, stdout=b'{"action": "none"}', stderr=b"")

    monkeypatch.setenv("SECRET_TOKEN", "x")
    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    assert loop.run_cli(["deliver-next"]) == {"action": "none"}
    assert seen["argv"][1:] == ["jobs", "deliver-next", "--json"]
    assert "shell" not in seen and seen["timeout"] == loop.CLI_TIMEOUT_SECONDS
    assert set(seen["env"]) <= {*loop.PASSED_ENV, loop.CONFIG_ENV}
    assert seen["env"][loop.CONFIG_ENV] == str(config.config_path())


def test_the_hermes_edge_calls_the_gateway_functions(loop, monkeypatch) -> None:
    sent = []

    class Db:
        def get_messages(self, target, *, latest, limit, include_ancestors, include_compacted):
            assert latest is True and limit == loop.HISTORY_SCAN
            assert include_ancestors is True and include_compacted is True
            return [{"content": "hello"}, {"content": f"note\nsound-scout-job:{target}"}]

    authority = types.SimpleNamespace(db=Db())
    # Hermes 3b42157858 `owner_for_home` returns `(authority, loop)`, never the pid.
    owners = {"HOME": (authority, None)}
    session_cron = stub(
        "gateway.session_cron",
        owner_for_home=owners.get,
        deliver_to_local_chat=lambda home, chat, text: sent.append((home, chat, text)),
    )
    viewer_close = stub(
        "gateway.session_viewer_close", _physical_target=lambda auth, chat: f"phys-{chat}"
    )
    monkeypatch.setitem(sys.modules, "gateway", stub("gateway"))
    monkeypatch.setitem(sys.modules, "gateway.session_cron", session_cron)
    monkeypatch.setitem(sys.modules, "gateway.session_viewer_close", viewer_close)
    constants = stub(
        "hermes_constants", get_hermes_home=lambda: "HOME", profile_name_for_home=lambda h: None
    )
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)

    assert loop.hermes_owner() is True
    assert loop.hermes_profile() == "default"
    assert loop.hermes_send("chat-1", "text") is None and sent == [("HOME", "chat-1", "text")]
    assert loop.hermes_history_has("c", "sound-scout-job:phys-c") is True
    assert loop.hermes_history_has("c", "sound-scout-job:c") is False

    owners.clear()
    assert loop.hermes_owner() is False
    with pytest.raises(RuntimeError):
        loop.hermes_history_has("c", "sound-scout-job:phys-c")


def hermes_with_history(monkeypatch, segments: dict[str, list[dict]], calls: list[dict]):
    """A fake gateway whose lineage read returns the parent segments only with
    `include_ancestors` and the rows compaction archived (`compacted: 1`) only with
    `include_compacted`, newest `limit` messages, as Hermes v0.23.1 `get_messages` does."""

    class Db:
        def get_messages(
            self, target, *, latest, limit, include_ancestors=False, include_compacted=False
        ):
            calls.append(
                {
                    "target": target,
                    "latest": latest,
                    "limit": limit,
                    "include_ancestors": include_ancestors,
                    "include_compacted": include_compacted,
                }
            )
            rows = [row for name in segments for row in segments[name]]
            if not include_ancestors:
                rows = segments[target]
            if not include_compacted:
                rows = [row for row in rows if not row.get("compacted")]
            return rows[-limit:] if latest else rows[:limit]

    authority = types.SimpleNamespace(db=Db())
    monkeypatch.setitem(sys.modules, "gateway", stub("gateway"))
    monkeypatch.setitem(
        sys.modules,
        "gateway.session_cron",
        stub("gateway.session_cron", owner_for_home=lambda home: (authority, None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "gateway.session_viewer_close",
        stub("gateway.session_viewer_close", _physical_target=lambda auth, chat: "tip"),
    )
    monkeypatch.setitem(
        sys.modules, "hermes_constants", stub("hermes_constants", get_hermes_home=lambda: "HOME")
    )


def test_the_history_check_reads_the_whole_lineage_and_finds_the_tag_in_a_parent(
    loop, monkeypatch
) -> None:
    calls = []
    segments = {
        "root": [{"content": "note\nsound-scout-job:j1"}],
        "tip": [{"content": "after the compression"}],
    }
    hermes_with_history(monkeypatch, segments, calls)

    assert loop.hermes_history_has("chat-1", "sound-scout-job:j1") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j2") is False
    assert calls[0] == {
        "target": "tip",
        "latest": True,
        "limit": loop.HISTORY_SCAN,
        "include_ancestors": True,
        "include_compacted": True,
    }


def test_the_history_check_finds_the_tag_in_a_row_compaction_archived(loop, monkeypatch) -> None:
    segments = {
        "tip": [
            {"role": "user", "content": "note\nsound-scout-job:j1", "compacted": 1},
            {"role": "assistant", "content": "summary of the earlier talk"},
        ]
    }
    hermes_with_history(monkeypatch, segments, [])

    assert loop.hermes_history_has("chat-1", "sound-scout-job:j1") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j2") is False


def test_the_tag_inside_the_job_status_tool_result_is_not_a_delivered_note(
    loop, monkeypatch
) -> None:
    status = json.dumps(
        {"delivery": {"state": "delivery_unconfirmed", "tag": "sound-scout-job:j1"}}
    )
    segments = {
        "tip": [
            {"role": "tool", "content": status},
            {"role": "tool", "content": "sound-scout-job:j1"},
            {"role": "assistant", "content": "Задание j1 (`sound-scout-job:j1`) не доставлено."},
        ]
    }
    hermes_with_history(monkeypatch, segments, [])

    assert loop.hermes_history_has("chat-1", "sound-scout-job:j1") is False


def test_the_tag_line_of_a_note_in_a_user_or_assistant_row_is_a_delivered_note(
    loop, monkeypatch
) -> None:
    # Hermes v0.23.1 writes a cron note into the chat as a `user` row; the agent may quote it.
    segments = {
        "tip": [
            {
                "role": "user",
                "content": "Sound Scout: анализ\nОтчёт: r/report.md\nsound-scout-job:j1",
            },
            {"role": "assistant", "content": "> Отчёт готов\n>\n  sound-scout-job:j2  \nok"},
            {"role": "user", "content": [{"type": "text", "text": "x\nsound-scout-job:j3"}]},
        ]
    }
    hermes_with_history(monkeypatch, segments, [])

    assert loop.hermes_history_has("chat-1", "sound-scout-job:j1") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j2") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j3") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j4") is False


def test_a_history_as_long_as_the_scan_without_the_tag_cannot_answer(loop, monkeypatch) -> None:
    rows = [{"content": f"message {n}"} for n in range(loop.HISTORY_SCAN + 5)]
    hermes_with_history(monkeypatch, {"tip": rows}, [])

    with pytest.raises(LookupError):
        loop.hermes_history_has("chat-1", "sound-scout-job:j1")


def test_the_history_check_of_one_attempt_never_matches_the_note_of_another(
    loop, monkeypatch
) -> None:
    segments = {
        "tip": [
            {"role": "user", "content": "Sound Scout: анализ\nsound-scout-job:j1-a2"},
            {"role": "user", "content": [{"type": "text", "text": "x\nsound-scout-job:j2-a3"}]},
            {"role": "user", "content": "Sound Scout: анализ\nsound-scout-job:j3"},
        ]
    }
    hermes_with_history(monkeypatch, segments, [])

    assert loop.hermes_history_has("chat-1", "sound-scout-job:j1-a2") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j1") is False
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j1-a3") is False
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j2-a3") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j2") is False
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j3") is True
    assert loop.hermes_history_has("chat-1", "sound-scout-job:j3-a2") is False


class NotedChats(Chats):
    """A chat whose history holds the notes this fake sent and confirmed (a timed-out note never
    arrives), checked by the loop's own match."""

    def __init__(self, loop, cfg: Config, *answers) -> None:
        super().__init__(cfg, *answers)
        self.loop = loop
        self.notes: list[tuple[str, str]] = []

    def send(self, chat_id: str, text: str):
        answer = super().send(chat_id, text)
        if answer is None:
            self.notes.append((chat_id, text))
        return answer

    def history_has(self, chat_id: str, tag: str) -> bool:
        self.checked.append((chat_id, tag))
        rows = [{"role": "user", "content": text} for chat, text in self.notes if chat == chat_id]
        return any(self.loop._is_note(row, tag) for row in rows)


@pytest.mark.slow
@pytest.mark.parametrize("ended", ["cancelled", "failed"])
def test_a_retried_attempt_gets_one_more_note_and_its_check_misses_the_first_note(
    loop, setup: Config, make_wav: WavFactory, ended: str
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-1", state="running")
    run_attempt(setup, job_id, ended)
    chats = NotedChats(loop, setup, None, TIMEOUT)

    assert ticks(loop, chats, 2) == ["delivered", "idle"]
    jobs.retry(job_id, setup)
    assert ticks(loop, chats, 1) == ["idle"]
    run_attempt(setup, job_id)
    assert ticks(loop, chats, 2) == ["unconfirmed", "idle"]

    first, second = f"sound-scout-job:{job_id}", f"sound-scout-job:{job_id}-a2"
    assert [(chat, text.splitlines()[-1]) for chat, text in chats.sent] == [
        ("chat-1", first),
        ("chat-1", second),
    ]
    assert chats.checked == [("chat-1", second)]
    job = job_file(setup, job_id)
    assert [item["tag"] for item in job["deliveries"]] == [first]
    assert job["delivery"]["state"] == "delivery_unconfirmed" and job["delivery"]["tag"] == second


# --- redelivery on the operator's request (#122) ---------------------------------------------


def requested(cfg: Config, wav: Path, **kwargs) -> str:
    from sound_scout import delivery

    job_id = unconfirmed_job(cfg, wav, **kwargs)
    delivery.request_redelivery(job_id, cfg)
    return job_id


@pytest.mark.slow
def test_a_redelivery_with_the_tag_in_history_sends_nothing(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = requested(setup, make_wav("recordings/a.wav"), session_id="chat-1")
    chats = Chats(setup, history=True)

    assert ticks(loop, chats) == ["delivered_by_history", "idle", "idle"]

    assert chats.sent == []
    assert chats.checked == [("chat-1", f"sound-scout-job:{job_id}")]
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == "delivered" and recorded["confirmed_by"] == "history"
    assert recorded["redelivery"] is True


@pytest.mark.slow
def test_a_redelivery_without_the_tag_sends_exactly_one_note_to_the_bound_chat(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = requested(setup, make_wav("recordings/a.wav"), session_id="chat-7")
    chats = Chats(setup, history=False)

    assert ticks(loop, chats, 4) == ["delivered", "idle", "idle", "idle"]

    assert [chat for chat, _ in chats.sent] == ["chat-7"]
    assert chats.sent[0][1].splitlines()[-1] == f"sound-scout-job:{job_id}"
    assert chats.checked == [("chat-7", f"sound-scout-job:{job_id}")]
    [marker] = chats.markers
    assert marker["state"] == "sending" and marker["redelivery"] is True
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == "delivered" and recorded["confirmed_by"] == "send"


def test_an_unreadable_history_sends_nothing_and_a_new_request_is_taken(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    from sound_scout import delivery

    job_id = requested(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup, history=LookupError("history longer than the scan"))

    assert ticks(loop, chats, 2) == ["unconfirmed", "idle"]

    assert chats.sent == []
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == "delivery_unconfirmed"
    assert recorded["error"] == loop.REDELIVERY_UNREAD
    assert delivery.request_redelivery(job_id, setup)["delivery"]["state"] == (
        "redelivery_requested"
    )
    assert len(job_file(setup, job_id)["redeliveries"]) == 2


@pytest.mark.slow
def test_a_timed_out_redelivery_without_the_tag_is_unconfirmed_after_one_send(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = requested(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup, TIMEOUT, history=False)

    assert ticks(loop, chats, 3) == ["unconfirmed", "idle", "idle"]

    assert len(chats.sent) == 1 and len(chats.checked) == 2
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == "delivery_unconfirmed" and recorded["redelivery"] is True


@pytest.mark.slow
def test_no_owner_in_a_redelivery_send_waits_and_checks_the_history_again(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = requested(setup, make_wav("recordings/a.wav"), session_id="chat-1")
    chats = Chats(setup, "local chat {}: no gateway in this process owns its session")

    assert ticks(loop, chats, 1) == ["released"]
    assert delivery_of(setup, job_id)["state"] == "redelivery_requested"

    assert ticks(loop, chats, 2) == ["delivered", "idle"]
    assert [chat for chat, _ in chats.sent] == ["chat-1", "chat-1"]
    assert len(chats.checked) == 2
    assert delivery_of(setup, job_id)["state"] == "delivered"


def test_the_hermes_edge_without_gateway_has_no_owner(loop, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gateway.session_cron", None)
    assert loop.hermes_owner() is False


def test_the_loop_imports_only_the_stdlib_and_parses_as_python_311() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 11))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module.split(".")[0])
    assert top_level <= set(sys.stdlib_module_names)
    assert "sound_scout" not in source


# --- acceptance criterion 1 of #122 through the real `sound-scout jobs redeliver` ------------


def jobs_redeliver(job_id: str) -> subprocess.CompletedProcess[str]:
    """The operator's real console command, with the configuration of `setup`."""
    return subprocess.run(
        [os.environ["SOUND_SCOUT_BIN"], "jobs", "redeliver", job_id],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
    )


@pytest.mark.slow
def test_criterion_1_jobs_redeliver_makes_the_loop_send_one_more_note_to_the_same_chat(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-7")
    chats = Chats(setup, TIMEOUT, history=False)
    assert ticks(loop, chats, 2) == ["unconfirmed", "idle"]
    assert delivery_of(setup, job_id)["state"] == "delivery_unconfirmed"
    assert len(chats.sent) == 1

    accepted = jobs_redeliver(job_id)
    assert accepted.returncode == 0, accepted.stderr

    assert ticks(loop, chats, 5) == ["delivered", "idle", "idle", "idle", "idle"]
    assert [chat for chat, _ in chats.sent] == ["chat-7", "chat-7"]
    assert chats.sent[1][1].splitlines()[-1] == f"sound-scout-job:{job_id}"
    assert delivery_of(setup, job_id)["state"] == "delivered"
    assert len(job_file(setup, job_id)["redeliveries"]) == 1

    refused = jobs_redeliver(job_id)
    assert refused.returncode == 1
    assert refused.stderr.startswith("sound-scout jobs: delivery_not_unconfirmed: ")
    assert ticks(loop, chats, 3) == ["idle", "idle", "idle"]
    assert len(chats.sent) == 2


@pytest.mark.slow
def test_criterion_1_jobs_redeliver_takes_no_job_but_a_delivery_unconfirmed_one(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-7")
    before = job_file(setup, job_id)

    refused = jobs_redeliver(job_id)

    assert refused.returncode == 1
    assert refused.stderr.startswith("sound-scout jobs: delivery_not_unconfirmed: ")
    assert job_file(setup, job_id) == before
    chats = Chats(setup)
    assert ticks(loop, chats) == ["delivered", "idle", "idle"]
    assert [chat for chat, _ in chats.sent] == ["chat-7"]


# --- acceptance criteria of #187 through the real runner, `jobs retry` and the loop ----------


def jobs_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """The operator's real `sound-scout jobs` command, with the configuration of `setup`."""
    return subprocess.run(
        [os.environ["SOUND_SCOUT_BIN"], "jobs", *args],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
    )


def bound_job(cfg: Config, wav: Path) -> str:
    binding = jobs.Binding(submitter="desktop", session_id="chat-1")
    return jobs.submit(jobs.SubmitRequest(str(wav)), binding, cfg)["job_id"]


def run_to_end(cfg: Config, job_id: str, ended: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real runner takes the queued attempt; its analysis fails, finishes or is cancelled
    while it runs (`jobs.cancel`, as the job card's button does)."""
    calls: list[str] = []
    if ended == "cancelled":
        monkeypatch.setattr(app, "analyze", substitute(calls, until_cancelled=True))
        runner = runner_thread(cfg)
        wait_until(lambda: len(calls) == 1)
        jobs.cancel(job_id, cfg)
        runner.join(timeout=30)
        assert not runner.is_alive()
    else:
        monkeypatch.setattr(app, "analyze", substitute(calls, fail_first=ended == "failed"))
        jobs.run_runner(cfg, exit_when_idle=True, poll_seconds=POLL)
    assert job_file(cfg, job_id)["state"] == ended


def last_lines(chats: Chats) -> list[tuple[str, str]]:
    return [(chat, text.splitlines()[-1]) for chat, text in chats.sent]


@pytest.mark.slow
@pytest.mark.parametrize("ended", ["cancelled", "failed"])
def test_criterion_1_after_jobs_retry_the_loop_sends_one_more_note_when_the_new_attempt_ends(
    loop, setup: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch, ended: str
) -> None:
    job_id = bound_job(setup, make_wav("recordings/a.wav"))
    run_to_end(setup, job_id, ended, monkeypatch)
    chats = NotedChats(loop, setup, None, TIMEOUT)
    assert ticks(loop, chats, 2) == ["delivered", "idle"]

    retried = jobs_cli("retry", job_id)
    assert retried.returncode == 0, retried.stderr
    assert ticks(loop, chats, 2) == ["idle", "idle"]
    run_to_end(setup, job_id, "finished", monkeypatch)
    # criterion 2: the second send times out; its history check must not take the first note
    assert ticks(loop, chats, 3) == ["unconfirmed", "idle", "idle"]

    first, second = f"sound-scout-job:{job_id}", f"sound-scout-job:{job_id}-a2"
    assert last_lines(chats) == [("chat-1", first), ("chat-1", second)]
    assert first not in chats.sent[1][1].splitlines()
    assert chats.checked == [("chat-1", second)]
    status = jobs.status(job_id, setup)
    assert status["state"] == "finished" and status["attempts"] == 2
    assert status["delivery"]["state"] == "delivery_unconfirmed"
    assert status["delivery"]["tag"] == second
    # criterion 3: one note per attempt; the console answers `none` after `deliver-end`
    again = jobs_cli("deliver-next", "--profile=default", "--loop=loop-1", "--json")
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout) == {"action": "none"}

    # criterion 5: `jobs redeliver` of the retried attempt's note keeps its tag; its history
    # check misses the first note, so the loop sends it once
    accepted = jobs_cli("redeliver", job_id)
    assert accepted.returncode == 0, accepted.stderr
    assert ticks(loop, chats, 3) == ["delivered", "idle", "idle"]
    assert last_lines(chats)[2:] == [("chat-1", second)]
    assert chats.checked == [("chat-1", second), ("chat-1", second)]
    assert delivery_of(setup, job_id)["state"] == "delivered"
    assert len(chats.sent) == 3


@pytest.mark.slow
def test_criterion_5_a_stopped_loop_of_a_retried_attempt_is_checked_by_its_own_tag(
    loop, setup: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = bound_job(setup, make_wav("recordings/a.wav"))
    run_to_end(setup, job_id, "failed", monkeypatch)
    chats = NotedChats(loop, setup)
    assert ticks(loop, chats, 1) == ["delivered"]
    assert jobs_cli("retry", job_id).returncode == 0
    run_to_end(setup, job_id, "finished", monkeypatch)
    stopped = jobs_cli(
        "deliver-next", "--profile=default", "--loop=stopped", "--idle-chat=chat-1", "--json"
    )
    assert json.loads(stopped.stdout)["tag"] == f"sound-scout-job:{job_id}-a2"

    assert ticks(loop, chats, 2) == ["unconfirmed", "idle"]

    assert chats.checked == [("chat-1", f"sound-scout-job:{job_id}-a2")]
    assert last_lines(chats) == [("chat-1", f"sound-scout-job:{job_id}")]
    marker = delivery_of(setup, job_id)
    assert marker["state"] == "delivery_unconfirmed"
    assert marker["tag"] == f"sound-scout-job:{job_id}-a2"


@pytest.mark.slow
def test_criterion_4_a_job_without_a_retry_gets_one_note_with_the_job_tag(
    loop, setup: Config, make_wav: WavFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = bound_job(setup, make_wav("recordings/a.wav"))
    run_to_end(setup, job_id, "finished", monkeypatch)
    chats = NotedChats(loop, setup)

    assert ticks(loop, chats, 3) == ["delivered", "idle", "idle"]

    assert last_lines(chats) == [("chat-1", f"sound-scout-job:{job_id}")]
    assert loop._is_note({"role": "user", "content": chats.sent[0][1]}, f"sound-scout-job:{job_id}")
    job = job_file(setup, job_id)
    assert job["delivery"]["tag"] == f"sound-scout-job:{job_id}" and "deliveries" not in job
    again = jobs_cli("deliver-next", "--profile=default", "--loop=loop-1", "--json")
    assert json.loads(again.stdout) == {"action": "none"}


# --- acceptance criteria of #190: no note for a seen attempt, none before the turn ends ------


class Turns:
    """The fake turn check: which chats have a turn, and every chat it was asked about."""

    def __init__(self, *busy: str, error: BaseException | None = None) -> None:
        self.busy = set(busy)
        self.error = error
        self.asked: list[str] = []

    def __call__(self, chat_id: str) -> bool:
        self.asked.append(chat_id)
        if self.error is not None:
            raise self.error
        return chat_id in self.busy


def ended_now(cfg: Config, wav: Path, session_id: str = "chat-1") -> str:
    """A bound job whose attempt 1 ended `finished` just now."""
    job_id = ended_job(cfg, wav, session_id=session_id, state="running")
    run_attempt(cfg, job_id)
    age(cfg, job_id, 0)
    return job_id


def age(cfg: Config, job_id: str, seconds: float) -> None:
    """Move the end of the job's last attempt `seconds` into the past."""
    from datetime import UTC, datetime, timedelta

    job = job_file(cfg, job_id)
    ended = datetime.now(UTC) - timedelta(seconds=seconds)
    job["attempts"][-1]["ended_at"] = ended.isoformat()
    path = cfg.data_root / "jobs" / job_id / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")


@pytest.mark.slow
def test_criterion_2_a_seen_attempt_gets_no_note_and_ends_seen_in_chat(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    from sound_scout import delivery

    job_id = ended_now(setup, make_wav("recordings/a.wav"))
    delivery.mark_seen(job_id, "chat-1", 1, setup)
    chats, turns = Chats(setup), Turns()

    assert ticks(loop, chats, 3, chat_busy=turns) == ["idle"] * 3

    assert chats.sent == [] and turns.asked == []
    recorded = delivery_of(setup, job_id)
    assert recorded["state"] == "seen_in_chat" and recorded["attempt"] == 1


@pytest.mark.slow
def test_criterion_3_a_running_turn_holds_the_note_and_its_end_sends_one(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_now(setup, make_wav("recordings/a.wav"))
    chats, turns = Chats(setup), Turns("chat-1")

    assert ticks(loop, chats, 3, chat_busy=turns) == ["waiting_turn"] * 3
    assert chats.sent == [] and delivery_of(setup, job_id) is None

    turns.busy.clear()
    assert ticks(loop, chats, 2, chat_busy=turns) == ["delivered", "idle"]
    assert [chat for chat, _ in chats.sent] == ["chat-1"]
    assert turns.asked == ["chat-1"] * 4


@pytest.mark.slow
@pytest.mark.parametrize("turns", [Turns("chat-1"), Turns(error=TimeoutError("owner stalled"))])
def test_criterion_3_without_the_turn_signal_the_note_goes_once_after_the_limit(
    loop, setup: Config, make_wav: WavFactory, turns: Turns, caplog: pytest.LogCaptureFixture
) -> None:
    job_id = ended_now(setup, make_wav("recordings/a.wav"))
    chats = Chats(setup)
    caplog.set_level(logging.WARNING)

    assert ticks(loop, chats, 2, chat_busy=turns) == ["waiting_turn"] * 2
    age(setup, job_id, 601)
    assert ticks(loop, chats, 3, chat_busy=turns) == ["delivered", "idle", "idle"]

    assert len(chats.sent) == 1 and len(turns.asked) == 2
    if turns.error is not None:
        assert "turn check failed: TimeoutError" in caplog.text and "chat-1" not in caplog.text


@pytest.mark.slow
def test_criterion_3_a_busy_chat_does_not_hold_the_note_of_another_chat(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    wav = make_wav("recordings/a.wav")
    held = ended_now(setup, wav, "chat-x")
    sent = ended_now(setup, wav, "chat-y")
    chats, turns = Chats(setup), Turns("chat-x")

    assert ticks(loop, chats, 2, chat_busy=turns) == ["delivered", "waiting_turn"]

    assert [(chat, text.splitlines()[-1]) for chat, text in chats.sent] == [
        ("chat-y", f"sound-scout-job:{sent}")
    ]
    assert delivery_of(setup, held) is None


@pytest.mark.slow
def test_criterion_4_a_retry_from_the_card_with_no_turn_gets_one_note(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    from sound_scout import delivery

    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-1", state="running")
    run_attempt(setup, job_id, "failed")
    delivery.mark_seen(job_id, "chat-1", 1, setup)
    chats, turns = Chats(setup), Turns()
    assert ticks(loop, chats, 1, chat_busy=turns) == ["idle"]

    assert jobs_cli("retry", job_id).returncode == 0
    run_attempt(setup, job_id)
    age(setup, job_id, 0)
    assert ticks(loop, chats, 3, chat_busy=turns) == ["delivered", "idle", "idle"]

    assert [(chat, text.splitlines()[-1]) for chat, text in chats.sent] == [
        ("chat-1", f"sound-scout-job:{job_id}-a2")
    ]
    assert turns.asked == ["chat-1"]


def test_a_tick_with_more_busy_chats_than_its_probes_waits(loop) -> None:
    calls = []

    def cli(args):
        calls.append(args)
        return {"action": "probe", "job_id": "a" * 32, "session_id": f"chat-{len(calls)}"}

    outcome = loop.tick(
        owner=lambda: True,
        profile=lambda: "default",
        loop_id="l",
        cli=cli,
        send=None,
        history_has=None,
        chat_busy=lambda chat_id: True,
        state={},
    )

    assert outcome == "waiting_turn" and len(calls) == loop.PROBES_PER_TICK
    assert calls[-1][3:] == [f"--busy-chat=chat-{n}" for n in sorted(range(1, 16), key=str)]


def test_the_turn_check_reads_the_pending_admissions_of_the_lineage_root_on_the_owner_loop(
    loop, monkeypatch
) -> None:
    import asyncio

    owner_loop = asyncio.new_event_loop()
    runner = threading.Thread(target=owner_loop.run_forever, daemon=True)
    runner.start()
    asked = []

    class Admissions:
        def list(self, *, session_id):
            asked.append((session_id, threading.current_thread() is runner))
            return [{"status": "started"}] if session_id == "root-busy" else []

    authority = types.SimpleNamespace(logical_owner=lambda chat: f"root-{chat}")
    owners = {"HOME": (authority, owner_loop)}
    monkeypatch.setitem(sys.modules, "gateway", stub("gateway"))
    monkeypatch.setitem(
        sys.modules, "gateway.session_cron", stub("gateway.session_cron", owner_for_home=owners.get)
    )
    persistence = stub(
        "gateway.session_persistence",
        authority_persistence=lambda auth: types.SimpleNamespace(admissions=Admissions()),
    )
    monkeypatch.setitem(sys.modules, "gateway.session_persistence", persistence)
    monkeypatch.setitem(
        sys.modules, "hermes_constants", stub("hermes_constants", get_hermes_home=lambda: "HOME")
    )
    try:
        assert loop.hermes_chat_busy("busy") is True
        assert loop.hermes_chat_busy("idle") is False
        assert asked == [("root-busy", True), ("root-idle", True)]
        owners.clear()
        with pytest.raises(RuntimeError):
            loop.hermes_chat_busy("busy")
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        runner.join(5)
        owner_loop.close()


def test_start_hands_the_hermes_turn_check_to_the_loop(loop) -> None:
    import inspect

    assert inspect.signature(loop.start).parameters["chat_busy"].default is loop.hermes_chat_busy
