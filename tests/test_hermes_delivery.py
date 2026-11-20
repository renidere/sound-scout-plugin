"""Delivery loop of the Hermes plugin (`hermes/plugin/sound-scout/delivery.py`, #102), without
Hermes.

`cli` is the real `sound-scout` of this venv over a temporary data root; `send` and
`history_has` are fakes that record their calls. No real chat is touched.
"""

import ast
import importlib.util
import json
import logging
import shutil
import sys
import threading
import types
from pathlib import Path

import pytest

from sound_scout import config, jobs
from sound_scout.config import Config
from tests.conftest import WavFactory
from tests.test_delivery import analyzed_job, ended_job, job_file

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
        job_id = text.splitlines()[-1].removeprefix("sound-scout-job:")
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


def ticks(loop, chats: Chats, n: int = 3, *, owner=True, cli=None, state=None, loop_id="loop-1"):
    state = {} if state is None else state
    return [
        loop.tick(
            owner=lambda: owner,
            profile=lambda: "default",
            loop_id=loop_id,
            cli=cli or loop.run_cli,
            send=chats.send,
            history_has=chats.history_has,
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


def test_with_the_owner_one_job_gets_exactly_one_note_in_its_chat(
    loop, setup: Config, make_wav: WavFactory
) -> None:
    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-7")
    chats = Chats(setup)

    assert ticks(loop, chats) == ["delivered", "idle", "idle"]

    assert [chat for chat, _ in chats.sent] == ["chat-7"]
    assert delivery_of(setup, job_id)["state"] == "delivered"
    assert delivery_of(setup, job_id)["confirmed_by"] == "send"


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


@pytest.mark.parametrize(
    ("history", "state"), [(True, "delivered"), (False, "delivery_unconfirmed")]
)
def test_a_marker_of_a_stopped_loop_is_checked_in_history_never_sent(
    loop, setup: Config, make_wav: WavFactory, history: bool, state: str
) -> None:
    from sound_scout import delivery

    job_id = ended_job(setup, make_wav("recordings/a.wav"), session_id="chat-1")
    delivery.next_delivery("default", "dead-loop", setup)
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
    assert "- содержание (ASR): `unavailable` — недоступно" in lines
    assert "- измерения (акустика): `completed` — выполнено" in lines
    assert "- звучание (audio review): `skipped` — пропущено" in lines
    assert "`complete`" not in lines[0]
    for phrase in ("готово", "всё выполнено", "все выполнено", "всё хорошо", "all done"):
        assert phrase not in text.lower()


# --- criterion 4: a deleted or unavailable chat is a delivery error, never another chat -------


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
        def get_messages(self, target, *, latest, limit):
            assert latest is True and limit == loop.HISTORY_SCAN
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
    assert loop.hermes_history_has("c", "sound-scout-job:phys-c") is False


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
