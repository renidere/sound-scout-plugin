"""The Hermes plugin package (#101, decision 0013 § 3.2, § 3.5), loaded without Hermes.

A stub `fastapi` records the routes; `Request` is the real Starlette request, built from an ASGI
scope and a `receive` that hands the body over in chunks. `SOUND_SCOUT_BIN` is the real
`sound-scout` of this venv, so `jobs limits`, `jobs submit`, `status` and `result` run for real
on synthesized audio; `_cli` is replaced only to observe it or to make it time out.
"""

import ast
import asyncio
import contextlib
import dataclasses
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import types
import uuid
from pathlib import Path

import pytest
from starlette.requests import Request

from sound_scout import app, config
from tests.conftest import WavFactory

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "hermes" / "plugin" / "sound-scout"
API = PACKAGE / "dashboard" / "plugin_api.py"
PLUGIN_JS = PACKAGE / "desktop" / "plugin.js"
DECISION = REPO / "docs" / "decisions" / "0015-spike-b-backend-package.md"
CLI = Path(sys.executable).with_name("sound-scout")
ROUTES = {("GET", "/health"), ("POST", "/jobs"), ("GET", "/jobs/{job_id}")}
BOUNDARY = "----SoundScoutTestBoundary7d1f"
MAX_INPUT = 20_000
UPLOAD_NAME = re.compile(r"[0-9a-f]{32}\.upload")
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)


class StubRouter:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], object] = {}

    def _route(self, method: str, path: str):
        def record(handler):
            self.routes[(method, path)] = handler
            return handler

        return record

    def get(self, path: str):
        return self._route("GET", path)

    def post(self, path: str):
        return self._route("POST", path)


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def api(monkeypatch):
    fastapi = _module("fastapi", APIRouter=StubRouter, Request=Request)
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    spec = importlib.util.spec_from_file_location("sound_scout_hermes_plugin_api", API)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_hermes(monkeypatch) -> None:
    for name in ("hermes_constants", "gateway", "gateway.session_cron"):
        monkeypatch.setitem(sys.modules, name, None)  # `import` of a None entry raises ImportError


def with_hermes(monkeypatch, *, home: Path, owner) -> None:
    session_cron = _module("gateway.session_cron", owner_for_home=lambda given: owner)
    monkeypatch.setitem(sys.modules, "gateway", _module("gateway", session_cron=session_cron))
    monkeypatch.setitem(sys.modules, "gateway.session_cron", session_cron)
    monkeypatch.setitem(
        sys.modules, "hermes_constants", _module("hermes_constants", get_hermes_home=lambda: home)
    )


@dataclasses.dataclass
class Station:
    root: Path
    staging: Path
    data: Path
    config: Path

    def write(self, *, staging: bool = True, max_input_bytes: int = MAX_INPUT, **jobs) -> None:
        lines = [
            f"allowed_roots = [{json.dumps(str(self.staging))}]",
            f"data_root = {json.dumps(str(self.data))}",
            f"[budgets]\nmax_duration_seconds = 30\nmax_input_bytes = {max_input_bytes}",
            '[asr]\nmodel = "absent/model"',
            "[jobs]",
            *(f"{key} = {value}" for key, value in jobs.items()),
        ]
        if staging:
            lines.append(f"staging_root = {json.dumps(str(self.staging))}")
        self.config.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def staged(self) -> list[Path]:
        return sorted(self.staging.iterdir())

    def outside(self) -> set[Path]:
        """Every path under the test dir except the staging root and the data root."""
        return {
            path
            for path in self.root.rglob("*")
            if not path.is_relative_to(self.staging) and not path.is_relative_to(self.data)
        }


@pytest.fixture
def station(tmp_path: Path, monkeypatch) -> Station:
    staging, data = tmp_path / "staging", tmp_path / "data"
    staging.mkdir()
    staging.chmod(0o700)
    data.mkdir()
    found = Station(tmp_path, staging, data, tmp_path / "sound-scout.toml")
    found.write()
    monkeypatch.setenv("SOUND_SCOUT_BIN", str(CLI))
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(found.config))
    without_hermes(monkeypatch)
    return found


@pytest.fixture
def audio(make_wav: WavFactory) -> bytes:
    return make_wav("tone.wav", seconds=0.5).read_bytes()


def multipart(
    data: bytes, *, filename: str = "voice.wav", name: str = "file", boundary: str = BOUNDARY
) -> bytes:
    """One part as Desktop `multipartBody` sends it."""
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    )
    return head.encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode()


@dataclasses.dataclass
class Posted:
    answer: dict
    receives: int
    chunks: int


def post(
    api,
    body: bytes,
    *,
    query: str = "",
    chunk: int = 4096,
    length: int | None | str = "auto",
    content_type: str | None = f"multipart/form-data; boundary={BOUNDARY}",
    drive=asyncio.run,
) -> Posted:
    chunks = [body[i : i + chunk] for i in range(0, len(body), chunk)] or [b""]
    receives = []

    async def receive():
        index = len(receives)
        receives.append(index)
        if index < len(chunks):
            more = index < len(chunks) - 1
            return {"type": "http.request", "body": chunks[index], "more_body": more}
        return {"type": "http.disconnect"}

    headers = []
    if content_type is not None:
        headers.append((b"content-type", content_type.encode("latin-1")))
    if length == "auto":
        length = len(body)
    if length is not None:
        headers.append((b"content-length", str(length).encode("ascii")))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/jobs",
        "query_string": query.encode("ascii"),
        "headers": headers,
    }
    handler = api.router.routes[("POST", "/jobs")]
    answer = drive(handler(Request(scope, receive)))
    return Posted(answer, len(receives), len(chunks))


def get_job(api, job_id: str) -> dict:
    return asyncio.run(api.router.routes[("GET", "/jobs/{job_id}")](job_id))


def health(api) -> dict:
    return asyncio.run(api.router.routes[("GET", "/health")]())


def job_json(station: Station, job_id: str) -> dict:
    return json.loads((station.data / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))


def set_job(station: Station, job_id: str, **fields) -> None:
    """Stand in for the runner: it changes the state and appends attempts."""
    path = station.data / "jobs" / job_id / "job.json"
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")) | fields))


def observe_cli(api, monkeypatch) -> list[list[str]]:
    calls = []
    real = api._cli

    def cli(args):
        calls.append(list(args))
        return real(args)

    monkeypatch.setattr(api, "_cli", cli)
    return calls


def refuse_cli(api, monkeypatch) -> None:
    def cli(args):
        raise AssertionError(f"sound-scout must not run: {args}")

    monkeypatch.setattr(api, "_cli", cli)


def assert_refused(answer: dict, error: str) -> None:
    assert answer["ok"] is False, answer
    assert answer["error"] == error, answer
    assert isinstance(answer["detail"], str) and answer["detail"]


# --- routes, health -------------------------------------------------------------------------------


def test_router_has_exactly_the_production_routes(api) -> None:
    assert set(api.router.routes) == ROUTES


def test_health_reports_the_budgets_of_the_configuration_and_a_ready_intake(
    api, station: Station
) -> None:
    station.write(max_input_bytes=12345)

    answer = health(api)

    assert answer["ok"] is True
    assert answer["plugin"] == "sound-scout" and answer["version"] == "0.1.0"
    assert isinstance(answer["pid"], int)
    assert answer["session_owner"] is None
    assert answer["budgets"] == {"max_duration_seconds": 30, "max_input_bytes": 12345}
    assert answer["intake"] == {"ready": True, "error": None, "detail": None}
    assert str(station.staging) not in json.dumps(answer)


@pytest.mark.parametrize("owner", [(object(), object()), None])
def test_health_reports_whether_this_process_owns_the_sessions(
    api, station: Station, monkeypatch, tmp_path: Path, owner
) -> None:
    with_hermes(monkeypatch, home=tmp_path / "hermes-home", owner=owner)
    answer = health(api)
    assert answer["session_owner"] is (owner is not None)
    assert answer["intake"]["ready"] is True


def write_script(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    "case",
    [
        "no_bin",
        "relative_bin",
        "not_executable",
        "exec_format",
        "not_json",
        "exit_2",
        "timeout",
        "config_error",
        "no_staging_root",
        "staging_0755",
        "staging_symlink",
        "staging_in_hermes_home",
    ],
)
def test_health_names_why_the_intake_is_not_ready(
    api, station: Station, monkeypatch, case: str
) -> None:
    bin_dir = station.root / "bin"
    bin_dir.mkdir()
    expected = {
        "no_bin": "not_configured",
        "relative_bin": "not_configured",
        "not_executable": "not_configured",
        "exec_format": "cli_unavailable",
        "not_json": "cli_failed",
        "exit_2": "cli_failed",
        "timeout": "cli_timeout",
        "config_error": "config_error",
        "no_staging_root": "staging_not_configured",
        "staging_0755": "staging_unavailable",
        "staging_symlink": "staging_unavailable",
        "staging_in_hermes_home": "staging_unavailable",
    }[case]
    if case == "no_bin":
        monkeypatch.delenv("SOUND_SCOUT_BIN")
    elif case == "relative_bin":
        monkeypatch.setenv("SOUND_SCOUT_BIN", "sound-scout")
    elif case == "not_executable":
        script = write_script(bin_dir / "sound-scout", "#!/bin/sh\n")
        script.chmod(0o644)
        monkeypatch.setenv("SOUND_SCOUT_BIN", str(script))
    elif case == "exec_format":
        script = write_script(bin_dir / "sound-scout", "\x7fELF not a program\n")
        monkeypatch.setenv("SOUND_SCOUT_BIN", str(script))
    elif case in ("not_json", "exit_2", "timeout"):
        body = {"not_json": "echo nope", "exit_2": "exit 2", "timeout": "exec sleep 5"}[case]
        monkeypatch.setenv(
            "SOUND_SCOUT_BIN", str(write_script(bin_dir / "s", f"#!/bin/sh\n{body}\n"))
        )
        monkeypatch.setattr(api, "CLI_TIMEOUT_S", 0.3)
    elif case == "config_error":
        station.config.write_text("allowed_roots = []\n", encoding="utf-8")
    elif case == "no_staging_root":
        station.write(staging=False)
    elif case == "staging_0755":
        station.staging.chmod(0o755)
    elif case == "staging_symlink":
        real = api._cli
        link = station.root / "link"
        link.symlink_to(station.staging)

        def cli(args):
            returncode, stdout, stderr = real(args)
            answer = json.loads(stdout) | {"staging_root": str(link)}
            return returncode, json.dumps(answer), stderr

        monkeypatch.setattr(api, "_cli", cli)
    elif case == "staging_in_hermes_home":
        with_hermes(monkeypatch, home=station.root, owner=None)

    answer = health(api)

    assert answer["ok"] is True
    assert answer["intake"]["ready"] is False
    assert answer["intake"]["error"] == expected, answer["intake"]
    assert str(station.root) not in json.dumps(answer)
    if case == "config_error":
        assert answer["intake"]["detail"] == api.CONFIG_ERROR_DETAIL
    budgets_known = case.startswith("staging") or case == "no_staging_root"
    assert (answer["budgets"] is not None) is budgets_known


# --- POST /jobs: the upload and the submit --------------------------------------------------------


def test_post_jobs_stages_the_upload_privately_and_submits_it_by_argv(
    api, station: Station, audio: bytes, monkeypatch
) -> None:
    calls = observe_cli(api, monkeypatch)
    query = "request_id=req-1&session_id=chat-1&profile=default&connection=desktop&layers=content"

    first = post(api, multipart(audio), query=query).answer

    assert first["ok"] is True, first
    assert first["request_id"] == "req-1" and first["reused"] is False
    assert first["state"] == "queued"
    assert first["size_bytes"] == len(audio)
    assert first["sha256"] == hashlib.sha256(audio).hexdigest()
    [staged] = station.staged()
    assert UPLOAD_NAME.fullmatch(staged.name)
    assert stat.S_IMODE(staged.stat().st_mode) == 0o600
    assert staged.read_bytes() == audio
    job = job_json(station, first["job_id"])
    assert job["request"]["input"] == {
        "path": str(staged),
        "size_bytes": len(audio),
        "sha256": first["sha256"],
    }
    assert job["request"]["layers"] == ["content"]
    assert job["binding"] == {
        "submitter": "hermes-plugin",
        "connection": "desktop",
        "profile": "default",
        "session_id": "chat-1",
    }
    assert calls[-1] == [
        "submit",
        "--json",
        "--submitter=hermes-plugin",
        "--request-id=req-1",
        "--session-id=chat-1",
        "--profile=default",
        "--connection=desktop",
        "--layers=content",
        "--",
        str(staged),
    ]

    again = post(api, multipart(audio), query=query).answer

    assert again["ok"] is True and again["reused"] is True
    assert again["job_id"] == first["job_id"]
    assert station.staged() == [staged]


def test_post_jobs_without_a_request_id_gets_a_new_one(api, station: Station, audio) -> None:
    answer = post(api, multipart(audio)).answer

    assert answer["ok"] is True, answer
    assert re.fullmatch(r"[0-9a-f]{32}", answer["request_id"])
    assert job_json(station, answer["job_id"])["request_id"] == answer["request_id"]


def test_an_option_like_request_id_stays_one_argv_item(
    api, station: Station, audio: bytes, monkeypatch
) -> None:
    calls = observe_cli(api, monkeypatch)

    answer = post(api, multipart(audio), query="request_id=--layers%3Dx").answer

    assert answer["ok"] is True, answer
    assert "--request-id=--layers=x" in calls[-1]
    job = job_json(station, answer["job_id"])
    assert job["request_id"] == "--layers=x"
    assert job["request"]["layers"] == sorted(app.LAYERS)


def test_get_job_of_a_queued_job_has_no_result_yet(api, station: Station, audio) -> None:
    job_id = post(api, multipart(audio)).answer["job_id"]

    answer = get_job(api, job_id)

    assert answer["ok"] is True
    assert answer["job"]["job_id"] == job_id and answer["job"]["state"] == "queued"
    assert answer["job"]["finished_stages"] == []
    assert answer["result"] is None


@needs_ffmpeg
def test_get_job_answers_the_state_the_finished_stages_and_the_compact_result(
    api, station: Station, make_wav: WavFactory
) -> None:
    wav = make_wav("voice.wav", seconds=1.5).read_bytes()
    station.write(max_input_bytes=len(wav) + 1)
    job_id = post(api, multipart(wav)).answer["job_id"]
    [staged] = station.staged()
    request_id = f"{job_id}-a1"
    analysis = app.analyze(app.AnalyzeRequest(str(staged), request_id=request_id), config.load())
    attempt = {"number": 1, "request_id": request_id, "run_id": analysis.run.run_id}
    set_job(station, job_id, state="finished", attempts=[attempt])

    answer = get_job(api, job_id)

    assert answer["ok"] is True, answer
    assert answer["job"]["state"] == "finished"
    stages = answer["job"]["finished_stages"]
    assert [item["stage"] for item in stages] == list(app.WORKER_STAGES)
    assert {"stage": "prepare", "status": "completed"} in stages
    assert answer["result"]["job_state"] == "finished" and answer["result"]["attempt"] == 1
    assert answer["result"]["status"] == analysis.result["status"] == "partial"
    assert len(json.dumps(answer["result"], ensure_ascii=False).encode("utf-8")) <= 4096
    assert str(station.root) not in json.dumps(answer)


@pytest.mark.parametrize("job_id", ["..%2F..", "../..", "--json", "F" * 32, "a" * 31, "a" * 33])
def test_get_job_with_a_malformed_id_is_job_not_found_without_running_the_cli(
    api, station: Station, monkeypatch, job_id: str
) -> None:
    refuse_cli(api, monkeypatch)
    assert_refused(get_job(api, job_id), "job_not_found")


def test_get_job_of_an_unknown_job_is_job_not_found_from_the_cli(api, station: Station) -> None:
    answer = get_job(api, uuid.uuid4().hex)
    assert_refused(answer, "job_not_found")
    assert str(station.root) not in answer["detail"]


# --- size -----------------------------------------------------------------------------------------


def test_a_content_length_over_the_limit_is_refused_before_reading_the_body(
    api, station: Station
) -> None:
    body = multipart(b"\0" * (MAX_INPUT + 70 * 1024))

    posted = post(api, body)

    assert_refused(posted.answer, "input_too_large")
    assert posted.receives == 0
    assert station.staged() == []


@pytest.mark.parametrize("length", [None, 500])
def test_an_oversize_body_without_a_true_length_is_refused_while_reading(
    api, station: Station, length: int | None
) -> None:
    body = multipart(b"\1" * (MAX_INPUT * 4))

    posted = post(api, body, length=length, chunk=1024)

    assert_refused(posted.answer, "input_too_large")
    assert posted.receives < posted.chunks
    assert station.staged() == []
    assert "request_id" in posted.answer


def test_a_file_of_exactly_max_input_bytes_is_accepted(
    api, station: Station, make_wav: WavFactory
) -> None:
    wav = make_wav("voice.wav", seconds=0.5).read_bytes()
    station.write(max_input_bytes=len(wav))

    answer = post(api, multipart(wav), length=None).answer

    assert answer["ok"] is True, answer
    assert answer["size_bytes"] == len(wav)
    station.write(max_input_bytes=len(wav) - 1)
    assert_refused(post(api, multipart(wav)).answer, "input_too_large")
    assert len(station.staged()) == 1


# --- symlinks and traversal -----------------------------------------------------------------------


def test_a_staging_root_swapped_for_a_symlink_after_limits_is_not_followed(
    api, station: Station, audio: bytes, monkeypatch
) -> None:
    elsewhere = station.root / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    real = api._cli

    def cli(args):
        answer = real(args)
        if args[0] == "limits":
            station.staging.rename(station.root / "moved")
            station.staging.symlink_to(elsewhere)
        return answer

    monkeypatch.setattr(api, "_cli", cli)

    posted = post(api, multipart(audio))

    assert_refused(posted.answer, "staging_unavailable")
    assert list(elsewhere.iterdir()) == []
    assert list((station.root / "moved").iterdir()) == []
    assert str(station.root) not in posted.answer["detail"]


@pytest.mark.parametrize("case", ["mode_0750", "mode_0755", "in_hermes_home", "not_allowed"])
def test_criterion_2_post_jobs_writes_only_into_a_private_allowed_staging_root_outside_hermes(
    api, station: Station, audio: bytes, monkeypatch, case: str
) -> None:
    expected = "config_error" if case == "not_allowed" else "staging_unavailable"
    if case.startswith("mode_"):
        station.staging.chmod(int(case.removeprefix("mode_"), 8))
    elif case == "in_hermes_home":
        with_hermes(monkeypatch, home=station.root, owner=None)
    else:
        elsewhere = station.root / "recordings"
        elsewhere.mkdir()
        station.write()
        text = station.config.read_text(encoding="utf-8")
        allowed = f"allowed_roots = [{json.dumps(str(station.staging))}]"
        station.config.write_text(
            text.replace(allowed, f"allowed_roots = [{json.dumps(str(elsewhere))}]"),
            encoding="utf-8",
        )
    before = station.outside()

    answer = post(api, multipart(audio)).answer

    assert_refused(answer, expected)
    assert station.staged() == []
    assert station.outside() == before
    assert not (station.data / "jobs").exists() or not list(
        (station.data / "jobs").glob("*/job.json")
    )
    assert str(station.root) not in json.dumps(answer)


@pytest.mark.parametrize("planted", ["symlink", "file"])
def test_an_upload_name_that_exists_is_a_conflict_and_nothing_is_written_through_it(
    api, station: Station, audio: bytes, monkeypatch, planted: str
) -> None:
    fixed = uuid.UUID(int=7)
    monkeypatch.setattr(api.uuid, "uuid4", lambda: fixed)
    victim = station.root / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    target = station.staging / f"{fixed.hex}.upload"
    if planted == "symlink":
        target.symlink_to(victim)
    else:
        target.write_text("earlier upload", encoding="utf-8")

    answer = post(api, multipart(audio), query="request_id=r").answer

    assert_refused(answer, "staging_conflict")
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert target.is_symlink() is (planted == "symlink")
    if planted == "file":
        assert target.read_text(encoding="utf-8") == "earlier upload"


@pytest.mark.parametrize(
    "filename", ["../../escape.wav", "/tmp/sound-scout-escape.wav", "x.wav\r\nX-Evil: 1"]
)
def test_the_client_file_name_is_never_used(
    api, station: Station, audio: bytes, filename: str
) -> None:
    before = station.outside()

    answer = post(api, multipart(audio, filename=filename)).answer

    assert answer["ok"] is True, answer
    [staged] = station.staged()
    assert UPLOAD_NAME.fullmatch(staged.name)
    assert station.outside() == before
    assert not Path("/tmp/sound-scout-escape.wav").exists()
    assert "escape" not in json.dumps(job_json(station, answer["job_id"]))


# --- multipart and query rules --------------------------------------------------------------------


def two_parts(audio: bytes) -> bytes:
    first = multipart(audio).removesuffix(f"--{BOUNDARY}--\r\n".encode())
    return first + multipart(audio, filename="second.wav")


@pytest.mark.parametrize(
    ("make", "content_type", "error"),
    [
        (lambda audio: two_parts(audio), None, "bad_multipart"),
        (lambda audio: multipart(audio, name="audio"), None, "bad_multipart"),
        (lambda audio: multipart(audio)[:-20], None, "bad_multipart"),
        (lambda audio: f"--{BOUNDARY}--\r\n".encode(), None, "bad_multipart"),
        (lambda audio: b"garbage without a boundary", None, "bad_multipart"),
        (lambda audio: multipart(b""), None, "empty_upload"),
        (lambda audio: audio, "audio/wav", "bad_multipart"),
        (lambda audio: multipart(audio), "multipart/form-data", "bad_multipart"),
    ],
    ids=[
        "two_parts",
        "not_file",
        "truncated",
        "no_part",
        "garbage",
        "empty",
        "not_multipart",
        "no_boundary",
    ],
)
def test_a_body_that_is_not_one_file_part_is_refused_and_leaves_nothing(
    api, station: Station, audio: bytes, make, content_type, error: str
) -> None:
    kwargs = {} if content_type is None else {"content_type": content_type}

    answer = post(api, make(audio), **kwargs).answer

    assert_refused(answer, error)
    assert station.staged() == []
    assert not (station.data / "jobs").exists() or not list(
        (station.data / "jobs").glob("*/job.json")
    )


@pytest.mark.parametrize(("length", "accepted"), [(256, True), (257, False)])
def test_a_boundary_the_parser_refuses_is_bad_multipart_before_the_upload_exists(
    api, station: Station, audio: bytes, monkeypatch, length: int, accepted: bool
) -> None:
    boundary = "b" * length
    created, create = [], api._create_upload
    monkeypatch.setattr(api, "_create_upload", lambda *args: created.append(args) or create(*args))

    answer = post(
        api,
        multipart(audio, boundary=boundary),
        content_type=f"multipart/form-data; boundary={boundary}",
    ).answer

    if accepted:
        assert answer["ok"] is True, answer
        assert len(station.staged()) == 1
    else:
        assert_refused(answer, "bad_multipart")
        assert created == []
        assert station.staged() == []


@pytest.mark.parametrize(
    "query",
    ["user=x", "request_id=a&request_id=b", "request_id=", "session_id=a%0Ab", "path=/etc/passwd"],
)
def test_a_bad_query_is_invalid_request_before_anything_runs(
    api, station: Station, audio: bytes, monkeypatch, query: str
) -> None:
    refuse_cli(api, monkeypatch)
    assert_refused(post(api, multipart(audio), query=query).answer, "invalid_request")
    assert station.staged() == []


def test_a_second_upload_while_one_is_in_progress_is_intake_busy(
    api, station: Station, audio: bytes, monkeypatch
) -> None:
    refuse_cli(api, monkeypatch)
    assert api._intake_lock.acquire(blocking=False)
    try:
        assert_refused(post(api, multipart(audio)).answer, "intake_busy")
    finally:
        api._intake_lock.release()


# --- off the event loop, CLI refusals -------------------------------------------------------------


def test_copy_hash_fsync_and_the_cli_run_off_the_event_loop(
    api, station: Station, audio: bytes, monkeypatch
) -> None:
    threads = {}

    def recording(name, function):
        def wrapper(*args, **kwargs):
            threads.setdefault(name, set()).add(threading.get_ident())
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(api._Sink, "feed", recording("feed", api._Sink.feed))
    monkeypatch.setattr(api.os, "fsync", recording("fsync", os.fsync))
    monkeypatch.setattr(api, "_cli", recording("cli", api._cli))
    monkeypatch.setattr(api, "_open_staging", recording("staging", api._open_staging))
    monkeypatch.setattr(api, "_create_upload", recording("create", api._create_upload))

    answer = post(api, multipart(audio), chunk=1024).answer

    assert answer["ok"] is True, answer
    assert set(threads) == {"feed", "fsync", "cli", "staging", "create"}
    loop_thread = threading.get_ident()  # asyncio.run drives the loop in this thread
    assert all(loop_thread not in idents for idents in threads.values()), threads


def test_a_cli_refusal_of_the_submit_removes_the_staged_copy(
    api, station: Station, make_wav: WavFactory
) -> None:
    station.write(max_queued_jobs=1)
    first = make_wav("a.wav", seconds=0.5).read_bytes()
    second = make_wav("b.wav", seconds=0.5, frequency=330.0).read_bytes()
    accepted = post(api, multipart(first)).answer
    [kept] = station.staged()

    refused = post(api, multipart(second), query="request_id=r-2").answer

    assert_refused(refused, "queue_full")
    assert refused["request_id"] == "r-2"
    assert station.staged() == [kept]
    assert accepted["ok"] is True
    assert str(station.root) not in json.dumps(refused)


@pytest.mark.parametrize(
    ("failure", "error"),
    [(subprocess.TimeoutExpired("sound-scout", 30), "cli_timeout"), ((2, "", ""), "cli_failed")],
)
def test_a_submit_that_may_have_created_the_job_keeps_the_staged_copy(
    api, station: Station, audio: bytes, monkeypatch, failure, error: str
) -> None:
    real = api._cli

    def cli(args):
        if args[0] != "submit":
            return real(args)
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(api, "_cli", cli)

    answer = post(api, multipart(audio), query="request_id=r-9").answer

    assert_refused(answer, error)
    assert answer["request_id"] == "r-9"
    [staged] = station.staged()
    assert staged.read_bytes() == audio


def test_a_request_cancelled_while_the_submit_runs_keeps_the_staged_copy(
    api, station: Station, audio: bytes, monkeypatch
) -> None:
    real = api._cli
    entered, release = threading.Event(), threading.Event()

    def cli(args):
        if args[0] == "submit":
            entered.set()
            assert release.wait(10)
        return real(args)

    def cancel_while_submitting(coroutine):
        async def main():
            task = asyncio.ensure_future(coroutine)
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            release.set()
            return task.cancelled()

        return asyncio.run(main())

    monkeypatch.setattr(api, "_cli", cli)

    cancelled = post(api, multipart(audio), drive=cancel_while_submitting).answer

    assert cancelled is True
    [staged] = station.staged()
    [job] = (station.data / "jobs").glob("*/job.json")
    assert json.loads(job.read_text(encoding="utf-8"))["request"]["input"]["path"] == str(staged)
    assert api._intake_lock.acquire(blocking=False)
    api._intake_lock.release()


@pytest.mark.parametrize("unlink_fails", [False, True])
def test_a_failed_fsync_of_the_upload_is_staging_unavailable_not_an_http_error(
    api, station: Station, audio: bytes, monkeypatch, unlink_fails: bool
) -> None:
    calls = observe_cli(api, monkeypatch)
    real_unlink = os.unlink

    def fsync(fd):
        raise OSError(errno.EIO, "Input/output error")

    def unlink(path, *, dir_fd=None):
        if unlink_fails and dir_fd is not None:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(api.os, "fsync", fsync)
    monkeypatch.setattr(api.os, "unlink", unlink)

    answer = post(api, multipart(audio), query="request_id=r-5").answer

    assert_refused(answer, "staging_unavailable")
    assert answer["request_id"] == "r-5"
    assert "Input/output error" in answer["detail"]
    assert [call[0] for call in calls] == ["limits"]
    assert len(station.staged()) == (1 if unlink_fails else 0)


def test_cli_runs_by_argv_with_an_allowlisted_environment(api, station: Station, monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setenv("OPENAI_API_KEY", "secret-from-hermes-env")
    monkeypatch.setattr(api.subprocess, "run", run)

    assert api._cli(["status", "--json", "--", "f" * 32]) == (0, "{}", "")

    assert seen["argv"] == [str(CLI), "jobs", "status", "--json", "--", "f" * 32]
    assert "shell" not in seen
    assert seen["timeout"] == api.CLI_TIMEOUT_S
    assert set(seen["env"]) <= {*api.CHILD_ENV, config.CONFIG_PATH_ENV}
    assert seen["env"][config.CONFIG_PATH_ENV] == str(station.config)


# --- the package ----------------------------------------------------------------------------------


def test_plugin_api_never_reads_hermes_attachments_or_paths_of_the_hermes_home() -> None:
    tree = ast.parse(API.read_text(encoding="utf-8"))
    strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not [text for text in strings if "attachments" in text.lower() or "~" in text]
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"expanduser", "home", "listdir", "scandir", "walk", "read_text"}


def test_plugin_api_imports_only_the_stdlib_fastapi_and_python_multipart() -> None:
    source = API.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 11))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module.split(".")[0])
    assert top_level - set(sys.stdlib_module_names) == {"fastapi", "python_multipart"}
    assert "sound_scout" not in source.replace("sound-scout", "")
    assert "/spike/" not in source


def _rest_calls(source: str) -> set[tuple[str, str]]:
    pattern = re.compile(
        r"ctx\.rest\(\s*[`'](?P<path>[^`']+)[`'](?:\s*,\s*\{\s*method:\s*'(?P<m>\w+)')?"
    )
    return {(match["m"] or "GET", match["path"]) for match in pattern.finditer(source)}


def test_desktop_plugin_calls_health_and_claims_the_directive(api) -> None:
    source = PLUGIN_JS.read_text(encoding="utf-8")
    assert "id: 'sound-scout'" in source and "name: 'Sound Scout'" in source
    imports = set(re.findall(r"^import .* from '([^']+)'", source, re.MULTILINE))
    assert imports == {"@hermes/plugin-sdk", "react/jsx-runtime"}
    assert source.count("import ") == 2
    assert _rest_calls(source) == {("GET", "/health")}
    assert _rest_calls(source) <= set(api.router.routes)
    for gone in ("/spike/", "deliver", "runner", "lastToken"):
        assert gone not in source
    assert f"const DIRECTIVE = '{api.DIRECTIVE}'" in source
    assert "name: DIRECTIVE" in source
    job_pattern = re.search(r"const SPIKE_JOB = /(.+)/\n", source)[1]
    assert re.fullmatch(job_pattern, f"spike-{uuid.uuid4().hex}")
    assert not re.fullmatch(job_pattern, "spike-../../etc")


def test_package_manifests_name_the_plugin_its_api_and_its_settings() -> None:
    manifest = json.loads((PACKAGE / "dashboard" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "sound-scout"
    assert manifest["api"] == API.name
    assert manifest["version"] == "0.1.0"
    assert manifest["tab"]["hidden"] is True
    plugin_yaml = (PACKAGE / "plugin.yaml").read_text(encoding="utf-8")
    assert re.search(r"^name: sound-scout$", plugin_yaml, re.MULTILINE)
    assert re.search(r"^version: 0\.1\.0$", plugin_yaml, re.MULTILINE)
    assert "spike" not in plugin_yaml.lower()
    assert not re.search(r"^kind:", plugin_yaml, re.MULTILINE)
    required = re.findall(r"^  - name: (\w+)\n    description: \S", plugin_yaml, re.MULTILINE)
    assert required == ["SOUND_SCOUT_BIN", "SOUND_SCOUT_CONFIG"]


def test_agent_half_registers_nothing(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "sound_scout_plugin_agent",
        PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.register(object()) is None


def _section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1 : end])


def test_criterion_2_instruction_enables_in_process_and_removes_the_package() -> None:
    text = DECISION.read_text(encoding="utf-8")
    instruction = _section(text, "## Инструкция владельцу")
    assert "hermes config get plugins.isolation" in instruction
    assert "`in_process`" in instruction
    assert "`host`" in instruction and "остановиться" in instruction
    install = instruction.index("cp -r <checkout sound-scout-plugin>/hermes/plugin/sound-scout")
    enable = instruction.index("hermes plugins enable sound-scout")
    restart = instruction.index("systemctl --user restart hermes-autonomous.service")
    assert install < enable < restart
    removal = _section(text, "## Удаление")
    remove = removal.index("hermes plugins remove sound-scout")
    assert remove < removal.index("systemctl --user restart hermes-autonomous.service")
    assert "hermes plugins list" in removal
    assert "desktop-plugins\\sound-scout" in removal


MOUNT_LINE = "Mounted plugin API routes: /api/plugins/sound-scout/"


def _install_step(text: str) -> str:
    lines = text.splitlines()
    start = lines.index("### 2. Установка (Linux, backend)")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("#"))
    return "\n".join(lines[start + 1 : end])


def _journal_lookups(step: str) -> list[str]:
    return [line for line in step.splitlines() if MOUNT_LINE in line and "journalctl" in line]


def test_install_step_finds_the_mount_line_in_the_log_files_not_the_journal() -> None:
    step = _install_step(DECISION.read_text(encoding="utf-8"))
    assert _journal_lookups(step) == []
    for log in ("gui.log", "agent.log"):
        assert f'grep "{MOUNT_LINE}" ~/.hermes/logs/{log}' in step
    assert "`hermes serve`" in step and "`hermes gateway run`" in step


def test_journal_lookups_flag_a_journal_grep_for_the_mount_line() -> None:
    journal = f'journalctl --user -u hermes-autonomous.service | grep "{MOUNT_LINE}"'
    assert _journal_lookups(journal) == [journal]
    assert _journal_lookups(f'grep "{MOUNT_LINE}" ~/.hermes/logs/gui.log') == []
    assert _journal_lookups("journalctl --user -u hermes-autonomous.service -n 50") == []


def test_criterion_6_decision_has_a_results_table_pending_the_owner() -> None:
    text = DECISION.read_text(encoding="utf-8")
    assert "результаты ожидают прогона владельцем" in text.splitlines()[2]
    rows = [
        line.split("|")[1:-1]
        for line in _section(text, "## Результаты").splitlines()
        if line.startswith("| ") and not line.startswith("| критерий")
    ]
    by_criterion = {cells[0].strip().split(".")[0]: cells[2].strip() for cells in rows}
    assert {"1", "2", "3", "4а", "4б", "4в", "5а", "5б", "5в", "6"} <= set(by_criterion)
    for criterion in ("3", "4а", "4б", "4в", "5а", "5б", "5в", "6"):
        assert by_criterion[criterion] == "ожидает прогона владельцем"
