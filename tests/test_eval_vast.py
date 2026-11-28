"""GPU stack of #147 against the fake provider API on 127.0.0.1: the key, the client, the one
create, the watchdog, the journal, the download check and the dry run.

Every test uses `vast_fake.FakeVast` and a key file in a temp dir. No test reaches Vast.ai, reads
the real key file or arms a real timer: the fakes prove the contract of the stack only.
"""

import contextlib
import dataclasses
import errno
import fcntl
import gc
import hashlib
import io
import itertools
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import wave
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config
from sound_scout.eval import gate0, gpu_ext, vast, vast_fake
from sound_scout.eval.gpu_host import hostrun
from sound_scout.smoke import speech_like_control

REPO = Path(__file__).resolve().parent.parent
DRY_STEPS = ["create", "ready", "weights", "smoke", "run", "extended", "download", "destroy"]


@pytest.fixture
def key_file(tmp_path: Path) -> tuple[Path, str]:
    key = "k3y-" + "x7" * 20
    path = tmp_path / "vast_api_key"
    path.write_text(key + "\n")
    path.chmod(0o600)
    return path, key


@pytest.fixture
def fake(key_file):
    server = vast_fake.FakeVast(key_file[1]).start()
    yield server
    server.stop()


def _journal(tmp_path: Path, fake, key_file, *, deadline_in: float = 3600.0, **extra):
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700, exist_ok=True)
    attempt_id = vast.new_attempt_id()
    journal = vast.Journal.create(gpu, attempt_id, key_file[1])
    opened = {
        "attempt_id": attempt_id,
        "label": vast.LABEL_PREFIX + attempt_id,
        "deadline_unix": 1_000_000.0 + deadline_in,
        "api_base": fake.url,
        "key_path": str(key_file[0]),
        "initial_credit": 10.0,
        "early_stop_usd": 2.5,
        **extra,
    }
    journal.append("opened", **opened)
    journal.append("armed", units=["u-watch", "u-cutoff"], argv=[], dry_run=True)
    return journal, opened


def _package(**changes) -> dict:
    return {
        "offer_id": vast_fake.OFFER_ID,
        "envelope": {"dph_total": {"lte": 1.533}, "num_gpus": {"eq": 1}},
        "ceiling_usd": 4.0,
        "early_stop_usd": 2.5,
        "deadline_minutes": 150,
        "disk_gb": 150,
        "in_gb": 102.95,
        "out_gb": 1.0,
        **changes,
    }


def _no_system(argv):
    raise AssertionError(f"no system command in this test: {argv}")


def _tick(journal, now, *, beat=True, **kw):
    """One watchdog tick at `now`; the agent's heartbeat is fresh unless `beat` is False."""
    if beat:
        (journal.dir / vast.HEARTBEAT_FILE).write_text(str(now))
    return vast.watch_tick(journal, vast._api_from_journal, runner=_no_system, now=now, **kw)


def _events(journal) -> list[str]:
    return [event["event"] for event in journal.replay()]


# --- the key -------------------------------------------------------------------------------


def test_key_file_0600_is_read_without_its_newline(key_file) -> None:
    assert vast.read_api_key(key_file[0]) == key_file[1]


@pytest.mark.parametrize(
    ("content", "mode"),
    [
        ("k3y-" + "x7" * 20, 0o644),
        ("", 0o600),
        ("k3y-x7x7x7x7 x7x7x7x7x7x7", 0o600),
        ("k" * 600, 0o600),
    ],
    ids=["mode-0644", "empty", "space", "over-512-bytes"],
)
def test_bad_key_files_are_refused_and_the_error_never_holds_the_key(
    tmp_path: Path, content, mode
) -> None:
    path = tmp_path / "key"
    path.write_text(content)
    path.chmod(mode)
    with pytest.raises(vast.KeyFileError) as caught:
        vast.read_api_key(path)
    for piece in {part for part in re.split(r"\s+", content) if len(part) >= 6}:
        assert piece not in str(caught.value)


def test_a_symlinked_key_file_is_refused(key_file, tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(key_file[0])
    with pytest.raises(vast.KeyFileError, match="cannot open"):
        vast.read_api_key(link)


def test_redact_replaces_every_copy_of_the_key() -> None:
    assert vast.redact("a SECRETKEY b SECRETKEY", "SECRETKEY") == "a <redacted> b <redacted>"
    assert vast.redact("text", None) == "text"


class _Echo(BaseHTTPRequestHandler):
    """Answers 400 with the Authorization header in the body, like a careless provider."""

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        body = json.dumps({"error": f"bad {self.headers.get('Authorization')}"}).encode()
        self.send_response(400)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    return server


def test_a_provider_error_body_with_the_key_reaches_no_exception_and_no_journal(
    tmp_path: Path, key_file, fake
) -> None:
    echo = _serve(_Echo)
    try:
        base = f"http://127.0.0.1:{echo.server_address[1]}"
        with pytest.raises(vast.ApiError) as caught:
            vast.Api(base, key_file[1]).user()
        assert key_file[1] not in str(caught.value) and "<redacted>" in str(caught.value)
        journal, _ = _journal(tmp_path, fake, key_file, api_base=base)
        journal.append("create_requested")
        assert _tick(journal, 1_000_001.0) == "error"
    finally:
        echo.shutdown()
    text = journal.path.read_text()
    assert "watch_error" in text and key_file[1] not in text


# --- the client ----------------------------------------------------------------------------


def test_the_key_goes_only_in_an_unredirected_header(monkeypatch, key_file, fake) -> None:
    seen = []

    class Spy(urllib.request.Request):
        def add_header(self, name, value):
            seen.append(("redirected", name))
            super().add_header(name, value)

        def add_unredirected_header(self, name, value):
            seen.append(("unredirected", name))
            super().add_unredirected_header(name, value)

    monkeypatch.setattr(vast.urllib.request, "Request", Spy)
    vast.Api(fake.url, key_file[1]).user()
    assert ("unredirected", "Authorization") in seen
    assert ("redirected", "Authorization") not in seen
    assert fake.log == [{"method": "GET", "path": "/api/v0/users/current/", "auth": True}]


@pytest.mark.slow
def test_a_redirect_is_an_error_and_the_second_server_gets_nothing(key_file) -> None:
    second = vast_fake.FakeVast(key_file[1]).start()
    first = vast_fake.FakeVast(key_file[1], redirect_to=second.url).start()
    try:
        with pytest.raises(vast.Redirected):
            vast.Api(first.url, key_file[1]).user()
        assert first.count("GET") == 1 and second.log == []
    finally:
        first.stop()
        second.stop()


def test_proxy_variables_are_ignored(monkeypatch, key_file, fake) -> None:
    proxy = vast_fake.FakeVast(key_file[1]).start()
    try:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
            monkeypatch.setenv(name, proxy.url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        assert vast.Api(fake.url, key_file[1]).user() == {"credit": 10.0}
        assert proxy.log == []
    finally:
        proxy.stop()


@pytest.mark.parametrize(
    "base", ["http://192.0.2.1:8080", "https://example.com", "http://127.0.0.1:1/x", "ftp://::1"]
)
def test_only_the_provider_or_loopback_http_is_a_base(base) -> None:
    with pytest.raises(vast.VastError, match="API base"):
        vast.Api(base, "k" * 20)


def test_the_provider_base_is_accepted_without_any_call() -> None:
    assert vast.Api(vast.API_BASE + "/", "k" * 20).base == vast.API_BASE


def test_instance_list_reads_every_page_and_keeps_only_the_label(key_file, fake) -> None:
    ids = [fake.add_instance("mine") for _ in range(3)]
    fake.add_instance("theirs")
    api = vast.Api(fake.url, key_file[1])
    assert [item["id"] for item in api.instances("mine")] == ids
    assert len(api.instances()) == 4
    assert fake.count("GET", "/api/v1/instances") == 3 + 4
    assert {entry["path"] for entry in fake.log} == {"/api/v1/instances/"}


def test_the_fake_redirects_the_instance_list_without_the_slash_and_the_api_refuses(
    key_file, fake
) -> None:
    api = vast.Api(fake.url, key_file[1])
    with pytest.raises(vast.Redirected, match="301"):
        api.call("GET", "/api/v1/instances")
    with pytest.raises(vast.Redirected, match="301"):
        api.call("GET", "/api/v1/instances", query={"select_filters": "{}"})
    assert fake.count("GET") == 2


def test_offer_is_read_by_ask_contract_id(monkeypatch, key_file, fake) -> None:
    api = vast.Api(fake.url, key_file[1])
    sent = []
    real_call = api.call
    monkeypatch.setattr(
        api,
        "call",
        lambda method, path, body=None, query=None: (
            sent.append((method, path, body)) or real_call(method, path, body, query)
        ),
    )
    assert api.offer(vast_fake.OFFER_ID, 60)["ask_contract_id"] == vast_fake.OFFER_ID
    method, path, body = sent[0]
    assert (method, path) == ("POST", "/api/v0/bundles/")
    assert body["ask_contract_id"] == {"eq": vast_fake.OFFER_ID} and "id" not in body
    with pytest.raises(vast.VastError, match="offer 7 is not available"):
        api.offer(7, 60)


def test_offer_sends_post_bundles_with_ask_contract_id_and_returns_that_offer(key_file) -> None:
    seen = []
    other = {**vast_fake.OFFER, "id": 99, "ask_contract_id": 99}

    class Recorder(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.command, self.path, body))
            wanted = (body.get("ask_contract_id") or {}).get("eq")
            offers = [o for o in (other, vast_fake.OFFER) if o["ask_contract_id"] == wanted]
            data = json.dumps({"offers": offers}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        api = vast.Api(f"http://127.0.0.1:{server.server_port}", key_file[1])
        assert api.offer(vast_fake.OFFER_ID, 60) == vast_fake.OFFER
    finally:
        server.shutdown()
        server.server_close()
    [(method, path, body)] = seen
    assert (method, path) == ("POST", "/api/v0/bundles/")
    assert body["ask_contract_id"] == {"eq": vast_fake.OFFER_ID} and "id" not in body


def test_offer_refuses_an_answer_for_another_offer(monkeypatch, key_file) -> None:
    api = vast.Api("http://127.0.0.1:9", key_file[1])
    other = {**vast_fake.OFFER, "id": 99, "ask_contract_id": 99}
    monkeypatch.setattr(api, "call", lambda *args, **kwargs: {"offers": [other]})
    with pytest.raises(vast.VastError, match=f"offer {vast_fake.OFFER_ID} is not available"):
        api.offer(vast_fake.OFFER_ID, 60)


def test_the_fake_has_no_offer_for_a_query_by_id(key_file, fake) -> None:
    api = vast.Api(fake.url, key_file[1])
    by_id = {"id": {"eq": vast_fake.OFFER_ID}, "type": "ondemand", "limit": 1}
    assert api.call("POST", "/api/v0/bundles/", by_id) == {"offers": []}
    by_contract = {"ask_contract_id": {"eq": vast_fake.OFFER_ID}, "limit": 1}
    assert api.call("POST", "/api/v0/bundles/", by_contract)["offers"] == [vast_fake.OFFER]


# --- the one create ------------------------------------------------------------------------


def test_payload_has_no_environment_the_fixed_onstart_and_no_key(key_file) -> None:
    payload = vast.create_payload(_package(), "sound-scout-gpu-x", 1795318954)
    assert payload["env"] == {}
    assert payload["onstart"] == vast.ONSTART.format(until_unix=1795318954)
    assert payload["image"] == vast.IMAGE_REF
    assert payload["runtype"] == "ssh_direct" and payload["cancel_unavail"] is True
    vast.check_no_secret(payload, key_file[1])
    with pytest.raises(vast.SecretLeak):
        vast.check_no_secret({**payload, "env": {"X": key_file[1]}}, key_file[1])
    for name in vast.SECRET_NAMES:
        with pytest.raises(vast.SecretLeak, match=name):
            vast.check_no_secret({**payload, "env": {name: "x"}}, key_file[1])


@pytest.mark.slow
@pytest.mark.parametrize(("until_in", "slept"), [(-100, False), (2, True)], ids=["past", "future"])
def test_onstart_sleeps_until_the_absolute_time_and_not_from_container_start(
    until_in, slept
) -> None:
    script = vast.ONSTART.format(until_unix=int(time.time()) + until_in)
    script = script.replace("kill -TERM 1", "echo fired")
    started = time.monotonic()
    done = subprocess.run(["sh", "-c", script], capture_output=True, timeout=30)
    elapsed = time.monotonic() - started
    assert done.returncode == 0 and done.stdout == b"fired\n", done.stderr
    assert (elapsed >= 1.0) is slept, elapsed


def test_a_file_with_the_key_never_goes_to_the_host(tmp_path: Path, key_file) -> None:
    leaky = tmp_path / "leaky.txt"
    leaky.write_text(f"token {key_file[1]}\n")
    host = vast.LocalHost(tmp_path / "root", {})
    (tmp_path / "root").mkdir()
    with pytest.raises(vast.SecretLeak):
        host.put(leaky, "/workspace/leaky.txt", key_file[1])
    assert not (tmp_path / "root" / "leaky.txt").exists()
    remote = vast.RemoteHost(tmp_path, "203.0.113.7", 22)
    with pytest.raises(vast.SecretLeak):
        remote.put(leaky, "/workspace/leaky.txt", key_file[1])


def test_the_host_files_carry_no_secret_name() -> None:
    for name in ("hostrun.py", "sitecustomize.py"):
        vast.check_no_secret((vast.HOST_FILES_DIR / name).read_bytes(), None)


def test_create_success_records_the_id_and_a_second_create_is_impossible(
    tmp_path: Path, key_file, fake
) -> None:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1])
    instance_id = vast.create_instance(api, journal, _package(), _no_system)
    assert instance_id == vast_fake.FIRST_INSTANCE_ID
    assert _events(journal)[-2:] == ["create_requested", "create_response"]
    with pytest.raises(vast.VastError, match="already sent its one create"):
        vast.create_instance(api, journal, _package(), _no_system)
    assert fake.count("PUT") == 1


@pytest.mark.parametrize("mode", ["drop", "500", "timeout"])
def test_a_lost_create_is_unknown_never_retried_and_the_tick_destroys_by_label(
    tmp_path: Path, key_file, fake, mode
) -> None:
    fake.create_mode = mode
    fake.timeout_s = 2.0
    journal, opened = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1], timeout=0.5)
    with pytest.raises(vast.CreateOutcomeUnknown):
        vast.create_instance(api, journal, _package(), _no_system)
    with pytest.raises(vast.VastError, match="already sent"):
        vast.create_instance(api, journal, _package(), _no_system)
    assert "create_response" not in _events(journal)
    assert _tick(journal, 1_000_001.0) == "closed"
    events = journal.replay()
    names = [event["event"] for event in events]
    for name in ("create_outcome_unknown", "adopted_by_label", "destroy_response"):
        assert name in names
    absence = next(event for event in events if event["event"] == "absence_verified")
    assert absence == {**absence, "instance_id": vast_fake.FIRST_INSTANCE_ID}
    assert absence["label"] == opened["label"]
    assert fake.count("PUT") == 1 and fake.count("DELETE") == 1
    assert not any(fake._visible(item) for item in fake.instances.values())


def test_a_signal_while_the_create_put_is_in_flight_journals_the_outcome_unknown(
    tmp_path: Path, key_file, fake, monkeypatch
) -> None:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1])

    def signalled(offer_id, payload):
        raise SystemExit(143)

    monkeypatch.setattr(api, "create", signalled)
    with pytest.raises(SystemExit):
        vast.create_instance(api, journal, _package(), _no_system)
    assert _events(journal)[-2:] == ["create_requested", "create_outcome_unknown"]
    assert "SystemExit" in journal.replay()[-1]["error"]


def test_a_lost_create_without_an_instance_is_searched_until_deadline_plus_15_min(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened = _journal(tmp_path, fake, key_file, deadline_in=600)
    journal.append("create_requested")
    journal.append("create_outcome_unknown", error="connection reset")
    deadline = opened["deadline_unix"]
    assert _tick(journal, deadline - 1) == "searching"
    assert _tick(journal, deadline + 899, cutoff=True) == "searching"
    assert _tick(journal, deadline + 900, cutoff=True) == "closed"
    assert journal.replay()[-1]["reason"] == "no_instance_found"
    assert fake.count("DELETE") == 0


@pytest.mark.parametrize(
    "change",
    [
        {"dph_total": math.nan},
        {"dph_total": math.inf},
        {"inet_down_cost": -1.0},
        {"dph_total": 9.0},
    ],
    ids=["nan", "inf", "negative", "over-ceiling-or-envelope"],
)
def test_a_bad_or_expensive_quote_is_refused_before_any_put(
    tmp_path: Path, key_file, fake, monkeypatch, change
) -> None:
    monkeypatch.setattr(vast_fake, "OFFER", {**vast_fake.OFFER, **change})
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1])
    with pytest.raises(vast.VastError, match="refused|finite|negative"):
        vast.create_instance(
            api, journal, _package(envelope={"dph_total": {"lte": 50}}), _no_system
        )
    assert fake.count("PUT") == 0 and "create_requested" not in _events(journal)


def test_estimate_follows_the_proposal_formula_and_refuses_bad_numbers() -> None:
    offer = {"dph_total": 1.236667, "inet_down_cost": 0.0026042, "inet_up_cost": 0.0039063}
    assert vast.estimate_usd(offer, 2.5, 102.946, 1.0) == pytest.approx(3.36, abs=0.01)
    for bad in (math.nan, math.inf, -0.1, True, "1"):
        with pytest.raises(vast.VastError):
            vast.estimate_usd({**offer, "dph_total": bad}, 2.5, 1, 1)


def test_create_is_refused_with_instances_already_on_the_account_or_without_timers(
    tmp_path: Path, key_file, fake
) -> None:
    api = vast.Api(fake.url, key_file[1])
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    fake.add_instance("someone-else")
    with pytest.raises(vast.VastError, match="already has instances"):
        vast.create_instance(api, journal, _package(), _no_system)
    bare = vast.Journal.create(tmp_path / "gpu", "bare", key_file[1])
    bare.append("opened", label="x", deadline_unix=2e9)
    with pytest.raises(vast.VastError, match="not armed"):
        vast.create_instance(api, bare, _package(), _no_system)
    assert fake.count("PUT") == 0


def test_create_checks_both_timers_are_active_when_armed_for_real(
    tmp_path: Path, key_file, fake
) -> None:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    journal.append("armed", units=["u-watch", "u-cutoff"], argv=[], dry_run=False)
    calls = []

    def runner(argv):
        calls.append(argv)
        return (3 if "u-cutoff.timer" in argv else 0), ""

    with pytest.raises(vast.VastError, match="u-cutoff.timer is not active"):
        vast.create_instance(vast.Api(fake.url, key_file[1]), journal, _package(), runner)
    assert calls[0] == ["systemctl", "--user", "is-active", "--quiet", "u-watch.timer"]
    assert fake.count("PUT") == 0


@pytest.mark.parametrize(
    "event",
    [
        ("closed", {"reason": "x"}),
        ("destroy_requested", {"instance_id": None, "reason": "agent"}),
        ("create_requested", {}),
    ],
    ids=["closed", "destroy-requested", "already-requested"],
)
def test_create_replays_the_journal_under_its_lock_and_refuses_a_changed_attempt(
    tmp_path: Path, key_file, fake, monkeypatch, event
) -> None:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1])
    real_offer = api.offer

    def offer_then_watchdog(*args):
        journal.append(event[0], **event[1])
        return real_offer(*args)

    monkeypatch.setattr(api, "offer", offer_then_watchdog)
    with pytest.raises(vast.VastError, match="create refused"):
        vast.create_instance(api, journal, _package(), _no_system)
    assert fake.count("PUT") == 0
    assert _events(journal).count("create_requested") == (event[0] == "create_requested")


def test_create_is_refused_after_the_deadline(tmp_path: Path, key_file, fake) -> None:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() - 1)
    with pytest.raises(vast.VastError, match="deadline has passed"):
        vast.create_instance(vast.Api(fake.url, key_file[1]), journal, _package(), _no_system)
    assert fake.log == []


# --- the watchdog --------------------------------------------------------------------------


def _created(tmp_path, fake, key_file, **kw):
    journal, opened = _journal(tmp_path, fake, key_file, **kw)
    instance_id = fake.add_instance(opened["label"], actual_status="running")
    journal.append("create_requested")
    journal.append("create_response", instance_id=instance_id)
    return journal, opened, instance_id


def test_the_deadline_destroys_by_id_once_and_a_later_tick_sends_nothing(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened, instance_id = _created(tmp_path, fake, key_file)
    deadline = opened["deadline_unix"]
    assert _tick(journal, deadline - 10) == "ok"
    assert fake.count("DELETE") == 0
    assert _tick(journal, deadline) == "closed"
    assert [e for e in fake.log if e["method"] == "DELETE"] == [
        {"method": "DELETE", "path": f"/api/v0/instances/{instance_id}/", "auth": True}
    ]
    assert journal.replay()[-1] == {**journal.replay()[-1], "event": "closed", "reason": "deadline"}
    assert _tick(journal, deadline + 60) == "closed"
    assert fake.count("DELETE") == 1


def test_cutoff_destroys_before_the_deadline_without_any_other_reason(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened, instance_id = _created(tmp_path, fake, key_file)
    assert _tick(journal, opened["deadline_unix"] - 3000, cutoff=True) == "closed"
    assert fake.count("DELETE") == 1


def test_a_destroy_answered_404_still_checks_absence_and_closes(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened, instance_id = _created(tmp_path, fake, key_file)
    del fake.instances[instance_id]
    assert _tick(journal, opened["deadline_unix"]) == "closed"
    statuses = [e["status"] for e in journal.replay() if e["event"] == "destroy_response"]
    assert statuses == ["absent"]


def test_an_instance_still_listed_after_destroy_keeps_the_attempt_open(
    tmp_path: Path, key_file, fake
) -> None:
    fake.destroy_lag = 3
    journal, opened, _ = _created(tmp_path, fake, key_file)
    assert _tick(journal, opened["deadline_unix"]) == "destroying"
    assert "closed" not in _events(journal)
    for _ in range(3):
        outcome = _tick(journal, opened["deadline_unix"] + 60)
    assert outcome == "closed"


def test_a_stale_heartbeat_means_the_agent_crashed_and_destroys(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened, _ = _created(tmp_path, fake, key_file)
    (journal.dir / vast.HEARTBEAT_FILE).write_text("1000000")
    assert _tick(journal, 1_000_000.0 + 299, beat=False) == "ok"
    assert _tick(journal, 1_000_000.0 + 301, beat=False) == "closed"
    assert "agent_lost" in journal.replay()[-1]["reason"]


def test_two_instances_with_the_label_are_both_destroyed_and_a_foreign_one_is_kept(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened, first = _created(tmp_path, fake, key_file)
    second = fake.add_instance(opened["label"])
    foreign = fake.add_instance("sound-scout-gpu-someone-else")
    assert _tick(journal, 1_000_001.0) == "closed"
    deleted = {e["path"] for e in fake.log if e["method"] == "DELETE"}
    assert deleted == {f"/api/v0/instances/{i}/" for i in (first, second)}
    assert fake._visible(fake.instances[foreign])
    assert "several_instances" in journal.replay()[-1]["reason"]


def test_spend_over_the_early_stop_destroys(tmp_path: Path, key_file, fake) -> None:
    journal, _, _ = _created(tmp_path, fake, key_file)
    fake.credit = 7.6
    assert _tick(journal, 1_000_001.0) == "ok"
    fake.credit = 7.5
    assert _tick(journal, 1_000_001.0) == "closed"
    assert journal.replay()[-1]["reason"] == "early_stop"


def test_a_tick_without_create_closes_at_the_deadline_and_makes_no_call(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened = _journal(tmp_path, fake, key_file)
    assert _tick(journal, opened["deadline_unix"] - 1) == "idle"
    assert _tick(journal, opened["deadline_unix"]) == "closed"
    assert journal.replay()[-1]["reason"] == "deadline_no_instance"
    assert fake.log == []


def test_a_failed_label_listing_still_destroys_the_known_id_and_keeps_the_attempt_open(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened, instance_id = _created(tmp_path, fake, key_file)
    failing = {"left": 2}

    class FlakyApi(vast.Api):
        def instances(self, label=None):
            if failing["left"]:
                failing["left"] -= 1
                raise vast.ApiError(503, "GET /api/v1/instances: HTTP 503")
            return super().instances(label)

    def factory(opened_):
        return FlakyApi(opened_["api_base"], key_file[1])

    def tick(now):
        (journal.dir / vast.HEARTBEAT_FILE).write_text(str(now))
        return vast.watch_tick(journal, factory, runner=_no_system, now=now)

    assert tick(opened["deadline_unix"] - 10) == "error"
    assert fake.count("DELETE") == 0
    assert tick(opened["deadline_unix"]) == "destroying"
    assert fake.count("DELETE") == 1
    assert "closed" not in _events(journal) and "watch_error" in _events(journal)
    assert tick(opened["deadline_unix"] + 60) == "closed"
    assert journal.replay()[-1]["reason"].startswith("deadline")
    assert not fake._visible(fake.instances[instance_id])


def test_a_tick_while_the_lock_is_held_is_skipped(tmp_path: Path, key_file, fake) -> None:
    journal, opened, _ = _created(tmp_path, fake, key_file)
    other = vast.Journal(journal.path)
    with journal.lock():
        assert vast.watch_tick(other, vast._api_from_journal, runner=_no_system) == "busy"
    assert fake.log == []


def test_closing_stops_the_real_timers_by_name(tmp_path: Path, key_file, fake) -> None:
    journal, opened, _ = _created(tmp_path, fake, key_file)
    journal.append("armed", units=["u-watch", "u-cutoff"], argv=[], dry_run=False)
    calls = []
    outcome = vast.watch_tick(
        journal,
        vast._api_from_journal,
        runner=lambda argv: calls.append(argv) or (0, ""),
        now=opened["deadline_unix"],
    )
    assert outcome == "closed"
    assert calls == [
        ["systemctl", "--user", "stop", "u-watch.timer"],
        ["systemctl", "--user", "stop", "u-cutoff.timer"],
    ]


def test_watch_cli_reads_the_key_itself_from_the_journal_path(
    tmp_path: Path, key_file, fake, capsys
) -> None:
    journal, opened, _ = _created(tmp_path, fake, key_file, deadline_in=-10)
    journal.path.parent.chmod(0o700)
    assert vast.main(["cutoff", "--journal", str(journal.path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"outcome": "closed"}
    assert fake.count("DELETE") == 1
    assert key_file[1] not in journal.path.read_text()


def test_watch_cli_refuses_a_journal_others_can_read(tmp_path: Path, key_file, fake, capsys):
    journal, _ = _journal(tmp_path, fake, key_file)
    journal.path.chmod(0o644)
    assert vast.main(["watch", "--journal", str(journal.path)]) == 2
    assert "0600" in capsys.readouterr().err


# --- the timers ----------------------------------------------------------------------------


def test_timer_argv_are_exact_with_utc_calendar_and_no_key(tmp_path: Path, key_file) -> None:
    journal = tmp_path / "gpu" / "A" / "journal.jsonl"
    watch = vast.watch_timer_argv(journal, "A", python="/opt/py/bin/python")
    cutoff = vast.cutoff_timer_argv(journal, "A", 1795318654.7, python="/opt/py/bin/python")
    tail = ["--", "/opt/py/bin/python", "-m", "sound_scout.eval.vast"]
    assert watch == [
        "systemd-run",
        "--user",
        "--unit=sound-scout-gpu-A-watch",
        "--collect",
        "--description=Sound Scout GPU watchdog A",
        "--on-active=60s",
        "--on-unit-active=60s",
        "--timer-property=AccuracySec=5s",
        *tail,
        "watch",
        "--journal",
        str(journal),
    ]
    assert cutoff == [
        "systemd-run",
        "--user",
        "--unit=sound-scout-gpu-A-cutoff",
        "--collect",
        "--description=Sound Scout GPU cutoff A",
        "--on-calendar=2026-11-22 03:37:34 UTC",
        "--on-calendar=2026-11-22 03:52:34 UTC",
        "--timer-property=AccuracySec=1s",
        *tail,
        "cutoff",
        "--journal",
        str(journal),
    ]
    joined = " ".join(watch + cutoff)
    assert key_file[1] not in joined and "--setenv" not in joined


def _armable(tmp_path, fake, key_file):
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700)
    journal = vast.Journal.create(gpu, "A", key_file[1])
    journal.append("opened", attempt_id="A", label="L", deadline_unix=2e9)
    return journal


def test_arm_refuses_without_linger_and_with_python_in_a_task_tree(
    tmp_path: Path, key_file, fake
) -> None:
    journal = _armable(tmp_path, fake, key_file)
    with pytest.raises(vast.VastError, match="Linger"):
        vast.arm(journal, lambda argv: (0, "Linger=no\n"), python="/opt/py/bin/python")
    with pytest.raises(vast.VastError, match="task tree"):
        vast.arm(journal, _no_system, python="/src/.work-task/147/.venv/bin/python")
    assert "armed" not in _events(journal)


def test_arm_starts_both_timers_checks_them_and_records_them(
    tmp_path: Path, key_file, fake
) -> None:
    journal = _armable(tmp_path, fake, key_file)
    calls = []

    def runner(argv):
        calls.append(argv)
        return 0, "Linger=yes\n"

    vast.arm(journal, runner, python="/opt/py/bin/python")
    assert calls[0][:2] == ["loginctl", "show-user"]
    assert [argv[2] for argv in calls[1:3]] == [
        "--unit=sound-scout-gpu-A-watch",
        "--unit=sound-scout-gpu-A-cutoff",
    ]
    assert calls[3:] == [
        ["systemctl", "--user", "is-active", "--quiet", f"sound-scout-gpu-A-{k}.timer"]
        for k in ("watch", "cutoff")
    ]
    armed = journal.replay()[-1]
    assert armed["event"] == "armed" and armed["dry_run"] is False


# --- the journal ---------------------------------------------------------------------------


def test_journal_fsyncs_every_event_and_is_private(
    tmp_path: Path, key_file, monkeypatch, fake
) -> None:
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700)
    journal = vast.Journal.create(gpu, "A", key_file[1])
    calls = []
    real = os.fsync
    monkeypatch.setattr(vast.os, "fsync", lambda fd: calls.append(fd) or real(fd))
    for number in range(3):
        journal.append("step", n=number)
    assert len(calls) == 3
    assert journal.path.stat().st_mode & 0o777 == 0o600
    assert journal.dir.stat().st_mode & 0o777 == 0o700
    journal.append("note", text=f"key is {key_file[1]}")
    assert key_file[1] not in journal.path.read_text()


def test_a_torn_last_line_is_skipped_and_dropped_before_the_next_event(
    tmp_path: Path, key_file
) -> None:
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700)
    journal = vast.Journal.create(gpu, "A", key_file[1])
    journal.append("opened", label="L")
    with open(journal.path, "a") as file:
        file.write('{"event": "create_req')
    assert _events(journal) == ["opened"]
    journal.append("closed", reason="x")
    assert _events(journal) == ["opened", "closed"]


def test_a_broken_line_in_the_middle_is_refused(tmp_path: Path, key_file) -> None:
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700)
    journal = vast.Journal.create(gpu, "A", key_file[1])
    with open(journal.path, "a") as file:
        file.write('{"event": "opened"}\nnot json\n{"event": "closed"}\n')
    with pytest.raises(vast.VastError, match="line 2 is broken"):
        journal.replay()


def test_open_refuses_while_an_earlier_attempt_is_not_closed(tmp_path: Path, key_file) -> None:
    gpu = tmp_path / "gpu"
    gpu.mkdir(mode=0o700)
    journal = vast.Journal.create(gpu, "A", key_file[1])
    journal.append("opened", label="L")
    with pytest.raises(vast.VastError, match="not closed"):
        vast.refuse_open_attempts(gpu)
    journal.append("closed", reason="done")
    vast.refuse_open_attempts(gpu)


# --- download ------------------------------------------------------------------------------


def _archive(tmp_path: Path, files: dict[str, bytes], sums: dict[str, bytes] | None = None):
    root = tmp_path / "pack"
    root.mkdir()
    sums = files if sums is None else sums
    lines = "".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in sums.items())
    archive = tmp_path / "results.tar"
    with tarfile.open(archive, "w") as tar:
        for name, data in {**files, vast.SUMS_FILE: lines.encode()}.items():
            (root / Path(name).name).write_bytes(data)
            tar.add(root / Path(name).name, arcname=name)
    return archive


def test_an_archive_matching_its_sums_is_accepted(tmp_path: Path) -> None:
    archive = _archive(tmp_path, {"a.json": b"{}", "q/run.json": b"[]"})
    assert vast.verify_archive(archive, tmp_path / "host") == 2


@pytest.mark.parametrize(
    ("files", "sums", "message"),
    [
        ({"a.json": b"{!}"}, {"a.json": b"{}"}, "changed"),
        ({"a.json": b"{}", "b.json": b"1"}, {"a.json": b"{}"}, "extra"),
        ({"a.json": b"{}"}, {"a.json": b"{}", "c.json": b"2"}, "missing"),
        ({"../escape.json": b"{}"}, None, "refused"),
    ],
    ids=["changed-byte", "extra-file", "missing-file", "path-out"],
)
def test_an_archive_that_differs_from_its_sums_is_refused(tmp_path, files, sums, message) -> None:
    archive = _archive(tmp_path, files, sums)
    with pytest.raises(vast.VastError, match=message):
        vast.verify_archive(archive, tmp_path / "host")
    assert not (tmp_path / "escape.json").exists()


def test_an_archive_with_a_link_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "results.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("link")
        info.type, info.linkname = tarfile.SYMTYPE, "a.json"
        tar.addfile(info)
    with pytest.raises(vast.VastError, match="link"):
        vast.verify_archive(archive, tmp_path / "host")


# --- runtime and pins ----------------------------------------------------------------------


def test_runtime_mismatches_name_each_component() -> None:
    good = {"vllm": "0.34.0", "torch": "2.14.0+cu130", "cuda": "13.0", "transformers": "5.22.1"}
    assert vast.runtime_mismatches(good, "0.34.0\n") == []
    assert vast.runtime_mismatches({**good, "cuda": "13.0.2"}, "0.34.0") == ["cuda"]
    bad = {**good, "torch": "2.13.0", "transformers": "5.23.0", "cuda": None}
    assert vast.runtime_mismatches(bad, "0.33.9") == ["vllm", "torch", "cuda", "transformers"]


def test_models_pin_both_revisions_without_spk_dict_and_with_one_hash_kind_each() -> None:
    assert vast.MODELS["qwen25omni"]["revision"] == "ae9e1690543ffd5c0221dc27f79834d0294cba00"
    assert vast.MODELS["qwen3omni"]["revision"] == "26291f793822fb6be9555850f06dfe95f2d7e695"
    sizes = {name: sum(f["size"] for f in m["files"]) for name, m in vast.MODELS.items()}
    assert sizes == {"qwen25omni": 22_379_037_779, "qwen3omni": 70_530_595_574}
    for model in vast.MODELS.values():
        assert all(entry["path"] != "spk_dict.pt" for entry in model["files"])
        for entry in model["files"]:
            assert re.fullmatch(r"[0-9a-f]{40}", entry["git_oid"])
            assert entry["lfs_sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", entry["lfs_sha256"])
            if entry["path"].endswith(".safetensors"):
                assert entry["lfs_sha256"] is not None


def test_vllm_argv_is_local_thinker_only_without_remote_code() -> None:
    argv = vast.vllm_argv("/workspace/models/qwen25omni", "qwen25omni")
    assert argv[:3] == ["vllm", "serve", "/workspace/models/qwen25omni"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8000"
    assert "--trust-remote-code" not in argv
    assert argv[argv.index("--limit-mm-per-prompt") + 1] == '{"audio":1,"image":0,"video":0}'
    assert argv[argv.index("--dtype") + 1] == "bfloat16"


def test_vllm_argv_turns_off_the_prefix_and_multimodal_processor_caches() -> None:
    # flag names of vLLM v0.34.0: vllm/engine/arg_utils.py (#161)
    argv = vast.vllm_argv("/workspace/models/qwen25omni", "qwen25omni")
    assert "--no-enable-prefix-caching" in argv and "--enable-prefix-caching" not in argv
    assert argv[argv.index("--mm-processor-cache-gb") + 1] == "0"


class _PilotClient:
    model = "qwen25omni"

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def text_tokens(self, text: str) -> int:
        return 56

    def request(self, route: str, body: dict, timeout: float):
        self.bodies.append(body)
        prompt = 56 + 25 * len(self.bodies)
        return 200, json.dumps({"usage": {"prompt_tokens": prompt}}), None


def test_pilot_request_sends_no_llama_only_field_to_vllm() -> None:
    client = _PilotClient()
    assert vast.run_pilot(client)["ok"] is True
    assert len(client.bodies) == 2
    for body in client.bodies:
        assert not gate0.LLAMA_ONLY_FIELDS & body.keys()
        assert body["temperature"] == 0 and body["seed"] == 0
        assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_pilot_clips_are_1_khz_tones_of_3_and_6_s() -> None:
    clips = vast.pilot_clips()
    for clip_id, seconds in (("X1", 3.0), ("X2", 6.0)):
        with wave.open(io.BytesIO(clips[clip_id])) as wav:
            assert wav.getnframes() == seconds * gate0.SAMPLE_RATE
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2") / 32767
        spectrum = np.abs(np.fft.rfft(pcm))
        peak_hz = np.fft.rfftfreq(len(pcm), 1 / gate0.SAMPLE_RATE)[np.argmax(spectrum)]
        assert peak_hz == pytest.approx(1000.0, abs=1.0)
        assert 20 * np.log10(np.max(np.abs(pcm))) == pytest.approx(-20.0, abs=0.05)


def test_dependencies_document_pins_the_image_and_runtime_of_the_code() -> None:
    text = (REPO / "docs" / "dependencies.md").read_text(encoding="utf-8")
    section = text.split("## 7. GPU-эталон (Vast.ai)", 1)[1]
    for value in (
        vast.IMAGE["index_digest"],
        vast.IMAGE["amd64_digest"],
        f"{vast.IMAGE['repository']}:{vast.IMAGE['tag']}",
        f"{vast.IMAGE['compressed_bytes']:,}".replace(",", " "),
        vast.IMAGE["config_digest"],
        f"CUDA_VERSION={vast.IMAGE['cuda_env']}",
        *(vast.RUNTIME.values()),
    ):
        assert value in section, value
    for model in vast.MODELS.values():
        assert model["repo"] in section and model["revision"] in section
        for entry in model["files"]:
            assert (entry["lfs_sha256"] or entry["git_oid"]) in section, entry["path"]
    assert "на станцию не загружается" in section and "#148" in section


def test_timing_document_gives_every_step_an_estimate_a_basis_and_names_the_pilot() -> None:
    text = (REPO / "docs" / "gpu-timing.md").read_text(encoding="utf-8")
    table = [line for line in text.splitlines() if line.startswith("| `")]
    steps = [re.match(r"\| `([a-z]+)`", line).group(1) for line in table]
    for step in DRY_STEPS:
        assert step in steps, step
    for line in table:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) >= 4 and all(cells[:4]), line
        assert re.search(r"\d", cells[1]), line
    assert "## Меряет только платный пилот" in text and "#148" in text
    assert "dry run" in text.lower()


def test_timing_document_records_attempt_1_of_the_pilot() -> None:
    text = (REPO / "docs" / "gpu-timing.md").read_text(encoding="utf-8")
    section = text.split("## Попытка 1 пилота #148", 1)[1].split("\n## ", 1)[0]
    measured = {}
    for line in section.splitlines():
        if line.startswith("| `"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            measured[cells[0].strip("`")] = cells[2]
    assert {step: measured[step] for step in ("ready", "preflight", "destroy")} == {
        "ready": "294,6",
        "preflight": "20,2",
        "destroy": "3,1",
    }
    assert "20261122T061200Z-66c4813a" in section and "#158" in section


def test_timing_document_records_attempt_3_of_the_pilot() -> None:
    text = (REPO / "docs" / "gpu-timing.md").read_text(encoding="utf-8")
    section = text.split("## Попытка 3 пилота #148", 1)[1].split("\n## ", 1)[0]
    measured = {}
    for line in section.splitlines():
        if line.startswith("| `"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            measured[cells[0].strip("`")] = (cells[2], cells[3])
    assert {step: value for step, (value, _) in measured.items()} == {
        "arm": "0,06",
        "create": "2,4",
        "ready": "45,0",
        "preflight": "29,3",
        "weights": "81,8",
        "smoke": "1201,1",
        "destroy": "2,9",
    }
    assert "22,38 GB" in measured["weights"][1] and "таймаут" in measured["smoke"][1]
    assert "20261122T161658Z-24071652" in section and "62000482" in section
    assert "1138" in section and "#160" in section and "не доказана" in section


def test_example_package_loads_and_holds_no_secret() -> None:
    path = REPO / "docs" / "examples" / "gpu-package.example.json"
    package, digest = vast.load_package(path, vast.MODELS, real=True)
    assert len(digest) == 64 and package["ceiling_usd"] == 3.0
    assert vast.offer_ids(package) == [34475479, 34475480]
    vast.check_no_secret(path.read_bytes(), None)


def _model_limit(index: int, **fields) -> dict:
    package = json.loads((REPO / "docs/examples/gpu-package.example.json").read_text())
    package["models"][index].update(fields)
    return {"models": package["models"]}


@pytest.mark.parametrize(
    "change",
    [
        {"ceiling_usd": math.inf},
        {"early_stop_usd": 5.0},
        {"envelope": {}},
        {"models": []},
        {"deadline_minutes": 121},
        {"soft_end_minutes": 105},
        {"soft_end_minutes": None, "weights_late_minutes": 25},
        {"spend_soft_usd": 2.3},
        {"offer_ids": []},
        {"offer_ids": [1, 1]},
        {"offer_ids": [1, 2, 3, 4]},
        {"offer_ids": [1], "offer_id": 1},
        {"extended": {"dir": "../x", "manifest_sha256": "0" * 64, "plan_sha256": "0" * 64}},
        {"extended": {"dir": "x", "manifest_sha256": "0" * 63}},
        {"closed_timeout_s": 0},
        _model_limit(0, slot_minutes=-1),
        _model_limit(1, health_timeout_s=0),
    ],
    ids=[
        "inf",
        "early-over-ceiling",
        "no-dph-bound",
        "no-models",
        "deadline-over-120",
        "soft-end-at-deadline",
        "weights-late-without-soft-end",
        "spend-soft-over-early-stop",
        "no-offers",
        "same-offer-twice",
        "four-offers",
        "offer-id-and-offer-ids",
        "extended-dir-outside",
        "extended-bad-pin",
        "zero-timeout",
        "negative-slot",
        "zero-health-timeout",
    ],
)
def test_a_malformed_package_is_refused(tmp_path: Path, change) -> None:
    package = json.loads((REPO / "docs/examples/gpu-package.example.json").read_text())
    path = tmp_path / "p.json"
    changed = {k: v for k, v in {**package, **change}.items() if v is not None}
    path.write_text(json.dumps(changed, allow_nan=True))
    with pytest.raises(vast.VastError):
        vast.load_package(path, vast.MODELS)


# --- the dry run ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dry_run_gives_the_full_timeline_against_the_fakes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VERSION", raising=False)
    sentinel = tmp_path / "real-key-must-not-be-read"
    monkeypatch.setattr(vast, "KEY_PATH", sentinel)
    read = []
    real_read = vast.read_api_key
    monkeypatch.setattr(
        vast, "read_api_key", lambda path=None: read.append(path) or real_read(path)
    )
    (tmp_path / "in").mkdir()
    (tmp_path / "data").mkdir()
    config_ = config.Config(
        allowed_roots=((tmp_path / "in").resolve(),), data_root=(tmp_path / "data").resolve()
    )
    out = tmp_path / "data" / "dry"
    result = vast.dry_run(out, config_)
    names = [step["name"] for step in result["steps"]]
    assert [name for name in names if name in DRY_STEPS] == [
        "create",
        "ready",
        "weights",
        "smoke",
        "run",
        "extended",
        "weights",
        "smoke",
        "run",
        "extended",
        "download",
        "destroy",
    ]
    assert [(m["model"], m["outcome"]) for m in result["models"]] == [
        ("qwen25omni", "ok"),
        ("qwen3omni", "ok"),
    ]
    assert all(step["ok"] for step in result["steps"])
    (preflight,) = [step for step in result["steps"] if step["name"] == "preflight"]
    assert _runtime_found(preflight["detail"]) == {
        "vllm": "0.34.0",
        "torch": "2.14.0",
        "cuda": "13.0",
        "transformers": "5.20.0",
        "driver": "595.91.07",
        "cuda_env": None,
    }
    for step in result["steps"]:
        assert step["elapsed_s"] >= 0 and step["finished_at"] >= step["started_at"]
    runs_ = [s for s in result["steps"] if s["name"] == "run"]
    assert {s["model"] for s in runs_} == {"qwen25omni", "qwen3omni"}
    assert all(s["detail"]["verdict"] == "not_applicable" for s in runs_)
    assert result["fake_api"] == {"put": 1, "delete": 1, "instances_left": []}
    assert result["closed"]["event"] == "closed"
    assert sentinel not in read and None not in read
    journal = vast.Journal(out / result["journal"])
    events = journal.replay()
    absence = [e for e in events if e["event"] == "absence_verified"]
    assert len(absence) == 1 and absence[0]["label"].startswith(vast.LABEL_PREFIX)
    armed = next(e for e in events if e["event"] == "armed")
    assert armed["dry_run"] is True and len(armed["argv"]) == 2
    assert json.loads((out / "timeline.json").read_text())["steps"] == result["steps"]
    log = [json.loads(line) for line in (out / "fake-api.jsonl").read_text().splitlines()]
    assert all(entry["auth"] for entry in log)
    key = (out / "fake-api-key").read_text().strip()
    for path in out.rglob("*"):
        if path.is_file() and path.name != "fake-api-key":
            assert key.encode() not in path.read_bytes(), path
    per_model = [
        f"{e['event']}:{e.get('name', '')}:{e['model']}"
        for e in events
        if e.get("model") and e["event"] in ("step", "server_evidence", "server_stopped")
    ]
    assert per_model == [
        f"{event}:{model}"
        for model in ("qwen25omni", "qwen3omni")
        for event in (
            "step:weights",
            "step:smoke",
            "step:run",
            "step:extended",
            "server_evidence:",
            "server_stopped:",
        )
    ]
    syncs = [e for e in events if e["event"] == "sync"]
    assert len(syncs) == 2 * len(gpu_ext.BLOCKS) and all(e["ok"] for e in syncs)
    assert all(e["returncode"] == 0 for e in events if e["event"] == "server_stopped")
    assert sorted(p.name for p in (out / "gpu" / result["attempt_id"]).glob("evidence-*")) == [
        "evidence-qwen25omni.json",
        "evidence-qwen3omni.json",
    ]
    kinds = [e["event"] for e in events]
    assert kinds.count("credit_after_close") == 1 and kinds[-1] == "credit_after_close"
    assert kinds.index("closed") < kinds.index("credit_after_close")
    credit = events[-1]
    assert set(credit) == CREDIT_FIELDS
    assert (credit["credit"], credit["initial_credit"], credit["delta_usd"]) == (10.0, 10.0, 0.0)
    assert credit["since_closed_s"] >= 0 and credit["t"] >= result["closed"]["t"]
    assert result["credit_after_close"] == credit
    credit_reads = sum(e["event"] == "credit_seen" for e in events)
    assert credit_reads == len(syncs)
    assert sum(e["path"] == "/api/v0/users/current/" for e in log) == 2 + credit_reads
    host = out / "gpu" / result["attempt_id"] / "host"
    assert (host / "guard-selftest.json").is_file()
    assert json.loads((host / "guard-selftest.json").read_text())["ok"] is True
    for model, day in (("qwen25omni", "2000-01-01"), ("qwen3omni", "2000-01-02")):
        assert "guard_probe: denied" in (host / model / "server.log").read_text()
        assert "socket.bind" in (host / model / "guard.log").read_text()
        run = json.loads((out / day / "attempts" / "1" / "run.json").read_text())
        assert run["runtime"] == "vllm" and run["served_model"] == model and run["argv"] is None
        assert run["remote"]["instance_id"] == vast_fake.FIRST_INSTANCE_ID


def test_ssh_ignores_the_user_config_forwards_nothing_and_carries_no_key(
    tmp_path: Path, key_file, monkeypatch
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return vast.proc.RunResult(0, b"", b"", False, False, 0.0)

    monkeypatch.setattr(vast.proc, "run", fake_run)
    host = vast.RemoteHost(tmp_path, "203.0.113.7", 40022)
    host.run(["python3", vast.HOSTRUN, "preflight", "--workspace", vast.WORKSPACE], 60)
    argv = calls[0]
    assert argv[:4] == ["ssh", "-a", "-x", "-F"] and argv[4] == "/dev/null"
    for option in (
        "IdentitiesOnly=yes",
        "StrictHostKeyChecking=accept-new",
        "BatchMode=yes",
        "ConnectTimeout=10",
    ):
        assert option in argv
    assert f"IdentityFile={tmp_path / vast.PRIVATE_DIR / 'id_ed25519'}" in argv
    assert argv[-2:] == [
        "root@203.0.113.7",
        "python3 /workspace/ss/hostrun.py preflight --workspace /workspace",
    ]
    joined = " ".join(argv)
    assert "SendEnv" not in joined and "ForwardAgent" not in joined
    assert key_file[1] not in joined
    assert vast.ssh_endpoint(
        {"public_ipaddr": "203.0.113.7", "ports": {"22/tcp": [{"HostPort": "40022"}]}}
    ) == ("203.0.113.7", 40022)
    with pytest.raises(vast.VastError, match="22/tcp"):
        vast.ssh_endpoint({"public_ipaddr": "203.0.113.7", "ports": {}})


class _SshHost:
    def __init__(self, failures: int) -> None:
        self.failures, self.calls = failures, []

    def run(self, argv, timeout):
        self.calls.append(argv)
        code = 255 if len(self.calls) <= self.failures else 0
        return vast.proc.RunResult(code, b"", b"", False, False, 0.0)


def test_ssh_is_polled_until_it_answers_and_a_dead_host_ends_in_time() -> None:
    intervals = {"ssh_timeout_s": 5.0, "ready_poll_s": 0.01}
    host = _SshHost(failures=2)
    vast.wait_ssh(host, intervals)
    assert host.calls == [["true"]] * 3
    dead = _SshHost(failures=10**9)
    with pytest.raises(vast.VastError, match="SSH not ready"):
        vast.wait_ssh(dead, {"ssh_timeout_s": 0.1, "ready_poll_s": 0.01})
    assert all(call == ["true"] for call in dead.calls)


def test_vllm_version_takes_the_last_line_so_log_lines_before_it_pass() -> None:
    found = {"vllm": "0.34.0", "torch": "2.14.0+cu130", "cuda": "13.0", "transformers": "5.20.0"}
    logged = "INFO 11-22 10:00:00 [__init__.py:1] Automatically detected platform cuda.\n0.34.0\n"
    assert vast.runtime_mismatches(found, logged) == []
    assert vast.runtime_mismatches(found, "0.34.0\nINFO done\n") == ["vllm"]
    assert vast.runtime_mismatches(found, "") == ["vllm"]


def _dry_config(tmp_path: Path) -> config.Config:
    (tmp_path / "in").mkdir()
    (tmp_path / "data").mkdir()
    return config.Config(
        allowed_roots=((tmp_path / "in").resolve(),), data_root=(tmp_path / "data").resolve()
    )


@pytest.mark.slow
def test_a_step_failing_after_create_still_destroys_and_closes_the_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(vast, "FAKE_DISTRIBUTIONS", {**vast.FAKE_DISTRIBUTIONS, "vllm": "0.33.9"})
    out = tmp_path / "data" / "dry"
    with pytest.raises(vast.VastError, match=r"runtime differs from the pin in \['vllm'\]"):
        vast.dry_run(out, _dry_config(tmp_path))
    log = [json.loads(line) for line in (out / "fake-api.jsonl").read_text().splitlines()]
    assert sum(e["method"] == "PUT" for e in log) == 1
    assert sum(e["method"] == "DELETE" for e in log) == 1
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    steps = {e["name"]: e["ok"] for e in events if e["event"] == "step"}
    assert steps["preflight"] is False and steps["destroy"] is True
    absence = [e["instance_id"] for e in events if e["event"] == "absence_verified"]
    assert absence == [vast_fake.FIRST_INSTANCE_ID]
    (closed,) = [e for e in events if e["event"] == "closed"]
    assert "destroy_requested" in closed["reason"]


USERS_PATH = "GET /api/v0/users/current/"
CREDIT_FIELDS = {"event", "t", "since_closed_s", "initial_credit", "credit", "delta_usd"}


def _no_secret_fields(event: dict, key: str) -> None:
    text = json.dumps(event)
    assert key not in text
    assert not any(name.lower() in text.lower() for name in (*vast.SECRET_NAMES, "key", "token"))


@pytest.mark.slow
def test_a_failed_credit_read_after_close_is_journaled_and_changes_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    real_user = vast.Api.user
    calls = []

    def user(self):
        calls.append(1)
        if len(calls) == 1:
            return real_user(self)
        raise vast.ApiError(503, f"{USERS_PATH}: HTTP 503: echo {self.key}")

    monkeypatch.setattr(vast.Api, "user", user)
    config_ = _dry_config(tmp_path)
    monkeypatch.setattr(vast.config, "load", lambda: config_)
    out = tmp_path / "data" / "dry"
    assert vast.main(["dry-run", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["closed"]["event"] == "closed"
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    seen = [e for e in events if e["event"] == "credit_seen"]
    assert len(calls) == 2 + len(seen) and seen
    assert all(e["error"].endswith(f"echo {vast.REDACTED}") for e in seen)
    kinds = [e["event"] for e in events]
    assert kinds[-3:] == ["closed", "step", "credit_after_close"]
    assert events[-2]["name"] == "destroy" and events[-2]["ok"] is True
    credit = events[-1]
    assert set(credit) == {"event", "t", "since_closed_s", "initial_credit", "error"}
    assert credit["error"] == f"ApiError: {USERS_PATH}: HTTP 503: echo {vast.REDACTED}"
    key = (out / "fake-api-key").read_text().strip()
    _no_secret_fields(credit, key)
    assert key.encode() not in journal_path.read_bytes()
    assert vast.summarize(events)["closed"] is not None
    timeline = json.loads((out / "timeline.json").read_text())
    assert timeline["credit_after_close"] == credit


def test_deploy_reads_the_credit_once_after_a_watchdog_close_and_the_watchdog_never_reads(
    tmp_path: Path, key_file, fake
) -> None:
    journal, opened, _ = _created(tmp_path, fake, key_file)
    api = vast.Api(fake.url, key_file[1])
    assert vast.read_credit_after_close(journal, api) is None
    assert fake.count("GET", "/api/v0/users") == 0
    assert _tick(journal, opened["deadline_unix"]) == "closed"
    assert fake.count("GET", "/api/v0/users") == 0
    fake.credit = 9.25
    assert vast.destroy_until_absent(journal, lambda _: api, _no_system, vast.INTERVALS) == "closed"
    credit = vast.read_credit_after_close(journal, api)
    assert fake.count("GET", "/api/v0/users") == 1
    assert set(credit) == CREDIT_FIELDS
    assert (credit["credit"], credit["delta_usd"]) == (9.25, 0.75)
    assert _events(journal)[-2:] == ["closed", "credit_after_close"]
    _no_secret_fields(credit, key_file[1])
    status = vast.status(journal)
    assert status["credit_after_close"] == credit
    assert status["closed"]["reason"] == "deadline"


@pytest.mark.slow
def test_deploy_reads_the_credit_one_time_when_the_watchdog_closed_the_attempt_mid_run(
    tmp_path: Path, monkeypatch
) -> None:
    out = tmp_path / "data" / "dry"

    def wait_ready(api, instance_id, intervals):
        (path,) = (out / "gpu").glob("*/journal.jsonl")
        journal = vast.Journal.open(path)
        closed = vast.watch_tick(journal, lambda _: api, runner=vast._refuse_system, cutoff=True)
        assert closed == "closed"
        raise vast.VastError("stopped after the watchdog close")

    monkeypatch.setattr(vast, "wait_ready", wait_ready)
    with pytest.raises(vast.VastError, match="stopped after the watchdog close"):
        vast.dry_run(out, _dry_config(tmp_path))
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    kinds = [e["event"] for e in events]
    assert kinds.count("closed") == 1 and kinds.count("credit_after_close") == 1
    assert kinds.index("closed") < kinds.index("credit_after_close") == len(kinds) - 1
    assert events[kinds.index("closed")]["reason"].startswith("deadline")
    assert set(events[-1]) == CREDIT_FIELDS
    log = [json.loads(line) for line in (out / "fake-api.jsonl").read_text().splitlines()]
    assert sum(e["path"] == "/api/v0/users/current/" for e in log) == 2


@pytest.mark.slow
def test_a_credit_journal_failure_after_close_keeps_the_original_failure_and_destroy(
    tmp_path: Path, monkeypatch
) -> None:
    real_append = vast.Journal.append

    def append(self, event, **fields):
        if event == "credit_after_close":
            raise KeyboardInterrupt("journal append interrupted")
        return real_append(self, event, **fields)

    def wait_ready(api, instance_id, intervals):
        raise vast.VastError("the original failure")

    monkeypatch.setattr(vast.Journal, "append", append)
    monkeypatch.setattr(vast, "wait_ready", wait_ready)
    out = tmp_path / "data" / "dry"
    with pytest.raises(vast.VastError, match="the original failure"):
        vast.dry_run(out, _dry_config(tmp_path))
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    assert [e["event"] for e in events][-2:] == ["closed", "step"]
    assert events[-1]["name"] == "destroy" and events[-1]["ok"] is True
    assert vast.summarize(events)["closed"] is not None


@pytest.mark.slow
def test_a_failed_credit_read_through_the_fake_api_redacts_the_echoed_key_and_keeps_the_outcome(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    real_answer = vast_fake.FakeVast._answer

    def answer(self, method, parts, body, query):
        if parts[:3] == ["api", "v0", "users"] and self.count("DELETE") > 0:
            return 503, {"error": "overloaded", "echo": f"Bearer {self.key}", "credit": None}
        return real_answer(self, method, parts, body, query)

    monkeypatch.setattr(vast_fake.FakeVast, "_answer", answer)
    config_ = _dry_config(tmp_path)
    monkeypatch.setattr(vast.config, "load", lambda: config_)
    out = tmp_path / "data" / "dry"
    assert vast.main(["dry-run", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["closed"]["reason"] == "destroy_requested"
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    steps = [e for e in events if e["event"] == "step"]
    assert all(step["ok"] for step in steps) and steps[-1]["name"] == "destroy"
    assert [e["event"] for e in events].count("credit_after_close") == 1
    credit = events[-1]
    assert set(credit) == {"event", "t", "since_closed_s", "initial_credit", "error"}
    assert credit["error"].startswith(f"ApiError: {USERS_PATH}: HTTP 503: ")
    assert vast.REDACTED in credit["error"]
    key = (out / "fake-api-key").read_text().strip()
    assert key not in json.dumps(credit)
    assert key.encode() not in journal_path.read_bytes()
    timeline = json.loads((out / "timeline.json").read_text())
    assert timeline["credit_after_close"] == credit
    assert timeline["fake_api"] == {"put": 1, "delete": 1, "instances_left": []}


def _runtime_found(detail: dict) -> dict:
    keys = ("vllm", "torch", "cuda", "transformers", "driver", "cuda_env")
    return {key: detail["found"][key] for key in keys}


@pytest.mark.slow
def test_a_torch_built_for_another_cuda_fails_preflight_with_every_found_value_journaled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUDA_VERSION", vast.RUNTIME["cuda"])
    monkeypatch.setattr(vast, "FAKE_TORCH_CUDA", "12.9")
    out = tmp_path / "data" / "dry"
    with pytest.raises(vast.VastError, match=r"runtime differs from the pin in \['cuda'\]"):
        vast.dry_run(out, _dry_config(tmp_path))
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    (step,) = [e for e in events if e["event"] == "step" and e["name"] == "preflight"]
    assert step["ok"] is False and "['cuda']" in step["detail"]["error"]
    assert _runtime_found(step["detail"]) == {
        "vllm": "0.34.0",
        "torch": "2.14.0",
        "cuda": "12.9",
        "transformers": "5.20.0",
        "driver": "595.91.07",
        "cuda_env": vast.RUNTIME["cuda"],
    }
    assert step["detail"]["vllm_cli"].splitlines()[-1] == "0.34.0"
    assert step["detail"]["gpu"].startswith("Fake A100")
    assert [e["event"] for e in events].count("absence_verified") == 1


@pytest.mark.slow
@pytest.mark.parametrize("session_cuda", ["12.4", "13.0.3", None])
def test_preflight_reads_cuda_from_torch_so_a_wrong_or_missing_ssh_cuda_version_passes(
    tmp_path: Path, monkeypatch, session_cuda
) -> None:
    if session_cuda is None:
        monkeypatch.delenv("CUDA_VERSION", raising=False)
    else:
        monkeypatch.setenv("CUDA_VERSION", session_cuda)
    result = vast.dry_run(tmp_path / "data" / "dry", _dry_config(tmp_path))
    (step,) = [s for s in result["steps"] if s["name"] == "preflight"]
    assert step["ok"] is True
    assert step["detail"]["found"]["cuda"] == vast.RUNTIME["cuda"] == "13.0"
    assert step["detail"]["found"]["cuda_env"] == session_cuda
    assert result["steps"][-1]["name"] == "destroy"


def test_driver_is_the_last_field_of_the_first_nvidia_smi_line() -> None:
    assert vast._driver("NVIDIA A40, 46068 MiB, 580.95.05\nNVIDIA A40, 46068 MiB, 1") == "580.95.05"
    assert vast._driver("") is None
    assert vast._driver("garbage") is None


@pytest.mark.slow
def test_a_failed_arm_closes_the_attempt_without_any_create(tmp_path: Path, monkeypatch) -> None:
    def broken_arm(*args, **kwargs):
        raise vast.VastError("arm refused: Linger is not on")

    monkeypatch.setattr(vast, "arm", broken_arm)
    out = tmp_path / "data" / "dry"
    with pytest.raises(vast.VastError, match="Linger"):
        vast.dry_run(out, _dry_config(tmp_path))
    log = [json.loads(line) for line in (out / "fake-api.jsonl").read_text().splitlines()]
    assert not any(e["method"] in ("PUT", "DELETE") for e in log)
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    (closed,) = [e for e in vast.Journal(journal_path).replay() if e["event"] == "closed"]
    assert closed["reason"] == "no_instance_created"


class _JsonHost:
    """Answers each host argv by its first word or `hostrun.py` command; `get` copies a file."""

    def __init__(self, stdout: dict[str, bytes], archive: Path | None = None) -> None:
        self.stdout, self.archive, self.got = stdout, archive, []

    def run(self, argv, timeout):
        name = argv[2] if argv[:2] == ["python3", vast.HOSTRUN] else argv[0]
        return vast.proc.RunResult(0, self.stdout[name], b"", False, False, 0.0)

    def get(self, remote, local, timeout=1800.0):
        self.got.append(remote)
        local.write_bytes(self.archive.read_bytes())


_FOUND = {
    **{name: vast.RUNTIME[name] for name in ("vllm", "torch", "cuda")},
    "transformers": "5.20.0",
    "disk_free_bytes": 10**12,
}
_NOT_OBJECTS = [b"Traceback: not json\n", b"[1, 2]\n", b'"text"\n', b"", b"[" * 200_000]
_NOT_OBJECT_IDS = ["not-json", "list", "string", "empty", "deeply-nested"]


def _preflight_host(found: bytes) -> _JsonHost:
    return _JsonHost(
        {
            "nvidia-smi": b"Fake A100, 81920 MiB, 595.91.07\n",
            "vllm": vast.RUNTIME["vllm"].encode() + b"\n",
            "preflight": found,
            "selftest": b'{"ok": true}\n',
        }
    )


def _preflight(host) -> dict:
    detail: dict = {}
    package = {"models": [{"name": "m"}], "disk_margin_bytes": 0}
    vast.preflight(host, package, {"m": {"files": [{"size": 1}]}}, 60, detail)
    return detail


def test_preflight_takes_the_json_object_the_host_prints() -> None:
    detail = _preflight(_preflight_host(json.dumps(_FOUND).encode()))
    assert detail["found"] == {**_FOUND, "driver": "595.91.07"}


@pytest.mark.parametrize("stdout", _NOT_OBJECTS, ids=_NOT_OBJECT_IDS)
def test_preflight_refuses_host_stdout_that_is_not_a_json_object(stdout) -> None:
    with pytest.raises(vast.VastError, match=r"^host: preflight printed "):
        _preflight(_preflight_host(stdout))


def test_download_takes_the_json_object_pack_prints(tmp_path: Path) -> None:
    archive = _archive(tmp_path, {"a.json": b"{}"})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    host = _JsonHost({"pack": json.dumps({"archive_sha256": digest}).encode()}, archive)
    (tmp_path / "attempt").mkdir()
    journal = vast.Journal.create(tmp_path / "attempt", vast.new_attempt_id(), "k3y")
    assert vast.download(host, tmp_path / "attempt", journal, 60) == {
        "archive_sha256": digest,
        "files": 1,
        "disk_free_bytes": None,
    }
    assert host.got == [f"{vast.WORKSPACE}/results.tar"]


@pytest.mark.parametrize("stdout", _NOT_OBJECTS, ids=_NOT_OBJECT_IDS)
def test_download_refuses_pack_stdout_that_is_not_a_json_object(tmp_path: Path, stdout) -> None:
    host = _JsonHost({"pack": stdout})
    (tmp_path / "attempt").mkdir()
    journal = vast.Journal.create(tmp_path / "attempt", vast.new_attempt_id(), "k3y")
    with pytest.raises(vast.VastError, match=r"^host: pack printed "):
        vast.download(host, tmp_path / "attempt", journal, 60)
    assert host.got == []


@pytest.mark.slow
@pytest.mark.parametrize("stdout", [b"Traceback: not json\n", b"[1, 2]\n"], ids=["text", "list"])
def test_a_preflight_without_a_json_object_is_one_vast_line_exit_2_and_still_destroys(
    tmp_path: Path, monkeypatch, capsys, stdout
) -> None:
    real_run = vast.LocalHost.run

    def run(self, argv, timeout):
        done = real_run(self, argv, timeout)
        if argv[:3] == ["python3", vast.HOSTRUN, "preflight"]:
            return dataclasses.replace(done, stdout=stdout)
        return done

    monkeypatch.setattr(vast.LocalHost, "run", run)
    config_ = _dry_config(tmp_path)
    monkeypatch.setattr(vast.config, "load", lambda: config_)
    out = tmp_path / "data" / "dry"
    assert vast.main(["dry-run", "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("vast: host: preflight printed ")
    assert captured.err.count("\n") == 1 and "Traceback (most recent" not in captured.err
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    steps = [e for e in events if e["event"] == "step"]
    (step,) = [e for e in steps if e["name"] == "preflight"]
    assert step["ok"] is False and step["detail"]["error"].startswith("VastError: host: preflight")
    assert [e["name"] for e in steps][-1] == "destroy" and steps[-1]["ok"] is True
    assert [e["event"] for e in events].count("absence_verified") == 1


@pytest.mark.parametrize("command", ["preflight", "pack"])
def test_the_refusal_names_the_command_and_never_echoes_host_stdout(
    tmp_path: Path, command
) -> None:
    secret = b"host-said-s3cr3t-token"
    with pytest.raises(vast.VastError) as raised:
        if command == "preflight":
            _preflight(_preflight_host(secret + b"\n"))
        else:
            (tmp_path / "attempt").mkdir()
            journal = vast.Journal.create(tmp_path / "attempt", vast.new_attempt_id(), "k3y")
            vast.download(_JsonHost({"pack": secret}), tmp_path / "attempt", journal, 60)
    assert str(raised.value) == f"host: {command} printed no JSON"
    assert secret.decode() not in str(raised.value)


# --- a server that fails (#160) ------------------------------------------------------------


def _with_fake_vllm_mode(monkeypatch, mode: dict) -> None:
    real = vast._fake_host_root

    def host_root(root: Path) -> dict:
        env = real(root)
        (root / vast_fake.FAKE_MODE_FILE).write_text(json.dumps(mode))
        return env

    monkeypatch.setattr(vast, "_fake_host_root", host_root)


def _failed_dry_run(tmp_path: Path, match: str) -> tuple[Path, list[dict]]:
    out = tmp_path / "data" / "dry"
    with pytest.raises(vast.VastError, match=match):
        vast.dry_run(out, _dry_config(tmp_path))
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    return journal_path.parent, vast.Journal(journal_path).replay()


def _after_failure(events: list[dict], step: str) -> list[str]:
    """Event names from the failed step on, the failed step as `step:<name>`."""
    names = [
        f"step:{e['name']}" if e["event"] == "step" else e["event"]
        for e in events
        if e["event"] != "step" or not e["ok"] or e["name"] == "destroy"
    ]
    return names[names.index(f"step:{step}") :]


def _no_key_anywhere(out: Path) -> None:
    key = (out / "fake-api-key").read_text().strip()
    for path in out.rglob("*"):
        if path.is_file() and path.name != "fake-api-key":
            assert key.encode() not in path.read_bytes(), path


@pytest.mark.slow
def test_a_server_that_exits_fails_smoke_fast_with_its_exit_code(tmp_path, monkeypatch) -> None:
    _with_fake_vllm_mode(monkeypatch, {"exit": 3, "line": "fake engine core died"})
    attempt, events = _failed_dry_run(
        tmp_path,
        r"no model gave a Gate 0 verdict: qwen25omni: smoke: .*exited before /health "
        r"\(reason None, exit code 3\)",
    )
    smokes = [e for e in events if e["event"] == "step" and e["name"] == "smoke"]
    assert [(e["model"], e["ok"]) for e in smokes] == [("qwen25omni", False), ("qwen3omni", False)]
    smoke = smokes[0]
    assert smoke["detail"]["server_exit"] == {"reason": None, "returncode": 3}
    assert smoke["elapsed_s"] < vast.DRY_INTERVALS["health_timeout_s"] / 4
    assert "state_errors" not in smoke["detail"]
    assert _after_failure(events, "smoke") == [
        "step:smoke",
        "server_evidence",
        "server_stopped",
        "model_outcome",
        "weights_started",
        "gpu_free",
        "step:smoke",
        "server_evidence",
        "server_stopped",
        "model_outcome",
        "download_verified",
        "destroy_requested",
        "destroy_response",
        "absence_verified",
        "closed",
        "step:destroy",
        "credit_after_close",
    ]
    evidence = next(e for e in events if e["event"] == "server_evidence")
    assert "fake engine core died" in evidence["tails"]["server.log"]
    assert '"returncode": 3' in evidence["tails"]["server-stats.jsonl"]


@pytest.mark.slow
def test_a_failed_smoke_journals_the_log_tail_and_the_process_list_before_destroy(
    tmp_path, monkeypatch
) -> None:
    _with_fake_vllm_mode(monkeypatch, {"hang": True, "child": True, "line": "fake loading forever"})
    monkeypatch.setattr(vast, "DRY_INTERVALS", {**vast.DRY_INTERVALS, "health_timeout_s": 4.0})
    attempt, events = _failed_dry_run(tmp_path, "did not answer /health in time")
    smoke = next(e for e in events if e["event"] == "step" and e["name"] == "smoke")
    assert "server_exit" not in smoke["detail"]
    assert _after_failure(events, "smoke")[:4] == [
        "step:smoke",
        "server_evidence",
        "server_stopped",
        "model_outcome",
    ]
    evidence = next(e for e in events if e["event"] == "server_evidence")
    assert evidence["model"] == "qwen25omni" and evidence["file"] == "evidence-qwen25omni.json"
    assert "fake loading forever" in evidence["tails"]["server.log"]
    assert all(
        len(tail.encode()) <= vast.EVIDENCE_TAIL_BYTES for tail in evidence["tails"].values()
    )
    assert (
        set(evidence["sizes"])
        == set(evidence["tails"])
        == {
            "server.log",
            "hostrun.log",
            "guard.log",
            "server-stats.jsonl",
        }
    )
    fakes = [
        p
        for p in evidence["processes"]
        if "vast_fake" in p["cmdline"] or "bin/vllm serve" in p["cmdline"]
    ]
    pids = {p["pid"] for p in fakes}
    assert any(p["ppid"] in pids for p in fakes), fakes
    path = attempt / "evidence-qwen25omni.json"
    assert path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(path.read_text())
    assert saved["kind"] == "server_evidence"
    assert saved["processes"] == evidence["processes"]
    _no_key_anywhere(tmp_path / "data" / "dry")


@pytest.mark.slow
def test_a_failed_run_also_journals_evidence(tmp_path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise vast.VastError("gate0 runner broke")

    monkeypatch.setattr(vast.gate0, "run_spike", broken)
    attempt, events = _failed_dry_run(tmp_path, "qwen25omni: run: gate0 runner broke")
    assert _after_failure(events, "run")[:4] == [
        "step:run",
        "server_evidence",
        "server_stopped",
        "model_outcome",
    ]
    outcomes = [
        (e["model"], e["outcome"], e["step"]) for e in events if e["event"] == "model_outcome"
    ]
    assert outcomes == [("qwen25omni", "failed", "run"), ("qwen3omni", "failed", "run")]
    evidence = next(e for e in events if e["event"] == "server_evidence")
    assert "serving qwen25omni" in evidence["tails"]["server.log"]
    assert (attempt / "evidence-qwen25omni.json").is_file()


@pytest.mark.slow
def test_a_server_that_dies_during_a_run_with_a_verdict_journals_evidence(
    tmp_path, monkeypatch
) -> None:
    out = tmp_path / "data" / "dry"

    def run_after_the_server_died(*args, served_model, **kwargs):
        if served_model == "qwen25omni":
            stats = out / "host-root" / "out" / served_model / "server-stats.jsonl"
            lines = [json.loads(line) for line in stats.read_text().splitlines()]
            (started,) = [line for line in lines if line["event"] == "started"]
            os.kill(started["pid"], 9)
            ends = time.monotonic() + 30
            while '"event": "exit"' not in stats.read_text() and time.monotonic() < ends:
                time.sleep(0.1)
        return {"verdict": "NOT READY", "reason": "every request failed"}

    monkeypatch.setattr(vast.gate0, "run_spike", run_after_the_server_died)
    vast.dry_run(out, _dry_config(tmp_path))
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    runs = {e["model"]: e for e in events if e["event"] == "step" and e["name"] == "run"}
    assert runs["qwen25omni"]["ok"] and runs["qwen25omni"]["detail"]["verdict"] == "NOT READY"
    assert runs["qwen25omni"]["detail"]["server_exit"] == {"reason": None, "returncode": -9}
    assert "server_exit" not in runs["qwen3omni"]["detail"]
    evidence, second = [e for e in events if e["event"] == "server_evidence"]
    assert evidence["model"] == "qwen25omni" and "processes" in evidence
    assert second["model"] == "qwen3omni" and '"returncode": -9' not in str(second["tails"])
    assert '"returncode": -9' in evidence["tails"]["server-stats.jsonl"]
    assert (journal_path.parent / "evidence-qwen25omni.json").is_file()
    assert events.index(runs["qwen25omni"]) < events.index(evidence)
    extended = next(e for e in events if e["event"] == "step" and e["name"] == "extended")
    assert (
        extended["model"] == "qwen25omni" and extended["detail"]["stop_reason"] == "server_exited"
    )
    assert all(b["sent"] == 0 for b in extended["detail"]["blocks"].values())


@pytest.mark.slow
def test_a_cancellation_during_evidence_after_a_run_still_destroys(tmp_path, monkeypatch) -> None:
    real_run = vast.LocalHost.run

    def run(self, argv, timeout):
        if argv[:3] == ["python3", vast.HOSTRUN, "evidence"]:
            raise KeyboardInterrupt
        return real_run(self, argv, timeout)

    monkeypatch.setattr(vast.LocalHost, "run", run)
    out = tmp_path / "data" / "dry"
    with pytest.raises(KeyboardInterrupt):
        vast.dry_run(out, _dry_config(tmp_path))
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    runs = [e for e in events if e["event"] == "step" and e["name"] in ("run", "extended")]
    assert [(e["model"], e["ok"]) for e in runs] == [("qwen25omni", True), ("qwen25omni", True)]
    assert not [e for e in events if e["event"] == "step" and e.get("model") == "qwen3omni"]
    (evidence,) = [e for e in events if e["event"] == "server_evidence"]
    assert evidence["error"] == "KeyboardInterrupt: "
    assert any(e["event"] == "download_verified" for e in events)
    names = [e["event"] for e in events]
    assert names.index("server_stopped") < names.index("destroy_requested")
    assert vast.summarize(events)["closed"] is not None


@pytest.mark.slow
@pytest.mark.parametrize("failure", ["not-json", "raises"])
def test_evidence_failure_never_blocks_destroy(tmp_path, monkeypatch, failure) -> None:
    _with_fake_vllm_mode(monkeypatch, {"exit": 3})
    real_run = vast.LocalHost.run

    def run(self, argv, timeout):
        if argv[:3] == ["python3", vast.HOSTRUN, "evidence"]:
            if failure == "raises":
                raise OSError("ssh dropped")
            done = real_run(self, argv, timeout)
            return dataclasses.replace(done, stdout=b"Traceback: not json\n")
        return real_run(self, argv, timeout)

    monkeypatch.setattr(vast.LocalHost, "run", run)
    attempt, events = _failed_dry_run(tmp_path, "exited before /health")
    evidence = next(e for e in events if e["event"] == "server_evidence")
    assert evidence["model"] == "qwen25omni" and "processes" not in evidence
    assert (
        evidence["error"]
        == {
            "not-json": "VastError: host: evidence printed no JSON",
            "raises": "OSError: ssh dropped",
        }[failure]
    )
    assert _after_failure(events, "smoke")[1:4] == [
        "server_evidence",
        "server_stopped",
        "model_outcome",
    ]
    assert _after_failure(events, "smoke")[-6:-2] == [
        "destroy_requested",
        "destroy_response",
        "absence_verified",
        "closed",
    ]
    assert [e["event"] for e in events[-2:]] == ["step", "credit_after_close"]
    assert events[-2]["name"] == "destroy" and events[-2]["ok"]
    assert not (attempt / "evidence-qwen25omni.json").exists()


class _StateHost:
    """Answers `hostrun.py state` with the next prepared answer; an exception is raised."""

    def __init__(self, answers: list) -> None:
        self.answers = answers

    def run(self, argv, timeout):
        assert argv[:3] == ["python3", vast.HOSTRUN, "state"]
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return vast.proc.RunResult(0, answer, b"", False, False, 0.0)


class _NeverHealthy:
    def request(self, route, timeout):
        return 0, b"", "refused"


def _state(**fields) -> bytes:
    base = {"kind": "serve_state", "started": True, "serve_alive": True, "exit": None}
    return json.dumps({**base, **fields}).encode()


@pytest.mark.parametrize(
    ("answer", "reason", "code"),
    [
        (_state(exit={"reason": "rss_limit", "returncode": -15, "t": 1.0}), "rss_limit", -15),
        (_state(serve_alive=False), "hostrun_gone", None),
        (_state(started=False, serve_alive=False), "not_started", None),
    ],
    ids=["exit", "hostrun-gone", "not-started"],
)
def test_wait_health_ends_when_the_state_shows_the_server_is_gone(answer, reason, code) -> None:
    host, detail = _StateHost([answer]), {}
    journal = vast.Journal(Path("unused"), "k3y")
    started = time.monotonic()
    with pytest.raises(vast.ServerExited) as raised:
        vast._wait_health(
            _NeverHealthy(),
            60.0,
            lambda: vast._state_or_none(host, "/workspace/out/m", journal, detail),
            0.0,
            0.0,
        )
    assert (raised.value.reason, raised.value.returncode) == (reason, code)
    assert time.monotonic() - started < 10


@pytest.mark.slow
def test_a_failing_state_call_is_counted_and_never_ends_the_wait() -> None:
    host = _StateHost([OSError("ssh k3y-secret dropped"), b"not json", _state()])
    detail: dict = {}
    journal = vast.Journal(Path("unused"), "k3y-secret")
    with pytest.raises(vast.VastError, match="did not answer /health in time"):
        vast._wait_health(
            _NeverHealthy(),
            2.5,
            lambda: vast._state_or_none(host, "/workspace/out/m", journal, detail),
            0.0,
            100.0,
        )
    assert detail["state_errors"]["count"] == 2
    assert detail["state_errors"]["last"] == "VastError: host: state printed no JSON"


# --- the two-model package of #149 ---------------------------------------------------------

DECISION_V1 = REPO / "docs/decisions/0024-gate0-qwen25-omni.md"
DECISION_V2 = REPO / "docs/decisions/0023-gate0-qwen3-omni.md"


def _reference_package(**changes) -> dict:
    """The shape of the 0028 package: both models, the cap and limits the owner approved."""
    return {
        **_package(),
        "ceiling_usd": 1.2,
        "early_stop_usd": 0.9,
        "deadline_minutes": 60,
        "disk_gb": 110,
        "disk_margin_bytes": 10 * 10**9,
        "models": [
            {
                "name": "qwen25omni",
                "gate0_date": "2026-11-19",
                "decision": str(DECISION_V1),
                "rss_limit_bytes": 18 * 10**9,
            },
            {
                "name": "qwen3omni",
                "gate0_date": "2026-11-21",
                "decision": str(DECISION_V2),
                "rss_limit_bytes": 26 * 10**9,
            },
        ],
        "preregistration": "docs/decisions/0028-gpu-reference-gate0.md",
        **changes,
    }


def _load(tmp_path: Path, package: dict, models=vast.MODELS) -> dict:
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    return vast.load_package(path, models)[0]


def test_the_reference_package_needs_110_gb_of_disk_and_refuses_less(tmp_path: Path) -> None:
    package = _load(tmp_path, _reference_package())
    assert vast.min_disk_gb(package, vast.MODELS) == 110
    with pytest.raises(vast.VastError, match="disk_gb 109 is below the 110 GB"):
        _load(tmp_path, _reference_package(disk_gb=109))
    single = _reference_package(disk_gb=40)
    single["models"] = single["models"][:1]
    assert vast.min_disk_gb(_load(tmp_path, single), vast.MODELS) == 40


def _model_change(index: int, **fields) -> dict:
    package = _reference_package()
    package["models"][index] = {**package["models"][index], **fields}
    return package


@pytest.mark.parametrize(
    ("package", "message"),
    [
        (_model_change(1, name="gemma"), "must be one of"),
        (_model_change(1, name="qwen25omni", rss_limit_bytes=10**9), "lists a model twice"),
        (_model_change(1, rss_limit_bytes=26 * 10**9 + 1), "over its 26000000000"),
        (_model_change(0, rss_limit_bytes=18 * 10**9 + 1), "over its 18000000000"),
        (_model_change(0, clips={"S0": "not a hash"}), "SHA-256 of their WAV"),
        (_model_change(0, clips={}), "SHA-256 of their WAV"),
    ],
    ids=["unknown-model", "twice", "qwen3omni-rss", "qwen25omni-rss", "bad-pin", "empty-pins"],
)
def test_a_reference_package_with_a_bad_model_entry_is_refused(
    tmp_path: Path, package, message
) -> None:
    with pytest.raises(vast.VastError, match=message):
        _load(tmp_path, package)


def test_a_model_missing_from_the_models_manifest_is_refused(tmp_path: Path) -> None:
    models = {"qwen25omni": vast.MODELS["qwen25omni"]}
    with pytest.raises(vast.VastError, match=r"must be one of \['qwen25omni'\]"):
        _load(tmp_path, _reference_package(), models)


def _controls_day(config_: config.Config, day: str, decision: Path) -> dict:
    return gate0.prepare_clips(day, config_, decision=decision, controls_only=True)


def _pins(manifest: dict) -> dict:
    return {clip["id"]: clip["wav_sha256"] for clip in manifest["clips"]}


@pytest.mark.parametrize(
    ("decisions", "message"),
    [
        ((DECISION_V1, DECISION_V1), "qwen3omni runs under GATE0-RULE v2, its decision holds v1"),
        ((DECISION_V2, DECISION_V2), "qwen25omni runs under GATE0-RULE v1, its decision holds v2"),
    ],
    ids=["v1-for-qwen3omni", "v2-for-qwen25omni"],
)
def test_a_model_under_another_rule_than_its_local_run_is_refused_before_any_call(
    tmp_path: Path, key_file, fake, decisions, message
) -> None:
    config_ = _dry_config(tmp_path)
    _controls_day(config_, "2026-11-19", DECISION_V1)
    _controls_day(config_, "2026-11-21", DECISION_V2)
    package = _reference_package()
    for entry, decision in zip(package["models"], decisions, strict=True):
        entry["decision"] = str(decision)
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    with pytest.raises(vast.VastError, match=message.replace("holds v", "holds GATE0-RULE v")):
        vast.run_attempt(
            path,
            api=vast.Api(fake.url, key_file[1]),
            make_host=None,
            config_=config_,
            runner=_no_system,
            dry_run=True,
            key_path=key_file[0],
        )
    assert fake.log == [] and not (config_.data_root / "gpu").exists()


def test_a_day_prepared_under_another_rule_or_another_clip_set_is_refused(tmp_path: Path) -> None:
    config_ = _dry_config(tmp_path)
    first = _controls_day(config_, "2026-11-19", DECISION_V1)
    _controls_day(config_, "2026-11-21", DECISION_V1)
    package = _reference_package()
    with pytest.raises(vast.VastError, match="2026-11-21 was prepared under another rule"):
        vast.check_days(package, config_)
    package["models"] = package["models"][:1]
    vast.check_days(package, config_)
    package["models"][0]["clips"] = _pins(first)
    with pytest.raises(vast.VastError, match="not the clip set the package pins"):
        vast.check_days(package, config_)
    missing = _reference_package(models=[{**package["models"][0], "gate0_date": "2026-11-30"}])
    with pytest.raises(
        vast.VastError, match="2026-11-30 is not prepared under the configured data_root"
    ):
        vast.check_days(missing, config_)


def test_an_absent_day_says_it_looked_under_the_configured_data_root_and_names_sound_scout_config(
    tmp_path: Path, key_file, fake
) -> None:
    config_ = _dry_config(tmp_path)
    _controls_day(config_, "2026-11-19", DECISION_V1)
    package = _reference_package()
    with pytest.raises(vast.VastError) as raised:
        vast.check_days(package, config_)
    message = str(raised.value)
    for part in ("2026-11-21", "configured data_root", "SOUND_SCOUT_CONFIG", "manifest.json"):
        assert part in message
    assert str(config_.data_root) not in message and str(tmp_path) not in message
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    with pytest.raises(vast.VastError) as through_deploy:
        vast.run_attempt(
            path,
            api=vast.Api(fake.url, key_file[1]),
            make_host=None,
            config_=config_,
            runner=_no_system,
            dry_run=True,
            key_path=key_file[0],
        )
    assert str(through_deploy.value) == message
    assert fake.log == [] and not (config_.data_root / "gpu").exists()


def test_a_full_day_off_its_clip_pins_is_refused(tmp_path: Path) -> None:
    config_ = _dry_config(tmp_path)
    v1, _ = _full_days(config_)
    package = _reference_package()
    package["models"] = package["models"][:1]
    package["models"][0]["clips"] = _pins(v1)
    vast.check_days(package, config_)
    wrong_hash = {**_pins(v1), "V2": "0" * 64}
    dropped = {k: v for k, v in _pins(v1).items() if k != "V2"}
    for clips in (wrong_hash, dropped):
        package["models"][0]["clips"] = clips
        with pytest.raises(vast.VastError, match="not the clip set the package pins"):
            vast.check_days(package, config_)
    package["models"][0]["clips"] = _pins(v1)
    wav = gate0.output_dir("2026-11-19", config_) / next(
        clip["file"] for clip in v1["clips"] if clip["id"] == "V2"
    )
    data = bytearray(wav.read_bytes())
    data[-1] ^= 1
    wav.write_bytes(bytes(data))
    with pytest.raises(vast.VastError, match="not the clip set the package pins"):
        vast.check_days(package, config_)


def _full_days(config_: config.Config) -> tuple[dict, dict]:
    """Full Gate 0 days as the local runs had them: 2026-11-19 under v1 with B1-B3, V1-V3 and the
    controls from synthetic voices, and 2026-11-21 under v2 as a byte copy of its clips."""

    def transcribe(path, run, config_):
        samples = np.frombuffer(gate0.wav_pcm(Path(path).read_bytes()), dtype="<i2") / 32767
        words = [
            {"word": " раз", "start": 0.1, "end": 0.5, "probability": 0.9},
            {"word": " два,", "start": 1.0, "end": 1.5, "probability": 0.9},
            {"word": " три", "start": 2.0, "end": 2.6, "probability": 0.9},
        ]
        return {
            "samples": samples,
            "sha256": "0" * 64,
            "duration_s": len(samples) / gate0.SAMPLE_RATE,
            "words": words,
            "transcript": "раз два, три",
        }

    for name in gate0.SOURCE_FILES:
        voice = speech_like_control(3.0, gate0.SAMPLE_RATE, f0=130.0)
        (config_.allowed_roots[0] / name).write_bytes(gate0.wav_bytes(gate0.pcm16(voice)))
    v1 = gate0.prepare_clips("2026-11-19", config_, decision=DECISION_V1, transcribe=transcribe)
    v2 = gate0.reuse_clips("2026-11-21", config_, source="2026-11-19", decision=DECISION_V2)
    return v1, v2


def _stack(tmp_path: Path, config_: config.Config, package: dict) -> Path:
    """`run_attempt` against the fakes, as the dry run drives it; the attempt directory."""
    key_path = tmp_path / "fake-api-key"
    key_path.write_text(secrets.token_urlsafe(32) + "\n")
    key_path.chmod(0o600)
    key = vast.read_api_key(key_path)
    env = vast._fake_host_root(tmp_path / "host-root")
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    fake = vast_fake.FakeVast(key).start()
    try:
        journal = vast.run_attempt(
            path,
            api=vast.Api(fake.url, key),
            make_host=lambda instance, attempt: vast.LocalHost(tmp_path / "host-root", env),
            config_=config_,
            runner=vast._refuse_system,
            intervals=vast.DRY_INTERVALS,
            dry_run=True,
            key_path=key_path,
            models=vast_fake.synthetic_models(),
        )
    finally:
        fake.stop()
    assert sum(e["method"] == "PUT" for e in fake.log) == 1
    assert not [i for i, inst in fake.instances.items() if fake._visible(inst)]
    return journal.dir


@pytest.mark.slow
def test_the_two_model_package_runs_each_full_day_under_its_rule_on_a_fresh_server(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(vast_fake, "OFFER", {**vast_fake.OFFER, "dph_total": 0.7})
    config_ = _dry_config(tmp_path)
    v1, v2 = _full_days(config_)
    assert [clip["id"] for clip in v1["clips"]] == [
        "B1", "V1", "B2", "V2", "B3", "V3", "S0", "N1", "T1",
    ]  # fmt: skip
    package = _reference_package(disk_gb=150, disk_margin_bytes=0)
    package["models"][0]["clips"] = _pins(v1)
    package["models"][1]["clips"] = _pins(v2)
    attempt = _stack(tmp_path, config_, package)
    events = vast.Journal(attempt / vast.JOURNAL_FILE).replay()
    for day, rule, served in (
        ("2026-11-19", "v1", "qwen25omni"),
        ("2026-11-21", "v2", "qwen3omni"),
    ):
        run = json.loads((config_.data_root / day / "attempts/1/run.json").read_text())
        assert run["rule"] == f"GATE0-RULE {rule}" and run["served_model"] == served
        assert run["requests_planned"] == 42
        trace = (config_.data_root / day / "attempts/1/trace.jsonl").read_text().splitlines()
        assert sum(json.loads(line).get("sent") is True for line in trace) == 42
    stopped = next(e for e in events if e["event"] == "server_stopped")
    evidence = next(e for e in events if e["event"] == "server_evidence")
    second_smoke = next(e for e in events if e["event"] == "step" and e.get("model") == "qwen3omni")
    assert evidence["model"] == stopped["model"] == "qwen25omni" and stopped["returncode"] == 0
    assert events.index(evidence) < events.index(stopped) < events.index(second_smoke)
    stats = {
        model: [
            json.loads(line)
            for line in (attempt / "host" / model / "server-stats.jsonl").read_text().splitlines()
        ]
        for model in ("qwen25omni", "qwen3omni")
    }
    (first_exit,) = [r for r in stats["qwen25omni"] if r["event"] == "exit"]
    (second_start,) = [r for r in stats["qwen3omni"] if r["event"] == "started"]
    assert first_exit["reason"] == "stopped" and first_exit["t"] <= second_start["t"]
    assert events[-2]["event"] == "step" and events[-2]["name"] == "destroy" and events[-2]["ok"]
    assert events[-1]["event"] == "credit_after_close"


@pytest.mark.slow
def test_a_server_that_does_not_stop_ends_the_attempt_before_the_second_model(
    tmp_path, monkeypatch
) -> None:
    real_run = vast.LocalHost.run

    def run(self, argv, timeout):
        done = real_run(self, argv, timeout)
        if argv[:3] == ["python3", vast.HOSTRUN, "stop"]:
            return dataclasses.replace(done, returncode=1)
        return done

    monkeypatch.setattr(vast.LocalHost, "run", run)
    attempt, events = _failed_dry_run(tmp_path, "server of qwen25omni did not stop")
    assert not [e for e in events if e["event"] == "step" and e.get("model") == "qwen3omni"]
    (outcome,) = [e for e in events if e["event"] == "model_outcome"]
    assert outcome["model"] == "qwen25omni" and outcome["outcome"] == "ok"
    assert any(e["event"] == "download_verified" for e in events)
    (stopped,) = [e for e in events if e["event"] == "server_stopped"]
    assert stopped["returncode"] == 1
    assert [e["event"] for e in events].count("absence_verified") == 1
    assert events[-2]["name"] == "destroy" and events[-2]["ok"]
    assert events[-1]["event"] == "credit_after_close"


@pytest.mark.parametrize(
    ("dph", "refused"), [(1.2, True), (0.7, False)], ids=["over-cap", "within-cap"]
)
def test_the_estimate_covers_the_whole_package_up_to_the_deadline_and_the_cutoff_grace(
    tmp_path: Path, key_file, fake, monkeypatch, dph, refused
) -> None:
    monkeypatch.setattr(vast_fake, "OFFER", {**vast_fake.OFFER, "dph_total": dph})
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1])
    package = _reference_package(envelope={"dph_total": {"lte": 2.0}})
    assert vast.estimate_hours(package) == pytest.approx(75 / 60)
    assert vast.estimate_usd(vast_fake.OFFER, 1.25, package["in_gb"], 1.0) == pytest.approx(
        1.25 * dph + 102.95 * 0.0026042 + 0.0039063
    )
    if refused:
        with pytest.raises(vast.VastError, match=r"estimate 1\.77 USD is over the ceiling"):
            vast.create_instance(api, journal, package, _no_system)
        assert fake.count("PUT") == 0
    else:
        assert vast.create_instance(api, journal, package, _no_system)
        assert fake.count("PUT") == 1


def test_the_0028_launch_names_the_gate0_config_and_one_package_place(
    tmp_path: Path, monkeypatch
) -> None:
    text = (REPO / "docs/decisions/0028-gpu-reference-gate0.md").read_text(encoding="utf-8")
    launch = text.split("\n## Запуск\n", 1)[1].split("\n## ", 1)[0]
    package_section = text.split("\n## Пакет\n", 1)[1].split("```json\n", 1)[0]
    place = "data/gate0/gpu/packages/gpu-ref-149.json"
    assert "SOUND_SCOUT_CONFIG=local/gate0.toml" in launch and "data/gate0/" in launch
    assert f"--package {place}" in launch and place in package_section
    assert re.findall(r"[\w<>/-]*gpu/packages/[\w.-]*", launch + package_section) == [
        place,
        place,
        "<data_root>/gpu/packages/",
    ]
    template = launch.split("```toml\n", 1)[1].split("```", 1)[0]
    template = "\n".join(line.strip() for line in template.splitlines())
    (tmp_path / "data/gate0").mkdir(parents=True)
    (tmp_path / "data/dev").mkdir()
    config_file = tmp_path / "local" / "gate0.toml"
    config_file.parent.mkdir()
    config_file.write_text(template.replace("<checkout>", str(tmp_path)), encoding="utf-8")
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(config_file))
    loaded = config.load()
    assert loaded.data_root == (tmp_path / "data/gate0").resolve()
    assert loaded.network.allow is False and loaded.backend == "none"


def test_the_0028_package_loads_with_a_quote_and_pins_the_rules_and_clips_it_lists(
    tmp_path: Path,
) -> None:
    text = (REPO / "docs/decisions/0028-gpu-reference-gate0.md").read_text(encoding="utf-8")
    block = text.split("\n## Пакет\n", 1)[1].split("```json\n", 1)[1].split("\n```", 1)[0]
    package = json.loads(block)
    assert "offer_id" not in package and "dph_total" not in package["envelope"]
    package["offer_id"] = vast_fake.OFFER_ID
    package["envelope"]["dph_total"] = {"lte": 0.9}
    loaded = _load(tmp_path, package)
    assert loaded["disk_gb"] == vast.min_disk_gb(loaded, vast.MODELS) == 110
    assert loaded["envelope"]["disk_space"] == {"gte": 110}
    assert (loaded["ceiling_usd"], loaded["early_stop_usd"], loaded["deadline_minutes"]) == (
        1.2,
        0.9,
        60,
    )
    assert [entry["name"] for entry in loaded["models"]] == ["qwen25omni", "qwen3omni"]
    for entry in loaded["models"]:
        rule = gate0.rule_sha256((REPO / entry["decision"]).read_text(encoding="utf-8"))
        assert gate0.rule_name(rule) == vast.GATE0_RUNS[entry["name"]]["rule"]
        assert rule in text
        assert entry["rss_limit_bytes"] == vast.GATE0_RUNS[entry["name"]]["rss_ceiling_bytes"]
        assert list(entry["clips"]) == ["B1", "V1", "B2", "V2", "B3", "V3", "S0", "N1", "T1"]
        for clip_id, digest in entry["clips"].items():
            assert re.search(rf"^\| `{clip_id}` \|.*\| `{digest}` \|", text, re.MULTILINE)
    assert loaded["models"][0]["clips"] == loaded["models"][1]["clips"]
    assert gate0.PROMPT_SHA256 in text and gate0.questions_sha256() in text
    for model in vast.MODELS.values():
        assert model["revision"] in text
    argv = vast.vllm_argv("/m", "m")
    assert "`--no-enable-prefix-caching`" in text and "`--mm-processor-cache-gb 0`" in text
    assert "--no-enable-prefix-caching" in argv


# --- the $3 package of #149: offers, the extended set, the soft stop ------------------------


def test_create_passes_over_a_gone_or_dear_offer_to_the_next_approved_one(
    tmp_path: Path, key_file, fake
) -> None:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    second, third = vast_fake.OFFER_ID + 1, vast_fake.OFFER_ID + 2
    fake.offers = {
        second: {**vast_fake.OFFER, "ask_contract_id": second, "dph_total": 9.0},
        third: {**vast_fake.OFFER, "ask_contract_id": third},
    }
    package = _package(offer_ids=[vast_fake.OFFER_ID, second, third], deadline_minutes=100)
    del package["offer_id"]
    api = vast.Api(fake.url, key_file[1])
    assert vast.create_instance(api, journal, package, _no_system)
    events = journal.replay()
    skipped = [(e["offer_id"], e["reason"]) for e in events if e["event"] == "offer_skipped"]
    assert skipped[0] == (vast_fake.OFFER_ID, f"offer {vast_fake.OFFER_ID} is not available")
    assert skipped[1][0] == second and "dph_total" in skipped[1][1]
    (requested,) = [e for e in events if e["event"] == "create_requested"]
    assert requested["offer_id"] == third
    assert requested["prices"]["dph_total"] == vast_fake.OFFER["dph_total"]
    assert fake.count("PUT") == 1 and fake.count("PUT", f"/api/v0/asks/{third}/") == 1


def test_no_approved_offer_means_no_put_and_names_every_reason(
    tmp_path: Path, key_file, fake
) -> None:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    fake.offers = {}
    package = _package(offer_ids=[11, 12], deadline_minutes=100)
    del package["offer_id"]
    with pytest.raises(vast.VastError, match="create refused: 11: offer 11 .*; 12: offer 12"):
        vast.create_instance(vast.Api(fake.url, key_file[1]), journal, package, _no_system)
    assert fake.count("PUT") == 0


@pytest.mark.slow
def test_the_extended_set_must_match_its_pins_and_hold_no_gate0_clip(tmp_path: Path) -> None:
    config_ = _dry_config(tmp_path)
    found = gpu_ext.fixture(config_.data_root / "gpu-ext-x")
    extended = {"dir": "gpu-ext-x", **{k: found[k] for k in ("manifest_sha256", "plan_sha256")}}
    package = {"models": [{"clips": {"B1": "f" * 64}}], "extended": extended}
    directory, manifest, entries = vast.check_extended(package, config_)
    assert directory.name == "gpu-ext-x" and len(entries) == sum(
        c["requests"] for c in found["counts"].values()
    )
    stolen = manifest["clips"][0]["wav_sha256"]
    with pytest.raises(vast.VastError, match="pilot-1 or of a Gate 0 day"):
        vast.check_extended({**package, "models": [{"clips": {"B1": stolen}}]}, config_)
    with pytest.raises(vast.VastError, match="differs from the SHA-256 the package pins"):
        vast.check_extended({**package, "extended": {**extended, "plan_sha256": "0" * 64}}, config_)
    assert vast.check_extended({"models": []}, config_) is None


def _soft_run(tmp_path: Path, fake, key_file, **package) -> vast.Run:
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1])
    return vast.Run(api, journal, {**_package(), **package}, {}, None, vast.DRY_INTERVALS, None)


def test_the_soft_stop_names_the_soft_end_the_spend_and_a_watchdog_destroy(
    tmp_path: Path, key_file, fake
) -> None:
    run = _soft_run(tmp_path, fake, key_file, spend_soft_usd=2.0)
    assert vast.soft_stop(run) is None
    run.soft_end = time.time() - 1
    assert vast.soft_stop(run) == "soft_end"
    run.soft_end = None
    run.journal.append(
        "create_requested",
        offer_id=1,
        prices={"dph_total": 1e7, "inet_down_cost": 0.0, "inet_up_cost": 0.0},
    )
    run.journal.append("create_response", instance_id=7)
    time.sleep(0.01)
    assert vast.spent_estimate(run) > 2.0 and vast.soft_stop(run) == "spend"
    run.package["spend_soft_usd"] = 1e9
    assert vast.soft_stop(run) is None
    fake.credit = 1.0
    run.package["spend_soft_usd"] = 9.0
    vast.check_credit(run)
    seen = [e for e in run.journal.replay() if e["event"] == "credit_seen"]
    assert seen[-1]["spent_usd"] == 9.0
    assert run.spend_seen and vast.soft_stop(run) == "spend"
    run.journal.append("destroy_requested", instance_id=7, reason="early_stop")
    assert vast.soft_stop(run) == "watchdog"
    run.fatal = "ssh_lost"
    assert vast.soft_stop(run) == "ssh_lost"


def test_the_0028_addendum_package_loads_with_a_quote_and_pins_the_set_and_its_code(
    tmp_path: Path,
) -> None:
    text = (REPO / "docs/decisions/0028-gpu-reference-gate0.md").read_text(encoding="utf-8")
    addendum = text.split("\n## Дополнение 2026-11-23: пакет $3 и расширенный набор\n", 1)[1]
    block = addendum.split("```json\n", 1)[1].split("\n```", 1)[0]
    package = json.loads(block)
    assert "offer_ids" not in package and "dph_total" not in package["envelope"]
    package["offer_ids"] = [vast_fake.OFFER_ID, vast_fake.OFFER_ID + 1]
    package["envelope"]["dph_total"] = {"lte": 1.2}
    loaded = _load(tmp_path, package)
    assert (loaded["ceiling_usd"], loaded["early_stop_usd"], loaded["deadline_minutes"]) == (
        3.0,
        2.25,
        100,
    )
    assert loaded["early_stop_usd"] == 0.75 * loaded["ceiling_usd"]
    assert loaded["deadline_minutes"] + vast.LOST_CREATE_GRACE_S / 60 <= 120
    assert loaded["soft_end_minutes"] == loaded["deadline_minutes"] - 15
    assert loaded["envelope"]["disk_space"] == {"gte": loaded["disk_gb"]} == {"gte": 150}
    worst = vast.estimate_usd(
        {"dph_total": 1.2, "inet_down_cost": 0.0043, "inet_up_cost": 0.0043},
        vast.estimate_hours(loaded),
        loaded["in_gb"],
        loaded["out_gb"],
    )
    assert worst == pytest.approx(2.748, abs=0.001) and worst <= loaded["ceiling_usd"] - 0.25
    assert [e["name"] for e in loaded["models"]] == ["qwen25omni", "qwen3omni"]
    assert loaded["models"][0]["slot_minutes"] == 25 and "slot_minutes" not in loaded["models"][1]
    for entry in loaded["models"]:
        assert list(entry["clips"]) == ["B1", "V1", "B2", "V2", "B3", "V3", "S0", "N1", "T1"]
    extended = loaded["extended"]
    for name in ("manifest.json", "plan.json"):
        digest = extended[f"{name.split('.')[0]}_sha256"]
        assert f"| `{name}` | `{digest}` |" in addendum
    assert f"`{gpu_ext.questions_sha256()}`" in addendum
    assert f"`{gate0.questions_sha256()}`" in addendum and f"`{gate0.PROMPT_SHA256}`" in addendum
    assert f"| `gpu_ext.py` при `prepare` (план и анализ) | `{gpu_ext.code_sha256()}` |" in addendum
    assert f"`{gpu_ext.UTTERANCES_SERIES_3_SHA256}`" in addendum
    rows = re.findall(r"^\| `(E-[0-9a-f]{8})` \| (E-[CPR]) \|", addendum, re.MULTILINE)
    assert len(rows) == 148 and len({r[0] for r in rows}) == 148
    blocks = [block for _, block in rows]
    assert (blocks.count("E-C"), blocks.count("E-P"), blocks.count("E-R")) == (12, 40, 96)
    assert "7085422087" in addendum and "7065928031" in addendum
    assert (
        "--approval https://github.com/renidere/sound-scout-plugin/issues/149#issuecomment-7085422087"
        in addendum
    )


def test_the_addendum_names_the_set_on_disk_when_the_station_has_it() -> None:
    root = REPO / "data" if (REPO / "data").is_dir() else REPO.parents[1] / "data"
    directory = root / "gate0" / "gpu-ext-149"
    if not (directory / "manifest.json").is_file():
        pytest.skip("the prepared set is on the station only")
    text = (REPO / "docs/decisions/0028-gpu-reference-gate0.md").read_text(encoding="utf-8")
    manifest, entries = gpu_ext.load_set(directory)
    assert f"`{hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest()}`" in text
    assert f"`{hashlib.sha256((directory / 'plan.json').read_bytes()).hexdigest()}`" in text
    assert manifest["code_sha256"] == gpu_ext.code_sha256()
    for clip in manifest["clips"]:
        assert f"| `{clip['id']}` | {clip['block']} |" in text and clip["wav_sha256"] in text
    assert len(entries) == 717


def test_a_request_may_run_only_two_minutes_past_the_soft_end(tmp_path: Path, key_file, fake):
    run = _soft_run(tmp_path, fake, key_file)
    assert vast.timeout_cap(run) == math.inf
    run.soft_end = time.time() + 600
    assert 715 <= vast.timeout_cap(run) <= 720
    run.soft_end = time.time() - 3600
    assert vast.timeout_cap(run) == vast.MIN_REQUEST_TIMEOUT_S


# --- the fixes of the verifier's round on #149 ---------------------------------------------


class _RecoverHost:
    """`state` and `true` fail while the network is gone (`down` calls), then answer;
    `ensure_tunnel` records whether it was forced and brings the tunnel back."""

    def __init__(self, down: int = 0, states: list | None = None) -> None:
        self.down, self.states, self.calls, self.reopened = down, states, [], []
        self.tunnel_up = False

    def run(self, argv, timeout):
        self.calls.append(argv[:3])
        if self.down > 0:
            self.down -= 1
            return vast.proc.RunResult(255, b"", b"", False, False, 0.0)
        if argv[:3] == ["python3", vast.HOSTRUN, "state"]:
            states = self.states or [_state()]
            answer = states.pop(0) if len(states) > 1 else states[0]
            return vast.proc.RunResult(0, answer, b"", False, False, 0.0)
        return vast.proc.RunResult(0, b"", b"", False, False, 0.0)

    def ensure_tunnel(self, force=False):
        self.reopened.append(force)
        self.tunnel_up = True
        return True


class _TunnelClient:
    """`/health` answers 200 only once the host's tunnel is back, after `late` more asks."""

    def __init__(self, host, late: int = 0, healthy: bool = False) -> None:
        self.host, self.late, self.healthy, self.asked = host, late, healthy, 0

    def request(self, route, timeout=5):
        self.asked += 1
        if self.healthy:
            return 200, b"", None
        if self.host.tunnel_up:
            if self.late <= 0:
                return 200, b"", None
            self.late -= 1
        return None, None, "ConnectionRefusedError"


def _recover_run(tmp_path, fake, key_file, host) -> vast.Run:
    run = _soft_run(tmp_path, fake, key_file)
    run.host = host
    return run


@pytest.mark.slow
def test_recover_waits_out_a_network_blip_then_reopens_the_tunnel_and_resends(
    tmp_path: Path, key_file, fake
) -> None:
    host = _RecoverHost(down=2)
    run = _recover_run(tmp_path, fake, key_file, host)
    client = _TunnelClient(host, late=2)
    assert vast.recover(run, client, "/workspace/out/m", "m") == "resend"
    # forced once, then a plain reopen after each failed /health while the server comes back
    assert host.reopened == [True, False, False] and run.fatal is None
    (check,) = [e for e in run.journal.replay() if e["event"] == "transport_check"]
    assert (check["tunnel_reopened"], check["health"], check["outcome"]) == (True, 200, "resend")


class _LostAgainHost(_RecoverHost):
    """The forced reopen comes while the network is gone again: the tunnel stays dead until a
    later plain reopen, once the network is back."""

    def ensure_tunnel(self, force=False):
        self.reopened.append(force)
        self.tunnel_up = not force
        return self.tunnel_up


@pytest.mark.slow
def test_recover_reopens_a_tunnel_a_second_loss_killed_while_it_waits_for_health(
    tmp_path: Path, key_file, fake
) -> None:
    host = _LostAgainHost()
    run = _recover_run(tmp_path, fake, key_file, host)
    client = _TunnelClient(host)
    assert vast.recover(run, client, "/workspace/out/m", "m") == "resend"
    assert host.reopened == [True, False]


def test_recover_after_a_timeout_through_a_tunnel_that_answers_resends_nothing(
    tmp_path: Path, key_file, fake
) -> None:
    host = _RecoverHost()
    run = _recover_run(tmp_path, fake, key_file, host)
    client = _TunnelClient(host, healthy=True)
    assert vast.recover(run, client, "/workspace/out/m", "m", after_timeout=True) == "timeout"
    assert host.reopened == [] and host.calls == []


def test_recover_after_a_timeout_restarts_a_stale_tunnel_and_resends(
    tmp_path: Path, key_file, fake
) -> None:
    host = _RecoverHost()
    run = _recover_run(tmp_path, fake, key_file, host)
    client = _TunnelClient(host)
    assert vast.recover(run, client, "/workspace/out/m", "m", after_timeout=True) == "resend"
    assert host.reopened == [True]
    (check,) = [e for e in run.journal.replay() if e["event"] == "transport_check"]
    assert check["after_timeout"] is True and check["tunnel_reopened"] is True


@pytest.mark.slow
def test_recover_ends_the_wait_when_the_server_exits_and_a_dead_host_is_ssh_lost(
    tmp_path: Path, key_file, fake
) -> None:
    exited = _state(exit={"reason": "exit", "returncode": 1, "t": 1.0})
    host = _RecoverHost(states=[_state(), _state(), exited])
    run = _recover_run(tmp_path, fake, key_file, host)
    started = time.monotonic()
    outcome = vast.recover(run, _NeverHealthy(), "/workspace/out/m", "m")
    assert outcome == "server_exited" and time.monotonic() - started < vast.RECOVER_HEALTH_S
    (tmp_path / "lost").mkdir()
    lost = _recover_run(tmp_path / "lost", fake, key_file, _RecoverHost(down=10**9))
    assert vast.recover(lost, _NeverHealthy(), "/workspace/out/m", "m") == "ssh_lost"
    assert lost.fatal == "ssh_lost"


_SLOW_SSH = """
import socket, sys, time
port = int(sys.argv[sys.argv.index("-L") + 1].split(":")[1])
time.sleep(0.7)
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen(4)
time.sleep(60)
"""


@pytest.mark.slow
def test_the_ssh_tunnel_returns_once_its_port_accepts_and_force_restarts_a_live_one(
    tmp_path: Path, monkeypatch
) -> None:
    host = vast.RemoteHost(tmp_path, "203.0.113.7", 22)
    monkeypatch.setattr(host, "ssh_argv", lambda *extra: [sys.executable, "-c", _SLOW_SSH, *extra])
    started = time.monotonic()
    port = host.tunnel(8000)
    assert time.monotonic() - started >= 0.6
    socket.create_connection(("127.0.0.1", port), 1).close()
    first = host._tunnels[-1][0]
    assert host.ensure_tunnel() is False
    assert host.ensure_tunnel(force=True) is True
    assert first.poll() is not None and host._forward == (8000, port)
    socket.create_connection(("127.0.0.1", port), 1).close()
    host.close()


def test_a_stalled_fake_tunnel_accepts_but_carries_nothing_and_its_stop_frees_the_port() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    forwarder = vast_fake.Forwarder(server.getsockname()[1])
    forwarder.stall()
    client = socket.create_connection(("127.0.0.1", forwarder.port), 2)
    client.settimeout(0.5)
    client.sendall(b"GET / HTTP/1.0\r\n\r\n")
    with pytest.raises(TimeoutError):
        client.recv(1)
    forwarder.stop()
    again = vast_fake.Forwarder(server.getsockname()[1], forwarder.port)
    assert again.port == forwarder.port and again.alive()
    again.stop()
    client.close()
    server.close()


def test_sigterm_and_sighup_cancel_the_attempt_once_and_later_signals_are_ignored(
    tmp_path: Path, key_file, fake
) -> None:
    run = _soft_run(tmp_path, fake, key_file)
    saved = vast.cancel_on_signals(run)
    try:
        with pytest.raises(SystemExit) as raised:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        assert raised.value.code == 128 + signal.SIGTERM
        assert run.fatal == "cancelled" and vast.soft_stop(run) == "cancelled"
        os.kill(os.getpid(), signal.SIGHUP)  # ignored now: the download and destroy go on
        time.sleep(0.05)
        assert [e["name"] for e in run.journal.replay() if e["event"] == "signal"] == ["SIGTERM"]
    finally:
        vast.restore_signals(saved)
    assert signal.getsignal(signal.SIGTERM) == saved[signal.SIGTERM]
    (tmp_path / "hup").mkdir()
    run2 = _soft_run(tmp_path / "hup", fake, key_file)
    saved = vast.cancel_on_signals(run2)
    try:
        with pytest.raises(SystemExit) as raised:
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(1)
        assert raised.value.code == 128 + signal.SIGHUP
    finally:
        vast.restore_signals(saved)


@pytest.mark.slow
def test_create_is_refused_when_the_credit_is_below_the_ceiling_plus_the_margin(
    tmp_path: Path,
) -> None:
    out = tmp_path / "data" / "dry"
    need = vast.DRY_PACKAGE["ceiling_usd"] + vast.CREDIT_MARGIN_USD
    scenario = {"fake": lambda fake: setattr(fake, "credit", need - 0.01)}
    with pytest.raises(vast.VastError, match="credit 4.99 USD is below the ceiling plus 1.00"):
        vast.dry_run(out, _dry_config(tmp_path), scenario=scenario)
    log = [json.loads(line) for line in (out / "fake-api.jsonl").read_text().splitlines()]
    assert not any(e["method"] in ("PUT", "DELETE") for e in log)
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    events = vast.Journal(journal_path).replay()
    (refused,) = [e for e in events if e["event"] == "credit_refused"]
    assert (refused["credit"], refused["need_usd"]) == (4.99, need)
    assert not any(e["event"] == "create_requested" for e in events)
    (closed,) = [e for e in events if e["event"] == "closed"]
    assert closed["reason"] == "no_instance_created"


@pytest.mark.slow
def test_an_extended_set_built_by_another_gpu_ext_is_refused_before_anything_opens(
    tmp_path: Path,
) -> None:
    out = tmp_path / "data" / "dry"
    with pytest.raises(vast.VastError, match="code_sha256"):
        vast.dry_run(out, _dry_config(tmp_path), scenario={"manifest_code_sha256": "0" * 64})
    assert not list((out / "gpu").glob("*/journal.jsonl"))
    assert (out / "fake-api.jsonl").read_text() == ""


def _spend_package(**changes) -> dict:
    package = json.loads((REPO / "docs/examples/gpu-package.example.json").read_text())
    package["extended"] = None
    return {k: v for k, v in {**package, **changes}.items() if v is not None}


def test_the_soft_spend_leaves_the_worst_tail_and_its_reserve_before_the_early_stop(
    tmp_path: Path,
) -> None:
    # 2 min request, 65 s check after it, 2 min sync, 1 min evidence, 2 min stop, 5 min download
    assert vast.spend_tail_s() == 785 and vast.SPEND_TAIL_RESERVE_S == 3 * 60
    loaded = _load(tmp_path, _spend_package(spend_soft_usd=1.9))
    assert loaded["spend_soft_usd"] == 1.9
    _load(tmp_path, _spend_package(spend_soft_usd=1.92))
    with pytest.raises(vast.VastError, match="leaves 0.300 USD before the early stop"):
        _load(tmp_path, _spend_package(spend_soft_usd=1.95))
    with pytest.raises(vast.VastError, match="leaves 0.150 USD before the early stop"):
        _load(tmp_path, _spend_package(spend_soft_usd=2.1))
    envelope = {**_spend_package()["envelope"], "dph_total": {"gte": 0.5}}
    with pytest.raises(vast.VastError, match="needs an envelope dph_total lte bound"):
        _load(tmp_path, _spend_package(spend_soft_usd=1.9, envelope=envelope))
    _load(tmp_path, _spend_package(spend_soft_usd=None, envelope=envelope))


def test_the_approved_package_bounds_every_wait_of_its_models_and_the_soft_spend() -> None:
    text = (REPO / "docs/decisions/0028-gpu-reference-gate0.md").read_text(encoding="utf-8")
    addendum = text.split("\n## Дополнение 2026-11-23: пакет $3 и расширенный набор\n", 1)[1]
    package = json.loads(addendum.split("```json\n", 1)[1].split("\n```", 1)[0])
    m1, m2 = package["models"]
    assert (m1["weights_timeout_s"], m1["health_timeout_s"], m1["slot_minutes"]) == (600, 300, 25)
    assert (m2["weights_timeout_s"], m2["health_timeout_s"]) == (1500, 600)
    assert package["spend_soft_usd"] == 1.9
    margin_s = (package["early_stop_usd"] - package["spend_soft_usd"]) / 1.2 * 3600
    assert margin_s >= vast.spend_tail_s() + vast.SPEND_TAIL_RESERVE_S
    assert "### Д.9. Правка 2026-11-23 (до аренды" in addendum
    assert "launch-argv --package data/gate0/gpu/packages/gpu-ref-149.json" in addendum


def test_a_truncated_day_is_its_model_outcome_incomplete_never_ok() -> None:
    assert vast._verdict_outcome({"verdict": "NOT READY"}) == {
        "outcome": "ok",
        "step": None,
        "reason": None,
    }
    assert vast._verdict_outcome({"verdict": "incomplete", "reason": "день не завершён"}) == {
        "outcome": "incomplete",
        "step": "run",
        "reason": "день не завершён",
    }


def test_the_watchdog_destroy_is_named_and_the_agents_own_is_not(
    tmp_path: Path, key_file, fake
) -> None:
    journal, _ = _journal(tmp_path, fake, key_file)
    journal.append("destroy_requested", instance_id=7, reason="agent")
    assert vast.watchdog_destroy(journal) is None
    journal.append("destroy_requested", instance_id=7, reason="early_stop")
    assert vast.watchdog_destroy(journal) == "early_stop"


def test_the_launch_argv_starts_deploy_from_the_checkout_with_its_config_and_python(
    tmp_path: Path,
) -> None:
    root, logs = tmp_path / "checkout", tmp_path / "logs"
    package, config_path = root / "data/p.json", root / "local/gate0.toml"
    argv = vast.launch_argv(
        package, "https://github.com/x/y/issues/1#c", config_path, logs, "T", root
    )
    assert argv[:4] == ["systemd-run", "--user", f"--unit={vast.LABEL_PREFIX}deploy-T", "--collect"]
    assert f"--working-directory={root}" in argv
    assert f"--setenv=SOUND_SCOUT_CONFIG={config_path}" in argv
    assert "--property=TimeoutStopSec=25min" in argv
    assert f"--property=StandardOutput=append:{logs / 'deploy.out'}" in argv
    assert f"--property=StandardError=append:{logs / 'deploy.err'}" in argv
    inhibit = argv.index("--") + 1
    assert argv[inhibit].endswith("systemd-inhibit") and "--mode=block" in argv[inhibit:]
    python = argv.index(str(root / ".venv" / "bin" / "python"))
    assert argv[python + 1 :] == [
        "-m", "sound_scout.eval.vast", "deploy", "--package", str(package),
        "--approval", "https://github.com/x/y/issues/1#c",
    ]  # fmt: skip


def test_launch_argv_refuses_without_an_approval_or_a_config_and_starts_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    package = tmp_path / "p.json"
    package.write_text("{}")
    monkeypatch.delenv("SOUND_SCOUT_CONFIG", raising=False)
    approval = "https://github.com/renidere/sound-scout-plugin/issues/149#issuecomment-1"
    argv = ["launch-argv", "--package", str(package), "--approval", approval]
    assert vast.main(argv) == 2
    assert "needs SOUND_SCOUT_CONFIG" in capsys.readouterr().err
    assert vast.main([*argv[:3], "--approval", "yes"]) == 2
    assert "--approval" in capsys.readouterr().err


def test_launch_argv_runs_the_deploy_checks_and_prints_the_argv(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dry = _dry_config(tmp_path)
    entries = vast._dry_days(dry, False)
    package = {**vast.DRY_PACKAGE, "models": entries}
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package))
    toml = tmp_path / "gate0.toml"
    toml.write_text(
        f'allowed_roots = ["{dry.allowed_roots[0]}"]\ndata_root = "{dry.data_root}"\n'
        'backend = "none"\n\n[network]\nallow = false\n'
    )
    monkeypatch.setenv("SOUND_SCOUT_CONFIG", str(toml))
    monkeypatch.setattr(vast, "MODELS", vast_fake.synthetic_models())
    approval = "https://github.com/renidere/sound-scout-plugin/issues/149#issuecomment-1"
    # a rental needs the weights limit of every model; the dry run's package has none
    assert vast.main(["launch-argv", "--package", str(path), "--approval", approval]) == 2
    assert "needs weights_timeout_s for a rental" in capsys.readouterr().err
    assert vast.load_package(path, vast_fake.synthetic_models())[0]["models"] == entries
    for entry in entries:
        entry["weights_timeout_s"] = 600
    path.write_text(json.dumps(package))
    assert vast.main(["launch-argv", "--package", str(path), "--approval", approval]) == 0
    printed = json.loads(capsys.readouterr().out)
    logs = Path(printed["logs"])
    assert logs.is_dir() and logs.parent == dry.data_root / "gpu" / "launches"
    assert oct(logs.stat().st_mode & 0o777) == "0o700" and not list(logs.iterdir())
    assert f"--setenv=SOUND_SCOUT_CONFIG={toml}" in printed["argv"]
    assert printed["package_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (dry.data_root / "gpu" / "launches" / "x").exists()


def test_a_request_is_cut_two_minutes_past_the_moment_the_clock_reaches_the_soft_spend(
    tmp_path: Path, key_file, fake
) -> None:
    run = _soft_run(tmp_path, fake, key_file, spend_soft_usd=1.9, in_gb=100.0, out_gb=0.0)
    assert vast.spend_end_unix(run) is None and vast.timeout_cap(run) == math.inf
    prices = {"dph_total": 1.2, "inet_down_cost": 0.004, "inet_up_cost": 0.004}
    created = run.journal.append("create_requested", offer_id=1, prices=prices)
    run.journal.append("create_response", instance_id=7)
    spend_end = created["t"] + (1.9 - 0.4) / 1.2 * 3600
    assert vast.spend_end_unix(run) == pytest.approx(spend_end)
    cap = spend_end + vast.REQUEST_PAST_SOFT_END_S - time.time()
    assert vast.timeout_cap(run) == pytest.approx(cap, abs=1)
    run.soft_end = time.time() + 60
    assert vast.timeout_cap(run) == pytest.approx(180, abs=1)


def test_recover_past_the_soft_stop_resends_nothing_but_keeps_the_rules_own_limits(
    tmp_path: Path, key_file, fake
) -> None:
    host = _RecoverHost(down=10**9)
    run = _recover_run(tmp_path, fake, key_file, host)
    run.soft_end = time.time() - 1
    assert vast.recover(run, _NeverHealthy(), "/workspace/out/m", "m") == "soft_end"
    assert host.calls == [["python3", vast.HOSTRUN, "state"]] and host.reopened == []
    assert run.fatal is None
    healthy = _TunnelClient(host, healthy=True)
    assert vast.recover(run, healthy, "/workspace/out/m", "m", after_timeout=True) == "timeout"
    killed = _state(exit={"reason": "rss_limit", "returncode": -9, "t": 1.0})
    (tmp_path / "late").mkdir()
    late = _recover_run(tmp_path / "late", fake, key_file, _RecoverHost(states=[killed]))
    late.soft_end = time.time() - 1
    assert vast.recover(late, _NeverHealthy(), "/workspace/out/m", "m") == "rss_limit"


def test_an_rss_kill_by_hostrun_is_the_rules_limit_not_a_server_exit(
    tmp_path: Path, key_file, fake
) -> None:
    killed = _state(exit={"reason": "rss_limit", "returncode": -9, "t": 1.0})
    run = _recover_run(tmp_path, fake, key_file, _RecoverHost(states=[killed]))
    assert vast.recover(run, _NeverHealthy(), "/workspace/out/m", "m") == "rss_limit"
    assert "rss_limit" in gate0.RULE_STOPS


def test_a_signal_inside_the_final_path_is_journaled_and_cuts_nothing(
    tmp_path: Path, key_file, fake
) -> None:
    run = _soft_run(tmp_path, fake, key_file)
    saved = vast.cancel_on_signals(run)
    try:
        vast.defer_signals(run, saved)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)
        signals = [e for e in run.journal.replay() if e["event"] == "signal"]
        assert [(e["name"], e.get("deferred")) for e in signals] == [("SIGTERM", True)]
        assert run.fatal == "cancelled"
    finally:
        vast.restore_signals(saved)
    assert vast.defer_signals(run, {}) is None


# A second process: `watch_tick` on the attempt's journal; `busy` only after 5 s of retries.
_WATCH_PROBE = """
import sys, time
from sound_scout.eval import vast

def no_api(_opened):
    raise vast.VastError("the probe has no API")

journal = vast.Journal.open(sys.argv[1])
for _ in range(50):
    outcome = vast.watch_tick(journal, no_api, runner=None)
    if outcome != "busy":
        break
    time.sleep(0.1)
print(outcome)
"""

# The child runs a dry run and, once armed, raises SIGTERM in the main thread inside a window of
# `Journal.lock()`: right after `flock` returns (`acquire`), right before the lock fd's `close`
# (`release`), or after the lock's body ends: at the start of `contextlib`'s `__exit__` (`exit`)
# or at the first line of the lock after its `yield` (`finally`, #173). `cancel` arms after the
# first Gate 0 block (the handler of `cancel_on_signals`), `defer` once the final download starts
# (the handler of `defer_signals`). Before the destroy, `_WATCH_PROBE` runs.
_LOCK_WINDOW_CHILD = """
import contextlib, fcntl, inspect, os, signal, subprocess, sys, threading
from pathlib import Path

from sound_scout import config
from sound_scout.eval import vast

out, window, when, probe_code = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
armed = []
real_flock, real_close, real_download = fcntl.flock, os.close, vast.download
real_destroy = vast.destroy_until_absent
lock_code = vast.Journal.lock.__wrapped__.__code__
exit_code = contextlib._GeneratorContextManager.__exit__.__code__
source, first = inspect.getsourcelines(vast.Journal.lock)
yield_line = first + next(i for i, line in enumerate(source) if "yield held" in line)


def is_lock(fd):
    try:
        return os.readlink(f"/proc/self/fd/{fd}").endswith("/" + vast.LOCK_FILE)
    except OSError:
        return False


def fire(frame=None):
    if armed and threading.current_thread() is threading.main_thread():
        armed.clear()
        sys.settrace(None)
        try:
            if frame is None:
                signal.raise_signal(signal.SIGTERM)  # the handler runs inside this call
            else:  # as the eval loop runs the handler at this point of `frame`
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, frame)
        finally:
            (out.parent / "fired").write_text(window)


def tracer(frame, event, _arg):
    if event != "call":
        return None
    if window == "exit" and frame.f_code is exit_code:
        lock = frame.f_locals["self"].gen
        if lock.gi_code is lock_code and lock.gi_frame.f_locals.get("fd") is not None:
            fire(frame)
        return None
    return after_yield if window == "finally" and frame.f_code is lock_code else None


def after_yield(frame, event, _arg):
    if event == "line" and frame.f_lineno > yield_line:
        fire(frame)
    return after_yield


def arm():
    armed.append(1)
    if window in ("exit", "finally"):
        sys.settrace(tracer)


def flock(fd, operation):
    real_flock(fd, operation)
    if window == "acquire" and operation & fcntl.LOCK_EX and is_lock(fd):
        fire()


def close(fd):
    if window == "release" and is_lock(fd):
        fire()
    real_close(fd)


def download(*args, **kwargs):
    final = kwargs.get("name", "results") == "results"
    if when == "defer" and final and not (out.parent / "fired").exists():
        arm()
    return real_download(*args, **kwargs)


def destroy_until_absent(journal, *args, **kwargs):
    probe = subprocess.run(
        [sys.executable, "-c", probe_code, str(journal.path)], capture_output=True, text=True
    )
    (out.parent / "probe").write_text(probe.stdout.strip() or probe.stderr)
    return real_destroy(journal, *args, **kwargs)


def first_block(run, **_):
    if not (out.parent / "fired").exists():
        arm()


fcntl.flock, os.close, vast.download = flock, close, download
vast.destroy_until_absent = destroy_until_absent
hooks = {"block_end": first_block} if when == "cancel" else {}
(out.parent / "in").mkdir()
config_ = config.Config(allowed_roots=((out.parent / "in").resolve(),), data_root=out.parent)
vast.dry_run(out, config_, scenario={"hooks": hooks})
"""


def _signal_in_lock_window(tmp_path: Path, window: str, when: str) -> tuple[int, list, str]:
    out = tmp_path / "dry"
    child = subprocess.Popen(
        [sys.executable, "-c", _LOCK_WINDOW_CHILD, str(out), window, when, _WATCH_PROBE],
        cwd=REPO,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        code = child.wait(240)
    except subprocess.TimeoutExpired:  # the handler blocked on the lock this thread holds
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        code = None
    assert vast.stop_local_sessions(out / "host-root"), "a host session outlived the dry run"
    assert code is not None, "the signal handler blocked on the journal lock"
    assert (tmp_path / "fired").read_text() == window
    (journal_path,) = (out / "gpu").glob("*/journal.jsonl")
    return code, vast.Journal(journal_path).replay(), (tmp_path / "probe").read_text()


def _download_and_destroy(events: list, tmp_path: Path) -> None:
    downloads = [e for e in events if e["event"] == "step" and e["name"] == "download"]
    assert [d["ok"] for d in downloads] == [True]
    assert any(e["event"] == "download_verified" for e in events)
    assert any(e["event"] == "absence_verified" for e in events)
    assert any(e["event"] == "closed" for e in events)
    timeline = json.loads((tmp_path / "dry" / "timeline.json").read_text())
    assert timeline["fake_api"]["instances_left"] == []


LOCK_WINDOWS = ["acquire", "release", "exit", "finally"]


@pytest.mark.slow
@pytest.mark.parametrize("window", LOCK_WINDOWS)
def test_a_cancel_signal_inside_the_journal_lock_still_reaches_download_and_destroy(
    tmp_path: Path, window: str
) -> None:
    code, events, probe = _signal_in_lock_window(tmp_path, window, "cancel")
    assert code == 128 + signal.SIGTERM
    signals = [e for e in events if e["event"] == "signal"]
    assert [(e["name"], e.get("deferred")) for e in signals] == [("SIGTERM", None)]
    first_download = next(
        i for i, e in enumerate(events) if e["event"] == "step" and e["name"] == "download"
    )
    assert events.index(signals[0]) < first_download
    models = {e["model"] for e in events if e["event"] == "step" and "model" in e}
    assert len(models) == 1  # the cancel ends the attempt: the second model never starts
    _download_and_destroy(events, tmp_path)
    assert probe == "error"  # the second process got the lock; only its API is missing


@pytest.mark.slow
@pytest.mark.parametrize("window", LOCK_WINDOWS)
def test_a_signal_inside_the_journal_lock_of_the_final_path_is_journaled_and_cuts_nothing(
    tmp_path: Path, window: str
) -> None:
    _, events, probe = _signal_in_lock_window(tmp_path, window, "defer")
    signals = [e for e in events if e["event"] == "signal"]
    assert [(e["name"], e.get("deferred")) for e in signals] == [("SIGTERM", True)]
    _download_and_destroy(events, tmp_path)
    assert probe == "error"


def test_a_signal_inside_the_lock_window_runs_its_handler_once_the_window_ends(
    tmp_path: Path, monkeypatch
) -> None:
    journal = vast.Journal.create(tmp_path, "a1", None)
    seen = []

    def handler(signum, _frame):
        seen.append((signum, vast._LOCK_WINDOW.pending, journal._held is not None))
        journal.append("signal", name=signal.Signals(signum).name)

    safe = vast._outside_lock_window(handler)
    real_flock = vast.fcntl.flock

    def flock(fd, operation):
        real_flock(fd, operation)
        if operation & vast.fcntl.LOCK_EX and not seen:
            safe(signal.SIGHUP, None)  # as the signal machinery calls it inside the window
            assert seen == []

    monkeypatch.setattr(vast.fcntl, "flock", flock)
    journal.append("one")
    assert seen == [(signal.SIGHUP, None, True)]
    assert [e["event"] for e in journal.replay()] == ["signal", "one"]
    assert vast._LOCK_WINDOW.pending is None
    safe(signal.SIGTERM, None)  # outside the window: at once
    assert seen[-1] == (signal.SIGTERM, None, False)


def test_a_signal_before_a_failed_lock_open_still_runs_its_handler(tmp_path: Path) -> None:
    journal = vast.Journal.create(tmp_path, "a1", None)
    seen = []
    safe = vast._outside_lock_window(
        lambda signum, _frame: seen.append(
            (signum, vast._LOCK_WINDOW.pending, journal._held is not None)
        )
    )

    class Missing:
        def __truediv__(self, name):
            safe(signal.SIGTERM, None)
            assert seen == []
            return tmp_path / "missing" / name

    journal.dir = Missing()
    with pytest.raises(FileNotFoundError), journal.lock():
        pass
    assert seen == [(signal.SIGTERM, None, False)]
    assert vast._LOCK_WINDOW.pending is None


@pytest.mark.parametrize("body_raises", [False, True])
def test_a_raising_signal_at_any_instruction_after_the_lock_body_still_closes_the_lock(
    tmp_path: Path, body_raises: bool
) -> None:
    """The handler runs as the signal machinery runs it, at each instruction in turn from the
    start of `contextlib`'s `__exit__` to the end of the lock's `finally` (#173)."""
    journal = vast.Journal.create(tmp_path, "a1", None)
    codes = (
        contextlib._GeneratorContextManager.__exit__.__code__,
        vast.Journal.lock.__wrapped__.__code__,
    )
    calls, state = [], {}

    def cancel(signum, _frame):
        calls.append(signum)
        raise SystemExit(128 + signum)

    safe = vast._outside_lock_window(cancel)

    def instruction(_code, _offset):
        if state["ended"]:
            state["seen"] += 1
            if state["seen"] == state["point"] + 1:
                safe(signal.SIGTERM, sys._getframe(1))  # its exception lands on this instruction

    monitoring, tool = sys.monitoring, sys.monitoring.DEBUGGER_ID
    monitoring.use_tool_id(tool, "lock-signal-test")
    monitoring.register_callback(tool, monitoring.events.INSTRUCTION, instruction)
    for code in codes:
        monitoring.set_local_events(tool, code, monitoring.events.INSTRUCTION)
    fds = sorted(os.listdir("/proc/self/fd"))

    def released(point: int) -> None:
        assert journal._held is None, point
        assert vast._LOCK_WINDOW.pending is None, point
        assert sorted(os.listdir("/proc/self/fd")) == fds, point
        fd = os.open(journal.dir / vast.LOCK_FILE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another open file: free, or raises
        finally:
            os.close(fd)

    try:
        for point in itertools.count():
            calls.clear()
            state.update(point=point, ended=False, seen=0)
            exited = False
            try:
                with journal.lock():
                    state["ended"] = True
                    if body_raises:
                        raise RuntimeError("the body")
            except SystemExit:
                exited = True
                released(point)  # while the exception lives, as in the attempt's `finally`
            except RuntimeError:
                assert body_raises
            if not calls:
                break  # past the last instruction
            assert (calls, exited) == ([signal.SIGTERM], True), point
            released(point)
    finally:
        for code in codes:
            monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool)
    assert point == state["seen"] > 30  # every instruction of `__exit__` and the lock was a point


def test_a_second_signal_inside_the_first_handler_at_the_lock_exit_still_closes_the_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """SIGTERM lands at the start of `contextlib`'s `__exit__`; SIGHUP lands in its handler
    before that handler opens the deferral. Both wait for the lock's close (#173)."""
    journal = vast.Journal.create(tmp_path, "a1", None)
    exit_code = contextlib._GeneratorContextManager.__exit__.__code__
    calls, fired = [], []

    def cancel(signum, _frame):
        calls.append(signum)
        raise SystemExit(128 + signum)

    safe = vast._outside_lock_window(cancel)
    real_exiting_lock = vast._exiting_lock

    def exiting_lock(frame):
        if fired == ["SIGTERM"]:
            fired.append("SIGHUP")
            safe(signal.SIGHUP, sys._getframe())  # it interrupts the first handler here
            assert calls == []
        return real_exiting_lock(frame)

    monkeypatch.setattr(vast, "_exiting_lock", exiting_lock)

    def instruction(_code, _offset):
        if not fired:
            fired.append("SIGTERM")
            safe(signal.SIGTERM, sys._getframe(1))

    monitoring, tool = sys.monitoring, sys.monitoring.DEBUGGER_ID
    monitoring.use_tool_id(tool, "lock-signal-test")
    monitoring.register_callback(tool, monitoring.events.INSTRUCTION, instruction)
    fds = sorted(os.listdir("/proc/self/fd"))
    exited = False
    try:
        with journal.lock():
            monitoring.set_local_events(tool, exit_code, monitoring.events.INSTRUCTION)
    except SystemExit:
        exited = True
        # while the exception lives: its traceback keeps the lock's generator from `close`
        assert journal._held is None
        assert vast._LOCK_WINDOW.pending is None
        assert sorted(os.listdir("/proc/self/fd")) == fds
        fd = os.open(journal.dir / vast.LOCK_FILE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        monitoring.set_local_events(tool, exit_code, 0)
        monitoring.register_callback(tool, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool)
    assert exited
    assert fired == ["SIGTERM", "SIGHUP"]
    assert calls == [signal.SIGHUP]  # the first deferred handler ends the window's run


_LOCK_PROBE = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("free")
except BlockingIOError:
    print("busy")
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("handlers", ["cancel", "defer"])
def test_a_signal_in_the_lock_enter_after_its_yield_runs_at_the_body_start(
    tmp_path: Path, handlers: str, signum: int
) -> None:
    """A real signal lands at each point of `contextlib`'s `__enter__` after `next(self.gen)`
    returns: the end of that call, where the interpreter checks for signals, and each
    instruction after it. Its handler runs once, at the start of the body, before the body's
    work, and the body runs outside the window (#178)."""
    journal = vast.Journal.create(tmp_path, "a1", None)
    run = vast.Run(None, journal, {}, {}, None, {}, None)
    enter_code = contextlib._GeneratorContextManager.__enter__.__code__
    state = {}

    def point(*_args):
        frame = sys._getframe(1)
        gen = getattr(frame.f_locals.get("self"), "gen", None)
        if state["armed"] and getattr(gen, "gi_code", None) is vast._LOCK_CODE and gen.gi_suspended:
            state["seen"] += 1
            if state["seen"] == state["point"] + 1:
                state.update(armed=False, fired=True)
                signal.raise_signal(signum)

    def released(at: int) -> None:
        assert journal._held is None, at
        assert vast._LOCK_WINDOW.pending is None, at
        assert sorted(os.listdir("/proc/self/fd")) == fds, at
        probe = subprocess.run(
            [sys.executable, "-c", _LOCK_PROBE, str(journal.dir / vast.LOCK_FILE)],
            capture_output=True,
            text=True,
        )
        assert probe.stdout.strip() == "free", (at, probe.stdout, probe.stderr)

    monitoring, tool = sys.monitoring, sys.monitoring.DEBUGGER_ID
    events = monitoring.events
    monitoring.use_tool_id(tool, "lock-signal-test")
    monitoring.register_callback(tool, events.C_RETURN, point)
    monitoring.register_callback(tool, events.INSTRUCTION, point)
    saved = vast.cancel_on_signals(run)
    fds = sorted(os.listdir("/proc/self/fd"))
    try:
        for at in itertools.count():
            vast.cancel_on_signals(run)  # the cancel ignores both signals once it ran
            if handlers == "defer":
                vast.defer_signals(run, saved)
            state.update(point=at, seen=0, armed=True, fired=False, window="unset")
            run.fatal = None
            start = len(journal.replay())
            exited = False
            monitoring.set_local_events(tool, enter_code, events.CALL | events.INSTRUCTION)
            try:
                with journal.lock():
                    state["armed"] = False
                    journal.append("body")
                    state["window"] = vast._LOCK_WINDOW.pending
            except SystemExit as error:
                exited = True
                assert error.code == 128 + signum, at
                released(at)  # while the exception lives, as in the attempt's `finally`
            finally:
                monitoring.set_local_events(tool, enter_code, 0)
            if not state["fired"]:
                break  # past the last point
            expected = [("signal", None)]
            if handlers == "defer":
                expected = [("signal", True), ("body", None)]
                assert state["window"] is None, at
            assert [
                (e["event"], e.get("deferred")) for e in journal.replay()[start:]
            ] == expected, at
            assert exited == (handlers == "cancel"), at
            assert run.fatal == "cancelled", at
            released(at)
    finally:
        vast.restore_signals(saved)
        monitoring.register_callback(tool, events.C_RETURN, None)
        monitoring.register_callback(tool, events.INSTRUCTION, None)
        monitoring.free_tool_id(tool)
    assert at == state["seen"] >= 2  # the end of `next(self.gen)` and `RETURN_VALUE`


def test_a_signal_held_in_the_create_lock_enter_stops_the_create_before_its_put(
    tmp_path: Path, key_file, fake, monkeypatch
) -> None:
    """A cancel held in the `__enter__` of the create's lock runs at the start of its body: no
    `create_requested` and no PUT after the signal, and the lock is free (#178)."""
    journal, _ = _journal(tmp_path, fake, key_file, deadline_unix=time.time() + 9000)
    api = vast.Api(fake.url, key_file[1])
    run = vast.Run(None, journal, {}, {}, None, {}, None)
    real_offer, real_lock = api.offer, vast.Journal.lock
    state = {"armed": False}

    def offer(*args):
        state["armed"] = True  # the next outer lock is the create's
        return real_offer(*args)

    class SignalInEnter:
        def __init__(self, manager) -> None:
            self.manager = manager

        def __enter__(self):
            outer = journal._held is None
            held = self.manager.__enter__()
            if state["armed"] and outer and journal._held is not None:  # the outer lock's flock
                state["armed"] = False
                vast._LOCK_WINDOW.pending = []  # as `_exiting_lock` opens it in `__enter__`
                signal.raise_signal(signal.SIGTERM)
                assert run.fatal is None  # held, not run
            return held

        def __exit__(self, *exc):
            return self.manager.__exit__(*exc)

    monkeypatch.setattr(api, "offer", offer)
    monkeypatch.setattr(vast.Journal, "lock", lambda self, *a: SignalInEnter(real_lock(self, *a)))
    saved = vast.cancel_on_signals(run)
    try:
        with pytest.raises(SystemExit) as error:
            vast.create_instance(api, journal, _package(), _no_system)
    finally:
        vast.restore_signals(saved)
    assert error.value.code == 128 + signal.SIGTERM
    assert run.fatal == "cancelled"
    assert fake.count("PUT") == 0
    assert "create_requested" not in _events(journal) and _events(journal)[-1] == "signal"
    assert journal._held is None and vast._LOCK_WINDOW.pending is None
    probe = subprocess.run(
        [sys.executable, "-c", _LOCK_PROBE, str(journal.dir / vast.LOCK_FILE)],
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "free", probe.stderr


def _flock_taken(journal: vast.Journal) -> bool:
    """Whether a second open file of `<attempt>/lock` fails to take the `flock`."""
    fd = os.open(journal.dir / vast.LOCK_FILE, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _cut_after_nested_enter(journal: vast.Journal, gens: list) -> None:
    nested = journal.lock()
    nested.__enter__()
    gens.append(weakref.ref(nested.gen))
    raise RuntimeError("after the nested enter")


def _cut_in_nested_exit(journal: vast.Journal, gens: list) -> None:
    """The exception lands on the first instruction of `contextlib`'s `__exit__` of the nested
    lock, before it ends the generator, as a raising signal handler can."""
    nested = journal.lock()
    gens.append(weakref.ref(nested.gen))
    exit_code = contextlib._GeneratorContextManager.__exit__.__code__
    monitoring, tool = sys.monitoring, sys.monitoring.DEBUGGER_ID

    def instruction(_code, _offset):
        if sys._getframe(1).f_locals.get("self") is nested:
            monitoring.set_local_events(tool, exit_code, 0)
            raise RuntimeError("in the nested exit")

    monitoring.use_tool_id(tool, "nested-lock-test")
    monitoring.register_callback(tool, monitoring.events.INSTRUCTION, instruction)
    try:
        with nested:
            monitoring.set_local_events(tool, exit_code, monitoring.events.INSTRUCTION)
    finally:
        monitoring.set_local_events(tool, exit_code, 0)
        monitoring.register_callback(tool, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool)


@pytest.mark.parametrize("late_close", ["before the next lock", "inside the next lock"])
@pytest.mark.parametrize("cut", [_cut_after_nested_enter, _cut_in_nested_exit])
def test_a_nested_journal_lock_cut_by_an_exception_keeps_the_flock_of_later_locks(
    tmp_path: Path, cut, late_close: str
) -> None:
    """The traceback keeps the cut nested generator at its `yield` past the outer close; its
    late close changes no lock state, and every later lock holds the `flock` (#182)."""
    journal = vast.Journal.create(tmp_path, "a1", None)
    gens = []
    with pytest.raises(RuntimeError) as kept, journal.lock():
        cut(journal, gens)
    (gen,) = gens
    assert gen() is not None and gen().gi_suspended  # cut: not ended by the outer close
    assert journal._held is None and not _flock_taken(journal)

    def close_late() -> None:
        nonlocal kept
        kept = None
        gc.collect()
        assert gen() is None  # the late close ran

    if late_close == "before the next lock":
        close_late()
    with journal.lock() as held:
        assert held and journal._held is not None and _flock_taken(journal)
        if late_close == "inside the next lock":
            close_late()
            assert journal._held is not None and _flock_taken(journal)
        with journal.lock(blocking=False) as nested:  # a second flock would fail, not hang
            assert nested and _flock_taken(journal)
        journal.append("next")  # a nested lock inside: no second flock, no deadlock
        assert journal._held is not None and _flock_taken(journal)
    assert journal._held is None and not _flock_taken(journal)
    with journal.lock() as held:
        assert held and _flock_taken(journal)
    assert journal._held is None and not _flock_taken(journal)
    assert [e["event"] for e in journal.replay()] == ["next"]


# --- transport recovery (#149, round 2) ----------------------------------------------------------


def _transport_run(tmp_path, fake, key_file, host, **intervals) -> vast.Run:
    run = _recover_run(tmp_path, fake, key_file, host)
    run.intervals = {**vast.DRY_INTERVALS, **intervals}
    return run


def _flaky(failures: int):
    calls = []

    def action(left: float) -> str:
        calls.append(left)
        if len(calls) <= failures:
            raise vast.TransportError("host: size: no SSH answer (exit 255)")
        return "done"

    return action, calls


def test_a_step_waits_out_the_host_then_runs_again(tmp_path: Path, key_file, fake) -> None:
    host = _RecoverHost(down=3)
    run = _transport_run(tmp_path, fake, key_file, host)
    action, calls = _flaky(1)
    assert vast.transport_retry(run, "weights m", action, time.time() + 60, tunnel=True) == "done"
    assert len(calls) == 2 and run.fatal is None and host.reopened == [True]
    (retry,) = [e for e in run.journal.replay() if e["event"] == "transport_retry"]
    assert (retry["what"], retry["outcome"], retry["tries"]) == ("weights m", "retry", 4)
    assert retry["tunnel_reopened"] is True and "exit 255" in retry["error"]


def test_a_host_silent_for_the_whole_window_is_ssh_lost(tmp_path: Path, key_file, fake) -> None:
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost(down=10**9), ssh_lost_s=0.3)
    action, calls = _flaky(10**9)
    started = time.monotonic()
    with pytest.raises(vast.HostLost, match="does not answer over SSH"):
        vast.transport_retry(run, "stop m", action, time.time() + 60)
    assert 0.3 <= time.monotonic() - started < 5 and len(calls) == 1
    assert run.fatal == "ssh_lost"
    (retry,) = [e for e in run.journal.replay() if e["event"] == "transport_retry"]
    assert retry["outcome"] == "ssh_lost" and retry["waited_s"] >= 0.3


class _SlowProbeHost(_RecoverHost):
    """Every probe uses its whole timeout, as an ssh that connects and then stalls."""

    def __init__(self, journal_on_probe=None) -> None:
        super().__init__(down=10**9)
        self.timeouts, self.on_probe = [], journal_on_probe

    def run(self, argv, timeout):
        self.timeouts.append(timeout)
        if self.on_probe is not None:
            self.on_probe()
        time.sleep(min(timeout, 0.2))
        return vast.proc.RunResult(-9, b"", b"", True, False, timeout)


@pytest.mark.slow
def test_the_ssh_window_never_runs_past_the_steps_own_end(tmp_path: Path, key_file, fake) -> None:
    host = _SlowProbeHost()
    run = _transport_run(tmp_path, fake, key_file, host, ssh_lost_s=300)
    action, calls = _flaky(10**9)
    started = time.monotonic()
    with pytest.raises(vast.TransportError) as raised:
        vast.transport_retry(run, "evidence m", action, time.time() + 1.5)
    # a window the step's end cut short is no ssh_lost of the attempt
    assert not isinstance(raised.value, vast.HostLost) and run.fatal is None
    assert time.monotonic() - started < 3 and len(calls) == 1
    assert max(host.timeouts) <= 1.5 and min(host.timeouts) >= 1.0
    (retry,) = [e for e in run.journal.replay() if e["event"] == "transport_retry"]
    assert retry["outcome"] == "step_end"


def test_a_watchdog_destroy_during_the_wait_ends_it_and_names_the_watchdog(
    tmp_path: Path, key_file, fake
) -> None:
    run = _transport_run(tmp_path, fake, key_file, None, ssh_lost_s=300)
    host = _SlowProbeHost(
        lambda: run.journal.append("destroy_requested", instance_id=7, reason="early_stop")
    )
    run.host = host
    action, _ = _flaky(10**9)
    with pytest.raises(vast.VastError, match="watchdog destroyed the instance") as raised:
        vast.transport_retry(run, "stop m", action, time.time() + 60)
    assert not isinstance(raised.value, vast.TransportError) and len(host.timeouts) == 1
    assert run.fatal is None


@pytest.mark.parametrize("stop", ["spend_seen", "cancelled", "ssh_lost"])
def test_once_stopped_for_any_reason_a_tail_step_gets_only_its_own_limit(
    tmp_path: Path, key_file, fake, stop
) -> None:
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost(), ssh_lost_s=300)
    run.soft_end = time.time() + 3600
    if stop == "spend_seen":
        run.package["spend_soft_usd"] = 1.0
        run.spend_seen = True
    else:
        run.fatal = stop
    now = time.time()
    assert vast.step_until(run, 60, tail=True) == pytest.approx(now + 60, abs=1)
    assert vast.step_until(run, 60) <= now + 1


def test_a_step_failing_while_the_host_answers_is_its_own_failure(
    tmp_path: Path, key_file, fake
) -> None:
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost())
    action, calls = _flaky(10**9)
    with pytest.raises(vast.TransportError) as raised:
        vast.transport_retry(run, "receipt m", action, time.time() + 60)
    assert not isinstance(raised.value, vast.HostLost)
    assert len(calls) == vast.SSH_RETRIES and run.fatal is None


def test_a_watchdog_destroy_ends_the_wait_at_once(tmp_path: Path, key_file, fake) -> None:
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost(down=10**9))
    run.journal.append("destroy_requested", instance_id=7, reason="early_stop")
    action, calls = _flaky(10**9)
    with pytest.raises(vast.VastError, match=r"watchdog destroyed the instance \(early_stop\)"):
        vast.transport_retry(run, "download", action, time.time() + 60)
    assert len(calls) == 1 and run.host.calls == []


def test_a_step_before_the_stop_gets_the_ssh_window_and_a_tail_step_its_own_limit(
    tmp_path: Path, key_file, fake
) -> None:
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost(), ssh_lost_s=300)
    now = time.time()
    assert vast.step_until(run, 60) == pytest.approx(now + 300, abs=1)
    assert vast.step_until(run, 600) == pytest.approx(now + 600, abs=1)
    run.soft_end = now + 100
    assert vast.step_until(run, 60) == pytest.approx(now + 100, abs=1)
    assert vast.step_until(run, 120, tail=True) == pytest.approx(now + 120, abs=1)
    run.soft_end = now - 10
    assert vast.step_until(run, 60) <= now + 1
    assert vast.step_until(run, 300, tail=True) == pytest.approx(now + 300, abs=1)


def test_a_lost_local_host_answers_as_ssh_and_a_spawn_starts_once(tmp_path: Path) -> None:
    host = vast.LocalHost(tmp_path / "root", {})
    (tmp_path / "root").mkdir()
    (tmp_path / "local.txt").write_text("x")
    host.spawn(["sh", "-c", "echo started"], f"{vast.WORKSPACE}/spawn.log")
    host.spawn(["sh", "-c", "echo again"], f"{vast.WORKSPACE}/spawn.log")
    assert len(host._spawned) == 1
    host.lose()
    done = host.run(["true"], 10)
    assert done.returncode == vast.SSH_EXIT
    with pytest.raises(vast.TransportError):
        vast._checked(done, "true")
    with pytest.raises(vast.TransportError):
        host.put(tmp_path / "local.txt", f"{vast.WORKSPACE}/x", "k3y-secret-value")
    with pytest.raises(vast.TransportError):
        host.get(f"{vast.WORKSPACE}/spawn.log", tmp_path / "copy")
    with pytest.raises(vast.TransportError):
        host.spawn(["true"], f"{vast.WORKSPACE}/other.log")
    host.close()


def test_a_remote_spawn_starts_nothing_when_its_log_exists_and_ssh_failing_is_transport(
    tmp_path: Path, monkeypatch
) -> None:
    host = vast.RemoteHost(tmp_path, "127.0.0.1", 22)
    seen = []

    def run(argv, timeout):
        seen.append(argv)
        return vast.proc.RunResult(255, b"", b"", False, False, 0.0)

    monkeypatch.setattr(host, "run", run)
    with pytest.raises(vast.TransportError):
        host.spawn(["hf", "download", "x y"], "/workspace/out/m/download-wrapper.log")
    (line,) = [argv[2] for argv in seen]
    assert line == (
        "test -e /workspace/out/m/download-wrapper.log || { nohup hf download 'x y' > "
        "/workspace/out/m/download-wrapper.log 2>&1 < /dev/null & }"
    )
    for code, timed_out, transport in ((255, False, True), (-15, False, True), (0, True, True),
                                       (1, False, False), (0, False, False)):  # fmt: skip
        done = vast.proc.RunResult(code, b"", b"", timed_out, False, 0.0)
        assert vast._transport_failed(done) is transport


def test_term_stop_covers_the_final_path_up_to_the_delete() -> None:
    minutes = int(vast.LAUNCH_TIMEOUT_STOP.removesuffix("min"))
    # evidence 1 min, stop 2 min, download 5 min, listing and DELETE 1 min each
    assert vast.stop_path_s() == 600 and minutes * 60 >= vast.stop_path_s()


@pytest.mark.slow
def test_the_ssh_window_counts_the_failed_try_itself(tmp_path: Path, key_file, fake) -> None:
    """A real ssh needs its ConnectTimeout to fail: that time is silence of the host too, so a
    host silent from the try on for the whole window is ssh_lost, though the wait was shorter."""
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost(down=10**9), ssh_lost_s=3.0)

    def slow_failure(left: float) -> None:
        time.sleep(1.5)
        raise vast.TransportError("host: state: no SSH answer (exit 255)")

    until = vast.step_until(run, 1.0)
    assert until == pytest.approx(time.time() + 3.0, abs=0.5)
    with pytest.raises(vast.HostLost):
        vast.transport_retry(run, "state m", slow_failure, until)
    assert run.fatal == "ssh_lost"


def test_a_pilot_whose_recovery_stops_fails_the_smoke_with_that_reason() -> None:
    dead = gate0.Client(vast._free_port(), "m")  # nothing listens: no HTTP answer
    with pytest.raises(vast.ModelFailed, match="pilot: ssh_lost") as raised:
        vast.run_pilot(dead, lambda after_timeout=False: "ssh_lost", timeout=lambda: 1.0)
    assert raised.value.step == "smoke"


def test_a_pilot_reads_its_timeout_for_every_request_and_a_resend() -> None:
    asked, timeouts = [], [3.0, 2.0, 1.0, 0.5]

    class Client:
        model = "m"

        def text_tokens(self, text):
            return 5

        def request(self, path, body=None, timeout=5):
            asked.append(timeout)
            if len(asked) == 1:
                return None, None, "ConnectionResetError: reset"
            return 200, json.dumps({"usage": {"prompt_tokens": 5 + 40 * len(asked)}}).encode(), None

    pilot = vast.run_pilot(Client(), lambda after_timeout=False: "resend", lambda: timeouts.pop(0))
    assert asked == [3.0, 2.0, 1.0] and pilot["ok"]
    assert [line["resent_after_tunnel"] for line in pilot["lines"]] == [True, False]


@pytest.mark.slow
def test_a_long_try_that_lost_its_ssh_late_still_gets_the_whole_window(
    tmp_path: Path, key_file, fake
) -> None:
    host = _SlowProbeHost()
    run = _transport_run(tmp_path, fake, key_file, host, ssh_lost_s=2.0)
    started = time.time()
    with pytest.raises(vast.HostLost):
        # the try began an hour ago and worked; its ssh noticed the loss only now
        vast.await_host(run, "verify m", started + 3600, "exit 255", since=started - 3600)
    assert time.time() - started >= 1.9 and min(host.timeouts) >= 1.0


@pytest.mark.slow
def test_a_try_that_used_its_timeout_counts_as_silence_to_the_steps_end(
    tmp_path: Path, key_file, fake
) -> None:
    """An ssh blackholed right after it connected takes the try's whole timeout to fail: the
    step's end then cuts the wait, and the silence since the try began is the window."""
    run = _transport_run(tmp_path, fake, key_file, _SlowProbeHost(), ssh_lost_s=3.0)
    started = time.time()
    with pytest.raises(vast.HostLost):
        vast.await_host(run, "state m", started + 1.5, "exit 255", since=started - 1.5)
    assert run.fatal == "ssh_lost" and time.time() - started < 2.5


@pytest.mark.slow
def test_a_long_try_that_worked_before_its_loss_is_no_ssh_lost_when_the_step_ends(
    tmp_path: Path, key_file, fake
) -> None:
    """A verify that ran 500 s and noticed its loss at once: the step's end 2 s later cuts the
    wait, and 2 s of silence is no ssh_lost of the attempt."""
    run = _transport_run(tmp_path, fake, key_file, _SlowProbeHost(), ssh_lost_s=300)
    started = time.time()
    with pytest.raises(vast.TransportError) as raised:
        vast.await_host(run, "verify m", started + 2.0, "exit 255", since=started - 500)
    assert not isinstance(raised.value, vast.HostLost) and run.fatal is None
    assert (
        vast.SSH_NOTICE_S == 60
        and "ServerAliveInterval=15" in vast.RemoteHost(tmp_path, "127.0.0.1", 22).ssh_argv()
    )


def test_a_retry_runs_with_what_is_left_of_a_short_step(tmp_path: Path, key_file, fake) -> None:
    """A step shorter than `MIN_STEP_S` (the dry run's GPU check, say) still runs again after a
    blip, with the seconds left and never past its end."""
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost(down=1))
    action, calls = _flaky(1)
    until = time.time() + 4.0
    assert vast.transport_retry(run, "gpu_free", action, until) == "done"
    assert calls[0] == vast.MIN_STEP_S and 3.0 < calls[1] < 4.0
    action, calls = _flaky(10**9)
    with pytest.raises(vast.TransportError):
        vast.transport_retry(run, "gpu_free", action, time.time() + 0.5)
    assert len(calls) == 1


# --- round 3 of the stack check (#165) -------------------------------------------------------


@pytest.mark.slow
def test_a_fork_the_station_refused_runs_the_step_again_after_a_growing_pause(
    tmp_path: Path, key_file, fake
) -> None:
    host = _RecoverHost()
    run = _transport_run(tmp_path, fake, key_file, host, ssh_retry_s=1.2)
    calls = []

    def action(left: float) -> str:
        calls.append(time.time())
        if len(calls) <= 3:
            raise OSError((errno.ENOMEM, errno.EAGAIN, errno.ENOMEM)[len(calls) - 1], "no fork")
        return "done"

    assert vast.transport_retry(run, "size m", action, time.time() + 30) == "done"
    pauses = [round(b - a, 1) for a, b in itertools.pairwise(calls)]
    assert pauses == [0.5, 1.0, 1.2]  # 0.5 s doubling, never longer than ssh_retry_s
    retries = [e for e in run.journal.replay() if e["event"] == "transport_retry"]
    assert [(e["outcome"], e["tries"]) for e in retries] == [("local_retry", n) for n in (1, 2, 3)]
    assert retries[0]["error"] == "OSError: ENOMEM" and not host.calls  # the host is not asked


@pytest.mark.slow
def test_a_refused_fork_ends_the_step_at_its_end_and_any_other_os_error_at_once(
    tmp_path: Path, key_file, fake
) -> None:
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost(), ssh_retry_s=0.2)
    calls = []

    def no_memory(left: float):
        calls.append(left)
        raise OSError(errno.ENOMEM, "no fork")

    started = time.time()
    with pytest.raises(OSError, match="no fork"):
        vast.transport_retry(run, "pack", no_memory, time.time() + 1.0)
    assert 0.6 < time.time() - started < 1.5 and len(calls) >= 4

    def no_space(left: float):
        raise OSError(errno.ENOSPC, "disk full")

    with pytest.raises(OSError, match="disk full"):
        vast.transport_retry(run, "pack", no_space, time.time() + 30)
    assert not [
        e for e in run.journal.replay() if e.get("what") == "pack" and "ENOSPC" in e["error"]
    ]


class _Reopens:
    def __init__(self) -> None:
        self.forced: list[bool] = []

    def __call__(self, force: bool = False) -> bool:
        self.forced.append(force)
        return force


@pytest.mark.slow
def test_the_health_wait_stops_at_once_on_the_soft_stop_and_skips_the_model() -> None:
    stops = iter([None, None, "soft_end"])
    started = time.monotonic()
    with pytest.raises(vast.ModelSkipped, match="smoke: stopped while the server loads: soft_end"):
        vast._wait_health(
            _NeverHealthy(), 60.0, lambda: None, 0.0, 100.0, _Reopens(), lambda: next(stops)
        )
    assert 1.5 < time.monotonic() - started < 4
    assert isinstance(vast.ModelSkipped("smoke", "x"), vast.ModelFailed)


@pytest.mark.slow
def test_the_health_wait_opens_a_live_but_stale_tunnel_anew_every_fourth_state_check() -> None:
    reopen = _Reopens()
    with pytest.raises(vast.VastError, match="did not answer /health in time"):
        vast._wait_health(_NeverHealthy(), 8.5, lambda: None, 0.0, 100.0, reopen)
    assert reopen.forced[:8] == [False, False, False, True, False, False, False, True]


def _models_run(tmp_path, fake, key_file) -> vast.Run:
    run = _soft_run(tmp_path, fake, key_file)
    run.package["models"] = [{"name": "m1"}, {"name": "m2"}]
    return run


def test_a_cancellation_after_the_gate0_day_keeps_the_verdict(
    tmp_path: Path, key_file, fake, monkeypatch
) -> None:
    def serve(run, entry, receipt, after_gate0, result):
        result.update(verdict="NOT READY", reason="C", stop_reason=None)
        run.journal.append("step", name="evidence", model=entry["name"], ok=False)
        raise SystemExit(143)

    monkeypatch.setattr(vast, "ensure_weights", lambda run, entry, late: Path("receipt"))
    monkeypatch.setattr(vast, "serve_model", serve)
    run = _models_run(tmp_path, fake, key_file)
    with pytest.raises(SystemExit):
        vast.run_models(run)
    outcomes = [e for e in run.journal.replay() if e["event"] == "model_outcome"]
    assert len(outcomes) == 1 and outcomes[0]["outcome"] == "ok"
    assert outcomes[0]["verdict"] == "NOT READY"
    assert outcomes[0]["cancelled"] == {"reason": "cancelled: SystemExit", "step": "evidence"}


def test_an_error_a_signal_caused_on_its_way_up_is_still_a_cancellation(
    tmp_path: Path, key_file, fake, monkeypatch
) -> None:
    def serve(run, entry, receipt, after_gate0, result):
        run.fatal, run.cancel_code = "cancelled", 143  # the signal came, then an EBADF won
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(vast, "ensure_weights", lambda run, entry, late: Path("receipt"))
    monkeypatch.setattr(vast, "serve_model", serve)
    run = _models_run(tmp_path, fake, key_file)
    with pytest.raises(SystemExit) as raised:
        vast.run_models(run)
    assert raised.value.code == 143
    outcomes = [e for e in run.journal.replay() if e["event"] == "model_outcome"]
    assert [(e["model"], e["outcome"], e["reason"]) for e in outcomes] == [
        ("m1", "cancelled", "cancelled: OSError")
    ]


def test_a_model_the_soft_stop_cut_in_smoke_is_skipped_not_failed(
    tmp_path: Path, key_file, fake, monkeypatch
) -> None:
    def serve(run, entry, receipt, after_gate0, result):
        raise vast.ModelSkipped("smoke", "stopped while the server loads: soft_end")

    monkeypatch.setattr(vast, "ensure_weights", lambda run, entry, late: Path("receipt"))
    monkeypatch.setattr(vast, "serve_model", serve)
    monkeypatch.setattr(vast, "wait_gpu_free", lambda run: {"memory_used_mib": 0})
    run = _models_run(tmp_path, fake, key_file)
    vast.run_models(run)
    outcomes = [e for e in run.journal.replay() if e["event"] == "model_outcome"]
    assert [(e["outcome"], e["step"]) for e in outcomes] == [("skipped", "smoke")] * 2
    assert "soft_end" in outcomes[0]["reason"]


def test_the_probe_of_the_host_counts_a_refused_fork_as_no_answer(
    tmp_path: Path, key_file, fake
) -> None:
    class Host(_RecoverHost):
        def run(self, argv, timeout):
            if not self.calls:
                self.calls.append(argv[:3])
                raise OSError(errno.EAGAIN, "no fork")
            return super().run(argv, timeout)

    host = Host()
    run = _transport_run(tmp_path, fake, key_file, host)
    assert vast.await_host(run, "size m", time.time() + 10, "exit 255") is False
    assert host.calls == [["true"], ["true"]]
    run = _transport_run(tmp_path, fake, key_file, _RecoverHost())
    run.host.run = lambda argv, timeout: (_ for _ in ()).throw(OSError(errno.EACCES, "denied"))
    with pytest.raises(OSError, match="denied"):
        vast.await_host(run, "size m", time.time() + 10, "exit 255")


def test_a_pilot_count_the_server_answered_without_is_not_asked_for_again() -> None:
    class Client:
        model, last_status, asked = "m", None, 0

        def text_tokens(self, text):
            self.asked += 1
            self.last_status = None if self.asked == 1 else 500
            return None

        def request(self, path, body=None, timeout=5):
            self.last_status = 200
            return 200, json.dumps({"usage": {"prompt_tokens": 50}}).encode(), None

    client, checks = Client(), []
    vast.run_pilot(client, lambda after_timeout=False: checks.append(1) or "resend")
    # lost once (no HTTP answer): asked again; the server answered 500 then: no more asks
    assert client.asked == 2 and checks == [1]


def test_a_signal_a_step_swallowed_still_cancels_after_the_model(
    tmp_path: Path, key_file, fake, monkeypatch
) -> None:
    def serve(run, entry, receipt, after_gate0, result):
        result.update(verdict="NOT READY", reason="C", stop_reason=None)
        run.fatal, run.cancel_code = "cancelled", 129  # SIGHUP, its exit noted as a block error

    monkeypatch.setattr(vast, "ensure_weights", lambda run, entry, late: Path("receipt"))
    monkeypatch.setattr(vast, "serve_model", serve)
    run = _models_run(tmp_path, fake, key_file)
    with pytest.raises(SystemExit) as raised:
        vast.run_models(run)
    assert raised.value.code == 129
    outcomes = [e for e in run.journal.replay() if e["event"] == "model_outcome"]
    assert len(outcomes) == 1 and outcomes[0]["outcome"] == "ok"
    assert outcomes[0]["cancelled"]["reason"] == "cancelled: swallowed"


def test_a_slow_reopen_of_the_dry_tunnel_counts_as_a_live_ssh_until_it_binds(tmp_path) -> None:
    """A plain `ensure_tunnel` while a slow reopen is in flight opens no second forwarder on
    its port (that bind failed with EADDRINUSE in a thread, CI run 45141440965)."""
    host = vast.LocalHost(tmp_path / "root", {})
    target = vast_fake.Forwarder(vast._free_port())
    port = host.tunnel(target.port)
    host.reopen_delay_s = 0.5
    host.kill_tunnels()
    assert host.ensure_tunnel(force=True) is True  # the slow reopen starts
    assert host.ensure_tunnel() is False and host.ensure_tunnel(force=True) is False
    deadline = time.monotonic() + 5
    while len(host._forwarders) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [f.port for f in host._forwarders][-1] == port and host._forwarders[-1].alive()
    assert host.ensure_tunnel() is False
    host.close()
    target.stop()


# A dry run in a child process: `failed` raises once the fake server answered its health check,
# before the first Gate 0 request (`pilot`); `killed` and `completed` run it unchanged.
_ENDING_CHILD = """
import sys
from pathlib import Path

from sound_scout import config
from sound_scout.eval import vast

out, ending = Path(sys.argv[1]), sys.argv[2]


def fail(run, **_):
    raise RuntimeError("the test fails the dry run after the server is up")


hooks = {"pilot": fail} if ending == "failed" else {}
(out.parent / "in").mkdir()
config_ = config.Config(allowed_roots=((out.parent / "in").resolve(),), data_root=out.parent)
vast.dry_run(out, config_, scenario={"hooks": hooks, "extended": False})
"""


def _processes_naming(path: Path, *, fake_vllm: bool = False) -> list[int]:
    """Live processes whose argv holds `path`: the dry run, its `hostrun serve` and fake
    `vllm serve`; with `fake_vllm`, only the fake `vllm serve`."""
    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        with contextlib.suppress(OSError):
            pid = int(cmdline.parent.name)
            argv = cmdline.read_bytes().split(b"\0")
            if fake_vllm and not any(
                a.endswith(b"/bin/vllm") and b == b"serve" for a, b in itertools.pairwise(argv)
            ):
                continue
            if any(str(path).encode() in a for a in argv) and hostrun._alive(pid):
                found.append(pid)
    return found


@pytest.mark.slow
@pytest.mark.parametrize("ending", ["completed", "failed", "killed"])
def test_a_dry_run_leaves_no_process_of_its_fake_host_however_it_ends(
    tmp_path: Path, ending: str
) -> None:
    """#198: a SIGKILLed dry run had the kernel SIGKILL its `hostrun serve`, and the fake vLLM
    server, in a session of its own, lived on."""
    out = tmp_path / "dry"
    child = subprocess.Popen(
        [sys.executable, "-c", _ENDING_CHILD, str(out), ending],
        cwd=REPO,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ends = time.monotonic() + 120
        while not list(out.glob("host-root/out/*/serve.pid")) and time.monotonic() < ends:
            assert child.poll() is None, f"the dry run exited {child.returncode} before serve"
            time.sleep(0.05)
        assert _processes_naming(out, fake_vllm=True), "no fake server process started"
        if ending == "killed":
            os.killpg(child.pid, signal.SIGKILL)
        code = child.wait(240)
        assert (code == 0) == (ending == "completed"), code
        ends = time.monotonic() + 30  # `serve` stops its session within `hostrun.GRACE_S`
        while _processes_naming(out) and time.monotonic() < ends:
            time.sleep(0.1)
        assert _processes_naming(out) == []
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(30)
        for pid in _processes_naming(out):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


# A server that ignores SIGTERM; it says so in the file `argv[1]`.
_IGNORES_SIGTERM = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], "w").close()
time.sleep(120)
"""


def _dry_host_with_serve(tmp_path: Path, server: list[str]) -> tuple[vast.LocalHost, Path]:
    """A dry host directory with the host files, its `hostrun serve` of `server` spawned; the
    host and the model's out directory."""
    root = tmp_path / "root"
    host = vast.LocalHost(root, vast._fake_host_root(root))
    (root / "ss" / "guard").mkdir(parents=True)
    (root / "ss" / "hostrun.py").write_bytes((vast.HOST_FILES_DIR / "hostrun.py").read_bytes())
    guard = vast.HOST_FILES_DIR / "sitecustomize.py"
    (root / "ss" / "guard" / "sitecustomize.py").write_bytes(guard.read_bytes())
    (root / "out" / "m").mkdir(parents=True)
    out = f"{vast.WORKSPACE}/out/m"
    serve = ["python3", vast.HOSTRUN, "serve", "--deadline", str(time.time() + 120)]
    serve += ["--rss-limit", str(1 << 40), "--stats", f"{out}/server-stats.jsonl"]
    serve += ["--log", f"{out}/server.log", "--pid-file", f"{out}/serve.pid", "--", *server]
    host.spawn(serve, f"{out}/hostrun.log")
    return host, root / "out" / "m"


def _session_alive(out: Path) -> list[int]:
    """Live processes of the server session `serve` recorded in `out`."""
    records = hostrun._records(out / "server-stats.jsonl")
    started = next(r for r in records if r["event"] == "started")
    return [pid for pid in hostrun.session_pids(started["pid"]) if hostrun._alive(pid)]


def _kill_recorded(out: Path) -> None:
    """SIGKILL to what is left of the test's own `serve` and server session, by identity."""
    identity = hostrun._serve_identity(out / "serve.pid")
    if identity is not None:
        hostrun._signal(*identity, signal.SIGKILL)
    for pid, start in hostrun._session_survivors(out / "server-stats.jsonl", set()) or []:
        hostrun._signal(pid, start, signal.SIGKILL)


@pytest.mark.slow
def test_close_of_the_dry_host_ends_a_server_that_ignores_sigterm(tmp_path, monkeypatch) -> None:
    """#203: `close` SIGKILLed `hostrun serve` after `STOP_GRACE_S`, before the `GRACE_S` of
    `serve` ended, and a server that ignores SIGTERM lived on in its own session."""
    ready = tmp_path / "ready"
    host, out = _dry_host_with_serve(tmp_path, [sys.executable, "-c", _IGNORES_SIGTERM, str(ready)])
    try:
        ends = time.monotonic() + 60
        while not (ready.exists() and (out / "serve.pid").exists()) and time.monotonic() < ends:
            time.sleep(0.05)
        assert _session_alive(out), "the server did not start"
        serve_ = hostrun._serve_identity(out / "serve.pid")
        monkeypatch.setattr(gate0, "STOP_GRACE_S", 1.0)  # `serve` is SIGKILLed inside its grace
        host.close()
        assert _session_alive(out) == []
        assert not hostrun._is_running(*serve_)
    finally:
        _kill_recorded(out)
        host.close()


def test_close_of_the_dry_host_keeps_a_session_that_outlived_it(tmp_path, monkeypatch) -> None:
    host = vast.LocalHost(tmp_path, {})
    host.close()
    assert host.sessions_stopped is True
    monkeypatch.setattr(vast, "stop_local_sessions", lambda root: False)
    host.close()
    monkeypatch.setattr(vast, "stop_local_sessions", lambda root: True)
    host.close()
    assert host.sessions_stopped is False  # a later clean close does not hide it


@pytest.mark.slow
def test_a_dry_run_fails_when_a_host_session_outlives_its_close(tmp_path, monkeypatch) -> None:
    """#203: `close` runs in deploy's `finally` and must not raise there; the dry run reports."""
    monkeypatch.setattr(vast, "stop_local_sessions", lambda root: False)
    out = tmp_path / "data" / "dry"
    with pytest.raises(vast.VastError, match="outlived the dry run"):
        vast.dry_run(out, _dry_config(tmp_path))
    timeline = json.loads((out / "timeline.json").read_text())
    assert "outlived the dry run" in timeline["failure"]
    assert timeline["steps"][-1]["name"] == "destroy"


def test_stop_local_sessions_fails_while_a_serve_does_not_exit(tmp_path) -> None:
    """A `serve` that outlives its SIGTERM is reported, not killed: only `stop` decides."""
    assert vast.stop_local_sessions(tmp_path) is True  # no `serve` at all
    ready = tmp_path / "ready"
    stuck = subprocess.Popen(
        [sys.executable, "-c", _IGNORES_SIGTERM, str(ready)], start_new_session=True
    )
    try:
        ends = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < ends:
            time.sleep(0.02)
        (tmp_path / "out" / "m").mkdir(parents=True)
        pid_file = tmp_path / "out" / "m" / "serve.pid"
        pid_file.write_text(f"{stuck.pid} {hostrun.start_ticks(stuck.pid)}")
        assert vast.stop_local_sessions(tmp_path, timeout=0.3) is False
        assert stuck.poll() is None
    finally:
        stuck.kill()
        stuck.wait(30)
