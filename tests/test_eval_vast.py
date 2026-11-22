"""GPU stack of #147 against the fake provider API on 127.0.0.1: the key, the client, the one
create, the watchdog, the journal, the download check and the dry run.

Every test uses `vast_fake.FakeVast` and a key file in a temp dir. No test reaches Vast.ai, reads
the real key file or arms a real timer: the fakes prove the contract of the stack only.
"""

import dataclasses
import hashlib
import io
import json
import math
import os
import re
import secrets
import subprocess
import tarfile
import threading
import time
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config
from sound_scout.eval import gate0, gpu_ext, vast, vast_fake
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
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
    package, digest = vast.load_package(path, vast.MODELS)
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
