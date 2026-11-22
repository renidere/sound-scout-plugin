"""GPU stack of #147 against the fake provider API on 127.0.0.1: the key, the client, the one
create, the watchdog, the journal, the download check and the dry run.

Every test uses `vast_fake.FakeVast` and a key file in a temp dir. No test reaches Vast.ai, reads
the real key file or arms a real timer: the fakes prove the contract of the stack only.
"""

import hashlib
import io
import json
import math
import os
import re
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
from sound_scout.eval import gate0, vast, vast_fake

REPO = Path(__file__).resolve().parent.parent
DRY_STEPS = ["create", "ready", "weights", "smoke", "run", "download", "destroy"]


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
    good = {"vllm": "0.34.0", "torch": "2.14.0+cu130", "cuda": "13.0.3", "transformers": "5.22.1"}
    assert vast.runtime_mismatches(good, "0.34.0\n") == []
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


def test_example_package_loads_and_holds_no_secret() -> None:
    path = REPO / "docs" / "examples" / "gpu-package.example.json"
    package, digest = vast.load_package(path, vast.MODELS)
    assert len(digest) == 64 and package["ceiling_usd"] == 4.0
    vast.check_no_secret(path.read_bytes(), None)


@pytest.mark.parametrize(
    "change",
    [{"ceiling_usd": math.inf}, {"early_stop_usd": 5.0}, {"envelope": {}}, {"models": []}],
    ids=["inf", "early-over-ceiling", "no-dph-bound", "no-models"],
)
def test_a_malformed_package_is_refused(tmp_path: Path, change) -> None:
    package = json.loads((REPO / "docs/examples/gpu-package.example.json").read_text())
    path = tmp_path / "p.json"
    path.write_text(json.dumps({**package, **change}, allow_nan=True))
    with pytest.raises(vast.VastError):
        vast.load_package(path, vast.MODELS)


# --- the dry run ---------------------------------------------------------------------------


def test_dry_run_gives_the_full_timeline_against_the_fakes(tmp_path: Path, monkeypatch) -> None:
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
        "smoke",
        "run",
        "download",
        "destroy",
    ]
    assert all(step["ok"] for step in result["steps"])
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
    found = {"vllm": "0.34.0", "torch": "2.14.0+cu130", "cuda": "13.0.3", "transformers": "5.20.0"}
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
