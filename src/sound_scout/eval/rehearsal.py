"""The rehearsal of the #149 package without money: every scenario of the plan on the fakes.

`python -m sound_scout.eval.rehearsal --out DIR [--only NAME ...]` runs each scenario as a dry
run of the whole two-model package (`vast.dry_run`: the fake Vast API, a directory for the host
with the real `hostrun.py serve` and guard, the fake vLLM, the real Gate 0 and extended-set
clients) and checks what the owner asked for (#149, comment 7085422087): a failure of one model
leaves the other's run whole, every failure keeps the journal and the answers already given,
results are downloaded and destroy is verified; a network loss at any kind of host step
(`STEP_KINDS`, one scenario each) costs a wait, never a model. A few scenarios are unit checks
(the set's refusals, the analysis numbers, `pack` of growing files). The report is
`DIR/report.json` and `DIR/report.md`; the exit code is 0 only when every check of every
scenario passed.

What it proves is the stack's order and contract on this station; never the image, vLLM, the
models, a real host or Vast.ai.
"""

import argparse
import contextlib
import json
import os
import random
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sound_scout import config
from sound_scout.eval import gate0, gpu_ext, vast, vast_fake

M1, M2 = "qwen25omni", "qwen3omni"
REPO1, REPO2 = "fake/Qwen2.5-Omni-7B", "fake/Qwen3-Omni"
PILOT_REQUESTS = 2
GATE0_REQUESTS = 42


@dataclass
class Outcome:
    name: str
    out: Path
    result: dict | None
    error: BaseException | None
    host: object = None
    watchdog: list = field(default_factory=list)

    @property
    def events(self) -> list[dict]:
        found = sorted((self.out / vast.GPU_DIR).glob(f"*/{vast.JOURNAL_FILE}"))
        return vast.Journal(found[-1]).replay() if found else []

    @property
    def attempt(self) -> Path | None:
        found = sorted((self.out / vast.GPU_DIR).glob(f"*/{vast.JOURNAL_FILE}"))
        return found[-1].parent if found else None

    def kinds(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["event"] == kind]

    def steps(self, name: str, model: str | None = None) -> list[dict]:
        return [
            e
            for e in self.kinds("step")
            if e["name"] == name and (model is None or e.get("model") == model)
        ]

    def model(self, name: str) -> dict:
        return next((e for e in self.kinds("model_outcome") if e["model"] == name), {})

    def ext_trace(self, model: str) -> list[dict]:
        found = sorted((self.out / "gpu-ext-fixture" / gpu_ext.ATTEMPTS_DIR).glob(f"*-{model}"))
        if not found or not (found[-1] / gpu_ext.TRACE_FILE).is_file():
            return []
        text = (found[-1] / gpu_ext.TRACE_FILE).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def gate0_trace(self, day: str) -> list[dict]:
        path = self.out / day / gate0.ATTEMPTS_DIR / "1" / gate0.TRACE_FILE
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _days(model: str) -> str:
    return dict((name, day) for name, day, _, _ in vast.FULL_DAYS)[model]


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def __call__(self, name: str, passed: bool, detail: object = None) -> None:
        self.items.append({"check": name, "passed": bool(passed), "detail": detail})


# --- common checks --------------------------------------------------------------------------


def destroyed(o: Outcome, c: Checks) -> None:
    fake = (o.result or {}).get("fake_api") or _timeline(o).get("fake_api") or {}
    c("one PUT at most", fake.get("put", 0) <= 1, fake.get("put"))
    c("no instance left on the fake account", fake.get("instances_left") == [], fake)
    closed = o.kinds("closed")
    c("attempt closed", bool(closed), closed[-1]["reason"] if closed else None)
    if o.kinds("create_response") or o.kinds("adopted_by_label"):
        c("absence verified", bool(o.kinds("absence_verified")))


def downloaded(o: Outcome, c: Checks) -> None:
    verified = o.kinds("download_verified")
    c("final download verified (SHA-256 of archive and files)", bool(verified))
    if verified and o.attempt is not None:
        c("host results on the station", (o.attempt / vast.HOST_DIR / "SHA256SUMS").is_file())


def no_secret(o: Outcome, c: Checks) -> None:
    key = (o.out / "fake-api-key").read_text().strip().encode()
    leaks = [
        str(p.relative_to(o.out))
        for p in o.out.rglob("*")
        if p.is_file() and p.name != "fake-api-key" and key in p.read_bytes()
    ]
    c("the key is in no file", not leaks, leaks[:3])


def _timeline(o: Outcome) -> dict:
    path = o.out / "timeline.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def model_ok(o: Outcome, c: Checks, model: str) -> None:
    outcome = o.model(model)
    c(f"{model}: outcome ok with a Gate 0 verdict", outcome.get("outcome") == "ok", outcome)
    sent = [line for line in o.gate0_trace(_days(model)) if line.get("sent") is True]
    c(f"{model}: Gate 0 sent all {GATE0_REQUESTS} requests", len(sent) == GATE0_REQUESTS, len(sent))


def model_failed(o: Outcome, c: Checks, model: str, step: str, *, kind: str = "failed") -> None:
    outcome = o.model(model)
    c(
        f"{model}: outcome {kind} at {step}",
        outcome.get("outcome") == kind and outcome.get("step") == step,
        outcome,
    )


def model_incomplete(o: Outcome, c: Checks, model: str, stop: str) -> None:
    """A day the stack stopped is `incomplete`, never a NOT READY of the model (0028)."""
    outcome = o.model(model)
    c(
        f"{model}: outcome incomplete ({stop}), no Gate 0 verdict",
        outcome.get("outcome") == "incomplete"
        and outcome.get("verdict") == "incomplete"
        and outcome.get("stop_reason") == stop,
        outcome,
    )
    path = o.out / _days(model) / gate0.ATTEMPTS_DIR / "1" / gate0.VERDICT_FILE
    verdict = json.loads(path.read_text()) if path.is_file() else {}
    c(
        f"{model}: verdict.json says incomplete and keeps the rule's reading apart",
        verdict.get("verdict") == "incomplete"
        and verdict.get("stop_reason") == stop
        and "rule_verdict" in verdict,
        {k: verdict.get(k) for k in ("verdict", "stop_reason", "rule_verdict", "unanswered")},
    )


def ext_complete(o: Outcome, c: Checks, model: str) -> None:
    trace = o.ext_trace(model)
    unsent = [line for line in trace if not line.get("sent")]
    answered = [line for line in trace if line.get("http_status") == 200]
    c(
        f"{model}: every extended request sent and answered",
        trace and not unsent and len(answered) == len(trace),
        {"lines": len(trace), "not_sent": len(unsent)},
    )


def evidence_of(o: Outcome, c: Checks, model: str) -> dict | None:
    found = [e for e in o.kinds("server_evidence") if e["model"] == model]
    c(f"{model}: server evidence journaled", bool(found))
    stopped = [e for e in o.kinds("server_stopped") if e["model"] == model]
    c(f"{model}: server stop journaled", bool(stopped))
    return found[-1] if found else None


def in_archive(o: Outcome, c: Checks, relative: str) -> None:
    path = None if o.attempt is None else o.attempt / vast.HOST_DIR / relative
    c(f"{relative} in the downloaded results", path is not None and path.is_file())


# --- the scenarios ----------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    title: str
    setup: dict | Callable[[], dict]
    check: Callable[[Outcome, Checks], None]
    watchdog: bool = False
    expect_error: str | None = None
    stale_s: float | None = None
    signal_when: Callable[[Path], bool] | None = None
    signal_group: bool = False


def _block_end(model: str, block: str, action: Callable) -> dict:
    def hook(run, **where) -> None:
        if where == {"model": model, "block": block}:
            action(run)

    return {"block_end": hook}


def _gate0_sent(trace: Path) -> int:
    if not trace.is_file():
        return 0
    sent = 0
    for text in trace.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(ValueError):
            sent += json.loads(text).get("sent") is True
    return sent


def _gate0_trace_path(data_root: Path, model: str) -> Path:
    return data_root / _days(model) / gate0.ATTEMPTS_DIR / "1" / gate0.TRACE_FILE


def _when_gate0_sent(model: str, count: int, action: Callable) -> dict:
    """A host hook: `action(run)` once, from a thread, when Gate 0 of `model` sent `count`."""

    def hook(run) -> None:
        trace = _gate0_trace_path(run.config_.data_root, model)

        def loop() -> None:
            ends = time.monotonic() + 600
            while time.monotonic() < ends:
                if _gate0_sent(trace) >= count:
                    action(run)
                    return
                time.sleep(0.02)

        threading.Thread(target=loop, daemon=True).start()

    return {"host": hook}


def check_success(o: Outcome, c: Checks) -> None:
    for model in (M1, M2):
        model_ok(o, c, model)
        ext_complete(o, c, model)
        evidence_of(o, c, model)
    order = [
        f"{e['name']}:{e.get('model')}"
        if e["event"] == "step"
        else f"{e['event']}:{e.get('model')}"
        for e in o.events
        if e.get("model") in (M1, M2)
        and e["event"] in ("step", "server_evidence", "server_stopped", "model_outcome")
    ]
    want = [
        f"{event}:{model}"
        for model in (M1, M2)
        for event in (
            "weights",
            "smoke",
            "run",
            "extended",
            "server_evidence",
            "server_stopped",
            "model_outcome",
        )
    ]
    c("journal order per model", order == want, order)
    syncs = o.kinds("sync")
    c(
        "one sync per block, all verified",
        len(syncs) == 10 and all(s["ok"] for s in syncs),
        len(syncs),
    )
    gate0_clips = {clip for model in (M1, M2) for clip in _clip_ids(o, model)}
    ext_clips = {line.get("clip") for model in (M1, M2) for line in o.ext_trace(model)} - {None}
    c(
        "B/V and the Gate 0 controls only in Gate 0",
        not gate0_clips & ext_clips,
        sorted(gate0_clips),
    )
    for model in (M1, M2):
        in_archive(o, c, f"{model}/server.log")
        in_archive(o, c, f"{model}/server-stats.jsonl")
    analyses = sorted((o.out / "gpu-ext-fixture" / gpu_ext.ATTEMPTS_DIR).glob("*/analysis.json"))
    c("analysis written for both models", len(analyses) == 2)
    started = [e for e in o.kinds("weights_started")]
    m2_start = next((e["t"] for e in started if e["model"] == M2), None)
    m1_run = next((e["finished_at"] for e in o.steps("run", M1)), None)
    c(
        "M2 weights start after the Gate 0 day of M1, during its blocks",
        m2_start is not None and m1_run is not None and m2_start >= m1_run,
    )
    c("GPU memory checked before M2", bool(o.kinds("gpu_free")))
    downloaded(o, c)
    destroyed(o, c)
    no_secret(o, c)


def _clip_ids(o: Outcome, model: str) -> set[str]:
    manifest = o.out / _days(model) / gate0.MANIFEST_FILE
    return {clip["id"] for clip in json.loads(manifest.read_text())["clips"]}


def check_m2_after_m1(step: str, extra: Callable[[Outcome, Checks], None] | None = None):
    def check(o: Outcome, c: Checks) -> None:
        model_failed(o, c, M1, step)
        model_ok(o, c, M2)
        ext_complete(o, c, M2)
        if extra is not None:
            extra(o, c)
        downloaded(o, c)
        destroyed(o, c)
        no_secret(o, c)

    return check


def _exit_extra(o: Outcome, c: Checks) -> None:
    found = evidence_of(o, c, M1)
    tail = (found or {}).get("tails", {}).get("server.log", "")
    c("M1 evidence holds the server's last line", "fake engine core died" in tail)
    in_archive(o, c, f"{M1}/server.log")
    smoke = o.steps("smoke", M1)
    c(
        "M1 smoke failed fast (before the health timeout)",
        bool(smoke) and smoke[0]["elapsed_s"] < 30,
        smoke[0]["elapsed_s"] if smoke else None,
    )


def check_crash(o: Outcome, c: Checks) -> None:
    trace = o.gate0_trace(_days(M1))
    sent = [line for line in trace if line.get("sent") is True]
    stopped = [
        line for line in trace if line.get("not_sent") in ("server_exited", "server_unreachable")
    ]
    c(
        "M1 Gate 0 stopped after the crash within 3 requests",
        len(sent) <= 5 + 3 - PILOT_REQUESTS,
        len(sent),
    )
    c(
        "the rest of M1 Gate 0 is not_sent: server_exited",
        len(stopped) >= GATE0_REQUESTS - 6,
        len(stopped),
    )
    run = o.steps("run", M1)
    c(
        "M1 run step took seconds, not the request timeout",
        bool(run) and run[0]["elapsed_s"] < 20,
        run[0]["elapsed_s"] if run else None,
    )
    ext = o.ext_trace(M1)
    c(
        "M1 extended blocks not sent after the crash",
        ext and all(not line.get("sent") for line in ext),
    )
    model_incomplete(o, c, M1, "server_exited")
    model_ok(o, c, M2)
    ext_complete(o, c, M2)
    downloaded(o, c)
    destroyed(o, c)


HANG_AT = PILOT_REQUESTS + GATE0_REQUESTS + 48 + 24 + 3


def check_hang(o: Outcome, c: Checks) -> None:
    ext = o.ext_trace(M1)
    er = [line for line in ext if line.get("block") == "E-R"]
    timeouts = [line for line in er if line.get("limit") == "request_timeout"]
    closed = [line for line in er if line.get("not_sent") == "block_timeouts"]
    c("two timeouts in M1 E-R", len(timeouts) == 2, len(timeouts))
    c("the rest of M1 E-R closed as block_timeouts", bool(closed), len(closed))
    ea = [line for line in ext if line.get("block") == "E-A"]
    c("M1 E-A still sent after the closed block", ea and all(line.get("sent") for line in ea))
    done = [e for e in o.kinds("block_done") if e["model"] == M1 and e["block"] == "E-R"]
    c(
        "M1 E-R took about two closed timeouts",
        bool(done) and done[0]["elapsed_s"] < 3 * 5.0 + 5,
        done[0]["elapsed_s"] if done else None,
    )
    model_ok(o, c, M2)
    downloaded(o, c)
    destroyed(o, c)


def check_weights_late(o: Outcome, c: Checks) -> None:
    model_ok(o, c, M1)
    ext_complete(o, c, M1)
    outcome = o.model(M2)
    c("M2 failed as weights_late", outcome.get("reason") == "weights_late", outcome)
    c("M2 never started a server", not o.steps("smoke", M2))
    downloaded(o, c)
    destroyed(o, c)


def check_soft_end(o: Outcome, c: Checks) -> None:
    model_ok(o, c, M1)
    ext = o.ext_trace(M2)
    er = [line for line in ext if line.get("block") == "E-R"]
    c("M2 E-R partly sent", any(line.get("sent") for line in er))
    c(
        "the rest of M2 is not_sent: soft_end",
        any(line.get("not_sent") == "soft_end" for line in er),
    )
    c(
        "M2 E-A and E-D not sent",
        all(
            line.get("not_sent") == "soft_end"
            for line in ext
            if line.get("block") in ("E-A", "E-D")
        ),
    )
    downloaded(o, c)
    destroyed(o, c)
    opened = o.kinds("opened")[0]
    closed = o.kinds("closed")
    c("closed before the deadline", bool(closed) and closed[-1]["t"] < opened["deadline_unix"])


def check_watchdog(reason: str):
    def check(o: Outcome, c: Checks) -> None:
        c(
            "the attempt ends in an error (deploy exits 2)",
            o.error is not None,
            None if o.error is None else str(o.error)[:200],
        )
        c(
            "the error names the watchdog's destroy, not a server that did not stop",
            o.error is not None and "did not stop" not in str(o.error),
            None if o.error is None else str(o.error)[:200],
        )
        closed = o.kinds("closed")
        c(
            f"the watchdog closed the attempt ({reason})",
            bool(closed) and reason in closed[-1]["reason"],
            closed[-1]["reason"] if closed else None,
        )
        c("earlier syncs are on the station", any(s["ok"] for s in o.kinds("sync")))
        trace = o.ext_trace(M1)
        c(
            "answers given before the destroy are on the station",
            any(line.get("http_status") == 200 for line in trace),
        )
        destroyed(o, c)
        no_secret(o, c)

    return check


def check_deadline(o: Outcome, c: Checks) -> None:
    # the host side as it is on the host (the fake host directory), since the destroy took it
    path = o.out / "host-root" / "out" / M1 / "server-stats.jsonl"
    stats = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    exits = [r.get("reason") for r in stats if r.get("event") == "exit"]
    c("serve stopped vLLM at the deadline (host side)", exits == ["deadline"], exits)
    check_watchdog("deadline")(o, c)


def check_create_lost(o: Outcome, c: Checks) -> None:
    c("create outcome unknown journaled", bool(o.kinds("create_outcome_unknown")))
    c("no model request: no smoke step", not o.steps("smoke"))
    c("the instance of the lost create was adopted by label", bool(o.kinds("adopted_by_label")))
    destroyed(o, c)


def check_spend_smooth(o: Outcome, c: Checks) -> None:
    lines = o.ext_trace(M1)
    c("M2 skipped on the spend", o.model(M2).get("reason") == "spend", o.model(M2))
    c(
        "the agent stopped itself on the soft spend",
        any(line.get("not_sent") == "spend" for line in lines),
    )
    first = next((e for e in o.kinds("destroy_requested")), {})
    c(
        "the watchdog ran, and the first destroy was the agent's own",
        bool(o.watchdog) and first.get("reason") == "agent",
        {"watchdog": o.watchdog, "first_destroy": first.get("reason")},
    )
    downloaded(o, c)
    destroyed(o, c)
    closed = o.kinds("closed")
    c(
        "closed by the agent's destroy, not the watchdog",
        bool(closed) and closed[-1]["reason"] == "destroy_requested",
        closed[-1]["reason"] if closed else None,
    )


def check_ssh_lost(o: Outcome, c: Checks) -> None:
    c("the error says the host is lost", o.error is not None)
    trace = o.ext_trace(M1)
    c(
        "M1 answers before the loss are on the station",
        any(line.get("http_status") == 200 for line in trace),
    )
    c(
        "the host was awaited for the SSH window, then ssh_lost",
        any(e.get("outcome") == "ssh_lost" for e in o.kinds("transport_retry")),
    )
    c("M2 never started", not o.steps("smoke", M2))
    destroyed(o, c)


def check_not_stopping(o: Outcome, c: Checks) -> None:
    c(
        "the stop failed and ended the attempt",
        o.error is not None and "did not stop" in str(o.error),
    )
    model_ok(o, c, M1)
    c("M2 never started", not o.steps("smoke", M2) and not o.steps("weights", M2))
    downloaded(o, c)
    destroyed(o, c)


def check_not_stopping_after_failure(o: Outcome, c: Checks) -> None:
    c("the failed stop ended the attempt", o.error is not None and "did not stop" in str(o.error))
    model_failed(o, c, M1, "smoke")
    c("M2 never started", not o.steps("smoke", M2) and not o.steps("weights", M2))
    downloaded(o, c)
    destroyed(o, c)


def check_extended_error(o: Outcome, c: Checks) -> None:
    outcome = o.model(M1)
    c(
        "M1 keeps its Gate 0 verdict",
        outcome.get("outcome") == "ok" and outcome.get("verdict"),
        outcome,
    )
    c("M1 notes the blocks' error", "RuntimeError" in str(outcome.get("extended_error")), outcome)
    model_ok(o, c, M2)
    ext_complete(o, c, M2)
    downloaded(o, c)
    destroyed(o, c)


def _broken_block(run) -> None:
    raise RuntimeError("a block broke")


def check_interrupt(o: Outcome, c: Checks) -> None:
    c("the cancellation propagated", isinstance(o.error, KeyboardInterrupt))
    model_ok(o, c, M1)
    evidence_of(o, c, M2)
    downloaded(o, c)
    destroyed(o, c)


def check_tunnel(o: Outcome, c: Checks) -> None:
    checks = o.kinds("transport_check")
    c(
        "the dead tunnel was reopened and the request resent",
        any(e["tunnel_reopened"] and e["outcome"] == "resend" for e in checks),
        checks[:2],
    )
    trace = o.ext_trace(M1)
    c("a line carries resent_after_tunnel", any(line.get("resent_after_tunnel") for line in trace))
    ext_complete(o, c, M1)
    model_ok(o, c, M2)
    downloaded(o, c)
    destroyed(o, c)


def check_gpu(o: Outcome, c: Checks) -> None:
    model_ok(o, c, M1)
    model_failed(o, c, M2, "gpu_free", kind="skipped")
    c("M2 never started a server", not o.steps("smoke", M2))
    downloaded(o, c)
    destroyed(o, c)


def check_offer(o: Outcome, c: Checks) -> None:
    skipped = o.kinds("offer_skipped")
    c(
        "the gone offer was passed over",
        [e["offer_id"] for e in skipped] == [vast_fake.OFFER_ID],
        skipped,
    )
    requested = o.kinds("create_requested")
    c(
        "one create on the next approved offer",
        [e["offer_id"] for e in requested] == [vast_fake.OFFER_ID + 1],
    )
    model_ok(o, c, M1)
    model_ok(o, c, M2)
    downloaded(o, c)
    destroyed(o, c)


def check_pack_blocked(o: Outcome, c: Checks) -> None:
    c("the failed final download is the attempt's error", o.error is not None)
    c("syncs before it are on the station", any(s["ok"] for s in o.kinds("sync")))
    model_ok(o, c, M1)
    model_ok(o, c, M2)
    download = o.steps("download")
    c("download step failed and was journaled", bool(download) and not download[0]["ok"])
    destroyed(o, c)


def check_compressed(o: Outcome, c: Checks) -> None:
    model_ok(o, c, M1)
    model_ok(o, c, M2)
    c("M1 blocks end at its slot", o.model(M1).get("extended_stop") == "slot_end", o.model(M1))
    c("M2 blocks end at the soft end", o.model(M2).get("extended_stop") == "soft_end", o.model(M2))
    opened = o.kinds("opened")[0]
    download = o.steps("download")
    c(
        "download verified before the deadline",
        bool(download)
        and download[0]["ok"]
        and download[0]["finished_at"] < opened["deadline_unix"],
    )
    c(
        "download after the soft end",
        bool(download) and download[0]["started_at"] >= opened["soft_end_unix"],
    )
    destroyed(o, c)


def _credit_jump(run) -> None:
    """3 USD gone at once (over the early stop), then the agent busy for a second, as inside a
    request, so the watchdog is the one that sees it."""
    run.api.call("POST", "/fake/credit", {"spend": 3.0})
    time.sleep(1.0)


def _spend_reached(run) -> None:
    """The clock-based estimate reaches the soft spend now; the credit, which the watchdog reads,
    does not move (as it lags on Vast.ai)."""
    run.package["spend_soft_usd"] = vast.spent_estimate(run)


def _lose_now(run) -> None:
    run.host.lose()


def _interrupt(run) -> None:
    raise KeyboardInterrupt


def _move_soft_end(run) -> None:
    run.soft_end = time.time() + 0.5


def _freeze(run) -> None:
    run.heartbeat.stop()
    ends = time.monotonic() + 60
    while time.monotonic() < ends and not vast.summarize(run.journal.replay())["closed"]:
        time.sleep(0.2)


def _block_stop(run) -> None:
    run.host.failing.append(["python3", vast.HOSTRUN, "stop"])


def _block_pack(run) -> None:
    (run.host.root / "results.tar").mkdir()


def _fallback_offer(fake) -> None:
    fake.offers = {
        vast_fake.OFFER_ID + 1: {**vast_fake.OFFER, "ask_contract_id": vast_fake.OFFER_ID + 1}
    }


def check_resent(after_timeout: bool = False):
    """The tunnel came back (reopened, and with `after_timeout` the stale one restarted) and
    the request went once more; nothing of M1 was lost."""

    def check(o: Outcome, c: Checks) -> None:
        checks = o.kinds("transport_check")
        c(
            "the tunnel was reopened and the request resent",
            any(
                e["tunnel_reopened"]
                and e["outcome"] == "resend"
                and e.get("after_timeout", False) == after_timeout
                for e in checks
            ),
            [(e.get("after_timeout"), e["tunnel_reopened"], e["outcome"]) for e in checks],
        )
        lines = o.ext_trace(M1) + o.gate0_trace(_days(M1))
        c("a line carries resent_after_tunnel", any(x.get("resent_after_tunnel") for x in lines))
        model_ok(o, c, M1)
        ext_complete(o, c, M1)
        model_ok(o, c, M2)
        ext_complete(o, c, M2)
        downloaded(o, c)
        destroyed(o, c)
        no_secret(o, c)

    return check


def _blip(run) -> None:
    """The network gone for 0.15 s: the tunnel dies with it, SSH answers again after."""
    run.host.lose()
    threading.Timer(0.15, lambda: setattr(run.host, "lost", False)).start()


def _slow_reopen(run) -> None:
    run.host.reopen_delay_s = 1.5
    run.host.kill_tunnels()


def _move_soft_end_now(run) -> None:
    run.soft_end = time.time()


def check_truncated(o: Outcome, c: Checks) -> None:
    c(
        "deploy ends in an error: no model gave a Gate 0 verdict",
        o.error is not None and "no model gave a Gate 0 verdict" in str(o.error),
        None if o.error is None else str(o.error)[:300],
    )
    model_incomplete(o, c, M1, "soft_end")
    trace = o.gate0_trace(_days(M1))
    c(
        "the rest of M1 Gate 0 is not_sent: soft_end",
        any(line.get("not_sent") == "soft_end" for line in trace),
    )
    c("M2 skipped on the soft end", o.model(M2).get("reason") == "soft_end", o.model(M2))
    downloaded(o, c)
    destroyed(o, c)


def check_credit_refused(o: Outcome, c: Checks) -> None:
    refused = o.kinds("credit_refused")
    c("the low credit is journaled before create", bool(refused), refused)
    c("no create asked", not o.kinds("create_requested"))
    fake = (o.result or {}).get("fake_api") or _timeline(o).get("fake_api") or {}
    c("no PUT on the fake account", fake.get("put") == 0, fake)
    closed = o.kinds("closed")
    c("attempt closed without an instance", bool(closed), closed[-1]["reason"] if closed else None)
    c("the error names the credit", o.error is not None and "credit" in str(o.error))


def check_code_refused(o: Outcome, c: Checks) -> None:
    c(
        "refused for the gpu_ext.py SHA-256 before anything opens",
        o.error is not None and "code_sha256" in str(o.error),
        None if o.error is None else str(o.error)[:300],
    )
    c("no journal opened", not o.events)
    fake = _timeline(o).get("fake_api") or {}
    c("no PUT and no API call on the fake account", fake.get("put") == 0, fake)


def check_sigterm(o: Outcome, c: Checks) -> None:
    c("deploy exited 128 + SIGTERM", o.error == 128 + signal.SIGTERM, o.error)
    c("the signal is journaled", [e["name"] for e in o.kinds("signal")] == ["SIGTERM"])
    outcome = o.model(M1)
    c("M1 outcome cancelled", outcome.get("outcome") == "cancelled", outcome)
    responses = o.out / _days(M1) / gate0.ATTEMPTS_DIR / "1" / gate0.RESPONSES_DIR
    kept = len(list(responses.glob("*.json"))) if responses.is_dir() else 0
    c("M1 Gate 0 answers before the signal are on the station", kept >= 5, kept)
    evidence_of(o, c, M1)
    c("M2 never started", not o.steps("weights", M2) and not o.steps("smoke", M2))
    downloaded(o, c)
    destroyed(o, c)
    no_secret(o, c)
    left = subprocess.run(["pgrep", "-f", str(o.out)], capture_output=True, text=True).stdout
    c("no process of the attempt left", not left.strip(), left.strip().splitlines()[:3])


def _journal_has(out: Path, event: str, **fields) -> bool:
    found = sorted((out / vast.GPU_DIR).glob(f"*/{vast.JOURNAL_FILE}"))
    if not found:
        return False
    with contextlib.suppress(vast.VastError):
        return any(
            e["event"] == event and all(e.get(k) == v for k, v in fields.items())
            for e in vast.Journal(found[-1]).replay()
        )
    return False


FINAL_PACK_MARK = "final-pack-started"


def _slow_final_pack(run) -> None:
    """The final `pack` runs 2 s in a subprocess and marks its start on the host, so the signal
    to the whole group (sent once the mark is there) kills it in flight, as `systemctl stop`
    kills an ssh or scp of the download."""
    real = run.host.run
    mark = f"{vast.WORKSPACE}/{FINAL_PACK_MARK}"

    def slow(argv, timeout):
        if argv[:3] == ["python3", vast.HOSTRUN, "pack"] and argv[-1].endswith("/results.tar"):
            argv = ["sh", "-c", f'touch {mark}; sleep 2; exec "$0" "$@"', *argv]
        return real(argv, timeout)

    run.host.run = slow


def check_sigterm_late(o: Outcome, c: Checks) -> None:
    c("deploy ended by the signal or after its path", o.error in (0, 143), o.error)
    signals = o.kinds("signal")
    c("one SIGTERM journaled", [e["name"] for e in signals] == ["SIGTERM"], signals)
    downloads = o.steps("download")
    retries = [e for e in o.kinds("transport_retry") if e["what"] == "download"]
    c(
        "the download killed in flight was tried once more and verified",
        [d["ok"] for d in downloads] == [True] and [r["outcome"] for r in retries] == ["retry"],
        {"steps": [d["ok"] for d in downloads], "retries": [r["error"] for r in retries]},
    )
    for model in (M1, M2):
        model_ok(o, c, model)
    downloaded(o, c)
    destroyed(o, c)
    no_secret(o, c)


# --- transport blips at every step (#149, round 2 of the stack check) --------------------------


def _fire(run, label: str) -> None:
    """Notes on the host that a scenario's hook fired, for its checks."""
    run.host.__dict__.setdefault("fired", []).append(label)


def _blip_for(seconds: float) -> Callable:
    def blip(run) -> None:
        run.host.lose()
        threading.Timer(seconds, lambda: setattr(run.host, "lost", False)).start()

    return blip


def _on_call(op: str, match: Callable[..., bool], action: Callable, nth: int = 1) -> Callable:
    """A host hook: `action(run)` right before the `nth` call of `host.<op>` whose arguments
    `match`; for `tunnel`, right after the tunnel opened."""

    def hook(run) -> None:
        real = getattr(run.host, op)
        seen = {"n": 0}

        def wrapped(*args, **kwargs):
            hit = match(*args, **kwargs)
            if hit:
                seen["n"] += 1
                hit = seen["n"] == nth
            if hit and op != "tunnel":
                _fire(run, f"{op} #{nth}")
                action(run)
            answer = real(*args, **kwargs)
            if hit and op == "tunnel":
                _fire(run, f"{op} #{nth}")
                action(run)
            return answer

        setattr(run.host, op, wrapped)

    return hook


def _hostrun(command: str, *needles: str) -> Callable[..., bool]:
    def match(argv, *_args, **_kwargs) -> bool:
        return argv[:3] == ["python3", vast.HOSTRUN, command] and all(
            any(needle in arg for arg in argv) for needle in needles
        )

    return match


def _argv0(*head: str, needle: str = "") -> Callable[..., bool]:
    def match(argv, *_args, **_kwargs) -> bool:
        return argv[: len(head)] == list(head) and any(needle in arg for arg in argv)

    return match


def _remote(suffix: str) -> Callable[..., bool]:
    return lambda remote, *_args, **_kwargs: suffix in remote


# every kind of host step of an attempt: (operation of the host, argument match, the most calls
# of this kind any attempt makes for sure, so the drawn call always comes)
STEP_KINDS = {
    "upload_mkdir": ("run", _argv0("mkdir", "-p", needle="/ss/guard"), 1),
    "upload_put": ("put", lambda *_a, **_k: True, 3),
    "upload_sha256sum": ("run", _argv0("sha256sum"), 1),
    "preflight_nvidia_smi": ("run", _argv0("nvidia-smi", needle="name,memory.total"), 1),
    "preflight_vllm_version": ("run", _argv0("vllm", "--version"), 1),
    "preflight_hostrun": ("run", _hostrun("preflight"), 1),
    "preflight_selftest": ("run", _hostrun("selftest"), 1),
    "fetch_mkdir": ("run", _argv0("mkdir", "-p", needle="/out/qwen"), 2),
    "fetch_spawn": ("spawn", lambda argv, log: log.endswith("download-wrapper.log"), 2),
    "weights_size": ("run", _hostrun("size"), 2),
    "weights_verify": ("run", _hostrun("weights"), 2),
    "weights_receipt": ("get", _remote("weights-receipt.json"), 2),
    "gpu_free": ("run", _argv0("nvidia-smi", needle="memory.used"), 1),
    "serve_spawn": ("spawn", lambda argv, log: log.endswith("/hostrun.log"), 2),
    "smoke_tunnel": ("tunnel", lambda remote_port, local_port=None: local_port is None, 2),
    "server_state": ("run", _hostrun("state"), 4),
    "sync_pack": ("run", _hostrun("pack", "/sync-"), 10),
    "sync_get": ("get", _remote("/sync-"), 10),
    "evidence": ("run", _hostrun("evidence"), 2),
    "stop": ("run", _hostrun("stop"), 2),
    "final_pack": ("run", _hostrun("pack", "/results.tar"), 1),
    "final_get": ("get", _remote("/results.tar"), 1),
    # through the tunnel: the pilot of a model (the n-th model) and a Gate 0 request
    "pilot": ("pilot", None, 2),
    "gate0_request": ("gate0", None, 2),
}


def _drawn(kind: str, most: int) -> tuple[int, float]:
    """The call and the blip length of a kind, drawn once with the kind as the seed."""
    rng = random.Random(f"149:{kind}")
    return rng.randint(1, most), round(rng.uniform(0.05, 1.0), 2)


def check_both_complete(o: Outcome, c: Checks) -> None:
    fired = getattr(o.host, "fired", None)
    c("the scenario's fault fired", bool(fired), fired)
    for model in (M1, M2):
        model_ok(o, c, model)
        ext_complete(o, c, model)
    downloaded(o, c)
    destroyed(o, c)
    no_secret(o, c)


def _step_kind_scenario(kind: str) -> Scenario:
    op, match, most = STEP_KINDS[kind]
    nth, seconds = _drawn(kind, most)
    model = (M1, M2)[(nth - 1) % 2]

    def blip(run, label: str = kind) -> None:
        _fire(run, label)
        _blip_for(seconds)(run)

    if op == "pilot":
        hooks = {"pilot": lambda run, **where: where["model"] == model and blip(run)}
    elif op == "gate0":
        hooks = _when_gate0_sent(model, 10, blip)
    else:
        hooks = {"host": _on_call(op, match, _blip_for(seconds), nth)}
    return Scenario(
        f"blip_step_{kind}",
        f"Обрыв сети {seconds} с на шаге {kind} (вызов {nth})",
        {"hooks": hooks},
        check_both_complete,
    )


def _incomplete_read_extra(o: Outcome, c: Checks) -> None:
    lines = o.gate0_trace(_days(M1))
    resent = [x for x in lines if x.get("resent_after_tunnel")]
    c(
        "the cut answer was sent once more",
        any("IncompleteRead" in str(x.get("first_error")) for x in resent),
        [x.get("first_error") for x in resent],
    )


def check_incomplete_read(o: Outcome, c: Checks) -> None:
    _incomplete_read_extra(o, c)
    for model in (M1, M2):
        model_ok(o, c, model)
        ext_complete(o, c, model)
    downloaded(o, c)
    destroyed(o, c)
    no_secret(o, c)


def _watchdog_at_final_pack(run) -> None:
    """3 USD gone just before the final `pack`; the agent busy for a second, so the watchdog
    destroys and the host is gone for the download."""

    def act(run) -> None:
        run.api.call("POST", "/fake/credit", {"spend": 3.0})
        time.sleep(1.0)

    _on_call("run", _hostrun("pack", "/results.tar"), act)(run)


def check_watchdog_in_download(o: Outcome, c: Checks) -> None:
    for model in (M1, M2):
        model_ok(o, c, model)
    first = next((e for e in o.kinds("destroy_requested")), {})
    c("the first destroy is the watchdog's early stop", "early_stop" in str(first.get("reason")))
    text = "" if o.error is None else str(o.error)
    c(
        "the error names the watchdog's early stop, not the pack",
        "early_stop" in text and "watchdog" in text and "pack" not in text,
        text[:300],
    )
    c("earlier syncs are on the station", any(s["ok"] for s in o.kinds("sync")))
    destroyed(o, c)
    no_secret(o, c)


def check_sigterm_create(o: Outcome, c: Checks) -> None:
    c("deploy exited 128 + SIGTERM", o.error == 128 + signal.SIGTERM, o.error)
    c("the signal is journaled", [e["name"] for e in o.kinds("signal")] == ["SIGTERM"])
    c("create was requested", bool(o.kinds("create_requested")))
    c("create outcome unknown journaled", bool(o.kinds("create_outcome_unknown")))
    c("the instance was adopted by label", bool(o.kinds("adopted_by_label")))
    destroyed(o, c)
    c("absence verified", bool(o.kinds("absence_verified")))
    no_secret(o, c)
    left = subprocess.run(["pgrep", "-f", str(o.out)], capture_output=True, text=True).stdout
    c("no process of the attempt left", not left.strip(), left.strip().splitlines()[:3])


ROUND2_SCENARIOS = [
    Scenario(
        "blip_m1_weights_poll",
        "Обрыв 0,15 с на первом опросе весов M1",
        {"hooks": {"host": _on_call("run", _hostrun("size", M1), _blip_for(0.15))}},
        check_both_complete,
    ),
    Scenario(
        "blip_m2_weights_poll",
        "Обрыв 0,15 с на первом опросе весов M2",
        {"hooks": {"host": _on_call("run", _hostrun("size", M2), _blip_for(0.15))}},
        check_both_complete,
    ),
    Scenario(
        "blip_m2_smoke_tunnel",
        "Обрыв 0,15 с через 0,5 с после туннеля M2, сервер M2 стартует 2 с",
        {
            "vllm": {M2: {"slow_start_s": 2}},
            "models": [{}, {"health_timeout_s": 20}],
            "hooks": {
                "host": _on_call(
                    "tunnel",
                    lambda remote_port, local_port=None: local_port is None,
                    lambda run: threading.Timer(0.5, _blip_for(0.15), (run,)).start(),
                    nth=2,
                )
            },
        },
        check_both_complete,
    ),
    Scenario(
        "blip_gpu_free",
        "Обрыв 0,15 с на первом nvidia-smi (preflight)",
        {"hooks": {"host": _on_call("run", _argv0("nvidia-smi"), _blip_for(0.15))}},
        check_both_complete,
    ),
    Scenario(
        "blip_final_download",
        "Обрыв 0,15 с на финальном scp results.tar",
        {"hooks": {"host": _on_call("get", _remote("/results.tar"), _blip_for(0.15))}},
        check_both_complete,
    ),
    Scenario(
        "outage_1s_gate0",
        "Сеть пропала на 1 с внутри Gate 0 M1",
        {"hooks": _when_gate0_sent(M1, 10, lambda run: (_fire(run, "gate0"), _blip_for(1.0)(run)))},
        check_both_complete,
    ),
    Scenario(
        "incomplete_read_mid_body",
        "M1: ответ Gate 0 оборван посреди тела (IncompleteRead)",
        {"vllm": {M1: {"truncate_at": [10]}}, "hooks": {"host": lambda run: _fire(run, "vllm")}},
        check_incomplete_read,
    ),
    Scenario(
        "sigterm_during_create_put",
        "SIGTERM, пока PUT create в полёте",
        {"create_mode": "timeout", "fake": lambda fake: setattr(fake, "timeout_s", 3.0)},
        check_sigterm_create,
        expect_error="any",
        signal_when=lambda out: _journal_has(out, "create_requested"),
    ),
    Scenario(
        "watchdog_destroy_in_download",
        "Скачок расхода перед финальным pack: удаляет сторож",
        {"hooks": {"host": _watchdog_at_final_pack}},
        check_watchdog_in_download,
        watchdog=True,
        expect_error="any",
    ),
]
STEP_KIND_SCENARIOS = [_step_kind_scenario(kind) for kind in STEP_KINDS]


REAL_DOC = vast.REPO_ROOT / "docs/decisions/0028-gpu-reference-gate0.md"
REAL_ENVELOPE_TOP = {"dph_total": 1.2, "inet_down_cost": 0.0043, "inet_up_cost": 0.0043}


def real_package() -> dict:
    """The money, time and request limits of the preregistered $3 package (0028, Д.2) with
    the envelope's price tops (Д.1); the models, days and set stay the dry run's."""
    text = REAL_DOC.read_text(encoding="utf-8")
    addendum = text.split("\n## Дополнение 2026-11-23: пакет $3 и расширенный набор\n", 1)[1]
    package = json.loads(addendum.split("```json\n", 1)[1].split("\n```", 1)[0])
    keep = {
        k: package[k]
        for k in (
            "ceiling_usd", "early_stop_usd", "spend_soft_usd", "deadline_minutes",
            "soft_end_minutes", "weights_late_minutes", "disk_gb", "in_gb", "out_gb",
            "gate0_request_timeout_s", "closed_timeout_s", "review_timeout_s",
        )
    }  # fmt: skip
    envelope = dict(package["envelope"])
    envelope.update({name: {"lte": top} for name, top in REAL_ENVELOPE_TOP.items()})
    fields = ("slot_minutes", "health_timeout_s", "weights_timeout_s")
    limits = [{k: v for k, v in entry.items() if k in fields} for entry in package["models"]]
    return {"package": {**keep, "envelope": envelope}, "models": limits}


def _real_offer(fake) -> None:
    """The worst offer the envelope admits: its top prices, the floors of the host."""
    fake.offers = {
        vast_fake.OFFER_ID: {
            **vast_fake.OFFER,
            **REAL_ENVELOPE_TOP,
            "cpu_ram": 128000,
            "disk_space": 200,
            "verification": "verified",
            "reliability2": 0.995,
        }
    }
    fake.spend_per_s = REAL_ENVELOPE_TOP["dph_total"] / 3600


def _spend_reaches_soft(run) -> None:
    """The credit seen at this block's end has gone down by the soft spend exactly."""
    initial = run.opened["initial_credit"]
    current = run.api.user()["credit"]
    run.api.call(
        "POST", "/fake/credit", {"spend": run.package["spend_soft_usd"] - (initial - current)}
    )


def check_spend_tail(o: Outcome, c: Checks) -> None:
    package = json.loads((o.out / "package.json").read_text())
    c(
        "the package carries the preregistered numbers",
        (package["ceiling_usd"], package["early_stop_usd"], package["spend_soft_usd"])
        == (3.0, 2.25, 1.9)
        and package["envelope"]["dph_total"] == {"lte": 1.2},
        {k: package.get(k) for k in ("ceiling_usd", "early_stop_usd", "spend_soft_usd")},
    )
    margin_min = (package["early_stop_usd"] - package["spend_soft_usd"]) / 1.2 * 60
    tail_min = (vast.spend_tail_s() + vast.SPEND_TAIL_RESERVE_S) / 60
    c(
        "the spend margin at 1.20 USD/h covers the worst tail with its reserve",
        margin_min >= tail_min,
        {"margin_min": round(margin_min, 2), "tail_and_reserve_min": tail_min},
    )
    old = {**package, "spend_soft_usd": 2.1}
    (o.out / "package-old.json").write_text(json.dumps(old))
    try:
        vast.load_package(o.out / "package-old.json", vast_fake.synthetic_models())
    except vast.VastError as error:
        c("the old spend_soft_usd 2.10 is refused", True, str(error)[:200])
    else:
        c("the old spend_soft_usd 2.10 is refused", False)
    for model in (M1, M2):
        model_ok(o, c, model)
    ext_complete(o, c, M1)
    c("M1 ran the 717-request plan", len(o.ext_trace(M1)) == 717, len(o.ext_trace(M1)))
    ext = o.ext_trace(M2)
    c(
        "M2 stopped itself on the soft spend after E-P",
        all(
            line.get("not_sent") == "spend"
            for line in ext
            if line["block"] in ("E-R", "E-A", "E-D")
        )
        and all(line.get("sent") for line in ext if line["block"] in ("E-C", "E-P")),
    )
    first = next((e for e in o.kinds("destroy_requested")), {})
    early = [e for e in o.kinds("destroy_requested") if "early_stop" in str(e.get("reason"))]
    c(
        "the watchdog ran, never stopped early, and the first destroy was the agent's own",
        bool(o.watchdog) and first.get("reason") == "agent" and not early,
        {"watchdog": o.watchdog, "first_destroy": first.get("reason")},
    )
    stopped = next(
        (e["t"] for e in o.kinds("block_done") if e["model"] == M2 and e["block"] == "E-P"), None
    )
    done = next((e["t"] for e in o.kinds("download_verified")), None)
    c(
        "the tail from the soft spend to the verified download ran here",
        stopped is not None and done is not None,
        None if stopped is None or done is None else {"tail_s_here": round(done - stopped, 2)},
    )
    downloaded(o, c)
    destroyed(o, c)
    closed = o.kinds("closed")
    c(
        "closed by the agent's destroy",
        bool(closed) and closed[-1]["reason"] == "destroy_requested",
        closed[-1]["reason"] if closed else None,
    )


SCENARIOS = [
    Scenario("success", "Обе модели, все блоки", {}, check_success),
    Scenario(
        "m1_bad_hash",
        "M1: неверный хеш весов",
        {"hf": {REPO1: {"bad_hash": True}}},
        check_m2_after_m1("weights"),
    ),
    Scenario(
        "m1_exit_before_health",
        "M1: сервер вышел до /health",
        {"vllm": {M1: {"exit": 3, "line": "fake engine core died"}}},
        check_m2_after_m1("smoke", _exit_extra),
    ),
    Scenario(
        "m1_pilot_no_audio_tokens",
        "M1: пилот без аудиотокенов",
        {"vllm": {M1: {"audio_tokens_per_s": 0}}},
        check_m2_after_m1("smoke"),
    ),
    Scenario(
        "m1_crash_in_gate0",
        "M1: vLLM падает после 5 ответов (внутри Gate 0)",
        {"vllm": {M1: {"crash_after": 5}}},
        check_crash,
    ),
    Scenario(
        "m1_hang_in_er",
        "M1: два зависших запроса в E-R",
        {"vllm": {M1: {"hang_at": [HANG_AT, HANG_AT + 2]}}},
        check_hang,
    ),
    Scenario(
        "m2_weights_late",
        "M2: медленная загрузка весов",
        {
            "hf": {REPO2: {"rate_bytes_s": 200}},
            "package": {"soft_end_minutes": 1.5, "weights_late_minutes": 1.2},
        },
        check_weights_late,
    ),
    Scenario(
        "m1_download_stalled",
        "M1: загрузка весов остановилась",
        {"hf": {REPO1: {"stall_after_bytes": 1000}}},
        check_m2_after_m1("weights"),
    ),
    Scenario(
        "soft_end_in_m2_er",
        "Мягкий конец внутри E-R модели M2",
        {
            "vllm": {M2: {"latency_s": {"closed": 0.05}}},
            "hooks": _block_end(M2, "E-P", _move_soft_end),
        },
        check_soft_end,
    ),
    Scenario(
        "no_heartbeat",
        "Агент замер: сторож удаляет по пульсу",
        {"hooks": _block_end(M1, "E-P", _freeze)},
        check_watchdog("agent_lost"),
        watchdog=True,
        expect_error="any",
        stale_s=2.0,
    ),
    Scenario(
        "hard_deadline",
        "Без мягкого конца: дедлайн, serve гасит vLLM, сторож удаляет",
        {
            "vllm": {M1: {"latency_s": {"closed": 0.2, "review": 0.2}}},
            "package": {
                "deadline_minutes": 0.5,
                "soft_end_minutes": None,
                "weights_late_minutes": None,
            },
        },
        check_deadline,
        watchdog=True,
        expect_error="any",
    ),
    Scenario(
        "create_lost",
        "Потерян ответ create",
        {"create_mode": "drop"},
        check_create_lost,
        expect_error="CreateOutcomeUnknown",
    ),
    Scenario(
        "spend_smooth",
        "Расход по часам доходит до мягкого порога раньше раннего порога сторожа",
        {"hooks": _block_end(M1, "E-C", _spend_reached)},
        check_spend_smooth,
        watchdog=True,
    ),
    Scenario(
        "spend_jump",
        "Скачок расхода выше раннего порога: удаляет сторож",
        {"hooks": _block_end(M1, "E-P", _credit_jump)},
        check_watchdog("early_stop"),
        watchdog=True,
        expect_error="any",
    ),
    Scenario(
        "ssh_lost",
        "Потеря SSH внутри run",
        {"hooks": _block_end(M1, "E-P", _lose_now)},
        check_ssh_lost,
        expect_error="AttemptFatal",
    ),
    Scenario(
        "server_not_stopping",
        "Сервер M1 не останавливается",
        {"hooks": {"host": _block_stop}},
        check_not_stopping,
        expect_error="AttemptFatal",
    ),
    Scenario(
        "server_not_stopping_after_failure",
        "M1: сервер вышел до /health и не останавливается",
        {
            "vllm": {M1: {"exit": 3, "line": "fake engine core died"}},
            "hooks": {"host": _block_stop},
        },
        check_not_stopping_after_failure,
        expect_error="AttemptFatal",
    ),
    Scenario(
        "extended_error",
        "M1: ошибка внутри блоков после Gate 0",
        {"hooks": _block_end(M1, "E-C", _broken_block)},
        check_extended_error,
    ),
    Scenario(
        "interrupt_in_m2",
        "KeyboardInterrupt внутри run M2",
        {"hooks": _block_end(M2, "E-C", _interrupt)},
        check_interrupt,
        expect_error="KeyboardInterrupt",
    ),
    Scenario(
        "tunnel_death",
        "Туннель умер внутри прогона M1",
        {"hooks": _block_end(M1, "E-C", lambda run: run.host.kill_tunnels())},
        check_tunnel,
    ),
    Scenario(
        "gpu_not_free", "GPU не освободилась к M2", {"gpu": {"memory_used_mib": 5000}}, check_gpu
    ),
    Scenario(
        "offer_fallback",
        "Первый offer исчез, второй в конверте",
        {
            "fake": _fallback_offer,
            "package": {"offer_ids": [vast_fake.OFFER_ID, vast_fake.OFFER_ID + 1]},
        },
        check_offer,
    ),
    Scenario(
        "final_pack_blocked",
        "Финальный pack не пишется на хосте (как полный диск)",
        {"hooks": {"host": _block_pack}},
        check_pack_blocked,
        expect_error="VastError",
    ),
    Scenario(
        "compressed_time",
        "Сжатое время: слот M1, мягкий конец M2",
        {
            "vllm": {
                M1: {"latency_s": {"closed": 0.02, "review": 0.1}, "review_tokens": 160},
                M2: {"latency_s": {"closed": 0.04, "review": 0.8}, "review_tokens": 450},
            },
            "models": [{"slot_minutes": 0.05}, {}],
            "package": {
                "deadline_minutes": 1.5,
                "soft_end_minutes": 0.75,
                "weights_late_minutes": 0.1,
            },
        },
        check_compressed,
    ),
    Scenario(
        "network_blip_m1",
        "M1: сеть пропала на 0,15 с, туннель умер",
        {"hooks": _block_end(M1, "E-C", _blip)},
        check_resent(),
    ),
    Scenario(
        "tunnel_slow_reopen",
        "M1: туннель умер, новый слушает порт через 1,5 с",
        {"hooks": _block_end(M1, "E-C", _slow_reopen)},
        check_resent(),
    ),
    Scenario(
        "half_alive_tunnel",
        "M1: полуживой туннель в Gate 0 — принимает, но не возит; таймаут запроса",
        {
            "hooks": _when_gate0_sent(M1, 10, lambda run: run.host.stall_tunnels()),
            "package": {"gate0_request_timeout_s": 8.0},
        },
        check_resent(after_timeout=True),
    ),
    Scenario(
        "sigterm_mid_gate0",
        "SIGTERM процессу deploy внутри Gate 0 M1",
        {},
        check_sigterm,
        expect_error="any",
        signal_when=lambda out: _gate0_sent(_gate0_trace_path(out, M1)) >= 5,
    ),
    Scenario(
        "sigterm_in_final_path",
        "SIGTERM всей группе процессов (как systemctl stop) во время финального download",
        {"hooks": {"host": _slow_final_pack}},
        check_sigterm_late,
        expect_error="any",
        signal_when=lambda out: (out / "host-root" / FINAL_PACK_MARK).exists(),
        signal_group=True,
    ),
    Scenario(
        "truncated_gate0_day",
        "Мягкий конец внутри Gate 0 M1: день не завершён, вердикта нет",
        {"hooks": _when_gate0_sent(M1, 10, _move_soft_end_now)},
        check_truncated,
        expect_error="VastError",
    ),
    Scenario(
        "spend_tail_real_numbers",
        "Числа пакета $3 (3,00/2,25/1,90, 1,20 $/ч), план 717: мягкий порог, хвост до сторожа",
        lambda: {
            **real_package(),
            "fake": _real_offer,
            "fixture": {"utterances": 96, "pairs": 20},
            "hooks": _block_end(M2, "E-P", _spend_reaches_soft),
        },
        check_spend_tail,
        watchdog=True,
    ),
    Scenario(
        "credit_below_ceiling",
        "Кредит ниже потолка + $1,00: отказ до create",
        {"fake": lambda fake: setattr(fake, "credit", 4.99)},
        check_credit_refused,
        expect_error="VastError",
    ),
    Scenario(
        "gpu_ext_sha_mismatch",
        "Набор построен другим gpu_ext.py: отказ до журнала",
        {"manifest_code_sha256": "0" * 64},
        check_code_refused,
        expect_error="VastError",
    ),
    *ROUND2_SCENARIOS,
    *STEP_KIND_SCENARIOS,
]


class Watchdog:
    """The timer of `vast.watch_tick` the dry run does not arm, in a thread: a tick every
    0.2 s; once it destroys, the host is lost for the station, as after a real destroy."""

    def __init__(self, out: Path, holder: dict) -> None:
        self.out, self.holder = out, holder
        self.outcomes: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(0.2):
            found = sorted((self.out / vast.GPU_DIR).glob(f"*/{vast.JOURNAL_FILE}"))
            if not found:
                continue
            journal = vast.Journal(found[-1])
            opened = vast.summarize(journal.replay())["opened"]
            if opened is None:
                continue
            key = vast.read_api_key(Path(opened["key_path"]))
            try:
                outcome = vast.watch_tick(
                    journal,
                    lambda opened, key=key: vast.Api(opened["api_base"], key),
                    runner=vast._refuse_system,
                )
            except Exception as error:
                outcome = f"error: {error}"
            if not self.outcomes or self.outcomes[-1] != outcome:
                self.outcomes.append(outcome)
            if outcome in ("destroying", "closed") and self.holder.get("host") is not None:
                self.holder["host"].lose()
            if outcome == "closed":
                return

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)


def _dry(scenario: Scenario, out: Path, holder: dict) -> dict:
    """The scenario's dry run; raises what the attempt raised."""
    setup = dict(scenario.setup() if callable(scenario.setup) else scenario.setup)
    setup.setdefault("full_days", True)
    hooks = dict(setup.get("hooks") or {})
    own_host = hooks.get("host")

    def host_hook(run) -> None:
        holder["host"] = run.host
        if own_host is not None:
            own_host(run)

    hooks["host"] = host_hook
    setup["hooks"] = hooks
    config_ = config.Config(allowed_roots=(out.parent.resolve(),), data_root=out.parent.resolve())
    saved_stale = vast.HEARTBEAT_STALE_S
    if scenario.stale_s is not None:
        vast.HEARTBEAT_STALE_S = scenario.stale_s
    try:
        return vast.dry_run(out, config_, scenario=setup)
    finally:
        vast.HEARTBEAT_STALE_S = saved_stale


def _descendants(pid: int) -> list[int]:
    """Every process below `pid` by `/proc/<n>/stat` parent links, whatever its session."""
    parents = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            fields = stat.read_text().rsplit(")", 1)[1].split()
            parents[int(stat.parent.name)] = int(fields[1])
    found, frontier = [], [pid]
    while frontier:
        below = [child for child, parent in parents.items() if parent in frontier]
        found += below
        frontier = below
    return found


def _signalled(scenario: Scenario, out: Path) -> int | None:
    """The dry run in a child process (`--child`), sent SIGTERM once `signal_when(out)` holds,
    as `systemctl stop` would signal `deploy`; returns its exit code."""
    log = open(out.parent / f"{scenario.name}.child.log", "wb")  # noqa: SIM115
    child = subprocess.Popen(
        [sys.executable, "-m", "sound_scout.eval.rehearsal", "--child", scenario.name,
         "--out", str(out)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )  # fmt: skip
    ends = time.monotonic() + 300
    try:
        while child.poll() is None and time.monotonic() < ends:
            if scenario.signal_when(out):
                if scenario.signal_group:  # as systemd's KillMode=control-group: every process
                    for pid in [child.pid, *_descendants(child.pid)]:
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGTERM)
                else:
                    child.send_signal(signal.SIGTERM)
                break
            time.sleep(0.05)
        return child.wait(300)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        return None
    finally:
        log.close()


def run_scenario(scenario: Scenario, out: Path) -> dict:
    """One scenario as a dry run, then its checks; never raises."""
    holder: dict = {}
    started = time.monotonic()
    error: BaseException | int | None = None
    watchdog = Watchdog(out, holder).start() if scenario.watchdog else None
    result = None
    try:
        if scenario.signal_when is not None:
            error = _signalled(scenario, out)
        else:
            result = _dry(scenario, out, holder)
    except BaseException as caught:
        error = caught
    finally:
        if watchdog is not None:
            time.sleep(0.5)
            watchdog.stop()
    ticks = [] if watchdog is None else watchdog.outcomes
    outcome = Outcome(scenario.name, out, result, error, holder.get("host"), ticks)
    checks = Checks()
    if scenario.expect_error is None:
        checks(
            "no error", error is None, None if error is None else f"{type(error).__name__}: {error}"
        )
    elif scenario.expect_error != "any":
        checks(
            f"error {scenario.expect_error}",
            error is not None and type(error).__name__ == scenario.expect_error,
            None if error is None else f"{type(error).__name__}: {error}",
        )
    try:
        scenario.check(outcome, checks)
    except Exception:
        checks("checks ran", False, traceback.format_exc()[-1500:])
    return {
        "scenario": scenario.name,
        "title": scenario.title,
        "passed": all(item["passed"] for item in checks.items),
        "elapsed_s": round(time.monotonic() - started, 3),
        "error": _error_text(error),
        "watchdog": None if watchdog is None else watchdog.outcomes,
        "models": _timeline(outcome).get("models"),
        "checks": checks.items,
        "dir": str(out),
    }


def _error_text(error: BaseException | int | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, int):
        return f"exit code {error}"
    return f"{type(error).__name__}: {error}"[:500]


# --- unit checks ------------------------------------------------------------------------------


def unit_refusals(out: Path) -> dict:
    """R12: the set refuses a clip of pilot-1 or Gate 0, a clip over 30 s, an excluded member,
    a role word in an id and a request for a clip outside the set (B1)."""
    checks = Checks()
    found = gpu_ext.fixture(out / "set")
    manifest, entries = gpu_ext.load_set(out / "set", manifest_sha256=found["manifest_sha256"])
    clips = manifest["clips"]
    cases = {
        "a pilot-1 or Gate 0 SHA-256": (dict(forbidden_sha256={clips[0]["wav_sha256"]}), clips),
        "a clip over 30 s": ({}, [{**clips[0], "duration_s": 30.5}, *clips[1:]]),
        "an excluded RINCO member": (
            dict(forbidden_members={next(c["member"] for c in clips if c.get("member"))}),
            clips,
        ),
        "a role word in a clip id": ({}, [{**clips[0], "id": "E-silence"}, *clips[1:]]),
    }
    for name, (kwargs, changed) in cases.items():
        try:
            gpu_ext.check_clips(changed, **kwargs)
        except gpu_ext.ExtError as error:
            checks(f"refuses {name}", True, str(error))
        else:
            checks(f"refuses {name}", False)
    try:
        gpu_ext.check_plan([{**entries[0], "clip": "B1"}], clips)
    except gpu_ext.ExtError as error:
        checks("refuses B1 in an extended block", True, str(error))
    else:
        checks("refuses B1 in an extended block", False)
    try:
        gpu_ext.load_set(out / "set", manifest_sha256="0" * 64)
    except gpu_ext.ExtError as error:
        checks("refuses a manifest off its pin", True, str(error))
    else:
        checks("refuses a manifest off its pin", False)
    checks(
        "Gate 0 questions unchanged (0028 § 4)",
        gate0.questions_sha256() == GATE0_QUESTIONS_SHA256,
        gate0.questions_sha256(),
    )
    checks(
        "Q-S parses as a closed answer", gate0.answer_of({"variant": "Q-S"}, _body("Нет.")) == "нет"
    )
    return _unit("refusals", "Отказы набора и неизменные вопросы Gate 0", checks)


GATE0_QUESTIONS_SHA256 = "34871259fa7ad301b1cd5ee24182664e79c3b1b7439bb1741204038e56cced82"


def _body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def unit_analysis(out: Path) -> dict:
    """R13: `analyze` on made-up answers: always «да» gives BA 0.5, right answers give 1.0."""
    checks = Checks()
    found = gpu_ext.fixture(out / "set")
    manifest, entries = gpu_ext.load_set(out / "set", manifest_sha256=found["manifest_sha256"])
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    for name, answer in (("always_yes", lambda e: "да"), ("right", None)):
        attempt = out / name
        (attempt / gate0.RESPONSES_DIR).mkdir(parents=True)
        lines = []
        for entry in entries:
            if entry["variant"] == "A":
                text = json.dumps(vast_fake.EMPTY_REVIEW)
            elif answer is not None:
                text = answer(entry)
            elif entry["variant"] in ("P", "P-text") and entry["block"] in ("E-R", "E-D"):
                text = "да" if entry.get("target") == "marked" else "нет"
            elif entry["variant"] == "Q-S":
                text = "да" if entry["take"] == "ok" else "нет"
            else:
                text = "нет речи"
            (attempt / gate0.RESPONSES_DIR / f"{entry['n']}.json").write_text(
                json.dumps(_body(text))
            )
            lines.append({**gpu_ext._keys(entry), "sent": True, "http_status": 200})
        (attempt / gpu_ext.TRACE_FILE).write_text(
            "".join(json.dumps(line) + "\n" for line in lines)
        )
        result = gpu_ext.analyze(out / "set", attempt)
        ba = result["R"]["R1_balanced_accuracy"]
        s1 = result["S"]["S1"]["share"]
        want = (0.5, 0.0) if name == "always_yes" else (1.0, 1.0)
        checks(f"{name}: BA {want[0]}, S1 {want[1]}", (ba, s1) == want, {"ba": ba, "s1": s1})
        if name == "right":
            checks(
                "right: R2 McNemar b = c = 0 (text answers right too)",
                result["R"]["R2_mcnemar"]["b"] == 0,
            )
            checks("repeats identical", result["D"]["identical_content"]["share"] == 1.0)
    del clips
    checks("McNemar exact p(10, 0) = 0.001953", round(gpu_ext.mcnemar_exact(10, 0), 6) == 0.001953)
    checks(
        "binomial P(X ≥ 8 | 10, 0.25) = 0.000416",
        round(gpu_ext.binomial_greater(8, 10, 0.25), 6) == 0.000416,
    )
    return _unit("analysis", "Числа анализа на заданных ответах", checks)


def unit_pack(out: Path) -> dict:
    """`hostrun.py pack` while a writer appends to a log: the archive matches its sums."""
    checks = Checks()
    root = out / "root"
    root.mkdir(parents=True)
    log = root / "server.log"
    log.write_text("start\n")
    stop = threading.Event()

    def writer() -> None:
        with open(log, "a") as handle:
            for _ in range(20_000):
                if stop.is_set():
                    return
                handle.write("x" * 255 + "\n")
                handle.flush()
                time.sleep(0.0005)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        for number in range(5):
            archive = out / f"pack-{number}.tar"
            done = subprocess.run(
                [
                    sys.executable,
                    str(vast.HOST_FILES_DIR / "hostrun.py"),
                    "pack",
                    "--root",
                    str(root),
                    "--out",
                    str(archive),
                ],
                capture_output=True,
                timeout=60,
                check=False,
            )
            ok = done.returncode == 0
            if ok:
                with contextlib.suppress(vast.VastError):
                    vast.verify_archive(archive, out / f"unpacked-{number}")
                    checks(f"pack {number} of a growing log verifies", True)
                    continue
            checks(f"pack {number} of a growing log verifies", False, done.stderr.decode()[-300:])
    finally:
        stop.set()
        thread.join(5)
    with tarfile.open(out / "pack-0.tar") as tar:
        checks("SHA256SUMS in the archive", "SHA256SUMS" in tar.getnames())
    return _unit("pack", "pack при растущем логе", checks)


def _unit(name: str, title: str, checks: Checks) -> dict:
    return {
        "scenario": name,
        "title": title,
        "passed": all(item["passed"] for item in checks.items),
        "elapsed_s": None,
        "error": None,
        "checks": checks.items,
    }


UNITS = {"refusals": unit_refusals, "analysis": unit_analysis, "pack": unit_pack}


def run(out: Path, only: list[str] | None = None) -> dict:
    out = Path(out).absolute()
    out.mkdir(mode=0o700, parents=True)
    results = []
    for scenario in SCENARIOS:
        if only and scenario.name not in only:
            continue
        results.append(run_scenario(scenario, out / scenario.name))
    for name, unit in UNITS.items():
        if only and name not in only:
            continue
        try:
            results.append(unit(out / f"unit-{name}"))
        except Exception:
            results.append(
                {"scenario": name, "title": name, "passed": False, "elapsed_s": None,
                 "error": traceback.format_exc()[-1500:], "checks": []}
            )  # fmt: skip
    report = {
        "kind": "gpu_rehearsal",
        "issue": "https://github.com/renidere/sound-scout-plugin/issues/149",
        "code": code_identity(),
        "passed": all(r["passed"] for r in results) and bool(results),
        "scenarios": results,
        "cannot_prove": CANNOT_PROVE,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(markdown(report))
    return report


def code_identity() -> dict:
    """The commit the rehearsal ran on and whether the tree differed from it."""

    def git(*args: str) -> str | None:
        done = subprocess.run(
            ["git", "-C", str(vast.REPO_ROOT), *args], capture_output=True, text=True, check=False
        )
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


CANNOT_PROVE = [
    "образ vllm/vllm-openai v0.34.0, vLLM, CUDA и драйвер: хост — каталог на станции",
    "время и память настоящих моделей: задержки поддельного vLLM заданы сценарием",
    "что модели слышат: ответы поддельного vLLM заданы сценарием (мок — не доказательство)",
    "поведение Vast.ai: API, биллинг и лаг кредита — поддельные",
    "настоящий SSH и его туннель: вместо него loopback-форвардер",
    "коды выхода настоящих ssh и scp при обрыве сети: выход 255 имитирует каталог-хост",
    "настоящий полный диск: вместо него недоступный путь архива",
    "скорость и Xet-кэш настоящего `hf download`",
    "таймеры systemd сторожа: в репетиции сторож — поток с тем же watch_tick",
]


def markdown(report: dict) -> str:
    lines = [
        "# Репетиция пакета #149",
        "",
        f"Итог: **{'пройдена' if report['passed'] else 'НЕ пройдена'}**.",
        "",
        f"Коммит: `{report['code']['commit']}`"
        + (" (дерево с изменениями)" if report["code"]["dirty"] else "")
        + ".",
        "",
        "| Сценарий | Что | Итог | Проверок | Время, с |",
        "|---|---|---|---|---|",
    ]
    for r in report["scenarios"]:
        passed = sum(item["passed"] for item in r["checks"])
        lines.append(
            f"| `{r['scenario']}` | {r['title']} | {'ok' if r['passed'] else 'FAIL'} | "
            f"{passed}/{len(r['checks'])} | {'—' if r['elapsed_s'] is None else r['elapsed_s']} |"
        )
    failed = [r for r in report["scenarios"] if not r["passed"]]
    for r in failed:
        lines += ["", f"## {r['scenario']}: не прошли", ""]
        lines += [
            f"- {i['check']}: {json.dumps(i['detail'], ensure_ascii=False)[:300]}"
            for i in r["checks"]
            if not i["passed"]
        ]
        if r.get("error"):
            lines.append(f"- ошибка: `{r['error'][:300]}`")
    lines += ["", "## Чего репетиция не доказывает", ""]
    lines += [f"- {item}" for item in report["cannot_prove"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.rehearsal")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        _dry(next(s for s in SCENARIOS if s.name == args.child), args.out.absolute(), {})
        return 0
    report = run(args.out, args.only)
    for r in report["scenarios"]:
        print(f"{'ok  ' if r['passed'] else 'FAIL'} {r['scenario']}")
    print(f"report: {args.out / 'report.md'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
