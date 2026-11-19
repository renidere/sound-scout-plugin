"""Spike B backend routes of the Sound Scout Hermes package (#96, decision 0015).

Hermes mounts `router` under `/api/plugins/sound-scout/` inside the gateway process once the owner
enables the package. The `/spike/*` routes are throwaway probes: #101 replaces this file.
Runs on the Hermes interpreter (Python >= 3.11), so it imports only the stdlib and FastAPI.
"""

import asyncio
import os
import subprocess
import threading
import time
import uuid

from fastapi import APIRouter

router = APIRouter()

PLUGIN = "sound-scout"
VERSION = "0.0.1"
STAGE = "spike-b"
UNIT = "sound-scout-spike-b.service"
START_ARGV = (
    "systemd-run",
    "--user",
    "--unit=sound-scout-spike-b",
    "--description=Sound Scout spike B probe",
    "--collect",
    "--quiet",
    "--",
    "/usr/bin/sleep",
    "900",
)
STATUS_ARGV = (
    "systemctl",
    "--user",
    "show",
    UNIT,
    "--property=LoadState,ActiveState,SubState,MainPID,ControlGroup,ActiveEnterTimestamp",
)
DIRECTIVE = "sound-scout-report"
MAX_DELAY_S = 120
MAX_CHAT_ID = 256
RUN_TIMEOUT_S = 15
STDERR_TAIL = 2000
DELIVER_THREAD = "sound-scout-spike-deliver"

_deliveries: dict[str, dict] = {}
_deliveries_lock = threading.Lock()


def _home():
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        return None
    return get_hermes_home()


def _session_owner(home) -> bool | None:
    if home is None:
        return None
    try:
        from gateway.session_cron import owner_for_home

        return owner_for_home(home) is not None
    except (ImportError, AttributeError):
        return None


def _deliver(home, chat_id: str, text: str) -> str | None:
    try:
        from gateway.session_cron import deliver_to_local_chat
    except (ImportError, AttributeError) as exc:
        return f"gateway.session_cron.deliver_to_local_chat unavailable: {exc!r}"
    return deliver_to_local_chat(home, chat_id, text)


def _run(argv) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return subprocess.run(
        list(argv), capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False, env=env
    )


def _refusal(error: str, detail: str) -> dict:
    return {"ok": False, "error": error, "detail": detail}


def note_text(token: str) -> str:
    return (
        f"Sound Scout · spike B: тестовая заметка {token}, доставлена из рабочего потока."
        f'\n\n::{DIRECTIVE}{{job="spike-{token}"}}'
    )


def _chat_id_problem(chat_id) -> str | None:
    if not isinstance(chat_id, str) or not chat_id:
        return "chat_id must be a non-empty string"
    if len(chat_id) > MAX_CHAT_ID:
        return f"chat_id is longer than {MAX_CHAT_ID} characters"
    if not chat_id.isprintable():
        return "chat_id contains control characters"
    return None


def _deliver_once(token: str, home, chat_id: str, text: str, delay_s: int) -> None:
    if delay_s:
        time.sleep(delay_s)
    try:
        error = _deliver(home, chat_id, text)
    except Exception as exc:  # a failed delivery is recorded, never retried
        error = repr(exc)
    with _deliveries_lock:
        _deliveries[token].update(state="delivered" if error is None else "failed", error=error)


async def _run_off_loop(argv) -> tuple[subprocess.CompletedProcess | None, dict | None]:
    try:
        return await asyncio.to_thread(_run, argv), None
    except FileNotFoundError as exc:
        return None, _refusal("systemd_run_missing", str(exc))
    except subprocess.TimeoutExpired as exc:
        return None, _refusal("timeout", str(exc))
    except OSError as exc:
        return None, _refusal("run_failed", str(exc))


@router.get("/health")
async def health():
    return {
        "ok": True,
        "plugin": PLUGIN,
        "stage": STAGE,
        "version": VERSION,
        "pid": os.getpid(),
        "session_owner": _session_owner(_home()),
    }


@router.post("/spike/deliver")
async def spike_deliver(body: dict):
    chat_id = body.get("chat_id")
    delay_s = body.get("delay_s", 0)
    problem = _chat_id_problem(chat_id)
    if problem:
        return _refusal("bad_chat_id", problem)
    if type(delay_s) is not int or not 0 <= delay_s <= MAX_DELAY_S:
        return _refusal("bad_delay", f"delay_s must be an integer from 0 to {MAX_DELAY_S}")
    token = uuid.uuid4().hex
    home = _home()
    with _deliveries_lock:
        _deliveries[token] = {"state": "pending", "error": None, "chat_id": chat_id}
    threading.Thread(
        target=_deliver_once,
        args=(token, home, chat_id, note_text(token), delay_s),
        name=DELIVER_THREAD,
        daemon=True,
    ).start()
    return {"ok": True, "token": token, "delay_s": delay_s}


@router.get("/spike/deliver/{token}")
async def spike_delivery(token: str):
    with _deliveries_lock:
        record = _deliveries.get(token)
        record = dict(record) if record else None
    if record is None:
        return _refusal("unknown_token", "no delivery with this token in this process")
    return {"ok": True, **record}


@router.post("/spike/runner")
async def spike_runner_start():
    done, refusal = await _run_off_loop(START_ARGV)
    if refusal:
        return refusal
    return {
        "ok": done.returncode == 0,
        "unit": UNIT,
        "returncode": done.returncode,
        "stderr": done.stderr[-STDERR_TAIL:],
    }


@router.get("/spike/runner")
async def spike_runner_status():
    done, refusal = await _run_off_loop(STATUS_ARGV)
    if refusal:
        return refusal
    properties = {}
    for line in done.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            properties[key] = value
    return {
        "ok": done.returncode == 0,
        "unit": UNIT,
        "returncode": done.returncode,
        "stderr": done.stderr[-STDERR_TAIL:],
        "properties": properties,
    }
