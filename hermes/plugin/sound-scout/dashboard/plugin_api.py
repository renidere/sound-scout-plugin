"""Backend routes of the Sound Scout Hermes package (#101, decision 0013 § 3.2, § 3.5).

Hermes mounts `router` under `/api/plugins/sound-scout/` inside the gateway process. The package
is a client of the `sound-scout` CLI: every action is `SOUND_SCOUT_BIN jobs ...` by argv with
`SOUND_SCOUT_CONFIG`, run in `asyncio.to_thread`, so budgets, roots and job logic live in one
place. `POST /jobs` takes the WAV as one JSON body `{"data_url": "data:audio/wav;base64,..."}`
(the cookie leg of Desktop `ctx.rest` carries only JSON, #168), checks its size, shape and base64
before any file exists, writes it into `<uuid4>.upload` under `jobs.staging_root` (created with
`O_EXCL|O_NOFOLLOW` in a directory opened with `O_NOFOLLOW`) and submits it. Every answer is
HTTP 200: a refusal is `{"ok": false, "error": <code>, "detail": <text>}`, because Desktop
`ctx.rest` drops the body of an HTTP error. The job card of the Desktop half (#108) reads
`GET /jobs/{job_id}/report` (`jobs report`, its larger output cap `REPORT_CLI_OUTPUT`) and acts by
`POST /jobs/{job_id}/cancel` and `POST /jobs/{job_id}/retry` on the one job of the path; their
bodies are not read. Runs on the Hermes interpreter (Python >= 3.11):
stdlib, FastAPI.
"""

import asyncio
import binascii
import contextlib
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import threading
import uuid

from fastapi import APIRouter, Request

router = APIRouter()

PLUGIN = "sound-scout"
VERSION = "0.1.0"
BIN_ENV = "SOUND_SCOUT_BIN"
CONFIG_ENV = "SOUND_SCOUT_CONFIG"
CHILD_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
CLI_TIMEOUT_S = 30
MAX_CLI_OUTPUT = 64 * 1024
REPORT_CLI_OUTPUT = 512 * 1024  # 256 KiB of report.md, Cyrillic, in JSON escapes
DATA_URL_PREFIX = "data:audio/wav;base64,"
JSON_OVERHEAD = 1024
JSON_MEDIA_TYPE = "application/json"
SUBMITTER = "hermes-plugin"
UPLOAD_SUFFIX = ".upload"
HERMES_DIR = ".hermes"
JOB_ID = re.compile(r"[0-9a-f]{32}")
CLI_REFUSAL = re.compile(r"sound-scout jobs: ([a-z_]+): (.*)")
JOB_OPTIONS = {
    "session_id": "--session-id",
    "profile": "--profile",
    "connection": "--connection",
    "layers": "--layers",
    "mode": "--mode",
    "previous_run_id": "--previous-run-id",
    "criterion": "--criterion",
}
QUERY_KEYS = ("request_id", *JOB_OPTIONS)
UNFINISHED = ("queued", "running")
CONFIG_ERROR_DETAIL = "the sound-scout configuration is invalid; run `sound-scout doctor`"

_intake_lock = threading.Lock()


class Refused(Exception):
    """A refusal answered as `{"ok": false, "error", "detail", **extra}` with HTTP 200."""

    def __init__(self, error: str, detail: str) -> None:
        super().__init__(f"{error}: {detail}")
        self.error = error
        self.detail = detail
        self.extra: dict = {}

    def answer(self) -> dict:
        return {"ok": False, "error": self.error, "detail": self.detail, **self.extra}


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


def _cli(args: list[str]) -> tuple[int, str, str]:
    """`SOUND_SCOUT_BIN jobs <args>` by argv with an allowlisted environment; blocks."""
    executable = os.environ.get(BIN_ENV, "")
    config = os.environ.get(CONFIG_ENV, "")
    if not os.path.isabs(executable) or not os.path.isabs(config):
        raise Refused("not_configured", f"{BIN_ENV} and {CONFIG_ENV} must be absolute paths")
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise Refused("not_configured", f"{BIN_ENV} is not an executable file")
    env = {name: os.environ[name] for name in CHILD_ENV if name in os.environ}
    env[CONFIG_ENV] = config
    done = subprocess.run(
        [executable, "jobs", *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=CLI_TIMEOUT_S,
        check=False,
        env=env,
    )
    return done.returncode, done.stdout, done.stderr


async def _call(
    args: list[str], *private: tuple[str, str], max_output: int = MAX_CLI_OUTPUT
) -> dict:
    """The JSON answer of one CLI call, or `Refused` with the CLI's reason code; `private` pairs
    of (absolute path, placeholder) are hidden in the detail."""
    try:
        returncode, stdout, stderr = await asyncio.to_thread(_cli, args)
    except subprocess.TimeoutExpired:
        raise Refused("cli_timeout", f"sound-scout took longer than {CLI_TIMEOUT_S} s") from None
    except OSError as error:
        raise Refused("cli_unavailable", f"cannot run sound-scout: {error.strerror}") from None
    if len(stdout) + len(stderr) > max_output:
        raise Refused("cli_failed", f"sound-scout printed more than {max_output // 1024} KiB")
    if returncode == 1:
        lines = stderr.strip().splitlines()
        refusal = CLI_REFUSAL.fullmatch(lines[-1]) if lines else None
        if refusal is None:
            raise Refused("cli_failed", "sound-scout exited 1 without a reason code")
        if refusal[1] == "config_error":  # the loader's message names the operator's paths
            raise Refused("config_error", CONFIG_ERROR_DETAIL)
        detail = refusal[2]
        config = os.environ.get(CONFIG_ENV)
        for path, placeholder in (*private, (config, "<config>")):
            if path:
                detail = detail.replace(path, placeholder)
        raise Refused(refusal[1], detail)
    if returncode != 0:
        raise Refused("cli_failed", f"sound-scout exited {returncode}")
    try:
        answer = json.loads(stdout)
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        raise Refused("cli_failed", "sound-scout printed no JSON object")
    return answer


async def _limits() -> dict:
    limits = await _call(["limits", "--json"])
    budgets = (limits.get("max_duration_seconds"), limits.get("max_input_bytes"))
    root = limits.get("staging_root")
    if not all(type(value) is int and value > 0 for value in budgets) or not (
        root is None or isinstance(root, str)
    ):
        raise Refused("cli_failed", "jobs limits answered without the budgets")
    return limits


def _open_staging(root: str) -> int:
    """The staging root opened without following a symlink: a private directory of this user,
    spelled by its real path, outside the Hermes home. Blocks."""
    home = _home()
    hermes_home = os.path.realpath(home) if home is not None else None
    if (
        not os.path.isabs(root)
        or os.path.realpath(root) != root
        or any(part.lower() == HERMES_DIR for part in root.split(os.sep))
        or (hermes_home is not None and os.path.commonpath([root, hermes_home]) == hermes_home)
    ):
        raise Refused("staging_unavailable", "the staging root is not a real path outside Hermes")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        dirfd = os.open(root, flags)
    except OSError as error:
        detail = f"cannot open the staging root: {error.strerror}"
        raise Refused("staging_unavailable", detail) from None
    info = os.fstat(dirfd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_mode & 0o777 != 0o700
        or info.st_uid != os.geteuid()
    ):
        os.close(dirfd)
        raise Refused(
            "staging_unavailable", "the staging root must be a 0700 directory of the gateway user"
        )
    return dirfd


def _check_staging(root: str) -> None:
    os.close(_open_staging(root))


def _create_upload(dirfd: int, name: str) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, 0o600, dir_fd=dirfd)
    except FileExistsError:
        raise Refused("staging_conflict", "the upload name already exists") from None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise Refused("staging_conflict", "the upload name is a symlink") from None
        detail = f"cannot create the upload: {error.strerror}"
        raise Refused("staging_unavailable", detail) from None


def _discard(dirfd: int, name: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(name, dir_fd=dirfd)


def _write_upload(fd: int, wav: bytes) -> str:
    """Write, fsync and close the upload; the SHA-256 of `wav`. Blocks."""
    try:
        try:
            view = memoryview(wav)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as error:
        detail = f"cannot write the upload: {error.strerror}"
        raise Refused("staging_unavailable", detail) from None
    return hashlib.sha256(wav).hexdigest()


def _body_cap(cap: int) -> int:
    """The largest JSON body that carries `cap` decoded bytes: base64, prefix and JSON syntax."""
    return 4 * -(-cap // 3) + len(DATA_URL_PREFIX) + JSON_OVERHEAD


def _decode(body: bytearray, cap: int) -> bytes:
    """The WAV of a `{"data_url": "data:audio/wav;base64,..."}` body. Empties `body` once it is
    parsed and decodes the base64 without a sliced copy, to keep the peak memory down. Blocks."""
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError):
        raise Refused("bad_request", "the body is not JSON") from None
    finally:
        body.clear()
    if not isinstance(parsed, dict) or list(parsed) != ["data_url"]:
        raise Refused("bad_request", 'the body must be an object with the one key "data_url"')
    data_url = parsed.pop("data_url")
    if not isinstance(data_url, str) or not data_url.startswith(DATA_URL_PREFIX):
        detail = f'"data_url" must be a string that starts with {DATA_URL_PREFIX}'
        raise Refused("bad_request", detail)
    try:
        encoded = data_url.encode("ascii")
    except UnicodeEncodeError:
        raise Refused("bad_base64", "the data URL is not valid base64") from None
    del data_url
    try:
        wav = binascii.a2b_base64(memoryview(encoded)[len(DATA_URL_PREFIX) :], strict_mode=True)
    except binascii.Error:
        raise Refused("bad_base64", "the data URL is not valid base64") from None
    if not wav:
        raise Refused("empty_upload", "the data URL holds no data")
    if len(wav) > cap:
        raise Refused("input_too_large", f"the file is larger than {cap} bytes")
    return wav


def _job_options(request: Request) -> dict[str, str]:
    items = request.query_params.multi_items()
    keys = [key for key, _ in items]
    for key, value in items:
        if key not in QUERY_KEYS or keys.count(key) > 1:
            raise Refused("invalid_request", f"unknown or repeated query parameter: {key[:64]!r}")
        if not value or not value.isprintable():
            raise Refused("invalid_request", f"{key} must be a non-empty printable string")
    return dict(items)


def _check_json(request: Request) -> None:
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type != JSON_MEDIA_TYPE:
        raise Refused("bad_request", f"the body must be {JSON_MEDIA_TYPE}")


def _content_length(request: Request) -> int | None:
    value = request.headers.get("content-length")
    if value is None:
        return None
    if not value.isdigit():
        raise Refused("bad_request", "Content-Length is not a number")
    return int(value)


@router.get("/health")
async def health():
    budgets, intake = None, {"ready": True, "error": None, "detail": None}
    try:
        limits = await _limits()
        budgets = {
            "max_duration_seconds": limits["max_duration_seconds"],
            "max_input_bytes": limits["max_input_bytes"],
        }
        if limits["staging_root"] is None:
            raise Refused("staging_not_configured", "jobs.staging_root is not set")
        await asyncio.to_thread(_check_staging, limits["staging_root"])
    except Refused as refusal:
        intake = {"ready": False, "error": refusal.error, "detail": refusal.detail}
    return {
        "ok": True,
        "plugin": PLUGIN,
        "version": VERSION,
        "pid": os.getpid(),
        "session_owner": _session_owner(_home()),
        "budgets": budgets,
        "intake": intake,
    }


@router.post("/jobs")
async def submit_job(request: Request):
    if not _intake_lock.acquire(blocking=False):
        return Refused("intake_busy", "another upload is in progress").answer()
    try:
        return await _intake(request)
    except Refused as refusal:
        return refusal.answer()
    finally:
        _intake_lock.release()


async def _intake(request: Request) -> dict:
    options = _job_options(request)
    request_id = options.pop("request_id", None) or uuid.uuid4().hex
    limits = await _limits()
    root = limits["staging_root"]
    if root is None:
        raise Refused("staging_not_configured", "jobs.staging_root is not set")
    try:
        _check_json(request)
        cap = limits["max_input_bytes"]
        body_cap = _body_cap(cap)
        length = _content_length(request)
        if length is not None and length > body_cap:
            raise Refused("input_too_large", f"the body is larger than {body_cap} bytes")
        body = await _receive(request, body_cap)
        wav = await asyncio.to_thread(_decode, body, cap)
        dirfd = await asyncio.to_thread(_open_staging, root)
        try:
            return await _stage_and_submit(dirfd, root, wav, request_id, options)
        finally:
            os.close(dirfd)
    except Refused as refusal:
        refusal.extra["request_id"] = request_id
        raise


async def _stage_and_submit(dirfd, root, wav, request_id, options) -> dict:
    name = uuid.uuid4().hex + UPLOAD_SUFFIX
    fd = await asyncio.to_thread(_create_upload, dirfd, name)
    staged = os.path.join(root, name)
    keep = False
    try:
        sha256 = await asyncio.to_thread(_write_upload, fd, wav)
        args = ["submit", "--json", f"--submitter={SUBMITTER}", f"--request-id={request_id}"]
        args += [f"{JOB_OPTIONS[key]}={value}" for key, value in options.items()]
        keep = True  # from here the job may exist: a cancelled request keeps the upload
        try:
            answer = await _call(
                [*args, "--", staged], (staged, "<upload>"), (root, "<staging root>")
            )
        except Refused as refusal:
            keep = refusal.error in ("cli_timeout", "cli_failed")
            raise
        keep = not answer.get("reused")
    finally:
        if not keep:
            await asyncio.to_thread(_discard, dirfd, name)
    return {
        "ok": True,
        "job_id": answer.get("job_id"),
        "request_id": answer.get("request_id"),
        "state": answer.get("state"),
        "reused": answer.get("reused"),
        "size_bytes": len(wav),
        "sha256": sha256,
    }


async def _receive(request: Request, body_cap: int) -> bytearray:
    """The body, read chunk by chunk; stop past `body_cap`."""
    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > body_cap:
                raise Refused("input_too_large", f"the body is larger than {body_cap} bytes")
            body += chunk
    except Refused:
        raise
    except Exception as error:  # the client went away in the middle of the body
        detail = f"the body could not be read: {type(error).__name__}"
        raise Refused("bad_request", detail) from None
    return body


def _check_job_id(job_id: str) -> None:
    if not JOB_ID.fullmatch(job_id):
        raise Refused("job_not_found", "a job id is 32 lowercase hex digits")


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    try:
        _check_job_id(job_id)
        job = await _call(["status", "--json", "--", job_id])
        result = None
        if job.get("state") not in UNFINISHED:
            try:
                result = await _call(["result", "--json", "--", job_id])
            except Refused as refusal:
                if refusal.error != "no_result":
                    raise
    except Refused as refusal:
        return refusal.answer()
    return {"ok": True, "job": job, "result": result}


@router.get("/jobs/{job_id}/report")
async def job_report(job_id: str):
    try:
        _check_job_id(job_id)
        answer = await _call(["report", "--json", "--", job_id], max_output=REPORT_CLI_OUTPUT)
    except Refused as refusal:
        return refusal.answer()
    return {"ok": True, "report": answer}


async def _act(action: str, job_id: str) -> dict:
    try:
        _check_job_id(job_id)
        answer = await _call([action, "--json", "--", job_id])
    except Refused as refusal:
        return refusal.answer()
    return {"ok": True, "job": answer}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    return await _act("cancel", job_id)


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    return await _act("retry", job_id)
