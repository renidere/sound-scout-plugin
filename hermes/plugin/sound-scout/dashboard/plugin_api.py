"""Backend routes of the Sound Scout Hermes package (#101, decision 0013 § 3.2, § 3.5).

Hermes mounts `router` under `/api/plugins/sound-scout/` inside the gateway process. The package
is a client of the `sound-scout` CLI: every action is `SOUND_SCOUT_BIN jobs ...` by argv with
`SOUND_SCOUT_CONFIG`, run in `asyncio.to_thread`, so budgets, roots and job logic live in one
place. `POST /jobs` streams one multipart part into `<uuid4>.upload` under `jobs.staging_root`
(created with `O_EXCL|O_NOFOLLOW` in a directory opened with `O_NOFOLLOW`; the client's file name
is never used) and submits it. Every answer is HTTP 200: a refusal is
`{"ok": false, "error": <code>, "detail": <text>}`, because Desktop `ctx.rest` drops the body of
an HTTP error. Runs on the Hermes interpreter (Python >= 3.11): stdlib, FastAPI, python-multipart.
"""

import asyncio
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
from python_multipart.multipart import MultipartParser, parse_options_header

router = APIRouter()

PLUGIN = "sound-scout"
VERSION = "0.1.0"
DIRECTIVE = "sound-scout-report"
BIN_ENV = "SOUND_SCOUT_BIN"
CONFIG_ENV = "SOUND_SCOUT_CONFIG"
CHILD_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
CLI_TIMEOUT_S = 30
MAX_CLI_OUTPUT = 64 * 1024
MULTIPART_OVERHEAD = 64 * 1024
SUBMITTER = "hermes-plugin"
UPLOAD_SUFFIX = ".upload"
FILE_FIELD = b"file"
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


async def _call(args: list[str], *private: tuple[str, str]) -> dict:
    """The JSON answer of one CLI call, or `Refused` with the CLI's reason code; `private` pairs
    of (absolute path, placeholder) are hidden in the detail."""
    try:
        returncode, stdout, stderr = await asyncio.to_thread(_cli, args)
    except subprocess.TimeoutExpired:
        raise Refused("cli_timeout", f"sound-scout took longer than {CLI_TIMEOUT_S} s") from None
    except OSError as error:
        raise Refused("cli_unavailable", f"cannot run sound-scout: {error.strerror}") from None
    if len(stdout) + len(stderr) > MAX_CLI_OUTPUT:
        raise Refused("cli_failed", "sound-scout printed more than 64 KiB")
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


def _discard(sink: "_Sink", dirfd: int, name: str) -> None:
    sink.abandon()
    with contextlib.suppress(OSError):
        os.unlink(name, dir_fd=dirfd)


class _Sink:
    """Writes the data of the one `file` part of a multipart body to `fd` and hashes it. Built
    before the upload exists, so a boundary the parser refuses leaves no file; `fd` is set once
    the upload is created."""

    def __init__(self, boundary: bytes, cap: int) -> None:
        self.fd = -1
        self.cap = cap
        self.size = 0
        self.sha256 = hashlib.sha256()
        self.parts = 0
        self.ended = False
        self.closed = False
        self._field = b""
        self._value = b""
        self._headers: dict[bytes, bytes] = {}
        self.parser = MultipartParser(
            boundary,
            {
                "on_part_begin": self._part_begin,
                "on_header_field": self._header_field,
                "on_header_value": self._header_value,
                "on_header_end": self._header_end,
                "on_headers_finished": self._headers_finished,
                "on_part_data": self._part_data,
                "on_end": self._end,
            },
        )

    def feed(self, chunk: bytes) -> None:
        try:
            self.parser.write(chunk)
        except Refused:
            raise
        except OSError as error:
            detail = f"cannot write the upload: {error.strerror}"
            raise Refused("staging_unavailable", detail) from None
        except Exception:  # python_multipart raises its own parse errors on a malformed body
            raise Refused("bad_multipart", "the body is not valid multipart/form-data") from None

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                try:
                    os.fsync(self.fd)
                finally:
                    os.close(self.fd)
            except OSError as error:
                detail = f"cannot write the upload: {error.strerror}"
                raise Refused("staging_unavailable", detail) from None

    def abandon(self) -> None:
        if not self.closed:
            self.closed = True
            os.close(self.fd)

    def _part_begin(self) -> None:
        self.parts += 1
        if self.parts > 1:
            raise Refused("bad_multipart", "the body must hold exactly one part")
        self._headers = {}

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def _header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def _header_end(self) -> None:
        self._headers[self._field.lower()] = self._value
        self._field, self._value = b"", b""

    def _headers_finished(self) -> None:
        disposition, params = parse_options_header(self._headers.get(b"content-disposition", b""))
        if disposition != b"form-data" or params.get(b"name") != FILE_FIELD:
            raise Refused("bad_multipart", 'the part must be form-data named "file"')

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        self.size += end - start
        if self.size > self.cap:
            raise Refused("input_too_large", f"the file is larger than {self.cap} bytes")
        view = memoryview(data)[start:end]
        self.sha256.update(view)
        while view:
            view = view[os.write(self.fd, view) :]

    def _end(self) -> None:
        self.ended = True


def _job_options(request: Request) -> dict[str, str]:
    items = request.query_params.multi_items()
    keys = [key for key, _ in items]
    for key, value in items:
        if key not in QUERY_KEYS or keys.count(key) > 1:
            raise Refused("invalid_request", f"unknown or repeated query parameter: {key[:64]!r}")
        if not value or not value.isprintable():
            raise Refused("invalid_request", f"{key} must be a non-empty printable string")
    return dict(items)


def _boundary(request: Request) -> bytes:
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise Refused("bad_multipart", "the body must be multipart/form-data with a boundary")
    return boundary


def _content_length(request: Request) -> int | None:
    value = request.headers.get("content-length")
    if value is None:
        return None
    if not value.isdigit():
        raise Refused("bad_multipart", "Content-Length is not a number")
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
    boundary = _boundary(request)
    cap = limits["max_input_bytes"]
    body_cap = cap + MULTIPART_OVERHEAD
    length = _content_length(request)
    if length is not None and length > body_cap:
        raise Refused("input_too_large", f"the body is larger than {body_cap} bytes")
    try:
        sink = _Sink(boundary, cap)
    except Exception:  # python_multipart refuses a boundary it cannot parse with
        raise Refused("bad_multipart", "the multipart boundary is not valid") from None
    dirfd = await asyncio.to_thread(_open_staging, root)
    try:
        return await _stage_and_submit(request, dirfd, root, sink, body_cap, request_id, options)
    finally:
        os.close(dirfd)


async def _stage_and_submit(request, dirfd, root, sink, body_cap, request_id, options) -> dict:
    name = uuid.uuid4().hex + UPLOAD_SUFFIX
    sink.fd = await asyncio.to_thread(_create_upload, dirfd, name)
    staged = os.path.join(root, name)
    keep = False
    try:
        await _receive(request, sink, body_cap)
        if sink.parts == 0:
            raise Refused("bad_multipart", 'the body holds no part named "file"')
        if not sink.ended:
            raise Refused("bad_multipart", "the body ended before its closing boundary")
        if sink.size == 0:
            raise Refused("empty_upload", "the file part holds no data")
        await asyncio.to_thread(sink.close)
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
    except Refused as refusal:
        refusal.extra["request_id"] = request_id
        raise
    finally:
        if not keep:
            await asyncio.to_thread(_discard, sink, dirfd, name)
    return {
        "ok": True,
        "job_id": answer.get("job_id"),
        "request_id": answer.get("request_id"),
        "state": answer.get("state"),
        "reused": answer.get("reused"),
        "size_bytes": sink.size,
        "sha256": sink.sha256.hexdigest(),
    }


async def _receive(request: Request, sink: _Sink, body_cap: int) -> None:
    """Feed the body to `sink` chunk by chunk, off the event loop; stop past `body_cap`."""
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > body_cap:
                raise Refused("input_too_large", f"the body is larger than {body_cap} bytes")
            if chunk:
                await asyncio.to_thread(sink.feed, chunk)
    except Refused:
        raise
    except Exception as error:  # the client went away in the middle of the body
        detail = f"the body could not be read: {type(error).__name__}"
        raise Refused("bad_multipart", detail) from None


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    if not JOB_ID.fullmatch(job_id):
        return Refused("job_not_found", "a job id is 32 lowercase hex digits").answer()
    try:
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
