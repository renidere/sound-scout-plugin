"""MCP stdio server `sound-scout serve` (R12, R08, R11): a thin adapter over the application layer.

Three tools, the same functions the CLI calls: `speech_capabilities` is `doctor.report`,
`speech_analyze` is `app.analyze` answered with `report.compact` (the document `analyze --json`
prints), `speech_compare` is `app.compare` answered with the comparison and its file (the document
`compare --json` prints). The descriptions and argument schemas in `TOOLS` are the ones
`docs/mcp.md` lists. Every schema closes its object (`additionalProperties: false`) and
`call` validates the arguments against it before anything runs: no argument names an allowed
root, the data root, a model, the backend or a budget, and an extra one is refused, so the
operator's `sound-scout.toml` is never overridden by a tool call (R12).

A refusal — invalid arguments, a path outside the allowed roots, a Windows path or a URL, an
input that is not audio, a busy data root, a request ID of another input, an unknown run — is a
tool result with `isError: true` and `{"error": {"code", "message"}}`, never a protocol error, so
the caller reads the reason; the codes are those the CLI prints. An unexpected error is
`internal_error` with the type only: its message may carry a path or a transcript. The
annotations say what a tool does: `speech_capabilities` reads, `speech_analyze` and
`speech_compare` write run and comparison files under the data root, so neither is read-only.

One tool call at a time: calls are serialized by a lock, and each runs in a thread so the server
keeps reading the protocol. A cancelled call (`notifications/cancelled`, or the client closing
stdin) cancels `app.analyze` through its `CancelToken` and the lock is held until the worker has
stopped. The server never asks the client for sampling. `stdio_server` points fd 1 at stderr
while it serves, so stdout carries only JSON-RPC; the SDK's OpenTelemetry middleware is removed.
Logging goes to stderr through one handler whose formatter replaces the data root, the allowed
roots, the configuration file and the home directory in every line, tracebacks included. A call
logs its tool, outcome or refusal code (with the reason of a refused path) and time; an unexpected
error logs its type only, never its message or traceback.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

import anyio
import anyio.to_thread
import jsonschema
import mcp_types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from sound_scout import __version__, app, comparison, config, doctor, ingest, paths, report, schema
from sound_scout._schema_messages import schema_message
from sound_scout.config import Config

SERVER_NAME = "sound-scout"
INSTRUCTIONS = (
    "Sound analysis of Russian speech in a local recording. Call speech_capabilities to see what "
    "is ready, speech_analyze for one recording, speech_compare for two analyzed attempts. Read "
    "every status: partial, not_assessed and unavailable are not success, and a finding is only "
    "as strong as its evidence_basis."
)
INVALID_ARGUMENTS = "invalid_arguments"
UNKNOWN_TOOL = "unknown_tool"
INTERNAL_ERROR = "internal_error"
CANCELLED = "cancelled"
PATH_DENIED = "path_denied"
PATH_HINT = (
    "path must be a local file on the host running sound-scout, inside an allowed root of the "
    "operator configuration; a Windows path, a URL or another host's path is not accepted"
)
HOME_PLACEHOLDER = "~"
CONFIG_PLACEHOLDER = "<config>"
LOGGER = logging.getLogger("sound_scout.mcp")

_IDENTIFIER = schema.SCHEMA["$defs"]["identifier_string"]


def _closed(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _run_id(role: str) -> dict:
    return {"type": "string", "minLength": 1, "description": f"run_id of the {role} attempt"}


TOOLS = (
    types.Tool(
        name="speech_capabilities",
        description=(
            "Report the state of the decoder, ASR, acoustic analysis and audio backend "
            "(ready, unavailable, stale, ...), the qualification flags validated_ru, "
            "diction_validated_ru, prosody_validated_ru and the network guard. Reads only."
        ),
        input_schema=_closed({}, []),
        annotations=types.ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="speech_analyze",
        description=(
            "Analyze one recording of Russian speech: transcript, acoustic measurements and the "
            "audio backend's observations, each with its own status. Writes a run directory "
            "under the server's data root and answers with a compact summary (statuses, "
            "coverage, up to three findings, limitations, links into the run directory). The "
            "same request_id with the same file returns the earlier result; with another file "
            "it is refused."
        ),
        input_schema=_closed(
            {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "the recording: a local path on this host, inside an "
                    "allowed root of the operator configuration",
                },
                "request_id": {
                    "type": "string",
                    "minLength": _IDENTIFIER["minLength"],
                    "maxLength": _IDENTIFIER["maxLength"],
                    "description": "the caller's request ID: a repeat with the same input "
                    "returns the same result",
                },
                "layers": {
                    "type": "array",
                    "items": {"enum": list(app.LAYERS)},
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": "layers to analyze (default: all three)",
                },
                "task": {
                    "type": "string",
                    "description": "the speaking task the recording answers",
                },
                "intent": {
                    "type": "string",
                    "description": "the speaker's intent for the recording, as the user stated "
                    "it; never text taken from the recording or its transcript",
                },
            },
            ["path", "request_id"],
        ),
        annotations=types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    types.Tool(
        name="speech_compare",
        description=(
            "Compare an initial and a control attempt, two analyzed runs, by one criterion: "
            "emphasis_on:<word>, pause_before:<word>, intelligibility_of:<word> or "
            "completeness_of_phrase. Writes compare/<compare_id>.json under the server's data "
            "root and answers with the verdict, its reasons and the file."
        ),
        input_schema=_closed(
            {
                "run_a": _run_id("initial"),
                "run_b": _run_id("control"),
                "criterion": {
                    "type": "string",
                    "minLength": 1,
                    "description": "emphasis_on:<word>, pause_before:<word>, "
                    "intelligibility_of:<word> or completeness_of_phrase",
                },
            },
            ["run_a", "run_b", "criterion"],
        ),
        annotations=types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    ),
)
_TOOLS = {tool.name: tool for tool in TOOLS}


class Refused(Exception):
    """A tool call refused with a reason code; the caller gets it as an error result."""

    def __init__(self, code: str, message: str, reason: str | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.reason = reason


def call(
    name: str,
    arguments: dict | None,
    loaded: Config,
    cancel: app.CancelToken | None = None,
) -> tuple[dict, bool]:
    """Run one tool: its answer and whether it is an error result. Never raises for a refusal."""
    started = time.monotonic()
    try:
        answer = _dispatch(name, {} if arguments is None else arguments, loaded, cancel)
    except Refused as refusal:
        why = refusal.code if refusal.reason is None else f"{refusal.code}: {refusal.reason}"
        LOGGER.info("%s refused: %s in %.2f s", name, why, time.monotonic() - started)
        return _error(refusal.code, refusal.message, name), True
    except Exception as error:
        kind = type(error).__name__
        LOGGER.error("%s failed: %s in %.2f s", name, kind, time.monotonic() - started)
        return _error(INTERNAL_ERROR, kind, name), True
    outcome = answer.get("status", "ok")
    LOGGER.info("%s: %s in %.2f s", name, outcome, time.monotonic() - started)
    return answer, False


def _dispatch(name: str, arguments: dict, loaded: Config, cancel: app.CancelToken | None) -> dict:
    tool = _TOOLS.get(name)
    if tool is None:
        raise Refused(UNKNOWN_TOOL, f"no tool {name!r}; the tools are {list(_TOOLS)}")
    errors = sorted(
        jsonschema.Draft202012Validator(tool.input_schema).iter_errors(arguments),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise Refused(INVALID_ARGUMENTS, "; ".join(_argument_error(e) for e in errors))
    if name == "speech_capabilities":
        return doctor.report(loaded)
    if name == "speech_analyze":
        return _analyze(arguments, loaded, cancel)
    return _compare(arguments, loaded)


def _argument_error(error: jsonschema.ValidationError) -> str:
    where = ".".join(str(part) for part in error.absolute_path)
    return f"{where}: {schema_message(error)}" if where else schema_message(error)


def _analyze(arguments: dict, loaded: Config, cancel: app.CancelToken | None) -> dict:
    fields = {
        "request_id": arguments["request_id"],
        "task": arguments.get("task"),
        "intent": arguments.get("intent"),
    }
    if "layers" in arguments:
        fields["layers"] = tuple(arguments["layers"])
    path = arguments["path"]
    try:
        analysis = app.analyze(app.AnalyzeRequest(path, **fields), loaded, cancel=cancel)
    except ingest.IngestError as error:
        raise Refused(error.code, paths.without_paths(error.message, path, loaded)) from error
    except tuple(kind for kind, _ in app.REFUSALS) as error:
        code = next(code for kind, code in app.REFUSALS if isinstance(error, kind))
        reason = error.reason if isinstance(error, paths.PathDenied) else None
        raise Refused(code, str(error), reason) from error
    run_dir = analysis.run.path.relative_to(loaded.data_root).as_posix()
    return report.compact(analysis.result, run_dir=run_dir, reused=analysis.reused)


def _compare(arguments: dict, loaded: Config) -> dict:
    try:
        compared = app.compare(
            arguments["run_a"], arguments["run_b"], arguments["criterion"], loaded
        )
    except comparison.CompareError as error:
        raise Refused(error.code, error.message) from error
    return {"path": compared.path.relative_to(loaded.data_root).as_posix(), **compared.result}


def _error(code: str, message: str, tool: str) -> dict:
    error = {"code": code, "message": " ".join(message.split())}
    if tool == "speech_analyze" and code == PATH_DENIED:
        error["hint"] = PATH_HINT
    return {"error": error}


def result(answer: dict, is_error: bool) -> types.CallToolResult:
    """The answer as one text block of JSON, the bytes `report.compact` measured."""
    text = json.dumps(answer, ensure_ascii=False)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], is_error=is_error
    )


# --- the server ------------------------------------------------------------------------------


def serve(loaded: Config) -> None:
    """Serve MCP over stdin/stdout until the client closes stdin."""
    configure_logging(loaded)
    anyio.run(_serve, loaded)


async def _serve(loaded: Config) -> None:
    server = build_server(loaded)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def build_server(loaded: Config) -> Server:
    """The MCP server of the three tools, without a transport; calls run one at a time."""
    lock = anyio.Lock()

    async def list_tools(ctx: object, params: object) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(TOOLS))

    async def call_tool(ctx: object, params: types.CallToolRequestParams) -> types.CallToolResult:
        async with lock:
            return result(*await _call_in_thread(params.name, params.arguments, loaded))

    server = Server(
        SERVER_NAME,
        version=__version__,
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
    server.middleware = []
    return server


async def _call_in_thread(name: str, arguments: dict | None, loaded: Config) -> tuple[dict, bool]:
    cancel = app.CancelToken()
    finished = threading.Event()
    decided = threading.Lock()
    started = False

    def work() -> tuple[dict, bool]:
        nonlocal started
        with decided:
            if cancel.cancelled:
                return _error(CANCELLED, "cancelled before it started", name), True
            started = True
        try:
            return call(name, arguments, loaded, cancel)
        finally:
            finished.set()

    try:
        return await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        with decided:
            cancel.cancel()
            running = started
        if running:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(finished.wait)
        raise


# --- logging ---------------------------------------------------------------------------------


class MaskingFormatter(logging.Formatter):
    """Formats a record, then replaces every private path in the whole text."""

    def __init__(self, replacements: dict[str, str]) -> None:
        super().__init__("sound-scout serve: %(levelname)s %(name)s: %(message)s")
        self._replacements = sorted(replacements.items(), key=lambda item: -len(item[0]))

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for private, placeholder in self._replacements:
            text = text.replace(private, placeholder)
        return text


def private_paths(loaded: Config) -> dict[str, str]:
    """The paths a log line must not show and what replaces each."""
    replacements = {str(root): paths.ALLOWED_ROOT_PLACEHOLDER for root in loaded.allowed_roots}
    replacements[str(loaded.data_root)] = paths.DATA_ROOT_PLACEHOLDER
    replacements[str(config.config_path().absolute())] = CONFIG_PLACEHOLDER
    home = Path.home()
    if home != Path(home.anchor):
        for spelling in {str(home), os.path.realpath(home)}:
            replacements[spelling] = HOME_PLACEHOLDER
    return replacements


def configure_logging(loaded: Config) -> None:
    """One stderr handler on the root logger, masking private paths; warnings go through it."""
    handler = logging.StreamHandler()
    handler.setFormatter(MaskingFormatter(private_paths(loaded)))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger("sound_scout").setLevel(logging.INFO)
    logging.captureWarnings(True)
