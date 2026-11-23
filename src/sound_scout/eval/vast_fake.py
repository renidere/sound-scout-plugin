"""Fakes of the GPU reference stack for tests and `vast dry-run` (#147): no network beyond
127.0.0.1, no real provider, no real model.

- `FakeVast` is the provider API on 127.0.0.1 and a free port: offer read, create (with failure
  modes: a dropped connection, HTTP 500, a timeout, all after the instance exists), show, list
  with `select_filters` and `next_token` pages, destroy, SSH key attach, account. It checks the
  bearer key and logs each request without its headers.
- `python -m sound_scout.eval.vast_fake vllm serve DIR …` is an OpenAI-compatible server in the
  form of vLLM (`/health`, `/version`, `/v1/models`, `/tokenize` with `messages`,
  `/v1/chat/completions` with `input_audio`): its audio tokens grow with the clip duration. On
  start it tries an internet `bind` and prints whether the guard denied it. A file
  `<root>/fake-vllm.json` of the host root (`/workspace`, outside `models/` and `out/`) sets a
  failure mode, since `hostrun.py serve` passes no foreign variable: `{"exit": N, "line": …}`
  prints the line and exits with N before `/health`; `{"hang": true, "child": bool, "line": …}`
  prints the line, optionally starts a `multiprocessing` child and never listens.
  The file may instead map a served model name to its own mode (`{"qwen3omni": {...}}`), and a
  mode may also slow the start (`slow_start_s`), crash the server after N completions
  (`crash_after`), never answer given completions (`hang_at`), cut the body of given completions
  in half and close the connection (`truncate_at`), delay answers
  (`latency_s: {closed, review}`), set the completion tokens of a review (`review_tokens`), the
  audio token rate (`audio_tokens_per_s`, 12.5 for `qwen3omni` and 25 otherwise by default) and
  the answers (`answers: {closed, review}`).
- `hf download …` writes the synthetic weights of `synthetic_models`; `fake-hf.json` maps a repo
  to a mode: `rate_bytes_s`, `fail` (exit code), `bad_hash`, `stall_after_bytes`.
- `nvidia-smi` prints one fake GPU; `fake-gpu.json` sets `memory_used_mib` (0 by default).
- `Forwarder` is a loopback TCP forwarder: the dry run's stand-in for the SSH tunnel, which a
  rehearsal can kill and the stack reopens on the same port.

A dry run with these fakes proves the order and the contract of the stack, never the image,
vLLM, the model or the provider's real behaviour.
"""

import base64
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import socket
import sys
import threading
import time
import urllib.parse
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
OFFER_ID = 4242
FIRST_INSTANCE_ID = 900001
SHUTDOWN_POLL_S = 0.05  # serve_forever polls for `stop` this often; the default 0.5 s waits
PAGE_SIZE = 1
AUDIO_TOKENS_PER_S = 25
QWEN3_AUDIO_TOKENS_PER_S = 12.5
MODEL_NAMES = ("qwen25omni", "qwen3omni")
VLLM_VERSION = "0.34.0"
MAX_MODEL_LEN = 8192
OFFER = {
    "id": OFFER_ID,
    "ask_contract_id": OFFER_ID,
    "gpu_name": "A100 SXM4",
    "num_gpus": 1,
    "gpu_ram": 81920,
    "compute_cap": 800,
    "cuda_max_good": 13.2,
    "cpu_arch": "amd64",
    "rentable": True,
    "dph_total": 1.2,
    "inet_down_cost": 0.0026042,
    "inet_up_cost": 0.0039063,
}
EMPTY_REVIEW = {
    "observations": [],
    "unlocalized_observations": [],
    "uncertainties": [],
    "limitations": [],
}


class FakeVast:
    """The provider API in one thread; `log` holds `{method, path, auth}` of every request."""

    def __init__(self, key: str, *, create_mode: str = "ok", redirect_to: str | None = None):
        self.key = key
        self.create_mode = create_mode
        self.redirect_to = redirect_to
        self.instances: dict[int, dict] = {}
        self.foreign: list[dict] = []
        self.log: list[dict] = []
        self.credit = 10.0
        self.offers: dict[int, dict] | None = None
        self.spend_per_s = 0.0
        self.credit_jumps: list[tuple[float, float]] = []
        self.created_at: float | None = None
        self.show_until_running = 1
        self.destroy_lag = 0
        self.timeout_s = 3.0
        self._next_id = FIRST_INSTANCE_ID
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((HOST, 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, args=(SHUTDOWN_POLL_S,), daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self._server.server_address[1]}"

    def start(self) -> "FakeVast":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def add_instance(self, label: str, **fields) -> int:
        """An instance created outside the attempt (another label, or a duplicate)."""
        with self._lock:
            instance_id = self._next_id
            self._next_id += 1
            self.instances[instance_id] = self._instance(instance_id, label, **fields)
        return instance_id

    def current_credit(self, now: float | None = None) -> float:
        """The base credit less a steady spend since the create and the jumps already due."""
        if self.created_at is None:
            return self.credit
        since = (time.time() if now is None else now) - self.created_at
        jumps = sum(usd for after, usd in self.credit_jumps if since >= after)
        return self.credit - self.spend_per_s * since - jumps

    def count(self, method: str, prefix: str = "") -> int:
        return sum(1 for e in self.log if e["method"] == method and e["path"].startswith(prefix))

    def _instance(self, instance_id: int, label: str, **fields) -> dict:
        return {
            "id": instance_id,
            "label": label,
            "actual_status": "loading",
            "shows": 0,
            "deleted": None,
            "public_ipaddr": HOST,
            "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "2222"}]},
            "dph_total": OFFER["dph_total"],
            **fields,
        }

    def _visible(self, instance: dict) -> bool:
        deleted = instance["deleted"]
        return deleted is None or deleted > 0

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _reply(self, status: int, body: object = None) -> None:
                data = json.dumps(body if body is not None else {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _route(self, method: str) -> None:
                url = urllib.parse.urlsplit(self.path)
                auth = self.headers.get("Authorization") == f"Bearer {fake.key}"
                with fake._lock:
                    fake.log.append({"method": method, "path": url.path, "auth": auth})
                if fake.redirect_to is not None:
                    self.send_response(302)
                    self.send_header("Location", fake.redirect_to + url.path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if url.path == "/api/v1/instances":
                    self.send_response(301)
                    self.send_header("Location", url.path + "/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if not auth:
                    return self._reply(401, {"error": "unauthorized"})
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null") if length else None
                query = urllib.parse.parse_qs(url.query)
                parts = [part for part in url.path.split("/") if part]
                with fake._lock:
                    outcome = fake._answer(method, parts, body, query)
                if outcome == "drop":
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                if outcome == "timeout":
                    time.sleep(fake.timeout_s)
                    return self._reply(200, {"success": True})
                self._reply(*outcome)

            def do_GET(self) -> None:
                self._route("GET")

            def do_PUT(self) -> None:
                self._route("PUT")

            def do_POST(self) -> None:
                self._route("POST")

            def do_DELETE(self) -> None:
                self._route("DELETE")

        return Handler

    def _answer(self, method, parts, body, query):
        if parts[:3] == ["api", "v0", "bundles"] and method == "POST":
            wanted = ((body or {}).get("ask_contract_id") or {}).get("eq")
            offers = self.offers if self.offers is not None else {OFFER_ID: OFFER}
            return 200, {"offers": [dict(offers[wanted])] if wanted in offers else []}
        if parts[:3] == ["api", "v0", "asks"] and method == "PUT":
            if self.created_at is None:
                self.created_at = time.time()
            instance_id = self._next_id
            self._next_id += 1
            self.instances[instance_id] = self._instance(instance_id, (body or {}).get("label"))
            self.instances[instance_id]["payload"] = body
            if self.create_mode == "ok":
                return 200, {"success": True, "new_contract": instance_id}
            if self.create_mode == "500":
                return 500, {"error": "internal"}
            return self.create_mode
        if parts == ["fake", "credit"] and method == "POST":
            self.credit -= float((body or {}).get("spend", 0))
            return 200, {"credit": self.current_credit()}
        if parts[:3] == ["api", "v0", "users"]:
            return 200, {"credit": self.current_credit()}
        if parts[:3] == ["api", "v1", "instances"] and method == "GET":
            filters = json.loads((query.get("select_filters") or ["{}"])[0])
            label = (filters.get("label") or {}).get("eq")
            found = []
            for instance in [*self.instances.values(), *self.foreign]:
                if self._visible(instance) and (label is None or instance["label"] == label):
                    found.append(self._public(instance))
                    if instance["deleted"] is not None:
                        instance["deleted"] -= 1
            start = int((query.get("next_token") or ["0"])[0])
            page = found[start : start + PAGE_SIZE]
            more = start + PAGE_SIZE < len(found)
            return 200, {"instances": page, "next_token": str(start + PAGE_SIZE) if more else None}
        if parts[:3] == ["api", "v0", "instances"] and len(parts) >= 4:
            instance = self.instances.get(int(parts[3]))
            if instance is None or not self._visible(instance):
                return 404, {"error": "no_such_instance"}
            if len(parts) == 5 and parts[4] == "ssh" and method == "POST":
                instance["ssh_key_attached"] = True
                return 200, {"success": True}
            if method == "DELETE":
                if instance["deleted"] is None:
                    instance["deleted"] = self.destroy_lag
                return 200, {"success": True}
            if method == "GET":
                instance["shows"] += 1
                if instance["shows"] > self.show_until_running:
                    instance["actual_status"] = "running"
                if instance["deleted"] is not None:
                    instance["deleted"] -= 1
                return 200, {"instances": self._public(instance)}
        return 404, {"error": "no_route"}

    @staticmethod
    def _public(instance: dict) -> dict:
        hidden = ("shows", "deleted", "payload")
        return {key: value for key, value in instance.items() if key not in hidden}


# --- synthetic weights ---------------------------------------------------------------------

SYNTHETIC_FILES = ("config.json", "model-00001-of-00001.safetensors")


def synthetic_bytes(repo: str, revision: str, path: str) -> bytes:
    return f"synthetic weights of {repo}@{revision}: {path}\n".encode() * 64


def synthetic_models() -> dict:
    """A `vast.MODELS`-shaped manifest whose files `hf download` of this fake writes."""
    models = {}
    for name, repo in (("qwen25omni", "fake/Qwen2.5-Omni-7B"), ("qwen3omni", "fake/Qwen3-Omni")):
        revision = hashlib.sha1(repo.encode()).hexdigest()
        files = []
        for path in SYNTHETIC_FILES:
            data = synthetic_bytes(repo, revision, path)
            lfs = path.endswith(".safetensors")
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            files.append(
                {
                    "path": path,
                    "size": len(data),
                    "lfs_sha256": hashlib.sha256(data).hexdigest() if lfs else None,
                    "git_oid": blob,
                }
            )
        models[name] = {"repo": repo, "revision": revision, "files": files}
    return models


FAKE_HF_FILE = "fake-hf.json"
FAKE_GPU_FILE = "fake-gpu.json"
ROOT_VARIABLE = "SS_FAKE_ROOT"


def fake_root() -> Path | None:
    """The fake host root the shims of `vast._fake_host_root` name, or None outside them."""
    root = os.environ.get(ROOT_VARIABLE)
    return Path(root) if root else None


def _root_mode(name: str) -> dict:
    root = fake_root()
    path = None if root is None else root / name
    return json.loads(path.read_text()) if path is not None and path.is_file() else {}


def hf_download(argv: list[str]) -> int:
    repo = argv[1]
    revision = argv[argv.index("--revision") + 1]
    target = Path(argv[argv.index("--local-dir") + 1])
    target.mkdir(parents=True, exist_ok=True)
    mode = _root_mode(FAKE_HF_FILE).get(repo, {})
    if "fail" in mode:
        print(f"fake hf download failure {mode['fail']}", file=sys.stderr)
        return int(mode["fail"])
    partial = target / ".cache" / "huggingface" / "download"
    partial.mkdir(parents=True, exist_ok=True)
    rate = mode.get("rate_bytes_s")
    stall = mode.get("stall_after_bytes")
    written = 0
    for path in SYNTHETIC_FILES:
        data = synthetic_bytes(repo, revision, path)
        if mode.get("bad_hash") and path.endswith(".safetensors"):
            data = data[:-1] + b"?"
        incomplete = partial / f"{path}.incomplete"
        with open(incomplete, "wb") as out:
            for start in range(0, len(data), 256):
                if stall is not None and written >= stall:
                    out.flush()
                    _sleep_forever()
                chunk = data[start : start + 256]
                out.write(chunk)
                out.flush()
                written += len(chunk)
                if rate:
                    time.sleep(len(chunk) / rate)
        os.replace(incomplete, target / path)
    print(f"downloaded {len(SYNTHETIC_FILES)} files")
    return 0


# --- the vLLM-shaped server ----------------------------------------------------------------


def _text_tokens(messages) -> int:
    count = 3
    for message in messages or []:
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for part in parts:
            if part.get("type") == "text":
                count += len(str(part.get("text", "")).split())
    return count


def _audio_seconds(messages) -> float:
    total = 0.0
    for message in messages or []:
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "input_audio":
                data = base64.b64decode(part["input_audio"]["data"])
                with wave.open(io.BytesIO(data)) as clip:
                    total += clip.getnframes() / clip.getframerate()
    return total


def _guard_probe() -> str:
    probe = socket.socket()
    try:
        probe.bind(("0.0.0.0", 0))
    except PermissionError:
        return "denied"
    finally:
        probe.close()
    return "allowed"


FAKE_MODE_FILE = "fake-vllm.json"


def _sleep_forever() -> None:
    while True:
        time.sleep(60)


def model_mode(model_dir: Path, served: str) -> dict:
    """The fake mode of this served model: its own entry, or a flat mode for every model."""
    path = model_dir.parents[1] / FAKE_MODE_FILE
    if not path.is_file():
        return {}
    mode = json.loads(path.read_text())
    keyed = bool(mode) and set(mode) <= set(MODEL_NAMES)
    return mode.get(served, {}) if keyed else mode


def _failure_mode(mode: dict) -> int | None:
    """The exit code of the `exit` mode; the `hang` mode never returns; None otherwise."""
    if "exit" not in mode and not mode.get("hang"):
        return None
    print(mode.get("line", "fake vllm failure mode"), flush=True)
    if "exit" in mode:
        return int(mode["exit"])
    if mode.get("child"):
        # fork: the shim of `vllm` has no `__main__` guard, so `spawn` would run it again
        multiprocessing.get_context("fork").Process(target=_sleep_forever).start()
    _sleep_forever()
    return None


def vllm(argv: list[str]) -> int:
    if argv[:1] == ["--version"]:
        print(VLLM_VERSION)
        return 0
    if argv[:1] != ["serve"] or len(argv) < 2:
        print("usage: vllm serve DIR --served-model-name N --port P", file=sys.stderr)
        return 2
    model_dir = Path(argv[1])
    served = argv[argv.index("--served-model-name") + 1]
    port = int(argv[argv.index("--port") + 1])
    host = argv[argv.index("--host") + 1]
    if not model_dir.is_dir():
        print(f"no model directory {model_dir.name}", file=sys.stderr)
        return 1
    print(f"guard_probe: {_guard_probe()}", flush=True)
    mode = model_mode(model_dir, served)
    code = _failure_mode(mode)
    if code is not None:
        return code
    time.sleep(float(mode.get("slow_start_s", 0)))
    default_rate = QWEN3_AUDIO_TOKENS_PER_S if served == "qwen3omni" else AUDIO_TOKENS_PER_S
    rate = float(mode.get("audio_tokens_per_s", default_rate))
    latency = mode.get("latency_s") or {}
    answers = mode.get("answers") or {}
    completions = {"count": 0}
    counter = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _reply(self, status: int, body: object = None) -> None:
            data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _cut_reply(self, status: int, body: object) -> None:
            """The headers and half the body, then the connection closes: `IncompleteRead`."""
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[: len(data) // 2])
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)

        def do_GET(self) -> None:
            if self.path == "/health":
                return self._reply(200)
            if self.path == "/version":
                return self._reply(200, {"version": VLLM_VERSION})
            if self.path == "/v1/models":
                return self._reply(200, {"object": "list", "data": [{"id": served}]})
            self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("model") != served:
                return self._reply(404, {"error": f"model {body.get('model')!r} does not exist"})
            messages = body.get("messages")
            if self.path == "/tokenize":
                count = _text_tokens(messages)
                return self._reply(
                    200,
                    {"count": count, "tokens": list(range(count)), "max_model_len": MAX_MODEL_LEN},
                )
            if self.path != "/v1/chat/completions":
                return self._reply(404, {"error": "not found"})
            with counter:
                completions["count"] += 1
                number = completions["count"]
            if "crash_after" in mode and number > int(mode["crash_after"]):
                print(f"fake engine crash after {mode['crash_after']} completions", flush=True)
                os._exit(1)
            if number in mode.get("hang_at", ()):
                time.sleep(3600)
            seconds = _audio_seconds(messages)
            prompt = _text_tokens(messages) + round(rate * seconds)
            json_answer = (body.get("response_format") or {}).get("type") == "json_object"
            if json_answer:
                content = json.dumps(answers.get("review", EMPTY_REVIEW), ensure_ascii=False)
                tokens = int(mode.get("review_tokens", 1))
            else:
                content, tokens = answers.get("closed", "тон"), 1
            tokens = min(tokens, int(body.get("max_tokens") or tokens))
            time.sleep(float(latency.get("review" if json_answer else "closed", 0)))
            reply = self._cut_reply if number in mode.get("truncate_at", ()) else self._reply
            reply(
                200,
                {
                    "id": "fake",
                    "object": "chat.completion",
                    "model": served,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": tokens},
                },
            )

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving {served} on {host}:{port}", flush=True)
    server.serve_forever()
    return 0


def nvidia_smi(argv: list[str]) -> int:
    if any(arg.startswith("--query-gpu=memory.used") for arg in argv):
        print(int(_root_mode(FAKE_GPU_FILE).get("memory_used_mib", 0)))
    else:
        print("Fake A100 SXM4, 81920 MiB, 595.91.07")
    return 0


class Forwarder:
    """127.0.0.1:`port` → 127.0.0.1:`target`, one thread per connection; `stop` closes the
    listener and every open connection, as a dead SSH tunnel would; `stall` keeps accepting and
    holding connections but carries no byte, as a half-dead one."""

    def __init__(self, target: int, port: int = 0) -> None:
        self.target = target
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((HOST, port))
        self._listener.listen(16)
        self.port = self._listener.getsockname()[1]
        self._open: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._stalled = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        return not self._stopped.is_set()

    def _accept(self) -> None:
        while not self._stopped.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            if self._stalled.is_set():
                with self._lock:
                    self._open.append(client)
                continue
            try:
                upstream = socket.create_connection((HOST, self.target), timeout=5)
            except OSError:
                client.close()
                continue
            upstream.settimeout(None)
            with self._lock:
                self._open += [client, upstream]
            for source, sink in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._copy, args=(source, sink), daemon=True).start()

    def _copy(self, source: socket.socket, sink: socket.socket) -> None:
        try:
            while data := source.recv(1 << 16):
                if not self._stalled.is_set():
                    sink.sendall(data)
        except OSError:
            pass
        finally:
            if not self._stalled.is_set():  # a stalled tunnel holds the client's end open
                for sock in (source, sink):
                    with contextlib.suppress(OSError):
                        sock.shutdown(socket.SHUT_RDWR)

    def stall(self) -> None:
        self._stalled.set()

    def stop(self) -> None:
        self._stopped.set()
        with contextlib.suppress(OSError):
            self._listener.shutdown(socket.SHUT_RDWR)  # wakes the accept, frees the port
        with contextlib.suppress(OSError):
            self._listener.close()
        self._thread.join(5)
        with self._lock:
            for sock in self._open:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                with contextlib.suppress(OSError):
                    sock.close()
            self._open.clear()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tools = {"vllm": vllm, "hf": lambda rest: hf_download(rest), "nvidia-smi": nvidia_smi}
    if not argv or argv[0] not in tools:
        print(f"usage: vast_fake {{{','.join(tools)}}} …", file=sys.stderr)
        return 2
    return tools[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
