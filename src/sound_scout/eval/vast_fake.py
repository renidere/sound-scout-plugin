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
- `hf download …` writes the synthetic weights of `synthetic_models`; `nvidia-smi` prints one
  fake GPU.

A dry run with these fakes proves the order and the contract of the stack, never the image,
vLLM, the model or the provider's real behaviour.
"""

import base64
import hashlib
import io
import json
import multiprocessing
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
PAGE_SIZE = 1
AUDIO_TOKENS_PER_S = 25
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
        self.show_until_running = 1
        self.destroy_lag = 0
        self.timeout_s = 3.0
        self._next_id = FIRST_INSTANCE_ID
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((HOST, 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

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
            return 200, {"offers": [dict(OFFER)] if wanted == OFFER_ID else []}
        if parts[:3] == ["api", "v0", "asks"] and method == "PUT":
            instance_id = self._next_id
            self._next_id += 1
            self.instances[instance_id] = self._instance(instance_id, (body or {}).get("label"))
            self.instances[instance_id]["payload"] = body
            if self.create_mode == "ok":
                return 200, {"success": True, "new_contract": instance_id}
            if self.create_mode == "500":
                return 500, {"error": "internal"}
            return self.create_mode
        if parts[:3] == ["api", "v0", "users"]:
            return 200, {"credit": self.credit}
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


def hf_download(argv: list[str]) -> int:
    repo = argv[1]
    revision = argv[argv.index("--revision") + 1]
    target = Path(argv[argv.index("--local-dir") + 1])
    target.mkdir(parents=True, exist_ok=True)
    for path in SYNTHETIC_FILES:
        (target / path).write_bytes(synthetic_bytes(repo, revision, path))
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


def _failure_mode(model_dir: Path) -> int | None:
    """The exit code of the `exit` mode; the `hang` mode never returns; None without a file."""
    path = model_dir.parents[1] / FAKE_MODE_FILE
    if not path.is_file():
        return None
    mode = json.loads(path.read_text())
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
    code = _failure_mode(model_dir)
    if code is not None:
        return code

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
            seconds = _audio_seconds(messages)
            prompt = _text_tokens(messages) + round(AUDIO_TOKENS_PER_S * seconds)
            json_answer = (body.get("response_format") or {}).get("type") == "json_object"
            content = json.dumps(EMPTY_REVIEW) if json_answer else "тон"
            self._reply(
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
                    "usage": {"prompt_tokens": prompt, "completion_tokens": 1},
                },
            )

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving {served} on {host}:{port}", flush=True)
    server.serve_forever()
    return 0


def nvidia_smi(argv: list[str]) -> int:
    if any(arg.startswith("--query-gpu=memory.used") for arg in argv):
        print("1024")
    else:
        print("Fake A100 SXM4, 81920 MiB, 595.91.07")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tools = {"vllm": vllm, "hf": lambda rest: hf_download(rest), "nvidia-smi": nvidia_smi}
    if not argv or argv[0] not in tools:
        print(f"usage: vast_fake {{{','.join(tools)}}} …", file=sys.stderr)
        return 2
    return tools[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
