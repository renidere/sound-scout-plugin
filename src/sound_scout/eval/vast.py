"""GPU reference runs on Vast.ai: pinned image, one create, watchdog, verified download (#147).

The decision is docs/decisions/0026-gpu-stack.md (the Decision comment of #147). One module on
stdlib `urllib`, `python -m sound_scout.eval.vast <command>`; it is not part of the `sound-scout`
CLI or of MCP, and it does not install `guard.install_network_guard`, because it talks to one
external host, `API_BASE`. Recording analysis stays offline (R09).

- `deploy --package P --approval URL` runs every step: `open` (append-only ownership journal
  under `<data_root>/gpu/<attempt>/`), `arm` (two `systemd-run --user` timers that need no agent),
  `create` (one PUT, never repeated), `ready`, `preflight`, `weights`, per model `smoke` and
  `run` (the Gate 0 client on this station through an SSH tunnel; each model on a server of its
  own, its host evidence journaled and the server stopped before the next), `download`
  (SHA-256 of the archive and of every file) and `destroy` with a check that the instance is
  gone by id and by label. Any failure destroys. Before anything opens, `check_days` holds each
  model to the rule of its local run (`GATE0_RUNS`) and to the clips its package entry pins.
- `watch --journal J` and `cutoff --journal J` are the timers' ticks; `destroy --journal J`
  destroys by hand; `status --journal J` reads the journal only.
- `dry-run [--out DIR]` runs the same steps against `vast_fake` on 127.0.0.1: a fake API, a fake
  host directory instead of SSH, a fake vLLM under the real `hostrun.py serve` and guard, the
  real Gate 0 client. It proves the order and the contract of the stack, not the image, vLLM, the
  model or Vast.ai; only the paid pilot #148 measures those.

The API key is read only from `KEY_PATH` (0600, owner only) inside each process. It never goes
into argv, the environment, a URL, the journal, a log, an exception or to the host.
"""

import argparse
import base64
import contextlib
import dataclasses
import fcntl
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import secrets
import shlex
import shutil
import socket
import stat
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sound_scout import config, paths, proc
from sound_scout.eval import gate0, vast_fake

API_BASE = "https://console.vast.ai"
KEY_PATH = Path.home() / ".config/vastai/vast_api_key"
KEY_MAX_BYTES = 512
KEY_PATTERN = re.compile(r"[\x21-\x7e]{16,256}")
SECRET_NAMES = ("VAST_API_KEY", "HF_TOKEN", "Authorization")
REDACTED = "<redacted>"
HTTP_TIMEOUT_S = 60.0
MAX_RESPONSE_BYTES = 16 << 20
MAX_PAGES = 100
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

IMAGE = {
    "repository": "vllm/vllm-openai",
    "tag": "v0.34.0",
    "index_digest": "sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1",
    "amd64_digest": "sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484",
    "compressed_bytes": 10036518499,
    # config blob of the amd64 manifest (#158): `ENV CUDA_VERSION`, torch from the cu130 index
    "config_digest": "sha256:0372147e2b3daa1baa5c6aa9b90ae5de2f926e25742650f91fa24dbca9cc645a",
    "cuda_env": "13.0.2",
}
IMAGE_REF = f"{IMAGE['repository']}:{IMAGE['tag']}@{IMAGE['index_digest']}"
# from the Dockerfile and requirements/common.txt of the tag; the exact transformers of the
# image is known only on the host: preflight records it, #148 pins it in docs/dependencies.md.
# `cuda` is `torch.version.cuda` of the cu130 wheel: an SSH session may lack the image ENV (#158)
RUNTIME = {
    "vllm": "0.34.0",
    "torch": "2.14.0",
    "cuda": "13.0",
    "python": "3.12",
    "transformers": ">=5.16.1,<5.23.0",
}
TRANSFORMERS_RANGE = ((5, 16, 1), (5, 23, 0))
VLLM_PORT = 8000


def _model(repo: str, revision: str, *files: tuple) -> dict:
    """Files as (path, size, LFS SHA-256 or None, git blob SHA-1), from the public HF API."""
    return {
        "repo": repo,
        "revision": revision,
        "files": [
            {"path": path, "size": size, "lfs_sha256": lfs, "git_oid": oid}
            for path, size, lfs, oid in files
        ],
    }


# bf16 weights of the candidates, revision and files from the public HF API (not computed here);
# `spk_dict.pt` (a pickle the talker needs) is left out and never downloaded. The talker and
# code2wav weights of Qwen3-Omni share its shards: vLLM loads only the thinker (its weights
# mapper drops `talker.` and `code2wav.`), as the local GGUF run of 0023 did
MODELS = {
    "qwen25omni": _model(
        "Qwen/Qwen2.5-Omni-7B",
        "ae9e1690543ffd5c0221dc27f79834d0294cba00",
        (".gitattributes", 1570, None, "52373fe24473b1aa44333d318f578ae6bf04b49b"),
        ("LICENSE", 11343, None, "9b1dd49441cd594560125cc7d79e886d0816cd8f"),
        ("README.md", 36838, None, "9eb4a8e90deb0e208a9db6fdf1ea2e4c6dbf518f"),
        ("added_tokens.json", 579, None, "4562256351969d3b68f81fd5623ae42a9c7fbb25"),
        ("chat_template.json", 1313, None, "e696060ab7ab6e20f60e4e09b18e5da4c1830000"),
        ("config.json", 13186, None, "633e1b2b157826cce16b51dee904750f02a7ac9c"),
        ("generation_config.json", 74, None, "1c1ac9d737684b833c80a438e30fc3ba53e89bd4"),
        ("merges.txt", 1671853, None, "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
        (
            "model-00001-of-00005.safetensors",
            4985055504,
            "5edb02fd7c98803239468375cc9dc1bff492865c2aa086b78f348597021d6cbc",
            "cbe9943da0e59f3ee03af9e9ed019c534e9abea6",
        ),
        (
            "model-00002-of-00005.safetensors",
            4991496800,
            "7c99b55c6e5bc63fd4b19d4dc23cdc3ddac4b0101bb3c0958cc2b5d05c2bbafe",
            "66318604f4ce70d3579cb9bbf5cc42f53a9ee5c8",
        ),
        (
            "model-00003-of-00005.safetensors",
            4991496904,
            "ad00c3ac296300db905934ed213c4077ff49b85d30c0099270c814e2c77ec812",
            "392f06b2ba109257f80458b24c3f2339b5c96f3c",
        ),
        (
            "model-00004-of-00005.safetensors",
            4969489824,
            "152bc7d81441eaba22547d8d96c03d32dd592ce0e0e1d0e449347a4b23a532d3",
            "04a3b9dbbb7e24ba2ffde279601f7f92adfb03f4",
        ),
        (
            "model-00005-of-00005.safetensors",
            2425322160,
            "b8b18276481ba8cdf4fe2c98ac4c7a2da6e0d1c8a51850d162a391760cb2b81e",
            "be5679a915f09aa07fa1c8b506772090fd4231da",
        ),
        ("model.safetensors.index.json", 233160, None, "e8728f00dd29744bb14d7ed4f374c4f05c75b6c5"),
        ("preprocessor_config.json", 667, None, "63c79f1c43bce173ab13d94096dea54b3688436f"),
        ("special_tokens_map.json", 832, None, "8c2a71eaac0f526df324364d953503def113b7a1"),
        (
            "tokenizer.json",
            11421870,
            "8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99",
            "0a8d651380b80b1775af303bdabc6da4e6774ce3",
        ),
        ("tokenizer_config.json", 6469, None, "22c6d3668a32e4fe6dc09331bff96e7cc3a2290f"),
        ("vocab.json", 2776833, None, "4783fe10ac3adce15ac8f358ef5462739852c569"),
    ),
    "qwen3omni": _model(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "26291f793822fb6be9555850f06dfe95f2d7e695",
        (".gitattributes", 1519, None, "a6344aac8c09253b3b630fb776ae94478aa0275b"),
        ("README.md", 90295, None, "64fde95f19d9f979785606ece4e484d896cd7738"),
        ("chat_template.json", 6772, None, "339ecf8a9f1ac161a9b59f833c85e6d92146b78c"),
        ("config.json", 13671, None, "18c714a7fa2d8881e8e1441ec3759b6338c9c5ea"),
        ("generation_config.json", 146, None, "59f9fe8f7db6b35dc244fc7eb52743b31533bbb5"),
        ("merges.txt", 1671839, None, "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"),
        (
            "model-00001-of-00015.safetensors",
            4997899632,
            "f95d142b727fbd0698f0ed9478e222d4f29467df855e21862e46bd1401049065",
            "c5c3d8c6cfb49d054a0b0e879535506b9cf7d1fa",
        ),
        (
            "model-00002-of-00015.safetensors",
            4997754216,
            "320c1f31f7ddc4efd2501891590eca2c481ce4eca3db33d70ebbb637dae85764",
            "0fe6554f32d0fa0f82fb151931526c7b6b45484d",
        ),
        (
            "model-00003-of-00015.safetensors",
            4997754216,
            "306f06b05ef032f1fa51eb62e72c3432324e7a084fa0d15ed17ff856adffcf33",
            "a33370662f17348245bad8594fa5ec8a268ef5a4",
        ),
        (
            "model-00004-of-00015.safetensors",
            4997755648,
            "4cb551f433bfb4d83a5843e433e74688d71a96eeb1b62217b37d719026d559b5",
            "4ade23e6f6df55d5dda3872f22b74fcc4f192dd8",
        ),
        (
            "model-00005-of-00015.safetensors",
            4997755792,
            "8aec5e8cc0e0b179a81f690af31a1149238f4b1d18ac60e3f690020fdf7d915f",
            "90603a01a044c3281d24ccb0dc44cfc7bfd198b5",
        ),
        (
            "model-00006-of-00015.safetensors",
            4997755792,
            "8ff40a3d9e9c59a35a10257c1c5b216f472e6a102a4d6059bd31183eee03cb4c",
            "f4ee22a063afd42a98fee514738dd0672be00240",
        ),
        (
            "model-00007-of-00015.safetensors",
            4997755792,
            "f6d84cc1d47178850f9ba1fe9c01f45ee63d9a8b9975f92821ee2b4a0387860b",
            "f5ae7d4e8b0b50565d2fdbaeecc42763e9016013",
        ),
        (
            "model-00008-of-00015.safetensors",
            4997755792,
            "9f369ccdc9d2d2a55df794005b23cdbce44494a72e4a0069a150ae471cfea720",
            "da77eb0f52bf3491341a00144fb2c1a0878155d7",
        ),
        (
            "model-00009-of-00015.safetensors",
            4997755792,
            "980f754b21c4b20f36e02d184b792f5625f7c9d24b8a5c9da94287cf3cfde11b",
            "35fa26106a398b8c3579297a080ddb4605b291ff",
        ),
        (
            "model-00010-of-00015.safetensors",
            4997755792,
            "5c5a798376321d74df2eb3f235e1c8c68669f09b0c753b5698a16e681204d808",
            "e08dfb813f79e2d3a57e8dc66e01edf0806ab909",
        ),
        (
            "model-00011-of-00015.safetensors",
            4997755792,
            "6f7c051bd391c89279b43e7342ff4a243ad2ba2144f4aa62233fd2b02d72fce0",
            "ac01862171a7352821d3ddd663ac371243476b84",
        ),
        (
            "model-00012-of-00015.safetensors",
            4997755792,
            "a8b09a9367cd31ebf928e87b9eaf002492379d9d21d352b480854c2fbe6d9978",
            "494bef648dd806a5f60c9fb3dd6a2cdcb48768c6",
        ),
        (
            "model-00013-of-00015.safetensors",
            4999771808,
            "d9b7441c3a33e066d109cd6338e74ed680de9aa407d4df66eba5cb3c15c6d738",
            "4e8b39ac6e58928db12f9d81599075e93fb4e095",
        ),
        (
            "model-00014-of-00015.safetensors",
            4996618552,
            "73f09f187ff7b469a8937b39b8ac772e539e0d61311526c345558c5e6a6eb09a",
            "5c9c360dc2cc73f9123a4ec79ed26910e6836cfb",
        ),
        (
            "model-00015-of-00015.safetensors",
            553698794,
            "68880d566a50643807a4e73f5227c4f468bb31532925ed5fae78a51bdd39492a",
            "d5258d7260e6830b12e03a68aba14f2f8a3f2301",
        ),
        ("model.safetensors.index.json", 2727350, None, "037226d18d29071d0ef1ca6b2a85c43be5ce61aa"),
        ("preprocessor_config.json", 603, None, "7df5b45379044a680218cb2b7ed4d01a6633f537"),
        ("tokenizer_config.json", 7344, None, "6ff9fd60cc623bb54bbd603cbd418c97a11528d7"),
        ("vocab.json", 2776833, None, "4783fe10ac3adce15ac8f358ef5462739852c569"),
    ),
}


# the rule and the memory ceiling of each model's local run: 0024 (Qwen2.5-Omni-7B, v1, 18e9)
# and 0023 (Qwen3-Omni-30B-A3B-Instruct, v2, 26e9); a package entry keeps both
GATE0_RUNS = {
    "qwen25omni": {"rule": gate0.RULE_V1, "rss_ceiling_bytes": 18 * 10**9},
    "qwen3omni": {"rule": gate0.RULE_V2, "rss_ceiling_bytes": 26 * 10**9},
}
DISK_STEP_GB = 10


def expected_files(models: Mapping, name: str) -> dict:
    """{path: {size, sha256 | git_oid}} of one model, the form of the host's weights receipt."""
    return {
        entry["path"]: {
            "size": entry["size"],
            **(
                {"sha256": entry["lfs_sha256"]}
                if entry["lfs_sha256"]
                else {"git_oid": entry["git_oid"]}
            ),
        }
        for entry in models[name]["files"]
    }


def vllm_argv(model_dir: str, served_name: str, port: int = VLLM_PORT) -> list[str]:
    """The vLLM server of one model: local directory, thinker only, no remote code, no caches."""
    return [
        "vllm",
        "serve",
        model_dir,
        "--served-model-name",
        served_name,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "1",
        "--seed",
        "0",
        "--enforce-eager",
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        '{"audio":1,"image":0,"video":0}',
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


# the second line of defence on the host: PID 1 ends 300 s after the deadline by the host clock,
# however late the container starts (stops the GPU, not the storage; whether billing stops is
# measured by #148)
ONSTART = '(t=$(( {until_unix} - $(date +%s) )); [ "$t" -gt 0 ] && sleep "$t"; kill -TERM 1) &\n'
ONSTART_MARGIN_S = 300
LABEL_PREFIX = "sound-scout-gpu-"
GPU_DIR = "gpu"
JOURNAL_FILE = "journal.jsonl"
LOCK_FILE = "lock"
HEARTBEAT_FILE = "heartbeat"
PRIVATE_DIR = ".private"
RECEIPTS_DIR = "receipts"
HOST_DIR = "host"
SUMS_FILE = "SHA256SUMS"
HEARTBEAT_STALE_S = 300.0
LOST_CREATE_GRACE_S = 15 * 60
BAD_STATUSES = frozenset({"exited", "unknown", "offline"})
WORKSPACE = "/workspace"
HOSTRUN = f"{WORKSPACE}/ss/hostrun.py"
HOST_FILES_DIR = Path(__file__).resolve().parent / "gpu_host"
REPO_ROOT = Path(__file__).resolve().parents[3]
INTERVALS = {
    "heartbeat_s": 60.0,
    "ready_poll_s": 10.0,
    "ready_timeout_s": 1800.0,
    "ssh_timeout_s": 600.0,
    "health_timeout_s": 1200.0,
    "server_check_s": 15.0,
    "server_start_s": 60.0,
    "destroy_poll_s": 10.0,
    "destroy_timeout_s": 900.0,
    "command_timeout_s": 600.0,
    "weights_timeout_s": 3 * 3600.0,
}
DRY_INTERVALS = {
    **INTERVALS,
    "heartbeat_s": 1.0,
    "ready_poll_s": 0.05,
    "ready_timeout_s": 30.0,
    "ssh_timeout_s": 30.0,
    "health_timeout_s": 60.0,
    "server_check_s": 0.2,
    "server_start_s": 10.0,
    "destroy_poll_s": 0.05,
    "destroy_timeout_s": 30.0,
    "command_timeout_s": 300.0,
    "weights_timeout_s": 120.0,
}
PILOT_TONE_HZ = 1000.0
PILOT_AMPLITUDE = 0.1


class VastError(Exception):
    """A step of the GPU stack refused or failed; the message never holds the key."""


class KeyFileError(VastError):
    pass


class SecretLeak(VastError):
    pass


class ApiError(VastError):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status


class Redirected(ApiError):
    pass


class CreateOutcomeUnknown(VastError):
    pass


class ServerExited(VastError):
    def __init__(self, reason: str | None, returncode: int | None) -> None:
        super().__init__(
            f"the vLLM server exited before /health (reason {reason}, exit code {returncode})"
        )
        self.reason, self.returncode = reason, returncode


# --- the key -------------------------------------------------------------------------------


def read_api_key(path: Path | None = None) -> str:
    """The key from a regular file of this user, mode 0600, at most 512 bytes."""
    path = Path(KEY_PATH if path is None else path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise KeyFileError(f"API key file {path.name}: cannot open ({error.strerror})") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise KeyFileError(f"API key file {path.name}: not a regular file")
        if info.st_uid != os.getuid():
            raise KeyFileError(f"API key file {path.name}: owned by another user")
        if info.st_mode & 0o077:
            raise KeyFileError(f"API key file {path.name}: mode must be 0600")
        if info.st_size > KEY_MAX_BYTES:
            raise KeyFileError(f"API key file {path.name}: over {KEY_MAX_BYTES} bytes")
        data = os.read(fd, KEY_MAX_BYTES + 1)
    finally:
        os.close(fd)
    text = data.decode("ascii", "replace")
    text = text[:-1] if text.endswith("\n") else text
    if len(data) > KEY_MAX_BYTES or not KEY_PATTERN.fullmatch(text):
        raise KeyFileError(f"API key file {path.name}: not one ASCII token of 16-256 characters")
    return text


def redact(text: str, key: str | None) -> str:
    return text.replace(key, REDACTED) if key else text


def check_no_secret(data: object, key: str | None) -> None:
    """Refuse a payload or a file for the host that holds the key or a secret's name."""
    if isinstance(data, bytes):
        text = data.decode("utf-8", "replace")
    elif isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if key and key in text:
        raise SecretLeak("the data for the provider or the host holds the API key")
    for name in SECRET_NAMES:
        if name in text:
            raise SecretLeak(f"the data for the provider or the host names {name}")


# --- the API -------------------------------------------------------------------------------


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A 3xx becomes an `HTTPError`: the request with the key is never sent again elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def check_base(base: str) -> str:
    url = urllib.parse.urlsplit(base)
    if base.rstrip("/") == API_BASE:
        return API_BASE
    if url.scheme == "http" and url.hostname in LOOPBACK_HOSTS and not url.path.strip("/"):
        return base.rstrip("/")
    raise VastError(f"API base must be {API_BASE} or http on loopback (a fake), got {base!r}")


class Api:
    """The provider API: no proxy, no redirect, one try per call; retries are the watchdog's
    next tick, and only for reads and destroy."""

    def __init__(self, base: str, key: str, timeout: float = HTTP_TIMEOUT_S) -> None:
        self.base = check_base(base)
        self.key = key
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def call(self, method: str, path: str, body: object = None, query: Mapping | None = None):
        url = self.base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        request.add_unredirected_header("Authorization", f"Bearer {self.key}")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                status = response.status
        except urllib.error.HTTPError as error:
            text = redact(error.read(4096).decode("utf-8", "replace"), self.key)
            if 300 <= error.code < 400:
                raise Redirected(
                    error.code, f"{method} {path}: redirect {error.code} refused"
                ) from None
            raise ApiError(
                error.code, f"{method} {path}: HTTP {error.code}: {text[:300]}"
            ) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            reason = getattr(error, "reason", error)
            message = redact(f"{type(reason).__name__}: {reason}", self.key)
            raise ApiError(None, f"{method} {path}: {message}") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ApiError(status, f"{method} {path}: answer over {MAX_RESPONSE_BYTES} bytes")
        try:
            answer = json.loads(raw) if raw else {}
        except ValueError:
            raise ApiError(status, f"{method} {path}: answer is not JSON") from None
        if not isinstance(answer, dict):
            raise ApiError(status, f"{method} {path}: answer is not a JSON object")
        return answer

    def user(self) -> dict:
        return self.call("GET", "/api/v0/users/current/")

    def offer(self, offer_id: int, disk_gb: int) -> dict:
        query = {
            "ask_contract_id": {"eq": offer_id},
            "type": "ondemand",
            "allocated_storage": disk_gb,
        }
        offers = self.call("POST", "/api/v0/bundles/", {**query, "limit": 1}).get("offers")
        if not isinstance(offers, list) or len(offers) != 1 or not isinstance(offers[0], dict):
            raise VastError(f"offer {offer_id} is not available")
        if offers[0].get("ask_contract_id") != offer_id:
            raise VastError(f"offer {offer_id} is not available")
        return offers[0]

    def create(self, offer_id: int, payload: Mapping) -> dict:
        return self.call("PUT", f"/api/v0/asks/{offer_id}/", payload)

    def instances(self, label: str | None = None) -> list[dict]:
        """Every instance of the account (all pages), or those with exactly this label."""
        query: dict = {}
        if label is not None:
            query["select_filters"] = json.dumps({"label": {"eq": label}})
        found = []
        for _ in range(MAX_PAGES):
            answer = self.call("GET", "/api/v1/instances/", query=query or None)
            page = answer.get("instances")
            if not isinstance(page, list):
                raise VastError("instance list without an `instances` list")
            found += [item for item in page if isinstance(item, dict)]
            token = answer.get("next_token")
            if not token:
                break
            query["next_token"] = token
        else:
            raise VastError(f"instance list over {MAX_PAGES} pages")
        if label is None:
            return found
        return [item for item in found if item.get("label") == label]

    def show(self, instance_id: int) -> dict | None:
        try:
            answer = self.call("GET", f"/api/v0/instances/{instance_id}/")
        except ApiError as error:
            if error.status == 404:
                return None
            raise
        instance = answer.get("instances")
        if isinstance(instance, list):
            instance = instance[0] if instance else None
        return instance if isinstance(instance, dict) and instance else None

    def destroy(self, instance_id: int) -> str:
        try:
            self.call("DELETE", f"/api/v0/instances/{instance_id}/")
        except ApiError as error:
            if error.status == 404:
                return "absent"
            raise
        return "destroyed"

    def attach_ssh(self, instance_id: int, public_key: str) -> None:
        self.call("POST", f"/api/v0/instances/{instance_id}/ssh/", {"ssh_key": public_key})


# --- the ownership journal -----------------------------------------------------------------


class Journal:
    """`<attempt>/journal.jsonl`: one `os.write` of one line per event, then `fsync`."""

    def __init__(self, path: Path, key: str | None = None) -> None:
        self.path = Path(path)
        self.dir = self.path.parent
        self.key = key
        self._held = 0

    @classmethod
    def create(cls, gpu: Path, attempt_id: str, key: str | None) -> "Journal":
        directory = gpu / attempt_id
        os.mkdir(directory, 0o700)
        os.chmod(directory, 0o700)
        fd = os.open(
            directory / JOURNAL_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        os.close(fd)
        _fsync_dir(directory)
        _fsync_dir(gpu)
        return cls(directory / JOURNAL_FILE, key)

    @classmethod
    def open(cls, path: Path) -> "Journal":
        """An existing journal of this user: a regular 0600 file in a 0700 directory."""
        path = Path(path).absolute()
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError as error:
            raise VastError(f"journal {path.name}: cannot open ({error.strerror})") from None
        try:
            info = os.fstat(fd)
        finally:
            os.close(fd)
        folder = os.lstat(path.parent)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or not stat.S_ISDIR(folder.st_mode)
            or folder.st_mode & 0o077
        ):
            raise VastError(f"journal {path.name}: must be a 0600 file of this user in a 0700 dir")
        return cls(path)

    @contextlib.contextmanager
    def lock(self, blocking: bool = True):
        """`flock` of `<attempt>/lock`, shared by the agent and both timers; yields whether held."""
        if self._held:
            self._held += 1
            try:
                yield True
            finally:
                self._held -= 1
            return
        fd = os.open(
            self.dir / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            self._held = 1
            try:
                yield True
            finally:
                self._held = 0
        finally:
            os.close(fd)

    def append(self, event: str, **fields) -> dict:
        record = {"event": event, "t": round(time.time(), 3), **fields}
        line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        line = redact(line, self.key)
        with self.lock():
            fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                size = os.fstat(fd).st_size
                if size and os.pread(fd, 1, size - 1) != b"\n":
                    # a torn last line never reached fsync: drop it before the next event
                    data = os.pread(fd, size, 0)
                    os.ftruncate(fd, data.rfind(b"\n") + 1)
                os.write(fd, (line + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        return record

    def replay(self) -> list[dict]:
        """The events in order; a torn last line is skipped, a broken line before it refused."""
        lines = self.path.read_bytes().split(b"\n")
        events = []
        for number, line in enumerate(lines[:-1], 1):
            try:
                event = json.loads(line)
            except ValueError:
                raise VastError(f"journal {self.path.name}: line {number} is broken") from None
            if not isinstance(event, dict) or "event" not in event:
                raise VastError(f"journal {self.path.name}: line {number} is not an event")
            events.append(event)
        return events


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def summarize(events: list[dict]) -> dict:
    state: dict = {
        "opened": None,
        "armed": None,
        "create_requested": False,
        "unknown": False,
        "ids": [],
        "destroy_requested": set(),
        "absent": set(),
        "closed": None,
        "steps": [],
    }
    for event in events:
        kind = event["event"]
        if kind in ("opened", "armed", "closed"):
            state[kind] = event
        elif kind == "create_requested":
            state["create_requested"] = True
        elif kind == "create_outcome_unknown":
            state["unknown"] = True
        elif kind in ("create_response", "adopted_by_label"):
            if event["instance_id"] not in state["ids"]:
                state["ids"].append(event["instance_id"])
        elif kind == "destroy_requested":
            state["destroy_requested"].add(event.get("instance_id"))
        elif kind == "absence_verified":
            state["absent"].add(event["instance_id"])
        elif kind == "step":
            state["steps"].append(event)
    return state


def gpu_root(config_: config.Config) -> Path:
    target = paths.resolve_output(config_.data_root, GPU_DIR, config_)
    with contextlib.suppress(FileExistsError):
        os.mkdir(target, 0o700)
    return target


def refuse_open_attempts(gpu: Path) -> None:
    for entry in sorted(gpu.iterdir()):
        journal = entry / JOURNAL_FILE
        if journal.is_file() and summarize(Journal(journal).replay())["closed"] is None:
            raise VastError(
                f"attempt {entry.name} is not closed: reconcile it with `status`/`destroy`; "
                "a new attempt needs a new approval, never a blind retry"
            )


def new_attempt_id(now: float | None = None) -> str:
    stamp = datetime.fromtimestamp(time.time() if now is None else now, UTC)
    return f"{stamp:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


# --- money and the one create --------------------------------------------------------------


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise VastError(f"{name} must be a finite number, got {value!r}")
    if value < 0:
        raise VastError(f"{name} must not be negative, got {value!r}")
    return float(value)


def estimate_usd(offer: Mapping, hours: float, in_gb: float, out_gb: float) -> float:
    """PROPOSAL § 7: dph_total × hours + inbound GB × inet_down_cost + outbound × inet_up_cost."""
    return (
        _number(offer.get("dph_total"), "dph_total") * _number(hours, "hours")
        + _number(in_gb, "in_gb") * _number(offer.get("inet_down_cost"), "inet_down_cost")
        + _number(out_gb, "out_gb") * _number(offer.get("inet_up_cost"), "inet_up_cost")
    )


def envelope_failures(offer: Mapping, envelope: Mapping) -> list[str]:
    """Fields of the offer outside the approved envelope ({field: {eq|lte|gte|in: value}})."""
    failed = []
    for field, rule in envelope.items():
        value = offer.get(field)
        for op, bound in rule.items():
            if op == "eq":
                ok = value == bound
            elif op == "in":
                ok = value in bound
            elif op in ("lte", "gte"):
                try:
                    number = _number(value, field)
                except VastError:
                    ok = False
                else:
                    ok = number <= bound if op == "lte" else number >= bound
            else:
                raise VastError(f"envelope {field}: unknown operator {op!r}")
            if not ok:
                failed.append(field)
    return failed


def create_payload(package: Mapping, label: str, onstart_until_unix: int) -> dict:
    return {
        "client_id": "me",
        "image": IMAGE_REF,
        "disk": package["disk_gb"],
        "runtype": "ssh_direct",
        "target_state": "running",
        "cancel_unavail": True,
        "label": label,
        "onstart": ONSTART.format(until_unix=int(onstart_until_unix)),
        "env": {},
    }


def _refuse_create(state: Mapping) -> None:
    if state["armed"] is None:
        raise VastError("create refused: the watchdog timers are not armed")
    if state["create_requested"]:
        raise VastError("create refused: this attempt already sent its one create")
    if state["closed"] is not None or state["destroy_requested"]:
        raise VastError("create refused: the attempt is closed or its destroy is requested")
    if time.time() >= state["opened"]["deadline_unix"]:
        raise VastError("create refused: the deadline has passed")


def create_instance(api: Api, journal: Journal, package: Mapping, runner) -> int:
    """The attempt's one PUT. Any outcome but `success` with an integer `new_contract` is
    `create_outcome_unknown`; the attempt never sends a second PUT."""
    state = summarize(journal.replay())
    opened, armed = state["opened"], state["armed"]
    _refuse_create(state)
    if not armed.get("dry_run"):
        for unit in armed["units"]:
            if runner(["systemctl", "--user", "is-active", "--quiet", f"{unit}.timer"])[0] != 0:
                raise VastError(f"create refused: {unit}.timer is not active")
    if api.instances():
        raise VastError("create refused: the account already has instances")
    offer = api.offer(package["offer_id"], package["disk_gb"])
    failed = envelope_failures(offer, package["envelope"])
    if failed:
        raise VastError(f"create refused: the offer is outside the envelope in {sorted(failed)}")
    cost = estimate_usd(
        offer, package["deadline_minutes"] / 60, package["in_gb"], package["out_gb"]
    )
    if cost > package["ceiling_usd"]:
        raise VastError(f"create refused: estimate {cost:.2f} USD is over the ceiling")
    until = int(opened["deadline_unix"]) + ONSTART_MARGIN_S
    payload = create_payload(package, opened["label"], until)
    check_no_secret(payload, api.key)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with journal.lock():
        _refuse_create(summarize(journal.replay()))
        journal.append(
            "create_requested",
            offer_id=package["offer_id"],
            payload_sha256=digest,
            estimate_usd=round(cost, 4),
        )
        try:
            answer = api.create(package["offer_id"], payload)
        except Exception as error:
            journal.append("create_outcome_unknown", error=redact(str(error), api.key))
            raise CreateOutcomeUnknown(f"create outcome unknown: {error}") from None
        instance_id = answer.get("new_contract")
        if answer.get("success") is True and type(instance_id) is int:
            journal.append("create_response", instance_id=instance_id)
            return instance_id
        journal.append("create_outcome_unknown", error="answer without success and new_contract")
        raise CreateOutcomeUnknown("create outcome unknown: answer without new_contract")


# --- the watchdog --------------------------------------------------------------------------


def unit_name(attempt_id: str, kind: str) -> str:
    return f"{LABEL_PREFIX}{attempt_id}-{kind}"


def _calendar(unix: float) -> str:
    return f"{datetime.fromtimestamp(int(unix), UTC):%Y-%m-%d %H:%M:%S} UTC"


def watch_timer_argv(journal: Path, attempt_id: str, python: str = sys.executable) -> list[str]:
    return [
        "systemd-run",
        "--user",
        f"--unit={unit_name(attempt_id, 'watch')}",
        "--collect",
        f"--description=Sound Scout GPU watchdog {attempt_id}",
        "--on-active=60s",
        "--on-unit-active=60s",
        "--timer-property=AccuracySec=5s",
        "--",
        python,
        "-m",
        "sound_scout.eval.vast",
        "watch",
        "--journal",
        str(Path(journal).absolute()),
    ]


def cutoff_timer_argv(
    journal: Path, attempt_id: str, deadline_unix: float, python: str = sys.executable
) -> list[str]:
    """Fires at the deadline and again when the search for a lost create ends."""
    return [
        "systemd-run",
        "--user",
        f"--unit={unit_name(attempt_id, 'cutoff')}",
        "--collect",
        f"--description=Sound Scout GPU cutoff {attempt_id}",
        f"--on-calendar={_calendar(deadline_unix)}",
        f"--on-calendar={_calendar(deadline_unix + LOST_CREATE_GRACE_S)}",
        "--timer-property=AccuracySec=1s",
        "--",
        python,
        "-m",
        "sound_scout.eval.vast",
        "cutoff",
        "--journal",
        str(Path(journal).absolute()),
    ]


def system(argv: list[str]) -> tuple[int, str]:
    """A short bounded system command: (returncode, stdout)."""
    done = proc.run(argv, timeout=60, max_output_bytes=1 << 16)
    return (124 if done.timed_out else done.returncode), done.stdout.decode("utf-8", "replace")


def arm(journal: Journal, runner, *, python: str = sys.executable, dry_run: bool = False) -> None:
    """Start both timers and record them; a dry run records the argv only (`dry_run: true`)."""
    opened = summarize(journal.replay())["opened"]
    attempt_id = opened["attempt_id"]
    argvs = [
        watch_timer_argv(journal.path, attempt_id, python),
        cutoff_timer_argv(journal.path, attempt_id, opened["deadline_unix"], python),
    ]
    units = [unit_name(attempt_id, "watch"), unit_name(attempt_id, "cutoff")]
    if not dry_run:
        if ".work-task" in Path(python).absolute().parts:
            raise VastError("arm refused: the timers' python lies in a task tree that goes away")
        code, out = runner(["loginctl", "show-user", str(os.getuid()), "-p", "Linger"])
        if code != 0 or out.strip() != "Linger=yes":
            raise VastError("arm refused: Linger is not on, the timers would stop at logout")
        for argv in argvs:
            if runner(argv)[0] != 0:
                raise VastError(f"arm refused: systemd-run failed for {argv[2]}")
        for unit in units:
            if runner(["systemctl", "--user", "is-active", "--quiet", f"{unit}.timer"])[0] != 0:
                raise VastError(f"arm refused: {unit}.timer is not active")
    journal.append("armed", units=units, argv=argvs, dry_run=dry_run)


def stop_timers(state: Mapping, runner) -> None:
    armed = state["armed"]
    if armed is None or armed.get("dry_run"):
        return
    for unit in armed["units"]:
        with contextlib.suppress(Exception):
            runner(["systemctl", "--user", "stop", f"{unit}.timer"])


def heartbeat_age(directory: Path, now: float) -> float:
    try:
        return now - float((directory / HEARTBEAT_FILE).read_text())
    except (OSError, ValueError):
        return math.inf


def _close(journal: Journal, state: Mapping, runner, reason: str) -> str:
    journal.append("closed", reason=reason)
    stop_timers(state, runner)
    return "closed"


def watch_tick(
    journal: Journal,
    api_factory: Callable[[Mapping], Api],
    *,
    runner,
    now: float | None = None,
    cutoff: bool = False,
) -> str:
    """One tick of the watchdog (also the agent's own destroy). Returns `busy`, `idle`, `ok`,
    `searching`, `destroying`, `closed` or `error`; destroy by id is idempotent."""
    now = time.time() if now is None else now
    with journal.lock(blocking=False) as held:
        if not held:
            return "busy"
        state = summarize(journal.replay())
        opened = state["opened"]
        if opened is None:
            raise VastError("journal without `opened`")
        if state["closed"] is not None:
            stop_timers(state, runner)
            return "closed"
        deadline = opened["deadline_unix"]
        if cutoff:
            now = max(now, deadline)
        if not state["create_requested"]:
            if now >= deadline:
                return _close(journal, state, runner, "deadline_no_instance")
            if state["destroy_requested"]:
                return _close(journal, state, runner, "no_instance_created")
            return "idle"
        try:
            api = api_factory(opened)
            journal.key = api.key
            return _owned_tick(journal, state, api, now, runner)
        except VastError as error:
            journal.append("watch_error", error=str(error))
            return "error"


def _owned_tick(journal: Journal, state: Mapping, api: Api, now: float, runner) -> str:
    opened = state["opened"]
    label, deadline = opened["label"], opened["deadline_unix"]
    try:
        owned = api.instances(label)
    except VastError as error:
        # the known ids are still destroyed; the attempt stays open until the listing recovers
        journal.append("watch_error", error=str(error))
        owned = None
    ids = list(state["ids"])
    for instance in owned or []:
        if instance.get("id") not in ids:
            journal.append("adopted_by_label", instance_id=instance["id"])
            ids.append(instance["id"])
    reasons = []
    if now >= deadline:
        reasons.append("deadline")
    if state["unknown"]:
        reasons.append("create_outcome_unknown")
    if owned is not None and len(owned) > 1:
        reasons.append("several_instances")
    if state["destroy_requested"]:
        reasons.append("destroy_requested")
    if heartbeat_age(journal.dir, now) > HEARTBEAT_STALE_S:
        reasons.append("agent_lost")
    if not reasons and opened.get("early_stop_usd") is not None:
        credit = _number(api.user().get("credit"), "credit")
        if opened["initial_credit"] - credit >= opened["early_stop_usd"]:
            reasons.append("early_stop")
    if not reasons:
        return "ok" if owned is not None else "error"
    if not ids:
        if owned is None:
            return "error"
        if now >= deadline + LOST_CREATE_GRACE_S:
            return _close(journal, state, runner, "no_instance_found")
        return "searching"
    reason = ",".join(reasons)
    for instance_id in ids:
        if instance_id in state["absent"]:
            continue
        if instance_id not in state["destroy_requested"]:
            journal.append("destroy_requested", instance_id=instance_id, reason=reason)
        status = api.destroy(instance_id)
        journal.append("destroy_response", instance_id=instance_id, status=status)
    if owned is None:
        return "destroying"
    listed = {item.get("id") for item in api.instances(label)}
    remaining = []
    for instance_id in ids:
        if instance_id in state["absent"]:
            continue
        if instance_id in listed or api.show(instance_id) is not None:
            remaining.append(instance_id)
        else:
            journal.append("absence_verified", instance_id=instance_id, label=label)
    if remaining or listed:
        return "destroying"
    return _close(journal, state, runner, reason)


def _api_from_journal(opened: Mapping) -> Api:
    return Api(opened["api_base"], read_api_key(Path(opened["key_path"])))


def destroy_until_absent(journal: Journal, api_factory, runner, intervals: Mapping) -> str:
    """Ask for destroy and tick until the attempt is closed or the time is up (the timers go on)."""
    state = summarize(journal.replay())
    if state["closed"] is not None:
        return "closed"
    for instance_id in state["ids"] or [None]:
        journal.append("destroy_requested", instance_id=instance_id, reason="agent")
    ends = time.monotonic() + intervals["destroy_timeout_s"]
    while True:
        outcome = watch_tick(journal, api_factory, runner=runner)
        if outcome == "closed":
            return outcome
        if time.monotonic() > ends:
            raise VastError(f"destroy not verified ({outcome}); the watchdog timers keep trying")
        time.sleep(intervals["destroy_poll_s"])


# --- hosts ---------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ssh_endpoint(instance: Mapping) -> tuple[str, int]:
    try:
        address = str(ipaddress.ip_address(instance["public_ipaddr"]))
        port = int(instance["ports"]["22/tcp"][0]["HostPort"])
    except (KeyError, IndexError, TypeError, ValueError):
        raise VastError("the instance has no public SSH port 22/tcp") from None
    return address, port


class RemoteHost:
    """The rented host over SSH with the attempt's own key and known_hosts, no user config."""

    def __init__(self, attempt: Path, address: str, port: int) -> None:
        self.attempt = attempt
        self.address, self.port = address, port
        self.server_port = VLLM_PORT
        self._tunnels: list = []
        private = attempt / PRIVATE_DIR
        self._options = [
            "-F",
            "/dev/null",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"IdentityFile={private / 'id_ed25519'}",
            "-o",
            f"UserKnownHostsFile={private / 'known_hosts'}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
        ]

    def ssh_argv(self, *extra: str) -> list[str]:
        return ["ssh", "-a", "-x", *self._options, "-p", str(self.port), *extra]

    def _dest(self) -> str:
        return f"root@{self.address}"

    def run(self, argv: list[str], timeout: float) -> proc.RunResult:
        return proc.run(
            [*self.ssh_argv(), self._dest(), shlex.join(argv)],
            timeout=timeout,
            max_output_bytes=1 << 20,
        )

    def spawn(self, argv: list[str], log: str) -> None:
        line = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        done = self.run(["sh", "-c", line], timeout=60)
        if done.returncode != 0:
            raise VastError(f"host: cannot start {argv[0]}")

    def put(self, local: Path, remote: str, key: str) -> None:
        check_no_secret(local.read_bytes(), key)
        target = f"{self._dest()}:{remote}"
        done = proc.run(
            ["scp", *self._options, "-P", str(self.port), str(local), target],
            timeout=600,
            max_output_bytes=1 << 16,
        )
        if done.returncode != 0:
            raise VastError(f"scp to the host failed for {Path(remote).name}")

    def get(self, remote: str, local: Path) -> None:
        source = f"{self._dest()}:{remote}"
        done = proc.run(
            ["scp", *self._options, "-P", str(self.port), source, str(local)],
            timeout=1800,
            max_output_bytes=1 << 16,
        )
        if done.returncode != 0:
            raise VastError(f"scp from the host failed for {Path(remote).name}")

    def tunnel(self, remote_port: int) -> int:
        local_port = _free_port()
        forward = f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"
        log = open(self.attempt / f"tunnel-{local_port}.log", "wb")  # noqa: SIM115
        child, copier = proc.start(
            [*self.ssh_argv("-N", "-L", forward), self._dest()], log=log, max_log_bytes=1 << 20
        )
        self._tunnels.append((child, copier, log))
        return local_port

    def close(self) -> None:
        for child, copier, log in self._tunnels:
            gate0.stop_server(child)
            copier.join(5)
            log.close()
        self._tunnels.clear()


class LocalHost(RemoteHost):
    """A directory standing in for the host in a dry run: the same host argv without SSH,
    `/workspace` mapped to `root`, the fake tools first on `PATH`."""

    def __init__(self, root: Path, env: Mapping[str, str]) -> None:
        self.root = root
        self.env = dict(env)
        self.server_port = _free_port()
        self._spawned: list = []

    def _map(self, text: str) -> str:
        return text.replace(WORKSPACE, str(self.root))

    def _argv(self, argv: list[str]) -> list[str]:
        return ["env", *(f"{k}={v}" for k, v in self.env.items()), *map(self._map, argv)]

    def run(self, argv: list[str], timeout: float) -> proc.RunResult:
        return proc.run(self._argv(argv), timeout=timeout, max_output_bytes=1 << 20)

    def spawn(self, argv: list[str], log: str) -> None:
        handle = open(self._map(log), "wb")  # noqa: SIM115
        child, copier = proc.start(self._argv(argv), log=handle, max_log_bytes=1 << 20)
        self._spawned.append((child, copier, handle))

    def put(self, local: Path, remote: str, key: str) -> None:
        check_no_secret(local.read_bytes(), key)
        shutil.copyfile(local, self._map(remote))

    def get(self, remote: str, local: Path) -> None:
        shutil.copyfile(self._map(remote), local)

    def tunnel(self, remote_port: int) -> int:
        return remote_port

    def close(self) -> None:
        for child, copier, handle in self._spawned:
            gate0.stop_server(child)
            copier.join(5)
            handle.close()
        self._spawned.clear()


# --- the steps -----------------------------------------------------------------------------

PACKAGE_NUMBERS = ("ceiling_usd", "early_stop_usd", "deadline_minutes", "in_gb", "out_gb")


def load_package(path: Path, models: Mapping) -> tuple[dict, str]:
    """The approved package (JSON outside git) and its SHA-256; refuses a malformed one."""
    data = Path(path).read_bytes()
    try:
        package = json.loads(data)
    except ValueError:
        raise VastError(f"package {Path(path).name} is not JSON") from None
    if not isinstance(package, dict):
        raise VastError("package must be a JSON object")
    for name in PACKAGE_NUMBERS:
        _number(package.get(name), name)
    for name in ("offer_id", "disk_gb"):
        if type(package.get(name)) is not int or package[name] <= 0:
            raise VastError(f"package {name} must be a positive integer")
    if package["early_stop_usd"] > package["ceiling_usd"] or package["ceiling_usd"] <= 0:
        raise VastError("package needs 0 < early_stop_usd <= ceiling_usd")
    envelope = package.get("envelope")
    if not isinstance(envelope, dict) or "dph_total" not in envelope:
        raise VastError("package envelope must bound at least dph_total")
    if not isinstance(package.get("preregistration"), str):
        raise VastError("package needs the preregistration decision document")
    entries = package.get("models")
    if not isinstance(entries, list) or not entries:
        raise VastError("package needs a non-empty list of models")
    known = sorted(set(models) & set(GATE0_RUNS))
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("name") not in known:
            raise VastError(f"package model must be one of {known}")
        if not isinstance(entry.get("gate0_date"), str) or not isinstance(
            entry.get("decision"), str
        ):
            raise VastError("package model needs gate0_date and decision")
        if type(entry.get("rss_limit_bytes")) is not int or entry["rss_limit_bytes"] <= 0:
            raise VastError("package model needs a positive rss_limit_bytes")
        ceiling = GATE0_RUNS[entry["name"]]["rss_ceiling_bytes"]
        if entry["rss_limit_bytes"] > ceiling:
            raise VastError(f"package rss_limit_bytes of {entry['name']} is over its {ceiling}")
        clips = entry.get("clips")
        if clips is not None and (
            not isinstance(clips, dict)
            or not clips
            or not all(
                isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in clips.values()
            )
        ):
            raise VastError("package model clips must map clip ids to SHA-256 of their WAV")
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        raise VastError("package lists a model twice")
    margin = package.setdefault("disk_margin_bytes", 10 * 10**9)
    if type(margin) is not int or margin < 0:
        raise VastError("package disk_margin_bytes must be a non-negative integer")
    need_gb = min_disk_gb(package, models)
    if package["disk_gb"] < need_gb:
        raise VastError(f"package disk_gb {package['disk_gb']} is below the {need_gb} GB it needs")
    return package, hashlib.sha256(data).hexdigest()


def min_disk_gb(package: Mapping, models: Mapping) -> int:
    """The weights of every model plus the margin, rounded up to whole `DISK_STEP_GB`."""
    need = package["disk_margin_bytes"] + sum(
        entry["size"] for model in package["models"] for entry in models[model["name"]]["files"]
    )
    return math.ceil(need / (DISK_STEP_GB * 10**9)) * DISK_STEP_GB


def check_days(package: Mapping, config_: config.Config) -> None:
    """Before anything is opened: each model's Gate 0 day is prepared under the rule of its local
    run, and a day the package pins by `clips` is exactly those clips, by id and by the SHA-256
    of each WAV on disk (never controls only)."""
    for entry in package["models"]:
        name, day = entry["name"], entry["gate0_date"]
        manifest_path = gate0.output_dir(day, config_) / gate0.MANIFEST_FILE
        if not manifest_path.is_file():
            raise VastError(f"Gate 0 day {day} is not prepared")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            rule = gate0.rule_sha256(Path(entry["decision"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, gate0.Gate0Error) as error:
            raise VastError(f"decision of {name}: {error}") from None
        want = GATE0_RUNS[name]["rule"]
        if gate0.rule_name(rule) != want:
            raise VastError(f"{name} runs under {want}, its decision holds {gate0.rule_name(rule)}")
        if manifest.get("rule_sha256") != rule:
            raise VastError(f"Gate 0 day {day} was prepared under another rule than {name}'s")
        clips = entry.get("clips")
        if clips is None:
            continue
        found = {
            clip["id"]: gate0._sha256_file(manifest_path.parent / clip["file"])
            for clip in manifest["clips"]
        }
        if manifest.get("controls_only") or found != clips:
            raise VastError(f"Gate 0 day {day} is not the clip set the package pins for {name}")


@contextlib.contextmanager
def _step(journal: Journal, name: str, **fields):
    started = time.time()
    detail: dict = {}
    try:
        yield detail
    except BaseException as error:
        detail["error"] = redact(f"{type(error).__name__}: {error}", journal.key)
        _record_step(journal, name, started, False, detail, fields)
        raise
    _record_step(journal, name, started, True, detail, fields)


def _record_step(journal, name, started, ok, detail, fields) -> None:
    finished = time.time()
    journal.append(
        "step",
        name=name,
        started_at=round(started, 3),
        finished_at=round(finished, 3),
        elapsed_s=round(finished - started, 3),
        ok=ok,
        detail=detail,
        **fields,
    )


class Heartbeat:
    """`<attempt>/heartbeat` (unix time) every interval from a thread; not in the journal."""

    def __init__(self, directory: Path, interval: float) -> None:
        self.directory = directory
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def beat(self) -> None:
        temporary = self.directory / f".{HEARTBEAT_FILE}.tmp"
        temporary.write_text(str(time.time()))
        os.replace(temporary, self.directory / HEARTBEAT_FILE)

    def start(self) -> "Heartbeat":
        self.beat()
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.beat()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)


def _keypair(attempt: Path) -> str:
    private = attempt / PRIVATE_DIR
    os.mkdir(private, 0o700)
    done = proc.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "sound-scout-gpu"]
        + ["-f", str(private / "id_ed25519")],
        timeout=60,
        max_output_bytes=1 << 16,
    )
    if done.returncode != 0:
        raise VastError("ssh-keygen failed")
    return (private / "id_ed25519.pub").read_text().strip()


def wait_ready(api: Api, instance_id: int, intervals: Mapping) -> dict:
    ends = time.monotonic() + intervals["ready_timeout_s"]
    while True:
        instance = api.show(instance_id)
        status = None if instance is None else instance.get("actual_status")
        if status in BAD_STATUSES:
            raise VastError(f"instance {instance_id} is {status}")
        if status == "running" and "22/tcp" in (instance.get("ports") or {}):
            return instance
        if time.monotonic() > ends:
            raise VastError(f"instance {instance_id} not running in time (last: {status})")
        time.sleep(intervals["ready_poll_s"])


def wait_ssh(host, intervals: Mapping) -> None:
    """sshd and the key from `attach_ssh` come up after `running`: poll `true` on the host
    until it answers or the time is up (never a second create)."""
    ends = time.monotonic() + intervals["ssh_timeout_s"]
    while True:
        done = host.run(["true"], timeout=30)
        if not done.timed_out and done.returncode == 0:
            return
        if time.monotonic() > ends:
            raise VastError(f"host: SSH not ready in time (exit {done.returncode})")
        time.sleep(intervals["ready_poll_s"])


def _checked(done: proc.RunResult, what: str) -> str:
    if done.timed_out or done.returncode != 0:
        raise VastError(f"host: {what} failed (exit {done.returncode})")
    return done.stdout.decode("utf-8", "replace")


def _checked_object(done: proc.RunResult, what: str) -> dict:
    """The stdout of a host command that must print one JSON object."""
    try:
        printed = json.loads(_checked(done, what))
    except (ValueError, RecursionError):
        raise VastError(f"host: {what} printed no JSON") from None
    if not isinstance(printed, dict):
        raise VastError(f"host: {what} printed JSON that is not an object")
    return printed


def _public(version: object) -> str | None:
    return version.split("+", 1)[0] if isinstance(version, str) else None


def runtime_mismatches(found: Mapping, vllm_cli: str) -> list[str]:
    """`vllm_cli` is the whole stdout of `vllm --version`: log lines may come before the
    version, so only its last non-empty line counts."""
    lines = vllm_cli.strip().splitlines()
    bad = []
    if (lines[-1].strip() if lines else "") != RUNTIME["vllm"] or _public(
        found.get("vllm")
    ) != RUNTIME["vllm"]:
        bad.append("vllm")
    if _public(found.get("torch")) != RUNTIME["torch"]:
        bad.append("torch")
    if found.get("cuda") != RUNTIME["cuda"]:
        bad.append("cuda")
    try:
        transformers = tuple(int(part) for part in _public(found.get("transformers")).split("."))
    except (AttributeError, ValueError):
        transformers = None
    low, high = TRANSFORMERS_RANGE
    if transformers is None or not low <= transformers < high:
        bad.append("transformers")
    return bad


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def upload_host_files(host, attempt: Path, models: Mapping, key: str, timeout: float) -> dict:
    """hostrun.py, the guard and the models manifest to the host, then SHA-256 read back."""
    manifest = attempt / "models.json"
    manifest.write_text(json.dumps(models, sort_keys=True, indent=2) + "\n")
    files = {
        HOSTRUN: HOST_FILES_DIR / "hostrun.py",
        f"{WORKSPACE}/ss/guard/sitecustomize.py": HOST_FILES_DIR / "sitecustomize.py",
        f"{WORKSPACE}/ss/models.json": manifest,
    }
    _checked(
        host.run(
            ["mkdir", "-p", f"{WORKSPACE}/ss/guard", f"{WORKSPACE}/out", f"{WORKSPACE}/models"],
            timeout,
        ),
        "mkdir",
    )
    for remote, local in files.items():
        host.put(local, remote, key)
    sums = _checked(host.run(["sha256sum", *files], timeout), "sha256sum").split()[::2]
    expected = [_sha256(local) for local in files.values()]
    if sums != expected:
        raise VastError("host files differ from the station's after copy")
    return {Path(remote).name: digest for remote, digest in zip(files, expected, strict=True)}


def _driver(gpu: str) -> str | None:
    """`driver_version`, the last field of the first `nvidia-smi` CSV line."""
    first = gpu.splitlines()[0] if gpu else ""
    return first.rsplit(",", 1)[-1].strip() if "," in first else None


def preflight(host, package: Mapping, models: Mapping, timeout: float, detail: dict) -> None:
    """Fills `detail` with every found value before any comparison: a refusal keeps them in the
    journal step."""
    detail["gpu"] = gpu = _checked(
        host.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            timeout,
        ),
        "nvidia-smi",
    ).strip()
    vllm_cli = _checked(host.run(["vllm", "--version"], timeout), "vllm --version")
    detail["vllm_cli"] = vllm_cli.strip()
    detail["found"] = found = _checked_object(
        host.run(["python3", HOSTRUN, "preflight", "--workspace", WORKSPACE], timeout),
        "preflight",
    )
    found["driver"] = _driver(gpu)
    selftest_out = host.run(
        ["python3", HOSTRUN, "selftest", "--out", f"{WORKSPACE}/out/guard-selftest.json"], timeout
    )
    try:
        selftest = json.loads(selftest_out.stdout)
    except ValueError:
        selftest = {"ok": False}
    need = sum(
        entry["size"] for model in package["models"] for entry in models[model["name"]]["files"]
    )
    detail["need_bytes"] = need + package["disk_margin_bytes"]
    detail["selftest"] = selftest
    bad = runtime_mismatches(found, vllm_cli)
    if bad:
        raise VastError(f"preflight: runtime differs from the pin in {bad}")
    if found.get("disk_free_bytes", 0) < detail["need_bytes"]:
        raise VastError("preflight: not enough free disk for the weights")
    if selftest.get("ok") is not True:
        raise VastError("preflight: the guard selftest failed")


def fetch_weights(host, attempt: Path, name: str, models: Mapping, intervals: Mapping) -> Path:
    spec = models[name]
    target = f"{WORKSPACE}/models/{name}"
    download = [
        "hf",
        "download",
        spec["repo"],
        "--revision",
        spec["revision"],
        "--local-dir",
        target,
        "--exclude",
        "spk_dict.pt",
    ]
    _checked(host.run(download, intervals["weights_timeout_s"]), f"hf download {name}")
    remote = f"{WORKSPACE}/out/{name}/weights-receipt.json"
    _checked(host.run(["mkdir", "-p", f"{WORKSPACE}/out/{name}"], 60), "mkdir")
    verify = ["python3", HOSTRUN, "weights", "--manifest", f"{WORKSPACE}/ss/models.json"]
    verify += ["--model", name, "--dir", target, "--out", remote]
    host.run(verify, intervals["weights_timeout_s"])
    receipts = attempt / RECEIPTS_DIR
    with contextlib.suppress(FileExistsError):
        os.mkdir(receipts, 0o700)
    local = receipts / f"{name}.json"
    host.get(remote, local)
    receipt = json.loads(local.read_text())
    if receipt.get("ok") is not True or receipt.get("files") != expected_files(models, name):
        raise VastError(f"weights of {name} do not match the pinned files")
    return local


def pilot_clips() -> dict[str, bytes]:
    """X1: 1 kHz, 3 s; X2: 1 kHz, 6 s, both at -20 dBFS peak (0.1): only the length differs."""
    clips = {}
    for clip_id, seconds in (("X1", 3.0), ("X2", 6.0)):
        t = np.arange(int(seconds * gate0.SAMPLE_RATE)) / gate0.SAMPLE_RATE
        samples = PILOT_AMPLITUDE * np.sin(2 * np.pi * PILOT_TONE_HZ * t)
        clips[clip_id] = gate0.wav_bytes(gate0.pcm16(samples))
    return clips


def run_pilot(client: gate0.Client) -> dict:
    """K1 on X1 and X2: audio tokens above zero and growing with the length (PROPOSAL § 5)."""
    lines = []
    text = gate0.QUESTIONS["K1"]
    text_tokens = client.text_tokens(text)
    for clip_id, wav in pilot_clips().items():
        audio = {
            "type": "input_audio",
            "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"},
        }
        body = {
            "model": client.model,
            "messages": [{"role": "user", "content": [audio, {"type": "text", "text": text}]}],
            "max_tokens": gate0.CLOSED_MAX_TOKENS,
            **gate0.sampling("vllm"),
        }
        status, raw, error = client.request(gate0.CHAT_ROUTE, body, timeout=gate0.REQUEST_TIMEOUT_S)
        try:
            prompt = json.loads(raw)["usage"]["prompt_tokens"] if status == 200 else None
        except (ValueError, KeyError, TypeError):
            prompt = None
        audio_tokens = (
            prompt - text_tokens if isinstance(prompt, int) and text_tokens is not None else None
        )
        lines.append(
            {
                "clip": clip_id,
                "http_status": status,
                "error": error,
                "prompt_tokens": prompt,
                "audio_tokens": audio_tokens,
            }
        )
    counts = [line["audio_tokens"] for line in lines]
    ok = all(isinstance(c, int) and c > 0 for c in counts) and counts[1] > counts[0]
    return {"text_tokens": text_tokens, "lines": lines, "ok": ok}


EVIDENCE_TAIL_BYTES = 8192


def server_state(host, out: str, timeout: float) -> dict:
    argv = ["python3", HOSTRUN, "state", "--stats", f"{out}/server-stats.jsonl"]
    return _checked_object(host.run([*argv, "--pid-file", f"{out}/serve.pid"], timeout), "state")


def _judge(state: Mapping | None, late: bool) -> None:
    """`ServerExited` when the state of `hostrun.py serve` shows the server will never answer."""
    if state is None:
        return
    ended = state.get("exit")
    if isinstance(ended, dict):
        raise ServerExited(ended.get("reason"), ended.get("returncode"))
    if state.get("started"):
        if not state.get("serve_alive"):
            raise ServerExited("hostrun_gone", None)
    elif late:
        raise ServerExited("not_started", None)


def _wait_health(
    client: gate0.Client,
    timeout: float,
    check: Callable[[], Mapping | None],
    check_every: float,
    start_grace: float,
) -> None:
    """`/health` every second until 200; every `check_every` also the server state from the
    host, so a server that exited fails at once instead of at `timeout`."""
    started = time.monotonic()
    ends, next_check = started + timeout, started + check_every
    while client.request("/health", timeout=5)[0] != 200:
        now = time.monotonic()
        if now >= next_check:
            next_check = now + check_every
            _judge(check(), now - started > start_grace)
        if time.monotonic() > ends:
            raise VastError("the vLLM server did not answer /health in time")
        time.sleep(1.0)


def _private_file(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def fetch_server_evidence(
    host, attempt: Path, journal: Journal, name: str, out: str, timeout: float = 60.0
) -> None:
    """Log tails and the process list of the host into `evidence-<name>.json` and the journal,
    while the server tree still lives. A failure is journaled, not raised, so destroy always
    follows; a cancellation (KeyboardInterrupt, SystemExit) is journaled and propagates."""
    try:
        found = _checked_object(
            host.run(["python3", HOSTRUN, "evidence", "--dir", out], timeout), "evidence"
        )
        files = found.get("files")
        if not isinstance(files, dict) or not isinstance(found.get("processes"), list):
            raise VastError("host: evidence printed no files or processes")
        file_name = f"evidence-{name}.json"
        text = json.dumps(found, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        _private_file(attempt / file_name, redact(text, journal.key))
        tails, sizes = {}, {}
        for path, entry in files.items():
            entry = entry if isinstance(entry, dict) else {}
            data = str(entry.get("tail", "")).encode()[-EVIDENCE_TAIL_BYTES:]
            tails[path] = redact(data.decode("utf-8", "replace"), journal.key)
            sizes[path] = entry.get("size")
        processes = json.loads(redact(json.dumps(found["processes"]), journal.key))
        journal.append(
            "server_evidence",
            model=name,
            processes=processes,
            tails=tails,
            sizes=sizes,
            file=file_name,
        )
    except BaseException as error:
        with contextlib.suppress(BaseException):
            journal.append(
                "server_evidence",
                model=name,
                error=redact(f"{type(error).__name__}: {error}", journal.key),
            )
        if not isinstance(error, Exception):
            raise


def _state_or_none(host, out: str, journal: Journal, detail: dict) -> dict | None:
    """The server state, or None after a failed `state` call, counted in `state_errors`."""
    try:
        return server_state(host, out, 60.0)
    except Exception as error:
        errors = detail.setdefault("state_errors", {"count": 0, "last": None})
        errors["count"] += 1
        errors["last"] = redact(f"{type(error).__name__}: {error}", journal.key)
        return None


def serve_and_run(host, attempt, journal, package, entry, models, receipt, iid, config_, intervals):
    """Steps `smoke` and `run` of one model on a server of its own. The host's evidence is
    journaled and the server stops whatever happens; a server that does not stop ends the
    attempt, so the next model never starts beside it."""
    name = entry["name"]
    out = f"{WORKSPACE}/out/{name}"
    deadline = summarize(journal.replay())["opened"]["deadline_unix"]
    serve = ["python3", HOSTRUN, "serve", "--deadline", str(deadline)]
    serve += ["--rss-limit", str(entry["rss_limit_bytes"]), "--stats", f"{out}/server-stats.jsonl"]
    serve += ["--log", f"{out}/server.log", "--guard-log", f"{out}/guard.log"]
    serve += ["--pid-file", f"{out}/serve.pid", "--"]
    serve += vllm_argv(f"{WORKSPACE}/models/{name}", name, host.server_port)
    fetched = False
    try:
        try:
            with _step(journal, "smoke", model=name) as detail:
                host.spawn(serve, f"{out}/hostrun.log")
                port = host.tunnel(host.server_port)
                client = gate0.Client(port, name)
                try:
                    _wait_health(
                        client,
                        intervals["health_timeout_s"],
                        lambda: _state_or_none(host, out, journal, detail),
                        intervals["server_check_s"],
                        intervals["server_start_s"],
                    )
                except ServerExited as error:
                    detail["server_exit"] = {"reason": error.reason, "returncode": error.returncode}
                    raise
                pilot = run_pilot(client)
                (attempt / f"pilot-{name}.json").write_text(json.dumps(pilot, indent=2) + "\n")
                detail.update(pilot_ok=pilot["ok"], tunnel_port=port)
                if not pilot["ok"]:
                    raise VastError(f"pilot of {name}: audio tokens absent or not growing")
            with _step(journal, "run", model=name) as detail:
                result = gate0.run_spike(
                    entry["gate0_date"],
                    config_,
                    decision=Path(entry["decision"]),
                    runtime="vllm",
                    server_port=port,
                    served_model=name,
                    weights=expected_files(models, name),
                    weights_receipt=receipt,
                    remote={"instance_id": iid, "tunnel_port": port},
                )
                detail.update(verdict=result["verdict"], reason=result["reason"])
                state = _state_or_none(host, out, journal, detail)
                try:
                    _judge(state, late=True)
                except ServerExited as error:
                    detail["server_exit"] = {"reason": error.reason, "returncode": error.returncode}
                fetched = True
                fetch_server_evidence(host, attempt, journal, name, out)
        except BaseException:
            if not fetched:
                fetch_server_evidence(host, attempt, journal, name, out)
            raise
    finally:
        stop = ["python3", HOSTRUN, "stop", "--pid-file", f"{out}/serve.pid"]
        stopped = host.run([*stop, "--stats", f"{out}/server-stats.jsonl"], 120)
        with contextlib.suppress(BaseException):
            journal.append(
                "server_stopped",
                model=name,
                returncode=stopped.returncode,
                timed_out=stopped.timed_out,
            )
    if stopped.timed_out or stopped.returncode != 0:
        raise VastError(f"the vLLM server of {name} did not stop; no next model starts")


def verify_archive(archive: Path, target: Path) -> int:
    """Unpack with the `data` filter, then every file against `SHA256SUMS`: an extra, missing
    or changed file, a link or a path out of `target` refuses."""
    os.mkdir(target, 0o700)
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
    except (tarfile.TarError, OSError) as error:
        raise VastError(f"archive refused: {type(error).__name__}") from None
    found = {}
    for path in target.rglob("*"):
        if path.is_symlink():
            raise VastError(f"archive holds a link: {path.relative_to(target)}")
        if path.is_file():
            found[path.relative_to(target).as_posix()] = path
    sums_path = found.pop(SUMS_FILE, None)
    if sums_path is None:
        raise VastError(f"archive without {SUMS_FILE}")
    listed = {}
    for line in sums_path.read_text().splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    if set(found) != set(listed):
        extra, missing = sorted(set(found) - set(listed)), sorted(set(listed) - set(found))
        raise VastError(f"archive files differ from {SUMS_FILE}: extra {extra}, missing {missing}")
    changed = sorted(name for name, path in found.items() if _sha256(path) != listed[name])
    if changed:
        raise VastError(f"archive files changed: {changed}")
    return len(found)


def download(host, attempt: Path, journal: Journal, timeout: float) -> dict:
    printed = _checked_object(
        host.run(
            ["python3", HOSTRUN, "pack", "--root", f"{WORKSPACE}/out"]
            + ["--out", f"{WORKSPACE}/results.tar"],
            timeout,
        ),
        "pack",
    )
    local = attempt / "results.tar"
    host.get(f"{WORKSPACE}/results.tar", local)
    digest = _sha256(local)
    if digest != printed.get("archive_sha256"):
        raise VastError("downloaded archive SHA-256 differs from the host's")
    files = verify_archive(local, attempt / HOST_DIR)
    journal.append("download_verified", archive_sha256=digest, files=files)
    return {"archive_sha256": digest, "files": files}


def run_attempt(
    package_path: Path,
    *,
    api: Api,
    make_host: Callable[[Mapping, Path], RemoteHost],
    config_: config.Config,
    runner,
    intervals: Mapping = INTERVALS,
    dry_run: bool = False,
    key_path: Path = KEY_PATH,
    models: Mapping = MODELS,
    approval_url: str | None = None,
) -> Journal:
    """Every step of one attempt; any failure ends in destroy and the absence check."""
    package, package_sha = load_package(package_path, models)
    if not dry_run and not (approval_url or "").startswith("https://github.com/"):
        raise VastError("deploy needs --approval with the link to the owner's approval")
    check_days(package, config_)
    gpu = gpu_root(config_)
    refuse_open_attempts(gpu)
    initial_credit = _number(api.user().get("credit"), "credit")
    attempt_id = new_attempt_id()
    journal = Journal.create(gpu, attempt_id, api.key)
    now = time.time()
    journal.append(
        "opened",
        attempt_id=attempt_id,
        label=LABEL_PREFIX + attempt_id,
        package_sha256=package_sha,
        approval_url=approval_url,
        preregistration=package["preregistration"],
        offer_id=package["offer_id"],
        envelope=package["envelope"],
        ceiling_usd=package["ceiling_usd"],
        early_stop_usd=package["early_stop_usd"],
        initial_credit=initial_credit,
        deadline_unix=round(now + package["deadline_minutes"] * 60, 3),
        image_ref=IMAGE_REF,
        models={e["name"]: models[e["name"]]["revision"] for e in package["models"]},
        python=sys.executable,
        api_base=api.base,
        key_path=str(Path(key_path).absolute()),
        dry_run=dry_run,
    )
    heartbeat = Heartbeat(journal.dir, intervals["heartbeat_s"]).start()
    attempt, host, failure = journal.dir, None, None
    timeout = intervals["command_timeout_s"]

    def tick(_: Mapping) -> Api:
        return api

    try:
        with _step(journal, "arm"):
            arm(journal, runner, dry_run=dry_run)
            if watch_tick(journal, tick, runner=runner) != "idle":
                raise VastError("the manual watchdog tick before create did not pass")
        with _step(journal, "create") as detail:
            public_key = _keypair(attempt)
            try:
                detail["instance_id"] = instance_id = create_instance(api, journal, package, runner)
            except CreateOutcomeUnknown:
                watch_tick(journal, tick, runner=runner)
                raise
        with _step(journal, "ready") as detail:
            instance = wait_ready(api, instance_id, intervals)
            api.attach_ssh(instance_id, public_key)
            host = make_host(instance, attempt)
            wait_ssh(host, intervals)
            detail["status"] = instance.get("actual_status")
        with _step(journal, "preflight") as detail:
            detail["host_files"] = upload_host_files(host, attempt, models, api.key, timeout)
            preflight(host, package, models, timeout, detail)
        receipts = {}
        with _step(journal, "weights") as detail:
            for entry in package["models"]:
                receipts[entry["name"]] = fetch_weights(
                    host, attempt, entry["name"], models, intervals
                )
            detail["models"] = sorted(receipts)
        for entry in package["models"]:
            serve_and_run(
                host,
                attempt,
                journal,
                package,
                entry,
                models,
                receipts[entry["name"]],
                instance_id,
                config_,
                intervals,
            )
        with _step(journal, "download") as detail:
            detail.update(download(host, attempt, journal, timeout))
    except BaseException as error:
        failure = error
    finally:
        if host is not None:
            host.close()
        try:
            with _step(journal, "destroy"):
                destroy_until_absent(journal, tick, runner, intervals)
        except BaseException as error:
            failure = failure or error
        heartbeat.stop()
    if failure is not None:
        raise failure
    return journal


def timeline(journal: Journal) -> dict:
    state = summarize(journal.replay())
    steps = [
        {
            key: step.get(key)
            for key in ("name", "model", "started_at", "finished_at", "elapsed_s", "ok", "detail")
        }
        for step in state["steps"]
    ]
    return {
        "attempt_id": state["opened"]["attempt_id"],
        "dry_run": state["opened"].get("dry_run"),
        "steps": steps,
        "closed": state["closed"],
    }


# --- the dry run ---------------------------------------------------------------------------

DRY_RUN_DAYS = (
    ("qwen25omni", "2000-01-01", "docs/decisions/0007-gate0-result.md", 17_500_000_000),
    ("qwen3omni", "2000-01-02", "docs/decisions/0023-gate0-qwen3-omni.md", 25_500_000_000),
)
SHIM = """#!{python}
import sys
from sound_scout.eval import vast_fake
sys.exit(vast_fake.main(["{tool}", *sys.argv[1:]]))
"""
PYTHON_SHIM = """#!/bin/sh
PYTHONPATH="{site}${{PYTHONPATH:+:$PYTHONPATH}}" exec "{python}" "$@"
"""
FAKE_DISTRIBUTIONS = {"vllm": "0.34.0", "torch": "2.14.0", "transformers": "5.20.0"}
FAKE_TORCH_CUDA = RUNTIME["cuda"]


def _fake_host_root(root: Path) -> dict:
    """The dry run's host: fake `vllm`, `hf`, `nvidia-smi` and a `python3` that sees fake package
    metadata of the pinned runtime and a `torch.version` with `FAKE_TORCH_CUDA`; no
    `CUDA_VERSION`, as in an SSH session without the image ENV."""
    bin_dir, site = root / "bin", root / "fake-site"
    for directory in (root, bin_dir, site):
        directory.mkdir(mode=0o700, exist_ok=True)
    for tool in ("vllm", "hf", "nvidia-smi"):
        (bin_dir / tool).write_text(SHIM.format(python=sys.executable, tool=tool))
        (bin_dir / tool).chmod(0o700)
    (bin_dir / "python3").write_text(PYTHON_SHIM.format(site=site, python=sys.executable))
    (bin_dir / "python3").chmod(0o700)
    for name, version in FAKE_DISTRIBUTIONS.items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (site / "torch").mkdir(exist_ok=True)
    (site / "torch" / "__init__.py").write_text("")
    (site / "torch" / "version.py").write_text(f"cuda = {FAKE_TORCH_CUDA!r}\n")
    return {"PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}"}


def _refuse_system(argv: list[str]) -> tuple[int, str]:
    raise VastError(f"a dry run starts no system command: {argv[0]}")


def dry_run(out: Path | None, config_: config.Config) -> dict:
    """The full timeline against the fakes; writes `timeline.json` and `fake-api.jsonl`."""
    if out is None:
        out = gpu_root(config_) / f"dry-run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = Path(out).absolute()
    out.mkdir(mode=0o700, parents=True)
    dry_config = dataclasses.replace(config_, data_root=out.resolve())
    key_file = out / "fake-api-key"
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, secrets.token_urlsafe(32).encode() + b"\n")
    os.close(fd)
    key = read_api_key(key_file)
    env = _fake_host_root(out / "host-root")
    entries = []
    for name, day, decision, rss in DRY_RUN_DAYS:
        document = REPO_ROOT / decision
        if not document.is_file():
            raise VastError(f"dry run needs the source tree: {decision} is missing")
        gate0.prepare_clips(day, dry_config, decision=document, controls_only=True)
        entries.append(
            {"name": name, "gate0_date": day, "decision": str(document), "rss_limit_bytes": rss}
        )
    package = {
        "offer_id": vast_fake.OFFER_ID,
        "envelope": {"dph_total": {"lte": 1.533}, "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 79000}},
        "ceiling_usd": 4.0,
        "early_stop_usd": 2.5,
        "deadline_minutes": 150,
        "disk_gb": 150,
        "in_gb": 102.95,
        "out_gb": 1.0,
        "disk_margin_bytes": 0,
        "models": entries,
        "preregistration": "dry run: no preregistration",
    }
    package_path = out / "package.json"
    package_path.write_text(json.dumps(package, indent=2) + "\n")
    fake = vast_fake.FakeVast(key).start()
    started = time.monotonic()
    journal = None
    try:
        journal = run_attempt(
            package_path,
            api=Api(fake.url, key),
            make_host=lambda instance, attempt: LocalHost(out / "host-root", env),
            config_=dry_config,
            runner=_refuse_system,
            intervals=DRY_INTERVALS,
            dry_run=True,
            key_path=key_file,
            models=vast_fake.synthetic_models(),
        )
    finally:
        log = "".join(json.dumps(entry) + "\n" for entry in fake.log)
        (out / "fake-api.jsonl").write_text(log)
        remaining = sorted(i for i, inst in fake.instances.items() if fake._visible(inst))
        fake.stop()
    result = {
        **timeline(journal),
        "elapsed_s": round(time.monotonic() - started, 3),
        "journal": str(journal.path.relative_to(out)),
        "fake_api": {
            "put": sum(1 for e in fake.log if e["method"] == "PUT"),
            "delete": sum(1 for e in fake.log if e["method"] == "DELETE"),
            "instances_left": remaining,
        },
        "note": "накладные расходы стека на станции; не время на Vast.ai",
    }
    (out / "timeline.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


# --- the CLI -------------------------------------------------------------------------------


def status(journal: Journal) -> dict:
    state = summarize(journal.replay())
    return {
        **timeline(journal),
        "instance_ids": state["ids"],
        "create_requested": state["create_requested"],
        "create_outcome_unknown": state["unknown"],
        "absence_verified": sorted(state["absent"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.vast")
    commands = parser.add_subparsers(dest="command", required=True)
    deploy = commands.add_parser("deploy")
    deploy.add_argument("--package", type=Path, required=True)
    deploy.add_argument("--approval", required=True)
    for name in ("watch", "cutoff", "destroy", "status"):
        commands.add_parser(name).add_argument("--journal", type=Path, required=True)
    commands.add_parser("dry-run").add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    key = None
    try:
        if args.command in ("watch", "cutoff"):
            journal = Journal.open(args.journal)
            outcome = watch_tick(
                journal, _api_from_journal, runner=system, cutoff=args.command == "cutoff"
            )
            print(json.dumps({"outcome": outcome}))
            return 1 if outcome == "error" else 0
        if args.command == "destroy":
            journal = Journal.open(args.journal)
            print(
                json.dumps(
                    {"outcome": destroy_until_absent(journal, _api_from_journal, system, INTERVALS)}
                )
            )
            return 0
        if args.command == "status":
            print(json.dumps(status(Journal.open(args.journal)), indent=2, ensure_ascii=False))
            return 0
        config_ = config.load()
        if args.command == "dry-run":
            result = dry_run(args.out, config_)
            print(
                json.dumps(
                    {"steps": [s["name"] for s in result["steps"]], "closed": result["closed"]}
                )
            )
            return 0
        key = read_api_key()
        journal = run_attempt(
            args.package,
            api=Api(API_BASE, key),
            make_host=lambda instance, attempt: RemoteHost(attempt, *ssh_endpoint(instance)),
            config_=config_,
            runner=system,
            approval_url=args.approval,
        )
        print(json.dumps(status(journal), ensure_ascii=False))
        return 0
    except (VastError, gate0.Gate0Error, config.ConfigError, paths.PathDenied) as error:
        print(f"vast: {redact(str(error), key)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
