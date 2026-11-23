"""GPU reference runs on Vast.ai: pinned image, one create, watchdog, verified download (#147).

The decision is docs/decisions/0026-gpu-stack.md (the Decision comment of #147). One module on
stdlib `urllib`, `python -m sound_scout.eval.vast <command>`; it is not part of the `sound-scout`
CLI or of MCP, and it does not install `guard.install_network_guard`, because it talks to one
external host, `API_BASE`. Recording analysis stays offline (R09).

- `deploy --package P --approval URL` runs every step: `open` (append-only ownership journal
  under `<data_root>/gpu/<attempt>/`), `arm` (two `systemd-run --user` timers that need no agent),
  `create` (one PUT, never repeated, on the first approved offer that fits), `ready`,
  `preflight`, then per model `weights`, `smoke`, `run` (the Gate 0 client on this station
  through an SSH tunnel) and `extended` (the blocks of the package's extended set, `gpu_ext`),
  each model on a server of its own, its host evidence journaled and the server stopped before
  the next; then `download` (SHA-256 of the archive and of every file) and `destroy` with a check
  that the instance is gone by id and by label. A failure of one model is its `model_outcome`
  and the next model still runs (#149); a failure before the models, a server that does not
  stop, a lost host, the soft end or a cancellation ends the attempt, always through download
  (while the host answers) and destroy. Before anything opens, `check_days` holds each model to
  the rule of its local run (`GATE0_RUNS`) and to the clips its package entry pins, and
  `check_extended` holds the extended set to its pins.
- `watch --journal J` and `cutoff --journal J` are the timers' ticks; `destroy --journal J`
  destroys by hand; `status --journal J` reads the journal only.
- `launch-argv --package P --approval URL` runs the checks `deploy` runs before any API call
  and prints the `systemd-run --user` argv that starts `deploy` outside the tool session
  (`launch_argv`); it starts nothing. SIGTERM and SIGHUP of `deploy` cancel the attempt through
  download and destroy (`cancel_on_signals`).
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
import errno
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
import signal
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
from sound_scout.eval import gate0, gpu_ext, vast_fake

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
# the fixed limits of the stack (#149: from the times #148 measured); a package may set the
# health and weights limits of each model and the request limits of its Gate 0 and its blocks
INTERVALS = {
    "heartbeat_s": 60.0,
    "ready_poll_s": 10.0,
    "ready_timeout_s": 900.0,
    "ssh_timeout_s": 300.0,
    "health_timeout_s": 1200.0,
    "server_check_s": 15.0,
    "server_start_s": 60.0,
    "destroy_poll_s": 10.0,
    "destroy_timeout_s": 900.0,
    "command_timeout_s": 600.0,
    "weights_timeout_s": 3 * 3600.0,
    "weights_poll_s": 10.0,
    "weights_stall_s": 300.0,
    "gpu_free_timeout_s": 60.0,
    "gpu_free_poll_s": 2.0,
    "ssh_retry_s": 30.0,
    "ssh_lost_s": 300.0,
    "evidence_timeout_s": 60.0,
    "stop_timeout_s": 120.0,
    "sync_timeout_s": 120.0,
    "download_timeout_s": 300.0,
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
    "weights_poll_s": 0.05,
    "weights_stall_s": 2.0,
    "gpu_free_timeout_s": 1.0,
    "gpu_free_poll_s": 0.1,
    "ssh_retry_s": 0.1,
    "ssh_lost_s": 5.0,
}
SSH_RETRIES = 3
SSH_EXIT = 255
SSH_PROBE_S = 30.0
SSH_ALIVE_S = 15
# the longest an ssh in flight takes to notice a lost network: OpenSSH counts four
# ServerAliveInterval probes before it gives up (its ServerAliveCountMax 3 is compared after the
# increment); a try with a shorter timeout notices at its timeout
SSH_NOTICE_S = 4 * SSH_ALIVE_S
MIN_STEP_S = 10.0
MIN_RETRY_S = 1.0
STATE_TIMEOUT_S = 60.0
RECOVER_HEALTH_S = 30.0
# a fork the station itself refused (no memory, too many processes): no fault of the host
LOCAL_ERRNOS = (errno.ENOMEM, errno.EAGAIN)
LOCAL_RETRY_S = 0.5
REOPEN_EVERY_CHECKS = 4
TUNNEL_READY_S = 15.0
CREDIT_MARGIN_USD = 1.0
SPEND_TAIL_RESERVE_S = 180.0
GPU_FREE_MIB = 1024
MAX_DEADLINE_MINUTES = 120
MAX_OFFERS = 3
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


class TransportError(VastError):
    """The host gave no answer over SSH: ssh exit 255, a killed or timed-out ssh, a failed scp."""


class HostLost(TransportError):
    """The host stayed silent for the whole SSH window: `ssh_lost` for the attempt."""


class CreateOutcomeUnknown(VastError):
    pass


class AttemptFatal(VastError):
    """A failure no next model can outlive: the attempt goes straight to download and destroy.
    `result` is what the model in hand still gave (its Gate 0 verdict), if anything."""

    def __init__(self, message: str, result: Mapping | None = None) -> None:
        super().__init__(message)
        self.result = dict(result or {})


class ModelFailed(VastError):
    """One model ends here; the next model of the package still runs (#149)."""

    def __init__(self, step: str, reason: str) -> None:
        super().__init__(f"{step}: {reason}")
        self.step, self.reason = step, reason


class ModelSkipped(ModelFailed):
    """The agent stopped (`soft_stop`) before this model gave anything: `skipped`, not failed."""


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
        "credit_after_close": None,
        "steps": [],
    }
    for event in events:
        kind = event["event"]
        if kind in ("opened", "armed", "closed", "credit_after_close"):
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


def offer_ids(package: Mapping) -> list[int]:
    """The approved offers in order: `offer_ids`, or the one `offer_id` of an older package."""
    found = package.get("offer_ids")
    return list(found) if found is not None else [package["offer_id"]]


def estimate_hours(package: Mapping) -> float:
    """Up to the deadline plus the 15 min the cutoff may still need: the worst case the
    watchdog lets happen, not the plan (#149)."""
    return (package["deadline_minutes"] + LOST_CREATE_GRACE_S / 60) / 60


def choose_offer(api: Api, journal: Journal, package: Mapping) -> tuple[int, dict, float]:
    """The first approved offer that is there, inside the envelope and under the ceiling at
    `estimate_hours`; every one passed over is journaled with its reason. A refusal here sends
    no PUT, so it is no rental."""
    reasons = []
    for offer_id in offer_ids(package):
        try:
            offer = api.offer(offer_id, package["disk_gb"])
        except VastError as error:
            reason = redact(str(error), api.key)
        else:
            failed = envelope_failures(offer, package["envelope"])
            cost = estimate_usd(offer, estimate_hours(package), package["in_gb"], package["out_gb"])
            if failed:
                reason = f"the offer is outside the envelope in {sorted(failed)}"
            elif cost > package["ceiling_usd"]:
                reason = f"estimate {cost:.2f} USD is over the ceiling"
            else:
                return offer_id, offer, cost
        journal.append("offer_skipped", offer_id=offer_id, reason=reason)
        reasons.append(f"{offer_id}: {reason}")
    raise VastError("create refused: " + "; ".join(reasons))


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
    offer_id, offer, cost = choose_offer(api, journal, package)
    until = int(opened["deadline_unix"]) + ONSTART_MARGIN_S
    payload = create_payload(package, opened["label"], until)
    check_no_secret(payload, api.key)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with journal.lock():
        _refuse_create(summarize(journal.replay()))
        journal.append(
            "create_requested",
            offer_id=offer_id,
            payload_sha256=digest,
            estimate_usd=round(cost, 4),
            prices={
                name: offer.get(name) for name in ("dph_total", "inet_down_cost", "inet_up_cost")
            },
        )
        try:
            answer = api.create(offer_id, payload)
        except BaseException as error:
            # a SIGTERM (SystemExit) in flight too: the PUT may have created the instance
            journal.append(
                "create_outcome_unknown",
                error=redact(f"{type(error).__name__}: {error}", api.key),
            )
            if not isinstance(error, Exception):
                raise
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


def read_credit_after_close(journal: Journal, api: Api) -> dict | None:
    """One read-only credit read after `closed` (0027 § 9 item 5); a failure is journaled, never
    raised: the attempt stays closed. An attempt still open gets no read here."""
    state = summarize(journal.replay())
    closed = state["closed"]
    if closed is None:
        return None
    initial = state["opened"]["initial_credit"]
    try:
        credit = _number(api.user().get("credit"), "credit")
    except Exception as error:
        return journal.append(
            "credit_after_close",
            since_closed_s=round(time.time() - closed["t"], 3),
            initial_credit=initial,
            error=redact(f"{type(error).__name__}: {error}", api.key),
        )
    return journal.append(
        "credit_after_close",
        since_closed_s=round(time.time() - closed["t"], 3),
        initial_credit=initial,
        credit=credit,
        delta_usd=round(initial - credit, 6),
    )


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
            f"ServerAliveInterval={SSH_ALIVE_S}",
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
        """Starts `argv` in the background once: an existing `log` means an earlier try whose
        answer was lost already started it, so a retry after a transport error starts nothing."""
        quoted = shlex.quote(log)
        line = f"test -e {quoted} || {{ nohup {shlex.join(argv)} > {quoted} 2>&1 < /dev/null & }}"
        _checked(self.run(["sh", "-c", line], timeout=60), f"start {argv[0]}")

    def put(self, local: Path, remote: str, key: str) -> None:
        check_no_secret(local.read_bytes(), key)
        target = f"{self._dest()}:{remote}"
        done = proc.run(
            ["scp", *self._options, "-P", str(self.port), str(local), target],
            timeout=600,
            max_output_bytes=1 << 16,
        )
        if done.returncode != 0:
            raise TransportError(f"scp to the host failed for {Path(remote).name}")

    def get(self, remote: str, local: Path, timeout: float = 1800.0) -> None:
        source = f"{self._dest()}:{remote}"
        done = proc.run(
            ["scp", *self._options, "-P", str(self.port), source, str(local)],
            timeout=timeout,
            max_output_bytes=1 << 16,
        )
        if done.returncode != 0:
            raise TransportError(f"scp from the host failed for {Path(remote).name}")

    def tunnel(self, remote_port: int, local_port: int | None = None) -> int:
        """`ssh -N -L` to the server port; returns once the local port accepts (or the ssh
        exited, or `TUNNEL_READY_S` passed: the caller's `/health` then decides)."""
        local_port = _free_port() if local_port is None else local_port
        forward = f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"
        log = open(self.attempt / f"tunnel-{local_port}.log", "ab")  # noqa: SIM115
        child, copier = proc.start(
            [*self.ssh_argv("-N", "-L", forward), self._dest()], log=log, max_log_bytes=1 << 20
        )
        self._tunnels.append((child, copier, log))
        self._forward = (remote_port, local_port)
        ends = time.monotonic() + TUNNEL_READY_S
        while child.poll() is None and time.monotonic() < ends:
            with (
                contextlib.suppress(OSError),
                socket.create_connection(("127.0.0.1", local_port), 1),
            ):
                break
            time.sleep(0.2)
        return local_port

    def ensure_tunnel(self, force: bool = False) -> bool:
        """A dead tunnel (the `ssh -N` exited), or with `force` a live one that no longer
        carries requests (a half-dead ssh), opens again on the same local port, so the client
        keeps its port; True when it was reopened."""
        if not self._tunnels:
            return False
        child, copier, _ = self._tunnels[-1]
        if child.poll() is None:
            if not force:
                return False
            gate0.stop_server(child)
            copier.join(5)
        remote_port, local_port = self._forward
        self.tunnel(remote_port, local_port)
        return True

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
        self._forwarders: list[vast_fake.Forwarder] = []
        self.lost = False
        self.failing: list[list[str]] = []
        self.reopen_delay_s = 0.0
        self._binding = threading.Event()  # a slow reopen in flight: its `ssh -N` still lives

    def _map(self, text: str) -> str:
        return text.replace(WORKSPACE, str(self.root))

    def _argv(self, argv: list[str]) -> list[str]:
        return ["env", *(f"{k}={v}" for k, v in self.env.items()), *map(self._map, argv)]

    def run(self, argv: list[str], timeout: float) -> proc.RunResult:
        if self.lost:  # as ssh: exit 255 when the host does not answer
            return proc.run(["sh", "-c", f"exit {SSH_EXIT}"], timeout=timeout, max_output_bytes=64)
        if any(argv[: len(prefix)] == prefix for prefix in self.failing):
            return proc.run(["false"], timeout=timeout, max_output_bytes=1 << 10)
        return proc.run(self._argv(argv), timeout=timeout, max_output_bytes=1 << 20)

    def spawn(self, argv: list[str], log: str) -> None:
        if self.lost:
            raise TransportError(f"host: start {argv[0]}: no SSH answer")
        if os.path.exists(self._map(log)):
            return
        handle = open(self._map(log), "wb")  # noqa: SIM115
        child, copier = proc.start(self._argv(argv), log=handle, max_log_bytes=1 << 20)
        self._spawned.append((child, copier, handle))

    def put(self, local: Path, remote: str, key: str) -> None:
        check_no_secret(local.read_bytes(), key)
        if self.lost:
            raise TransportError(f"scp to the host failed for {Path(remote).name}")
        shutil.copyfile(local, self._map(remote))

    def get(self, remote: str, local: Path, timeout: float = 1800.0) -> None:
        if self.lost:
            raise TransportError(f"scp from the host failed for {Path(remote).name}")
        shutil.copyfile(self._map(remote), local)

    def tunnel(self, remote_port: int, local_port: int | None = None) -> int:
        """A loopback forwarder stands in for `ssh -L`, so a rehearsal can kill it; with
        `reopen_delay_s` a reopened one binds only that much later, as a slow `ssh -N`."""
        if local_port and self.reopen_delay_s:
            delay, self.reopen_delay_s = self.reopen_delay_s, 0.0

            def later() -> None:
                time.sleep(delay)
                try:
                    self._forwarders.append(vast_fake.Forwarder(remote_port, local_port))
                finally:
                    self._binding.clear()

            self._binding.set()
            threading.Thread(target=later, daemon=True).start()
            return local_port
        forwarder = vast_fake.Forwarder(remote_port, local_port or 0)
        self._forwarders.append(forwarder)
        return forwarder.port

    def ensure_tunnel(self, force: bool = False) -> bool:
        if self.lost or not self._forwarders or self._binding.is_set():
            return False
        last = self._forwarders[-1]
        if last.alive():
            if not force:
                return False
            last.stop()
        self.tunnel(last.target, last.port)
        return True

    def kill_tunnels(self) -> None:
        for forwarder in self._forwarders:
            forwarder.stop()

    def stall_tunnels(self) -> None:
        """The tunnel stays up and accepts but carries nothing, as a half-dead `ssh -N`."""
        for forwarder in self._forwarders:
            forwarder.stall()

    def lose(self) -> None:
        """The host is gone for the station, as after a network loss or a destroy."""
        self.lost = True
        self.kill_tunnels()

    def close(self) -> None:
        self.kill_tunnels()
        self._forwarders.clear()
        for child, copier, handle in self._spawned:
            gate0.stop_server(child)
            copier.join(5)
            handle.close()
        self._spawned.clear()


# --- the steps -----------------------------------------------------------------------------

PACKAGE_NUMBERS = ("ceiling_usd", "early_stop_usd", "deadline_minutes", "in_gb", "out_gb")


def load_package(path: Path, models: Mapping, *, real: bool = False) -> tuple[dict, str]:
    """The approved package (JSON outside git) and its SHA-256; refuses a malformed one. A
    `real` package (for a rental) must also set `weights_timeout_s` of every model: the stack's
    3 h default is no limit for a $3 package."""
    data = Path(path).read_bytes()
    try:
        package = json.loads(data)
    except ValueError:
        raise VastError(f"package {Path(path).name} is not JSON") from None
    if not isinstance(package, dict):
        raise VastError("package must be a JSON object")
    for name in PACKAGE_NUMBERS:
        _number(package.get(name), name)
    if type(package.get("disk_gb")) is not int or package["disk_gb"] <= 0:
        raise VastError("package disk_gb must be a positive integer")
    ids = package.get("offer_ids", [package.get("offer_id")])
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= MAX_OFFERS
        or len(set(ids)) != len(ids)
        or not all(type(i) is int and i > 0 for i in ids)
        or ("offer_ids" in package and "offer_id" in package)
    ):
        raise VastError(
            f"package needs offer_id or offer_ids: 1 to {MAX_OFFERS} distinct positive integers"
        )
    if package["early_stop_usd"] > package["ceiling_usd"] or package["ceiling_usd"] <= 0:
        raise VastError("package needs 0 < early_stop_usd <= ceiling_usd")
    if not 0 < package["deadline_minutes"] <= MAX_DEADLINE_MINUTES:
        raise VastError(f"package deadline_minutes must be in (0, {MAX_DEADLINE_MINUTES}]")
    _check_limits(package)
    envelope = package.get("envelope")
    if not isinstance(envelope, dict) or "dph_total" not in envelope:
        raise VastError("package envelope must bound at least dph_total")
    _check_spend_margin(package)
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
        for name in ("slot_minutes", "health_timeout_s", "weights_timeout_s"):
            if entry.get(name) is not None and _number(entry[name], name) <= 0:
                raise VastError(f"package model {name} must be a positive number")
        if real and entry.get("weights_timeout_s") is None:
            raise VastError(f"package model {entry['name']} needs weights_timeout_s for a rental")
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


def _check_limits(package: Mapping) -> None:
    """The optional limits of a package with model isolation and an extended set (#149)."""
    deadline = package["deadline_minutes"]
    soft = package.get("soft_end_minutes")
    if soft is not None and not 0 < _number(soft, "soft_end_minutes") < deadline:
        raise VastError("package soft_end_minutes must lie between 0 and deadline_minutes")
    late = package.get("weights_late_minutes")
    if late is not None and (soft is None or not 0 <= _number(late, "weights_late_minutes") < soft):
        raise VastError("package weights_late_minutes needs soft_end_minutes above it")
    spend = package.get("spend_soft_usd")
    if spend is not None and not 0 < _number(spend, "spend_soft_usd") <= package["early_stop_usd"]:
        raise VastError("package needs 0 < spend_soft_usd <= early_stop_usd")
    for name in ("gate0_request_timeout_s", "closed_timeout_s", "review_timeout_s"):
        if package.get(name) is not None and _number(package[name], name) <= 0:
            raise VastError(f"package {name} must be a positive number")
    extended = package.get("extended")
    if extended is None:
        return
    if (
        not isinstance(extended, dict)
        or not isinstance(extended.get("dir"), str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", extended["dir"])
        or not all(
            isinstance(extended.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", extended[k])
            for k in ("manifest_sha256", "plan_sha256")
        )
    ):
        raise VastError(
            "package extended needs dir (a name under data_root), manifest_sha256, plan_sha256"
        )


RECOVER_PAST_STOP_S = 5.0 + STATE_TIMEOUT_S


def spend_tail_s(intervals: Mapping = INTERVALS) -> float:
    """The longest the agent runs after it stops itself (soft end or soft spend) until destroy:
    the request in flight, the check after it (`/health` and the server state, so a rule limit
    keeps its name), the block's sync, the evidence, the server stop, the download — each
    retry after a transport error inside its own limit (`step_until`, `tail=True`)."""
    return (
        REQUEST_PAST_SOFT_END_S
        + RECOVER_PAST_STOP_S
        + sum(
            intervals[name]
            for name in (
                "sync_timeout_s",
                "evidence_timeout_s",
                "stop_timeout_s",
                "download_timeout_s",
            )
        )
    )


def stop_path_s(intervals: Mapping = INTERVALS) -> float:
    """The longest `deploy` runs after SIGTERM until its DELETE is sent: the evidence, the
    server stop, the final download and the first destroy tick (the label listing and the
    DELETE, `HTTP_TIMEOUT_S` each). `TimeoutStopSec` must cover it; the absence check after the
    DELETE is the watchdog's when SIGKILL comes first (0028 Д.10)."""
    names = ("evidence_timeout_s", "stop_timeout_s", "download_timeout_s")
    return sum(intervals[name] for name in names) + 2 * HTTP_TIMEOUT_S


def _check_spend_margin(package: Mapping) -> None:
    """Between the soft spend and the watchdog's early stop the worst tail plus
    `SPEND_TAIL_RESERVE_S` must fit at the envelope's top `dph_total`; otherwise the watchdog
    could destroy the instance during the final download (#149)."""
    spend = package.get("spend_soft_usd")
    if spend is None:
        return
    top = package["envelope"]["dph_total"]
    top = top.get("lte") if isinstance(top, dict) else None
    if top is None:
        raise VastError("package spend_soft_usd needs an envelope dph_total lte bound")
    need = _number(top, "dph_total lte") * (spend_tail_s() + SPEND_TAIL_RESERVE_S) / 3600
    if package["early_stop_usd"] - spend < need - 1e-9:
        raise VastError(
            f"package spend_soft_usd {spend} leaves {package['early_stop_usd'] - spend:.3f} USD "
            f"before the early stop; the tail needs {need:.3f} USD at {top} USD/h"
        )


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
            raise VastError(
                f"Gate 0 day {day} is not prepared under the configured data_root "
                f"(no {day}/{gate0.MANIFEST_FILE}); set {config.CONFIG_PATH_ENV} to the "
                "configuration whose data_root holds the Gate 0 days"
            )
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


def check_extended(package: Mapping, config_: config.Config) -> tuple[Path, dict, list] | None:
    """The extended set of the package, by the SHA-256 of its manifest and plan and of every
    clip, refused if a Gate 0 clip the package pins is in it (B/V go only in Gate 0)."""
    extended = package.get("extended")
    if extended is None:
        return None
    directory = paths.resolve_output(config_.data_root, extended["dir"], config_)
    pinned = {
        digest for entry in package["models"] for digest in (entry.get("clips") or {}).values()
    }
    try:
        manifest, entries = gpu_ext.load_set(
            directory,
            manifest_sha256=extended["manifest_sha256"],
            plan_sha256=extended["plan_sha256"],
            forbidden_sha256=pinned,
        )
    except (gpu_ext.ExtError, OSError, ValueError) as error:
        raise VastError(f"extended set {extended['dir']}: {error}") from None
    if manifest.get("code_sha256") != gpu_ext.code_sha256():
        raise VastError(
            f"extended set {extended['dir']}: gpu_ext.py differs from the code the set was "
            "built and preregistered with (code_sha256)"
        )
    return directory, manifest, entries


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


def _transport_failed(done: proc.RunResult) -> bool:
    """ssh itself failed: its exit 255, or the ssh killed (by its timeout or a signal)."""
    return done.timed_out or done.returncode == SSH_EXIT or done.returncode < 0


def _answered(done: proc.RunResult, what: str) -> proc.RunResult:
    if _transport_failed(done):
        raise TransportError(f"host: {what}: no SSH answer (exit {done.returncode})")
    return done


def _checked(done: proc.RunResult, what: str) -> str:
    if _answered(done, what).returncode != 0:
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
    selftest_out = _answered(
        host.run(
            ["python3", HOSTRUN, "selftest", "--out", f"{WORKSPACE}/out/guard-selftest.json"],
            timeout,
        ),
        "selftest",
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


def _model_out(name: str) -> str:
    return f"{WORKSPACE}/out/{name}"


def start_fetch(host, journal: Journal, name: str, models: Mapping) -> None:
    """`hf download` of one model in the background on the host, its exit code then written to
    `download.rc`; the Xet chunk cache is off, so nothing outside the model directory grows."""
    spec, out = models[name], _model_out(name)
    _checked(host.run(["mkdir", "-p", out, f"{WORKSPACE}/models"], 60), "mkdir")
    download = ["env", "HF_XET_CHUNK_CACHE_SIZE_BYTES=0", "hf", "download", spec["repo"]]
    download += ["--revision", spec["revision"], "--local-dir", f"{WORKSPACE}/models/{name}"]
    download += ["--exclude", "spk_dict.pt"]
    script = (
        f"{shlex.join(download)} > {shlex.quote(out + '/download.log')} 2>&1 & "
        f"echo $! > {shlex.quote(out + '/download.pid')}; wait $!; "
        f"echo $? > {shlex.quote(out + '/download.rc')}"
    )
    host.spawn(["sh", "-c", script], f"{out}/download-wrapper.log")
    journal.append("weights_started", model=name)


def fetch_size(host, name: str, timeout: float = 60.0) -> dict:
    argv = ["python3", HOSTRUN, "size", "--dir", f"{WORKSPACE}/models/{name}"]
    argv += ["--rc-file", f"{_model_out(name)}/download.rc"]
    return _checked_object(host.run(argv, min(60.0, timeout)), "size")


def _stop_fetch(host, name: str) -> None:
    pid = f"{_model_out(name)}/download.pid"
    with contextlib.suppress(Exception):
        host.run(["sh", "-c", f"kill $(cat {shlex.quote(pid)})"], 30)


def finish_fetch(
    run: "Run",
    name: str,
    *,
    timeout: float,
    late_unix: float | None = None,
    detail: dict | None = None,
) -> Path:
    """Wait for the background download: bytes must keep growing (partial files included) and
    the exit code must be 0, else `ModelFailed`; then the host hashes every file and the receipt
    must match the pins. `late_unix` ends the wait as `weights_late`. Every host call goes
    through `transport_retry`, so a network loss costs the wait, not the model."""
    host, intervals, models = run.host, run.intervals, run.models
    detail = {} if detail is None else detail
    started = time.monotonic()
    ends = time.time() + timeout

    def until() -> float:
        bound = step_until(run, ends - time.time())
        return bound if late_unix is None else min(bound, late_unix)

    grown_at, last = started, -1
    while True:
        found = transport_retry(
            run, f"weights {name}", lambda left: fetch_size(host, name, left), until()
        )
        detail.update(bytes=found["bytes"], disk_free_bytes=found["disk_free_bytes"])
        now = time.monotonic()
        if found["bytes"] > last:
            grown_at, last = now, found["bytes"]
        if found["rc"] is not None:
            if found["rc"] != 0:
                raise ModelFailed("weights", f"hf download exited with {found['rc']}")
            break
        if now - grown_at > intervals["weights_stall_s"]:
            _stop_fetch(host, name)
            raise ModelFailed("weights", "download_stalled")
        if now - started > timeout:
            _stop_fetch(host, name)
            raise ModelFailed("weights", "download_timeout")
        if late_unix is not None and time.time() >= late_unix:
            _stop_fetch(host, name)
            raise ModelFailed("weights", "weights_late")
        time.sleep(intervals["weights_poll_s"])
    detail["download_s"] = round(time.monotonic() - started, 3)
    target = f"{WORKSPACE}/models/{name}"
    remote = f"{_model_out(name)}/weights-receipt.json"
    verify = ["python3", HOSTRUN, "weights", "--manifest", f"{WORKSPACE}/ss/models.json"]
    verify += ["--model", name, "--dir", target, "--out", remote]
    transport_retry(
        run,
        f"verify {name}",
        lambda left: _answered(host.run(verify, min(timeout, left)), "weights"),
        step_until(run, timeout),
    )
    receipts = run.attempt / RECEIPTS_DIR
    with contextlib.suppress(FileExistsError):
        os.mkdir(receipts, 0o700)
    local = receipts / f"{name}.json"
    with contextlib.suppress(FileNotFoundError):
        local.unlink()
    try:
        transport_retry(
            run,
            f"receipt {name}",
            lambda left: host.get(remote, local, min(STATE_TIMEOUT_S, left)),
            step_until(run, STATE_TIMEOUT_S),
        )
        receipt = json.loads(local.read_text())
    except (VastError, OSError, ValueError):
        raise ModelFailed("weights", "no weights receipt from the host") from None
    if receipt.get("ok") is not True or receipt.get("files") != expected_files(models, name):
        raise ModelFailed("weights", f"weights of {name} do not match the pinned files")
    detail["verified_s"] = round(time.monotonic() - started, 3)
    return local


def pilot_clips() -> dict[str, bytes]:
    """X1: 1 kHz, 3 s; X2: 1 kHz, 6 s, both at -20 dBFS peak (0.1): only the length differs."""
    clips = {}
    for clip_id, seconds in (("X1", 3.0), ("X2", 6.0)):
        t = np.arange(int(seconds * gate0.SAMPLE_RATE)) / gate0.SAMPLE_RATE
        samples = PILOT_AMPLITUDE * np.sin(2 * np.pi * PILOT_TONE_HZ * t)
        clips[clip_id] = gate0.wav_bytes(gate0.pcm16(samples))
    return clips


def _no_answer(client) -> bool:
    return hasattr(client, "last_status") and client.last_status is None


def run_pilot(
    client: gate0.Client,
    transport_check: Callable | None = None,
    timeout: Callable[[], float] = lambda: gate0.REQUEST_TIMEOUT_S,
) -> dict:
    """K1 on X1 and X2: audio tokens above zero and growing with the length (PROPOSAL § 5). A
    request without any HTTP answer goes to `transport_check` (`recover`) and, each time the
    server answers again, once more (`gate0.send_checked`); any other answer of
    `transport_check` (the host lost, the server gone, a stop) is `ModelFailed` at `smoke` with
    that reason. `timeout`
    is read for every request, so `timeout_cap` cuts a resend and the second clip too."""
    lines = []
    text = gate0.QUESTIONS["K1"]
    text_tokens = client.text_tokens(text)
    # only a count lost with its HTTP answer goes again: an answered failure stays None
    while text_tokens is None and transport_check is not None and _no_answer(client):
        decision = transport_check()
        if decision != "resend":
            raise ModelFailed("smoke", f"pilot: {decision}")
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

        def send(write_request: bool = True, body=body) -> dict:
            status, raw, error = client.request(gate0.CHAT_ROUTE, body, timeout=timeout())
            return {"sent": True, "http_status": status, "raw": raw, "error": error}

        answer, decision = gate0.send_checked(send, transport_check)
        if decision is not None:
            raise ModelFailed("smoke", f"pilot: {decision}")
        status, raw = answer["http_status"], answer["raw"]
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
                "error": answer["error"],
                "prompt_tokens": prompt,
                "audio_tokens": audio_tokens,
                "resent_after_tunnel": answer.get("resent_after_tunnel", False),
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
    reopen: Callable[..., object] = lambda force=False: None,
    stop: Callable[[], str | None] = lambda: None,
) -> None:
    """`/health` every second until 200; every `check_every` also the server state from the
    host, so a server that exited fails at once instead of at `timeout`. After each failed
    `/health`, `reopen` opens a tunnel whose `ssh` exited again (a network loss kills it while
    the server still loads); every `REOPEN_EVERY_CHECKS`-th state check it opens the tunnel
    anew even when its `ssh` lives, since a half-dead `ssh -N` accepts and carries nothing.
    A reason `stop` names (`soft_stop`) ends the wait at once: `ModelSkipped` (#165)."""
    started = time.monotonic()
    ends, next_check = started + timeout, started + check_every
    checks = 0
    while client.request("/health", timeout=5)[0] != 200:
        reason = stop()
        if reason is not None:
            raise ModelSkipped("smoke", f"stopped while the server loads: {reason}")
        now = time.monotonic()
        force = False
        if now >= next_check:
            next_check = now + check_every
            _judge(check(), now - started > start_grace)
            checks += 1
            force = checks % REOPEN_EVERY_CHECKS == 0
        reopen(force=force)
        if time.monotonic() > ends:
            raise VastError("the vLLM server did not answer /health in time")
        time.sleep(1.0)


def _private_file(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def fetch_server_evidence(run: "Run", name: str, out: str) -> None:
    """Log tails and the process list of the host into `evidence-<name>.json` and the journal,
    while the server tree still lives. A failure is journaled, not raised, so destroy always
    follows; a cancellation (KeyboardInterrupt, SystemExit) is journaled and propagates."""
    host, attempt, journal = run.host, run.attempt, run.journal
    timeout = run.intervals["evidence_timeout_s"]
    try:
        found = transport_retry(
            run,
            f"evidence {name}",
            lambda left: _checked_object(
                host.run(["python3", HOSTRUN, "evidence", "--dir", out], min(timeout, left)),
                "evidence",
            ),
            step_until(run, timeout, tail=True),
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
        return server_state(host, out, STATE_TIMEOUT_S)
    except Exception as error:
        errors = detail.setdefault("state_errors", {"count": 0, "last": None})
        errors["count"] += 1
        errors["last"] = redact(f"{type(error).__name__}: {error}", journal.key)
        return None


@dataclasses.dataclass
class Run:
    """What the steps of one attempt share once it is opened."""

    api: Api
    journal: Journal
    package: dict
    models: Mapping
    config_: config.Config
    intervals: Mapping
    runner: Callable
    extended: tuple | None = None
    hooks: Mapping = dataclasses.field(default_factory=dict)
    host: object = None
    instance_id: int | None = None
    heartbeat: object = None
    fetching: set = dataclasses.field(default_factory=set)
    syncs: int = 0
    spend_seen: bool = False
    fatal: str | None = None
    cancel_code: int | None = None
    soft_end: float | None = None

    @property
    def attempt(self) -> Path:
        return self.journal.dir

    @property
    def opened(self) -> dict:
        return summarize(self.journal.replay())["opened"]

    def hook(self, name: str, **fields) -> None:
        if name in self.hooks:
            self.hooks[name](self, **fields)


CANCEL_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def cancel_on_signals(run: Run) -> dict:
    """SIGTERM and SIGHUP (a `systemctl stop`, a closed terminal) end the attempt like Ctrl-C:
    `SystemExit` takes the `finally` path, so the results are downloaded and the instance
    destroyed. The first signal counts; later ones are ignored, so they cannot cut that path
    short; `run.fatal` keeps the stop even where the exception is swallowed. Only the main
    thread can set handlers: elsewhere nothing changes."""
    if threading.current_thread() is not threading.main_thread():
        return {}

    def cancel(signum: int, _frame) -> None:
        for number in CANCEL_SIGNALS:
            signal.signal(number, signal.SIG_IGN)
        run.fatal = "cancelled"
        run.cancel_code = 128 + signum
        with contextlib.suppress(Exception):
            run.journal.append("signal", name=signal.Signals(signum).name)
        raise SystemExit(128 + signum)

    return {number: signal.signal(number, cancel) for number in CANCEL_SIGNALS}


def defer_signals(run: Run, saved: Mapping) -> None:
    """Inside the `finally` of the attempt (download, destroy) a signal only is journaled: that
    path is what the signal asks for, so it must not be cut short."""
    if not saved:
        return

    def note(signum: int, _frame) -> None:
        run.fatal = run.fatal or "cancelled"
        with contextlib.suppress(Exception):
            run.journal.append("signal", name=signal.Signals(signum).name, deferred=True)

    for number in CANCEL_SIGNALS:
        if signal.getsignal(number) is not signal.SIG_IGN:
            signal.signal(number, note)


def restore_signals(saved: Mapping) -> None:
    for number, handler in saved.items():
        signal.signal(number, handler)


def soft_end_unix(opened: Mapping) -> float | None:
    return opened.get("soft_end_unix")


def spent_estimate(run: Run, now: float | None = None) -> float | None:
    """The spend so far by the clock and the create's prices (the credit lags behind it):
    dph_total × hours since the create + all inbound and outbound traffic of the package."""
    state = summarize(run.journal.replay())
    requested = next((e for e in run.journal.replay() if e["event"] == "create_requested"), None)
    if requested is None or not state["ids"]:
        return None
    prices = requested.get("prices") or {}
    try:
        hours = ((time.time() if now is None else now) - requested["t"]) / 3600
        return estimate_usd(prices, hours, run.package["in_gb"], run.package["out_gb"])
    except VastError:
        return None


def soft_stop(run: Run) -> str | None:
    """Why the agent itself should end now, before the watchdog has to: the soft end, the soft
    spend (by clock or by the credit seen), a destroy the watchdog asked for, a fatal loss."""
    if run.fatal is not None:
        return run.fatal
    state = summarize(run.journal.replay())
    if state["closed"] is not None or state["destroy_requested"]:
        return "watchdog"
    soft = run.soft_end
    if soft is not None and time.time() >= soft:
        return "soft_end"
    limit = run.package.get("spend_soft_usd")
    if limit is not None:
        if run.spend_seen:
            return "spend"
        estimate = spent_estimate(run)
        if estimate is not None and estimate >= limit:
            return "spend"
    return None


REQUEST_PAST_SOFT_END_S = 120.0
MIN_REQUEST_TIMEOUT_S = 10.0


def spend_end_unix(run: Run) -> float | None:
    """When the clock estimate (`spent_estimate`) reaches `spend_soft_usd`, by the create's
    prices; None without a soft spend or a create."""
    limit = run.package.get("spend_soft_usd")
    if limit is None:
        return None
    requested = next((e for e in run.journal.replay() if e["event"] == "create_requested"), None)
    if requested is None:
        return None
    try:
        prices = requested.get("prices") or {}
        dph = _number(prices.get("dph_total"), "dph_total")
        traffic = estimate_usd(prices, 0.0, run.package["in_gb"], run.package["out_gb"])
    except VastError:
        return None
    if dph <= 0:
        return None
    return requested["t"] + (limit - traffic) / dph * 3600


def timeout_cap(run: Run) -> float:
    """A request may run at most `REQUEST_PAST_SOFT_END_S` past the soft end or the moment the
    clock estimate reaches the soft spend, whichever comes first, so the tail (in-flight
    request, evidence, stop, download) fits before the deadline and before the early stop."""
    ends = [t for t in (run.soft_end, spend_end_unix(run)) if t is not None]
    if not ends:
        return math.inf
    return max(MIN_REQUEST_TIMEOUT_S, min(ends) + REQUEST_PAST_SOFT_END_S - time.time())


def check_credit(run: Run) -> None:
    """The credit read only at a block's end; a failed read is journaled and never stops."""
    initial = run.opened["initial_credit"]
    try:
        credit = _number(run.api.user().get("credit"), "credit")
    except Exception as error:
        run.journal.append(
            "credit_seen", error=redact(f"{type(error).__name__}: {error}", run.api.key)
        )
        return
    spent = round(initial - credit, 6)
    estimate = spent_estimate(run)
    run.journal.append(
        "credit_seen",
        credit=credit,
        spent_usd=spent,
        estimate_usd=None if estimate is None else round(estimate, 4),
    )
    limit = run.package.get("spend_soft_usd")
    if limit is not None and spent >= limit:
        run.spend_seen = True


def stop_at(run: Run) -> float:
    """The latest the agent works before it stops itself: the soft end or the moment the clock
    estimate reaches the soft spend, else the deadline."""
    ends = [t for t in (run.soft_end, spend_end_unix(run)) if t is not None]
    return min(ends) if ends else run.opened["deadline_unix"]


def step_until(run: Run, own_s: float, *, tail: bool = False) -> float:
    """Until when a step may retry after transport errors. Before the stop: its own limit, but
    at least the SSH window `ssh_lost_s`, since a lost network is not the step's fault, and
    never past `stop_at`. Once the agent has stopped for any reason `soft_stop` names (also a
    credit-seen spend, a signal, `ssh_lost`, the watchdog), a `tail` step (sync, evidence,
    server stop, final download) gets only its own limit — what `spend_tail_s` and
    `stop_path_s` count — and any other step nothing more."""
    now = time.time()
    stop = now if soft_stop(run) is not None else stop_at(run)
    until = min(stop, now + max(own_s, run.intervals["ssh_lost_s"]))
    return max(until, now + own_s) if tail else until


def _watchdog_stop(run: Run, what: str, error: str) -> None:
    destroyed = watchdog_destroy(run.journal)
    if destroyed is not None:
        run.journal.append("transport_retry", what=what, error=error, outcome="watchdog")
        raise VastError(f"{what}: the watchdog destroyed the instance ({destroyed})")


def await_host(
    run: Run,
    what: str,
    until: float,
    error: str,
    *,
    tunnel: bool = False,
    since: float | None = None,
) -> bool:
    """After a transport error: `true` on the host every `ssh_retry_s` until it answers, for at
    most `ssh_lost_s` and never past `until` (no probe outlives it); then, with `tunnel`, the
    tunnel opened again (a live but stale `ssh` restarted). Only a host silent for the whole
    `ssh_lost_s` window is `ssh_lost` for the attempt (`HostLost`); a window `until` cut short
    is a `TransportError` of this step alone. A destroy the watchdog asks for, before or during
    the wait, ends it at once. Journaled as `transport_retry`; True when the host answered the
    first `true` (no outage seen). The wait itself gets the whole window from now; the silence
    that makes `ssh_lost` counts from the start of the failed try (`since`), because an ssh
    needs up to its timeout to notice a lost network and that time was silence too, but no
    earlier than `SSH_NOTICE_S` before now: before that a long try still had its connection."""
    _watchdog_stop(run, what, error)
    started = time.time()
    silent_from = started if since is None else max(min(since, started), started - SSH_NOTICE_S)
    ends = min(until, started + run.intervals["ssh_lost_s"])
    tries = 0
    while True:
        tries += 1
        probe = min(SSH_PROBE_S, max(1.0, ends - time.time()))
        try:
            done = run.host.run(["true"], timeout=probe)
        except OSError as refused:  # the station refused the fork: a probe without an answer
            if refused.errno not in LOCAL_ERRNOS:
                raise
            done = None
        if done is not None and not _transport_failed(done) and done.returncode == 0:
            break
        _watchdog_stop(run, what, error)
        if time.time() >= ends:
            waited = round(time.time() - started, 3)
            lost = time.time() - silent_from >= run.intervals["ssh_lost_s"] - 1.0
            if lost:
                run.fatal = run.fatal or "ssh_lost"
            run.journal.append(
                "transport_retry",
                what=what,
                error=error,
                outcome="ssh_lost" if lost else "step_end",
                waited_s=waited,
            )
            if lost:
                raise HostLost(f"the host does not answer over SSH ({what}, {waited:.0f} s)")
            raise TransportError(f"{what}: no SSH answer before the step's end ({waited:.0f} s)")
        time.sleep(max(0.0, min(run.intervals["ssh_retry_s"], ends - time.time())))
    reopened = False
    if tunnel:
        try:
            reopened = bool(run.host.ensure_tunnel(force=True))
        except OSError as refused:  # a refused fork: the next try finds the tunnel dead
            if refused.errno not in LOCAL_ERRNOS:
                raise
    run.journal.append(
        "transport_retry",
        what=what,
        error=error,
        outcome="retry",
        tries=tries,
        waited_s=round(time.time() - started, 3),
        tunnel_reopened=reopened,
    )
    return tries == 1


def transport_retry(
    run: Run, what: str, action: Callable[[float], object], until: float, *, tunnel: bool = False
):
    """The one recovery path of every host step: `action(left)` (`left`: the seconds to
    `until`; the first try gets at least `MIN_STEP_S`); after a `TransportError` the host is
    awaited (`await_host`) and the step runs again with what is left before `until` — the
    step's own end (`step_until`) — while that is at least `MIN_RETRY_S`. A step that failed
    `SSH_RETRIES` times while the host answered at once is the step's own failure (a missing
    remote file, say), and its last error is raised. A fork the station refused
    (`LOCAL_ERRNOS`) runs the step again after a pause that doubles from `LOCAL_RETRY_S` up to
    `ssh_retry_s`, while the pause ends before `until`; the host is not asked (#165)."""
    at_once = local = 0
    left = max(MIN_STEP_S, until - time.time())
    while True:
        tried = time.time()
        try:
            return action(left)
        except HostLost:
            raise
        except TransportError as error:
            failure = error
        except OSError as error:
            if error.errno not in LOCAL_ERRNOS:
                raise
            local += 1
            pause = min(run.intervals["ssh_retry_s"], LOCAL_RETRY_S * 2 ** (local - 1))
            if time.time() + pause >= until:
                raise
            run.journal.append(
                "transport_retry",
                what=what,
                error=f"{type(error).__name__}: {errno.errorcode.get(error.errno)}",
                outcome="local_retry",
                tries=local,
                pause_s=pause,
            )
            time.sleep(pause)
            left = max(MIN_RETRY_S, until - time.time())
            continue
        if time.time() >= until:
            raise failure
        error = redact(str(failure), run.journal.key)
        if await_host(run, what, until, error, tunnel=tunnel, since=tried):
            at_once += 1
            if at_once >= SSH_RETRIES:
                raise failure
        left = until - time.time()
        if left < MIN_RETRY_S:
            raise failure


def _health_within(run: Run, client: gate0.Client, seconds: float, state: Callable) -> int | None:
    """`/health` every second until 200 or `seconds` (never past `timeout_cap`) have passed;
    every `server_check_s` the server state too, so a server that exited ends the wait
    (`ServerExited`). After a failed `/health` a tunnel whose `ssh` exited opens again: a
    second loss right after the reopen kills it once more (#165)."""
    started = time.monotonic()
    ends = started + min(seconds, timeout_cap(run))
    next_check = started + run.intervals["server_check_s"]
    while True:
        health = client.request("/health", timeout=5)[0]
        now = time.monotonic()
        if health == 200 or now >= ends:
            return health
        if now >= next_check:
            next_check = now + run.intervals["server_check_s"]
            _judge(state(), late=True)
        run.host.ensure_tunnel()
        time.sleep(1.0)


def _exit_outcome(error: ServerExited) -> str:
    """`rss_limit` when `hostrun.py serve` killed vLLM for its RSS — the rule's own limit, as
    in 0007 § 3 — and `server_exited` for any other exit."""
    return "rss_limit" if error.reason == "rss_limit" else "server_exited"


def recover(
    run: Run, client: gate0.Client, out: str, name: str, after_timeout: bool = False
) -> str:
    """After a request without any HTTP answer (or, `after_timeout`, one that ran out of time):
    first the host (`await_host` up to `ssh_lost_s`, never past `stop_at`), then the server's
    state, then the
    tunnel — `/health` through the one in hand, else it is opened again (a live but stale `ssh`
    restarted) and `/health` asked for up to `RECOVER_HEALTH_S`. `resend` only when the server
    answers again; after a timeout a tunnel that still carries `/health` means the request
    itself was slow: `timeout`, nothing is resent."""
    if after_timeout and client.request("/health", timeout=5)[0] == 200:
        run.journal.append(
            "transport_check",
            model=name,
            after_timeout=True,
            tunnel_reopened=False,
            health=200,
            outcome="timeout",
        )
        return "timeout"
    detail: dict = {}
    state = _state_or_none(run.host, out, run.journal, detail)
    # past the soft end or the soft spend nothing is resent, so the tail stays bounded; a
    # rule limit (the timeout above, an RSS kill here) still keeps its own name
    stopped = soft_stop(run)
    outcome, reopened, health = None, False, None
    if state is None and stopped is None:
        try:
            await_host(run, f"request {name}", step_until(run, 0.0), "no server state")
            state = _state_or_none(run.host, out, run.journal, detail)
        except HostLost:
            outcome = "ssh_lost"
        except VastError:
            outcome = soft_stop(run) or "server_unreachable"
        stopped = soft_stop(run)  # the wait may have run into the soft end
    if outcome is None:
        try:
            _judge(state, late=True)
        except ServerExited as error:
            outcome = _exit_outcome(error)
    if outcome is None and stopped is not None:
        outcome = stopped
    if outcome is None and not after_timeout:
        health = client.request("/health", timeout=5)[0]
    if outcome is None and health != 200:
        reopened = run.host.ensure_tunnel(force=True)
        try:
            health = _health_within(
                run,
                client,
                RECOVER_HEALTH_S,
                lambda: _state_or_none(run.host, out, run.journal, detail),
            )
        except ServerExited as error:
            outcome = _exit_outcome(error)
    if outcome is None:
        outcome = "resend" if health == 200 else "server_unreachable"
    run.journal.append(
        "transport_check",
        model=name,
        after_timeout=after_timeout,
        tunnel_reopened=reopened,
        health=health,
        outcome=outcome,
    )
    return outcome


def wait_gpu_free(run: Run) -> dict:
    """GPU memory below `GPU_FREE_MIB` before the next server: the driver may free the last
    one's memory late, and vLLM checks the free memory at start."""
    limit = run.intervals["gpu_free_timeout_s"]
    ends = time.monotonic() + limit
    argv = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    while True:
        done = transport_retry(
            run,
            "gpu_free",
            lambda left: _answered(run.host.run(argv, min(SSH_PROBE_S, left)), "nvidia-smi"),
            step_until(run, limit),
        )
        try:
            used = (
                sum(int(v) for v in done.stdout.decode().split()) if done.returncode == 0 else None
            )
        except ValueError:
            used = None
        if used is not None and used < GPU_FREE_MIB:
            return {"memory_used_mib": used}
        if time.monotonic() > ends:
            raise ModelFailed("gpu_free", f"GPU memory still used: {used} MiB")
        time.sleep(run.intervals["gpu_free_poll_s"])


def sync(run: Run, label: str) -> None:
    """The host's logs so far into `sync-<n>.tar` and `host-sync-<n>/`; a failure is journaled."""
    run.syncs += 1
    number = run.syncs
    started = time.time()
    limit = run.intervals["sync_timeout_s"]
    try:
        found = transport_retry(
            run,
            f"sync-{number}",
            lambda left: download(
                run.host, run.attempt, run.journal, min(limit, left), name=f"sync-{number}"
            ),
            step_until(run, limit, tail=True),
        )
    except Exception as error:
        run.journal.append(
            "sync",
            n=number,
            label=label,
            ok=False,
            error=redact(f"{type(error).__name__}: {error}", run.journal.key),
        )
        return
    run.journal.append(
        "sync", n=number, label=label, ok=True, elapsed_s=round(time.time() - started, 3), **found
    )


def _latest_attempt(day: str, config_: config.Config) -> Path | None:
    attempts = gate0.output_dir(day, config_) / gate0.ATTEMPTS_DIR
    numbered = [p for p in attempts.iterdir() if p.name.isdigit()] if attempts.is_dir() else []
    return max(numbered, key=lambda p: int(p.name), default=None)


def run_extended(run: Run, entry: Mapping, client: gate0.Client, out: str) -> dict:
    """The blocks of the extended set on the model's live server, until the model's slot, the
    soft end or a stop; the host's logs sync after each block."""
    name = entry["name"]
    directory, manifest, entries = run.extended
    slot = entry.get("slot_minutes")
    slot_end = None if slot is None else time.time() + slot * 60
    initial = None
    if soft_stop(run) is None:  # past the stop every block is not_sent: no probe in the tail
        try:
            _judge(_state_or_none(run.host, out, run.journal, {}), late=True)
        except ServerExited:
            initial = "server_exited"

    def stop() -> str | None:
        reason = initial or soft_stop(run)
        if reason is None and slot_end is not None and time.time() >= slot_end:
            reason = "slot_end"
        return reason

    def block_end(block: str, record: dict) -> None:
        run.journal.append("block_done", model=name, block=block, **record)
        run.hook("block_end", model=name, block=block)
        sync(run, f"{name}:{block}")
        if soft_stop(run) is None:
            check_credit(run)

    attempt_name = f"{run.opened['attempt_id']}-{name}"
    target = gpu_ext.new_attempt(directory, attempt_name, run.config_)
    with _step(run.journal, "extended", model=name) as detail:
        detail["attempt"] = attempt_name
        summary = gpu_ext.run_blocks(
            directory,
            manifest,
            entries,
            target,
            client,
            run.config_,
            stop_check=stop,
            transport_check=lambda after_timeout=False: recover(
                run, client, out, name, after_timeout
            ),
            on_block_end=block_end,
            timeout_cap=lambda: timeout_cap(run),
            timeouts={
                "closed": run.package.get("closed_timeout_s", gpu_ext.CLOSED_TIMEOUT_S),
                "review": run.package.get("review_timeout_s", gpu_ext.REVIEW_TIMEOUT_S),
            },
        )
        record = {
            "kind": "gpu_ext_run",
            "model": name,
            "attempt": run.opened["attempt_id"],
            "instance_id": run.instance_id,
            "extended": run.package["extended"],
            "summary": summary,
        }
        gate0._write_json(target, gpu_ext.RUN_FILE, record, run.config_)
        detail.update(stop_reason=summary["stop_reason"], blocks=summary["blocks"])
    try:
        found = gpu_ext.analyze(
            directory, target, gate0_attempt=_latest_attempt(entry["gate0_date"], run.config_)
        )
        gate0._write_json(target, gpu_ext.ANALYSIS_FILE, found, run.config_)
    except Exception as error:
        run.journal.append(
            "analysis_failed", model=name, error=redact(f"{type(error).__name__}: {error}", None)
        )
    return summary


def serve_model(
    run: Run,
    entry: Mapping,
    receipt: Path,
    after_gate0=lambda: None,
    result: dict | None = None,
) -> dict:
    """Steps `smoke`, `run` (its Gate 0 day) and `extended` of one model on a server of its
    own. The host's evidence is journaled and the server stops whatever happens; a server that
    does not stop is `AttemptFatal`, so the next model never starts beside it. A failure of
    this model is `ModelFailed` with its step. `result` (the caller's dict) holds the Gate 0
    verdict as soon as the day is done, so a cancellation after it still keeps it."""
    host, journal, intervals = run.host, run.journal, run.intervals
    name = entry["name"]
    out = _model_out(name)
    deadline = run.opened["deadline_unix"]
    serve = ["python3", HOSTRUN, "serve", "--deadline", str(deadline)]
    serve += ["--rss-limit", str(entry["rss_limit_bytes"]), "--stats", f"{out}/server-stats.jsonl"]
    serve += ["--log", f"{out}/server.log", "--guard-log", f"{out}/guard.log"]
    serve += ["--pid-file", f"{out}/serve.pid", "--"]
    serve += vllm_argv(f"{WORKSPACE}/models/{name}", name, host.server_port)
    result = {} if result is None else result
    fetched = False
    health_s = entry.get("health_timeout_s") or intervals["health_timeout_s"]

    def state_now(detail: dict) -> dict | None:
        """The server state while it loads: a lost network is awaited and the tunnel reopened
        (`transport_retry`); any other failure is counted and read as unknown."""
        try:
            return transport_retry(
                run,
                f"state {name}",
                lambda left: server_state(host, out, min(STATE_TIMEOUT_S, left)),
                step_until(run, STATE_TIMEOUT_S),
                tunnel=True,
            )
        except HostLost:
            raise
        except VastError as error:
            if watchdog_destroy(journal) is not None:
                raise
            errors = detail.setdefault("state_errors", {"count": 0, "last": None})
            errors["count"] += 1
            errors["last"] = redact(f"{type(error).__name__}: {error}", journal.key)
            return None

    try:
        try:
            with _step(journal, "smoke", model=name) as detail:
                transport_retry(
                    run,
                    f"serve {name}",
                    lambda left: host.spawn(serve, f"{out}/hostrun.log"),
                    step_until(run, intervals["server_start_s"]),
                )
                port = host.tunnel(host.server_port)
                client = gate0.Client(port, name)
                try:
                    _wait_health(
                        client,
                        health_s,
                        lambda: state_now(detail),
                        intervals["server_check_s"],
                        intervals["server_start_s"],
                        host.ensure_tunnel,
                        lambda: soft_stop(run),
                    )
                except ServerExited as error:
                    detail["server_exit"] = {"reason": error.reason, "returncode": error.returncode}
                    raise ModelFailed("smoke", str(error)) from None
                except ModelSkipped:
                    raise
                except VastError as error:
                    raise ModelFailed("smoke", str(error)) from None
                run.hook("pilot", model=name)
                pilot = run_pilot(
                    client,
                    lambda after_timeout=False: recover(run, client, out, name, after_timeout),
                    lambda: min(gate0.REQUEST_TIMEOUT_S, timeout_cap(run)),
                )
                (run.attempt / f"pilot-{name}.json").write_text(json.dumps(pilot, indent=2) + "\n")
                detail.update(pilot_ok=pilot["ok"], tunnel_port=port)
                if not pilot["ok"]:
                    raise ModelFailed(
                        "smoke", f"pilot of {name}: audio tokens absent or not growing"
                    )
            with _step(journal, "run", model=name) as detail:
                verdict = gate0.run_spike(
                    entry["gate0_date"],
                    run.config_,
                    decision=Path(entry["decision"]),
                    runtime="vllm",
                    server_port=port,
                    served_model=name,
                    weights=expected_files(run.models, name),
                    weights_receipt=receipt,
                    remote={"instance_id": run.instance_id, "tunnel_port": port},
                    request_timeout=run.package.get(
                        "gate0_request_timeout_s", gate0.REQUEST_TIMEOUT_S
                    ),
                    stop_check=lambda: soft_stop(run),
                    transport_check=lambda after_timeout=False: recover(
                        run, client, out, name, after_timeout
                    ),
                    timeout_cap=lambda: timeout_cap(run),
                )
                detail.update(
                    verdict=verdict["verdict"],
                    reason=verdict["reason"],
                    stop_reason=verdict.get("stop_reason"),
                )
                result.update(
                    verdict=verdict["verdict"],
                    reason=verdict["reason"],
                    stop_reason=verdict.get("stop_reason"),
                )
                if soft_stop(run) is None:  # past the stop the tail takes no extra probe
                    try:
                        _judge(_state_or_none(host, out, journal, detail), late=True)
                    except ServerExited as error:
                        detail["server_exit"] = {
                            "reason": error.reason,
                            "returncode": error.returncode,
                        }
            after_gate0()
            if run.extended is not None:
                try:
                    summary = run_extended(run, entry, client, out)
                    result["extended_stop"] = summary["stop_reason"]
                except Exception as error:
                    # the Gate 0 verdict stays this model's result; the blocks' failure is noted
                    result["extended_error"] = redact(
                        f"{type(error).__name__}: {error}", run.api.key
                    )
            fetched = True
            fetch_server_evidence(run, name, out)
        except BaseException:
            if not fetched:
                fetch_server_evidence(run, name, out)
            raise
    finally:
        stop = ["python3", HOSTRUN, "stop", "--pid-file", f"{out}/serve.pid"]
        stop += ["--stats", f"{out}/server-stats.jsonl"]
        last: dict = {}
        limit = intervals["stop_timeout_s"]

        def stop_server(left: float) -> str:
            last["done"] = done = host.run(stop, min(limit, left))
            return _checked(done, "stop")

        failed = None
        try:
            transport_retry(run, f"stop {name}", stop_server, step_until(run, limit, tail=True))
        except VastError as error:
            failed = error
        done = last.get("done")
        with contextlib.suppress(BaseException):
            journal.append(
                "server_stopped",
                model=name,
                returncode=None if done is None else done.returncode,
                timed_out=None if done is None else done.timed_out,
                error=None if failed is None else redact(str(failed), journal.key),
            )
        # a server left running is fatal whether or not this model failed first; only a
        # cancellation in flight keeps its own exception
        cancelled = isinstance(sys.exc_info()[1], KeyboardInterrupt | SystemExit)
        if failed is not None and not cancelled:
            destroyed = watchdog_destroy(journal)
            if destroyed is None and (
                run.fatal == "ssh_lost" or isinstance(failed, TransportError)
            ):
                raise AttemptFatal(
                    f"the host does not answer over SSH; the server of {name} could not be "
                    "stopped from here",
                    result,
                )
            if destroyed is not None:
                raise AttemptFatal(
                    f"the watchdog destroyed the instance ({destroyed}); the server of {name} "
                    "ended with it",
                    result,
                )
            raise AttemptFatal(
                f"the vLLM server of {name} did not stop; no next model starts", result
            )
    return result


def watchdog_destroy(journal: Journal) -> str | None:
    """The reason of a destroy the watchdog asked for, if any (the agent's own is `agent`)."""
    reasons = [
        e.get("reason")
        for e in journal.replay()
        if e["event"] == "destroy_requested" and e.get("reason") != "agent"
    ]
    return reasons[0] if reasons else None


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


def download(
    host, attempt: Path, journal: Journal, timeout: float, *, name: str = "results"
) -> dict:
    """`hostrun.py pack` of `/workspace/out` into `<name>.tar`, copied to the station, its
    SHA-256 and every file checked into `host/` (the final one) or `host-<name>/` (a sync)."""
    remote = f"{WORKSPACE}/{name}.tar"
    ends = time.monotonic() + timeout
    printed = _checked_object(
        host.run(
            ["python3", HOSTRUN, "pack", "--root", f"{WORKSPACE}/out", "--out", remote], timeout
        ),
        "pack",
    )
    local = attempt / f"{name}.tar"
    left = ends - time.monotonic()
    if left <= 0:
        raise VastError(f"no time left to copy {name}.tar within {timeout:.0f} s")
    host.get(remote, local, left)
    digest = _sha256(local)
    if digest != printed.get("archive_sha256"):
        raise VastError("downloaded archive SHA-256 differs from the host's")
    target = attempt / (HOST_DIR if name == "results" else f"host-{name}")
    if target.exists():  # an earlier try of this download, cut short: kept beside it
        number = 1
        while target.with_name(f"{target.name}-failed-{number}").exists():
            number += 1
        os.rename(target, target.with_name(f"{target.name}-failed-{number}"))
    files = verify_archive(local, target)
    found = {
        "archive_sha256": digest,
        "files": files,
        "disk_free_bytes": printed.get("disk_free_bytes"),
    }
    if name == "results":
        journal.append("download_verified", **found)
    return found


def ensure_weights(run: Run, entry: Mapping, late_unix: float | None) -> Path:
    name = entry["name"]
    with _step(run.journal, "weights", model=name) as detail:
        if name not in run.fetching:
            start_weights(run, name)
        return finish_fetch(
            run,
            name,
            timeout=entry.get("weights_timeout_s") or run.intervals["weights_timeout_s"],
            late_unix=late_unix,
            detail=detail,
        )


def start_weights(run: Run, name: str) -> None:
    transport_retry(
        run,
        f"start weights {name}",
        lambda left: start_fetch(run.host, run.journal, name, run.models),
        step_until(run, run.intervals["command_timeout_s"]),
    )
    run.fetching.add(name)


def _failed_step(journal: Journal, name: str) -> str | None:
    failed = [
        e["name"]
        for e in journal.replay()
        if e["event"] == "step" and e.get("model") == name and not e["ok"]
    ]
    return failed[-1] if failed else None


def _verdict_outcome(result: Mapping) -> dict:
    """`ok` for a Gate 0 verdict; `incomplete` for a day the stack stopped (no verdict, 0028)."""
    if result.get("verdict") == "incomplete":
        return {"outcome": "incomplete", "step": "run", "reason": result.get("reason")}
    return {"outcome": "ok", "step": None, "reason": None}


def _cancelled(run: Run, name: str, result: Mapping, reason: str) -> None:
    """A cancellation once the model's Gate 0 day is done keeps its verdict (`cancelled` names
    the signal and the step it cut); before that the model is `cancelled` (0028 Д.9 п. 6)."""
    step = _failed_step(run.journal, name)
    if result.get("verdict"):
        outcome = {"model": name, **_verdict_outcome(result), **result}
        outcome["cancelled"] = {"reason": reason, "step": step}
    else:
        outcome = {"model": name, "outcome": "cancelled", "step": step, "reason": reason}
    run.journal.append("model_outcome", **outcome)


def run_models(run: Run) -> list[dict]:
    """Each model in turn, isolated: a failure of one is its `model_outcome` and the next still
    runs; only `AttemptFatal`, a lost host and a cancellation end the loop. The next model's
    weights start downloading in the background once this model's Gate 0 day is done."""
    entries = run.package["models"]
    outcomes = []
    soft = run.soft_end
    late_minutes = run.package.get("weights_late_minutes")

    def start_next(index: int) -> None:
        # past the stop the next model never runs: its weights would only lengthen the tail
        if index + 1 < len(entries) and soft_stop(run) is None:
            name = entries[index + 1]["name"]
            if name not in run.fetching:
                try:
                    start_weights(run, name)
                except Exception as error:
                    run.journal.append("weights_start_failed", model=name, error=str(error))

    for index, entry in enumerate(entries):
        name = entry["name"]
        reason = soft_stop(run)
        if reason is not None:
            outcome = {"model": name, "outcome": "skipped", "step": None, "reason": reason}
        else:
            result: dict = {}
            try:
                if index > 0:
                    run.journal.append("gpu_free", model=name, **wait_gpu_free(run))
                late = None
                if index > 0 and soft is not None and late_minutes is not None:
                    late = soft - late_minutes * 60
                receipt = ensure_weights(run, entry, late)
                serve_model(run, entry, receipt, lambda index=index: start_next(index), result)
                outcome = {"model": name, **_verdict_outcome(result), **result}
            except AttemptFatal as error:
                kept = error.result
                if kept.get("verdict"):
                    outcome = {"model": name, **_verdict_outcome(kept), **kept}
                else:
                    step = _failed_step(run.journal, name)
                    outcome = {
                        "model": name,
                        "outcome": "failed",
                        "step": step,
                        "reason": str(error),
                    }
                outcome["fatal"] = str(error)
                run.journal.append("model_outcome", **outcome)
                raise
            except (KeyboardInterrupt, SystemExit) as error:
                _cancelled(run, name, result, f"cancelled: {type(error).__name__}")
                raise
            except Exception as error:
                if run.fatal == "cancelled":
                    # the signal's SystemExit lost to an error it caused on its way up
                    _cancelled(run, name, result, f"cancelled: {type(error).__name__}")
                    raise SystemExit(run.cancel_code or 128 + signal.SIGTERM) from error
                step = getattr(error, "step", None) or _failed_step(run.journal, name)
                skipped = step == "gpu_free" or isinstance(error, ModelSkipped)
                kind = "skipped" if skipped else "failed"
                message = redact(getattr(error, "reason", None) or str(error), run.api.key)
                outcome = {"model": name, "outcome": kind, "step": step, "reason": message}
                if run.fatal == "ssh_lost":  # `await_host` gave the host its whole window
                    run.journal.append("model_outcome", **outcome)
                    raise AttemptFatal("the host does not answer over SSH") from None
            else:
                if run.fatal == "cancelled":  # a step swallowed the signal's exit (the blocks)
                    _cancelled(run, name, result, "cancelled: swallowed")
                    raise SystemExit(run.cancel_code or 128 + signal.SIGTERM)
        run.journal.append("model_outcome", **outcome)
        outcomes.append(outcome)
        if reason is None:
            start_next(index)
    return outcomes


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
    hooks: Mapping | None = None,
) -> Journal:
    """Every step of one attempt. Each model is isolated (`run_models`); whatever happens the
    host's results are downloaded while the host answers and the watchdog is not destroying,
    then destroy and the absence check follow. Raises when no model gave a Gate 0 verdict."""
    package, package_sha = load_package(package_path, models, real=not dry_run)
    if not dry_run and not (approval_url or "").startswith("https://github.com/"):
        raise VastError("deploy needs --approval with the link to the owner's approval")
    check_days(package, config_)
    extended = check_extended(package, config_)
    gpu = gpu_root(config_)
    refuse_open_attempts(gpu)
    initial_credit = _number(api.user().get("credit"), "credit")
    attempt_id = new_attempt_id()
    journal = Journal.create(gpu, attempt_id, api.key)
    now = time.time()
    soft = package.get("soft_end_minutes")
    journal.append(
        "opened",
        attempt_id=attempt_id,
        label=LABEL_PREFIX + attempt_id,
        package_sha256=package_sha,
        approval_url=approval_url,
        preregistration=package["preregistration"],
        offer_id=offer_ids(package)[0],
        offer_ids=offer_ids(package),
        envelope=package["envelope"],
        ceiling_usd=package["ceiling_usd"],
        early_stop_usd=package["early_stop_usd"],
        spend_soft_usd=package.get("spend_soft_usd"),
        initial_credit=initial_credit,
        deadline_unix=round(now + package["deadline_minutes"] * 60, 3),
        soft_end_unix=None if soft is None else round(now + soft * 60, 3),
        image_ref=IMAGE_REF,
        models={e["name"]: models[e["name"]]["revision"] for e in package["models"]},
        extended=package.get("extended"),
        python=sys.executable,
        api_base=api.base,
        key_path=str(Path(key_path).absolute()),
        dry_run=dry_run,
    )
    run = Run(api, journal, package, models, config_, intervals, runner, extended, hooks or {})
    run.soft_end = soft_end_unix(run.opened)
    run.heartbeat = Heartbeat(journal.dir, intervals["heartbeat_s"]).start()
    attempt, failure, outcomes = journal.dir, None, []
    timeout = intervals["command_timeout_s"]
    saved_signals = cancel_on_signals(run)

    def tick(_: Mapping) -> Api:
        return api

    try:
        need = round(package["ceiling_usd"] + CREDIT_MARGIN_USD, 6)
        if initial_credit < need:
            journal.append("credit_refused", credit=initial_credit, need_usd=need)
            raise VastError(
                f"the account credit {initial_credit:.2f} USD is below the ceiling plus "
                f"{CREDIT_MARGIN_USD:.2f} USD ({need:.2f} USD); no create"
            )
        with _step(journal, "arm"):
            arm(journal, runner, dry_run=dry_run)
            if watch_tick(journal, tick, runner=runner) != "idle":
                raise VastError("the manual watchdog tick before create did not pass")
        with _step(journal, "create") as detail:
            public_key = _keypair(attempt)
            try:
                detail["instance_id"] = run.instance_id = create_instance(
                    api, journal, package, runner
                )
            except CreateOutcomeUnknown:
                watch_tick(journal, tick, runner=runner)
                raise
        with _step(journal, "ready") as detail:
            instance = wait_ready(api, run.instance_id, intervals)
            api.attach_ssh(run.instance_id, public_key)
            run.host = make_host(instance, attempt)
            run.hook("host")
            wait_ssh(run.host, intervals)
            detail["status"] = instance.get("actual_status")
        with _step(journal, "preflight") as detail:
            detail["host_files"] = transport_retry(
                run,
                "upload",
                lambda left: upload_host_files(
                    run.host, attempt, models, api.key, min(timeout, left)
                ),
                step_until(run, timeout),
            )
            transport_retry(
                run,
                "preflight",
                lambda left: preflight(run.host, package, models, min(timeout, left), detail),
                step_until(run, timeout),
            )
        outcomes = run_models(run)
    except BaseException as error:
        failure = error
    finally:
        defer_signals(run, saved_signals)
        if run.host is not None:
            state = summarize(journal.replay())
            if state["closed"] is None and not state["destroy_requested"]:
                # a network loss, and `systemctl stop` (KillMode=control-group) killing the ssh
                # or scp in flight, are transport errors: retried within the download's limit
                limit = intervals["download_timeout_s"]
                try:
                    with _step(journal, "download") as detail:
                        detail.update(
                            transport_retry(
                                run,
                                "download",
                                lambda left: download(run.host, attempt, journal, min(limit, left)),
                                step_until(run, limit, tail=True),
                            )
                        )
                except BaseException as error:
                    destroyed = watchdog_destroy(journal)
                    cut = destroyed and VastError(
                        f"final download cut: the watchdog destroyed the instance ({destroyed}); "
                        "only the syncs and the client's answers are on the station"
                    )
                    failure = failure or cut or error
            else:
                journal.append("download_skipped", reason="the watchdog is destroying the instance")
                failure = failure or VastError(
                    "final download skipped: the watchdog destroyed the instance; only the "
                    "syncs and the client's answers are on the station"
                )
            run.host.close()
        try:
            with _step(journal, "destroy"):
                destroy_until_absent(journal, tick, runner, intervals)
        except BaseException as error:
            failure = failure or error
        run.heartbeat.stop()
        try:
            read_credit_after_close(journal, api)
        except BaseException as error:
            failure = failure or error
        restore_signals(saved_signals)
    if failure is not None:
        raise failure
    if not any(o["outcome"] == "ok" for o in outcomes):
        reasons = "; ".join(f"{o['model']}: {o['step']}: {o['reason']}" for o in outcomes)
        raise VastError(f"no model gave a Gate 0 verdict: {reasons}")
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
    events = journal.replay()
    return {
        "attempt_id": state["opened"]["attempt_id"],
        "dry_run": state["opened"].get("dry_run"),
        "steps": steps,
        "models": [
            {k: v for k, v in e.items() if k not in ("event", "t")}
            for e in events
            if e["event"] == "model_outcome"
        ],
        "syncs": [
            {k: e.get(k) for k in ("n", "label", "ok", "files")}
            for e in events
            if e["event"] == "sync"
        ],
        "closed": state["closed"],
        "credit_after_close": state["credit_after_close"],
    }


# --- the dry run ---------------------------------------------------------------------------

DRY_RUN_DAYS = (
    ("qwen25omni", "2000-01-01", "docs/decisions/0007-gate0-result.md", 17_500_000_000),
    ("qwen3omni", "2000-01-02", "docs/decisions/0023-gate0-qwen3-omni.md", 25_500_000_000),
)
SHIM = """#!{python}
import os
import sys
os.environ.setdefault("SS_FAKE_ROOT", "{root}")
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
        (bin_dir / tool).write_text(SHIM.format(python=sys.executable, tool=tool, root=root))
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


FULL_DAYS = (
    ("qwen25omni", "2000-01-01", "docs/decisions/0024-gate0-qwen25-omni.md", 17_500_000_000),
    ("qwen3omni", "2000-01-02", "docs/decisions/0023-gate0-qwen3-omni.md", 25_500_000_000),
)
DRY_PACKAGE = {
    "offer_ids": [vast_fake.OFFER_ID],
    "envelope": {"dph_total": {"lte": 1.533}, "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 79000}},
    "ceiling_usd": 4.0,
    "early_stop_usd": 2.5,
    "spend_soft_usd": 2.0,
    "deadline_minutes": 100,
    "soft_end_minutes": 85,
    "weights_late_minutes": 1,
    "disk_gb": 150,
    "in_gb": 102.95,
    "out_gb": 1.0,
    "disk_margin_bytes": 0,
    "closed_timeout_s": 5.0,
    "review_timeout_s": 5.0,
    "gate0_request_timeout_s": 30.0,
    "preregistration": "dry run: no preregistration",
}


def _dry_transcribe(path, run, config_):
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


def _dry_days(dry_config: config.Config, full: bool) -> list[dict]:
    """The Gate 0 days of a dry run: controls only, or full days of synthetic voices as the
    local runs had them (B1–B3, V1–V3 and the controls; the second day a byte copy)."""
    entries = []
    if full:
        from sound_scout.smoke import speech_like_control

        for name in gate0.SOURCE_FILES:
            voice = speech_like_control(3.0, gate0.SAMPLE_RATE, f0=130.0)
            (dry_config.allowed_roots[0] / name).write_bytes(gate0.wav_bytes(gate0.pcm16(voice)))
        (first, day, decision, rss), (second, day2, decision2, rss2) = FULL_DAYS
        v1 = gate0.prepare_clips(
            day, dry_config, decision=REPO_ROOT / decision, transcribe=_dry_transcribe
        )
        v2 = gate0.reuse_clips(day2, dry_config, source=day, decision=REPO_ROOT / decision2)
        for name, day_, document, limit, manifest in (
            (first, day, decision, rss, v1),
            (second, day2, decision2, rss2, v2),
        ):
            pins = {
                clip["id"]: gate0._sha256_file(gate0.output_dir(day_, dry_config) / clip["file"])
                for clip in manifest["clips"]
            }
            entries.append(
                {
                    "name": name,
                    "gate0_date": day_,
                    "decision": str(REPO_ROOT / document),
                    "rss_limit_bytes": limit,
                    "clips": pins,
                }
            )
        return entries
    for name, day, decision, rss in DRY_RUN_DAYS:
        document = REPO_ROOT / decision
        if not document.is_file():
            raise VastError(f"dry run needs the source tree: {decision} is missing")
        gate0.prepare_clips(day, dry_config, decision=document, controls_only=True)
        entries.append(
            {"name": name, "gate0_date": day, "decision": str(document), "rss_limit_bytes": rss}
        )
    return entries


def dry_run(out: Path | None, config_: config.Config, *, scenario: Mapping | None = None) -> dict:
    """The full timeline against the fakes; writes `timeline.json` and `fake-api.jsonl`.

    `scenario` (the rehearsal's knobs, all optional): `full_days`; `extended` (False: no
    extended set); `fixture` (arguments of `gpu_ext.fixture`, e.g. the 717-request shape);
    `manifest_code_sha256` (the set as another `gpu_ext.py` built it); `vllm`, `hf`, `gpu`
    (the fake mode files); `package` and `models` (changes
    to the package and to each model entry); `create_mode`; `fake` (a callable on the
    `FakeVast`); `hooks` (of `run_attempt`); `intervals`. A failure still writes both files
    and then propagates."""
    scenario = scenario or {}
    if out is None:
        out = gpu_root(config_) / f"dry-run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = Path(out).absolute()
    out.mkdir(mode=0o700, parents=True)
    (out / "in").mkdir(mode=0o700)
    dry_config = dataclasses.replace(
        config_, data_root=out.resolve(), allowed_roots=((out / "in").resolve(),)
    )
    key_file = out / "fake-api-key"
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, secrets.token_urlsafe(32).encode() + b"\n")
    os.close(fd)
    key = read_api_key(key_file)
    root = out / "host-root"
    env = _fake_host_root(root)
    for name, file in (
        ("vllm", vast_fake.FAKE_MODE_FILE),
        ("hf", vast_fake.FAKE_HF_FILE),
        ("gpu", vast_fake.FAKE_GPU_FILE),
    ):
        if name in scenario:
            (root / file).write_text(json.dumps(scenario[name]))
    entries = _dry_days(dry_config, scenario.get("full_days", False))
    for entry, change in zip(entries, scenario.get("models", ()), strict=False):
        entry.update(change)
    package = {**DRY_PACKAGE, "models": entries}
    if scenario.get("extended", True):
        found = gpu_ext.fixture(out / "gpu-ext-fixture", **scenario.get("fixture", {}))
        if "manifest_code_sha256" in scenario:
            # the set as built by another gpu_ext.py: the package pins this manifest
            manifest_path = out / "gpu-ext-fixture" / gpu_ext.MANIFEST_FILE
            manifest = json.loads(manifest_path.read_bytes())
            manifest["code_sha256"] = scenario["manifest_code_sha256"]
            manifest_path.write_bytes(gpu_ext._json_bytes(manifest))
            found["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        package["extended"] = {
            "dir": "gpu-ext-fixture",
            "manifest_sha256": found["manifest_sha256"],
            "plan_sha256": found["plan_sha256"],
        }
    package.update(scenario.get("package", {}))
    package = {k: v for k, v in package.items() if v is not None}
    package_path = out / "package.json"
    package_path.write_text(json.dumps(package, indent=2) + "\n")
    fake = vast_fake.FakeVast(key, create_mode=scenario.get("create_mode", "ok"))
    if "fake" in scenario:
        scenario["fake"](fake)
    fake.start()
    started = time.monotonic()
    journal, failure = None, None
    try:
        journal = run_attempt(
            package_path,
            api=Api(fake.url, key),
            make_host=lambda instance, attempt: LocalHost(root, env),
            config_=dry_config,
            runner=_refuse_system,
            intervals={**DRY_INTERVALS, **scenario.get("intervals", {})},
            dry_run=True,
            key_path=key_file,
            models=vast_fake.synthetic_models(),
            hooks=scenario.get("hooks"),
        )
    except BaseException as error:
        failure = error
    finally:
        log = "".join(json.dumps(entry) + "\n" for entry in fake.log)
        (out / "fake-api.jsonl").write_text(log)
        remaining = sorted(i for i, inst in fake.instances.items() if fake._visible(inst))
        fake.stop()
    found_journals = sorted((out / GPU_DIR).glob(f"*/{JOURNAL_FILE}"))
    if journal is None and found_journals:
        journal = Journal(found_journals[-1])
    result = {
        **(timeline(journal) if journal is not None else {"steps": [], "closed": None}),
        "elapsed_s": round(time.monotonic() - started, 3),
        "journal": None if journal is None else str(journal.path.relative_to(out)),
        "fake_api": {
            "put": sum(1 for e in fake.log if e["method"] == "PUT"),
            "delete": sum(1 for e in fake.log if e["method"] == "DELETE"),
            "instances_left": remaining,
        },
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
        "note": "накладные расходы стека на станции; не время на Vast.ai",
    }
    (out / "timeline.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if failure is not None:
        raise failure
    return result


# --- the CLI -------------------------------------------------------------------------------

LAUNCH_TIMEOUT_STOP = "25min"


def launch_argv(
    package: Path, approval: str, config_path: Path, logs: Path, stamp: str, root: Path = REPO_ROOT
) -> list[str]:
    """The `systemd-run --user` argv of `deploy`, which outlives the tool session (up to the
    deadline + 15 min): the checkout as working directory (the package's decision paths are
    relative), `SOUND_SCOUT_CONFIG` absolute, the checkout's own `.venv` python, `systemd-inhibit`
    against sleep, stdout and stderr appended under `logs`. A `systemctl --user stop` sends
    SIGTERM, which `deploy` turns into download and destroy; `TimeoutStopSec` covers that path
    up to the DELETE (`stop_path_s`) before SIGKILL; the watchdog timers finish the absence
    check if it comes."""
    return [
        "systemd-run",
        "--user",
        f"--unit={LABEL_PREFIX}deploy-{stamp}",
        "--collect",
        f"--description=Sound Scout GPU deploy {package.name}",
        f"--working-directory={root}",
        f"--setenv={config.CONFIG_PATH_ENV}={config_path}",
        f"--property=TimeoutStopSec={LAUNCH_TIMEOUT_STOP}",
        f"--property=StandardOutput=append:{logs / 'deploy.out'}",
        f"--property=StandardError=append:{logs / 'deploy.err'}",
        "--",
        shutil.which("systemd-inhibit") or "systemd-inhibit",
        "--what=sleep:idle",
        "--who=sound-scout",
        "--why=GPU attempt: results are downloaded and the instance destroyed first",
        "--mode=block",
        str(root / ".venv" / "bin" / "python"),
        "-m",
        "sound_scout.eval.vast",
        "deploy",
        "--package",
        str(package),
        "--approval",
        approval,
    ]


def prepare_launch(package: Path, approval: str, root: Path = REPO_ROOT) -> dict:
    """The checks `deploy` runs before any API call (package, days, extended set and its code),
    then the launch argv and a fresh 0700 directory for its output; nothing is started."""
    if not approval.startswith("https://github.com/"):
        raise VastError("launch needs --approval with the link to the owner's approval")
    env = os.environ.get(config.CONFIG_PATH_ENV)
    if not env:
        raise VastError(f"launch needs {config.CONFIG_PATH_ENV} (0028, «Запуск»)")
    config_path = Path(env).absolute()
    if not (root / ".venv" / "bin" / "python").is_file():
        raise VastError(f"no .venv/bin/python in {root}")
    config_ = config.load()
    package = Path(package).absolute()
    loaded, package_sha = load_package(package, MODELS, real=True)
    check_days(loaded, config_)
    check_extended(loaded, config_)
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    logs = gpu_root(config_) / "launches" / stamp
    os.makedirs(logs.parent, 0o700, exist_ok=True)
    os.mkdir(logs, 0o700)
    argv = launch_argv(package, approval, config_path, logs, stamp, root)
    return {
        "argv": argv,
        "unit": argv[2].split("=", 1)[1],
        "logs": str(logs),
        "package_sha256": package_sha,
    }


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
    launch = commands.add_parser("launch-argv")
    launch.add_argument("--package", type=Path, required=True)
    launch.add_argument("--approval", required=True)
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
        if args.command == "launch-argv":
            print(json.dumps(prepare_launch(args.package, args.approval), ensure_ascii=False))
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
