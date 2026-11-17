"""Honest component state for `doctor` / `speech_capabilities` (R01) and the smoke-receipt.

`report` reads the state of the four components — `decoder`, `asr`, `acoustics`,
`audio_review` — without the network, without inference and without writing anything. Presence
of a binary, a package or weights never means `ready`: it gives `unvalidated` until a
smoke-receipt saved by the component's own smoke test (in its own task) matches the current
fingerprint. The fingerprint is the versions, the model snapshot, a hash of the component's
configuration section, the platform and the package version; any change makes the receipt stale.

Presence is established cheaply and read-only: `shutil.which` plus `ffmpeg -version` /
`ffprobe -version` through the bounded `proc.run` (the only two child processes of `doctor`),
`importlib.metadata.version` without importing the heavy packages, and `stat` of the snapshot
files in the local Hugging Face cache. `audio_review` is `permission_required` while no backend
is selected (Gate 0); missing weights are `unavailable`, because downloading them needs the
owner's permission (R09). No absolute path appears in a fingerprint: it goes into the receipt
and into the report.

`read_receipt` / `write_receipt` fix the receipt format, `<data_root>/receipts/<component>.json`
(mode 0600 in a 0700 directory). `write_receipt` is the only writer and `report` never calls
it; a receipt with a test double, an absolute path or a hand-edited fingerprint is refused or
read as invalid. The network guard is installed by the process entry point (CLI, MCP), not here:
`report` only tells whether it is active.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

from sound_scout import __version__, guard, paths, proc
from sound_scout.config import MIB, Config

COMPONENTS = ("decoder", "asr", "acoustics", "audio_review")
STATUSES = ("ready", "unavailable", "unvalidated", "permission_required")
RECEIPTS_DIR_NAME = "receipts"
RECEIPT_VERSION = 1
TEST_DOUBLE = "test-double"
MAX_RECEIPT_BYTES = 1 * MIB
VERSION_OUTPUT_BYTES = 64 * 1024
DISTRIBUTIONS = {
    "asr": ("faster-whisper", "ctranslate2"),
    "acoustics": ("praat-parselmouth", "onnxruntime", "faster-whisper"),
}
DECODER_TOOLS = ("ffmpeg", "ffprobe")
WEIGHT_FILES = ("model.bin", "config.json")
VAD_PACKAGE = "faster_whisper"
VAD_ASSET = re.compile(r"silero.*\.onnx$")
DIR_MODE = 0o700
GATE0_REASON = "audio backend is not selected or allowed (backend = none, Gate 0)"
_VERSION_LINE = re.compile(r"^(ffmpeg|ffprobe) version (\S+)")
_PLAIN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class ReceiptInvalid(Exception):
    """The receipt file exists but cannot be trusted; the message says why."""


class ReceiptRejected(Exception):
    """`write_receipt` refuses to write this receipt; the message says why."""


def installed_version(distribution: str) -> str | None:
    """Version of an installed distribution from its metadata, without importing it."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def package_dir(import_name: str) -> Path | None:
    """Directory of an importable package, found without importing it."""
    try:
        spec = find_spec(import_name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def hf_hub_cache() -> Path:
    """The local Hugging Face cache, resolved in the same order as `huggingface_hub`."""
    if cache := os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.path.expandvars(os.path.expanduser(cache)))
    xdg_cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    home = os.environ.get("HF_HOME") or os.path.join(xdg_cache, "huggingface")
    return Path(os.path.expandvars(os.path.expanduser(home))) / "hub"


def platform_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def component_config(config: Config, component: str) -> dict[str, object]:
    """The configuration section a component depends on; hashed into its fingerprint."""
    match component:
        case "decoder":
            return asdict(config.budgets)
        case "asr":
            return asdict(config.asr)
        case "acoustics":
            return {}
        case "audio_review":
            return {"backend": config.backend}
    raise ValueError(f"unknown component {component!r}; known: {list(COMPONENTS)}")


def config_hash(section: Mapping[str, object]) -> str:
    return _sha256(section)


def fingerprint_hash(fields: Mapping[str, object]) -> str:
    """SHA-256 of canonical JSON over every fingerprint field except `hash`."""
    return _sha256({key: value for key, value in fields.items() if key != "hash"})


def fingerprint(component: str, config: Config) -> dict[str, object]:
    """Current fingerprint of the component, `hash` last; `null` stands for what is missing."""
    fields, _ = _probe(component, config)
    return _with_hash(component, config, fields)


def component_status(component: str, config: Config, data_root: Path) -> dict[str, object]:
    """`status`, `reason`, `fingerprint`, `last_smoke_at` of one component, read-only."""
    fields, reason = _probe(component, config)
    current = _with_hash(component, config, fields)
    if component == "audio_review" and config.backend == "none":
        return _status("permission_required", GATE0_REASON, current, None)
    if reason is not None:
        return _status("unavailable", reason, current, None)
    try:
        receipt = read_receipt(component, data_root)
    except ReceiptInvalid as error:
        return _status("unvalidated", f"smoke-receipt is invalid: {error}", current, None)
    if receipt is None:
        return _status(
            "unvalidated",
            f"no smoke-receipt ({RECEIPTS_DIR_NAME}/{component}.json under the data root)",
            current,
            None,
        )
    smoke_at = receipt["smoke_at"]
    saved = receipt["fingerprint"]
    if saved["hash"] != current["hash"]:
        changed = ", ".join(_changed_keys(saved, current))
        reason = f"smoke-receipt of {smoke_at} is stale: changed {changed}"
        return _status("unvalidated", reason, current, smoke_at)
    return _status("ready", None, current, smoke_at)


def qualification_flags(data_root: Path) -> dict[str, bool]:
    """Quality qualification (R01), separate from smoke tests; no source of truth exists yet."""
    return {"validated_ru": False, "diction_validated_ru": False, "prosody_validated_ru": False}


def report(config: Config) -> dict[str, object]:
    """The whole diagnostic picture; reads receipts, writes nothing, installs no guard."""
    return {
        "sound_scout": __version__,
        "components": {
            component: component_status(component, config, config.data_root)
            for component in COMPONENTS
        },
        **qualification_flags(config.data_root),
        "network_guard": {
            "active": guard.network_guard_active(),
            "network_allow": config.network.allow,
            "offline_env": {name: os.environ.get(name) for name in guard.OFFLINE_ENV},
        },
    }


def read_receipt(component: str, data_root: Path) -> dict[str, object] | None:
    """The receipt of a component: `None` when there is none, `ReceiptInvalid` when untrusted."""
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; known: {list(COMPONENTS)}")
    receipts_dir = data_root / RECEIPTS_DIR_NAME
    path = receipts_dir / f"{component}.json"
    try:
        dir_mode = os.lstat(receipts_dir).st_mode
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ReceiptInvalid(f"cannot read receipt: {error.strerror}") from None
    if stat.S_ISLNK(dir_mode):
        raise ReceiptInvalid(f"{RECEIPTS_DIR_NAME} directory is a symlink")
    if not stat.S_ISDIR(dir_mode):
        raise ReceiptInvalid(f"{RECEIPTS_DIR_NAME} is not a directory")
    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ReceiptInvalid(f"cannot read receipt: {error.strerror}") from None
    if stat.S_ISLNK(file_stat.st_mode):
        raise ReceiptInvalid("receipt file is a symlink")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ReceiptInvalid("receipt is not a regular file")
    if file_stat.st_size > MAX_RECEIPT_BYTES:
        raise ReceiptInvalid(f"receipt is larger than {MAX_RECEIPT_BYTES} bytes")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as source:
            raw = source.read(MAX_RECEIPT_BYTES + 1)
    except OSError as error:
        raise ReceiptInvalid(f"cannot read receipt: {error.strerror}") from None
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ReceiptInvalid(f"receipt is larger than {MAX_RECEIPT_BYTES} bytes")
    try:
        receipt = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ReceiptInvalid("receipt is not valid JSON") from None
    _check_receipt(receipt, component)
    return receipt


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not valid JSON")


def write_receipt(
    component: str, config: Config, smoke: Mapping[str, object], *, now: datetime | None = None
) -> Path:
    """Save the receipt of a smoke test that just passed; the only writer of receipts.

    Refused (`ReceiptRejected`): an unknown component, `audio_review` without a backend, a
    component that is not even present, a test double or an absolute path inside the receipt.
    The file is written privately under the data root and replaces the old receipt atomically.
    """
    if component not in COMPONENTS:
        raise ReceiptRejected(f"unknown component {component!r}; known: {list(COMPONENTS)}")
    if component == "audio_review" and config.backend == "none":
        raise ReceiptRejected(GATE0_REASON)
    fields, reason = _probe(component, config)
    if reason is not None:
        raise ReceiptRejected(f"component is unavailable: {reason}")
    current = _with_hash(component, config, fields)
    if _contains_test_double(current):
        raise ReceiptRejected(f"fingerprint contains the string {TEST_DOUBLE!r}")
    if _contains_test_double(smoke):
        raise ReceiptRejected(f"smoke contains the string {TEST_DOUBLE!r}")
    if (found := _absolute_path_in(smoke)) is not None:
        raise ReceiptRejected(f"smoke contains an absolute path: {found!r}")
    stamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "component": component,
        "status": "ok",
        "smoke_at": stamp,
        "fingerprint": current,
        "smoke": dict(smoke),
    }
    try:
        text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ReceiptRejected(f"smoke is not JSON-serializable: {error}") from None
    receipts_dir = config.data_root / RECEIPTS_DIR_NAME
    _ensure_private_dir(receipts_dir)
    staging_name = f".{component}.json.{uuid.uuid4().hex}"
    target = receipts_dir / f"{component}.json"
    with paths.open_output(receipts_dir, staging_name, config) as out:
        try:
            out.write(text.encode("utf-8") + b"\n")
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            os.unlink(receipts_dir / staging_name)
            raise
    os.replace(receipts_dir / staging_name, target)
    return target


def _probe(component: str, config: Config) -> tuple[dict[str, object], str | None]:
    """Component-specific fingerprint fields and the first reason it is not present."""
    match component:
        case "decoder":
            return _probe_decoder(config)
        case "asr":
            return _probe_asr(config)
        case "acoustics":
            return _probe_acoustics()
        case "audio_review":
            backend = {"name": config.backend, "model": None, "revision": None}
            return {"backend": backend}, None
    raise ValueError(f"unknown component {component!r}; known: {list(COMPONENTS)}")


def _probe_decoder(config: Config) -> tuple[dict[str, object], str | None]:
    versions: dict[str, str | None] = {}
    reason = None
    for tool in DECODER_TOOLS:
        executable = shutil.which(tool)
        if executable is None:
            versions[tool] = None
            reason = reason or f"{tool} not found on PATH"
            continue
        versions[tool], failure = _tool_version(executable, tool, config)
        if failure is not None:
            reason = reason or f"{tool} -version failed: {failure}"
    return {"versions": versions}, reason


def _tool_version(executable: str, tool: str, config: Config) -> tuple[str | None, str | None]:
    try:
        result = proc.run(
            [executable, "-version"],
            timeout=config.budgets.max_subprocess_seconds,
            max_output_bytes=VERSION_OUTPUT_BYTES,
            memory_bytes=config.budgets.max_subprocess_memory_mib * MIB,
        )
    except OSError as error:
        return None, error.strerror or str(error)
    if result.timed_out:
        return None, "timed out"
    if result.returncode != 0:
        return None, str(result.returncode)
    match = _VERSION_LINE.match(result.stdout.decode("utf-8", errors="replace"))
    if match is None or match.group(1) != tool:
        return None, "unrecognized output"
    return match.group(2), None


def _probe_asr(config: Config) -> tuple[dict[str, object], str | None]:
    versions, reason = _distribution_versions("asr")
    model, model_reason = _snapshot(config.asr.model, config.asr.revision)
    return {"versions": versions, "model": model}, reason or model_reason


def _probe_acoustics() -> tuple[dict[str, object], str | None]:
    versions, reason = _distribution_versions("acoustics")
    vad: dict[str, object] = {"asset": None, "size": None}
    assets = package_dir(VAD_PACKAGE)
    if assets is not None:
        for name, size in _regular_files(assets / "assets").items():
            if VAD_ASSET.match(name):
                vad = {"asset": name, "size": size}
                break
    if vad["asset"] is None:
        reason = reason or f"Silero VAD asset not found in {VAD_PACKAGE}"
    return {"versions": versions, "vad": vad}, reason


def _distribution_versions(component: str) -> tuple[dict[str, str | None], str | None]:
    versions = {name: installed_version(name) for name in DISTRIBUTIONS[component]}
    missing = next((name for name, found in versions.items() if found is None), None)
    return versions, None if missing is None else f"package {missing} is not installed"


def _snapshot(repo: str, revision: str | None) -> tuple[dict[str, object], str | None]:
    """Snapshot of the model in the local HF cache: `repo`, `revision`, `files`, read-only."""
    model: dict[str, object] = {"repo": repo, "revision": None, "files": {}}
    model_dir = hf_hub_cache() / f"models--{repo.replace('/', '--')}"
    snapshots = _subdirectories(model_dir / "snapshots")
    not_cached = (
        f"model weights not in the local cache: {repo}; downloading needs the owner's permission"
    )
    if revision is not None:
        matching = [name for name in snapshots if name.startswith(revision)]
        if len(matching) != 1:
            return model, f"asr.revision {revision} is not in the local cache"
        chosen = matching[0]
    elif (ref := _ref_main(model_dir)) in snapshots:
        chosen = ref
    elif len(snapshots) == 1:
        chosen = snapshots[0]
    elif not snapshots:
        return model, not_cached
    else:
        return model, f"several snapshots of {repo} in the cache and no asr.revision set"
    files = _regular_files(model_dir / "snapshots" / chosen)
    model = {"repo": repo, "revision": chosen, "files": files}
    if any(name not in files for name in WEIGHT_FILES):
        return model, not_cached
    return model, None


def _ref_main(model_dir: Path) -> str | None:
    try:
        fd = os.open(model_dir / "refs" / "main", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as source:
        ref = source.read(256).decode("utf-8", errors="replace").strip()
    return ref if _PLAIN_NAME.match(ref) and ref not in (".", "..") else None


def _subdirectories(directory: Path) -> list[str]:
    try:
        entries = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    except OSError:
        return []
    return entries


def _regular_files(directory: Path) -> dict[str, int]:
    """Names and sizes of the regular files in a directory (a symlink to a blob counts)."""
    files: dict[str, int] = {}
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        return files
    for entry in entries:
        try:
            info = entry.stat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            files[entry.name] = info.st_size
    return files


def _with_hash(component: str, config: Config, fields: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "component": component,
        "sound_scout": __version__,
        "platform": platform_info(),
        "config_hash": config_hash(component_config(config, component)),
        **fields,
    }
    result["hash"] = fingerprint_hash(result)
    return result


def _status(
    status: str, reason: str | None, current: dict[str, object], last_smoke_at: object
) -> dict[str, object]:
    return {
        "status": status,
        "reason": reason,
        "fingerprint": current,
        "last_smoke_at": last_smoke_at,
    }


def _changed_keys(saved: Mapping[str, object], current: Mapping[str, object]) -> list[str]:
    before, after = _flatten(saved), _flatten(current)
    keys = (set(before) | set(after)) - {"hash"}
    return sorted(key for key in keys if before.get(key) != after.get(key))


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flat: dict[str, object] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def _check_receipt(receipt: object, component: str) -> None:
    if not isinstance(receipt, dict):
        raise ReceiptInvalid("receipt is not a JSON object")
    if receipt.get("receipt_version") != RECEIPT_VERSION:
        raise ReceiptInvalid(f"receipt_version is not {RECEIPT_VERSION}")
    if receipt.get("component") != component:
        raise ReceiptInvalid(f"receipt is for component {receipt.get('component')!r}")
    if receipt.get("status") != "ok":
        raise ReceiptInvalid(f"receipt status is {receipt.get('status')!r}, not 'ok'")
    for field in ("smoke_at", "fingerprint", "smoke"):
        if field not in receipt:
            raise ReceiptInvalid(f"receipt has no {field}")
    if not isinstance(receipt["smoke_at"], str):
        raise ReceiptInvalid("smoke_at is not a string")
    saved, smoke = receipt["fingerprint"], receipt["smoke"]
    if not isinstance(saved, dict) or not isinstance(smoke, dict):
        raise ReceiptInvalid("fingerprint and smoke must be objects")
    try:
        expected = fingerprint_hash(saved)
    except ValueError:
        raise ReceiptInvalid("fingerprint is not canonical JSON") from None
    if not isinstance(saved.get("hash"), str) or saved["hash"] != expected:
        raise ReceiptInvalid("fingerprint hash does not match its fields")
    if _contains_test_double(saved) or _contains_test_double(smoke):
        raise ReceiptInvalid(f"receipt contains the string {TEST_DOUBLE!r}")


def _contains_test_double(value: object) -> bool:
    return _any_string(value, lambda text: text == TEST_DOUBLE)


def _absolute_path_in(value: object) -> str | None:
    found: list[str] = []

    def is_absolute(text: str) -> bool:
        if text.startswith("/") or _WINDOWS_ABSOLUTE.match(text):
            found.append(text)
            return True
        return False

    _any_string(value, is_absolute)
    return found[0] if found else None


def _any_string(value: object, predicate) -> bool:
    if isinstance(value, str):
        return predicate(value)
    if isinstance(value, Mapping):
        return any(_any_string(item, predicate) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_any_string(item, predicate) for item in value)
    return False


def _ensure_private_dir(path: Path) -> None:
    try:
        os.mkdir(path, DIR_MODE)
    except FileExistsError:
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ReceiptRejected(f"{RECEIPTS_DIR_NAME} is a symlink or not a directory") from None
        return
    os.chmod(path, DIR_MODE)


def _sha256(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
