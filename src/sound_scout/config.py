"""Operator configuration from `sound-scout.toml`: one loader and validator for CLI and MCP.

Allowed roots, the data root, budgets, the network policy and the backend are set by the
operator here, never by arguments of a tool call (R09, R10, R12). Secrets do not belong in
this file: an unknown field is rejected.

`DENIED_NAME_PATTERNS` lives here rather than in `paths` because both modules need it and
`paths` imports `config`: the loader rejects an allowed root that has a denied component in its
resolved path, and `paths` refuses every input path whose component below the root matches.
"""

import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_FILE_NAME = "sound-scout.toml"
EXAMPLE_FILE_NAME = "sound-scout.example.toml"
CONFIG_PATH_ENV = "SOUND_SCOUT_CONFIG"
KNOWN_BACKENDS = ("none",)
# Самое большее число каналов, которое версия обрабатывает (R02): `channels` работает
# с одним или двумя; `budgets.max_channels` не может поднять этот предел. Константа здесь,
# а не в `channels`, потому что `channels` импортирует `ingest`, а тот — `config`.
MAX_CHANNELS = 2
MIB = 1024 * 1024
MODEL_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
MODEL_REVISION = re.compile(r"[0-9a-f]{7,40}")

# Names never read as input, in any position of the path, even inside an allowed root:
# operator secrets and config, credential and agent-profile directories, playlist and script
# sources. Matched case-insensitively against each path component.
DENIED_NAME_PATTERNS = (
    ".env*",
    CONFIG_FILE_NAME,
    ".ssh",
    ".hermes",
    "*.m3u",
    "*.m3u8",
    "*.pls",
    "*.xspf",
    "*.cue",
    "*.ffconcat",
    "*.sdp",
)


class ConfigError(Exception):
    """The configuration file is missing or invalid; the message names the file and the field."""


@dataclass(frozen=True)
class Budgets:
    """Engineering limits of the version (R10, R11); every one is documented in the example file.

    `max_duration_seconds` also bounds the decoded samples: `decoded_samples_limit` is this
    many seconds at the stream's sample rate and channel count. `max_subprocess_seconds` and
    `max_subprocess_memory_mib` bound one decoder child process (`proc.run`: wall time and
    `RLIMIT_AS`), `worker_deadline_seconds` the whole analysis worker.
    """

    max_duration_seconds: int = 180
    max_input_bytes: int = 100 * MIB
    max_channels: int = MAX_CHANNELS
    worker_deadline_seconds: int = 180
    max_subprocess_seconds: int = 60
    max_subprocess_memory_mib: int = 2048


def decoded_samples_limit(budgets: Budgets, sample_rate: int, channels: int) -> int:
    """Most interleaved samples a decoder may produce: `max_duration_seconds` at this format."""
    if sample_rate <= 0 or channels <= 0:
        raise ValueError(
            f"sample_rate and channels must be positive, got {sample_rate}, {channels}"
        )
    return budgets.max_duration_seconds * sample_rate * channels


@dataclass(frozen=True)
class Network:
    allow: bool = False


@dataclass(frozen=True)
class Asr:
    """The ASR model as `owner/name` on the Hugging Face hub and, optionally, its snapshot.

    Only an already installed local snapshot is used (`local_files_only`); the default is
    the model installed on the owner's workstation. A new model or revision is a download
    and needs the owner's permission (R09). `revision` picks the snapshot when the local
    cache holds several.
    """

    model: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    revision: str | None = None


@dataclass(frozen=True)
class Config:
    allowed_roots: tuple[Path, ...]
    data_root: Path
    budgets: Budgets = Budgets()
    network: Network = Network()
    backend: str = "none"
    asr: Asr = Asr()


def config_path() -> Path:
    """`SOUND_SCOUT_CONFIG` when set, otherwise `sound-scout.toml` in the current directory."""
    env_path = os.environ.get(CONFIG_PATH_ENV)
    return Path(env_path) if env_path else Path.cwd() / CONFIG_FILE_NAME


def load() -> Config:
    path = config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(
            f"configuration file not found: {path} - copy {EXAMPLE_FILE_NAME} to "
            f"{CONFIG_FILE_NAME} in the current directory and fill in the paths, "
            f"or point {CONFIG_PATH_ENV} at the file"
        ) from None
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path}: cannot read configuration file: {error}") from None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from None
    try:
        return _parse(data)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None


def _parse(data: dict[str, object]) -> Config:
    _reject_unknown(data, {field.name for field in fields(Config)}, prefix="")
    allowed_roots = _directories(_require(data, "allowed_roots"), "allowed_roots")
    data_root = _directory(_require(data, "data_root"), "data_root")
    for index, root in enumerate(allowed_roots):
        # Запрещённые имена — про чтение входа (R10): корень с таким компонентом не поддерживается
        # `paths` (каждый путь под ним отвечает ROOT_HAS_DENIED_NAME), поэтому отказ здесь, при
        # загрузке. `data_root` не проверяется: оттуда не читают, туда только пишут.
        denied = next((part for part in root.parts if is_denied_name(part)), None)
        if denied is not None:
            raise ConfigError(
                f"allowed_roots[{index}]: contains the denied name {denied!r}: {root}"
            )
        if data_root.is_relative_to(root) or root.is_relative_to(data_root):
            raise ConfigError(
                f"data_root and allowed_roots[{index}] must not contain each other: "
                f"{data_root} and {root}"
            )
    return Config(
        allowed_roots=allowed_roots,
        data_root=data_root,
        budgets=_budgets(data.get("budgets", {})),
        network=_network(data.get("network", {})),
        backend=_backend(data.get("backend", "none")),
        asr=_asr(data.get("asr", {})),
    )


def is_denied_name(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in DENIED_NAME_PATTERNS)


def _budgets(value: object) -> Budgets:
    table = _table(value, "budgets")
    _reject_unknown(table, {field.name for field in fields(Budgets)}, prefix="budgets")
    budgets = Budgets(
        **{name: _positive_int(item, f"budgets.{name}") for name, item in table.items()}
    )
    if budgets.max_channels > MAX_CHANNELS:
        raise ConfigError(
            f"budgets.max_channels: the version handles at most {MAX_CHANNELS} channels, "
            f"got {budgets.max_channels}"
        )
    return budgets


def _network(value: object) -> Network:
    table = _table(value, "network")
    _reject_unknown(table, {field.name for field in fields(Network)}, prefix="network")
    return Network(**{name: _bool(item, f"network.{name}") for name, item in table.items()})


def _asr(value: object) -> Asr:
    table = _table(value, "asr")
    _reject_unknown(table, {field.name for field in fields(Asr)}, prefix="asr")
    asr = Asr()
    if "model" in table:
        model = table["model"]
        if not isinstance(model, str) or not MODEL_ID.fullmatch(model):
            raise ConfigError(f"asr.model: expected a model id like owner/name, got {model!r}")
        asr = Asr(model=model, revision=asr.revision)
    if "revision" in table:
        revision = table["revision"]
        if not isinstance(revision, str) or not MODEL_REVISION.fullmatch(revision):
            raise ConfigError(
                f"asr.revision: expected a hex commit of 7 to 40 characters, got {revision!r}"
            )
        asr = Asr(model=asr.model, revision=revision)
    return asr


def _backend(value: object) -> str:
    if not isinstance(value, str) or value not in KNOWN_BACKENDS:
        raise ConfigError(f"backend: expected one of {list(KNOWN_BACKENDS)}, got {value!r}")
    return value


def _require(data: dict[str, object], name: str) -> object:
    if name not in data:
        raise ConfigError(f"{name}: required field is missing")
    return data[name]


def _reject_unknown(table: dict[str, object], known: set[str], prefix: str) -> None:
    for key in table:
        if key not in known:
            raise ConfigError(f"unknown field: {f'{prefix}.{key}' if prefix else key}")


def _table(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field}: expected a table, got {value!r}")
    return value


def _directories(value: object, field: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field}: expected a non-empty list of absolute directories")
    return tuple(_directory(item, f"{field}[{index}]") for index, item in enumerate(value))


def _directory(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise ConfigError(f"{field}: expected a path string, got {value!r}")
    path = Path(value)
    if not path.is_absolute():
        raise ConfigError(f"{field}: expected an absolute path, got {value!r}")
    if not path.is_dir():
        raise ConfigError(f"{field}: not an existing directory: {value}")
    return path.resolve()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field}: expected a positive integer, got {value!r}")
    return value


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field}: expected true or false, got {value!r}")
    return value
