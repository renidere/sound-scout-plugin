"""Operator configuration from `sound-scout.toml`: one loader and validator for CLI and MCP.

Allowed roots, the data root, budgets, the network policy and the backend are set by the
operator here, never by arguments of a tool call (R09, R10, R12). Secrets do not belong in
this file: an unknown field is rejected.
"""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_FILE_NAME = "sound-scout.toml"
EXAMPLE_FILE_NAME = "sound-scout.example.toml"
CONFIG_PATH_ENV = "SOUND_SCOUT_CONFIG"
KNOWN_BACKENDS = ("none",)
MIB = 1024 * 1024


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
    max_channels: int = 2
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
class Config:
    allowed_roots: tuple[Path, ...]
    data_root: Path
    budgets: Budgets = Budgets()
    network: Network = Network()
    backend: str = "none"


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
    )


def _budgets(value: object) -> Budgets:
    table = _table(value, "budgets")
    _reject_unknown(table, {field.name for field in fields(Budgets)}, prefix="budgets")
    return Budgets(**{name: _positive_int(item, f"budgets.{name}") for name, item in table.items()})


def _network(value: object) -> Network:
    table = _table(value, "network")
    _reject_unknown(table, {field.name for field in fields(Network)}, prefix="network")
    return Network(**{name: _bool(item, f"network.{name}") for name, item in table.items()})


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
