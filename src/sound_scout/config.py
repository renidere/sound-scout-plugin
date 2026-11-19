"""Operator configuration from `sound-scout.toml`: one loader and validator for CLI and MCP.

Allowed roots, the data root, budgets, the network policy and the backend are set by the
operator here, never by arguments of a tool call (R09, R10, R12). Secrets do not belong in
this file: an unknown field is rejected.

`DENIED_NAME_PATTERNS` and the root rule `root_denied_name` live here rather than in `paths`
because both modules need them and `paths` imports `config`: the loader rejects an allowed root
that `root_denied_name` bars in its resolved path, `paths` refuses every path under such a root
with `ROOT_HAS_DENIED_NAME` and every input path whose component below the root matches.
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

# Names never read as input in any component below an allowed root, and not allowed in the
# root itself except for the Hermes media directories `root_denied_name` permits: operator
# secrets and config, credential and agent-profile directories, playlist and script sources.
# Matched case-insensitively against each path component.
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

# The only directories under `.hermes` an allowed root may be (issue #66, owner decision A):
# the media caches where Hermes saves incoming audio, in the home or in a profile home.
# Credentials and config next to them stay denied, and below the root every name is checked.
HERMES_DIR = ".hermes"
HERMES_MEDIA_DIRS = (("cache", "audio"), ("audio_cache",))
HERMES_PROFILES_DIR = "profiles"


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
class Jobs:
    """Admission of the job queue (`sound_scout.jobs`, decision 0013 § 6): a submit that would
    put more than `max_queued_jobs` jobs in `queued`, or more than `max_staging_bytes` of input
    in the jobs not yet ended (`queued`, `running`), is refused with its reason."""

    max_queued_jobs: int = 10
    max_staging_bytes: int = 1024 * MIB


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
    jobs: Jobs = Jobs()


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
    given_roots = _require(data, "allowed_roots")
    allowed_roots = _directories(given_roots, "allowed_roots")
    data_root = _directory(_require(data, "data_root"), "data_root")
    for index, root in enumerate(allowed_roots):
        # Запрещённые имена — про чтение входа (R10): корень с таким компонентом не поддерживается
        # `paths` (каждый путь под ним отвечает ROOT_HAS_DENIED_NAME), поэтому отказ здесь, при
        # загрузке. `data_root` не проверяется: оттуда не читают, туда только пишут.
        denied = root_denied_name(root)
        if denied is not None:
            raise ConfigError(
                f"allowed_roots[{index}]: contains the denied name {denied!r}: {root}"
            )
        if not _hermes_dir_given_directly(root, Path(os.path.normpath(given_roots[index]))):
            raise ConfigError(
                f"allowed_roots[{index}]: a Hermes media directory must be given by its own "
                f"path, not through a symlink: {root}"
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
        jobs=_jobs(data.get("jobs", {})),
    )


def is_denied_name(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in DENIED_NAME_PATTERNS)


def root_denied_name(root: Path) -> str | None:
    """The component of the resolved `root` that bars it as an allowed root, or `None`.

    Any denied name bars a root, except one `.hermes` followed by exactly a Hermes media
    directory: `cache/audio`, `audio_cache` or either under `profiles/<name>/`.
    """
    parts = root.parts
    hermes = _hermes_index(root)
    if hermes is not None and _is_hermes_media_dir(parts[hermes + 1 :]):
        parts = parts[:hermes] + parts[hermes + 1 :]
    return next((part for part in parts if is_denied_name(part)), None)


def _hermes_index(path: Path) -> int | None:
    """Index of the only `.hermes` component of `path`; `None` when there is none or several."""
    found = [index for index, part in enumerate(path.parts) if part.lower() == HERMES_DIR]
    return found[0] if len(found) == 1 else None


def _is_hermes_media_dir(tail: tuple[str, ...]) -> bool:
    if len(tail) > 2 and tail[0] == HERMES_PROFILES_DIR:
        tail = tail[2:]
    return tail in HERMES_MEDIA_DIRS


def _hermes_dir_given_directly(root: Path, given: Path) -> bool:
    """A resolved root under `.hermes` must be spelled from `.hermes` down the same way: a
    symlink such as `media` -> `.hermes` would let a path without `.hermes` select it."""
    hermes = _hermes_index(root)
    if hermes is None:
        return True
    tail = root.parts[hermes:]
    return given.parts[-len(tail) :] == tail


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


def _jobs(value: object) -> Jobs:
    table = _table(value, "jobs")
    _reject_unknown(table, {field.name for field in fields(Jobs)}, prefix="jobs")
    return Jobs(**{name: _positive_int(item, f"jobs.{name}") for name, item in table.items()})


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
