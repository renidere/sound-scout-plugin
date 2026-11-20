import dataclasses
import json
import re
from pathlib import Path

import pytest

from sound_scout import channels, config, paths
from sound_scout.config import (
    MAX_CHANNELS,
    Asr,
    Budgets,
    Config,
    ConfigError,
    Jobs,
    Network,
    decoded_samples_limit,
    load,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "sound-scout.example.toml"
COMMENTED_FIELD_LINE = re.compile(r"# ?(\[[\w.]+\]|[\w.]+ = .*)")


def toml_string(path: Path) -> str:
    return json.dumps(str(path))


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    recordings = tmp_path / "recordings"
    data = tmp_path / "data"
    recordings.mkdir()
    data.mkdir()
    return recordings, data


@pytest.fixture
def minimal_toml(roots: tuple[Path, Path]) -> str:
    recordings, data = roots
    return f"allowed_roots = [{toml_string(recordings)}]\ndata_root = {toml_string(data)}\n"


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write the text as `sound-scout.toml` in a fresh current directory, env unset."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv(config.CONFIG_PATH_ENV, raising=False)

    def _write(text: str) -> Path:
        path = cwd / config.CONFIG_FILE_NAME
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def test_loads_from_current_directory_with_defaults(
    write_config, minimal_toml: str, roots: tuple[Path, Path]
) -> None:
    write_config(minimal_toml)
    loaded = load()
    assert loaded == Config(allowed_roots=(roots[0].resolve(),), data_root=roots[1].resolve())
    assert loaded.budgets == Budgets(
        max_duration_seconds=180,
        max_input_bytes=100 * 1024 * 1024,
        max_channels=2,
        worker_deadline_seconds=180,
        max_subprocess_seconds=60,
        max_subprocess_memory_mib=2048,
    )
    assert loaded.network == Network(allow=False)
    assert loaded.backend == "none"
    assert loaded.asr == Asr(model="mobiuslabsgmbh/faster-whisper-large-v3-turbo", revision=None)


def test_loads_from_env_path_over_current_directory(
    write_config, minimal_toml: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config("allowed_roots = 'broken'\n")
    env_path = tmp_path / "elsewhere" / "custom.toml"
    env_path.parent.mkdir()
    env_path.write_text(minimal_toml + 'backend = "none"\n[budgets]\nmax_channels = 1\n')
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(env_path))
    assert load().budgets.max_channels == 1


def test_all_fields_are_read_with_explicit_types(write_config, minimal_toml: str) -> None:
    write_config(
        minimal_toml
        + 'backend = "none"\n'
        + "[budgets]\nmax_duration_seconds = 30\nmax_input_bytes = 1024\n"
        + "max_channels = 1\nworker_deadline_seconds = 60\n"
        + "max_subprocess_seconds = 10\nmax_subprocess_memory_mib = 256\n"
        + "[network]\nallow = true\n"
        + '[asr]\nmodel = "Systran/faster-whisper-small"\nrevision = "0a363e91abcdef0"\n'
    )
    loaded = load()
    assert loaded.budgets == Budgets(30, 1024, 1, 60, 10, 256)
    assert loaded.network.allow is True
    assert loaded.asr == Asr(model="Systran/faster-whisper-small", revision="0a363e91abcdef0")


def test_jobs_admission_limits_are_read_and_default(write_config, minimal_toml: str) -> None:
    write_config(minimal_toml)
    assert load().jobs == Jobs() == Jobs(max_queued_jobs=10, max_staging_bytes=1024 * config.MIB)
    write_config(minimal_toml + "[jobs]\nmax_queued_jobs = 3\nmax_staging_bytes = 4096\n")
    assert load().jobs == Jobs(max_queued_jobs=3, max_staging_bytes=4096)
    assert Jobs().staging_retention_seconds == 604800
    write_config(minimal_toml + "[jobs]\nstaging_retention_seconds = 60\n")
    assert load().jobs.staging_retention_seconds == 60


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("[jobs]\nmax_queued_jobs = 0\n", "jobs.max_queued_jobs: expected a positive integer"),
        ('[jobs]\nmax_staging_bytes = "1G"\n', "jobs.max_staging_bytes: expected a positive"),
        ("[jobs]\nrunners = 2\n", "unknown field: jobs.runners"),
        ("[jobs]\nstaging_retention_seconds = 0\n", "jobs.staging_retention_seconds: expected"),
        ("[jobs]\nstaging_retention_seconds = -1\n", "jobs.staging_retention_seconds: expected"),
        ('[jobs]\nstaging_retention_seconds = "7d"\n', "jobs.staging_retention_seconds: expected"),
        ("jobs = 3\n", "jobs: expected a table"),
    ],
)
def test_invalid_jobs_table_is_rejected_by_field(
    write_config, minimal_toml: str, extra: str, message: str
) -> None:
    write_config(minimal_toml + extra)
    with pytest.raises(ConfigError, match=message):
        load()


def test_staging_root_is_unset_by_default_and_read_when_it_is_an_allowed_root(
    write_config, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    recordings, data = roots
    staging = tmp_path / "staging"
    staging.mkdir()
    head = f"data_root = {toml_string(data)}\n"
    write_config(f"allowed_roots = [{toml_string(recordings)}]\n" + head)
    assert load().jobs.staging_root is None
    both = f"allowed_roots = [{toml_string(recordings)}, {toml_string(staging)}]\n"
    write_config(both + head + f"[jobs]\nstaging_root = {toml_string(staging)}\n")
    assert load().jobs.staging_root == staging.resolve()


@pytest.mark.parametrize("where", ["outside", "inside_a_root", "missing", "relative"])
def test_staging_root_must_be_one_of_the_allowed_roots(
    write_config, roots: tuple[Path, Path], tmp_path: Path, where: str
) -> None:
    recordings, data = roots
    staging = {
        "outside": tmp_path / "staging",
        "inside_a_root": recordings / "staging",
        "missing": tmp_path / "absent",
    }.get(where)
    if where in ("outside", "inside_a_root"):
        staging.mkdir()
    value = '"staging"' if where == "relative" else toml_string(staging)
    write_config(
        f"allowed_roots = [{toml_string(recordings)}]\ndata_root = {toml_string(data)}\n"
        f"[jobs]\nstaging_root = {value}\n"
    )
    message = {
        "outside": "jobs.staging_root: must be one of allowed_roots",
        "inside_a_root": "jobs.staging_root: must be one of allowed_roots",
        "missing": "jobs.staging_root: not an existing directory",
        "relative": "jobs.staging_root: expected an absolute path",
    }[where]
    with pytest.raises(ConfigError, match=message):
        load()


@pytest.mark.parametrize(
    "parts", [(".hermes", "cache", "audio"), (".Hermes", "audio_cache"), ("media", ".HERMES")]
)
def test_staging_root_under_hermes_is_refused_even_as_an_allowed_media_dir(
    write_config, roots: tuple[Path, Path], tmp_path: Path, parts: tuple[str, ...]
) -> None:
    _, data = roots
    staging = tmp_path.joinpath(*parts)
    staging.mkdir(parents=True)
    write_config(
        f"allowed_roots = [{toml_string(staging)}]\ndata_root = {toml_string(data)}\n"
        f"[jobs]\nstaging_root = {toml_string(staging)}\n"
    )
    with pytest.raises(ConfigError, match=r"jobs\.staging_root|allowed_roots\[0\]"):
        load()
    if parts[-1] in ("audio", "audio_cache"):
        write_config(f"allowed_roots = [{toml_string(staging)}]\ndata_root = {toml_string(data)}\n")
        assert load().allowed_roots == (staging.resolve(),)
        write_config(
            f"allowed_roots = [{toml_string(staging)}]\ndata_root = {toml_string(data)}\n"
            f"[jobs]\nstaging_root = {toml_string(staging)}\n"
        )
        with pytest.raises(ConfigError, match=r"jobs\.staging_root: must not be under \.hermes"):
            load()


def test_asr_model_alone_leaves_revision_unset(write_config, minimal_toml: str) -> None:
    write_config(minimal_toml + '[asr]\nmodel = "owner/model.v2-ru"\n')
    assert load().asr == Asr(model="owner/model.v2-ru", revision=None)


def test_missing_file_names_example_and_env(write_config) -> None:
    with pytest.raises(ConfigError) as error:
        load()
    message = str(error.value)
    assert config.CONFIG_FILE_NAME in message
    assert config.EXAMPLE_FILE_NAME in message
    assert config.CONFIG_PATH_ENV in message


def test_missing_env_file_names_that_path(
    write_config, minimal_toml: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(minimal_toml)
    missing = tmp_path / "missing.toml"
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(missing))
    with pytest.raises(ConfigError, match=re.escape(str(missing))):
        load()


def test_directory_as_config_path_is_a_config_error(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(tmp_path))
    with pytest.raises(ConfigError, match=rf"{re.escape(str(tmp_path))}: cannot read"):
        load()


def test_non_utf8_content_is_a_config_error(write_config) -> None:
    path = write_config("")
    path.write_bytes(b"allowed_roots = [\xff]\n")
    with pytest.raises(ConfigError, match=rf"{re.escape(str(path))}: cannot read"):
        load()


def test_invalid_toml_names_the_file(write_config) -> None:
    path = write_config("allowed_roots = [\n")
    with pytest.raises(ConfigError, match=rf"{re.escape(str(path))}: not valid TOML"):
        load()


@pytest.mark.parametrize("missing", ["allowed_roots", "data_root"])
def test_required_field_missing_is_named(write_config, minimal_toml: str, missing: str) -> None:
    text = "".join(line for line in minimal_toml.splitlines(True) if not line.startswith(missing))
    write_config(text)
    with pytest.raises(ConfigError, match=rf"{missing}: required field is missing"):
        load()


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        ('api_key = "sk-secret"\n', "api_key"),
        ("[budgets]\nwall_time_seconds = 10\n", r"budgets\.wall_time_seconds"),
        ("[network]\nproxy = 'x'\n", r"network\.proxy"),
        ("[backend]\nname = 'x'\n", "backend"),
        ("[asr]\ndevice = 'cuda'\n", r"asr\.device"),
    ],
)
def test_unknown_field_is_rejected_by_name(
    write_config, minimal_toml: str, extra: str, field: str
) -> None:
    write_config(minimal_toml + extra)
    with pytest.raises(ConfigError, match=field):
        load()


def test_relative_paths_are_rejected_by_field(write_config, roots: tuple[Path, Path]) -> None:
    write_config(f'allowed_roots = ["recordings"]\ndata_root = {toml_string(roots[1])}\n')
    with pytest.raises(ConfigError, match=r"allowed_roots\[0\]: expected an absolute path"):
        load()
    write_config(f'allowed_roots = [{toml_string(roots[0])}]\ndata_root = "data"\n')
    with pytest.raises(ConfigError, match="data_root: expected an absolute path"):
        load()


def test_nonexistent_directory_is_rejected(write_config, roots: tuple[Path, Path]) -> None:
    gone = roots[1] / "not-there"
    write_config(f"allowed_roots = [{toml_string(roots[0])}]\ndata_root = {toml_string(gone)}\n")
    with pytest.raises(ConfigError, match="data_root: not an existing directory"):
        load()


def test_data_root_inside_allowed_root_is_rejected(write_config, roots: tuple[Path, Path]) -> None:
    inside = roots[0] / "runs"
    inside.mkdir()
    write_config(f"allowed_roots = [{toml_string(roots[0])}]\ndata_root = {toml_string(inside)}\n")
    with pytest.raises(ConfigError, match=r"data_root and allowed_roots\[0\]"):
        load()


def test_allowed_root_inside_data_root_is_rejected(write_config, roots: tuple[Path, Path]) -> None:
    inside = roots[1] / "incoming"
    inside.mkdir()
    write_config(
        f"allowed_roots = [{toml_string(roots[0])}, {toml_string(inside)}]\n"
        f"data_root = {toml_string(roots[1])}\n"
    )
    with pytest.raises(ConfigError, match=r"data_root and allowed_roots\[1\]"):
        load()


def test_same_directory_for_both_is_rejected(write_config, roots: tuple[Path, Path]) -> None:
    write_config(
        f"allowed_roots = [{toml_string(roots[0])}]\ndata_root = {toml_string(roots[0])}\n"
    )
    with pytest.raises(ConfigError, match=r"data_root and allowed_roots\[0\]"):
        load()


def test_symlinked_roots_are_compared_resolved(
    write_config, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    link = tmp_path / "link-to-recordings"
    link.symlink_to(roots[0], target_is_directory=True)
    inside = roots[0] / "runs"
    inside.mkdir()
    write_config(f"allowed_roots = [{toml_string(link)}]\ndata_root = {toml_string(inside)}\n")
    with pytest.raises(ConfigError, match=r"data_root and allowed_roots\[0\]"):
        load()


@pytest.mark.parametrize(
    ("relative", "denied"),
    [
        (Path(".hermes") / "audio", ".hermes"),
        (Path(".ssh"), ".ssh"),
        (Path("home") / ".env.local" / "recordings", ".env.local"),
    ],
)
def test_allowed_root_with_a_denied_component_is_rejected_at_load(
    write_config, roots: tuple[Path, Path], tmp_path: Path, relative: Path, denied: str
) -> None:
    bad = tmp_path / relative
    bad.mkdir(parents=True)
    write_config(
        f"allowed_roots = [{toml_string(roots[0])}, {toml_string(bad)}]\n"
        f"data_root = {toml_string(roots[1])}\n"
    )
    with pytest.raises(ConfigError) as error:
        load()
    message = str(error.value)
    assert "allowed_roots[1]" in message
    assert repr(denied) in message
    assert str(bad.resolve()) in message


def test_allowed_root_with_an_ordinary_name_loads(
    write_config, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    ordinary = tmp_path / "hermes-env" / "ssh-recordings.m3u-backup"
    ordinary.mkdir(parents=True)
    write_config(
        f"allowed_roots = [{toml_string(ordinary)}]\ndata_root = {toml_string(roots[1])}\n"
    )
    assert load().allowed_roots == (ordinary.resolve(),)


HERMES_MEDIA_ROOTS = [
    Path(".hermes") / "cache" / "audio",
    Path(".hermes") / "audio_cache",
    Path(".hermes") / "profiles" / "work" / "cache" / "audio",
    Path(".hermes") / "profiles" / "work" / "audio_cache",
]


@pytest.mark.parametrize("relative", HERMES_MEDIA_ROOTS)
def test_hermes_media_directory_loads_as_an_allowed_root(
    write_config, roots: tuple[Path, Path], tmp_path: Path, relative: Path
) -> None:
    media = tmp_path / relative
    media.mkdir(parents=True)
    write_config(f"allowed_roots = [{toml_string(media)}]\ndata_root = {toml_string(roots[1])}\n")
    assert load().allowed_roots == (media.resolve(),)


def test_hermes_media_directory_through_a_symlinked_home_loads(
    write_config, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    media = tmp_path / "real-home" / ".hermes" / "cache" / "audio"
    media.mkdir(parents=True)
    (tmp_path / "home").symlink_to(tmp_path / "real-home", target_is_directory=True)
    given = tmp_path / "home" / ".hermes" / "cache" / "audio"
    write_config(f"allowed_roots = [{toml_string(given)}]\ndata_root = {toml_string(roots[1])}\n")
    assert load().allowed_roots == (media.resolve(),)


def hermes_root_error(write_config, roots: tuple[Path, Path], given: Path) -> str:
    write_config(
        f"allowed_roots = [{toml_string(roots[0])}, {toml_string(given)}]\n"
        f"data_root = {toml_string(roots[1])}\n"
    )
    with pytest.raises(ConfigError) as error:
        load()
    message = str(error.value)
    assert "allowed_roots[1]" in message
    return message


@pytest.mark.parametrize(
    ("relative", "denied"),
    [
        (Path(".hermes"), ".hermes"),
        (Path(".hermes") / "attachments", ".hermes"),
        (Path(".hermes") / "sessions", ".hermes"),
        (Path(".hermes") / "cache", ".hermes"),
        (Path(".hermes") / "cache" / "audio" / "sub", ".hermes"),
        (Path(".hermes") / "profiles" / "cache" / "audio", ".hermes"),
        (Path(".hermes") / "x" / ".hermes" / "cache" / "audio", ".hermes"),
        (Path(".ssh") / ".hermes" / "cache" / "audio", ".ssh"),
        (Path(".hermes") / "profiles" / ".env" / "cache" / "audio", ".env"),
        (Path("sound-scout.toml") / ".hermes" / "audio_cache", "sound-scout.toml"),
        (Path("list.m3u") / ".hermes" / "cache" / "audio", "list.m3u"),
    ],
)
def test_hermes_root_outside_the_media_directories_is_rejected(
    write_config, roots: tuple[Path, Path], tmp_path: Path, relative: Path, denied: str
) -> None:
    bad = tmp_path / relative
    bad.mkdir(parents=True)
    message = hermes_root_error(write_config, roots, bad)
    assert repr(denied) in message
    assert str(bad.resolve()) in message


def test_symlink_into_hermes_is_not_a_media_root(
    write_config, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    (tmp_path / ".hermes" / "cache" / "audio").mkdir(parents=True)
    (tmp_path / "media").symlink_to(tmp_path / ".hermes", target_is_directory=True)
    message = hermes_root_error(write_config, roots, tmp_path / "media" / "cache" / "audio")
    assert "through a symlink" in message


def test_media_directory_symlinked_to_hermes_itself_is_rejected(
    write_config, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    (tmp_path / ".hermes" / "cache").mkdir(parents=True)
    audio = tmp_path / ".hermes" / "cache" / "audio"
    audio.symlink_to(tmp_path / ".hermes", target_is_directory=True)
    message = hermes_root_error(write_config, roots, audio)
    assert repr(".hermes") in message
    assert str((tmp_path / ".hermes").resolve()) in message


def test_data_root_with_a_denied_component_is_not_checked(
    write_config, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    data = tmp_path / ".hermes" / "sound-scout-data"
    data.mkdir(parents=True)
    write_config(f"allowed_roots = [{toml_string(roots[0])}]\ndata_root = {toml_string(data)}\n")
    assert load().data_root == data.resolve()


def test_denied_names_are_shared_with_paths() -> None:
    assert paths.DENIED_NAME_PATTERNS is config.DENIED_NAME_PATTERNS
    assert paths.is_denied_name is config.is_denied_name
    assert paths.root_denied_name is config.root_denied_name
    assert config.is_denied_name(".HERMES") and config.is_denied_name(config.CONFIG_FILE_NAME)
    assert not config.is_denied_name("recordings")


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ('allowed_roots = "/abs"\n', "allowed_roots: expected a non-empty list"),
        ("allowed_roots = []\n", "allowed_roots: expected a non-empty list"),
        ("allowed_roots = [1]\n", r"allowed_roots\[0\]: expected a path string"),
        ("data_root = 1\n", "data_root: expected a path string"),
        ("budgets = 1\n", "budgets: expected a table"),
        ("[budgets]\nmax_duration_seconds = 0\n", r"budgets\.max_duration_seconds"),
        ("[budgets]\nmax_input_bytes = -1\n", r"budgets\.max_input_bytes"),
        ("[budgets]\nmax_channels = true\n", r"budgets\.max_channels"),
        ('[budgets]\nworker_deadline_seconds = "60"\n', r"budgets\.worker_deadline_seconds"),
        ("[budgets]\nmax_duration_seconds = 1.5\n", r"budgets\.max_duration_seconds"),
        ("[budgets]\nmax_subprocess_seconds = 0\n", r"budgets\.max_subprocess_seconds"),
        ("[budgets]\nmax_subprocess_memory_mib = -5\n", r"budgets\.max_subprocess_memory_mib"),
        ('[network]\nallow = "yes"\n', r"network\.allow: expected true or false"),
        ('backend = "whisper"\n', "backend: expected one of"),
        ("backend = 1\n", "backend: expected one of"),
        ("asr = 1\n", "asr: expected a table"),
        ('[asr]\nmodel = "x"\n', r"asr\.model: expected a model id like owner/name"),
        ('[asr]\nmodel = "a/b/c"\n', r"asr\.model: expected a model id"),
        ('[asr]\nmodel = "../etc/passwd"\n', r"asr\.model: expected a model id"),
        ("[asr]\nmodel = 1\n", r"asr\.model: expected a model id"),
        ('[asr]\nrevision = "zz"\n', r"asr\.revision: expected a hex commit"),
        ('[asr]\nrevision = "abc123"\n', r"asr\.revision: expected a hex commit"),
        ("[asr]\nrevision = 123456789\n", r"asr\.revision: expected a hex commit"),
        ('[asr]\nmodel = "owner/name\\n"\n', r"asr\.model: expected a model id"),
        ('[asr]\nrevision = "0123456789abcdef\\n"\n', r"asr\.revision: expected a hex commit"),
    ],
)
def test_wrong_types_and_values_are_named(
    write_config, minimal_toml: str, text: str, field: str
) -> None:
    lines = [line for line in minimal_toml.splitlines(True) if line.split(" ")[0] not in text]
    write_config("".join(lines) + text)
    with pytest.raises(ConfigError, match=field):
        load()


def test_max_channels_above_the_version_limit_is_refused_at_load(
    write_config, minimal_toml: str
) -> None:
    write_config(minimal_toml + "[budgets]\nmax_channels = 3\n")
    with pytest.raises(ConfigError, match=r"budgets\.max_channels: .*at most 2 channels, got 3"):
        load()


@pytest.mark.parametrize("value", [1, 2])
def test_max_channels_up_to_the_version_limit_loads(
    write_config, minimal_toml: str, value: int
) -> None:
    write_config(minimal_toml + f"[budgets]\nmax_channels = {value}\n")
    assert load().budgets.max_channels == value


def test_max_channels_limit_is_the_one_constant_channels_checks() -> None:
    assert MAX_CHANNELS == channels.MAX_CHANNELS == 2
    assert Budgets().max_channels == MAX_CHANNELS


@pytest.mark.parametrize(
    ("seconds", "sample_rate", "channels", "expected"),
    [(180, 16_000, 1, 2_880_000), (30, 48_000, 2, 2_880_000), (1, 8_000, 1, 8_000)],
)
def test_decoded_samples_limit_is_duration_times_rate_times_channels(
    seconds: int, sample_rate: int, channels: int, expected: int
) -> None:
    budgets = Budgets(max_duration_seconds=seconds)
    assert decoded_samples_limit(budgets, sample_rate, channels) == expected


@pytest.mark.parametrize(("sample_rate", "channels"), [(0, 1), (16_000, 0), (-1, 2)])
def test_decoded_samples_limit_rejects_non_positive_format(sample_rate: int, channels: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        decoded_samples_limit(Budgets(), sample_rate, channels)


def test_config_is_frozen(write_config, minimal_toml: str) -> None:
    loaded = write_config(minimal_toml) and load()
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.backend = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.budgets.max_duration_seconds = 1  # type: ignore[misc]


def example_config_text(allowed_root: Path, data_root: Path) -> str:
    """Uncomment the field lines of the example and substitute the illustrative paths."""
    field_lines = []
    for line in EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        if match := COMMENTED_FIELD_LINE.fullmatch(line):
            field_lines.append(match.group(1))
    text = "\n".join(field_lines) + "\n"
    return (
        text.replace('"/path/to/recordings"', toml_string(allowed_root))
        .replace('"/path/to/sound-scout-data"', toml_string(data_root))
        .replace('"/path/to/sound-scout-staging"', toml_string(allowed_root))
    )


def test_example_config_loads_after_substituting_paths(
    write_config, roots: tuple[Path, Path]
) -> None:
    text = example_config_text(*roots)
    assert "/path/to/" not in text
    write_config(text)
    loaded = load()
    assert loaded.allowed_roots == (roots[0].resolve(),)
    assert loaded.data_root == roots[1].resolve()
    assert loaded.budgets == Budgets()
    assert loaded.network == Network()
    assert loaded.backend == "none"
    assert loaded.asr.model == Asr().model
    assert re.fullmatch(r"[0-9a-f]{40}", loaded.asr.revision or "")


def test_example_names_the_allowed_range_next_to_max_channels() -> None:
    lines = EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
    (field_line,) = [
        line for line in lines if COMMENTED_FIELD_LINE.fullmatch(line) and "max_channels" in line
    ]
    assert re.search(r"1 или 2", field_line), field_line
    assert str(MAX_CHANNELS) in field_line


def test_example_config_documents_every_field() -> None:
    text = example_config_text(Path("/a"), Path("/b"))
    for field in dataclasses.fields(Config):
        assert re.search(rf"^\[?{field.name}\b", text, re.MULTILINE), field.name
    sections = (("budgets", Budgets), ("network", Network), ("asr", Asr), ("jobs", Jobs))
    for section, dataclass_type in sections:
        body = text.split(f"[{section}]", 1)[1]
        for field in dataclasses.fields(dataclass_type):
            assert re.search(rf"^{field.name} = ", body, re.MULTILINE), f"{section}.{field.name}"
