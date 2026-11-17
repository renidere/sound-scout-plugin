import os
import stat
from pathlib import Path

import pytest

from sound_scout import paths
from sound_scout.config import CONFIG_FILE_NAME, Config
from sound_scout.paths import (
    PathDenied,
    open_input,
    open_output,
    resolve_input,
    resolve_output,
)

GOOD = b"RIFF good"
SECRET = b"secret"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    directory = tmp_path.resolve() / "recordings"
    directory.mkdir()
    return directory


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    directory = tmp_path.resolve() / "data"
    directory.mkdir()
    return directory


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    directory = tmp_path.resolve() / "outside"
    directory.mkdir()
    (directory / "secret.wav").write_bytes(SECRET)
    return directory


@pytest.fixture
def config(root: Path, data_root: Path) -> Config:
    return Config(allowed_roots=(root,), data_root=data_root)


@pytest.fixture
def good(root: Path) -> Path:
    path = root / "good.wav"
    path.write_bytes(GOOD)
    return path


def denied(path: object, config: Config) -> str:
    with pytest.raises(PathDenied) as info:
        resolve_input(path, config)  # type: ignore[arg-type]
    assert info.value.path == str(path)
    return info.value.reason


# resolve_input: accepted paths


def test_regular_file_inside_root_resolves_to_itself(good: Path, config: Config) -> None:
    assert resolve_input(good, config) == good
    assert resolve_input(str(good), config) == good


def test_relative_path_and_dotdot_staying_inside_root(
    good: Path, root: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "sub").mkdir()
    monkeypatch.chdir(root)
    assert resolve_input("good.wav", config) == good
    assert resolve_input("sub/../good.wav", config) == good
    assert resolve_input(root / "sub" / ".." / "good.wav", config) == good


def test_symlink_inside_root_resolves_to_its_target(good: Path, root: Path, config: Config) -> None:
    link = root / "link.wav"
    link.symlink_to(good)
    assert resolve_input(link, config) == good


def test_file_in_second_root_is_accepted(tmp_path: Path, config: Config) -> None:
    other = tmp_path.resolve() / "other"
    other.mkdir()
    wav = other / "a.wav"
    wav.write_bytes(GOOD)
    two_roots = Config(allowed_roots=(*config.allowed_roots, other), data_root=config.data_root)
    assert resolve_input(wav, two_roots) == wav
    assert denied(wav, config) == paths.OUTSIDE_ROOTS


def test_root_spelled_through_a_symlink_matches_the_resolved_root(
    tmp_path: Path, good: Path, root: Path, config: Config
) -> None:
    alias = tmp_path.resolve() / "alias"
    alias.symlink_to(root)
    assert resolve_input(alias / "good.wav", config) == good
    assert denied(alias / "missing.wav", config) == paths.NOT_FOUND


# resolve_input: outside the roots


def test_dotdot_outward_is_outside_roots(root: Path, outside: Path, config: Config) -> None:
    (root / "sub").mkdir()
    assert denied(root / ".." / "outside" / "secret.wav", config) == paths.OUTSIDE_ROOTS
    assert denied(root / "sub" / ".." / ".." / "outside" / "secret.wav", config) == (
        paths.OUTSIDE_ROOTS
    )
    assert denied(outside / "secret.wav", config) == paths.OUTSIDE_ROOTS


def test_root_prefix_of_another_directory_is_outside(
    tmp_path: Path, root: Path, config: Config
) -> None:
    sibling = tmp_path.resolve() / "recordings-other"
    sibling.mkdir()
    (sibling / "a.wav").write_bytes(GOOD)
    assert denied(sibling / "a.wav", config) == paths.OUTSIDE_ROOTS


def test_symlink_to_file_outside_roots(root: Path, outside: Path, config: Config) -> None:
    link = root / "link.wav"
    link.symlink_to(outside / "secret.wav")
    assert denied(link, config) == paths.SYMLINK_OUT


def test_symlink_directory_leading_outside(root: Path, outside: Path, config: Config) -> None:
    (root / "linkdir").symlink_to(outside)
    assert denied(root / "linkdir" / "secret.wav", config) == paths.SYMLINK_OUT
    assert denied(root / "linkdir", config) == paths.SYMLINK_OUT


def test_symlink_directory_inside_root_is_not_a_regular_file(root: Path, config: Config) -> None:
    (root / "sub").mkdir()
    (root / "linkdir").symlink_to(root / "sub")
    assert denied(root / "linkdir", config) == paths.NOT_REGULAR_FILE


def test_outside_path_gets_the_same_reason_whether_or_not_it_exists(
    root: Path, outside: Path, config: Config
) -> None:
    assert denied(outside / "secret.wav", config) == paths.OUTSIDE_ROOTS
    assert denied(outside / "missing.wav", config) == paths.OUTSIDE_ROOTS
    (root / "linkdir").symlink_to(outside)
    assert denied(root / "linkdir" / "secret.wav", config) == paths.SYMLINK_OUT
    assert denied(root / "linkdir" / "missing.wav", config) == paths.SYMLINK_OUT


def test_dotdot_after_a_symlinked_directory_follows_the_symlink_not_the_text(
    root: Path, outside: Path, config: Config
) -> None:
    (outside / "sub").mkdir()
    (root / "linkdir").symlink_to(outside / "sub")
    (root / "x.wav").write_bytes(GOOD)
    (outside / "x.wav").write_bytes(SECRET)
    assert denied(root / "linkdir" / ".." / "x.wav", config) == paths.SYMLINK_OUT


# resolve_input: not a regular file


def test_directory_is_rejected(root: Path, config: Config) -> None:
    (root / "sub").mkdir()
    assert denied(root / "sub", config) == paths.NOT_REGULAR_FILE
    assert denied(root, config) == paths.NOT_REGULAR_FILE


def test_fifo_is_rejected(root: Path, config: Config) -> None:
    fifo = root / "pipe.wav"
    os.mkfifo(fifo)
    assert denied(fifo, config) == paths.NOT_REGULAR_FILE


@pytest.mark.skipif(not Path("/dev/zero").exists(), reason="no /dev/zero here")
def test_device_is_rejected_inside_a_root_and_through_a_symlink(
    root: Path, data_root: Path, config: Config
) -> None:
    dev = Config(allowed_roots=(Path("/dev"),), data_root=data_root)
    assert denied("/dev/zero", dev) == paths.NOT_REGULAR_FILE
    (root / "zero.wav").symlink_to("/dev/zero")
    assert denied(root / "zero.wav", config) == paths.SYMLINK_OUT
    assert denied("/dev/zero", config) == paths.OUTSIDE_ROOTS


def test_missing_file_is_denied(root: Path, config: Config) -> None:
    assert denied(root / "missing.wav", config) == paths.NOT_FOUND


def test_dangling_symlink_and_symlink_loop_are_denied(root: Path, config: Config) -> None:
    (root / "dangling.wav").symlink_to(root / "missing.wav")
    assert denied(root / "dangling.wav", config) == paths.NOT_FOUND
    (root / "loop.wav").symlink_to(root / "loop.wav")
    assert denied(root / "loop.wav", config).startswith("cannot resolve")


# resolve_input: denied names inside an allowed root


@pytest.mark.parametrize(
    "name", [".env", ".env.local", ".envrc", CONFIG_FILE_NAME, "SOUND-SCOUT.TOML", "list.m3u8"]
)
def test_secret_config_and_playlist_names_are_denied(name: str, root: Path, config: Config) -> None:
    path = root / name
    path.write_bytes(GOOD)
    assert denied(path, config) == paths.DENIED_NAME


def test_denied_name_applies_to_any_component_and_to_the_symlink_target(
    root: Path, config: Config
) -> None:
    (root / ".env").mkdir()
    inner = root / ".env" / "tone.wav"
    inner.write_bytes(GOOD)
    assert denied(inner, config) == paths.DENIED_NAME
    secret = root / CONFIG_FILE_NAME
    secret.write_bytes(GOOD)
    (root / "link.wav").symlink_to(secret)
    assert denied(root / "link.wav", config) == paths.DENIED_NAME


def test_ssh_and_hermes_under_a_fake_home_are_denied_even_inside_a_root(
    tmp_path: Path, data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path.resolve() / "home"
    for directory in (".ssh", ".hermes"):
        (home / directory).mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_bytes(SECRET)
    (home / ".hermes" / "config.yaml").write_bytes(SECRET)
    (home / "voice.wav").write_bytes(GOOD)
    monkeypatch.setenv("HOME", str(home))
    assert Path.home() == home
    config = Config(allowed_roots=(home,), data_root=data_root)
    assert denied(Path("~/.ssh/id_ed25519").expanduser(), config) == paths.DENIED_NAME
    assert denied(Path("~/.hermes/config.yaml").expanduser(), config) == paths.DENIED_NAME
    assert resolve_input(home / "voice.wav", config) == home / "voice.wav"


def test_allowed_root_with_a_denied_component_fails_loudly(tmp_path: Path, data_root: Path) -> None:
    hermes_root = tmp_path.resolve() / ".hermes" / "recordings"
    hermes_root.mkdir(parents=True)
    wav = hermes_root / "voice.wav"
    wav.write_bytes(GOOD)
    config = Config(allowed_roots=(hermes_root,), data_root=data_root)
    assert denied(wav, config) == paths.ROOT_HAS_DENIED_NAME
    with pytest.raises(PathDenied) as info:
        open_input(wav, config)
    assert info.value.reason == paths.ROOT_HAS_DENIED_NAME
    assert paths.ROOT_HAS_DENIED_NAME != paths.DENIED_NAME


def test_denied_name_list_lives_in_one_place() -> None:
    for pattern in (".env*", CONFIG_FILE_NAME, ".ssh", ".hermes"):
        assert pattern in paths.DENIED_NAME_PATTERNS
    assert paths.is_denied_name(".hermes") and paths.is_denied_name("x.M3U")
    assert not paths.is_denied_name("tone.wav")


# resolve_input: non-file sources are rejected before the filesystem


def test_file_url_to_an_existing_allowed_file_is_still_rejected(good: Path, config: Config) -> None:
    assert denied(f"file://{good}", config) == paths.NOT_A_FILE_PATH


@pytest.mark.parametrize(
    "source",
    [
        "http://example.invalid/a.wav",
        "https://example.invalid/a.wav",
        "rtsp://example.invalid/stream",
        "rtmp://example.invalid/live",
        "ftp://example.invalid/a.wav",
        "file:///etc/passwd",
        "concat:a.wav|b.wav",
        "playlist:list",
        "hls:list.m3u8",
        "pipe:0",
        "data:audio/wav;base64,UklGRg==",
        "subfile,,start,0,end,1,,:a.wav",
        "-",
        "",
        "rec\0ording.wav",
    ],
)
def test_non_file_sources_are_rejected_before_any_filesystem_or_decoder_call(
    source: str, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"touched before rejection: {args[:1]}")

    with monkeypatch.context() as patch:
        for module, name in (
            (os, "open"),
            (os, "stat"),
            (os, "lstat"),
            (os, "access"),
            (os, "listdir"),
            (os, "scandir"),
            (os.path, "realpath"),
            (os.path, "exists"),
            (os.path, "isfile"),
            (subprocess, "run"),
            (subprocess, "Popen"),
            (shutil, "which"),
        ):
            patch.setattr(module, name, forbidden)
        patch.setattr("builtins.open", forbidden)
        with pytest.raises(PathDenied) as info:
            resolve_input(source, config)
    assert info.value.reason == paths.NOT_A_FILE_PATH
    assert info.value.path == source


def test_colon_after_the_first_separator_is_a_plain_path(root: Path, config: Config) -> None:
    wav = root / "take 12:30.wav"
    wav.write_bytes(GOOD)
    assert resolve_input(wav, config) == wav


# open_input: by descriptor, swap after resolve fails


def test_open_input_reads_the_resolved_file(good: Path, config: Config) -> None:
    resolved = resolve_input(good, config)
    with open_input(resolved, config) as stream:
        assert stream.read() == GOOD
        assert stat.S_ISREG(os.fstat(stream.fileno()).st_mode)
        assert os.get_blocking(stream.fileno())
    assert stream.closed


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_open_input_needs_only_search_permission_on_ancestors(root: Path, config: Config) -> None:
    sub = root / "sub"
    sub.mkdir()
    wav = sub / "tone.wav"
    wav.write_bytes(GOOD)
    sub.chmod(0o111)
    try:
        with pytest.raises(PermissionError):
            os.listdir(sub)
        with open_input(resolve_input(wav, config), config) as stream:
            assert stream.read() == GOOD
    finally:
        sub.chmod(0o755)


def test_symlink_replaced_after_resolve_does_not_redirect_the_read(
    good: Path, root: Path, outside: Path, config: Config
) -> None:
    link = root / "link.wav"
    link.symlink_to(good)
    resolved = resolve_input(link, config)
    link.unlink()
    link.symlink_to(outside / "secret.wav")
    with open_input(resolved, config) as stream:
        assert stream.read() == GOOD


def test_file_replaced_by_symlink_after_resolve_is_denied(
    good: Path, outside: Path, config: Config
) -> None:
    resolved = resolve_input(good, config)
    good.unlink()
    good.symlink_to(outside / "secret.wav")
    with pytest.raises(PathDenied) as info:
        open_input(resolved, config)
    assert info.value.reason == paths.CHANGED_AFTER_RESOLVE


def test_descriptor_open_refuses_a_swap_that_a_by_name_open_follows(
    good: Path, outside: Path, config: Config
) -> None:
    resolved = resolve_input(good, config)
    assert stat.S_ISREG(os.lstat(resolved).st_mode)
    good.unlink()
    good.symlink_to(outside / "secret.wav")
    # The swap is real: by name the path still "is a regular file" and reads the secret.
    assert stat.S_ISREG(os.stat(resolved).st_mode)
    with open(resolved, "rb") as by_name:
        assert by_name.read() == SECRET
    with pytest.raises(PathDenied) as info:
        open_input(resolved, config)
    assert info.value.reason == paths.CHANGED_AFTER_RESOLVE


def test_open_input_checks_the_descriptor_not_the_name(
    good: Path, outside: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = resolve_input(good, config)

    def by_name(*args: object, **kwargs: object) -> None:
        raise AssertionError("checked by name")

    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", by_name)
        patch.setattr(os, "lstat", by_name)
        patch.setattr(os.path, "realpath", by_name)
        patch.setattr("builtins.open", by_name)
        stream = open_input(resolved, config)
    with stream:
        good.unlink()
        good.symlink_to(outside / "secret.wav")
        assert stat.S_ISREG(os.fstat(stream.fileno()).st_mode)
        assert stream.read() == GOOD
    with open(resolved, "rb") as by_name_now:
        assert by_name_now.read() == SECRET


def test_directory_replaced_by_symlink_after_resolve_is_denied(
    root: Path, outside: Path, config: Config
) -> None:
    sub = root / "sub"
    sub.mkdir()
    (sub / "secret.wav").write_bytes(GOOD)
    resolved = resolve_input(sub / "secret.wav", config)
    sub.rename(root / "sub.moved")
    sub.symlink_to(outside)
    with pytest.raises(PathDenied) as info:
        open_input(resolved, config)
    assert info.value.reason == paths.CHANGED_AFTER_RESOLVE


def test_file_replaced_by_fifo_after_resolve_is_denied_without_blocking(
    good: Path, config: Config
) -> None:
    resolved = resolve_input(good, config)
    good.unlink()
    os.mkfifo(good)
    with pytest.raises(PathDenied) as info:
        open_input(resolved, config)
    assert info.value.reason == paths.NOT_REGULAR_FILE


def test_file_removed_after_resolve_is_denied(good: Path, config: Config) -> None:
    resolved = resolve_input(good, config)
    good.unlink()
    with pytest.raises(PathDenied) as info:
        open_input(resolved, config)
    assert info.value.reason == paths.CHANGED_AFTER_RESOLVE


def test_open_input_refuses_a_path_it_did_not_resolve(
    root: Path, outside: Path, config: Config
) -> None:
    for path in (outside / "secret.wav", root / ".." / "outside" / "secret.wav"):
        with pytest.raises(PathDenied) as info:
            open_input(path, config)
        assert info.value.reason == paths.OUTSIDE_ROOTS
    (root / ".env").write_bytes(SECRET)
    with pytest.raises(PathDenied) as info:
        open_input(root / ".env", config)
    assert info.value.reason == paths.DENIED_NAME


# resolve_output / open_output: only inside the data root


@pytest.fixture
def run_dir(data_root: Path) -> Path:
    directory = data_root / "runs" / "0123456789abcdef0123456789abcdef"
    directory.mkdir(parents=True)
    return directory


def test_plain_name_inside_the_run_dir(run_dir: Path, config: Config) -> None:
    assert resolve_output(run_dir, "report.json", config) == run_dir / "report.json"
    assert resolve_output(run_dir, "clip-001_v2.wav", config) == run_dir / "clip-001_v2.wav"


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../report.json",
        "sub/report.json",
        "/etc/passwd",
        "/report.json",
        "sub\\report.json",
        "report\0.json",
        "report .json",
        "report\n.json",
    ],
)
def test_output_names_with_separators_dotdot_or_absolute_are_rejected(
    name: str, run_dir: Path, config: Config
) -> None:
    with pytest.raises(PathDenied) as info:
        resolve_output(run_dir, name, config)
    assert info.value.reason == paths.BAD_OUTPUT_NAME


def test_run_dir_outside_the_data_root_is_rejected(
    root: Path, outside: Path, data_root: Path, config: Config
) -> None:
    for directory in (outside, root, data_root.parent, data_root / "runs" / ".." / ".."):
        with pytest.raises(PathDenied) as info:
            resolve_output(directory, "report.json", config)
        assert info.value.reason == paths.OUTSIDE_DATA_ROOT, directory


def test_run_dir_symlinked_outside_the_data_root_is_rejected(
    outside: Path, data_root: Path, config: Config
) -> None:
    (outside / "run").mkdir()
    link = data_root / "runs"
    link.symlink_to(outside)
    with pytest.raises(PathDenied) as info:
        resolve_output(link / "run", "report.json", config)
    assert info.value.reason == paths.SYMLINK_OUT


def test_run_dir_must_be_an_existing_directory(data_root: Path, config: Config) -> None:
    with pytest.raises(PathDenied) as info:
        resolve_output(data_root / "missing", "report.json", config)
    assert info.value.reason == paths.NOT_FOUND
    (data_root / "file").write_bytes(GOOD)
    with pytest.raises(PathDenied) as info:
        resolve_output(data_root / "file", "report.json", config)
    assert info.value.reason == paths.NOT_A_DIRECTORY


def test_open_output_creates_a_private_file_once(run_dir: Path, config: Config) -> None:
    with open_output(run_dir, "report.json", config) as stream:
        stream.write(b"{}\n")
    target = run_dir / "report.json"
    assert target.read_bytes() == b"{}\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        open_output(run_dir, "report.json", config)


def test_open_output_refuses_a_planted_symlink_and_a_run_dir_outside_the_data_root(
    run_dir: Path, outside: Path, config: Config
) -> None:
    (run_dir / "report.json").symlink_to(outside / "report.json")
    with pytest.raises(FileExistsError):
        open_output(run_dir, "report.json", config)
    assert not (outside / "report.json").exists()
    with pytest.raises(PathDenied) as info:
        open_output(outside, "report.json", config)
    assert info.value.reason == paths.OUTSIDE_DATA_ROOT


def test_open_output_refuses_a_run_dir_swapped_for_a_symlink_after_resolve(
    run_dir: Path, data_root: Path, outside: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = run_dir / "report.json"
    (outside / run_dir.name).mkdir()
    monkeypatch.setattr(paths, "resolve_output", lambda *args: target)
    runs = data_root / "runs"
    runs.rename(data_root / "runs.moved")
    runs.symlink_to(outside)
    with pytest.raises(PathDenied) as info:
        open_output(run_dir, "report.json", config)
    assert info.value.reason == paths.CHANGED_AFTER_RESOLVE
    assert info.value.path == str(target)
    assert not (outside / run_dir.name / "report.json").exists()
    assert sorted(os.listdir(outside)) == [run_dir.name, "secret.wav"]
