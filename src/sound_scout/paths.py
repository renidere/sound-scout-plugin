"""Path boundaries (R10): input only from the allowed roots, output only under the data root.

`resolve_input` turns a caller-supplied source into a canonical absolute path or raises
`PathDenied` with the reason. A URL, an ffmpeg pseudo-protocol (`concat:`, `pipe:`, `data:`,
`subfile,…:`), `-` or any other non-file source is rejected before the filesystem is touched.
The path is resolved with its `..` components intact (collapsing them lexically could select a
different file through a symlinked directory); the resolved path must lie inside one of
`Config.allowed_roots`, which the loader stores resolved too, so a root spelled through a
symlink still matches. It must be a regular file (no directory, device or FIFO), and no
component below the root may be a denied name (`config.DENIED_NAME_PATTERNS`): `.env*` and the
operator config, `.ssh` and `.hermes` directories, and — by file name only, a first filter
before the decoder's own format and protocol whitelists — playlist and script files that would
make the decoder pull in other files or the network. A denied name is denied in any position,
so `~/.ssh/*` and `~/.hermes/*` are rejected wherever the home directory is, even inside an
allowed root. An allowed root that itself contains a denied component is not a supported
configuration: `config.load()` rejects it, and a `Config` built by hand with such a root has
every path under it refused loudly with `ROOT_HAS_DENIED_NAME` instead of looking like a denied
file. A path from a transcript gets no exception: the caller resolves it like any other.

`open_input` opens a resolved path by descriptor: every directory component is opened with
`O_PATH` (search permission is enough, no read permission on ancestors) and `O_NOFOLLOW`
relative to the previous one, then the file likewise, so a symlink planted anywhere in the path
after `resolve_input` fails instead of being followed, and the regular-file check is made with
`fstat` on the open descriptor, never by name. The path is checked against the roots again, so
the helper cannot be pointed outside them.

`resolve_output` names a file inside a run directory under `Config.data_root`: the name is one
plain component (no separators, no `..`, not absolute) and the run directory must lie under the
data root after symlink resolution. `open_output` creates that file exclusively (`O_EXCL`,
`O_NOFOLLOW`, mode 0600) through the same descriptor walk.
"""

import errno
import fcntl
import os
import re
import stat
from pathlib import Path
from typing import BinaryIO

from sound_scout.config import DENIED_NAME_PATTERNS as DENIED_NAME_PATTERNS
from sound_scout.config import Config
from sound_scout.config import is_denied_name as is_denied_name

OUTPUT_NAME = re.compile(r"[A-Za-z0-9._-]+")
FILE_MODE = 0o600

NOT_A_FILE_PATH = "not a file path"
OUTSIDE_ROOTS = "outside allowed roots"
OUTSIDE_DATA_ROOT = "outside data root"
SYMLINK_OUT = "symlink leads outside"
NOT_FOUND = "does not exist"
NOT_REGULAR_FILE = "not a regular file"
NOT_A_DIRECTORY = "not a directory"
DENIED_NAME = "denied name"
ROOT_HAS_DENIED_NAME = "allowed root contains a denied name"
BAD_OUTPUT_NAME = "invalid output name"
CHANGED_AFTER_RESOLVE = "path changed after resolve"
INPUT_PLACEHOLDER = "<input>"
ALLOWED_ROOT_PLACEHOLDER = "<allowed root>"
DATA_ROOT_PLACEHOLDER = "<data root>"


class PathDenied(Exception):
    """The path is refused by policy; `reason` says why, `path` is the path that was checked.

    For `resolve_input` and `resolve_output` that is the source as given; for a descriptor walk
    that fails after resolution it is the resolved target.
    """

    def __init__(self, path: object, reason: str) -> None:
        super().__init__(f"{reason}: {path}")
        self.path = str(path)
        self.reason = reason


def resolve_input(path: str | os.PathLike[str], config: Config) -> Path:
    """Canonical absolute path of an input file inside an allowed root, or `PathDenied`."""
    source = os.fspath(path)
    if not _is_file_path(source):
        raise PathDenied(source, NOT_A_FILE_PATH)
    given = os.path.join(os.getcwd(), source)
    lexical = Path(os.path.normpath(given))
    try:
        resolved = Path(os.path.realpath(given, strict=True))
    except OSError as error:
        if not _under_any(Path(os.path.realpath(given)), config.allowed_roots):
            raise PathDenied(source, _outside_reason(lexical, config.allowed_roots)) from None
        if error.errno == errno.ENOENT:
            raise PathDenied(source, NOT_FOUND) from None
        raise PathDenied(source, f"cannot resolve: {error.strerror}") from None
    root = _root_of(resolved, config.allowed_roots)
    if root is None:
        raise PathDenied(source, _outside_reason(lexical, config.allowed_roots))
    _require_allowed_names(resolved, root, source)
    if not stat.S_ISREG(os.lstat(resolved).st_mode):
        raise PathDenied(source, NOT_REGULAR_FILE)
    return resolved


def open_input(resolved: Path, config: Config) -> BinaryIO:
    """Open a path from `resolve_input` for reading by descriptor; see the module docstring."""
    source = os.fspath(resolved)
    root = _root_of(resolved, config.allowed_roots) if _is_canonical(resolved) else None
    if root is None:
        raise PathDenied(source, OUTSIDE_ROOTS)
    _require_allowed_names(resolved, root, source)
    fd = _open_nofollow(resolved, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PathDenied(source, NOT_REGULAR_FILE)
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb")


def resolve_output(run_dir: Path, name: str, config: Config) -> Path:
    """Path of a new file `name` inside `run_dir`, which must lie under the data root."""
    if not isinstance(name, str) or not OUTPUT_NAME.fullmatch(name) or name in (".", ".."):
        raise PathDenied(name, BAD_OUTPUT_NAME)
    source = os.fspath(run_dir)
    if not _is_file_path(source):
        raise PathDenied(source, NOT_A_FILE_PATH)
    given = Path(os.path.abspath(source))
    if not given.is_relative_to(config.data_root):
        raise PathDenied(source, OUTSIDE_DATA_ROOT)
    try:
        resolved = Path(os.path.realpath(given, strict=True))
    except FileNotFoundError:
        raise PathDenied(source, NOT_FOUND) from None
    except OSError as error:
        raise PathDenied(source, f"cannot resolve: {error.strerror}") from None
    if not resolved.is_relative_to(config.data_root):
        raise PathDenied(source, SYMLINK_OUT)
    if not stat.S_ISDIR(os.lstat(resolved).st_mode):
        raise PathDenied(source, NOT_A_DIRECTORY)
    return resolved / name


def open_output(run_dir: Path, name: str, config: Config) -> BinaryIO:
    """Create `name` inside `run_dir` exclusively, mode 0600, and open it for writing."""
    target = resolve_output(run_dir, name, config)
    fd = _open_nofollow(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.fchmod(fd, FILE_MODE)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "wb")


def without_paths(message: str, source: str, config: Config) -> str:
    """`message` with the input's absolute path, the allowed roots and the data root replaced by
    placeholders: ffprobe and ffmpeg name the file they read in the stderr a refusal quotes."""
    given = os.path.join(os.getcwd(), source)
    hidden = {
        os.path.realpath(given): INPUT_PLACEHOLDER,
        os.path.normpath(given): INPUT_PLACEHOLDER,
    }
    hidden |= {str(root): ALLOWED_ROOT_PLACEHOLDER for root in config.allowed_roots}
    hidden[str(config.data_root)] = DATA_ROOT_PLACEHOLDER
    for text in sorted(hidden, key=len, reverse=True):
        message = message.replace(text, hidden[text])
    return message


def _is_file_path(source: str) -> bool:
    """A plain path: not empty, no NUL, not stdin, no scheme (`x:` before the first `/`)."""
    if not source or "\0" in source or source == "-":
        return False
    return ":" not in source.split("/", 1)[0]


def _is_canonical(path: Path) -> bool:
    source = os.fspath(path)
    return path.is_absolute() and os.path.normpath(source) == source


def _under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    return _root_of(path, roots) is not None


def _root_of(path: Path, roots: tuple[Path, ...]) -> Path | None:
    return next((root for root in roots if path.is_relative_to(root)), None)


def _outside_reason(lexical: Path, roots: tuple[Path, ...]) -> str:
    """A path that looks inside a root but resolves outside left it through a symlink."""
    return SYMLINK_OUT if _under_any(lexical, roots) else OUTSIDE_ROOTS


def _require_allowed_names(path: Path, root: Path, source: str) -> None:
    if any(is_denied_name(part) for part in root.parts):
        raise PathDenied(source, ROOT_HAS_DENIED_NAME)
    for part in path.relative_to(root).parts:
        if is_denied_name(part):
            raise PathDenied(source, DENIED_NAME)


def _open_nofollow(path: Path, flags: int) -> int:
    """Open an absolute, normalized path by descriptor, refusing a symlink in any component."""
    search_only = getattr(os, "O_PATH", os.O_RDONLY)
    directory_flags = search_only | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    *directories, name = path.parts
    try:
        dir_fd = os.open(directories[0], directory_flags)
        try:
            for part in directories[1:]:
                child = os.open(part, directory_flags, dir_fd=dir_fd)
                os.close(dir_fd)
                dir_fd = child
            return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, FILE_MODE, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOENT, errno.ENOTDIR):
            raise PathDenied(path, CHANGED_AFTER_RESOLVE) from None
        raise
