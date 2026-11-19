"""Download and verify the open datasets of the pilot corpus (#78) by the pinned table `SOURCES`.

    uv run python scripts/fetch_corpus_sources.py fetch --corpus-dir data/corpus [NAME...]
    uv run python scripts/fetch_corpus_sources.py verify --corpus-dir data/corpus

`fetch` downloads each file of `sound_scout.eval.corpus.SOURCES` (or only the NAMEs given), one
after another, into `<corpus-dir>/sources/<dataset>/<file>`: HTTPS only, directories 0700, files
0600, no symlink anywhere in the path. The body streams into `<file>.part` while its publisher
checksum (Zenodo MD5, git blob SHA-1) and SHA-256 are computed; after `fsync` the file is read
back from disk and both are computed again. The file is renamed into place only when the size,
the publisher checksum, the stream and the disk digests and, once pinned, the SHA-256 all agree;
otherwise `.part` is removed and the exit code is 1. No automatic retry. A file already in place
is verified, not downloaded again. On the first download the SHA-256 is not pinned yet: the file
is accepted by the publisher checksum and its SHA-256 is printed, to be written into `SOURCES`
and docs/dependencies.md § 5. `verify` re-hashes every file from disk against the table.

The network code lives here, outside the installed package (R09); only the corpus directory is
written, and only under file names taken from the table.
"""

import argparse
import hashlib
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from sound_scout.eval.corpus import (
    HASH_CHUNK,
    SOURCES,
    Source,
    check_source,
    digest_mismatches,
    file_digests,
    publisher_hasher,
)

DIR_MODE = 0o700
FILE_MODE = 0o600
TIMEOUT_S = 60.0
USER_AGENT = "sound-scout-corpus-fetch"

Opener = Callable[[str], object]


class FetchError(Exception):
    pass


def default_opener(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)


def require_https(url: str) -> None:
    if urllib.parse.urlsplit(url).scheme != "https":
        raise FetchError(f"only HTTPS is accepted, got {url!r}")


def safe_dir(corpus_dir: Path, *parts: str) -> Path:
    """Create `corpus_dir/parts...` with 0700; refuse a symlink anywhere on the path."""
    if os.path.realpath(corpus_dir) != os.path.abspath(corpus_dir):
        raise FetchError("--corpus-dir: a symlink in the path is refused")
    path = Path(os.path.abspath(corpus_dir))
    path.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise FetchError(f"{part}: a symlink in the path is refused")
        path.mkdir(mode=DIR_MODE, exist_ok=True)
    return path


def fetch_one(corpus_dir: Path, source: Source, opener: Opener = default_opener) -> str:
    """Download, verify twice and move one file into place; return its SHA-256."""
    require_https(source.url)
    directory = safe_dir(corpus_dir, "sources", source.dataset)
    target = directory / source.name
    if os.path.lexists(target):
        problems = check_source(corpus_dir, source, require_pin=source.sha256 is not None)
        if problems:
            raise FetchError("; ".join(problems) + " (already in place: delete it to download)")
        return file_digests(target, source)[2]
    part = directory / f"{source.name}.part"
    if os.path.lexists(part):
        os.unlink(part)
    try:
        stream_digests = _download(source, part, opener)
        disk_digests = file_digests(part, source)
        problems = digest_mismatches(source, *stream_digests, require_pin=source.sha256 is not None)
        if disk_digests != stream_digests:
            problems.append(
                f"the file read back from disk {disk_digests[1:]} differs from the stream "
                f"{stream_digests[1:]}"
            )
        if problems:
            raise FetchError(f"{source.relative_path}: " + "; ".join(problems))
        os.rename(part, target)
    finally:
        if os.path.lexists(part):
            os.unlink(part)
    return disk_digests[2]


def _download(source: Source, part: Path, opener: Opener) -> tuple[int, str, str]:
    publisher = publisher_hasher(source)
    sha256 = hashlib.sha256()
    size = 0
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    with os.fdopen(fd, "wb") as out, opener(source.url) as response:
        final_url = getattr(response, "geturl", lambda: source.url)()
        require_https(final_url)
        while chunk := response.read(HASH_CHUNK):
            size += len(chunk)
            if size > source.size:
                raise FetchError(f"{source.relative_path}: more than the pinned {source.size} B")
            publisher.update(chunk)
            sha256.update(chunk)
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    algorithm = source.checksum.split(":", 1)[0]
    return size, f"{algorithm}:{publisher.hexdigest()}", sha256.hexdigest()


def selected(names: list[str]) -> list[Source]:
    known = {source.name: source for source in SOURCES}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise FetchError(f"not in SOURCES: {', '.join(unknown)}")
    return [known[name] for name in names] if names else list(SOURCES)


def main(argv: list[str] | None = None, opener: Opener = default_opener) -> int:
    parser = argparse.ArgumentParser(prog="scripts/fetch_corpus_sources.py")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download and verify the pinned files")
    fetch.add_argument("--corpus-dir", type=Path, required=True)
    fetch.add_argument("names", nargs="*", help="file names from SOURCES (default: all)")
    verify = commands.add_parser("verify", help="re-hash every pinned file from disk")
    verify.add_argument("--corpus-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        failed = False
        for source in SOURCES:
            problems = check_source(args.corpus_dir, source)
            failed = failed or bool(problems)
            print("\n".join(problems) if problems else f"{source.relative_path}: ok")
        return 1 if failed else 0
    try:
        sources = selected(args.names)
    except FetchError as error:
        print(f"fetch: {error}", file=sys.stderr)
        return 2
    for source in sources:
        try:
            sha256 = fetch_one(args.corpus_dir, source, opener)
        except (FetchError, OSError) as error:
            print(f"fetch: {error}", file=sys.stderr)
            return 1
        print(f"{source.relative_path}: ok sha256 {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
