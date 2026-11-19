"""scripts/fetch_corpus_sources.py (#78): pinned, twice-verified downloads; no network in tests."""

import dataclasses
import hashlib
import importlib.util
import io
import os
import stat
from pathlib import Path

import pytest

from sound_scout.eval import corpus

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "fetch_corpus_sources.py"

spec = importlib.util.spec_from_file_location("fetch_corpus_sources", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

BODY = b"RIFF fake corpus file " * 1000


class Response(io.BytesIO):
    def __init__(self, data: bytes, url: str) -> None:
        super().__init__(data)
        self.url = url

    def geturl(self) -> str:
        return self.url


def opener_for(data: bytes, calls: list[str] | None = None, final_url: str | None = None):
    def opener(url: str) -> Response:
        if calls is not None:
            calls.append(url)
        return Response(data, final_url or url)

    return opener


def md5_source(data: bytes = BODY, **changes) -> corpus.Source:
    source = corpus.Source(
        dataset="demand",
        name="TEST_16k.zip",
        url="https://zenodo.org/records/1/files/TEST_16k.zip?download=1",
        size=len(data),
        checksum="md5:" + hashlib.md5(data).hexdigest(),
        sha256=None,
        version="1.0",
        license="CC BY-SA 3.0",
    )
    return dataclasses.replace(source, **changes)


def blob_source(data: bytes = BODY) -> corpus.Source:
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return md5_source(
        data, dataset="ms-snsd", name="Babble_9.wav", checksum=f"git-blob-sha1:{blob}"
    )


def test_git_blob_sha1_matches_the_known_vector_of_git_hash_object() -> None:
    # `printf 'hello\n' | git hash-object --stdin`
    source = md5_source(b"hello\n", checksum="git-blob-sha1:x")
    hasher = corpus.publisher_hasher(source)
    hasher.update(b"hello\n")
    assert hasher.hexdigest() == "ce013625030ba8dba906f756967f9e9ca394464a"


@pytest.mark.parametrize("make", [md5_source, blob_source])
def test_fetch_places_the_file_0600_in_0700_directories(tmp_path: Path, make) -> None:
    source = make()
    sha256 = fetch.fetch_one(tmp_path, source, opener_for(BODY))
    target = tmp_path / source.relative_path
    assert target.read_bytes() == BODY
    assert sha256 == hashlib.sha256(BODY).hexdigest()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "sources").stat().st_mode) == 0o700
    assert not (target.parent / f"{source.name}.part").exists()


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (md5_source(checksum="md5:" + "0" * 32), "publisher checksum"),
        (
            dataclasses.replace(blob_source(), checksum="git-blob-sha1:" + "0" * 40),
            "publisher checksum",
        ),
        (md5_source(size=len(BODY) + 1), "size"),
        (md5_source(size=len(BODY) - 1), "more than the pinned"),
        (md5_source(sha256="0" * 64), "SHA-256"),
    ],
)
def test_a_mismatch_removes_the_part_file_and_keeps_nothing(tmp_path, source, reason) -> None:
    with pytest.raises(fetch.FetchError, match=reason):
        fetch.fetch_one(tmp_path, source, opener_for(BODY))
    assert list((tmp_path / "sources" / source.dataset).iterdir()) == []


def test_main_exits_1_and_names_the_mismatch(tmp_path, monkeypatch, capsys) -> None:
    bad = md5_source(checksum="md5:" + "0" * 32)
    monkeypatch.setattr(fetch, "SOURCES", (bad,))
    code = fetch.main(["fetch", "--corpus-dir", str(tmp_path)], opener=opener_for(BODY))
    assert code == 1
    assert "publisher checksum" in capsys.readouterr().err
    assert list((tmp_path / "sources" / "demand").iterdir()) == []


def test_a_file_changed_between_the_stream_and_the_disk_is_refused(tmp_path, monkeypatch) -> None:
    real_fsync = os.fsync

    def flip_a_byte_then_fsync(fd: int) -> None:
        os.pwrite(fd, b"X", 10)
        real_fsync(fd)

    monkeypatch.setattr(fetch.os, "fsync", flip_a_byte_then_fsync)
    source = md5_source()
    with pytest.raises(fetch.FetchError, match="read back from disk"):
        fetch.fetch_one(tmp_path, source, opener_for(BODY))
    assert not (tmp_path / source.relative_path).exists()


def test_a_file_in_place_is_verified_not_downloaded(tmp_path: Path) -> None:
    source = md5_source(sha256=hashlib.sha256(BODY).hexdigest())
    fetch.fetch_one(tmp_path, source, opener_for(BODY))
    calls: list[str] = []
    assert fetch.fetch_one(tmp_path, source, opener_for(BODY, calls)) == source.sha256
    assert calls == []
    (tmp_path / source.relative_path).write_bytes(BODY[:-1] + b"Y")
    with pytest.raises(fetch.FetchError, match="already in place"):
        fetch.fetch_one(tmp_path, source, opener_for(BODY, calls))
    assert calls == []


@pytest.mark.parametrize(
    "url", ["http://zenodo.org/x.zip", "file:///etc/passwd", "ftp://example.org/x.zip"]
)
def test_only_https_is_accepted(tmp_path: Path, url: str) -> None:
    calls: list[str] = []
    with pytest.raises(fetch.FetchError, match="only HTTPS"):
        fetch.fetch_one(tmp_path, md5_source(url=url), opener_for(BODY, calls))
    assert calls == []


def test_a_redirect_away_from_https_is_refused(tmp_path: Path) -> None:
    source = md5_source()
    with pytest.raises(fetch.FetchError, match="only HTTPS"):
        fetch.fetch_one(tmp_path, source, opener_for(BODY, final_url="http://example.org/x"))
    assert not (tmp_path / source.relative_path).exists()


def test_a_symlink_in_the_path_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "corpus").symlink_to(real)
    with pytest.raises(fetch.FetchError, match="symlink"):
        fetch.fetch_one(tmp_path / "corpus", md5_source(), opener_for(BODY))
    inner = tmp_path / "inner"
    (inner / "sources").mkdir(parents=True)
    (inner / "sources" / "demand").symlink_to(real)
    with pytest.raises(fetch.FetchError, match="symlink"):
        fetch.fetch_one(inner, md5_source(), opener_for(BODY))
    assert list(real.iterdir()) == []


def test_verify_rehashes_every_pinned_file(tmp_path, monkeypatch, capsys) -> None:
    good = md5_source(sha256=hashlib.sha256(BODY).hexdigest())
    missing = dataclasses.replace(good, name="OTHER_16k.zip")
    monkeypatch.setattr(fetch, "SOURCES", (good,))
    assert fetch.main(["fetch", "--corpus-dir", str(tmp_path)], opener=opener_for(BODY)) == 0
    assert fetch.main(["verify", "--corpus-dir", str(tmp_path)]) == 0
    assert "TEST_16k.zip: ok" in capsys.readouterr().out
    monkeypatch.setattr(fetch, "SOURCES", (good, missing))
    assert fetch.main(["verify", "--corpus-dir", str(tmp_path)]) == 1
    assert "OTHER_16k.zip: missing" in capsys.readouterr().out


def test_unknown_names_are_refused(tmp_path: Path, capsys) -> None:
    assert fetch.main(["fetch", "--corpus-dir", str(tmp_path), "nope.zip"]) == 2
    assert "not in SOURCES" in capsys.readouterr().err


def test_the_committed_table_pins_every_file_over_https() -> None:
    names = [source.name for source in corpus.SOURCES]
    assert len(names) == len(set(names))
    for source in corpus.SOURCES:
        assert source.url.startswith("https://")
        assert source.sha256 is not None and len(source.sha256) == 64, source.name
        assert source.dataset in ("rinco", "demand", "ms-snsd")
