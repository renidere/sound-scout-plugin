import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from sound_scout import __version__, runs
from sound_scout.runs import (
    RequestIndexCorrupt,
    RequestMismatch,
    RequestRunMissing,
    Run,
    complete_run,
    identity_hash,
    open_run,
)

IDENTITY = {
    "audio": "sha256:" + "a" * 64,
    "model": {"name": "whisper", "revision": "r1"},
    "preprocessing": {"mono": True, "sample_rate": 16_000},
    "ranges": [[0.0, 12.5]],
    "prompt_schema": None,
    "task_context": None,
}
HEX_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def contents_and_mtimes(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {
        entry.name: (entry.read_bytes(), entry.stat().st_mtime_ns)
        for entry in sorted(directory.iterdir())
    }


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def permissive_umask() -> Iterator[None]:
    previous = os.umask(0o000)
    try:
        yield
    finally:
        os.umask(previous)


def test_identity_hash_is_canonical_sha256_of_r11_fields() -> None:
    digest = identity_hash({})
    canonical = json.dumps(
        dict.fromkeys(runs.IDENTITY_FIELDS), sort_keys=True, separators=(",", ":")
    )
    assert set(runs.IDENTITY_FIELDS) == {
        "audio",
        "model",
        "preprocessing",
        "ranges",
        "prompt_schema",
        "task_context",
    }
    assert digest == hashlib.sha256(canonical.encode()).hexdigest()
    assert identity_hash({"audio": None, "ranges": None}) == digest
    reordered = {key: IDENTITY[key] for key in reversed(list(IDENTITY))}
    reordered["model"] = {"revision": "r1", "name": "whisper"}
    assert identity_hash(reordered) == identity_hash(IDENTITY)
    assert identity_hash({"audio": "other"}) != digest


def test_identity_hash_rejects_unknown_fields_and_nan() -> None:
    with pytest.raises(ValueError, match="unknown identity fields: \\['tenant'\\]"):
        identity_hash({"tenant": "x"})
    with pytest.raises(ValueError):
        identity_hash({"ranges": float("nan")})


def test_open_run_creates_private_run_dir_with_request_and_manifest(
    data_root: Path, permissive_umask: None
) -> None:
    run = open_run("req/../../etc:passwd", IDENTITY, data_root)

    assert run == Run(
        run_id=run.run_id,
        path=data_root / "runs" / run.run_id,
        request_id="req/../../etc:passwd",
        identity_hash=identity_hash(IDENTITY),
        reused=False,
        complete=False,
        created=True,
    )
    assert HEX_RUN_ID.match(run.run_id)
    assert run.path.is_dir() and run.path.resolve().is_relative_to(data_root.resolve())
    for private_dir in (data_root / "runs", data_root / "runs" / "by-request", run.path):
        assert mode_of(private_dir) == 0o700, private_dir
    request = json.loads((run.path / "request.json").read_text())
    assert request == {
        "request_id": "req/../../etc:passwd",
        "identity_hash": identity_hash(IDENTITY),
        "created_at": request["created_at"],
        "package_version": __version__,
        "task_context": None,
    }
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$", request["created_at"])
    assert json.loads((run.path / "manifest.json").read_text()) == {
        "run_id": run.run_id,
        "complete": False,
    }
    for name in ("request.json", "manifest.json"):
        assert mode_of(run.path / name) == 0o600, name
    index_entries = list((data_root / "runs" / "by-request").iterdir())
    assert [entry.read_text().strip() for entry in index_entries] == [run.run_id]
    assert mode_of(index_entries[0]) == 0o600
    assert not any("passwd" in str(path) for path in data_root.rglob("*"))


def test_request_json_records_the_task_context_of_the_identity(data_root: Path) -> None:
    context = {"layers": ["sound"], "task": "прочитать абзац", "intent": None}

    run = open_run("with-task", {**IDENTITY, "task_context": context}, data_root)
    without = open_run("without-task", {"audio": None}, data_root)

    assert json.loads((run.path / "request.json").read_text())["task_context"] == context
    assert json.loads((without.path / "request.json").read_text())["task_context"] is None


def test_open_run_rejects_empty_request_id(data_root: Path) -> None:
    with pytest.raises(ValueError, match="request_id"):
        open_run("", IDENTITY, data_root)
    assert not (data_root / "runs").exists() or not list((data_root / "runs").iterdir())


def test_run_ids_are_unique_per_request(data_root: Path) -> None:
    first = open_run("one", IDENTITY, data_root)
    second = open_run("two", IDENTITY, data_root)
    assert first.run_id != second.run_id
    assert first.path != second.path


def test_repeat_with_same_identity_reuses_complete_run_without_work(data_root: Path) -> None:
    work_calls = 0

    def analyze(request_id: str) -> Run:
        nonlocal work_calls
        run = open_run(request_id, IDENTITY, data_root)
        if run.reused:
            return run
        work_calls += 1
        return complete_run(run)

    first = analyze("req-1")
    before = {name: (first.path / name).read_bytes() for name in os.listdir(first.path)}

    second = analyze("req-1")

    assert work_calls == 1
    assert second == Run(
        first.run_id,
        first.path,
        "req-1",
        first.identity_hash,
        reused=True,
        complete=True,
        created=False,
    )
    assert {name: (first.path / name).read_bytes() for name in os.listdir(first.path)} == before
    assert [p.name for p in (data_root / "runs").iterdir() if p.name != "by-request"] == [
        first.run_id
    ]


def test_complete_run_marks_manifest_and_keeps_file_private(
    data_root: Path, permissive_umask: None
) -> None:
    run = open_run("req", IDENTITY, data_root)
    done = complete_run(run)
    assert done.complete is True and done.reused is False and done.run_id == run.run_id
    assert json.loads((run.path / "manifest.json").read_text()) == {
        "run_id": run.run_id,
        "complete": True,
    }
    assert mode_of(run.path / "manifest.json") == 0o600
    assert sorted(os.listdir(run.path)) == ["manifest.json", "request.json"]
    assert complete_run(done).complete is True


def test_same_request_with_different_identity_is_rejected_unchanged(data_root: Path) -> None:
    run = complete_run(open_run("req", IDENTITY, data_root))
    snapshot = contents_and_mtimes(run.path)

    with pytest.raises(RequestMismatch, match="'req'"):
        open_run("req", {**IDENTITY, "ranges": [[0.0, 13.0]]}, data_root)
    with pytest.raises(RequestMismatch):
        open_run("req", {}, data_root)

    assert contents_and_mtimes(run.path) == snapshot
    assert [p.name for p in (data_root / "runs").iterdir() if p.name != "by-request"] == [
        run.run_id
    ]


def test_partial_run_on_repeat_is_not_a_ready_result(data_root: Path) -> None:
    partial = open_run("req", IDENTITY, data_root)

    again = open_run("req", IDENTITY, data_root)

    assert again.reused is False and again.complete is False
    assert again.run_id == partial.run_id and again.path == partial.path
    assert [p.name for p in (data_root / "runs").iterdir() if p.name != "by-request"] == [
        partial.run_id
    ]
    assert complete_run(again).complete is True
    assert open_run("req", IDENTITY, data_root).reused is True


def test_existing_run_directory_is_never_overwritten(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = complete_run(open_run("one", IDENTITY, data_root))
    snapshot = {name: (first.path / name).read_bytes() for name in os.listdir(first.path)}
    monkeypatch.setattr(runs, "new_run_id", lambda: first.run_id)

    with pytest.raises(FileExistsError):
        open_run("two", IDENTITY, data_root)

    assert {name: (first.path / name).read_bytes() for name in os.listdir(first.path)} == snapshot
    assert open_run("one", IDENTITY, data_root).reused is True
    assert len(list((data_root / "runs" / "by-request").iterdir())) == 1


def test_concurrent_claim_of_one_request_yields_one_run(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_new_run_id = runs.new_run_id
    inner: list[Run] = []

    def racing_new_run_id() -> str:
        if not inner:
            monkeypatch.setattr(runs, "new_run_id", real_new_run_id)
            inner.append(complete_run(open_run("req", IDENTITY, data_root)))
            monkeypatch.setattr(runs, "new_run_id", racing_new_run_id)
        return real_new_run_id()

    monkeypatch.setattr(runs, "new_run_id", racing_new_run_id)

    outer = open_run("req", IDENTITY, data_root)

    assert outer.run_id == inner[0].run_id and outer.reused is True
    assert outer.created is False and inner[0].created is True
    assert [p.name for p in (data_root / "runs").iterdir() if p.name != "by-request"] == [
        inner[0].run_id
    ]
    assert [p.name for p in (data_root / "runs" / "by-request").iterdir()] == [
        hashlib.sha256(b"req").hexdigest()
    ]


def test_runs_dir_must_be_a_directory(data_root: Path) -> None:
    (data_root / "runs").write_text("not a directory")
    with pytest.raises(FileExistsError):
        open_run("req", IDENTITY, data_root)


def test_identity_hash_equals_independent_sha256_of_sorted_canonical_json() -> None:
    # The R11 fields are spelled out here on purpose: the expected document must not come from
    # the module under test, and a field the caller did not pass is present as null.
    given = {"model": {"revision": "r1", "name": "whisper"}, "audio": "sha256:abc"}
    document = {
        "audio": "sha256:abc",
        "model": {"name": "whisper", "revision": "r1"},
        "preprocessing": None,
        "prompt_schema": None,
        "ranges": None,
        "task_context": None,
    }
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    assert list(json.loads(text)) == sorted(document)
    assert identity_hash(given) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert identity_hash({**given, "ranges": None}) == identity_hash(given)
    assert identity_hash({**given, "ranges": []}) != identity_hash(given)


def test_run_id_contains_no_raw_request_id_or_identity_values(data_root: Path) -> None:
    request_id = "REQ-zygote-7Q/../../etc:passwd"
    identity = {
        "audio": "sha256:XYZZY-recording",
        "model": {"name": "whisper-large-v3", "revision": "REV-plugh"},
        "preprocessing": {"mono": True},
        "ranges": [[0.0, 1.5]],
        "prompt_schema": "PROMPT-xyzzy",
        "task_context": "CTX-frobnicate",
    }
    raw_values = [request_id, "zygote", "passwd", "XYZZY", "plugh", "xyzzy", "frobnicate"]

    run = open_run(request_id, identity, data_root)

    assert run.run_id and HEX_RUN_ID.match(run.run_id)
    assert not any(value in run.run_id for value in raw_values)
    assert not any(value.lower() in run.run_id for value in raw_values)
    assert run.run_id != hashlib.sha256(request_id.encode()).hexdigest()
    assert run.run_id != identity_hash(identity)
    assert not any(
        value in str(path.relative_to(data_root))
        for path in data_root.rglob("*")
        for value in raw_values
    )
    assert json.loads((run.path / "request.json").read_text())["request_id"] == request_id


def test_run_ids_are_unique_across_repeated_opens_and_roots(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("a", "b", "c")]
    for root in roots:
        root.mkdir()
    seen = {open_run("same-request", IDENTITY, root).run_id for root in roots}
    assert len(seen) == 3
    assert open_run("same-request", IDENTITY, roots[0]).run_id in seen


def test_partial_run_manifest_stays_incomplete_until_complete_run(data_root: Path) -> None:
    first = open_run("req", IDENTITY, data_root)
    assert json.loads((first.path / "manifest.json").read_text())["complete"] is False

    repeat = open_run("req", IDENTITY, data_root)

    assert (repeat.reused, repeat.complete) == (False, False)
    assert json.loads((repeat.path / "manifest.json").read_text())["complete"] is False
    assert repeat.path == first.path and sorted(os.listdir(repeat.path)) == [
        "manifest.json",
        "request.json",
    ]


def test_modes_are_owner_only_under_permissive_umask(
    data_root: Path, permissive_umask: None
) -> None:
    assert os.umask(0o000) == 0o000
    probe = data_root / "umask-probe"
    probe.mkdir()
    assert mode_of(probe) == 0o777, "umask is not permissive; the mode check would be vacuous"

    run = complete_run(open_run("req", IDENTITY, data_root))

    assert mode_of(run.path) == 0o700
    assert mode_of(data_root / "runs") == 0o700
    assert mode_of(data_root / "runs" / "by-request") == 0o700
    for file in run.path.iterdir():
        assert mode_of(file) == 0o600, file.name
    for file in (data_root / "runs" / "by-request").iterdir():
        assert mode_of(file) == 0o600, file.name


def test_existing_run_directory_is_not_overwritten_on_id_collision(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, permissive_umask: None
) -> None:
    first = open_run("one", IDENTITY, data_root)
    (first.path / "report.json").write_text("{}")
    snapshot = contents_and_mtimes(first.path)
    monkeypatch.setattr(runs, "new_run_id", lambda: first.run_id)

    with pytest.raises(FileExistsError):
        open_run("two", IDENTITY, data_root)

    assert contents_and_mtimes(first.path) == snapshot
    assert mode_of(first.path) == 0o700
    assert len(list((data_root / "runs" / "by-request").iterdir())) == 1
    monkeypatch.undo()
    second = open_run("two", IDENTITY, data_root)
    assert second.run_id != first.run_id and second.reused is False
    assert contents_and_mtimes(first.path) == snapshot


def test_created_is_true_only_for_the_call_that_claimed_the_request(data_root: Path) -> None:
    fresh = open_run("req", IDENTITY, data_root)
    assert (fresh.created, fresh.reused, fresh.complete) == (True, False, False)

    partial = open_run("req", IDENTITY, data_root)
    assert (partial.created, partial.reused, partial.complete) == (False, False, False)
    assert partial.path == fresh.path

    done = complete_run(partial)
    assert (done.created, done.reused, done.complete) == (False, False, True)
    assert complete_run(fresh).created is True

    repeat = open_run("req", IDENTITY, data_root)
    assert (repeat.created, repeat.reused, repeat.complete) == (False, True, True)


def test_index_entry_is_published_atomically_without_leftovers(
    data_root: Path, permissive_umask: None
) -> None:
    run = open_run("req", IDENTITY, data_root)
    index_dir = data_root / "runs" / "by-request"
    entries = list(index_dir.iterdir())
    assert [entry.name for entry in entries] == [hashlib.sha256(b"req").hexdigest()]
    assert entries[0].read_text() == run.run_id
    assert mode_of(entries[0]) == 0o600
    assert entries[0].stat().st_nlink == 1
    assert not entries[0].is_symlink()


@pytest.mark.parametrize("content", ["", "\n", "../request", "not-a-run-id"])
def test_index_entry_without_run_id_is_an_error_not_a_lookup_of_runs_dir(
    data_root: Path, content: str
) -> None:
    index_dir = data_root / "runs" / "by-request"
    index_dir.mkdir(parents=True)
    (index_dir / hashlib.sha256(b"req").hexdigest()).write_text(content)
    # If the empty run_id resolved to `runs/` itself, this file would pass as the request.
    (data_root / "runs" / "request.json").write_text(
        json.dumps({"identity_hash": identity_hash(IDENTITY)})
    )
    (data_root / "runs" / "manifest.json").write_text(json.dumps({"complete": True}))

    with pytest.raises(RequestIndexCorrupt, match=hashlib.sha256(b"req").hexdigest()):
        open_run("req", IDENTITY, data_root)

    assert sorted(p.name for p in (data_root / "runs").iterdir()) == [
        "by-request",
        "manifest.json",
        "request.json",
    ]
    assert open_run("other", IDENTITY, data_root).created is True


def test_index_entry_pointing_at_missing_run_dir_is_an_explicit_error(data_root: Path) -> None:
    request_id = "REQ-zygote-7Q"
    run = complete_run(open_run(request_id, IDENTITY, data_root))
    other = open_run("other", IDENTITY, data_root)
    shutil.rmtree(run.path)
    runs_dir = data_root / "runs"
    index_dir = runs_dir / "by-request"
    entry_name = hashlib.sha256(request_id.encode()).hexdigest()
    runs_before = sorted(p.name for p in runs_dir.iterdir())
    index_before = contents_and_mtimes(index_dir)
    other_before = contents_and_mtimes(other.path)

    with pytest.raises(RequestRunMissing) as info:
        open_run(request_id, IDENTITY, data_root)

    assert isinstance(info.value, RequestIndexCorrupt)
    assert not isinstance(info.value, FileNotFoundError)
    assert run.run_id in str(info.value) and entry_name in str(info.value)
    assert request_id not in str(info.value) and "zygote" not in str(info.value)
    assert sorted(p.name for p in runs_dir.iterdir()) == runs_before
    assert contents_and_mtimes(index_dir) == index_before
    assert contents_and_mtimes(other.path) == other_before
    assert not run.path.exists()

    with pytest.raises(RequestRunMissing):
        open_run(request_id, IDENTITY, data_root)
    assert open_run("other", IDENTITY, data_root).path == other.path
    fresh = open_run("fresh", IDENTITY, data_root)
    assert fresh.created is True and fresh.run_id != run.run_id
    assert sorted(p.name for p in index_dir.iterdir()) == sorted(
        [entry_name, hashlib.sha256(b"other").hexdigest(), hashlib.sha256(b"fresh").hexdigest()]
    )


def test_symlinked_runs_or_index_directory_is_rejected(data_root: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (data_root / "runs").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        open_run("req", IDENTITY, data_root)
    assert list(elsewhere.iterdir()) == []

    (data_root / "runs").unlink()
    (data_root / "runs").mkdir(mode=0o700)
    (data_root / "runs" / "by-request").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        open_run("req", IDENTITY, data_root)
    assert list(elsewhere.iterdir()) == []
    assert [p.name for p in (data_root / "runs").iterdir()] == ["by-request"]


def test_symlinked_run_directory_is_rejected_on_repeat(data_root: Path, tmp_path: Path) -> None:
    run = complete_run(open_run("req", IDENTITY, data_root))
    moved = tmp_path / "moved"
    run.path.rename(moved)
    run.path.symlink_to(moved, target_is_directory=True)
    snapshot = contents_and_mtimes(moved)

    with pytest.raises(FileExistsError, match="symlink"):
        open_run("req", IDENTITY, data_root)

    assert contents_and_mtimes(moved) == snapshot
    run.path.unlink()
    moved.rename(run.path)
    assert open_run("req", IDENTITY, data_root).reused is True


def test_leftover_staging_file_does_not_block_or_bind_the_request(data_root: Path) -> None:
    index_dir = data_root / "runs" / "by-request"
    index_dir.mkdir(parents=True, mode=0o700)
    leftover = index_dir / f".{hashlib.sha256(b'req').hexdigest()}.deadbeef"
    leftover.write_text("0" * 32)

    run = open_run("req", IDENTITY, data_root)

    assert run.created is True and run.run_id != "0" * 32
    assert sorted(p.name for p in index_dir.iterdir()) == sorted(
        [leftover.name, hashlib.sha256(b"req").hexdigest()]
    )


def test_module_docstring_tells_operator_to_remove_entry_with_run_dir_and_not_expect_repair() -> (
    None
):
    doc = runs.__doc__ or ""
    assert "RequestRunMissing" in doc
    assert re.search(r"operator removes the index entry together with the run directory", doc)
    assert re.search(r"`open_run` does\s+not repair such an entry", doc)
