"""Open STT corrupted_asr candidates (#81): the archive read safely, a dev-only pool, no verdict.

The archive is built here from numpy tones and short texts, and the ASR is a stub that returns
fixed texts: these tests prove which files are read and transcribed and what enters the manifest,
not what the model hears.
"""

import dataclasses
import hashlib
import io
import json
import shutil
import tarfile
import wave
from pathlib import Path

import numpy as np
import pytest

from sound_scout.config import Config
from sound_scout.eval import corpus, labels, manifest, openstt

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

RATE = 16_000
REFERENCE = "сегодня мы обсудим новый проект компании"
SAME = "Сегодня, мы обсудим новый проект компании!"
WRONG = "сегодня вы обсудили новые проекты кампании"


def wav(seconds: float, frequency: float = 220.0) -> bytes:
    t = np.arange(round(seconds * RATE)) / RATE
    samples = np.round(0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(samples.tobytes())
    return buffer.getvalue()


def stem(i: int) -> str:
    return f"{i:012x}"


def member(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def build_archive(path: Path, phrases: dict[str, tuple[float, str]], extra=()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for i, (name, (seconds, text)) in enumerate(phrases.items()):
            folder = f"asr_calls_2_val/{i % 3:x}a/{i % 5:x}b"
            for info, data in (
                member(f"{folder}/{name}.wav", wav(seconds, 200 + i)),
                member(f"{folder}/{name}.txt", text.encode()),
            ):
                archive.addfile(info, io.BytesIO(data))
        for info, data in extra:
            archive.addfile(info, io.BytesIO(data) if data is not None else None)


@dataclasses.dataclass
class Setup:
    corpus_dir: Path
    manifest_path: Path
    archive: Path
    calls: list

    def transcriber(self, texts: dict[str, str]):
        def transcribe(path: Path, sha256: str, name: str):
            self.calls.append(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
            model = {"repo": "o/m", "revision": "r"}
            info = {"text": texts.get(name, REFERENCE), "run": f"r/{name}", "model": model}
            return None, {**info, "raw_sha256": "0" * 64}

        return transcribe


def pin(monkeypatch, archive: Path) -> None:
    data = archive.read_bytes()
    source = corpus.Source(
        "open-stt", corpus.OPEN_STT_ARCHIVE, "https://example.org/a.tar.gz", len(data),
        "md5:" + hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest(),
        corpus.OPEN_STT_VERSION, corpus.OPEN_STT_LICENSE,
    )  # fmt: skip
    monkeypatch.setattr(corpus, "EXTRA_SOURCES", (source,))


def make(tmp_path: Path, monkeypatch, phrases: dict, extra=()) -> Setup:
    corpus_dir = tmp_path / "corpus"
    archive = corpus_dir / "sources" / "open-stt" / corpus.OPEN_STT_ARCHIVE
    build_archive(archive, phrases, extra)
    pin(monkeypatch, archive)
    root = corpus_dir / "pilot-x"
    (root / "clips").mkdir(parents=True)
    (root / "labels").mkdir()
    held = wav(2.0, 500)
    (root / "clips" / "held.wav").write_bytes(held)
    record = {
        "id": "held", "path": "clips/held.wav", "sha256": hashlib.sha256(held).hexdigest(),
        "duration_s": 2.0, "split": "heldout", "source_group": "held",
        "categories": ["pronunciation"], "negative_control": False,
    }  # fmt: skip
    path = root / "manifest.json"
    document = {"schema_version": "1.0", "name": "pilot-x", "note": corpus.MANIFEST_NOTE,
                "records": [record]}  # fmt: skip
    path.write_text(json.dumps(document), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps({"clips": []}), encoding="utf-8")
    return Setup(corpus_dir, path, archive, [])


def phrases(count: int, seconds: float = 3.0) -> dict[str, tuple[float, str]]:
    return {stem(i): (seconds, REFERENCE) for i in range(count)}


# --- text rules ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        (REFERENCE, SAME, 0.0),
        ("Ёлка, стоит!", "елка стоит", 0.0),
        ("раз два три четыре", "раз три четыре", 0.25),
        ("раз два три четыре", "раз два три четыре пять", 0.25),
        ("раз два три четыре", "один два три пять", 0.5),
        ("раз два", "", 1.0),
    ],
)
def test_normalized_word_error_rate_on_known_vectors(reference, hypothesis, expected) -> None:
    assert openstt.wer(reference, hypothesis) == pytest.approx(expected)


def test_normalize_lowers_maps_yo_and_drops_punctuation() -> None:
    assert openstt.normalize("  Ещё, РАЗ —  и всё! ") == "еще раз и все"


# --- the archive --------------------------------------------------------------------------------


def test_only_well_named_regular_members_within_the_limits_are_read(tmp_path, monkeypatch) -> None:
    link = tarfile.TarInfo(f"asr_calls_2_val/0a/0b/{stem(90)}.wav")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    big = member(f"asr_calls_2_val/0a/0b/{stem(91)}.wav", b"\0" * (openstt.MAX_WAV_BYTES + 1))
    escape = member(f"asr_calls_2_val/../{stem(92)}.wav", wav(3.0))
    other = member("README.txt", b"hello")
    setup = make(tmp_path, monkeypatch, phrases(2), [(link, None), big, escape, other])
    found, stats = openstt.scan(setup.archive)
    assert set(found) == {stem(0), stem(1)}
    assert stats["skipped_name_or_type"] == 3
    assert stats["skipped_size"] == 1
    assert all(found[s].duration_s == pytest.approx(3.0) for s in found)


def test_the_pool_is_deterministic_and_keeps_only_eligible_phrases(tmp_path, monkeypatch) -> None:
    items = phrases(6)
    items[stem(10)] = (1.0, REFERENCE)
    items[stem(11)] = (20.0, REFERENCE)
    items[stem(12)] = (3.0, "три слова тут")
    items[stem(13)] = (3.0, "звонок номер 5 от клиента")
    items[stem(14)] = (3.0, "call me back please now")
    setup = make(tmp_path, monkeypatch, items)
    found, _ = openstt.scan(setup.archive)
    first = [p.stem for p in openstt.pool(found)]
    assert first == [p.stem for p in openstt.pool(openstt.scan(setup.archive)[0])]
    assert set(first) == {stem(i) for i in range(6)}
    assert first == sorted(first, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    monkeypatch.setattr(openstt, "POOL_MAX", 4)
    assert [p.stem for p in openstt.pool(found)] == first[:4]


# --- add ----------------------------------------------------------------------------------------


def test_candidates_are_dev_corrupted_asr_and_never_confirmed(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(6))
    wrong = {stem(i): WRONG for i in (1, 3, 4)}
    added, summary = openstt.add(setup.corpus_dir, setup.manifest_path, setup.transcriber(wrong))
    assert summary["candidates"] == 3 and summary["pool"] == 6 and summary["transcribed"] == 6
    records = json.loads(setup.manifest_path.read_text(encoding="utf-8"))["records"]
    new = [r for r in records if r["id"].startswith("ostt-")]
    assert [r for r, _ in added] == new
    assert {r["id"] for r in new} == {f"ostt-{stem(i)}" for i in (1, 3, 4)}
    for r in new:
        assert (r["split"], r["control_kind"], r["asr_error_status"]) == (
            "dev",
            "corrupted_asr",
            "candidate",
        )
        assert r["negative_control"] is True and r["categories"] == []
        assert "не подтверждена" in r["note"] and "CC BY-NC 4.0" in r["note"]
        template = setup.manifest_path.parent / "labels" / f"{r['id']}.json"
        assert not labels.validate(template).ok
    assert "confirmed" not in setup.manifest_path.read_text(encoding="utf-8")
    report = manifest.validate(setup.manifest_path)
    assert report.structure_ok, report.errors
    assert all(wer >= openstt.WER_MIN for _, wer in added)


def test_the_asr_stops_at_five_candidates(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(10))
    wrong = {stem(i): WRONG for i in range(10)}
    added, summary = openstt.add(setup.corpus_dir, setup.manifest_path, setup.transcriber(wrong))
    assert len(added) == 5 and summary["transcribed"] == 5 and len(setup.calls) == 5


def test_the_asr_sees_only_pool_clips_never_a_heldout_record(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(5))
    openstt.add(
        setup.corpus_dir,
        setup.manifest_path,
        setup.transcriber({stem(i): WRONG for i in range(5)}),
    )
    pool_dir = setup.manifest_path.parent / openstt.POOL_DIR
    assert setup.calls and all(path.parent == pool_dir for path in setup.calls)
    assert {path.stem for path in setup.calls} <= {stem(i) for i in range(5)}


def test_fewer_than_three_leave_the_manifest_byte_for_byte(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(6))
    before = setup.manifest_path.read_bytes()
    labels_before = sorted((setup.manifest_path.parent / "labels").iterdir())
    with pytest.raises(openstt.Shortfall, match="2 of 6"):
        openstt.add(
            setup.corpus_dir,
            setup.manifest_path,
            setup.transcriber({stem(0): WRONG, stem(2): WRONG}),
        )
    assert setup.manifest_path.read_bytes() == before
    assert sorted((setup.manifest_path.parent / "labels").iterdir()) == labels_before
    selection = setup.manifest_path.parent / openstt.SELECTION_FILE
    assert json.loads(selection.read_text(encoding="utf-8"))["candidates"] == 2


def test_every_failed_asr_call_is_an_error_not_a_shortfall(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(4))
    before = setup.manifest_path.read_bytes()

    def failing(path: Path, sha256: str, name: str):
        setup.calls.append(path)
        return "PathDenied: the input is outside the allowed roots", {}

    with pytest.raises(openstt.OpenSttError, match="no ASR call completed.*PathDenied") as error:
        openstt.add(setup.corpus_dir, setup.manifest_path, failing)
    assert not isinstance(error.value, openstt.Shortfall)
    assert setup.manifest_path.read_bytes() == before
    assert len(setup.calls) == 4
    selection = json.loads(
        (setup.manifest_path.parent / openstt.SELECTION_FILE).read_text(encoding="utf-8")
    )
    assert selection["transcribed"] == 0 and selection["asr_failed"] == 4


def test_failed_asr_calls_are_counted_apart_from_the_transcribed(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(4))
    good = setup.transcriber({stem(0): WRONG})

    def flaky(path: Path, sha256: str, name: str):
        if name in (stem(1), stem(2)):
            return "asr is failed: decoder", {}
        return good(path, sha256, name)

    with pytest.raises(openstt.Shortfall, match="1 of 2 transcribed.*2 ASR calls failed"):
        openstt.add(setup.corpus_dir, setup.manifest_path, flaky)


def test_main_exits_2_when_no_asr_call_completes(tmp_path, monkeypatch, capsys) -> None:
    setup = make(tmp_path, monkeypatch, phrases(4))
    monkeypatch.setattr(openstt.config, "load", lambda: Config((tmp_path,), tmp_path))
    monkeypatch.setattr(openstt.guard, "install_network_guard", lambda config_: None)
    monkeypatch.setattr(
        openstt, "local_transcriber", lambda c, n: lambda path, sha, name: ("asr is failed", {})
    )
    argv = ["add", "--corpus-dir", str(setup.corpus_dir), "--manifest", str(setup.manifest_path)]
    assert openstt.main(argv) == 2
    assert "no ASR call completed" in capsys.readouterr().err


@pytest.mark.parametrize("taken", ["clips/ostt-0123.wav", "labels/ostt-0123.json",
                                   "asr-review/ostt-0123.json"])  # fmt: skip
def test_existing_ostt_files_are_refused_before_anything_is_written(
    tmp_path, monkeypatch, taken
) -> None:
    setup = make(tmp_path, monkeypatch, phrases(4))
    root = setup.manifest_path.parent
    (root / taken).parent.mkdir(exist_ok=True)
    (root / taken).write_bytes(b"x")
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    wrong = {stem(i): WRONG for i in range(4)}
    with pytest.raises(openstt.OpenSttError, match=f"already there, nothing written: {taken}"):
        openstt.add(setup.corpus_dir, setup.manifest_path, setup.transcriber(wrong))
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before
    assert setup.calls == []


def test_a_tampered_archive_is_refused_before_anything_is_written(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(4))
    with open(setup.archive, "ab") as out:
        out.write(b"x")
    before = setup.manifest_path.read_bytes()
    with pytest.raises(openstt.OpenSttError, match="does not match its pin"):
        openstt.add(setup.corpus_dir, setup.manifest_path, setup.transcriber({}))
    assert setup.manifest_path.read_bytes() == before
    assert not (setup.manifest_path.parent / openstt.POOL_DIR).exists()
    assert setup.calls == []


def test_a_second_add_and_a_fixed_manifest_are_refused(tmp_path, monkeypatch) -> None:
    setup = make(tmp_path, monkeypatch, phrases(4))
    wrong = {stem(i): WRONG for i in range(4)}
    openstt.add(setup.corpus_dir, setup.manifest_path, setup.transcriber(wrong))
    with pytest.raises(openstt.OpenSttError, match="already holds ostt-"):
        openstt.add(setup.corpus_dir, setup.manifest_path, setup.transcriber(wrong))
    document = json.loads(setup.manifest_path.read_text(encoding="utf-8"))
    document["note"] = "зафиксирован"
    setup.manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(openstt.OpenSttError, match="no draft"):
        openstt.add(setup.corpus_dir, setup.manifest_path, setup.transcriber(wrong))


def test_main_prints_ids_wer_and_status_but_no_text(tmp_path, monkeypatch, capsys) -> None:
    setup = make(tmp_path, monkeypatch, phrases(4))
    wrong = {stem(i): WRONG for i in range(4)}
    monkeypatch.setattr(openstt.config, "load", lambda: Config((tmp_path,), tmp_path))
    monkeypatch.setattr(openstt.guard, "install_network_guard", lambda config_: None)
    monkeypatch.setattr(openstt, "local_transcriber", lambda c, n: setup.transcriber(wrong))
    reviewed = []
    monkeypatch.setattr(
        openstt.asr_review,
        "review",
        lambda path, config_: reviewed.append(path) or {"reviewed": [], "skipped_heldout": []},
    )
    argv = ["add", "--corpus-dir", str(setup.corpus_dir), "--manifest", str(setup.manifest_path)]
    assert openstt.main(argv) == 0
    out = capsys.readouterr().out
    for word in (*REFERENCE.split(), *WRONG.split()):
        assert word not in out
    lines = [json.loads(line) for line in out.splitlines()]
    assert sum(1 for line in lines if line.get("asr_error_status") == "candidate") == 4
    assert reviewed == [setup.manifest_path]
    assert openstt.main(argv) == 2
