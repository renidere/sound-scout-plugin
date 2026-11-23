"""Pilot corpus `prepare` (#78) on synthetic RINCO / DEMAND / MS-SNSD stand-ins; no network.

The stand-ins only have the shape of the real datasets (zip layout, ELAN tiers, WAV files):
these tests prove the contract of `prepare`, not anything about the real recordings.
"""

import dataclasses
import hashlib
import io
import json
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sound_scout import cli
from sound_scout.eval import corpus, gate0, labels, manifest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

RATE = 16_000
TRANSCRIBE_MARK = "SECRET-TRANSCRIBE-MARK"
PHRASES = ((0.5, 3.0), (3.6, 6.0), (6.6, 9.0))
FOLDERS = ("1-Questions", "2-Imperatives", "3-Statements")


def wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(np.round(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def speech_like(seconds: float, seed: int) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    voice = 0.3 * np.sin(2 * np.pi * (150 + 10 * seed) * t)
    gate = np.zeros_like(t)
    for start, end in PHRASES:
        gate[(t >= start) & (t < end)] = 1.0
    return voice * gate


def noise(seconds: float, seed: int, level: float = 0.05) -> np.ndarray:
    return level * np.random.default_rng(seed).standard_normal(round(seconds * RATE))


def eaf(media: str, tiers: dict[str, list[tuple[float, float, str]]]) -> bytes:
    slots, body, n = [], [], 0
    for tier_id, annotations in tiers.items():
        items = []
        for start, end, value in annotations:
            n += 1
            slots.append(f'<TIME_SLOT TIME_SLOT_ID="ts{n}a" TIME_VALUE="{round(start * 1000)}"/>')
            slots.append(f'<TIME_SLOT TIME_SLOT_ID="ts{n}b" TIME_VALUE="{round(end * 1000)}"/>')
            items.append(
                f'<ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a{n}" TIME_SLOT_REF1="ts{n}a" '
                f'TIME_SLOT_REF2="ts{n}b"><ANNOTATION_VALUE>{value}</ANNOTATION_VALUE>'
                "</ALIGNABLE_ANNOTATION></ANNOTATION>"
            )
        body.append(f'<TIER TIER_ID="{tier_id}" LINGUISTIC_TYPE_REF="t">{"".join(items)}</TIER>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?><ANNOTATION_DOCUMENT><HEADER>'
        f'<MEDIA_DESCRIPTOR MEDIA_URL="file:///x/{media}" RELATIVE_MEDIA_URL="./{media}"/>'
        f"</HEADER><TIME_ORDER>{''.join(slots)}</TIME_ORDER>{''.join(body)}</ANNOTATION_DOCUMENT>"
    ).encode()


def rinco_tiers(int_phrase_id: str = "Int phrase") -> dict:
    return {
        "ID": [(0.5, 9.0, "q1f")],
        "Transcribe": [(s, e, TRANSCRIBE_MARK) for s, e in PHRASES],
        int_phrase_id: [(s, e, "фраза") for s, e in PHRASES],
        "Accent": [(s, e, "Ты *игра/ешь?") for s, e in PHRASES],
        "Pause": [(3.0, 3.6, "0,6"), (6.0, 6.6, "техническая пауза")],
    }


def rinco_zip(path: Path, files: int = 13, tiers=rinco_tiers) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(files):
            folder = FOLDERS[i % len(FOLDERS)]
            stem = f"x-{i:02d}"
            archive.writestr(f"INTON-EAF/{folder}/{stem}.wav", wav(speech_like(10.0, i)))
            tier_set = tiers("INT-PHRASE" if i == 0 else "Int phrase")
            archive.writestr(f"INTON-EAF/{folder}/{stem}.eaf", eaf(f"{stem}.wav", tier_set))
    return path


def build_sources(corpus_dir: Path) -> tuple[corpus.Source, ...]:
    """Stand-in files for every row of the real table, pinned like the real ones."""
    table = []
    for i, real in enumerate(corpus.SOURCES):
        path = corpus_dir / real.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if real.name == corpus.RINCO_ARCHIVE:
            rinco_zip(path)
        elif real.name.endswith(".pdf"):
            path.write_bytes(b"%PDF stand-in " + real.name.encode())
        elif real.dataset == "demand":
            room = real.name.split("_")[0]
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(f"{room}/{corpus.DEMAND_CHANNEL}", wav(noise(76.0, i)))
                archive.writestr(f"{room}/ch02.wav", wav(noise(1.0, i)))
        else:
            seconds = 20.0 if real.name == corpus.MIX_NOISE else 5.0
            path.write_bytes(wav(noise(seconds, i, 0.1)))
        data = path.read_bytes()
        table.append(
            dataclasses.replace(
                real,
                url="https://example.invalid/" + real.name,
                size=len(data),
                checksum="md5:" + hashlib.md5(data).hexdigest(),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    return tuple(table)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, dict, tuple[corpus.Source, ...]]:
    corpus_dir = tmp_path_factory.mktemp("corpus")
    sources = build_sources(corpus_dir)
    result = corpus.prepare(corpus_dir, "pilot-t", sources)
    return corpus_dir, result, sources


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, RATE)
        frames = source.readframes(source.getnframes())
    return np.frombuffer(frames, "<i2") / 32767


@pytest.mark.slow
def test_prepare_writes_22_clips_and_lists_19_without_the_speech_backgrounds(built) -> None:
    corpus_dir, result, _ = built
    out = corpus_dir / "pilot-t"
    records = result["records"]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == result
    assert result["note"] == corpus.MANIFEST_NOTE
    assert len(list((out / "clips").glob("*.wav"))) == 22
    assert len(records) == 19
    assert sum(r["negative_control"] for r in records) == 7
    for record in records:
        data = (out / record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert record["duration_s"] == pytest.approx(len(read_wav(out / record["path"])) / RATE)
        assert "pair_id" not in record
    kinds = {r["id"]: r.get("control_kind") for r in records if r["negative_control"]}
    assert kinds["ctl-silence"] == "silence" and kinds["ctl-pink"] == "noise"
    assert kinds["ctl-asrnoise-0db"] == kinds["ctl-asrnoise-m5db"] == "corrupted_asr"
    assert [k for k in kinds.values() if k == "noise"] == ["noise"] * 4
    statuses = {r["id"]: r.get("asr_error_status") for r in records}
    assert statuses["ctl-asrnoise-0db"] == statuses["ctl-asrnoise-m5db"] == "candidate"
    assert [i for i, s in statuses.items() if s is not None] == [i for i, _ in corpus.MIX_SNRS_DB]
    assert not list(corpus_dir.glob(".pilot-t.*")), "the staging directory is gone"


def test_speech_backgrounds_keep_their_audio_but_are_no_noise_controls(built) -> None:
    corpus_dir, result, _ = built
    out = corpus_dir / "pilot-t"
    assert corpus.EXCLUDED_CONTROLS == (
        "ctl-snsd-babble",
        "ctl-snsd-neighbor-2",
        "ctl-snsd-neighbor-4",
    )
    listed = {r["id"] for r in result["records"]}
    provenance = {
        c["id"]: c for c in json.loads((out / "provenance.json").read_text("utf-8"))["clips"]
    }
    for clip_id in corpus.EXCLUDED_CONTROLS:
        assert clip_id not in listed
        assert clip_id in result["note"]
        assert (out / "clips" / f"{clip_id}.wav").is_file()
        assert not (out / "labels" / f"{clip_id}.json").exists()
        assert provenance[clip_id]["excluded"] == corpus.SPEECH_BACKGROUND
    assert "разборчивая речь" in result["note"] and "не контроль noise" in result["note"]


BACKGROUND_RECORD = {
    "duration_s": 15.0,
    "split": "dev",
    "categories": [],
    "negative_control": True,
    "control_kind": "noise",
}


def earlier_manifest(result: dict, provenance: dict) -> dict:
    """The #78 draft: every clip listed, the #78 note, no `asr_error_status`."""
    earlier = {**result, "note": corpus.DRAFT_NOTE, "records": []}
    for clip in provenance["clips"]:
        record = next((r for r in result["records"] if r["id"] == clip["id"]), None)
        if record is None:
            record = {
                **BACKGROUND_RECORD,
                "id": clip["id"],
                "path": f"clips/{clip['id']}.wav",
                "sha256": clip["sha256"],
                "source_group": clip["id"],
            }
        earlier["records"].append({k: v for k, v in record.items() if k != "asr_error_status"})
    return earlier


def test_revise_brings_an_earlier_draft_to_the_same_manifest_and_is_idempotent(
    built, tmp_path: Path, capsys
) -> None:
    corpus_dir, result, _ = built
    provenance = json.loads((corpus_dir / "pilot-t" / "provenance.json").read_text("utf-8"))
    earlier = earlier_manifest(result, provenance)
    assert corpus.revise(earlier) == result
    assert corpus.revise(result) == result

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(earlier, ensure_ascii=False), encoding="utf-8")
    assert corpus.main(["revise", "--manifest", str(path)]) == 0
    assert capsys.readouterr().out == (
        "dropped: ctl-snsd-babble, ctl-snsd-neighbor-2, ctl-snsd-neighbor-4; "
        "marked excluded: none; templates deleted: none\n"
    )
    assert json.loads(path.read_text("utf-8")) == result
    assert corpus.main(["revise", "--manifest", str(path)]) == 0
    assert capsys.readouterr().out == NOTHING_CHANGED
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]
    assert corpus.main(["revise", "--manifest", str(tmp_path / "missing.json")]) == 2


NOTHING_CHANGED = "dropped: none; marked excluded: none; templates deleted: none\n"


def old_corpus(built, tmp_path: Path, start: str = "78-draft") -> tuple[Path, Path]:
    """A copy of the fresh corpus with labels/ and provenance.json in their #78 form.

    The manifest is the #78 draft, or for `start="84-revised"` the one #84 already revised.
    """
    corpus_dir, result, _ = built
    fresh = corpus_dir / "pilot-t"
    old = shutil.copytree(fresh, tmp_path / "pilot-t")
    provenance = json.loads((old / "provenance.json").read_text("utf-8"))
    start_manifest = earlier_manifest(result, provenance) if start == "78-draft" else result
    (old / "manifest.json").write_text(json.dumps(start_manifest, ensure_ascii=False), "utf-8")
    for clip in provenance["clips"]:
        clip.pop("excluded", None)
    (old / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False), "utf-8")
    for clip_id in corpus.EXCLUDED_CONTROLS:
        audio = old / "clips" / f"{clip_id}.wav"
        sha256 = hashlib.sha256(audio.read_bytes()).hexdigest()
        template = corpus.label_template(clip_id, sha256, len(read_wav(audio)) / RATE)
        (old / "labels" / f"{clip_id}.json").write_text(json.dumps(template), "utf-8")
    return fresh, old


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.mark.parametrize("start", ["78-draft", "84-revised"])
def test_revise_brings_a_78_corpus_to_the_labels_and_provenance_of_a_fresh_prepare(
    built, tmp_path: Path, capsys, start
) -> None:
    fresh, old = old_corpus(built, tmp_path, start)
    assert len(list((old / "labels").iterdir())) == 22
    manifest_path = str(old / "manifest.json")

    assert corpus.main(["revise", "--manifest", manifest_path]) == 0
    excluded = ", ".join(corpus.EXCLUDED_CONTROLS)
    dropped = excluded if start == "78-draft" else "none"
    assert capsys.readouterr().out == (
        f"dropped: {dropped}; marked excluded: {excluded}; templates deleted: {excluded}\n"
    )
    manifest = json.loads((old / "manifest.json").read_text("utf-8"))
    assert manifest == json.loads((fresh / "manifest.json").read_text("utf-8"))
    label_ids = sorted(p.stem for p in (old / "labels").iterdir())
    assert label_ids == sorted(r["id"] for r in manifest["records"])
    assert snapshot(old / "labels") == snapshot(fresh / "labels")
    assert json.loads((old / "provenance.json").read_text("utf-8")) == json.loads(
        (fresh / "provenance.json").read_text("utf-8")
    )
    assert snapshot(old / "clips") == snapshot(fresh / "clips")
    assert sorted(p.name for p in old.iterdir()) == sorted(p.name for p in fresh.iterdir())

    before = snapshot(old)
    assert corpus.main(["revise", "--manifest", manifest_path]) == 0
    assert capsys.readouterr().out == NOTHING_CHANGED
    assert snapshot(old) == before


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda t: {**t, "annotator": "listener-a"}, id="signed"),
        pytest.param(lambda t: {**t, "events": [{"category": "pause"}]}, id="events"),
        pytest.param(lambda t: {**t, "clip_sha256": "0" * 64}, id="other-clip"),
        pytest.param(lambda t: "not json {", id="not-json"),
    ],
)
def test_revise_refuses_a_label_file_a_person_filled_and_changes_nothing(
    built, tmp_path: Path, capsys, change
) -> None:
    _, old = old_corpus(built, tmp_path)
    label = old / "labels" / "ctl-snsd-neighbor-2.json"
    changed = change(json.loads(label.read_text("utf-8")))
    label.write_text(changed if isinstance(changed, str) else json.dumps(changed), "utf-8")
    before = snapshot(old)

    assert corpus.main(["revise", "--manifest", str(old / "manifest.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "labels/ctl-snsd-neighbor-2.json" in captured.err
    assert "ctl-snsd-babble" not in captured.err and "changed nothing" in captured.err
    assert snapshot(old) == before


def test_revise_refuses_a_template_whose_audio_is_gone(built, tmp_path: Path) -> None:
    _, old = old_corpus(built, tmp_path)
    (old / "clips" / "ctl-snsd-babble.wav").unlink()
    before = snapshot(old)
    with pytest.raises(corpus.CorpusError, match=r"labels/ctl-snsd-babble\.json"):
        corpus.revise_file(old / "manifest.json")
    assert snapshot(old) == before


def test_revise_keeps_a_note_someone_wrote_and_adds_the_exclusion_reason_once() -> None:
    assert corpus.revise({"records": []})["note"] == corpus.MANIFEST_NOTE
    assert corpus.revise({"note": None, "records": []})["note"] == corpus.MANIFEST_NOTE
    revised = corpus.revise({"note": "held-out фиксирует владелец", "records": []})
    assert revised["note"] == f"held-out фиксирует владелец; {corpus.EXCLUSION_NOTE}"
    assert corpus.revise(revised) == revised
    held = {"note": f"{corpus.EXCLUSION_NOTE}; заметка владельца", "records": []}
    assert corpus.revise(held) == held


def test_revise_keeps_a_status_a_person_set_and_never_confirms() -> None:
    record = {"id": "c", "negative_control": True, "control_kind": "corrupted_asr"}
    confirmed = {**record, "asr_error_status": "confirmed"}
    revised = corpus.revise({"records": [record, confirmed]})["records"]
    assert [r["asr_error_status"] for r in revised] == ["candidate", "confirmed"]


def test_candidates_are_bound_to_the_member_and_interval_with_rule_categories(built) -> None:
    corpus_dir, result, sources = built
    provenance = json.loads((corpus_dir / "pilot-t" / "provenance.json").read_text("utf-8"))
    clips = {c["id"]: c for c in provenance["clips"]}
    archive_sha = next(s.sha256 for s in sources if s.name == corpus.RINCO_ARCHIVE)
    candidates = [r for r in result["records"] if not r["negative_control"]]
    assert len(candidates) == 12
    for record in candidates:
        clip = clips[record["id"]]
        assert corpus.WINDOW_MIN_S <= record["duration_s"] <= corpus.WINDOW_MAX_S
        assert clip["member"].startswith("INTON-EAF/") and clip["member"].endswith(".wav")
        assert clip["source_sha256"] == archive_sha
        assert clip["start_s"] == pytest.approx(0.2) and clip["end_s"] == pytest.approx(9.3)
        assert record["duration_s"] == pytest.approx(clip["end_s"] - clip["start_s"], abs=1e-3)
        assert record["categories"] == ["semantic_stress", "pause_phrasing", "completeness"]
        assert record["source_group"] == "rinco-" + Path(clip["member"]).stem
        assert clip["member"] in record["note"] and "не утверждены разметкой" in record["note"]
        tiers = {a["tier"] for a in clip["annotations"]}
        assert tiers == {"Accent", "Pause", "Int phrase"}
        pause = next(a for a in clip["annotations"] if a["value"] == "0,6")
        assert pause["start"] == pytest.approx(2.8) and pause["end"] == pytest.approx(3.4)


def test_the_transcribe_tier_reaches_no_output_file(built) -> None:
    corpus_dir, _, _ = built
    for path in (corpus_dir / "pilot-t").rglob("*"):
        if path.is_file():
            assert TRANSCRIBE_MARK.encode() not in path.read_bytes(), path.name


def test_silence_pink_noise_and_background_controls(built) -> None:
    corpus_dir, _, _ = built
    clips = corpus_dir / "pilot-t" / "clips"
    assert not read_wav(clips / "ctl-silence.wav").any()
    pink = read_wav(clips / "ctl-pink.wav")
    expected = gate0.pink_noise(corpus.CONTROL_SECONDS, seed=corpus.PINK_SEED)
    assert np.max(np.abs(pink - expected)) < 1e-4
    assert corpus.PINK_SEED != gate0.NOISE_SEED
    assert np.max(np.abs(pink - gate0.pink_noise(corpus.CONTROL_SECONDS))) > 1e-3
    assert len(read_wav(clips / "ctl-demand-dkitchen.wav")) == 15 * RATE
    assert len(read_wav(clips / "ctl-snsd-babble.wav")) == 5 * RATE


def test_speech_babble_mixes_hit_their_snr(built) -> None:
    corpus_dir, result, _ = built
    out = corpus_dir / "pilot-t"
    provenance = json.loads((out / "provenance.json").read_text("utf-8"))
    clips = {c["id"]: c for c in provenance["clips"]}
    candidate_groups = {r["source_group"] for r in result["records"] if not r["negative_control"]}
    for clip_id, snr in corpus.MIX_SNRS_DB:
        clip = clips[clip_id]
        record = next(r for r in result["records"] if r["id"] == clip_id)
        assert record["split"] == "dev" and record["source_group"] not in candidate_groups
        assert corpus.ASR_UNCONFIRMED in record["note"]
        assert clip["noise"]["snr_db"] == snr and clip["noise"]["seed"] == corpus.MIX_SEED
        with zipfile.ZipFile(out.parent / f"sources/rinco/{corpus.RINCO_ARCHIVE}") as archive:
            source = archive.read(clip["member"])
        speech = np.frombuffer(source[44:], "<i2") / 32767
        start = round(clip["start_s"] * RATE)
        mixed = read_wav(out / record["path"])
        speech = speech[start : start + len(mixed)] * clip["noise"]["scale"]
        measured = 20 * np.log10(np.sqrt(np.mean(speech**2) / np.mean((mixed - speech) ** 2)))
        assert measured == pytest.approx(snr, abs=0.5)
        assert np.max(np.abs(mixed)) <= corpus.PEAK_LIMIT + 1e-4


def test_mix_scales_down_a_peak_over_the_limit() -> None:
    speech = np.full(RATE, 0.9)
    mixed, info = corpus.mix_at_snr(speech, noise(2.0, 1, 0.5), -5.0, 3)
    assert info["scale"] < 1.0 and np.max(np.abs(mixed)) == pytest.approx(corpus.PEAK_LIMIT)
    with pytest.raises(corpus.CorpusError, match="noise is"):
        corpus.mix_at_snr(speech, noise(0.5, 1), 0.0, 3)


def test_draft_manifest_names_the_clip_shortfall_and_the_unconfirmed_candidates(
    built, capsys
) -> None:
    corpus_dir, result, _ = built
    path = str(corpus_dir / "pilot-t" / "manifest.json")
    assert cli.main(["eval", "check-manifest", path, "--draft"]) == 1
    out = capsys.readouterr().out
    shortfall = f"$.records: 19 clips, the corpus needs at least {manifest.MIN_CLIPS}"
    assert out.splitlines()[-1] == f"{manifest.DRAFT_SHORT}: {shortfall}"
    assert manifest.DRAFT_UNSOUND not in out
    assert cli.main(["eval", "check-manifest", path]) == 1
    report = manifest.validate(path)
    assert [e for e in report.errors if e not in report.label_dependent] == [shortfall]
    assert report.count_shortfalls == [shortfall]
    waiting = [e for e in report.label_dependent if "corrupted_asr candidate" in e]
    assert len(waiting) == 2 and all(e in out for e in waiting)
    assert not any("negative controls" in e for e in report.errors), "5 confirmed controls remain"
    splits: dict[str, set[str]] = {}
    for record in result["records"]:
        splits.setdefault(record["source_group"], set()).add(record["split"])
    assert all(len(seen) == 1 for seen in splits.values())
    heldout_candidates = [
        r for r in result["records"] if r["split"] == "heldout" and not r["negative_control"]
    ]
    assert len(heldout_candidates) == corpus.HELDOUT_CANDIDATE_GROUPS
    plain = ["ctl-silence", "ctl-pink", *(c for c, _, _ in corpus.DEMAND_CONTROLS)]
    plain += list(corpus.EXCLUDED_CONTROLS)
    split = corpus.split_by_group(plain, len(plain) // 2)
    listed = [r for r in result["records"] if r.get("control_kind") in ("silence", "noise")]
    assert {r["id"]: r["split"] for r in listed} == {r["id"]: split[r["id"]] for r in listed}


def test_label_templates_are_empty_and_invalid_until_a_person_signs_them(built) -> None:
    corpus_dir, result, _ = built
    out = corpus_dir / "pilot-t"
    for record in result["records"]:
        path = out / "labels" / f"{record['id']}.json"
        template = json.loads(path.read_text("utf-8"))
        assert template == corpus.label_template(
            record["id"], record["sha256"], record["duration_s"]
        )
        assert template["model_outputs_seen"] is False and template["events"] == []
        report = labels.validate(path)
        assert sorted(e.split(":")[0] for e in report.errors) == ["$.annotated_at", "$.annotator"]
    filled = dict(template, annotator="listener-a", annotated_at="2026-11-19")
    assert labels.validation_report(filled).ok


def test_a_second_prepare_is_refused(built) -> None:
    corpus_dir, _, sources = built
    with pytest.raises(corpus.CorpusError, match="already exists"):
        corpus.prepare(corpus_dir, "pilot-t", sources)


def test_a_source_off_its_pin_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    sources = build_sources(tmp_path)
    babble = tmp_path / "sources" / "ms-snsd" / corpus.MIX_NOISE
    babble.write_bytes(babble.read_bytes()[:-2] + b"\0\1")
    with pytest.raises(corpus.CorpusError, match="do not match their pins"):
        corpus.prepare(tmp_path, "pilot-x", sources)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sources"]


def test_an_archive_member_over_the_limit_is_refused(tmp_path, monkeypatch) -> None:
    sources = build_sources(tmp_path)
    monkeypatch.setattr(corpus, "MAX_MEMBER_BYTES", 1000)
    with pytest.raises(corpus.CorpusError, match="over the limit"):
        corpus.prepare(tmp_path, "pilot-x", sources)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sources"]


def test_eaf_without_the_tiers_names_the_tiers_it_found(tmp_path: Path) -> None:
    def without_accent(int_phrase_id: str) -> dict:
        tiers = rinco_tiers(int_phrase_id)
        del tiers["Accent"]
        return tiers

    path = rinco_zip(tmp_path / "r.zip", files=2, tiers=without_accent)
    with zipfile.ZipFile(path) as archive, pytest.raises(corpus.CorpusError) as error:
        corpus.rinco_files(archive)
    message = str(error.value)
    assert "['Accent']" in message and "Transcribe" in message and "Pause" in message
    assert TRANSCRIBE_MARK not in message and "0,6" not in message


def test_parse_eaf_inherits_ref_times_and_counts_unaligned() -> None:
    data = """<ANNOTATION_DOCUMENT><TIME_ORDER>
      <TIME_SLOT TIME_SLOT_ID="t1" TIME_VALUE="1000"/>
      <TIME_SLOT TIME_SLOT_ID="t2" TIME_VALUE="2500"/>
      <TIME_SLOT TIME_SLOT_ID="t3"/></TIME_ORDER>
      <TIER TIER_ID="ID"><ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="t1"
        TIME_SLOT_REF2="t2"><ANNOTATION_VALUE>q1</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION>
        </ANNOTATION></TIER>
      <TIER TIER_ID="accent-f"><ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a2" ANNOTATION_REF="a1">
        <ANNOTATION_VALUE>*да/</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION></TIER>
      <TIER TIER_ID="Pause"><ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a3" TIME_SLOT_REF1="t2"
        TIME_SLOT_REF2="t3"><ANNOTATION_VALUE>0,4</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION>
        </ANNOTATION></TIER>
      <TIER TIER_ID="Transcribe"><ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a4" ANNOTATION_REF="a1">
        <ANNOTATION_VALUE>secret</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION></TIER>
    </ANNOTATION_DOCUMENT>""".encode()
    found, unaligned, tier_ids = corpus.parse_eaf(data)
    assert found["accent"] == [corpus.Annotation("accent", 1.0, 2.5, "*да/")]
    assert found["pause"] == [] and found["int_phrase"] == [] and unaligned == 1
    assert tier_ids == {"ID", "accent-f", "Pause", "Transcribe"}
    assert corpus.tier_kind("Int-phrase") == "int_phrase" and corpus.tier_kind("Transcribe") is None
    assert corpus.tier_kind("Int-phrase-cp-1") is None


@pytest.mark.parametrize(
    ("value", "counts"),
    [("0,4", True), ("0.25 (э-э)", True), ("0,1", False), ("техническая пауза (1 с.)", False)],
)
def test_only_measured_pauses_of_at_least_0_2_s_count(value: str, counts: bool) -> None:
    assert corpus.pause_counts(corpus.Annotation("pause", 0.0, 1.0, value)) is counts


def test_windows_never_cut_an_annotation_and_need_context_around_a_pause() -> None:
    annotations = {
        "int_phrase": [corpus.Annotation("int_phrase", s, e, "ф") for s, e in PHRASES],
        "accent": [corpus.Annotation("accent", 0.5, 6.0, "*да/")],
        "pause": [corpus.Annotation("pause", 3.0, 3.6, "0,6")],
    }
    found = corpus.windows(annotations, 10.0)
    assert [(round(w.start, 1), round(w.end, 1)) for w in found] == [(0.2, 6.3), (0.2, 9.3)]
    assert all(w.categories[0] == "semantic_stress" for w in found)
    short = corpus.windows(annotations, 10.0, 1.0, 3.5)
    assert [(round(w.start, 1), round(w.end, 1), w.categories) for w in short] == [
        (6.3, 9.3, ("completeness",))
    ]


@pytest.mark.parametrize(
    ("pause", "counts"),
    [((0.25, 0.45), False), ((3.05, 3.25), False), ((1.5, 1.9), True)],
)
def test_a_pause_counts_only_with_0_3_s_of_context_inside_the_window(
    pause: tuple[float, float], counts: bool
) -> None:
    annotations = {
        "int_phrase": [corpus.Annotation("int_phrase", 0.5, 3.0, "ф")],
        "accent": [],
        "pause": [corpus.Annotation("pause", *pause, "0,2")],
    }
    [window] = corpus.windows(annotations, 10.0, 1.0, 10.0)
    assert (round(window.start, 1), round(window.end, 1)) == (0.2, 3.3)
    assert any(a.tier == "pause" for a in window.annotations)
    assert ("pause_phrasing" in window.categories) is counts
    assert "completeness" in window.categories
