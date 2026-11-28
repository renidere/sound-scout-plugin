"""prosody-confirm-1 (#199): rule check, frozen files, selection, freshness, import, verdict.

All audio, archives and sheets here are synthetic and the transcriber is a stand-in that names
voiced runs from a fixed script: these tests prove the contract of
`sound_scout.eval.prosody_confirm` (what is chosen, checked, recorded and refused), never that the
index finds prominent words. Only the real run recorded in docs/eval can say that.
"""

import csv
import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config
from sound_scout.eval import corpus, gate0, openstt, prosody
from sound_scout.eval import prosody_confirm as pc
from tests.test_eval_corpus import eaf
from tests.test_eval_prosody import words_signal

REPO = Path(__file__).resolve().parent.parent
RATE = pc.SAMPLE_RATE
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

RINCO_ACCENT = "Мама /мыла раму вчера"
RINCO_WORDS = ["мама", "мыла", "раму", "вчера"]
REFERENCE = "сегодня мы обсудим новый проект компании"
REFERENCE_WORDS = REFERENCE.split()
SCRIPT = {4: RINCO_WORDS, 6: REFERENCE_WORDS}
RECEIPT = {
    "repo": prosody.ASR_REPO,
    "revision": prosody.ASR_REVISION,
    "device": "cpu",
    "compute_type": "int8",
    "files": [{"name": "model.bin", "sha256": "e" * 64}],
    "local_files_only": True,
    "versions": {"faster-whisper": "1.2.1", "ctranslate2": "4.9.0"},
}


# --- stand-ins -----------------------------------------------------------------------------


def voiced_runs(samples: np.ndarray, threshold: float = 0.05) -> list[tuple[float, float]]:
    frame = RATE // 100
    count = len(samples) // frame
    levels = [np.sqrt(np.mean(samples[i * frame : (i + 1) * frame] ** 2)) for i in range(count)]
    runs, start = [], None
    for i, level in enumerate([*levels, 0.0]):
        if level > threshold and start is None:
            start = i
        elif level <= threshold and start is not None:
            runs.append((start / 100, i / 100))
            start = None
    return runs


def script_transcriber(receipt: dict = RECEIPT, calls: list | None = None):
    """Names the voiced runs of a cut from `SCRIPT` by their count; other cuts get no words."""

    def transcribe(samples, run_dir: Path) -> dict:
        (run_dir / "asr").mkdir(parents=True)
        (run_dir / "asr" / "model-receipt.json").write_text(json.dumps(receipt))
        if calls is not None:
            calls.append(run_dir.name)
        runs = voiced_runs(np.asarray(samples))
        names = SCRIPT.get(len(runs), [])
        pairs = zip(names, runs, strict=True) if names else ()
        words = [{"word": f" {n}", "start": s, "end": e} for n, (s, e) in pairs]
        return prosody.transcribed("completed", None, words, trusted_segments=int(bool(words)))

    return transcribe


def make_config(tmp_path: Path) -> config.Config:
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return config.Config(allowed_roots=(tmp_path / "in",), data_root=(tmp_path / "data").resolve())


def wav16(samples: np.ndarray) -> bytes:
    return gate0.wav_bytes(gate0.pcm16(samples))


def stem(i: int) -> str:
    return f"{i:012x}"


def rinco_zip(path: Path, files: int = 4, utterances: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(files):
            name = f"INTON-EAF/1-Questions/x-{i}"
            signal = np.concatenate([words_signal(4, raised=2)[0] for _ in range(utterances)])
            archive.writestr(f"{name}.wav", wav16(signal))
            span = len(signal) / RATE / utterances
            tiers = {
                "Accent": [(k * span, (k + 1) * span, RINCO_ACCENT) for k in range(utterances)],
                "Int phrase": [(0.0, span, "фраза")],
                "Pause": [(0.0, 0.1, "0,3")],
            }
            archive.writestr(f"{name}.eaf", eaf(f"x-{i}.wav", tiers))
    return path


def open_stt_tar(path: Path, count: int = 10) -> dict[str, int]:
    """`count` six-word phrases (word `i % 6 + 1` raised), one of five words and one in Latin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raised = {}
    entries = [(stem(i), REFERENCE, i % 6 + 1) for i in range(count)]
    entries += [(stem(90), "сегодня мы обсудим новый проект", 1), (stem(91), "hello world", 1)]
    with tarfile.open(path, "w:gz") as archive:
        for i, (name, text, word) in enumerate(entries):
            folder = f"asr_calls_2_val/{i % 3:x}a/{i % 5:x}b"
            audio = wav16(words_signal(len(text.split()), raised=word)[0])
            for member, data in (
                (f"{folder}/{name}.wav", audio),
                (f"{folder}/{name}.txt", text.encode()),
            ):
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            raised[name] = word
    return raised


def phrase_member(archive_path: Path, name: str) -> str:
    with tarfile.open(archive_path, "r:gz") as archive:
        return next(m.name for m in archive if m.name.endswith(f"/{name}.wav"))


@pytest.fixture
def station(tmp_path, monkeypatch):
    """A data root with RINCO, Open STT, pilot-1 and series 3, at the sizes of a small test."""
    settings = make_config(tmp_path)
    root = settings.data_root
    monkeypatch.setattr(pc, "q1", lambda _: [])
    monkeypatch.setattr(prosody, "memory_scope", lambda: (str(prosody.MEMORY_MAX_BYTES), "0"))
    monkeypatch.setattr(prosody, "pin_cpus", lambda: [0])
    monkeypatch.setattr(pc, "FRESH_COUNT", 4)
    monkeypatch.setattr(pc, "CALIBRATION_COUNT", 2)
    monkeypatch.setattr(pc, "R0_UTTERANCES", 3)
    monkeypatch.setattr(openstt, "POOL_MAX", 2)
    monkeypatch.setitem(prosody.SERIES[3], "excluded", ())
    rinco = rinco_zip(prosody.rinco_path(settings))
    tar = pc.open_stt_path(settings)
    raised = open_stt_tar(tar)
    pilot = root / prosody.PILOT_PROVENANCE
    (pilot.parent / "clips").mkdir(parents=True)
    (pilot.parent / "clips" / "ctl.wav").write_bytes(wav16(np.zeros(RATE)))
    clips = [
        {"kind": "rinco_candidate", "member": "INTON-EAF/1-Questions/x-0.wav",
         "source": f"sources/rinco/{corpus.RINCO_ARCHIVE}", "sha256": "1" * 64},
        {"kind": "openstt_asr_candidate", "member": phrase_member(tar, stem(9)),
         "source": "sources/open-stt/x", "sha256": "2" * 64},
    ]  # fmt: skip
    pilot.write_text(json.dumps({"clips": clips}))
    series = pc.series_3_dir(settings)
    (series / "main" / "runs").mkdir(parents=True)
    with zipfile.ZipFile(rinco) as archive:
        excluded = prosody.series_exclusion(settings, 3, archive)["members"]
        files, counts = prosody.material(archive, excluded)
        prosody.run_material(
            archive, files, script_transcriber(), series / "main" / "runs",
            series / "main" / "utterances.jsonl", float("inf"),
        )  # fmt: skip
    (series / "prepare.json").write_text(json.dumps({"material": counts}))
    return settings, raised


def filled_sheet(settings, raised: dict, calibration_agrees: bool = True) -> bytes:
    """The template with the raised word of each clip marked; calibration as RINCO or inverted."""
    out = settings.data_root / pc.OUTPUT_DIR
    items = {i["id"]: i for i in json.loads((out / pc.ALIAS_FILE).read_text())["items"]}
    rows = list(csv.DictReader(io.StringIO((out / pc.PACKAGE_DIR / pc.SHEET_FILE).read_text())))
    for row in rows:
        item, number = items[row["id"]], int(row["word_no"])
        if item["kind"] == pc.FRESH:
            row["marked"] = str(int(number == raised[item["stem"]]))
        else:
            agrees = item["rinco_marked"][number - 1] == calibration_agrees
            row["marked"] = str(int(agrees))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, pc.SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


# --- the rule and the frozen files ---------------------------------------------------------


def test_rule_sha256_of_0034_is_the_one_recorded_before_the_run() -> None:
    assert pc.check_rule(REPO / pc.DECISION_DOC) == pc.RULE_SHA256
    text = (REPO / pc.DECISION_DOC).read_text(encoding="utf-8")
    assert f"SHA-256 блока: `{pc.RULE_SHA256}`" in text
    assert pc.OWNER_DECISION.rsplit("-", 1)[-1] in text


def test_every_step_refuses_to_start_when_the_rule_block_changed(tmp_path: Path) -> None:
    edited = tmp_path / "0034.md"
    text = (REPO / pc.DECISION_DOC).read_text(encoding="utf-8")
    edited.write_text(text.replace("AUC >= 0,75", "AUC >= 0,70", 1), encoding="utf-8")
    settings = make_config(tmp_path)
    for step in (pc.prepare, pc.run_check_fresh, pc.run_sanity, pc.run_main):
        with pytest.raises(pc.ConfirmError, match="SHA-256"):
            step(settings, edited)
    with pytest.raises(pc.ConfirmError, match="SHA-256"):
        pc.import_sheet(settings, tmp_path / "sheet.csv", edited)
    assert not (settings.data_root / pc.OUTPUT_DIR).exists()


def test_rule_block_must_be_one_prosody_confirm_rule_v1_1_block() -> None:
    text = (REPO / pc.DECISION_DOC).read_text(encoding="utf-8")
    for broken in (text.replace(pc.RULE_BEGIN, ""), text + pc.RULE_END):
        with pytest.raises(pc.ConfirmError):
            pc.rule_sha256(broken)
    with pytest.raises(pc.ConfirmError, match="PROSODY-CONFIRM-RULE v1.1"):
        pc.rule_sha256(text.replace("PROSODY-CONFIRM-RULE v1.1 (", "PROSODY-CONFIRM-RULE v1 ("))


def test_frozen_pins_are_those_of_series_3_and_0034() -> None:
    doc = (REPO / pc.DECISION_DOC).read_text(encoding="utf-8")
    for name, sha in pc.FROZEN_SHA256.items():
        assert f"| `{name}` | `{sha}` |" in doc
        assert pc.FROZEN_MODULES[name].__file__.endswith(name.removeprefix("src/"))
    series_3 = "d7471142730a749f57db6911ca698ee79ef6d14341f4dda5ff456fb2f9bea565"
    assert pc.FROZEN_SHA256["src/sound_scout/eval/prosody.py"] == series_3
    assert pc.PROSODY_RULE_SHA256 == prosody.RULE_SHA256
    assert pc.OPEN_STT_SHA256 in (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    assert pc.open_stt_source().sha256 == pc.OPEN_STT_SHA256


def test_a_foreign_sha256_of_a_frozen_file_is_named(monkeypatch) -> None:
    monkeypatch.setitem(pc.FROZEN_SHA256, "src/sound_scout/asr.py", "0" * 64)
    problems = pc.frozen_problems()
    assert len(problems) == 1 and "src/sound_scout/asr.py" in problems[0]


def test_prepare_with_a_foreign_frozen_file_gives_invalid_before_any_data(
    tmp_path, monkeypatch
) -> None:
    settings = make_config(tmp_path)
    monkeypatch.setattr(pc, "q1", lambda _: [])
    monkeypatch.setitem(pc.FROZEN_SHA256, "src/sound_scout/acoustics.py", "0" * 64)
    result = pc.prepare(settings, REPO / pc.DECISION_DOC)
    assert result["verdict"] == pc.INVALID and "acoustics.py" in result["reason"]
    out = settings.data_root / pc.OUTPUT_DIR
    assert sorted(p.name for p in out.iterdir()) == [pc.RESULT_FILE]
    with pytest.raises(pc.ConfirmError, match="verdict is out"):
        pc.run_sanity(settings, REPO / pc.DECISION_DOC)


def test_q1_adds_the_open_stt_archive_to_the_q1_of_prosody(tmp_path, monkeypatch) -> None:
    settings = make_config(tmp_path)
    monkeypatch.setattr(prosody, "q1", lambda _: ["network"])
    problems = pc.q1(settings)
    assert problems[0] == "network"
    assert any(corpus.OPEN_STT_ARCHIVE in p and "missing" in p for p in problems)


# --- selection -----------------------------------------------------------------------------


def phrase(i: int, text: str = REFERENCE, seconds: float = 3.0) -> openstt.Phrase:
    name = stem(i)
    return openstt.Phrase(name, f"asr_calls_2_val/0/0/{name}.wav", f"t/{name}.txt", text, seconds)


def test_select_fresh_orders_by_hash_and_leaves_out_pool_pilot_and_short_texts(monkeypatch) -> None:
    monkeypatch.setattr(openstt, "POOL_MAX", 3)
    monkeypatch.setattr(pc, "FRESH_COUNT", 4)
    phrases = {p.stem: p for p in [phrase(i) for i in range(12)]}
    phrases[stem(20)] = phrase(20, "раз два три четыре пять")  # eligible, five words
    phrases[stem(21)] = phrase(21, seconds=20.0)  # too long
    pool = {p.stem for p in openstt.pool(phrases)}
    pilot = phrases[stem(5)].wav_member if stem(5) not in pool else phrases[stem(6)].wav_member
    chosen, selection = pc.select_fresh(phrases, {pilot})
    names = [p.stem for p in chosen]
    assert len(names) == 4 and not set(names) & pool and stem(20) not in names
    assert pilot not in {p.wav_member for p in chosen} and stem(21) not in names
    rest = [
        s
        for s in phrases
        if s not in pool | {stem(20), stem(21)} and phrases[s].wav_member != pilot
    ]
    assert names == sorted(rest, key=lambda s: pc.order_key(pc.SELECT_PREFIX, s))[:4]
    counts = selection["counts"]
    assert counts["min_words"] == 12 and counts["outside_pool_81"] == 12 - len(pool - {stem(20)})
    assert counts["outside_pilot_1"] == len(rest)
    assert selection["pool"] == sorted(phrases[s].wav_member for s in pool)
    monkeypatch.setattr(pc, "FRESH_COUNT", len(rest) + 1)
    with pytest.raises(pc.ConfirmError, match=f"{len(rest)} fresh clips available"):
        pc.select_fresh(phrases, {pilot})


def rinco_file(i: int, values: list[str]) -> corpus.RincoFile:
    annotations = [corpus.Annotation("Accent", k, k + 1.0, v) for k, v in enumerate(values)]
    member = f"INTON-EAF/1-Questions/f-{i}.wav"
    return corpus.RincoFile(
        "1-Questions", member, member[:-4] + ".eaf", 10.0, {"accent": annotations}
    )


def test_calibration_takes_one_mixed_utterance_of_4_to_20_words_per_file(monkeypatch) -> None:
    monkeypatch.setattr(pc, "CALIBRATION_COUNT", 3)
    files = [
        rinco_file(i, ["мама /мыла раму вчера", "мама мыла", "/мама /мыла /раму /вчера",
                       " ".join(["/слово"] + ["слово"] * 20), "мама мыла раму /вчера утром"])
        for i in range(4)
    ]  # fmt: skip
    chosen = pc.calibration(files)
    assert len(chosen) == 3 and len({u.member for u in chosen}) == 3
    assert {u.number for u in chosen} <= {0, 4}
    candidates = [u for f in files for u in prosody.utterances(f) if u.number in (0, 4)]
    best = sorted(
        candidates, key=lambda u: pc.order_key(pc.AGREEMENT_PREFIX, f"{u.member}{u.number}")
    )
    assert chosen[0] == best[0]
    monkeypatch.setattr(pc, "CALIBRATION_COUNT", 5)
    with pytest.raises(pc.ConfirmError, match="4 calibration utterances"):
        pc.calibration(files)


def test_reference_words_keep_the_order_of_the_normalized_text() -> None:
    assert pc.reference_words("Ещё раз, и всё!") == ["еще", "раз", "и", "все"]
    assert pc.reference_words("  а  - б ") == ["а", "б"]


def test_a_hyphenated_reference_word_is_one_word_and_aligns_with_the_asr_word() -> None:
    words = pc.reference_words("Кто-то сказал: всё-таки нет")
    assert words == ["ктото", "сказал", "всетаки", "нет"]
    asr_forms = [prosody.normalize(w) for w in (" Кто-то", " сказал,", " всё-таки", " нет.")]
    assert prosody.align(words, asr_forms) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert [prosody.normalize(w) for w in words] == words


# --- freshness -----------------------------------------------------------------------------


def fresh_entry(name: str, member_sha="a" * 64, wav_sha="b" * 64, pcm_sha="c" * 64) -> dict:
    return {"stem": name, "member": f"asr_calls_2_val/0/0/{name}.wav", "member_sha256": member_sha,
            "wav_sha256": wav_sha, "pcm_sha256": pcm_sha}  # fmt: skip


def freshness_root(tmp_path) -> tuple[config.Config, bytes, bytes]:
    settings = make_config(tmp_path)
    archive = rinco_zip(prosody.rinco_path(settings), files=1)
    with zipfile.ZipFile(archive) as opened:
        rinco_wav = opened.read("INTON-EAF/1-Questions/x-0.wav")
    pilot = settings.data_root / prosody.PILOT_PROVENANCE
    (pilot.parent / openstt.POOL_DIR).mkdir(parents=True)
    pool_wav = wav16(np.full(800, 0.1))
    (pilot.parent / openstt.POOL_DIR / "p.wav").write_bytes(pool_wav)
    record = {"kind": "openstt_asr_candidate", "member": "asr_calls_2_val/1/1/used.wav",
              "sha256": "d" * 64}  # fmt: skip
    pilot.write_text(json.dumps({"clips": [record]}))
    return settings, rinco_wav, pool_wav


def test_check_fresh_passes_a_set_that_shares_nothing(tmp_path) -> None:
    settings, _, _ = freshness_root(tmp_path)
    found = pc.check_fresh(settings, [fresh_entry("n1")], [])
    assert found["fresh"] and found["overlap"] == []
    assert found["references"] == {"rinco": 1, "pilot_1": 3, "gate0": len(prosody.SANITY_CLIPS)}


def test_check_fresh_rejects_every_kind_of_reused_file(tmp_path) -> None:
    settings, rinco_wav, pool_wav = freshness_root(tmp_path)
    sha = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    reused = [
        fresh_entry("rinco", member_sha=sha(rinco_wav)),
        fresh_entry("pool-file", wav_sha=sha(pool_wav)),
        fresh_entry("pool-pcm", pcm_sha=sha(gate0.wav_pcm(pool_wav))),
        fresh_entry("provenance", wav_sha="d" * 64),
        fresh_entry("gate0", pcm_sha=next(iter(prosody.SANITY_CLIPS.values()))),
        {**fresh_entry("pilot-member"), "member": "asr_calls_2_val/1/1/used.wav"},
        {**fresh_entry("pool-member"), "member": "asr_calls_2_val/2/2/pool.wav"},
    ]
    found = pc.check_fresh(
        settings, [*reused, fresh_entry("clean")], ["asr_calls_2_val/2/2/pool.wav"]
    )
    assert not found["fresh"]
    assert found["overlap"] == sorted(c["stem"] for c in reused)


# --- kappa, import and the verdict ---------------------------------------------------------


def test_kappa_of_known_labelings() -> None:
    assert pc.kappa([(True, True), (False, False)] * 3) == pytest.approx(1.0)
    assert pc.kappa([(True, False), (False, True)] * 3) == pytest.approx(-1.0)
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    assert pc.kappa(pairs) == pytest.approx(0.0)
    assert pc.kappa([(True, True)] * 4) is None
    assert pc.kappa([]) is None


ITEMS = [
    {"id": "P-00000001", "kind": "fresh", "words": ["раз", "два", "три"]},
    {"id": "P-00000002", "kind": "calibration", "words": ["мама", "мыла"],
     "rinco_marked": [True, False]},
]  # fmt: skip


def sheet(rows: list[tuple], delimiter: str = ",", bom: bool = False) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(pc.SHEET_COLUMNS)
    writer.writerows(rows)
    return ("﻿" if bom else "").encode() + buffer.getvalue().encode()


GOOD = [
    ("P-00000001", 1, "раз", "0", ""), ("P-00000001", 2, "два", "1", ""),
    ("P-00000001", 3, "три", "0", ""), ("P-00000002", 1, "мама", "1", ""),
    ("P-00000002", 2, "мыла", "0", ""),
]  # fmt: skip


def labels_of(rows, **kwargs) -> dict:
    return pc.validate_sheet(pc.read_sheet(sheet(rows, **kwargs)), ITEMS)


def test_a_filled_sheet_gives_the_marks_of_each_item() -> None:
    labels = labels_of(GOOD)
    assert labels["P-00000001"] == {"skip_reason": None, "marked": [False, True, False]}
    assert labels_of(GOOD, delimiter=";", bom=True) == labels
    assert pc.agreement(ITEMS, labels) == {
        "kappa": pytest.approx(1.0),
        "words": 2,
        "skipped_items": 0,
    }


def test_a_skipped_item_needs_empty_marks_and_a_reason() -> None:
    rows = [r[:3] + ("", "текст не совпадает" if r[1] == 1 else "") if r[0] == "P-00000001" else r
            for r in GOOD]  # fmt: skip
    assert labels_of(rows)["P-00000001"] == {"skip_reason": "текст не совпадает", "marked": None}
    with_marks = [r[:4] + ("плохой звук",) if r[0] == "P-00000001" else r for r in GOOD]
    with pytest.raises(pc.ImportRefused) as refused:
        labels_of(with_marks)
    assert refused.value.code == "partial_item"


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        (GOOD[:-1], "missing_row"),
        ([*GOOD[:-1], GOOD[-1][:3] + ("да", "")], "bad_marked"),
        ([*GOOD[:-1], GOOD[-1][:3] + ("2", "")], "bad_marked"),
        ([*GOOD, ("P-99999999", 1, "икс", "0", "")], "unknown_id"),
        ([*GOOD, ("P-00000002", 3, "раму", "0", "")], "unknown_word"),
        ([*GOOD, GOOD[0]], "duplicate_row"),
        ([GOOD[0][:2] + ("Раз", "0", ""), *GOOD[1:]], "word_changed"),
        ([*GOOD[:3], GOOD[3][:3] + ("", ""), GOOD[4]], "partial_item"),
        ([*GOOD[:3], GOOD[3][:3] + ("", ""), GOOD[4][:3] + ("", "")], "unlabeled_item"),
        ([r[:3] + ("", "") for r in GOOD], "unfilled_template"),
    ],
)
def test_import_refuses_a_bad_sheet_with_a_code(rows, code) -> None:
    with pytest.raises(pc.ImportRefused) as refused:
        labels_of(rows)
    assert refused.value.code == code


def test_import_refuses_a_bad_header_and_a_short_row() -> None:
    with pytest.raises(pc.ImportRefused, match="bad_header") as refused:
        pc.read_sheet("id,мыла,marked\n".encode())
    assert refused.value.problems == [("cells", 3)] and "мыла" not in str(refused.value)
    with pytest.raises(pc.ImportRefused, match="bad_header"):
        pc.read_sheet(b"")
    with pytest.raises(pc.ImportRefused, match="bad_row"):
        pc.read_sheet(sheet([("P-00000001", 1, "раз")]))
    with pytest.raises(pc.ImportRefused, match="not_utf8"):
        pc.read_sheet(b"\xff\xfe")


@pytest.mark.parametrize(
    ("coverage", "pairs", "clips", "auc_", "ci", "expected"),
    [
        (0.69, 5000, 160, 0.90, (0.85, 0.95), pc.INCONCLUSIVE),
        (0.80, 1199, 160, 0.90, (0.85, 0.95), pc.INCONCLUSIVE),
        (0.80, 5000, 59, 0.90, (0.85, 0.95), pc.INCONCLUSIVE),
        (0.70, 1200, 60, 0.75, (0.70, 0.80), pc.PASS),
        (0.80, 5000, 160, 0.749, (0.72, 0.78), pc.FAIL),
        (0.80, 5000, 160, 0.76, (0.69, 0.80), pc.FAIL),
        (0.80, 5000, 160, None, None, pc.FAIL),
    ],
)
def test_decide_follows_the_rule(coverage, pairs, clips, auc_, ci, expected) -> None:
    assert pc.decide(coverage, pairs, clips, auc_, ci) == expected


def clip_record(clip_id: str, marked_scores: list[float], plain_scores: list[float]) -> dict:
    words = [{"marked": True, "score": {"index": s, "z_f0": s, "z_rms": 0.0, "z_dur": 0.0}}
             for s in marked_scores]  # fmt: skip
    words += [{"marked": False, "score": {"index": s, "z_f0": 0.0, "z_rms": s, "z_dur": 0.0}}
              for s in plain_scores]  # fmt: skip
    return {"member": clip_id, "folder": "open-stt", "words": words, "asr_words": len(words),
            "asr_excluded": 0, "asr_unmatched": 0}  # fmt: skip


def test_the_metric_bootstraps_by_clip_and_counts_clips_with_pairs() -> None:
    records = [clip_record(f"P-{i}", [1.0], [0.0, 2.0]) for i in range(5)]
    records.append(clip_record("P-plain", [], [0.0]))
    found = pc.confirm_metric(records)
    per_clip = [prosody.pair_counts(r["words"]) for r in records]
    assert found["auc"] == pytest.approx(0.5) and found["pairs"] == 10
    assert found["ci95"] == list(prosody.bootstrap_ci(per_clip))
    assert found["clips"] == 6 and found["clips_with_pairs"] == 5
    assert "files" not in found and "verdict" not in found


# --- sanity stage parts --------------------------------------------------------------------


def test_r0_differences_name_a_changed_index_or_scored_state() -> None:
    old = [clip_record("m", [1.0], [0.0]) | {"utterance": 0}]
    same = [json.loads(json.dumps(old[0]))]
    same[0]["words"][0]["score"]["index"] += 1e-7
    assert pc.r0_differences(old, same) == []
    changed = json.loads(json.dumps(same))
    changed[0]["words"][0]["score"]["index"] += 1e-5
    assert pc.r0_differences(old, changed) == ["m#0 word 1: index differs"]
    unscored = json.loads(json.dumps(same))
    unscored[0]["words"][1]["score"] = None
    assert pc.r0_differences(old, unscored) == ["m#0 word 2: scored state differs"]
    assert pc.r0_differences(old, []) == ["m#0: not reproduced"]


def test_receipt_problems_name_each_differing_field(tmp_path) -> None:
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(RECEIPT))
    versions = {"faster-whisper": "9", "ctranslate2": "4.9.0"}
    new.write_text(json.dumps(RECEIPT | {"compute_type": "float32", "versions": versions}))
    assert pc.receipt_problems(old, old) == []
    assert pc.receipt_problems(new, old) == [
        "receipt compute_type differs", "receipt version of faster-whisper differs"
    ]  # fmt: skip
    assert pc.receipt_problems(tmp_path / "none.json", old) == [
        "model receipt unreadable: FileNotFoundError"
    ]


def test_the_controls_are_five_seconds_of_silence_pink_noise_and_a_tone() -> None:
    found = pc.controls()
    assert set(found) == {"silence", "pink_noise", "tone_440"}
    assert all(len(s) == 5 * RATE for s in found.values())
    assert not found["silence"].any()
    assert np.array_equal(found["pink_noise"], pc.controls()["pink_noise"])


# --- the steps on a synthetic station ------------------------------------------------------


@needs_ffmpeg
def test_prepare_builds_a_blind_package_once_and_records_no_text(station) -> None:
    settings, raised = station
    result = pc.prepare(settings, REPO / pc.DECISION_DOC)
    out = settings.data_root / pc.OUTPUT_DIR
    assert result["fresh_check"]["fresh"] and result["package"]["items"] == 6
    assert result["package"]["fresh"] == 4 and result["package"]["calibration"] == 2
    chosen = [c["stem"] for c in result["fresh_clips"]]
    assert stem(9) not in chosen and stem(90) not in chosen and stem(91) not in chosen
    assert not set(chosen) & {Path(m).stem for m in result["pool_81_members"]}
    listen = sorted(p.stem for p in (out / pc.PACKAGE_DIR / pc.LISTEN_DIR).iterdir())
    aliases = json.loads((out / pc.ALIAS_FILE).read_text())["items"]
    assert listen == sorted(i["id"] for i in aliases)
    assert all(i.startswith("P-") and len(i) == 10 for i in listen)
    package_files = {p.name for p in (out / pc.PACKAGE_DIR).iterdir()}
    assert package_files == {pc.LISTEN_DIR, pc.SHEET_FILE, pc.LABELING_DOC.name}
    sheet_text = (out / pc.PACKAGE_DIR / pc.SHEET_FILE).read_text()
    assert "/" not in sheet_text and "calibration" not in sheet_text and "fresh" not in sheet_text
    prepared = (out / pc.PREPARE_FILE).read_text()
    assert "обсудим" not in prepared and "мыла" not in prepared
    assert result["fresh_list_sha256"] == prosody.list_sha256(
        [f"{c['stem']} {c['wav_sha256']}" for c in result["fresh_clips"]]
    )
    with pytest.raises(pc.ConfirmError, match="prepare runs once"):
        pc.prepare(settings, REPO / pc.DECISION_DOC)
    assert pc.run_check_fresh(settings, REPO / pc.DECISION_DOC)["fresh"]
    clip = out / pc.CLIPS_DIR / f"{chosen[0]}.wav"
    clip.write_bytes(clip.read_bytes() + b"\0\0")
    assert pc.run_check_fresh(settings, REPO / pc.DECISION_DOC)["clip_files"] != "ok"
    assert json.loads((out / pc.RESULT_FILE).read_text())["verdict"] == pc.INVALID


@needs_ffmpeg
def test_prepare_with_a_reused_file_gives_invalid_and_no_package(station) -> None:
    settings, _ = station
    tar = pc.open_stt_path(settings)
    clips = settings.data_root / prosody.PILOT_PROVENANCE.parent / "clips"
    with tarfile.open(tar, "r:gz") as archive:
        for member in archive:
            if member.name.endswith(".wav"):
                (clips / Path(member.name).name).write_bytes(archive.extractfile(member).read())
    result = pc.prepare(settings, REPO / pc.DECISION_DOC)
    assert result["verdict"] == pc.INVALID and "check-fresh" in result["reason"]
    assert not (settings.data_root / pc.OUTPUT_DIR / pc.PACKAGE_DIR).exists()


@needs_ffmpeg
def test_prepare_import_sanity_and_run_give_one_verdict_below_the_sample_minimum(station) -> None:
    settings, raised = station
    decision = REPO / pc.DECISION_DOC
    pc.prepare(settings, decision)
    out = settings.data_root / pc.OUTPUT_DIR
    transcribe = script_transcriber()
    with pytest.raises(pc.ConfirmError, match="no imported labels"):
        pc.run_main(settings, decision, transcribe)
    template = out / pc.PACKAGE_DIR / pc.SHEET_FILE
    with pytest.raises(pc.ImportRefused) as refused:
        pc.import_sheet(settings, template, decision)
    assert refused.value.code == "unfilled_template"
    filled = settings.data_root / "filled.csv"
    filled.write_bytes(filled_sheet(settings, raised))
    imported = pc.import_sheet(settings, filled, decision)
    assert imported["agreement"]["kappa"] == pytest.approx(1.0)
    with pytest.raises(pc.ImportRefused) as refused:  # the first accepted sheet is final
        pc.import_sheet(settings, filled, decision)
    assert refused.value.code == "already_imported"
    with pytest.raises(pc.ConfirmError, match="no passed sanity attempt"):
        pc.run_main(settings, decision, transcribe)
    attempt = pc.run_sanity(settings, decision, transcribe)
    assert attempt["passed"], attempt
    assert attempt["r0_utterances"] == 3 and attempt["controls_scored_words"] == {
        "silence": 0, "pink_noise": 0, "tone_440": 0
    }  # fmt: skip
    with pytest.raises(pc.ImportRefused) as refused:
        pc.import_sheet(settings, filled, decision)
    assert refused.value.code == "already_imported"
    result = pc.run_main(settings, decision, transcribe)
    assert result["verdict"] == pc.INCONCLUSIVE  # 4 clips: far below 1 200 pairs and 60 clips
    assert result["clips"] == 4 and result["clips_with_pairs"] == 4 and result["pairs"] == 20
    assert result["coverage"] == pytest.approx(1.0) and result["auc"] == pytest.approx(1.0)
    with pytest.raises(pc.ConfirmError, match="verdict is out"):
        pc.run_main(settings, decision, transcribe)


@needs_ffmpeg
def test_kappa_below_0_40_gives_invalid_without_a_transcription(station) -> None:
    settings, raised = station
    decision = REPO / pc.DECISION_DOC
    pc.prepare(settings, decision)
    filled = settings.data_root / "filled.csv"
    filled.write_bytes(filled_sheet(settings, raised, calibration_agrees=False))
    assert pc.import_sheet(settings, filled, decision)["agreement"]["kappa"] < pc.KAPPA_MIN
    assert pc.run_sanity(settings, decision, script_transcriber())["passed"]
    calls: list = []
    result = pc.run_main(settings, decision, script_transcriber(calls=calls))
    assert result["verdict"] == pc.INVALID and "каппа" in result["reason"]
    assert calls == [] and not (settings.data_root / pc.OUTPUT_DIR / pc.MAIN_DIR).exists()


@needs_ffmpeg
def test_a_changed_imported_sheet_gives_invalid(station) -> None:
    settings, raised = station
    decision = REPO / pc.DECISION_DOC
    pc.prepare(settings, decision)
    filled = settings.data_root / "filled.csv"
    filled.write_bytes(filled_sheet(settings, raised))
    pc.import_sheet(settings, filled, decision)
    assert pc.run_sanity(settings, decision, script_transcriber())["passed"]
    copy = settings.data_root / pc.OUTPUT_DIR / pc.IMPORT_DIR / pc.SHEET_FILE
    copy.write_bytes(copy.read_bytes().replace(b",1,", b",0,", 1))
    result = pc.run_main(settings, decision, script_transcriber())
    assert result["verdict"] == pc.INVALID and "sheet SHA-256" in result["reason"]


@needs_ffmpeg
def test_two_failed_sanity_attempts_give_invalid_and_stop(station) -> None:
    settings, _ = station
    decision = REPO / pc.DECISION_DOC
    pc.prepare(settings, decision)
    path = pc.series_3_dir(settings) / "main" / "utterances.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first = min(records, key=lambda r: (r["member"], r["utterance"]))
    first["words"][0]["score"]["index"] += 0.01
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    other = script_transcriber(receipt=RECEIPT | {"compute_type": "float32"})
    one = pc.run_sanity(settings, decision, other)
    assert not one["passed"] and one["r0_differences"] and one["receipt_problems"]
    two = pc.run_sanity(settings, decision, script_transcriber())
    assert not two["passed"] and two["attempt"] == 2
    result = json.loads((settings.data_root / pc.OUTPUT_DIR / pc.RESULT_FILE).read_text())
    assert result["verdict"] == pc.INVALID
    with pytest.raises(pc.ConfirmError, match="verdict is out"):
        pc.run_sanity(settings, decision, script_transcriber())


@needs_ffmpeg
@pytest.mark.parametrize("first_passed", [False, True])
def test_a_last_sanity_attempt_killed_before_its_record_gives_invalid(
    station, monkeypatch, first_passed
) -> None:
    settings, _ = station
    decision = REPO / pc.DECISION_DOC
    pc.prepare(settings, decision)
    out = settings.data_root / pc.OUTPUT_DIR
    if first_passed:  # attempt 1 passed with the code before a fix, attempt 2 died
        with monkeypatch.context() as patched:
            patched.setattr(pc, "code_sha256", lambda: "1" * 64)
            assert pc.run_sanity(settings, decision, script_transcriber())["passed"]
    else:
        other = script_transcriber(receipt=RECEIPT | {"compute_type": "float32"})
        assert not pc.run_sanity(settings, decision, other)["passed"]
    assert not (out / pc.RESULT_FILE).exists()
    (out / pc.SANITY_DIR / "work-2").mkdir()  # killed: a work directory and no record
    calls: list = []
    result = pc.run_sanity(settings, decision, script_transcriber(calls=calls))
    assert result["verdict"] == pc.INVALID and "2 попытки" in result["reason"] and calls == []
    assert json.loads((out / pc.RESULT_FILE).read_text())["verdict"] == pc.INVALID
    assert not (out / pc.SANITY_DIR / "attempt-2.json").exists()
    with pytest.raises(pc.ConfirmError, match="verdict is out"):
        pc.run_sanity(settings, decision, script_transcriber())


@needs_ffmpeg
def test_a_main_run_left_without_a_result_is_inconclusive_and_not_rerun(station) -> None:
    settings, raised = station
    decision = REPO / pc.DECISION_DOC
    pc.prepare(settings, decision)
    filled = settings.data_root / "filled.csv"
    filled.write_bytes(filled_sheet(settings, raised))
    pc.import_sheet(settings, filled, decision)
    assert pc.run_sanity(settings, decision, script_transcriber())["passed"]

    def broken(samples, run_dir):
        raise MemoryError

    assert pc.run_main(settings, decision, broken)["verdict"] == pc.INCONCLUSIVE
    with pytest.raises(pc.ConfirmError, match="verdict is out"):
        pc.run_main(settings, decision, script_transcriber())


def test_import_and_run_need_prepare_first(tmp_path) -> None:
    settings = make_config(tmp_path)
    decision = REPO / pc.DECISION_DOC
    with pytest.raises(pc.ConfirmError, match="prepare"):
        pc.import_sheet(settings, tmp_path / "sheet.csv", decision)
    with pytest.raises(pc.ConfirmError, match="prepare"):
        pc.run_main(settings, decision)


def test_the_cli_names_the_refusal_code_and_prints_no_sheet_text(
    tmp_path, monkeypatch, capsys
) -> None:
    settings = make_config(tmp_path)
    (settings.data_root / prosody.CORPUS_DIR / "sources").mkdir(parents=True)
    (settings.data_root / prosody.CORPUS_DIR / "sources" / "placeholder").write_bytes(b"")
    monkeypatch.setattr(config, "load", lambda: settings)
    out = settings.data_root / pc.OUTPUT_DIR
    out.mkdir()
    (out / pc.PREPARE_FILE).write_text(json.dumps({"package": {"sheet_sha256": "0" * 64}}))
    (out / pc.ALIAS_FILE).write_text(json.dumps({"items": ITEMS}))
    bad = tmp_path / "bad.csv"
    bad.write_bytes(sheet(GOOD[:-1]))
    assert pc.main(["--decision", str(REPO / pc.DECISION_DOC), "import", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "missing_row" in err and "P-00000002 2" in err and "мыла" not in err
    header = tmp_path / "header.csv"
    header.write_bytes("id,word_no,мыла,marked\n".encode())
    assert pc.main(["--decision", str(REPO / pc.DECISION_DOC), "import", str(header)]) == 2
    err = capsys.readouterr().err
    assert "bad_header" in err and "cells 4" in err and "мыла" not in err
    good = tmp_path / "good.csv"
    good.write_bytes(sheet(GOOD))
    assert pc.main(["--decision", str(REPO / pc.DECISION_DOC), "import", str(good)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["agreement"]["kappa"] == pytest.approx(1.0) and "мама" not in json.dumps(printed)


def ready_for_run(station, raised: dict | None = None):
    """Prepare, import a filled sheet (fresh marks from `raised`) and pass one sanity attempt."""
    settings, station_raised = station
    decision = REPO / pc.DECISION_DOC
    pc.prepare(settings, decision)
    filled = settings.data_root / "filled.csv"
    filled.write_bytes(filled_sheet(settings, raised or station_raised))
    pc.import_sheet(settings, filled, decision)
    assert pc.run_sanity(settings, decision, script_transcriber())["passed"]
    return settings, decision, settings.data_root / pc.OUTPUT_DIR


@needs_ffmpeg
def test_run_writes_pass_when_the_sample_minimum_holds(station, monkeypatch) -> None:
    monkeypatch.setattr(pc, "MIN_PAIRS", 20)
    monkeypatch.setattr(pc, "MIN_CLIPS_WITH_PAIRS", 4)
    settings, decision, out = ready_for_run(station)
    result = pc.run_main(settings, decision, script_transcriber())
    assert result["verdict"] == pc.PASS and result["pairs"] == 20
    assert json.loads((out / pc.RESULT_FILE).read_text())["verdict"] == pc.PASS


@needs_ffmpeg
def test_run_writes_fail_when_the_marked_words_are_not_the_raised_ones(station, monkeypatch):
    monkeypatch.setattr(pc, "MIN_PAIRS", 20)
    monkeypatch.setattr(pc, "MIN_CLIPS_WITH_PAIRS", 4)
    _, raised = station
    wrong = {name: word % 6 + 1 for name, word in raised.items()}
    settings, decision, out = ready_for_run(station, wrong)
    result = pc.run_main(settings, decision, script_transcriber())
    assert result["verdict"] == pc.FAIL and result["auc"] < prosody.PASS_AUC
    assert json.loads((out / pc.RESULT_FILE).read_text())["verdict"] == pc.FAIL


@needs_ffmpeg
def test_a_main_run_broken_by_an_error_is_invalid_and_not_rerun(station) -> None:
    settings, decision, _ = ready_for_run(station)

    def broken(samples, run_dir):
        raise RuntimeError("decoder died")

    result = pc.run_main(settings, decision, broken)
    assert result["verdict"] == pc.INVALID and "прерван" in result["reason"]
    with pytest.raises(pc.ConfirmError, match="verdict is out"):
        pc.run_main(settings, decision, script_transcriber())


@needs_ffmpeg
def test_a_main_run_killed_before_its_result_is_inconclusive(station) -> None:
    settings, decision, out = ready_for_run(station)
    (out / pc.MAIN_DIR).mkdir()
    (out / pc.MAIN_DIR / pc.STARTED_FILE).write_text("{}")
    calls: list = []
    result = pc.run_main(settings, decision, script_transcriber(calls=calls))
    assert result["verdict"] == pc.INCONCLUSIVE and calls == []


@needs_ffmpeg
def test_a_main_run_stopped_by_the_wall_time_ceiling_is_inconclusive(station) -> None:
    settings, decision, _ = ready_for_run(station)
    result = pc.run_main(settings, decision, script_transcriber(), wall_seconds=-1)
    assert result["verdict"] == pc.INCONCLUSIVE and "потолком" in result["reason"]


@needs_ffmpeg
def test_a_q1_violation_at_the_main_run_is_invalid_without_a_transcription(
    station, monkeypatch
) -> None:
    settings, decision, out = ready_for_run(station)
    monkeypatch.setattr(pc, "q1", lambda _: ["the network guard is not active"])
    calls: list = []
    result = pc.run_main(settings, decision, script_transcriber(calls=calls))
    assert result["verdict"] == pc.INVALID and "Q1" in result["reason"]
    assert calls == [] and not (out / pc.MAIN_DIR).exists()


def eval_config(tmp_path: Path) -> Path:
    """An eval config file as `docs/eval/PROSODY-CONFIG.md` says, for the real `config.load`."""
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    path = tmp_path / "eval.toml"
    path.write_text(
        f"allowed_roots = [{json.dumps(str(tmp_path / 'in'))}]\n"
        f"data_root = {json.dumps(str(tmp_path / 'data'))}\n[network]\nallow = false\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("command", ["prepare", "check-fresh", "sanity", "run", "import"])
@pytest.mark.parametrize("sources", ["missing", "empty"])
def test_the_cli_refuses_a_data_root_without_corpus_sources_and_writes_nothing(
    tmp_path, monkeypatch, capsys, command, sources
) -> None:
    path = eval_config(tmp_path)
    if sources == "empty":
        (tmp_path / "data" / prosody.CORPUS_DIR / "sources").mkdir(parents=True)
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    argv = [command, str(tmp_path / "sheet.csv")] if command == "import" else [command]
    before = sorted(tmp_path.rglob("*"))
    assert pc.main(argv) == 2
    err = capsys.readouterr().err
    assert f"prosody-confirm: {path}: data_root has no corpus sources" in err
    assert config.CONFIG_PATH_ENV in err and "docs/eval/PROSODY-CONFIG.md" in err
    assert sorted(tmp_path.rglob("*")) == before


def test_the_cli_goes_past_the_check_when_the_data_root_has_corpus_sources(
    tmp_path, monkeypatch, capsys
) -> None:
    path = eval_config(tmp_path)
    sources = tmp_path / "data" / prosody.CORPUS_DIR / "sources"
    sources.mkdir(parents=True)
    (sources / "placeholder").write_bytes(b"")
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    assert pc.main(["check-fresh"]) == 2
    err = capsys.readouterr().err
    assert "run `prepare` first" in err and "corpus sources" not in err
    assert not (tmp_path / "data" / pc.OUTPUT_DIR / pc.RESULT_FILE).exists()


def test_the_cli_refuses_a_write_into_a_read_only_archive_with_exit_2(
    tmp_path, monkeypatch, capsys
) -> None:
    path = eval_config(tmp_path)
    sources = tmp_path / "data" / prosody.CORPUS_DIR / "sources"
    sources.mkdir(parents=True)
    (sources / "placeholder").write_bytes(b"")
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))

    def read_only(*_args):
        raise PermissionError(13, "Permission denied", "result.json")

    monkeypatch.setattr(pc, "run_sanity", read_only)
    assert pc.main(["sanity"]) == 2
    assert "prosody-confirm: [Errno 13] Permission denied: 'result.json'" in capsys.readouterr().err
