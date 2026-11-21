"""prosody-measure-1 (#134): the rule check, words, the index, the metric and the step contract.

All audio here is synthetic and the transcriber is a stand-in: these tests prove the contract of
`sound_scout.eval.prosody` (what is computed, recorded and refused), never that the index
separates real accents. That is only the real run recorded in docs/eval.
"""

import hashlib
import io
import json
import re
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sound_scout import config
from sound_scout.eval import corpus, prosody
from tests.test_eval_corpus import eaf

REPO = Path(__file__).resolve().parent.parent
RATE = prosody.SAMPLE_RATE
PIN_CPUS = prosody.pin_cpus
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def voice(seconds: float, f0: float = 120.0, level: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return level * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))


def words_signal(count: int, raised: int | None, word_s: float = 0.4, gap_s: float = 0.1):
    """`count` voiced words with gaps; word number `raised` (1-based) higher, louder and longer."""
    pieces, words, at = [np.zeros(round(0.3 * RATE))], [], 0.3
    for number in range(1, count + 1):
        hot = number == raised
        length = word_s * (1.6 if hot else 1.0)
        pieces.append(voice(length, 190.0 if hot else 120.0, 0.4 if hot else 0.2))
        words.append({"word": f"слово{number}", "start": at, "end": at + length})
        at += length
        pieces.append(np.zeros(round(gap_s * RATE)))
        at += gap_s
    pieces.append(np.zeros(round(0.3 * RATE)))
    return np.concatenate(pieces), words


def wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm(samples))
    return buffer.getvalue()


def pcm(samples: np.ndarray) -> bytes:
    return np.round(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def hand_track(f0: list[float], rms: list[float], speech: list[bool] | None = None) -> dict:
    times = np.arange(len(f0)) * 0.01
    return {
        "times_s": times,
        "speech": np.array(speech if speech is not None else [True] * len(f0)),
        "f0_hz": np.array(f0, dtype=np.float64),
        "rms_dbfs": np.array(rms, dtype=np.float64),
    }


def make_config(tmp_path: Path) -> config.Config:
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return config.Config(allowed_roots=(tmp_path / "in",), data_root=(tmp_path / "data").resolve())


# --- the rule and the pins -----------------------------------------------------------------


def test_rule_sha256_of_the_decision_is_the_one_recorded_before_the_run() -> None:
    assert prosody.check_rule(REPO / prosody.DECISION_DOC) == prosody.RULE_SHA256
    text = (REPO / prosody.DECISION_DOC).read_text(encoding="utf-8")
    assert f"SHA-256 блока: `{prosody.RULE_SHA256}`" in text


def test_every_step_refuses_to_start_when_the_rule_block_changed(tmp_path: Path) -> None:
    edited = tmp_path / "0019.md"
    text = (REPO / prosody.DECISION_DOC).read_text(encoding="utf-8")
    edited.write_text(text.replace("AUC >= 0,75", "AUC >= 0,70", 1), encoding="utf-8")
    settings = make_config(tmp_path)
    for step in (prosody.prepare, prosody.run_sanity, prosody.run_main):
        with pytest.raises(prosody.ProsodyError, match="SHA-256"):
            step(settings, edited)
    assert not (settings.data_root / prosody.OUTPUT_DIR).exists()


def test_rule_block_must_be_one_prosody_rule_v1_block() -> None:
    text = (REPO / prosody.DECISION_DOC).read_text(encoding="utf-8")
    for broken in (text.replace(prosody.RULE_BEGIN, ""), text + prosody.RULE_BEGIN):
        with pytest.raises(prosody.ProsodyError):
            prosody.rule_sha256(broken)
    with pytest.raises(prosody.ProsodyError, match="PROSODY-RULE v1"):
        prosody.rule_sha256(text.replace("PROSODY-RULE v1 (", "PROSODY-RULE v2 ("))


def test_pins_are_the_ones_of_the_dependencies_doc() -> None:
    doc = (REPO / "docs/dependencies.md").read_text(encoding="utf-8")
    for value in (prosody.ASR_REPO, prosody.ASR_REVISION, prosody.RINCO_SHA256):
        assert value in doc
    for name, sha in prosody.ASR_WEIGHTS_SHA256.items():
        assert f"`{name}` SHA-256 `{sha}`" in doc
    assert prosody.rinco_source().sha256 == prosody.RINCO_SHA256


def test_sanity_clips_and_target_words_are_the_table_of_decision_0010() -> None:
    doc = (REPO / "docs/decisions/0010-gate0-borealis.md").read_text(encoding="utf-8")
    for clip_id, sha in prosody.SANITY_CLIPS.items():
        assert re.search(rf"\| `{clip_id}` \|.*`{sha}`", doc)
    for pair in prosody.SANITY_PAIRS:
        start, end = pair["interval"]
        row = rf"\| `{pair['base']}` \|.*\| № {pair['word']}, {start:.2f}–{end:.2f} с \|"
        assert re.search(row, doc)


# --- words and alignment -------------------------------------------------------------------


def test_rinco_words_read_marks_and_drop_tokens_without_letters() -> None:
    words = prosody.rinco_words("Ваня с *жено/й (.) при+ехал? | Ёлки\\ сто́ит 2")
    forms = ["ваня", "с", "женой", "приехал", "елки", "стоит", "2"]
    assert [w["form"] for w in words] == forms
    assert [w["marked"] for w in words] == [False, False, True, True, True, False, False]
    assert prosody.rinco_words("(.) | …") == []


def test_normalize_and_vowel_count() -> None:
    assert prosody.normalize(" Ещё,") == "еще"
    assert prosody.normalize("кто-то!") == "ктото"
    assert prosody.vowel_count("вздр") == 1
    assert prosody.vowel_count("молоко") == 3


def test_align_keeps_order_and_skips_extra_words() -> None:
    asr_ = ["ну", "мама", "мыла", "эту", "раму"]
    rinco = ["мама", "мыла", "раму", "вчера"]
    assert prosody.align(asr_, rinco) == [(1, 0), (2, 1), (4, 2)]
    assert prosody.align(["раму", "мама"], ["мама", "раму"]) in ([(1, 0)], [(0, 1)])
    assert prosody.align([], ["мама"]) == []


# --- the index -----------------------------------------------------------------------------


def test_raw_features_follow_the_rule_formulas() -> None:
    f0 = [100.0] * 10 + [200.0] * 10 + [0.0] * 10
    rms = [-30.0] * 10 + [-20.0] * 10 + [np.nan] * 10
    track = hand_track(f0, rms)
    words = [
        {"form": "мама", "start": 0.0, "end": 0.1},
        {"form": "рама", "start": 0.1, "end": 0.2},
        {"form": "вздр", "start": 0.2, "end": 0.3},
    ]
    features = prosody.raw_features(track, words)
    median_f0 = float(np.median([100.0] * 10 + [200.0] * 10))
    assert features[0]["f0"] == pytest.approx(12 * np.log2(100 / median_f0))
    assert features[1]["f0"] == pytest.approx(12 * np.log2(200 / median_f0))
    assert features[2]["f0"] == 0.0  # no voiced frame
    assert features[1]["rms"] == pytest.approx(-20 - (-25))
    assert features[2]["rms"] == 0.0  # digital silence only
    durations = [np.log(0.1 / 2), np.log(0.1 / 2), np.log(0.1 / 1)]
    assert features[2]["dur"] == pytest.approx(durations[2] - np.median(durations))


def test_index_is_the_equal_weight_sum_of_z_scores_and_zero_without_spread() -> None:
    scores = prosody.index([{"f0": 0, "rms": 1, "dur": 5}, {"f0": 2, "rms": 1, "dur": 1}])
    assert scores[0]["z_rms"] == scores[1]["z_rms"] == 0.0
    assert scores[1]["z_f0"] == pytest.approx(1.0)
    assert scores[1]["index"] == pytest.approx(1.0 + 0.0 - 1.0)


def test_score_utterance_excludes_empty_asr_words_and_counts_unmatched() -> None:
    track = hand_track([120.0] * 50, [-30.0] * 50)
    asr_words = [
        {"word": " Мама", "start": 0.0, "end": 0.1},
        {"word": " ну", "start": 0.1, "end": 0.1},  # zero duration: excluded before matching
        {"word": " мыла", "start": 0.1, "end": 0.2},
        {"word": " эту", "start": 0.2, "end": 0.3},
        {"word": " раму.", "start": 0.6, "end": 0.7},  # past the track: no frames
    ]
    rinco = prosody.rinco_words("мама /мыла раму")
    scored = prosody.score_utterance(track, asr_words, rinco)
    assert scored["asr_excluded"] == 2
    assert scored["asr_unmatched"] == 1
    assert [w["score"] is not None for w in scored["words"]] == [True, True, False]


def test_track_index_rises_on_the_raised_word_of_a_synthetic_signal() -> None:
    samples, words = words_signal(5, raised=3)
    scores = prosody.clip_scores(samples, words)
    values = [s["index"] for s in scores]
    assert values.index(max(values)) == 2


# --- the metric ----------------------------------------------------------------------------


def word(marked: bool, value: float | None) -> dict:
    return {"form": "x", "marked": marked, "score": None if value is None else {"index": value}}


def test_pair_counts_and_auc_count_ties_as_half() -> None:
    words = [word(True, 2.0), word(True, 1.0), word(False, 1.0), word(False, None)]
    assert prosody.pair_counts(words) == (1.5, 2)
    assert prosody.auc([(1.5, 2), (0.0, 2)]) == pytest.approx(0.375)
    assert prosody.auc([]) is None


def test_bootstrap_ci_is_seeded_and_resamples_files() -> None:
    per_file = [(9.0, 10), (5.0, 10), (8.0, 10), (7.0, 10)]
    first = prosody.bootstrap_ci(per_file)
    assert first == prosody.bootstrap_ci(per_file)
    assert first[0] <= 0.725 <= first[1]
    assert prosody.bootstrap_ci([(3.0, 4)]) == (0.75, 0.75)
    assert prosody.bootstrap_ci([]) is None


@pytest.mark.parametrize(
    ("coverage", "auc_", "ci", "expected"),
    [
        (0.70, 0.75, (0.70, 0.80), prosody.PASS),
        (0.90, 0.74, (0.71, 0.80), prosody.FAIL),
        (0.90, 0.80, (0.69, 0.90), prosody.FAIL),
        (0.90, None, None, prosody.FAIL),
        (0.69, 0.99, (0.95, 1.0), prosody.INCONCLUSIVE),
    ],
)
def test_verdict_follows_the_rule(coverage, auc_, ci, expected) -> None:
    assert prosody.verdict(coverage, auc_, ci) == expected


def test_metric_coverage_counts_unmatched_marked_words_in_the_denominator() -> None:
    records = [
        {"member": "a.wav", "folder": "1-Questions", "asr_words": 3, "asr_excluded": 0,
         "asr_unmatched": 0, "words": [word(True, 2.0), word(False, 1.0), word(True, None)]},
        {"member": "b.wav", "folder": "3-Statements", "asr_words": 0, "asr_excluded": 0,
         "asr_unmatched": 0, "skipped": "too_short", "words": [word(True, None)]},
    ]  # fmt: skip
    for record in records:
        for w in record["words"]:
            if w["score"]:
                w["score"].update(z_f0=w["score"]["index"], z_rms=0.0, z_dur=0.0)
    found = prosody.metric(records)
    assert found["coverage"] == pytest.approx(1 / 3)
    assert found["verdict"] == prosody.INCONCLUSIVE
    assert found["auc"] == 1.0
    assert found["auc_by_feature"]["z_rms"] == 0.5
    assert found["auc_by_section"] == {"1-Questions": 1.0, "3-Statements": None}
    assert found["utterances_not_scored"] == {"too_short": 1}
    assert found["unmatched_rinco_words"] == 2


# --- material ------------------------------------------------------------------------------

ACCENT = "Мама /мыла раму"


def rinco_archive(path: Path, files: int = 3, utterances: int = 2) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(files):
            stem = f"INTON-EAF/1-Questions/x-{i}"
            signal = np.concatenate([words_signal(3, raised=2)[0] for _ in range(utterances)])
            archive.writestr(f"{stem}.wav", wav_bytes(signal))
            span = len(signal) / RATE / utterances
            tiers = {
                "Accent": [(k * span, (k + 1) * span, ACCENT) for k in range(utterances)]
                + [(0.0, 0.05, "(.)")],
                "Int phrase": [(0.0, span, "фраза")],
                "Pause": [(0.0, 0.1, "0,3")],
            }
            archive.writestr(f"{stem}.eaf", eaf(f"x-{i}.wav", tiers))
    return path


def provenance(members: list[str]) -> dict:
    clips = [{"kind": "rinco_candidate", "member": members[0],
              "source": f"sources/rinco/{corpus.RINCO_ARCHIVE}"},
             {"kind": "corrupted_asr", "member": members[0],
              "source": f"sources/rinco/{corpus.RINCO_ARCHIVE}"},
             {"kind": "openstt_asr_candidate", "member": "asr_calls_2_val/x.wav", "source": "x"},
             {"kind": "noise", "member": "DLIVING/ch01.wav", "source": "y"}]  # fmt: skip
    return {"clips": clips}


def test_excluded_members_are_the_rinco_files_of_pilot_1_with_a_list_hash() -> None:
    found = prosody.excluded_members(provenance(["INTON-EAF/1-Questions/x-0.wav"]))
    assert found == ["INTON-EAF/1-Questions/x-0.wav"]
    expected = hashlib.sha256(b"INTON-EAF/1-Questions/x-0.wav\n").hexdigest()
    assert prosody.list_sha256(found) == expected


def test_material_leaves_out_the_pilot_files_and_counts_the_rest(tmp_path: Path) -> None:
    with zipfile.ZipFile(rinco_archive(tmp_path / "r.zip")) as archive:
        files, counts = prosody.material(archive, ["INTON-EAF/1-Questions/x-0.wav"])
        assert "INTON-EAF/1-Questions/x-0.wav" not in {f.wav_member for f in files}
        assert counts["excluded_files"] == 1 and counts["files"] == 2
        assert counts["utterances"] == 4 and counts["words"] == 12 and counts["marked_words"] == 4
        assert counts["mixed_utterances"] == 4
        with pytest.raises(prosody.ProsodyError, match="does not pair"):
            prosody.material(archive, ["INTON-EAF/9-None/y.wav"])


# --- precondition Q1 and the scope ---------------------------------------------------------


def test_q1_fails_with_an_open_network_and_a_wrong_rinco_archive(tmp_path: Path) -> None:
    settings = make_config(tmp_path)
    settings = config.Config(
        allowed_roots=settings.allowed_roots,
        data_root=settings.data_root,
        network=config.Network(allow=True),
    )
    path = prosody.rinco_path(settings)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the archive")
    problems = " | ".join(prosody.q1(settings))
    assert "network guard is not active" in problems
    assert corpus.RINCO_ARCHIVE in problems


def test_memory_scope_reads_the_cgroup_and_scope_problems_need_8_gb_without_swap(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc/self"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/user.slice/run-x.scope\n")
    group = tmp_path / "cg/user.slice/run-x.scope"
    group.mkdir(parents=True)
    (group / "memory.max").write_text("8000000000\n")
    (group / "memory.swap.max").write_text("0\n")
    scope = prosody.memory_scope(tmp_path / "proc", tmp_path / "cg")
    assert scope == ("8000000000", "0")
    assert prosody.scope_problems(scope) == []
    assert len(prosody.scope_problems(("max", "max"))) == 2
    assert prosody.scope_problems(("8000000001", "0"))
    assert prosody.scope_problems(None)
    assert prosody.memory_scope(tmp_path / "none", tmp_path / "cg") is None


def test_pin_cpus_keeps_at_most_eight(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = []
    monkeypatch.setattr(prosody.os, "sched_getaffinity", lambda _: set(range(16)))
    monkeypatch.setattr(prosody.os, "sched_setaffinity", lambda _, cpus: pinned.append(cpus))
    assert prosody.pin_cpus() == list(range(8))
    assert pinned == [list(range(8))]


# --- sanity stage --------------------------------------------------------------------------


def gate0_clips(root: Path, variant_raised: bool = True) -> tuple[dict, dict]:
    """Synthetic stand-ins for the nine Gate 0 clips, their PCM hashes and their ASR words."""
    root.mkdir(parents=True, exist_ok=True)
    hashes, words = {}, {}
    for pair in prosody.SANITY_PAIRS:
        for role in ("base", "variant"):
            raised = pair["word"] if role == "variant" and variant_raised else None
            samples, found = words_signal(16, raised)
            data = wav_bytes(samples)
            (root / f"{pair[role]}.wav").write_bytes(data)
            hashes[pair[role]] = hashlib.sha256(pcm(samples)).hexdigest()
            words[pair[role]] = found
    for clip_id in prosody.SANITY_CONTROLS:
        samples = np.zeros(RATE) if clip_id == "S0" else voice(1.0, 440.0)
        (root / f"{clip_id}.wav").write_bytes(wav_bytes(samples))
        hashes[clip_id] = hashlib.sha256(pcm(samples)).hexdigest()
        words[clip_id] = []
    return hashes, words


def fake_transcriber(words: dict, calls: list | None = None):
    def transcribe(samples, run_dir):
        if calls is not None:
            calls.append((run_dir.name, len(samples)))
        found = [dict(w) for w in words.get(run_dir.name, ACCENT_WORDS)]
        return prosody.transcribed("completed", None, found, trusted_segments=int(bool(found)))

    return transcribe


def test_read_clip_refuses_a_clip_whose_pcm_differs(tmp_path: Path) -> None:
    path = tmp_path / "B1.wav"
    path.write_bytes(wav_bytes(voice(0.5)))
    with pytest.raises(prosody.ProsodyError, match="PCM SHA-256"):
        prosody.read_clip(path, "0" * 64)
    good = hashlib.sha256(pcm(voice(0.5))).hexdigest()
    assert len(prosody.read_clip(path, good)) == RATE // 2


def test_sanity_passes_when_variants_rise_and_controls_have_no_words(tmp_path: Path) -> None:
    hashes, words = gate0_clips(tmp_path / "clips")
    clips = {c: prosody.read_clip(tmp_path / "clips" / f"{c}.wav", h) for c, h in hashes.items()}
    found = prosody.sanity(clips, fake_transcriber(words), tmp_path)
    assert found["s1"] and found["s2"] and found["passed"]
    assert found["pairs"][1]["expected_variant_interval"] == [4.22, 4.924]


def test_sanity_fails_s1_when_variants_do_not_rise_and_s2_on_scored_control(tmp_path) -> None:
    hashes, words = gate0_clips(tmp_path / "clips", variant_raised=False)
    words["T1"] = [{"word": "ля", "start": 0.1, "end": 0.6}]
    clips = {c: prosody.read_clip(tmp_path / "clips" / f"{c}.wav", h) for c, h in hashes.items()}
    found = prosody.sanity(clips, fake_transcriber(words), tmp_path)
    assert not found["s1"] and not found["s2"] and not found["passed"]
    assert found["controls_scored_words"]["T1"] == 1


def sanity_setup(tmp_path, monkeypatch, variant_raised=True):
    settings = make_config(tmp_path)
    hashes, words = gate0_clips(settings.data_root / prosody.GATE0_CLIPS, variant_raised)
    monkeypatch.setattr(prosody, "SANITY_CLIPS", hashes)
    monkeypatch.setattr(prosody, "q1", lambda _: [])
    monkeypatch.setattr(prosody, "rinco_problems", lambda _: [])
    monkeypatch.setattr(prosody, "memory_scope", lambda: (str(prosody.MEMORY_MAX_BYTES), "0"))
    monkeypatch.setattr(prosody, "pin_cpus", lambda: [0])
    archive = prosody.rinco_path(settings)
    archive.parent.mkdir(parents=True)
    rinco_archive(archive)
    members = ["INTON-EAF/1-Questions/x-0.wav"]
    provenance_path = settings.data_root / prosody.PILOT_PROVENANCE
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(json.dumps(provenance(members)))
    return settings, words


@needs_ffmpeg
def test_prepare_records_the_rule_and_the_exclusion_list_once(tmp_path, monkeypatch) -> None:
    settings, _ = sanity_setup(tmp_path, monkeypatch)
    record = prosody.prepare(settings, REPO / prosody.DECISION_DOC)
    assert record["rule_sha256"] == prosody.RULE_SHA256
    assert record["exclusion"]["count"] == 1
    assert record["exclusion"]["sha256"] == prosody.list_sha256(["INTON-EAF/1-Questions/x-0.wav"])
    assert record["material"]["files"] == 2
    with pytest.raises(prosody.ProsodyError, match="once"):
        prosody.prepare(settings, REPO / prosody.DECISION_DOC)


def test_prepare_with_a_failed_q1_gives_invalid(tmp_path, monkeypatch) -> None:
    settings, _ = sanity_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(prosody, "q1", lambda _: ["ASR model.bin SHA-256 differs"])
    record = prosody.prepare(settings, REPO / prosody.DECISION_DOC)
    assert record["verdict"] == prosody.INVALID and "Q1" in record["reason"]
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.run_sanity(settings, REPO / prosody.DECISION_DOC)


@needs_ffmpeg
def test_two_failed_sanity_attempts_give_invalid_and_stop(tmp_path, monkeypatch) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch, variant_raised=False)
    decision = REPO / prosody.DECISION_DOC
    with pytest.raises(prosody.ProsodyError, match="prepare"):
        prosody.run_sanity(settings, decision, fake_transcriber(words))
    prosody.prepare(settings, decision)
    first = prosody.run_sanity(settings, decision, fake_transcriber(words))
    assert first["attempt"] == 1 and not first["passed"]
    assert first["code_sha256"] == prosody.code_sha256()
    second = prosody.run_sanity(settings, decision, fake_transcriber(words))
    assert second["attempt"] == 2 and not second["passed"]
    out = settings.data_root / prosody.OUTPUT_DIR
    result = json.loads((out / prosody.RESULT_FILE).read_text())
    assert result["verdict"] == prosody.INVALID
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.run_sanity(settings, decision, fake_transcriber(words))
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.run_main(settings, decision, fake_transcriber(words), scope=False)


@needs_ffmpeg
def test_a_sanity_attempt_records_its_memory_scope_and_at_most_eight_cpus(
    tmp_path, monkeypatch
) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    pinned = []
    monkeypatch.setattr(prosody, "pin_cpus", PIN_CPUS)
    monkeypatch.setattr(prosody.os, "sched_getaffinity", lambda _: set(range(16)))
    monkeypatch.setattr(prosody.os, "sched_setaffinity", lambda _, cpus: pinned.append(cpus))
    prosody.prepare(settings, decision)
    record = prosody.run_sanity(settings, decision, fake_transcriber(words))
    assert pinned == [list(range(8))]
    attempt = settings.data_root / prosody.OUTPUT_DIR / prosody.SANITY_DIR / "attempt-1.json"
    saved = json.loads(attempt.read_text())
    assert saved["cpus"] == record["cpus"] == list(range(8))
    assert saved["memory_scope"] == [str(prosody.MEMORY_MAX_BYTES), "0"]


@needs_ffmpeg
def test_a_sanity_attempt_outside_the_memory_scope_is_refused_and_not_counted(
    tmp_path, monkeypatch
) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    prosody.prepare(settings, decision)
    pinned = []
    monkeypatch.setattr(prosody, "pin_cpus", lambda: pinned.append(1) or [0])
    sanity_dir = settings.data_root / prosody.OUTPUT_DIR / prosody.SANITY_DIR
    for scope in (("max", "0"), (str(prosody.MEMORY_MAX_BYTES), "max"), None):
        monkeypatch.setattr(prosody, "memory_scope", lambda scope=scope: scope)
        with pytest.raises(prosody.ProsodyError, match="resource scope"):
            prosody.run_sanity(settings, decision, fake_transcriber(words))
    assert pinned == [] and list(sanity_dir.iterdir()) == []
    assert not (settings.data_root / prosody.OUTPUT_DIR / prosody.RESULT_FILE).exists()
    monkeypatch.setattr(prosody, "memory_scope", lambda: (str(prosody.MEMORY_MAX_BYTES), "0"))
    assert prosody.run_sanity(settings, decision, fake_transcriber(words))["attempt"] == 1


# --- main run ------------------------------------------------------------------------------

ACCENT_WORDS = words_signal(3, raised=2)[1]
for _w, _form in zip(ACCENT_WORDS, ("мама", "мыла", "раму"), strict=True):
    _w["word"] = f" {_form.capitalize()}"


def ready_for_main(tmp_path, monkeypatch):
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    monkeypatch.setattr(prosody, "pin_cpus", lambda: [0])
    prosody.prepare(settings, decision)
    assert prosody.run_sanity(settings, decision, fake_transcriber(words))["passed"]
    return settings, decision


@needs_ffmpeg
def test_main_run_scores_every_utterance_and_computes_the_metric_once(
    tmp_path, monkeypatch
) -> None:
    settings, decision = ready_for_main(tmp_path, monkeypatch)
    calls = []
    result = prosody.run_main(settings, decision, fake_transcriber({}, calls), scope=False)
    assert len(calls) == 4  # two files of two utterances; the pilot file and "(.)" left out
    assert result["coverage"] == 1.0 and result["utterances"] == 4
    assert result["auc"] == 1.0 and result["ci95"] == [1.0, 1.0]
    assert result["verdict"] == prosody.PASS
    out = settings.data_root / prosody.OUTPUT_DIR
    lines = (out / prosody.MAIN_DIR / prosody.UTTERANCES_FILE).read_text().splitlines()
    assert len(lines) == 4
    assert json.loads((out / prosody.RESULT_FILE).read_text())["auc"] == 1.0
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False)


@needs_ffmpeg
def test_main_run_needs_a_passed_sanity_attempt_of_this_code(tmp_path, monkeypatch) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    prosody.prepare(settings, decision)
    with pytest.raises(prosody.ProsodyError, match="sanity"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False)
    prosody.run_sanity(settings, decision, fake_transcriber(words))
    monkeypatch.setattr(prosody, "code_sha256", lambda: "f" * 64)
    with pytest.raises(prosody.ProsodyError, match="sanity"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False)


@needs_ffmpeg
def test_main_run_refuses_without_the_memory_scope(tmp_path, monkeypatch) -> None:
    settings, decision = ready_for_main(tmp_path, monkeypatch)
    with pytest.raises(prosody.ProsodyError, match="memory.max"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=("max", "0"))
    assert not (settings.data_root / prosody.OUTPUT_DIR / prosody.MAIN_DIR).exists()


@needs_ffmpeg
def test_main_run_over_the_wall_time_ceiling_is_inconclusive(tmp_path, monkeypatch) -> None:
    settings, decision = ready_for_main(tmp_path, monkeypatch)
    result = prosody.run_main(settings, decision, fake_transcriber({}), scope=False, wall_seconds=0)
    assert result["verdict"] == prosody.INCONCLUSIVE
    assert "wall time" in result["reason"]
    assert "auc" not in result
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False)


@needs_ffmpeg
def test_a_main_run_that_breaks_is_recorded_invalid_and_not_rerun(tmp_path, monkeypatch) -> None:
    settings, decision = ready_for_main(tmp_path, monkeypatch)

    def broken(samples, run_dir):
        raise RuntimeError("decoder gone")

    result = prosody.run_main(settings, decision, broken, scope=False)
    assert result["verdict"] == prosody.INVALID and "decoder gone" in result["reason"]
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False)


@needs_ffmpeg
def test_a_main_run_left_without_a_result_is_not_started_again(tmp_path, monkeypatch) -> None:
    settings, decision = ready_for_main(tmp_path, monkeypatch)
    main_dir = settings.data_root / prosody.OUTPUT_DIR / prosody.MAIN_DIR
    main_dir.mkdir()
    with pytest.raises(prosody.ProsodyError, match="starts once"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False)
    (main_dir / prosody.STARTED_FILE).write_text("{}")
    calls = []
    result = prosody.run_main(settings, decision, fake_transcriber({}, calls), scope=False)
    assert result["verdict"] == prosody.INCONCLUSIVE and "без результата" in result["reason"]
    assert calls == []


@needs_ffmpeg
def test_a_main_run_out_of_memory_is_inconclusive(tmp_path, monkeypatch) -> None:
    settings, decision = ready_for_main(tmp_path, monkeypatch)

    def hungry(samples, run_dir):
        raise MemoryError

    result = prosody.run_main(settings, decision, hungry, scope=False)
    assert result["verdict"] == prosody.INCONCLUSIVE and "памяти" in result["reason"]


@needs_ffmpeg
def test_a_main_run_that_ends_past_the_ceiling_is_inconclusive(tmp_path, monkeypatch) -> None:
    settings, decision = ready_for_main(tmp_path, monkeypatch)
    slow = fake_transcriber({})

    def late(samples, run_dir):
        found = slow(samples, run_dir)
        clock[0] += 1000.0
        return found

    clock = [0.0]
    monkeypatch.setattr(prosody.time, "monotonic", lambda: clock[0])
    result = prosody.run_main(settings, decision, late, scope=False, wall_seconds=3500)
    assert result["verdict"] == prosody.INCONCLUSIVE, result
    assert result["utterances_done"] == 4, result


@needs_ffmpeg
def test_a_broken_sanity_attempt_is_recorded_and_counts(tmp_path, monkeypatch) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    prosody.prepare(settings, decision)

    def failing(samples, run_dir):
        return prosody.transcribed("failed", "RuntimeError: boom")

    first = prosody.run_sanity(settings, decision, failing)
    assert first["attempt"] == 1 and not first["passed"] and "boom" in first["error"]
    (settings.data_root / prosody.OUTPUT_DIR / prosody.SANITY_DIR / "work-2").mkdir()
    with pytest.raises(prosody.ProsodyError, match="2 sanity attempts made"):
        prosody.run_sanity(settings, decision, fake_transcriber(words))


def test_s2_fails_on_a_trusted_control_segment_without_scored_words(tmp_path) -> None:
    hashes, words = gate0_clips(tmp_path / "clips")
    clips = {c: prosody.read_clip(tmp_path / "clips" / f"{c}.wav", h) for c, h in hashes.items()}
    plain = fake_transcriber(words)

    def transcribe(samples, run_dir):
        found = plain(samples, run_dir)
        if run_dir.name == "N1":
            found["trusted_segments"] = 1  # a trusted segment whose words are all degenerate
        return found

    found = prosody.sanity(clips, transcribe, tmp_path)
    assert found["controls_scored_words"]["N1"] == 0
    assert not found["s2"] and found["controls_trusted_segments"]["N1"] == 1


def test_reused_model_loads_once_and_restores_the_asr_functions(monkeypatch, tmp_path) -> None:
    from sound_scout import asr

    loads = []
    monkeypatch.setattr(asr, "_load_model", lambda path: loads.append(path) or object())
    original = asr._load_model
    (tmp_path / "w.bin").write_bytes(b"w")
    with prosody._reused_model():
        first = asr._load_model(tmp_path)
        assert asr._load_model(tmp_path) is first
        assert asr._sha256_file(tmp_path / "w.bin") == hashlib.sha256(b"w").hexdigest()
    assert loads == [tmp_path]
    assert asr._load_model is original


def test_run_report_names_the_verdict_the_hashes_and_holds_no_private_paths() -> None:
    report = (REPO / "docs/eval/2026-11-21-prosody-measure-1.md").read_text(encoding="utf-8")
    assert "**Вердикт: INVALID.** Санитарная стадия не пройдена за 2 попытки" in report
    assert "result.manual-2026-11-21T0833Z.json" in report
    assert "отступление от буквы правила" not in report
    assert prosody.RULE_SHA256 in report and prosody.RINCO_SHA256 in report
    assert "23a224baf9e3c07e7bcd4b03d6c1eec41a200eee65aa0d422a3bd61b339ce3c7" in report
    assert not re.search(r"/(home|Users)/|~/|[A-Za-z]:\\", report)


# --- series --------------------------------------------------------------------------------


def test_series_2_names_the_asr_rule_of_135_and_the_rinco_files_it_read() -> None:
    second = prosody.SERIES[2]
    assert second["dir"] == "prosody-measure-1-series-2" != prosody.SERIES[1]["dir"]
    (change,) = second["data_changes"]
    assert change["issue"].endswith("/issues/135")
    assert re.fullmatch(r"[0-9a-f]{40}", change["commit"])
    decision = (REPO / change["decision"]).read_text(encoding="utf-8")
    listed = re.findall(r"^- `(INTON-EAF/[^`]+)`$", decision.split("## Файлы RINCO")[1], re.M)
    assert sorted(listed) == list(second["excluded"])


@needs_ffmpeg
def test_series_2_runs_in_its_own_directory_after_a_decided_series_1(tmp_path, monkeypatch) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    first = settings.data_root / prosody.SERIES[1]["dir"]
    first.mkdir()
    (first / prosody.RESULT_FILE).write_text('{"verdict": "INVALID"}')
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.prepare(settings, decision)
    extra = ("INTON-EAF/1-Questions/x-1.wav",)
    monkeypatch.setitem(prosody.SERIES, 2, dict(prosody.SERIES[2], excluded=extra))
    record = prosody.prepare(settings, decision, series=2)
    members = ["INTON-EAF/1-Questions/x-0.wav", *extra]
    assert record["series"] == 2 and record["data_changes"] == list(
        prosody.SERIES[2]["data_changes"]
    )
    assert record["exclusion"] == {
        "members": members,
        "count": 2,
        "sha256": prosody.list_sha256(members),
    }
    assert record["material"]["files"] == 1
    attempt = prosody.run_sanity(settings, decision, fake_transcriber(words), series=2)
    assert attempt["attempt"] == 1 and attempt["passed"] and attempt["series"] == 2
    calls = []
    result = prosody.run_main(
        settings, decision, fake_transcriber({}, calls), scope=False, series=2
    )
    assert len(calls) == 2 and result["verdict"] == prosody.PASS
    assert result["exclusion_sha256"] == prosody.list_sha256(members)
    second = settings.data_root / prosody.SERIES[2]["dir"]
    assert sorted(p.name for p in second.iterdir()) == [
        "main",
        "prepare.json",
        "result.json",
        "sanity",
    ]
    assert sorted(p.name for p in first.iterdir()) == [prosody.RESULT_FILE]


def test_an_unapproved_series_is_refused_before_any_file(tmp_path, monkeypatch) -> None:
    settings, _ = sanity_setup(tmp_path, monkeypatch)
    with pytest.raises(prosody.ProsodyError, match="series 4 is not approved"):
        prosody.prepare(settings, REPO / prosody.DECISION_DOC, series=4)
    assert not any(p.name.startswith("prosody-measure-1") for p in settings.data_root.iterdir())
    with pytest.raises(SystemExit):
        prosody.main(["prepare", "--series", "4"])


def test_the_cli_passes_the_series_to_the_step(tmp_path, monkeypatch, capsys) -> None:
    seen = []
    monkeypatch.setattr(prosody.config, "load", lambda: make_config(tmp_path))
    monkeypatch.setattr(prosody.guard, "install_network_guard", lambda _: True)
    monkeypatch.setattr(
        prosody, "run_sanity", lambda c, d, series: seen.append(series) or {"passed": True}
    )
    assert prosody.main(["sanity", "--series", "2"]) == 0
    assert seen == [2] and json.loads(capsys.readouterr().out) == {"passed": True}


def test_series_2_has_a_180_min_ceiling_and_series_1_keeps_90() -> None:
    assert prosody.main_wall_seconds(1) == prosody.MAIN_WALL_SECONDS == 90 * 60
    assert prosody.main_wall_seconds(2) == 180 * 60
    assert prosody.SERIES[2]["ceiling_decision"].endswith("#issuecomment-7062644385")
    assert prosody.MAX_SANITY_ATTEMPTS == 2
    assert not prosody.SERIES[1].get("sanity_after_pass")


def series_2_passed(tmp_path, monkeypatch, variant_raised=True):
    """Series 2 prepared, with a passed attempt 1 made by code of another SHA-256."""
    settings, words = sanity_setup(tmp_path, monkeypatch, variant_raised)
    monkeypatch.setitem(prosody.SERIES, 2, dict(prosody.SERIES[2], excluded=()))
    decision = REPO / prosody.DECISION_DOC
    prosody.prepare(settings, decision, series=2)
    good = gate0_clips(tmp_path / "good")[1]
    current = prosody.code_sha256()
    monkeypatch.setattr(prosody, "code_sha256", lambda: "a" * 64)
    first = prosody.run_sanity(settings, decision, fake_transcriber(good), series=2)
    assert first["passed"] and first["main_wall_seconds"] == 180 * 60
    monkeypatch.setattr(prosody, "code_sha256", lambda: current)
    return settings, decision, words


@needs_ffmpeg
def test_series_2_makes_attempt_2_with_new_code_after_a_pass_and_runs_main_with_it(
    tmp_path, monkeypatch
) -> None:
    settings, decision, words = series_2_passed(tmp_path, monkeypatch)
    with pytest.raises(prosody.ProsodyError, match="sanity"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False, series=2)
    second = prosody.run_sanity(settings, decision, fake_transcriber(words), series=2)
    assert second["attempt"] == 2 and second["passed"]
    assert second["code_sha256"] == prosody.code_sha256()
    assert second["main_wall_seconds"] == 180 * 60
    assert second["ceiling_decision"] == prosody.SERIES[2]["ceiling_decision"]
    with pytest.raises(prosody.ProsodyError, match="of this code has passed already"):
        prosody.run_sanity(settings, decision, fake_transcriber(words), series=2)
    result = prosody.run_main(settings, decision, fake_transcriber({}), scope=False, series=2)
    assert result["verdict"] == prosody.PASS and result["main_wall_seconds"] == 180 * 60
    out = settings.data_root / prosody.SERIES[2]["dir"]
    started = json.loads((out / prosody.MAIN_DIR / prosody.STARTED_FILE).read_text())
    assert started["main_wall_seconds"] == 180 * 60


@needs_ffmpeg
def test_series_2_attempt_2_after_a_pass_refuses_the_same_code(tmp_path, monkeypatch) -> None:
    settings, decision, words = series_2_passed(tmp_path, monkeypatch)
    monkeypatch.setattr(prosody, "code_sha256", lambda: "a" * 64)
    with pytest.raises(prosody.ProsodyError, match="of this code has passed already"):
        prosody.run_sanity(settings, decision, fake_transcriber(words), series=2)
    assert not (settings.data_root / prosody.SERIES[2]["dir"] / "sanity" / "work-2").exists()


@needs_ffmpeg
def test_series_2_failed_attempt_2_after_a_pass_gives_invalid(tmp_path, monkeypatch) -> None:
    settings, decision, words = series_2_passed(tmp_path, monkeypatch, variant_raised=False)
    second = prosody.run_sanity(settings, decision, fake_transcriber(words), series=2)
    assert second["attempt"] == 2 and not second["passed"]
    out = settings.data_root / prosody.SERIES[2]["dir"]
    result = json.loads((out / prosody.RESULT_FILE).read_text())
    assert result["verdict"] == prosody.INVALID and "2 попытки" in result["reason"]
    assert result["main_wall_seconds"] == 180 * 60
    with pytest.raises(prosody.ProsodyError, match="verdict is out"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False, series=2)


@needs_ffmpeg
def test_series_1_refuses_a_second_attempt_after_a_pass_even_with_new_code(
    tmp_path, monkeypatch
) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    prosody.prepare(settings, decision)
    first = prosody.run_sanity(settings, decision, fake_transcriber(words))
    assert first["passed"] and first["main_wall_seconds"] == 90 * 60
    assert "ceiling_decision" not in first
    monkeypatch.setattr(prosody, "code_sha256", lambda: "f" * 64)
    with pytest.raises(prosody.ProsodyError, match="passed already"):
        prosody.run_sanity(settings, decision, fake_transcriber(words))


@needs_ffmpeg
@pytest.mark.parametrize(("series", "done", "verdict"), [(1, 3, "INCONCLUSIVE"), (2, 4, "PASS")])
def test_the_main_run_stops_at_the_ceiling_of_its_series(
    tmp_path, monkeypatch, series, done, verdict
) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    monkeypatch.setitem(prosody.SERIES, 2, dict(prosody.SERIES[2], excluded=()))
    decision = REPO / prosody.DECISION_DOC
    prosody.prepare(settings, decision, series=series)
    assert prosody.run_sanity(settings, decision, fake_transcriber(words), series=series)["passed"]
    slow, clock = fake_transcriber({}), [0.0]

    def late(samples, run_dir):
        clock[0] += 2500.0  # four utterances: 10 000 s, past 90 min and inside 180 min
        return slow(samples, run_dir)

    monkeypatch.setattr(prosody.time, "monotonic", lambda: clock[0])
    result = prosody.run_main(settings, decision, late, scope=False, series=series)
    assert result["verdict"] == verdict, result
    assert result["main_wall_seconds"] == prosody.main_wall_seconds(series)
    if series == 1:
        assert result["utterances_done"] == done and "90 мин" in result["reason"]
    else:
        assert result["utterances"] == done


def test_series_2_report_names_the_verdict_the_numbers_and_holds_no_private_paths() -> None:
    name = "docs/eval/2026-11-21-prosody-measure-1-series-2.md"
    report = (REPO / name).read_text(encoding="utf-8")
    assert "**Вердикт: INCONCLUSIVE.** Coverage 0,488 меньше порога 0,70" in report
    assert "AUC 0,762, 95 % CI 0,742–0,778" in report
    assert prosody.RULE_SHA256 in report and prosody.RINCO_SHA256 in report
    assert "23a224baf9e3c07e7bcd4b03d6c1eec41a200eee65aa0d422a3bd61b339ce3c7" in report
    assert "docs/eval/2026-11-21-prosody-measure-1.md" in report
    assert not re.search(r"/(home|Users)/|~/|[A-Za-z]:\\", report)
    decision = (REPO / prosody.DECISION_DOC).read_text(encoding="utf-8")
    assert name in decision


# --- series 3: the EAF rules E1 and E2 -----------------------------------------------------

QUESTIONS = "INTON-EAF/1-Questions"


def add_rule_files(archive_path: Path) -> None:
    """A whole-second grid EAF, a WAV with two EAF files and a one-annotation whole-second EAF."""
    signal = words_signal(3, raised=2)[0]
    grid = [(0.0, 1.0, ACCENT), (1.0, 2.0, ACCENT)]
    aligned = [(0.0, 1.13, ACCENT), (1.13, 2.207, ACCENT)]
    with zipfile.ZipFile(archive_path, "a") as archive:
        for stem in ("g-0", "d-0", "s-0"):
            archive.writestr(f"{QUESTIONS}/{stem}.wav", wav_bytes(signal))
        archive.writestr(f"{QUESTIONS}/g-0.eaf", eaf("g-0.wav", {"Accent": grid}))
        archive.writestr(f"{QUESTIONS}/d-0.eaf", eaf("d-0.wav", {"Accent": aligned}))
        archive.writestr(f"{QUESTIONS}/d-0--v1.eaf", eaf("d-0.wav", {"Accent": aligned}))
        archive.writestr(f"{QUESTIONS}/s-0.eaf", eaf("s-0.wav", {"Accent": [(0.0, 2.0, ACCENT)]}))


def test_rules_e1_and_e2_read_only_the_pairing_and_the_annotation_times(tmp_path: Path) -> None:
    path = rinco_archive(tmp_path / "r.zip")
    add_rule_files(path)
    with zipfile.ZipFile(path) as archive:
        files, _ = corpus.rinco_files(archive)
    assert prosody.rule_exclusions(files) == {
        f"{QUESTIONS}/d-0--v1.eaf": prosody.RULE_E2,
        f"{QUESTIONS}/d-0.eaf": prosody.RULE_E2,
        f"{QUESTIONS}/d-0.wav": prosody.RULE_E2,
        f"{QUESTIONS}/g-0.eaf": prosody.RULE_E1,
    }
    by_eaf = {f.eaf_member: f for f in files}
    assert prosody.without_time_alignment(by_eaf[f"{QUESTIONS}/g-0.eaf"])
    assert not prosody.without_time_alignment(by_eaf[f"{QUESTIONS}/s-0.eaf"])
    assert not prosody.without_time_alignment(by_eaf[f"{QUESTIONS}/x-0.eaf"])
    assert prosody.rule_exclusions([by_eaf[f"{QUESTIONS}/x-1.eaf"]]) == {}


def test_material_leaves_out_an_excluded_eaf_or_wav_member(tmp_path: Path) -> None:
    path = rinco_archive(tmp_path / "r.zip")
    add_rule_files(path)
    excluded = [f"{QUESTIONS}/d-0.wav", f"{QUESTIONS}/g-0.eaf", f"{QUESTIONS}/x-0.wav"]
    with zipfile.ZipFile(path) as archive:
        files, counts = prosody.material(archive, excluded)
        assert sorted(f.eaf_member for f in files) == [
            f"{QUESTIONS}/s-0.eaf",
            f"{QUESTIONS}/x-1.eaf",
            f"{QUESTIONS}/x-2.eaf",
        ]
        assert counts["paired_files"] == 7 and counts["excluded_files"] == 4
        with pytest.raises(prosody.ProsodyError, match="does not pair"):
            prosody.material(archive, [f"{QUESTIONS}/none.eaf"])


def test_series_3_has_its_own_dir_the_limits_of_series_2_and_names_both_data_changes() -> None:
    third = prosody.SERIES[3]
    dirs = {prosody.SERIES[n]["dir"] for n in (1, 2, 3)}
    assert third["dir"] == "prosody-measure-1-series-3" and len(dirs) == 3
    assert prosody.main_wall_seconds(3) == 180 * 60
    assert third["owner_decision"].endswith("/issues/142#issuecomment-7064744998")
    assert third["eaf_rules"] and not third.get("sanity_after_pass")
    assert third["excluded"] == prosody.SERIES[2]["excluded"]
    asr_rule, eaf_rules = third["data_changes"]
    assert asr_rule == prosody.SERIES[2]["data_changes"][0]
    assert eaf_rules["issue"].endswith("/issues/142")
    assert re.fullmatch(r"[0-9a-f]{40}", eaf_rules["commit"])
    assert prosody.MAX_SANITY_ATTEMPTS == 2
    assert prosody.RULE_SHA256.startswith("b6ce9fcf") and prosody.RULE_SHA256.endswith("28cc65")
    assert not prosody.SERIES[1].get("eaf_rules") and not prosody.SERIES[2].get("eaf_rules")


@needs_ffmpeg
def test_series_3_prepare_records_the_rule_exclusions_and_the_main_run_uses_them(
    tmp_path, monkeypatch
) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    add_rule_files(prosody.rinco_path(settings))
    monkeypatch.setitem(prosody.SERIES, 3, dict(prosody.SERIES[3], excluded=()))
    decision = REPO / prosody.DECISION_DOC
    record = prosody.prepare(settings, decision, series=3)
    rules = prosody.rule_exclusions(
        corpus.rinco_files(zipfile.ZipFile(prosody.rinco_path(settings)))[0]
    )
    members = sorted([*rules, f"{QUESTIONS}/x-0.wav"])
    assert len(members) == 5 and rules[f"{QUESTIONS}/g-0.eaf"] == prosody.RULE_E1
    assert record["exclusion"] == {
        "members": members,
        "count": 5,
        "sha256": prosody.list_sha256(members),
        "rules": rules,
    }
    assert record["material"]["files"] == 3 and record["material"]["excluded_files"] == 4
    assert record["data_changes"] == list(prosody.SERIES[3]["data_changes"])
    assert prosody.run_sanity(settings, decision, fake_transcriber(words), series=3)["passed"]
    calls = []
    result = prosody.run_main(
        settings, decision, fake_transcriber({}, calls), scope=False, series=3
    )
    assert len(calls) == 5  # x-1 and x-2 with 2 utterances each, s-0 with one
    assert result["exclusion_sha256"] == prosody.list_sha256(members)
    assert result["main_wall_seconds"] == 180 * 60
    out = settings.data_root / prosody.SERIES[3]["dir"]
    assert (out / prosody.RESULT_FILE).exists()
    assert not (settings.data_root / prosody.SERIES[2]["dir"]).exists()
    assert not (settings.data_root / prosody.SERIES[1]["dir"]).exists()


@needs_ffmpeg
def test_series_3_main_run_refuses_when_the_rule_exclusions_changed_since_prepare(
    tmp_path, monkeypatch
) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    add_rule_files(prosody.rinco_path(settings))
    monkeypatch.setitem(prosody.SERIES, 3, dict(prosody.SERIES[3], excluded=()))
    decision = REPO / prosody.DECISION_DOC
    prosody.prepare(settings, decision, series=3)
    assert prosody.run_sanity(settings, decision, fake_transcriber(words), series=3)["passed"]
    monkeypatch.setattr(prosody, "rule_exclusions", lambda files: {})
    with pytest.raises(prosody.ProsodyError, match="exclusion list changed"):
        prosody.run_main(settings, decision, fake_transcriber({}), scope=False, series=3)
    assert not (settings.data_root / prosody.SERIES[3]["dir"] / prosody.MAIN_DIR).exists()


@needs_ffmpeg
def test_series_3_main_run_with_a_broken_rinco_archive_is_invalid_before_reading_it(
    tmp_path, monkeypatch
) -> None:
    settings, words = sanity_setup(tmp_path, monkeypatch)
    decision = REPO / prosody.DECISION_DOC
    monkeypatch.setitem(prosody.SERIES, 3, dict(prosody.SERIES[3], excluded=()))
    prosody.prepare(settings, decision, series=3)
    assert prosody.run_sanity(settings, decision, fake_transcriber(words), series=3)["passed"]
    prosody.rinco_path(settings).write_bytes(b"not a zip")
    monkeypatch.setattr(prosody, "rinco_problems", lambda _: ["RINCO SHA-256 differs"])
    result = prosody.run_main(settings, decision, fake_transcriber({}), scope=False, series=3)
    assert result["verdict"] == prosody.INVALID and result["reason"].startswith("Q1: RINCO")
    assert not (settings.data_root / prosody.SERIES[3]["dir"] / prosody.MAIN_DIR).exists()


def test_rinco_problems_name_a_wrong_archive(tmp_path: Path) -> None:
    settings = make_config(tmp_path)
    archive = prosody.rinco_path(settings)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"not the RINCO archive")
    assert prosody.rinco_problems(settings)


def test_series_3_report_names_the_verdict_the_post_hoc_caveat_and_holds_no_private_paths() -> None:
    name = "docs/eval/2026-11-21-prosody-measure-1-series-3.md"
    report = (REPO / name).read_text(encoding="utf-8")
    assert "**Вердикт: PASS, post hoc.**" in report
    assert "AUC 0,761, 95 % CI 0,737–0,783, coverage 0,807" in report
    flat = " ".join(report.split())
    assert "более слабое свидетельство" in flat and "не независимое подтверждение" in flat
    assert "PASS сам по себе не добавляет `word_prominence`" in flat
    assert "только бутстреп-интервал" in flat
    assert "AUC сопоставленной части правилами не задаётся" not in flat
    assert prosody.RULE_SHA256 in report and prosody.RINCO_SHA256 in report
    assert "8fa0199c114129685ff3e7fb870027ac8086843cc140868e5180e393efe7cd38" in report
    assert not re.search(r"/(home|Users)/|~/|[A-Za-z]:\\", report)
    decision = (REPO / prosody.DECISION_DOC).read_text(encoding="utf-8")
    assert name in decision and "**Серия 3 (2026-11-21): PASS, post hoc**" in decision


def test_series_2_report_names_memory_scope_only_in_the_files_that_hold_it() -> None:
    report = (REPO / "docs/eval/2026-11-21-prosody-measure-1-series-2.md").read_text(
        encoding="utf-8"
    )
    flat = " ".join(report.split())
    assert "В `attempt-2.json` и `main/started.json`: `memory_scope` = [`8000000000`, `0`]." in flat
    holders = re.findall(r"В ([^:]*): `memory_scope`", flat)
    assert holders and all("result.json" not in h for h in holders)
    assert "В `attempt-2.json`, `main/started.json` и `result.json`: `cpus` = 0–7" in flat
