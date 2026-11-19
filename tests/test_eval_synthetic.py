"""Synthetic diction defects (#81): the defect, its interval from the alignment, the corpus rows.

The voice here is a fake with the Piper interface: each phoneme id is a known number of samples,
so the interval of the target word is known before the synthesis. These tests prove the contract
(where the interval comes from, what enters the manifest, what never becomes a label), not that
the defect is audible: only a person listening to the real clips can say that.
"""

import hashlib
import json
import os
import shutil
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from sound_scout.config import Config
from sound_scout.eval import corpus, labels, manifest, synthetic

REPO_ROOT = Path(__file__).resolve().parent.parent
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
RATE = 22_050
FILLER = "tʌ"
QUIET = frozenset({"t", "ʌ", " ", ".", ",", "^", "$", "_"})


def symbols() -> list[str]:
    found = set(FILLER) | set(" .,^$_") | {"m", "u", "k", "ˈ", "ɑ"}
    for sentence in synthetic.SENTENCES:
        found |= set(unicodedata.normalize("NFD", sentence.correct_ipa + sentence.spoken_ipa))
    return sorted(found)


@dataclass
class FakeConfig:
    phoneme_id_map: dict
    sample_rate: int = RATE
    noise_scale: float = 0.5
    length_scale: float = 1.0
    noise_w_scale: float = 0.333


@dataclass
class FakeVoice:
    """Piper's interface: espeak words → phonemes, phonemes → ids, ids → audio + alignment."""

    alignment: bool = True
    merge_words: bool = False
    drop_symbol: str | None = None
    calls: list = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = {symbol: [i] for i, symbol in enumerate(symbols())}
        if self.drop_symbol:
            ids.pop(self.drop_symbol)
        self.config = FakeConfig(ids)
        self.correct = {s.word: s.correct_ipa for s in synthetic.SENTENCES}

    def phonemize(self, text: str) -> list[list[str]]:
        words = []
        for word in text.split():
            bare = word.strip(".,")
            words.append(self.correct.get(bare, FILLER) + word[len(bare) :])
        if self.merge_words:
            words[0:2] = [words[0] + words[1]]
        return [list(unicodedata.normalize("NFD", " ".join(words)))]

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        m = self.config.phoneme_id_map
        ids = m["^"] + m["_"]
        for phoneme in phonemes:
            ids += m[phoneme] + m["_"]
        return ids + m["$"]

    def samples_of(self, phoneme_id: int) -> int:
        return 256 * (1 + phoneme_id % 4)

    def phoneme_ids_to_audio(self, ids, syn_config, include_alignments=False):
        by_id = {v[0]: k for k, v in self.config.phoneme_id_map.items()}
        pieces, loud = [], False
        for phoneme_id in ids:
            symbol = by_id[phoneme_id]
            loud = (symbol not in QUIET) or (symbol == "_" and loud)
            n = self.samples_of(phoneme_id)
            t = np.arange(n) / RATE
            pieces.append(0.5 * np.sin(2 * np.pi * 300 * t) if loud else np.zeros(n))
        audio = np.concatenate(pieces).astype(np.float32)
        self.calls.append(list(ids))
        if not include_alignments:
            return audio
        if not self.alignment:
            return audio, None
        return audio, np.array([self.samples_of(i) for i in ids], dtype=np.int64)


def probe_sentence(**changes) -> synthetic.Sentence:
    base = synthetic.Sentence(
        "probe", "Раз два три мука четыре пять шесть.", 3, "lexical_stress", "homograph",
        "mˈuka", "mukˈɑ", "мУка → мукА",
    )  # fmt: skip
    return replace(base, **changes)


@pytest.fixture(autouse=True)
def piper_free_config(monkeypatch) -> None:
    monkeypatch.setattr(synthetic, "_synthesis_config", lambda: None)


# --- one clip -----------------------------------------------------------------------------------


def test_the_interval_is_the_samples_before_and_through_the_target_word() -> None:
    voice = FakeVoice()
    result = synthetic.synthesize(voice, probe_sentence())
    ids = voice.calls[-1]
    samples = [voice.samples_of(i) for i in ids]
    target = 2 + 2 * len("tʌ tʌ tʌ ")  # BOS, PAD, then every phoneme with its PAD
    spoken = len(unicodedata.normalize("NFD", "mukˈɑ"))
    assert result.start_s == pytest.approx(sum(samples[:target]) / RATE)
    assert result.end_s == pytest.approx(sum(samples[: target + 2 * spoken]) / RATE)
    assert result.sample_rate == RATE
    assert np.max(np.abs(result.samples)) == pytest.approx(1.0)


@needs_ffmpeg
def test_the_interval_holds_within_10_ms_after_resampling_to_16_khz(tmp_path: Path) -> None:
    result = synthetic.synthesize(FakeVoice(), probe_sentence())
    resampled = synthetic.resample(result.samples, result.sample_rate, tmp_path)
    loud = np.flatnonzero(np.abs(resampled) > 0.05)
    assert loud[0] / corpus.SAMPLE_RATE == pytest.approx(result.start_s, abs=0.01)
    assert (loud[-1] + 1) / corpus.SAMPLE_RATE == pytest.approx(result.end_s, abs=0.01)
    assert len(resampled) / corpus.SAMPLE_RATE == pytest.approx(len(result.samples) / RATE, 0.001)


def test_a_stress_error_changes_only_the_target_word() -> None:
    voice = FakeVoice()
    sentence = synthetic.SENTENCES[0]
    original = voice.phonemize(sentence.text)[0]
    result = synthetic.synthesize(voice, sentence)
    before, after = "".join(original).split(" "), "".join(result.phonemes).split(" ")
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert changed == [sentence.target_index]
    assert after[sentence.target_index].rstrip(".,") == sentence.spoken_ipa


@pytest.mark.parametrize(
    ("voice", "sentence", "reason"),
    [
        (FakeVoice(), probe_sentence(correct_ipa="mukˈɑ"), "espeak gives 'mˈuka'"),
        (FakeVoice(merge_words=True), probe_sentence(), "6 phoneme words for 7 words"),
        (FakeVoice(alignment=False), probe_sentence(), "alignment is not available"),
        (FakeVoice(), probe_sentence(spoken_ipa="mukˈɑ@"), r"\['@'\] are not in the voice"),
        (FakeVoice(drop_symbol="ɑ"), probe_sentence(), r"\['ɑ'\] are not in the voice"),
    ],
)
def test_the_synthesis_refuses_instead_of_guessing(voice, sentence, reason: str) -> None:
    with pytest.raises(synthetic.SyntheticError, match=reason):
        synthetic.synthesize(voice, sentence)


def test_audio_without_tuple_alignment_is_refused() -> None:
    voice = FakeVoice()
    voice.phoneme_ids_to_audio = lambda ids, cfg, include_alignments=False: np.ones(10)
    with pytest.raises(synthetic.SyntheticError, match="alignment is not available"):
        synthetic.synthesize(voice, probe_sentence())


def test_a_voice_off_its_pin_is_refused_before_it_loads(tmp_path: Path) -> None:
    with pytest.raises(synthetic.SyntheticError, match="does not match its pins"):
        synthetic.load_voice(tmp_path)
    directory = tmp_path / "sources" / synthetic.VOICE_DATASET
    directory.mkdir(parents=True)
    for source in synthetic.voice_sources():
        (directory / source.name).write_bytes(b"x" * source.size)
    with pytest.raises(synthetic.SyntheticError, match="publisher checksum"):
        synthetic.check_voice(tmp_path)


# --- the corpus ---------------------------------------------------------------------------------


@pytest.fixture
def draft(tmp_path: Path) -> Path:
    root = tmp_path / "pilot-x"
    for name in ("clips", "labels"):
        (root / name).mkdir(parents=True)
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "pilot-x",
                "note": corpus.MANIFEST_NOTE,
                "records": [],
            }  # fmt: skip
        ),
        encoding="utf-8",
    )
    (root / "provenance.json").write_text(json.dumps({"clips": []}), encoding="utf-8")
    return path


def records_of(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


@needs_ffmpeg
def test_add_marks_every_clip_synthetic_pronunciation_and_splits_by_group(
    tmp_path: Path, draft: Path
) -> None:
    added = synthetic.add(tmp_path, draft, FakeVoice())
    records = records_of(draft)
    assert [r for r, _ in added] == records and len(records) == 16
    assert all(r["synthetic"] is True for r in records)
    assert all(r["categories"] == ["pronunciation"] for r in records)
    assert not any(r["negative_control"] or "control_kind" in r for r in records)
    splits = [r["split"] for r in records]
    assert (splits.count("heldout"), splits.count("dev")) == (10, 6)
    by_group: dict[str, set] = {}
    for r in records:
        by_group.setdefault(r["source_group"], set()).add(r["split"])
    assert all(len(seen) == 1 for seen in by_group.values())
    for record in records:
        data = (draft.parent / record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    assert (draft.parent / f"manifest.json{synthetic.BACKUP_SUFFIX}").exists()
    provenance = json.loads((draft.parent / "provenance.json").read_text(encoding="utf-8"))
    assert [c["kind"] for c in provenance["clips"]] == [synthetic.PROVENANCE_KIND] * 16
    assert not list(tmp_path.glob(".synthetic-81.*"))


@needs_ffmpeg
def test_defect_files_are_no_labels_and_label_templates_stay_empty(
    tmp_path: Path, draft: Path
) -> None:
    synthetic.add(tmp_path, draft, FakeVoice())
    root = draft.parent
    for record in records_of(draft):
        template = root / "labels" / f"{record['id']}.json"
        content = json.loads(template.read_text(encoding="utf-8"))
        assert content["events"] == [] and content["annotator"] == ""
        assert not labels.validate(template).ok
        defect_path = root / synthetic.SYNTHETIC_DIR / f"{record['id']}.json"
        defect = json.loads(defect_path.read_text(encoding="utf-8"))
        assert defect["kind"] == "synthetic_defect" and defect["not_a_human_label"] is True
        assert defect["clip_sha256"] == record["sha256"]
        [item] = defect["defects"]
        assert item["type"] in ("lexical_stress", "word_distortion")
        assert 0 < item["start_s"] < item["end_s"] < record["duration_s"]
        assert item["interval_method"] == "piper_alignment"
        assert not labels.validate(defect_path).ok
    assert not list((root / "labels").glob("*synthetic*"))
    assert {p.name for p in (root / "labels").iterdir()} == {
        f"{r['id']}.json" for r in records_of(draft)
    }


@needs_ffmpeg
def test_the_draft_with_synthetic_clips_has_no_structural_error(
    tmp_path: Path, draft: Path
) -> None:
    synthetic.add(tmp_path, draft, FakeVoice())
    report = manifest.validate(draft)
    assert report.structure_ok, report.errors
    assert report.heldout_coverage["pronunciation"] == 10


@needs_ffmpeg
def test_a_second_add_and_a_fixed_manifest_are_refused_unchanged(
    tmp_path: Path, draft: Path
) -> None:
    synthetic.add(tmp_path, draft, FakeVoice())
    before = draft.read_bytes()
    with pytest.raises(synthetic.SyntheticError, match="already holds syn-"):
        synthetic.add(tmp_path, draft, FakeVoice())
    assert draft.read_bytes() == before
    fixed = json.loads(before)
    fixed["note"], fixed["records"] = "зафиксирован", []
    draft.write_text(json.dumps(fixed), encoding="utf-8")
    with pytest.raises(synthetic.SyntheticError, match="no draft"):
        synthetic.add(tmp_path, draft, FakeVoice())


def test_a_failing_sentence_writes_nothing(tmp_path: Path, draft: Path) -> None:
    before = draft.read_bytes()
    with pytest.raises(synthetic.SyntheticError, match="alignment is not available"):
        synthetic.add(tmp_path, draft, FakeVoice(alignment=False))
    assert draft.read_bytes() == before
    assert not list((draft.parent / "clips").iterdir())
    assert not (draft.parent / synthetic.SYNTHETIC_DIR).exists()


@needs_ffmpeg
def test_main_prints_only_id_split_and_interval(tmp_path, draft, monkeypatch, capsys) -> None:
    monkeypatch.setattr(synthetic.config, "load", lambda: Config((tmp_path,), tmp_path))
    monkeypatch.setattr(synthetic.guard, "install_network_guard", lambda config_: None)
    monkeypatch.setattr(synthetic, "load_voice", lambda corpus_dir: FakeVoice())
    code = synthetic.main(["add", "--corpus-dir", str(tmp_path), "--manifest", str(draft)])
    assert code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 16
    assert all(set(line) == {"id", "split", "start_s", "end_s"} for line in lines)
    assert synthetic.main(["add", "--corpus-dir", str(tmp_path), "--manifest", str(draft)]) == 2
    assert "already holds" in capsys.readouterr().err


# --- the table ----------------------------------------------------------------------------------


def vowel_before_stress(ipa: str) -> int:
    return sum(1 for ch in ipa[: ipa.index(synthetic.STRESS)] if ch in synthetic.VOWELS)


def test_the_sentence_table_holds_its_invariants() -> None:
    table = synthetic.SENTENCES
    assert len({s.id for s in table}) == 16
    kinds = [s.defect for s in table]
    assert (kinds.count("lexical_stress"), kinds.count("word_distortion")) == (8, 8)
    for s in table:
        assert 8 <= len(s.words) <= 14, s.id
        assert 2 <= s.target_index <= len(s.words) - 3, s.id
        assert s.spoken_ipa != s.correct_ipa, s.id
        assert s.word and s.word.isalpha(), s.id
        if s.defect == "lexical_stress":
            assert s.spoken_ipa.count(synthetic.STRESS) == 1, s.id
            assert vowel_before_stress(s.spoken_ipa) != vowel_before_stress(s.correct_ipa), s.id


@pytest.mark.slow
def test_the_real_piper_voice_aligns_one_sentence(monkeypatch) -> None:
    """Contract with the real voice: the alignment covers the audio. Not a proof of quality."""
    pytest.importorskip("onnx")
    piper = pytest.importorskip("piper")
    corpus_dir = Path(os.environ.get("SOUND_SCOUT_CORPUS_DIR", REPO_ROOT / "data" / "corpus"))
    if not (corpus_dir / synthetic.voice_sources()[0].relative_path).exists():
        pytest.skip("the pinned Piper voice is not in the corpus directory")
    monkeypatch.setattr(synthetic, "_synthesis_config", piper.SynthesisConfig)
    voice = synthetic.load_voice(corpus_dir)
    result = synthetic.synthesize(voice, synthetic.SENTENCES[0])
    assert 0 < result.start_s < result.end_s < len(result.samples) / result.sample_rate
