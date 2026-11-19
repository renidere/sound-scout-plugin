"""Synthetic diction defects for the draft pilot corpus: Piper `denis` with known errors (#81).

`python -m sound_scout.eval.synthetic add --corpus-dir DIR --manifest DIR/<name>/manifest.json`
needs the `corpus-tts` extra (`uv run --extra corpus-tts ...`). It re-hashes the three files of
the Piper voice against `corpus.EXTRA_SOURCES`, loads the voice on the CPU with in-memory
alignment output (the `onnx` package; the model file is not changed) and synthesizes the 16
sentences of `SENTENCES`, one defect per clip: a lexical stress error (the stress mark `ˈ` moved
and the vowels of the stressed syllable changed) or a word distortion (a changed phoneme string).
The defect is put in at the espeak-ng phoneme level, so the interval of the target word comes
from the alignment Piper returns (samples per phoneme id), never from ASR and never estimated:
without alignment the command refuses to work.

The clips join the draft manifest as `synthetic: true`, category `pronunciation` — by
construction, not confirmed by labels. The defect file is `synthetic/<clip id>.json`, never in
`labels/`, and it is no human label (`not_a_human_label: true`; the label schema refuses it);
`labels/<clip id>.json` is the empty template a person fills by ear. No ASR and no other model
runs on these clips: the only model is the TTS that makes them. The rules and the word table are
docs/decisions/0012-synthetic-diction-and-openstt.md.
"""

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sound_scout import config, guard, proc
from sound_scout.eval import corpus, gate0

FORMAT_VERSION = 1
KIND = "synthetic_defect"
PROVENANCE_KIND = "synthetic_tts"
SYNTHETIC_DIR = "synthetic"
ID_PREFIX = "syn-"
SYNTHETIC_HELDOUT_GROUPS = 10
BACKUP_SUFFIX = ".bak-81"
VOICE_DATASET = "piper-denis"
VOICE_MODEL = "ru_RU-denis-medium.onnx"
VOICE_CONFIG = "ru_RU-denis-medium.onnx.json"
RUNTIME = {"package": "piper-tts", "version": "1.8.0", "alignment": "onnx (в памяти)"}
INTERVAL_METHOD = "piper_alignment"
PUNCTUATION = frozenset(",.?!;:")
WORD_SEPARATOR = " "
STRESS = "ˈ"
VOWELS = frozenset("aeiouyɑɐɛɪʊʌəɨɔæɵʉ")
BOS, EOS, PAD = "^", "$", "_"
RESAMPLE_TIMEOUT_S = 60.0
ISSUE = "https://github.com/renidere/sound-scout-plugin/issues/81"
DECISION = "docs/decisions/0012-synthetic-diction-and-openstt.md"


class SyntheticError(Exception):
    """The synthesis or the corpus change cannot go on; the message names the reason."""


@dataclass(frozen=True)
class Sentence:
    """One clip: the text, the target word, its espeak-ng phonemes as said right and as spoken."""

    id: str
    text: str
    target_index: int
    defect: str
    subtype: str
    correct_ipa: str
    spoken_ipa: str
    note: str

    @property
    def words(self) -> list[str]:
        return self.text.split()

    @property
    def word(self) -> str:
        return self.words[self.target_index].strip("".join(PUNCTUATION))


def _s(id_, text, index, defect, subtype, correct, spoken, note) -> Sentence:
    return Sentence(id_, text, index, defect, subtype, correct, spoken, note)


SENTENCES: tuple[Sentence, ...] = (
    _s("stress-zamok", "На двери висит старый замок и никто не может его открыть.", 4,
       "lexical_stress", "homograph", "zamˈok", "zˈɑmʌk", "замОк → зАмок"),
    _s("stress-muka", "Для него это была настоящая мука и долгое ожидание ответа.", 5,
       "lexical_stress", "homograph", "mˈuka", "mukˈɑ", "мУка → мукА"),
    _s("stress-atlas", "На полке лежит старый атлас мира и несколько книг.", 4,
       "lexical_stress", "homograph", "ˈɑtɭʌs", "atɭˈɑs", "Атлас → атлАс"),
    _s("stress-khlopok", "Этот южный регион выращивает хлопок и пшеницу уже много лет.", 4,
       "lexical_stress", "homograph", "xɭʌpˈok", "xɭˈopʌk", "хлопОк → хлОпок"),
    _s("stress-zvonit", "Каждое утро мне звонит старый друг из другого города.", 3,
       "lexical_stress", "orthoepic", "zvʌnʲˈit", "zvˈonʲit", "звонИт → звОнит"),
    _s("stress-torty", "На праздник мы купили торты и свежие фрукты для гостей.", 4,
       "lexical_stress", "orthoepic", "tˈorty", "tʌrtˈy", "тОрты → тортЫ"),
    _s("stress-katalog", "В магазине можно взять каталог новых товаров на этот сезон.", 4,
       "lexical_stress", "orthoepic", "kʌtaɭˈok", "kʌtˈɑɭʌk", "каталОг → катАлог"),
    _s("stress-dokument", "Секретарь отправил документ в отдел кадров еще вчера вечером.", 2,
       "lexical_stress", "orthoepic", "dʌkumʲˈent", "dʌkˈumʲint", "докумЕнт → докУмент"),
    _s("drop-biblioteka", "Наша городская библиотека открыта каждый день до семи часов.", 2,
       "word_distortion", "syllable_drop", "bʲibɭʲiʌtʲˈeka", "bʲibɭʲitʲˈeka",
       "библиотека → библитека"),
    _s("drop-soprovozhdenie", "Для поездки нужно сопровождение опытного гида и хорошая карта.", 3,
       "word_distortion", "syllable_drop", "sʌprʌvʌʒdʲˈenʲijɪ", "sʌprʌvˈoʒnʲijɪ",
       "сопровождение → сопровожние"),
    _s("drop-velosiped", "Летом мой брат купил новый велосипед и часто катается по парку.", 5,
       "word_distortion", "syllable_drop", "vʲiɭʌsʲipʲˈet", "vʲiɭsʲipʲˈet",
       "велосипед → велсипед"),
    _s("slur-interesnogo",
       "Сегодня на лекции было много интересного и полезного для студентов.", 5,
       "word_distortion", "ending_slur", "ɪntʲirʲˈesnʌvʌ", "ɪntʲirʲˈesn",
       "интересного → интересн"),
    _s("slur-specialistov",
       "Компания пригласила несколько опытных специалистов для работы над проектом.", 4,
       "word_distortion", "ending_slur", "spʲitsyaɭʲˈistʌv", "spʲitsyaɭʲˈis",
       "специалистов → специалис"),
    _s("slur-zamechatelnymi",
       "Соседи всегда были замечательными и добрыми друзьями нашей семьи.", 3,
       "word_distortion", "ending_slur", "zʌmʲitʃʲˈɑtʲiɭnymʲɪ", "zʌmʲitʃʲˈɑtʲiɭn",
       "замечательными → замечательн"),
    _s("subst-ryba", "Вчера на рынке свежая рыба продавалась по хорошей цене.", 4,
       "word_distortion", "consonant_substitution", "rˈyba", "ɭˈyba", "рыба → лыба (р → л)"),
    _s("subst-shuba", "Зимой эта теплая шуба спасает от сильного мороза и ветра.", 3,
       "word_distortion", "consonant_substitution", "ʃˈuba", "sˈuba", "шуба → суба (ш → с)"),
)  # fmt: skip


@dataclass(frozen=True)
class Synthesis:
    samples: np.ndarray
    sample_rate: int
    start_s: float
    end_s: float
    phonemes: list[str]


def voice_sources() -> tuple[corpus.Source, ...]:
    return tuple(s for s in corpus.EXTRA_SOURCES if s.dataset == VOICE_DATASET)


def check_voice(corpus_dir: Path) -> None:
    """Every file of the voice matches its pin, or `SyntheticError` names the mismatch."""
    problems = [p for source in voice_sources() for p in corpus.check_source(corpus_dir, source)]
    if problems:
        raise SyntheticError("the Piper voice does not match its pins: " + "; ".join(problems))


def load_voice(corpus_dir: Path):
    """The pinned Piper voice on the CPU with alignment output; piper is imported only here."""
    check_voice(corpus_dir)
    directory = Path(corpus_dir) / "sources" / VOICE_DATASET
    try:
        from piper import PiperVoice
    except ImportError as error:
        raise SyntheticError(
            "piper-tts is not installed: run with `uv run --extra corpus-tts`"
        ) from error
    return PiperVoice.load(
        directory / VOICE_MODEL,
        config_path=directory / VOICE_CONFIG,
        use_cuda=False,
        include_alignments=True,
    )


def word_spans(phonemes: list[str]) -> list[tuple[int, int]]:
    """(first, last) index of each word's phonemes, punctuation excluded, split on the space."""
    spans, current = [], []
    for index, phoneme in enumerate([*phonemes, WORD_SEPARATOR]):
        if phoneme == WORD_SEPARATOR:
            if current:
                spans.append((current[0], current[-1]))
            current = []
        elif phoneme not in PUNCTUATION:
            current.append(index)
    return spans


def phoneme_samples(phonemes: list[str], ids: list[int], id_samples, id_map) -> list[int]:
    """Samples of BOS, of each phoneme and of EOS, the way `PiperVoice.synthesize` 1.8.0 adds
    them up: every phoneme owns its ids and the PAD after them; EOS owns its ids only."""
    counts, position = [], 0
    for phoneme in itertools.chain([BOS], phonemes, [EOS]):
        expected = list(id_map[phoneme]) + (list(id_map[PAD]) if phoneme != EOS else [])
        if list(ids[position : position + len(expected)]) != expected:
            raise SyntheticError("the phoneme ids do not follow the phonemes")
        counts.append(int(np.sum(id_samples[position : position + len(expected)])))
        position += len(expected)
    if position != len(ids):
        raise SyntheticError("the phoneme ids do not follow the phonemes")
    return counts


def synthesize(voice, sentence: Sentence) -> Synthesis:
    """One clip with the defect of `sentence` in its target word and that word's interval."""
    sentences = voice.phonemize(sentence.text)
    if len(sentences) != 1:
        raise SyntheticError(f"{sentence.id}: espeak gave {len(sentences)} sentences, not one")
    phonemes = list(sentences[0])
    spans = word_spans(phonemes)
    if len(spans) != len(sentence.words):
        raise SyntheticError(
            f"{sentence.id}: {len(spans)} phoneme words for {len(sentence.words)} words of text "
            "(espeak merged or split a word)"
        )
    first, last = spans[sentence.target_index]
    actual = "".join(phonemes[first : last + 1])
    if actual != sentence.correct_ipa:
        raise SyntheticError(
            f"{sentence.id}: espeak gives {actual!r} for the target word, the table "
            f"{sentence.correct_ipa!r}"
        )
    spoken = list(unicodedata.normalize("NFD", sentence.spoken_ipa))
    id_map = voice.config.phoneme_id_map
    unknown = sorted({p for p in spoken if p not in id_map})
    if unknown:
        raise SyntheticError(f"{sentence.id}: {unknown} are not in the voice's phoneme map")
    phonemes = phonemes[:first] + spoken + phonemes[last + 1 :]
    last = first + len(spoken) - 1
    ids = voice.phonemes_to_ids(phonemes)
    result = voice.phoneme_ids_to_audio(ids, _synthesis_config(), include_alignments=True)
    if not isinstance(result, tuple) or result[1] is None or len(result[1]) != len(ids):
        raise SyntheticError(
            f"{sentence.id}: alignment is not available (the onnx package is needed)"
        )
    audio, id_samples = np.asarray(result[0], dtype=np.float64).reshape(-1), result[1]
    counts = phoneme_samples(phonemes, ids, id_samples, id_map)
    if sum(counts) != len(audio):
        raise SyntheticError(
            f"{sentence.id}: the alignment covers {sum(counts)} samples, the audio {len(audio)}"
        )
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        raise SyntheticError(f"{sentence.id}: the voice returned silence")
    rate = voice.config.sample_rate
    start = sum(counts[: 1 + first])
    end = sum(counts[: 1 + last + 1])
    return Synthesis(audio / peak, rate, start / rate, end / rate, phonemes)


def _synthesis_config():
    from piper import SynthesisConfig

    return SynthesisConfig()


def resample(samples: np.ndarray, rate: int, work: Path) -> np.ndarray:
    """`samples` at `rate` → 16 kHz mono through the bounded ffmpeg; the time scale is kept."""
    raw = work / "input.f32"
    raw.write_bytes(np.asarray(samples, "<f4").tobytes())
    try:
        argv = proc.ffmpeg_argv(
            "-v", "error", "-threads", corpus.DECODE_THREADS, "-f", "f32le", "-ar", str(rate),
            "-ac", "1", "-i", str(raw), "-ac", "1", "-ar", str(corpus.SAMPLE_RATE),
            "-f", "f32le", "pipe:1",
        )  # fmt: skip
        limit = int(len(samples) / rate * corpus.SAMPLE_RATE * 4) + (1 << 16)
        result = proc.run(argv, timeout=RESAMPLE_TIMEOUT_S, max_output_bytes=limit)
    finally:
        raw.unlink()
    if result.timed_out or result.truncated or result.returncode != 0:
        reason = result.stderr.decode("utf-8", "replace").strip()[-300:]
        raise SyntheticError(f"ffmpeg could not resample: {reason or 'timed out or too long'}")
    return np.frombuffer(result.stdout, "<f4").astype(np.float64)


def add(corpus_dir: Path, manifest_path: Path, voice) -> list[tuple[dict, dict]]:
    """Synthesize every sentence, then add the clips to the draft manifest.

    Returns (manifest record, defect) per clip.
    Refuses, changing nothing, a manifest past its draft (no `corpus.DRAFT_NOTE`) or one that
    already holds `syn-*` records, and any sentence that does not synthesize as the table says.
    """
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus.DRAFT_NOTE not in (document.get("note") or ""):
        raise SyntheticError("the manifest is no draft (its note lacks corpus.DRAFT_NOTE)")
    if any(r["id"].startswith(ID_PREFIX) for r in document["records"]):
        raise SyntheticError(f"the manifest already holds {ID_PREFIX}* records")
    work = Path(tempfile.mkdtemp(prefix=".synthetic-81.", dir=corpus_dir))
    try:
        made = []
        for sentence in SENTENCES:
            synthesis = synthesize(voice, sentence)
            samples = resample(synthesis.samples, synthesis.sample_rate, work)
            made.append((sentence, synthesis, gate0.wav_bytes(gate0.pcm16(samples))))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    clip_ids = [ID_PREFIX + sentence.id for sentence, _, _ in made]
    taken = [
        path
        for clip_id in clip_ids
        for path in (f"clips/{clip_id}.wav", f"labels/{clip_id}.json", f"synthetic/{clip_id}.json")
        if os.path.lexists(root / path)
    ]
    if taken:
        raise SyntheticError(f"already there, nothing written: {', '.join(taken)}")
    split = corpus.split_by_group(clip_ids, SYNTHETIC_HELDOUT_GROUPS)
    voice_pins = [
        {"file": s.name, "sha256": s.sha256, "revision": corpus.PIPER_REVISION}
        for s in voice_sources()
    ]
    parameters = {
        "noise_scale": voice.config.noise_scale,
        "length_scale": voice.config.length_scale,
        "noise_w": voice.config.noise_w_scale,
        "normalize": "audio / max|audio| (как у Piper по умолчанию)",
        "source_sample_rate": voice.config.sample_rate,
        "sample_rate": corpus.SAMPLE_RATE,
        "resampler": "ffmpeg",
    }
    (root / SYNTHETIC_DIR).mkdir(mode=corpus.DIR_MODE, exist_ok=True)
    records, defects, provenance = [], [], []
    for (sentence, synthesis, data), clip_id in zip(made, clip_ids, strict=True):
        sha256 = hashlib.sha256(data).hexdigest()
        duration = (len(data) - 44) / 2 / corpus.SAMPLE_RATE
        defect = {
            "type": sentence.defect,
            "subtype": sentence.subtype,
            "word": sentence.word,
            "word_index": sentence.target_index,
            "correct_ipa": sentence.correct_ipa,
            "spoken_ipa": sentence.spoken_ipa,
            "start_s": round(synthesis.start_s, 3),
            "end_s": round(synthesis.end_s, 3),
            "interval_method": INTERVAL_METHOD,
            "note": sentence.note,
        }
        corpus._write_bytes(root / "clips" / f"{clip_id}.wav", data)
        corpus._write_json(
            root / SYNTHETIC_DIR / f"{clip_id}.json",
            {
                "format_version": FORMAT_VERSION,
                "kind": KIND,
                "not_a_human_label": True,
                "clip_id": clip_id,
                "clip_sha256": sha256,
                "text": sentence.text,
                "defects": [defect],
                "voice": voice_pins,
                "runtime": RUNTIME,
                "synthesis": parameters,
            },
        )
        corpus._write_json(
            root / "labels" / f"{clip_id}.json", corpus.label_template(clip_id, sha256, duration)
        )
        records.append({
            "id": clip_id, "path": f"clips/{clip_id}.wav", "sha256": sha256,
            "duration_s": duration, "split": split[clip_id], "source_group": clip_id,
            "categories": ["pronunciation"], "negative_control": False, "synthetic": True,
            "note": f"синтетика Piper denis: {sentence.defect}/{sentence.subtype}, слово "
                    f"«{sentence.word}», {defect['start_s']:.3f}–{defect['end_s']:.3f} с; "
                    "категория — по построению, не утверждена разметкой",
        })  # fmt: skip
        defects.append(defect)
        provenance.append({
            "id": clip_id, "sha256": sha256, "kind": PROVENANCE_KIND, "issue": ISSUE,
            "decision": DECISION, "text": sentence.text, "defects": [defect],
            "voice": voice_pins, "runtime": RUNTIME, "synthesis": parameters,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })  # fmt: skip
    provenance_path = root / "provenance.json"
    if provenance_path.exists():
        whole = json.loads(provenance_path.read_text(encoding="utf-8"))
        whole["clips"] = [*whole.get("clips", []), *provenance]
        corpus._replace_json(provenance_path, whole)
    backup = manifest_path.with_name(manifest_path.name + BACKUP_SUFFIX)
    if not os.path.lexists(backup):
        shutil.copyfile(manifest_path, backup)
    corpus._replace_json(manifest_path, {**document, "records": document["records"] + records})
    return list(zip(records, defects, strict=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.synthetic")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("add", help="synthesize the defect clips into a draft corpus")
    command.add_argument("--corpus-dir", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        guard.install_network_guard(config.load())
        voice = load_voice(args.corpus_dir)
        added = add(args.corpus_dir, args.manifest, voice)
    except (SyntheticError, corpus.CorpusError, config.ConfigError, OSError, ValueError) as error:
        print(f"synthetic: {error}", file=sys.stderr)
        return 2
    for record, defect in added:
        print(json.dumps({"id": record["id"], "split": record["split"],
                          "start_s": defect["start_s"], "end_s": defect["end_s"]}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
