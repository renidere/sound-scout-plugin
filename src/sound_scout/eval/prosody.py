"""`prosody-measure-1`: the fixed word prominence index `PROSODY-RULE v1` against RINCO Accent.

The protocol is `docs/decisions/0019-next-audio-experiment.md`, revision 2, sections 4.1–4.8; the
rule is the block between its `prosody-rule` markers, and every command refuses to start unless
the SHA-256 of that block is `RULE_SHA256`, the value recorded before any run. Nothing here is
fitted to data: the features, weights, metric and thresholds are the rule's.

Three steps, each its own command, all in the directory of one series under `<data_root>/`
(outside git). A series is one owner-approved run of the rule (`SERIES`): the second one, after
the ASR trust rule of #135, sits in `prosody-measure-1-series-2/` and never reads the first; the
third one also leaves out what the EAF rules E1 and E2 of 0019 «Серия 3» name (`rule_exclusions`).

- `prepare` checks precondition Q1 (network guard, the ASR weights and the RINCO archive against
  the pins of `docs/dependencies.md`), lists the RINCO files of `pilot-1` (dev and held-out, from
  `corpus/pilot-1/provenance.json`), those the series names and, in series 3, those the EAF
  rules name (`series_exclusion`), that the experiment leaves out, with the SHA-256 of that
  list, and counts the material that stays. It writes `prepare.json` with the data changes of
  the series (`data_changes`).
- `sanity` is one attempt of the sanity stage S1/S2 on the Gate 0 clips, under the same memory
  scope and CPU pin as the main run; each attempt is recorded with its number, the SHA-256 of
  this file, the cgroup memory limits, the CPUs it ran on and the main-run ceiling. The second
  failed attempt writes the verdict INVALID and nothing more can run. Series 2 may also make its
  second attempt after a passed first one, with code of another SHA-256 (`sanity_after_pass`).
- `run` is the main run, once: it needs a passed sanity attempt made by this very code, runs ASR
  and the index utterance by utterance under the wall-time ceiling of the series (90 min; series
  2 and 3 — 180 min, `main_wall_seconds`), then computes the
  metric (AUC, its bootstrap CI and coverage) exactly once and writes `result.json` with the
  verdict. A second `run` is refused whatever the first one ended with.

A sanity attempt and the main run must sit in a cgroup with `memory.max` ≤ `MEMORY_MAX_BYTES`
and no swap (`systemd-run --user --scope -p MemoryMax=8000000000 -p MemorySwapMax=0`); each pins
itself to at most `MAX_CPUS` CPUs. The ASR model is loaded once per process and the weight files
hashed once per process (`_reused_model`); the model, its parameters and the receipts
`asr.transcribe` writes are those of the analysis path.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import sys
import time
import uuid
import wave
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sound_scout import acoustics, asr, config, doctor, guard, paths, runs
from sound_scout.eval import corpus

DECISION_DOC = Path("docs/decisions/0019-next-audio-experiment.md")
RULE_BEGIN = "<!-- prosody-rule:begin -->"
RULE_END = "<!-- prosody-rule:end -->"
RULE_HEADER = "```text\nPROSODY-RULE v1 "
RULE_SHA256 = "b6ce9fcfbf24b8fa706b2470d47b9967f896fcf677fe5b9326b72bf6af28cc65"

# Pins of docs/dependencies.md (§ 4 weights, § 5 RINCO); a test checks they are written there.
ASR_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
ASR_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
ASR_WEIGHTS_SHA256 = {
    "model.bin": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
}
RINCO_SHA256 = "243688e816bfdf9adbfd65ef5c6104acccd0742ec9bb78b00edb2a299a7148f1"

# The RINCO files #135 read to check its rule (decision 0021, «Файлы RINCO»).
FILES_READ_BY_135 = (
    "INTON-EAF/1-Questions/q-1122.wav",
    "INTON-EAF/2-Imperatives/i-1113.wav",
    "INTON-EAF/2-Imperatives/i-12124_2.wav",
    "INTON-EAF/3-Statements/s-218.WAV",
    "INTON-EAF/3-Statements/s-221.WAV",
)
ASR_RULE_135 = {
    "change": "ASR trust rule: no_vad_speech flags suspect_hallucination",
    "issue": "https://github.com/renidere/sound-scout-plugin/issues/135",
    "commit": "fe18c4aa99c622d23f3eaa31ac06bb62c020b71d",
    "decision": "docs/decisions/0021-asr-vad-hallucination.md",
}
SERIES_3_DECISION = (
    "https://github.com/renidere/sound-scout-plugin/issues/142#issuecomment-7064744998"
)

# One series is one owner-approved run in a directory of its own; nothing carries over between
# series. `data_changes` are changes of the data definition since series 1, with their commit.
SERIES = {
    1: {"dir": "prosody-measure-1", "excluded": (), "data_changes": ()},
    2: {
        "dir": "prosody-measure-1-series-2",
        "owner_decision": "https://github.com/renidere/sound-scout-plugin/issues/134#issuecomment-7060595460",
        # The owner raised the main-run ceiling of this series and allowed a second sanity attempt
        # with the code of that ceiling after a passed first one.
        "ceiling_decision": "https://github.com/renidere/sound-scout-plugin/issues/134#issuecomment-7062644385",
        "main_wall_seconds": 180 * 60,
        "sanity_after_pass": True,
        "excluded": FILES_READ_BY_135,
        "data_changes": (ASR_RULE_135,),
    },
    3: {
        "dir": "prosody-measure-1-series-3",
        # The owner approved one series 3 with the limits of series 2 (ceiling 180 min).
        "owner_decision": SERIES_3_DECISION,
        "ceiling_decision": SERIES_3_DECISION,
        "main_wall_seconds": 180 * 60,
        "eaf_rules": True,
        "excluded": FILES_READ_BY_135,
        "data_changes": (
            ASR_RULE_135,
            {
                "change": "EAF rules E1 (no time alignment) and E2 (WAV with more than one EAF)",
                "issue": "https://github.com/renidere/sound-scout-plugin/issues/142",
                "commit": "14ba9273756b30068654976e65df6c2935dc24cb",
                "decision": "docs/decisions/0019-next-audio-experiment.md (Серия 3)",
            },
        ),
    },
}
OUTPUT_DIR = SERIES[1]["dir"]
CORPUS_DIR = "corpus"
PILOT_PROVENANCE = Path("corpus/pilot-1/provenance.json")
PILOT_RINCO_KINDS = ("rinco_candidate", "corrupted_asr")
GATE0_CLIPS = Path("gate0/2026-11-19/clips")
PREPARE_FILE = "prepare.json"
SANITY_DIR = "sanity"
MAIN_DIR = "main"
RUNS_DIR = "runs"
STARTED_FILE = "started.json"
UTTERANCES_FILE = "utterances.jsonl"
RESULT_FILE = "result.json"
DIR_MODE = 0o700
FILE_MODE = 0o600
FORMAT_VERSION = 1

SAMPLE_RATE = 16_000
TONE_MARKS = "/\\"
LOUDNESS_MARK = "+"
MARKS = TONE_MARKS + LOUDNESS_MARK
VOWELS = frozenset("аеиоуыэюя")
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 0
CI_PERCENTILES = (2.5, 97.5)
PASS_COVERAGE = 0.70
PASS_AUC = 0.75
PASS_CI_LOW = 0.70
MAX_SANITY_ATTEMPTS = 2
MAIN_WALL_SECONDS = 90 * 60
MEMORY_MAX_BYTES = 8_000_000_000
MAX_CPUS = 8
FEATURES = ("f0", "rms", "dur")
RULE_E1 = "E1: EAF without time alignment"
RULE_E2 = "E2: WAV with more than one EAF"

# Decision 0010 § 2.1: the Gate 0 clips, the PCM SHA-256 of each, the target word number per pair.
SANITY_CLIPS = {
    "B1": "06f9208aecefd9d3b179842776ef80370056c52b82c336a87a81c38e50afeca0",
    "V1": "4e3c0ea813709bd8f25ebacd378128257994f25bf9b5e4979723def229383eb6",
    "B2": "e54a56e0afc6ee47f26072acf14ea4dd46d6778a4658e6a42d5928758bdda26a",
    "V2": "50de4d80de5b896f2655b0a7cf1192b2af85cd8b80c0dc469182a68b855960a7",
    "B3": "e0ca641fae9e81961240056c24f89eea3b7d2c554ca7b32165b842c8fc7d9df7",
    "V3": "7a50cb7ab764abe3fb27e89763af567d957f39eed0854e4a1fa335ce0aa49c85",
    "S0": "ac2878215ad33205d1732578c89f25ededaad1b676911c540825bcf4baa42d9b",
    "N1": "00f375bf3faedf91e2cd48ec34eee4ee8f28da159cf2b43d6d65451ccdcf9f5f",
    "T1": "7e66e958d585baca033c7e2ae741a4f7e514c22f709917e293bca064b8ad9276",
}
SANITY_PAIRS = (
    {"pair": 1, "base": "B1", "variant": "V1", "word": 15, "interval": (8.90, 9.26), "factor": 1.0},
    {"pair": 2, "base": "B2", "variant": "V2", "word": 6, "interval": (4.22, 4.66), "factor": 1.6},
    {"pair": 3, "base": "B3", "variant": "V3", "word": 3, "interval": (0.56, 1.00), "factor": 1.3},
)
SANITY_CONTROLS = ("S0", "N1", "T1")

PASS, FAIL, INCONCLUSIVE, INVALID = "PASS", "FAIL", "INCONCLUSIVE", "INVALID"

# (samples, run dir) -> {"status", "reason", "words", "trusted_segments", "flagged_words"}
Transcriber = Callable[[np.ndarray, Path], dict]


class ProsodyError(Exception):
    """The step cannot start or go on as the protocol says; nothing was computed after it."""


# --- the rule ------------------------------------------------------------------------------


def rule_sha256(document: str) -> str:
    """SHA-256 of the text between the markers, stripped at the edges, as UTF-8."""
    if document.count(RULE_BEGIN) != 1 or document.count(RULE_END) != 1:
        raise ProsodyError(f"the decision document must hold one {RULE_BEGIN} … {RULE_END} block")
    begin = document.index(RULE_BEGIN) + len(RULE_BEGIN)
    end = document.index(RULE_END)
    block = document[begin:end].strip()
    if end < begin or not block.startswith(RULE_HEADER):
        raise ProsodyError("the rule block does not start with PROSODY-RULE v1")
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def check_rule(decision: Path) -> str:
    """The rule SHA-256 when it is the one recorded before the run; otherwise refuse to start."""
    found = rule_sha256(decision.read_text(encoding="utf-8"))
    if found != RULE_SHA256:
        raise ProsodyError(f"PROSODY-RULE v1 SHA-256 is {found}, recorded {RULE_SHA256}")
    return found


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# --- words ---------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lower case, ё → е, letters and digits only: no punctuation, marks or combining accents."""
    return "".join(ch for ch in text.casefold().replace("ё", "е") if ch.isalnum())


def rinco_words(value: str) -> list[dict]:
    """The words of an Accent annotation: tokens that keep a letter or digit without marks."""
    words = []
    for token in value.split():
        form = normalize(token)
        if form:
            words.append({"form": form, "marked": any(mark in token for mark in MARKS)})
    return words


def vowel_count(form: str) -> int:
    return max(1, sum(ch in VOWELS for ch in form))


def align(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    """Order-preserving alignment of equal forms (longest common subsequence), deterministic."""
    rows, cols = len(left), len(right)
    table = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if left[i] == right[j]:
                table[i, j] = table[i + 1, j + 1] + 1
            else:
                table[i, j] = max(table[i + 1, j], table[i, j + 1])
    pairs, i, j = [], 0, 0
    while i < rows and j < cols:
        if left[i] == right[j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1, j] >= table[i, j + 1]:
            i += 1
        else:
            j += 1
    return pairs


# --- the index -----------------------------------------------------------------------------


def track(samples: np.ndarray) -> dict:
    """`acoustics.acoustic_track` of a 16 kHz cut with its speech mask from the VAD spans."""
    mono = np.asarray(samples, dtype=np.float64)
    return acoustics.acoustic_track(mono, SAMPLE_RATE, acoustics.speech_activity(mono, SAMPLE_RATE))


def frames_of(track_: dict, word: dict) -> np.ndarray:
    times = track_["times_s"]
    return (times >= word["start"]) & (times < word["end"])


def kept_words(track_: dict, words: list[dict]) -> tuple[list[dict], int]:
    """ASR words with a positive duration and at least one frame (rule step 3), and the rest."""
    kept = [w for w in words if w["end"] - w["start"] > 0 and bool(frames_of(track_, w).any())]
    return kept, len(words) - len(kept)


def raw_features(track_: dict, words: list[dict]) -> list[dict]:
    """f0 (semitones), rms (dB) and dur (log) of each word against its utterance (rule step 5)."""
    f0, rms = track_["f0_hz"], track_["rms_dbfs"]
    voiced = f0 > 0
    f0_median = float(np.median(f0[voiced])) if voiced.any() else None
    speech_levels = rms[track_["speech"] & np.isfinite(rms)]
    rms_median = float(np.median(speech_levels)) if len(speech_levels) else None
    durations = [math.log((w["end"] - w["start"]) / vowel_count(w["form"])) for w in words]
    dur_median = float(np.median(durations)) if durations else 0.0
    result = []
    for word, duration in zip(words, durations, strict=True):
        frames = frames_of(track_, word)
        word_f0 = f0[frames & voiced]
        word_rms = rms[frames & np.isfinite(rms)]
        result.append(
            {
                "f0": 12 * math.log2(float(word_f0.max()) / f0_median)
                if len(word_f0) and f0_median
                else 0.0,
                "rms": float(word_rms.max()) - rms_median
                if len(word_rms) and rms_median is not None
                else 0.0,
                "dur": duration - dur_median,
            }
        )
    return result


def zscores(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    sigma = float(array.std()) if len(array) else 0.0
    if sigma == 0:
        return [0.0] * len(array)
    return [float(v) for v in (array - array.mean()) / sigma]


def index(features: list[dict]) -> list[dict]:
    """z-scores of each feature within the utterance and their equal-weight sum (rule step 6)."""
    z = {name: zscores([f[name] for f in features]) for name in FEATURES}
    return [
        {**{f"z_{name}": z[name][i] for name in FEATURES}, "index": sum(z[n][i] for n in FEATURES)}
        for i in range(len(features))
    ]


def score_utterance(track_: dict, asr_words: list[dict], words: list[dict]) -> dict:
    """Rule steps 3–6 for one utterance: RINCO words with their index, or None when unmatched."""
    forms = [dict(w, form=normalize(w["word"])) for w in asr_words]
    kept, dropped = kept_words(track_, forms)
    matches = align([w["form"] for w in kept], [w["form"] for w in words])
    matched = [kept[i] for i, _ in matches]
    scores = index(raw_features(track_, matched)) if matched else []
    scored = [dict(w, score=None) for w in words]
    for (_, j), score in zip(matches, scores, strict=True):
        scored[j]["score"] = score
    return {
        "words": scored,
        "asr_words": len(asr_words),
        "asr_excluded": dropped,
        "asr_unmatched": len(kept) - len(matches),
    }


# --- the metric ----------------------------------------------------------------------------


def pair_counts(utterance_words: list[dict], key: str = "index") -> tuple[float, int]:
    """(wins, pairs) over marked × unmarked scored words of one utterance; a tie counts 0.5."""
    marked = [w["score"][key] for w in utterance_words if w["marked"] and w["score"]]
    plain = [w["score"][key] for w in utterance_words if not w["marked"] and w["score"]]
    wins = sum(1.0 if m > p else 0.5 if m == p else 0.0 for m in marked for p in plain)
    return wins, len(marked) * len(plain)


def auc(counts: list[tuple[float, int]]) -> float | None:
    pairs = sum(p for _, p in counts)
    return sum(w for w, _ in counts) / pairs if pairs else None


def bootstrap_ci(per_file: list[tuple[float, int]]) -> tuple[float, float] | None:
    """95 % CI by resampling RINCO files: 1000 repeats, seed 0, percentiles 2.5 and 97.5."""
    if not per_file:
        return None
    wins = np.array([w for w, _ in per_file], dtype=np.float64)
    pairs = np.array([p for _, p in per_file], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPEATS):
        chosen = rng.integers(0, len(per_file), size=len(per_file))
        total = pairs[chosen].sum()
        if total:
            values.append(wins[chosen].sum() / total)
    if not values:
        return None
    low, high = np.percentile(values, CI_PERCENTILES)
    return float(low), float(high)


def verdict(coverage: float, auc_: float | None, ci: tuple[float, float] | None) -> str:
    if coverage < PASS_COVERAGE:
        return INCONCLUSIVE
    if auc_ is not None and ci is not None and auc_ >= PASS_AUC and ci[0] >= PASS_CI_LOW:
        return PASS
    return FAIL


def metric(utterances: list[dict]) -> dict:
    """AUC, CI, coverage and the side numbers of the rule over every utterance record."""
    by_file: dict[str, list[tuple[float, int]]] = {}
    by_folder: dict[str, list[tuple[float, int]]] = {}
    by_feature: dict[str, list[tuple[float, int]]] = {f"z_{n}": [] for n in FEATURES}
    marked = matched_marked = words = matched = 0
    for record in utterances:
        counts = pair_counts(record["words"])
        by_file.setdefault(record["member"], []).append(counts)
        by_folder.setdefault(record["folder"], []).append(counts)
        for key, found in by_feature.items():
            found.append(pair_counts(record["words"], key))
        words += len(record["words"])
        matched += sum(1 for w in record["words"] if w["score"])
        marked += sum(1 for w in record["words"] if w["marked"])
        matched_marked += sum(1 for w in record["words"] if w["marked"] and w["score"])
    per_file = [(sum(w for w, _ in c), sum(p for _, p in c)) for _, c in sorted(by_file.items())]
    auc_ = auc(per_file)
    ci = bootstrap_ci(per_file)
    coverage = matched_marked / marked if marked else 0.0
    return {
        "verdict": verdict(coverage, auc_, ci),
        "auc": auc_,
        "ci95": list(ci) if ci else None,
        "coverage": coverage,
        "pairs": sum(p for _, p in per_file),
        "files": len(by_file),
        "utterances": len(utterances),
        "words": words,
        "marked_words": marked,
        "matched_words": matched,
        "matched_marked_words": matched_marked,
        "unmatched_rinco_words": words - matched,
        "utterances_without_pairs": sum(1 for r in utterances if not pair_counts(r["words"])[1]),
        "asr_words": sum(r["asr_words"] for r in utterances),
        "asr_excluded_words": sum(r["asr_excluded"] for r in utterances),
        "asr_unmatched_words": sum(r["asr_unmatched"] for r in utterances),
        "asr_flagged_words": sum(r.get("asr_flagged_words", 0) for r in utterances),
        "utterances_not_scored": {
            reason: sum(1 for r in utterances if r.get("skipped") == reason)
            for reason in sorted({r["skipped"] for r in utterances if r.get("skipped")})
        },
        "auc_by_section": {k: auc(v) for k, v in sorted(by_folder.items())},
        "auc_by_feature": {k: auc(v) for k, v in by_feature.items()},
    }


# --- material ------------------------------------------------------------------------------


def excluded_members(provenance: dict, extra: tuple[str, ...] = ()) -> list[str]:
    """The RINCO WAV members of `pilot-1` (dev and held-out) and `extra`, sorted and unique."""
    return sorted(
        {
            clip["member"]
            for clip in provenance["clips"]
            if clip.get("kind") in PILOT_RINCO_KINDS
            and str(clip.get("source", "")).endswith(corpus.RINCO_ARCHIVE)
        }
        | set(extra)
    )


def series_excluded(config_: config.Config, series: int) -> list[str]:
    provenance = _read_json(config_.data_root / PILOT_PROVENANCE)
    return excluded_members(provenance, SERIES[series]["excluded"])


def without_time_alignment(file: corpus.RincoFile) -> bool:
    """Rule E1: at least two annotations, and every start and end of them on a whole second."""
    found = [a for annotations in file.annotations.values() for a in annotations]
    times = [t for a in found for t in (a.start, a.end)]
    return len(found) >= 2 and all(round(t * 1000) % 1000 == 0 for t in times)


def rule_exclusions(files: list[corpus.RincoFile]) -> dict[str, str]:
    """EAF and WAV members the rules E1 and E2 of 0019 «Серия 3» leave out, with the rule.

    Both read only the EAF/WAV pairing and the annotation times, never a text or a sample.
    """
    eafs_per_wav = Counter(f.wav_member for f in files)
    shared = [f for f in files if eafs_per_wav[f.wav_member] > 1]
    found = {m: RULE_E2 for f in shared for m in (f.wav_member, f.eaf_member)}
    found |= {f.eaf_member: RULE_E1 for f in files if without_time_alignment(f)}
    return dict(sorted(found.items()))


def series_exclusion(config_: config.Config, series: int, archive: zipfile.ZipFile) -> dict:
    """Every member the series leaves out (WAV or EAF), its count and the SHA-256 of the list."""
    rules = (
        rule_exclusions(corpus.rinco_files(archive)[0]) if SERIES[series].get("eaf_rules") else {}
    )
    members = sorted(set(series_excluded(config_, series)) | set(rules))
    record = {"members": members, "count": len(members), "sha256": list_sha256(members)}
    return {**record, "rules": rules} if rules else record


def list_sha256(members: list[str]) -> str:
    return hashlib.sha256("".join(f"{m}\n" for m in members).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Utterance:
    member: str
    folder: str
    number: int
    start: float
    end: float
    words: tuple[dict, ...]


def utterances(file: corpus.RincoFile) -> list[Utterance]:
    """Accent annotations of one RINCO file with at least one word."""
    found = []
    for number, annotation in enumerate(file.annotations["accent"]):
        words = rinco_words(annotation.value)
        if words:
            found.append(
                Utterance(
                    file.wav_member,
                    file.folder,
                    number,
                    annotation.start,
                    annotation.end,
                    tuple(words),
                )
            )
    return found


def material(archive: zipfile.ZipFile, excluded: list[str]) -> tuple[list[corpus.RincoFile], dict]:
    """The paired RINCO files without those whose WAV or EAF member is excluded, and counts."""
    files, stats = corpus.rinco_files(archive)
    left_out = set(excluded)
    missing = sorted(left_out - {m for f in files for m in (f.wav_member, f.eaf_member)})
    if missing:
        raise ProsodyError(
            f"the exclusion names RINCO members the archive does not pair: {missing}"
        )
    kept = [f for f in files if not {f.wav_member, f.eaf_member} & left_out]
    found = [u for f in kept for u in utterances(f)]
    words = [w for u in found for w in u.words]
    counts = {
        "paired_files": len(files),
        "excluded_files": len(files) - len(kept),
        "files": len(kept),
        "utterances": len(found),
        "words": len(words),
        "marked_words": sum(w["marked"] for w in words),
        "mixed_utterances": sum(
            1 for u in found if 0 < sum(w["marked"] for w in u.words) < len(u.words)
        ),
        "audio_minutes": round(sum(f.duration_s for f in kept) / 60, 2),
        "archive": stats,
    }
    return kept, counts


# --- precondition Q1 and the resource scope ------------------------------------------------


def rinco_source() -> corpus.Source:
    return next(s for s in corpus.SOURCES if s.name == corpus.RINCO_ARCHIVE)


def rinco_path(config_: config.Config) -> Path:
    return config_.data_root / CORPUS_DIR / rinco_source().relative_path


def q1(config_: config.Config) -> list[str]:
    """Problems of precondition Q1; an empty list means the network is closed and the pins hold."""
    problems = []
    if not guard.install_network_guard(config_):
        problems.append("the network guard is not active (network.allow is true)")
    if config_.asr.model != ASR_REPO:
        problems.append(f"asr.model is {config_.asr.model}, pinned {ASR_REPO}")
    model, reason = doctor.snapshot(config_.asr.model, config_.asr.revision)
    if reason is not None:
        problems.append(f"ASR weights: {reason}")
    elif model["revision"] != ASR_REVISION:
        problems.append(f"ASR revision {model['revision']}, pinned {ASR_REVISION}")
    else:
        snapshot = doctor.snapshot_dir(model["repo"], model["revision"])
        for name, pinned in ASR_WEIGHTS_SHA256.items():
            found = asr._sha256_file(snapshot / name)
            if found != pinned:
                problems.append(f"ASR {name} SHA-256 {found}, pinned {pinned}")
    return problems + rinco_problems(config_)


def rinco_problems(config_: config.Config) -> list[str]:
    """The RINCO part of Q1: the pin of `corpus.SOURCES` and the archive on disk."""
    source = rinco_source()
    problems = []
    if source.sha256 != RINCO_SHA256:
        problems.append(f"corpus.SOURCES pins RINCO {source.sha256}, dependencies {RINCO_SHA256}")
    return problems + corpus.check_source(config_.data_root / CORPUS_DIR, source)


def memory_scope(proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")):
    """(memory.max, memory.swap.max) of this process's cgroup v2, as text, or None when unknown."""
    try:
        line = next(
            entry
            for entry in (proc_root / "self/cgroup").read_text().splitlines()
            if entry.startswith("0::")
        )
        group = cgroup_root / line[3:].lstrip("/")
        return (
            (group / "memory.max").read_text().strip(),
            (group / "memory.swap.max").read_text().strip(),
        )
    except (OSError, StopIteration):
        return None


def scope_problems(scope: tuple[str, str] | None) -> list[str]:
    if scope is None:
        return ["the cgroup memory limits cannot be read"]
    memory, swap = scope
    problems = []
    if not memory.isdigit() or int(memory) > MEMORY_MAX_BYTES:
        problems.append(f"memory.max is {memory}, needs at most {MEMORY_MAX_BYTES}")
    if swap != "0":
        problems.append(f"memory.swap.max is {swap}, needs 0")
    return problems


def pin_cpus() -> list[int]:
    """Keep this process on at most `MAX_CPUS` of the CPUs it may use; returns them."""
    allowed = sorted(os.sched_getaffinity(0))[:MAX_CPUS]
    os.sched_setaffinity(0, allowed)
    return allowed


@contextlib.contextmanager
def _reused_model() -> Iterator[None]:
    """One model load and one hash per weight file in this process instead of one per utterance."""
    load, sha = asr._load_model, asr._sha256_file
    asr._load_model = functools.cache(load)
    seen: dict[tuple, str] = {}

    def hashed(path: Path) -> str:
        info = os.stat(path)
        key = (os.fspath(path), info.st_size, info.st_mtime_ns)
        if key not in seen:
            seen[key] = sha(path)
        return seen[key]

    asr._sha256_file = hashed
    try:
        yield
    finally:
        asr._load_model, asr._sha256_file = load, sha


# --- ASR -----------------------------------------------------------------------------------


def transcribed(status, reason=None, words=(), trusted_segments=0, flagged_words=0) -> dict:
    return {
        "status": status,
        "reason": reason,
        "words": list(words),
        "trusted_segments": trusted_segments,
        "flagged_words": flagged_words,
    }


def local_transcriber(config_: config.Config) -> Transcriber:
    """`asr.transcribe` in a new run directory of its own; the words of trusted segments only.

    Words of segments flagged `suspect_hallucination` are not used (decision 0005; rule S2 reads
    an evaluated word as one of a trusted segment) and are counted as `flagged_words`.
    """

    def transcribe(samples: np.ndarray, run_dir: Path) -> dict:
        os.mkdir(run_dir, DIR_MODE)
        run = runs.Run(f"prosody-{uuid.uuid4().hex}", run_dir, OUTPUT_DIR, "", False, False, True)
        transcription = asr.transcribe(np.asarray(samples, dtype=np.float32), run, config_)
        if transcription.status != "completed":
            return transcribed(transcription.status, transcription.reason)
        trusted, flagged = asr.trusted_segments(transcription.raw)
        words = [
            {"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
            for segment in trusted
            for w in segment.get("words") or []
        ]
        return transcribed(
            "completed",
            None,
            words,
            trusted_segments=len(trusted),
            flagged_words=sum(len(segment.get("words") or []) for segment in flagged),
        )

    return transcribe


# --- sanity stage --------------------------------------------------------------------------


def read_clip(path: Path, pcm_sha256: str) -> np.ndarray:
    """A Gate 0 clip (16 kHz mono 16-bit WAV) as float samples, refused when its PCM differs."""
    with wave.open(os.fspath(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            2,
        ):
            raise ProsodyError(f"{path.name}: expected 16 kHz mono 16-bit PCM")
        pcm = source.readframes(source.getnframes())
    if hashlib.sha256(pcm).hexdigest() != pcm_sha256:
        raise ProsodyError(f"{path.name}: PCM SHA-256 differs from decision 0010 § 2.1")
    return np.frombuffer(pcm, "<i2").astype(np.float64) / 32768


def clip_scores(samples: np.ndarray, words: list[dict]) -> list[dict | None]:
    """Index of each ASR word of a whole clip (the clip is the utterance); None when excluded."""
    if not words:
        return []
    track_ = track(samples)
    forms = [dict(w, form=normalize(w["word"])) for w in words]
    kept, _ = kept_words(track_, forms)
    scores = iter(index(raw_features(track_, kept)) if kept else [])
    kept_ids = {id(w) for w in kept}
    return [next(scores) if id(w) in kept_ids else None for w in forms]


def sanity(clips: dict[str, np.ndarray], transcribe: Transcriber, work: Path) -> dict:
    """S1 and S2 over the Gate 0 clips with their own ASR and timestamps."""
    words, segments, results, started = {}, {}, {}, time.monotonic()
    for clip_id, samples in clips.items():
        found = transcribe(samples, work / clip_id)
        if found["status"] != "completed":
            raise ProsodyError(f"{clip_id}: ASR is {found['status']}: {found['reason']}")
        words[clip_id], segments[clip_id] = found["words"], found["trusted_segments"]
    pairs = []
    for pair in SANITY_PAIRS:
        entry = {"pair": pair["pair"], "word": pair["word"]}
        for role in ("base", "variant"):
            clip_id = pair[role]
            scores = clip_scores(clips[clip_id], words[clip_id])
            number = pair["word"]
            score = scores[number - 1] if len(scores) >= number else None
            target = words[clip_id][number - 1] if len(words[clip_id]) >= number else None
            entry[role] = {
                "clip": clip_id,
                "asr_words": len(words[clip_id]),
                "target": None if target is None else {k: target[k] for k in ("start", "end")},
                "index": None if score is None else score["index"],
            }
        start, end = pair["interval"]
        stretched = round(start + (end - start) * pair["factor"], 3)
        entry["expected_variant_interval"] = [start, stretched]
        b, v = entry["base"]["index"], entry["variant"]["index"]
        entry["variant_higher"] = b is not None and v is not None and v > b
        pairs.append(entry)
    controls = {
        c: sum(s is not None for s in clip_scores(clips[c], words[c])) for c in SANITY_CONTROLS
    }
    results["s1"] = all(p["variant_higher"] for p in pairs)
    trusted = {c: segments[c] for c in SANITY_CONTROLS}
    results["s2"] = not any(controls.values()) and not any(trusted.values())
    return {
        "pairs": pairs,
        "controls_scored_words": controls,
        "controls_trusted_segments": trusted,
        **results,
        "passed": results["s1"] and results["s2"],
        "elapsed_s": round(time.monotonic() - started, 1),
    }


# --- main run ------------------------------------------------------------------------------


def run_material(
    archive: zipfile.ZipFile,
    files: list[corpus.RincoFile],
    transcribe: Transcriber,
    work: Path,
    log: Path,
    deadline: float,
) -> tuple[list[dict], bool]:
    """Every utterance of `files` scored and appended to `log`; (records, stopped by the limit)."""
    clock = time.monotonic
    records = []
    for number, file in enumerate(files):
        wav = work / f"source-{number}.wav"
        samples = corpus.decode(corpus.copy_member(archive, file.wav_member, wav))
        wav.unlink()
        for utterance in utterances(file):
            if clock() > deadline:
                return records, True
            record = {
                "member": utterance.member,
                "folder": utterance.folder,
                "utterance": utterance.number,
                "start": utterance.start,
                "end": utterance.end,
            }
            cut = samples[round(utterance.start * SAMPLE_RATE) : round(utterance.end * SAMPLE_RATE)]
            words = [dict(w, score=None) for w in utterance.words]
            if len(cut) < SAMPLE_RATE * acoustics.MIN_TRACK_SECONDS:
                record |= {"skipped": "too_short", "words": words, "asr_words": 0,
                           "asr_excluded": 0, "asr_unmatched": 0}  # fmt: skip
            else:
                found = transcribe(cut, work / f"utt-{len(records):05d}")
                record["asr_flagged_words"] = found["flagged_words"]
                if found["status"] != "completed":
                    record |= {"skipped": f"asr_{found['status']}", "reason": found["reason"],
                               "words": words, "asr_words": 0, "asr_excluded": 0,
                               "asr_unmatched": 0}  # fmt: skip
                else:
                    record |= score_utterance(track(cut), found["words"], list(utterance.words))
            records.append(record)
            with open(log, "a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records, clock() > deadline


# --- files ---------------------------------------------------------------------------------


def _mkdir(path: Path) -> Path:
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, DIR_MODE)
    return path


def _write_json(path: Path, data: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main_wall_seconds(series: int) -> float:
    return SERIES[series].get("main_wall_seconds", MAIN_WALL_SECONDS)


def output_dir(config_: config.Config, series: int = 1) -> Path:
    return _mkdir(paths.resolve_output(config_.data_root, SERIES[series]["dir"], config_))


def _header(decision: Path, series: int = 1) -> dict:
    if series not in SERIES:
        raise ProsodyError(f"series {series} is not approved; approved: {sorted(SERIES)}")
    approved = SERIES[series]
    return {
        "format_version": FORMAT_VERSION,
        "experiment": "prosody-measure-1",
        "series": series,
        "decision": "docs/decisions/0019-next-audio-experiment.md (редакция 2)",
        **({"owner_decision": approved["owner_decision"]} if "owner_decision" in approved else {}),
        "data_changes": list(approved["data_changes"]),
        "main_wall_seconds": main_wall_seconds(series),
        **(
            {"ceiling_decision": approved["ceiling_decision"]}
            if "ceiling_decision" in approved
            else {}
        ),
        "rule_sha256": check_rule(decision),
        "code_sha256": code_sha256(),
        "at": _now(),
    }


def _invalid(out: Path, header: dict, reason: str) -> dict:
    result = {**header, "verdict": INVALID, "reason": reason}
    _write_json(out / RESULT_FILE, result)
    return result


def _stopped(out: Path, header: dict, reason: str) -> dict:
    result = {**header, "verdict": INCONCLUSIVE, "reason": reason}
    _write_json(out / RESULT_FILE, result)
    return result


def _refuse_if_decided(out: Path) -> None:
    if (out / RESULT_FILE).exists():
        raise ProsodyError(f"{RESULT_FILE} exists: the verdict is out, nothing more runs")


# --- commands ------------------------------------------------------------------------------


def prepare(config_: config.Config, decision: Path = DECISION_DOC, series: int = 1) -> dict:
    header = _header(decision, series)
    out = output_dir(config_, series)
    _refuse_if_decided(out)
    if (out / PREPARE_FILE).exists():
        raise ProsodyError(f"{PREPARE_FILE} exists; prepare runs once")
    problems = q1(config_)
    if problems:
        return _invalid(out, header, "Q1: " + "; ".join(problems))
    with zipfile.ZipFile(rinco_path(config_)) as archive:
        exclusion = series_exclusion(config_, series, archive)
        _, counts = material(archive, exclusion["members"])
    record = {**header, "q1": "ok", "exclusion": exclusion, "material": counts}
    _write_json(out / PREPARE_FILE, record)
    return record


def _checked_scope(scope: tuple[str, str] | None | bool) -> tuple[str, str] | None | bool:
    """The cgroup memory limits (read when `scope` is True); refuse when they break the rule."""
    if scope is True:
        scope = memory_scope()
    issues = [] if scope is False else scope_problems(scope)
    if issues:
        raise ProsodyError("resource scope: " + "; ".join(issues))
    return scope


def run_sanity(
    config_: config.Config,
    decision: Path = DECISION_DOC,
    transcribe=None,
    scope: tuple[str, str] | None | bool = True,
    series: int = 1,
) -> dict:
    header = _header(decision, series)
    out = output_dir(config_, series)
    _refuse_if_decided(out)
    if not (out / PREPARE_FILE).exists():
        raise ProsodyError("run `prepare` first")
    attempts = _mkdir(out / SANITY_DIR)
    passed = [a for a in map(_read_json, attempts.glob("attempt-*.json")) if a["passed"]]
    after_pass = SERIES[series].get("sanity_after_pass", False)
    if passed and not (
        after_pass and all(a["code_sha256"] != header["code_sha256"] for a in passed)
    ):
        raise ProsodyError("a sanity attempt of this code has passed already; run the main step")
    started = len(list(attempts.glob("work-*")))  # an attempt that died without a record counts
    if started >= MAX_SANITY_ATTEMPTS:
        raise ProsodyError(f"{started} sanity attempts made, none passed")
    scope = _checked_scope(scope)
    cpus = pin_cpus()
    problems = q1(config_)
    if problems:
        return _invalid(out, header, "Q1: " + "; ".join(problems))
    number = started + 1
    work = attempts / f"work-{number}"
    os.mkdir(work, DIR_MODE)
    try:
        clips = {
            clip_id: read_clip(config_.data_root / GATE0_CLIPS / f"{clip_id}.wav", sha)
            for clip_id, sha in SANITY_CLIPS.items()
        }
        with _reused_model():
            found = sanity(clips, transcribe or local_transcriber(config_), work)
    except Exception as error:  # a broken attempt is a failed attempt, recorded and counted
        found = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    record = {**header, "attempt": number, "cpus": cpus, "memory_scope": scope, **found}
    _write_json(attempts / f"attempt-{number}.json", record)
    if not record["passed"] and number >= MAX_SANITY_ATTEMPTS:
        _invalid(out, header, f"санитарная стадия не пройдена за {number} попытки")
    return record


def run_main(
    config_: config.Config,
    decision: Path = DECISION_DOC,
    transcribe=None,
    scope: tuple[str, str] | None | bool = True,
    wall_seconds: float | None = None,
    series: int = 1,
) -> dict:
    header = _header(decision, series)
    if wall_seconds is None:
        wall_seconds = header["main_wall_seconds"]
    out = output_dir(config_, series)
    _refuse_if_decided(out)
    main_dir = out / MAIN_DIR
    if (main_dir / STARTED_FILE).exists():
        reason = "основной прогон остановлен без результата (лимит ресурсов или kill); повтора нет"
        return _stopped(out, header, reason)
    if main_dir.exists():
        raise ProsodyError(f"{MAIN_DIR}/ exists: the main run starts once")
    prepared = _read_json(out / PREPARE_FILE) if (out / PREPARE_FILE).exists() else None
    if prepared is None:
        raise ProsodyError("run `prepare` first")
    attempts = sorted((out / SANITY_DIR).glob("attempt-*.json"))
    passed = [a for a in map(_read_json, attempts) if a["passed"]]
    if not passed or passed[-1]["code_sha256"] != header["code_sha256"]:
        raise ProsodyError("no passed sanity attempt of this code; run `sanity` first")
    scope = _checked_scope(scope)
    problems = rinco_problems(config_)  # before the archive is read for the exclusion list
    if problems:
        return _invalid(out, header, "Q1: " + "; ".join(problems))
    with zipfile.ZipFile(rinco_path(config_)) as archive:
        excluded = series_exclusion(config_, series, archive)["members"]
    if list_sha256(excluded) != prepared["exclusion"]["sha256"]:
        raise ProsodyError("the exclusion list changed since `prepare`")
    _mkdir(main_dir)
    cpus = pin_cpus()
    started = time.monotonic()
    _write_json(main_dir / STARTED_FILE, {**header, "cpus": cpus, "memory_scope": scope})
    problems = q1(config_)
    if problems:
        return _invalid(out, header, "Q1: " + "; ".join(problems))
    try:
        with zipfile.ZipFile(rinco_path(config_)) as archive:
            files, counts = material(archive, excluded)
            if counts != prepared["material"]:
                raise ProsodyError("the material differs from `prepare`")
            with _reused_model():
                records, stopped = run_material(
                    archive,
                    files,
                    transcribe or local_transcriber(config_),
                    _mkdir(main_dir / RUNS_DIR),
                    main_dir / UTTERANCES_FILE,
                    started + wall_seconds,
                )
    except MemoryError:
        return _stopped(out, header, "основной прогон остановлен лимитом памяти")
    except Exception as error:  # the main run starts once: its failure is recorded, not retried
        return _invalid(out, header, f"основной прогон прерван: {type(error).__name__}: {error}")
    elapsed = round(time.monotonic() - started, 1)
    if stopped:
        result = {
            **header,
            "verdict": INCONCLUSIVE,
            "reason": f"остановлено потолком wall time {wall_seconds / 60:g} мин",
            "utterances_done": len(records),
            "elapsed_s": elapsed,
        }
    else:
        result = {**header, **metric(records), "elapsed_s": elapsed, "cpus": cpus}
    result["exclusion_sha256"] = prepared["exclusion"]["sha256"]
    _write_json(out / RESULT_FILE, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.prosody")
    parser.add_argument("command", choices=("prepare", "sanity", "run"))
    parser.add_argument("--decision", type=Path, default=DECISION_DOC)
    parser.add_argument("--series", type=int, choices=sorted(SERIES), default=1)
    args = parser.parse_args(argv)
    try:
        config_ = config.load()
        guard.install_network_guard(config_)
        step = {"prepare": prepare, "sanity": run_sanity, "run": run_main}[args.command]
        result = step(config_, args.decision, series=args.series)
    except (ProsodyError, corpus.CorpusError, config.ConfigError, paths.PathDenied) as error:
        print(f"prosody: {error}", file=sys.stderr)
        return 2
    keys = ("verdict", "reason", "passed", "s1", "s2", "auc", "ci95", "coverage", "q1")
    print(json.dumps({k: result[k] for k in keys if k in result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
