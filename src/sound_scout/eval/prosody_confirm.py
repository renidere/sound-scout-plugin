"""`prosody-confirm-1`: one confirmatory run of the `PROSODY-RULE v1` index on fresh data (#199).

The protocol is `docs/decisions/0034-word-prominence-confirm.md`; the rule is the block between
its `prosody-confirm-rule` markers, and every command refuses to start unless the SHA-256 of that
block is `RULE_SHA256`. The pipeline of series 3 is frozen: this module calls `prosody` (and
through it `asr` and `acoustics`) and copies nothing. `prepare`, `sanity` and `run` check the
SHA-256 of those files against `FROZEN_SHA256`, and a mismatch makes the verdict INVALID;
`check-fresh` and `import` run no part of the pipeline and check only the rule block.

Commands, all in `<data_root>/prosody-confirm-1/` (outside git):

- `prepare` (once): precondition Q1, the 160 fresh Open STT clips (`select_fresh`), the 40
  calibration utterances of series 3 (`calibration`), `check_fresh`, and the blind label package
  `label-package/` (`listen/<id>.wav`, `sheet.csv`, the instruction). The ids, sources and words
  of the items stay in `alias-map.json`, next to the package, not in it.
- `check-fresh`: the freshness check again, over `prepare.json` and the clip files on disk.
- `import <sheet.csv>`: the sheet a person filled, checked row by row (`ImportRefused` with a
  code), Cohen's kappa on the calibration items, `labels.json` and the SHA-256 of the sheet. The
  first accepted sheet is final.
- `sanity`: one attempt of R0 (20 utterances of series 3 again, same index) and S2' (silence,
  pink noise, tone); at most two attempts, an attempt killed before its record included.
- `run` (once): the main run over the labelled fresh clips and the verdict from `decide`.

Recordings of real calls and their texts never leave the data directory: stdout carries counts,
codes and the verdict only.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

from sound_scout import acoustics, asr, config, guard, paths
from sound_scout.eval import corpus, gate0, openstt, prosody

REPO = Path(__file__).resolve().parents[3]
DECISION_DOC = Path("docs/decisions/0034-word-prominence-confirm.md")
LABELING_DOC = REPO / "docs/eval/PROSODY-LABELING.md"
RULE_BEGIN = "<!-- prosody-confirm-rule:begin -->"
RULE_END = "<!-- prosody-confirm-rule:end -->"
RULE_VERSION = "PROSODY-CONFIRM-RULE v1.1"
RULE_HEADER = f"```text\n{RULE_VERSION} "
RULE_SHA256 = "ecec5226938b280ca001298a2b250d436afb3523bfcf3ac327c02c7405acd16a"
OWNER_DECISION = "https://github.com/renidere/sound-scout-plugin/issues/90#issuecomment-7196234132"
SPEC = "https://github.com/renidere/sound-scout-plugin/issues/199#issuecomment-7196416113"

# The files of series 3 at db786cd (0034 § 2) and the PROSODY-RULE v1 block of 0019.
FROZEN_SHA256 = {
    "src/sound_scout/eval/prosody.py": (
        "d7471142730a749f57db6911ca698ee79ef6d14341f4dda5ff456fb2f9bea565"
    ),
    "src/sound_scout/asr.py": "11c8afdcd3e1f2d92250d339551573f5a85e609c7a6d1244eb026d8ffa978d21",
    "src/sound_scout/acoustics.py": (
        "4a4067bc3fab5d40a357112111f0bb158b36eb46756689d2b90e13c3e2c182e4"
    ),
}
FROZEN_MODULES = {
    "src/sound_scout/eval/prosody.py": prosody,
    "src/sound_scout/asr.py": asr,
    "src/sound_scout/acoustics.py": acoustics,
}
PROSODY_RULE_SHA256 = "b6ce9fcfbf24b8fa706b2470d47b9967f896fcf677fe5b9326b72bf6af28cc65"
OPEN_STT_SHA256 = "aea64611133129e8c9b5d829953f5bfb9cc2d1bd95b9e429fa8e00e2c83f6461"

OUTPUT_DIR = "prosody-confirm-1"
SERIES_3 = 3
PREPARE_FILE = "prepare.json"
ALIAS_FILE = "alias-map.json"
CLIPS_DIR = "clips"
PACKAGE_DIR = "label-package"
LISTEN_DIR = "listen"
SHEET_FILE = "sheet.csv"
IMPORT_DIR = "import"
LABELS_FILE = "labels.json"
SANITY_DIR = "sanity"
MAIN_DIR = "main"
RUNS_DIR = "runs"
STARTED_FILE = "started.json"
CLIPS_LOG = "clips.jsonl"
RESULT_FILE = "result.json"
DIR_MODE = 0o700
FORMAT_VERSION = 1

SAMPLE_RATE = prosody.SAMPLE_RATE
FRESH_COUNT = 160
CALIBRATION_COUNT = 40
FRESH_MIN_WORDS = 6
CALIBRATION_WORDS = (4, 20)
SELECT_PREFIX = "prosody-confirm-1:"
AGREEMENT_PREFIX = "prosody-confirm-1:agreement:"
SHUFFLE_SEED = 0
R0_UTTERANCES = 20
R0_TOLERANCE = 1e-6
CONTROL_SECONDS = 5.0
CONTROL_SEED = 0
RECEIPT_KEYS = ("repo", "revision", "device", "compute_type", "files", "local_files_only")
RECEIPT_VERSIONS = ("faster-whisper", "ctranslate2")
MAX_SANITY_ATTEMPTS = 2
SANITY_WALL_SECONDS = 15 * 60
MAIN_WALL_SECONDS = 60 * 60
MIN_PAIRS = 1200
MIN_CLIPS_WITH_PAIRS = 60
KAPPA_MIN = 0.40
SHEET_COLUMNS = ("id", "word_no", "word", "marked", "skip_reason")
FRESH, CALIBRATION = "fresh", "calibration"
PASS, FAIL, INCONCLUSIVE, INVALID = (
    prosody.PASS,
    prosody.FAIL,
    prosody.INCONCLUSIVE,
    prosody.INVALID,
)


class ConfirmError(Exception):
    """The step cannot start or go on as 0034 says; nothing was computed after it."""


class ImportRefused(ConfirmError):
    """The sheet is not taken; `code` names why, `problems` list the rows (id, word_no, code)."""

    def __init__(self, code: str, problems: list[tuple] = ()):
        self.code, self.problems = code, list(problems)
        shown = "; ".join(" ".join(str(p) for p in problem) for problem in self.problems[:10])
        more = f" (+{len(self.problems) - 10})" if len(self.problems) > 10 else ""
        super().__init__(f"{code}: {shown}{more}" if shown else code)


# --- the rule and the frozen files ---------------------------------------------------------


def rule_sha256(document: str) -> str:
    """SHA-256 of the text between the markers, stripped at the edges, as UTF-8."""
    if document.count(RULE_BEGIN) != 1 or document.count(RULE_END) != 1:
        raise ConfirmError(f"the decision document must hold one {RULE_BEGIN} … {RULE_END} block")
    begin = document.index(RULE_BEGIN) + len(RULE_BEGIN)
    end = document.index(RULE_END)
    block = document[begin:end].strip()
    if end < begin or not block.startswith(RULE_HEADER):
        raise ConfirmError(f"the rule block does not start with {RULE_VERSION}")
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def check_rule(decision: Path) -> str:
    found = rule_sha256(decision.read_text(encoding="utf-8"))
    if found != RULE_SHA256:
        raise ConfirmError(f"{RULE_VERSION} SHA-256 is {found}, recorded {RULE_SHA256}")
    return found


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def frozen_problems() -> list[str]:
    """The frozen files and the PROSODY-RULE v1 block against their pins; empty when they hold."""
    problems = []
    for name, module in FROZEN_MODULES.items():
        found = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if found != FROZEN_SHA256[name]:
            problems.append(f"{name} SHA-256 {found}, pinned {FROZEN_SHA256[name]}")
    try:
        found = prosody.rule_sha256((REPO / prosody.DECISION_DOC).read_text(encoding="utf-8"))
    except (OSError, prosody.ProsodyError) as error:
        found = f"unreadable ({error})"
    if found != PROSODY_RULE_SHA256:
        problems.append(f"PROSODY-RULE v1 SHA-256 {found}, pinned {PROSODY_RULE_SHA256}")
    return problems


# --- precondition Q1 -----------------------------------------------------------------------


def open_stt_source() -> corpus.Source:
    return next(s for s in corpus.EXTRA_SOURCES if s.name == corpus.OPEN_STT_ARCHIVE)


def open_stt_path(config_: config.Config) -> Path:
    return config_.data_root / prosody.CORPUS_DIR / open_stt_source().relative_path


def require_corpus_sources(config_: config.Config) -> None:
    """Refuse a config whose `data_root` has no corpus sources, before a command writes (#201)."""
    sources = config_.data_root / prosody.CORPUS_DIR / "sources"
    try:
        found = any(sources.iterdir())
    except OSError:
        found = False
    if not found:
        raise config.ConfigError(
            f"{config.config_path()}: data_root has no corpus sources ({sources}); point "
            f"{config.CONFIG_PATH_ENV} at the eval config of docs/eval/PROSODY-CONFIG.md"
        )


def q1(config_: config.Config) -> list[str]:
    """`prosody.q1` (guard, ASR weights, RINCO) and the Open STT archive against its pin."""
    source = open_stt_source()
    problems = prosody.q1(config_)
    if source.sha256 != OPEN_STT_SHA256:
        problems.append(f"corpus pins Open STT {source.sha256}, 0034 {OPEN_STT_SHA256}")
    return problems + corpus.check_source(config_.data_root / prosody.CORPUS_DIR, source)


# --- words ---------------------------------------------------------------------------------


def reference_words(reference: str) -> list[str]:
    """The words of an Open STT reference text, as the sheet shows them and in their order.

    Split on whitespace and normalized as ASR words are (`prosody.normalize`), so that a
    hyphenated word stays one word on both sides of the alignment.
    """
    return [form for token in reference.split() if (form := prosody.normalize(token))]


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa of two binary labelings; None when there is no word or no chance spread."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    chance = left * right + (1 - left) * (1 - right)
    return None if chance == 1 else (observed - chance) / (1 - chance)


# --- selection -----------------------------------------------------------------------------


def order_key(prefix: str, value: str) -> str:
    return hashlib.sha256((prefix + value).encode("utf-8")).hexdigest()


def pilot_members(provenance: dict) -> set[str]:
    return {clip["member"] for clip in provenance["clips"] if clip.get("member")}


def select_fresh(phrases: dict[str, openstt.Phrase], excluded: set[str]) -> tuple[list, dict]:
    """The fresh clips of 0034 in their order, the pool #81 members and the counts of each cut."""
    pool = openstt.pool(phrases)
    pool_stems = {p.stem for p in pool}
    eligible = [p for p in phrases.values() if openstt.eligible(p)]
    long_enough = [p for p in eligible if len(reference_words(p.reference)) >= FRESH_MIN_WORDS]
    outside_pool = [p for p in long_enough if p.stem not in pool_stems]
    available = [p for p in outside_pool if p.wav_member not in excluded]
    available.sort(key=lambda p: order_key(SELECT_PREFIX, p.stem))
    if len(available) < FRESH_COUNT:
        raise ConfirmError(f"{len(available)} fresh clips available, {FRESH_COUNT} needed")
    counts = {
        "phrases": len(phrases),
        "eligible": len(eligible),
        "min_words": len(long_enough),
        "outside_pool_81": len(outside_pool),
        "outside_pilot_1": len(available),
        "chosen": FRESH_COUNT,
    }
    return available[:FRESH_COUNT], {"counts": counts, "pool": sorted(p.wav_member for p in pool)}


def extract_fresh(path: Path, chosen: list[openstt.Phrase], clips_dir: Path, work: Path) -> dict:
    """Each chosen WAV as 16 kHz mono PCM16 in `clips_dir/<stem>.wav`, with its three SHA-256."""
    wanted = {p.wav_member: p.stem for p in chosen}
    found: dict[str, dict] = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            stem = wanted.get(member.name)
            if stem is None or not member.isreg():
                continue
            data = openstt._read(member, archive, openstt.MAX_WAV_BYTES)
            if data is None:
                raise ConfirmError(f"{member.name}: over the limit or shorter than declared")
            temporary = work / f"{stem}.wav"
            corpus._write_bytes(temporary, data)
            samples = corpus.decode(temporary)
            temporary.unlink()
            pcm = gate0.pcm16(samples)
            clip = gate0.wav_bytes(pcm)
            corpus._write_bytes(clips_dir / f"{stem}.wav", clip)
            found[stem] = {
                "member_sha256": hashlib.sha256(data).hexdigest(),
                "wav_sha256": hashlib.sha256(clip).hexdigest(),
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            }
    missing = sorted(set(wanted.values()) - set(found))
    if missing:
        raise ConfirmError(f"{len(missing)} chosen WAVs were not found on the second pass")
    return found


def calibration(files: list[corpus.RincoFile]) -> list[prosody.Utterance]:
    """40 utterances of 4–20 words with both kinds of words, at most one per WAV, in hash order."""
    low, high = CALIBRATION_WORDS
    candidates = [
        u
        for f in files
        for u in prosody.utterances(f)
        if low <= len(u.words) <= high and 0 < sum(w["marked"] for w in u.words) < len(u.words)
    ]
    candidates.sort(key=lambda u: order_key(AGREEMENT_PREFIX, f"{u.member}{u.number}"))
    chosen, used = [], set()
    for utterance in candidates:
        if utterance.member not in used:
            chosen.append(utterance)
            used.add(utterance.member)
        if len(chosen) == CALIBRATION_COUNT:
            return chosen
    raise ConfirmError(
        f"{len(chosen)} calibration utterances available, {CALIBRATION_COUNT} needed"
    )


def series_3_material(config_: config.Config, archive: zipfile.ZipFile) -> list[corpus.RincoFile]:
    """The RINCO files of series 3, refused when their counts differ from its `prepare.json`."""
    excluded = prosody.series_exclusion(config_, SERIES_3, archive)["members"]
    files, counts = prosody.material(archive, excluded)
    recorded = _read_json(series_3_dir(config_) / prosody.PREPARE_FILE)["material"]
    if counts != recorded:
        raise ConfirmError("the RINCO material differs from series 3 prepare.json")
    return files


def series_3_dir(config_: config.Config) -> Path:
    return config_.data_root / prosody.SERIES[SERIES_3]["dir"]


def cut_wav(samples: np.ndarray, start: float, end: float) -> bytes:
    cut = samples[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)]
    return gate0.wav_bytes(gate0.pcm16(cut))


# --- freshness -----------------------------------------------------------------------------


def _wav_hashes(path: Path) -> set[str]:
    data = path.read_bytes()
    found = {hashlib.sha256(data).hexdigest()}
    with contextlib.suppress(gate0.Gate0Error, wave.Error, EOFError, ValueError):
        found.add(hashlib.sha256(gate0.wav_pcm(data)).hexdigest())
    return found


def reference_hashes(config_: config.Config) -> dict[str, set[str]]:
    """SHA-256 of everything the fresh set must not repeat, by where it comes from."""
    with zipfile.ZipFile(prosody.rinco_path(config_)) as archive:
        rinco = {
            hashlib.sha256(corpus.read_member(archive, n, corpus.MAX_MEMBER_BYTES)).hexdigest()
            for n in archive.namelist()
            if n.lower().endswith(".wav")
        }
    pilot_root = config_.data_root / prosody.PILOT_PROVENANCE.parent
    pilot = {
        sha
        for folder in ("clips", openstt.POOL_DIR)
        for wav in sorted((pilot_root / folder).glob("*.wav"))
        for sha in _wav_hashes(wav)
    }
    provenance = _read_json(config_.data_root / prosody.PILOT_PROVENANCE)
    pilot |= {clip["sha256"] for clip in provenance["clips"] if clip.get("sha256")}
    return {"rinco": rinco, "pilot_1": pilot, "gate0": set(prosody.SANITY_CLIPS.values())}


def check_fresh(config_: config.Config, fresh: list[dict], pool_81: list[str]) -> dict:
    """Every hash and member of the fresh clips against the earlier material; `fresh` when none."""
    references = reference_hashes(config_)
    provenance = _read_json(config_.data_root / prosody.PILOT_PROVENANCE)
    reused_members = (pilot_members(provenance) | set(pool_81)) & {c["member"] for c in fresh}
    overlap = sorted(
        {
            clip["stem"]
            for clip in fresh
            for key in ("member_sha256", "wav_sha256", "pcm_sha256")
            for hashes in references.values()
            if clip[key] in hashes
        }
        | {clip["stem"] for clip in fresh if clip["member"] in reused_members}
    )
    return {
        "fresh": not overlap,
        "overlap": overlap,
        "references": {name: len(hashes) for name, hashes in references.items()},
        "clips": len(fresh),
    }


def clip_file_problems(out: Path, fresh: list[dict]) -> list[str]:
    problems = []
    for clip in fresh:
        path = out / CLIPS_DIR / f"{clip['stem']}.wav"
        found = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if found != clip["wav_sha256"]:
            problems.append(f"clip {clip['stem']}: SHA-256 {found}, prepare {clip['wav_sha256']}")
    return problems


def list_sha256(lines: list[str]) -> str:
    return prosody.list_sha256(lines)


# --- the label package ---------------------------------------------------------------------


def neutral_ids(count: int, rng: np.random.Generator) -> list[str]:
    ids: list[str] = []
    while len(ids) < count:
        candidate = f"P-{int(rng.integers(0, 2**32)):08x}"
        if candidate not in ids:
            ids.append(candidate)
    return ids


def sheet_bytes(items: list[dict]) -> bytes:
    """The empty sheet: one row per word of each item, in package order."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SHEET_COLUMNS)
    for item in items:
        for number, word in enumerate(item["words"], 1):
            writer.writerow((item["id"], number, word, "", ""))
    return buffer.getvalue().encode("utf-8")


def build_package(out: Path, items: list[dict], audio: dict[int, bytes], labeling_doc: Path):
    """`label-package/` and `alias-map.json`; items get neutral ids in a seeded shuffle."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    order = [int(i) for i in rng.permutation(len(items))]
    ids = neutral_ids(len(items), rng)
    package = _mkdir(out / PACKAGE_DIR)
    listen = _mkdir(package / LISTEN_DIR)
    placed = []
    for item_id, index in zip(ids, order, strict=True):
        corpus._write_bytes(listen / f"{item_id}.wav", audio[index])
        placed.append({"id": item_id, **items[index]})
    sheet = sheet_bytes(placed)
    corpus._write_bytes(package / SHEET_FILE, sheet)
    corpus._write_bytes(package / labeling_doc.name, labeling_doc.read_bytes())
    _write_json(out / ALIAS_FILE, {"format_version": FORMAT_VERSION, "items": placed})
    return {
        "items": len(placed),
        "fresh": sum(i["kind"] == FRESH for i in placed),
        "calibration": sum(i["kind"] == CALIBRATION for i in placed),
        "rows": sum(len(i["words"]) for i in placed),
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
    }


# --- import --------------------------------------------------------------------------------


def read_sheet(data: bytes) -> list[dict]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ImportRefused("not_utf8") from None
    first = text.split("\n", 1)[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    header = next(reader, None)
    if header is None or tuple(h.strip() for h in header) != SHEET_COLUMNS:
        raise ImportRefused("bad_header", [("cells", len(header or ()))])
    rows = []
    for line, row in enumerate(reader, 2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(SHEET_COLUMNS):
            raise ImportRefused("bad_row", [(f"line {line}", len(row))])
        rows.append(dict(zip(SHEET_COLUMNS, (cell.strip() for cell in row), strict=True)))
    return rows


def validate_sheet(rows: list[dict], items: list[dict]) -> dict[str, dict]:
    """Per item: its marks or its skip reason; `ImportRefused` with every problem otherwise."""
    expected = {(i["id"], n): w for i in items for n, w in enumerate(i["words"], 1)}
    seen: dict[tuple[str, int], dict] = {}
    problems: list[tuple] = []
    for row in rows:
        try:
            key = (row["id"], int(row["word_no"]))
        except ValueError:
            problems.append((row["id"], row["word_no"], "bad_word_no"))
            continue
        if key not in expected:
            known = any(i["id"] == row["id"] for i in items)
            problems.append((*key, "unknown_word" if known else "unknown_id"))
        elif key in seen:
            problems.append((*key, "duplicate_row"))
        else:
            seen[key] = row
            if row["word"] != expected[key]:
                problems.append((*key, "word_changed"))
            elif row["marked"] not in ("", "0", "1"):
                problems.append((*key, "bad_marked"))
    problems += [(*key, "missing_row") for key in expected if key not in seen]
    if problems:
        raise ImportRefused(_one_code(problems, "bad_rows"), problems)
    labels, unlabeled = {}, []
    for item in items:
        found = [seen[(item["id"], n)] for n in range(1, len(item["words"]) + 1)]
        marks = [r["marked"] for r in found]
        reasons = [r["skip_reason"] for r in found if r["skip_reason"]]
        if all(marks) and not reasons:
            labels[item["id"]] = {"skip_reason": None, "marked": [m == "1" for m in marks]}
        elif not any(marks) and reasons:
            labels[item["id"]] = {"skip_reason": reasons[0], "marked": None}
        elif not any(marks):
            unlabeled.append((item["id"], "unlabeled_item"))
        else:
            problems.append((item["id"], "partial_item"))
    if len(unlabeled) == len(items):
        raise ImportRefused("unfilled_template")
    if problems or unlabeled:
        raise ImportRefused(_one_code(problems + unlabeled, "bad_items"), problems + unlabeled)
    return labels


def _one_code(problems: list[tuple], mixed: str) -> str:
    codes = {problem[-1] for problem in problems}
    return codes.pop() if len(codes) == 1 else mixed


def agreement(items: list[dict], labels: dict[str, dict]) -> dict:
    """Cohen's kappa of the person against RINCO over the calibration items they did not skip."""
    pairs, skipped = [], 0
    for item in items:
        if item["kind"] != CALIBRATION:
            continue
        found = labels[item["id"]]
        if found["marked"] is None:
            skipped += 1
            continue
        pairs += list(zip(found["marked"], item["rinco_marked"], strict=True))
    return {"kappa": kappa(pairs), "words": len(pairs), "skipped_items": skipped}


# --- the verdict ---------------------------------------------------------------------------


def decide(coverage: float, pairs: int, clips_with_pairs: int, auc_, ci) -> str:
    """Rule steps 2–4 after every INVALID check passed and the run was not stopped."""
    if (
        coverage < prosody.PASS_COVERAGE
        or pairs < MIN_PAIRS
        or clips_with_pairs < MIN_CLIPS_WITH_PAIRS
    ):
        return INCONCLUSIVE
    if (
        auc_ is not None
        and ci is not None
        and auc_ >= prosody.PASS_AUC
        and ci[0] >= prosody.PASS_CI_LOW
    ):
        return PASS
    return FAIL


def confirm_metric(records: list[dict]) -> dict:
    """`prosody.metric` over the clip records (a clip is its own file) plus the clips with pairs."""
    found = prosody.metric(records)
    for key in ("verdict", "auc_by_section", "files"):
        found.pop(key)
    found["clips"] = len(records)
    found["clips_with_pairs"] = sum(1 for r in records if prosody.pair_counts(r["words"])[1])
    return found


# --- sanity stage --------------------------------------------------------------------------


def r0_keys(config_: config.Config) -> list[dict]:
    """The first `R0_UTTERANCES` records of series 3 in (member, utterance) order."""
    path = series_3_dir(config_) / prosody.MAIN_DIR / prosody.UTTERANCES_FILE
    with open(path, encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    records.sort(key=lambda r: (r["member"], r["utterance"]))
    return records[:R0_UTTERANCES]


def r0_differences(expected: list[dict], found: list[dict]) -> list[str]:
    """Words whose scored state or index differ beyond `R0_TOLERANCE`; empty when reproduced."""
    by_key = {(r["member"], r["utterance"]): r for r in found}
    problems = []
    for record in expected:
        key = (record["member"], record["utterance"])
        again = by_key.get(key)
        if again is None or len(again["words"]) != len(record["words"]):
            problems.append(f"{key[0]}#{key[1]}: not reproduced")
            continue
        for number, (old, new) in enumerate(zip(record["words"], again["words"], strict=True), 1):
            if (old["score"] is None) != (new["score"] is None):
                problems.append(f"{key[0]}#{key[1]} word {number}: scored state differs")
            elif old["score"] and abs(old["score"]["index"] - new["score"]["index"]) > R0_TOLERANCE:
                problems.append(f"{key[0]}#{key[1]} word {number}: index differs")
    return problems


def receipt_problems(found: Path, recorded: Path) -> list[str]:
    try:
        new, old = _read_json(found), _read_json(recorded)
    except (OSError, ValueError) as error:
        return [f"model receipt unreadable: {type(error).__name__}"]
    problems = [f"receipt {k} differs" for k in RECEIPT_KEYS if new.get(k) != old.get(k)]
    for name in RECEIPT_VERSIONS:
        if new.get("versions", {}).get(name) != old.get("versions", {}).get(name):
            problems.append(f"receipt version of {name} differs")
    return problems


def controls() -> dict[str, np.ndarray]:
    return {
        "silence": gate0.silence(CONTROL_SECONDS),
        "pink_noise": gate0.pink_noise(CONTROL_SECONDS, seed=CONTROL_SEED),
        "tone_440": gate0.tone(CONTROL_SECONDS),
    }


def sanity(config_: config.Config, transcribe: prosody.Transcriber, work: Path, deadline: float):
    """R0 over 20 utterances of series 3 and S2' over the synthetic controls."""
    started = time.monotonic()
    expected = r0_keys(config_)
    members = {r["member"] for r in expected}
    with zipfile.ZipFile(prosody.rinco_path(config_)) as archive:
        files = [f for f in series_3_material(config_, archive) if f.wav_member in members]
        records, stopped = prosody.run_material(
            archive, files, transcribe, _mkdir(work / "r0"), work / "r0.jsonl", deadline
        )
    differences = r0_differences(expected, records)
    first = next(iter(sorted((work / "r0").glob("utt-*"))), None)
    receipt = asr.MODEL_RECEIPT_FILE_NAME
    recorded = (
        series_3_dir(config_) / prosody.MAIN_DIR / prosody.RUNS_DIR / "utt-00000/asr" / receipt
    )
    receipts = (
        ["no R0 transcription"]
        if first is None
        else receipt_problems(first / "asr" / receipt, recorded)
    )
    scored = {}
    for name, samples in controls().items():
        found = transcribe(samples, work / f"control-{name}")
        if found["status"] != "completed":
            scored[name] = f"asr_{found['status']}"
            continue
        scored[name] = sum(s is not None for s in prosody.clip_scores(samples, found["words"]))
    r0 = not stopped and not differences and not receipts
    s2 = all(count == 0 for count in scored.values())
    return {
        "r0": r0,
        "r0_utterances": len(expected),
        "r0_differences": differences,
        "r0_stopped": stopped,
        "receipt_problems": receipts,
        "s2": s2,
        "controls_scored_words": scored,
        "passed": r0 and s2,
        "elapsed_s": round(time.monotonic() - started, 1),
    }


# --- main run ------------------------------------------------------------------------------


def read_wav(path: Path, wav_sha256: str) -> np.ndarray:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != wav_sha256:
        raise ConfirmError(f"{path.name}: SHA-256 differs from prepare.json")
    return np.frombuffer(gate0.wav_pcm(data), "<i2").astype(np.float64) / 32768


def run_clips(
    out: Path,
    fresh: list[dict],
    aliases: dict[str, dict],
    labels: dict[str, dict],
    transcribe: prosody.Transcriber,
    work: Path,
    log: Path,
    deadline: float,
) -> tuple[list[dict], int, bool]:
    """Every labelled fresh clip scored and appended to `log`; (records, skipped, stopped)."""
    records, skipped = [], 0
    for number, clip in enumerate(fresh):
        alias = aliases[clip["stem"]]
        found_labels = labels[alias["id"]]
        if found_labels["marked"] is None:
            skipped += 1
            continue
        if time.monotonic() > deadline:
            return records, skipped, True
        samples = read_wav(out / CLIPS_DIR / f"{clip['stem']}.wav", clip["wav_sha256"])
        words = [
            {"form": prosody.normalize(w), "marked": m, "score": None}
            for w, m in zip(alias["words"], found_labels["marked"], strict=True)
        ]
        record = {"member": alias["id"], "folder": "open-stt"}
        empty = {"words": words, "asr_words": 0, "asr_excluded": 0, "asr_unmatched": 0}
        if len(samples) < SAMPLE_RATE * acoustics.MIN_TRACK_SECONDS:
            record |= {"skipped": "too_short", **empty}
        else:
            found = transcribe(samples, work / f"clip-{number:03d}")
            record["asr_flagged_words"] = found["flagged_words"]
            if found["status"] != "completed":
                record |= {"skipped": f"asr_{found['status']}", "reason": found["reason"], **empty}
            else:
                plain = [{k: w[k] for k in ("form", "marked")} for w in words]
                record |= prosody.score_utterance(prosody.track(samples), found["words"], plain)
        records.append(record)
        with open(log, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records, skipped, time.monotonic() > deadline


# --- files ---------------------------------------------------------------------------------


def _mkdir(path: Path) -> Path:
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, DIR_MODE)
    return path


def _write_json(path: Path, data: object) -> None:
    prosody._write_json(path, data)


def _replace_bytes(path: Path, data: bytes) -> None:
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    corpus._write_bytes(staged, data)
    os.replace(staged, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def output_dir(config_: config.Config) -> Path:
    return _mkdir(paths.resolve_output(config_.data_root, OUTPUT_DIR, config_))


def _header(decision: Path) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "experiment": OUTPUT_DIR,
        "decision": str(DECISION_DOC),
        "owner_decision": OWNER_DECISION,
        "spec": SPEC,
        "rule_sha256": check_rule(decision),
        "code_sha256": code_sha256(),
        "frozen_sha256": FROZEN_SHA256,
        "at": prosody._now(),
    }


def _result(out: Path, header: dict, verdict: str, reason: str, **extra) -> dict:
    result = {**header, "verdict": verdict, "reason": reason, **extra}
    _write_json(out / RESULT_FILE, result)
    return result


def _refuse_if_decided(out: Path) -> None:
    if (out / RESULT_FILE).exists():
        raise ConfirmError(f"{RESULT_FILE} exists: the verdict is out, nothing more runs")


def _invalid_reasons(config_: config.Config) -> list[str]:
    return [f"Q1: {p}" for p in q1(config_)] + [f"frozen: {p}" for p in frozen_problems()]


def _prepared(out: Path) -> dict:
    if not (out / PREPARE_FILE).exists():
        raise ConfirmError("run `prepare` first")
    return _read_json(out / PREPARE_FILE)


def _sanity_attempts(out: Path) -> list[dict]:
    return [_read_json(p) for p in sorted((out / SANITY_DIR).glob("attempt-*.json"))]


# --- commands ------------------------------------------------------------------------------


def prepare(
    config_: config.Config, decision: Path = DECISION_DOC, labeling_doc: Path = LABELING_DOC
) -> dict:
    header = _header(decision)
    out = output_dir(config_)
    _refuse_if_decided(out)
    if (out / PREPARE_FILE).exists():
        raise ConfirmError(f"{PREPARE_FILE} exists; prepare runs once")
    for leftover in (CLIPS_DIR, PACKAGE_DIR, ALIAS_FILE):
        if (out / leftover).exists():
            raise ConfirmError(f"{leftover} is left from a broken prepare; remove it by hand")
    if not labeling_doc.is_file():
        raise ConfirmError(f"{labeling_doc.name} is missing")
    reasons = _invalid_reasons(config_)
    if reasons:
        return _result(out, header, INVALID, "; ".join(reasons))
    provenance = _read_json(config_.data_root / prosody.PILOT_PROVENANCE)
    phrases, scan_stats = openstt.scan(open_stt_path(config_))
    chosen, selection = select_fresh(phrases, pilot_members(provenance))
    work = _mkdir(out / ".prepare-work")
    clips_dir = _mkdir(out / CLIPS_DIR)
    hashes = extract_fresh(open_stt_path(config_), chosen, clips_dir, work)
    fresh = [
        {
            "stem": p.stem,
            "member": p.wav_member,
            **hashes[p.stem],
            "duration_s": round(p.duration_s, 3),
            "words": len(reference_words(p.reference)),
        }
        for p in chosen
    ]
    items = [
        {"kind": FRESH, "stem": p.stem, "wav_sha256": hashes[p.stem]["wav_sha256"],
         "words": reference_words(p.reference)}
        for p in chosen
    ]  # fmt: skip
    audio = {i: (clips_dir / f"{p.stem}.wav").read_bytes() for i, p in enumerate(chosen)}
    with zipfile.ZipFile(prosody.rinco_path(config_)) as archive:
        agreement_set = calibration(series_3_material(config_, archive))
        decoded: dict[str, np.ndarray] = {}
        for utterance in agreement_set:
            source = work / f"rinco-{len(decoded)}.wav"
            decoded[utterance.member] = corpus.decode(
                corpus.copy_member(archive, utterance.member, source)
            )
            source.unlink()
    calibration_items = []
    for utterance in agreement_set:
        clip = cut_wav(decoded[utterance.member], utterance.start, utterance.end)
        audio[len(items)] = clip
        item = {
            "kind": CALIBRATION,
            "member": utterance.member,
            "utterance": utterance.number,
            "wav_sha256": hashlib.sha256(clip).hexdigest(),
            "words": [w["form"] for w in utterance.words],
            "rinco_marked": [w["marked"] for w in utterance.words],
        }
        items.append(item)
        calibration_items.append(item)
    os.rmdir(work)
    fresh_check = check_fresh(config_, fresh, selection["pool"])
    record = {
        **header,
        "q1": "ok",
        "frozen": "ok",
        "open_stt_scan": dict(scan_stats),
        "selection": selection["counts"],
        "pool_81_members": selection["pool"],
        "fresh_clips": fresh,
        "fresh_list_sha256": list_sha256([f"{c['stem']} {c['wav_sha256']}" for c in fresh]),
        "calibration": [
            {k: c[k] for k in ("member", "utterance", "wav_sha256")}
            | {"words": len(c["words"]), "marked_words": sum(c["rinco_marked"])}
            for c in calibration_items
        ],
        "calibration_list_sha256": list_sha256(
            [f"{c['member']} {c['utterance']} {c['wav_sha256']}" for c in calibration_items]
        ),
        "fresh_check": fresh_check,
    }
    if not fresh_check["fresh"]:
        _write_json(out / PREPARE_FILE, record)
        return _result(out, header, INVALID, f"check-fresh: {len(fresh_check['overlap'])} reused")
    record["package"] = build_package(out, items, audio, labeling_doc)
    _write_json(out / PREPARE_FILE, record)
    return record


def run_check_fresh(config_: config.Config, decision: Path = DECISION_DOC) -> dict:
    header = _header(decision)
    out = output_dir(config_)
    prepared = _prepared(out)
    problems = clip_file_problems(out, prepared["fresh_clips"])
    found = check_fresh(config_, prepared["fresh_clips"], prepared["pool_81_members"])
    found |= {"clip_files": problems or "ok"}
    if (problems or not found["fresh"]) and not (out / RESULT_FILE).exists():
        _result(out, header, INVALID, "check-fresh: " + (f"{problems}" if problems else "reused"))
    return found


def import_sheet(config_: config.Config, sheet: Path, decision: Path = DECISION_DOC) -> dict:
    header = _header(decision)
    out = output_dir(config_)
    _refuse_if_decided(out)
    prepared = _prepared(out)
    if (out / MAIN_DIR).exists():
        raise ImportRefused("run_started")
    if (out / LABELS_FILE).exists():
        raise ImportRefused("already_imported")
    data = sheet.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha == prepared["package"]["sheet_sha256"]:
        raise ImportRefused("unfilled_template")
    items = _read_json(out / ALIAS_FILE)["items"]
    labels = validate_sheet(read_sheet(data), items)
    found = agreement(items, labels)
    imported = _mkdir(out / IMPORT_DIR)
    _replace_bytes(imported / SHEET_FILE, data)
    record = {
        **header,
        "sheet_sha256": sha,
        "agreement": found,
        "items": labels,
        "counts": {
            kind: {
                "labelled": sum(
                    1 for i in items if i["kind"] == kind and labels[i["id"]]["marked"] is not None
                ),
                "skipped": sum(
                    1 for i in items if i["kind"] == kind and labels[i["id"]]["marked"] is None
                ),
            }
            for kind in (FRESH, CALIBRATION)
        },
    }
    _replace_bytes(
        out / LABELS_FILE,
        (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    return {k: record[k] for k in ("sheet_sha256", "agreement", "counts")}


def run_sanity(
    config_: config.Config,
    decision: Path = DECISION_DOC,
    transcribe=None,
    scope: tuple[str, str] | None | bool = True,
    wall_seconds: float = SANITY_WALL_SECONDS,
) -> dict:
    header = _header(decision)
    out = output_dir(config_)
    _refuse_if_decided(out)
    _prepared(out)
    attempts = _mkdir(out / SANITY_DIR)
    if any(
        a["passed"] and a["code_sha256"] == header["code_sha256"] for a in _sanity_attempts(out)
    ):
        raise ConfirmError("a sanity attempt of this code has passed already; run the main step")
    started = len(list(attempts.glob("work-*")))  # an attempt that died without a record counts
    if started >= MAX_SANITY_ATTEMPTS:
        reason = f"санитарная стадия не пройдена за {started} попытки с этим кодом"
        return _result(out, header, INVALID, reason, attempts=started)
    scope = prosody._checked_scope(scope)
    cpus = prosody.pin_cpus()
    reasons = _invalid_reasons(config_)
    if reasons:
        return _result(out, header, INVALID, "; ".join(reasons))
    number = started + 1
    work = attempts / f"work-{number}"
    os.mkdir(work, DIR_MODE)
    try:
        with prosody._reused_model():
            found = sanity(
                config_,
                transcribe or prosody.local_transcriber(config_),
                work,
                time.monotonic() + wall_seconds,
            )
    except Exception as error:  # a broken attempt is a failed attempt, recorded and counted
        found = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    record = {**header, "attempt": number, "cpus": cpus, "memory_scope": scope,
              "wall_seconds": wall_seconds, **found}  # fmt: skip
    _write_json(attempts / f"attempt-{number}.json", record)
    if not record["passed"] and number >= MAX_SANITY_ATTEMPTS:
        _result(out, header, INVALID, f"санитарная стадия не пройдена за {number} попытки")
    return record


def run_main(
    config_: config.Config,
    decision: Path = DECISION_DOC,
    transcribe=None,
    scope: tuple[str, str] | None | bool = True,
    wall_seconds: float = MAIN_WALL_SECONDS,
) -> dict:
    header = _header(decision)
    out = output_dir(config_)
    _refuse_if_decided(out)
    main_dir = out / MAIN_DIR
    if (main_dir / STARTED_FILE).exists():
        reason = "основной прогон остановлен без результата (лимит ресурсов или kill); повтора нет"
        return _result(out, header, INCONCLUSIVE, reason)
    if main_dir.exists():
        raise ConfirmError(f"{MAIN_DIR}/ exists: the main run starts once")
    prepared = _prepared(out)
    if not (out / LABELS_FILE).exists():
        raise ConfirmError("no imported labels: run `import <sheet.csv>` after the labelling")
    passed = [a for a in _sanity_attempts(out) if a["passed"]]
    if not passed or passed[-1]["code_sha256"] != header["code_sha256"]:
        raise ConfirmError("no passed sanity attempt of this code; run `sanity` first")
    scope = prosody._checked_scope(scope)
    labels = _read_json(out / LABELS_FILE)
    reasons = _invalid_reasons(config_)
    imported = out / IMPORT_DIR / SHEET_FILE
    found_sha = hashlib.sha256(imported.read_bytes()).hexdigest() if imported.is_file() else None
    if found_sha != labels["sheet_sha256"]:
        reasons.append(f"sheet SHA-256 {found_sha}, recorded at import {labels['sheet_sha256']}")
    reasons += clip_file_problems(out, prepared["fresh_clips"])
    fresh_check = check_fresh(config_, prepared["fresh_clips"], prepared["pool_81_members"])
    if not fresh_check["fresh"]:
        reasons.append(f"check-fresh: {len(fresh_check['overlap'])} reused clips")
    agreement_ = labels["agreement"]
    if agreement_["kappa"] is None or agreement_["kappa"] < KAPPA_MIN:
        reasons.append(f"каппа {agreement_['kappa']} < {KAPPA_MIN}")
    if reasons:
        return _result(out, header, INVALID, "; ".join(reasons), agreement=agreement_)
    aliases = {i["stem"]: i for i in _read_json(out / ALIAS_FILE)["items"] if i["kind"] == FRESH}
    _mkdir(main_dir)
    cpus = prosody.pin_cpus()
    started = time.monotonic()
    _write_json(main_dir / STARTED_FILE, {**header, "cpus": cpus, "memory_scope": scope})
    try:
        with prosody._reused_model():
            records, skipped, stopped = run_clips(
                out,
                prepared["fresh_clips"],
                aliases,
                labels["items"],
                transcribe or prosody.local_transcriber(config_),
                _mkdir(main_dir / RUNS_DIR),
                main_dir / CLIPS_LOG,
                started + wall_seconds,
            )
    except MemoryError:
        return _result(out, header, INCONCLUSIVE, "основной прогон остановлен лимитом памяти")
    except Exception as error:  # the main run starts once: its failure is recorded, not retried
        reason = f"основной прогон прерван: {type(error).__name__}: {error}"
        return _result(out, header, INVALID, reason)
    elapsed = round(time.monotonic() - started, 1)
    common = {"agreement": agreement_, "skipped_by_labeler": skipped, "elapsed_s": elapsed}
    if stopped:
        reason = f"остановлено потолком wall time {wall_seconds / 60:g} мин"
        return _result(out, header, INCONCLUSIVE, reason, clips_done=len(records), **common)
    found = confirm_metric(records)
    verdict = decide(
        found["coverage"], found["pairs"], found["clips_with_pairs"], found["auc"], found["ci95"]
    )
    result = {**header, "verdict": verdict, **found, **common, "cpus": cpus,
              "fresh_list_sha256": prepared["fresh_list_sha256"],
              "sheet_sha256": labels["sheet_sha256"]}  # fmt: skip
    _write_json(out / RESULT_FILE, result)
    return result


SUMMARY_KEYS = (
    "verdict", "reason", "passed", "r0", "s2", "auc", "ci95", "coverage", "pairs",
    "clips_with_pairs", "q1", "fresh", "overlap", "references", "clip_files", "sheet_sha256",
    "agreement", "counts", "selection", "package", "fresh_list_sha256", "calibration_list_sha256",
)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.prosody_confirm")
    parser.add_argument("--decision", type=Path, default=DECISION_DOC)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check-fresh", "sanity", "run"):
        commands.add_parser(name)
    commands.add_parser("import").add_argument("sheet", type=Path)
    args = parser.parse_args(argv)
    try:
        config_ = config.load()
        require_corpus_sources(config_)
        guard.install_network_guard(config_)
        if args.command == "import":
            result = import_sheet(config_, args.sheet, args.decision)
        else:
            step = {"prepare": prepare, "check-fresh": run_check_fresh, "sanity": run_sanity,
                    "run": run_main}[args.command]  # fmt: skip
            result = step(config_, args.decision)
    except (
        ConfirmError,
        prosody.ProsodyError,
        corpus.CorpusError,
        openstt.OpenSttError,
        config.ConfigError,
        paths.PathDenied,
        OSError,  # a read-only archive, such as a frozen data directory (#201)
    ) as error:
        print(f"prosody-confirm: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in SUMMARY_KEYS if k in result}, ensure_ascii=False))
    if args.command == "check-fresh" and not (result["fresh"] and result["clip_files"] == "ok"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
