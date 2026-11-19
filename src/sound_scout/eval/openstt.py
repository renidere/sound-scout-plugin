"""Control "speech with ASR errors" from Open STT `asr_calls_2_val`: dev-only candidates (#81).

`python -m sound_scout.eval.openstt add --corpus-dir DIR --manifest DIR/<name>/manifest.json`
re-hashes the archive against `corpus.EXTRA_SOURCES` and reads it as a stream, never extracting it
by its own paths: only regular files named like `MEMBER` are taken, within the size limits; the
rest is counted and skipped. A pool of up to `POOL_MAX` phrases (2.5–15 s, at least four words,
Cyrillic only) in `sha256(stem)` order is decoded into `<name>/openstt-pool/`. Every Open STT clip
is `dev` by rule: no held-out record of the manifest is read or opened. The local ASR runs over
the pool the `analyze` way (`asr_review.transcribe_file`: the configured cached model, offline,
runs under `<data_root>/openstt-select/<name>/`) until `MAX_ASR_CANDIDATES` phrases differ from
the human transcript of Open STT by a WER of at least `WER_MIN`, or the pool ends. The
configuration must allow `<name>/openstt-pool/` as input, for example with the corpus directory as
an allowed root.

A difference is no confirmation: each phrase enters the manifest as a `corrupted_asr` control
with `asr_error_status: candidate`, and only a person who fills `asr-review/<id>.json` (written by
`asr_review.review` right after, with the human fields empty) makes it `confirmed` — never this
code. Fewer than `MIN_ASR_CANDIDATES` differing phrases leave the manifest unchanged, exit code 1;
when no ASR call completes, the first problem is reported with exit code 2.
The reference and ASR texts stay in `openstt-selection.json` next to the corpus, outside git;
stdout carries only ids, WER and status: these are recordings of real calls.
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import wave
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sound_scout import config, guard, paths
from sound_scout.eval import asr_review, corpus, gate0, manifest

MEMBER = re.compile(r"asr_calls_2_val/[0-9a-f]+/[0-9a-f]+/([0-9a-f]{12})\.(wav|txt)")
MAX_WAV_BYTES = 4 << 20
MAX_TXT_BYTES = 4 << 10
WAV_HEAD_BYTES = 4096
POOL_MIN_S = 2.5
POOL_MAX_S = 15.0
MIN_WORDS = 4
POOL_MAX = 120
WER_MIN = 0.25
MIN_ASR_CANDIDATES = 3
MAX_ASR_CANDIDATES = 5
MIN_FREE_BYTES = 1 << 30
ID_PREFIX = "ostt-"
POOL_DIR = "openstt-pool"
RUNS_DIR = "openstt-select"
SELECTION_FILE = "openstt-selection.json"
BACKUP_SUFFIX = ".bak-81-openstt"
PROVENANCE_KIND = "openstt_asr_candidate"
ISSUE = "https://github.com/renidere/sound-scout-plugin/issues/81"
DECISION = "docs/decisions/0012-synthetic-diction-and-openstt.md"
_NOT_CYRILLIC = re.compile(r"[0-9A-Za-z]")
_NOT_LETTER = re.compile(r"[^\w\s]|[\d_]")


class OpenSttError(Exception):
    """The archive or the manifest cannot be used; nothing was changed."""


class Shortfall(OpenSttError):
    """Fewer than `MIN_ASR_CANDIDATES` phrases differ; the manifest is unchanged."""


@dataclass
class Phrase:
    stem: str
    wav_member: str = ""
    txt_member: str = ""
    reference: str = ""
    duration_s: float = 0.0


Transcriber = Callable[[Path, str, str], tuple[str | None, dict]]


def normalize(text: str) -> str:
    """Lower case, ё → е, letters and single spaces only."""
    text = _NOT_LETTER.sub(" ", text.lower().replace("ё", "е"))
    return " ".join(text.split())


def wer(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance over the reference word count, both texts normalized."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        raise ValueError("an empty reference has no WER")
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            substitution = previous[j - 1] + (word != other)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[-1] / len(ref)


def _read(member: tarfile.TarInfo, archive: tarfile.TarFile, limit: int) -> bytes | None:
    stream = archive.extractfile(member)
    if stream is None:
        return None
    data = stream.read(limit + 1)
    return None if len(data) > limit or len(data) != member.size else data


def scan(path: Path) -> tuple[dict[str, Phrase], Counter]:
    """Pass 1: the reference text and the WAV header duration of every well-named pair."""
    phrases: dict[str, Phrase] = {}
    stats: Counter = Counter()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            match = MEMBER.fullmatch(member.name)
            if not member.isreg() or match is None or ".." in member.name.split("/"):
                stats["skipped_name_or_type"] += 1
                continue
            stem, extension = match.groups()
            limit = MAX_WAV_BYTES if extension == "wav" else MAX_TXT_BYTES
            if member.size > limit:
                stats["skipped_size"] += 1
                continue
            phrase = phrases.setdefault(stem, Phrase(stem))
            if extension == "txt":
                data = _read(member, archive, limit)
                try:
                    phrase.reference = data.decode("utf-8").strip() if data else ""
                except UnicodeDecodeError:
                    stats["skipped_text"] += 1
                    continue
                phrase.txt_member = member.name
            else:
                stream = archive.extractfile(member)
                head = stream.read(WAV_HEAD_BYTES) if stream else b""
                try:
                    with wave.open(io.BytesIO(head), "rb") as source:
                        phrase.duration_s = source.getnframes() / source.getframerate()
                except (wave.Error, EOFError, ZeroDivisionError):
                    stats["skipped_wav_header"] += 1
                    continue
                phrase.wav_member = member.name
            stats["members"] += 1
    return phrases, stats


def eligible(phrase: Phrase) -> bool:
    return (
        bool(phrase.wav_member and phrase.txt_member)
        and POOL_MIN_S <= phrase.duration_s <= POOL_MAX_S
        and not _NOT_CYRILLIC.search(phrase.reference)
        and len(normalize(phrase.reference).split()) >= MIN_WORDS
        and all("а" <= ch <= "я" or ch == " " for ch in normalize(phrase.reference))
    )


def pool(phrases: dict[str, Phrase]) -> list[Phrase]:
    """Up to `POOL_MAX` eligible phrases in `sha256(stem)` order: one archive, one pool."""
    chosen = [p for p in phrases.values() if eligible(p)]
    chosen.sort(key=lambda p: hashlib.sha256(p.stem.encode()).hexdigest())
    return chosen[:POOL_MAX]


def extract_pool(path: Path, chosen: list[Phrase], pool_dir: Path, work: Path) -> dict[str, str]:
    """Pass 2: each pool WAV, read by name into a file the code names, as 16 kHz mono PCM16."""
    wanted = {p.wav_member: p.stem for p in chosen}
    hashes: dict[str, str] = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            stem = wanted.get(member.name)
            if stem is None or not member.isreg():
                continue
            data = _read(member, archive, MAX_WAV_BYTES)
            if data is None:
                raise OpenSttError(f"{member.name}: over the limit or shorter than declared")
            temporary = work / f"{stem}.wav"
            corpus._write_bytes(temporary, data)
            samples = corpus.decode(temporary)
            temporary.unlink()
            clip = gate0.wav_bytes(gate0.pcm16(samples))
            target = pool_dir / f"{stem}.wav"
            staged = pool_dir / f".{stem}.wav.{os.getpid()}.tmp"
            corpus._write_bytes(staged, clip)
            os.replace(staged, target)
            hashes[stem] = hashlib.sha256(clip).hexdigest()
    missing = sorted(set(wanted.values()) - set(hashes))
    if missing:
        raise OpenSttError(f"{len(missing)} pool WAVs were not found on the second pass")
    return hashes


def select(chosen: list[Phrase], hashes: dict[str, str], pool_dir: Path, transcribe: Transcriber):
    """ASR over the pool in order until `MAX_ASR_CANDIDATES` differ; (candidates, every outcome)."""
    candidates, outcomes = [], []
    for phrase in chosen:
        path = pool_dir / f"{phrase.stem}.wav"
        problem, info = transcribe(path, hashes[phrase.stem], phrase.stem)
        outcome = {
            "stem": phrase.stem,
            "member": phrase.wav_member,
            "duration_s": phrase.duration_s,
            "reference": phrase.reference,
            "asr_status": "completed" if problem is None else "failed",
            "asr_problem": problem,
            "asr_text": info.get("text"),
            "wer": None if problem is not None else round(wer(phrase.reference, info["text"]), 4),
            "asr": {k: v for k, v in info.items() if k != "text"},
            "sha256": hashes[phrase.stem],
        }
        outcomes.append(outcome)
        if outcome["wer"] is not None and outcome["wer"] >= WER_MIN:
            candidates.append(outcome)
            if len(candidates) >= MAX_ASR_CANDIDATES:
                break
    candidates.sort(key=lambda c: (-c["wer"], c["stem"]))
    return candidates, outcomes


def local_transcriber(config_: config.Config, manifest_name: str) -> Transcriber:
    runs_dir = asr_review._mkdir(
        asr_review._mkdir(config_.data_root / RUNS_DIR, config_) / manifest_name, config_
    )

    def transcribe(path: Path, sha256: str, name: str) -> tuple[str | None, dict]:
        problem, transcription, run_dir = asr_review.transcribe_file(
            path, sha256, name, runs_dir, config_, kind=RUNS_DIR
        )
        if transcription is None:
            return problem, {}
        raw = transcription.raw
        return None, {
            "text": raw["text"],
            "run": run_dir.relative_to(config_.data_root.resolve()).as_posix(),
            "raw_sha256": transcription.raw_sha256,
            "model": {key: raw["model"][key] for key in ("repo", "revision")},
        }

    return transcribe


def add(corpus_dir: Path, manifest_path: Path, transcribe: Transcriber) -> tuple[list, dict]:
    """Select the candidates and add them to the draft manifest; ([(record, WER)], summary).

    `Shortfall` when fewer than `MIN_ASR_CANDIDATES` differ: `openstt-selection.json` records
    the pool outcomes, the manifest and `labels/` stay as they were.
    """
    corpus_dir, manifest_path = Path(corpus_dir), Path(manifest_path)
    root = manifest_path.parent
    source = next(s for s in corpus.EXTRA_SOURCES if s.name == corpus.OPEN_STT_ARCHIVE)
    problems = corpus.check_source(corpus_dir, source)
    if problems:
        raise OpenSttError("the archive does not match its pin: " + "; ".join(problems))
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus.DRAFT_NOTE not in (document.get("note") or ""):
        raise OpenSttError("the manifest is no draft (its note lacks corpus.DRAFT_NOTE)")
    if any(r["id"].startswith(ID_PREFIX) for r in document["records"]):
        raise OpenSttError(f"the manifest already holds {ID_PREFIX}* records")
    taken = sorted(
        path.relative_to(root).as_posix()
        for folder in ("clips", "labels", manifest.ASR_REVIEW_DIR)
        for path in (root / folder).glob(ID_PREFIX + "*")
    )
    if taken:
        raise OpenSttError(f"already there, nothing written: {', '.join(taken)}")
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise OpenSttError(f"not enough free space: need {MIN_FREE_BYTES} B, free {free} B")
    archive = corpus_dir / source.relative_path
    phrases, stats = scan(archive)
    chosen = pool(phrases)
    pool_dir = root / POOL_DIR
    pool_dir.mkdir(mode=corpus.DIR_MODE, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".openstt-81.", dir=root))
    try:
        hashes = extract_pool(archive, chosen, pool_dir, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    candidates, outcomes = select(chosen, hashes, pool_dir, transcribe)
    completed = [o for o in outcomes if o["asr_status"] == "completed"]
    summary = {
        "archive": source.relative_path,
        "archive_sha256": source.sha256,
        "scan": dict(stats),
        "pairs": sum(1 for p in phrases.values() if p.wav_member and p.txt_member),
        "pool": len(chosen),
        "transcribed": len(completed),
        "asr_failed": len(outcomes) - len(completed),
        "candidates": len(candidates),
    }
    selection = {
        "format_version": 1,
        "issue": ISSUE,
        "decision": DECISION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "license": source.license,
        "rule": {"wer_min": WER_MIN, "pool_max": POOL_MAX, "min": MIN_ASR_CANDIDATES,
                 "max": MAX_ASR_CANDIDATES, "duration_s": [POOL_MIN_S, POOL_MAX_S],
                 "min_words": MIN_WORDS},
        **summary,
        "outcomes": outcomes,
    }  # fmt: skip
    corpus._replace_json(root / SELECTION_FILE, selection)
    if outcomes and not completed:
        raise OpenSttError(
            f"no ASR call completed ({len(outcomes)} failed), the first: "
            f"{outcomes[0]['asr_problem']}; the manifest is unchanged"
        )
    if len(candidates) < MIN_ASR_CANDIDATES:
        raise Shortfall(
            f"{len(candidates)} of {len(completed)} transcribed pool phrases differ by WER ≥ "
            f"{WER_MIN} ({summary['asr_failed']} ASR calls failed), at least "
            f"{MIN_ASR_CANDIDATES} needed; the manifest is unchanged"
        )
    records, provenance = [], []
    for candidate in candidates:
        clip_id = ID_PREFIX + candidate["stem"]
        data = (pool_dir / f"{candidate['stem']}.wav").read_bytes()
        corpus._write_bytes(root / "clips" / f"{clip_id}.wav", data)
        duration = (len(data) - 44) / 2 / corpus.SAMPLE_RATE
        corpus._write_json(
            root / "labels" / f"{clip_id}.json",
            corpus.label_template(clip_id, candidate["sha256"], duration),
        )
        records.append({
            "id": clip_id, "path": f"clips/{clip_id}.wav", "sha256": candidate["sha256"],
            "duration_s": duration, "split": "dev", "source_group": clip_id, "categories": [],
            "negative_control": True, "control_kind": "corrupted_asr",
            "asr_error_status": "candidate",
            "note": f"{corpus.ASR_UNCONFIRMED}; Open STT asr_calls_2_val {candidate['member']}, "
                    f"CC BY-NC 4.0; автоматическое расхождение ASR с эталоном (WER "
                    f"{candidate['wer']:.2f}) — только отбор",
        })  # fmt: skip
        provenance.append({
            "id": clip_id, "sha256": candidate["sha256"], "kind": PROVENANCE_KIND, "issue": ISSUE,
            "decision": DECISION, "source": source.relative_path,
            "source_sha256": source.sha256, "member": candidate["member"],
            "wer": candidate["wer"], "asr": candidate["asr"],
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
    return list(zip(records, [c["wer"] for c in candidates], strict=True)), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.openstt")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("add", help="add dev corrupted_asr candidates from Open STT")
    command.add_argument("--corpus-dir", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config_ = config.load()
        guard.install_network_guard(config_)
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
        transcribe = local_transcriber(config_, document["name"])
        added, summary = add(args.corpus_dir, args.manifest, transcribe)
        reviewed = asr_review.review(args.manifest, config_)
    except Shortfall as error:
        print(f"openstt: {error}", file=sys.stderr)
        return 1
    except (
        OpenSttError,
        asr_review.ReviewError,
        corpus.CorpusError,
        config.ConfigError,
        paths.PathDenied,
        tarfile.TarError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"openstt: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary))
    sheets = {o["id"]: o["outcome"] for o in reviewed["reviewed"]}
    for record, wer_value in added:
        print(json.dumps({"id": record["id"], "wer": wer_value,
                          "asr_error_status": record["asr_error_status"],
                          "sheet": sheets.get(record["id"])}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
