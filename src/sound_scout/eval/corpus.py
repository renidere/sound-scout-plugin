"""Pilot qualification corpus from open datasets: the pinned sources and `prepare` (#78).

`SOURCES` is the pinned table of the files the owner allowed to download on 2026-11-19: URL, size,
the publisher's checksum and the SHA-256 this repository recorded. It is plain data: the download
itself is `scripts/fetch_corpus_sources.py`, outside the installed package (R09).

`python -m sound_scout.eval.corpus prepare --corpus-dir DIR --name pilot-1` re-hashes every
source against `SOURCES`, then builds `DIR/<name>/`: negative controls (silence, seeded pink
noise, DEMAND backgrounds, MS-SNSD babble and neighbour speech, RINCO speech mixed with babble),
prosody candidates cut from RINCO by its Accent / Pause / Int phrase tiers, a draft manifest,
`provenance.json` and an empty label template per clip. The MS-SNSD background clips carry
intelligible speech (the owner listened on 2026-11-19, #84): their audio is written and kept, but
they are no noise controls and stay out of the manifest (`EXCLUDED_CONTROLS`); the speech + babble
mixes enter it as `corrupted_asr` candidates (`asr_error_status: candidate`) until a person
confirms the ASR error. `revise --manifest PATH` applies the same two rules to a draft manifest
built before them, in place and idempotently; it replaces a missing or #78 draft `note` with
`MANIFEST_NOTE` and appends the exclusion reason to any other note. Archive members are read by
name into temporary files the code names, never extracted by the archive's own paths; the
Transcribe tier of RINCO (an automatic transcript, a model output) is never read. No model, no
ASR, no network. The category mapping and the rules are docs/decisions/0009-corpus-pilot-1.md.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import wave
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np

from sound_scout import __version__, proc
from sound_scout.eval import gate0

HASH_CHUNK = 1 << 20
RINCO_VERSION = "v0.1, DOI 10.5281/zenodo.8119302"
DEMAND_VERSION = "1.0, DOI 10.5281/zenodo.1227121"
MS_SNSD_COMMIT = "fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742"
RINCO_LICENSE = "CC BY 4.0"
DEMAND_LICENSE = "CC BY-SA 3.0 (текст записи; поле Zenodo — CC BY 4.0, берётся более строгая)"
MS_SNSD_LICENSE = "CC0 (шумы Freesound по README MS-SNSD; код MIT)"
_ZENODO = "https://zenodo.org/records/{record}/files/{name}?download=1"
_MS_SNSD = f"https://raw.githubusercontent.com/microsoft/MS-SNSD/{MS_SNSD_COMMIT}/"


@dataclass(frozen=True)
class Source:
    """One pinned file; `checksum` is the publisher's `algorithm:hex` (md5 or git-blob-sha1)."""

    dataset: str
    name: str
    url: str
    size: int
    checksum: str
    sha256: str | None
    version: str
    license: str

    @property
    def relative_path(self) -> str:
        return f"sources/{self.dataset}/{self.name}"


def _zenodo(record, dataset, name, size, md5, sha256, version, license_) -> Source:
    url = _ZENODO.format(record=record, name=name)
    return Source(dataset, name, url, size, f"md5:{md5}", sha256, version, license_)


def _snsd(member: str, size: int, blob: str, sha256: str | None) -> Source:
    name = member.rsplit("/", 1)[1]
    return Source(
        "ms-snsd",
        name,
        _MS_SNSD + member,
        size,
        f"git-blob-sha1:{blob}",
        sha256,
        f"commit {MS_SNSD_COMMIT}, {member}",
        MS_SNSD_LICENSE,
    )


SOURCES: tuple[Source, ...] = (
    _zenodo(
        8119302,
        "rinco",
        "RINCO_Russian_Intonation_Corpus-v01.zip",
        291339598,
        "63ff0a4366e82cfaac21a220dba5f27b",
        "243688e816bfdf9adbfd65ef5c6104acccd0742ec9bb78b00edb2a299a7148f1",
        RINCO_VERSION,
        RINCO_LICENSE,
    ),
    _zenodo(
        8119302,
        "rinco",
        "help-markup-ru.pdf",
        101972,
        "ba07834ca3e32193198688eb1644e479",
        "78ebf0c95050f86abe4aeb9daa142f70dfb1a8f42b6d5357078f37af1e094218",
        RINCO_VERSION,
        RINCO_LICENSE,
    ),
    _zenodo(
        8119302,
        "rinco",
        "rubrikator-2011.pdf",
        217274,
        "7cf8332d1b7ebf46cce2ef5501f3bee5",
        "58eb4b0a4d953f2298486fa3d32ba052d56dd17132e87043419399262fa2e189",
        RINCO_VERSION,
        RINCO_LICENSE,
    ),
    _zenodo(
        1227121,
        "demand",
        "DLIVING_16k.zip",
        80170627,
        "46741384d9e434a0bd8b3ec1830b6052",
        "2b1726fe06e41551ce2397f2aaf3e4fb692c912d81914b04708df0bbb5252338",
        DEMAND_VERSION,
        DEMAND_LICENSE,
    ),
    _zenodo(
        1227121,
        "demand",
        "OOFFICE_16k.zip",
        88995191,
        "7b61cc2d182d5a654cb9c3101ddd4041",
        "7570c952f621990d0d71b8398107d9a78066254f883a5569abec9c233ffda358",
        DEMAND_VERSION,
        DEMAND_LICENSE,
    ),
    _zenodo(
        1227121,
        "demand",
        "DKITCHEN_16k.zip",
        110501049,
        "7ffbf52d7f4699f96927846103dc8788",
        "9d68e46d2709847c59580770e2f8aa160607ebff7475788174aa1080332cc845",
        DEMAND_VERSION,
        DEMAND_LICENSE,
    ),
    _snsd(
        "noise_test/Babble_2.wav",
        981122,
        "3c426baec1c75b4866f88e256b4e55165ae24ccb",
        "b7f9872b1e4aef0949672d577cdd00132273c2b53efb4a23add549ee0cbbad90",
    ),
    _snsd(
        "noise_test/Babble_6.wav",
        686764,
        "8efd80a6a452657dc7b8097d1a0c59560d195fdf",
        "c355c815d3cc34e0826fb1ba72fab8c3d221d7be4b25e629932b8a7e083d4af5",
    ),
    _snsd(
        "noise_train/NeighborSpeaking_2.wav",
        706052,
        "aba9083a05dc576ed56feb5dfa89bd6f67586a31",
        "dff6281d248beeff5458b4fcb71387c19f62982eecafd00b78e3a0a6a8c6dc52",
    ),
    _snsd(
        "noise_train/NeighborSpeaking_4.wav",
        760012,
        "f8009b35aff8a2bcbd6a069852a8e773e60c966d",
        "8b179254d5b8c039563cf5206531dbcc0bb24ef7bf077456c503d17d4d083d47",
    ),
)


class CorpusError(Exception):
    """The corpus cannot be built or a source does not match its pin; the message says why."""


def publisher_hasher(source: Source):
    """A hash object for the publisher's checksum; a git blob hash starts with its header."""
    algorithm, _ = source.checksum.split(":", 1)
    if algorithm == "md5":
        return hashlib.md5(usedforsecurity=False)
    if algorithm == "git-blob-sha1":
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {source.size}\0".encode())
        return digest
    raise ValueError(f"unknown checksum algorithm {algorithm!r}")


def file_digests(path: Path, source: Source) -> tuple[int, str, str]:
    """Size, `algorithm:hex` publisher checksum and SHA-256 of the file as it is on disk now."""
    publisher = publisher_hasher(source)
    sha256 = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while chunk := stream.read(HASH_CHUNK):
            size += len(chunk)
            publisher.update(chunk)
            sha256.update(chunk)
    algorithm = source.checksum.split(":", 1)[0]
    return size, f"{algorithm}:{publisher.hexdigest()}", sha256.hexdigest()


def digest_mismatches(
    source: Source, size: int, checksum: str, sha256: str, *, require_pin: bool = True
) -> list[str]:
    """Every way the measured digests differ from the pinned ones.

    An unpinned SHA-256 is a mismatch unless `require_pin` is false (the first download only).
    """
    problems = []
    if size != source.size:
        problems.append(f"size {size}, pinned {source.size}")
    if checksum != source.checksum:
        problems.append(f"publisher checksum {checksum}, pinned {source.checksum}")
    if source.sha256 is None:
        if require_pin:
            problems.append(f"SHA-256 {sha256} is not pinned in SOURCES yet")
    elif sha256 != source.sha256:
        problems.append(f"SHA-256 {sha256}, pinned {source.sha256}")
    return problems


def check_source(corpus_dir: Path, source: Source, *, require_pin: bool = True) -> list[str]:
    """Re-hash one source from disk against its pins; an empty list means it matches."""
    path = corpus_dir / source.relative_path
    if path.is_symlink() or not path.is_file():
        return [f"{source.relative_path}: missing or not a regular file"]
    digests = file_digests(path, source)
    problems = digest_mismatches(source, *digests, require_pin=require_pin)
    return [f"{source.relative_path}: {problem}" for problem in problems]


# --- prepare -------------------------------------------------------------------------------

SAMPLE_RATE = 16_000
RINCO_ARCHIVE = "RINCO_Russian_Intonation_Corpus-v01.zip"
RINCO_ROOT = "INTON-EAF/"
RINCO_FOLDERS = ("1-Questions", "2-Imperatives", "3-Statements", "4-News", "6-Stories")
# Tier IDs as they are in the real archive, compared case-insensitively with '-' read as a space;
# a speaker suffix ` f` / ` m` is allowed. Transcribe (automatic) is never among them.
RINCO_TIERS = {"accent": ("accent",), "pause": ("pause",), "int_phrase": ("int phrase",)}
TIER_LABELS = {"accent": "Accent", "pause": "Pause", "int_phrase": "Int phrase"}
TIER_CATEGORIES = {
    "accent": "semantic_stress",
    "pause": "pause_phrasing",
    "int_phrase": "completeness",
}
ACCENT_MARK = re.compile(r"[/\\+]")
PAUSE_VALUE = re.compile(r"\s*(\d+(?:[.,]\d+)?)")
PAUSE_MIN_S = 0.2
WINDOW_MARGIN_S = 0.3
PAUSE_CONTEXT_S = 0.3
WINDOW_MIN_S = 5.0
WINDOW_MAX_S = 30.0
WINDOW_TARGET_S = 12.0
CANDIDATES = 12
MIN_CANDIDATES = 8
HELDOUT_CANDIDATE_GROUPS = 8
MEDIA_SLACK_S = 0.05
CONTROL_SECONDS = 10.0
BACKGROUND_SECONDS = 15.0
PINK_SEED = 78
DEMAND_OFFSET_S = 60.0
DEMAND_CHANNEL = "ch01.wav"
DEMAND_CONTROLS = (
    ("ctl-demand-dliving", "DLIVING_16k.zip", "жилая комната"),
    ("ctl-demand-ooffice", "OOFFICE_16k.zip", "офис"),
    ("ctl-demand-dkitchen", "DKITCHEN_16k.zip", "кухня"),
)
SNSD_CONTROLS = (
    ("ctl-snsd-babble", "Babble_2.wav"),
    ("ctl-snsd-neighbor-2", "NeighborSpeaking_2.wav"),
    ("ctl-snsd-neighbor-4", "NeighborSpeaking_4.wav"),
)
MIX_NOISE = "Babble_6.wav"
MIX_SNRS_DB = (("ctl-asrnoise-0db", 0.0), ("ctl-asrnoise-m5db", -5.0))
MIX_SEED = 78
MIX_MIN_S = 8.0
MIX_MAX_S = 15.0
PEAK_LIMIT = 0.99
MAX_MEMBER_BYTES = 128 << 20
MAX_EAF_BYTES = 4 << 20
MAX_DECODE_S = 900.0
DECODE_TIMEOUT_S = 120.0
DECODE_THREADS = "2"
DIR_MODE = 0o700
FILE_MODE = 0o600
FORMAT_VERSION = 1
EXCLUDED_CONTROLS = tuple(clip_id for clip_id, _ in SNSD_CONTROLS)
SPEECH_BACKGROUND = (
    "в фоне разборчивая речь, преимущественно английская (прослушивание владельца 2026-11-19, "
    "#84): клип с разборчивыми словами — не контроль noise"
)
DRAFT_NOTE = "черновик #78: разметки нет, категории RINCO не утверждены, held-out не зафиксирован"
EXCLUSION_NOTE = (
    f"исключены из состава: {', '.join(EXCLUDED_CONTROLS)} — {SPEECH_BACKGROUND}; их аудио "
    "сохранено в clips/"
)
MANIFEST_NOTE = f"{DRAFT_NOTE}; {EXCLUSION_NOTE}"
ASR_UNCONFIRMED = "кандидат, ошибка ASR не подтверждена"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class Annotation:
    tier: str
    start: float
    end: float
    value: str


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    categories: tuple[str, ...]
    annotations: tuple[Annotation, ...]

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class RincoFile:
    folder: str
    wav_member: str
    eaf_member: str
    duration_s: float
    annotations: dict[str, list[Annotation]]

    @property
    def group(self) -> str:
        stem = PurePosixPath(self.wav_member).stem.lower()
        return "rinco-" + re.sub(r"[^a-z0-9._-]+", "-", stem)


def tier_kind(tier_id: str) -> str | None:
    """The `RINCO_TIERS` kind of a tier ID, or None for every other tier (Transcribe among them)."""
    name = " ".join(tier_id.casefold().replace("-", " ").split())
    name = re.sub(r" [fm]$", "", name)
    return next((kind for kind, names in RINCO_TIERS.items() if name in names), None)


def parse_eaf(data: bytes) -> tuple[dict[str, list[Annotation]], int, set[str]]:
    """Accent / Pause / Int phrase annotations in seconds, the unaligned count, every tier ID.

    Times come from `TIME_ORDER` (ms); a `REF_ANNOTATION` takes the time of the annotation it
    refers to. Values are read only from the tiers `tier_kind` knows.
    """
    root = ET.fromstring(data)
    slots = {}
    for slot in root.iter("TIME_SLOT"):
        value = slot.get("TIME_VALUE")
        slots[slot.get("TIME_SLOT_ID")] = None if value is None else int(value) / 1000
    times: dict[str, tuple[float | None, float | None]] = {}
    refs: dict[str, str] = {}
    for element in root.iter("ALIGNABLE_ANNOTATION"):
        ends = (slots.get(element.get("TIME_SLOT_REF1")), slots.get(element.get("TIME_SLOT_REF2")))
        times[element.get("ANNOTATION_ID")] = ends
    for element in root.iter("REF_ANNOTATION"):
        refs[element.get("ANNOTATION_ID")] = element.get("ANNOTATION_REF")
    found: dict[str, list[Annotation]] = {kind: [] for kind in RINCO_TIERS}
    tier_ids = set()
    unaligned = 0
    for tier in root.iter("TIER"):
        tier_ids.add(tier.get("TIER_ID", ""))
        kind = tier_kind(tier.get("TIER_ID", ""))
        if kind is None:
            continue
        for element in tier.iter():
            if element.tag not in ("ALIGNABLE_ANNOTATION", "REF_ANNOTATION"):
                continue
            start, end = _annotation_time(element.get("ANNOTATION_ID"), times, refs)
            if start is None or end is None or end <= start:
                unaligned += 1
                continue
            value = (element.findtext("ANNOTATION_VALUE") or "").strip()
            found[kind].append(Annotation(kind, start, end, value))
    for annotations in found.values():
        annotations.sort(key=lambda a: (a.start, a.end))
    return found, unaligned, tier_ids


def _annotation_time(annotation_id, times, refs) -> tuple[float | None, float | None]:
    seen = set()
    while annotation_id in refs and annotation_id not in seen:
        seen.add(annotation_id)
        annotation_id = refs[annotation_id]
    return times.get(annotation_id, (None, None))


def pause_counts(annotation: Annotation) -> bool:
    """A marked pause between phrases: a duration value ≥ 0.2 s; technical pauses have none."""
    match = PAUSE_VALUE.match(annotation.value)
    return bool(match) and float(match.group(1).replace(",", ".")) >= PAUSE_MIN_S


def windows(
    annotations: dict[str, list[Annotation]],
    duration_s: float,
    min_s: float = WINDOW_MIN_S,
    max_s: float = WINDOW_MAX_S,
) -> list[Window]:
    """Every run of adjacent Int phrases with margins, `min_s`–`max_s` long, cutting nothing."""
    phrases = annotations["int_phrase"]
    every = [a for kind in RINCO_TIERS for a in annotations[kind]]
    result = []
    for i, first in enumerate(phrases):
        for last in phrases[i:]:
            start = max(0.0, first.start - WINDOW_MARGIN_S)
            end = min(duration_s, last.end + WINDOW_MARGIN_S)
            if end - start > max_s:
                break
            if end - start < min_s or any(_cuts(a, start, end) for a in every):
                continue
            inside = tuple(a for a in every if start <= a.start and a.end <= end)
            categories = _categories(inside, start, end)
            if categories:
                result.append(Window(start, end, categories, inside))
    return result


def _cuts(annotation: Annotation, start: float, end: float) -> bool:
    return annotation.start < start < annotation.end or annotation.start < end < annotation.end


def _categories(inside: tuple[Annotation, ...], start: float, end: float) -> tuple[str, ...]:
    present = set()
    for a in inside:
        if a.tier == "accent" and ACCENT_MARK.search(a.value):
            present.add("accent")
        if (
            a.tier == "pause"
            and pause_counts(a)
            and a.start - start >= PAUSE_CONTEXT_S
            and end - a.end >= PAUSE_CONTEXT_S
        ):
            present.add("pause")
        if a.tier == "int_phrase" and start < a.end <= end:
            present.add("int_phrase")
    return tuple(TIER_CATEGORIES[kind] for kind in RINCO_TIERS if kind in present)


def best_window(windows_: list[Window]) -> Window | None:
    key = lambda w: (-len(w.categories), abs(w.length - WINDOW_TARGET_S), w.start)  # noqa: E731
    return min(windows_, key=key, default=None)


def select_candidates(files: list[RincoFile]) -> list[tuple[RincoFile, Window]]:
    """At most one window per file, ordered by categories, folder round-robin, length, path."""
    best = [(f, w) for f in files if (w := best_window(windows(f.annotations, f.duration_s)))]
    by_folder: dict[tuple[int, str], list[tuple[RincoFile, Window]]] = {}
    for item in sorted(
        best, key=lambda fw: (abs(fw[1].length - WINDOW_TARGET_S), fw[0].wav_member)
    ):
        by_folder.setdefault((len(item[1].categories), item[0].folder), []).append(item)
    ranked = []
    for (count, folder), items in by_folder.items():
        for rank, (file, window) in enumerate(items):
            ranked.append(((-count, rank, _folder_order(folder)), file, window))
    ranked.sort(key=lambda r: (*r[0], abs(r[2].length - WINDOW_TARGET_S), r[1].wav_member))
    return [(file, window) for _, file, window in ranked[:CANDIDATES]]


def _folder_order(folder: str) -> int:
    return RINCO_FOLDERS.index(folder) if folder in RINCO_FOLDERS else len(RINCO_FOLDERS)


def read_member(archive: zipfile.ZipFile, member: str, limit: int) -> bytes:
    """The bytes of one archive member by name, refused over `limit` declared or actually read."""
    try:
        info = archive.getinfo(member)
    except KeyError:
        raise CorpusError(f"{member}: not in the archive") from None
    if info.file_size > limit:
        raise CorpusError(f"{member}: {info.file_size} B, over the limit of {limit} B")
    chunks, read = [], 0
    with archive.open(info) as source:
        while chunk := source.read(HASH_CHUNK):
            read += len(chunk)
            if read > min(limit, info.file_size):
                raise CorpusError(f"{member}: more bytes than declared or allowed ({limit} B)")
            chunks.append(chunk)
    return b"".join(chunks)


def copy_member(archive: zipfile.ZipFile, member: str, target: Path) -> Path:
    """One member into `target`, a name the code chose; never a path taken from the archive."""
    data = read_member(archive, member, MAX_MEMBER_BYTES)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
    return target


def decode(path: Path) -> np.ndarray:
    """The audio as 16 kHz mono float samples through the bounded ffmpeg; no gain."""
    argv = proc.ffmpeg_argv(
        "-v", "error", "-threads", DECODE_THREADS, "-i", str(path), "-map", "0:a:0",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    )  # fmt: skip
    limit = int(MAX_DECODE_S * SAMPLE_RATE * 4)
    result = proc.run(argv, timeout=DECODE_TIMEOUT_S, max_output_bytes=limit)
    if result.timed_out or result.truncated or result.returncode != 0:
        reason = "timed out" if result.timed_out else "over the length limit"
        if not (result.timed_out or result.truncated):
            reason = result.stderr.decode("utf-8", "replace").strip()[-300:]
        raise CorpusError(f"{path.name}: ffmpeg could not decode it: {reason}")
    return np.frombuffer(result.stdout, "<f4").astype(np.float64)


def wav_duration(data_head: zipfile.ZipExtFile) -> float | None:
    try:
        with wave.open(data_head, "rb") as source:
            return source.getnframes() / source.getframerate()
    except (wave.Error, EOFError):
        return None


def rinco_files(archive: zipfile.ZipFile) -> tuple[list[RincoFile], dict]:
    """Every EAF paired with its WAV in the same folder (media name, else the EAF stem)."""
    members = [n for n in archive.namelist() if n.startswith(RINCO_ROOT) and not n.endswith("/")]
    wavs = {n.lower(): n for n in members if n.lower().endswith(".wav")}
    files, tier_ids, kinds_seen = [], set(), set()
    stats = {"eaf": 0, "unpaired": 0, "media_mismatch": 0, "unaligned_annotations": 0}
    for eaf in sorted(n for n in members if n.lower().endswith(".eaf")):
        stats["eaf"] += 1
        data = read_member(archive, eaf, MAX_EAF_BYTES)
        try:
            annotations, unaligned, ids = parse_eaf(data)
        except (ET.ParseError, ValueError) as error:
            raise CorpusError(f"{eaf}: not a readable EAF: {error}") from None
        stats["unaligned_annotations"] += unaligned
        tier_ids |= ids
        kinds_seen |= {kind for kind, found in annotations.items() if found}
        wav = _paired_wav(eaf, data, wavs)
        if wav is None:
            stats["unpaired"] += 1
            continue
        with archive.open(wav) as head:
            duration = wav_duration(head)
        last = max((a.end for found in annotations.values() for a in found), default=0.0)
        if duration is None or last > duration + MEDIA_SLACK_S:
            stats["media_mismatch"] += 1
            continue
        folder = eaf[len(RINCO_ROOT) :].split("/", 1)[0]
        files.append(RincoFile(folder, wav, eaf, duration, annotations))
    missing = [TIER_LABELS[kind] for kind in RINCO_TIERS if kind not in kinds_seen]
    if missing:
        raise CorpusError(
            f"no EAF has the tiers {missing}; tiers found: {', '.join(sorted(tier_ids))}"
        )
    return files, stats


def _paired_wav(eaf: str, data: bytes, wavs: dict[str, str]) -> str | None:
    folder = eaf.rsplit("/", 1)[0]
    names = [m.group(1) for m in re.finditer(rb'RELATIVE_MEDIA_URL="([^"]+)"', data)]
    names += [m.group(1) for m in re.finditer(rb'MEDIA_URL="([^"]+)"', data)]
    for name in names:
        base = name.decode("utf-8", "replace").replace("\\", "/").rsplit("/", 1)[-1]
        if base.lower().endswith(".wav") and f"{folder}/{base}".lower() in wavs:
            return wavs[f"{folder}/{base}".lower()]
    return wavs.get(f"{eaf[:-4]}.wav".lower())


def mix_at_snr(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int
) -> tuple[np.ndarray, dict]:
    """Speech plus a seeded slice of noise at `snr_db` (RMS over the window), peak kept ≤ 0.99."""
    if len(noise) < len(speech):
        raise CorpusError(f"the noise is {len(noise)} samples, the speech {len(speech)}")
    offset = int(np.random.default_rng(seed).integers(0, len(noise) - len(speech) + 1))
    piece = noise[offset : offset + len(speech)]
    gain = _rms(speech) / (_rms(piece) * 10 ** (snr_db / 20))
    mixed = speech + piece * gain
    peak = float(np.max(np.abs(mixed)))
    scale = PEAK_LIMIT / peak if peak > PEAK_LIMIT else 1.0
    return mixed * scale, {
        "snr_db": snr_db,
        "seed": seed,
        "noise_offset_s": offset / SAMPLE_RATE,
        "noise_gain": gain,
        "scale": scale,
    }


def _rms(samples: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(samples**2)))
    if value == 0.0:
        raise CorpusError("a signal to mix is digital silence")
    return value


def split_by_group(groups: list[str], heldout: int) -> dict[str, str]:
    ordered = sorted(set(groups), key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    return {group: "heldout" if i < heldout else "dev" for i, group in enumerate(ordered)}


def label_template(clip_id: str, sha256: str, duration_s: float) -> dict:
    """Empty on purpose: invalid until a person writes `annotator` and `annotated_at`."""
    return {
        "schema_version": "1.0",
        "clip_id": clip_id,
        "clip_sha256": sha256,
        "clip_duration_s": duration_s,
        "annotator": "",
        "annotated_at": "",
        "model_outputs_seen": False,
        "events": [],
        "note": None,
    }


def prepare(corpus_dir: Path, name: str, sources: tuple[Source, ...] = SOURCES) -> dict:
    """Build `corpus_dir/<name>/`; refuses an existing one and any source off its pin."""
    if not _IDENTIFIER.fullmatch(name):
        raise CorpusError(f"--name {name!r} is not an identifier")
    corpus_dir = Path(corpus_dir)
    out = corpus_dir / name
    if os.path.lexists(out):
        raise CorpusError(f"{name}/ already exists; prepare never overwrites: delete it first")
    problems = [p for source in sources for p in check_source(corpus_dir, source)]
    if problems:
        raise CorpusError("sources do not match their pins: " + "; ".join(problems))
    by_name = {source.name: source for source in sources}
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=corpus_dir))
    try:
        work = staging / ".work"
        work.mkdir(mode=DIR_MODE)
        pending, provenance = _build(corpus_dir, by_name, work)
        shutil.rmtree(work)
        manifest = _write(staging, name, pending, provenance, by_name)
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _source_path(corpus_dir: Path, by_name: dict[str, Source], name: str) -> Path:
    if name not in by_name:
        raise CorpusError(f"{name} is not in the source table")
    return corpus_dir / by_name[name].relative_path


def _build(corpus_dir: Path, by_name: dict[str, Source], work: Path) -> tuple[list, dict]:
    """(id, samples, record fields, provenance) per clip, and the corpus-wide provenance."""
    controls = [
        ("ctl-silence", gate0.silence(CONTROL_SECONDS), "silence",
         "тишина: 10 с цифровых нулей", {"generator": "gate0.silence"}),
        ("ctl-pink", gate0.pink_noise(CONTROL_SECONDS, seed=PINK_SEED), "noise",
         f"розовый шум: 10 с, seed {PINK_SEED}",
         {"generator": "gate0.pink_noise", "seed": PINK_SEED, "rms_dbfs": gate0.NOISE_DBFS}),
    ]  # fmt: skip
    for clip_id, archive_name, room in DEMAND_CONTROLS:
        with zipfile.ZipFile(_source_path(corpus_dir, by_name, archive_name)) as archive:
            member = _channel_member(archive)
            samples = decode(copy_member(archive, member, work / f"{clip_id}.wav"))
        start = round(DEMAND_OFFSET_S * SAMPLE_RATE)
        piece = samples[start : start + round(BACKGROUND_SECONDS * SAMPLE_RATE)]
        if len(piece) < round(BACKGROUND_SECONDS * SAMPLE_RATE):
            raise CorpusError(f"{archive_name} {member}: shorter than {DEMAND_OFFSET_S + 15} s")
        end_s = DEMAND_OFFSET_S + BACKGROUND_SECONDS
        controls.append((
            clip_id, piece, "noise",
            f"DEMAND {room} ({archive_name}, {DEMAND_CHANNEL}), {DEMAND_OFFSET_S:g}–{end_s:g} с, "
            "без усиления",
            {"source": by_name[archive_name].relative_path, "member": member,
             "start_s": DEMAND_OFFSET_S, "end_s": end_s},
        ))  # fmt: skip
    for clip_id, file_name in SNSD_CONTROLS:
        samples = decode(_source_path(corpus_dir, by_name, file_name))
        piece = samples[: round(BACKGROUND_SECONDS * SAMPLE_RATE)]
        note = f"MS-SNSD {file_name}, 0–{len(piece) / SAMPLE_RATE:g} с"
        controls.append((
            clip_id, piece, "noise", note,
            {"source": by_name[file_name].relative_path, "start_s": 0.0,
             "end_s": len(piece) / SAMPLE_RATE},
        ))  # fmt: skip

    with zipfile.ZipFile(_source_path(corpus_dir, by_name, RINCO_ARCHIVE)) as archive:
        files, stats = rinco_files(archive)
        chosen = select_candidates(files)
        if len(chosen) < MIN_CANDIDATES:
            raise CorpusError(f"{len(chosen)} RINCO candidates, at least {MIN_CANDIDATES} needed")
        used = {file.group for file, _ in chosen}
        mix_source = next(
            ((f, w) for f in files if f.group not in used
             if (w := best_window(windows(f.annotations, f.duration_s, MIX_MIN_S, MIX_MAX_S)))),
            None,
        )  # fmt: skip
        if mix_source is None:
            raise CorpusError("no RINCO file outside the candidates has an 8–15 s speech window")
        candidates = []
        for i, (file, window) in enumerate([*chosen, mix_source]):
            samples = decode(copy_member(archive, file.wav_member, work / f"rinco-{i}.wav"))
            piece = samples[round(window.start * SAMPLE_RATE) : round(window.end * SAMPLE_RATE)]
            candidates.append((file, window, piece))
    archive_sha256 = by_name[RINCO_ARCHIVE].sha256
    pending = [
        (clip_id, samples, {"negative_control": True, "control_kind": kind, "categories": [],
                            "source_group": clip_id, "note": note}, {"kind": kind, **details})
        for clip_id, samples, kind, note, details in controls
    ]  # fmt: skip
    for file, window, piece in candidates[:-1]:
        pending.append((
            file.group, piece,
            {"negative_control": False, "categories": list(window.categories),
             "source_group": file.group,
             "note": f"RINCO {file.wav_member} {window.start:.2f}–{window.end:.2f} с; категории — "
                     "кандидаты по слоям Accent/Pause/Int phrase, не утверждены разметкой"},
            _rinco_provenance(file, window, archive_sha256),
        ))  # fmt: skip
    file, window, speech = candidates[-1]
    babble = decode(_source_path(corpus_dir, by_name, MIX_NOISE))
    for clip_id, snr in MIX_SNRS_DB:
        mixed, mix = mix_at_snr(speech, babble, snr, MIX_SEED)
        pending.append((
            clip_id, mixed,
            {"negative_control": True, "control_kind": "corrupted_asr", "categories": [],
             "source_group": file.group,
             "note": f"{ASR_UNCONFIRMED}; RINCO {file.wav_member} {window.start:.2f}–"
                     f"{window.end:.2f} с + {MIX_NOISE}, SNR {snr:g} дБ"},
            {**_rinco_provenance(file, window, archive_sha256), "kind": "corrupted_asr",
             "noise": {"source": by_name[MIX_NOISE].relative_path, **mix}},
        ))  # fmt: skip
    provenance = {
        "rinco": {**stats, "paired_files": len(files), "tiers": RINCO_TIERS,
                  "not_read": "Transcribe (автоматическая транскрипция, выход модели)"},
        "parameters": {
            "sample_rate": SAMPLE_RATE, "window_s": [WINDOW_MIN_S, WINDOW_MAX_S],
            "window_margin_s": WINDOW_MARGIN_S, "pause_context_s": PAUSE_CONTEXT_S,
            "pause_min_s": PAUSE_MIN_S, "candidates": CANDIDATES,
            "heldout_candidate_groups": HELDOUT_CANDIDATE_GROUPS, "pink_seed": PINK_SEED,
            "demand_offset_s": DEMAND_OFFSET_S, "mix_seed": MIX_SEED,
            "mix_window_s": [MIX_MIN_S, MIX_MAX_S], "peak_limit": PEAK_LIMIT,
            "decoder": "ffmpeg → 16 kHz mono, усиления нет (кроме смесей corrupted_asr)",
        },
    }  # fmt: skip
    return pending, provenance


def _channel_member(archive: zipfile.ZipFile) -> str:
    found = [n for n in archive.namelist() if n.lower().rsplit("/", 1)[-1] == DEMAND_CHANNEL]
    if len(found) != 1:
        raise CorpusError(f"expected one {DEMAND_CHANNEL} in the archive, found {len(found)}")
    return found[0]


def _rinco_provenance(file: RincoFile, window: Window, archive_sha256: str | None) -> dict:
    return {
        "kind": "rinco_candidate",
        "source": f"sources/rinco/{RINCO_ARCHIVE}",
        "source_sha256": archive_sha256,
        "member": file.wav_member,
        "eaf_member": file.eaf_member,
        "start_s": round(window.start, 3),
        "end_s": round(window.end, 3),
        "candidate_categories": list(window.categories),
        "annotations": [
            {
                "tier": TIER_LABELS[a.tier],
                "start": round(a.start - window.start, 3),
                "end": round(a.end - window.start, 3),
                "value": a.value,
            }
            for a in window.annotations
        ],
    }


def _write(staging: Path, name: str, pending: list, provenance: dict, by_name: dict) -> dict:
    candidate_groups = [f["source_group"] for _, _, f, _ in pending if not f["negative_control"]]
    split = split_by_group(candidate_groups, HELDOUT_CANDIDATE_GROUPS)
    plain = [
        f["source_group"]
        for _, _, f, _ in pending
        if f["negative_control"] and f["control_kind"] != "corrupted_asr"
    ]
    split |= split_by_group(plain, len(set(plain)) // 2)
    clips_dir = staging / "clips"
    labels_dir = staging / "labels"
    clips_dir.mkdir(mode=DIR_MODE)
    labels_dir.mkdir(mode=DIR_MODE)
    records, clip_provenance = [], []
    for clip_id, samples, fields, details in pending:
        data = gate0.wav_bytes(gate0.pcm16(samples))
        _write_bytes(clips_dir / f"{clip_id}.wav", data)
        sha256 = hashlib.sha256(data).hexdigest()
        duration = (len(data) - 44) / 2 / SAMPLE_RATE
        group_split = split.get(fields["source_group"], "dev")
        records.append({
            "id": clip_id, "path": f"clips/{clip_id}.wav", "sha256": sha256,
            "duration_s": duration, "split": group_split, **fields,
        })  # fmt: skip
        if clip_id in EXCLUDED_CONTROLS:
            details = {**details, "excluded": SPEECH_BACKGROUND}
        else:
            template = label_template(clip_id, sha256, duration)
            _write_json(labels_dir / f"{clip_id}.json", template)
        clip_provenance.append({"id": clip_id, "sha256": sha256, **details})
    manifest = revise({"schema_version": "1.0", "name": name, "records": records})
    _write_json(staging / "manifest.json", manifest)
    _write_json(
        staging / "provenance.json",
        {
            "format_version": FORMAT_VERSION,
            "issue": "https://github.com/renidere/sound-scout-plugin/issues/78",
            "decision": "docs/decisions/0009-corpus-pilot-1.md",
            "name": name,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sound_scout_version": __version__,
            "sources": [
                {
                    "path": s.relative_path,
                    "sha256": s.sha256,
                    "checksum": s.checksum,
                    "version": s.version,
                    "license": s.license,
                }
                for s in by_name.values()
            ],  # fmt: skip
            **provenance,
            "clips": clip_provenance,
        },
    )
    return manifest


def revise(manifest: dict) -> dict:
    """The draft without `EXCLUDED_CONTROLS`, every corrupted_asr control with a status.

    Idempotent; a status already present is kept and a missing one is `candidate`: confirming
    an ASR error is a person's step (`manifest.sheet_problems`), never this function's. A missing
    note or the #78 draft note becomes `MANIFEST_NOTE`; any other note is kept, with
    `EXCLUSION_NOTE` appended when it does not hold it yet.
    """
    records = []
    for record in manifest["records"]:
        if record["id"] in EXCLUDED_CONTROLS:
            continue
        if record.get("control_kind") == "corrupted_asr":
            record = {**record, "asr_error_status": record.get("asr_error_status", "candidate")}
        records.append(record)
    note = manifest.get("note")
    if note in (None, DRAFT_NOTE):
        note = MANIFEST_NOTE
    elif EXCLUSION_NOTE not in note:
        note = f"{note}; {EXCLUSION_NOTE}"
    return {**manifest, "note": note, "records": records}


def revise_file(path: Path) -> tuple[list[str], bool]:
    """Apply `revise` to the manifest at `path` in place; the ids it dropped, whether it wrote."""
    path = Path(path)
    before = json.loads(path.read_text(encoding="utf-8"))
    after = revise(before)
    if after == before:
        return [], False
    kept = {record["id"] for record in after["records"]}
    dropped = [record["id"] for record in before["records"] if record["id"] not in kept]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(temporary, after)
    os.replace(temporary, path)
    return dropped, True


def _write_bytes(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    with os.fdopen(fd, "wb") as out:
        out.write(data)


def _write_json(path: Path, data: object) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    _write_bytes(path, (text + "\n").encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="build the draft corpus from verified sources")
    command.add_argument("--corpus-dir", type=Path, required=True)
    command.add_argument("--name", required=True)
    command = commands.add_parser(
        "revise", help="drop the excluded controls and mark corrupted_asr candidates in place"
    )
    command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "revise":
            dropped, written = revise_file(args.manifest)
            print(f"dropped: {', '.join(dropped) or 'none'}; written: {str(written).lower()}")
            return 0
        manifest = prepare(args.corpus_dir, args.name)
    except (CorpusError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"corpus: {error}", file=sys.stderr)
        return 2
    for record in manifest["records"]:
        print(
            json.dumps(
                {k: record[k] for k in ("id", "split", "duration_s", "categories")},
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
