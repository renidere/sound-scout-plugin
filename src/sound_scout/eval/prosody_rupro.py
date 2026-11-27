"""`prosody-rupro-1`: one confirmatory run of the `PROSODY-RULE v1` index on RuPro (#199).

The protocol is `docs/decisions/0037-word-prominence-rupro.md`; the rule is the block between its
`prosody-rupro-rule` markers, and `survey`, `prepare`, `check-fresh`, `sanity` and `run`
(0037 § 3) refuse to start unless the SHA-256 of that block is `RULE_SHA256`. The labels are the
pitch accents of RuPro (`word_pa` of the OSF table `ip_pos.csv`), the sound and the phrase
intervals come from the RUEG 1.0 release on Zenodo. The index pipeline of series 3 is not
changed: this module calls `prosody` (and through it `asr` and `acoustics`) and the sanity stage
of `prosody_confirm`, and copies nothing.

Commands, all in `<data_root>/prosody-rupro-1/` (outside git):

- `fetch`: the pinned sources into `sources/`: the OSF table whole, and from the 4.4 GB RUEG ZIP
  only the pinned members, read by HTTP range (`RangeReader` under `zipfile`). Needs the network.
- `verify`: the sources on disk against their pins.
- `survey`: import and join of the table, the EXB files and the RuPro graph, the interval checks,
  the exclusions and the counts, written to `survey/<UTC time>.json`. No clip, no ASR, no index.
- `freeze <tree>` / `verify-frozen <tree>`: the manifest `FROZEN.json` of an unchanged git tree with
  its own `.venv`, and the check that the modules the tree's interpreter imports lie in that tree
  and carry the manifest and decision pins (`EXPERIMENTS`).
- `prepare`, `check-fresh`, `sanity`, `run`: the confirmatory run itself. They run only from the
  frozen tree `prosody-rupro-1` (`frozen_state`); `prepare` cuts the phrase clips once, `run`
  scores them once and writes the verdict from `decide`.

Corpus words and texts never leave the data directory: stdout carries counts, codes and verdicts.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import statistics
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sound_scout import acoustics, config, guard, paths, proc
from sound_scout.eval import corpus, gate0, prosody, prosody_confirm

REPO = Path(__file__).resolve().parents[3]
DECISION_DOC = Path("docs/decisions/0037-word-prominence-rupro.md")
RULE_BEGIN = "<!-- prosody-rupro-rule:begin -->"
RULE_END = "<!-- prosody-rupro-rule:end -->"
RULE_VERSION = "PROSODY-RUPRO-RULE v1"
RULE_HEADER = f"```text\n{RULE_VERSION} "
RULE_SHA256 = "6571cfe492e65cd4ef520068c51f0c3e9fedef5945d9aec769a5dec8c77df68c"
OWNER_DECISION = "https://github.com/renidere/sound-scout-plugin/issues/199#issuecomment-7228019900"
SPEC = "https://github.com/renidere/sound-scout-plugin/issues/199#issuecomment-7228283735"

# --- sources (docs/dependencies.md § 8) ----------------------------------------------------

OSF_URL = "https://osf.io/download/fpma5/?version=1"
TABLE_FILE = "ip_pos.csv"
TABLE_BYTES = 2_556_687
TABLE_SHA256 = "f4fc22629c93fb702bc509d0beb07705c428fcc5459659ea50ee2e9a8391eea7"
RUEG_URL = "https://zenodo.org/records/11234583/files/RUEG-1.0_corpora.zip?download=1"
RUEG_BYTES = 4_390_834_445
RUEG_MD5 = "7bfc43c230144ffbfa444f085867b1b6"
RUEG_DIR = "rueg"
LICENSE_MEMBER = "RUEG-1.0_corpora/LICENSE"
RUPRO_MEMBER = "RUEG-1.0_corpora/prosody/ru/RUEG-RuPro.zip"
EXB_PREFIX = "RUEG-1.0_corpora/core/ru/exb/RUEG/RU/MONOLINGUAL/"
PINNED_MEMBERS = {
    LICENSE_MEMBER: (7_048, "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"),
    RUPRO_MEMBER: (117_338_883, "9e3839e581998e2067bdab3566bcd38b7a89268a1d8c872179151d5e9942a758"),
}
# The 80 monolingual documents of the table, and every `<doc>.exb` / `<doc>.meta` with its SHA-256
# (`list_sha256` of sorted "member sha256" lines; each file's own hash is in the survey).
DOCS_SHA256 = "2289dd65704362d325dcca36125fed9486ce7a19313f0e53c609cdfc5a436b3c"
EXB_LIST_SHA256 = "84a071a351237dafd77f834c6a41f5f9278452e25ba5eb4d89726ff73c7ffaf0"
DOC_COUNT = 80
GRAPH_MEMBER = "RUEG-RuPro/RUEG-RuPro.graphml"
GRAPH_PREFIX = "RUEG-RuPro/RU/MONOLINGUAL/"
MP3_MEMBER = "RUEG-RuPro/RUEG-RuPro/RU/MONOLINGUAL/{doc}/{doc}.mp3"
FETCH_TIMEOUT_S = 60.0
FETCH_CHUNK = 1 << 20

# --- the table -----------------------------------------------------------------------------

DOC = "1_meta_annis:doc"
IP_ID, IP_VALUE = "1_id", "1_anno_default_ns:ip"
BILINGUAL, SETTING = "1_meta_speaker-bilingual", "1_meta_setting"
WORD_PA = "3_anno_default_ns:word_pa"
TOKEN_ID, DIPL = "4_id", "4_anno_default_ns:dipl"
POS = "5_anno_default_ns:pos"
TABLE_COLUMNS = (DOC, IP_ID, IP_VALUE, BILINGUAL, SETTING, WORD_PA, TOKEN_ID, DIPL, POS)
IP_LABEL = "IP"
NO_ACCENT = "None"
ACCENT_PART = re.compile(r"[!^]?[LH]\*?(\+[!^]?[LH]\*?)?")
PAUSE = re.compile(r"\((-+|\d+(?:[.,]\d+)?|N\.N)\)")
HESITATION = re.compile(r"[эаымнхг]{1,4}")
# Three documents were matched by hand when the source was checked; they test the import only.
IMPORT_CHECK_DOCS = ("RUmo01FR_fsR", "RUmo01FR_isR", "RUmo02FR_fsR")
MAX_EDGE_DISAGREEMENT_S = 1.0
SETTINGS = {"fsR": "formal", "isR": "informal"}

# The exclusion reasons in the order the rule checks them; the first one an IP meets is its reason.
REASONS = (
    "import_check",
    "ip_value",
    "ambiguous_ip",
    "ambiguous_label",
    "unmatched_token",
    "ambiguous_match",
    "no_rupro_times",
    "label_mismatch",
    "bad_interval",
    "interval_disagree",
)

# --- the run -------------------------------------------------------------------------------

OUTPUT_DIR = "prosody-rupro-1"
SOURCES_DIR = "sources"
SURVEY_DIR = "survey"
PREPARE_FILE = "prepare.json"
PREPARE_SHA_FILE = "prepare.json.sha256"
CLIPS_DIR = "clips"
SANITY_DIR = "sanity"
MAIN_DIR = "main"
RUNS_DIR = "runs"
STARTED_FILE = "started.json"
CLIPS_LOG = "clips.jsonl"
RESULT_FILE = "result.json"
FROZEN_FILE = "FROZEN.json"
DIR_MODE = 0o700
FILE_MODE = 0o600
FORMAT_VERSION = 1
SAMPLE_RATE = prosody.SAMPLE_RATE
MAX_SANITY_ATTEMPTS = 2
SANITY_WALL_SECONDS = 15 * 60
MAIN_WALL_SECONDS = 180 * 60
MIN_PAIRS = 1200
MIN_IPS_WITH_PAIRS = 60
MIN_SPEAKERS_WITH_PAIRS = 20
PASS, FAIL, INCONCLUSIVE, INVALID = (
    prosody.PASS,
    prosody.FAIL,
    prosody.INCONCLUSIVE,
    prosody.INVALID,
)

# --- frozen trees --------------------------------------------------------------------------

CONFIRM_PATH = "src/sound_scout/eval/prosody_confirm.py"
RUPRO_PATH = "src/sound_scout/eval/prosody_rupro.py"
CONFIRM_SHA256 = "8009636aad669e7c172f902e6edf6e34e679539b1793e71e9aa6087572693675"
CONFIRM_DECISION = "docs/decisions/0034-word-prominence-confirm.md"
# The pins of 0034 § 2: the files of series 3 and the module of prosody-confirm-1 (§ 7).
PINNED_FILES = {**prosody_confirm.FROZEN_SHA256, CONFIRM_PATH: CONFIRM_SHA256}
MODULES = {
    "src/sound_scout/eval/prosody.py": "sound_scout.eval.prosody",
    "src/sound_scout/asr.py": "sound_scout.asr",
    "src/sound_scout/acoustics.py": "sound_scout.acoustics",
    CONFIRM_PATH: "sound_scout.eval.prosody_confirm",
    RUPRO_PATH: "sound_scout.eval.prosody_rupro",
}
EXPERIMENTS = {
    # The old experiment, kept as it was when its labelling was replaced (0037 «Заморозка»).
    "prosody-confirm-1": {
        "commit": "f37df5a395be79b8a52c105fa2ea891c99bcc052",
        "modules": tuple(m for m in MODULES if m != RUPRO_PATH),
        "documents": (str(prosody.DECISION_DOC), CONFIRM_DECISION),
        "data": ("prosody-confirm-1",),
    },
    # This run: the commit that lands 0037 and this module on main (fixed when the tree is made).
    "prosody-rupro-1": {
        "commit": None,
        "modules": tuple(MODULES),
        "documents": (str(prosody.DECISION_DOC), CONFIRM_DECISION, str(DECISION_DOC)),
        "data": (),
    },
}
PROBE = (
    "import importlib, importlib.metadata as m, json, sys\n"
    "found = {n: importlib.import_module(n).__file__ for n in sys.argv[1:]}\n"
    "print(json.dumps({'modules': found, 'prefix': sys.prefix,"
    " 'python': sys.version.split()[0],"
    " 'versions': {p: m.version(p) for p in ('faster-whisper', 'ctranslate2')}}))\n"
)
PROBE_TIMEOUT_S = 120.0
GIT_TIMEOUT_S = 30.0


class RuproError(Exception):
    """The step cannot start or go on as 0037 says; nothing was computed after it."""


# --- the rule ------------------------------------------------------------------------------


def rule_sha256(document: str) -> str:
    """SHA-256 of the text between the markers, stripped at the edges, as UTF-8."""
    if document.count(RULE_BEGIN) != 1 or document.count(RULE_END) != 1:
        raise RuproError(f"the decision document must hold one {RULE_BEGIN} … {RULE_END} block")
    begin = document.index(RULE_BEGIN) + len(RULE_BEGIN)
    end = document.index(RULE_END)
    block = document[begin:end].strip()
    if end < begin or not block.startswith(RULE_HEADER):
        raise RuproError(f"the rule block does not start with {RULE_VERSION}")
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def check_rule(decision: Path) -> str:
    found = rule_sha256(decision.read_text(encoding="utf-8"))
    if found != RULE_SHA256:
        raise RuproError(f"{RULE_VERSION} SHA-256 is {found}, recorded {RULE_SHA256}")
    return found


def code_sha256() -> str:
    return sha256_file(Path(__file__))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(FETCH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def list_sha256(lines: list[str]) -> str:
    return prosody.list_sha256(lines)


# --- labels --------------------------------------------------------------------------------


def accent(value: str) -> int | None:
    """0 for `None`, 1 for a pitch accent (each comma part a tone with one `*`), else None."""
    if value == NO_ACCENT:
        return 0
    parts = value.split(",")
    if all(ACCENT_PART.fullmatch(p.strip()) and p.count("*") == 1 for p in parts):
        return 1
    return None


def event_class(text: str) -> str:
    """The kind of an EXB event the table has no row for."""
    stripped = text.strip()
    if "UNK" in stripped:
        return "unk"
    if stripped.startswith("[") and stripped.endswith("]"):
        return "nonspeech"
    if stripped.endswith("/"):
        return "cutoff"
    if HESITATION.fullmatch(prosody.normalize(stripped)):
        return "hesitation"
    return "other"


# --- the sources ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """One `dipl` token of the table: its rows (file line numbers), IPs and raw `word_pa`."""

    token_id: str
    text: str
    rows: tuple[int, ...]
    ips: tuple[str, ...]
    word_pa: tuple[str, ...]
    pos: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    start: float | None
    end: float | None
    text: str


@dataclass(frozen=True)
class GraphToken:
    text: str
    start: float | None
    end: float | None
    word_pa: str | None


def read_table(data: bytes) -> tuple[dict[str, dict], dict]:
    """The monolingual documents of `ip_pos.csv`: setting, tokens in order and their IPs."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")
    missing = [c for c in TABLE_COLUMNS if c not in (reader.fieldnames or ())]
    if missing:
        raise RuproError(f"{TABLE_FILE}: columns missing: {missing}")
    rows: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    total = 0
    for line, row in enumerate(reader, 2):
        total += 1
        if row[BILINGUAL] == "no":
            rows[row[DOC]].append((line, row))
    docs = {}
    for doc, found in rows.items():
        settings = {r[SETTING] for _, r in found}
        if len(settings) != 1:
            raise RuproError(f"{doc}: {len(settings)} settings in the table")
        order: list[str] = []
        by_token: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        ip_values: dict[str, set[str]] = defaultdict(set)
        for line, row in found:
            if row[TOKEN_ID] not in by_token:
                order.append(row[TOKEN_ID])
            by_token[row[TOKEN_ID]].append((line, row))
            ip_values[row[IP_ID]].add(row[IP_VALUE])
        if [int(t) for t in order] != sorted(int(t) for t in order):
            raise RuproError(f"{doc}: table tokens are not in the order of their ids")
        tokens = []
        for token_id in order:
            entries = by_token[token_id]
            texts = {r[DIPL] for _, r in entries}
            if len(texts) != 1:
                raise RuproError(f"{doc}: token {token_id} has {len(texts)} texts")
            tokens.append(
                Token(
                    token_id,
                    texts.pop(),
                    tuple(line for line, _ in entries),
                    tuple(sorted({r[IP_ID] for _, r in entries}, key=int)),
                    tuple(sorted({r[WORD_PA] for _, r in entries})),
                    tuple(r[POS] for _, r in entries),
                )
            )
        docs[doc] = {
            "setting": settings.pop(),
            "tokens": tokens,
            "rows": len(found),
            "ip_values": {ip: sorted(values) for ip, values in ip_values.items()},
        }
    return docs, {"rows": total, "monolingual_rows": sum(len(f) for f in rows.values())}


def parse_exb(data: bytes) -> list[Event]:
    """The events of the one `dipl` transcription tier, in the order of the timeline."""
    root = ET.fromstring(data)
    order, times = {}, {}
    for number, point in enumerate(root.iter("tli")):
        order[point.get("id")] = number
        value = point.get("time")
        times[point.get("id")] = float(value) if value is not None else None
    tiers = [t for t in root.iter("tier") if t.get("category") == "dipl" and t.get("type") == "t"]
    if len(tiers) != 1:
        raise RuproError(f"EXB has {len(tiers)} dipl transcription tiers, expected 1")
    events = sorted(tiers[0].iter("event"), key=lambda e: order.get(e.get("start"), -1))
    return [Event(times.get(e.get("start")), times.get(e.get("end")), e.text or "") for e in events]


def parse_meta(text: str) -> str | None:
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "speaker-id":
            return value.strip()
    return None


def _time_parts(value: str) -> tuple[float | None, float | None]:
    start, _, end = value.partition("-")
    return (float(start) if start else None, float(end) if end else None)


def parse_graph(stream, docs: set[str]) -> dict[str, list[GraphToken]]:
    """The `dipl` tokens of each document of the RuPro graph, with their time and `word_pa`.

    A `dipl` token gets its time through the `base` token that points to it (WebMAUS words
    on the Praat timeline) and its `word_pa` from the span that covers the same timeline tokens.
    """
    keys: dict[str, str] = {}
    texts: dict[str, str] = {}
    times: dict[str, str] = {}
    word_pa: dict[str, str] = {}
    is_base: set[str] = set()
    cover: dict[str, list[str]] = defaultdict(list)
    span_cover: dict[str, list[str]] = defaultdict(list)
    following: dict[str, str] = {}
    pointer: dict[str, list[str]] = defaultdict(list)
    wanted = tuple(f"{GRAPH_PREFIX}{d}#" for d in sorted(docs))
    for _, element in ET.iterparse(stream):
        tag = element.tag
        if tag == "key":
            keys[element.get("id")] = element.get("attr.name")
        elif tag == "node":
            node = element.get("id")
            if node.startswith(wanted):
                for data in element:
                    name, value = keys.get(data.get("key")), data.text or ""
                    if name == "default_ns::dipl":
                        texts[node] = value
                    elif name == "annis::time":
                        times[node] = value
                    elif name == "default_ns::word_pa":
                        word_pa[node] = value
                    elif name == "default_ns::base":
                        is_base.add(node)
            element.clear()
        elif tag == "edge":
            source, label = element.get("source"), element.get("label")
            if source.startswith(wanted):
                target = element.get("target")
                if label == "Ordering/default_ns/dipl":
                    following[source] = target
                elif label == "Pointing/default_ns/a":
                    pointer[target].append(source)
                elif label == "Coverage/VIRTUAL/timelineRelationMapping":
                    cover[source].append(target)
                elif label == "Coverage/default_ns/":
                    span_cover[source].append(target)
            element.clear()
    span_of = {}
    for span, covered in span_cover.items():
        if span in word_pa:
            span_of[frozenset(covered)] = word_pa[span]
    found: dict[str, list[GraphToken]] = {}
    firsts = set(texts) - set(following.values())
    for first in firsts:
        doc = first.split("#", 1)[0].rsplit("/", 1)[1]
        if doc in found:
            raise RuproError(f"{doc}: the graph has more than one dipl chain")
        chain, node = [], first
        while node is not None:
            chain.append(node)
            node = following.get(node)
        tokens = []
        for node in chain:
            bases = [b for b in pointer.get(node, ()) if b in is_base]
            start = end = label = None
            if len(bases) == 1:
                covered = cover.get(bases[0], [])
                parts = [_time_parts(times.get(t, "")) for t in covered]
                starts = [s for s, _ in parts if s is not None]
                ends = [e for _, e in parts if e is not None]
                if covered and starts and ends:
                    start, end = min(starts), max(ends)
                label = span_of.get(frozenset(covered))
            tokens.append(GraphToken(texts[node], start, end, label))
        found[doc] = tokens
    return found


# --- the join ------------------------------------------------------------------------------


def align_both(left: list[str], right: list[str]) -> tuple[dict[int, int], set[int]]:
    """The leftmost-first order-preserving match of equal texts and the tokens whose match
    differs from the rightmost-first one (both are longest common subsequences)."""
    first = dict(prosody.align(left, right))
    last = {
        len(left) - 1 - i: len(right) - 1 - j for i, j in prosody.align(left[::-1], right[::-1])
    }
    differ = {i for i in set(first) | set(last) if first.get(i) != last.get(i)}
    return first, differ


def is_pause(text: str) -> bool:
    return bool(PAUSE.fullmatch(text.strip()))


def join_doc(
    doc: str,
    table: dict,
    events: list[Event],
    graph: list[GraphToken],
    mp3_seconds: float | None,
    speaker: str,
) -> tuple[list[dict], dict]:
    """The IPs of one document with words, intervals, provenance and exclusion reasons."""
    tokens: list[Token] = table["tokens"]
    speech = [i for i, e in enumerate(events) if not is_pause(e.text)]
    match, differ = align_both(
        [t.text.strip() for t in tokens], [events[i].text.strip() for i in speech]
    )
    event_of = {t: speech[e] for t, e in match.items()}
    same_length = len(graph) == len(events)
    pairs = zip(events, graph, strict=True) if same_length else ()
    text_differs = {i for i, (e, g) in enumerate(pairs) if e.text.strip() != g.text.strip()}
    overlapping = set()
    for i, (event, after) in enumerate(zip(events, events[1:], strict=False)):
        if event.end is not None and after.start is not None and event.end > after.start:
            overlapping |= {i, i + 1}
    matched_events = set(event_of.values())
    unmatched = Counter(event_class(events[i].text) for i in speech if i not in matched_events)
    by_ip: dict[str, list[int]] = defaultdict(list)
    for number, token in enumerate(tokens):
        for ip in token.ips:
            by_ip[ip].append(number)
    ips = []
    for ip, members in sorted(by_ip.items(), key=lambda item: (item[1][0], int(item[0]))):
        reasons, words = [], []
        exb, rupro = [], []
        for number in members:
            token = tokens[number]
            event = event_of.get(number)
            paired = same_length and event is not None and event not in text_differs
            partner = graph[event] if paired else None
            raw = token.word_pa[0] if len(token.word_pa) == 1 else None
            label = accent(raw) if raw is not None else None
            if event is not None:
                exb.append((events[event].start, events[event].end, event in overlapping))
            if partner is not None:
                rupro.append((partner.start, partner.end))
            words.append(
                {
                    "token_id": token.token_id,
                    "rows": list(token.rows),
                    "text": token.text,
                    "form": prosody.normalize(token.text),
                    "word_pa": raw if raw is not None else list(token.word_pa),
                    "marked": label,
                    "graph_word_pa": partner.word_pa if partner is not None else None,
                    "exb_event": event,
                }
            )
            checks = {
                "ambiguous_ip": len(token.ips) > 1,
                "ambiguous_label": label is None,
                "unmatched_token": event is None and number not in differ,
                "ambiguous_match": number in differ,
                "no_rupro_times": event is not None
                and (partner is None or partner.start is None or partner.end is None),
                "label_mismatch": partner is not None and partner.word_pa != raw,
            }
            reasons += [name for name, hit in checks.items() if hit]
        if doc in IMPORT_CHECK_DOCS:
            reasons.append("import_check")
        if table["ip_values"].get(ip) != [IP_LABEL]:
            reasons.append("ip_value")
        exb_span = rupro_span = cut = None
        if exb and all(s is not None and e is not None for s, e, _ in exb):
            exb_span = (min(s for s, _, _ in exb), max(e for _, e, _ in exb))
        if rupro and len(rupro) == len(members) and all(None not in r for r in rupro):
            rupro_span = (min(s for s, _ in rupro), max(e for _, e in rupro))
        complete = exb_span is not None and len(exb) == len(members)
        if complete:
            bad = any(s >= e or flagged for s, e, flagged in exb)
            bad |= rupro_span is not None and any(s >= e for s, e in rupro)
            cut = (
                (min(exb_span[0], rupro_span[0]), max(exb_span[1], rupro_span[1]))
                if rupro_span is not None
                else exb_span
            )
            bad |= cut[0] < 0 or mp3_seconds is None or cut[1] > mp3_seconds
            if bad:
                reasons.append("bad_interval")
            if rupro_span is not None and (
                abs(exb_span[0] - rupro_span[0]) > MAX_EDGE_DISAGREEMENT_S
                or abs(exb_span[1] - rupro_span[1]) > MAX_EDGE_DISAGREEMENT_S
            ):
                reasons.append("interval_disagree")
        elif exb and len(exb) == len(members):
            reasons.append("bad_interval")
        reasons = [r for r in REASONS if r in reasons]
        marked = sum(1 for w in words if w["marked"] == 1)
        plain = sum(1 for w in words if w["marked"] == 0)
        ips.append(
            {
                "doc": doc,
                "ip": ip,
                "speaker": speaker,
                "setting": doc.rsplit("_", 1)[-1],
                "exb": list(exb_span) if exb_span else None,
                "rupro": list(rupro_span) if rupro_span else None,
                "cut": list(cut) if cut else None,
                "words": words,
                "reasons": reasons,
                "reason": reasons[0] if reasons else None,
                "pairs": marked * plain,
            }
        )
    stats = {
        "tokens": len(tokens),
        "exb_events": len(events),
        "exb_pauses": len(events) - len(speech),
        "matched_tokens": len(match) - len(differ & set(match)),
        "unmatched_tokens": len(tokens) - len(set(match) | differ),
        "ambiguous_tokens": len(differ),
        "unmatched_exb_events": dict(sorted(unmatched.items())),
        "graph_tokens": len(graph),
        "graph_counts_equal": same_length,
        "graph_texts_differ": len(text_differs),
        "mp3_seconds": mp3_seconds,
    }
    return ips, stats


def edge_disagreements(ips: list[dict]) -> list[float]:
    both = [ip for ip in ips if ip["exb"] and ip["rupro"]]
    return [abs(ip["exb"][k] - ip["rupro"][k]) for ip in both for k in (0, 1)]


def summarize(ips: list[dict], stats: dict[str, dict]) -> dict:
    """The counts of the import that the protocol records before any ASR or index."""
    kept = [ip for ip in ips if not ip["reasons"]]
    words = [w for ip in kept for w in ip["words"]]
    with_pairs = [ip for ip in kept if ip["pairs"]]
    lengths = [ip["cut"][1] - ip["cut"][0] for ip in kept]
    edges = edge_disagreements(ips)
    labels = Counter(
        "none" if w["marked"] == 0 else "accent" if w["marked"] == 1 else "ambiguous"
        for ip in ips
        for w in ip["words"]
    )
    compared = [w for ip in ips for w in ip["words"] if w["graph_word_pa"] is not None]
    unmatched: Counter = Counter()
    for found in stats.values():
        unmatched.update(found["unmatched_exb_events"])
    return {
        "docs": len(stats),
        "speakers": len({ip["speaker"] for ip in ips}),
        "ips": len(ips),
        "table_tokens": sum(s["tokens"] for s in stats.values()),
        "word_labels": dict(sorted(labels.items())),
        "exb_events": sum(s["exb_events"] for s in stats.values()),
        "exb_pauses": sum(s["exb_pauses"] for s in stats.values()),
        "matched_tokens": sum(s["matched_tokens"] for s in stats.values()),
        "unmatched_tokens": sum(s["unmatched_tokens"] for s in stats.values()),
        "ambiguous_tokens": sum(s["ambiguous_tokens"] for s in stats.values()),
        "unmatched_exb_events": dict(sorted(unmatched.items())),
        "docs_with_graph_counts_equal": sum(s["graph_counts_equal"] for s in stats.values()),
        "graph_texts_differ": sum(s["graph_texts_differ"] for s in stats.values()),
        "graph_labels_compared": len(compared),
        "graph_labels_equal": sum(1 for w in compared if w["graph_word_pa"] == w["word_pa"]),
        "edge_disagreement_s": {
            "edges": len(edges),
            "median": round(statistics.median(edges), 3) if edges else None,
            "p95": round(float(np.percentile(edges, 95)), 3) if edges else None,
            "over_1s": sum(1 for e in edges if e > MAX_EDGE_DISAGREEMENT_S),
        },
        "excluded_by_reason": {r: sum(1 for ip in ips if ip["reason"] == r) for r in REASONS},
        "excluded_with_reason": {r: sum(1 for ip in ips if r in ip["reasons"]) for r in REASONS},
        "kept": {
            "ips": len(kept),
            "docs": len({ip["doc"] for ip in kept}),
            "speakers": len({ip["speaker"] for ip in kept}),
            "words": len(words),
            "marked_words": sum(1 for w in words if w["marked"] == 1),
            "pairs": sum(ip["pairs"] for ip in kept),
            "ips_with_pairs": len(with_pairs),
            "docs_with_pairs": len({ip["doc"] for ip in with_pairs}),
            "speakers_with_pairs": len({ip["speaker"] for ip in with_pairs}),
            "audio_minutes": round(sum(lengths) / 60, 2),
            "median_ip_s": round(statistics.median(lengths), 3) if lengths else None,
            "median_words": statistics.median(len(ip["words"]) for ip in kept) if kept else None,
            "by_setting": {
                s: {
                    "ips": sum(1 for ip in kept if ip["setting"] == s),
                    "pairs": sum(ip["pairs"] for ip in kept if ip["setting"] == s),
                }
                for s in SETTINGS
            },
        },
    }


# --- reading the sources from disk ---------------------------------------------------------


def sources_dir(config_: config.Config) -> Path:
    return config_.data_root / OUTPUT_DIR / SOURCES_DIR


def member_path(sources: Path, member: str) -> Path:
    return sources / RUEG_DIR / member


def exb_members(docs: list[str]) -> list[str]:
    return [f"{EXB_PREFIX}{d}.{kind}" for d in sorted(docs) for kind in ("exb", "meta")]


def verify(sources: Path) -> tuple[list[str], dict]:
    """The sources on disk against their pins: (problems, SHA-256 of each file read)."""
    problems, hashes = [], {}
    table = sources / TABLE_FILE
    if not table.is_file():
        return [f"{TABLE_FILE}: missing"], hashes
    hashes[TABLE_FILE] = sha256_file(table)
    if table.stat().st_size != TABLE_BYTES or hashes[TABLE_FILE] != TABLE_SHA256:
        problems.append(f"{TABLE_FILE}: size or SHA-256 differs from the pin")
    for member, (size, sha) in PINNED_MEMBERS.items():
        path = member_path(sources, member)
        if not path.is_file():
            problems.append(f"{member}: missing")
            continue
        hashes[member] = sha256_file(path)
        if path.stat().st_size != size or hashes[member] != sha:
            problems.append(f"{member}: size or SHA-256 differs from the pin")
    if problems:
        return problems, hashes
    docs, _ = read_table(table.read_bytes())
    if len(docs) != DOC_COUNT or list_sha256(sorted(docs)) != DOCS_SHA256:
        problems.append(f"the table names {len(docs)} monolingual documents, not the pinned list")
        return problems, hashes
    lines = []
    for member in exb_members(list(docs)):
        path = member_path(sources, member)
        if not path.is_file():
            problems.append(f"{member}: missing")
            continue
        hashes[member] = sha256_file(path)
        lines.append(f"{member} {hashes[member]}")
    if not problems and list_sha256(lines) != EXB_LIST_SHA256:
        problems.append("the EXB and meta files differ from the pinned list")
    return problems, hashes


def mp3_name(doc: str) -> str:
    return MP3_MEMBER.format(doc=doc)


def load(sources: Path, work: Path, decode: bool = True) -> dict:
    """Table, EXB, meta, graph and decoded audio of every monolingual document."""
    table_bytes = (sources / TABLE_FILE).read_bytes()
    docs, table_stats = read_table(table_bytes)
    exb, speakers = {}, {}
    for doc in docs:
        exb[doc] = parse_exb(member_path(sources, f"{EXB_PREFIX}{doc}.exb").read_bytes())
        meta = member_path(sources, f"{EXB_PREFIX}{doc}.meta").read_text(encoding="utf-8")
        speakers[doc] = parse_meta(meta)
        if speakers[doc] != doc.split("_", 1)[0]:
            raise RuproError(f"{doc}: speaker-id of the meta file differs from the document name")
    audio, mp3 = {}, {}
    with zipfile.ZipFile(member_path(sources, RUPRO_MEMBER)) as archive:
        with archive.open(GRAPH_MEMBER) as stream:
            graph = parse_graph(stream, set(docs))
        for doc in sorted(docs):
            data = corpus.read_member(archive, mp3_name(doc), corpus.MAX_MEMBER_BYTES)
            mp3[doc] = hashlib.sha256(data).hexdigest()
            if decode:
                temporary = work / f"{doc}.mp3"
                corpus._write_bytes(temporary, data)
                try:
                    audio[doc] = corpus.decode(temporary)
                finally:
                    temporary.unlink()
    missing = sorted(set(docs) - set(graph))
    if missing:
        raise RuproError(f"{len(missing)} documents are not in the RuPro graph")
    return {
        "table_sha256": hashlib.sha256(table_bytes).hexdigest(),
        "table": docs,
        "table_stats": table_stats,
        "exb": exb,
        "speakers": speakers,
        "graph": graph,
        "audio": audio,
        "mp3_sha256": mp3,
    }


def join(loaded: dict) -> tuple[list[dict], dict[str, dict]]:
    ips, stats = [], {}
    for doc in sorted(loaded["table"]):
        samples = loaded["audio"].get(doc)
        found, stats[doc] = join_doc(
            doc,
            loaded["table"][doc],
            loaded["exb"][doc],
            loaded["graph"][doc],
            len(samples) / SAMPLE_RATE if samples is not None else None,
            loaded["speakers"][doc],
        )
        ips += found
    return ips, stats


# --- freshness -----------------------------------------------------------------------------


def confirm_hashes(config_: config.Config) -> set[str]:
    """Every SHA-256 the old experiment `prosody-confirm-1` recorded for its clips."""
    path = config_.data_root / prosody_confirm.OUTPUT_DIR / prosody_confirm.PREPARE_FILE
    if not path.is_file():
        raise RuproError(f"{prosody_confirm.OUTPUT_DIR}/{prosody_confirm.PREPARE_FILE} is missing")
    prepared = _read_json(path)
    found = {
        clip[key]
        for clip in prepared["fresh_clips"]
        for key in ("member_sha256", "wav_sha256", "pcm_sha256")
    }
    return found | {c["wav_sha256"] for c in prepared["calibration"]}


def check_fresh(config_: config.Config, hashes: dict[str, str], docs: list[str]) -> dict:
    """Each MP3 and clip hash against series 1–3, pilot-1, Gate 0 and prosody-confirm-1."""
    references = prosody_confirm.reference_hashes(config_)
    references["prosody_confirm_1"] = confirm_hashes(config_)
    provenance = _read_json(config_.data_root / prosody.PILOT_PROVENANCE)
    named = json.dumps(provenance["clips"], ensure_ascii=False)
    in_pilot = sorted(d for d in docs if d in named)
    overlap = sorted(
        {name for name, sha in hashes.items() for found in references.values() if sha in found}
    )
    return {
        "fresh": not overlap and not in_pilot,
        "overlap": overlap,
        "docs_in_pilot_1": in_pilot,
        "references": {name: len(found) for name, found in references.items()},
        "hashes": len(hashes),
    }


# --- frozen trees --------------------------------------------------------------------------


def _git(tree: Path, *args: str) -> str:
    result = proc.run(
        ["git", "-C", os.fspath(tree), *args], timeout=GIT_TIMEOUT_S, max_output_bytes=1 << 20
    )
    if result.returncode != 0 or result.timed_out:
        raise RuproError(f"git {args[0]} failed in the tree")
    return result.stdout.decode("utf-8")


def probe(tree: Path, modules: list[str]) -> dict:
    """`__file__` of each module as the tree's own interpreter (`python -I`) imports it."""
    python = tree / ".venv/bin/python"
    if not python.exists():
        raise RuproError("the tree has no .venv/bin/python")
    result = proc.run(
        [os.fspath(python), "-I", "-c", PROBE, *modules],
        timeout=PROBE_TIMEOUT_S,
        max_output_bytes=1 << 20,
    )
    if result.returncode != 0 or result.timed_out:
        tail = result.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        raise RuproError(f"the tree's interpreter could not import the modules: {tail}")
    return json.loads(result.stdout)


def _data_files(data_root: Path, names: tuple[str, ...]) -> dict[str, str]:
    found = {}
    for name in names:
        for path in sorted((data_root / name).rglob("*")):
            if path.is_file() and not path.is_symlink():
                found[path.relative_to(data_root).as_posix()] = sha256_file(path)
    return found


def freeze(tree: Path, experiment: str, data_root: Path, probe_=probe) -> dict:
    """`FROZEN.json` of a clean tree: commit, files, interpreter, versions and data hashes."""
    spec = EXPERIMENTS[experiment]
    manifest_path = tree / FROZEN_FILE
    if manifest_path.exists():
        raise RuproError(f"{FROZEN_FILE} exists; a tree is frozen once")
    commit = _git(tree, "rev-parse", "HEAD").strip()
    if spec["commit"] is not None and commit != spec["commit"]:
        raise RuproError(f"the tree is at {commit}, {experiment} is pinned to {spec['commit']}")
    if _git(tree, "status", "--porcelain", "--untracked-files=no").strip():
        raise RuproError("the tree has changes to tracked files")
    tracked = _git(tree, "ls-files", "-z", "src", "uv.lock", "pyproject.toml", *spec["documents"])
    files = {name: sha256_file(tree / name) for name in sorted(filter(None, tracked.split("\0")))}
    found = probe_(tree, [MODULES[m] for m in spec["modules"]])
    manifest = {
        "format_version": FORMAT_VERSION,
        "experiment": experiment,
        "decision": str(DECISION_DOC),
        "commit": commit,
        "python": found["python"],
        "versions": found["versions"],
        "modules": {m: MODULES[m] for m in spec["modules"]},
        "files": files,
        "data": _data_files(data_root, spec["data"]),
        "at": prosody._now(),
    }
    corpus._write_json(manifest_path, manifest)
    return manifest


def verify_frozen(
    tree: Path, data_root: Path, pins: dict[str, str] = PINNED_FILES, probe_=probe
) -> dict:
    """The frozen tree as its manifest and the 0034 pins say; `verified` with `problems`."""
    tree = tree.resolve()
    if not (tree / FROZEN_FILE).is_file():
        raise RuproError(f"not a frozen tree: {FROZEN_FILE} is missing")
    manifest = _read_json(tree / FROZEN_FILE)
    spec = EXPERIMENTS[manifest["experiment"]]
    problems = []
    if spec["commit"] is not None and manifest["commit"] != spec["commit"]:
        problems.append(f"manifest commit {manifest['commit']}, pinned {spec['commit']}")
    try:
        if _git(tree, "rev-parse", "HEAD").strip() != manifest["commit"]:
            problems.append("the tree is not at the manifest commit")
        if _git(tree, "status", "--porcelain", "--untracked-files=no").strip():
            problems.append("the tree has changes to tracked files")
    except RuproError as error:
        problems.append(str(error))
    for name, sha in manifest["files"].items():
        path = tree / name
        if not path.is_file() or sha256_file(path) != sha:
            problems.append(f"{name}: SHA-256 differs from the manifest")
    for name, sha in pins.items():
        if name in manifest["modules"] and manifest["files"].get(name) != sha:
            problems.append(f"{name}: manifest SHA-256 is not the 0034 pin")
    for document, check, pinned in (
        (str(prosody.DECISION_DOC), prosody.rule_sha256, prosody_confirm.PROSODY_RULE_SHA256),
        (CONFIRM_DECISION, prosody_confirm.rule_sha256, prosody_confirm.RULE_SHA256),
    ):
        try:
            found = check((tree / document).read_text(encoding="utf-8"))
        except (OSError, prosody.ProsodyError, prosody_confirm.ConfirmError):
            found = None
        if found != pinned:
            problems.append(f"{document}: rule block SHA-256 is not the pin")
    try:
        found = probe_(tree, list(manifest["modules"].values()))
    except RuproError as error:
        found = None
        problems.append(str(error))
    for name, module in manifest["modules"].items() if found else ():
        location = Path(found["modules"].get(module) or "/").resolve()
        if location != tree / name:
            problems.append(f"{module} is imported from outside the tree")
        elif sha256_file(location) != manifest["files"].get(name):
            problems.append(f"{module}: the imported file differs from the manifest")
    if found and Path(found["prefix"]).resolve() != (tree / ".venv").resolve():
        problems.append("the interpreter is not the tree's .venv")
    if found and (found["python"], found["versions"]) != (manifest["python"], manifest["versions"]):
        problems.append("Python or package versions differ from the manifest")
    data = _data_files(data_root, spec["data"])
    if data != manifest["data"]:
        problems.append("the data files differ from the manifest")
    return {
        "verified": not problems,
        "experiment": manifest["experiment"],
        "commit": manifest["commit"],
        "uv_lock_sha256": manifest["files"].get("uv.lock"),
        "python": manifest["python"],
        "modules": len(manifest["modules"]),
        "files": len(manifest["files"]),
        "data_files": len(manifest["data"]),
        "problems": problems,
    }


def frozen_state(tree: Path = REPO) -> tuple[list[str], dict]:
    """This process against the frozen tree `prosody-rupro-1` it must run from."""
    manifest_path = tree / FROZEN_FILE
    if not manifest_path.is_file():
        return [f"not a frozen tree: {FROZEN_FILE} is missing"], {}
    manifest = _read_json(manifest_path)
    if manifest.get("experiment") != OUTPUT_DIR:
        return [f"the frozen tree is {manifest.get('experiment')}, not {OUTPUT_DIR}"], {}
    problems = []
    try:
        if _git(tree, "rev-parse", "HEAD").strip() != manifest["commit"]:
            problems.append("the frozen tree is not at the manifest commit")
        if _git(tree, "status", "--porcelain", "--untracked-files=no").strip():
            problems.append("the frozen tree has changes to tracked files")
    except RuproError as error:
        problems.append(str(error))
    for name, sha in manifest["files"].items():
        path = tree / name
        if not path.is_file() or sha256_file(path) != sha:
            problems.append(f"{name}: SHA-256 differs from the manifest")
    for name, module_name in manifest["modules"].items():
        module = sys.modules.get(module_name)
        loaded = __file__ if name == RUPRO_PATH else getattr(module, "__file__", None)
        location = Path(loaded or "/").resolve()  # this module may run as `__main__`
        if location != (tree / name).resolve():
            problems.append(f"{module_name} is loaded from outside the frozen tree")
        elif sha256_file(location) != manifest["files"].get(name):
            problems.append(f"{module_name}: SHA-256 differs from the manifest")
        if name in PINNED_FILES and manifest["files"].get(name) != PINNED_FILES[name]:
            problems.append(f"{name}: SHA-256 is not the 0034 pin")
    if Path(sys.prefix).resolve() != (tree / ".venv").resolve():
        problems.append("the interpreter is not the frozen tree's .venv")
    info = {"commit": manifest["commit"], "manifest_sha256": sha256_file(manifest_path)}
    return problems, info


# --- the metric and the verdict ------------------------------------------------------------


def rupro_metric(records: list[dict]) -> dict:
    """`prosody.metric` with the speaker as the cluster, plus the minimum counts of the rule."""
    found = prosody.metric(records)
    found.pop("verdict")
    found["speakers"] = found.pop("files")
    found["auc_by_setting"] = found.pop("auc_by_section")
    with_pairs = [r for r in records if prosody.pair_counts(r["words"])[1]]
    found["ips"] = len(records)
    found["ips_with_pairs"] = len(with_pairs)
    found["speakers_with_pairs"] = len({r["member"] for r in with_pairs})
    by_doc: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for record in records:
        by_doc[record["doc"]].append(prosody.pair_counts(record["words"]))
    per_doc = [(sum(w for w, _ in c), sum(p for _, p in c)) for _, c in sorted(by_doc.items())]
    ci = prosody.bootstrap_ci(per_doc)
    found["ci95_by_doc"] = list(ci) if ci else None
    return found


def decide(found: dict) -> str:
    """Rule steps 2–4 after every INVALID check passed and the run was not stopped."""
    if (
        found["coverage"] < prosody.PASS_COVERAGE
        or found["pairs"] < MIN_PAIRS
        or found["ips_with_pairs"] < MIN_IPS_WITH_PAIRS
        or found["speakers_with_pairs"] < MIN_SPEAKERS_WITH_PAIRS
    ):
        return INCONCLUSIVE
    auc_, ci = found["auc"], found["ci95"]
    if auc_ is not None and ci is not None and auc_ >= prosody.PASS_AUC:
        return PASS if ci[0] >= prosody.PASS_CI_LOW else FAIL
    return FAIL


# --- HTTP range reading for fetch ----------------------------------------------------------


Opener = Callable[..., object]


class RangeReader(io.RawIOBase):
    """A seekable read-only view of a remote file of known size by HTTP `Range` requests."""

    def __init__(self, url: str, size: int, opener: Opener = urllib.request.urlopen):
        self.url, self.size, self.opener, self.position = url, size, opener, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size or not len(buffer):
            return 0
        last = min(self.size, self.position + len(buffer)) - 1
        span = f"bytes={self.position}-{last}"
        request = urllib.request.Request(self.url, headers={"Range": span})
        with self.opener(request, timeout=FETCH_TIMEOUT_S) as response:
            _refuse_plain_http(response)
            if response.status != 206:
                raise RuproError(f"range request answered {response.status}, expected 206")
            data = response.read(last - self.position + 2)
        if len(data) != last - self.position + 1:
            raise RuproError("range request returned another length than asked")
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def _refuse_plain_http(response) -> None:
    """A redirect may leave HTTPS; the sources are read only over HTTPS."""
    if not response.geturl().startswith("https://"):
        raise RuproError("the source answered from a URL that is not HTTPS")


def _mkdirs(path: Path) -> Path:
    missing = []
    for parent in (path, *path.parents):
        if parent.is_dir():
            break
        missing.append(parent)
    for parent in reversed(missing):
        _mkdir(parent)
    return path


def _store(target: Path, chunks, size: int, sha256: str | None) -> str:
    """Write the chunks to `target` through a temporary file; refuse a wrong size or SHA-256."""
    _mkdirs(target.parent)
    staged = target.with_name(f".{target.name}.{os.getpid()}.part")
    digest, written = hashlib.sha256(), 0
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        with os.fdopen(os.open(staged, flags, FILE_MODE), "wb") as out:
            for chunk in chunks:
                written += len(chunk)
                if written > size:
                    raise RuproError(f"{target.name}: longer than {size} B")
                digest.update(chunk)
                out.write(chunk)
        found = digest.hexdigest()
        if written != size or (sha256 is not None and found != sha256):
            raise RuproError(f"{target.name}: size {written} or SHA-256 {found} is not the pin")
        os.replace(staged, target)
        return found
    finally:
        with contextlib.suppress(FileNotFoundError):
            staged.unlink()


def _chunks(stream):
    while chunk := stream.read(FETCH_CHUNK):
        yield chunk


def fetch(sources: Path, opener: Opener = urllib.request.urlopen) -> dict:
    """The pinned sources into `sources/`; files that already verify are not read again."""
    fetched = []
    table = sources / TABLE_FILE
    if not (table.is_file() and sha256_file(table) == TABLE_SHA256):
        with opener(OSF_URL, timeout=FETCH_TIMEOUT_S) as response:
            _refuse_plain_http(response)
            _store(table, _chunks(response), TABLE_BYTES, TABLE_SHA256)
        fetched.append(TABLE_FILE)
    docs, _ = read_table(table.read_bytes())
    if list_sha256(sorted(docs)) != DOCS_SHA256:
        raise RuproError("the table names other monolingual documents than the pinned list")
    reader = io.BufferedReader(RangeReader(RUEG_URL, RUEG_BYTES, opener), FETCH_CHUNK)
    with zipfile.ZipFile(reader) as archive:
        for member in [*PINNED_MEMBERS, *exb_members(list(docs))]:
            info = archive.getinfo(member)
            size, sha = PINNED_MEMBERS.get(member, (info.file_size, None))
            path = member_path(sources, member)
            done = path.is_file() and path.stat().st_size == size
            if done and (sha is None or sha256_file(path) == sha):
                continue  # a member without its own pin is checked by the list in `verify`
            with archive.open(info) as stream:
                _store(path, _chunks(stream), size, sha)
            fetched.append(member)
    problems, _ = verify(sources)
    if problems:
        raise RuproError("; ".join(problems))
    return {"fetched": len(fetched), "verified": True}


# --- files ---------------------------------------------------------------------------------


def _mkdir(path: Path) -> Path:
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, DIR_MODE)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def output_dir(config_: config.Config) -> Path:
    return _mkdir(paths.resolve_output(config_.data_root, OUTPUT_DIR, config_))


def _header(decision: Path, frozen: dict) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "experiment": OUTPUT_DIR,
        "decision": str(DECISION_DOC),
        "owner_decision": OWNER_DECISION,
        "spec": SPEC,
        "rule_sha256": check_rule(decision),
        "code_sha256": code_sha256(),
        "frozen": frozen,
        "at": prosody._now(),
    }


def _result(out: Path, header: dict, verdict: str, reason: str, **extra) -> dict:
    result = {**header, "verdict": verdict, "reason": reason, **extra}
    prosody._write_json(out / RESULT_FILE, result)
    return result


def _refuse_if_decided(out: Path) -> None:
    if (out / RESULT_FILE).exists():
        raise RuproError(f"{RESULT_FILE} exists: the verdict is out, nothing more runs")


def _invalid_reasons(config_: config.Config) -> list[str]:
    problems, _ = verify(sources_dir(config_))
    return [f"Q1: {p}" for p in prosody.q1(config_) + problems] + [
        f"frozen: {p}" for p in prosody_confirm.frozen_problems()
    ]


def _frozen_or_refuse(tree: Path) -> tuple[list[str], dict]:
    problems, info = frozen_state(tree)
    if not info:
        raise RuproError("; ".join(problems) + "; prepare, sanity and run go from the frozen tree")
    return problems, info


def _prepared(out: Path) -> dict:
    if not (out / PREPARE_FILE).exists():
        raise RuproError("run `prepare` first")
    return _read_json(out / PREPARE_FILE)


def _prepare_problems(out: Path, prepared: dict) -> list[str]:
    recorded = (out / PREPARE_SHA_FILE).read_text(encoding="utf-8").strip()
    problems = []
    if sha256_file(out / PREPARE_FILE) != recorded:
        problems.append(f"{PREPARE_FILE} changed after it was written")
    for clip in prepared["clips"]:
        path = out / CLIPS_DIR / f"{clip['id']}.wav"
        if not path.is_file() or sha256_file(path) != clip["wav_sha256"]:
            problems.append(f"clip {clip['id']}: SHA-256 differs from {PREPARE_FILE}")
    return problems


def _prepared_elsewhere(prepared: dict, header: dict) -> list[str]:
    """`prepare.json` written by other code or from another frozen tree than this step's."""
    problems = []
    if prepared["code_sha256"] != header["code_sha256"]:
        problems.append(f"{PREPARE_FILE} was written by other code")
    if prepared["frozen"] != header["frozen"]:
        problems.append(f"{PREPARE_FILE} was written from another frozen tree")
    return problems


def _sanity_attempts(out: Path) -> list[dict]:
    return [_read_json(p) for p in sorted((out / SANITY_DIR).glob("attempt-*.json"))]


def clip_id(ip: dict) -> str:
    return f"{ip['doc']}-{ip['ip']}"


def public_ip(ip: dict) -> dict:
    """An IP as `prepare.json` keeps it: provenance of each label, no audio."""
    keys = ("doc", "ip", "speaker", "setting", "exb", "rupro", "cut", "pairs")
    return {k: ip[k] for k in keys} | {
        "words": [
            {k: w[k] for k in ("token_id", "rows", "form", "word_pa", "marked", "graph_word_pa")}
            for w in ip["words"]
        ]
    }


# --- commands ------------------------------------------------------------------------------


def survey(config_: config.Config, decision: Path = DECISION_DOC) -> dict:
    """Import, join, interval checks and exclusions on the real sources; no clip and no index."""
    rule = check_rule(decision)
    sources = sources_dir(config_)
    problems, hashes = verify(sources)
    if problems:
        raise RuproError("; ".join(problems))
    out = _mkdir(output_dir(config_) / SURVEY_DIR)
    work = _mkdir(out / f".work-{os.getpid()}")
    try:
        loaded = load(sources, work)
    finally:
        os.rmdir(work)
    ips, stats = join(loaded)
    found = summarize(ips, stats)
    fresh = check_fresh(
        config_, {f"{d}.mp3": sha for d, sha in loaded["mp3_sha256"].items()}, list(stats)
    )
    record = {
        "format_version": FORMAT_VERSION,
        "experiment": OUTPUT_DIR,
        "kind": "survey: import, join, intervals and exclusions; no clip, ASR or index",
        "rule_sha256": rule,
        "code_sha256": code_sha256(),
        "at": prosody._now(),
        "sources": hashes,
        "table": loaded["table_stats"],
        "counts": found,
        "fresh_mp3": fresh,
        "docs": stats,
        "ips": [public_ip(ip) | {"reasons": ip["reasons"]} for ip in ips],
    }
    name = f"{record['at'].replace(':', '').replace('+0000', 'Z')}.json"
    prosody._write_json(out / name, record)
    return {"survey": name, **found, "fresh": fresh["fresh"]}


def prepare(config_: config.Config, decision: Path = DECISION_DOC, tree: Path = REPO) -> dict:
    frozen_problems, frozen = _frozen_or_refuse(tree)
    header = _header(decision, frozen)
    out = output_dir(config_)
    _refuse_if_decided(out)
    if (out / PREPARE_FILE).exists():
        raise RuproError(f"{PREPARE_FILE} exists; prepare runs once")
    if (out / CLIPS_DIR).exists():
        raise RuproError(f"{CLIPS_DIR} is left from a broken prepare; remove it by hand")
    reasons = [f"frozen: {p}" for p in frozen_problems] + _invalid_reasons(config_)
    if reasons:
        return _result(out, header, INVALID, "; ".join(reasons))
    work = _mkdir(out / ".prepare-work")
    loaded = load(sources_dir(config_), work)
    os.rmdir(work)
    ips, stats = join(loaded)
    kept = [ip for ip in ips if not ip["reasons"]]
    clips_dir = _mkdir(out / CLIPS_DIR)
    clips, hashes = [], {f"{d}.mp3": sha for d, sha in loaded["mp3_sha256"].items()}
    for ip in kept:
        samples = loaded["audio"][ip["doc"]]
        start, end = ip["cut"]
        pcm = gate0.pcm16(samples[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)])
        wav = gate0.wav_bytes(pcm)
        corpus._write_bytes(clips_dir / f"{clip_id(ip)}.wav", wav)
        clip = {
            "id": clip_id(ip),
            "wav_sha256": hashlib.sha256(wav).hexdigest(),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        }
        clips.append(clip)
        hashes[f"{clip['id']}.pcm"] = clip["pcm_sha256"]
        hashes[f"{clip['id']}.wav"] = clip["wav_sha256"]
    fresh = check_fresh(config_, hashes, list(stats))
    record = {
        **header,
        "q1": "ok",
        "sources": verify(sources_dir(config_))[1],
        "table_sha256": loaded["table_sha256"],
        "mp3_sha256": loaded["mp3_sha256"],
        "counts": summarize(ips, stats),
        "docs": stats,
        "ips": [public_ip(ip) | {"reasons": ip["reasons"]} for ip in ips],
        "clips": clips,
        "clip_list_sha256": list_sha256([f"{c['id']} {c['wav_sha256']}" for c in clips]),
        "fresh_check": fresh,
    }
    prosody._write_json(out / PREPARE_FILE, record)
    corpus._write_bytes(out / PREPARE_SHA_FILE, (sha256_file(out / PREPARE_FILE) + "\n").encode())
    if not fresh["fresh"]:
        return _result(out, header, INVALID, "check-fresh: an MP3 or a clip was used before")
    return record


def run_check_fresh(
    config_: config.Config, decision: Path = DECISION_DOC, tree: Path = REPO
) -> dict:
    frozen_problems, frozen = _frozen_or_refuse(tree)
    header = _header(decision, frozen)
    out = output_dir(config_)
    prepared = _prepared(out)
    hashes = {f"{d}.mp3": sha for d, sha in prepared["mp3_sha256"].items()}
    hashes |= {f"{c['id']}.pcm": c["pcm_sha256"] for c in prepared["clips"]}
    hashes |= {f"{c['id']}.wav": c["wav_sha256"] for c in prepared["clips"]}
    found = check_fresh(config_, hashes, list(prepared["docs"]))
    problems = frozen_problems + _prepare_problems(out, prepared)
    found["files"] = problems or "ok"
    if (problems or not found["fresh"]) and not (out / RESULT_FILE).exists():
        _result(out, header, INVALID, "check-fresh: " + ("; ".join(problems) or "reused"))
    return found


def run_sanity(
    config_: config.Config,
    decision: Path = DECISION_DOC,
    transcribe=None,
    scope: tuple[str, str] | None | bool = True,
    wall_seconds: float = SANITY_WALL_SECONDS,
    tree: Path = REPO,
) -> dict:
    frozen_problems, frozen = _frozen_or_refuse(tree)
    header = _header(decision, frozen)
    out = output_dir(config_)
    _refuse_if_decided(out)
    prepared = _prepared(out)
    attempts = _mkdir(out / SANITY_DIR)
    if any(
        a["passed"] and a["code_sha256"] == header["code_sha256"] for a in _sanity_attempts(out)
    ):
        raise RuproError("a sanity attempt of this code has passed already; run the main step")
    started = len(list(attempts.glob("work-*")))  # an attempt that died without a record counts
    if started >= MAX_SANITY_ATTEMPTS:
        reason = f"санитарная стадия не пройдена за {started} попытки с этим кодом"
        return _result(out, header, INVALID, reason, attempts=started)
    scope = prosody._checked_scope(scope)
    cpus = prosody.pin_cpus()
    reasons = [f"frozen: {p}" for p in frozen_problems] + _invalid_reasons(config_)
    reasons += _prepared_elsewhere(prepared, header)
    if reasons:
        return _result(out, header, INVALID, "; ".join(reasons))
    number = started + 1
    work = attempts / f"work-{number}"
    os.mkdir(work, DIR_MODE)
    try:
        with prosody._reused_model():
            found = prosody_confirm.sanity(
                config_,
                transcribe or prosody.local_transcriber(config_),
                work,
                time.monotonic() + wall_seconds,
            )
    except Exception as error:  # a broken attempt is a failed attempt, recorded and counted
        found = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    record = {**header, "attempt": number, "cpus": cpus, "memory_scope": scope,
              "wall_seconds": wall_seconds, **found}  # fmt: skip
    prosody._write_json(attempts / f"attempt-{number}.json", record)
    if not record["passed"] and number >= MAX_SANITY_ATTEMPTS:
        _result(out, header, INVALID, f"санитарная стадия не пройдена за {number} попытки")
    return record


def read_clip(path: Path, wav_sha256: str) -> np.ndarray:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != wav_sha256:
        raise RuproError(f"{path.name}: SHA-256 differs from {PREPARE_FILE}")
    return np.frombuffer(gate0.wav_pcm(data), "<i2").astype(np.float64) / 32768


def run_clips(
    out: Path,
    prepared: dict,
    transcribe: prosody.Transcriber,
    work: Path,
    log: Path,
    deadline: float,
) -> tuple[list[dict], bool]:
    """Every kept IP of `prepare.json` scored and appended to `log`; (records, stopped)."""
    clips = {c["id"]: c for c in prepared["clips"]}
    records = []
    for ip in (ip for ip in prepared["ips"] if not ip["reasons"]):
        if time.monotonic() > deadline:
            return records, True
        clip = clips[clip_id(ip)]
        samples = read_clip(out / CLIPS_DIR / f"{clip['id']}.wav", clip["wav_sha256"])
        words = [{"form": w["form"], "marked": bool(w["marked"])} for w in ip["words"]]
        record = {"member": ip["speaker"], "folder": ip["setting"]} | {
            "doc": ip["doc"],
            "ip": ip["ip"],
        }
        empty = {"words": [dict(w, score=None) for w in words], "asr_words": 0,
                 "asr_excluded": 0, "asr_unmatched": 0}  # fmt: skip
        if len(samples) < SAMPLE_RATE * acoustics.MIN_TRACK_SECONDS:
            record |= {"skipped": "too_short", **empty}
        else:
            found = transcribe(samples, work / f"clip-{len(records):05d}")
            record["asr_flagged_words"] = found["flagged_words"]
            if found["status"] != "completed":
                record |= {"skipped": f"asr_{found['status']}", "reason": found["reason"], **empty}
            else:
                record |= prosody.score_utterance(prosody.track(samples), found["words"], words)
        records.append(record)
        with open(log, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records, time.monotonic() > deadline


def run_main(
    config_: config.Config,
    decision: Path = DECISION_DOC,
    transcribe=None,
    scope: tuple[str, str] | None | bool = True,
    wall_seconds: float = MAIN_WALL_SECONDS,
    tree: Path = REPO,
) -> dict:
    frozen_problems, frozen = _frozen_or_refuse(tree)
    header = _header(decision, frozen)
    out = output_dir(config_)
    _refuse_if_decided(out)
    main_dir = out / MAIN_DIR
    if (main_dir / STARTED_FILE).exists():
        reason = "основной прогон остановлен без результата (лимит ресурсов или kill); повтора нет"
        return _result(out, header, INCONCLUSIVE, reason)
    if main_dir.exists():
        raise RuproError(f"{MAIN_DIR}/ exists: the main run starts once")
    prepared = _prepared(out)
    passed = [a for a in _sanity_attempts(out) if a["passed"]]
    if not passed or passed[-1]["code_sha256"] != header["code_sha256"]:
        raise RuproError("no passed sanity attempt of this code; run `sanity` first")
    scope = prosody._checked_scope(scope)
    reasons = [f"frozen: {p}" for p in frozen_problems] + _invalid_reasons(config_)
    reasons += _prepare_problems(out, prepared) + _prepared_elsewhere(prepared, header)
    if prepared["rule_sha256"] != header["rule_sha256"]:
        reasons.append(f"{PREPARE_FILE} was written under another rule")
    if not prepared["fresh_check"]["fresh"]:
        reasons.append("check-fresh found reused files at prepare")
    if reasons:
        return _result(out, header, INVALID, "; ".join(reasons))
    _mkdir(main_dir)
    cpus = prosody.pin_cpus()
    started = time.monotonic()
    prosody._write_json(main_dir / STARTED_FILE, {**header, "cpus": cpus, "memory_scope": scope})
    try:
        with prosody._reused_model():
            records, stopped = run_clips(
                out,
                prepared,
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
    if stopped:
        reason = f"остановлено потолком wall time {wall_seconds / 60:g} мин"
        return _result(out, header, INCONCLUSIVE, reason, ips_done=len(records), elapsed_s=elapsed)
    found = rupro_metric(records)
    result = {**header, "verdict": decide(found), **found, "elapsed_s": elapsed, "cpus": cpus,
              "clip_list_sha256": prepared["clip_list_sha256"],
              "excluded_by_reason": prepared["counts"]["excluded_by_reason"]}  # fmt: skip
    prosody._write_json(out / RESULT_FILE, result)
    return result


SUMMARY_KEYS = (
    "verdict", "reason", "verified", "problems", "experiment", "commit", "uv_lock_sha256",
    "python", "modules", "files", "data_files", "passed", "r0", "s2", "auc", "ci95", "coverage",
    "pairs", "ips_with_pairs", "speakers_with_pairs", "fresh", "overlap", "fetched", "survey",
    "kept", "excluded_by_reason",
)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.prosody_rupro")
    parser.add_argument("--decision", type=Path, default=DECISION_DOC)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify", "survey", "prepare", "check-fresh", "sanity", "run"):
        commands.add_parser(name)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("tree", type=Path)
    frozen.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    commands.add_parser("verify-frozen").add_argument("tree", type=Path)
    args = parser.parse_args(argv)
    try:
        config_ = config.load()
        if args.command != "fetch":
            guard.install_network_guard(config_)
        elif not config_.network.allow:
            raise RuproError("fetch needs the network: [network] allow = true in its own config")
        if args.command == "fetch":
            result = fetch(_mkdir(output_dir(config_) / SOURCES_DIR))
        elif args.command == "verify":
            problems, _ = verify(sources_dir(config_))
            result = {"verified": not problems, "problems": problems}
        elif args.command == "freeze":
            manifest = freeze(args.tree, args.experiment, config_.data_root)
            result = verify_frozen(args.tree, config_.data_root) | {"files": len(manifest["files"])}
        elif args.command == "verify-frozen":
            result = verify_frozen(args.tree, config_.data_root)
        else:
            step = {"survey": survey, "prepare": prepare, "check-fresh": run_check_fresh,
                    "sanity": run_sanity, "run": run_main}[args.command]  # fmt: skip
            result = step(config_, args.decision)
    except (
        RuproError,
        prosody.ProsodyError,
        corpus.CorpusError,
        config.ConfigError,
        paths.PathDenied,
        OSError,
        KeyError,
        ET.ParseError,
    ) as error:
        print(f"prosody-rupro: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in SUMMARY_KEYS if k in result}, ensure_ascii=False))
    if args.command in ("verify", "verify-frozen", "freeze") and not result["verified"]:
        return 1
    if args.command == "check-fresh" and not (result["fresh"] and result["files"] == "ok"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
