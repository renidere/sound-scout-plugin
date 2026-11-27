"""Forced choice of the stressed word: Qwen2.5-Omni-7B, audio against text only, local CPU (#205).

The preregistration is docs/decisions/0036-forced-choice-stress.md; its block
`FORCED-CHOICE-RULE v1` has the SHA-256 `RULE_SHA256`, and the code refuses to run under any
other block. Three commands, `python -m sound_scout.eval.forced_choice <cmd>`:

- `prepare` sends nothing to a model and starts no server: it checks the pins of the weights and
  of the runtime by SHA-256, selects 20 utterances of E-R of `gpu-ext-149` by the fixed rule
  (`select`) and writes `set.json` (clips, transcripts, word pairs) and `plan.json` (the 86
  requests) to `<data_root>/gate0/forced-choice-205/`. Both files stay outside git; the
  preregistration pins their SHA-256.
- `run` checks the rule, the pins, the files and the limits (cgroup `memory.max` <= 18e9 without
  swap, at most 8 CPUs, `MemAvailable`, no other inference), starts `llama-server` b12359 with
  the runtime `qwen25omni` of `gate0` on 127.0.0.1 and a free port, and sends each request of the
  plan once, in order. A request over its time, the total time limit, the RSS watchdog or a
  server exit stops the run; the rest are `not_sent`. Nothing is retried. A second attempt
  needs `--owner-decision <link>`.
- `analyze [--attempt K]` computes the preregistered numbers and the conclusion from the saved
  attempt alone (`analysis.json` without words, the local `report.md` with them).

A test with a fake server proves this contract only, never what the model hears (CLAUDE.md).
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sound_scout import config, guard, paths, proc, review
from sound_scout.eval import gate0, gpu_ext, prosody

DECISION_DOC = Path("docs/decisions/0036-forced-choice-stress.md")
RULE_BEGIN = "<!-- forced-choice-rule:begin -->"
RULE_END = "<!-- forced-choice-rule:end -->"
RULE_VERSION = "FORCED-CHOICE-RULE v1"
RULE_HEADER = f"```text\n{RULE_VERSION} "
RULE_SHA256 = "366445600bcf6d032ca473a075364c62bd6767a894ab90922c5d5ca4203f3de5"

# what `prepare` wrote on the station (0036 section 2); `run` refuses other files
SET_SHA256 = "500c60fcb1755bad365022dac6e3f79b5ce7b22b6b40e84547115e921a76cabe"
PLAN_SHA256 = "68da1cd065b865cd49732e59b2bc37b3c8d816279202e2e49e26ad3c9ed803f9"

SET_DIR = Path("gate0/gpu-ext-149")
CONTROL_DIR = Path("gate0/2026-11-19")
PILOT_DIR = Path("corpus/pilot-1")
OUT_DIR = Path("gate0/forced-choice-205")
SET_FILE = "set.json"
PLAN_FILE = "plan.json"
RUN_FILE = "run.json"
TRACE_FILE = "trace.jsonl"
ANALYSIS_FILE = "analysis.json"
REPORT_FILE = "report.md"
SERVER_LOG = "server.log"
ATTEMPTS_DIR = "attempts"
REQUESTS_DIR = "requests"
RESPONSES_DIR = "responses"
FORMAT_VERSION = 1
# the 0028 GPU attempt of the same runtime family: its P answers, described only (0036 section 6)
GPU_ATTEMPT = SET_DIR / "attempts" / "20261123T054739Z-0ae97bef-qwen25omni"

UTTERANCES = 20
CONTROLS = ("S0", "N1", "T1")
ORDERS = ("MU", "UM")
MODALITIES = ("audio", "text")
MAX_REQUESTS = 86
TEMPLATE = (
    "Вот расшифровка записи: «{transcript}». Какое из двух слов выделено голосом сильнее "
    "остальных: 1 — «{w1}»; 2 — «{w2}»? Ответь одной цифрой: 1, 2 или 0 — если определить "
    "невозможно или на записи нет речи."
)
CONTROL_TRANSCRIPT = "сегодня вечером"
CONTROL_WORDS = ("сегодня", "вечером")
ANSWERS = ("0", "1", "2")
INVALID = "invalid"
SELECT_COUNTS = ("e_r", "pilot_1", "second_of_wav", "targets_not_once", "valid", "selected")
OUTCOMES = ("correct", "wrong", "zero", "invalid", "timeout", "error", "not_sent")

RUNTIME = "qwen25omni"
THREADS = 8
MAX_CPUS = 8
MEMORY_CAP_BYTES = gate0.QWEN25OMNI_MEMORY_CAP_BYTES
RSS_LIMIT_BYTES = gate0.QWEN25OMNI_RSS_LIMIT_BYTES
MIN_AVAILABLE_BYTES = gate0.QWEN25OMNI_MIN_AVAILABLE_BYTES
REQUEST_TIMEOUT_S = 120.0
WALL_LIMIT_S = 4500.0
LOAD_TIMEOUT_S = gate0.LOAD_TIMEOUT_S
EXIT_WAIT_S = 1.0
MAX_TOKENS = gate0.CLOSED_MAX_TOKENS
SAMPLING = gate0.sampling(RUNTIME)
SERVER_LOG_MAX_BYTES = gate0.SERVER_LOG_MAX_BYTES

BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 0
CI_PERCENTILES = (2.5, 97.5)
CHANCE = 0.5
CONTROL_NOTE = "контроли не пройдены: ответ из текста вместо звука не исключён"
BUSY_MODULES = (b"sound_scout.eval.prosody", b"sound_scout.eval.prosody_confirm")


class ForcedChoiceError(Exception):
    """The experiment cannot start or go on as 0036 says; nothing was sent after it."""


@dataclass(frozen=True)
class Pins:
    """The local weights and runtime the run may use, each by SHA-256 (docs/dependencies.md)."""

    model_dir: Path = gate0.QWEN25OMNI_MODEL_DIR
    weights: Mapping[str, str] = field(default_factory=lambda: dict(gate0.QWEN25OMNI_WEIGHTS))
    llama_dir: Path = gate0.DEFAULT_LLAMA_DIR
    archive: Path = gate0.DEFAULT_LLAMA_DIR.parent / "llama-b12359-bin-ubuntu-x64.tar.gz"
    archive_sha256: str = "0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e"
    server_sha256: str = "cf07a7d6b6d2c61064105cc0a81360cb5830aacd3d1a949b80eaea3a96fa409f"
    server_version: str = "version: 1.0.0-dev (build 12359, commit b1b770d90)"


@dataclass(frozen=True)
class Sources:
    """The read-only inputs of `prepare` by SHA-256: the set of #149, the controls' day and
    `pilot-1` (0036 section 2)."""

    set_manifest_sha256: str = "6cd25e0503cebd921ea594d9236403acacbdbfe881d321892dbca3f2f2bb3aaf"
    set_plan_sha256: str = "ab822b01d67c840443dcec1ca8620c492939154083024cc2b9f0b843f456da21"
    control_manifest_sha256: str = (
        "b00e9cb16a5dae9ffe5d01ddbed7d6741494a387b12e4de5e426a0989e5c8c1b"
    )
    pilot_provenance_sha256: str = (
        "d72d8152c24542996b00f48800dedc35ba8a080cfb5977cdd7db2f0f4175fadf"
    )
    pilot_manifest_sha256: str = "bae774d9c6d2ae680cefcab49a5d152c7b638d52bf1057825ec74c2d4b4e0444"


# --- the rule ------------------------------------------------------------------------------


def rule_block(document: str) -> str:
    if document.count(RULE_BEGIN) != 1 or document.count(RULE_END) != 1:
        raise ForcedChoiceError(f"the decision must hold one {RULE_BEGIN} … {RULE_END} block")
    begin = document.index(RULE_BEGIN) + len(RULE_BEGIN)
    end = document.index(RULE_END)
    block = document[begin:end].strip()
    if end < begin or not block.startswith(RULE_HEADER):
        raise ForcedChoiceError(f"the rule block does not start with {RULE_VERSION}")
    return block


def rule_sha256(document: str) -> str:
    return hashlib.sha256(rule_block(document).encode("utf-8")).hexdigest()


def check_rule(decision: Path) -> str:
    """The rule SHA-256 when it is the one pinned before the run; otherwise refuse."""
    try:
        document = decision.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ForcedChoiceError(f"cannot read the decision {decision.name}: {error}") from None
    found = rule_sha256(document)
    if found != RULE_SHA256:
        raise ForcedChoiceError(f"{RULE_VERSION} SHA-256 is {found}, pinned {RULE_SHA256}")
    if TEMPLATE not in flat(rule_block(document)):
        raise ForcedChoiceError("the question template of the code is not the one of the rule")
    return found


def flat(text: str) -> str:
    return " ".join(text.split())


# --- pins ----------------------------------------------------------------------------------


def check_pins(pins: Pins) -> tuple[dict, list[str]]:
    """Every pinned file against its SHA-256, the unpacked runtime against its archive; hashing
    only: nothing is downloaded, replaced or run. Returns the findings and the problems."""
    found: dict = {"weights": {}, "archive": None, "server": None, "runtime_tree": None}
    problems = []
    for name, expected in pins.weights.items():
        sha = gate0._sha256_file(pins.model_dir / name)
        found["weights"][name] = sha
        if sha is None:
            problems.append(f"weight file {name} is missing in {pins.model_dir}")
        elif sha != expected:
            problems.append(f"weight file {name} has SHA-256 {sha}, pinned {expected}")
    found["archive"] = gate0._sha256_file(pins.archive)
    if found["archive"] is None:
        problems.append(f"runtime archive {pins.archive.name} is missing")
    elif found["archive"] != pins.archive_sha256:
        problems.append(f"runtime archive has SHA-256 {found['archive']}, pinned")
    else:
        found["runtime_tree"] = _tree_differences(pins.archive, pins.llama_dir.parent)
        problems += [
            f"runtime file {name} differs from the archive" for name in found["runtime_tree"]
        ]
    found["server"] = gate0._sha256_file(pins.llama_dir / gate0.SERVER_BINARY)
    if found["server"] is None:
        problems.append(f"runtime {gate0.SERVER_BINARY} is missing in {pins.llama_dir}")
    elif found["server"] != pins.server_sha256:
        problems.append(f"{gate0.SERVER_BINARY} has SHA-256 {found['server']}, pinned")
    return found, problems


def _tree_differences(archive: Path, base: Path) -> list[str]:
    """Members of the archive whose unpacked file under `base` differs or is missing, and files
    of the unpacked top directories the archive does not hold."""
    differs, members, tops = [], set(), set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name.rstrip("/")
            members.add(name)
            tops.add(Path(name).parts[0])
            target = base / name
            if member.isdir():
                if not target.is_dir() or target.is_symlink():
                    differs.append(name)
            elif member.issym():
                if not target.is_symlink() or os.readlink(target) != member.linkname:
                    differs.append(name)
            elif member.isfile():
                digest = hashlib.sha256()
                source = tar.extractfile(member)
                while chunk := source.read(1 << 20):
                    digest.update(chunk)
                if target.is_symlink() or gate0._sha256_file(target) != digest.hexdigest():
                    differs.append(name)
            else:
                differs.append(name)
    for top in sorted(tops):
        for path in sorted((base / top).rglob("*")):
            name = path.relative_to(base).as_posix()
            if name not in members:
                differs.append(name)
    return differs


# --- the set and the plan ------------------------------------------------------------------


def pilot_exclusions(data_root: Path) -> tuple[set[str], set[str], set[str]]:
    """RINCO members of `pilot-1`, SHA-256 of all its clips, and SHA-256 of its held-out ones."""
    provenance = _read_json(data_root / PILOT_DIR / "provenance.json")
    manifest = _read_json(data_root / PILOT_DIR / "manifest.json")
    members = {c["member"] for c in provenance["clips"] if c.get("member")}
    hashes = {c["sha256"] for c in provenance["clips"]} | {r["sha256"] for r in manifest["records"]}
    heldout = {r["sha256"] for r in manifest["records"] if r.get("split") == "heldout"}
    return members, hashes, heldout


def select(
    clips: list[Mapping], members: set[str], hashes: set[str], count: int = UTTERANCES
) -> tuple[list[dict], dict]:
    """The rule of 0036 section 3: E-R in manifest order; skip a clip of `pilot-1` (member or
    WAV SHA-256); one utterance per WAV, the first; both target forms once in the transcript;
    the first `count`. Old model answers and acoustic values are never read. Refuses below
    `count` with the shortfall."""
    counts = Counter(dict.fromkeys(SELECT_COUNTS, 0))
    seen, chosen = set(), []
    for position, clip in enumerate(c for c in clips if c["block"] == "E-R"):
        counts["e_r"] += 1
        if clip["member"] in members or clip["wav_sha256"] in hashes:
            counts["pilot_1"] += 1
            continue
        if clip["member"] in seen:
            counts["second_of_wav"] += 1
            continue
        seen.add(clip["member"])
        words = clip["transcript"].split()
        pair = clip["targets"]
        if pair["marked"] == pair["unmarked"] or any(words.count(pair[k]) != 1 for k in pair):
            counts["targets_not_once"] += 1
            continue
        counts["valid"] += 1
        if len(chosen) < count:
            chosen.append({"position": position, **clip})
    counts["selected"] = len(chosen)
    if len(chosen) < count:
        raise ForcedChoiceError(
            f"only {len(chosen)} valid utterances, the run needs {count}: "
            f"{count - len(chosen)} missing ({dict(counts)}); the run does not start"
        )
    return chosen, dict(counts)


def build_plan(utterances: list[Mapping], controls: list[Mapping]) -> list[dict]:
    """6 control requests (2 orders each, audio), then per utterance audio MU, audio UM, text MU,
    text UM. `MU`: 1 = the marked word; `UM`: 1 = the unmarked one."""
    entries = []
    for clip in controls:
        for order, words in zip(ORDERS, (CONTROL_WORDS, CONTROL_WORDS[::-1]), strict=True):
            text = TEMPLATE.format(transcript=CONTROL_TRANSCRIPT, w1=words[0], w2=words[1])
            entries.append(_entry("control", clip["id"], "audio", order, "0", text))
    for clip in utterances:
        for modality in MODALITIES:
            for order in ORDERS:
                first, second = ("marked", "unmarked") if order == "MU" else ("unmarked", "marked")
                text = TEMPLATE.format(
                    transcript=clip["transcript"],
                    w1=clip[first],
                    w2=clip[second],
                )
                expected = "1" if order == "MU" else "2"
                entries.append(_entry("utterance", clip["id"], modality, order, expected, text))
    for number, entry in enumerate(entries, 1):
        entry["n"] = number
    if len(entries) > MAX_REQUESTS:
        raise ForcedChoiceError(f"the plan holds {len(entries)} requests, at most {MAX_REQUESTS}")
    return entries


def _entry(kind, clip, modality, order, expected, text) -> dict:
    return {
        "kind": kind,
        "clip": clip,
        "modality": modality,
        "order": order,
        "expected": expected,
        "text": text,
        "max_tokens": MAX_TOKENS,
    }


def build_set(data_root: Path, sources: Sources = Sources()) -> tuple[dict, dict]:
    """`set.json` and `plan.json` contents from the pinned read-only inputs."""
    set_dir = data_root / SET_DIR
    manifest, _ = gpu_ext.load_set(
        set_dir,
        manifest_sha256=sources.set_manifest_sha256,
        plan_sha256=sources.set_plan_sha256,
    )
    for name, pin in (
        ("provenance.json", sources.pilot_provenance_sha256),
        ("manifest.json", sources.pilot_manifest_sha256),
    ):
        if gate0._sha256_file(data_root / PILOT_DIR / name) != pin:
            raise ForcedChoiceError(f"pilot-1 {name} differs from its pinned SHA-256")
    members, hashes, heldout = pilot_exclusions(data_root)
    chosen, counts = select(manifest["clips"], members, hashes)
    control_manifest = data_root / CONTROL_DIR / gate0.MANIFEST_FILE
    if gate0._sha256_file(control_manifest) != sources.control_manifest_sha256:
        raise ForcedChoiceError("the controls' manifest differs from its pinned SHA-256")
    day = _read_json(control_manifest)
    controls = []
    for clip_id in CONTROLS:
        clip = next((c for c in day["clips"] if c["id"] == clip_id), None)
        if clip is None:
            raise ForcedChoiceError(f"control {clip_id} is not in the controls' manifest")
        if gate0._sha256_file(data_root / CONTROL_DIR / clip["file"]) != clip["wav_sha256"]:
            raise ForcedChoiceError(f"control {clip_id} differs from its SHA-256")
        controls.append(_clip_record(CONTROL_DIR, clip))
    utterances = [
        {
            **_clip_record(SET_DIR, clip),
            "position": clip["position"],
            "transcript": clip["transcript"],
            "marked": clip["targets"]["marked"],
            "unmarked": clip["targets"]["unmarked"],
        }
        for clip in chosen
    ]
    taken = [c["id"] for c in utterances + controls if c["wav_sha256"] in heldout]
    if taken:
        raise ForcedChoiceError(f"Gate 1 held-out clips in the set: {taken}")
    data = {
        "format_version": FORMAT_VERSION,
        "kind": "forced_choice_set",
        "sources": {
            "set_manifest_sha256": sources.set_manifest_sha256,
            "set_plan_sha256": sources.set_plan_sha256,
            "control_manifest_sha256": sources.control_manifest_sha256,
            "pilot_provenance_sha256": sources.pilot_provenance_sha256,
            "pilot_manifest_sha256": sources.pilot_manifest_sha256,
        },
        "counts": counts,
        "utterances": utterances,
        "controls": controls,
    }
    plan = {
        "format_version": FORMAT_VERSION,
        "kind": "forced_choice_plan",
        "entries": build_plan(utterances, controls),
    }
    return data, plan


def _clip_record(directory: Path, clip: Mapping) -> dict:
    return {
        "id": clip["id"],
        "file": (directory / clip["file"]).as_posix(),
        "wav_sha256": clip["wav_sha256"],
        "pcm_sha256": clip["pcm_sha256"],
        "duration_s": clip["duration_s"],
    }


def prepare(config_: config.Config, *, pins: Pins = Pins(), sources: Sources = Sources()) -> dict:
    """The pin check and the selection; no server, no request. Writes `set.json` and
    `plan.json` once: a file already there must hold the same bytes."""
    data_root = config_.data_root
    found, problems = check_pins(pins)
    if problems:
        raise ForcedChoiceError("pins do not hold, nothing is downloaded: " + "; ".join(problems))
    data, plan = build_set(data_root, sources)
    out = _out_dir(config_)
    hashes = {}
    for name, content in ((SET_FILE, data), (PLAN_FILE, plan)):
        raw = _json_bytes(content)
        path = out / name
        if path.exists():
            if path.read_bytes() != raw:
                raise ForcedChoiceError(f"{name} is there with other content; remove it first")
        else:
            with paths.open_output(out, name, config_) as file:
                file.write(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    entries = plan["entries"]
    return {
        "counts": data["counts"],
        "controls": len(data["controls"]),
        "requests": len(entries),
        "audio_requests": sum(e["modality"] == "audio" for e in entries),
        "set_sha256": hashes[SET_FILE],
        "plan_sha256": hashes[PLAN_FILE],
        "pins": "ok",
        "weights_sha256": found["weights"],
    }


def load_prepared(
    data_root: Path, set_sha256: str = SET_SHA256, plan_sha256: str = PLAN_SHA256
) -> tuple[dict, list[dict]]:
    """`set.json` and `plan.json` against their pins, every clip on disk, the plan against the
    set (texts, order, at most `MAX_REQUESTS`, one audio file per audio request)."""
    out = data_root / OUT_DIR
    loaded = {}
    for name, pin in ((SET_FILE, set_sha256), (PLAN_FILE, plan_sha256)):
        try:
            raw = (out / name).read_bytes()
        except OSError:
            raise ForcedChoiceError(f"{name} is missing; run prepare first") from None
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ForcedChoiceError(f"{name} differs from the SHA-256 0036 pins")
        loaded[name] = json.loads(raw)
    data, entries = loaded[SET_FILE], loaded[PLAN_FILE]["entries"]
    if len(data["utterances"]) != UTTERANCES or [c["id"] for c in data["controls"]] != list(
        CONTROLS
    ):
        raise ForcedChoiceError("set.json does not hold 20 utterances and the controls S0 N1 T1")
    for clip in data["utterances"] + data["controls"]:
        if gate0._sha256_file(data_root / clip["file"]) != clip["wav_sha256"]:
            raise ForcedChoiceError(f"clip {clip['id']} differs from its SHA-256")
    if entries != build_plan(data["utterances"], data["controls"]):
        raise ForcedChoiceError("plan.json is not the plan of set.json under this code")
    return data, entries


# --- the run -------------------------------------------------------------------------------


def busy_processes() -> list[str]:
    """Other inference, a prosody run (#199) or a pytest on this host; this run waits for them."""
    found = gate0._other_inference()
    own = str(os.getpid())
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or entry.name == own:
            continue
        with contextlib.suppress(OSError):
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            if any(arg.endswith(b"pytest") or arg == b"pytest" for arg in argv[:3]):
                found.append("pytest")
            found += [m.decode() for m in BUSY_MODULES if m in argv]
    return sorted(set(found))


def parse_answer(body: object) -> str:
    """`1`, `2` or `0` when the content is exactly that digit after stripping blanks and the
    edges of `gate0._EDGES`; `invalid` otherwise, also for an empty or a length-cut answer."""
    try:
        finish = body["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return INVALID
    content = gate0._content(body)
    if finish == "length" or content is None:
        return INVALID
    answer = content.strip(gate0._EDGES)
    return answer if answer in ANSWERS else INVALID


def outcome_of(line: Mapping | None, body: object, expected: str) -> str:
    """The one outcome of a request from its trace line and its saved response body."""
    if not line or not line.get("sent"):
        return "not_sent"
    if line.get("limit") in ("request_timeout", "wall_limit"):
        return "timeout"
    if line.get("limit") or line.get("http_status") != 200:
        return "error"
    answer = parse_answer(body)
    if answer == INVALID:
        return INVALID
    if answer == expected:
        return "correct"
    return "zero" if answer == "0" else "wrong"


def run(
    config_: config.Config,
    *,
    decision: Path = DECISION_DOC,
    pins: Pins = Pins(),
    set_sha256: str = SET_SHA256,
    plan_sha256: str = PLAN_SHA256,
    request_timeout: float = REQUEST_TIMEOUT_S,
    wall_limit: float = WALL_LIMIT_S,
    load_timeout: float = LOAD_TIMEOUT_S,
    rss_limit: int = RSS_LIMIT_BYTES,
    cgroup_root: Path = gate0.CGROUP_ROOT,
    meminfo: Path = gate0.MEMINFO,
    busy: Callable[[], list[str]] = busy_processes,
    pin_cpus: Callable[[], list[int]] = prosody.pin_cpus,
    owner_decision: str | None = None,
) -> dict:
    """One attempt under `<out>/attempts/<k>/`; every check before the server start refuses
    without an attempt. A finished attempt already there refuses a new one unless
    `owner_decision` cites the owner's decision for it (0036 section 3). Returns the analysis
    of the attempt."""
    data_root = config_.data_root
    rule = check_rule(decision)
    data, entries = load_prepared(data_root, set_sha256, plan_sha256)
    earlier = finished_attempts(data_root)
    if earlier and not (owner_decision or "").startswith("https://"):
        raise ForcedChoiceError(
            f"attempts {earlier} already ran; a new one needs --owner-decision <link>"
        )
    cgroup = gate0.memory_cap(cgroup_root)
    gate0.check_memory_cap(cgroup, MEMORY_CAP_BYTES, "forced-choice")
    cpus = pin_cpus()
    if not 0 < len(cpus) <= MAX_CPUS:
        raise ForcedChoiceError(f"the run keeps to at most {MAX_CPUS} CPUs, got {cpus}")
    host = gate0.check_host_memory(MIN_AVAILABLE_BYTES, meminfo)
    others = busy()
    if others:
        raise ForcedChoiceError(f"other heavy work runs on this host: {others}; wait for it")
    found, problems = check_pins(pins)
    binary = pins.llama_dir / gate0.SERVER_BINARY
    version = None if problems else gate0._server_version([str(binary), "--version"])
    if not problems and version != pins.server_version:
        problems.append(f"{gate0.SERVER_BINARY} --version says {version!r}")
    if problems:
        raise ForcedChoiceError("pins do not hold, nothing is downloaded: " + "; ".join(problems))

    out = _out_dir(config_)
    attempt = gate0._new_attempt(out, config_)
    port = gate0._free_port()
    argv = gate0.server_argv(binary, pins.model_dir, port, THREADS, gate0.LLAMA_RUNTIMES[RUNTIME])
    client = gate0.Client(port, None, SAMPLING["chat_template_kwargs"])
    clips = {clip["id"]: clip for clip in data["utterances"] + data["controls"]}
    record: dict = {
        "format_version": FORMAT_VERSION,
        "kind": "forced_choice_run",
        "attempt": attempt.name,
        "started_at": gate0._now(),
        "code": {**gate0._code_identity(), "forced_choice_sha256": _sha256_self()},
        "rule_sha256": rule,
        "set_sha256": set_sha256,
        "plan_sha256": plan_sha256,
        "runtime": RUNTIME,
        "pins": found,
        "argv": argv,
        "server_version": version,
        "threads": THREADS,
        "cpus": cpus,
        "cpu": gate0._cpu_model(),
        "loadavg": os.getloadavg(),
        "cgroup": cgroup,
        "host_memory": host,
        "limits": {
            "request_timeout_s": request_timeout,
            "wall_limit_s": wall_limit,
            "load_timeout_s": load_timeout,
            "rss_limit_bytes": rss_limit,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
        },
        "sampling": SAMPLING,
        "requests_planned": len(entries),
        "earlier_attempts": earlier,
        "owner_decision": owner_decision if earlier else None,
    }
    gate0._mkdir(attempt / REQUESTS_DIR, config_)
    gate0._mkdir(attempt / RESPONSES_DIR, config_)
    stop = None
    with (
        paths.open_output(attempt, TRACE_FILE, config_) as trace,
        paths.open_output(attempt, SERVER_LOG, config_) as log,
    ):
        started = time.monotonic()
        deadline = started + wall_limit
        process, copier = proc.start(argv, log=log, max_log_bytes=SERVER_LOG_MAX_BYTES)
        watch = gate0.RssWatch(process, rss_limit).start()
        try:
            stop = gate0._wait_ready(client, process, watch, min(load_timeout, wall_limit))
            record["model_load_s"] = round(time.monotonic() - started, 3)
            for entry in entries:
                remaining = deadline - time.monotonic()
                if stop is None and remaining <= 0:
                    stop = "wall_limit"
                if stop is not None:
                    gate0._trace(trace, {**_keys(entry), "sent": False, "not_sent": stop})
                    continue
                timeout = min(request_timeout, remaining)

                line = _send(entry, clips, data_root, client, watch, attempt, config_, timeout)
                if line.get("limit") == "request_timeout" and timeout < request_timeout:
                    line["limit"] = "wall_limit"
                line["outcome"] = outcome_of(line, _body(attempt, entry["n"]), entry["expected"])
                gate0._trace(trace, line)
                if line.get("http_status") is None and not line.get("limit"):
                    # a dying server closes the socket a moment before it can be reaped
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(EXIT_WAIT_S)
                exited = process.poll() is not None
                stop = line.get("limit") or ("server_exited" if exited else None)
        except BaseException as error:
            cancelled = isinstance(error, KeyboardInterrupt | SystemExit)
            stop = stop or ("cancelled" if cancelled else "error")
            raise
        finally:
            status = gate0.proc_status(process.pid)
            watch.stop()
            gate0.stop_server(process)
            copier.join(gate0.STOP_GRACE_S)
            record.update(
                {
                    "finished_at": gate0._now(),
                    "wall_s": round(time.monotonic() - started, 3),
                    "stop_reason": stop,
                    "complete": stop is None,
                    "rss_exceeded_bytes": watch.exceeded,
                    "rss_peak_bytes": watch.peak,
                    "vm_hwm_bytes": status.get("VmHWM"),
                    "server_returncode": process.returncode,
                    "http_calls": client.calls,
                }
            )
            record["cgroup"].update(gate0.memory_usage(cgroup, cgroup_root))
            gate0._write_json(attempt, RUN_FILE, record, config_)
    return analyze_attempt(config_, attempt.name, set_sha256=set_sha256, plan_sha256=plan_sha256)


def finished_attempts(data_root: Path) -> list[str]:
    """Attempts that wrote their `run.json`, in number order."""
    attempts = data_root / OUT_DIR / ATTEMPTS_DIR
    if not attempts.is_dir():
        return []
    found = [p for p in attempts.iterdir() if p.name.isdigit() and (p / RUN_FILE).is_file()]
    return [p.name for p in sorted(found, key=lambda p: int(p.name))]


def _keys(entry: Mapping) -> dict:
    keys = ("n", "kind", "clip", "modality", "order", "expected")
    return {key: entry[key] for key in keys}


def _send(entry, clips, data_root, client, watch, attempt, config_, timeout) -> dict:
    line = {**_keys(entry), "sent": True, "timeout_s": round(timeout, 3)}
    content: list[dict] = [{"type": "text", "text": entry["text"]}]
    redacted: list[dict] = [{"type": "text", "text": entry["text"]}]
    if entry["modality"] == "audio":
        clip = clips[entry["clip"]]
        wav = (data_root / clip["file"]).read_bytes()
        pcm = gate0.wav_pcm(wav)
        duration = len(pcm) / 2 / gate0.SAMPLE_RATE
        payload = review.Clip(
            clip["id"], pcm, gate0.SAMPLE_RATE, 1, "s16le", duration, clip["pcm_sha256"]
        )
        content.insert(0, gate0._audio_part(payload, wav))
        hashes = {
            "wav_sha256": hashlib.sha256(wav).hexdigest(),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "duration_s": duration,
        }
        line.update(hashes)
        redacted.insert(0, {"type": "input_audio", "input_audio": {"format": "wav", **hashes}})
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": entry["max_tokens"],
        **SAMPLING,
    }
    gate0._write_json(
        attempt / REQUESTS_DIR,
        f"{entry['n']}.json",
        {**body, "messages": [{"role": "user", "content": redacted}]},
        config_,
    )
    line["rss_before_bytes"] = watch.current()
    watch.reset_window()
    started = time.monotonic()
    status, raw, error = client.request(gate0.CHAT_ROUTE, body, timeout=timeout)
    line["wall_s"] = round(time.monotonic() - started, 3)
    line.update(
        {
            "rss_after_bytes": watch.current(),
            "rss_peak_bytes": watch.window_peak,
            "http_status": status,
            "error": error,
        }
    )
    parsed = None
    if raw is not None:
        with paths.open_output(attempt / RESPONSES_DIR, f"{entry['n']}.json", config_) as file:
            file.write(raw)
        line["response_sha256"] = hashlib.sha256(raw).hexdigest()
        with contextlib.suppress(ValueError):
            parsed = json.loads(raw)
    if isinstance(parsed, dict):
        usage = parsed.get("usage") or {}
        choice = (parsed.get("choices") or [{}])[0]
        line.update(
            {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                "timings": parsed.get("timings"),
            }
        )
    line["answer"] = parse_answer(parsed)
    if watch.exceeded is not None:
        line["limit"] = "rss_limit"
    elif line["wall_s"] > timeout or (error is not None and "timed out" in error):
        line["limit"] = "request_timeout"
    return line


# --- the analysis --------------------------------------------------------------------------


def analyze_attempt(
    config_: config.Config,
    name: str | None = None,
    *,
    set_sha256: str = SET_SHA256,
    plan_sha256: str = PLAN_SHA256,
) -> dict:
    """`analyze` on attempt `name` (the last one by default); writes `analysis.json` and the
    local `report.md` into the attempt."""
    data_root = config_.data_root
    attempts = data_root / OUT_DIR / ATTEMPTS_DIR
    numbers = (
        [int(p.name) for p in attempts.iterdir() if p.name.isdigit()] if attempts.is_dir() else []
    )
    name = name or (str(max(numbers)) if numbers else "")
    attempt = attempts / name
    if not name.isdigit() or not (attempt / RUN_FILE).is_file():
        raise ForcedChoiceError(f"attempt {name or '-'} not found; run `run` first")
    record = _read_json(attempt / RUN_FILE)
    if (record.get("set_sha256"), record.get("plan_sha256")) != (set_sha256, plan_sha256):
        raise ForcedChoiceError("the attempt ran on other set or plan files than these pins")
    data, entries = load_prepared(data_root, set_sha256, plan_sha256)
    lines = [json.loads(text) for text in (attempt / TRACE_FILE).read_text("utf-8").splitlines()]
    result = analyze(record, entries, lines, lambda n: _body(attempt, n))
    result["gpu_0028"] = describe_0028(data_root / GPU_ATTEMPT, data["utterances"])
    _replace(attempt, ANALYSIS_FILE, _json_bytes(result), config_)
    _replace(attempt, REPORT_FILE, report_text(result, data).encode("utf-8"), config_)
    return result


def analyze(
    record: Mapping, entries: list[Mapping], lines: list[Mapping], body_of: Callable
) -> dict:
    """The numbers and the conclusion of the rule of 0036 section 3 from the saved answers."""
    by_n = {line["n"]: line for line in lines}
    rows = []
    for entry in entries:
        line = by_n.get(entry["n"])
        sent = bool(line and line.get("sent"))
        body = body_of(entry["n"]) if sent else None
        outcome = outcome_of(line, body, entry["expected"])
        answer = parse_answer(body) if outcome not in ("not_sent", "timeout", "error") else None
        rows.append({**_keys(entry), "outcome": outcome, "answer": answer, "line": line or {}})
    utterance_rows = [row for row in rows if row["kind"] == "utterance"]
    clips = list(dict.fromkeys(row["clip"] for row in utterance_rows))
    score = {m: np.array([_score(utterance_rows, c, m) for c in clips]) for m in MODALITIES}
    conditions = {}
    for modality in MODALITIES:
        chosen = [row for row in utterance_rows if row["modality"] == modality]
        conditions[modality] = {
            "accuracy": _share([row["outcome"] == "correct" for row in chosen]),
            "accuracy_by_order": {
                order: _share(
                    [row["outcome"] == "correct" for row in chosen if row["order"] == order]
                )
                for order in ORDERS
            },
            "outcomes": _outcome_shares(chosen),
            "order_disagreement": _share(
                [_disagree(_named(chosen, c, "MU"), _named(chosen, c, "UM")) for c in clips]
            ),
            "answer_1_among_1_2": _share(
                [row["answer"] == "1" for row in chosen if row["answer"] in ("1", "2")]
            ),
        }
    boot = bootstrap(score["audio"], score["text"])
    conditions["audio"]["accuracy_ci"] = boot["audio"]
    conditions["text"]["accuracy_ci"] = boot["text"]
    gain = float(np.mean(score["audio"] - score["text"])) if clips else None
    control_rows = [row for row in rows if row["kind"] == "control"]
    controls_passed = bool(control_rows) and all(row["answer"] == "0" for row in control_rows)
    complete = record.get("stop_reason") is None and all(r["outcome"] != "not_sent" for r in rows)
    if not complete:
        verdict = "incomplete"
    elif boot["gain"] is not None and boot["gain"][0] > 0 and boot["audio"][0] > CHANCE:
        verdict = "audio advantage shown"
    else:
        verdict = "not shown"
    return {
        "format_version": FORMAT_VERSION,
        "kind": "forced_choice_analysis",
        "attempt": record.get("attempt"),
        "rule_sha256": record.get("rule_sha256"),
        "verdict": verdict,
        "stop_reason": record.get("stop_reason"),
        "complete": complete,
        "controls_passed": controls_passed,
        "note": None if controls_passed else CONTROL_NOTE,
        "requests": {
            "planned": len(entries),
            "sent": sum(bool(r["line"].get("sent")) for r in rows),
        },
        "conditions": conditions,
        "outcomes_all_utterances": _outcome_shares(utterance_rows),
        "audio_gain": {"mean": _round(gain), "ci": boot["gain"]},
        "bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "percentiles": list(CI_PERCENTILES),
            "cluster": "utterance",
        },
        "scores": {
            clip: {"audio": float(score["audio"][k]), "text": float(score["text"][k])}
            for k, clip in enumerate(clips)
        },
        "audio_tokens": _audio_tokens(utterance_rows, clips),
        "controls": [
            {k: row[k] for k in ("n", "clip", "order", "outcome", "answer")} for row in control_rows
        ],
        "outcomes": [
            {k: row[k] for k in ("n", "kind", "clip", "modality", "order", "outcome", "answer")}
            for row in rows
        ],
    }


def _score(rows: list[Mapping], clip: str, modality: str) -> float:
    found = [r for r in rows if r["clip"] == clip and r["modality"] == modality]
    return sum(r["outcome"] == "correct" for r in found) / len(ORDERS)


def _disagree(first: str | None, second: str | None) -> bool:
    """Two orders agree only when both name the same word or both answer 0."""
    return first is None or second is None or first != second


def _named(rows: list[Mapping], clip: str, order: str) -> str | None:
    """The word an answer named: `marked`, `unmarked`, `zero`; None without a valid answer."""
    row = next((r for r in rows if r["clip"] == clip and r["order"] == order), None)
    answer = None if row is None else row["answer"]
    if answer == "0":
        return "zero"
    if answer not in ("1", "2"):
        return None
    return "marked" if (answer == "1") == (order == "MU") else "unmarked"


def bootstrap(audio: np.ndarray, text: np.ndarray) -> dict:
    """95 % percentile CI of the audio accuracy, the text accuracy and their mean difference,
    one draw of utterances (clusters of 4 answers) for all three."""
    if len(audio) == 0:
        return {"audio": None, "text": None, "gain": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, len(audio), size=(BOOTSTRAP_REPEATS, len(audio)))
    found = {}
    for name, values in (("audio", audio[index]), ("text", text[index])):
        found[name] = values.mean(axis=1)
    found["gain"] = (audio[index] - text[index]).mean(axis=1)
    return {
        name: [_round(float(x)) for x in np.percentile(values, CI_PERCENTILES)]
        for name, values in found.items()
    }


def _audio_tokens(rows: list[Mapping], clips: list[str]) -> dict:
    """Prompt tokens of an audio request minus those of the text request of the same order."""
    differences = []
    for clip in clips:
        for order in ORDERS:
            tokens = {
                r["modality"]: r["line"].get("prompt_tokens")
                for r in rows
                if r["clip"] == clip and r["order"] == order
            }
            if all(isinstance(tokens.get(m), int) for m in MODALITIES):
                differences.append(tokens["audio"] - tokens["text"])
    if not differences:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": len(differences),
        "median": float(np.median(differences)),
        "min": min(differences),
        "max": max(differences),
    }


def _outcome_shares(rows: list[Mapping]) -> dict:
    counts = Counter(row["outcome"] for row in rows)
    return {name: _share_of(counts[name], len(rows)) for name in OUTCOMES}


def _share(values: list[bool]) -> dict:
    return _share_of(sum(values), len(values))


def _share_of(count: int, total: int) -> dict:
    return {"count": count, "of": total, "share": _round(count / total) if total else None}


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def describe_0028(attempt: Path, utterances: list[Mapping]) -> dict | None:
    """Descriptive only: the answers P and P-text of the 0028 GPU attempt on the same clips."""
    trace = attempt / gate0.TRACE_FILE
    if not trace.is_file():
        return None
    ids = {clip["id"] for clip in utterances}
    counts: dict = {}
    for text in trace.read_text("utf-8").splitlines():
        line = json.loads(text)
        clip = line.get("clip") or line.get("about")
        if line.get("block") != "E-R" or line.get("variant") not in ("P", "P-text"):
            continue
        if clip in ids:
            key = f"{line['variant']}/{line.get('target')}"
            counts.setdefault(key, Counter())[str(line.get("answer"))] += 1
    return {
        "attempt": attempt.name,
        "answers": {k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
    }


def report_text(result: Mapping, data: Mapping) -> str:
    """The local report with the words; it stays in the attempt, outside git."""
    audio, text = result["conditions"]["audio"], result["conditions"]["text"]
    lines = [
        f"# Выбор выделенного слова — попытка {result['attempt']}",
        "",
        f"Вывод: **{result['verdict']}**. Остановка: {result['stop_reason'] or 'нет'}.",
        f"Контроли: {'пройдены' if result['controls_passed'] else result['note']}.",
        "",
        "| условие | точность | CI 95 % | MU | UM | расхождение порядков | доля `1` |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, item in (("audio", audio), ("text", text)):
        lines.append(
            f"| {name} | {item['accuracy']['share']} | {item['accuracy_ci']} | "
            f"{item['accuracy_by_order']['MU']['share']} | "
            f"{item['accuracy_by_order']['UM']['share']} | "
            f"{item['order_disagreement']['share']} | {item['answer_1_among_1_2']['share']} |"
        )
    gain = result["audio_gain"]
    lines += ["", f"Выигрыш audio G = {gain['mean']}, CI 95 % {gain['ci']}.", ""]
    lines += ["| исход | audio | text |", "|---|---|---|"]
    for name in OUTCOMES:
        lines.append(
            f"| {name} | {audio['outcomes'][name]['count']} | {text['outcomes'][name]['count']} |"
        )
    lines += [
        "",
        "## Контроли",
        "",
        "| n | клип | порядок | ответ | исход |",
        "|---|---|---|---|---|",
    ]
    for row in result["controls"]:
        lines.append(
            f"| {row['n']} | {row['clip']} | {row['order']} | {row['answer']} | {row['outcome']} |"
        )
    by_clip = {clip["id"]: clip for clip in data["utterances"]}
    outcomes = {(r["clip"], r["modality"], r["order"]): r for r in result["outcomes"]}
    lines += [
        "",
        "## Реплики (локально, вне git)",
        "",
        "| клип | отмеченное | неотмеченное | audio MU | audio UM | text MU | text UM |",
        "|---|---|---|---|---|---|---|",
    ]
    for clip_id, clip in by_clip.items():
        cells = [
            str(outcomes.get((clip_id, m, o), {}).get("answer")) for m in MODALITIES for o in ORDERS
        ]
        lines.append(
            f"| {clip_id} | {clip['marked']} | {clip['unmarked']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


# --- files ---------------------------------------------------------------------------------


def _out_dir(config_: config.Config) -> Path:
    return gate0._mkdir(config_.data_root / OUT_DIR, config_)


def _replace(directory: Path, name: str, raw: bytes, config_: config.Config) -> None:
    """A derived file `analyze` writes anew each time: the old one goes first."""
    with contextlib.suppress(FileNotFoundError):
        (directory / name).unlink()
    with paths.open_output(directory, name, config_) as file:
        file.write(raw)


def _body(attempt: Path, n: int) -> object:
    try:
        return json.loads((attempt / RESPONSES_DIR / f"{n}.json").read_bytes())
    except (OSError, ValueError):
        return None


def _json_bytes(data: object) -> bytes:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ForcedChoiceError(f"{path.name} is missing") from None


def _sha256_self() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.forced_choice")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--decision", type=Path, default=DECISION_DOC)
    run_parser.add_argument("--owner-decision", help="link to the owner's decision for a new try")
    commands.add_parser("analyze").add_argument("--attempt")
    args = parser.parse_args(argv)
    try:
        config_ = config.load()
        if args.command == "prepare":
            guard.install_network_guard(config_)
            print(json.dumps(prepare(config_), ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            # no network guard: it refuses loopback too; the client has no proxy (gate0)
            result = run(config_, decision=args.decision, owner_decision=args.owner_decision)
        else:
            guard.install_network_guard(config_)
            result = analyze_attempt(config_, args.attempt)
    except (ForcedChoiceError, gate0.Gate0Error, gpu_ext.ExtError) as error:
        print(f"forced_choice: {error}", file=sys.stderr)
        return 2
    except (config.ConfigError, paths.PathDenied) as error:
        print(f"forced_choice: {error}", file=sys.stderr)
        return 2
    summary = {k: result[k] for k in ("verdict", "stop_reason", "controls_passed", "note")}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
