"""The extended set of the GPU reference run #149: open and synthetic data only.

The owner allowed it with the $3 package (#149, comment 7085422087): Piper synthetic speech
and RINCO without the files of `pilot-1`, never the Gate 1 held-out. The preregistration is
docs/decisions/0028-gpu-reference-gate0.md, section «Дополнение 2026-11-23». The blocks, in the
order a model gets them after its Gate 0 day:

- `E-C`: 12 generated clips without speech (silence, noises, tone, sweep, chord, clicks,
  speech-shaped noise, two Piper phrases played backwards); K1, K2, P, A on each.
- `E-P`: 20 Piper sentences, each as said right (`ok`) and with a defect in its target word
  (`defect`); Q-S, K1, A on each take, the two takes of a pair side by side.
- `E-R`: 96 RINCO utterances; P on one marked and one unmarked Accent word by audio, the same
  two by transcript only (P-text), K1.
- `E-A`: A on the 16 longest utterances of `E-R`, in `E-R` order.
- `E-D`: the requests of the first 10 utterances of `E-R` again, and A on the 20 s silence and
  on both takes of the first `E-P` pair: how often the same request gives the same bytes.

Clip ids are neutral (`E-<8 hex>`): the A prompt carries the id, so a role word in it would
give the answer away. The roles stay in `manifest.json` on the station. `prepare` builds the set
once from the pinned RINCO archive and Piper voice (`uv run --extra corpus-tts`), `fixture`
builds a small set of the same shape from generators alone for the dry run and the tests, and
`run_blocks` sends the plan to a vLLM server through `gate0._send`. `analyze` gives the
preregistered numbers from the saved trace and responses alone.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

from sound_scout import config, paths, review
from sound_scout.eval import corpus, gate0, prosody, synthetic

SET_NAME = "gpu-ext-149"
SALT = "gpu-ext-149"
FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
PLAN_FILE = "plan.json"
CLIPS_DIR = "clips"
ATTEMPTS_DIR = "attempts"
RUN_FILE = "run.json"
TRACE_FILE = "trace.jsonl"
ANALYSIS_FILE = "analysis.json"
BLOCKS = ("E-C", "E-P", "E-R", "E-A", "E-D")
ID_PATTERN = re.compile(r"E-[0-9a-f]{8}")
SAMPLE_RATE = gate0.SAMPLE_RATE
MAX_CLIP_S = gate0.MAX_CLIP_SECONDS

QUESTIONS = {
    "Q-S": "Правильно ли произнесено слово «{word}» на этой записи? Ответь одним из вариантов: "
    "да; нет; нет речи — если на записи нет речи.",
}
CONTROL_WORD = "сегодня"
REVIEW_MAX_TOKENS = 512
CLOSED_TIMEOUT_S = 60.0
REVIEW_TIMEOUT_S = 180.0
PROBE_HEALTH_S = 5.0
PROBE_TEXT_S = 15.0
BLOCK_TIMEOUTS = 2

ER_COUNT = 96
ER_PER_WAV = 2
ER_MIN_S, ER_MAX_S = 2.0, 30.0
ER_MIN_VOWELS = 2
EA_COUNT = 16
ED_UTTERANCES = 10
PIPER_NOISE = {"noise_scale": 0.0, "noise_w": 0.0}

HIT_MARGIN_S = 0.3
MIN_UTTERANCES = 40
MIN_PAIRS = 10
BOOTSTRAP_REPEATS = 1000
BOOTSTRAP_SEED = 0
CI_PERCENTILES = (2.5, 97.5)
ALPHA = 0.05
CHANCE_PAIR = 0.25
UTTERANCES_SERIES_3_SHA256 = "a20dd7e0efa82a379bf3d31dc8e189d7e357b30c58049ddfc60671ef7739f89c"


class ExtError(Exception):
    """The extended set cannot be built, loaded or checked; nothing is sent after it."""


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def questions_sha256() -> str:
    text = json.dumps(QUESTIONS, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def neutral_id(block: str, role: str) -> str:
    return "E-" + hashlib.sha256(f"{SALT}|{block}|{role}".encode()).hexdigest()[:8]


def _order(key: str) -> str:
    return hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()


# --- clips without speech ------------------------------------------------------------------


def _rms_scaled(samples: np.ndarray, dbfs: float) -> np.ndarray:
    return samples * (10 ** (dbfs / 20) / np.sqrt(np.mean(samples**2)))


def _fade(samples: np.ndarray, seconds: float = 0.01) -> np.ndarray:
    frames = round(seconds * SAMPLE_RATE)
    samples = samples.copy()
    samples[:frames] *= np.linspace(0, 1, frames)
    samples[-frames:] *= np.linspace(1, 0, frames)
    return samples


def _time(seconds: float) -> np.ndarray:
    return np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE


def white_noise(seconds: float, seed: int, dbfs: float) -> np.ndarray:
    return _rms_scaled(
        np.random.default_rng(seed).standard_normal(round(seconds * SAMPLE_RATE)), dbfs
    )


def brown_noise(seconds: float, seed: int, dbfs: float) -> np.ndarray:
    steps = np.random.default_rng(seed).standard_normal(round(seconds * SAMPLE_RATE))
    walk = np.cumsum(steps)
    walk -= np.convolve(walk, np.ones(801) / 801, mode="same")
    return _rms_scaled(walk, dbfs)


def sweep(seconds: float = 10.0, low: float = 100.0, high: float = 4000.0) -> np.ndarray:
    t = _time(seconds)
    rate = math.log(high / low) / seconds
    return _fade(0.1 * np.sin(2 * np.pi * low * (np.exp(rate * t) - 1) / rate))


def chord(seconds: float = 10.0) -> np.ndarray:
    t = _time(seconds)
    notes = (261.63, 329.63, 392.0)
    return _fade(sum(0.05 * np.sin(2 * np.pi * f * t) for f in notes))


def clicks(seconds: float = 10.0, rate_hz: float = 4.0) -> np.ndarray:
    samples = np.zeros(round(seconds * SAMPLE_RATE))
    width = round(0.001 * SAMPLE_RATE)
    for start in range(0, len(samples), round(SAMPLE_RATE / rate_hz)):
        samples[start : start + width] = 0.5
    return samples


def tone_hz(hz: float, seconds: float = 10.0) -> np.ndarray:
    return _fade(0.1 * np.sin(2 * np.pi * hz * _time(seconds)))


def speech_shaped_noise(speech: Iterable[np.ndarray], seconds: float, seed: int) -> np.ndarray:
    """Noise with the long-term average spectrum of `speech` (frames of 1024, Hann), -30 dBFS."""
    frame = 1024
    window = np.hanning(frame)
    power = np.zeros(frame // 2 + 1)
    count = 0
    for samples in speech:
        for start in range(0, len(samples) - frame, frame // 2):
            power += np.abs(np.fft.rfft(samples[start : start + frame] * window)) ** 2
            count += 1
    if count == 0:
        raise ExtError("no speech to take the long-term spectrum from")
    frames = round(seconds * SAMPLE_RATE)
    shape = np.interp(
        np.fft.rfftfreq(frames, 1 / SAMPLE_RATE),
        np.fft.rfftfreq(frame, 1 / SAMPLE_RATE),
        np.sqrt(power / count),
    )
    rng = np.random.default_rng(seed)
    spectrum = (rng.standard_normal(len(shape)) + 1j * rng.standard_normal(len(shape))) * shape
    spectrum[0] = 0
    return _rms_scaled(np.fft.irfft(spectrum, frames), -30.0)


def no_speech_clips(speech: list[np.ndarray], reversed_: list[tuple[np.ndarray, str]]) -> list:
    """(role, samples, generator record, P word) of `E-C`; `reversed_` are (speech, its word)."""
    pink = gate0.pink_noise(10.0, seed=1) * 10 ** (10 / 20)
    found = [
        ("silence-3", gate0.silence(3.0), {"kind": "silence", "seconds": 3.0}),
        ("silence-20", gate0.silence(20.0), {"kind": "silence", "seconds": 20.0}),
        ("pink", pink, {"kind": "pink_noise", "seed": 1, "dbfs": -20.0, "seconds": 10.0}),
        (
            "white",
            white_noise(10.0, 2, -30.0),
            {"kind": "white_noise", "seed": 2, "dbfs": -30.0, "seconds": 10.0},
        ),
        (
            "brown",
            brown_noise(10.0, 3, -30.0),
            {"kind": "brown_noise", "seed": 3, "dbfs": -30.0, "seconds": 10.0},
        ),
        ("tone", tone_hz(220.0), {"kind": "tone", "hz": 220.0, "seconds": 10.0}),
        ("sweep", sweep(), {"kind": "log_sweep", "hz": [100.0, 4000.0], "seconds": 10.0}),
        ("chord", chord(), {"kind": "chord", "hz": [261.63, 329.63, 392.0], "seconds": 10.0}),
        ("clicks", clicks(), {"kind": "clicks", "rate_hz": 4.0, "seconds": 10.0}),
        (
            "speech-noise",
            speech_shaped_noise(speech, 10.0, 4),
            {"kind": "speech_shaped_noise", "seed": 4, "dbfs": -30.0, "seconds": 10.0},
        ),
    ]
    clips = [(role, samples, record, CONTROL_WORD) for role, samples, record in found]
    for number, (samples, word) in enumerate(reversed_, 1):
        record = {"kind": "reversed_speech", "of": number}
        clips.append((f"rv-{number}", samples[::-1].copy(), record, word))
    return clips


# --- the set: clips, manifest, plan --------------------------------------------------------


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)


def _json_bytes(data: object) -> bytes:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _clip(out: Path, block: str, role: str, samples: np.ndarray, **fields) -> dict:
    clip_id = neutral_id(block, role)
    pcm = gate0.pcm16(samples)
    wav = gate0.wav_bytes(pcm)
    name = f"{CLIPS_DIR}/{clip_id}.wav"
    _write_new(out / name, wav)
    return {
        "id": clip_id,
        "block": block,
        "role": role,
        "file": name,
        "wav_sha256": hashlib.sha256(wav).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "duration_s": round(len(pcm) / 2 / SAMPLE_RATE, 6),
        **fields,
    }


def transcript_of(value: str) -> str:
    """The Accent annotation as P-text gets it: every token through `prosody.normalize` (lower
    case, ё → е, letters and digits only), so no mark, accent sign, capital stress letter or
    punctuation of the annotation reaches the text; single spaces."""
    return " ".join(form for form in map(prosody.normalize, value.split()) if form)


def eligible_value(value: str) -> bool:
    """An Accent annotation without bracketed notes (`(г,)`, `()`): those glue letters onto
    the words, and the forms would not be words."""
    return not set(value) & set("()[]{}")


def _vowels(form: str) -> int:
    return sum(ch in prosody.VOWELS for ch in form)


def targets(utterance_id: str, words: list[dict]) -> dict | None:
    """One marked and one unmarked target: a form found once in the utterance with at least
    two vowels, the first by SHA-256 of `<utterance id>|<form>`; None without both."""
    counts: dict[str, int] = {}
    for word in words:
        counts[word["form"]] = counts.get(word["form"], 0) + 1
    chosen = {}
    for marked, key in ((True, "marked"), (False, "unmarked")):
        forms = sorted(
            {
                w["form"]
                for w in words
                if w["marked"] is marked
                and counts[w["form"]] == 1
                and _vowels(w["form"]) >= ER_MIN_VOWELS
            },
            key=lambda form: hashlib.sha256(f"{utterance_id}|{form}".encode()).hexdigest(),
        )
        if not forms:
            return None
        chosen[key] = forms[0]
    return chosen


def select_utterances(candidates: list[dict], count: int = ER_COUNT) -> list[dict]:
    """`candidates` (member, number, start, end, words, value) → at most `count`, at most
    `ER_PER_WAV` per WAV, in the order of SHA-256 of `gpu-ext-149|<member>#<number>`."""
    ordered = sorted(candidates, key=lambda c: _order(f"{c['member']}#{c['number']}"))
    chosen, per_wav = [], {}
    for candidate in ordered:
        duration = candidate["end"] - candidate["start"]
        if not ER_MIN_S <= duration <= ER_MAX_S:
            continue
        utterance_id = f"{candidate['member']}#{candidate['number']}"
        found = targets(utterance_id, list(candidate["words"]))
        if found is None or per_wav.get(candidate["member"], 0) >= ER_PER_WAV:
            continue
        per_wav[candidate["member"]] = per_wav.get(candidate["member"], 0) + 1
        chosen.append({**candidate, "targets": found})
        if len(chosen) == count:
            break
    return chosen


def pair_order(sentences: Iterable) -> list:
    """Stress and distortion pairs alternate, each kind in the order of SHA-256 of its id, so a
    cut block keeps both kinds."""
    stress = sorted(
        (s for s in sentences if s.defect == "lexical_stress"), key=lambda s: _order(s.id)
    )
    other = sorted(
        (s for s in sentences if s.defect != "lexical_stress"), key=lambda s: _order(s.id)
    )
    ordered = []
    for index in range(max(len(stress), len(other))):
        ordered += [kind[index] for kind in (stress, other) if index < len(kind)]
    return ordered


def build_plan(clips: list[dict]) -> list[dict]:
    """The fixed, ordered requests of every block from the manifest clips alone."""
    prompt = review.load_prompt()
    by_block: dict[str, list[dict]] = {block: [] for block in ("E-C", "E-P", "E-R")}
    for clip in clips:
        by_block[clip["block"]].append(clip)
    entries: list[dict] = []

    def add(block, variant, clip, text, *, json_response=False, **fields) -> dict:
        entry = {
            "block": block,
            "variant": variant,
            "clip": None if clip is None else clip["id"],
            "modality": "text" if clip is None else "audio",
            "pair": fields.pop("pair", None),
            "take": fields.pop("take", None),
            "duration_s": None if clip is None else clip["duration_s"],
            "text": text,
            "max_tokens": REVIEW_MAX_TOKENS if json_response else gate0.CLOSED_MAX_TOKENS,
            "json_response": json_response,
            **fields,
        }
        if json_response and clip is not None:
            entry["text"] = gate0.review_prompt(prompt, clip["id"], clip["duration_s"])
        entries.append(entry)
        return entry

    for clip in by_block["E-C"]:
        word = clip["word"]
        add("E-C", "K1", clip, gate0.QUESTIONS["K1"])
        add("E-C", "K2", clip, gate0.QUESTIONS["K2"])
        add("E-C", "P", clip, gate0.QUESTIONS["P"].format(word=word))
        add("E-C", "A", clip, None, json_response=True)
    pairs: dict[int, dict] = {}
    for clip in by_block["E-P"]:
        pairs.setdefault(clip["pair"], {})[clip["take"]] = clip
    for number in sorted(pairs):
        takes = [pairs[number]["ok"], pairs[number]["defect"]]
        for variant in ("Q-S", "K1"):
            for clip in takes:
                text = (
                    QUESTIONS["Q-S"].format(word=clip["word"])
                    if variant == "Q-S"
                    else gate0.QUESTIONS["K1"]
                )
                add("E-P", variant, clip, text, pair=number, take=clip["take"])
        for clip in takes:
            add("E-P", "A", clip, None, json_response=True, pair=number, take=clip["take"])
    utterance_entries: list[list[dict]] = []
    for item, clip in enumerate(by_block["E-R"], 1):
        group = []
        for target in ("marked", "unmarked"):
            word = clip["targets"][target]
            group.append(
                add(
                    "E-R",
                    "P",
                    clip,
                    gate0.QUESTIONS["P"].format(word=word),
                    item=item,
                    target=target,
                )
            )
        for target in ("marked", "unmarked"):
            word = clip["targets"][target]
            text = gate0.QUESTIONS["P-text"].format(transcript=clip["transcript"], word=word)
            entry = add("E-R", "P-text", None, text, item=item, target=target)
            entry["about"] = clip["id"]
            group.append(entry)
        group.append(add("E-R", "K1", clip, gate0.QUESTIONS["K1"], item=item))
        utterance_entries.append(group)
    longest = sorted(by_block["E-R"], key=lambda clip: -clip["duration_s"])[:EA_COUNT]
    long_ones = [clip for clip in by_block["E-R"] if clip in longest]
    for clip in long_ones:
        add("E-A", "A", clip, None, json_response=True)
    for group in utterance_entries[:ED_UTTERANCES]:
        for original in group:
            repeat = {k: v for k, v in original.items() if k != "n"}
            entries.append({**repeat, "block": "E-D", "repeat_of": id(original)})
    silence = next((c for c in by_block["E-C"] if c["role"] == "silence-20"), None)
    first_pair = pairs[min(pairs)] if pairs else {}
    for clip in [silence, first_pair.get("ok"), first_pair.get("defect")]:
        if clip is None:
            continue
        original = next(
            e
            for e in entries
            if e["clip"] == clip["id"] and e["variant"] == "A" and e["block"] != "E-D"
        )
        repeat = {k: v for k, v in original.items() if k != "n"}
        entries.append({**repeat, "block": "E-D", "repeat_of": id(original)})
    numbers = {}
    for number, entry in enumerate(entries, 1):
        numbers[id(entry)] = number
        entry["n"] = number
    for entry in entries:
        if "repeat_of" in entry:
            entry["repeat_of"] = numbers[entry["repeat_of"]]
    return entries


def write_set(out: Path, clips: list[dict], sources: dict, *, fixture: bool) -> dict:
    """`manifest.json` and `plan.json` beside the clips already in `out/clips`."""
    entries = build_plan(clips)
    plan = {"format_version": FORMAT_VERSION, "kind": "gpu_ext_plan", "entries": entries}
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "gpu_ext_set",
        "set": SET_NAME,
        "fixture": fixture,
        "code_sha256": code_sha256(),
        "synthetic_code_sha256": hashlib.sha256(Path(synthetic.__file__).read_bytes()).hexdigest(),
        "prompt_sha256": gate0.PROMPT_SHA256,
        "gate0_questions_sha256": gate0.questions_sha256(),
        "questions_sha256": questions_sha256(),
        "review_max_tokens": REVIEW_MAX_TOKENS,
        "timeouts_s": {"closed": CLOSED_TIMEOUT_S, "review": REVIEW_TIMEOUT_S},
        "sources": sources,
        "clips": clips,
        "counts": block_counts(entries),
    }
    _write_new(out / PLAN_FILE, _json_bytes(plan))
    _write_new(out / MANIFEST_FILE, _json_bytes(manifest))
    return {
        "manifest_sha256": hashlib.sha256((out / MANIFEST_FILE).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256((out / PLAN_FILE).read_bytes()).hexdigest(),
        "counts": manifest["counts"],
    }


def block_counts(entries: list[Mapping]) -> dict:
    counts = {block: {"requests": 0, "A": 0} for block in BLOCKS}
    for entry in entries:
        counts[entry["block"]]["requests"] += 1
        counts[entry["block"]]["A"] += entry["variant"] == "A"
    return counts


def _new_dir(out: Path) -> Path:
    out = Path(out)
    out.mkdir(mode=0o700, parents=True)
    (out / CLIPS_DIR).mkdir(mode=0o700)
    return out


def prepare(corpus_root: Path, out: Path, voice=None) -> dict:
    """The set from RINCO (series 3 material without `pilot-1`) and the Piper voice."""
    corpus_root = Path(corpus_root).resolve()
    corpus_config = config.Config(allowed_roots=(corpus_root,), data_root=corpus_root)
    archive_path = prosody.rinco_path(corpus_config)
    problems = prosody.rinco_problems(corpus_config)
    if problems:
        raise ExtError("RINCO does not match its pin: " + "; ".join(problems))
    voice = voice or synthetic.load_voice(corpus_root / prosody.CORPUS_DIR)
    out = _new_dir(out)
    with zipfile.ZipFile(archive_path) as archive:
        exclusion = prosody.series_exclusion(corpus_config, 3, archive)
        files, counts = prosody.material(archive, exclusion["members"])
        candidates = []
        values = {}
        for file in files:
            for number, annotation in enumerate(file.annotations["accent"]):
                words = prosody.rinco_words(annotation.value)
                if not words or not eligible_value(annotation.value):
                    continue
                values[(file.wav_member, number)] = annotation.value
                candidates.append(
                    {
                        "member": file.wav_member,
                        "number": number,
                        "start": annotation.start,
                        "end": annotation.end,
                        "words": words,
                    }
                )
        chosen = select_utterances(candidates)
        decoded: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory(prefix=".gpu-ext-") as scratch:
            for index, member in enumerate(sorted({c["member"] for c in chosen})):
                copy = corpus.copy_member(archive, member, Path(scratch) / f"{index}.wav")
                decoded[member] = corpus.decode(copy)
                copy.unlink()
    pairs = []
    for sentence in synthetic.GPU_EXT_SENTENCES:
        takes = {}
        for take in synthetic.TAKES:
            made = synthetic.synthesize(voice, sentence, take=take, **PIPER_NOISE)
            with tempfile.TemporaryDirectory(prefix=".gpu-ext-") as scratch:
                samples = synthetic.resample(made.samples, made.sample_rate, Path(scratch))
            takes[take] = (made, samples)
        pairs.append((sentence, takes))
    reversed_ = [
        (takes["ok"][1], sentence.word) for sentence, takes in (pairs[0], pairs[len(pairs) // 2])
    ]
    clips = _no_speech(out, [takes["ok"][1] for _, takes in pairs], reversed_)
    clips += _piper_clips(out, pairs)
    for candidate in chosen:
        samples = decoded[candidate["member"]]
        start, end = round(candidate["start"] * SAMPLE_RATE), round(candidate["end"] * SAMPLE_RATE)
        value = values[(candidate["member"], candidate["number"])]
        clips.append(_rinco_clip(out, candidate, samples[start:end], value))
    sources = {
        "rinco": {
            "archive": corpus.RINCO_ARCHIVE,
            "sha256": prosody.RINCO_SHA256,
            "version": corpus.RINCO_VERSION,
            "license": corpus.RINCO_LICENSE,
            "material": "prosody.material, серия 3 (без pilot-1, q-1122, E1/E2)",
            "excluded": {k: exclusion[k] for k in ("count", "sha256")},
            "material_counts": {k: counts[k] for k in ("files", "utterances", "words")},
            "candidates": len(candidates),
        },
        "piper": {
            "voice": [
                {"file": s.name, "sha256": s.sha256, "revision": corpus.PIPER_REVISION}
                for s in synthetic.voice_sources()
            ],
            "runtime": synthetic.RUNTIME,
            "synthesis": {**PIPER_NOISE, "resampler": "ffmpeg", "sample_rate": SAMPLE_RATE},
            "sentences": [s.id for s in synthetic.GPU_EXT_SENTENCES],
        },
    }
    check_clips(
        clips, forbidden_sha256=pilot_sha256(corpus_root), forbidden_members=exclusion["members"]
    )
    return write_set(out, clips, sources, fixture=False)


def _no_speech(out: Path, speech: list[np.ndarray], reversed_: list) -> list[dict]:
    clips = []
    for role, samples, record, word in no_speech_clips(speech, reversed_):
        clips.append(_clip(out, "E-C", role, samples, generator=record, word=word))
    return clips


def _piper_clips(out: Path, pairs: list) -> list[dict]:
    clips = []
    takes_of = {sentence.id: takes for sentence, takes in pairs}
    for number, sentence in enumerate(pair_order([s for s, _ in pairs]), 1):
        found = takes_of[sentence.id]
        for take in synthetic.TAKES:
            made, samples = found[take]
            clips.append(
                _clip(
                    out,
                    "E-P",
                    f"{sentence.id}-{take}",
                    samples,
                    pair=number,
                    take=take,
                    sentence=sentence.id,
                    word=sentence.word,
                    defect=sentence.defect,
                    subtype=sentence.subtype,
                    ipa=sentence.correct_ipa if take == "ok" else sentence.spoken_ipa,
                    word_start_s=round(made.start_s, 3),
                    word_end_s=round(made.end_s, 3),
                    interval_method=synthetic.INTERVAL_METHOD,
                )
            )
    return clips


def _rinco_clip(out: Path, candidate: Mapping, samples: np.ndarray, value: str) -> dict:
    utterance = f"{candidate['member']}#{candidate['number']}"
    return _clip(
        out,
        "E-R",
        utterance,
        samples,
        member=candidate["member"],
        utterance=candidate["number"],
        start_s=round(candidate["start"], 3),
        end_s=round(candidate["end"], 3),
        targets=candidate["targets"],
        transcript=transcript_of(value),
        marked_share=round(
            sum(w["marked"] for w in candidate["words"]) / len(candidate["words"]), 4
        ),
    )


def pilot_sha256(corpus_root: Path) -> set[str]:
    """SHA-256 of every clip of `pilot-1`: its provenance records and its `clips/` files."""
    pilot = Path(corpus_root) / prosody.PILOT_PROVENANCE
    found = {c["sha256"] for c in json.loads(pilot.read_text(encoding="utf-8"))["clips"]}
    for path in (pilot.parent / "clips").glob("*"):
        if path.is_file():
            found.add(gate0._sha256_file(path))
    return found


def check_clips(
    clips: list[Mapping],
    *,
    forbidden_sha256: Iterable[str] = (),
    forbidden_members: Iterable[str] = (),
) -> None:
    """Refuse a set with a role in a clip id, a clip over 30 s, a clip of `pilot-1` or of the
    Gate 0 days (by SHA-256 of its WAV) or an utterance of an excluded RINCO member."""
    ids = [clip["id"] for clip in clips]
    if len(set(ids)) != len(ids):
        raise ExtError("the set holds a clip id twice")
    bad = [clip_id for clip_id in ids if not ID_PATTERN.fullmatch(clip_id)]
    if bad:
        raise ExtError(f"clip ids must be neutral E-<8 hex>, got {bad[:3]}")
    long_ = [clip["id"] for clip in clips if clip["duration_s"] > MAX_CLIP_S]
    if long_:
        raise ExtError(f"clips over {MAX_CLIP_S} s: {long_[:3]}")
    forbidden = set(forbidden_sha256)
    taken = [clip["id"] for clip in clips if clip["wav_sha256"] in forbidden]
    if taken:
        raise ExtError(f"clips of pilot-1 or of a Gate 0 day: {taken[:3]}")
    members = set(forbidden_members)
    excluded = [clip["id"] for clip in clips if clip.get("member") in members]
    if excluded:
        raise ExtError(f"utterances of excluded RINCO members: {excluded[:3]}")


def check_plan(entries: list[Mapping], clips: list[Mapping]) -> None:
    """Every request names a clip of the set (never `B1`…`T1`) and a known block."""
    known = {clip["id"] for clip in clips}
    for entry in entries:
        if entry["block"] not in BLOCKS:
            raise ExtError(f"request {entry['n']}: unknown block {entry['block']!r}")
        for name in ("clip", "about"):
            if entry.get(name) is not None and entry[name] not in known:
                raise ExtError(f"request {entry['n']}: clip {entry[name]!r} is not in the set")


def load_set(directory: Path, *, manifest_sha256=None, plan_sha256=None, forbidden_sha256=()):
    """The manifest and plan of a set, checked against their pins and every clip on disk."""
    directory = Path(directory)
    data = {}
    for name, pin in ((MANIFEST_FILE, manifest_sha256), (PLAN_FILE, plan_sha256)):
        path = directory / name
        if not path.is_file():
            raise ExtError(f"extended set: {name} is missing")
        raw = path.read_bytes()
        if pin is not None and hashlib.sha256(raw).hexdigest() != pin:
            raise ExtError(f"extended set: {name} differs from the SHA-256 the package pins")
        data[name] = json.loads(raw)
    manifest, plan = data[MANIFEST_FILE], data[PLAN_FILE]
    for clip in manifest["clips"]:
        if gate0._sha256_file(directory / clip["file"]) != clip["wav_sha256"]:
            raise ExtError(f"extended set: clip {clip['id']} differs from its SHA-256")
    check_clips(manifest["clips"], forbidden_sha256=forbidden_sha256)
    check_plan(plan["entries"], manifest["clips"])
    if manifest.get("questions_sha256") != questions_sha256():
        raise ExtError("extended set: the questions differ from this code's")
    if manifest.get("prompt_sha256") != gate0.PROMPT_SHA256:
        raise ExtError("extended set: prompt A differs from the pinned one")
    return manifest, plan["entries"]


# --- a fixture of the same shape for the dry run ------------------------------------------


def fixture(out: Path, *, utterances: int = 12, pairs: int = 4) -> dict:
    """A small set from generators alone (no RINCO, no Piper): the same blocks and plan code."""
    from sound_scout.smoke import speech_like_control

    out = _new_dir(out)
    speech = [
        speech_like_control(3.0 + (k % 4), SAMPLE_RATE, f0=110.0 + 7 * k) for k in range(utterances)
    ]
    takes = [
        {
            "ok": speech_like_control(3.0, SAMPLE_RATE, f0=120.0 + 5 * k),
            "defect": speech_like_control(3.0, SAMPLE_RATE, f0=180.0 + 5 * k),
        }
        for k in range(pairs)
    ]
    reversed_ = [(takes[0]["ok"], "слово"), (takes[-1]["ok"], "слово")]
    clips = _no_speech(out, [t["ok"] for t in takes], reversed_)
    for number, pair in enumerate(takes, 1):
        for take, samples in pair.items():
            clips.append(
                _clip(
                    out,
                    "E-P",
                    f"fixture-{number}-{take}",
                    samples,
                    pair=number,
                    take=take,
                    sentence=f"fixture-{number}",
                    word="слово",
                    defect="lexical_stress" if number % 2 else "word_distortion",
                    word_start_s=1.0,
                    word_end_s=1.5,
                )
            )
    for k, samples in enumerate(speech):
        words = [
            {"form": "первое", "marked": True},
            {"form": "второе", "marked": False},
            {"form": f"слово{k}", "marked": False},
        ]
        candidate = {
            "member": f"fixture/u-{k // 2}.wav",
            "number": k % 2,
            "start": 0.0,
            "end": len(samples) / SAMPLE_RATE,
            "words": words,
            "targets": {"marked": "первое", "unmarked": "второе"},
        }
        clips.append(_rinco_clip(out, candidate, samples, "/первое второе слово"))
    check_clips(clips)
    return write_set(out, clips, {"fixture": "generators only"}, fixture=True)


# --- the run -------------------------------------------------------------------------------


def _not_sent(entry: Mapping, reason: str) -> dict:
    return {**_keys(entry), "sent": False, "not_sent": reason}


def _keys(entry: Mapping) -> dict:
    line = gate0._trace_keys(entry)
    for name in ("block", "item", "target", "repeat_of"):
        if name in entry:
            line[name] = entry[name]
    return line


def probe(client: gate0.Client) -> bool:
    """The server still answers: `/health` in 5 s and a one-token text request in 15 s."""
    if client.request("/health", timeout=PROBE_HEALTH_S)[0] != 200:
        return False
    body = {
        "model": client.model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "да"}]}],
        "max_tokens": 1,
        **gate0.sampling("vllm"),
    }
    return client.request(gate0.CHAT_ROUTE, body, timeout=PROBE_TEXT_S)[0] == 200


def new_attempt(set_dir: Path, name: str, config_: config.Config) -> Path:
    attempts = gate0._mkdir(Path(set_dir) / ATTEMPTS_DIR, config_)
    target = paths.resolve_output(attempts, name, config_)
    os.mkdir(target, 0o700)
    for sub in (gate0.REQUESTS_DIR, gate0.RESPONSES_DIR):
        os.mkdir(target / sub, 0o700)
    return target


def run_blocks(
    set_dir: Path,
    manifest: Mapping,
    entries: list[Mapping],
    attempt: Path,
    client: gate0.Client,
    config_: config.Config,
    *,
    stop_check: Callable[[], str | None] = lambda: None,
    transport_check: Callable[[], str] | None = None,
    on_block_end: Callable[[str, dict], None] = lambda block, summary: None,
    timeouts: Mapping | None = None,
    probe_: Callable[[gate0.Client], bool] = probe,
    timeout_cap: Callable[[], float] = lambda: math.inf,
) -> dict:
    """Send the plan block by block, each request once. A timeout is followed by a probe: a
    server that does not answer it ends the model, a second timeout ends the block. A transport
    error goes to `transport_check` (`gate0.send_checked`). `stop_check` runs before every
    request; its reason marks the rest `not_sent`; `timeout_cap` lowers a request's timeout."""
    timeouts = {"closed": CLOSED_TIMEOUT_S, "review": REVIEW_TIMEOUT_S, **(timeouts or {})}
    started = time.time()
    summary: dict = {"blocks": {}, "stop_reason": None}
    stop = None
    with paths.open_output(attempt, TRACE_FILE, config_) as trace:
        for block in BLOCKS:
            block_entries = [e for e in entries if e["block"] == block]
            block_started = time.time()
            counts = {"sent": 0, "not_sent": 0, "timeouts": 0, "transport_stops": 0}
            block_stop = None
            for entry in block_entries:
                if stop is None:
                    stop = stop_check()
                reason = stop or block_stop
                if reason is not None:
                    gate0._trace(trace, _not_sent(entry, reason))
                    counts["not_sent"] += 1
                    continue
                limit = timeouts["review" if entry["json_response"] else "closed"]

                def send(write_request: bool = True, entry=entry, limit=limit) -> dict:
                    timeout = min(limit, timeout_cap())
                    return gate0._send(
                        entry,
                        Path(set_dir),
                        manifest,
                        client,
                        gate0.NoWatch(),
                        attempt,
                        config_,
                        timeout,
                        gate0.sampling("vllm"),
                        write_request=write_request,
                    )

                line, transport_stop = gate0.send_checked(send, transport_check)
                for name in ("block", "item", "target", "repeat_of"):
                    if name in entry:
                        line[name] = entry[name]
                gate0._trace(trace, line)
                counts["sent"] += 1
                if transport_stop is not None:
                    counts["transport_stops"] += 1
                    stop = transport_stop
                elif line.get("limit") == "request_timeout":
                    counts["timeouts"] += 1
                    if not probe_(client):
                        stop = "server_unresponsive"
                    elif counts["timeouts"] >= BLOCK_TIMEOUTS:
                        block_stop = "block_timeouts"
            record = {
                **counts,
                "planned": len(block_entries),
                "block_stop": block_stop,
                "started_at": round(block_started, 3),
                "elapsed_s": round(time.time() - block_started, 3),
            }
            summary["blocks"][block] = record
            if counts["sent"]:
                on_block_end(block, record)
    summary.update(stop_reason=stop, elapsed_s=round(time.time() - started, 3))
    return summary


# --- the analysis --------------------------------------------------------------------------


def _lines(attempt: Path) -> list[dict]:
    path = Path(attempt) / TRACE_FILE
    if not path.is_file():
        return []
    return [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines() if text]


def _body(attempt: Path, n: int) -> object:
    path = Path(attempt) / gate0.RESPONSES_DIR / f"{n}.json"
    with contextlib.suppress(OSError, ValueError):
        return json.loads(path.read_bytes())
    return None


def balanced_accuracy(items: list[tuple[bool, bool]]) -> float | None:
    """(truth, predicted «да») → (TPR + TNR) / 2; None without both classes."""
    positives = [p for t, p in items if t]
    negatives = [p for t, p in items if not t]
    if not positives or not negatives:
        return None
    return (sum(positives) / len(positives) + sum(not p for p in negatives) / len(negatives)) / 2


def bootstrap_ba(groups: Mapping[str, list[tuple[bool, bool]]]) -> tuple[float, float] | None:
    """Percentile CI of the balanced accuracy, resampling whole files with replacement."""
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPEATS):
        chosen = rng.integers(0, len(keys), len(keys))
        items = [item for index in chosen for item in groups[keys[index]]]
        value = balanced_accuracy(items)
        if value is not None:
            values.append(value)
    if not values:
        return None
    low, high = np.percentile(values, CI_PERCENTILES)
    return float(low), float(high)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: the binomial test of b against b + c at 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def binomial_greater(k: int, n: int, p: float) -> float:
    """P(X ≥ k) for X ~ Bin(n, p): the one-sided exact test against chance."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def auc(scored: list[tuple[float, bool]]) -> float | None:
    positives = [s for s, marked in scored if marked]
    negatives = [s for s, marked in scored if not marked]
    if not positives or not negatives:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in positives for q in negatives)
    return wins / (len(positives) * len(negatives))


def _answered(line: Mapping | None) -> bool:
    return bool(line) and line.get("sent") is True and line.get("http_status") == 200


def _observations(attempt: Path, line: Mapping) -> dict:
    """The A answer of one line: valid, counts and the pronunciation intervals."""
    content = gate0._content(_body(attempt, line["n"]))
    if content is None:
        return {"valid": False, "observations": 0, "intervals": []}
    response, _, _ = review.validate_response(content, line["duration_s"])
    if response is None:
        return {"valid": False, "observations": 0, "intervals": []}
    intervals = [
        (float(item["start_seconds"]), float(item["end_seconds"]))
        for item in response.observations
        if item["category"] == "pronunciation"
    ]
    count = len(response.observations) + len(response.unlocalized_observations)
    return {"valid": True, "observations": count, "intervals": intervals}


def _hit(intervals: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(a < end + HIT_MARGIN_S and b > start - HIT_MARGIN_S for a, b in intervals)


def _share(values: list[bool]) -> dict:
    return {"n": len(values), "share": round(sum(values) / len(values), 4) if values else None}


def analyze(
    set_dir: Path,
    attempt: Path,
    *,
    utterances_path: Path | None = None,
    gate0_attempt: Path | None = None,
) -> dict:
    """The preregistered numbers of one model's attempt from its saved files alone."""
    manifest = json.loads((Path(set_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    lines = _lines(attempt)
    for line in lines:
        if line.get("sent") and line["variant"] != "A":
            line["answer"] = gate0.answer_of(line, _body(attempt, line["n"]))
    by_n = {line["n"]: line for line in lines}
    result: dict = {
        "format_version": FORMAT_VERSION,
        "kind": "gpu_ext_analysis",
        "code_sha256": code_sha256(),
        "blocks": _timing(lines),
    }
    result["R"] = _prosody(lines, clips, utterances_path)
    result["S"] = _pairs(attempt, lines, clips)
    result["C"] = _no_speech_answers(attempt, lines, clips, gate0_attempt)
    result["P_and_K"] = _closed_shares(lines, clips)
    result["A_on_speech"] = _review_on_speech(attempt, lines)
    result["D"] = _repeats(attempt, lines, by_n)
    return result


def _timing(lines: list[dict]) -> dict:
    found = {}
    for block in BLOCKS:
        mine = [line for line in lines if line.get("block") == block]
        sent = [line for line in mine if line.get("sent")]
        walls = {"closed": [], "A": []}
        for line in sent:
            if isinstance(line.get("wall_s"), int | float):
                walls["A" if line["variant"] == "A" else "closed"].append(line["wall_s"])
        found[block] = {
            "planned": len(mine),
            "sent": len(sent),
            "answered": sum(_answered(line) for line in mine),
            "not_sent": sorted({line["not_sent"] for line in mine if not line.get("sent")}),
            "mean_wall_s": {
                k: round(float(np.mean(v)), 3) if v else None for k, v in walls.items()
            },
            "completion_tokens_A": [
                line.get("completion_tokens") for line in sent if line["variant"] == "A"
            ][:5],
        }
    return found


def _prosody(lines: list[dict], clips: Mapping, utterances_path: Path | None) -> dict:
    er = [line for line in lines if line.get("block") == "E-R"]
    audio = {(line["item"], line["target"]): line for line in er if line["variant"] == "P"}
    text = {(line["item"], line["target"]): line for line in er if line["variant"] == "P-text"}
    items = sorted({item for item, _ in audio})
    groups: dict[str, list[tuple[bool, bool]]] = {}
    complete = 0
    paired = {"b": 0, "c": 0, "n": 0}
    scored = []
    rows = _utterance_rows(utterances_path)
    for item in items:
        marked, unmarked = audio.get((item, "marked")), audio.get((item, "unmarked"))
        if not (_answered(marked) and _answered(unmarked)):
            continue
        complete += 1
        clip = clips[marked["clip"]]
        for line, truth in ((marked, True), (unmarked, False)):
            groups.setdefault(clip["member"], []).append((truth, line["answer"] == "да"))
            other = text.get((item, line["target"]))
            if _answered(other):
                audio_right = (line["answer"] == "да") is truth
                text_right = (other["answer"] == "да") is truth
                paired["n"] += 1
                paired["b"] += audio_right and not text_right
                paired["c"] += text_right and not audio_right
            row = rows.get((clip["member"], clip["utterance"]))
            form = clip["targets"]["marked" if truth else "unmarked"]
            score = next(
                (
                    w["score"]["index"]
                    for w in (row or {}).get("words", [])
                    if w["form"] == form and w.get("score")
                ),
                None,
            )
            if isinstance(score, int | float):
                scored.append((float(score), truth))
    flat = [item for group in groups.values() for item in group]
    ba = balanced_accuracy(flat)
    ci = bootstrap_ba(groups)
    p = mcnemar_exact(paired["b"], paired["c"])
    if complete < MIN_UTTERANCES:
        verdict = "insufficient"
    elif ci is not None and ci[0] > 0.5 and p < ALPHA and paired["b"] > paired["c"]:
        verdict = "hears"
    else:
        verdict = "not_shown"
    return {
        "utterances_complete": complete,
        "files": len(groups),
        "R1_balanced_accuracy": None if ba is None else round(ba, 4),
        "R1_ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "R2_mcnemar": {**paired, "p": round(p, 6)},
        "verdict": verdict,
        "prosody_rule_v1_auc": None if auc(scored) is None else round(auc(scored), 4),
        "prosody_rule_v1_words": len(scored),
        "utterances_sha256": None
        if utterances_path is None
        else gate0._sha256_file(Path(utterances_path)),
    }


def _utterance_rows(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {}
    rows = {}
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(text)
        rows[(row["member"], row["utterance"])] = row
    return rows


def _pairs(attempt: Path, lines: list[dict], clips: Mapping) -> dict:
    ep = [line for line in lines if line.get("block") == "E-P"]
    by_key = {(line["pair"], line["take"], line["variant"]): line for line in ep}
    pairs = sorted({line["pair"] for line in ep})
    s1, s2s3 = [], []
    by_kind: dict[str, list[bool]] = {}
    for pair in pairs:
        ok, defect = by_key.get((pair, "ok", "Q-S")), by_key.get((pair, "defect", "Q-S"))
        if _answered(ok) and _answered(defect):
            right = ok["answer"] == "да" and defect["answer"] == "нет"
            s1.append(right)
            by_kind.setdefault(clips[ok["clip"]]["defect"], []).append(right)
        a_ok, a_defect = by_key.get((pair, "ok", "A")), by_key.get((pair, "defect", "A"))
        if _answered(a_ok) and _answered(a_defect):
            hits = []
            for line in (a_defect, a_ok):
                clip = clips[line["clip"]]
                found = _observations(attempt, line)
                hits.append(_hit(found["intervals"], clip["word_start_s"], clip["word_end_s"]))
            s2s3.append(tuple(hits))
    k, n = sum(s1), len(s1)
    b = sum(hit and not false for hit, false in s2s3)
    c = sum(false and not hit for hit, false in s2s3)
    p1 = binomial_greater(k, n, CHANCE_PAIR) if n else None
    verdict = "insufficient" if n < MIN_PAIRS else "above_chance" if p1 < ALPHA else "not_shown"
    return {
        "S1": {
            "pairs_complete": n,
            "right": k,
            "share": round(k / n, 4) if n else None,
            "p_vs_0_25": None if p1 is None else round(p1, 6),
            "by_kind": {kind: _share(v) for kind, v in sorted(by_kind.items())},
            "verdict": verdict,
        },
        "S2_hits": _share([hit for hit, _ in s2s3]),
        "S3_false_alarms": _share([false for _, false in s2s3]),
        "S2_vs_S3_mcnemar": {"b": b, "c": c, "p": round(mcnemar_exact(b, c), 6)},
    }


def _no_speech_answers(
    attempt: Path, lines: list[dict], clips: Mapping, gate0_attempt: Path | None
) -> dict:
    ec = [line for line in lines if line.get("block") == "E-C"]
    reviews = []
    for line in ec:
        if line["variant"] == "A" and _answered(line):
            reviews.append(_observations(attempt, line))
    valid = [r for r in reviews if r["valid"]]
    gate0_controls = []
    if gate0_attempt is not None and (Path(gate0_attempt) / gate0.TRACE_FILE).is_file():
        for line in _lines(gate0_attempt):
            if line.get("clip") in gate0.CONTROLS and line["variant"] == "A" and _answered(line):
                gate0_controls.append(_observations(gate0_attempt, line))
    together = valid + [r for r in gate0_controls if r["valid"]]
    p_lines = [line for line in ec if line["variant"] == "P" and _answered(line)]
    plain = [line for line in p_lines if not clips[line["clip"]]["role"].startswith("rv-")]
    rv = [line for line in p_lines if clips[line["clip"]]["role"].startswith("rv-")]
    return {
        "A_any_observation": _share([r["observations"] > 0 for r in valid]),
        "A_invalid": sum(not r["valid"] for r in reviews),
        "A_any_observation_with_gate0_controls": _share([r["observations"] > 0 for r in together]),
        "P_no_speech": _share([line["answer"] == "нет речи" for line in plain]),
        "P_reversed_no_or_no_speech": _share(
            [line["answer"] in ("нет", "нет речи") for line in rv]
        ),
        "K_speech": _share(
            [
                line["answer"] == "речь"
                for line in ec
                if line["variant"] in ("K1", "K2") and _answered(line)
            ]
        ),
    }


def _closed_shares(lines: list[dict], clips: Mapping) -> dict:
    er = [line for line in lines if line.get("block") == "E-R" and _answered(line)]
    return {
        "E_R_P_no_speech": _share([x["answer"] == "нет речи" for x in er if x["variant"] == "P"]),
        "E_R_K1_speech": _share([x["answer"] == "речь" for x in er if x["variant"] == "K1"]),
        "E_P_K1_speech": _share(
            [
                x["answer"] == "речь"
                for x in lines
                if x.get("block") == "E-P" and x["variant"] == "K1" and _answered(x)
            ]
        ),
    }


def _review_on_speech(attempt: Path, lines: list[dict]) -> dict:
    ea = [line for line in lines if line.get("block") == "E-A" and _answered(line)]
    found = [(_observations(attempt, line), line["duration_s"]) for line in ea]
    valid = [(r, d) for r, d in found if r["valid"]]
    minutes = sum(d for _, d in valid) / 60
    count = sum(r["observations"] for r, _ in valid)
    return {
        "answered": len(found),
        "valid": len(valid),
        "observations": count,
        "speech_minutes": round(minutes, 3),
        "observations_per_minute": round(count / minutes, 3) if minutes else None,
        "note": "верхняя граница ложных тревог, не доказанные ошибки",
    }


def _repeats(attempt: Path, lines: list[dict], by_n: Mapping) -> dict:
    same = []
    for line in lines:
        if line.get("block") != "E-D" or not _answered(line):
            continue
        original = by_n.get(line["repeat_of"])
        if not _answered(original):
            continue
        first = gate0._content(_body(attempt, original["n"]))
        second = gate0._content(_body(attempt, line["n"]))
        same.append(first is not None and first == second)
    return {"identical_content": _share(same)}


# --- the CLI -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.gpu_ext")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_ = commands.add_parser("prepare", help="build the set from RINCO and Piper once")
    prepare_.add_argument("--corpus-root", type=Path, required=True)
    prepare_.add_argument("--out", type=Path, required=True)
    analyze_ = commands.add_parser("analyze", help="the preregistered numbers of one attempt")
    analyze_.add_argument("--set", type=Path, required=True)
    analyze_.add_argument("--attempt", type=Path, required=True)
    analyze_.add_argument("--utterances", type=Path)
    analyze_.add_argument("--gate0-attempt", type=Path)
    # deploy already wrote analysis.json without the series-3 AUC; a second pass gets its own file
    analyze_.add_argument("--name", default=ANALYSIS_FILE)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(json.dumps(prepare(args.corpus_root, args.out), indent=2, ensure_ascii=False))
            return 0
        found = analyze(
            args.set,
            args.attempt,
            utterances_path=args.utterances,
            gate0_attempt=args.gate0_attempt,
        )
        if Path(args.name).name != args.name or not args.name.endswith(".json"):
            raise ExtError("--name must be a plain file name ending in .json")
        _write_new(Path(args.attempt) / args.name, _json_bytes(found))
        print(json.dumps(found, indent=2, ensure_ascii=False))
        return 0
    except (
        ExtError,
        gate0.Gate0Error,
        prosody.ProsodyError,
        corpus.CorpusError,
        synthetic.SyntheticError,
        OSError,
    ) as error:
        print(f"gpu_ext: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
