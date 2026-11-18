"""Gate 0 spike of the audio backend: clips, fixed questions, one run, a mechanical verdict (#23).

The protocol is docs/decisions/0007-gate0-result.md (sections 0–3, written before the run) and
the Decision comment of #23 it copies. Three commands, `python -m sound_scout.eval.gate0 <cmd>`:

- `prepare --date D [--sources DIR] [--natural DIR]` builds the clips under
  `<data_root>/D/`: three synthetic pairs from the development recordings (a neutral Praat PSOLA
  resynthesis `B<i>` and the same Manipulation with F0 and/or duration changed on one
  mechanically chosen word, `V<i>`), or with `--natural` the owner's six natural recordings, plus
  digital silence, pink noise and a pure tone. Word boundaries come from the configured local ASR
  (`asr.transcribe`, offline); the words and the transcript stay in `words.json`, the manifest
  carries only hashes, durations, word numbers and intervals. It records the SHA-256 of the
  verdict rule as the decision document holds it (between `RULE_BEGIN` and `RULE_END`), of the
  prompt and of the fixed questions. The audio backend is not touched.
- `run --date D` checks the weights, the prompt, the rule and the clips against what was
  recorded, starts the local `llama-server` with local `-m`/`--mmproj` only, on 127.0.0.1 and a
  free port, sends the fixed list of requests one at a time and once each, and stops the server
  whatever happens. The empty input `E0` goes through the same `review.check_payload` as every
  audio request and is refused before any HTTP call. Every request leaves a trace line with the
  SHA-256 of the WAV and of the PCM actually sent, the server's token counts and timings, the
  client wall time and the server's RSS; the raw answer is kept verbatim. A request over the
  wall time limit or a server RSS over the limit ends the run: the rest are recorded as not sent.
  The run talks to nothing but the server it started (no proxy); the process-wide network guard
  is not installed here because it refuses loopback too.
- `verdict --date D` recomputes the verdict from the saved answers alone (`decide`).

A test with a fake server proves this contract only, never what the model hears (CLAUDE.md).
"""

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from sound_scout import asr, channels, config, guard, ingest, paths, prepare, proc, review, runs

MODEL_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
MODEL_REVISION = "8903cb408c7df39cde2ed15fac397e6d7be92951"
MODEL_FILE = "gemma-4-E4B-it-Q8_0.gguf"
MMPROJ_FILE = "mmproj-gemma-4-E4B-it-BF16.gguf"
WEIGHTS = {
    MODEL_FILE: "49008e7cc7edadf154c8748904a89ee86eb6905d220aac568f6b7b92ca6aa0c6",
    MMPROJ_FILE: "fc1c6a1b566041bf41268b388bf59f0431ab86b3f4a7e34e996c0b4772509e37",
}
DEFAULT_MODEL_DIR = (
    Path.home() / ".cache/sound-scout/models/ggml-org--gemma-4-E4B-it-GGUF" / MODEL_REVISION
)
DEFAULT_LLAMA_DIR = Path.home() / ".local/opt/llama.cpp/b12359/llama-b12359"
SERVER_BINARY = "llama-server"
PROMPT_SHA256 = "53b9fd53bfeb1eb78b2b3a837824807563a84561b0bccc2e9cd1b1157a0917ca"
DECISION_DOC = Path("docs/decisions/0007-gate0-result.md")
RULE_BEGIN = "<!-- gate0-rule:begin -->"
RULE_END = "<!-- gate0-rule:end -->"
RULE_HEADER = "```text\nGATE0-RULE v1 "

SAMPLE_RATE = 16_000
MAX_CLIPS = 10
MAX_CLIP_SECONDS = 30.0
CONTROL_SECONDS = 10.0
NOISE_DBFS = -30.0
NOISE_SEED = 0
TONE_HZ = 440.0
TONE_AMPLITUDE = 0.1
TONE_FADE_S = 0.01
PITCH_TIME_STEP = 0.01
PITCH_FLOOR_HZ = 75
PITCH_CEILING_HZ = 600
PEAK_LIMIT = 0.99
DURATION_EDGE_S = 0.001
WORD_MIN_PROBABILITY = 0.80
WORD_MIN_S = 0.25
WORD_MAX_S = 1.0
SOURCE_FILES = ("H01-listen-plus12dB.wav", "H02-listen-plus12dB.wav", "H03-listen-plus12dB.wav")
MANIPULATIONS = (
    {"semitones": 4.0, "duration_factor": 1.0},
    {"semitones": 0.0, "duration_factor": 1.6},
    {"semitones": 3.0, "duration_factor": 1.3},
)
NATURAL_PAIRS = 3
NATURAL_TAKES = ("a", "b")
PAIRS_FILE = "pairs.json"
CONTROLS = ("S0", "N1", "T1")

HOST = "127.0.0.1"
CHAT_ROUTE = "/v1/chat/completions"
CONTEXT_TOKENS = 4096
THREADS_IDLE = 12
THREADS_BUSY = 8
LOAD_TIMEOUT_S = 300.0
REQUEST_TIMEOUT_S = 600.0
SERVICE_TIMEOUT_S = 60.0
RSS_LIMIT_BYTES = 18 * 10**9
RSS_POLL_S = 0.5
STOP_GRACE_S = 10.0
DOWNLOAD_FLAGS = frozenset(
    {"-hf", "-hfr", "--hf-repo", "-hff", "--hf-file", "-mu", "--model-url", "-mmu", "--mmproj-url"}
)
CLOSED_MAX_TOKENS = 8
REVIEW_MAX_TOKENS = 1024
SAMPLING = {
    "temperature": 0,
    "seed": 0,
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
MIN_AUDIO_TOKENS_PER_S = 20.0

K_OPTIONS = ("речь", "музыка", "тон", "шум", "тишина")
P_OPTIONS = ("да", "нет", "нет речи")
INVALID = "invalid"
QUESTIONS = {
    "K1": "Что слышно на этой записи? Ответь одним словом из списка: речь, музыка, тон, шум, "
    "тишина.",
    "K2": "Что слышно на этой записи? Ответь одним словом из списка: тишина, шум, тон, музыка, "
    "речь.",
    "P": "Выделено ли на этой записи слово «{word}» голосом сильнее соседних слов? Ответь одним "
    "из вариантов: да; нет; нет речи — если на записи нет речи.",
    "P-text": "Вот расшифровка записи: «{transcript}». Выделено ли в ней слово «{word}» голосом "
    "сильнее соседних слов? Ответь одним из вариантов: да; нет; нет речи — если на записи нет "
    "речи.",
    "A-transcript": "\n\nРасшифровка (вместо аудио): «{transcript}»",
    "A-trusted": "\n\nДоверенные поля: {trusted}",
    "A-answer": "\n\nОтвет — один JSON-объект с полями observations, unlocalized_observations, "
    "uncertainties, limitations; category — одно из: pronunciation, repetition_self_correction, "
    "semantic_stress, pause_phrasing, completeness.",
}

MANIFEST_FILE = "manifest.json"
WORDS_FILE = "words.json"
CLIPS_DIR = "clips"
PREP_DIR = "prep"
ATTEMPTS_DIR = "attempts"
RUN_FILE = "run.json"
TRACE_FILE = "trace.jsonl"
VERDICT_FILE = "verdict.json"
SERVER_LOG = "server.log"
SERVER_LOG_MAX_BYTES = 16 << 20
REQUESTS_DIR = "requests"
RESPONSES_DIR = "responses"
DIR_MODE = 0o700
FORMAT_VERSION = 1
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_EDGES = " \t\r\n\"'«»„“”`*_#>~.,!?;:()[]{}-—–…"
_WORD_EDGES = " \t\r\n\"'«»„“”`*_.,!?;:()[]{}-—–…"


class Gate0Error(Exception):
    """The spike cannot go on as the protocol says; nothing was sent to the model after it."""


# --- the rule and the fixed texts ---------------------------------------------------------


def rule_block(document: str) -> str:
    """The verdict rule as the decision document holds it: the text between the markers."""
    if document.count(RULE_BEGIN) != 1 or document.count(RULE_END) != 1:
        raise Gate0Error(f"the decision document must hold one {RULE_BEGIN} … {RULE_END} block")
    begin = document.index(RULE_BEGIN) + len(RULE_BEGIN)
    end = document.index(RULE_END)
    block = document[begin:end].strip()
    if end < begin or not block.startswith(RULE_HEADER):
        raise Gate0Error("the rule block does not start with GATE0-RULE v1")
    return block


def rule_sha256(document: str) -> str:
    return hashlib.sha256(rule_block(document).encode("utf-8")).hexdigest()


def questions_sha256() -> str:
    text = json.dumps(QUESTIONS, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def review_prompt(prompt: review.Prompt, clip_id: str, duration_s: float, transcript=None) -> str:
    """Variant A: the audio review instruction, the trusted fields and the answer format."""
    trusted = json.dumps(
        {"clip_id": clip_id, "duration_s": duration_s, "task": None, "intent": None},
        ensure_ascii=False,
    )
    inserted = "" if transcript is None else QUESTIONS["A-transcript"].format(transcript=transcript)
    return (
        prompt.text
        + inserted
        + QUESTIONS["A-trusted"].format(trusted=trusted)
        + QUESTIONS["A-answer"]
    )


def parse_closed(text: str, options: tuple[str, ...]) -> str:
    """The one option a closed answer names, or `INVALID` (none or several)."""
    normalized = text.lower().replace("ё", "е").strip(_EDGES)
    found = set()
    for option in sorted(options, key=len, reverse=True):
        words = r"\s+".join(re.escape(word) for word in option.split())
        pattern = re.compile(rf"(?<!\w){words}(?!\w)")
        if pattern.search(normalized):
            found.add(option)
            normalized = pattern.sub(" ", normalized)
    return found.pop() if len(found) == 1 else INVALID


def clean_word(word: str) -> str:
    return word.strip(_WORD_EDGES)


def _same_word(word: str) -> str:
    return word.casefold().replace("ё", "е")


# --- signals -------------------------------------------------------------------------------


def silence(seconds: float = CONTROL_SECONDS) -> np.ndarray:
    return np.zeros(round(seconds * SAMPLE_RATE))


def pink_noise(seconds: float = CONTROL_SECONDS, seed: int = NOISE_SEED) -> np.ndarray:
    """1/f noise shaped in the frequency domain, scaled to `NOISE_DBFS` RMS."""
    frames = round(seconds * SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    bins = frames // 2 + 1
    spectrum = rng.standard_normal(bins) + 1j * rng.standard_normal(bins)
    frequencies = np.fft.rfftfreq(frames, 1 / SAMPLE_RATE)
    frequencies[0] = frequencies[1]
    spectrum /= np.sqrt(frequencies)
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, frames)
    return noise * (10 ** (NOISE_DBFS / 20) / np.sqrt(np.mean(noise**2)))


def tone(seconds: float = CONTROL_SECONDS) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    signal_ = TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * t)
    fade = round(TONE_FADE_S * SAMPLE_RATE)
    signal_[:fade] *= np.linspace(0, 1, fade)
    signal_[-fade:] *= np.linspace(1, 0, fade)
    return signal_


def pcm16(samples: np.ndarray) -> bytes:
    return np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_bytes(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


def wav_pcm(data: bytes) -> bytes:
    """The `data` block of a 16 kHz mono s16 WAV; `Gate0Error` for any other format."""
    with wave.open(io.BytesIO(data), "rb") as source:
        layout = (source.getnchannels(), source.getsampwidth(), source.getframerate())
        if layout != (1, 2, SAMPLE_RATE):
            raise Gate0Error(f"clip WAV is {layout}, expected (1, 2, {SAMPLE_RATE})")
        return source.readframes(source.getnframes())


def f0_medians(samples: np.ndarray, words: list[dict]) -> list[float | None]:
    """Median F0 (Praat `To Pitch (ac)`) of the voiced frames inside each word; None if none."""
    import parselmouth

    sound = parselmouth.Sound(np.asarray(samples, dtype=np.float64), sampling_frequency=SAMPLE_RATE)
    pitch = sound.to_pitch_ac(
        time_step=PITCH_TIME_STEP, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    frequency = pitch.selected_array["frequency"]
    times = pitch.xs()
    medians = []
    for word in words:
        inside = frequency[(times >= word["start"]) & (times <= word["end"]) & (frequency > 0)]
        medians.append(float(np.median(inside)) if len(inside) else None)
    return medians


def choose_target(words: list[dict]) -> int | None:
    """Index of the target word: not first or last, probability and length in range, lowest
    median F0 (earlier on a tie); None when no word qualifies."""
    candidates = [
        (word["f0_median_hz"], index)
        for index, word in enumerate(words)
        if 0 < index < len(words) - 1
        and word["probability"] >= WORD_MIN_PROBABILITY
        and WORD_MIN_S <= word["end"] - word["start"] <= WORD_MAX_S
        and word.get("f0_median_hz") is not None
    ]
    return min(candidates)[1] if candidates else None


def build_pair(
    samples: np.ndarray, start: float, end: float, semitones: float, duration_factor: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """(base, variant, scale): a neutral PSOLA resynthesis and the same Manipulation with the
    word's F0 raised and/or its duration stretched; both scaled by one factor to a 0.99 peak."""
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(np.asarray(samples, dtype=np.float64), sampling_frequency=SAMPLE_RATE)
    manipulation = call(sound, "To Manipulation", PITCH_TIME_STEP, PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
    base = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    if semitones:
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(pitch_tier, "Multiply frequencies", start, end, 2 ** (semitones / 12))
        call([manipulation, pitch_tier], "Replace pitch tier")
    if duration_factor != 1.0:
        tier = call("Create DurationTier", "stretch", 0, sound.duration)
        for time_s, factor in (
            (start - DURATION_EDGE_S, 1.0),
            (start, duration_factor),
            (end, duration_factor),
            (end + DURATION_EDGE_S, 1.0),
        ):
            call(tier, "Add point", time_s, factor)
        call([manipulation, tier], "Replace duration tier")
    variant = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    peak = max(float(np.max(np.abs(base))), float(np.max(np.abs(variant))))
    scale = PEAK_LIMIT / peak if peak > PEAK_LIMIT else 1.0
    return base * scale, variant * scale, scale


# --- prepare -------------------------------------------------------------------------------


def output_dir(day: str, config_: config.Config) -> Path:
    if not _DATE.fullmatch(day):
        raise Gate0Error(f"--date must be YYYY-MM-DD, got {day!r}")
    try:
        date.fromisoformat(day)
    except ValueError as error:
        raise Gate0Error(f"--date {day!r} is not a date: {error}") from None
    return paths.resolve_output(config_.data_root, day, config_)


def prepare_clips(
    day: str,
    config_: config.Config,
    *,
    sources: Path | None = None,
    natural: Path | None = None,
    decision: Path = DECISION_DOC,
    transcribe: Callable[[Path, runs.Run, config.Config], dict] | None = None,
) -> dict:
    """Build the clips, the words and the manifest under `<data_root>/<day>/`; never overwrites."""
    rule = rule_sha256(decision.read_text(encoding="utf-8"))
    prompt = review.load_prompt()
    if prompt.sha256 != PROMPT_SHA256:
        raise Gate0Error(f"prompt A is {prompt.sha256}, the protocol fixes {PROMPT_SHA256}")
    out = output_dir(day, config_)
    _mkdir(out, config_)
    if os.path.lexists(out / MANIFEST_FILE):
        raise Gate0Error(f"{day}/{MANIFEST_FILE} already exists; prepare never overwrites")
    leftovers = [name for name in (CLIPS_DIR, PREP_DIR, WORDS_FILE) if os.path.lexists(out / name)]
    if leftovers:
        raise Gate0Error(
            f"{day}/ holds {leftovers} from a prepare that stopped midway; "
            f"delete the directory {day}/ and run prepare again"
        )
    prep_dir = _mkdir(out / PREP_DIR, config_)
    transcribe = transcribe or transcribe_source
    if natural is None:
        mode = "synthetic"
        pending, pairs, sources_info, words = _synthetic(
            sources or config_.allowed_roots[0], prep_dir, config_, transcribe
        )
    else:
        mode = "natural"
        pending, pairs, sources_info, words = _natural(natural, prep_dir, config_)
    for clip_id, samples in (("S0", silence()), ("N1", pink_noise()), ("T1", tone())):
        pending.append((clip_id, samples, {"role": "control"}))
    if len(pending) > MAX_CLIPS:
        raise Gate0Error(f"{len(pending)} clips, the owner's limit is {MAX_CLIPS}")
    too_long = [
        clip_id for clip_id, samples, _ in pending if len(samples) / SAMPLE_RATE > MAX_CLIP_SECONDS
    ]
    if too_long:
        raise Gate0Error(f"clips over {MAX_CLIP_SECONDS} s: {too_long}")
    clips_dir = _mkdir(out / CLIPS_DIR, config_)
    clips = [
        _write_clip(clips_dir, clip_id, samples, config_, **fields)
        for clip_id, samples, fields in pending
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "gate0_manifest",
        "mode": mode,
        "date": day,
        "created_at": _now(),
        "code": _code_identity(),
        "rule_sha256": rule,
        "prompt": {"name": prompt.name, "sha256": prompt.sha256},
        "questions_sha256": questions_sha256(),
        "sources": sources_info,
        "clips": clips,
        "pairs": pairs,
        "limits": {"max_clips": MAX_CLIPS, "max_clip_seconds": MAX_CLIP_SECONDS},
        "note": "V-клипы синтетически изменены (Praat PSOLA) и не заменяют естественные пары",
    }
    _write_json(out, WORDS_FILE, words, config_)
    _write_json(out, MANIFEST_FILE, manifest, config_)
    return manifest


def transcribe_source(path: Path, run: runs.Run, config_: config.Config) -> dict:
    """The source at 16 kHz mono and its trusted ASR words, through the `analyze` path."""
    loaded = load_mono16k(path, run, config_)
    transcription = asr.transcribe(loaded["samples"], run, config_)
    if transcription.status != "completed":
        raise Gate0Error(f"asr is {transcription.status}: {transcription.reason}")
    trusted, _ = asr.trusted_segments(transcription.raw)
    loaded["words"] = [
        {key: word[key] for key in ("word", "start", "end", "probability")}
        for segment in trusted
        for word in segment.get("words") or []
    ]
    loaded["transcript"] = " ".join(segment["text"].strip() for segment in trusted).strip()
    return loaded


def load_mono16k(path: Path, run: runs.Run, config_: config.Config) -> dict:
    source = ingest.decode(path, run, config_)
    audio = channels.load(source)
    mono, _ = channels.choose_mono(audio, channels.report(audio))
    samples = prepare.mono16k(mono, source.probe.sample_rate, run, config_).samples
    return {"samples": samples, "sha256": source.sha256, "duration_s": source.duration_s}


def _synthetic(sources: Path, prep_dir, config_, transcribe) -> tuple:
    clips, pairs, sources_info, words_out = [], [], [], {}
    for number, (name, manipulation) in enumerate(zip(SOURCE_FILES, MANIPULATIONS, strict=True), 1):
        source_id = f"H{number:02d}"
        loaded = transcribe(sources / name, _scratch_run(prep_dir, source_id, config_), config_)
        samples, words = loaded["samples"], loaded["words"]
        for word, median in zip(words, f0_medians(samples, words), strict=True):
            word["f0_median_hz"] = median
        target = choose_target(words)
        sources_info.append(
            {"id": source_id, "sha256": loaded["sha256"], "duration_s": loaded["duration_s"]}
        )
        words_out[source_id] = {
            "transcript": loaded["transcript"],
            "words": words,
            "target_index": None if target is None else target + 1,
            "target_word": None if target is None else clean_word(words[target]["word"]),
        }
        pair = {
            "pair": number,
            "source": source_id,
            "base": f"B{number}",
            "variant": f"V{number}",
            "manipulation": manipulation,
            "target": None,
        }
        if target is not None:
            word = words[target]
            base, variant, scale = build_pair(samples, word["start"], word["end"], **manipulation)
            pair["target"] = {
                "index": target + 1,
                "start": word["start"],
                "end": word["end"],
                "f0_median_hz": word["f0_median_hz"],
            }
            pair["peak_scale"] = scale
            for clip_id, signal_, role in (
                (pair["base"], base, "base"),
                (pair["variant"], variant, "variant"),
            ):
                fields = {"role": role, "pair": number, "source": source_id}
                clips.append(
                    (clip_id, signal_, {**fields, "synthetic_manipulation": role == "variant"})
                )
        pairs.append(pair)
    return clips, pairs, sources_info, words_out


def _natural(natural: Path, prep_dir, config_) -> tuple:
    pairs_path = paths.resolve_input(natural / PAIRS_FILE, config_)
    with paths.open_input(pairs_path, config_) as file:
        spec = json.loads(file.read().decode("utf-8"))
    clips, pairs, sources_info, words_out = [], [], [], {}
    for number in range(1, NATURAL_PAIRS + 1):
        pair_id = f"P{number}"
        entry = spec.get(pair_id) if isinstance(spec, dict) else None
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise Gate0Error(f"{PAIRS_FILE}: {pair_id} needs text, a and b")
        phrase = entry["text"].split()
        pair = {"pair": number}
        words_out[pair_id] = {"text": entry["text"]}
        for take in NATURAL_TAKES:
            index = entry.get(take)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 1 <= index <= len(phrase)
            ):
                raise Gate0Error(
                    f"{PAIRS_FILE}: {pair_id}.{take} must be a word number of the text"
                )
            words_out[pair_id][take] = {"index": index, "word": clean_word(phrase[index - 1])}
            pair[f"word_{take}"] = {"index": index}
        if len({_same_word(words_out[pair_id][take]["word"]) for take in NATURAL_TAKES}) == 1:
            raise Gate0Error(f"{PAIRS_FILE}: {pair_id} stresses the same word in both takes")
        for take in NATURAL_TAKES:
            clip_id = f"{pair_id}-{take}"
            loaded = load_mono16k(
                natural / f"{clip_id}.wav", _scratch_run(prep_dir, clip_id, config_), config_
            )
            sources_info.append(
                {"id": clip_id, "sha256": loaded["sha256"], "duration_s": loaded["duration_s"]}
            )
            fields = {"role": "natural", "pair": number, "take": take, "source": clip_id}
            clips.append((clip_id, loaded["samples"], fields))
            pair[take] = clip_id
        pairs.append(pair)
    return clips, pairs, sources_info, words_out


def _write_clip(clips_dir: Path, clip_id: str, samples, config_, **fields) -> dict:
    pcm = pcm16(samples)
    data = wav_bytes(pcm)
    name = f"{clip_id}.wav"
    with paths.open_output(clips_dir, name, config_) as out:
        out.write(data)
    frames = len(pcm) // 2
    return {
        "id": clip_id,
        **{key: value for key, value in fields.items() if value is not None},
        "file": f"{CLIPS_DIR}/{name}",
        "sample_rate": SAMPLE_RATE,
        "frames": frames,
        "duration_s": frames / SAMPLE_RATE,
        "wav_sha256": hashlib.sha256(data).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
    }


def _scratch_run(prep_dir: Path, name: str, config_) -> runs.Run:
    run_dir = _mkdir(prep_dir / name, config_)
    return runs.Run(f"gate0-{uuid.uuid4().hex}", run_dir, "gate0", "", False, False, True)


# --- the request list ----------------------------------------------------------------------


def plan(manifest: Mapping, words: Mapping, prompt: review.Prompt) -> list[dict]:
    """The fixed, ordered request list of the spike (E0 is separate and comes first)."""
    clips = manifest["clips"]
    by_id = {clip["id"]: clip for clip in clips}
    closed = []
    reviews = []
    if manifest["mode"] == "synthetic":
        pairs = [pair for pair in manifest["pairs"] if pair["target"] is not None]
        for pair in pairs:
            word = words[pair["source"]]["target_word"]
            transcript = words[pair["source"]]["transcript"]
            for clip_id in (pair["base"], pair["variant"]):
                closed.append(("P", clip_id, pair["pair"], None, word, None))
            closed.append(("P-text", None, pair["pair"], None, word, transcript))
            reviews.append((pair["base"], transcript))
        control_word = (
            (pairs[0]["pair"], None, words[pairs[0]["source"]]["target_word"]) if pairs else None
        )
    else:
        pairs = manifest["pairs"]
        for pair in pairs:
            phrase = words[f"P{pair['pair']}"]
            word_a, word_b = phrase["a"]["word"], phrase["b"]["word"]
            for clip_id, take, word in (
                (pair["a"], "a", word_a),
                (pair["b"], "a", word_a),
                (pair["b"], "b", word_b),
                (pair["a"], "b", word_b),
            ):
                closed.append(("P", clip_id, pair["pair"], take, word, None))
            for take, word in (("a", word_a), ("b", word_b)):
                closed.append(("P-text", None, pair["pair"], take, word, phrase["text"]))
            reviews.append((pair["a"], phrase["text"]))
        first = words["P1"]["a"]["word"] if pairs else None
        control_word = (1, "a", first) if first is not None else None
    if control_word is not None:
        for clip_id in CONTROLS:
            closed.append(("P", clip_id, control_word[0], control_word[1], control_word[2], None))
    entries = []
    for variant in ("K1", "K2"):
        entries += [_entry(variant, clip, QUESTIONS[variant], CLOSED_MAX_TOKENS) for clip in clips]
    for variant in ("P", "P-text"):
        for kind, clip_id, pair_number, take, word, transcript in closed:
            if kind != variant:
                continue
            if kind == "P":
                entries.append(
                    _entry(
                        "P",
                        by_id[clip_id],
                        QUESTIONS["P"].format(word=word),
                        CLOSED_MAX_TOKENS,
                        pair=pair_number,
                        take=take,
                    )
                )
            else:
                entries.append(
                    _entry(
                        "P-text",
                        None,
                        QUESTIONS["P-text"].format(transcript=transcript, word=word),
                        CLOSED_MAX_TOKENS,
                        pair=pair_number,
                        take=take,
                    )
                )
    for clip in clips:
        entries.append(
            _entry(
                "A",
                clip,
                review_prompt(prompt, clip["id"], clip["duration_s"]),
                REVIEW_MAX_TOKENS,
                json_response=True,
            )
        )
    for clip_id, transcript in reviews:
        clip = by_id[clip_id]
        entry = _entry(
            "A-text",
            None,
            review_prompt(prompt, clip_id, clip["duration_s"], transcript),
            REVIEW_MAX_TOKENS,
            json_response=True,
            pair=clip.get("pair"),
        )
        entry["duration_s"] = clip["duration_s"]
        entry["about"] = clip_id
        entries.append(entry)
    for number, entry in enumerate(entries, 1):
        entry["n"] = number
    return entries


def _entry(variant, clip, text, max_tokens, *, json_response=False, pair=None, take=None) -> dict:
    return {
        "variant": variant,
        "clip": None if clip is None else clip["id"],
        "modality": "text" if clip is None else "audio",
        "pair": pair if pair is not None or clip is None else clip.get("pair"),
        "take": take,
        "duration_s": None if clip is None else clip["duration_s"],
        "text": text,
        "max_tokens": max_tokens,
        "json_response": json_response,
    }


def answer_of(entry: Mapping, body: object) -> object:
    """The parsed answer of one saved response body: an option, `INVALID`, or the A summary."""
    content = _content(body)
    if entry["variant"] in ("K1", "K2"):
        return INVALID if content is None else parse_closed(content, K_OPTIONS)
    if entry["variant"] in ("P", "P-text"):
        return INVALID if content is None else parse_closed(content, P_OPTIONS)
    if content is None:
        return {"valid": False, "errors": ["no message content"]}
    response, errors, _ = review.validate_response(content, entry["duration_s"])
    if response is None:
        return {"valid": False, "errors": errors[:5]}
    return {
        "valid": True,
        "observations": len(response.observations),
        "unlocalized_observations": len(response.unlocalized_observations),
        "categories": sorted(
            {
                str(item["category"])
                for item in response.observations + response.unlocalized_observations
            }
        ),
    }


def _content(body: object) -> str | None:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


# --- the run -------------------------------------------------------------------------------


def server_argv(binary: Path, model_dir: Path, port: int, threads: int) -> list[str]:
    argv = [
        str(binary),
        "-m",
        str(model_dir / MODEL_FILE),
        "--mmproj",
        str(model_dir / MMPROJ_FILE),
        "--host",
        HOST,
        "--port",
        str(port),
        "--jinja",
        "-c",
        str(CONTEXT_TOKENS),
        "--parallel",
        "1",
        "--no-webui",
        "-t",
        str(threads),
    ]
    if DOWNLOAD_FLAGS.intersection(argv):
        raise Gate0Error("the server argv must not carry a download flag")
    return argv


class Client:
    """HTTP to the local server only, without proxies; `calls` counts every request made."""

    def __init__(self, port: int) -> None:
        self.base = f"http://{HOST}:{port}"
        self.calls = 0
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path: str, body: object = None, timeout: float = SERVICE_TIMEOUT_S):
        """(status, raw bytes, error); status None when no HTTP answer came."""
        self.calls += 1
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method="GET" if body is None else "POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return response.status, response.read(), None
        except urllib.error.HTTPError as error:
            return error.code, error.read(), f"HTTP {error.code}"
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            return None, None, f"{type(reason).__name__}: {reason}"

    def text_tokens(self, text: str) -> int | None:
        """Prompt tokens of the same message without audio, by template and tokenizer only."""
        message = {"role": "user", "content": [{"type": "text", "text": text}]}
        status, raw, _ = self.request(
            "/apply-template",
            {"messages": [message], "chat_template_kwargs": SAMPLING["chat_template_kwargs"]},
        )
        try:
            prompt = json.loads(raw)["prompt"] if status == 200 else None
            if not isinstance(prompt, str):
                return None
            status, raw, _ = self.request(
                "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}
            )
            tokens = json.loads(raw)["tokens"] if status == 200 else None
        except (ValueError, KeyError, TypeError):
            return None
        return len(tokens) if isinstance(tokens, list) else None


class RssWatch:
    """Polls the server's VmRSS; over the limit it stops the server and remembers why."""

    def __init__(self, process: subprocess.Popen, limit: int) -> None:
        self.process = process
        self.limit = limit
        self.peak = 0
        self.window_peak = 0
        self.exceeded: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "RssWatch":
        self._thread.start()
        return self

    def current(self) -> int | None:
        return proc_status(self.process.pid).get("VmRSS")

    def reset_window(self) -> None:
        self.window_peak = self.current() or 0

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while True:
            rss = self.current()
            if rss is not None:
                self.peak = max(self.peak, rss)
                self.window_peak = max(self.window_peak, rss)
                if rss > self.limit:
                    self.exceeded = rss
                    stop_server(self.process)
                    return
            if self._stop.wait(RSS_POLL_S):
                return


def proc_status(pid: int) -> dict[str, int]:
    """VmRSS and VmHWM of a process in bytes; empty when it is gone."""
    values = {}
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        name, _, rest = line.partition(":")
        if name in ("VmRSS", "VmHWM"):
            values[name] = int(rest.split()[0]) * 1024
    return values


def stop_server(process: subprocess.Popen) -> None:
    """SIGTERM to the server's process group, SIGKILL after `STOP_GRACE_S`; idempotent."""
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_spike(
    day: str,
    config_: config.Config,
    *,
    decision: Path = DECISION_DOC,
    model_dir: Path = DEFAULT_MODEL_DIR,
    llama_dir: Path = DEFAULT_LLAMA_DIR,
    threads: int | None = None,
    weights: Mapping[str, str] = WEIGHTS,
    rss_limit: int = RSS_LIMIT_BYTES,
    request_timeout: float = REQUEST_TIMEOUT_S,
    load_timeout: float = LOAD_TIMEOUT_S,
) -> dict:
    """One attempt of the spike under `<data_root>/<day>/attempts/<k>/`; returns the verdict."""
    out = output_dir(day, config_)
    manifest = _read_json(out / MANIFEST_FILE)
    words = _read_json(out / WORDS_FILE)
    prompt = review.load_prompt()
    try:
        document = decision.read_bytes()
        rule = rule_sha256(decision.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        reason = getattr(error, "strerror", None) or error
        raise Gate0Error(f"cannot read the decision document {decision.name}: {reason}") from None
    attempt = _new_attempt(out, config_)
    record: dict = {
        "format_version": FORMAT_VERSION,
        "kind": "gate0_run",
        "attempt": attempt.name,
        "started_at": _now(),
        "code": _code_identity(),
        "decision_sha256": hashlib.sha256(document).hexdigest(),
    }
    found = {name: _sha256_file(model_dir / name) for name in weights}
    clips_found = {
        clip["id"]: _sha256_file(out / clip["file"]) == clip["wav_sha256"]
        for clip in manifest["clips"]
    }
    checks = {
        "weights": found == dict(weights),
        "prompt": prompt.sha256 == PROMPT_SHA256 == manifest["prompt"]["sha256"],
        "rule": rule == manifest["rule_sha256"],
        "questions": questions_sha256() == manifest["questions_sha256"],
        "clips": all(clips_found.values()),
    }
    record["preconditions"] = {
        "checks": checks,
        "weights_sha256": found,
        "prompt_sha256": prompt.sha256,
        "clips_match": clips_found,
    }
    if not all(checks.values()):
        record["stop_reason"] = "precondition_failed"
        _write_json(attempt, RUN_FILE, record, config_)
        _write_json(attempt, VERDICT_FILE, evaluate(attempt), config_)
        failed = [name for name, passed in checks.items() if not passed]
        raise Gate0Error(f"run refused before the server start: {failed} do not match")

    entries = plan(manifest, words, prompt)
    binary = llama_dir / SERVER_BINARY
    others = _other_inference()
    threads = threads or (THREADS_BUSY if others else THREADS_IDLE)
    port = _free_port()
    argv = server_argv(binary, model_dir, port, threads)
    record.update(
        {
            "argv": argv,
            "server_version": _server_version(binary),
            "threads": threads,
            "other_inference": others,
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "loadavg": os.getloadavg(),
            "limits": {
                "request_timeout_s": request_timeout,
                "load_timeout_s": load_timeout,
                "rss_limit_bytes": rss_limit,
                "rss_poll_s": RSS_POLL_S,
            },
            "sampling": SAMPLING,
            "requests_planned": len(entries),
        }
    )
    _mkdir(attempt / REQUESTS_DIR, config_)
    _mkdir(attempt / RESPONSES_DIR, config_)
    client = Client(port)
    stop_reason = None
    with (
        paths.open_output(attempt, TRACE_FILE, config_) as trace,
        paths.open_output(attempt, SERVER_LOG, config_) as log,
    ):
        process, copier = proc.start(argv, log=log, max_log_bytes=SERVER_LOG_MAX_BYTES)
        watch = RssWatch(process, rss_limit).start()
        try:
            started = time.monotonic()
            stop_reason = _wait_ready(client, process, watch, load_timeout)
            record["model_load_s"] = round(time.monotonic() - started, 3)
            _trace(trace, _empty_input(client))
            for entry in entries:
                if stop_reason is not None:
                    _trace(trace, {**_trace_keys(entry), "sent": False, "not_sent": stop_reason})
                    continue
                line = _send(entry, out, manifest, client, watch, attempt, config_, request_timeout)
                _trace(trace, line)
                stop_reason = line.get("limit") or (
                    "server_exited" if process.poll() is not None else None
                )
        finally:
            status = proc_status(process.pid)
            watch.stop()
            stop_server(process)
            copier.join(STOP_GRACE_S)
            record.update(
                {
                    "finished_at": _now(),
                    "stop_reason": stop_reason,
                    "rss_exceeded_bytes": watch.exceeded,
                    "rss_peak_bytes": watch.peak,
                    "vm_hwm_bytes": status.get("VmHWM"),
                    "server_returncode": process.returncode,
                    "http_calls": client.calls,
                }
            )
            _write_json(attempt, RUN_FILE, record, config_)
    result = evaluate(attempt)
    _write_json(attempt, VERDICT_FILE, result, config_)
    return result


def _wait_ready(client: Client, process, watch: RssWatch, timeout: float) -> str | None:
    deadline = time.monotonic() + timeout
    while True:
        if watch.exceeded is not None:
            return "rss_limit"
        if process.poll() is not None:
            return "server_exited"
        status, _, _ = client.request("/health", timeout=5)
        if status == 200:
            return None
        if time.monotonic() > deadline:
            return "load_timeout"
        time.sleep(RSS_POLL_S)


def _empty_input(client: Client) -> dict:
    """E0: an empty payload goes through the audio request path and is refused before HTTP."""
    before = client.calls
    clip = review.Clip("E0", b"", SAMPLE_RATE, 1, "s16le", 0.0, hashlib.sha256(b"").hexdigest())
    line = {"n": 0, "variant": "E0", "clip": "E0", "modality": "audio", "sent": False}
    try:
        _audio_part(clip, b"")
    except review.AudioPayloadMissing as error:
        line.update({"rejected": True, "error": str(error)})
    else:
        line.update({"rejected": False})
    line["server_calls"] = client.calls - before
    return line


def _audio_part(clip: review.Clip, wav: bytes) -> dict:
    review.check_payload(clip)
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"},
    }


def _send(entry, out, manifest, client, watch, attempt, config_, request_timeout) -> dict:
    line = {**_trace_keys(entry), "sent": True}
    content: list[dict] = [{"type": "text", "text": entry["text"]}]
    redacted: list[dict] = [{"type": "text", "text": entry["text"]}]
    if entry["modality"] == "audio":
        clip = next(item for item in manifest["clips"] if item["id"] == entry["clip"])
        wav = (out / clip["file"]).read_bytes()
        pcm = wav_pcm(wav)
        payload = review.Clip(
            clip["id"], pcm, SAMPLE_RATE, 1, "s16le", len(pcm) / 2 / SAMPLE_RATE, clip["pcm_sha256"]
        )
        content.insert(0, _audio_part(payload, wav))
        hashes = {
            "wav_sha256": hashlib.sha256(wav).hexdigest(),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "wav_bytes": len(wav),
            "duration_s": payload.duration_s,
            "sample_rate": SAMPLE_RATE,
        }
        line.update(hashes)
        redacted.insert(0, {"type": "input_audio", "input_audio": {"format": "wav", **hashes}})
        line["text_tokens"] = client.text_tokens(entry["text"])
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": entry["max_tokens"],
        **SAMPLING,
    }
    if entry["json_response"]:
        body["response_format"] = {"type": "json_object"}
    _write_json(
        attempt / REQUESTS_DIR,
        f"{entry['n']}.json",
        {**body, "messages": [{"role": "user", "content": redacted}]},
        config_,
    )
    line["rss_before_bytes"] = watch.current()
    watch.reset_window()
    started = time.monotonic()
    status, raw, error = client.request(CHAT_ROUTE, body, timeout=request_timeout)
    line["wall_s"] = round(time.monotonic() - started, 3)
    line.update(
        {
            "rss_after_bytes": watch.current(),
            "rss_peak_bytes": watch.window_peak,
            "http_status": status,
            "error": error,
        }
    )
    if raw is not None:
        with paths.open_output(attempt / RESPONSES_DIR, f"{entry['n']}.json", config_) as file:
            file.write(raw)
        line["response_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        parsed = json.loads(raw) if raw is not None else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        usage = parsed.get("usage") or {}
        choice = (parsed.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        line.update(
            {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "timings": parsed.get("timings"),
                "reasoning_content_present": bool(message.get("reasoning_content")),
            }
        )
    if entry["modality"] == "audio" and line.get("text_tokens") is not None:
        prompt_tokens = line.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            line["audio_tokens"] = prompt_tokens - line["text_tokens"]
    line["answer"] = answer_of(entry, parsed)
    if watch.exceeded is not None:
        line["limit"] = "rss_limit"
    elif line["wall_s"] > request_timeout or (error is not None and "timed out" in error):
        line["limit"] = "request_timeout"
    return line


def _trace_keys(entry: Mapping) -> dict:
    keys = ("n", "variant", "clip", "modality", "pair", "take", "duration_s", "max_tokens")
    line = {key: entry[key] for key in keys}
    if "about" in entry:
        line["about"] = entry["about"]
    return line


def _trace(trace, line: dict) -> None:
    trace.write((json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    trace.flush()


# --- the verdict ---------------------------------------------------------------------------


def evaluate(attempt: Path) -> dict:
    """The verdict of one attempt from its saved files alone: answers re-parsed from responses."""
    out = attempt.parent.parent
    manifest = _read_json(out / MANIFEST_FILE)
    record = _read_json(attempt / RUN_FILE)
    trace = attempt / TRACE_FILE
    texts = trace.read_text(encoding="utf-8").splitlines() if trace.is_file() else []
    lines = []
    for text in texts:
        line = json.loads(text)
        response = attempt / RESPONSES_DIR / f"{line['n']}.json"
        if line.get("sent") and line["variant"] != "E0":
            body = None
            if response.is_file():
                with contextlib.suppress(ValueError):
                    body = json.loads(response.read_bytes())
            line["answer"] = answer_of(line, body)
        lines.append(line)
    return decide(manifest, record, lines)


def decide(manifest: Mapping, record: Mapping, lines: list[Mapping]) -> dict:
    """GATE0-RULE v1 over the trace of one attempt (docs/decisions/0007, section 3)."""
    by_key = {(line["variant"], line["clip"], line.get("take")): line for line in lines}
    e0 = by_key.get(("E0", "E0", None))
    checks = (record.get("preconditions") or {}).get("checks") or {}
    route = audio_route(lines)
    preconditions = {
        "P1": bool(e0 and e0.get("rejected") is True and e0.get("server_calls") == 0),
        "P2": checks.get("weights") is True and checks.get("prompt") is True,
        "P3": route["passed"],
    }
    take = "a" if manifest["mode"] == "natural" else None
    controls = []
    for clip in CONTROLS:
        for variant in ("K1", "K2", "P", "A"):
            line = by_key.get((variant, clip, take if variant == "P" else None))
            answer = line.get("answer") if line else None
            if variant in ("K1", "K2"):
                passed = _passed(line) and answer not in ("речь", INVALID)
            elif variant == "P":
                passed = _passed(line) and answer in ("нет", "нет речи")
            else:
                passed = (
                    _passed(line)
                    and answer.get("valid") is True
                    and answer.get("observations") == 0
                    and answer.get("unlocalized_observations") == 0
                )
            controls.append({"clip": clip, "variant": variant, "answer": answer, "passed": passed})
    c_passed = all(item["passed"] for item in controls)
    result: dict = {
        "rule": "GATE0-RULE v1",
        "mode": manifest["mode"],
        "preconditions": preconditions,
        "audio_route": route,
        "controls": controls,
        "C": c_passed,
        "stop_reason": record.get("stop_reason"),
    }
    if manifest["mode"] == "synthetic":
        pairs = []
        for pair in manifest["pairs"]:
            base = by_key.get(("P", pair["base"], None))
            variant = by_key.get(("P", pair["variant"], None))
            answers = {"base": _closed(base), "variant": _closed(variant)}
            distinguished = (
                pair["target"] is not None
                and answers["variant"] == "да"
                and answers["base"] == "нет"
            )
            pairs.append({"pair": pair["pair"], **answers, "distinguished": distinguished})
        result["synthetic_pairs"] = pairs
        result["S"] = sum(item["distinguished"] for item in pairs) >= 2
        second, outcome = result["S"], "HELD"
        second_reason = "S: различено меньше 2 из 3 синтетических пар"
        held_reason = "C и S проходят: нет вердикта без естественных пар"
    else:
        pairs = []
        for pair in manifest["pairs"]:
            answers = {
                "a_in_a": _closed(by_key.get(("P", pair["a"], "a"))),
                "a_in_b": _closed(by_key.get(("P", pair["b"], "a"))),
                "b_in_b": _closed(by_key.get(("P", pair["b"], "b"))),
                "b_in_a": _closed(by_key.get(("P", pair["a"], "b"))),
            }
            distinguished = answers == {
                "a_in_a": "да",
                "a_in_b": "нет",
                "b_in_b": "да",
                "b_in_a": "нет",
            }
            pairs.append({"pair": pair["pair"], **answers, "distinguished": distinguished})
        result["natural_pairs"] = pairs
        result["N"] = sum(item["distinguished"] for item in pairs) >= 2
        second, outcome = result["N"], "READY"
        second_reason = "N: различено меньше 2 из 3 естественных пар"
        held_reason = "C и N проходят"
    if not all(preconditions.values()):
        failed = [name for name, passed in preconditions.items() if not passed]
        result["verdict"], result["reason"] = "NOT READY", f"протокол не выполнен: {failed}"
    elif not c_passed:
        failed = [f"{item['variant']}({item['clip']})" for item in controls if not item["passed"]]
        result["verdict"] = "NOT READY"
        result["reason"] = f"C: не прошли {len(failed)} из 12 пунктов контролей: {failed}"
    elif not second:
        result["verdict"], result["reason"] = "NOT READY", second_reason
    else:
        result["verdict"], result["reason"] = outcome, held_reason
    return result


def audio_route(lines: list[Mapping]) -> dict:
    """P3: audio tokens of each tokenized request; the K1 slope covers the requests without them."""
    answered = [
        line
        for line in lines
        if line.get("modality") == "audio" and line.get("sent") and line.get("http_status") == 200
    ]
    if not answered:
        return {"method": None, "passed": False, "reason": "no answered audio request"}
    tokenized = [line for line in answered if line.get("text_tokens") is not None]
    failing = [
        line["n"]
        for line in tokenized
        if not isinstance(line.get("audio_tokens"), int)
        or line["audio_tokens"] < MIN_AUDIO_TOKENS_PER_S * line["duration_s"]
    ]
    rates = [
        line["audio_tokens"] / line["duration_s"]
        for line in tokenized
        if isinstance(line.get("audio_tokens"), int)
    ]
    route: dict = {
        "method": "tokenize",
        "passed": not failing,
        "failing": failing,
        "min_tokens_per_s": round(min(rates), 2) if rates else None,
        "max_tokens_per_s": round(max(rates), 2) if rates else None,
    }
    untokenized = [line["n"] for line in answered if line.get("text_tokens") is None]
    if not untokenized:
        return route
    route["method"] = "tokenize+k1_slope" if tokenized else "k1_slope"
    route["untokenized"] = untokenized
    points = [
        (line["duration_s"], line["prompt_tokens"])
        for line in answered
        if line["variant"] == "K1" and isinstance(line.get("prompt_tokens"), int)
    ]
    if len({duration for duration, _ in points}) < 2:
        route.update(passed=False, reason="too few K1 points")
        return route
    durations, tokens = np.array(points, dtype=float).T
    route["slope"] = float(np.polyfit(durations, tokens, 1)[0])
    route["passed"] = not failing and route["slope"] >= MIN_AUDIO_TOKENS_PER_S
    return route


def _passed(line: Mapping | None) -> bool:
    return bool(
        line
        and line.get("sent")
        and line.get("http_status") == 200
        and line.get("finish_reason") != "length"
        and line.get("answer") != INVALID
    )


def _closed(line: Mapping | None) -> str:
    return line["answer"] if _passed(line) else INVALID


# --- small helpers -------------------------------------------------------------------------


def _mkdir(path: Path, config_: config.Config) -> Path:
    target = paths.resolve_output(path.parent, path.name, config_)
    with contextlib.suppress(FileExistsError):
        os.mkdir(target, DIR_MODE)
    return target


def _new_attempt(out: Path, config_: config.Config) -> Path:
    attempts = _mkdir(out / ATTEMPTS_DIR, config_)
    number = 1
    while True:
        target = paths.resolve_output(attempts, str(number), config_)
        try:
            os.mkdir(target, DIR_MODE)
        except FileExistsError:
            number += 1
            continue
        return target


def _write_json(directory: Path, name: str, data: object, config_: config.Config) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with paths.open_output(directory, name, config_) as out:
        out.write((text + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Gate0Error(f"{path.name} is missing; run prepare first") from None


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            while chunk := source.read(1 << 20):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _code_identity() -> dict:
    here = Path(__file__).resolve()
    identity: dict = {"gate0_sha256": _sha256_file(here), "commit": None, "dirty": None}
    git = ["git", "-C", str(here.parent)]
    head = _output([*git, "rev-parse", "HEAD"], timeout=10)
    status = _output([*git, "status", "--porcelain"], timeout=10)
    if head is not None:
        identity["commit"] = head.strip()
    if status is not None:
        identity["dirty"] = bool(status.strip())
    return identity


def _server_version(binary: Path) -> str | None:
    output = _output([str(binary), "--version"], timeout=30, stderr=True)
    lines = (output or "").splitlines()
    return next((line.strip() for line in lines if line.startswith("version:")), None)


def _output(argv: list[str], *, timeout: float, stderr: bool = False) -> str | None:
    """Stdout (and stderr) of a short bounded child that succeeded, else None."""
    try:
        done = proc.run(argv, timeout=timeout, max_output_bytes=1 << 20)
    except OSError:
        return None
    if done.timed_out or done.returncode != 0:
        return None
    return (done.stdout + (done.stderr if stderr else b"")).decode("utf-8", "replace")


def _other_inference() -> list[str]:
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            with contextlib.suppress(OSError):
                name = (entry / "comm").read_text().strip()
                if name.startswith("llama-"):
                    found.append(name)
    return sorted(found)


def _cpu_model() -> str | None:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.gate0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "verdict"):
        command = commands.add_parser(name)
        command.add_argument("--date", required=True)
        command.add_argument("--decision", type=Path, default=DECISION_DOC)
    commands.choices["prepare"].add_argument("--sources", type=Path)
    commands.choices["prepare"].add_argument("--natural", type=Path)
    commands.choices["run"].add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    commands.choices["run"].add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    commands.choices["run"].add_argument("--threads", type=int)
    commands.choices["verdict"].add_argument("--attempt")
    args = parser.parse_args(argv)
    try:
        config_ = config.load()
        if args.command == "prepare":
            guard.install_network_guard(config_)
            manifest = prepare_clips(
                args.date,
                config_,
                sources=args.sources,
                natural=args.natural,
                decision=args.decision,
            )
            for clip in manifest["clips"]:
                print(json.dumps({k: clip[k] for k in ("id", "duration_s", "pcm_sha256")}))
            for pair in manifest["pairs"]:
                print(json.dumps({k: v for k, v in pair.items() if k != "f0_median_hz"}))
            return 0
        if args.command == "run":
            result = run_spike(
                args.date,
                config_,
                decision=args.decision,
                model_dir=args.model_dir,
                llama_dir=args.llama_dir,
                threads=args.threads,
            )
        else:
            guard.install_network_guard(config_)
            attempts = output_dir(args.date, config_) / ATTEMPTS_DIR
            found = attempts.iterdir() if attempts.is_dir() else ()
            numbers = [int(p.name) for p in found if p.name.isdigit()]
            name = args.attempt or (str(max(numbers)) if numbers else "")
            if not name.isdigit() or not (attempts / name / RUN_FILE).is_file():
                raise Gate0Error(f"{args.date}: attempt {name or '-'} not found; run `run` first")
            result = evaluate(attempts / name)
    except (Gate0Error, config.ConfigError, paths.PathDenied) as error:
        print(f"gate0: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps({"verdict": result["verdict"], "reason": result["reason"]}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
