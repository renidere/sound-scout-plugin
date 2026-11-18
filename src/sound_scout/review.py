"""Audio backend adapter: call contract, payload check, provenance, response validator.

One `AudioBackend` protocol, one implementation path (R03, R05, R10, T03). `run_review` feeds
a `Clip` to the backend and refuses (`AudioPayloadMissing`) before the call when the payload is
empty or its hash does not match the clip, so a transcript-only or empty "review" can never be
recorded as a completed audio review. Each call leaves a verifiable trace in
`review/<clip id>/` of the run directory: `provenance.json` (R03: backend/model/revision,
duration and sha256 of the payload actually passed, preprocessing, ranges on the clip's own
time scale — `findings.build` maps them onto the source through `clips/<id>.json` — prompt hash,
settings, request IDs, response hash and validity) and `response.raw.txt` (the model output
verbatim, never rewritten). A call that fails or returns the wrong type leaves
`provenance.json` with `response.valid=false` and the error, so the run keeps the trace and never
calls twice.

The model's JSON is validated independently by `validate_response`: only the four response
fields of `AUDIO_REVIEW_PROMPT.md`, the sound categories of the result contract
(`schema.SOUND_CATEGORIES`), timecodes inside the clip, qualitative confidence notes.
Server-only fields (`human_verified`, `approved`, `cost`, …) and numeric confidence are rejected.
An observation without timecodes is accepted only in `unlocalized_observations` and is not a
finding; an interval spanning the whole clip is marked `whole_clip` and is not the localisation
of a specific error. Instruction-like text in the
trusted fields or in the model output is never interpreted: it is listed in `suspicious_content`
by path and pattern, without repeating the text; a key the contract does not name is model
text too and appears in paths and errors only as `<unexpected field #n>`.

`cover` reviews requested source ranges window by window: `windows` cuts each range into
overlapping windows no longer than the backend's own `limits["max_audio_seconds"]` (a backend
without that limit gets one window per range), each window becomes its own clip and its own
review record, and a failed backend call stops the walk, so the remaining windows stay unreviewed
and nothing is called twice (R03, R11). The request carries no transcript: the first pass is blind.

The prompt is the packaged copy of `docs/spec/AUDIO_REVIEW_PROMPT.md` (byte-equal, checked by
the tests); `load_prompt` reads it and takes its SHA-256 as the prompt version. The backend
decides nothing about provenance, stage status or validity: the server writes those.
"""

import contextlib
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Protocol

from sound_scout import __version__, clips, guard, paths, schema
from sound_scout.config import MIB, Config
from sound_scout.doctor import TEST_DOUBLE
from sound_scout.runs import Run

REVIEW_DIR_NAME = "review"
PROVENANCE_FILE_NAME = "provenance.json"
RAW_RESPONSE_FILE_NAME = "response.raw.txt"
PROVENANCE_VERSION = 1
PROMPT_FILE_NAME = "AUDIO_REVIEW_PROMPT.md"
DEFAULT_PROMPT_PATH = files("sound_scout").joinpath(PROMPT_FILE_NAME)
CATEGORIES = schema.SOUND_CATEGORIES
RESPONSE_FIELDS = ("observations", "unlocalized_observations", "uncertainties", "limitations")
OBSERVATION_FIELDS = (
    "category",
    "observation",
    "start_seconds",
    "end_seconds",
    "confidence_note",
    "impact",
    "correction",
)
TIMECODE_FIELDS = ("start_seconds", "end_seconds")
UNLOCALIZED_FIELDS = tuple(name for name in OBSERVATION_FIELDS if name not in TIMECODE_FIELDS)
TEXT_FIELDS = ("observation", "confidence_note", "impact")
TRUSTED_FIELD_NAMES = ("clip_id", "duration_s", "task", "intent")
# How app puts the response's limitations and uncertainties into `provenance.limitations`.
MODEL_LIMITATION = "audio backend (заявлено моделью), ограничение: "
MODEL_UNCERTAINTY = "audio backend (заявлено моделью), неопределённость: "
SERVER_ONLY_FIELDS = {
    "human_verified": "human_verified is set only by a human evidence record (R05)",
    "approved": "approval is never taken from a model response (R09)",
    "consent": "consent is never taken from a model response (R09)",
    "cost": "cost is set by the server (R09)",
    "provenance": "provenance is set by the server (R03)",
    "status": "stage status is set by the server (R11)",
    "confidence": "confidence is a qualitative confidence_note, not a number (R05)",
    "confidence_percent": "confidence is a qualitative confidence_note, not a number (R05)",
    "probability": "confidence is a qualitative confidence_note, not a number (R05)",
}
NAMED_KEYS = frozenset(RESPONSE_FIELDS + OBSERVATION_FIELDS + TRUSTED_FIELD_NAMES) | frozenset(
    SERVER_ONLY_FIELDS
)
MAX_RESPONSE_BYTES = 1 * MIB
TIME_EPS = 0.001
MAX_AUDIO_SECONDS_LIMIT = "max_audio_seconds"
WINDOW_OVERLAP_S = 2.0
WINDOW_PAD_MS = 0
WHOLE_CLIP_FRACTION = 0.95
DIR_MODE = 0o700
SUSPICIOUS_PATTERNS = {
    "approval_flag": re.compile(
        r"\b(approved|approval|consent|human_verified|validated_ru)\b\s*[:=]\s*(true|1|yes|да)\b",
        re.IGNORECASE,
    ),
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|prompt|system)\b"
        r"|\b(игнорир|забуд|отмен)\w*\b.{0,40}\b(инструкци|правил|промпт)\w*",
        re.IGNORECASE | re.DOTALL,
    ),
    "run_command": re.compile(
        r"\b(run|execute|exec|launch|invoke|call)\b.{0,30}\b(command|shell|script|tool|function|mcp)s?\b"
        r"|\b(выполни|запусти|исполни|вызови)\w*\b.{0,30}\b(команд|скрипт|инструмент|функци)\w*",
        re.IGNORECASE | re.DOTALL,
    ),
    "shell_syntax": re.compile(
        r"(^|[\s;&|(`])(rm\s+-[a-z]*f|sudo|curl|wget|chmod|bash|sh|powershell|cmd\.exe)\b"
        r"|\$\(|`[^`\n]+`",
        re.IGNORECASE,
    ),
    "secret_request": re.compile(
        r"\b(api[_ -]?keys?|secrets?|passwords?|tokens?|credentials?|private keys?)\b"
        r"|\b(парол|секрет|ключ(и|ей|а)? api|токен|учётн)\w*",
        re.IGNORECASE,
    ),
    "path_or_url": re.compile(
        r"\b(https?|ftp|file)://|(^|[\s\"'(])(/home/|/etc/|/root/|~/|[A-Za-z]:\\)",
        re.IGNORECASE,
    ),
}
_PERCENT = re.compile(r"\d\s*(%|percent|процент)", re.IGNORECASE)


class AudioPayloadMissing(Exception):
    """The clip carries no audio payload or a payload that is not the clip's; nothing was called."""


class ReviewRecordExists(Exception):
    """The run already holds a review record for this clip; nothing was called or overwritten."""


# What `run_review` raises for a bad payload or path or a denied network attempt, not a failed
# call; `cover` passes it on.
_NOT_A_CALL_FAILURE = (AudioPayloadMissing, clips.ClipError, paths.PathDenied, guard.NetworkDenied)


@dataclass(frozen=True)
class Clip:
    """A fragment of the decoded original, as the backend receives it (`clip_from_cut`).

    `payload` is raw PCM (`sample_format`, `sample_rate`, `channels`), `sha256` is the hash of
    exactly those bytes, `duration_s` is the length in seconds on the clip's own time scale.
    """

    id: str
    payload: bytes
    sample_rate: int
    channels: int
    sample_format: str
    duration_s: float
    sha256: str
    preprocessing: tuple[str, ...] = ()


def make_clip(
    clip_id: str,
    payload: bytes,
    sample_rate: int,
    *,
    channels: int = 1,
    sample_format: str = "f32le",
    bytes_per_sample: int = 4,
    preprocessing: tuple[str, ...] = (),
) -> Clip:
    """A `Clip` with `duration_s` and `sha256` derived from the payload itself."""
    frames = len(payload) // (bytes_per_sample * channels)
    return Clip(
        id=clip_id,
        payload=payload,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        duration_s=frames / sample_rate,
        sha256=hashlib.sha256(payload).hexdigest(),
        preprocessing=preprocessing,
    )


def clip_from_cut(clip: clips.Clip) -> Clip:
    """The backend payload of a clip `clips.cut` wrote: its samples as float32, same id."""
    return make_clip(
        clip.id,
        clips.pcm_f32le(clip),
        clip.record["sample_rate"],
        channels=clip.record["channels"],
    )


@dataclass(frozen=True)
class Prompt:
    text: str
    sha256: str
    name: str = PROMPT_FILE_NAME


def load_prompt(path: Path | Traversable = DEFAULT_PROMPT_PATH) -> Prompt:
    """The audio review prompt and its SHA-256 (the prompt version in provenance)."""
    data = path.read_bytes()
    return Prompt(data.decode("utf-8"), hashlib.sha256(data).hexdigest(), path.name)


@dataclass(frozen=True)
class ReviewRequest:
    """Everything the backend gets: the audio, the trusted fields and the prompt. No tools."""

    clip_id: str
    payload: bytes
    sample_rate: int
    channels: int
    sample_format: str
    duration_s: float
    prompt: str
    prompt_sha256: str
    task: str | None = None
    intent: str | None = None
    response_fields: tuple[str, ...] = RESPONSE_FIELDS

    @property
    def trusted_fields(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "duration_s": self.duration_s,
            "task": self.task,
            "intent": self.intent,
        }


@dataclass(frozen=True)
class RawReviewResponse:
    """What the backend returned, verbatim; `request_id` is the backend's own, when it has one."""

    text: str
    request_id: str | None = None


@dataclass(frozen=True)
class BackendDescription:
    name: str
    model: str | None
    revision: str | None
    input_modalities: tuple[str, ...]
    limits: Mapping[str, object] = field(default_factory=dict)
    settings: Mapping[str, object] = field(default_factory=dict)


class AudioBackend(Protocol):
    def describe(self) -> BackendDescription: ...

    def review(self, request: ReviewRequest) -> RawReviewResponse: ...


@dataclass(frozen=True)
class ReviewResponse:
    """The validated model output; items are plain dicts, `whole_clip` added to observations."""

    observations: tuple[dict[str, object], ...]
    unlocalized_observations: tuple[dict[str, object], ...]
    uncertainties: tuple[str, ...]
    limitations: tuple[str, ...]
    suspicious_content: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ReviewOutcome:
    """One backend call: `response` is None when `errors` is not empty."""

    provenance: dict[str, object]
    provenance_path: Path
    raw_response_path: Path
    response: ReviewResponse | None
    errors: tuple[str, ...]
    suspicious_content: tuple[dict[str, str], ...]


def check_payload(clip: Clip) -> None:
    """Raise `AudioPayloadMissing` unless the clip carries the audio its hash names (T03)."""
    if not isinstance(clip.payload, bytes | bytearray | memoryview) or len(clip.payload) == 0:
        raise AudioPayloadMissing(f"clip {clip.id!r} has no audio payload")
    if not _number(clip.duration_s) or clip.duration_s <= 0:
        raise AudioPayloadMissing(
            f"clip {clip.id!r} has duration {clip.duration_s}, needs a finite number > 0"
        )
    actual = hashlib.sha256(clip.payload).hexdigest()
    if actual != clip.sha256:
        raise AudioPayloadMissing(
            f"clip {clip.id!r} payload sha256 {actual[:12]}… does not match the clip's "
            f"{str(clip.sha256)[:12]}…"
        )


def build_request(
    clip: Clip, prompt: Prompt, *, task: str | None = None, intent: str | None = None
) -> ReviewRequest:
    check_payload(clip)
    return ReviewRequest(
        clip_id=clip.id,
        payload=bytes(clip.payload),
        sample_rate=clip.sample_rate,
        channels=clip.channels,
        sample_format=clip.sample_format,
        duration_s=clip.duration_s,
        prompt=prompt.text,
        prompt_sha256=prompt.sha256,
        task=task,
        intent=intent,
    )


def run_review(
    backend: AudioBackend,
    clip: Clip,
    run: Run,
    config: Config,
    prompt: Prompt,
    *,
    task: str | None = None,
    intent: str | None = None,
) -> ReviewOutcome:
    """Review one clip: check the payload, call the backend once, record the trace, validate.

    Order matters: the payload check, the record-exists check, the serialisation of the static
    part of the provenance and the creation of the record directory come before the call, so a
    refused call never reaches the model and a call is never made that cannot be recorded. The
    raw response and `provenance.json` are written even when the response is invalid. When the
    call raises or returns the wrong type, `provenance.json` records the failure
    (`response.valid=false`, error type and message, no raw file) and the exception propagates.
    Once the raw response is written it stays, even if the provenance cannot be written.
    `ranges` are on the clip's own time scale; `findings.build` maps them onto the source.
    """
    request = build_request(clip, prompt, task=task, intent=intent)
    review_dir = paths.resolve_output(run.path, REVIEW_DIR_NAME, config)
    clip_dir = review_dir / clip.id
    if not paths.OUTPUT_NAME.fullmatch(clip.id) or clip.id in (".", ".."):
        raise paths.PathDenied(clip.id, paths.BAD_OUTPUT_NAME)
    for name in (PROVENANCE_FILE_NAME, RAW_RESPONSE_FILE_NAME):
        if os.path.lexists(clip_dir / name):
            raise ReviewRecordExists(f"{REVIEW_DIR_NAME}/{clip.id}/{name} is already in the run")
    description = backend.describe()
    suspicious = list(scan_suspicious(request.trusted_fields, "trusted"))
    provenance: dict[str, object] = {
        "provenance_version": PROVENANCE_VERSION,
        "sound_scout": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run.run_id,
        "call_id": uuid.uuid4().hex,
        "clip_id": clip.id,
        "backend": {
            "name": description.name,
            "model": description.model,
            "revision": description.revision,
            "input_modalities": list(description.input_modalities),
            "limits": dict(description.limits),
            "settings": dict(description.settings),
        },
        "payload": {
            "sha256": clip.sha256,
            "bytes": len(clip.payload),
            "duration_s": clip.duration_s,
            "sample_rate": clip.sample_rate,
            "channels": clip.channels,
            "sample_format": clip.sample_format,
        },
        "preprocessing": list(clip.preprocessing),
        "ranges": {"clip": [{"start": 0.0, "end": clip.duration_s}]},
        "prompt": {"name": prompt.name, "sha256": prompt.sha256},
        "trusted_fields": {"task": task, "intent": intent},
    }
    json.dumps(provenance, allow_nan=False)
    _make_record_dir(clip_dir)

    try:
        raw = backend.review(request)
        if not isinstance(raw, RawReviewResponse) or not isinstance(raw.text, str):
            raise TypeError("backend.review must return a RawReviewResponse with text")
    except Exception as error:
        provenance["response"] = {
            "valid": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "suspicious_content": suspicious,
        }
        _write_provenance(clip_dir, config, provenance)
        raise

    response, errors, found = validate_response(raw.text, clip.duration_s)
    suspicious += found
    if response is not None:
        response = ReviewResponse(
            response.observations,
            response.unlocalized_observations,
            response.uncertainties,
            response.limitations,
            tuple(suspicious),
        )
    raw_bytes = raw.text.encode("utf-8")
    with paths.open_output(clip_dir, RAW_RESPONSE_FILE_NAME, config) as out:
        out.write(raw_bytes)
    provenance["backend_request_id"] = raw.request_id
    provenance["response"] = {
        "file": RAW_RESPONSE_FILE_NAME,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "valid": response is not None,
        "errors": list(errors),
        "suspicious_content": suspicious,
    }
    _write_provenance(clip_dir, config, provenance)
    return ReviewOutcome(
        provenance,
        clip_dir / PROVENANCE_FILE_NAME,
        clip_dir / RAW_RESPONSE_FILE_NAME,
        response,
        tuple(errors),
        tuple(suspicious),
    )


@dataclass(frozen=True)
class Covered:
    """What `cover` did: the planned windows on the source scale, one outcome per window called,
    and the window whose call failed with the error, after which nothing more was called."""

    windows: list[tuple[float, float]]
    outcomes: list[ReviewOutcome]
    failure: tuple[tuple[float, float], str] | None


def windows(
    ranges: list[tuple[float, float]], max_s: object, overlap_s: float = WINDOW_OVERLAP_S
) -> list[tuple[float, float]]:
    """Windows covering each range: none longer than `max_s`, neighbours overlapping by at least
    `overlap_s`, the last one aligned to the range end; `max_s=None` keeps each range whole."""
    if max_s is not None and not (_number(max_s) and max_s > 2 * overlap_s):
        raise ValueError(
            f"backend limit {MAX_AUDIO_SECONDS_LIMIT} must be a finite number of seconds "
            f"> {2 * overlap_s}"
        )
    planned: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if max_s is None or end - start <= max_s:
            planned.append((start, end))
            continue
        step = max_s - overlap_s
        n = 0
        # A later window ending within TIME_EPS of the range end would duplicate the last one.
        while n == 0 or start + n * step + max_s < end - TIME_EPS:
            planned.append((start + n * step, start + n * step + max_s))
            n += 1
        planned.append((end - max_s, end))
    return planned


def cover(
    backend: AudioBackend,
    run: Run,
    config: Config,
    prompt: Prompt,
    ranges: list[tuple[float, float]],
    *,
    task: str | None = None,
    intent: str | None = None,
    before_call: Callable[[], None] | None = None,
    after_window: Callable[[list[ReviewOutcome]], None] | None = None,
) -> Covered:
    """Review `ranges` of the source window by window, each window its own clip and record.

    The window length is the backend's `limits["max_audio_seconds"]`. An invalid response does
    not stop the walk (`findings.build` leaves that window out of `reviewed`); a failed call does:
    it is recorded by `run_review`, never repeated, and the windows after it are not called (R11).
    `before_call` runs right before every backend call, so a caller can stop the walk there;
    `after_window` gets the outcomes so far after every window reviewed, so a caller can keep them
    before the next call.
    """
    limit = backend.describe().limits.get(MAX_AUDIO_SECONDS_LIMIT)
    planned = windows(ranges, limit)
    outcomes: list[ReviewOutcome] = []
    cut_ids: set[str] = set()
    for window in planned:
        cut = clips.cut(run, *window, WINDOW_PAD_MS, config=config)
        if cut.id in cut_ids:  # two windows less than a frame apart are one clip
            continue
        cut_ids.add(cut.id)
        clip = clip_from_cut(cut)
        if before_call is not None:
            before_call()
        try:
            outcomes.append(
                run_review(backend, clip, run, config, prompt, task=task, intent=intent)
            )
        except _NOT_A_CALL_FAILURE:
            raise
        except Exception as error:
            return Covered(planned, outcomes, (window, f"{type(error).__name__}: {error}"))
        if after_window is not None:
            after_window(list(outcomes))
    return Covered(planned, outcomes, None)


def validate_response(
    text: str, clip_duration_s: float
) -> tuple[ReviewResponse | None, list[str], list[dict[str, str]]]:
    """Parse and check the model output; (response, errors, suspicious_content).

    `response` is None exactly when `errors` is not empty. `suspicious_content` is reported in
    both cases and is not an error: the text is data that was not interpreted.
    """
    errors: list[str] = []
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return None, [f"$: response is larger than {MAX_RESPONSE_BYTES} bytes"], []
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        return None, [f"$: response is not valid JSON: {error}"], list(scan_suspicious(text, "$"))
    suspicious = list(scan_suspicious(data, "$"))
    if not isinstance(data, dict):
        return None, ["$: response must be a JSON object"], suspicious
    errors += _field_errors("$", data, RESPONSE_FIELDS)
    for name in RESPONSE_FIELDS:
        if name in data and not isinstance(data[name], list):
            errors.append(f"$.{name}: must be a list")
    if errors:
        return None, errors, suspicious
    observations = []
    for i, item in enumerate(data["observations"]):
        checked, found = _observation(f"$.observations[{i}]", item, clip_duration_s)
        observations.append(checked)
        errors += found
    unlocalized = []
    for i, item in enumerate(data["unlocalized_observations"]):
        checked, found = _unlocalized(f"$.unlocalized_observations[{i}]", item)
        unlocalized.append(checked)
        errors += found
    for name in ("uncertainties", "limitations"):
        errors += [
            f"$.{name}[{i}]: must be a non-empty string"
            for i, item in enumerate(data[name])
            if not _text(item)
        ]
    if errors:
        return None, errors, suspicious
    response = ReviewResponse(
        tuple(observations),
        tuple(unlocalized),
        tuple(data["uncertainties"]),
        tuple(data["limitations"]),
        tuple(suspicious),
    )
    return response, [], suspicious


def _observation(path: str, item: object, duration: float) -> tuple[dict[str, object], list[str]]:
    if not isinstance(item, dict):
        return {}, [f"{path}: must be an object"]
    errors = _field_errors(path, item, OBSERVATION_FIELDS) + _common_errors(path, item)
    for name in TIMECODE_FIELDS:
        if item.get(name) is None:
            errors.append(
                f"{path}.{name}: an observation without timecodes belongs to "
                "unlocalized_observations"
            )
        elif not _number(item[name]):
            errors.append(f"{path}.{name}: must be a finite number of seconds")
    if errors:
        return {}, errors
    start, end = item["start_seconds"], item["end_seconds"]
    if start < 0:
        errors.append(f"{path}.start_seconds: must be >= 0")
    if start >= end:
        errors.append(f"{path}: start_seconds must be less than end_seconds")
    if end > duration + TIME_EPS:
        errors.append(f"{path}.end_seconds: exceeds the clip duration {duration}")
    if errors:
        return {}, errors
    checked = {name: item[name] for name in OBSERVATION_FIELDS}
    checked["whole_clip"] = end - start >= WHOLE_CLIP_FRACTION * duration
    return checked, []


def _unlocalized(path: str, item: object) -> tuple[dict[str, object], list[str]]:
    if not isinstance(item, dict):
        return {}, [f"{path}: must be an object"]
    allowed = UNLOCALIZED_FIELDS + TIMECODE_FIELDS
    errors = _field_errors(path, item, allowed, required=UNLOCALIZED_FIELDS)
    errors += _common_errors(path, item)
    errors += [
        f"{path}.{name}: an unlocalized observation carries no timecodes; use observations"
        for name in TIMECODE_FIELDS
        if item.get(name) is not None
    ]
    if errors:
        return {}, errors
    return {name: item[name] for name in UNLOCALIZED_FIELDS}, []


def _common_errors(path: str, item: dict) -> list[str]:
    errors = []
    if "category" in item and item["category"] not in CATEGORIES:
        errors.append(f"{path}.category: must be one of {list(CATEGORIES)}")
    errors += [
        f"{path}.{name}: must be a non-empty string"
        for name in TEXT_FIELDS
        if name in item and not _text(item[name])
    ]
    if "correction" in item and item["correction"] is not None and not _text(item["correction"]):
        errors.append(f"{path}.correction: must be a non-empty string or null")
    note = item.get("confidence_note")
    if isinstance(note, str) and _PERCENT.search(note):
        errors.append(
            f"{path}.confidence_note: confidence is qualitative, a percentage is not accepted"
        )
    return errors


def _field_errors(
    path: str, item: dict, allowed: tuple[str, ...], *, required: tuple[str, ...] | None = None
) -> list[str]:
    errors = []
    for i, name in enumerate(item):
        if name not in allowed:
            why = SERVER_ONLY_FIELDS.get(name, "not a field of the response contract")
            errors.append(f"{path}.{_key_label(name, i)}: unexpected field, {why}")
    errors += [f"{path}.{name}: required" for name in (required or allowed) if name not in item]
    return errors


def _key_label(key: object, index: int) -> str:
    """A key the contract names is repeated; any other key is model text and is only counted."""
    return key if isinstance(key, str) and key in NAMED_KEYS else f"<unexpected field #{index}>"


def scan_suspicious(value: object, path: str) -> Iterator[dict[str, str]]:
    """Yield `{"path", "pattern"}` for every string matching an instruction-like pattern.

    The text itself is never repeated: what matched is data, not an instruction (R10, T07).
    """
    if isinstance(value, str):
        for name, pattern in SUSPICIOUS_PATTERNS.items():
            if pattern.search(value):
                yield {"path": path, "pattern": name}
    elif isinstance(value, Mapping):
        for i, (key, item) in enumerate(value.items()):
            label = f"{path}.{_key_label(key, i)}"
            if isinstance(key, str) and key not in NAMED_KEYS:
                yield from scan_suspicious(key, label)
            yield from scan_suspicious(item, label)
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            yield from scan_suspicious(item, f"{path}[{i}]")


def _make_record_dir(clip_dir: Path) -> None:
    for directory in (clip_dir.parent, clip_dir):
        try:
            os.mkdir(directory, DIR_MODE)
        except FileExistsError:
            continue
        os.chmod(directory, DIR_MODE)


def _write_provenance(clip_dir: Path, config: Config, provenance: dict[str, object]) -> None:
    text = json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    out = paths.open_output(clip_dir, PROVENANCE_FILE_NAME, config)
    try:
        with out:
            out.write(text.encode("utf-8") + b"\n")
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(clip_dir / PROVENANCE_FILE_NAME)
        raise


def is_test_double(provenance: Mapping[str, object]) -> bool:
    """True when the provenance names the test double; such a run proves nothing (T03, T09)."""
    backend = provenance.get("backend")
    return isinstance(backend, Mapping) and backend.get("name") == TEST_DOUBLE


def _reject_constant(name: str) -> object:
    raise ValueError(f"{name} is not valid JSON")


def _number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
