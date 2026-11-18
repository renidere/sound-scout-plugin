"""Initial and control attempt compared by one criterion (R07, T10); see 0006-compare-rules.md.

`compare` reads two complete `analyze` runs by `run_id` (never by path: R10), checks each
`result.json` with `findings.validate` against the files of its run and compares them by one
criterion of `CRITERIA`, each bound to one audio-review category. The verdict comes only from the
audio model's findings of that category inside the criterion's windows (and its unlocalized
observations): measurements, duration, pauses, level, channel, format, gain, task, intent and
backend go to `conditions_diff` and can only make the result `inconclusive`. Every reason is
listed at once. So `improved` never follows from a shorter recording or fewer pauses: they are not
in the rule. The audio model's own limitations and uncertainties of each attempt are carried into
`limitations`; they never change the verdict.

The word of a criterion is looked up in the raw transcript (`asr/raw.json`, only the segments
`asr.trusted_segments` trusts), never in the model's text. The result is checked against
`compare.v1.schema.json` and written to `<data_root>/compare/<compare_id>.json` (0600,
exclusive). Nothing is run: no lock, no model, no backend, no network.
"""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from itertools import pairwise
from pathlib import Path

from sound_scout import asr, findings, ingest, paths, report, review, runs, schema
from sound_scout._schema_messages import schema_errors
from sound_scout.config import Config

SCHEMA_VERSION = "1.0"
SCHEMA_FILE_NAME = "compare.v1.schema.json"
SCHEMA_BYTES = files("sound_scout").joinpath(SCHEMA_FILE_NAME).read_bytes()
SCHEMA: dict = json.loads(SCHEMA_BYTES)
COMPARE_DIR_NAME = "compare"
DIR_MODE = 0o700
CRITERIA = {
    "emphasis_on": "semantic_stress",
    "pause_before": "pause_phrasing",
    "intelligibility_of": "pronunciation",
    "completeness_of_phrase": "completeness",
}
WHOLE_RECORDING = "completeness_of_phrase"
MAX_WORD_LENGTH = 64
LEVEL_DIFF_LIMIT_DB = 6.0

CRITERION_INVALID = "criterion_invalid"
RUN_NOT_FOUND = "run_not_found"
RUN_INCOMPLETE = "run_incomplete"
SAME_RUN = "same_run"
RESULT_INVALID = "result_invalid"

GAIN_REASON = "gain и микрофон записи инструменту не известны; косвенно — по различию уровня"
PAUSES_NOTE = "в вердикт не входит"
NO_TASK_LIMITATION = "задача ни для одной попытки не указана — совпадение задания не проверено"
LIMITATIONS = (
    "Более короткая запись или меньшее число пауз не считается улучшением: длительность и паузы "
    "— условия, вердикт по ним не выводится. Паузы во второй попытке могут быть связаны с новым "
    "вопросом или обдумыванием нового содержания",
    "Один удачный контроль не доказывает устойчивый навык: различайте исправление после "
    "подсказки, сохранение позже и перенос в другой контекст",
    "Отсутствие замечания аудиомодели во второй попытке — не доказательство правильности; "
    "not_assessed не равно «всё хорошо»",
    f"gain и микрофон неизвестны: различие уровня больше {LEVEL_DIFF_LIMIT_DB:g} dB делает "
    "сравнение недостоверным, меньшее различие условий записи не исключено",
)
_EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$")
_FINDING_FIELDS = (
    "id",
    "category",
    "polarity",
    "start",
    "end",
    "observation",
    "evidence_basis",
    "evidence_refs",
)
_UNLOCALIZED_FIELDS = (
    "id",
    "category",
    "observation",
    "reason",
    "correction",
    "evidence_basis",
    "evidence_refs",
)


class CompareError(Exception):
    """The comparison is refused before anything is written; `code` names why."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Criterion:
    raw: str
    kind: str
    word: str | None
    category: str


@dataclass(frozen=True)
class Attempt:
    """One complete, validated run; a field read from a file that is not there is None."""

    run_id: str
    result: dict
    result_sha256: str
    task_context: dict | None
    source_record: dict | None
    words: list[tuple[float, float, str]] | None


@dataclass(frozen=True)
class Comparison:
    result: dict
    path: Path


def parse_criterion(text: object) -> Criterion:
    """One of the four fixed forms, or `CompareError` (`criterion_invalid`)."""
    if not isinstance(text, str):
        raise CompareError(CRITERION_INVALID, "the criterion must be a string")
    raw = text.strip()
    kind, separator, word = raw.partition(":")
    forms = ", ".join(f"{name}:<слово>" for name in CRITERIA if name != WHOLE_RECORDING)
    if kind not in CRITERIA:
        raise CompareError(
            CRITERION_INVALID, f"unknown criterion; known: {forms}, {WHOLE_RECORDING}"
        )
    if kind == WHOLE_RECORDING:
        if separator:
            raise CompareError(CRITERION_INVALID, f"{WHOLE_RECORDING} takes no word")
        return Criterion(raw, kind, None, CRITERIA[kind])
    word = word.strip()
    if not 1 <= len(word) <= MAX_WORD_LENGTH or any(char.isspace() for char in word):
        raise CompareError(
            CRITERION_INVALID,
            f"{kind} takes one word of 1..{MAX_WORD_LENGTH} characters without spaces",
        )
    if not normalized_word(word):
        raise CompareError(CRITERION_INVALID, f"the word of {kind} has no letters or digits")
    return Criterion(f"{kind}:{word}", kind, word, CRITERIA[kind])


def normalized_word(text: str) -> str:
    """`casefold`, `ё→е`, punctuation at the edges dropped: how criterion and ASR words match."""
    return _EDGE_PUNCTUATION.sub("", text.casefold().replace("ё", "е"))


def load_attempt(run_id: object, config: Config) -> Attempt:
    """A complete run under `<data_root>/runs/` whose `result.json` passes `findings.validate`."""
    if not isinstance(run_id, str) or not runs.RUN_ID_PATTERN.match(run_id):
        raise CompareError(RUN_NOT_FOUND, "a run_id is 32 lowercase hex characters")
    runs_dir = config.data_root / runs.RUNS_DIR_NAME
    run_dir = runs_dir / run_id
    if not (_is_real_dir(runs_dir) and _is_real_dir(run_dir)):
        raise CompareError(RUN_NOT_FOUND, f"run {run_id} is not in the data root")
    manifest = _read_json(run_dir / runs.MANIFEST_FILE_NAME)
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise CompareError(RUN_INCOMPLETE, f"run {run_id} did not complete")
    result_bytes = _read_bytes(run_dir / report.RESULT_FILE_NAME)
    if result_bytes is None:
        raise CompareError(RUN_INCOMPLETE, f"run {run_id} has no {report.RESULT_FILE_NAME}")
    request = _read_json(run_dir / runs.REQUEST_FILE_NAME)
    request = request if isinstance(request, dict) else {}
    try:
        document = json.loads(result_bytes)
    except ValueError:
        document = None
    run = runs.Run(
        run_id,
        run_dir,
        str(request.get("request_id")),
        str(request.get("identity_hash")),
        reused=False,
        complete=True,
        created=False,
    )
    errors = ["$: not JSON"] if document is None else findings.validation_errors(document, run=run)
    if not errors and document["run_id"] != run_id:
        errors = ["$.run_id: names another run"]
    if errors:
        # the count only: no text of the document gets into the message
        raise CompareError(
            RESULT_INVALID,
            f"{report.RESULT_FILE_NAME} of run {run_id} does not match the contract "
            f"({len(errors)} violations)",
        )
    task_context = request.get("task_context")
    source_record = _read_json(run_dir / ingest.SOURCE_FILE_NAME)
    return Attempt(
        run_id,
        document,
        hashlib.sha256(result_bytes).hexdigest(),
        task_context if isinstance(task_context, dict) else None,
        source_record if isinstance(source_record, dict) else None,
        _words(_read_json(run_dir / asr.ASR_DIR_NAME / asr.RAW_FILE_NAME)),
    )


def compare(run_a: object, run_b: object, criterion_text: object, config: Config) -> Comparison:
    """Compare attempt A (initial) with B (control) by one criterion and write the result."""
    criterion = parse_criterion(criterion_text)
    if run_a == run_b:
        raise CompareError(SAME_RUN, "the two attempts are the same run")
    a, b = load_attempt(run_a, config), load_attempt(run_b, config)
    result = build(criterion, a, b)
    errors = schema._non_finite_numbers(SCHEMA, result) + schema_errors(SCHEMA, result)
    if errors:
        raise ValueError("comparison does not match compare.v1:\n" + "\n".join(errors))
    compare_dir = config.data_root / COMPARE_DIR_NAME
    _ensure_private_dir(compare_dir)
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    name = f"{result['compare_id']}.json"
    with paths.open_output(compare_dir, name, config) as out:
        out.write(text.encode("utf-8"))
    return Comparison(result, compare_dir / name)


def build(criterion: Criterion, a: Attempt, b: Attempt) -> dict:
    """The comparison document; nothing is read or written here."""
    sides = {"a": _side(criterion, a), "b": _side(criterion, b)}
    conditions = _conditions(a, b)
    reasons = _reasons(criterion, sides, conditions, a, b)
    if reasons:
        verdict = "inconclusive"
    elif _has_issue(criterion, sides["b"]["evidence"]):
        verdict = "not_improved"
    else:
        verdict = "improved"
    synthetic = any(attempt.result["provenance"]["synthetic"] for attempt in (a, b))
    return {
        "schema_version": SCHEMA_VERSION,
        "compare_id": runs.new_run_id(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "criterion": {
            "raw": criterion.raw,
            "kind": criterion.kind,
            "word": criterion.word,
            "category": criterion.category,
        },
        "verdict": verdict,
        "reasons": reasons,
        "attempts": {
            label: {key: value for key, value in side.items() if not key.startswith("_")}
            for label, side in sides.items()
        },
        "conditions_diff": conditions,
        "limitations": _limitations(criterion, sides, conditions, a, b),
        "synthetic": synthetic,
    }


# --- one attempt -----------------------------------------------------------------------------


def _side(criterion: Criterion, attempt: Attempt) -> dict:
    result = attempt.result
    review_stage = result["stages"]["audio_review"]
    review_completed = review_stage["status"] == "completed"
    windows = _windows(criterion, attempt)
    reviewed = (
        review_completed
        and bool(windows)
        and all(_inside(window, result["coverage"]["reviewed"]) for window in windows)
    )
    found = [] if windows is None else windows
    findings_in = [
        {key: item[key] for key in _FINDING_FIELDS}
        for item in result["findings"]
        if item["category"] == criterion.category
        and (criterion.kind == WHOLE_RECORDING or any(_overlaps(item, w) for w in found))
    ]
    unlocalized = [
        {key: item[key] for key in _UNLOCALIZED_FIELDS}
        for item in result["unlocalized"]
        if item["category"] == criterion.category
    ]
    measurements = []
    if criterion.kind == "pause_before":
        measurements = [
            {"name": "speech_spans", **span, "evidence_basis": "measurement"}
            for span in _speech_spans(result) or []
            if any(_overlaps(span, window) for window in found)
        ]
    return {
        "run_id": attempt.run_id,
        "request_id": result["request_id"],
        "result_sha256": attempt.result_sha256,
        "status": result["status"],
        "audio_review": {"status": review_stage["status"], "reason": review_stage["reason"]},
        "windows": found,
        "reviewed": reviewed,
        "word_occurrences": None if criterion.word is None else len(found),
        "evidence": {
            "findings": findings_in,
            "unlocalized": unlocalized,
            "measurements": measurements,
        },
        "_review_completed": review_completed,
        "_word_missing": criterion.word is not None and not found,
    }


def _windows(criterion: Criterion, attempt: Attempt) -> list[dict] | None:
    """Windows on the source scale; None when there is no transcript to look the word up in."""
    if criterion.word is None:
        return [dict(item) for item in attempt.result["coverage"]["requested"]]
    if attempt.words is None:
        return None
    target = normalized_word(criterion.word)
    windows = []
    for index, (start, end, word) in enumerate(attempt.words):
        if word != target:
            continue
        if criterion.kind == "pause_before":
            start = attempt.words[index - 1][1] if index else 0.0
        windows.append({"start": start, "end": end})
    return windows


def _words(raw: object) -> list[tuple[float, float, str]] | None:
    try:
        trusted = asr.trusted_segments(raw)[0]
    except ValueError:
        return None
    words = [
        (float(word["start"]), float(word["end"]), normalized_word(word["word"]))
        for segment in trusted
        for word in segment.get("words") or []
        if isinstance(word, dict)
        and isinstance(word.get("word"), str)
        and _is_number(word.get("start"))
        and _is_number(word.get("end"))
    ]
    return sorted(words)


def _has_issue(criterion: Criterion, evidence: dict) -> bool:
    if any(item["polarity"] == "negative" for item in evidence["findings"]):
        return True
    return criterion.kind == WHOLE_RECORDING and _corrected_unlocalized(evidence)


def _corrected_unlocalized(evidence: dict) -> bool:
    return any(item["correction"] is not None for item in evidence["unlocalized"])


# --- conditions ------------------------------------------------------------------------------


def _conditions(a: Attempt, b: Attempt) -> dict:
    duration = [attempt.result["source"]["duration_s"] for attempt in (a, b)]
    level = [_level(attempt.result) for attempt in (a, b)]
    level_delta = None if None in level else round(level[1] - level[0], 6)
    tasks = [_context_text(attempt, "task") for attempt in (a, b)]
    intents = [_context_text(attempt, "intent") for attempt in (a, b)]
    return {
        "duration_s": {
            "a": duration[0],
            "b": duration[1],
            "delta": round(duration[1] - duration[0], 6),
            "source": "result.source.duration_s",
        },
        "pauses": {
            "a": _pauses(a.result),
            "b": _pauses(b.result),
            "source": "result.measurements.speech_spans",
            "note": PAUSES_NOTE,
        },
        "level_dbfs": {
            "a": level[0],
            "b": level[1],
            "delta": level_delta,
            "limit_db": LEVEL_DIFF_LIMIT_DB,
            "exceeds": None if level_delta is None else abs(level_delta) > LEVEL_DIFF_LIMIT_DB,
            "source": "result.measurements.rms_dbfs.value.p50",
        },
        "channel": _pair(
            [_channel(a.result), _channel(b.result)],
            "result.source.channels, result.provenance.transformations.channel_choice",
        ),
        "format": _pair([_format(a), _format(b)], "source.json"),
        "gain": {"a": None, "b": None, "known": False, "reason": GAIN_REASON},
        "task": {
            "a": tasks[0],
            "b": tasks[1],
            "known": [a.task_context is not None, b.task_context is not None],
            "differs": None if None in (a.task_context, b.task_context) else tasks[0] != tasks[1],
            "source": "request.json.task_context.task",
        },
        "intent": {
            "a": intents[0],
            "b": intents[1],
            "known": [a.task_context is not None, b.task_context is not None],
            "differs": None
            if None in (a.task_context, b.task_context)
            else intents[0] != intents[1],
            "source": "request.json.task_context.intent",
        },
        "backend": _pair([_backend(a.result), _backend(b.result)], "result.provenance"),
    }


def _pair(values: list[dict | None], source: str) -> dict:
    differs = None if None in values else values[0] != values[1]
    return {"a": values[0], "b": values[1], "differs": differs, "source": source}


def _level(result: dict) -> float | None:
    for measurement in result["measurements"]:
        if measurement["name"] == "rms_dbfs" and isinstance(measurement["value"], dict):
            p50 = measurement["value"].get("p50")
            return float(p50) if _is_number(p50) else None
    return None


def _speech_spans(result: dict) -> list[dict] | None:
    for measurement in result["measurements"]:
        if measurement["name"] == "speech_spans" and isinstance(measurement["value"], list):
            return [{"start": span["start"], "end": span["end"]} for span in measurement["value"]]
    return None


def _pauses(result: dict) -> dict | None:
    spans = _speech_spans(result)
    if spans is None:
        return None
    gaps = [max(0.0, nxt["start"] - prev["end"]) for prev, nxt in pairwise(spans)]
    return {"speech_spans": len(spans), "pause_total_s": round(sum(gaps), 6)}


def _channel(result: dict) -> dict | None:
    for transformation in result["provenance"]["transformations"]:
        if transformation.get("name") == "channel_choice":
            settings = transformation.get("settings") or {}
            return {
                "source_channels": result["source"]["channels"],
                "method": settings.get("method"),
                "selected_channel": settings.get("selected_channel"),
            }
    return None


def _format(attempt: Attempt) -> dict | None:
    record = attempt.source_record
    if record is None:
        return None
    return {key: record.get(key) for key in ("container", "codec", "sample_rate")}


def _backend(result: dict) -> dict:
    provenance = result["provenance"]
    return {
        key: provenance[key]
        for key in ("backend", "model", "revision", "prompt_sha256", "synthetic")
    }


def _context_text(attempt: Attempt, key: str) -> str | None:
    text = (attempt.task_context or {}).get(key)
    return " ".join(text.split()) if isinstance(text, str) else None


# --- the rule --------------------------------------------------------------------------------


def _reasons(
    criterion: Criterion, sides: dict, conditions: dict, a: Attempt, b: Attempt
) -> list[dict]:
    reasons: list[dict] = []

    def add(code: str, attempt: str | None, message: str) -> None:
        reasons.append({"code": code, "attempt": attempt, "message": message})

    for label, side in sides.items():
        review = side["audio_review"]
        if not side["_review_completed"]:
            add(
                "audio_review_missing",
                label,
                f"попытка {label.upper()}: audio review не выполнен "
                f"({review['status']}: {review['reason']})",
            )
        if side["_word_missing"]:
            add(
                "word_not_found",
                label,
                f"попытка {label.upper()}: слово критерия не найдено в сыром транскрипте "
                "(или ASR не выполнен)",
            )
        elif side["_review_completed"] and not side["reviewed"]:
            add(
                "criterion_not_reviewed",
                label,
                f"попытка {label.upper()}: окно критерия не входит в прослушанный диапазон",
            )
    level = conditions["level_dbfs"]
    for label in ("a", "b"):
        if level[label] is None:
            add("level_unknown", label, f"попытка {label.upper()}: уровень речи не измерен")
    if level["exceeds"]:
        add(
            "level_differs",
            None,
            f"уровень речи различается на {abs(level['delta']):.1f} dB "
            f"(больше {LEVEL_DIFF_LIMIT_DB:g} dB)",
        )
    for name, what in (("channel", "канал"), ("format", "формат записи")):
        entry = conditions[name]
        for label in ("a", "b"):
            if entry[label] is None:
                add(f"{name}_unknown", label, f"попытка {label.upper()}: {what} неизвестен")
        if entry["differs"]:
            add(f"{name}_differs", None, f"{what} попыток различается")
    for label, attempt in (("a", a), ("b", b)):
        if attempt.task_context is None:
            add(
                "task_unknown",
                label,
                f"попытка {label.upper()}: задача не записана в request.json (старый run)",
            )
    if conditions["task"]["differs"]:
        add("task_differs", None, "задачи попыток различаются")
    if conditions["intent"]["differs"]:
        add("intent_differs", None, "замысел (intent) попыток различается")
    if conditions["backend"]["differs"]:
        add("backend_differs", None, "audio backend, модель, ревизия или промпт различаются")
    if a.result["source"]["sha256"] == b.result["source"]["sha256"]:
        add("same_recording", None, "обе попытки — одна и та же запись")
    side_a, side_b = sides["a"], sides["b"]
    if (
        criterion.word is not None
        and side_b["_review_completed"]
        and _corrected_unlocalized(side_b["evidence"])
    ):
        add(
            "unlocalized_observation",
            "b",
            "попытка B: у аудиомодели есть замечание этой категории без таймкодов — "
            "нельзя исключить ту же проблему",
        )
    if side_a["reviewed"] and not _has_issue(criterion, side_a["evidence"]):
        add(
            "no_baseline_issue",
            "a",
            "в попытке A нет замечания по критерию — показывать улучшение не на чем",
        )
    return reasons


def _limitations(
    criterion: Criterion, sides: dict, conditions: dict, a: Attempt, b: Attempt
) -> list[str]:
    limitations = [f"Сравнение только по критерию {criterion.raw}; это не общая оценка речи"]
    limitations += LIMITATIONS
    provenances = [attempt.result["provenance"] for attempt in (a, b)]
    limitations += sorted({item["note"] for item in provenances if item["synthetic"]})
    task = conditions["task"]
    if task["a"] is None and task["b"] is None:
        limitations.append(NO_TASK_LIMITATION)
    counts = [sides[label]["word_occurrences"] or 0 for label in ("a", "b")]
    if max(counts) > 1:
        limitations.append(
            f"слово критерия встречается больше одного раза (A: {counts[0]}, B: {counts[1]}); "
            "сопоставлены все вхождения"
        )
    for label, attempt in (("a", a), ("b", b)):
        limitations += [
            f"попытка {label.upper()}: {text}"
            for text in attempt.result["provenance"]["limitations"]
            if text.startswith((review.MODEL_LIMITATION, review.MODEL_UNCERTAINTY))
        ]
    return limitations


# --- files -----------------------------------------------------------------------------------


def _inside(window: dict, ranges: list[dict]) -> bool:
    return any(
        item["start"] - schema.TIME_EPS <= window["start"]
        and window["end"] <= item["end"] + schema.TIME_EPS
        for item in ranges
    )


def _overlaps(item: dict, window: dict) -> bool:
    return item["start"] < window["end"] and item["end"] > window["start"]


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_real_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _ensure_private_dir(path: Path) -> None:
    try:
        os.mkdir(path, DIR_MODE)
    except FileExistsError:
        if not _is_real_dir(path):
            raise FileExistsError(f"{path} is not a directory or is a symlink") from None
    else:
        os.chmod(path, DIR_MODE)


def _read_bytes(path: Path) -> bytes | None:
    """The bytes of a regular file; None when it is missing, a symlink or not a file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as file:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        return file.read()


def _read_json(path: Path) -> object:
    data = _read_bytes(path)
    if data is None:
        return None
    try:
        return json.loads(data)
    except ValueError:
        return None
