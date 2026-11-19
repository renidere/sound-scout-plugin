"""Evaluation runner of Gate 1 (docs/eval/PROTOCOL.md § 6–10; T04–T06, R01, R05, R11).

`evaluate` reads what already exists — the corpus manifest, the human labels of every clip and
the ready `analyze` results — and counts; it never calls a backend, a model or anything else
that infers. Results are found per clip as `<results>/<clip id>/result.json`, the run directory
of that clip's analysis (or a copy of it), so the artifacts a result cites resolve relative to
it. Labels are `<labels>/<clip id>.json` as docs/eval/LABELING.md describes. A result counts only
when it is `complete` with its ASR and audio review completed, passes the contract and the check
of its run files (`findings.validation_errors` with the run), and was made on the clip of the
manifest; all results of a run must share one backend, model, revision, prompt and schema.
Anything else disqualifies the run rather than being scored.

Matching (§ 6): a finding is a candidate for an event of the same category when it is no longer
than the category's maximum, does not leave the event by more than the tolerance on either side
and meets the category's hit condition. Whether the finding agrees with what the labeller heard
is a person's judgement (§ 6: the labels carry free text): it comes from the agreements file,
one `agrees` per candidate pair of a `clear` event, bound to the sha256 of the result and of the
labels it was made on. A candidate without a current judgement never counts as a match; it is
listed as pending and the run is NOT QUALIFIED until a person judged it. Only findings with
`evidence_basis: audio_model_observation` are scored; any other basis on a scored clip
disqualifies. Pairs are assigned one to one, the largest overlap first. A finding matched to an
`uncertain` event is neither TP nor FP; an `uncertain` event is outside recall.

Negative controls (§ 9) are counted on every control of the manifest, both splits, by the kind
its manifest record names. On every kind: each finding of the five categories (the adapter carries
the backend's own uncertainty into `uncertainties` and `unlocalized`, never into `findings`, so a
finding is a confident claim) and each transcript correction with `certainty: clear`. On
`silence` and `noise` also each segment of the raw transcript with text that is not flagged
`suspect_hallucination` (`asr.trusted_segments`, the one rule of docs/decisions/0005); on
`corrupted_asr` the speech is real, so its raw transcript words are not a fabrication. A
control without a kind (`manifest.validate` rejects it) is counted as silence: it fails closed.
A `corrupted_asr` candidate whose ASR error no person confirmed (`manifest.is_unconfirmed`) is no
control: it is neither loaded nor counted, and the manifest error it carries disqualifies the run.

The thresholds below mirror the protocol's tables; a test asserts they match its text. They are
the owner's to change in the protocol, never here.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath

from sound_scout import asr, findings
from sound_scout._schema_messages import schema_errors
from sound_scout.doctor import TEST_DOUBLE
from sound_scout.eval import labels, manifest
from sound_scout.eval.manifest import CATEGORIES, CATEGORY_GROUPS, MIN_HELDOUT_EVENTS_PER_GROUP
from sound_scout.runs import Run
from sound_scout.schema import _non_finite_numbers

REPORT_VERSION = "1.0"
RESULT_FILE_NAME = "result.json"
DEFAULT_OUTPUT_DIR = Path("docs/eval")
PRECISION_MIN = 0.80
RECALL_MIN = 0.70
MAX_CONFIDENT_FABRICATIONS = 0
MIN_OVERLAP_SHARE = 0.5
END_TOLERANCE_S = 0.5
HEARING_BASIS = "audio_model_observation"
REQUIRED_STAGES = ("asr", "audio_review")
_EPSILON = 1e-9
SMALL_SAMPLE_WARNING = (
    "Малый пилот, не статистическое доказательство качества для всех людей, микрофонов и "
    "ситуаций (T06): числа получены на нескольких десятках клипов, пороги — предложение "
    "владельцу, не нормы."
)


@dataclass(frozen=True)
class MatchRule:
    """§ 6 per category: `hit` is `overlap` (≥ 50 % of the event inside the finding),
    `midpoint` (the finding holds the event's middle) or `end` (the finding holds a moment
    within ± 0.5 s of the event's end)."""

    hit: str
    tolerance_s: float
    max_length_s: float


MATCH_RULES = {
    "pronunciation": MatchRule("overlap", 0.5, 3.0),
    "repetition_self_correction": MatchRule("overlap", 1.0, 6.0),
    "semantic_stress": MatchRule("overlap", 1.5, 6.0),
    "pause_phrasing": MatchRule("midpoint", 1.5, 6.0),
    "completeness": MatchRule("end", 2.0, 8.0),
}
GROUP_OF = {name: group for group, names in CATEGORY_GROUPS.items() for name in names}


class EvalError(Exception):
    """The run cannot produce a report at all (unreadable manifest or agreements, report exists)."""


@dataclass(frozen=True)
class _Judgement:
    agrees: bool
    result_sha256: str
    labels_sha256: str


@dataclass
class _Clip:
    record: dict
    events: list[dict]
    result: dict | None
    result_sha256: str | None = None
    labels_sha256: str | None = None


def evaluate(
    manifest_path: str | Path,
    labels_dir: str | Path,
    results_dir: str | Path,
    split: str,
    *,
    agreements_path: str | Path | None = None,
    today: date | None = None,
) -> dict:
    """The report document of one run over `split`; nothing is written."""
    if split not in manifest.SPLITS:
        raise EvalError(f"split must be one of {list(manifest.SPLITS)}, got {split!r}")
    manifest_bytes, records = _load_manifest(Path(manifest_path))
    agreements = _load_agreements(agreements_path)
    reasons: list[str] = []
    if split != "heldout":
        reasons.append("прогон на dev-наборе: квалификация только на held-out (§ 5)")
    manifest_errors = manifest.validation_report(
        json.loads(manifest_bytes), Path(manifest_path).parent
    ).errors
    reasons += [f"манифест: {error}" for error in manifest_errors]
    input_errors: list[str] = []
    evaluated = sorted(
        (
            r
            for r in records
            if (r["split"] == split or r["negative_control"]) and not manifest.is_unconfirmed(r)
        ),
        key=lambda r: r["id"],
    )
    clips = [_load_clip(r, Path(labels_dir), Path(results_dir), input_errors) for r in evaluated]
    reasons += input_errors
    scored = [clip for clip in clips if clip.record["split"] == split]
    controls = [clip for clip in clips if clip.record["negative_control"]]

    counts = {name: _empty_counts() for name in CATEGORIES}
    failures: dict[str, list[dict]] = {"missed_events": [], "false_findings": []}
    pending: list[dict] = []
    used_judgements: set[tuple[str, str, int]] = set()
    abstention = {group: {"clips": 0, "of": 0} for group in CATEGORY_GROUPS}
    for clip in scored:
        _score_clip(clip, agreements, counts, failures, pending, used_judgements)
        for group, names in CATEGORY_GROUPS.items():
            if any(name in names for name in clip.record["categories"]):
                abstention[group]["of"] += 1
                abstention[group]["clips"] += _group_abstained(clip.result or {}, group)
    unused = sorted(set(agreements) - used_judgements)
    reasons += [
        f"согласие: клип {c!r}, finding {f!r}, событие {e} — не пара-кандидат этого прогона"
        for c, f, e in unused
    ]
    reasons += [
        f"согласие: клип {c!r}, finding {f!r}, событие {e} — решение принято о другом результате "
        "или другой разметке (sha256), его нужно принять заново"
        for c, f, e in sorted({(p["clip_id"], p["finding_id"], p["event"]) for p in pending})
        if (c, f, e) in agreements
    ]
    other_basis = sorted(
        (clip.record["id"], f["id"])
        for clip in scored
        for f in (clip.result or {}).get("findings", [])
        if f["category"] in MATCH_RULES and f["evidence_basis"] != HEARING_BASIS
    )
    reasons += [
        f"результат клипа {c!r}: finding {f!r} основан не на прослушивании "
        f"(evidence_basis не {HEARING_BASIS}) и не засчитывается"
        for c, f in other_basis
    ]
    if pending:
        reasons.append(
            f"{len(pending)} пар-кандидатов ждут решения человека о согласии наблюдений (§ 6)"
        )
    fabrications, fabrication_list = _control_fabrications(controls, Path(results_dir), reasons)
    failures["control_fabrications"] = fabrication_list
    doubles = sorted(c.record["id"] for c in clips if c.result and _is_test_double(c.result))
    if doubles:
        reasons.append(
            f"{len(doubles)} результатов получены test-double/synthetic backend: мок проверяет "
            "контракт, а не способность слышать (R01, T03)"
        )
    settings = _settings(clips)
    if len(settings) > 1:
        reasons.append(
            f"результаты получены {len(settings)} разными настройками (backend, модель, ревизия, "
            "промпт, схема): квалификация — одних настроек"
        )

    groups = {
        group: _group_summary(names, counts, abstention[group], fabrications)
        for group, names in CATEGORY_GROUPS.items()
    }
    statuses = qualification({g: s["passed"] for g, s in groups.items()}, reasons)
    return {
        "report_version": REPORT_VERSION,
        "date": (today or date.today()).isoformat(),
        "split": split,
        "manifest": {
            "name": json.loads(manifest_bytes)["name"],
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "protocol": {
            "precision_min": PRECISION_MIN,
            "recall_min": RECALL_MIN,
            "max_confident_fabrications": MAX_CONFIDENT_FABRICATIONS,
            "min_heldout_clear_events_per_group": MIN_HELDOUT_EVENTS_PER_GROUP,
            "match_rules": {
                name: {
                    "hit": rule.hit,
                    "tolerance_s": rule.tolerance_s,
                    "max_length_s": rule.max_length_s,
                }
                for name, rule in MATCH_RULES.items()
            },
        },
        "warning": SMALL_SAMPLE_WARNING,
        "clips": {
            "scored": len(scored),
            "synthetic": sum(1 for clip in scored if clip.record.get("synthetic")),
            "negative_controls": len(controls),
        },
        "categories": counts,
        "groups": groups,
        "negative_controls": {"clips": len(controls), "confident_fabrications": fabrications},
        "failures": failures,
        "pending_agreement": pending,
        "settings": settings,
        "not_qualified_reasons": reasons,
        **statuses,
    }


def qualification(passed: dict[str, bool], reasons: list[str]) -> dict[str, bool]:
    """§ 10: a group is validated when it passed § 7 and § 9 and nothing disqualifies the run."""
    diction = passed["diction"] and not reasons
    prosody = passed["prosody"] and not reasons
    return {
        "diction_validated_ru": diction,
        "prosody_validated_ru": prosody,
        "validated_ru": diction and prosody,
    }


def candidate(category: str, finding: dict, event: dict) -> bool:
    """§ 6 interval rules: length, tolerance on both sides and the category's hit condition."""
    rule = MATCH_RULES[category]
    f_start, f_end, e_start, e_end = finding["start"], finding["end"], event["start"], event["end"]
    if f_end - f_start > rule.max_length_s + _EPSILON:
        return False
    slack = rule.tolerance_s + _EPSILON
    if f_start < e_start - slack or f_end > e_end + slack:
        return False
    if rule.hit == "overlap":
        return _overlap(finding, event) >= MIN_OVERLAP_SHARE * (e_end - e_start) - _EPSILON
    if rule.hit == "midpoint":
        middle = (e_start + e_end) / 2
        return f_start - _EPSILON <= middle <= f_end + _EPSILON
    window = END_TOLERANCE_S + _EPSILON
    return f_start <= e_end + window and f_end >= e_end - window


def _overlap(finding: dict, event: dict) -> float:
    return max(0.0, min(finding["end"], event["end"]) - max(finding["start"], event["start"]))


def _empty_counts() -> dict:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "events": 0,
        "clear_events": 0,
        "uncertain_events": 0,
        "uncertain_matches": 0,
        "clips": 0,
        "abstention_clips": 0,
        "unlocalized": 0,
    }


def _score_clip(
    clip: _Clip,
    agreements: dict[tuple[str, str, int], _Judgement],
    counts: dict,
    failures: dict[str, list[dict]],
    pending: list[dict],
    used: set[tuple[str, str, int]],
) -> None:
    clip_id = clip.record["id"]
    result = clip.result or {}
    all_findings = sorted(
        (
            f
            for f in result.get("findings", [])
            if f["category"] in MATCH_RULES and f["evidence_basis"] == HEARING_BASIS
        ),
        key=lambda f: (f["start"], f["end"], f["id"]),
    )
    for item in result.get("unlocalized", []):
        if item["category"] in counts:
            counts[item["category"]]["unlocalized"] += 1
    for name in clip.record["categories"]:
        counts[name]["clips"] += 1
        if _abstained(result, name):
            counts[name]["abstention_clips"] += 1
    for name in CATEGORIES:
        events = [(i, e) for i, e in enumerate(clip.events) if e["category"] == name]
        found = [f for f in all_findings if f["category"] == name]
        tally = counts[name]
        tally["events"] += len(events)
        tally["clear_events"] += sum(1 for _, e in events if e["certainty"] == "clear")
        tally["uncertain_events"] += sum(1 for _, e in events if e["certainty"] == "uncertain")
        eligible = []
        disagreed: set[str] = set()
        waiting: set[str] = set()
        for finding in found:
            for index, event in events:
                if not candidate(name, finding, event):
                    continue
                key = (clip_id, finding["id"], index)
                if event["certainty"] == "uncertain":
                    if key in agreements:
                        used.add(key)
                    eligible.append((finding, index, event))
                    continue
                judgement = agreements.get(key)
                if judgement is not None:
                    used.add(key)
                if judgement is None or (judgement.result_sha256, judgement.labels_sha256) != (
                    clip.result_sha256,
                    clip.labels_sha256,
                ):
                    waiting.add(finding["id"])
                    pending.append(
                        {
                            "clip_id": clip_id,
                            "finding_id": finding["id"],
                            "event": index,
                            "category": name,
                        }
                    )
                    continue
                if judgement.agrees:
                    eligible.append((finding, index, event))
                else:
                    disagreed.add(finding["id"])
        eligible.sort(key=lambda c: (-_overlap(c[0], c[2]), c[0]["start"], c[0]["id"], c[1]))
        matched_findings: dict[str, dict] = {}
        matched_events: set[int] = set()
        for finding, index, event in eligible:
            if finding["id"] in matched_findings or index in matched_events:
                continue
            matched_findings[finding["id"]] = event
            matched_events.add(index)
        for finding in found:
            event = matched_findings.get(finding["id"])
            if event is None:
                tally["fp"] += 1
                failures["false_findings"].append(
                    {
                        "clip_id": clip_id,
                        "finding_id": finding["id"],
                        "category": name,
                        "start": finding["start"],
                        "end": finding["end"],
                        "why": _why_false(name, finding, events, disagreed, waiting),
                    }
                )
            elif event["certainty"] == "clear":
                tally["tp"] += 1
            else:
                tally["uncertain_matches"] += 1
        for index, event in events:
            if event["certainty"] == "clear" and index not in matched_events:
                tally["fn"] += 1
                failures["missed_events"].append(
                    {
                        "clip_id": clip_id,
                        "event": index,
                        "category": name,
                        "start": event["start"],
                        "end": event["end"],
                    }
                )


def _why_false(
    name: str, finding: dict, events: list, disagreed: set[str], waiting: set[str]
) -> str:
    if finding["id"] in waiting:
        return "agreement_pending"
    if finding["id"] in disagreed:
        return "observation_disagrees"
    if not events:
        return "no_event"
    if finding["end"] - finding["start"] > MATCH_RULES[name].max_length_s + _EPSILON:
        return "too_long"
    if any(candidate(name, finding, event) for _, event in events):
        return "event_already_matched"
    return "outside_tolerance"


def _abstained(result: dict, category: str) -> bool:
    """Per category: the group is `not_assessed`, or the clip has no finding of `category`."""
    if _status(result, GROUP_OF[category]) != "assessed":
        return True
    return not any(f["category"] == category for f in result.get("findings", []))


def _group_abstained(result: dict, group: str) -> bool:
    """§ 8: the group is `not_assessed`, or its `findings` are empty while `assessed`."""
    if _status(result, group) != "assessed":
        return True
    names = CATEGORY_GROUPS[group]
    return not any(f["category"] in names for f in result.get("findings", []))


def _status(result: dict, group: str) -> str | None:
    return result.get("assessment_status", {}).get(group, {}).get("status")


def _group_summary(
    names: tuple[str, ...], counts: dict, abstention: dict[str, int], fabrications: dict
) -> dict:
    tp = sum(counts[name]["tp"] for name in names)
    fp = sum(counts[name]["fp"] for name in names)
    fn = sum(counts[name]["fn"] for name in names)
    clear = sum(counts[name]["clear_events"] for name in names)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    failed = []
    if precision is None:
        failed.append("нет ни одного замечания группы: precision не определена (§ 7)")
    elif precision < PRECISION_MIN:
        failed.append(f"precision {precision:.2f} < {PRECISION_MIN:.2f}")
    if recall is None:
        failed.append("нет отчётливых событий группы: recall не определён")
    elif recall < RECALL_MIN:
        failed.append(f"recall {recall:.2f} < {RECALL_MIN:.2f}")
    failed += [
        f"категория {name!r} без отчётливых событий: не представлена (§ 7)"
        for name in names
        if counts[name]["clear_events"] == 0
    ]
    if clear < MIN_HELDOUT_EVENTS_PER_GROUP:
        failed.append(f"{clear} отчётливых событий, нужно не меньше {MIN_HELDOUT_EVENTS_PER_GROUP}")
    total = sum(fabrications.values())
    if total > MAX_CONFIDENT_FABRICATIONS:
        failed.append(f"{total} уверенных выдумок на отрицательных контролях (§ 9)")
    return {
        "categories": list(names),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "clear_events": clear,
        "precision": precision,
        "recall": recall,
        "abstention": {
            **abstention,
            "share": abstention["clips"] / abstention["of"] if abstention["of"] else None,
        },
        "passed": not failed,
        "failed": failed,
    }


def _control_fabrications(
    controls: list[_Clip], results_dir: Path, reasons: list[str]
) -> tuple[dict[str, int], list[dict]]:
    counts = {"diction": 0, "prosody": 0, "transcript": 0}
    listed: list[dict] = []
    for clip in controls:
        if clip.result is None:
            continue
        clip_id = clip.record["id"]
        for finding in sorted(clip.result["findings"], key=lambda f: f["id"]):
            if finding["category"] in GROUP_OF:
                counts[GROUP_OF[finding["category"]]] += 1
                listed.append({"clip_id": clip_id, "kind": "finding", "ref": finding["id"]})
        for i, correction in enumerate(clip.result["transcript"]["corrections"]):
            if correction["certainty"] == "clear":
                counts["transcript"] += 1
                listed.append({"clip_id": clip_id, "kind": "clear_correction", "ref": str(i)})
        if clip.record.get("control_kind") == "corrupted_asr":
            continue
        for i in _confident_raw_segments(clip, results_dir / clip_id, reasons):
            counts["transcript"] += 1
            listed.append({"clip_id": clip_id, "kind": "raw_transcript_words", "ref": str(i)})
    return counts, listed


def _confident_raw_segments(clip: _Clip, run_dir: Path, reasons: list[str]) -> list[int]:
    """Indices of the trusted raw transcript segments (`asr.trusted_segments`); the file was
    checked against its hash with the run when the result was loaded."""
    raw = clip.result["transcript"]["raw"]
    if raw is None:
        return []
    clip_id = clip.record["id"]
    artifact = next(a for a in clip.result["artifacts"] if a["id"] == raw["artifact_ref"])
    try:
        raw_doc = json.loads((run_dir / PurePosixPath(artifact["path"])).read_bytes())
        trusted = {id(segment) for segment in asr.trusted_segments(raw_doc)[0]}
        return [i for i, segment in enumerate(raw_doc["segments"]) if id(segment) in trusted]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        reasons.append(
            f"контроль {clip_id!r}: сырой транскрипт не проверен ({type(error).__name__}), "
            "§ 9 не подтверждён"
        )
        return []


def _is_test_double(result: dict) -> bool:
    provenance = result["provenance"]
    return provenance["backend"] == TEST_DOUBLE or provenance["synthetic"] is True


def _settings(clips: list[_Clip]) -> list[dict]:
    seen = {
        (
            p["backend"],
            p["model"],
            p["revision"],
            p["prompt_sha256"],
            p["schema_sha256"],
        )
        for c in clips
        if c.result
        for p in [c.result["provenance"]]
    }
    keys = ("backend", "model", "revision", "prompt_sha256", "schema_sha256")
    return [dict(zip(keys, row, strict=True)) for row in sorted(seen, key=str)]


def _load_manifest(path: Path) -> tuple[bytes, list[dict]]:
    try:
        data = path.read_bytes()
        document = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvalError(f"manifest cannot be read: {type(error).__name__}") from error
    errors = _non_finite_numbers(manifest.SCHEMA, document)
    errors += schema_errors(manifest.SCHEMA, document)
    if errors:
        raise EvalError("manifest does not match its schema:\n" + "\n".join(errors))
    return data, document["records"]


def _load_clip(record: dict, labels_dir: Path, results_dir: Path, errors: list[str]) -> _Clip:
    clip_id = record["id"]
    events: list[dict] = []
    labels_sha256 = None
    if not record["negative_control"]:
        path = labels_dir / f"{clip_id}.json"
        report = labels.validate(path)
        if not report.ok:
            errors.append(f"разметка клипа {clip_id!r} не принята ({len(report.errors)} ошибок)")
        else:
            data = path.read_bytes()
            document = json.loads(data)
            if document["clip_id"] != clip_id or document["clip_sha256"] != record["sha256"]:
                errors.append(f"разметка клипа {clip_id!r} описывает другой клип")
            else:
                events = document["events"]
                labels_sha256 = hashlib.sha256(data).hexdigest()
    result, result_sha256 = _load_result(record, results_dir, errors)
    return _Clip(record, events, result, result_sha256, labels_sha256)


def _load_result(
    record: dict, results_dir: Path, errors: list[str]
) -> tuple[dict | None, str | None]:
    clip_id = record["id"]
    run_dir = results_dir / clip_id
    try:
        data = (run_dir / RESULT_FILE_NAME).read_bytes()
        result = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"результат клипа {clip_id!r} не прочитан ({type(error).__name__})")
        return None, None
    invalid = findings.validation_errors(result)
    if invalid:
        errors.append(
            f"результат клипа {clip_id!r} не проходит контракт результата ({len(invalid)} ошибок)"
        )
        return None, None
    unfinished = [
        name for name in REQUIRED_STAGES if result["stages"][name]["status"] != "completed"
    ]
    if result["status"] != "complete" or unfinished:
        errors.append(
            f"результат клипа {clip_id!r} не завершён (status {result['status']!r}, "
            f"не выполнены этапы: {', '.join(unfinished) or 'нет'})"
        )
        return None, None
    if result["source"]["sha256"] != record["sha256"]:
        errors.append(f"результат клипа {clip_id!r} получен на другом источнике (sha256)")
        return None, None
    run = Run(result["run_id"], run_dir, "", "", reused=True, complete=True, created=False)
    unbacked = findings.validation_errors(result, run=run)
    if unbacked:
        errors.append(
            f"результат клипа {clip_id!r} не подтверждён файлами run ({len(unbacked)} ошибок)"
        )
        return None, None
    return result, hashlib.sha256(data).hexdigest()


def _load_agreements(path: str | Path | None) -> dict[tuple[str, str, int], _Judgement]:
    """`{"judgements": [{"clip_id", "finding_id", "event", "agrees", "result_sha256",
    "labels_sha256"}]}`, written by a person about the result and labels with those hashes."""
    if path is None:
        return {}
    try:
        document = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvalError(f"agreements cannot be read: {type(error).__name__}") from error
    items = document.get("judgements") if isinstance(document, dict) else None
    if not isinstance(items, list):
        raise EvalError("agreements: '$.judgements' must be a list")
    judgements: dict[tuple[str, str, int], _Judgement] = {}
    fields = {
        "clip_id": str,
        "finding_id": str,
        "event": int,
        "agrees": bool,
        "result_sha256": str,
        "labels_sha256": str,
    }
    for i, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != set(fields):
            raise EvalError(f"agreements: $.judgements[{i}] must have exactly {sorted(fields)}")
        for name, kind in fields.items():
            value = item[name]
            if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
                raise EvalError(f"agreements: $.judgements[{i}].{name} must be {kind.__name__}")
        key = (item["clip_id"], item["finding_id"], item["event"])
        if key in judgements:
            raise EvalError(f"agreements: $.judgements[{i}] repeats an earlier judgement")
        judgements[key] = _Judgement(item["agrees"], item["result_sha256"], item["labels_sha256"])
    return judgements


def write_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """`<date>-report.md` and `<date>-report.json` in `output_dir`, never overwriting a report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['date']}-report"
    markdown_path, json_path = directory / f"{stem}.md", directory / f"{stem}.json"
    for path in (markdown_path, json_path):
        if path.exists():
            raise EvalError(f"{path.name} already exists in the output directory; not overwritten")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    return markdown_path, json_path


def format_markdown(report: dict) -> str:
    def number(value: float | None) -> str:
        return "не определено" if value is None else f"{value:.2f}"

    lines = [
        f"# Отчёт evaluation runner, {report['date']}",
        "",
        f"> **Предупреждение.** {report['warning']}",
        "",
        f"- **Часть корпуса:** `{report['split']}`; "
        f"клипов в подсчёте: {report['clips']['scored']}, "
        f"из них синтетических (TTS): {report['clips']['synthetic']}; "
        f"отрицательных контролей (обе части): {report['clips']['negative_controls']}.",
        f"- **Манифест:** `{report['manifest']['name']}`, sha256 `{report['manifest']['sha256']}`.",
        f"- **Пороги (docs/eval/PROTOCOL.md § 7):** precision ≥ {PRECISION_MIN:.2f}, "
        f"recall ≥ {RECALL_MIN:.2f}, уверенных выдумок на контролях — "
        f"{MAX_CONFIDENT_FABRICATIONS}.",
        "",
        "## Статусы",
        "",
    ]
    for flag in ("validated_ru", "diction_validated_ru", "prosody_validated_ru"):
        lines.append(f"- `{flag}`: `{json.dumps(report[flag])}`")
    if report["not_qualified_reasons"]:
        lines += ["", "**NOT QUALIFIED:**", ""]
        lines += [f"- {reason}" for reason in report["not_qualified_reasons"]]
    lines += ["", "## Группы", ""]
    lines.append("| Группа | TP | FP | FN | отчётливых событий | precision | recall | abstention |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, group in report["groups"].items():
        abstention = group["abstention"]
        lines.append(
            f"| {name} | {group['tp']} | {group['fp']} | {group['fn']} | {group['clear_events']} "
            f"| {number(group['precision'])} | {number(group['recall'])} "
            f"| {abstention['clips']} из {abstention['of']} |"
        )
    for name, group in report["groups"].items():
        verdict = (
            "проходит § 7 и § 9"
            if group["passed"]
            else "не проходит: " + "; ".join(group["failed"])
        )
        lines.append(f"\n- **{name}:** {verdict}.")
    lines += ["", "## Категории", ""]
    lines.append(
        "| Категория | TP | FP | FN | событий | uncertain | совпадений с uncertain "
        "| abstention (клипов) | незалокализованных |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, c in report["categories"].items():
        lines.append(
            f"| {name} | {c['tp']} | {c['fp']} | {c['fn']} | {c['events']} | "
            f"{c['uncertain_events']} | {c['uncertain_matches']} | "
            f"{c['abstention_clips']} из {c['clips']} | {c['unlocalized']} |"
        )
    fabrications = report["negative_controls"]["confident_fabrications"]
    lines += [
        "",
        "## Отрицательные контроли (§ 9)",
        "",
        f"Контролей: {report['negative_controls']['clips']}. Уверенных выдумок — дикция: "
        f"{fabrications['diction']}, просодия: {fabrications['prosody']}, транскрипт: "
        f"{fabrications['transcript']} (порог — {MAX_CONFIDENT_FABRICATIONS} в каждой строке).",
        "",
        "## Failures",
        "",
    ]
    failures = report["failures"]
    lines.append("Пропущенные события:")
    lines += [
        f"- `{e['clip_id']}` событие {e['event']} `{e['category']}` {e['start']}–{e['end']} с"
        for e in failures["missed_events"]
    ] or ["- нет"]
    lines.append("\nЛожные findings:")
    lines += [
        f"- `{f['clip_id']}` `{f['finding_id']}` `{f['category']}` {f['start']}–{f['end']} с "
        f"({f['why']})"
        for f in failures["false_findings"]
    ] or ["- нет"]
    lines.append("\nВыдумки на контролях:")
    lines += [
        f"- `{f['clip_id']}` {f['kind']} `{f['ref']}`" for f in failures["control_fabrications"]
    ] or ["- нет"]
    if report["pending_agreement"]:
        lines.append("\nЖдут решения человека о согласии наблюдений (§ 6):")
        lines += [
            f"- `{p['clip_id']}` `{p['finding_id']}` ↔ событие {p['event']} `{p['category']}`"
            for p in report["pending_agreement"]
        ]
    lines += ["", "## Настройки", ""]
    lines += [
        "- " + ", ".join(f"{key}: `{value}`" for key, value in row.items())
        for row in report["settings"]
    ] or ["- результатов нет"]
    return "\n".join(lines) + "\n"
