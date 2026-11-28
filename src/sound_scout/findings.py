"""Findings of the audio review on the source timescale, and their validation (R05, R06, § 4).

`build` turns the outcomes of `review.run_review` into the result contract's `findings`,
`unlocalized`, `transcript.corrections` and `artifacts`. For every outcome it loads the clip the
provenance names (`clips/<id>.json`), checks that the backend heard exactly that clip (the hash of
the PCM payload), checks the review record on disk, and moves the observation's timecodes from the
clip's own scale onto the source. The model's text is carried over verbatim. An observation without
timecodes, or a negative one spanning the whole clip, does not localise an error and goes to
`unlocalized`, never to `findings`. Polarity follows the structure, not the wording: an observation
with a correction is `negative`, one without is `neutral`; `build` never claims `positive`.

An ASR/audio-model disagreement is kept, not settled: a negative `pronunciation` finding over
words of the raw transcript adds a correction with both versions and `certainty: "uncertain"`.
Only words of trusted segments (`asr.trusted_segments`) count: a segment flagged
`suspect_hallucination` is not speech, so a made-up phrase is never quoted as `original`.
The raw transcript is never rewritten and no "more literary" form is chosen (R05). Measurements
never become findings here: F0 or a pause length is not an auditory conclusion (R04, R06).

`validate` is `schema.validation_errors` (every rule the document itself proves) plus, with `run`,
the files behind the evidence: each cited artifact, the raw transcript and the paired review
provenance exist as regular files inside the run with the recorded sha256, and the provenance
names a valid response of this run, the cited clip and the backend the result claims, so a test
double never passes for a real backend; `transcript.suspect_segments` is exactly what
`asr.suspect_segments` makes of the run's `asr/raw.json`, so no flagged segment is dropped; and
the `findings`, `unlocalized` and audio `transcript.corrections` citing a response are exactly
what `build`'s own code makes of the stored `response.raw.txt` files on their clips' source ranges
(duplicates of overlapping windows merged the same way), up to the placeholders `report.write`
puts in place of private values (`REDACTIONS`). These errors name the path and the rule, never the
text.
"""

import contextlib
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from sound_scout import asr, clips, review, schema
from sound_scout.config import Config
from sound_scout.doctor import TEST_DOUBLE
from sound_scout.runs import Run

STAGE = "audio_review"
RESPONSE_INVALID = "review_response_invalid"
CLIP_MISSING = "review_clip_missing"
PAYLOAD_MISMATCH = "review_payload_mismatch"
RECORD_MISMATCH = "review_record_mismatch"
INTERVAL_EMPTY = "review_interval_empty"
WINDOW_BOUNDARY = "window_boundary"
TRANSCRIPT_ARTIFACT_ID = "transcript-raw"
DISAGREEMENT_CATEGORY = "pronunciation"
DISAGREEMENT_NOTE = (
    "версия аудиомодели — описание наблюдения; ASR-вариант не заменён, выбор не сделан"
)
# What `report.write` puts in a result in place of private paths and the user name.
REDACTIONS = ("<private path>", "<user>")
_REDACTION_SPLIT = re.compile("|".join(map(re.escape, REDACTIONS)))
_HASH_CHUNK = 1 << 20
_CLIP_RECORD_KEYS = ("kind", "path", "sha256", "source_start", "source_end")


@dataclass(frozen=True)
class Built:
    """What `build` contributes to a result; `reviewed` is merged and sorted source ranges."""

    findings: list[dict]
    unlocalized: list[dict]
    corrections: list[dict]
    artifacts: list[dict]
    reviewed: list[dict]
    errors: list[dict]
    synthetic: bool


def build(
    run: Run,
    review_results: Iterable[review.ReviewOutcome],
    measurements: object,
    transcript: Mapping | None,
    *,
    config: Config,
    requested: list[tuple[float, float]] | None = None,
) -> Built:
    """Findings of `review_results` on the source timescale; nothing is written to disk.

    `transcript` is `asr.read_raw(run)` or None when ASR did not run. `measurements` is accepted
    and not used: measurements never turn into auditory findings (R04, R06). An outcome that
    cannot be trusted is skipped with an `errors[]` entry and adds nothing to `reviewed`; an
    observation whose interval is empty once clamped to the clip on the source scale is skipped
    with an `errors[]` entry too. Mixing test-double and real backend outcomes raises `ValueError`,
    and so does a transcript whose segments `asr.trusted_segments` refuses.

    Outcomes of overlapping windows (`review.cover`) share their duplicates: the same observation
    heard in the overlap of two or more windows becomes one finding citing all of them
    (`_merge_window_duplicates`). With `requested`, the source ranges asked for, a finding cut by
    the inner edge of its window adds a `window_boundary` entry to `errors[]`, unless a window of
    the finding it was merged with heard across that edge.
    """
    outcomes = list(review_results)
    if len({review.is_test_double(outcome.provenance) for outcome in outcomes}) > 1:
        raise ValueError("test-double and real backend outcomes cannot make one result")
    words = _words(transcript)
    raw_sha256 = _transcript_sha256(run, transcript) if transcript is not None else None
    made_all: list[_Made] = []
    unlocalized: list[dict] = []
    artifacts: list[dict] = []
    reviewed: list[tuple[float, float]] = []
    errors: list[dict] = []
    for outcome in outcomes:
        checked = _checked_record(run, outcome, config)
        if isinstance(checked, dict):
            errors.append(checked)
            continue
        clip, response_artifact, provenance_artifact = checked
        artifacts += [clip.artifact, response_artifact, provenance_artifact]
        reviewed.append((clip.source_start, clip.source_end))
        refs = [
            {"type": "clip", "ref": clip.id, "sha256": clip.sha256},
            {
                "type": "audio_model_observation",
                "ref": response_artifact["id"],
                "sha256": response_artifact["sha256"],
            },
        ]
        assert outcome.response is not None
        made = _from_response(outcome.response, clip, refs)
        made_all.append(made)
        unlocalized += made.unlocalized
        errors += made.errors
    found, into = _merge_window_duplicates(made_all)
    corrections = _corrections(found, words, raw_sha256)
    if requested is not None:
        errors += _boundary_errors(made_all, into, requested)
    if corrections:
        artifacts.append(
            {
                "id": TRANSCRIPT_ARTIFACT_ID,
                "kind": "transcript_raw",
                "path": f"{asr.ASR_DIR_NAME}/{asr.RAW_FILE_NAME}",
                "sha256": raw_sha256,
                "source_start": None,
                "source_end": None,
            }
        )
    return Built(
        found,
        unlocalized,
        corrections,
        artifacts,
        _merged(reviewed),
        errors,
        bool(outcomes) and review.is_test_double(outcomes[0].provenance),
    )


def validate(result: object, *, run: Run | None = None) -> None:
    """Raise `schema.ResultValidationError` listing every violation, as `schema.validate` does."""
    errors = validation_errors(result, run=run)
    if errors:
        raise schema.ResultValidationError(errors)


def validation_errors(result: object, *, run: Run | None = None) -> list[str]:
    """`schema.validation_errors`, then — on a sound document and with `run` — the run's files."""
    errors = schema.validation_errors(result)
    if errors or run is None:
        return errors
    assert isinstance(result, dict)
    return _run_errors(result, run)


def _checked_record(
    run: Run, outcome: review.ReviewOutcome, config: Config
) -> tuple[clips.Clip, dict, dict] | dict:
    """(clip, review_response artifact, review_provenance artifact), or an `errors[]` entry."""
    provenance = outcome.provenance
    clip_id = provenance.get("clip_id")
    if outcome.response is None:
        return _error(
            RESPONSE_INVALID,
            f"the backend response for clip {clip_id!r} is not valid ({len(outcome.errors)} "
            "errors recorded in its review provenance)",
        )
    try:
        clip = clips.load(run, clip_id, config=config)
    except clips.ClipError as error:
        return _error(CLIP_MISSING, f"the reviewed clip {clip_id!r} is not usable: {error}")
    payload = provenance.get("payload")
    if (
        not isinstance(payload, Mapping)
        or payload.get("sha256") != hashlib.sha256(clips.pcm_f32le(clip)).hexdigest()
        or payload.get("sample_rate") != clip.record["sample_rate"]
        or payload.get("channels") != clip.record["channels"]
    ):
        return _error(
            PAYLOAD_MISMATCH,
            f"the payload the backend heard is not clip {clip_id!r}: hash, rate or channels differ",
        )
    record_dir = f"{review.REVIEW_DIR_NAME}/{clip_id}"
    response_path = f"{record_dir}/{review.RAW_RESPONSE_FILE_NAME}"
    provenance_path = f"{record_dir}/{review.PROVENANCE_FILE_NAME}"
    response_sha256 = _file_sha256(run, response_path)
    provenance_sha256 = _file_sha256(run, provenance_path)
    recorded = provenance.get("response")
    if (
        response_sha256 is None
        or provenance_sha256 is None
        or not isinstance(recorded, Mapping)
        or recorded.get("sha256") != response_sha256
        or _read_json(run, provenance_path) != json.loads(json.dumps(provenance))
    ):
        return _error(
            RECORD_MISMATCH,
            f"the review record in {record_dir}/ does not match the outcome for clip {clip_id!r}",
        )
    span = {"source_start": clip.source_start, "source_end": clip.source_end}
    response_artifact = {
        "id": f"review-{clip_id}",
        "kind": "review_response",
        "path": response_path,
        "sha256": response_sha256,
        **span,
    }
    provenance_artifact = {
        "id": f"review-{clip_id}-provenance",
        "kind": "review_provenance",
        "path": provenance_path,
        "sha256": provenance_sha256,
        **span,
    }
    return clip, response_artifact, provenance_artifact


@dataclass(frozen=True)
class _Made:
    """What one response makes; `span` is its clip's source range, `corrections` are filled in
    only after the findings of all windows are merged."""

    findings: list[dict]
    unlocalized: list[dict]
    corrections: list[dict]
    errors: list[dict]
    clip_id: str
    span: tuple[float, float]


def _from_response(response: review.ReviewResponse, clip: clips.Clip, refs: list[dict]) -> _Made:
    """What one validated response makes on the source scale; `build` and `validate` share it."""
    made = _Made([], [], [], [], clip.id, (clip.source_start, clip.source_end))
    localized, unplaced = [], []
    for obs in response.observations:
        if obs["whole_clip"] and _polarity(obs) == "negative":
            unplaced.append((obs, "whole_clip"))
        else:
            localized.append(obs)
    unplaced += [(obs, "no_timecodes") for obs in response.unlocalized_observations]
    for n, obs in enumerate(localized, 1):
        finding = _finding(f"f-{clip.id}-{n}", obs, clip, refs)
        if finding["start"] >= finding["end"]:
            made.errors.append(
                _error(
                    INTERVAL_EMPTY,
                    f"observation {n} of clip {clip.id!r} lies past the clip's end on the "
                    "source timescale; it is not a finding",
                )
            )
            continue
        made.findings.append(finding)
    for n, (obs, reason) in enumerate(unplaced, 1):
        made.unlocalized.append(_unlocalized(f"u-{clip.id}-{n}", obs, reason, refs))
    return made


def _polarity(observation: Mapping) -> str:
    return "neutral" if observation["correction"] is None else "negative"


def _finding(finding_id: str, obs: Mapping, clip: clips.Clip, refs: list[dict]) -> dict:
    return {
        "id": finding_id,
        "category": obs["category"],
        "observation": obs["observation"],
        "start": round(clip.source_start + obs["start_seconds"], 6),
        "end": round(min(clip.source_start + obs["end_seconds"], clip.source_end), 6),
        "evidence_basis": "audio_model_observation",
        "evidence_refs": [dict(ref) for ref in refs],
        "confidence_note": obs["confidence_note"],
        "impact": obs["impact"],
        "correction": obs["correction"],
        "polarity": _polarity(obs),
    }


def _unlocalized(item_id: str, obs: Mapping, reason: str, refs: list[dict]) -> dict:
    return {
        "id": item_id,
        "category": obs["category"],
        "observation": obs["observation"],
        "reason": reason,
        "evidence_basis": "audio_model_observation",
        "evidence_refs": [dict(ref) for ref in refs],
        "confidence_note": obs["confidence_note"],
        "impact": obs["impact"],
        "correction": obs["correction"],
    }


def _merge_window_duplicates(made: list[_Made]) -> tuple[list[dict], dict[str, str]]:
    """The findings of all windows with the duplicates of overlapping windows merged, and the id
    each finding went into; `build` and `validate` share it.

    For clips A and B with `A.start < B.start` and an overlap `Z = [B.start, A.end]`, a finding of
    A and one of B with the same category and polarity, whose intervals meet and both meet `Z`,
    are one observation heard twice. Such pairs join transitively, so an event heard in three
    windows is one finding too, but a group never holds two findings of one clip. A group keeps its
    earliest finding, by `(clip start, start, id)`, with the union of the intervals and the
    references of the others appended in that order; the order of the rest is kept, and the
    result does not depend on the order of `made`.
    """
    items = [(finding, m.span, m.clip_id) for m in made for finding in m.findings]
    order = sorted(range(len(items)), key=lambda i: (items[i][1][0], *_place(items[i][0])))
    rank = {i: n for n, i in enumerate(order)}
    parent = list(range(len(items)))
    clips_in = [{clip_id} for _, _, clip_id in items]

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for n, i in enumerate(order):
        for j in order[n + 1 :]:
            a, b = root(i), root(j)
            if a == b or clips_in[a] & clips_in[b] or not _duplicates(items[i], items[j]):
                continue
            keep, gone = sorted((a, b), key=rank.__getitem__)
            parent[gone] = keep
            clips_in[keep] |= clips_in[gone]
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(root(i), []).append(i)
    found, into = [], {}
    for i, (finding, _, _) in enumerate(items):
        members = groups.get(i)
        for m in members or []:
            into[items[m][0]["id"]] = finding["id"]
        if members is None:
            continue
        if len(members) == 1:
            found.append(finding)
            continue
        refs: list[dict] = []
        for m in members:
            refs += [dict(r) for r in items[m][0]["evidence_refs"] if r not in refs]
        found.append(
            {
                **finding,
                "start": min(items[m][0]["start"] for m in members),
                "end": max(items[m][0]["end"] for m in members),
                "evidence_refs": refs,
            }
        )
    return found, into


def _place(finding: dict) -> tuple[float, str]:
    return finding["start"], finding["id"]


def _duplicates(a: tuple, b: tuple) -> bool:
    """`a` and `b` (finding, clip span, clip id), `a`'s clip starting first, are one observation
    heard in the overlap of their clips."""
    (fa, (a_start, a_end), _), (fb, (b_start, _), _) = a, b
    zone = (b_start, a_end)
    return (
        a_start < b_start
        and zone[1] - zone[0] > schema.TIME_EPS
        and (fa["category"], fa["polarity"]) == (fb["category"], fb["polarity"])
        and _meet((fa["start"], fa["end"]), (fb["start"], fb["end"]))
        and _meet((fa["start"], fa["end"]), zone)
        and _meet((fb["start"], fb["end"]), zone)
    )


def _meet(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] + schema.TIME_EPS and b[0] < a[1] + schema.TIME_EPS


def _boundary_errors(
    made: list[_Made], into: dict[str, str], requested: list[tuple[float, float]]
) -> list[dict]:
    """A finding touching an edge of its window that is not an edge of a requested range may be
    cut by the window (R03). The mark goes to the finding it was merged into, unless a window of
    another finding in that merge holds the edge inside, so the event was heard across it."""
    edges = [edge for interval in requested for edge in interval]
    spans: dict[str, list[tuple[float, float]]] = {}
    for m in made:
        for finding in m.findings:
            spans.setdefault(into[finding["id"]], []).append(m.span)
    errors = []
    for m in made:
        for finding in m.findings:
            merged_id = into[finding["id"]]
            for at, edge in ((finding["start"], m.span[0]), (finding["end"], m.span[1])):
                inner = all(abs(edge - outer) > schema.TIME_EPS for outer in edges)
                heard_across = any(
                    start + schema.TIME_EPS < edge < end - schema.TIME_EPS
                    for start, end in spans[merged_id]
                )
                if inner and abs(at - edge) <= schema.TIME_EPS and not heard_across:
                    errors.append(
                        {
                            "stage": STAGE,
                            "code": WINDOW_BOUNDARY,
                            "message": f"finding {merged_id} touches the inner edge of "
                            f"window {m.clip_id} at {round(edge, 6)} s; its interval may be cut",
                        }
                    )
    return errors


def _corrections(found: list[dict], words: list[Mapping], raw_sha256: str | None) -> list[dict]:
    """An ASR/audio-model disagreement for each negative pronunciation finding over words."""
    corrections = []
    for finding in found:
        heard = _overlapping(words, finding["start"], finding["end"])
        if heard and (finding["category"], finding["polarity"]) == (
            DISAGREEMENT_CATEGORY,
            "negative",
        ):
            corrections.append(_disagreement(finding, heard, raw_sha256))
    return corrections


def _disagreement(finding: dict, heard: list[Mapping], raw_sha256: str | None) -> dict:
    """Both versions side by side; the raw transcript stays as it is, nothing is chosen."""
    return {
        "start": finding["start"],
        "end": finding["end"],
        "original": " ".join(str(word["word"]).strip() for word in heard),
        "corrected": finding["observation"],
        "source": "audio_model_observation",
        "evidence_refs": [
            *(dict(r) for r in finding["evidence_refs"] if r["type"] == "audio_model_observation"),
            {"type": "transcript", "ref": TRANSCRIPT_ARTIFACT_ID, "sha256": raw_sha256},
        ],
        "certainty": "uncertain",
        "note": DISAGREEMENT_NOTE,
    }


def _words(transcript: Mapping | None) -> list[Mapping]:
    if transcript is None:
        return []
    return [
        word
        for segment in asr.trusted_segments(transcript)[0]
        for word in segment.get("words") or []
        if _number(word.get("start")) and _number(word.get("end"))
    ]


def _overlapping(words: list[Mapping], start: float, end: float) -> list[Mapping]:
    return [word for word in words if word["start"] < end and word["end"] > start]


def _transcript_sha256(run: Run, transcript: Mapping) -> str:
    """The hash of `asr/raw.json`, which must be the transcript passed in."""
    path = f"{asr.ASR_DIR_NAME}/{asr.RAW_FILE_NAME}"
    sha256 = _file_sha256(run, path)
    if sha256 is None:
        raise asr.AsrError(asr.TRANSCRIPT_MISSING, f"{path} is not in the run")
    if _read_json(run, path) != json.loads(json.dumps(transcript)):
        raise ValueError(f"the transcript passed in is not the run's {path}")
    return sha256


def _merged(ranges: list[tuple[float, float]]) -> list[dict]:
    merged: list[list[float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + schema.TIME_EPS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": start, "end": end} for start, end in merged]


def _error(code: str, message: str) -> dict:
    return {"stage": STAGE, "code": code, "message": message}


def _run_errors(result: dict, run: Run) -> list[str]:
    artifacts = {a["id"]: a for a in result["artifacts"]}
    index = {a["id"]: i for i, a in enumerate(result["artifacts"])}
    claims = schema.evidence_claims(result)
    to_check = {
        ref["ref"] for _, _, _, refs in claims for ref in refs if ref["type"] != "measurement"
    }
    if result["transcript"]["raw"] is not None:
        to_check.add(result["transcript"]["raw"]["artifact_ref"])
    words = _word_refs(result)
    to_check |= {ref["artifact_ref"] for _, ref in words}
    cited = {
        response["id"]
        for _, _, _, refs in claims
        for response in schema.cited_review_responses(refs, artifacts)
    }
    # Every paired response is rebuilt, cited or not, so dropping all it makes is noticed.
    responses = {a["id"]: a for a in result["artifacts"] if a["kind"] == "review_response"}
    pairs = {rid: schema.review_provenance_for(result, r) for rid, r in responses.items()}
    pairs = {rid: pair for rid, pair in pairs.items() if pair is not None or rid in cited}
    to_check |= {rid for rid, pair in pairs.items() if pair is not None}
    to_check |= {pair["id"] for pair in pairs.values() if pair is not None}
    errors: list[str] = []
    broken: set[str] = set()
    for artifact_id in sorted(to_check, key=index.__getitem__):
        found = _file_errors(run, f"$.artifacts[{index[artifact_id]}]", artifacts[artifact_id])
        if found:
            broken.add(artifact_id)
        errors += found
    raw_ref = result["transcript"]["raw"]
    if raw_ref is not None and raw_ref["artifact_ref"] not in broken:
        errors += _suspect_segment_errors(result, run, artifacts[raw_ref["artifact_ref"]])
    errors += _word_ref_errors(run, words, artifacts, broken)
    records = {}
    clip_records: set[str] = set()
    for response_id, pair in pairs.items():
        if pair is None or pair["id"] in broken or response_id in broken:
            continue
        path = f"$.artifacts[{index[pair['id']]}]"
        record = _read_json(run, pair["path"])
        if not isinstance(record, dict):
            errors.append(f"{path}: review provenance is not a JSON object")
            continue
        records[response_id] = record
        errors += _provenance_errors(path, record, responses[response_id], result)
    for claim_path, _, _, refs in claims:
        clip_refs = {ref["ref"] for ref in refs if ref["type"] == "clip"}
        for response in schema.cited_review_responses(refs, artifacts):
            record = records.get(response["id"])
            if record is None:
                continue
            clip = artifacts.get(record.get("clip_id"))
            if (
                clip is None
                or clip["kind"] != "clip"
                or (clip["source_start"], clip["source_end"])
                != (response["source_start"], response["source_end"])
                or (clip_refs and clip["id"] not in clip_refs)
            ):
                errors.append(
                    f"{claim_path}: review_response {response['id']!r} was made for clip "
                    f"{record.get('clip_id')!r}, not the clip this claim cites with its range"
                )
            elif clip["id"] not in clip_records:
                clip_records.add(clip["id"])
                errors += _clip_record_errors(run, f"$.artifacts[{index[clip['id']]}]", clip)
    for response_id in records.keys() - cited:
        record, response = records[response_id], responses[response_id]
        clip = artifacts.get(record.get("clip_id"))
        if (
            clip is None
            or clip["kind"] != "clip"
            or (clip["source_start"], clip["source_end"])
            != (response["source_start"], response["source_end"])
        ):
            errors.append(
                f"$.artifacts[{index[response_id]}]: review_response was made for clip "
                f"{record.get('clip_id')!r}, not a clip of this result with its range"
            )
        elif clip["id"] not in clip_records:
            clip_records.add(clip["id"])
            errors += _clip_record_errors(run, f"$.artifacts[{index[clip['id']]}]", clip)
    return errors + _content_errors(result, run, responses, records, broken)


def _content_errors(
    result: dict, run: Run, responses: dict, records: dict, broken: set[str]
) -> list[str]:
    """Findings, unlocalized and audio corrections are what `_from_response` makes of the cited,
    stored responses; an entry citing a response that cannot be rebuilt is left to the rules
    that already reject it."""
    artifacts = {a["id"]: a for a in result["artifacts"]}
    index = {a["id"]: i for i, a in enumerate(result["artifacts"])}
    errors: list[str] = []
    words: list[Mapping] | None = []
    raw_sha256 = None
    raw_ref = result["transcript"]["raw"]
    if raw_ref is not None:
        words = None
        raw_artifact = artifacts[raw_ref["artifact_ref"]]
        raw = None if raw_artifact["id"] in broken else _read_json(run, raw_artifact["path"])
        if isinstance(raw, dict):
            raw_sha256 = raw_artifact["sha256"]
            with contextlib.suppress(ValueError, TypeError, AttributeError):
                words = _words(raw)
    rebuilt: dict[str, _Made] = {}
    for response_id, record in records.items():
        made = _rebuilt(run, responses[response_id], record, artifacts)
        if isinstance(made, str):
            errors.append(f"$.artifacts[{index[response_id]}]: {made}")
        elif made is not None:
            rebuilt[response_id] = made
    # Findings and corrections are what the responses make together, duplicates of overlapping
    # windows merged; each is checked against the response it cites first.
    found, _ = _merge_window_duplicates(list(rebuilt.values()))
    corrections = _corrections(found, words or [], raw_sha256)
    expected = {
        response_id: _Made(
            [f for f in found if _first_response(f["evidence_refs"]) == response_id],
            made.unlocalized,
            [c for c in corrections if _first_response(c["evidence_refs"]) == response_id],
            [],
            made.clip_id,
            made.span,
        )
        for response_id, made in rebuilt.items()
    }
    groups = (
        ("findings", "$.findings", result["findings"], "evidence_basis"),
        ("unlocalized", "$.unlocalized", result["unlocalized"], "evidence_basis"),
        ("corrections", "$.transcript.corrections", result["transcript"]["corrections"], "source"),
    )
    for field, path, entries, basis in groups:
        cited: dict[str, list[int]] = {response_id: [] for response_id in expected}
        for i, entry in enumerate(entries):
            if entry[basis] != "audio_model_observation":
                continue
            cites = schema.cited_review_responses(entry["evidence_refs"], artifacts)
            response_id = next((r["id"] for r in cites if r["id"] in expected), None)
            if response_id is not None:
                cited[response_id].append(i)
        if field == "corrections" and words is None:
            if any(cited.values()):
                errors.append(f"{path}: the raw transcript gives no words to check them against")
            continue
        for response_id, positions in cited.items():
            made = getattr(expected[response_id], field)
            for n, i in enumerate(positions):
                if n >= len(made):
                    errors.append(f"{path}[{i}]: the review response it cites makes no such entry")
                elif not _same(made[n], entries[i]):
                    errors.append(f"{path}[{i}]: does not match the review response it cites")
            if len(made) > len(positions):
                errors.append(
                    f"{path}: {len(made) - len(positions)} entries the review response at "
                    f"$.artifacts[{index[response_id]}] makes are missing"
                )
    return errors


def _first_response(refs: list[dict]) -> str | None:
    return next((r["ref"] for r in refs if r["type"] == "audio_model_observation"), None)


def _rebuilt(run: Run, response: dict, record: dict, artifacts: dict) -> _Made | str | None:
    """What the stored response makes, a rule it breaks, or None when its clip is not usable."""
    clip_artifact = artifacts.get(record.get("clip_id"))
    if clip_artifact is None or clip_artifact["kind"] != "clip":
        return None
    if _clip_record_errors(run, "", clip_artifact) or (
        clip_artifact["source_start"],
        clip_artifact["source_end"],
    ) != (response["source_start"], response["source_end"]):
        return None
    relative = f"{clips.CLIPS_DIR_NAME}/{clip_artifact['id']}.json"
    clip_record = _read_json(run, relative)
    if not isinstance(clip_record, dict):
        return None
    payload = record.get("payload")
    duration = payload.get("duration_s") if isinstance(payload, dict) else None
    if not _number(duration) or duration <= 0:
        return "review provenance records no payload duration to check the response against"
    try:
        text = (run.path / response["path"]).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "review response is not readable UTF-8 text"
    parsed, _, _ = review.validate_response(text, duration)
    if parsed is None:
        return "review response does not pass the response validator"
    clip = clips.Clip(
        clip_artifact["id"],
        clips.CLIP_KIND,
        run.path / clip_record["path"],
        run.path / relative,
        clip_record,
        reused=True,
    )
    refs = [
        {"type": "clip", "ref": clip_artifact["id"], "sha256": clip_artifact["sha256"]},
        {"type": "audio_model_observation", "ref": response["id"], "sha256": response["sha256"]},
    ]
    return _from_response(parsed, clip, refs)


def _same(expected: object, actual: object) -> bool:
    """Equal, allowing the placeholders `report.write` puts in place of private values."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _same(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(map(_same, expected, actual))
    if isinstance(expected, str) and isinstance(actual, str):
        return _redacted_from(expected, actual)
    if _number(expected) and _number(actual):
        return expected == actual
    return type(expected) is type(actual) and expected == actual


def _redacted_from(expected: str, actual: str) -> bool:
    """`actual` is `expected` with some non-empty parts replaced by `REDACTIONS`.

    Leftmost matching of the literal parts, so a crafted document cannot make it backtrack."""
    parts = _REDACTION_SPLIT.split(actual)
    if len(parts) == 1:
        return expected == actual
    first, *middle, last = parts
    if not expected.startswith(first):
        return False
    cursor, end = len(first) + 1, len(expected) - len(last)
    for part in middle:
        found = expected.find(part, cursor, end)
        if found < 0:
            return False
        cursor = found + len(part) + 1
    return cursor <= end and expected.endswith(last)


def _suspect_segment_errors(result: dict, run: Run, raw_artifact: dict) -> list[str]:
    """`transcript.suspect_segments` restates the flagged segments of the hashed raw file."""
    raw = _read_json(run, raw_artifact["path"])
    try:
        expected = asr.suspect_segments(raw, result["source"]["duration_s"])
    except ValueError as error:
        return [f"$.transcript.raw: {error}"]
    if result["transcript"]["suspect_segments"] != expected:
        return [
            "$.transcript.suspect_segments: does not match the suspect_hallucination segments "
            "of the raw transcript"
        ]
    return []


def _word_refs(result: dict) -> list[tuple[str, dict]]:
    """(path, transcript_ref) of every value per word in `measurements[]`."""
    return [
        (f"$.measurements[{i}].value[{j}].transcript_ref", item["transcript_ref"])
        for i, measurement in enumerate(result["measurements"])
        if isinstance(measurement["value"], list)
        for j, item in enumerate(measurement["value"])
        if "transcript_ref" in item
    ]


def _word_ref_errors(
    run: Run, words: list[tuple[str, dict]], artifacts: dict, broken: set[str]
) -> list[str]:
    """Each pointer names a word of the hashed raw transcript file it cites."""
    errors = []
    raws: dict[str, object] = {}
    for path, ref in words:
        if ref["artifact_ref"] in broken:
            continue
        if ref["artifact_ref"] not in raws:
            raws[ref["artifact_ref"]] = _read_json(run, artifacts[ref["artifact_ref"]]["path"])
        found = raws[ref["artifact_ref"]]
        positions = map(int, re.findall(r"\d+", ref["pointer"]))
        for key, at in zip(("segments", "words"), positions, strict=True):
            items = found.get(key) if isinstance(found, dict) else None
            found = items[at] if isinstance(items, list) and at < len(items) else None
        if not isinstance(found, dict):
            errors.append(f"{path}.pointer: {ref['pointer']} is not a word of the raw transcript")
    return errors


def _clip_record_errors(run: Run, path: str, clip: dict) -> list[str]:
    """The clip artifact is what `clips/<id>.json` of the run records, source range included."""
    relative = f"{clips.CLIPS_DIR_NAME}/{clip['id']}.json"
    record = _read_json(run, relative) if _file_sha256(run, relative) is not None else None
    if not isinstance(record, dict):
        return [f"{path}: the run holds no clip record {relative}"]
    if any(record.get(key) != clip[key] for key in _CLIP_RECORD_KEYS):
        return [f"{path}: clip artifact does not match its record {relative} (path, hash or range)"]
    return []


def _provenance_errors(path: str, record: dict, response: dict, result: dict) -> list[str]:
    errors = []
    recorded = record.get("response")
    if not isinstance(recorded, dict) or recorded.get("valid") is not True:
        errors.append(f"{path}: review provenance does not record a valid response")
    elif recorded.get("sha256") != response["sha256"]:
        errors.append(
            f"{path}: review provenance response sha256 does not match {response['id']!r}"
        )
    if record.get("run_id") != result["run_id"]:
        errors.append(f"{path}: review provenance belongs to another run")
    backend = record.get("backend")
    name = backend.get("name") if isinstance(backend, dict) else None
    claimed = result["provenance"]["backend"]
    if name == TEST_DOUBLE and claimed != TEST_DOUBLE:
        errors.append(
            f"{path}: the review came from the test double, but provenance.backend is "
            f"{claimed!r}; such a result is synthetic and names backend {TEST_DOUBLE!r}"
        )
    elif name != TEST_DOUBLE and name != claimed:
        errors.append(
            f"{path}: review backend {name!r} does not match provenance.backend {claimed!r}"
        )
    return errors


def _file_errors(run: Run, path: str, artifact: dict) -> list[str]:
    """The artifact is a regular file inside the run directory with the recorded sha256."""
    target = run.path / artifact["path"]
    try:
        mode = os.lstat(target).st_mode
    except FileNotFoundError:
        return [f"{path}: file is missing"]
    except OSError as error:
        return [f"{path}: file is not readable: {error.strerror}"]
    if not stat.S_ISREG(mode):
        return [f"{path}: not a regular file"]
    if not target.resolve().is_relative_to(run.path.resolve()):
        return [f"{path}: file lies outside the run directory"]
    if _file_sha256(run, artifact["path"]) != artifact["sha256"]:
        return [f"{path}: sha256 does not match the file"]
    return []


def _file_sha256(run: Run, relative: str) -> str | None:
    """The sha256 of a regular file inside the run (no symlink), None when there is none."""
    target = run.path / relative
    try:
        if not stat.S_ISREG(os.lstat(target).st_mode):
            return None
        if not target.resolve().is_relative_to(run.path.resolve()):
            return None
        digest = hashlib.sha256()
        with open(target, "rb") as source:
            while chunk := source.read(_HASH_CHUNK):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _read_json(run: Run, relative: str) -> object:
    try:
        return json.loads((run.path / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
