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
The raw transcript is never rewritten and no "more literary" form is chosen (R05). Measurements
never become findings here: F0 or a pause length is not an auditory conclusion (R04, R06).

`validate` is `schema.validation_errors` (every rule the document itself proves) plus, with `run`,
the files behind the evidence: each cited artifact, the raw transcript and the paired review
provenance exist as regular files inside the run with the recorded sha256, and the provenance
names a valid response of this run, the cited clip and the backend the result claims, so a test
double never passes for a real backend.
"""

import hashlib
import json
import os
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
TRANSCRIPT_ARTIFACT_ID = "transcript-raw"
DISAGREEMENT_CATEGORY = "pronunciation"
DISAGREEMENT_NOTE = (
    "версия аудиомодели — описание наблюдения; ASR-вариант не заменён, выбор не сделан"
)
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
) -> Built:
    """Findings of `review_results` on the source timescale; nothing is written to disk.

    `transcript` is `asr.read_raw(run)` or None when ASR did not run. `measurements` is accepted
    and not used: measurements never turn into auditory findings (R04, R06). An outcome that
    cannot be trusted is skipped with an `errors[]` entry and adds nothing to `reviewed`; an
    observation whose interval is empty once clamped to the clip on the source scale is skipped
    with an `errors[]` entry too. Mixing test-double and real backend outcomes raises `ValueError`.
    """
    outcomes = list(review_results)
    if len({review.is_test_double(outcome.provenance) for outcome in outcomes}) > 1:
        raise ValueError("test-double and real backend outcomes cannot make one result")
    words = _words(transcript)
    raw_sha256 = _transcript_sha256(run, transcript) if transcript is not None else None
    findings: list[dict] = []
    unlocalized: list[dict] = []
    corrections: list[dict] = []
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
        localized, unplaced = [], []
        for obs in outcome.response.observations:
            if obs["whole_clip"] and _polarity(obs) == "negative":
                unplaced.append((obs, "whole_clip"))
            else:
                localized.append(obs)
        unplaced += [(obs, "no_timecodes") for obs in outcome.response.unlocalized_observations]
        for n, obs in enumerate(localized, 1):
            finding = _finding(f"f-{clip.id}-{n}", obs, clip, refs)
            if finding["start"] >= finding["end"]:
                errors.append(
                    _error(
                        INTERVAL_EMPTY,
                        f"observation {n} of clip {clip.id!r} lies past the clip's end on the "
                        "source timescale; it is not a finding",
                    )
                )
                continue
            findings.append(finding)
            heard = _overlapping(words, finding["start"], finding["end"])
            if heard and (finding["category"], finding["polarity"]) == (
                DISAGREEMENT_CATEGORY,
                "negative",
            ):
                corrections.append(_disagreement(finding, heard, refs[1], raw_sha256))
        for n, (obs, reason) in enumerate(unplaced, 1):
            unlocalized.append(_unlocalized(f"u-{clip.id}-{n}", obs, reason, refs))
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
        findings,
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


def _disagreement(
    finding: dict, heard: list[Mapping], review_ref: dict, raw_sha256: str | None
) -> dict:
    """Both versions side by side; the raw transcript stays as it is, nothing is chosen."""
    return {
        "start": finding["start"],
        "end": finding["end"],
        "original": " ".join(str(word["word"]).strip() for word in heard),
        "corrected": finding["observation"],
        "source": "audio_model_observation",
        "evidence_refs": [
            dict(review_ref),
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
        for segment in transcript.get("segments") or []
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
    responses = {
        response["id"]: response
        for _, _, _, refs in claims
        for response in schema.cited_review_responses(refs, artifacts)
    }
    pairs = {rid: schema.review_provenance_for(result, r) for rid, r in responses.items()}
    to_check |= {pair["id"] for pair in pairs.values() if pair is not None}
    errors: list[str] = []
    broken: set[str] = set()
    for artifact_id in sorted(to_check, key=index.__getitem__):
        found = _file_errors(run, f"$.artifacts[{index[artifact_id]}]", artifacts[artifact_id])
        if found:
            broken.add(artifact_id)
        errors += found
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
