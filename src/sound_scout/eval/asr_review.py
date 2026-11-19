"""Local ASR over the dev `corrupted_asr` candidates of a corpus, for a person to compare (#84).

`python -m sound_scout.eval.asr_review --manifest DIR/manifest.json` takes the records with
`split: dev` and `control_kind: corrupted_asr` (`candidates`) before any path is touched: a
`heldout` record is never resolved, opened, decoded or transcribed (docs/eval/PROTOCOL.md § 5).
Each candidate goes the `analyze` way — the input policy and the decoder (`ingest.decode`: the
clips directory must be an allowed root of the configuration), the working mono at 16 kHz, the
local ASR (`asr.transcribe`, offline, the configured cached model) — in a run of its own under
`<data_root>/asr-review/<manifest name>/`, where the raw transcript and the model receipt stay.
A clip whose hash differs from its record gets no ASR and no sheet. A manifest that breaks its
schema, a record rule, the uniqueness of ids and hashes or the dev/held-out split of a source group
runs nothing; the draft shortfalls (coverage, counts, unconfirmed candidates) do not stop it.

The review sheet `DIR/asr-review/<clip id>.json` (next to the corpus, outside git) holds the raw
ASR text and segments and a `human` section with every field empty: what the person heard, the
verdict (`asr_error` or `no_asr_error`), who and when. This code never fills it, never compares
the two texts and never touches the manifest: a difference of texts is no confirmation, and only
a sheet a person filled lets the manifest say `confirmed` (`manifest.sheet_problems`). A sheet
that exists is never rewritten; an ASR stage that is not `completed` writes no sheet.
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sound_scout import asr, channels, config, guard, ingest, paths, prepare, runs
from sound_scout._schema_messages import schema_errors
from sound_scout.eval import gate0, manifest
from sound_scout.schema import _non_finite_numbers

FORMAT_VERSION = 1
SHEET_KIND = "asr_review_sheet"
RUNS_DIR = "asr-review"
DIR_MODE = 0o700
FILE_MODE = 0o600
INSTRUCTIONS = (
    "Сначала прослушайте клип, не читая asr.text, и впишите услышанное в human.transcript. Затем "
    "сравните с asr.text и поставьте human.verdict: asr_error — локальный ASR ошибся, "
    "no_asr_error — не ошибся; заполните human.reviewer и human.reviewed_at (YYYY-MM-DD). "
    "Различие текстов само по себе ошибку ASR не подтверждает. Только по заполненному листу с "
    "verdict asr_error в манифесте ставится asr_error_status: confirmed."
)
_LOAD_ERRORS = (
    ingest.IngestError,
    paths.PathDenied,
    channels.ChannelError,
    prepare.PrepareError,
    asr.AsrError,
    OSError,
)


class ReviewError(Exception):
    """The manifest cannot be used; nothing was run."""


def candidates(records: list[dict]) -> tuple[list[dict], list[str]]:
    """The dev corrupted_asr records to transcribe, and the ids of the held-out ones left alone."""
    kind = [r for r in records if r.get("control_kind") == "corrupted_asr"]
    return [r for r in kind if r["split"] == "dev"], [r["id"] for r in kind if r["split"] != "dev"]


def review(manifest_path: str | Path, config_: config.Config) -> dict:
    """Transcribe each dev candidate once and write its sheet; returns what happened per clip."""
    manifest_path = Path(manifest_path)
    data, document = _load_manifest(manifest_path)
    root = manifest_path.parent
    selected, heldout = candidates(document["records"])
    runs_dir = _mkdir(_mkdir(config_.data_root / RUNS_DIR, config_) / document["name"], config_)
    sheets_dir = root / manifest.ASR_REVIEW_DIR
    outcomes = []
    for record in selected:
        sheet_path = sheets_dir / f"{record['id']}.json"
        outcome = {"id": record["id"], "sheet": f"{manifest.ASR_REVIEW_DIR}/{sheet_path.name}"}
        if os.path.lexists(sheet_path):
            outcomes.append({**outcome, "outcome": "exists", "reason": None})
            continue
        problem, sheet = _transcribe(record, root, runs_dir, config_)
        if problem is not None:
            outcomes.append({**outcome, "outcome": "failed", "reason": problem, "sheet": None})
            continue
        sheet["manifest"] = {"name": document["name"], "sha256": hashlib.sha256(data).hexdigest()}
        with contextlib.suppress(FileExistsError):
            os.mkdir(sheets_dir, DIR_MODE)
        _write_json(sheet_path, sheet)
        outcomes.append({**outcome, "outcome": "written", "reason": None})
    return {"reviewed": outcomes, "skipped_heldout": heldout}


def _transcribe(
    record: dict, root: Path, runs_dir: Path, config_: config.Config
) -> tuple[str | None, dict]:
    run_dir = _mkdir(runs_dir / f"{record['id']}-{uuid.uuid4().hex[:12]}", config_)
    run = runs.Run(f"asr-review-{uuid.uuid4().hex}", run_dir, RUNS_DIR, "", False, False, True)
    try:
        loaded = gate0.load_mono16k(root / record["path"], run, config_)
        if loaded["sha256"] != record["sha256"]:
            return "the clip's sha256 differs from its manifest record", {}
        transcription = asr.transcribe(loaded["samples"], run, config_)
    except _LOAD_ERRORS as error:
        return f"{type(error).__name__}: {error}", {}
    if transcription.status != "completed":
        return f"asr is {transcription.status}: {transcription.reason}", {}
    raw = transcription.raw
    return None, {
        "format_version": FORMAT_VERSION,
        "kind": SHEET_KIND,
        "issue": "https://github.com/renidere/sound-scout-plugin/issues/84",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "clip_id": record["id"],
        "clip_path": record["path"],
        "clip_sha256": record["sha256"],
        "split": record["split"],
        "control_kind": record["control_kind"],
        "asr": {
            "status": transcription.status,
            "run": run_dir.relative_to(config_.data_root.resolve()).as_posix(),
            "raw_sha256": transcription.raw_sha256,
            "model": {key: raw["model"][key] for key in ("repo", "revision")},
            "flags": raw["flags"],
            "text": raw["text"],
            "segments": [
                {
                    key: segment.get(key)
                    for key in ("start", "end", "text", asr.SUSPECT_HALLUCINATION)
                }
                for segment in raw["segments"]
            ],
        },
        "human": {
            "transcript": "",
            "verdict": None,
            "reviewer": "",
            "reviewed_at": "",
            "note": None,
        },
        "verdicts": list(manifest.ASR_ERROR_VERDICTS),
        "instructions": INSTRUCTIONS,
    }


def _load_manifest(path: Path) -> tuple[bytes, dict]:
    try:
        data = path.read_bytes()
        document = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewError(f"manifest cannot be read: {type(error).__name__}") from error
    errors = _non_finite_numbers(manifest.SCHEMA, document)
    errors += schema_errors(manifest.SCHEMA, document)
    if errors:
        raise ReviewError("manifest does not match its schema:\n" + "\n".join(errors))
    records = document["records"]
    errors = (
        manifest._record_errors(records)
        + manifest._uniqueness_errors(records)
        + manifest._split_errors(records)
    )
    if errors:
        raise ReviewError("manifest records are not consistent:\n" + "\n".join(errors))
    return data, document


def _mkdir(path: Path, config_: config.Config) -> Path:
    target = paths.resolve_output(path.parent, path.name, config_)
    with contextlib.suppress(FileExistsError):
        os.mkdir(target, DIR_MODE)
    return target


def _write_json(path: Path, document: dict) -> None:
    text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sound_scout.eval.asr_review")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config_ = config.load()
        guard.install_network_guard(config_)
        result = review(args.manifest, config_)
    except (ReviewError, config.ConfigError, paths.PathDenied) as error:
        print(f"asr_review: {error}", file=sys.stderr)
        return 2
    for outcome in result["reviewed"]:
        print(json.dumps(outcome, ensure_ascii=False))
    print(json.dumps({"skipped_heldout": result["skipped_heldout"]}))
    return 1 if any(o["outcome"] == "failed" for o in result["reviewed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
