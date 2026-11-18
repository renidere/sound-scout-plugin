"""Evaluation runner (issue #28): matching by docs/eval/PROTOCOL.md § 6, metrics § 7,
abstention § 8, negative controls § 9, statuses § 10, the report and the `eval run` command.

Results are made by the real pipeline with `FakeBackend` (review → findings.build), so they are
valid under the result contract and marked test-double: they check the arithmetic and the
matching rules, never that the tool hears (CLAUDE.md: a mock proves a contract)."""

import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sound_scout import asr, findings, ingest, review
from sound_scout.cli import main
from sound_scout.clips import cut
from sound_scout.config import Config
from sound_scout.eval import runner
from sound_scout.eval.runner import EvalError, candidate, evaluate, qualification, write_report
from sound_scout.runs import open_run
from tests.test_clips import RATE
from tests.test_findings import document
from tests.test_ingest import write_wav
from tests.test_review import FakeBackend

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = REPO_ROOT / "docs" / "eval" / "PROTOCOL.md"
DURATION = 10.0
TODAY = date(2026, 11, 18)
IDENTITY = {"audio": None, "model": None, "preprocessing": None, "ranges": None}

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)


def sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(category: str, start: float, end: float, certainty: str = "clear") -> dict:
    return {
        "category": category,
        "start": start,
        "end": end,
        "description": "синтетическое событие",
        "certainty": certainty,
        "comment": "синтетика: неоднозначно" if certainty == "uncertain" else None,
    }


def observation(category: str, start: float, end: float) -> dict:
    return {
        "category": category,
        "observation": "синтетическое наблюдение",
        "start_seconds": start,
        "end_seconds": end,
        "confidence_note": "синтетика",
        "impact": "синтетика",
        "correction": "синтетика",
    }


# --- § 6 interval rules -------------------------------------------------------------------------


def span(start: float, end: float) -> dict:
    return {"start": start, "end": end}


@pytest.mark.parametrize(
    ("category", "found", "labelled", "expected"),
    [
        ("pronunciation", (1.9, 2.6), (2.0, 2.5), True),
        ("pronunciation", (2.25, 2.6), (2.0, 2.5), True),  # exactly half the event
        ("pronunciation", (2.3, 2.6), (2.0, 2.5), False),  # less than half
        ("pronunciation", (1.4, 2.5), (2.0, 2.5), False),  # 0.6 s before the event
        ("pronunciation", (0.6, 3.8), (1.0, 3.4), False),  # 3.2 s > 3 s, inside the tolerance
        ("repetition_self_correction", (4.0, 7.0), (5.0, 6.0), True),
        ("repetition_self_correction", (0.4, 6.6), (0.5, 6.5), False),  # 6.2 s > 6 s
        ("semantic_stress", (0.0, 2.0), (1.0, 1.5), True),
        ("pause_phrasing", (3.2, 3.8), (3.0, 4.0), True),
        ("pause_phrasing", (3.0, 3.4), (3.0, 4.0), False),  # misses the middle
        ("completeness", (7.6, 8.3), (7.0, 8.0), True),
        ("completeness", (8.4, 9.0), (7.0, 8.0), True),  # within ± 0.5 s of the end
        ("completeness", (7.0, 7.4), (7.0, 8.0), False),
        ("completeness", (5.0, 13.5), (7.0, 12.0), False),  # 8.5 s > 8 s
    ],
)
def test_candidate_follows_the_protocol_table(
    category: str, found: tuple, labelled: tuple, expected: bool
) -> None:
    assert candidate(category, span(*found), span(*labelled)) is expected


def protocol_rows() -> dict[str, tuple[float, float]]:
    """(tolerance, max length) per category from the § 6 table of the protocol text."""
    rows = {}
    for line in PROTOCOL.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\| `(\w+)` \|[^|]+\| ≤ ([\d,]+) с[^|]*\| (\d+) с \|", line)
        if match:
            rows[match[1]] = (float(match[2].replace(",", ".")), float(match[3]))
    return rows


def test_constants_mirror_the_protocol_text() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert protocol_rows() == {
        name: (rule.tolerance_s, rule.max_length_s) for name, rule in runner.MATCH_RULES.items()
    }
    assert "| дикция | ≥ 0,80 | ≥ 0,70 | 0 уверенных выдумок |" in text
    assert "| просодия | ≥ 0,80 | ≥ 0,70 | 0 уверенных выдумок |" in text
    assert (runner.PRECISION_MIN, runner.RECALL_MIN) == (0.80, 0.70)
    assert runner.MAX_CONFIDENT_FABRICATIONS == 0
    assert "пересечение ≥ 50 % длительности события" in text and runner.MIN_OVERLAP_SHARE == 0.5
    assert "момент конца фразы ± 0,5 с" in text and runner.END_TOLERANCE_S == 0.5
    assert "finding содержит середину паузы" in text
    assert runner.MATCH_RULES["pause_phrasing"].hit == "midpoint"
    assert runner.MATCH_RULES["completeness"].hit == "end"


# --- § 10 statuses ------------------------------------------------------------------------------


def test_overall_flag_is_true_only_when_both_groups_are() -> None:
    both = qualification({"diction": True, "prosody": True}, [])
    assert both == {
        "diction_validated_ru": True,
        "prosody_validated_ru": True,
        "validated_ru": True,
    }
    one = qualification({"diction": True, "prosody": False}, [])
    assert one == {
        "diction_validated_ru": True,
        "prosody_validated_ru": False,
        "validated_ru": False,
    }


def test_any_disqualifying_reason_clears_every_flag() -> None:
    flags = qualification({"diction": True, "prosody": True}, ["прогон на dev-наборе"])
    assert not any(flags.values())


# --- the corpus, made through FakeBackend -------------------------------------------------------


class Corpus:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.root, self.data = base / "recordings", base / "data"
        self.root.mkdir()
        self.data.mkdir()
        self.config = Config(allowed_roots=(self.root,), data_root=self.data)
        self.labels, self.results = base / "labels", base / "results"
        self.labels.mkdir()
        self.results.mkdir()
        self.records: list[dict] = []
        self.docs: dict[str, dict] = {}

    def add(
        self,
        clip_id: str,
        split: str,
        categories: list[str],
        events: list[dict],
        observations: list[dict],
        *,
        raw_segments: list[dict] | None = None,
        assessed: bool = True,
        control_kind: str = "silence",
        clear_correction: bool = False,
    ) -> None:
        control = not categories
        record = {
            "id": clip_id,
            "path": f"audio/{clip_id}.wav",
            "sha256": sha(clip_id),
            "duration_s": DURATION,
            "split": split,
            "source_group": f"g-{clip_id}",
            "categories": categories,
            "negative_control": control,
        }
        if control:
            record["control_kind"] = control_kind
        self.records.append(record)
        if not control:
            self.write_labels(clip_id, events)
        self.write_result(clip_id, observations, raw_segments, assessed, clear_correction)

    def write_labels(self, clip_id: str, events: list[dict]) -> None:
        labels = {
            "schema_version": "1.0",
            "clip_id": clip_id,
            "clip_sha256": sha(clip_id),
            "clip_duration_s": DURATION,
            "annotator": "listener-a",
            "annotated_at": "2026-11-18",
            "model_outputs_seen": False,
            "events": events,
        }
        (self.labels / f"{clip_id}.json").write_text(json.dumps(labels), encoding="utf-8")

    def write_result(
        self,
        clip_id: str,
        observations: list[dict],
        raw_segments,
        assessed: bool,
        clear_correction: bool = False,
    ) -> None:
        run = open_run(f"req-{clip_id}", IDENTITY, self.data)
        t = np.arange(int(DURATION * RATE)) / RATE
        tone = (0.25 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
        ingest.decode(write_wav(self.root / f"{clip_id}.wav", tone, RATE), run, self.config)
        clip = cut(run, 0.0, DURATION, config=self.config)
        response = {
            "observations": observations,
            "unlocalized_observations": [],
            "uncertainties": [],
            "limitations": [],
        }
        outcome = review.run_review(
            FakeBackend(text=json.dumps(response, ensure_ascii=False)),
            review.clip_from_cut(clip),
            run,
            self.config,
            review.load_prompt(),
        )
        transcript = None
        if raw_segments is not None:
            (run.path / "asr").mkdir()
            raw = {"kind": "transcript_raw", "segments": raw_segments}
            (run.path / "asr" / "raw.json").write_text(json.dumps(raw), "utf-8")
            transcript = asr.read_raw(run)
        built = findings.build(run, [outcome], [], transcript, config=self.config)
        doc = document(run, built, outcome)
        doc["source"].update(id=clip_id, sha256=sha(clip_id), duration_s=DURATION)
        if transcript is not None and doc["transcript"]["raw"] is None:
            raw_path = run.path / "asr" / "raw.json"
            doc["artifacts"].append(
                {
                    "id": "transcript-raw",
                    "kind": "transcript_raw",
                    "path": "asr/raw.json",
                    "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "source_start": None,
                    "source_end": None,
                }
            )
            doc["transcript"]["raw"] = {"artifact_ref": "transcript-raw"}
        if transcript is not None:
            doc["transcript"]["suspect_segments"] = asr.suspect_segments(transcript, DURATION)
        if clear_correction:
            ref = doc["transcript"]["raw"]["artifact_ref"]
            raw_sha256 = next(a["sha256"] for a in doc["artifacts"] if a["id"] == ref)
            doc["transcript"]["corrections"].append(
                {
                    "start": 1.0,
                    "end": 2.0,
                    "original": "слово",
                    "corrected": "другое слово",
                    "source": "transcript",
                    "evidence_refs": [{"type": "transcript", "ref": ref, "sha256": raw_sha256}],
                    "certainty": "clear",
                    "note": None,
                }
            )
        status = {"status": "assessed", "reason": None}
        if not assessed:
            status = {"status": "not_assessed", "reason": "синтетика: отказ"}
        doc["assessment_status"].update(diction=status, prosody=dict(status))
        doc["status"] = "complete"
        assert findings.validation_errors(doc) == []
        (run.path / "result.json").write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
        (self.results / clip_id).symlink_to(run.path, target_is_directory=True)
        self.docs[clip_id] = doc

    def finding_id(self, clip_id: str, category: str, start: float) -> str:
        return next(
            f["id"]
            for f in self.docs[clip_id]["findings"]
            if f["category"] == category and abs(f["start"] - start) < 1e-6
        )

    def write_manifest(self) -> Path:
        path = self.base / "manifest.json"
        manifest = {"schema_version": "1.0", "name": "synthetic", "records": self.records}
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def write_agreements(
        self, judgements: list[tuple[str, str, float, int, bool]], name: str = "agreements.json"
    ) -> Path:
        path = self.base / name
        items = [
            {
                "clip_id": clip_id,
                "finding_id": self.finding_id(clip_id, category, start),
                "event": index,
                "agrees": agrees,
                "result_sha256": file_sha(self.results / clip_id / "result.json"),
                "labels_sha256": file_sha(self.labels / f"{clip_id}.json"),
            }
            for clip_id, category, start, index, agrees in judgements
        ]
        path.write_text(json.dumps({"judgements": items}), encoding="utf-8")
        return path

    def evaluate(self, split: str = "heldout", agreements: Path | None = None) -> dict:
        return evaluate(
            self.write_manifest(),
            self.labels,
            self.results,
            split,
            agreements_path=agreements,
            today=TODAY,
        )


JUDGEMENTS = [
    ("h-diction", "pronunciation", 1.9, 0, True),
    ("h-diction", "repetition_self_correction", 4.8, 1, True),
    ("h-prosody", "semantic_stress", 0.9, 0, False),
    ("h-prosody", "pause_phrasing", 3.2, 1, True),
    ("h-prosody", "completeness", 7.6, 2, True),
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are not installed")
    corpus = Corpus(tmp_path_factory.mktemp("eval"))
    corpus.add(
        "h-diction",
        "heldout",
        ["pronunciation", "repetition_self_correction"],
        [
            event("pronunciation", 2.0, 2.5),
            event("repetition_self_correction", 5.0, 6.0),
            event("pronunciation", 8.0, 8.4, "uncertain"),
        ],
        [
            observation("pronunciation", 1.9, 2.6),
            observation("repetition_self_correction", 4.8, 6.2),
            observation("pronunciation", 8.0, 8.3),
        ],
    )
    corpus.add(
        "h-prosody",
        "heldout",
        ["semantic_stress", "pause_phrasing", "completeness"],
        [
            event("semantic_stress", 1.0, 1.5),
            event("pause_phrasing", 3.0, 4.0),
            event("completeness", 7.0, 8.0),
        ],
        [
            observation("semantic_stress", 0.9, 1.6),
            observation("pause_phrasing", 1.0, 1.5),
            observation("pause_phrasing", 3.2, 3.8),
            observation("completeness", 7.6, 8.3),
        ],
    )
    corpus.add(
        "h-long",
        "heldout",
        ["pronunciation"],
        [event("pronunciation", 1.0, 3.4)],
        [observation("pronunciation", 0.6, 3.8)],
    )
    corpus.add(
        "h-abstain",
        "heldout",
        ["completeness"],
        [event("completeness", 4.0, 5.0)],
        [],
        assessed=False,
    )
    corpus.add(
        "d-dev",
        "dev",
        ["pronunciation"],
        [event("pronunciation", 2.0, 2.5)],
        [observation("pronunciation", 2.0, 2.5), observation("pronunciation", 6.0, 6.5)],
    )
    corpus.add(
        "c-silence",
        "heldout",
        [],
        [],
        [observation("pronunciation", 4.0, 4.5)],
        raw_segments=[{"text": "слово", "words": [], "suspect_hallucination": False}],
    )
    corpus.add(
        "c-dev-noise",
        "dev",
        [],
        [],
        [observation("completeness", 4.0, 5.0)],
        raw_segments=[
            {
                "start": 0.5,
                "end": 1.5,
                "text": "шум",
                "words": [],
                "hallucination_signals": ["no_speech_prob"],
                "suspect_hallucination": True,
            }
        ],
        control_kind="noise",
    )
    return corpus


@pytest.fixture(scope="module")
def report(corpus: Corpus) -> dict:
    return corpus.evaluate(agreements=corpus.write_agreements(JUDGEMENTS))


def counts(report: dict, category: str) -> tuple[int, int, int]:
    c = report["categories"][category]
    return c["tp"], c["fp"], c["fn"]


@needs_ffmpeg
def test_matching_counts_tp_fp_fn_per_category(report: dict) -> None:
    assert counts(report, "pronunciation") == (1, 2, 1)  # h-long too long, control finding
    assert counts(report, "repetition_self_correction") == (1, 0, 0)
    assert counts(report, "semantic_stress") == (0, 1, 1)  # observation disagrees
    assert counts(report, "pause_phrasing") == (1, 1, 0)  # a finding of another category's place
    assert counts(report, "completeness") == (1, 0, 1)  # the abstained clip's event is missed
    pronunciation = report["categories"]["pronunciation"]
    assert (pronunciation["events"], pronunciation["clear_events"]) == (3, 2)
    assert (pronunciation["uncertain_events"], pronunciation["uncertain_matches"]) == (1, 1)


@needs_ffmpeg
def test_groups_report_precision_recall_and_abstention_separately(report: dict) -> None:
    diction, prosody = report["groups"]["diction"], report["groups"]["prosody"]
    assert (diction["tp"], diction["fp"], diction["fn"]) == (2, 2, 1)
    assert diction["precision"] == pytest.approx(0.5)
    assert diction["recall"] == pytest.approx(2 / 3)
    assert (prosody["tp"], prosody["fp"], prosody["fn"]) == (2, 2, 2)
    assert prosody["precision"] == pytest.approx(0.5)
    assert prosody["recall"] == pytest.approx(0.5)
    assert prosody["abstention"] == {"clips": 1, "of": 2, "share": 0.5}
    assert diction["abstention"] == {"clips": 0, "of": 2, "share": 0.0}
    assert report["categories"]["completeness"]["abstention_clips"] == 1
    assert not diction["passed"] and not prosody["passed"]


@needs_ffmpeg
def test_a_finding_longer_than_the_category_maximum_is_not_a_match(report: dict) -> None:
    long = [f for f in report["failures"]["false_findings"] if f["clip_id"] == "h-long"]
    assert [f["why"] for f in long] == ["too_long"]
    assert any(e["clip_id"] == "h-long" for e in report["failures"]["missed_events"])


@needs_ffmpeg
def test_dev_clips_stay_out_of_a_heldout_run(report: dict) -> None:
    referenced = {f["clip_id"] for f in report["failures"]["false_findings"]}
    referenced |= {e["clip_id"] for e in report["failures"]["missed_events"]}
    assert "d-dev" not in referenced
    assert report["clips"] == {"scored": 5, "negative_controls": 2}


@needs_ffmpeg
def test_negative_controls_count_confident_fabrications_in_both_splits(report: dict) -> None:
    assert report["negative_controls"] == {
        "clips": 2,
        "confident_fabrications": {"diction": 1, "prosody": 1, "transcript": 1},
    }
    kinds = {(f["clip_id"], f["kind"]) for f in report["failures"]["control_fabrications"]}
    assert kinds == {
        ("c-silence", "finding"),
        ("c-silence", "raw_transcript_words"),
        ("c-dev-noise", "finding"),
    }
    assert any("уверенных выдумок" in reason for reason in report["groups"]["diction"]["failed"])


CONTROL_RAW = [{"text": "слово", "words": [], "suspect_hallucination": False}]


def control_report(
    base: Path, kind: str, *, findings: bool = False, clear_correction: bool = False
) -> dict:
    base.mkdir()
    corpus = Corpus(base)
    observations = [observation("pronunciation", 4.0, 4.5)] if findings else []
    corpus.add(
        "c-control",
        "heldout",
        [],
        [],
        observations,
        raw_segments=CONTROL_RAW,
        control_kind=kind,
        clear_correction=clear_correction,
    )
    return corpus.evaluate()


def fabrication_kinds(report: dict) -> list[str]:
    return [f["kind"] for f in report["failures"]["control_fabrications"]]


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["silence", "noise"])
def test_silence_and_noise_controls_count_raw_words_findings_and_clear_corrections(
    tmp_path: Path, kind: str
) -> None:
    words = control_report(tmp_path / "words", kind)
    assert words["negative_controls"]["confident_fabrications"] == {
        "diction": 0,
        "prosody": 0,
        "transcript": 1,
    }
    assert fabrication_kinds(words) == ["raw_transcript_words"]
    everything = control_report(tmp_path / "all", kind, findings=True, clear_correction=True)
    assert everything["negative_controls"]["confident_fabrications"] == {
        "diction": 1,
        "prosody": 0,
        "transcript": 2,
    }
    assert sorted(fabrication_kinds(everything)) == [
        "clear_correction",
        "finding",
        "raw_transcript_words",
    ]


@needs_ffmpeg
def test_corrupted_asr_control_counts_findings_and_clear_corrections_not_raw_words(
    tmp_path: Path,
) -> None:
    words = control_report(tmp_path / "words", "corrupted_asr")
    assert words["negative_controls"] == {
        "clips": 1,
        "confident_fabrications": {"diction": 0, "prosody": 0, "transcript": 0},
    }
    assert fabrication_kinds(words) == []
    corrected = control_report(tmp_path / "corrected", "corrupted_asr", clear_correction=True)
    assert corrected["negative_controls"]["confident_fabrications"] == {
        "diction": 0,
        "prosody": 0,
        "transcript": 1,
    }
    assert fabrication_kinds(corrected) == ["clear_correction"]
    found = control_report(tmp_path / "found", "corrupted_asr", findings=True)
    assert found["negative_controls"]["confident_fabrications"]["diction"] == 1
    assert fabrication_kinds(found) == ["finding"]


def test_confident_raw_segments_are_the_trusted_ones_by_their_index(tmp_path: Path) -> None:
    raw_path = tmp_path / "asr" / "raw.json"
    raw_path.parent.mkdir()
    result = {
        "transcript": {"raw": {"artifact_ref": "transcript-raw"}},
        "artifacts": [{"id": "transcript-raw", "path": "asr/raw.json"}],
    }
    clip = runner._Clip({"id": "c-control"}, [], result)
    segments = [
        {"text": " ", "suspect_hallucination": False},
        {"text": "слово", "suspect_hallucination": False},
        {
            "text": "выдумка",
            "suspect_hallucination": True,
            "hallucination_signals": ["silent_span"],
        },
        {"text": "ещё", "suspect_hallucination": False},
    ]
    raw_path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    reasons: list[str] = []
    assert runner._confident_raw_segments(clip, tmp_path, reasons) == [1, 3]
    assert reasons == []

    del segments[1]["suspect_hallucination"]
    raw_path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    assert runner._confident_raw_segments(clip, tmp_path, reasons) == []
    assert reasons == [
        "контроль 'c-control': сырой транскрипт не проверен (ValueError), § 9 не подтверждён"
    ]


@needs_ffmpeg
def test_a_control_whose_raw_segment_has_no_flag_is_not_confirmed(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path)
    corpus.add("c-control", "heldout", [], [], [], raw_segments=CONTROL_RAW)
    results = tmp_path / "copied-results"
    shutil.copytree(corpus.results, results, symlinks=False)
    raw = results / "c-control" / "asr" / "raw.json"
    raw.chmod(0o600)
    raw.write_text(json.dumps({"segments": [{"text": "слово", "words": []}]}), encoding="utf-8")
    result_path = results / "c-control" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    artifact = next(a for a in result["artifacts"] if a["id"] == "transcript-raw")
    artifact["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    report = evaluate(corpus.write_manifest(), corpus.labels, results, "heldout", today=TODAY)

    assert report["negative_controls"]["confident_fabrications"]["transcript"] == 0
    assert any(
        "результат клипа 'c-control' не подтверждён файлами run" in r
        for r in report["not_qualified_reasons"]
    )


@needs_ffmpeg
def test_a_control_without_a_kind_counts_raw_words_and_fails_closed(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path)
    corpus.add("c-control", "heldout", [], [], [], raw_segments=CONTROL_RAW)
    del corpus.records[0]["control_kind"]
    report = corpus.evaluate()
    assert report["negative_controls"]["confident_fabrications"]["transcript"] == 1
    assert fabrication_kinds(report) == ["raw_transcript_words"]


@needs_ffmpeg
def test_test_double_results_are_never_validated(report: dict) -> None:
    assert not report["validated_ru"]
    assert not report["diction_validated_ru"] and not report["prosody_validated_ru"]
    assert any("test-double" in reason for reason in report["not_qualified_reasons"])
    assert report["settings"][0]["backend"] == "test-double"


@needs_ffmpeg
def test_candidates_without_a_judgement_are_pending_and_never_matched(corpus: Corpus) -> None:
    report = corpus.evaluate()
    assert len(report["pending_agreement"]) == len(JUDGEMENTS)
    assert sum(report["groups"][g]["tp"] for g in ("diction", "prosody")) == 0
    assert any("ждут решения человека" in r for r in report["not_qualified_reasons"])
    why = {f["why"] for f in report["failures"]["false_findings"] if f["clip_id"] == "h-prosody"}
    assert "agreement_pending" in why


@needs_ffmpeg
def test_a_judgement_of_no_candidate_pair_is_reported(corpus: Corpus, tmp_path: Path) -> None:
    path = tmp_path / "agreements.json"
    stale = {**JUDGED, "clip_id": "h-diction", "finding_id": "f-nowhere"}
    path.write_text(json.dumps({"judgements": [stale]}), encoding="utf-8")
    report = corpus.evaluate(agreements=path)
    assert any("не пара-кандидат" in r for r in report["not_qualified_reasons"])


@needs_ffmpeg
def test_dev_run_is_not_qualified(corpus: Corpus) -> None:
    report = corpus.evaluate("dev")
    assert report["clips"]["scored"] == 2
    assert any("dev-наборе" in r for r in report["not_qualified_reasons"])


@needs_ffmpeg
def test_the_runner_is_deterministic(corpus: Corpus, report: dict) -> None:
    again = corpus.evaluate(agreements=corpus.base / "agreements.json")
    assert json.dumps(again, sort_keys=True) == json.dumps(report, sort_keys=True)


@needs_ffmpeg
def test_missing_or_foreign_inputs_disqualify_and_are_named(corpus: Corpus, tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    shutil.copytree(corpus.labels, labels)
    (labels / "h-long.json").unlink()
    foreign = json.loads((labels / "h-prosody.json").read_text(encoding="utf-8"))
    foreign["clip_sha256"] = sha("another clip")
    (labels / "h-prosody.json").write_text(json.dumps(foreign), encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    for path in corpus.results.iterdir():
        if path.name != "h-diction":
            (results / path.name).symlink_to(path.resolve(), target_is_directory=True)
    report = evaluate(corpus.write_manifest(), labels, results, "heldout", today=TODAY)
    reasons = "\n".join(report["not_qualified_reasons"])
    assert "разметка клипа 'h-long' не принята" in reasons
    assert "разметка клипа 'h-prosody' описывает другой клип" in reasons
    assert "результат клипа 'h-diction' не прочитан" in reasons
    assert not report["validated_ru"]


@needs_ffmpeg
def test_report_files_carry_the_warning_flags_and_failures(report: dict, tmp_path: Path) -> None:
    markdown_path, json_path = write_report(report, tmp_path / "out")
    assert markdown_path.name == "2026-11-18-report.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    text = markdown_path.read_text(encoding="utf-8")
    assert "Малый пилот, не статистическое доказательство" in text
    for flag in ("validated_ru", "diction_validated_ru", "prosody_validated_ru"):
        assert f"`{flag}`: `false`" in text
    assert "`h-long`" in text and "too_long" in text and "`c-silence`" in text
    assert "синтетическое" not in text  # observation and label texts stay out of the report
    with pytest.raises(EvalError, match="already exists"):
        write_report(report, tmp_path / "out")


# --- inputs the runner refuses ------------------------------------------------------------------


def test_unreadable_or_invalid_manifest_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="cannot be read"):
        evaluate(tmp_path / "absent.json", tmp_path, tmp_path, "heldout")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "1.0", "name": "x"}), encoding="utf-8")
    with pytest.raises(EvalError, match="does not match its schema"):
        evaluate(path, tmp_path, tmp_path, "heldout")


def test_unknown_split_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="split must be one of"):
        evaluate(tmp_path / "m.json", tmp_path, tmp_path, "test")


JUDGED = {
    "clip_id": "a",
    "finding_id": "f",
    "event": 0,
    "agrees": True,
    "result_sha256": sha("result"),
    "labels_sha256": sha("labels"),
}


@pytest.mark.parametrize(
    ("agreements", "message"),
    [
        ({"judgements": {}}, "must be a list"),
        ({"judgements": [{"clip_id": "a"}]}, "must have exactly"),
        (
            {"judgements": [{**JUDGED, "event": True}]},
            "event must be int",
        ),
        (
            {"judgements": [{**JUDGED, "labels_sha256": 1}]},
            "labels_sha256 must be str",
        ),
        (
            {"judgements": [JUDGED] * 2},
            "repeats",
        ),
    ],
)
def test_malformed_agreements_are_refused(tmp_path: Path, agreements: dict, message: str) -> None:
    path = tmp_path / "agreements.json"
    path.write_text(json.dumps(agreements), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "1.0", "name": "x", "records": []}), encoding="utf-8"
    )
    with pytest.raises(EvalError, match=message):
        evaluate(manifest, tmp_path, tmp_path, "heldout", agreements_path=path)


# --- `sound-scout eval run` ---------------------------------------------------------------------


@needs_ffmpeg
def test_cli_eval_run_writes_the_report_and_prints_the_flags(
    corpus: Corpus, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "eval",
        "run",
        "--manifest",
        str(corpus.write_manifest()),
        "--labels",
        str(corpus.labels),
        "--results",
        str(corpus.results),
        "--split",
        "heldout",
        "--agreements",
        str(corpus.base / "agreements.json"),
        "--output-dir",
        str(tmp_path),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "validated_ru: false" in out and "prosody_validated_ru: false" in out
    reports = sorted(p.suffix for p in tmp_path.glob("*-report.*"))
    assert reports == [".json", ".md"]
    assert main(argv) == 1
    assert "already exists" in capsys.readouterr().err


def test_cli_eval_run_reports_an_unreadable_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["eval", "run", "--manifest", str(tmp_path / "absent.json"), "--labels", "l"]
    argv += ["--results", "r", "--split", "heldout", "--output-dir", str(tmp_path / "out")]
    assert main(argv) == 1
    assert capsys.readouterr().err.startswith("eval error: manifest cannot be read")
    assert not (tmp_path / "out").exists()


def test_cli_eval_run_requires_its_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["eval", "run", "--split", "heldout"])
    assert excinfo.value.code == 2
    assert "--manifest" in capsys.readouterr().err


@needs_ffmpeg
def test_a_control_whose_raw_transcript_does_not_verify_disqualifies(
    corpus: Corpus, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    shutil.copytree(corpus.results, results, symlinks=False)
    raw = results / "c-silence" / "asr" / "raw.json"
    raw.chmod(0o600)
    raw.write_text(json.dumps({"kind": "transcript_raw", "segments": []}), encoding="utf-8")
    report = evaluate(corpus.write_manifest(), corpus.labels, results, "heldout", today=TODAY)
    assert report["negative_controls"]["confident_fabrications"]["transcript"] == 0
    assert any(
        "результат клипа 'c-silence' не подтверждён файлами run" in r
        for r in report["not_qualified_reasons"]
    )


# --- verifier: the report a reader sees and a group that meets every threshold ------------------


@needs_ffmpeg
def test_report_markdown_lists_counts_per_category_and_per_group(
    report: dict, tmp_path: Path
) -> None:
    markdown_path, _ = write_report(report, tmp_path)
    text = markdown_path.read_text(encoding="utf-8")
    assert "| pronunciation | 1 | 2 | 1 | 3 | 1 | 1 | 0 из 2 | 0 |" in text
    assert "| completeness | 1 | 0 | 1 | 2 | 0 | 0 | 1 из 2 | 0 |" in text
    assert "| diction | 2 | 2 | 1 | 3 | 0.50 | 0.67 | 0 из 2 |" in text
    assert "| prosody | 2 | 2 | 2 | 4 | 0.50 | 0.50 | 1 из 2 |" in text
    assert "дикция: 1, просодия: 1, транскрипт: 1 (порог — 0 в каждой строке)" in text
    assert "- `h-abstain` событие 0 `completeness` 4.0–5.0 с" in text


@needs_ffmpeg
def test_overall_flag_needs_both_groups_when_one_group_meets_every_threshold(
    tmp_path: Path,
) -> None:
    good = Corpus(tmp_path)
    spans = [
        ("pronunciation", 0.5, 1.0),
        ("pronunciation", 2.0, 2.5),
        ("pronunciation", 4.0, 4.5),
        ("repetition_self_correction", 6.0, 7.0),
        ("repetition_self_correction", 8.0, 9.0),
    ]
    good.add(
        "h-good",
        "heldout",
        ["pronunciation", "repetition_self_correction"],
        [event(name, start, end) for name, start, end in spans],
        [observation(name, start, end) for name, start, end in spans],
    )
    good.add("c-quiet", "heldout", [], [], [])
    judgements = [("h-good", name, start, i, True) for i, (name, start, _) in enumerate(spans)]
    report = good.evaluate(agreements=good.write_agreements(judgements))
    diction, prosody = report["groups"]["diction"], report["groups"]["prosody"]
    assert (diction["tp"], diction["fp"], diction["fn"]) == (5, 0, 0)
    assert (diction["precision"], diction["recall"]) == (1.0, 1.0)
    assert diction["passed"] and not prosody["passed"]
    assert report["negative_controls"]["confident_fabrications"] == {
        "diction": 0,
        "prosody": 0,
        "transcript": 0,
    }
    assert qualification({g: s["passed"] for g, s in report["groups"].items()}, []) == {
        "diction_validated_ru": True,
        "prosody_validated_ru": False,
        "validated_ru": False,
    }
    # The synthetic corpus is below the manifest minimums and made by a test double: no flag.
    assert not any(report[flag] for flag in ("validated_ru", "diction_validated_ru"))
    markdown_path, _ = write_report(report, tmp_path / "out")
    assert "- **diction:** проходит § 7 и § 9." in markdown_path.read_text(encoding="utf-8")


# --- review findings: inputs that disqualify instead of being scored ----------------------------


def copied_results(corpus: Corpus, tmp_path: Path) -> Path:
    results = tmp_path / "results"
    shutil.copytree(corpus.results, results, symlinks=False)
    return results


def rewrite_result(results: Path, clip_id: str, change) -> None:
    path = results / clip_id / "result.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.chmod(0o600)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def reasons_of(corpus: Corpus, results: Path) -> str:
    report = evaluate(corpus.write_manifest(), corpus.labels, results, "heldout", today=TODAY)
    return "\n".join(report["not_qualified_reasons"])


@needs_ffmpeg
def test_results_of_different_settings_disqualify(
    corpus: Corpus, report: dict, tmp_path: Path
) -> None:
    assert len(report["settings"]) == 1
    assert not any("разными настройками" in r for r in report["not_qualified_reasons"])
    results = copied_results(corpus, tmp_path)
    rewrite_result(results, "h-long", lambda doc: doc["provenance"].update(revision="other"))
    assert "результаты получены 2 разными настройками" in reasons_of(corpus, results)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("clip_id", "change", "message"),
    [
        (
            "c-silence",
            lambda doc: doc.update(status="partial"),
            "результат клипа 'c-silence' не завершён (status 'partial', не выполнены этапы: нет)",
        ),
        (
            "h-abstain",
            lambda doc: doc["stages"]["asr"].update(
                requested=False, status="skipped", reason="синтетика: не запрошен"
            ),
            "результат клипа 'h-abstain' не завершён (status 'complete', не выполнены этапы: asr)",
        ),
    ],
)
def test_an_unfinished_result_is_refused(
    corpus: Corpus, report: dict, tmp_path: Path, clip_id: str, change, message: str
) -> None:
    assert not any("не завершён" in r for r in report["not_qualified_reasons"])
    results = copied_results(corpus, tmp_path)
    rewrite_result(results, clip_id, change)
    assert message in reasons_of(corpus, results)


@needs_ffmpeg
def test_a_finding_not_based_on_hearing_is_not_scored(corpus: Corpus, tmp_path: Path) -> None:
    results = copied_results(corpus, tmp_path)
    heard = corpus.finding_id("h-diction", "pronunciation", 1.9)
    target = "human-verified-1"
    label = results / "h-diction" / "human-label.json"
    label.write_text("{}", encoding="utf-8")

    def human_verified(doc: dict) -> None:
        digest = file_sha(label)
        doc["artifacts"].append(
            {
                "id": "human-label",
                "kind": "human_label",
                "path": "human-label.json",
                "sha256": digest,
                "source_start": None,
                "source_end": None,
            }
        )
        # a copy next to the backend's own finding: that one must stay as its response made it
        finding = json.loads(json.dumps(next(f for f in doc["findings"] if f["id"] == heard)))
        finding.update(id=target, evidence_basis="human_verified")
        finding["evidence_refs"].append(
            {"type": "human_label", "ref": "human-label", "sha256": digest}
        )
        doc["findings"].append(finding)

    rewrite_result(results, "h-diction", human_verified)
    report = evaluate(corpus.write_manifest(), corpus.labels, results, "heldout", today=TODAY)
    reasons = "\n".join(report["not_qualified_reasons"])
    assert f"результат клипа 'h-diction': finding {target!r} основан не на прослушивании" in reasons
    scored = report["pending_agreement"] + report["failures"]["false_findings"]
    assert ("h-diction", target) not in {(f["clip_id"], f["finding_id"]) for f in scored}
    assert not any(
        "основан не на прослушивании" in r for r in reasons_of(corpus, corpus.results).splitlines()
    )


@needs_ffmpeg
def test_a_judgement_made_on_other_content_is_not_reused(corpus: Corpus) -> None:
    path = corpus.write_agreements(JUDGEMENTS, "outdated.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["judgements"][0]["labels_sha256"] = sha("earlier labels")
    path.write_text(json.dumps(document), encoding="utf-8")
    report = corpus.evaluate(agreements=path)
    stale = document["judgements"][0]
    assert [(p["clip_id"], p["finding_id"], p["event"]) for p in report["pending_agreement"]] == [
        (stale["clip_id"], stale["finding_id"], stale["event"])
    ]
    assert report["categories"]["pronunciation"]["tp"] == 0
    assert any("решение принято о другом результате" in r for r in report["not_qualified_reasons"])
    assert not any("не пара-кандидат" in r for r in report["not_qualified_reasons"])


@needs_ffmpeg
def test_a_judgement_of_an_uncertain_pair_is_ignored(corpus: Corpus, report: dict) -> None:
    path = corpus.write_agreements(
        [*JUDGEMENTS, ("h-diction", "pronunciation", 8.0, 2, True)], "u.json"
    )
    again = corpus.evaluate(agreements=path)
    assert again["not_qualified_reasons"] == report["not_qualified_reasons"]
    assert again["categories"] == report["categories"]


@pytest.fixture(scope="module")
def rivals(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are not installed")
    corpus = Corpus(tmp_path_factory.mktemp("rivals"))
    corpus.add(
        "h-rivals",
        "heldout",
        ["pronunciation", "repetition_self_correction"],
        [event("pronunciation", 2.0, 3.0), event("repetition_self_correction", 6.0, 7.0)],
        [observation("pronunciation", 2.4, 3.1), observation("pronunciation", 1.9, 2.8)],
    )
    corpus.add(
        "h-other",
        "heldout",
        ["pronunciation"],
        [event("pronunciation", 2.0, 2.5)],
        [observation("repetition_self_correction", 5.0, 6.0)],
    )
    judgements = [("h-rivals", "pronunciation", start, 0, True) for start in (2.4, 1.9)]
    report = corpus.evaluate(agreements=corpus.write_agreements(judgements))
    return {
        "report": report,
        "larger": corpus.finding_id("h-rivals", "pronunciation", 1.9),
        "smaller": corpus.finding_id("h-rivals", "pronunciation", 2.4),
    }


@needs_ffmpeg
def test_pairs_are_assigned_one_to_one_by_largest_overlap(rivals: dict) -> None:
    report = rivals["report"]
    assert counts(report, "pronunciation") == (1, 1, 1)  # h-other's event is missed
    false = report["failures"]["false_findings"]
    assert [
        (f["clip_id"], f["finding_id"], f["why"]) for f in false if f["category"] == "pronunciation"
    ] == [("h-rivals", rivals["smaller"], "event_already_matched")]
    assert report["pending_agreement"] == []


@needs_ffmpeg
def test_group_abstention_needs_no_finding_of_the_whole_group(rivals: dict) -> None:
    report = rivals["report"]
    # h-other carries pronunciation only and has a repetition finding: the group did not abstain.
    assert report["groups"]["diction"]["abstention"] == {"clips": 0, "of": 2, "share": 0.0}
    repetition = report["categories"]["repetition_self_correction"]
    assert (repetition["abstention_clips"], repetition["clips"]) == (1, 1)
    pronunciation = report["categories"]["pronunciation"]
    assert (pronunciation["abstention_clips"], pronunciation["clips"]) == (1, 2)
