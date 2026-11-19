"""`report`: full `result.json` and Russian `report.md`, the compact answer, the priority rule,
cost semantics and private values kept out (issue #32)."""

import getpass
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from sound_scout import findings, ingest, report, schema
from sound_scout.clips import Clip, cut
from sound_scout.config import Config
from sound_scout.findings import build
from sound_scout.report import AUDIO_MODEL_PHRASE, COMPACT_MAX_BYTES, ReportError
from sound_scout.runs import Run, open_run
from sound_scout.schema import ResultValidationError
from tests.test_clips import RATE, burst
from tests.test_findings import (
    IDENTITY,
    RESPONSE,
    SOURCE_DURATION,
    SUSPECT_WORD,
    document,
    raw_segment,
    reviewed,
    with_raw_artifact,
    write_raw,
)
from tests.test_ingest import write_wav
from tests.test_schema import EXAMPLE_DOC

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)

NOT_ASSESSED_REASON = EXAMPLE_DOC["assessment_status"]["diction"]["reason"]
ABSOLUTE_PATH = re.compile(r"(?<![\w.:/\\~-])/[^\s/]+(?:/[^\s/]*)+|[A-Za-z]:\\")
BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{120,}")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    root, data_root = tmp_path / "recordings", tmp_path / "data"
    root.mkdir()
    data_root.mkdir()
    return Config(allowed_roots=(root,), data_root=data_root)


@pytest.fixture
def run(config: Config) -> Run:
    run = open_run("req-1", IDENTITY, config.data_root)
    source = write_wav(config.allowed_roots[0] / "burst.wav", burst(), RATE)
    ingest.decode(source, run, config)
    return run


@pytest.fixture
def clip(run: Run, config: Config) -> Clip:
    return cut(run, 1.0, 1.5, config=config)


@pytest.fixture
def heard(run: Run, config: Config, clip: Clip) -> dict:
    """A result with a completed (test-double) audio review, as #33 will assemble it."""
    outcome = reviewed(run, config, clip)
    return document(run, build(run, [outcome], [], None, config=config), outcome)


@pytest.fixture
def unheard(run: Run, clip: Clip) -> dict:
    """A result whose audio review is unavailable: one measurement finding with its clip."""
    doc = deepcopy(EXAMPLE_DOC)
    doc["run_id"] = run.run_id
    doc["source"]["duration_s"] = SOURCE_DURATION
    doc["coverage"] = {
        "requested": [{"start": 0.0, "end": SOURCE_DURATION}],
        "reviewed": [],
        "unreviewed": [{"start": 0.0, "end": SOURCE_DURATION}],
    }
    doc["transcript"] = {"raw": None, "corrections": [], "suspect_segments": []}
    doc["measurements"][0]["settings"] = {"window_s": SOURCE_DURATION}
    finding = deepcopy(EXAMPLE_DOC["findings"][0])
    finding.update(start=1.0, end=1.5)
    finding["evidence_refs"][1] = {"type": "clip", "ref": clip.id, "sha256": clip.sha256}
    doc["findings"] = [finding]
    doc["artifacts"] = [clip.artifact]
    schema.validate(doc)
    return doc


def written(run: Run) -> tuple[dict, str]:
    result = json.loads((run.path / "result.json").read_text(encoding="utf-8"))
    return result, (run.path / "report.md").read_text(encoding="utf-8")


def compact_size(answer: dict) -> int:
    return len(json.dumps(answer, ensure_ascii=False).encode("utf-8"))


def strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return []


# --- write ------------------------------------------------------------------------------------


def test_write_puts_the_full_result_and_a_russian_report_in_the_run(
    run: Run, config: Config, heard: dict
) -> None:
    returned = report.write(run, heard, config=config)
    result, markdown = written(run)
    assert result == returned
    findings.validate(result, run=run)
    kinds = {a["kind"]: a["path"] for a in result["artifacts"]}
    assert kinds["result_json"] == "result.json" and kinds["report_md"] == "report.md"
    assert result["findings"] == heard["findings"]
    assert result["unlocalized"] == heard["unlocalized"]
    for heading in ("## Этапы", "## Охват", "## Измерения", "## Транскрипт", "## Ограничения"):
        assert heading in markdown
    for heading in ("## Пределы", "## Ошибки", "## Стоимость", "## Оценка по слоям"):
        assert heading in markdown
    assert "Статус: `partial`" in markdown
    for finding in heard["findings"]:
        assert f"`{finding['id']}`" in markdown
        assert finding["observation"] in markdown
        if finding["correction"]:
            assert f"Действие: {finding['correction']}" in markdown
    for item in heard["unlocalized"]:
        assert f"`{item['id']}`" in markdown
    clip_path = next(a["path"] for a in heard["artifacts"] if a["kind"] == "clip")
    assert f"[`{clip_path}`]({clip_path})" in markdown
    assert f"{AUDIO_MODEL_PHRASE}: {heard['findings'][0]['observation']}" in markdown
    assert "`not_assessed` — не оценивалось" in markdown and NOT_ASSESSED_REASON in markdown
    assert "syllables/s" in markdown and "synthetic: a rate over one window" in markdown


def test_written_result_json_is_full_by_schema_and_report_md_covers_every_section(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, [{"start": 1.1, "end": 1.2, "word": " приме", "probability": 0.9}])
    outcome = reviewed(run, config, clip)
    result = document(run, build(run, [outcome], [], transcript, config=config), outcome)
    result["errors"] = [{"stage": "asr", "code": "asr_failed", "message": "decoder stopped"}]
    result["cost"] = report.cost()

    report.write(run, result, config=config)
    written_result, markdown = written(run)

    schema.validate(written_result)
    for key in result.keys() - {"artifacts", "source"}:
        assert written_result[key] == result[key], key
    assert written_result["artifacts"][: len(result["artifacts"])] == result["artifacts"]
    raw = next(a for a in result["artifacts"] if a["id"] == "transcript-raw")
    assert f"Сырой транскрипт ASR: [`{raw['path']}`]({raw['path']})" in markdown
    measurement = result["measurements"][0]
    assert f"`{measurement['name']}`:" in markdown and measurement["unit"] in markdown
    for limitation in measurement["limitations"]:
        assert f"ограничение: {limitation}" in markdown
    assert f"- содержание: `{result['assessment_status']['content']['status']}`" in markdown
    for name in result["limits"]:
        assert f"- `{name}`:" in markdown
    assert "`asr` / `asr_failed`: decoder stopped" in markdown
    assert "- 0 (`actual`, local)" in markdown
    correction = result["transcript"]["corrections"][0]
    assert f"«{correction['original']}» → «{correction['corrected']}»" in markdown


def test_flagged_asr_segments_are_listed_as_intervals_without_their_text(
    run: Run, config: Config, clip: Clip
) -> None:
    flagged = raw_segment([SUSPECT_WORD], ["silent_span", "no_speech_prob"])
    transcript = write_raw(run, [], [flagged])
    outcome = reviewed(run, config, clip)
    result = with_raw_artifact(
        run,
        document(run, build(run, [outcome], [], transcript, config=config), outcome),
        transcript,
    )

    report.write(run, result, config=config)
    written_result, markdown = written(run)

    assert written_result["transcript"]["suspect_segments"] == [
        {
            "start": 1.1,
            "end": 1.2,
            "signals": ["silent_span", "no_speech_prob"],
            "certainty": "uncertain",
        }
    ]
    assert report.SUSPECT_SEGMENTS_LEAD in markdown
    assert "- 1.10–1.20 с: признаки `silent_span`, `no_speech_prob`." in markdown
    assert markdown.index(report.SUSPECT_SEGMENTS_LEAD) > markdown.index("## Транскрипт")
    assert markdown.index(report.SUSPECT_SEGMENTS_LEAD) < markdown.index("## Наблюдения")
    assert "выдумка" not in markdown
    assert "выдумка" not in json.dumps(written_result, ensure_ascii=False)
    assert "выдумка" not in json.dumps(report.compact(written_result), ensure_ascii=False)


def test_report_without_flagged_segments_has_no_suspect_heading(
    run: Run, config: Config, clip: Clip
) -> None:
    transcript = write_raw(run, [{"start": 1.1, "end": 1.2, "word": " приме", "probability": 0.9}])
    outcome = reviewed(run, config, clip)
    report.write(
        run,
        document(run, build(run, [outcome], [], transcript, config=config), outcome),
        config=config,
    )
    _, markdown = written(run)
    assert report.SUSPECT_SEGMENTS_LEAD not in markdown


def test_report_without_audio_review_says_partial_and_not_assessed(
    run: Run, config: Config, unheard: dict
) -> None:
    report.write(run, unheard, config=config)
    _, markdown = written(run)
    assert "Статус: `partial`" in markdown
    for label in ("дикция", "просодия"):
        assert f"{label}: `not_assessed` — не оценивалось" in markdown
    assert NOT_ASSESSED_REASON in markdown
    assert "аудиомодель отмечает" not in markdown.lower()
    assert "Измерение показывает: Темп речи" in markdown
    assert "| audio review | да | `unavailable` | audio backend is not selected" in markdown
    answer = report.compact(unheard)
    assert answer["status"] == "partial"
    assert answer["stages"]["audio_review"]["status"] == "unavailable"
    for name in ("diction", "prosody"):
        assert answer["assessment_status"][name] == {
            "status": "not_assessed",
            "reason": NOT_ASSESSED_REASON,
        }
    assert any(text.startswith("diction: not_assessed") for text in answer["limitations"])


def test_audio_model_wording_needs_a_completed_audio_review(heard: dict) -> None:
    assert AUDIO_MODEL_PHRASE in report.render(heard)
    heard["stages"]["audio_review"] = {
        "requested": True,
        "status": "failed",
        "reason": "backend crashed",
        "elapsed_ms": None,
    }
    assert AUDIO_MODEL_PHRASE.lower() not in report.render(heard).lower()


def test_write_rejects_an_invalid_result_and_writes_nothing(
    run: Run, config: Config, heard: dict
) -> None:
    broken = deepcopy(heard)
    del broken["cost"]
    with pytest.raises(ResultValidationError):
        report.write(run, broken, config=config)
    missing = deepcopy(heard)
    clip_path = next(a["path"] for a in heard["artifacts"] if a["kind"] == "clip")
    (run.path / clip_path).unlink()
    with pytest.raises(ResultValidationError, match="file is missing"):
        report.write(run, missing, config=config)
    assert not (run.path / "result.json").exists() and not (run.path / "report.md").exists()


def test_write_rejects_a_result_of_another_run(run: Run, config: Config, heard: dict) -> None:
    heard["run_id"] = "another-run"
    with pytest.raises(ReportError, match="another-run"):
        report.write(run, heard, config=config)


def test_write_never_overwrites_a_report(run: Run, config: Config, heard: dict) -> None:
    report.write(run, heard, config=config)
    before = (run.path / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        report.write(run, heard, config=config)
    assert (run.path / "result.json").read_bytes() == before


def test_write_leaves_neither_file_when_the_second_one_fails(
    run: Run, config: Config, heard: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open_output = report.paths.open_output

    def failing_on_report_md(run_dir: Path, name: str, config: Config):
        if name == "report.md":
            raise OSError("disk full")
        return real_open_output(run_dir, name, config)

    monkeypatch.setattr(report.paths, "open_output", failing_on_report_md)
    with pytest.raises(OSError, match="disk full"):
        report.write(run, heard, config=config)
    assert not (run.path / "result.json").exists() and not (run.path / "report.md").exists()
    monkeypatch.setattr(report.paths, "open_output", real_open_output)
    report.write(run, heard, config=config)
    assert (run.path / "result.json").exists() and (run.path / "report.md").exists()


def test_phoneme_notation_passes_write_and_compact_unchanged(
    run: Run, config: Config, clip: Clip
) -> None:
    observation = "Звук /ш/ смазан, /ть/ звучит как /ч/, безударный (/о/) и пара /р/-/л/."
    response = {
        **RESPONSE,
        "observations": [{**obs, "observation": observation} for obs in RESPONSE["observations"]],
    }
    outcome = reviewed(run, config, clip, response)
    heard = document(run, build(run, [outcome], [], None, config=config), outcome)
    returned = report.write(run, heard, config=config)
    result, markdown = written(run)
    assert all(f["observation"] == observation for f in returned["findings"])
    assert all(f["observation"] == observation for f in result["findings"])
    assert observation in markdown
    assert all(f["observation"] == observation for f in report.compact(returned)["findings"])


def test_a_private_path_heard_in_a_response_is_redacted_and_still_matches_it(
    run: Run, config: Config, clip: Clip
) -> None:
    observation = f"модель упоминает {run.path}/notes.txt и путь /srv/media/take.wav"
    response = {
        **RESPONSE,
        "observations": [{**RESPONSE["observations"][0], "observation": observation}],
    }
    outcome = reviewed(run, config, clip, response)
    returned = report.write(
        run, document(run, build(run, [outcome], [], None, config=config), outcome), config=config
    )
    result, _ = written(run)
    assert result["findings"][0]["observation"] == returned["findings"][0]["observation"]
    assert str(run.path) not in result["findings"][0]["observation"]
    assert "<private path>" in result["findings"][0]["observation"]
    findings.validate(result, run=run)


# --- compact ----------------------------------------------------------------------------------


def test_compact_is_small_prioritised_and_linked_relatively(heard: dict) -> None:
    answer = report.compact(heard)
    assert compact_size(answer) <= COMPACT_MAX_BYTES
    assert answer["status"] == heard["status"]
    assert set(answer["stages"]) == set(schema.STAGES)
    assert answer["coverage"]["reviewed"] == heard["coverage"]["reviewed"]
    assert [f["id"] for f in answer["findings"]] == [
        f["id"] for f in report.prioritized(heard["findings"])[:3]
    ]
    assert answer["findings"][0]["polarity"] == "negative"
    assert answer["findings_total"] == len(heard["findings"])
    assert answer["unlocalized_total"] == len(heard["unlocalized"])
    assert answer["links"]["report"] == "report.md" and answer["links"]["result"] == "result.json"
    clip_path = next(a["path"] for a in heard["artifacts"] if a["kind"] == "clip")
    assert answer["links"]["clips"] == [clip_path]
    assert "synthetic" in answer["limitations"][0]


def test_compact_stays_under_4_kib_without_pcm_base64_or_absolute_paths(
    heard: dict, tmp_path: Path
) -> None:
    template = heard["findings"][0]
    heard["findings"] = [
        {**template, "id": f"f-{n}", "observation": "о" * 3000, "impact": "в" * 3000}
        for n in range(30)
    ]
    heard["coverage"]["requested"] = [{"start": n / 10, "end": n / 10 + 0.05} for n in range(25)]
    heard["provenance"]["limitations"] = ["л" * 2000] * 20
    heard["errors"] = [{"stage": "asr", "code": "x" * 60, "message": "м" * 5000}] * 20
    heard["source"]["path"] = str(tmp_path / "recordings" / "burst.wav")
    heard["stages"]["asr"]["reason"] = f"model at {tmp_path}/models/whisper missing; " + "A" * 500
    answer = report.compact(heard)
    assert compact_size(answer) <= COMPACT_MAX_BYTES
    assert len(answer["findings"]) == 3 and answer["findings_total"] == 30
    assert answer["coverage"]["ranges_truncated"] is True
    text = json.dumps(answer, ensure_ascii=False)
    assert str(tmp_path) not in text
    assert not any(ABSOLUTE_PATH.search(s) for s in strings(answer))
    assert not any(BASE64_RUN.search(s) for s in strings(answer))


def test_compact_counts_the_head_of_the_caller_in_the_4_kib_limit(heard: dict) -> None:
    template = heard["findings"][0]
    heard["findings"] = [
        {**template, "id": f"f-{n}", "observation": "о" * 3000, "impact": "в" * 3000}
        for n in range(30)
    ]
    heard["provenance"]["limitations"] = ["л" * 2000] * 20
    head = {"job_id": "j" * 32, "job_state": "finished", "attempt": 1, "note": "н" * 300}
    answer = report.compact(heard, run_dir="runs/x", reused=False, head=head)
    assert compact_size(answer) <= COMPACT_MAX_BYTES
    assert list(answer)[:6] == ["job_id", "job_state", "attempt", "note", "run_dir", "reused"]
    assert answer["note"] == head["note"]


def test_priority_is_impact_on_intelligibility_then_confidence_then_time() -> None:
    def item(id_: str, polarity: str, category: str, basis: str, start: float = 1.0) -> dict:
        return {
            "id": id_,
            "polarity": polarity,
            "category": category,
            "evidence_basis": basis,
            "start": start,
        }

    ordered = report.prioritized(
        [
            item("neutral-pron-human", "neutral", "pronunciation", "human_verified"),
            item("neg-tempo", "negative", "tempo", "measurement"),
            item("neg-stress", "negative", "semantic_stress", "audio_model_observation"),
            item("neg-pron-model-late", "negative", "pronunciation", "audio_model_observation", 2),
            item("neg-pron-model", "negative", "pronunciation", "audio_model_observation"),
            item("neg-pron-human", "negative", "pronunciation", "human_verified", 3),
            item("positive", "positive", "pronunciation", "human_verified"),
        ]
    )
    assert [f["id"] for f in ordered] == [
        "neg-pron-human",
        "neg-pron-model",
        "neg-pron-model-late",
        "neg-stress",
        "neg-tempo",
        "neutral-pron-human",
        "positive",
    ]


def test_priority_rule_is_written_in_the_documentation() -> None:
    decision = Path(__file__).resolve().parent.parent / "docs/decisions/0004-report-priority.md"
    text = decision.read_text(encoding="utf-8")
    for name in (*report.INTELLIGIBILITY_ORDER, *report.CONFIDENCE_ORDER, *report.POLARITY_ORDER):
        assert f"`{name}`" in text


# --- cost -------------------------------------------------------------------------------------


def test_local_cost_is_zero_and_marked_local(run: Run, config: Config, unheard: dict) -> None:
    unheard["cost"] = report.cost()
    assert unheard["cost"]["kind"] == "actual" and unheard["cost"]["amount"] == 0
    assert unheard["cost"]["provider"] == "local" and "local" in unheard["cost"]["note"]
    report.write(run, unheard, config=config)
    _, markdown = written(run)
    assert "- 0 (`actual`, local); провайдер: local; local:" in markdown
    assert report.compact(unheard)["cost"] == {
        "kind": "actual",
        "amount": 0,
        "currency": None,
        "provider": "local",
    }


def test_external_cost_without_data_is_unknown(run: Run, config: Config, unheard: dict) -> None:
    unheard["cost"] = report.cost("cloud-provider")
    assert unheard["cost"]["kind"] == "unknown" and unheard["cost"]["amount"] is None
    report.write(run, unheard, config=config)
    _, markdown = written(run)
    assert "неизвестна (`unknown`); провайдер: cloud-provider" in markdown
    zero = {**unheard["cost"], "kind": "actual", "amount": 0}
    assert any("zero actual cost" in e for e in schema.validation_errors({**unheard, "cost": zero}))


def test_external_cost_carries_its_currency(unheard: dict) -> None:
    unheard["cost"] = report.cost("cloud-provider", amount=0.42, currency="USD", estimated=True)
    assert unheard["cost"]["kind"] == "estimated" and unheard["cost"]["currency"] == "USD"
    schema.validate(unheard)
    assert "- 0.42 USD (`estimated`); провайдер: cloud-provider." in report.render(unheard)
    with pytest.raises(ValueError, match="currency"):
        report.cost("cloud-provider", amount=1.0)


# --- private values ---------------------------------------------------------------------------


def test_report_holds_no_absolute_paths_user_name_or_config_contents(
    run: Run, config: Config, heard: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "operator" / "sound-scout.toml"
    config_file.parent.mkdir()
    config_text = (
        f'allowed_roots = ["{config.allowed_roots[0]}"]\ndata_root = "{config.data_root}"\n'
    )
    config_file.write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("SOUND_SCOUT_CONFIG", str(config_file))
    user = getpass.getuser()
    heard["source"]["path"] = str(config.allowed_roots[0] / "burst.wav")
    heard["stages"]["prepare"]["reason"] = f"read by {user} from {Path.home()}/notes.txt"
    heard["errors"] = [
        {"stage": "asr", "code": "asr_failed", "message": f"{config_file}: {config_text}"},
        {"stage": "report", "code": "other", "message": f"run {run.path} and C:\\Users\\{user}"},
    ]
    returned = report.write(run, heard, config=config)
    answer = report.compact(returned)
    result, markdown = written(run)
    for text in (json.dumps(result, ensure_ascii=False), markdown):
        for private in (str(tmp_path), str(Path.home()), str(config_file), user, "C:\\"):
            assert private not in text
    for shown in (returned, answer):
        assert "path" not in shown.get("source", {})
        assert not any(ABSOLUTE_PATH.search(s) for s in strings(shown))
    assert not ABSOLUTE_PATH.search(markdown)
    assert "<private path>" in markdown and "<user>" in markdown
