"""`app.compare` (issue #34): criteria, windows, the verdict rule, conditions, the written file.

Attempts are real `app.analyze` runs over audio synthesized in the temp dir, executed in process,
with test doubles in three places: `FakeBackend` for the audio review, a `FakeModel` for ASR
(known words with timecodes) and fixed speech spans for the detector. They prove the comparison
rules and the contract, not that the tool hears anything (R01, T03); the results are synthetic.
"""

import copy
import dataclasses
import json
import os
import shutil
import stat
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from sound_scout import acoustics, app, asr, comparison, review
from sound_scout.app import AnalyzeRequest, analyze
from sound_scout.comparison import CompareError, normalized_word, parse_criterion
from sound_scout.config import Asr, Config
from tests.test_asr import REPO, REVISION, FakeModel, fake_snapshot, restore_offline_env
from tests.test_asr import segment as asr_segment
from tests.test_asr import word as asr_word
from tests.test_review import FakeBackend

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are not installed",
)

LAYERS = ("content", "measurements", "sound")
TASK = "прочитать фразу про маму"
INTENT = "подчеркнуть, что мыла именно мама"
WORDS = [(0.2, 0.5, " Мама"), (0.6, 0.9, " мыла"), (1.0, 1.4, " раму.")]
SPANS = [{"start": 0.1, "end": 1.5}]
FIXED_LIMITATIONS = (
    "Более короткая запись или меньшее число пауз не считается улучшением",
    "Паузы во второй попытке могут быть связаны с новым вопросом",
    "Один удачный контроль не доказывает устойчивый навык",
    "Отсутствие замечания аудиомодели во второй попытке — не доказательство правильности",
    "gain и микрофон неизвестны",
)


def observation(category: str, start: float | None, end: float | None, correction: str | None):
    item = {
        "category": category,
        "observation": "слово звучит не так, как задумано",
        "confidence_note": "слышно отчётливо",
        "impact": "смысл фразы смещается",
        "correction": correction,
    }
    if start is not None:
        item |= {"start_seconds": start, "end_seconds": end}
    return item


STRESS = observation("semantic_stress", 0.6, 0.9, "сделать ударение на этом слове")
NEUTRAL_STRESS = observation("semantic_stress", 0.6, 0.9, None)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    base = tmp_path.resolve()
    recordings, data_root, hf_cache = base / "recordings", base / "data", base / "hf"
    for directory in (recordings, data_root):
        directory.mkdir()
    fake_snapshot(hf_cache, REVISION)
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))
    restore_offline_env(monkeypatch)
    return Config(allowed_roots=(recordings,), data_root=data_root, asr=Asr(model=REPO))


@pytest.fixture
def make_attempt(config: Config, monkeypatch: pytest.MonkeyPatch):
    """Analyse one synthesized recording in process; returns its `run_id`."""

    def execute_in_process(request, config, run, backend, audio_sha256, cancel):
        app._run_stages(request, config, run, backend, audio_sha256)
        return None

    monkeypatch.setattr(app, "_stop_signal", None)
    monkeypatch.setattr(app, "_execute", execute_in_process)
    counter = iter(range(1000))

    def make(
        *,
        observations: tuple = (),
        unlocalized: tuple = (),
        task: str | None = TASK,
        intent: str | None = None,
        uncertainties: tuple = (),
        model_limitations: tuple = (),
        amplitude: float = 0.5,
        channels: int = 1,
        sample_rate: int = 16_000,
        seconds: float = 2.0,
        frequency: float = 220.0,
        words: list = WORDS,
        suspect: bool = False,
        spans: list[dict] = SPANS,
        backend: bool = True,
        path: Path | None = None,
    ) -> str:
        index = next(counter)
        if path is None:
            path = config.allowed_roots[0] / f"attempt-{index}.wav"
            write_wav(path, seconds, sample_rate, frequency, amplitude, channels)
        segment = asr_segment(
            0.0,
            seconds,
            "".join(text for _, _, text in words),
            [asr_word(start, end, text) for start, end, text in words],
            no_speech_prob=0.9 if suspect else 0.01,
        )
        model = FakeModel([segment])
        monkeypatch.setattr(asr, "_load_model", lambda snapshot_dir: model)
        monkeypatch.setattr(acoustics, "speech_activity", lambda mono, rate: list(spans))
        response = {
            "observations": list(observations),
            "unlocalized_observations": list(unlocalized),
            "uncertainties": list(uncertainties),
            "limitations": list(model_limitations),
        }
        request = AnalyzeRequest(
            str(path), request_id=f"req-{index}", layers=LAYERS, task=task, intent=intent
        )
        fake = FakeBackend(text=json.dumps(response)) if backend else None
        analysis = analyze(request, config, backend=fake)
        assert analysis.result["stages"]["asr"]["status"] == "completed"
        return analysis.run.run_id

    return make


def write_wav(
    path: Path, seconds: float, rate: int, frequency: float, amplitude: float, channels: int
) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    tone = amplitude * np.sin(2 * np.pi * frequency * t) * np.iinfo(np.int16).max
    samples = np.repeat(tone.astype(np.int16)[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())
    return path


def codes(result: dict) -> set[tuple[str, str | None]]:
    return {(reason["code"], reason["attempt"]) for reason in result["reasons"]}


def compare_files(config: Config) -> list[Path]:
    directory = config.data_root / "compare"
    return sorted(directory.iterdir()) if directory.exists() else []


def run_dir(config: Config, run_id: str) -> Path:
    return config.data_root / "runs" / run_id


def rewrite_json(path: Path, change) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    os.chmod(path, 0o600)
    path.write_text(json.dumps(document), encoding="utf-8")


# --- the criterion ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "word", "category"),
    [
        ("emphasis_on:мыла", "emphasis_on", "мыла", "semantic_stress"),
        ("pause_before: раму ", "pause_before", "раму", "pause_phrasing"),
        ("intelligibility_of:Мама", "intelligibility_of", "Мама", "pronunciation"),
        ("completeness_of_phrase", "completeness_of_phrase", None, "completeness"),
    ],
)
def test_the_four_criterion_forms_parse(text: str, kind: str, word, category: str) -> None:
    criterion = parse_criterion(text)

    assert (criterion.kind, criterion.word, criterion.category) == (kind, word, category)


@pytest.mark.parametrize(
    "text",
    [
        "loudness_of:мыла",
        "emphasis_on:",
        "emphasis_on:  ",
        "completeness_of_phrase:мама",
        "pause_before:мама мыла",
        "emphasis_on:" + "а" * 65,
        "emphasis_on:...",
        "",
        None,
    ],
)
def test_a_criterion_outside_the_fixed_forms_is_refused(text: object) -> None:
    with pytest.raises(CompareError) as error:
        parse_criterion(text)

    assert error.value.code == "criterion_invalid"


def test_words_match_after_casefold_yo_and_edge_punctuation() -> None:
    assert normalized_word(" «Ёлка,»") == normalized_word("елка") == "елка"
    assert normalized_word("из-за") == "из-за"


# --- verdicts --------------------------------------------------------------------------------


def test_improved_needs_an_issue_in_a_and_none_in_b_and_carries_both_attempts(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(observations=(NEUTRAL_STRESS,), frequency=230.0)

    compared = app.compare(run_a, run_b, "emphasis_on:мыла", config)

    result = compared.result
    assert result["verdict"] == "improved", result["reasons"]
    assert result["reasons"] == []
    a, b = result["attempts"]["a"], result["attempts"]["b"]
    assert (a["run_id"], b["run_id"]) == (run_a, run_b)
    assert a["windows"] == b["windows"] == [{"start": 0.6, "end": 0.9}]
    assert a["reviewed"] and b["reviewed"] and a["word_occurrences"] == 1
    assert [f["polarity"] for f in a["evidence"]["findings"]] == ["negative"]
    assert [f["polarity"] for f in b["evidence"]["findings"]] == ["neutral"]
    assert a["evidence"]["findings"][0]["evidence_basis"] == "audio_model_observation"
    conditions = result["conditions_diff"]
    assert set(conditions) == {
        "duration_s",
        "pauses",
        "level_dbfs",
        "channel",
        "format",
        "gain",
        "task",
        "intent",
        "backend",
    }
    assert conditions["gain"]["known"] is False and conditions["gain"]["a"] is None
    assert conditions["task"]["a"] == conditions["task"]["b"] == TASK
    assert conditions["intent"]["a"] is None and conditions["intent"]["differs"] is False
    assert conditions["format"]["a"] == {
        "container": "wav",
        "codec": "pcm_s16le",
        "sample_rate": 16000,
    }
    assert conditions["channel"]["differs"] is False
    assert result["synthetic"] is True
    assert app.SYNTHETIC_NOTE in result["limitations"]
    assert result["limitations"][0].startswith("Сравнение только по критерию emphasis_on:мыла")
    for text in FIXED_LIMITATIONS:
        assert any(text in limitation for limitation in result["limitations"]), text
    assert not any(text.startswith("попытка") for text in result["limitations"])


def test_the_file_is_written_under_the_data_root_0600_and_matches_the_schema(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)

    compared = app.compare(run_a, run_b, "emphasis_on:мыла", config)

    path = compared.path
    assert path == config.data_root / "compare" / f"{compared.result['compare_id']}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == compared.result
    assert comparison.schema_errors(comparison.SCHEMA, written) == []
    assert comparison.schema_errors(comparison.SCHEMA, written | {"extra": 1}) != []
    again = app.compare(run_a, run_b, "emphasis_on:мыла", config)
    assert again.path != path and len(compare_files(config)) == 2


def test_not_improved_when_b_still_has_a_negative_finding(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(observations=(STRESS,), frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "not_improved" and result["reasons"] == []


def test_pause_before_uses_the_gap_before_the_word_and_shows_spans_as_measurements(
    config: Config, make_attempt
) -> None:
    pause = observation("pause_phrasing", 0.5, 0.6, "не делать паузу перед этим словом")
    run_a = make_attempt(observations=(pause,))
    run_b = make_attempt(frequency=230.0)

    result = app.compare(run_a, run_b, "pause_before:мыла", config).result

    assert result["verdict"] == "improved", result["reasons"]
    a = result["attempts"]["a"]
    assert a["windows"] == [{"start": 0.5, "end": 0.9}]
    assert a["evidence"]["measurements"] == [
        {"name": "speech_spans", "start": 0.1, "end": 1.5, "evidence_basis": "measurement"}
    ]


def test_completeness_counts_an_unlocalized_observation_with_a_correction(
    config: Config, make_attempt
) -> None:
    unfinished = observation("completeness", None, None, "договорить фразу до конца")
    run_a = make_attempt(unlocalized=(unfinished,))
    run_b = make_attempt(unlocalized=(unfinished,), frequency=230.0)
    run_c = make_attempt(frequency=240.0)

    still = app.compare(run_a, run_b, "completeness_of_phrase", config).result
    fixed = app.compare(run_a, run_c, "completeness_of_phrase", config).result

    assert still["verdict"] == "not_improved" and still["reasons"] == []
    assert fixed["verdict"] == "improved" and fixed["reasons"] == []
    assert still["attempts"]["a"]["windows"] == [{"start": 0.0, "end": 2.0}]
    assert still["attempts"]["a"]["word_occurrences"] is None


# --- inconclusive ----------------------------------------------------------------------------


def test_backend_none_makes_both_attempts_audio_review_missing(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(backend=False)
    run_b = make_attempt(backend=False, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("audio_review_missing", "a"), ("audio_review_missing", "b")}
    assert result["synthetic"] is False
    assert "backend = none" in result["attempts"]["a"]["audio_review"]["reason"]


@pytest.mark.parametrize("task_b", ["пересказать абзац", None])
def test_a_different_or_missing_task_of_one_attempt_is_task_differs(
    config: Config, make_attempt, task_b: str | None
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(task=task_b, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("task_differs", None)}
    assert result["conditions_diff"]["task"]["differs"] is True


@pytest.mark.parametrize("intent_b", ["показать удивление", None])
def test_a_different_or_missing_intent_of_one_attempt_is_intent_differs(
    config: Config, make_attempt, intent_b: str | None
) -> None:
    run_a = make_attempt(observations=(STRESS,), intent=INTENT)
    run_b = make_attempt(intent=intent_b, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("intent_differs", None)}
    intent = result["conditions_diff"]["intent"]
    assert (intent["a"], intent["b"], intent["differs"]) == (INTENT, intent_b, True)
    assert intent["source"] == "request.json.task_context.intent"


def test_same_intent_spelled_with_other_spaces_is_the_same_intent(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,), intent=INTENT)
    run_b = make_attempt(intent=f" {INTENT.replace(' ', '  ')}  ", frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "improved", result["reasons"]
    assert result["conditions_diff"]["intent"]["b"] == INTENT


def test_the_model_caveats_of_both_attempts_reach_limitations_without_changing_the_verdict(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,), model_limitations=("слышен только фрагмент",))
    run_b = make_attempt(
        observations=(NEUTRAL_STRESS,),
        uncertainties=("ударение могло прийтись на соседнее слово",),
        frequency=230.0,
    )

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "improved", result["reasons"]
    caveats = [text for text in result["limitations"] if text.startswith("попытка")]
    assert caveats == [
        "попытка A: " + review.MODEL_LIMITATION + "слышен только фрагмент",
        "попытка B: " + review.MODEL_UNCERTAINTY + "ударение могло прийтись на соседнее слово",
    ]


def test_same_task_spelled_with_other_spaces_is_the_same_task(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(task=f"  {TASK.replace(' ', '   ')} ", frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "improved", result["reasons"]


def test_no_task_for_either_attempt_is_a_limitation_not_a_reason(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,), task=None)
    run_b = make_attempt(task=None, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "improved", result["reasons"]
    assert comparison.NO_TASK_LIMITATION in result["limitations"]


def test_an_old_request_json_without_task_context_is_task_unknown(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)
    rewrite_json(run_dir(config, run_a) / "request.json", lambda d: d.pop("task_context"))

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("task_unknown", "a")}
    assert result["conditions_diff"]["task"]["differs"] is None


@pytest.mark.parametrize(("gain_db", "differs"), [(7.0, True), (5.0, False)])
def test_a_level_difference_over_6_db_is_level_differs(
    config: Config, make_attempt, gain_db: float, differs: bool
) -> None:
    run_a = make_attempt(observations=(STRESS,), amplitude=0.5 / 10 ** (gain_db / 20))
    run_b = make_attempt()

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    level = result["conditions_diff"]["level_dbfs"]
    assert level["delta"] == pytest.approx(gain_db, abs=0.05)
    assert level["exceeds"] is differs
    if differs:
        assert result["verdict"] == "inconclusive"
        assert codes(result) == {("level_differs", None)}
    else:
        assert result["verdict"] == "improved", result["reasons"]


def test_mono_against_stereo_is_channel_differs(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(channels=2, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("channel_differs", None)}
    channel = result["conditions_diff"]["channel"]
    assert (channel["a"]["source_channels"], channel["b"]["source_channels"]) == (1, 2)


def test_another_sample_rate_is_format_differs(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(sample_rate=22_050, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("format_differs", None)}


def test_wav_against_ogg_opus_is_format_differs(config: Config, make_attempt) -> None:
    wav = write_wav(config.allowed_roots[0] / "control.wav", 2.0, 16_000, 230.0, 0.5, 1)
    ogg = config.allowed_roots[0] / "control.ogg"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(wav), "-c:a", "libopus", str(ogg)],
        check=True,
    )
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(path=ogg)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert ("format_differs", None) in codes(result)
    fmt = result["conditions_diff"]["format"]
    assert fmt["differs"] is True
    assert (fmt["a"]["container"], fmt["b"]["container"]) == ("wav", "ogg")
    assert fmt["b"]["codec"] == "opus"


def test_a_missing_source_json_is_format_unknown(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)
    (run_dir(config, run_b) / "source.json").unlink()

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert codes(result) == {("format_unknown", "b")}
    assert result["conditions_diff"]["format"]["b"] is None


def test_a_word_absent_from_the_transcript_is_word_not_found(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:кот", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("word_not_found", "a"), ("word_not_found", "b")}
    assert result["attempts"]["a"]["evidence"]["findings"] == []


def test_words_of_a_suspect_hallucination_segment_are_not_looked_up(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(suspect=True, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert codes(result) == {("word_not_found", "b")}


def test_a_raw_transcript_without_the_trust_flag_is_result_invalid(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)
    rewrite_json(
        run_dir(config, run_b) / "asr" / "raw.json",
        lambda d: d["segments"][0].pop("suspect_hallucination"),
    )

    with pytest.raises(CompareError) as error:
        app.compare(run_a, run_b, "emphasis_on:мыла", config)

    assert error.value.code == "result_invalid"
    assert compare_files(config) == []


def test_a_window_outside_the_reviewed_range_is_criterion_not_reviewed(
    config: Config, make_attempt
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    late = [(0.2, 0.5, " Мама"), (2.1, 2.4, " мыла")]
    run_b = make_attempt(words=late, frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert codes(result) == {("criterion_not_reviewed", "b")}
    assert result["attempts"]["b"]["reviewed"] is False


def test_nothing_to_improve_in_a_is_no_baseline_issue(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(NEUTRAL_STRESS,))
    run_b = make_attempt(frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {("no_baseline_issue", "a")}


def test_an_unlocalized_issue_of_the_category_in_b_is_unlocalized_observation(
    config: Config, make_attempt
) -> None:
    somewhere = observation("semantic_stress", None, None, "переставить ударение во фразе")
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(unlocalized=(somewhere,), frequency=230.0)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert codes(result) == {("unlocalized_observation", "b")}


def test_the_same_recording_twice_is_same_recording(config: Config, make_attempt) -> None:
    path = write_wav(config.allowed_roots[0] / "once.wav", 2.0, 16_000, 220.0, 0.5, 1)
    run_a = make_attempt(observations=(STRESS,), path=path)
    run_b = make_attempt(path=path)

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert codes(result) == {("same_recording", None)}


def test_every_reason_is_listed_at_once(config: Config, make_attempt) -> None:
    run_a = make_attempt(backend=False, amplitude=0.1)
    run_b = make_attempt(
        backend=False, channels=2, task="другая задача", intent=INTENT, sample_rate=22_050
    )

    result = app.compare(run_a, run_b, "emphasis_on:кот", config).result

    assert result["verdict"] == "inconclusive"
    assert codes(result) == {
        ("audio_review_missing", "a"),
        ("audio_review_missing", "b"),
        ("word_not_found", "a"),
        ("word_not_found", "b"),
        ("level_differs", None),
        ("channel_differs", None),
        ("format_differs", None),
        ("task_differs", None),
        ("intent_differs", None),
    }
    assert all(reason["message"] for reason in result["reasons"])


@pytest.mark.parametrize(("a_observations", "b_observations"), [((STRESS,), (STRESS,)), ((), ())])
def test_a_shorter_control_with_fewer_pauses_is_never_improved_by_that_alone(
    config: Config, make_attempt, a_observations: tuple, b_observations: tuple
) -> None:
    long_spans = [
        {"start": 0.1, "end": 0.5},
        {"start": 0.9, "end": 1.2},
        {"start": 1.6, "end": 1.9},
    ]
    run_a = make_attempt(observations=a_observations, seconds=2.0, spans=long_spans)
    run_b = make_attempt(
        observations=b_observations, seconds=1.2, spans=[{"start": 0.1, "end": 1.1}]
    )

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    conditions = result["conditions_diff"]
    assert conditions["duration_s"]["delta"] < 0
    assert conditions["pauses"]["a"]["speech_spans"] == 3
    assert conditions["pauses"]["b"] == {"speech_spans": 1, "pause_total_s": 0.0}
    assert result["verdict"] != "improved"


def test_a_shorter_control_with_fewer_pauses_and_the_same_issue_is_not_improved(
    config: Config, make_attempt
) -> None:
    long_spans = [
        {"start": 0.1, "end": 0.5},
        {"start": 0.9, "end": 1.2},
        {"start": 1.6, "end": 1.9},
    ]
    run_a = make_attempt(observations=(STRESS,), seconds=2.0, spans=long_spans)
    run_b = make_attempt(observations=(STRESS,), seconds=1.2, spans=[{"start": 0.1, "end": 1.1}])

    result = app.compare(run_a, run_b, "emphasis_on:мыла", config).result

    assert result["conditions_diff"]["duration_s"]["delta"] < 0
    assert result["verdict"] == "not_improved" and result["reasons"] == []


def test_intelligibility_of_a_word_is_compared_through_app_compare(
    config: Config, make_attempt
) -> None:
    unclear = observation("pronunciation", 0.2, 0.5, "произнести слово отчётливее")
    clear = observation("pronunciation", 0.2, 0.5, None)
    run_a = make_attempt(observations=(unclear,))
    run_b = make_attempt(observations=(clear,), frequency=230.0)

    result = app.compare(run_a, run_b, "intelligibility_of:мама", config).result

    assert result["criterion"]["category"] == "pronunciation"
    assert result["verdict"] == "improved", result["reasons"]
    assert result["attempts"]["a"]["windows"] == [{"start": 0.2, "end": 0.5}]
    assert [f["polarity"] for f in result["attempts"]["a"]["evidence"]["findings"]] == ["negative"]


def test_another_backend_revision_is_backend_differs(config: Config, make_attempt) -> None:
    a = comparison.load_attempt(make_attempt(observations=(STRESS,)), config)
    b = comparison.load_attempt(make_attempt(frequency=230.0), config)
    other = copy.deepcopy(b.result)
    other["provenance"]["revision"] = "another-revision"

    result = comparison.build(
        parse_criterion("emphasis_on:мыла"), a, dataclasses.replace(b, result=other)
    )

    assert codes(result) == {("backend_differs", None)}
    assert result["conditions_diff"]["backend"]["b"]["revision"] == "another-revision"


def test_the_packaged_schema_is_the_copy_in_schema_dir() -> None:
    root = REPO_ROOT / "schema" / "compare.v1.schema.json"

    assert root.read_bytes() == comparison.SCHEMA_BYTES


# --- refusals --------------------------------------------------------------------------------


def test_refusals_raise_compare_error_and_write_nothing(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)
    absent = "0" * 32
    cases = {
        "criterion_invalid": (run_a, run_b, "loudness_of:мыла"),
        "same_run": (run_a, run_a, "emphasis_on:мыла"),
        "run_not_found": (run_a, absent, "emphasis_on:мыла"),
    }
    for code, (first, second, criterion) in cases.items():
        with pytest.raises(CompareError) as error:
            app.compare(first, second, criterion, config)
        assert error.value.code == code
    for bad in ("../" + run_a, run_a.upper(), "", None):
        with pytest.raises(CompareError) as error:
            app.compare(run_a, bad, "emphasis_on:мыла", config)
        assert error.value.code == "run_not_found"
    assert compare_files(config) == []


def test_an_incomplete_run_is_refused(config: Config, make_attempt) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)
    rewrite_json(run_dir(config, run_b) / "manifest.json", lambda d: d.update(complete=False))

    with pytest.raises(CompareError) as error:
        app.compare(run_a, run_b, "emphasis_on:мыла", config)

    assert error.value.code == "run_incomplete"
    assert compare_files(config) == []


def test_a_symlink_in_place_of_a_run_directory_is_refused(
    config: Config, make_attempt, tmp_path: Path
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)
    elsewhere = tmp_path / "elsewhere"
    shutil.move(run_dir(config, run_b), elsewhere)
    run_dir(config, run_b).symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(CompareError) as error:
        app.compare(run_a, run_b, "emphasis_on:мыла", config)

    assert error.value.code == "run_not_found"
    assert compare_files(config) == []


@pytest.mark.parametrize(
    "spoil",
    [
        lambda path: rewrite_json(path, lambda d: d["provenance"].update(synthetic=False)),
        lambda path: rewrite_json(path, lambda d: d["artifacts"][0].update(sha256="0" * 64)),
        lambda path: (os.chmod(path, 0o600), path.write_text("{not json", encoding="utf-8")),
    ],
)
def test_a_spoiled_result_json_is_result_invalid_without_its_text(
    config: Config, make_attempt, spoil
) -> None:
    run_a = make_attempt(observations=(STRESS,))
    run_b = make_attempt(frequency=230.0)
    spoil(run_dir(config, run_a) / "result.json")

    with pytest.raises(CompareError) as error:
        app.compare(run_a, run_b, "emphasis_on:мыла", config)

    assert error.value.code == "result_invalid"
    assert "слово звучит" not in str(error.value)
    assert compare_files(config) == []
