"""ASR review sheets of corrupted_asr candidates (#84): only dev clips, empty human fields.

The ASR is a stub here: these tests prove the contract (which clips are opened and transcribed,
what the sheet holds, that nothing is decided for the person), not what the model hears.
"""

import builtins
import hashlib
import io
import json
import os
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from sound_scout import asr, ingest
from sound_scout.config import Config
from sound_scout.eval import asr_review, manifest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

RATE = 16_000
RAW = {
    "model": {"repo": "owner/model", "revision": "rev1"},
    "flags": [],
    "text": "распознанный текст",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": " распознанный текст", "suspect_hallucination": False}
    ],
}


def wav_bytes(seconds: float, frequency: float) -> bytes:
    t = np.arange(round(seconds * RATE)) / RATE
    samples = np.round(0.3 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(samples.tobytes())
    return buffer.getvalue()


def control(clip_id: str, split: str, kind: str, data: bytes) -> dict:
    item = {
        "id": clip_id,
        "path": f"clips/{clip_id}.wav",
        "sha256": hashlib.sha256(data).hexdigest(),
        "duration_s": 2.0,
        "split": split,
        "source_group": f"g-{clip_id}",
        "categories": [],
        "negative_control": True,
        "control_kind": kind,
    }
    if kind == "corrupted_asr":
        item["asr_error_status"] = "candidate"
    return item


class Setup:
    def __init__(self, base: Path) -> None:
        self.root = base / "corpus" / "pilot-t"
        self.clips = self.root / "clips"
        self.clips.mkdir(parents=True)
        self.data = base / "data"
        self.data.mkdir()
        self.config = Config(allowed_roots=(self.clips,), data_root=self.data)
        records = []
        for clip_id, split, kind, frequency in (
            ("c-dev", "dev", "corrupted_asr", 220.0),
            ("c-held", "heldout", "corrupted_asr", 330.0),
            ("n-dev", "dev", "noise", 440.0),
        ):
            data = wav_bytes(2.0, frequency)
            (self.clips / f"{clip_id}.wav").write_bytes(data)
            records.append(control(clip_id, split, kind, data))
        # Opening the held-out clip at all would fail: the code must never try.
        (self.clips / "c-held.wav").chmod(0)
        self.manifest = self.root / "manifest.json"
        self.write(records)

    def write(self, records: list[dict]) -> None:
        document = {"schema_version": "1.0", "name": "pilot-t", "records": records}
        self.manifest.write_text(json.dumps(document), encoding="utf-8")

    def records(self) -> list[dict]:
        return json.loads(self.manifest.read_text("utf-8"))["records"]

    def sheet(self, clip_id: str) -> Path:
        return self.root / manifest.ASR_REVIEW_DIR / f"{clip_id}.json"


@pytest.fixture
def setup(tmp_path: Path):
    made = Setup(tmp_path)
    yield made
    (made.clips / "c-held.wav").chmod(0o600)


@pytest.fixture
def transcribed(monkeypatch) -> list[int]:
    calls: list[int] = []

    def fake(samples, run, config_):
        calls.append(len(samples))
        return asr.Transcription("completed", None, 5, raw_sha256="ab" * 32, raw=RAW)

    monkeypatch.setattr(asr, "transcribe", fake)
    return calls


@pytest.fixture
def opened(monkeypatch) -> list[str]:
    """Every path the process opens or hands to the decoder, as text."""
    seen: list[str] = []
    real_open, real_os_open, real_decode = builtins.open, os.open, ingest.decode

    def spy_open(file, *args, **kwargs):
        seen.append(str(file))
        return real_open(file, *args, **kwargs)

    def spy_os_open(path, *args, **kwargs):
        seen.append(str(path))
        return real_os_open(path, *args, **kwargs)

    def spy_decode(path, run, config_):
        seen.append(str(path))
        return real_decode(path, run, config_)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    monkeypatch.setattr(ingest, "decode", spy_decode)
    return seen


def test_only_dev_corrupted_asr_candidates_are_transcribed_and_heldout_is_never_opened(
    setup: Setup, transcribed: list[int], opened: list[str]
) -> None:
    result = asr_review.review(setup.manifest, setup.config)
    assert result == {
        "reviewed": [
            {
                "id": "c-dev",
                "sheet": "asr-review/c-dev.json",
                "outcome": "written",
                "reason": None,
            }
        ],
        "skipped_heldout": ["c-held"],
    }
    assert transcribed == [2 * RATE]
    assert any(path.endswith("c-dev.wav") for path in opened)
    assert not any("c-held" in path for path in opened), "a held-out clip was opened"
    assert not any("n-dev" in path for path in opened), "a non-candidate was opened"
    assert sorted(p.name for p in setup.sheet("c-dev").parent.iterdir()) == ["c-dev.json"]


def test_the_sheet_holds_the_raw_asr_and_empty_human_fields_and_decides_nothing(
    setup: Setup, transcribed: list[int]
) -> None:
    before = setup.manifest.read_bytes()
    asr_review.review(setup.manifest, setup.config)
    assert setup.manifest.read_bytes() == before, "the manifest is never changed"
    path = setup.sheet("c-dev")
    assert path.stat().st_mode & 0o777 == 0o600
    sheet = json.loads(path.read_text("utf-8"))
    record = setup.records()[0]
    assert sheet["clip_id"] == "c-dev" and sheet["clip_sha256"] == record["sha256"]
    assert sheet["split"] == "dev" and sheet["control_kind"] == "corrupted_asr"
    assert sheet["manifest"] == {"name": "pilot-t", "sha256": hashlib.sha256(before).hexdigest()}
    assert sheet["asr"]["text"] == RAW["text"]
    assert sheet["asr"]["segments"] == RAW["segments"]
    assert sheet["asr"]["model"] == RAW["model"] and sheet["asr"]["status"] == "completed"
    assert sheet["asr"]["run"].startswith("asr-review/pilot-t/c-dev-")
    assert (setup.data / sheet["asr"]["run"]).is_dir()
    assert sheet["human"] == {
        "transcript": "",
        "verdict": None,
        "reviewer": "",
        "reviewed_at": "",
        "note": None,
    }
    assert "asr_error" not in json.dumps(sheet["human"])
    assert manifest.sheet_problems(sheet, record), "an unsigned sheet confirms nothing"
    report = manifest.validate(setup.manifest)
    assert any("'c-dev' is a corrupted_asr candidate" in e for e in report.label_dependent)


def test_an_existing_sheet_is_never_rewritten(setup: Setup, transcribed: list[int]) -> None:
    asr_review.review(setup.manifest, setup.config)
    path = setup.sheet("c-dev")
    signed = path.read_text("utf-8").replace('"transcript": ""', '"transcript": "услышано"')
    path.write_text(signed, encoding="utf-8")
    again = asr_review.review(setup.manifest, setup.config)
    assert [o["outcome"] for o in again["reviewed"]] == ["exists"]
    assert transcribed == [2 * RATE], "the ASR ran once"
    assert path.read_text("utf-8") == signed


def test_an_asr_stage_that_did_not_complete_writes_no_sheet(setup: Setup, monkeypatch) -> None:
    def unavailable(samples, run, config_):
        return asr.Transcription("unavailable", asr.MODEL_UNAVAILABLE_REASON, 1)

    monkeypatch.setattr(asr, "transcribe", unavailable)
    result = asr_review.review(setup.manifest, setup.config)
    assert result["reviewed"] == [
        {
            "id": "c-dev",
            "sheet": None,
            "outcome": "failed",
            "reason": f"asr is unavailable: {asr.MODEL_UNAVAILABLE_REASON}",
        }
    ]
    assert not setup.sheet("c-dev").exists()


def test_a_clip_off_its_manifest_hash_is_not_transcribed(
    setup: Setup, transcribed: list[int]
) -> None:
    records = setup.records()
    records[0]["sha256"] = "0" * 64
    setup.write(records)
    result = asr_review.review(setup.manifest, setup.config)
    assert result["reviewed"][0]["outcome"] == "failed"
    assert result["reviewed"][0]["reason"] == "the clip's sha256 differs from its manifest record"
    assert transcribed == [] and not setup.sheet("c-dev").exists()


def test_clips_outside_the_allowed_roots_are_refused(
    setup: Setup, transcribed: list[int], tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    config_ = Config(allowed_roots=(elsewhere,), data_root=setup.data)
    result = asr_review.review(setup.manifest, config_)
    assert result["reviewed"][0]["outcome"] == "failed"
    assert result["reviewed"][0]["reason"].startswith("PathDenied")
    assert transcribed == [] and not setup.sheet("c-dev").exists()


def test_a_manifest_off_its_schema_is_refused_before_anything_runs(
    setup: Setup, transcribed: list[int]
) -> None:
    records = setup.records()
    records[0]["asr_error_status"] = "maybe"
    setup.write(records)
    with pytest.raises(asr_review.ReviewError, match="does not match its schema"):
        asr_review.review(setup.manifest, setup.config)
    assert transcribed == [] and not (setup.data / asr_review.RUNS_DIR).exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda records: records[2].update(id="c-dev"), r"\$\.records\[2\]\.id: duplicate"),
        (lambda records: records[2].update(sha256=records[0]["sha256"]), "sha256: duplicate"),
        (
            lambda records: records[2].update(source_group="g-c-held"),
            "'g-c-held' is present in both dev and heldout",
        ),
        (lambda records: records[2].update(path="../n-dev.wav"), "must not contain '..'"),
    ],
)
def test_a_manifest_with_inconsistent_records_is_refused_before_anything_runs(
    setup: Setup, transcribed: list[int], change, message: str
) -> None:
    records = setup.records()
    change(records)
    setup.write(records)
    with pytest.raises(asr_review.ReviewError, match=message):
        asr_review.review(setup.manifest, setup.config)
    assert transcribed == [] and not (setup.data / asr_review.RUNS_DIR).exists()


def test_main_prints_outcomes_without_text_and_fails_on_a_failed_clip(
    setup: Setup, monkeypatch, capsys
) -> None:
    toml = setup.root.parent / "sound-scout.toml"
    toml.write_text(
        f'allowed_roots = ["{setup.clips}"]\ndata_root = "{setup.data}"\n', encoding="utf-8"
    )
    monkeypatch.setenv("SOUND_SCOUT_CONFIG", str(toml))
    monkeypatch.setattr(asr_review.guard, "install_network_guard", lambda config_: None)

    def unavailable(samples, run, config_):
        return asr.Transcription("unavailable", "нет модели", 1)

    monkeypatch.setattr(asr, "transcribe", unavailable)
    assert asr_review.main(["--manifest", str(setup.manifest)]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines == [
        {
            "id": "c-dev",
            "sheet": None,
            "outcome": "failed",
            "reason": "asr is unavailable: нет модели",
        },
        {"skipped_heldout": ["c-held"]},
    ]
    monkeypatch.setattr(
        asr,
        "transcribe",
        lambda samples, run, config_: asr.Transcription("completed", None, 1, raw=RAW),
    )
    assert asr_review.main(["--manifest", str(setup.manifest)]) == 0
    out = capsys.readouterr().out
    assert RAW["text"] not in out and '"outcome": "written"' in out
    assert asr_review.main(["--manifest", str(setup.root / "missing.json")]) == 2
