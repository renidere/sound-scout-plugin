"""Cleanup of the plugin's staging copies (`sound_scout.staging.sweep`, #122).

Jobs are submitted with a synthesized WAV saved as `<uuid4>.upload` in the staging root, as the
plugin's `POST /jobs` does; their states are written as the runner and the delivery loop would.
"""

import json
import os
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sound_scout import config, staging
from sound_scout.config import Config, Jobs
from tests.conftest import WavFactory
from tests.test_delivery import ABSENT_ASR_MODEL, ended_job, job_file, set_job

RETENTION = 3600
DELIVERED = {"state": "delivered", "tag": "t", "confirmed_by": "send"}


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = tmp_path.resolve()
    for name in ("recordings", "staging", "data"):
        (base / name).mkdir()
    return Config(
        allowed_roots=(base / "recordings", base / "staging"),
        data_root=base / "data",
        asr=config.Asr(model=ABSENT_ASR_MODEL),
        jobs=Jobs(staging_root=base / "staging", staging_retention_seconds=RETENTION),
    )


def upload(cfg: Config, make_wav: WavFactory) -> Path:
    path = cfg.jobs.staging_root / f"{uuid.uuid4().hex}.upload"
    shutil.move(make_wav(f"recordings/{uuid.uuid4().hex}.wav"), path)
    return path


def staged_job(cfg: Config, path: Path, state: str, delivery: dict | None = None, **kw) -> str:
    job_id = ended_job(cfg, path, state=state, **kw)
    set_job(cfg, job_id, delivery=delivery)
    return job_id


def later(seconds: float) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=seconds)


def make_old(path: Path) -> None:
    old = (datetime.now(UTC) - timedelta(seconds=2 * RETENTION)).timestamp()
    os.utime(path, (old, old), follow_symlinks=False)


def test_a_finished_and_delivered_job_loses_its_copy_at_once(
    cfg: Config, make_wav: WavFactory
) -> None:
    path = upload(cfg, make_wav)
    job_id = staged_job(cfg, path, "finished", DELIVERED)

    assert staging.sweep(cfg) == [{"name": path.name, "reason": "delivered"}]

    assert not path.exists()
    assert job_file(cfg, job_id)["staging_removed_at"]
    assert staging.sweep(cfg, now=later(2 * RETENTION)) == []


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_a_failed_or_cancelled_job_keeps_its_delivered_copy_until_the_retention_ends(
    cfg: Config, make_wav: WavFactory, state: str
) -> None:
    path = upload(cfg, make_wav)
    job_id = staged_job(cfg, path, state, DELIVERED)

    assert staging.sweep(cfg) == []
    assert staging.sweep(cfg, now=later(RETENTION - 60)) == []
    assert path.exists()

    assert staging.sweep(cfg, now=later(RETENTION + 1)) == [
        {"name": path.name, "reason": "retention"}
    ]
    assert not path.exists() and job_file(cfg, job_id)["staging_removed_at"]


@pytest.mark.parametrize(
    ("delivery", "session_id"),
    [
        (None, "chat-1"),
        ({"state": "delivery_unconfirmed", "tag": "t"}, "chat-1"),
        ({"state": "delivery_failed", "tag": "t"}, "chat-1"),
        (None, None),
    ],
)
def test_an_ended_job_without_a_delivered_note_loses_its_copy_only_after_the_retention(
    cfg: Config, make_wav: WavFactory, delivery: dict | None, session_id: str | None
) -> None:
    path = upload(cfg, make_wav)
    staged_job(cfg, path, "finished", delivery, session_id=session_id)

    assert staging.sweep(cfg) == []
    assert path.exists()
    assert staging.sweep(cfg, now=later(RETENTION + 1)) == [
        {"name": path.name, "reason": "retention"}
    ]
    assert not path.exists()


@pytest.mark.parametrize("state", ["queued", "running"])
def test_a_queued_or_running_job_never_loses_its_copy(
    cfg: Config, make_wav: WavFactory, state: str
) -> None:
    path = upload(cfg, make_wav)
    staged_job(cfg, path, state)

    assert staging.sweep(cfg, now=later(100 * RETENTION)) == []
    assert path.exists()


def test_a_copy_named_by_an_ended_and_a_queued_job_stays(cfg: Config, make_wav: WavFactory) -> None:
    path = upload(cfg, make_wav)
    staged_job(cfg, path, "finished", DELIVERED)
    queued = staged_job(cfg, path, "queued")

    assert staging.sweep(cfg, now=later(2 * RETENTION)) == []
    assert path.exists()

    set_job(cfg, queued, state="finished", delivery=DELIVERED)
    assert staging.sweep(cfg) == [{"name": path.name, "reason": "delivered"}]
    assert not path.exists()


def test_an_input_of_another_allowed_root_is_never_touched(
    cfg: Config, make_wav: WavFactory
) -> None:
    path = cfg.allowed_roots[0] / f"{uuid.uuid4().hex}.upload"
    shutil.move(make_wav("recordings/x.wav"), path)
    make_old(path)
    job_id = staged_job(cfg, path, "finished", DELIVERED)

    assert staging.sweep(cfg, now=later(2 * RETENTION)) == []
    assert path.exists() and "staging_removed_at" not in job_file(cfg, job_id)


def test_only_old_regular_upload_files_are_orphans_and_nothing_else_is_touched(
    cfg: Config, make_wav: WavFactory, tmp_path: Path
) -> None:
    root = cfg.jobs.staging_root
    outside = make_wav("outside.wav")
    link = root / f"{uuid.uuid4().hex}.upload"
    link.symlink_to(outside)
    folder = root / f"{uuid.uuid4().hex}.upload"
    folder.mkdir()
    others = [root / "notes.wav", root / f"{uuid.uuid4().hex.upper()}.upload"]
    for path in others:
        path.write_bytes(b"x")
    fresh = upload(cfg, make_wav)
    old = upload(cfg, make_wav)
    for path in (link, folder, *others, old):
        make_old(path)

    removed = staging.sweep(cfg)

    assert removed == [{"name": old.name, "reason": "orphan"}]
    assert not old.exists() and fresh.exists()
    assert link.is_symlink() and outside.exists() and folder.is_dir()
    assert all(path.exists() for path in others)
    assert str(tmp_path.resolve()) not in json.dumps(removed)


def test_without_a_staging_root_nothing_is_done(cfg: Config, make_wav: WavFactory) -> None:
    path = upload(cfg, make_wav)
    staged_job(cfg, path, "finished", DELIVERED)
    without = Config(
        allowed_roots=cfg.allowed_roots, data_root=cfg.data_root, asr=cfg.asr, jobs=Jobs()
    )

    assert staging.sweep(without, now=later(2 * RETENTION)) == []
    assert path.exists()


def test_a_staging_root_swapped_for_a_symlink_is_not_followed(
    cfg: Config, make_wav: WavFactory
) -> None:
    path = upload(cfg, make_wav)
    staged_job(cfg, path, "finished", DELIVERED)
    root = cfg.jobs.staging_root
    real = root.with_name("staging-real")
    root.rename(real)
    root.symlink_to(real)

    assert staging.sweep(cfg, now=later(2 * RETENTION)) == []
    assert (real / path.name).exists()


def test_an_unreadable_job_json_stops_the_sweep(cfg: Config, make_wav: WavFactory) -> None:
    path = upload(cfg, make_wav)
    staged_job(cfg, path, "finished", DELIVERED)
    orphan = upload(cfg, make_wav)
    make_old(orphan)
    broken = staged_job(cfg, upload(cfg, make_wav), "finished")
    (cfg.data_root / "jobs" / broken / "job.json").write_text("{not json", encoding="utf-8")

    assert staging.sweep(cfg, now=later(2 * RETENTION)) == []
    assert path.exists() and orphan.exists()


def test_criterion_4_a_delivered_copy_swapped_for_a_symlink_into_another_root_is_not_followed(
    cfg: Config, make_wav: WavFactory
) -> None:
    path = upload(cfg, make_wav)
    job_id = staged_job(cfg, path, "finished", DELIVERED)
    target = make_wav("recordings/kept.wav")
    content = target.read_bytes()
    path.unlink()
    path.symlink_to(target)

    assert staging.sweep(cfg, now=later(2 * RETENTION)) == []

    assert path.is_symlink() and path.resolve() == target
    assert target.read_bytes() == content
    assert "staging_removed_at" not in job_file(cfg, job_id)
