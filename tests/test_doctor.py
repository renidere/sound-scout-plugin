"""`doctor` is tested against a fake environment in tmp dirs: fake `ffmpeg`/`ffprobe` on PATH,
a fake Hugging Face cache with a few-byte snapshot, patched package metadata. No real model, no
recording; the one test on the real machine makes no assumption about the statuses."""

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sound_scout import doctor
from sound_scout.config import Asr, Budgets, Config
from sound_scout.doctor import (
    COMPONENTS,
    STATUSES,
    ReceiptRejected,
    component_status,
    fingerprint,
    fingerprint_hash,
    read_receipt,
    report,
    write_receipt,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
VERSIONS = {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.5",
    "praat-parselmouth": "0.4.7",
    "onnxruntime": "1.31.1",
}
HEAVY_MODULES = (
    "faster_whisper",
    "ctranslate2",
    "parselmouth",
    "onnxruntime",
    "huggingface_hub",
    "torch",
)
SMOKE = {"test": "tests/test_doctor.py::fake", "input": "synthetic", "elapsed_ms": 12}


@dataclass
class FakeEnv:
    bin_dir: Path
    hf_cache: Path
    package: Path
    data_root: Path
    versions: dict[str, str | None]
    config: Config

    def write_tool(self, tool: str, version: str = "0.test", exit_code: int = 0) -> None:
        script = self.bin_dir / tool
        script.write_text(
            f'#!/bin/sh\necho "{tool} version {version} Copyright (c) test"\nexit {exit_code}\n'
        )
        script.chmod(0o755)

    def add_snapshot(self, revision: str, repo: str = "o/n", files=("model.bin", "config.json")):
        snapshot = self.hf_cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        for name in files:
            (snapshot / name).write_bytes(b"x" * 8)
        return snapshot


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeEnv:
    base = tmp_path.resolve()
    bin_dir, hf_cache, package = base / "bin", base / "hf", base / "pkg" / "faster_whisper"
    recordings, data_root = base / "recordings", base / "data"
    for directory in (bin_dir, hf_cache, package / "assets", recordings, data_root):
        directory.mkdir(parents=True)
    (package / "assets" / "silero_vad_v6.onnx").write_bytes(b"onnx")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))
    versions: dict[str, str | None] = dict(VERSIONS)
    monkeypatch.setattr(doctor, "installed_version", lambda name: versions.get(name))
    monkeypatch.setattr(
        doctor, "package_dir", lambda name: package if name == "faster_whisper" else None
    )
    fake = FakeEnv(
        bin_dir,
        hf_cache,
        package,
        data_root,
        versions,
        Config(allowed_roots=(recordings,), data_root=data_root, asr=Asr(model="o/n")),
    )
    fake.write_tool("ffmpeg")
    fake.write_tool("ffprobe")
    fake.add_snapshot(REVISION)
    return fake


@pytest.fixture
def permissive_umask() -> Iterator[None]:
    previous = os.umask(0o000)
    try:
        yield
    finally:
        os.umask(previous)


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def statuses(config: Config) -> dict[str, str]:
    return {name: component["status"] for name, component in report(config)["components"].items()}


def test_report_structure_on_the_real_machine(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    result = report(Config(allowed_roots=(tmp_path,), data_root=data_root))
    assert set(result) == {
        "sound_scout",
        "components",
        "validated_ru",
        "diction_validated_ru",
        "prosody_validated_ru",
        "network_guard",
    }
    assert tuple(result["components"]) == COMPONENTS
    for name, component in result["components"].items():
        assert list(component) == ["status", "reason", "fingerprint", "last_smoke_at"]
        assert component["status"] in STATUSES
        assert component["status"] != "ready"
        assert isinstance(component["reason"], str)
        assert component["fingerprint"]["component"] == name
        assert component["fingerprint"]["hash"] == fingerprint_hash(component["fingerprint"])
        assert component["last_smoke_at"] is None
    assert (result["validated_ru"], result["diction_validated_ru"]) == (False, False)
    assert result["prosody_validated_ru"] is False
    assert set(result["network_guard"]) == {"active", "network_allow", "offline_env"}
    assert result["network_guard"]["active"] is False
    assert set(result["network_guard"]["offline_env"]) == {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
    assert not (data_root / "receipts").exists()
    json.dumps(result)


def test_presence_without_receipt_is_unvalidated_not_ready(env: FakeEnv) -> None:
    result = report(env.config)["components"]
    for name in ("decoder", "asr", "acoustics"):
        assert result[name]["status"] == "unvalidated", name
        assert (
            result[name]["reason"] == f"no smoke-receipt (receipts/{name}.json under the data root)"
        )
        assert result[name]["last_smoke_at"] is None
    assert result["decoder"]["fingerprint"]["versions"] == {"ffmpeg": "0.test", "ffprobe": "0.test"}
    assert result["asr"]["fingerprint"]["model"] == {
        "repo": "o/n",
        "revision": REVISION,
        "files": {"config.json": 8, "model.bin": 8},
    }
    assert result["acoustics"]["fingerprint"]["vad"] == {"asset": "silero_vad_v6.onnx", "size": 4}


def test_real_presence_without_receipt_is_unvalidated_not_ready(tmp_path: Path) -> None:
    """No fakes: whatever is really installed on this machine stays `unvalidated` without a
    receipt. CI installs ffmpeg, so the decoder is present there and here."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    result = report(Config(allowed_roots=(tmp_path,), data_root=data_root))["components"]
    present = [
        name
        for name, component in result.items()
        if component["status"] not in ("unavailable", "permission_required")
    ]
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        assert "decoder" in present
        versions = result["decoder"]["fingerprint"]["versions"]
        assert versions["ffmpeg"] and versions["ffprobe"]
    for name in present:
        assert result[name]["status"] == "unvalidated", name
        assert (
            result[name]["reason"] == f"no smoke-receipt (receipts/{name}.json under the data root)"
        )
        assert result[name]["last_smoke_at"] is None
    assert not (data_root / "receipts").exists()


def test_audio_review_is_permission_required_until_a_backend_is_chosen(env: FakeEnv) -> None:
    receipts = env.data_root / "receipts"
    receipts.mkdir()
    current = fingerprint("audio_review", env.config)
    (receipts / "audio_review.json").write_text(
        json.dumps(
            {
                "receipt_version": 1,
                "component": "audio_review",
                "status": "ok",
                "smoke_at": "2026-11-17T00:00:00+00:00",
                "fingerprint": current,
                "smoke": SMOKE,
            }
        )
    )
    component = component_status("audio_review", env.config, env.data_root)
    assert component["status"] == "permission_required"
    assert (
        component["reason"] == "audio backend is not selected or allowed (backend = none, Gate 0)"
    )
    assert component["last_smoke_at"] is None
    assert component["fingerprint"]["backend"] == {"name": "none", "model": None, "revision": None}


def test_no_ffmpeg_on_path_is_unavailable(env: FakeEnv) -> None:
    (env.bin_dir / "ffmpeg").unlink()
    component = component_status("decoder", env.config, env.data_root)
    assert component["status"] == "unavailable"
    assert component["reason"] == "ffmpeg not found on PATH"
    assert component["fingerprint"]["versions"] == {"ffmpeg": None, "ffprobe": "0.test"}


def test_failing_ffprobe_is_unavailable_with_its_exit_code(env: FakeEnv) -> None:
    env.write_tool("ffprobe", exit_code=3)
    component = component_status("decoder", env.config, env.data_root)
    assert component["status"] == "unavailable"
    assert component["reason"] == "ffprobe -version failed: 3"
    assert component["fingerprint"]["versions"]["ffprobe"] is None


@pytest.mark.parametrize(
    ("component", "distribution"),
    [("asr", "ctranslate2"), ("acoustics", "praat-parselmouth"), ("acoustics", "onnxruntime")],
)
def test_missing_package_is_unavailable(env: FakeEnv, component: str, distribution: str) -> None:
    env.versions[distribution] = None
    result = component_status(component, env.config, env.data_root)
    assert result["status"] == "unavailable"
    assert result["reason"] == f"package {distribution} is not installed"
    assert result["fingerprint"]["versions"][distribution] is None


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"HF_HUB_CACHE": "/hub-cache", "HUGGINGFACE_HUB_CACHE": "/legacy"}, "/hub-cache"),
        ({"HUGGINGFACE_HUB_CACHE": "/legacy", "HF_HOME": "/hf-home"}, "/legacy"),
        ({"HF_HOME": "/hf-home", "XDG_CACHE_HOME": "/xdg"}, "/hf-home/hub"),
        ({"XDG_CACHE_HOME": "/xdg"}, "/xdg/huggingface/hub"),
        ({"HF_HUB_CACHE": "~/hub-cache", "HOME": "/users/me"}, "/users/me/hub-cache"),
        ({"HF_HOME": "$XDG_CACHE_HOME/hf", "XDG_CACHE_HOME": "/xdg"}, "/xdg/hf/hub"),
        ({"HOME": "/users/me"}, "/users/me/.cache/huggingface/hub"),
    ],
)
def test_hf_hub_cache_follows_the_huggingface_hub_order(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], expected: str
) -> None:
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert doctor.hf_hub_cache() == Path(expected)


def test_empty_hf_cache_is_unavailable_and_names_the_permission(
    env: FakeEnv, tmp_path, monkeypatch
):
    empty = tmp_path / "empty-cache"
    empty.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(empty))
    component = component_status("asr", env.config, env.data_root)
    assert component["status"] == "unavailable"
    assert component["reason"] == (
        "model weights not in the local cache: o/n; downloading needs the owner's permission"
    )
    assert component["fingerprint"]["model"] == {"repo": "o/n", "revision": None, "files": {}}


def test_snapshot_without_weights_is_unavailable(env: FakeEnv) -> None:
    config = replace(env.config, asr=Asr(model="o/other"))
    env.add_snapshot(REVISION, repo="o/other", files=("config.json",))
    component = component_status("asr", config, env.data_root)
    assert component["status"] == "unavailable"
    assert component["reason"].startswith("model weights not in the local cache: o/other")
    assert component["fingerprint"]["model"]["files"] == {"config.json": 8}


def test_two_snapshots_without_a_revision_are_ambiguous(env: FakeEnv) -> None:
    env.add_snapshot(OTHER_REVISION)
    component = component_status("asr", env.config, env.data_root)
    assert component["status"] == "unavailable"
    assert component["reason"] == "several snapshots of o/n in the cache and no asr.revision set"


def test_refs_main_picks_among_several_snapshots(env: FakeEnv) -> None:
    env.add_snapshot(OTHER_REVISION)
    refs = env.hf_cache / "models--o--n" / "refs"
    refs.mkdir()
    (refs / "main").write_text(OTHER_REVISION + "\n")
    component = component_status("asr", env.config, env.data_root)
    assert component["status"] == "unvalidated"
    assert component["fingerprint"]["model"]["revision"] == OTHER_REVISION


def test_configured_revision_not_in_cache_is_unavailable(env: FakeEnv) -> None:
    config = replace(env.config, asr=Asr(model="o/n", revision=OTHER_REVISION[:12]))
    component = component_status("asr", config, env.data_root)
    assert component["status"] == "unavailable"
    assert component["reason"] == f"asr.revision {OTHER_REVISION[:12]} is not in the local cache"


def test_configured_revision_prefix_selects_the_snapshot(env: FakeEnv) -> None:
    env.add_snapshot(OTHER_REVISION)
    config = replace(env.config, asr=Asr(model="o/n", revision=REVISION[:7]))
    component = component_status("asr", config, env.data_root)
    assert component["status"] == "unvalidated"
    assert component["fingerprint"]["model"]["revision"] == REVISION


def test_missing_vad_asset_is_unavailable(env: FakeEnv) -> None:
    (env.package / "assets" / "silero_vad_v6.onnx").unlink()
    component = component_status("acoustics", env.config, env.data_root)
    assert component["status"] == "unavailable"
    assert component["reason"] == "Silero VAD asset not found in faster_whisper"
    assert component["fingerprint"]["vad"] == {"asset": None, "size": None}


def test_fingerprint_has_no_absolute_paths_and_a_matching_hash(env: FakeEnv) -> None:
    for component in COMPONENTS:
        current = fingerprint(component, env.config)
        text = json.dumps(current)
        assert str(env.bin_dir) not in text and str(env.hf_cache) not in text
        assert str(env.package) not in text and str(Path.home()) not in text
        assert list(current)[-1] == "hash"
        assert current["hash"] == fingerprint_hash(current)
        assert current["sound_scout"] == doctor.__version__
        assert set(current["platform"]) == {"system", "machine", "python"}


@pytest.fixture
def ready(env: FakeEnv, permissive_umask: None) -> dict[str, Path]:
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    return {
        name: write_receipt(name, env.config, SMOKE, now=now)
        for name in ("decoder", "asr", "acoustics")
    }


def test_matching_receipt_is_ready_with_its_time(env: FakeEnv, ready: dict[str, Path]) -> None:
    result = report(env.config)["components"]
    for name in ready:
        assert result[name]["status"] == "ready", name
        assert result[name]["reason"] is None
        assert result[name]["last_smoke_at"] == "2026-11-17T12:00:00+00:00"
    assert result["audio_review"]["status"] == "permission_required"


def assert_stale(component: dict, *changed: str) -> None:
    assert component["status"] == "unvalidated"
    assert component["reason"].startswith(
        "smoke-receipt of 2026-11-17T12:00:00+00:00 is stale: changed "
    )
    for key in changed:
        assert key in component["reason"]
    assert component["last_smoke_at"] == "2026-11-17T12:00:00+00:00"


def test_new_ffmpeg_version_makes_the_receipt_stale(env: FakeEnv, ready: dict[str, Path]) -> None:
    env.write_tool("ffmpeg", version="1.test")
    assert_stale(component_status("decoder", env.config, env.data_root), "versions.ffmpeg")
    assert component_status("asr", env.config, env.data_root)["status"] == "ready"


def test_changed_budgets_make_the_decoder_receipt_stale(
    env: FakeEnv, ready: dict[str, Path]
) -> None:
    config = replace(env.config, budgets=Budgets(max_subprocess_seconds=5))
    assert_stale(component_status("decoder", config, env.data_root), "config_hash")


def test_new_snapshot_makes_the_asr_receipt_stale(env: FakeEnv, ready: dict[str, Path]) -> None:
    env.add_snapshot(OTHER_REVISION)
    config = replace(env.config, asr=Asr(model="o/n", revision=OTHER_REVISION))
    assert_stale(component_status("asr", config, env.data_root), "config_hash", "model.revision")


def test_new_package_version_makes_the_receipt_stale(env: FakeEnv, ready: dict[str, Path]) -> None:
    env.versions["onnxruntime"] = "2.0.0"
    assert_stale(component_status("acoustics", env.config, env.data_root), "versions.onnxruntime")


def test_other_platform_makes_the_receipt_stale(
    env: FakeEnv, ready: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {**doctor.platform_info(), "machine": "other-arch"}
    monkeypatch.setattr(doctor, "platform_info", lambda: info)
    assert_stale(component_status("decoder", env.config, env.data_root), "platform.machine")


def test_other_sound_scout_version_makes_the_receipt_stale(
    env: FakeEnv, ready: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "__version__", "9.9.9")
    assert_stale(component_status("asr", env.config, env.data_root), "sound_scout")


def corrupt(path: Path, mutate) -> None:
    receipt = json.loads(path.read_text())
    mutate(receipt)
    path.write_text(json.dumps(receipt))


def set_component(receipt: dict) -> None:
    receipt["component"] = "asr"


def set_failed(receipt: dict) -> None:
    receipt["status"] = "failed"


def edit_fingerprint(receipt: dict) -> None:
    receipt["fingerprint"]["versions"]["ffmpeg"] = "edited"


def plant_test_double(receipt: dict) -> None:
    receipt["smoke"]["note"] = "test-double"


def drop_smoke(receipt: dict) -> None:
    del receipt["smoke"]


def set_version(receipt: dict) -> None:
    receipt["receipt_version"] = 2


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (set_component, "for component 'asr'"),
        (set_failed, "status is 'failed'"),
        (edit_fingerprint, "hash does not match"),
        (plant_test_double, "test-double"),
        (drop_smoke, "no smoke"),
        (set_version, "receipt_version"),
    ],
)
def test_untrusted_receipt_is_invalid(env: FakeEnv, ready: dict[str, Path], mutate, why: str):
    corrupt(ready["decoder"], mutate)
    component = component_status("decoder", env.config, env.data_root)
    assert component["status"] == "unvalidated"
    assert component["reason"].startswith("smoke-receipt is invalid: ")
    assert why in component["reason"]
    assert component["last_smoke_at"] is None


def test_broken_json_receipt_is_invalid(env: FakeEnv, ready: dict[str, Path]) -> None:
    ready["asr"].write_text("{not json")
    component = component_status("asr", env.config, env.data_root)
    assert component["status"] == "unvalidated"
    assert component["reason"] == "smoke-receipt is invalid: receipt is not valid JSON"


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        ('{"fingerprint": {"hash": NaN}}', "receipt is not valid JSON"),
        ('{"fingerprint": {"hash": Infinity}}', "receipt is not valid JSON"),
        ("[" * 100_000 + "]" * 100_000, "receipt is not valid JSON"),
        ("$fingerprint:1e400", "fingerprint is not canonical JSON"),
    ],
)
def test_corrupted_receipt_is_invalid_not_a_crash(
    env: FakeEnv, ready: dict[str, Path], raw: str, why: str
) -> None:
    if raw.startswith("$fingerprint:"):
        receipt = json.loads(ready["decoder"].read_text())
        receipt["fingerprint"]["versions"]["ffmpeg"] = "$"
        raw = json.dumps(receipt).replace('"$"', raw.removeprefix("$fingerprint:"))
    ready["decoder"].write_text(raw)
    component = component_status("decoder", env.config, env.data_root)
    assert component["status"] == "unvalidated"
    assert component["reason"] == f"smoke-receipt is invalid: {why}"
    assert component["last_smoke_at"] is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_receipt_is_unvalidated_not_a_crash(env: FakeEnv, ready: dict[str, Path]):
    ready["asr"].chmod(0o000)
    component = component_status("asr", env.config, env.data_root)
    assert component["status"] == "unvalidated"
    assert component["reason"] == "smoke-receipt is invalid: cannot read receipt: Permission denied"
    assert component["last_smoke_at"] is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unsearchable_receipts_dir_is_unvalidated_not_a_crash(env: FakeEnv, ready: dict[str, Path]):
    receipts_dir = ready["asr"].parent
    receipts_dir.chmod(0o000)
    try:
        component = component_status("asr", env.config, env.data_root)
    finally:
        receipts_dir.chmod(0o700)
    assert component["status"] == "unvalidated"
    assert component["reason"] == "smoke-receipt is invalid: cannot read receipt: Permission denied"


def test_symlinked_receipt_is_invalid(env: FakeEnv, ready: dict[str, Path]) -> None:
    target = env.data_root / "elsewhere.json"
    ready["acoustics"].rename(target)
    ready["acoustics"].symlink_to(target)
    component = component_status("acoustics", env.config, env.data_root)
    assert component["status"] == "unvalidated"
    assert component["reason"] == "smoke-receipt is invalid: receipt file is a symlink"


def test_symlinked_receipts_dir_is_invalid(env: FakeEnv) -> None:
    real = env.data_root / "real-receipts"
    real.mkdir()
    (env.data_root / "receipts").symlink_to(real, target_is_directory=True)
    component = component_status("decoder", env.config, env.data_root)
    assert component["reason"] == "smoke-receipt is invalid: receipts directory is a symlink"
    with pytest.raises(ReceiptRejected, match="symlink"):
        write_receipt("decoder", env.config, SMOKE)


def test_write_receipt_is_private_and_replaces_the_old_one(env: FakeEnv, permissive_umask: None):
    first = write_receipt("decoder", env.config, SMOKE, now=datetime(2026, 1, 1, tzinfo=UTC))
    assert first == env.data_root / "receipts" / "decoder.json"
    assert mode_of(first.parent) == 0o700 and mode_of(first) == 0o600
    receipt = json.loads(first.read_text())
    assert receipt["receipt_version"] == 1 and receipt["status"] == "ok"
    assert receipt["smoke_at"] == "2026-01-01T00:00:00+00:00"
    assert receipt["smoke"] == SMOKE
    assert receipt["fingerprint"] == fingerprint("decoder", env.config)
    assert read_receipt("decoder", env.data_root) == receipt
    text = first.read_text()
    assert str(env.data_root) not in text and str(Path.home()) not in text

    second = write_receipt("decoder", env.config, SMOKE, now=datetime(2026, 2, 2, tzinfo=UTC))
    assert second == first
    assert json.loads(first.read_text())["smoke_at"] == "2026-02-02T00:00:00+00:00"
    assert sorted(os.listdir(first.parent)) == ["decoder.json"]
    assert component_status("decoder", env.config, env.data_root)["status"] == "ready"


def test_write_receipt_stamps_now_in_utc(env: FakeEnv) -> None:
    before = datetime.now(UTC).replace(microsecond=0)
    path = write_receipt("asr", env.config, SMOKE)
    stamped = datetime.fromisoformat(json.loads(path.read_text())["smoke_at"])
    assert stamped.tzinfo is not None and stamped >= before


@pytest.mark.parametrize(
    ("component", "smoke", "why"),
    [
        ("decoder", {**SMOKE, "note": "test-double"}, "test-double"),
        ("audio_review", SMOKE, "Gate 0"),
        ("asr", {**SMOKE, "note": "/home/someone/recording.wav"}, "absolute path"),
        ("asr", {**SMOKE, "paths": [r"C:\Users\x\a.wav"]}, "absolute path"),
        ("mixer", SMOKE, "unknown component"),
        ("decoder", {**SMOKE, "when": datetime(2026, 1, 1, tzinfo=UTC)}, "not JSON"),
    ],
)
def test_write_receipt_refuses(env: FakeEnv, component: str, smoke: dict, why: str) -> None:
    with pytest.raises(ReceiptRejected, match=why):
        write_receipt(component, env.config, smoke)
    assert not (env.data_root / "receipts").exists()


def test_write_receipt_refuses_a_test_double_fingerprint(env: FakeEnv) -> None:
    env.write_tool("ffmpeg", version="test-double")
    with pytest.raises(ReceiptRejected, match="fingerprint contains"):
        write_receipt("decoder", env.config, SMOKE)


def test_write_receipt_refuses_an_unavailable_component(env: FakeEnv) -> None:
    (env.bin_dir / "ffprobe").unlink()
    with pytest.raises(ReceiptRejected, match="unavailable: ffprobe not found on PATH"):
        write_receipt("decoder", env.config, SMOKE)
    assert not (env.data_root / "receipts").exists()


def test_read_receipt_is_none_without_a_file(env: FakeEnv) -> None:
    assert read_receipt("decoder", env.data_root) is None
    (env.data_root / "receipts").mkdir()
    assert read_receipt("decoder", env.data_root) is None
    with pytest.raises(ValueError, match="unknown component"):
        read_receipt("mixer", env.data_root)


CHILD = """
import json, os, subprocess, sys
from pathlib import Path
from sound_scout import doctor, guard
from sound_scout.config import Asr, Config

base = Path({base!r})
package = base / "pkg" / "faster_whisper"
versions = {versions!r}
doctor.installed_version = lambda name: versions.get(name)
doctor.package_dir = lambda name: package if name == "faster_whisper" else None
config = Config(
    allowed_roots=(base / "recordings",), data_root=base / "data", asr=Asr(model="o/n")
)
watched = [base / "data", base / "hf", base / "bin", base / "pkg"]

def snapshot():
    seen = []
    for root in watched:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in [""] + sorted(filenames):
                path = os.path.join(dirpath, name) if name else dirpath
                info = os.lstat(path)
                seen.append((os.path.relpath(path, base), info.st_size, info.st_mtime_ns))
    return seen

spawned, imported = [], []
def audit(event, args):
    if event == "subprocess.Popen":
        spawned.append(list(args[1]))
    elif event == "import":
        imported.append(args[0])

assert guard.install_network_guard(config) is True
sys.addaudithook(audit)
before = snapshot()
try:
    result = doctor.report(config)
    denied = None
except guard.NetworkDenied as error:
    result, denied = None, str(error)
after = snapshot()
print(json.dumps({{
    "same": before == after,
    "receipts_exists": (base / "data" / "receipts").exists(),
    "denied": denied,
    "spawned": spawned,
    "heavy_imports": sorted(set(imported) & set({heavy!r})),
    "statuses": {{k: v["status"] for k, v in result["components"].items()}} if result else None,
    "guard_active": result["network_guard"]["active"] if result else None,
}}))
"""


def test_report_opens_no_network_runs_no_inference_and_writes_nothing(env: FakeEnv) -> None:
    base = env.data_root.parent
    script = CHILD.format(base=str(base), versions=VERSIONS, heavy=list(HEAVY_MODULES))
    child_env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=child_env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["denied"] is None
    assert outcome["same"] is True
    assert outcome["receipts_exists"] is False
    assert sorted(outcome["spawned"]) == [
        [str(env.bin_dir / "ffmpeg"), "-version"],
        [str(env.bin_dir / "ffprobe"), "-version"],
    ]
    assert outcome["heavy_imports"] == []
    assert outcome["statuses"] == {
        "decoder": "unvalidated",
        "asr": "unvalidated",
        "acoustics": "unvalidated",
        "audio_review": "permission_required",
    }
    assert outcome["guard_active"] is True
