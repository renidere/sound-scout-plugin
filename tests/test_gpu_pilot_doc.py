"""Decision 0027 preregisters the approved #148 package byte for byte before the rental."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from sound_scout.eval import gate0, vast

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs/decisions/0027-gpu-pilot.md"
PACKAGE_SHA256 = "1ef5be76305ccd41adafe4601b5f34da16bc36166d8774469f967c82df24c105"
ISSUE = "https://github.com/renidere/sound-scout-plugin/issues/148#issuecomment-"
QUOTED_OFFER = {
    "dph_total": 0.582222,
    "inet_down_cost": 0.0026042,
    "inet_up_cost": 0.0039063,
    "num_gpus": 1,
    "gpu_ram": 46068,
    "cpu_ram": 512984,
    "compute_cap": 860,
    "cuda_max_good": 13.2,
    "cpu_arch": "amd64",
    "verification": "verified",
    "reliability2": 0.9961,
}


def text() -> str:
    return DOC.read_text(encoding="utf-8")


def package_block() -> bytes:
    blocks = re.findall(r"```json\n(.*?)```", text(), re.DOTALL)
    assert len(blocks) == 1
    return blocks[0].encode("utf-8")


def table_rows(document: str) -> dict[str, list[str]]:
    rows = {}
    for line in document.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| `") and len(cells) == 5:
            rows[cells[0].strip("`")] = cells
    return rows


def test_package_block_is_the_approved_file_and_the_stack_accepts_it(tmp_path: Path) -> None:
    block = package_block()
    assert hashlib.sha256(block).hexdigest() == PACKAGE_SHA256
    assert f"`{PACKAGE_SHA256}`" in text()
    path = tmp_path / "gpu-pilot-148.json"
    path.write_bytes(block)
    package, sha = vast.load_package(path, vast.MODELS)
    assert sha == PACKAGE_SHA256
    assert package["preregistration"].startswith("docs/decisions/0027-gpu-pilot.md ")
    assert package["offer_id"] == 55773524 and "`55773524`" in text()
    assert [entry["name"] for entry in package["models"]] == ["qwen25omni"]
    assert (REPO / package["models"][0]["decision"]).is_file()


def test_a_changed_package_no_longer_matches_the_preregistration(tmp_path: Path) -> None:
    changed = json.loads(package_block())
    changed["ceiling_usd"] = 4.0
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed, indent=2, ensure_ascii=False), encoding="utf-8")
    assert vast.load_package(path, vast.MODELS)[1] != PACKAGE_SHA256


def test_quoted_offer_fits_the_envelope_and_the_ceiling() -> None:
    package = json.loads(package_block())
    assert vast.envelope_failures(QUOTED_OFFER, package["envelope"]) == []
    hours = package["deadline_minutes"] / 60
    at_deadline = vast.estimate_usd(QUOTED_OFFER, hours, package["in_gb"], package["out_gb"])
    assert round(at_deadline, 2) == 0.67 and at_deadline <= package["ceiling_usd"]
    top = {field: rule["lte"] for field, rule in package["envelope"].items() if "lte" in rule}
    assert vast.estimate_usd(top, hours, package["in_gb"], package["out_gb"]) <= 1.0
    for line in (
        "**Потолок пакета — 1,00 USD**",
        "**Ранний порог — 0,65 USD**",
        "**Дедлайн — 60 мин от `opened`**",
        "**Окно старта — до 2026-11-23 05:00 UTC**",
        "| На дедлайне, 60 мин (проверка стека перед create) | 0,67 |",
    ):
        assert line in text(), line


def test_over_envelope_offer_is_refused() -> None:
    package = json.loads(package_block())
    dearer = {**QUOTED_OFFER, "dph_total": 0.65, "gpu_ram": 24000}
    assert vast.envelope_failures(dearer, package["envelope"]) == ["dph_total", "gpu_ram"]


def test_clip_table_matches_the_generators() -> None:
    rows = table_rows(text())
    pcm = {
        "S0": gate0.pcm16(gate0.silence()),
        "N1": gate0.pcm16(gate0.pink_noise()),
        "T1": gate0.pcm16(gate0.tone()),
    }
    wavs = {clip_id: gate0.wav_bytes(data) for clip_id, data in pcm.items()}
    wavs.update(vast.pilot_clips())
    pcm.update({clip_id: data[44:] for clip_id, data in vast.pilot_clips().items()})
    assert set(wavs) == {"S0", "N1", "T1", "X1", "X2"}
    for clip_id, data in wavs.items():
        assert rows[clip_id][3] == f"`{hashlib.sha256(data).hexdigest()}`", clip_id
        assert rows[clip_id][4] == f"`{hashlib.sha256(pcm[clip_id]).hexdigest()}`", clip_id


@pytest.mark.parametrize(
    "pinned",
    [
        vast.IMAGE_REF,
        vast.IMAGE["amd64_digest"],
        vast.MODELS["qwen25omni"]["revision"],
        vast.MODELS["qwen25omni"]["repo"],
    ],
)
def test_image_and_model_are_the_pinned_ones(pinned: str) -> None:
    assert f"`{pinned}" in text()


def test_model_size_matches_the_pinned_files() -> None:
    files = vast.MODELS["qwen25omni"]["files"]
    assert f"{len(files)} файлов, 22 379 037 779 байт" in text()
    assert sum(entry["size"] for entry in files) == 22_379_037_779


def test_links_resolve_and_no_personal_paths() -> None:
    for comment in ("7074534146", "7074848111"):
        assert f"{ISSUE}{comment}" in text()
    for name in set(re.findall(r"`(00\d\d)`", text())):
        assert len(list((REPO / "docs/decisions").glob(f"{name}-*.md"))) == 1, name
    assert not re.search(r"/(home|Users)/|[A-Za-z]:\\", text())
    assert "аренды не было, расход — 0 USD" in text()
