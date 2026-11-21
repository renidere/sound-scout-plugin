"""Spike A of issue #95: the diagnostic Desktop probe, its SHA-256 check and decision 0016.

The node suite runs the probe against a fake SDK: it proves the probe's contract, not what the real
Hermes Desktop or gateway do. Those results come only from the owner's run (decision 0016).
"""

import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import assert_node_suite_passes, declared_node_tests

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN = REPO_ROOT / "hermes" / "probe" / "sound-scout-probe-a" / "plugin.js"
SCRIPT = REPO_ROOT / "scripts" / "probe_a_check_attach.py"
NODE_SUITE = REPO_ROOT / "tests" / "js" / "probe_a.test.mjs"
LOADER = REPO_ROOT / "tests" / "js" / "load_plugin.mjs"
DECISION = REPO_ROOT / "docs" / "decisions" / "0016-spike-a-desktop-probe.md"
RPC_METHODS = ("session.resume", "file.attach", "prompt.submit", "prompt.receipt")
PENDING = "ожидает прогона владельцем"

spec = importlib.util.spec_from_file_location("probe_a_check_attach", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.fixture(scope="module")
def reference() -> bytes:
    return check.synthetic_wav()


def run_check(capsys, *args: str) -> tuple[int, dict]:
    code = check.main(list(args))
    return code, json.loads(capsys.readouterr().out)


def tail(text: str) -> str:
    return "\n".join(text.splitlines()[-40:])


def test_probe_node_suite_passes(node: str) -> None:
    assert assert_node_suite_passes(node, NODE_SUITE, REPO_ROOT) >= 43


def test_declared_tests_count_each_loop_element() -> None:
    source = """test('one', () => {})
for (const [name, x] of [['a, b', [1, 2]], ['c\\'d', { y: [3] }], ['e', 4],]) {
  test(`loop ${name}`, () => {
    for (const z of [1, 2, 3]) {}
  })
}
for (const a of ['p', 'q']) {
  for (const b of [1, 2, 3]) {
    test(`${a}${b}`, () => {})
  }
}
test('two', async () => {
  for (const round of [1, 2]) {}
})
"""
    assert declared_node_tests(source) == 2 + 3 + 6


def test_a_suite_without_test_calls_fails_the_check(node: str, tmp_path: Path) -> None:
    suite = tmp_path / "empty.test.mjs"
    suite.write_text("import test from 'node:test'\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="declares no test"):
        assert_node_suite_passes(node, suite, REPO_ROOT)


def test_a_suite_that_runs_fewer_tests_than_it_declares_fails(node: str, tmp_path: Path) -> None:
    suite = tmp_path / "short.test.mjs"
    suite.write_text(
        "import test from 'node:test'\ntest('a', () => {})\nif (false) {\ntest('b', () => {})\n}\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="declared 2"):
        assert_node_suite_passes(node, suite, REPO_ROOT)


@pytest.mark.parametrize(
    "source",
    [
        "test('a', () => {})\nif (false) {\n  test('b', () => {})\n}\n",
        "for (const x of [1, 2]) {\ntest('b', () => {})\n}\n",
    ],
    ids=["indented-outside-loop", "not-deeper-than-loop"],
)
def test_an_uncounted_test_call_fails_closed(source: str) -> None:
    with pytest.raises(ValueError, match="test\\(\\) call outside a counted loop"):
        declared_node_tests(source)


def test_probe_wav_is_byte_identical_to_the_backend_generator(node: str, reference: bytes) -> None:
    program = (
        "import { createHash } from 'node:crypto';"
        f"import {{ loadPlugin }} from {json.dumps(LOADER.as_uri())};"
        "const bytes = await (await loadPlugin()).syntheticWav();"
        "console.log(createHash('sha256').update(bytes).digest('hex'), bytes.length);"
    )
    result = subprocess.run(
        [node, "--experimental-vm-modules", "--input-type=module", "-e", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, tail(result.stderr)
    digest, size = result.stdout.split()
    assert (digest, int(size)) == (hashlib.sha256(reference).hexdigest(), len(reference))


def test_synthetic_wav_is_30_s_of_stereo_pcm16_at_48_khz(reference: bytes) -> None:
    assert len(reference) == 5_760_044
    assert check.parse_wav(reference) == {
        "format": 1,
        "channels": 2,
        "sample_rate": 48_000,
        "bits": 16,
        "frames": 1_440_000,
        "seconds": 30.0,
    }
    assert struct.unpack_from("<hh", reference, 44 + 4 * 25) == (8000, 4000)
    assert struct.unpack_from("<hh", reference, 44 + 4 * 75) == (-8000, 0)


def test_matching_file_passes_both_checks(tmp_path: Path, capsys, reference: bytes) -> None:
    path = tmp_path / "sound-scout-probe-a-0123456789ab.wav"
    path.write_bytes(reference)
    digest = hashlib.sha256(reference).hexdigest()
    code, out = run_check(capsys, str(path), "--sha256", digest.upper())
    assert code == 0
    assert out["ok"] and out["matches_log"] and out["matches_synthetic"]
    assert (out["bytes"], out["sha256"], out["wav"]["seconds"]) == (len(reference), digest, 30.0)


def test_one_changed_byte_fails_the_synthetic_check(tmp_path: Path, capsys, reference) -> None:
    changed = bytearray(reference)
    changed[1000] ^= 1
    path = tmp_path / "x.wav"
    path.write_bytes(changed)
    code, out = run_check(capsys, str(path), "--sha256", hashlib.sha256(changed).hexdigest())
    assert code == 1
    assert out["matches_log"] and not out["matches_synthetic"] and not out["ok"]


def test_a_wrong_log_digest_fails_the_log_check(tmp_path: Path, capsys, reference: bytes) -> None:
    path = tmp_path / "x.wav"
    path.write_bytes(reference)
    code, out = run_check(capsys, str(path), "--sha256", "0" * 64)
    assert code == 1
    assert not out["matches_log"] and out["matches_synthetic"]


def test_refusals(tmp_path: Path, capsys, reference: bytes) -> None:
    good = "a" * 64
    wav = tmp_path / "x.wav"
    wav.write_bytes(reference)
    link = tmp_path / "link.wav"
    link.symlink_to(wav)
    big = tmp_path / "big.wav"
    big.touch()
    os.truncate(big, check.MAX_BYTES + 1)
    cases = [
        ((str(wav), "--sha256", "abc"), "bad_sha256"),
        ((str(wav), "--sha256", "g" * 64), "bad_sha256"),
        ((str(tmp_path / "missing.wav"), "--sha256", good), "not_found"),
        ((str(link), "--sha256", good), "symlink"),
        ((str(tmp_path), "--sha256", good), "not_regular"),
        ((str(big), "--sha256", good), "too_large"),
    ]
    for args, error in cases:
        assert run_check(capsys, *args) == (2, {"ok": False, "error": error}), error


def test_unreadable_paths_are_refused_without_the_path(tmp_path: Path, capsys, reference) -> None:
    wav = tmp_path / "private-chat.wav"
    wav.write_bytes(reference)
    under_file = wav / "x.wav"
    cases = [(under_file, "unreadable")]
    locked = tmp_path / "locked-chat.wav"
    locked.write_bytes(reference)
    locked.chmod(0)
    if os.geteuid() != 0:
        cases.append((locked, "unreadable"))
    try:
        for path, error in cases:
            code, out = run_check(capsys, str(path), "--sha256", "a" * 64)
            assert (code, out) == (2, {"ok": False, "error": error}), path.name
    finally:
        locked.chmod(0o600)


def test_a_symlink_swapped_in_after_lstat_is_refused_as_symlink(
    tmp_path: Path, capsys, monkeypatch, reference: bytes
) -> None:
    wav = tmp_path / "x.wav"
    wav.write_bytes(reference)
    link = tmp_path / "link.wav"
    link.symlink_to(wav)
    monkeypatch.setattr(check.os, "lstat", os.stat)
    refusal = (2, {"ok": False, "error": "symlink"})
    assert run_check(capsys, str(link), "--sha256", "a" * 64) == refusal


def test_output_holds_neither_the_path_nor_the_content(tmp_path: Path, capsys) -> None:
    chat_name = "private-chat-name"
    path = tmp_path / f"{chat_name}.wav"
    path.write_bytes(b"RIFF secret transcript words")
    code, out = run_check(capsys, str(path), "--sha256", "0" * 64)
    text = json.dumps(out)
    assert code == 1 and out["wav"] is None
    assert chat_name not in text and str(tmp_path) not in text and "transcript" not in text


def test_the_script_runs_isolated_as_the_owner_runs_it(tmp_path: Path, reference: bytes) -> None:
    path = tmp_path / "x.wav"
    path.write_bytes(reference)
    digest = hashlib.sha256(reference).hexdigest()
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(path), "--sha256", digest],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_decision_0016_records_the_owner_run_results(reference: bytes) -> None:
    text = DECISION.read_text(encoding="utf-8")
    for needle in (
        hashlib.sha256(reference).hexdigest(),
        "hermes/probe/sound-scout-probe-a/plugin.js",
        "scripts/probe_a_check_attach.py",
        "sound-scout-probe-a",
        *RPC_METHODS,
        "\n## Инструкция владельцу\n",
        "\n## Правило анализа\n",
        "\n## Топология прогона\n",
    ):
        assert needle in text, needle
    status = next(line for line in text.splitlines() if line.startswith("- **Статус:**"))
    assert "результаты прогона 2026-11-21" in status
    status_block = text.split("- **Статус:**", 1)[1].split("\n- **", 1)[0]
    assert "(2а)" in status_block and "(4б)" in status_block
    assert PENDING not in text
    assert "ожидают прогона" not in text

    results = text.split("\n## Результаты", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in results.splitlines() if line.startswith("|")][2:]
    assert len(rows) == 11
    assert not any("неперенесённый" in row for row in rows)
    for row in rows:
        cell = row.rstrip(" |").rsplit("|", 1)[1].strip()
        assert len(cell) > 40, row
    partial = [row for row in rows if "Измерено частично" in row]
    assert [row.split("|")[1].strip()[:2] for row in partial] == ["2а", "4б"]
    assert "probe-a-log.clipboard.jsonl" in results
    assert "state.db" in results

    conclusions = text.split("\n## Выводы для #103\n", 1)[1].split("\n## ", 1)[0]
    for needle in (
        "`prompt.submit` с `submission_id`",
        "`composer.submit` — не путь доставки",
        "fail-closed",
        "`retainProfile` → `session.resume`",
        "`file.attach`",
        "Открыто:",
        "ACK для скрытой цели не записан",
        "предварительный",
        "«удаляй»",
    ):
        assert needle in conclusions, needle

    assert not re.search(r"/home/[a-z]|/root/|[A-Za-z]:\\Users\\[A-Za-z]", text)
    filled = results + conclusions
    assert "completed" not in filled, "not a session_admissions status value"
    assert not re.search(r"local-[0-9a-f]{9,}", filled), "session ids must be shortened"
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-", filled), "raw connectionId"
    assert not re.search(r"\b[0-9a-f]{32}\b", filled), "raw admission id"
    assert set(re.findall(r"conn-\w+", filled)) == {"conn-1", "conn-2"}


def test_decision_0016_instruction_fits_the_gateway_topology() -> None:
    text = DECISION.read_text(encoding="utf-8")
    instruction = text.split("\n## Инструкция владельцу\n", 1)[1].split("\n## ", 1)[0]
    for needle in (
        hashlib.sha256(PLUGIN.read_bytes()).hexdigest(),
        "Get-FileHash",
        "H:\\SoundScout\\desktop-plugins\\sound-scout-probe-a\\",
        "H:\\SoundScout\\probe-a\\probe-a-log.jsonl",
        "H:\\SoundScout\\probe-a\\notes.txt",
        "Open Desktop plugins folder",
        "%LOCALAPPDATA%\\hermes",
        "scripts/probe_a_check_attach.py",
        "session_admissions",
        "H:\\SoundScout\\probe-a\\probe-a-log.clipboard.jsonl",
        "adopt_legacy_session",
        "не перенесённые на этапе B, не открывать",
    ):
        assert needle in instruction, needle
    assert "Открыть другой чат" not in instruction
    assert "«probe-a test» до первого сообщения" not in instruction
    topology = text.split("\n## Топология прогона\n", 1)[1].split("\n## ", 1)[0]
    assert ":9119" in topology
    assert "SSH backend нет" in topology
    assert "комментарий к" not in instruction
    assert "Журнал в issue не вставлять" in instruction
    labels = set(re.findall(r"label: '(Probe A: [^']+)'", PLUGIN.read_text(encoding="utf-8")))
    assert len(labels) == 8
    told = set(re.findall(r"`(Probe A: [^`…]+)`", instruction))
    assert told == labels, told ^ labels


def test_plugin_is_one_esm_file_apart_from_the_spike_b_package() -> None:
    assert [p.name for p in PLUGIN.parent.iterdir()] == ["plugin.js"]
    source = PLUGIN.read_text(encoding="utf-8")
    assert re.search(r"^export default \{\n  id: PROBE_ID,", source, re.M)
    assert "export const PROBE_ID = 'sound-scout-probe-a'" in source
    spike_b = (REPO_ROOT / "hermes/plugin/sound-scout/desktop/plugin.js").read_text()
    assert "id: 'sound-scout'," in spike_b
