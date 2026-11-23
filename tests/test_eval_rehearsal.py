"""The rehearsal of the #149 package: every scenario as a dry run of both models on the fakes.

Each scenario runs the whole stack (`vast.dry_run`: the fake Vast API, a host directory with
the real `hostrun.py serve` and guard, the fake vLLM, the real Gate 0 and extended clients) and
its own checks (`rehearsal.SCENARIOS`); a failing check names itself in the assertion. The
fakes prove the order and the contract of the stack, never the image, vLLM, a model or Vast.ai.
"""

import json
import re

import pytest

from sound_scout.eval import rehearsal


def _failed(result: dict) -> list:
    return [item for item in result["checks"] if not item["passed"]]


@pytest.mark.slow
@pytest.mark.parametrize("scenario", rehearsal.SCENARIOS, ids=lambda s: s.name)
def test_scenario(tmp_path, scenario) -> None:
    result = rehearsal.run_scenario(scenario, tmp_path / scenario.name)
    assert result["checks"], "a scenario without checks proves nothing"
    assert not _failed(result), json.dumps(_failed(result), ensure_ascii=False)[:2000]


@pytest.mark.parametrize("name", sorted(rehearsal.UNITS))
def test_unit(tmp_path, name) -> None:
    result = rehearsal.UNITS[name](tmp_path / name)
    assert result["checks"] and not _failed(result), _failed(result)


def test_every_plan_scenario_has_a_rehearsal() -> None:
    names = {s.name for s in rehearsal.SCENARIOS} | set(rehearsal.UNITS)
    assert {
        "success", "m1_bad_hash", "m1_exit_before_health", "m1_pilot_no_audio_tokens",
        "m1_crash_in_gate0", "m1_hang_in_er", "m2_weights_late", "m1_download_stalled",
        "soft_end_in_m2_er", "no_heartbeat", "hard_deadline", "create_lost", "spend_smooth",
        "spend_jump", "ssh_lost", "server_not_stopping", "interrupt_in_m2", "tunnel_death",
        "gpu_not_free", "offer_fallback", "final_pack_blocked", "compressed_time",
        "server_not_stopping_after_failure", "extended_error",
        "network_blip_m1", "tunnel_slow_reopen", "half_alive_tunnel", "sigterm_mid_gate0",
        "sigterm_in_final_path", "truncated_gate0_day", "spend_tail_real_numbers",
        "credit_below_ceiling", "gpu_ext_sha_mismatch",
        "refusals", "analysis", "pack",
    } <= names  # fmt: skip


def test_the_report_lists_every_scenario_and_what_it_cannot_prove(tmp_path) -> None:
    report = rehearsal.run(tmp_path / "r", only=["refusals"])
    assert report["passed"] and [r["scenario"] for r in report["scenarios"]] == ["refusals"]
    text = (tmp_path / "r" / "report.md").read_text()
    assert "Чего репетиция не доказывает" in text and "`refusals`" in text
    saved = json.loads((tmp_path / "r" / "report.json").read_text())
    assert saved["cannot_prove"] and re.fullmatch(r"[0-9a-f]{40}", saved["code"]["commit"])
    assert f"Коммит: `{saved['code']['commit']}`" in text


def test_the_real_numbers_scenario_reads_the_preregistered_package() -> None:
    found = rehearsal.real_package()
    package = found["package"]
    assert (package["ceiling_usd"], package["early_stop_usd"], package["spend_soft_usd"]) == (
        3.0,
        2.25,
        1.9,
    )
    assert package["envelope"]["dph_total"] == {"lte": 1.2}
    assert found["models"][0]["weights_timeout_s"] == 600
    assert found["models"][1]["weights_timeout_s"] == 1500
