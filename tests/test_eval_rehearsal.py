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
        "blip_m1_weights_poll", "blip_m2_weights_poll", "blip_m2_smoke_tunnel", "blip_gpu_free",
        "blip_final_download", "outage_1s_gate0", "incomplete_read_mid_body",
        "sigterm_during_create_put", "watchdog_destroy_in_download",
        "refusals", "analysis", "pack",
    } <= names  # fmt: skip


def test_a_blip_is_rehearsed_at_every_kind_of_host_step() -> None:
    names = {s.name for s in rehearsal.SCENARIOS}
    kinds = {
        "upload_mkdir", "upload_put", "upload_sha256sum", "preflight_nvidia_smi",
        "preflight_vllm_version", "preflight_hostrun", "preflight_selftest", "fetch_mkdir",
        "fetch_spawn", "weights_size", "weights_verify", "weights_receipt", "gpu_free",
        "serve_spawn", "smoke_tunnel", "server_state", "sync_pack", "sync_get", "evidence",
        "stop", "final_pack", "final_get", "pilot", "gate0_request",
    }  # fmt: skip
    assert set(rehearsal.STEP_KINDS) == kinds
    assert {f"blip_step_{kind}" for kind in kinds} <= names
    # the draw is seeded by the kind: the same call and length every run
    assert [rehearsal._drawn(k, 3) for k in kinds] == [rehearsal._drawn(k, 3) for k in kinds]


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
