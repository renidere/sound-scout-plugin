"""The rehearsal of the #149 package: every scenario as a dry run of both models on the fakes.

Each scenario runs the whole stack (`vast.dry_run`: the fake Vast API, a host directory with
the real `hostrun.py serve` and guard, the fake vLLM, the real Gate 0 and extended clients) and
its own checks (`rehearsal.SCENARIOS`); a failing check names itself in the assertion. The
fakes prove the order and the contract of the stack, never the image, vLLM, a model or Vast.ai.

CI runs the scenarios and the units in the `rehearsal` job, one shard per runner (#167):
`REHEARSAL_SHARD=k/n` keeps every n-th name from the k-th; without it all of them run.
"""

import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sound_scout.eval import rehearsal, vast
from sound_scout.eval.gpu_host import hostrun

REPO_ROOT = Path(__file__).resolve().parent.parent
REHEARSED = [s.name for s in rehearsal.SCENARIOS] + sorted(rehearsal.UNITS)


def shard(names: list[str], spec: str) -> list[str]:
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", spec)
    if not match or int(match[1]) > int(match[2]):
        raise ValueError(f"REHEARSAL_SHARD must be k/n with 1 <= k <= n, got {spec!r}")
    return names[int(match[1]) - 1 :: int(match[2])]


SELECTED = (
    set(shard(REHEARSED, os.environ["REHEARSAL_SHARD"]))
    if os.environ.get("REHEARSAL_SHARD")
    else set(REHEARSED)
)


def _failed(result: dict) -> list:
    return [item for item in result["checks"] if not item["passed"]]


@pytest.mark.slow
@pytest.mark.rehearsal
@pytest.mark.parametrize(
    "scenario", [s for s in rehearsal.SCENARIOS if s.name in SELECTED], ids=lambda s: s.name
)
def test_scenario(tmp_path, scenario) -> None:
    result = rehearsal.run_scenario(scenario, tmp_path / scenario.name)
    assert result["checks"], "a scenario without checks proves nothing"
    assert not _failed(result), json.dumps(_failed(result), ensure_ascii=False)[:2000]


@pytest.mark.slow
@pytest.mark.rehearsal
@pytest.mark.parametrize("name", sorted(set(rehearsal.UNITS) & SELECTED))
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
    # the draw is seeded by the kind alone: the same call and length in every process
    code = (
        "from sound_scout.eval import rehearsal as r; "
        "print([r._drawn(k, m) for k, (_, _, m) in sorted(r.STEP_KINDS.items())])"
    )
    drawn = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2")
    }  # fmt: skip
    assert len(drawn) == 1
    assert drawn == {
        f"{[rehearsal._drawn(k, m) for k, (_, _, m) in sorted(rehearsal.STEP_KINDS.items())]}\n"
    }


@pytest.mark.slow
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


def test_round3_faults_are_rehearsed() -> None:
    """The faults the round-3 stack check found (#165): a second loss while the request goes
    again, a refused fork, a soft end in the /health wait, a half-dead smoke tunnel, a loss that
    takes most of the SSH window."""
    names = {s.name for s in rehearsal.SCENARIOS}
    assert {
        "double_blip_m1_gate0", "double_blip_m1_block", "double_blip_m2_block",
        "real_double_blip_m1_gate0", "flap_m1_gate0", "cut_gate0_twice", "cut_block_twice",
        "enomem_size", "enomem_final_pack", "eagain_final_pack", "soft_in_m2_smoke",
        "half_2m_m1_pilot", "half_2m_m2_health", "stall_m2_health",
        "outage_60pct_m1_gate0", "outage_60pct_m2_block", "outage_60pct_m2_health",
        "outage_60pct_stop", "outage_60pct_final_get",
    } <= names  # fmt: skip
    real = next(s for s in rehearsal.SCENARIOS if s.name == "real_double_blip_m1_gate0")
    assert real.setup["intervals"] == {"ssh_lost_s": 300.0, "ssh_retry_s": 30.0}


def test_the_extended_check_counts_against_the_plan(tmp_path) -> None:
    out = tmp_path / "o"
    attempt = out / "gpu-ext-fixture" / "attempts" / "x-m"
    attempt.mkdir(parents=True)
    (out / "gpu-ext-fixture" / "plan.json").write_text(json.dumps({"entries": [{}, {}, {}]}))
    lines = [{"sent": True, "http_status": 200}] * 2
    (attempt / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    outcome = rehearsal.Outcome("t", out, None, None)
    checks = rehearsal.Checks()
    rehearsal.ext_complete(outcome, checks, "m")
    assert not checks.items[0]["passed"]  # two answered lines of three planned is no pass
    lines = [{"sent": True, "http_status": 200}] * 3
    (attempt / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    checks = rehearsal.Checks()
    rehearsal.ext_complete(outcome, checks, "m")
    assert all(item["passed"] for item in checks.items)


def ci_shards() -> list[str]:
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())["jobs"]
    job = jobs["rehearsal"]
    test = next(step for step in job["steps"] if step.get("name") == "Test")
    assert test["run"] == "uv run pytest -m rehearsal"
    assert test["env"]["REHEARSAL_SHARD"] == "${{ matrix.shard }}"
    assert job["strategy"]["fail-fast"] is False
    return job["strategy"]["matrix"]["shard"]


def test_the_ci_shards_run_every_scenario_and_unit_exactly_once() -> None:
    """#167: a scenario added later lands in a shard by itself; a shard missing from the CI
    matrix, or one counted twice, fails here."""
    shards = ci_shards()
    assert len(shards) >= 2 and all(spec.endswith(f"/{len(shards)}") for spec in shards)
    picked = [name for spec in shards for name in shard(REHEARSED, spec)]
    assert sorted(picked) == sorted(REHEARSED) and len(picked) == len(set(picked))
    assert len(REHEARSED) == len(set(REHEARSED)) == len(rehearsal.SCENARIOS) + len(rehearsal.UNITS)


def test_a_shard_picks_every_nth_name_and_refuses_a_bad_spec() -> None:
    names = [f"s{i}" for i in range(7)]
    assert shard(names, "1/3") == ["s0", "s3", "s6"]
    assert shard(names, "3/3") == ["s2", "s5"]
    assert shard(names, "1/1") == names
    for spec in ("0/3", "4/3", "3", "1/0", "a/b", " 1/3", "1/3 "):
        with pytest.raises(ValueError, match="REHEARSAL_SHARD"):
            shard(names, spec)


@pytest.mark.slow
def test_the_shard_env_selects_the_collected_scenarios(tmp_path) -> None:
    def collected(spec: str | None) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "REHEARSAL_SHARD"}
        if spec is not None:
            env["REHEARSAL_SHARD"] = spec
        return subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "rehearsal",
             "-p", "no:cacheprovider", str(Path(__file__))],
            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
        )  # fmt: skip

    def names(done: subprocess.CompletedProcess) -> list[str]:
        assert done.returncode == 0, done.stdout + done.stderr
        return re.findall(r"::test_(?:scenario|unit)\[([^\]]+)\]$", done.stdout, re.MULTILINE)

    assert sorted(names(collected(None))) == sorted(REHEARSED)
    assert names(collected("2/3")) == shard(REHEARSED, "2/3")
    refused = collected("4/3")
    assert refused.returncode != 0 and "REHEARSAL_SHARD must be k/n" in refused.stdout + (
        refused.stderr
    )


def _serving(out: Path) -> bool:
    """Holds until a `serve` of the dry run recorded its server session; never signals."""
    ends = time.monotonic() + 120
    while time.monotonic() < ends:
        for pid_file in out.glob("host-root/out/*/serve.pid"):
            records = hostrun._records(pid_file.with_name("server-stats.jsonl"))
            if any(r.get("event") == "started" for r in records):
                return False
        time.sleep(0.05)
    raise AssertionError("no `serve` of the dry run started")


@pytest.mark.slow
def test_a_timed_out_signalled_child_leaves_no_process_of_its_host_sessions(
    tmp_path, monkeypatch
) -> None:
    """#203: on a timeout `_signalled` SIGKILLed the child group only; `serve` and the server
    live in sessions of their own."""
    scenario = next(s for s in rehearsal.SCENARIOS if s.name == "sigterm_mid_gate0")
    scenario = dataclasses.replace(scenario, signal_when=_serving)
    monkeypatch.setattr(rehearsal, "CHILD_WAIT_S", 1.0)
    out = tmp_path / "dry"
    try:
        assert rehearsal._signalled(scenario, out) is None
        pid_files = list(out.glob("host-root/out/*/serve.pid"))
        assert pid_files
        for pid_file in pid_files:
            serve_ = hostrun._serve_identity(pid_file)
            assert serve_ is not None and not hostrun._is_running(*serve_)
            stats = pid_file.with_name("server-stats.jsonl")
            assert hostrun._session_survivors(stats, set()) == []
    finally:
        vast.stop_local_sessions(out / "host-root")


def test_signalled_fails_when_a_host_session_outlives_the_child(tmp_path, monkeypatch) -> None:
    scenario = next(s for s in rehearsal.SCENARIOS if s.name == "sigterm_mid_gate0")
    scenario = dataclasses.replace(scenario, signal_when=lambda out: False)
    monkeypatch.setattr(rehearsal, "CHILD_WAIT_S", 0.2)
    monkeypatch.setattr(vast, "stop_local_sessions", lambda root: False)
    with pytest.raises(RuntimeError, match="outlived the dry run"):
        rehearsal._signalled(scenario, tmp_path / "dry")


@pytest.mark.slow
def test_signalled_kills_the_child_and_its_host_sessions_when_signal_when_raises(
    tmp_path, monkeypatch
) -> None:
    """The child and the sessions of its fake host do not outlive an error of `signal_when`."""
    children = []
    real_popen = rehearsal.subprocess.Popen
    monkeypatch.setattr(
        rehearsal.subprocess,
        "Popen",
        lambda *a, **k: children.append(real_popen(*a, **k)) or children[-1],
    )

    def broken(out: Path) -> bool:
        _serving(out)
        raise ValueError("signal_when broke")

    scenario = next(s for s in rehearsal.SCENARIOS if s.name == "sigterm_mid_gate0")
    scenario = dataclasses.replace(scenario, signal_when=broken)
    out = tmp_path / "dry"
    try:
        with pytest.raises(ValueError, match="signal_when broke"):
            rehearsal._signalled(scenario, out)
        (child,) = children
        assert child.poll() is not None
        for pid_file in out.glob("host-root/out/*/serve.pid"):
            serve_ = hostrun._serve_identity(pid_file)
            assert serve_ is not None and not hostrun._is_running(*serve_)
            stats = pid_file.with_name("server-stats.jsonl")
            assert hostrun._session_survivors(stats, set()) == []
    finally:
        vast.stop_local_sessions(out / "host-root")
