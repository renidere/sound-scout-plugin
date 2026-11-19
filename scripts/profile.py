"""Resource profile of `sound-scout analyze` against the UX budget of R11 (T10, issue #41).

    uv run python scripts/profile.py --output <profile.json outside the repo> [--durations 30,180]

Each input is analyzed twice in a row by the real CLI started as a child process, `cold` (its
first run in this session) and `warm` (the repeat): synthetic speech-like audio of each duration
and, when `SOUND_SCOUT_DEV_AUDIO` names a file, the development recording. Wall time is measured
from the start of the child to `os.wait4`, whose rusage gives the peak RSS of the process tree;
stage times come from the run's `work/stage-*.json` and `asr/raw.json`.

The runs use a temporary configuration and data root (0700, removed at the end) with
`backend = "none"`, the operator's `[asr]` and a worker deadline raised so the actual time is
measured instead of `deadline_exceeded`. Offline: `HF_HUB_OFFLINE=1`, the network guard of the CLI.
`profile.json` holds durations, statuses and timings only: never a path, a file name, a SHA-256,
a run id or text of the recording.
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from pathlib import Path

import numpy as np

from sound_scout import __version__, config, guard, proc

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_VERSION = 1
PROFILE_DEADLINE_SECONDS = 900
DEFAULT_DURATIONS = (30.0, 180.0)
PHASES = ("cold", "warm")
STAGES = ("prepare", "acoustics", "asr", "audio_review")
NOT_RUN = ("unavailable", "skipped")
RESULT_STATUSES = ("complete", "partial")
SYNTHETIC_RATE = 48_000
SYLLABLE_PERIOD_S = 3.0
VOICED_S = 1.2
BUDGET_TARGETS = ((30, 60), (180, 180))
BUDGET = {
    "source": "docs/spec/REQUIREMENTS.md R11",
    "targets": [{"max_input_s": i, "max_processing_s": p} for i, p in BUDGET_TARGETS],
    "scope": "без audio_review: backend = none "
    "(Gate 0 NOT READY, docs/decisions/0007-gate0-result.md)",
}
NOTES = (
    "page cache не сбрасывается (для этого нужен root): cold — первый прогон входа в сессии, "
    "а не холодный диск",
    "каждый analyze запускает новый worker и загружает ASR заново: warm отличается от cold "
    "только кэшем ОС, тёплой модели в процессе нет ни в CLI, ни в MCP",
    "peak_rss_mib — ru_maxrss самого большого процесса дерева (os.wait4), а не сумма",
    "wall_ms — от запуска sound-scout analyze до его завершения, включая старт Python",
    "audio_review не измерен: backend = none, Gate 0 NOT READY",
)
TOOL_TIMEOUT_SECONDS = 30
OUTPUT_LIMIT = 1 << 16
ABSOLUTE_PATH = re.compile(r"(?<![\w.~-])/[^\s'\"(),;]+")
REFUSAL = re.compile(r"^sound-scout analyze: ([a-z_]+):", re.MULTILINE)


class ProfileError(Exception):
    """A request the script refuses before anything runs."""


def budget_for(duration_s: float) -> int | None:
    """The R11 processing budget in seconds of an input this long; `None` above 180 s."""
    return next((p for i, p in BUDGET_TARGETS if duration_s <= i), None)


def synthetic_speech(seconds: float, rate: int = SYNTHETIC_RATE) -> np.ndarray:
    """The formant control of `smoke` repeated every 3 s: 1.2 s of vowels, 1.8 s of silence."""
    from sound_scout import smoke

    vowels = smoke.speech_like_control(VOICED_S, rate)
    audio = np.zeros(int(seconds * rate), dtype=np.float32)
    start = 0.0
    while start + VOICED_S <= seconds + 1e-9:
        audio += smoke.place(vowels, start, seconds, rate)
        start += SYLLABLE_PERIOD_S
    return audio


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def redacted(text: str | None) -> str | None:
    """A stage reason without absolute paths: they may name the recording or its directory."""
    return None if text is None else ABSOLUTE_PATH.sub("<path>", text)


def stage_times(run_dir: Path, answer_stages: dict) -> dict:
    """Status and times of each stage from the run's records, no text and no path."""
    timing = _read_json(run_dir / "asr" / "raw.json", {}).get("timing", {})
    stages = {}
    for name in STAGES:
        record = _read_json(run_dir / "work" / f"stage-{name}.json", None)
        if record is None:
            status = (answer_stages.get(name) or {}).get("status")
            stages[name] = {"status": status, "elapsed_ms": None, "reason": "нет записи этапа"}
            continue
        entry = record["entry"]
        stage = {
            "status": entry["status"],
            "elapsed_ms": None if entry["status"] in NOT_RUN else entry["elapsed_ms"],
        }
        if name in ("prepare", "acoustics"):
            stage["steps"] = {step["name"]: step["elapsed_ms"] for step in record["steps"]}
        else:
            completed_asr = name == "asr" and entry["status"] == "completed"
            stage["model_load_ms"] = timing.get("model_load_ms") if completed_asr else None
            stage["inference_ms"] = timing.get("inference_ms") if completed_asr else None
        if entry["status"] != "completed":
            stage["reason"] = redacted(entry["reason"])
        stages[name] = stage
    return stages


def run_analyze(path: Path, slug: str, phase: str, env: dict, data_root: Path) -> dict:
    """One `sound-scout analyze` child: exit code, wall time, peak RSS of its tree, stage times."""
    argv = [sys.executable, "-m", "sound_scout", "analyze", str(path), "--json"]
    argv += ["--request-id", f"profile-{slug}-{phase}-{uuid.uuid4().hex}"]
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        started = time.monotonic()
        child = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env, cwd=data_root.parent
        )
        _, status, usage = os.wait4(child.pid, 0)
        wall_ms = round((time.monotonic() - started) * 1000)
        child.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read().decode("utf-8", "replace")
    try:
        answer = json.loads(stdout)
    except ValueError:
        answer = {}
    run = {
        "phase": phase,
        "exit_code": child.returncode,
        "result_status": answer.get("status"),
        "wall_ms": wall_ms,
        "peak_rss_mib": round(usage.ru_maxrss / 1024, 1),
        "within_budget": None,
    }
    if "run_dir" in answer:
        run["stages"] = stage_times(data_root / answer["run_dir"], answer.get("stages", {}))
        prepare = _read_json(data_root / answer["run_dir"] / "work" / "stage-prepare.json", {})
        duration = (prepare.get("fragment", {}).get("source") or {}).get("duration_s")
        if duration is not None:
            run["duration_s"] = round(duration, 1)
    else:
        refusal = REFUSAL.search(stderr)
        run["error"] = refusal.group(1) if refusal else "no result"
    return run


def profile_input(
    label: str, kind: str, path: Path, slug: str, duration_s: float | None, env: dict, root: Path
) -> dict:
    runs = [run_analyze(path, slug, phase, env, root) for phase in PHASES]
    if duration_s is None:
        duration_s = next((run["duration_s"] for run in runs if "duration_s" in run), None)
    for run in runs:
        run.pop("duration_s", None)
    budget_s = None if duration_s is None else budget_for(duration_s)
    for run in runs:
        if budget_s is not None and run["result_status"] in RESULT_STATUSES:
            run["within_budget"] = run["wall_ms"] / 1000 <= budget_s
    item = {
        "label": label,
        "kind": kind,
        "duration_s": duration_s,
        "budget_s": budget_s,
        "status": "measured",
        "runs": runs,
    }
    failed = [run for run in runs if run["result_status"] not in RESULT_STATUSES]
    if failed:
        item["status"] = "failed"
        item["reason"] = "; ".join(
            f"{run['phase']}: exit {run['exit_code']}, {run.get('error') or run['result_status']}"
            for run in failed
        )
    return item


def host_info() -> dict:
    """The station: CPU, memory, OS and tool versions; no host name, user or path."""
    cpuinfo = _read_text("/proc/cpuinfo")
    meminfo = _read_text("/proc/meminfo")
    cpu = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    memory = re.search(r"^MemTotal:\s*(\d+) kB$", meminfo, re.MULTILINE)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": cpu.group(1).strip() if cpu else None,
        "logical_cpus": os.cpu_count(),
        "memory_total_mib": int(memory.group(1)) // 1024 if memory else None,
        "python": platform.python_version(),
        "uv": _tool_version(["uv", "--version"], 1),
        "ffmpeg": _tool_version(["ffmpeg", "-version"], 2),
    }


def config_text(roots: list[Path], data_root: Path, asr: config.Asr) -> str:
    lines = [
        f"allowed_roots = [{', '.join(json.dumps(str(root)) for root in roots)}]",
        f"data_root = {json.dumps(str(data_root))}",
        'backend = "none"',
        "",
        "[asr]",
        f"model = {json.dumps(asr.model)}",
    ]
    if asr.revision is not None:
        lines.append(f"revision = {json.dumps(asr.revision)}")
    lines += [
        "",
        "[budgets]",
        f"worker_deadline_seconds = {PROFILE_DEADLINE_SECONDS}",
        f"max_subprocess_seconds = {PROFILE_DEADLINE_SECONDS}",
    ]
    return "\n".join(lines) + "\n"


def operator_asr() -> config.Asr:
    """`[asr]` of the operator's configuration when there is one, otherwise the default."""
    try:
        return config.load().asr
    except config.ConfigError:
        return config.Asr()


def checked_output(output: str) -> Path:
    path = Path(output).expanduser().resolve()
    if path.is_relative_to(REPO_ROOT):
        raise ProfileError(f"--output must be outside the repository {REPO_ROOT}")
    if not path.parent.is_dir():
        raise ProfileError(f"--output: no such directory: {path.parent}")
    return path


def parse_durations(text: str) -> tuple[float, ...]:
    try:
        durations = tuple(float(item) for item in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds like 30,180, got {text!r}") from None
    longest = BUDGET_TARGETS[-1][0]
    if not durations or any(not 0 < d <= longest for d in durations):
        raise argparse.ArgumentTypeError(f"durations must be in (0, {longest}] s, got {text!r}")
    return durations


def profile(durations: tuple[float, ...]) -> dict:
    asr = operator_asr()
    recording = os.environ.get("SOUND_SCOUT_DEV_AUDIO")
    recording_path = Path(recording).resolve() if recording else None
    work = Path(tempfile.mkdtemp(prefix="sound-scout-profile-"))
    try:
        synthetic_dir, data_root = work / "in", work / "data"
        synthetic_dir.mkdir(mode=0o700)
        data_root.mkdir(mode=0o700)
        roots = [synthetic_dir]
        if recording_path is not None and recording_path.is_file():
            roots.append(recording_path.parent)
        config_path = work / "sound-scout.toml"
        config_path.write_text(config_text(roots, data_root, asr), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != "SOUND_SCOUT_DEV_AUDIO"}
        env.update(
            SOUND_SCOUT_CONFIG=str(config_path), HF_HUB_OFFLINE="1", SOUND_SCOUT_NO_NETWORK="1"
        )
        notes = [*NOTES, f"load average (1 мин) перед замером: {os.getloadavg()[0]:.2f}"]
        inputs = []
        for seconds in durations:
            path = synthetic_dir / f"synthetic-{seconds:g}s.wav"
            write_wav(path, synthetic_speech(seconds), SYNTHETIC_RATE)
            label = f"synthetic {seconds:g} s"
            slug = f"synthetic-{seconds:g}".replace(".", "_")
            inputs.append(profile_input(label, "synthetic", path, slug, seconds, env, data_root))
        if recording_path is not None and recording_path.is_file():
            inputs.append(
                profile_input(
                    "development recording",
                    "development",
                    recording_path,
                    "development",
                    None,
                    env,
                    data_root,
                )
            )
        else:
            reason = (
                "SOUND_SCOUT_DEV_AUDIO не задана"
                if recording_path is None
                else "SOUND_SCOUT_DEV_AUDIO не указывает на файл"
            )
            inputs.append(
                {
                    "label": "development recording",
                    "kind": "development",
                    "status": "skipped",
                    "reason": reason,
                }
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "profile_version": PROFILE_VERSION,
        "created_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sound_scout_version": __version__,
        "commit": _tool_version(["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"]),
        "host": host_info(),
        "config": {
            "backend": "none",
            "asr_model": asr.model,
            "asr_revision": asr.revision,
            "worker_deadline_seconds": PROFILE_DEADLINE_SECONDS,
        },
        "budget": BUDGET,
        "notes": notes,
        "inputs": inputs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output", required=True, help="profile.json, outside the repository")
    parser.add_argument(
        "--durations",
        type=parse_durations,
        default=DEFAULT_DURATIONS,
        help="seconds of synthetic audio, comma-separated (default: 30,180)",
    )
    args = parser.parse_args(argv)
    try:
        output = checked_output(args.output)
    except ProfileError as error:
        print(f"profile.py: {error}", file=sys.stderr)
        return 2
    guard.disable_telemetry()
    result = profile(args.durations)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in result["inputs"]:
        walls = ", ".join(
            f"{run['phase']} {run['wall_ms'] / 1000:.1f} s" for run in item.get("runs", [])
        )
        print(f"{item['label']}: {item['status']}" + (f" ({walls})" if walls else ""))
    print(f"profile: {output}")
    return 0 if all(item["status"] != "failed" for item in result["inputs"]) else 1


def _read_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _tool_version(argv: list[str], field: int | None = None) -> str | None:
    """`field` of the first output line of a tool, or the whole line; `None` without the tool."""
    try:
        done = proc.run(argv, timeout=TOOL_TIMEOUT_SECONDS, max_output_bytes=OUTPUT_LIMIT)
    except OSError:
        return None
    lines = done.stdout.decode("utf-8", "replace").splitlines()
    if done.returncode != 0 or not lines:
        return None
    words = lines[0].split()
    if field is None:
        return lines[0].strip()
    return words[field] if len(words) > field else None


if __name__ == "__main__":
    sys.exit(main())
