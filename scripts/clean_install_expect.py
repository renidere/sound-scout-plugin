"""What scripts/clean-install-check.sh expects of `doctor --json` and `analyze --json`.

    clean_install_expect.py <doctor.json> <analyze.json> <analyze exit code>
        `ok` and exit 0, or one `fail: <field>: expected …, got …` line per deviation and exit 1
    clean_install_expect.py --detail doctor|analyze <file.json>
        the one-line detail of that step for the summary

The clean install has `backend = none` and maybe no ASR model: `asr` may be `completed` or
`unavailable`, `audio_review` must be `unavailable` and the result therefore `partial` (exit 3).
Standard library only: it runs in the clone before anything is known to work.
"""

import json
import sys

COMPONENTS = ("decoder", "asr", "acoustics", "audio_review")
STAGES = ("prepare", "acoustics", "asr", "audio_review")
PARTIAL_EXIT_CODE = 3
USAGE = (
    "usage: clean_install_expect.py <doctor.json> <analyze.json> <analyze exit code>\n"
    "       clean_install_expect.py --detail doctor|analyze <file.json>"
)


def check(doctor: object, analyze: object, exit_code: int) -> list[str]:
    failures: list[str] = []

    def expect(field: str, got: object, allowed: tuple) -> None:
        if got not in allowed:
            wanted = " or ".join(json.dumps(item) for item in allowed)
            failures.append(f"fail: {field}: expected {wanted}, got {json.dumps(got)}")

    components = _get(doctor, "components")
    for name in COMPONENTS:
        if not isinstance(components, dict) or name not in components:
            failures.append(f"fail: doctor.components.{name}: expected present, got missing")
    expect(
        "doctor.components.audio_review.status",
        _get(doctor, "components", "audio_review", "status"),
        ("permission_required",),
    )
    expect("doctor.network_guard.active", _get(doctor, "network_guard", "active"), (True,))
    expect("analyze.status", _get(analyze, "status"), ("partial",))
    expect("analyze exit code", exit_code, (PARTIAL_EXIT_CODE,))
    for stage in ("prepare", "acoustics"):
        expect(f"analyze.stages.{stage}", _status(analyze, stage), ("completed",))
    expect("analyze.stages.asr", _status(analyze, "asr"), ("completed", "unavailable"))
    expect("analyze.stages.audio_review", _status(analyze, "audio_review"), ("unavailable",))
    return failures


def detail(step: str, document: object) -> str:
    if step == "doctor":
        return " ".join(
            f"{name}={_get(document, 'components', name, 'status')}" for name in COMPONENTS
        )
    stages = " ".join(f"{name}={_status(document, name)}" for name in STAGES)
    return f"status={_get(document, 'status')} {stages}"


def _get(document: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


def _status(analyze: object, stage: str) -> object:
    return _get(analyze, "stages", stage, "status")


def _load(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as source:
            return json.load(source)
    except (OSError, ValueError):
        return None


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "--detail" and argv[1] in ("doctor", "analyze"):
        print(detail(argv[1], _load(argv[2])))
        return 0
    if len(argv) != 3 or not argv[2].lstrip("-").isdigit():
        print(USAGE, file=sys.stderr)
        return 2
    failures = check(_load(argv[0]), _load(argv[1]), int(argv[2]))
    print("\n".join(failures) if failures else "ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
