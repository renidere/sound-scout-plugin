"""Console entry point `sound-scout`: argument parsing and the `doctor` report.

`doctor` loads the operator configuration, installs the network guard for the rest of the
process, prints `doctor.report` as a table or JSON and exits 0 whatever the statuses are: it is
a diagnosis, not a check. A configuration error is one line on stderr and exit code 1. The
other subcommands are not implemented yet and exit 2.
"""

import argparse
import json
import sys

from sound_scout import __version__, config, doctor, guard

SUBCOMMANDS = {
    "doctor": "report the state of the decoder, ASR, acoustic analysis and audio backend",
    "analyze": "analyze one recording and write a run directory under the data root",
    "compare": "compare two analyzed recordings",
    "serve": "run the MCP stdio server",
}
NOT_IMPLEMENTED_EXIT_CODE = 2
CONFIG_ERROR_EXIT_CODE = 1
TABLE_COLUMNS = ("component", "status", "last smoke", "fingerprint", "reason")
FINGERPRINT_DIGITS = 12
QUALIFICATION_FLAGS = ("validated_ru", "diction_validated_ru", "prosody_validated_ru")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound-scout",
        description="Sound analysis of Russian speech for the Hermes agent.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, help_text in SUBCOMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        if name == "doctor":
            subparser.add_argument(
                "--json", action="store_true", help="print the report as JSON instead of a table"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return run_doctor(as_json=args.json)
    print(f"sound-scout {args.command}: not implemented", file=sys.stderr)
    return NOT_IMPLEMENTED_EXIT_CODE


def run_doctor(*, as_json: bool) -> int:
    try:
        loaded = config.load()
    except config.ConfigError as error:
        print(f"config error: {' '.join(str(error).split())}", file=sys.stderr)
        return CONFIG_ERROR_EXIT_CODE
    guard.install_network_guard(loaded)
    report = doctor.report(loaded)
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(report))
    return 0


def format_report(report: dict) -> str:
    """The human table: one row per component, then the qualification flags and the guard."""
    rows = [TABLE_COLUMNS]
    for name, component in report["components"].items():
        rows.append(
            (
                name,
                component["status"],
                component["last_smoke_at"] or "-",
                component["fingerprint"]["hash"][:FINGERPRINT_DIGITS],
                component["reason"] or "-",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(TABLE_COLUMNS) - 1)]
    lines = []
    for row in rows:
        padded = [cell.ljust(width) for cell, width in zip(row[:-1], widths, strict=True)]
        lines.append(" | ".join([*padded, row[-1]]).rstrip())
    for flag in QUALIFICATION_FLAGS:
        lines.append(f"{flag}: {json.dumps(report[flag])}")
    network_guard = report["network_guard"]
    state = "active" if network_guard["active"] else "inactive"
    if not network_guard["active"] and network_guard["network_allow"]:
        state += " (network.allow = true)"
    lines.append(f"network_guard: {state}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
