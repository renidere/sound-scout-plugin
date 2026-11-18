"""Console entry point `sound-scout`: argument parsing and the `doctor` report.

`doctor` loads the operator configuration, installs the network guard for the rest of the
process, prints `doctor.report` as a table or JSON and exits 0 whatever the statuses are: it is
a diagnosis, not a check. A configuration error is one line on stderr and exit code 1.
`eval run` counts ready `analyze` results against the human labels of the corpus
(`sound_scout.eval.runner`), writes `<date>-report.md` and `.json` into `--output-dir` and exits
0 whatever the qualification flags are. An unreadable or invalid manifest or agreements file, or
a report that already exists, is one line on stderr and exit code 1; missing or invalid labels
and results are NOT QUALIFIED reasons in the report. `smoke [decoder] [acoustics]` loads the
same configuration, installs the network guard, runs the real components on a synthetic signal
(`sound_scout.smoke`, imported only here: it loads Praat and the speech detector) and saves the
smoke-receipts `doctor` reads; one line per component, exit 0 when every receipt was written,
1 otherwise. The other subcommands are not implemented yet and exit 2.
"""

import argparse
import json
import sys

from sound_scout import __version__, config, doctor, guard
from sound_scout.eval import runner

SUBCOMMANDS = {
    "doctor": "report the state of the decoder, ASR, acoustic analysis and audio backend",
    "analyze": "analyze one recording and write a run directory under the data root",
    "compare": "compare two analyzed recordings",
    "serve": "run the MCP stdio server",
    "eval": "evaluate ready analyze results against the corpus labels (Gate 1)",
    "smoke": "run the real decoder and acoustics on synthetic audio and save their "
    "smoke-receipts for doctor",
}
# The names of `smoke.COMPONENTS`, repeated so that building the parser does not import `smoke`.
SMOKE_COMPONENTS = ("decoder", "acoustics")
NOT_IMPLEMENTED_EXIT_CODE = 2
CONFIG_ERROR_EXIT_CODE = 1
EVAL_ERROR_EXIT_CODE = 1
SMOKE_FAILED_EXIT_CODE = 1
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
        if name == "eval":
            _add_eval_arguments(subparser)
        if name == "smoke":
            subparser.add_argument(
                "components",
                nargs="*",
                choices=SMOKE_COMPONENTS,
                help="components to smoke-test (default: all of them)",
            )
    return parser


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="eval_command", required=True, metavar="action")
    run = actions.add_parser(
        "run",
        help="match findings to labels by docs/eval/PROTOCOL.md and write the report",
        description="match findings to labels by docs/eval/PROTOCOL.md and write the report",
    )
    run.add_argument("--manifest", required=True, help="corpus manifest.json")
    run.add_argument("--labels", required=True, help="directory of <clip id>.json label files")
    run.add_argument(
        "--results", required=True, help="directory of <clip id>/result.json run directories"
    )
    run.add_argument("--split", required=True, choices=["dev", "heldout"])
    run.add_argument(
        "--agreements", help="a person's observation-agreement judgements of candidate pairs"
    )
    run.add_argument(
        "--output-dir",
        default=str(runner.DEFAULT_OUTPUT_DIR),
        help=f"where the report goes (default: {runner.DEFAULT_OUTPUT_DIR})",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return run_doctor(as_json=args.json)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "smoke":
        return run_smoke(tuple(dict.fromkeys(args.components)) or SMOKE_COMPONENTS)
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


def run_smoke(components: tuple[str, ...]) -> int:
    try:
        loaded = config.load()
    except config.ConfigError as error:
        print(f"config error: {' '.join(str(error).split())}", file=sys.stderr)
        return CONFIG_ERROR_EXIT_CODE
    guard.install_network_guard(loaded)
    from sound_scout import smoke

    outcomes = smoke.run(loaded, components)
    for name, outcome in outcomes.items():
        if isinstance(outcome, smoke.SmokeFailed):
            print(f"{name}: failed: {' '.join(outcome.reason.split())}")
        else:
            print(f"{name}: ok ({outcome.relative_to(loaded.data_root).as_posix()})")
    failed = any(isinstance(outcome, smoke.SmokeFailed) for outcome in outcomes.values())
    return SMOKE_FAILED_EXIT_CODE if failed else 0


def run_eval(args: argparse.Namespace) -> int:
    try:
        report = runner.evaluate(
            args.manifest,
            args.labels,
            args.results,
            args.split,
            agreements_path=args.agreements,
        )
        markdown_path, json_path = runner.write_report(report, args.output_dir)
    except runner.EvalError as error:
        print(f"eval error: {' '.join(str(error).split())}", file=sys.stderr)
        return EVAL_ERROR_EXIT_CODE
    for flag in QUALIFICATION_FLAGS:
        print(f"{flag}: {json.dumps(report[flag])}")
    print(f"report: {markdown_path}")
    print(f"report json: {json_path}")
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
