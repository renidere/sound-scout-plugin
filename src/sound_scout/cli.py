"""Console entry point `sound-scout`: argument parsing and printing over the application layer.

Every subcommand that needs the operator configuration loads it with `config.load()` (the file
in the current directory or in `SOUND_SCOUT_CONFIG`) and installs the network guard for the rest
of the process. No flag sets an allowed root, the data root, a model, the backend or a budget:
those are the operator's, never arguments (R12). A refused request is one line on stderr,
`sound-scout <command>: <reason code>: <message>`, with exit code 1 and no traceback; `--verbose`
adds the configuration file, the absolute run directory and the traceback on stderr; without it
an ingest refusal names the input, the allowed roots and the data root by placeholders, not by
their absolute paths (ffprobe and ffmpeg quote them). Stdout carries only the answer.

`analyze` calls `app.analyze` and prints `report.compact` of the result (`--json`, with
`run_dir` and `reused`) or a summary of it; the run directory is printed relative to the data
root. The exit code follows the result: 0 `complete`, 3 `partial`, 4 `blocked`, 1 `failed`.
`compare` calls `app.compare` and prints the verdict, its reasons and the file
`compare/<id>.json` (`--json`: the comparison with `path`), exit 0 whatever the verdict.
`jobs submit|status|result|cancel` calls `sound_scout.jobs` and prints its answer as
`key: value` lines (`--json`: the answer); `result` puts `job_id`, `job_state` and `attempt`
before the summary of the compact answer, as `analyze` prints it. The exit code is 0 whatever
the job state or the result status is, 1 on a refusal.
`doctor` prints `doctor.report` as a table or JSON and exits 0 whatever the statuses are: it is a
diagnosis, not a check. `serve` runs the MCP stdio server (`sound_scout.mcp_server`) over the
same functions until the client closes stdin, then exits 0.
`eval run` counts ready `analyze` results against the human labels of the corpus
(`sound_scout.eval.runner`), writes `<date>-report.md` and `.json` into `--output-dir` and exits 0
whatever the qualification flags are. An unreadable or invalid manifest or agreements file, or a
report that already exists, is a refusal with the reason code `eval_error`; missing or invalid
labels and results are NOT QUALIFIED reasons in the report. `eval check-manifest <manifest>
[--draft]` prints the whole `sound_scout.eval.manifest` report and exits 1 on any error; with
`--draft` the errors that wait for human labels or the owner's recordings stand in a block of
their own and only the others fail. A qualifying manifest ends with 'manifest qualifies' and
exits 0; for a manifest that does not qualify, the last `--draft` line is one of three verdicts:
the structure is sound (exit 0), the structure is sound but short of clips or controls (exit 1),
or the structure is not sound (exit 1). `eval import-labels <submission> --package <package>`
imports one returned working copy of a blind labeling package (`sound_scout.eval.label_import`):
on success it prints the counts and the new `<package>-returns/<NNN>` directory, exit 0; each
refusal is one `refused: <file>: <code>` line on stderr (a file may have several; validator errors
indented below it), nothing is written and the command ends with `eval_error`, exit 1.
`smoke [decoder] [acoustics] [asr]` runs the real components (`sound_scout.smoke`) — `decoder`
and `acoustics` on a synthetic signal, `asr` on the development recording named by
`SOUND_SCOUT_DEV_AUDIO`, only when asked for — and saves the smoke-receipts `doctor` reads; one
line per component, exit 0 when every receipt was written, 1 otherwise. `app`, `smoke` and
`mcp_server` are imported only by the subcommands that run them: they load Praat and the speech
detector.
"""

import argparse
import json
import sys
import traceback

from sound_scout import __version__, config, doctor, guard
from sound_scout.eval import label_import, manifest, runner

SUBCOMMANDS = {
    "doctor": "report the state of the decoder, ASR, acoustic analysis and audio backend",
    "analyze": "analyze one recording and write a run directory under the data root",
    "compare": "compare two analyzed recordings",
    "serve": "run the MCP stdio server",
    "eval": "evaluate ready analyze results against the corpus labels (Gate 1)",
    "smoke": "run the real decoder and acoustics on synthetic audio (and, when named, ASR on the "
    "recording in SOUND_SCOUT_DEV_AUDIO) and save their smoke-receipts for doctor",
    "jobs": "queue an analysis and follow it: submit, status, result, cancel",
}
# `smoke.COMPONENTS` and `smoke.DEFAULT_COMPONENTS`, repeated so that building the parser does
# not import `smoke`.
SMOKE_COMPONENTS = ("decoder", "acoustics", "asr")
SMOKE_DEFAULT_COMPONENTS = ("decoder", "acoustics")
# `app.LAYERS`, repeated so that building the parser does not import `app`.
ANALYZE_LAYERS = ("content", "measurements", "sound")
ANALYZE_EXIT_CODES = {"complete": 0, "partial": 3, "blocked": 4, "failed": 1}
ERROR_EXIT_CODE = 1
SMOKE_FAILED_EXIT_CODE = 1
CONFIG_ERROR = "config_error"
EVAL_ERROR = "eval_error"
INTERNAL_ERROR = "internal_error"
TABLE_COLUMNS = ("component", "status", "last smoke", "fingerprint", "reason")
FINGERPRINT_DIGITS = 12
QUALIFICATION_FLAGS = ("validated_ru", "diction_validated_ru", "prosody_validated_ru")


class CliError(Exception):
    """A request the application layer refused; `code` is the reason code printed on stderr."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound-scout",
        description="Sound analysis of Russian speech for the Hermes agent.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--verbose",
        action="store_true",
        help="print diagnostics to stderr: the configuration file, the run directory, tracebacks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, help_text in SUBCOMMANDS.items():
        parents = [] if name in ("eval", "jobs") else [common]
        subparser = subparsers.add_parser(
            name, help=help_text, description=help_text, parents=parents
        )
        if name == "doctor":
            subparser.add_argument(
                "--json", action="store_true", help="print the report as JSON instead of a table"
            )
        if name == "analyze":
            _add_analyze_arguments(subparser)
        if name == "compare":
            _add_compare_arguments(subparser)
        if name == "eval":
            _add_eval_arguments(subparser, common)
        if name == "jobs":
            _add_jobs_arguments(subparser, common)
        if name == "smoke":
            subparser.add_argument(
                "components",
                nargs="*",
                choices=SMOKE_COMPONENTS,
                help="components to smoke-test (default: decoder and acoustics; asr reads "
                "the development recording in SOUND_SCOUT_DEV_AUDIO)",
            )
    return parser


def _add_analyze_arguments(parser: argparse.ArgumentParser, *, with_json: bool = True) -> None:
    parser.add_argument("file", help="the recording, inside one of the configured allowed roots")
    parser.add_argument(
        "--request-id",
        help="the caller's request ID: a repeat with the same input returns the same result",
    )
    parser.add_argument(
        "--layers",
        type=_layer_list,
        help=f"comma-separated layers to analyze (default: {','.join(ANALYZE_LAYERS)})",
    )
    parser.add_argument("--task", help="the speaking task the recording answers")
    parser.add_argument(
        "--intent",
        help="the speaker's intent for the recording, as the user stated it; never text taken "
        "from the recording or its transcript",
    )
    if with_json:
        parser.add_argument(
            "--json",
            action="store_true",
            help="print the compact answer as JSON instead of a summary",
        )


def _add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_a", help="run_id of the initial attempt")
    parser.add_argument("run_b", help="run_id of the control attempt")
    parser.add_argument(
        "--criterion",
        required=True,
        help="emphasis_on:<word>, pause_before:<word>, intelligibility_of:<word> or "
        "completeness_of_phrase",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the comparison as JSON instead of a summary"
    )


def _add_jobs_arguments(parser: argparse.ArgumentParser, common: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="jobs_command", required=True, metavar="action")
    as_json = argparse.ArgumentParser(add_help=False)
    as_json.add_argument("--json", action="store_true", help="print the answer as JSON")
    submit_help = "queue one recording for analysis and print its job_id at once"
    submit = actions.add_parser(
        "submit", help=submit_help, description=submit_help, parents=[common, as_json]
    )
    _add_analyze_arguments(submit, with_json=False)
    submit.add_argument("--connection", help="the connection the job came through, stored as is")
    submit.add_argument("--profile", help="the agent profile of the chat, stored as is")
    submit.add_argument("--session-id", help="the stored session ID of the chat, stored as is")
    submit.add_argument(
        "--submitter", default="cli", help="who submitted the job, stored as is (default: cli)"
    )
    for action, help_text in (
        ("status", "print the state of a job"),
        ("result", "print the compact answer of the run of a job"),
        ("cancel", "cancel a job: a queued job at once, a running one by its cancel marker"),
    ):
        command = actions.add_parser(
            action, help=help_text, description=help_text, parents=[common, as_json]
        )
        command.add_argument("job_id", help="the job_id that submit printed")


def _layer_list(text: str) -> tuple[str, ...]:
    """The items as written; `app.analyze` is the one validator of the list."""
    return tuple(item.strip() for item in text.split(","))


def _add_eval_arguments(parser: argparse.ArgumentParser, common: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="eval_command", required=True, metavar="action")
    run = actions.add_parser(
        "run",
        help="match findings to labels by docs/eval/PROTOCOL.md and write the report",
        description="match findings to labels by docs/eval/PROTOCOL.md and write the report",
        parents=[common],
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
    check = actions.add_parser(
        "check-manifest",
        help="validate a corpus manifest by docs/eval/PROTOCOL.md § 3 and print the whole report",
        description="validate a corpus manifest by docs/eval/PROTOCOL.md § 3 and print the "
        "whole report",
        parents=[common],
    )
    check.add_argument("manifest", help="corpus manifest.json")
    check.add_argument(
        "--draft",
        action="store_true",
        help="exit 0 when only checks that need human labels or the owner's recordings remain; "
        "for a manifest that does not qualify, the last line says: structure sound, structure "
        "sound but short of clips or controls (exit 1), or structure not sound (exit 1)",
    )
    imports = actions.add_parser(
        "import-labels",
        help="import one returned working copy of a blind labeling package into "
        "<package>-returns/<NNN>/ next to the package",
        description="import one returned working copy of a blind labeling package into "
        "<package>-returns/<NNN>/ next to the package; any refused file writes nothing",
        parents=[common],
    )
    imports.add_argument("submission", help="the returned folder holding labels/ and review.csv")
    imports.add_argument(
        "--package",
        required=True,
        help="the blind labeling package holding operator-private/alias-map.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "doctor": run_doctor,
        "analyze": run_analyze,
        "compare": run_compare,
        "serve": run_serve,
        "eval": run_eval,
        "smoke": run_smoke,
        "jobs": run_jobs,
    }
    try:
        return commands[args.command](args)
    except config.ConfigError as error:
        return _refused(args, CONFIG_ERROR, str(error), error)
    except CliError as error:
        return _refused(args, error.code, error.message, error)
    except Exception as error:
        # Only the type: the message of an unexpected error may carry a path or a transcript.
        return _refused(args, INTERNAL_ERROR, type(error).__name__, error)


def _refused(args: argparse.Namespace, code: str, message: str, error: Exception) -> int:
    print(f"sound-scout {args.command}: {code}: {' '.join(message.split())}", file=sys.stderr)
    if args.verbose:
        traceback.print_exception(error, file=sys.stderr)
    return ERROR_EXIT_CODE


def _load_config(args: argparse.Namespace) -> config.Config:
    loaded = config.load()
    if args.verbose:
        print(f"configuration: {config.config_path().absolute()}", file=sys.stderr)
    guard.install_network_guard(loaded)
    return loaded


def run_doctor(args: argparse.Namespace) -> int:
    loaded = _load_config(args)
    report = doctor.report(loaded)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(report))
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    loaded = _load_config(args)
    from sound_scout import app, ingest, paths, report

    fields = {"request_id": args.request_id, "task": args.task, "intent": args.intent}
    if args.layers is not None:
        fields["layers"] = args.layers
    try:
        analysis = app.analyze(app.AnalyzeRequest(args.file, **fields), loaded)
    except ingest.IngestError as error:
        message = error.message
        if not args.verbose:
            message = paths.without_paths(message, args.file, loaded)
        raise CliError(error.code, message) from error
    except tuple(kind for kind, _ in app.REFUSALS) as error:
        code = next(code for kind, code in app.REFUSALS if isinstance(error, kind))
        raise CliError(code, str(error)) from error
    if args.verbose:
        print(f"run directory: {analysis.run.path}", file=sys.stderr)
    run_dir = analysis.run.path.relative_to(loaded.data_root).as_posix()
    answer = report.compact(analysis.result, run_dir=run_dir, reused=analysis.reused)
    if args.json:
        print(json.dumps(answer, indent=2, ensure_ascii=False))
    else:
        print(format_analysis(answer, run_dir, reused=analysis.reused))
    return ANALYZE_EXIT_CODES[analysis.result["status"]]


def run_compare(args: argparse.Namespace) -> int:
    loaded = _load_config(args)
    from sound_scout import app, comparison

    try:
        compared = app.compare(args.run_a, args.run_b, args.criterion, loaded)
    except comparison.CompareError as error:
        raise CliError(error.code, error.message) from error
    path = compared.path.relative_to(loaded.data_root).as_posix()
    if args.json:
        print(json.dumps({"path": path, **compared.result}, indent=2, ensure_ascii=False))
    else:
        print(format_comparison(compared.result, path))
    return 0


def run_jobs(args: argparse.Namespace) -> int:
    loaded = _load_config(args)
    from sound_scout import ingest, jobs, paths

    try:
        if args.jobs_command == "submit":
            fields = {"request_id": args.request_id, "task": args.task, "intent": args.intent}
            if args.layers is not None:
                fields["layers"] = args.layers
            binding = jobs.Binding(
                submitter=args.submitter,
                connection=args.connection,
                profile=args.profile,
                session_id=args.session_id,
            )
            answer = jobs.submit(jobs.SubmitRequest(args.file, **fields), binding, loaded)
        else:
            call = {"status": jobs.status, "result": jobs.result, "cancel": jobs.cancel}
            answer = call[args.jobs_command](args.job_id, loaded)
    except jobs.JobError as error:
        raise CliError(error.code, error.message) from error
    except ingest.IngestError as error:
        message = error.message
        if not args.verbose:
            message = paths.without_paths(message, args.file, loaded)
        raise CliError(error.code, message) from error
    except tuple(kind for kind, _ in jobs.REFUSALS) as error:
        code = next(code for kind, code in jobs.REFUSALS if isinstance(error, kind))
        raise CliError(code, str(error)) from error
    if args.json:
        print(json.dumps(answer, indent=2, ensure_ascii=False))
    elif args.jobs_command == "result":
        print(f"job_id: {answer['job_id']}\njob_state: {answer['job_state']}")
        print(f"attempt: {_plain(answer['attempt'])}")
        print(format_analysis(answer, answer["run_dir"], reused=False))
    else:
        print("\n".join(f"{key}: {_plain(value)}" for key, value in answer.items()))
    return 0


def run_serve(args: argparse.Namespace) -> int:
    loaded = _load_config(args)
    from sound_scout import mcp_server

    mcp_server.serve(loaded)
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    components = tuple(dict.fromkeys(args.components)) or SMOKE_DEFAULT_COMPONENTS
    loaded = _load_config(args)
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
    if args.eval_command == "check-manifest":
        checked = manifest.validate(args.manifest)
        print(manifest.format_report(checked, draft=args.draft))
        passed = checked.draft_ok if args.draft else checked.ok
        return 0 if passed else ERROR_EXIT_CODE
    if args.eval_command == "import-labels":
        return run_import_labels(args)
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
        raise CliError(EVAL_ERROR, str(error)) from error
    for flag in QUALIFICATION_FLAGS:
        print(f"{flag}: {json.dumps(report[flag])}")
    print(f"report: {markdown_path}")
    print(f"report json: {json_path}")
    return 0


def run_import_labels(args: argparse.Namespace) -> int:
    try:
        summary = label_import.import_submission(args.submission, args.package)
    except label_import.ImportRefused as error:
        for refusal in error.refusals:
            print(f"refused: {refusal.file}: {refusal.code}", file=sys.stderr)
            for detail in refusal.details:
                print(f"  {_one_line(detail)}", file=sys.stderr)
        raise CliError(EVAL_ERROR, str(error)) from error
    except label_import.LabelImportError as error:
        raise CliError(EVAL_ERROR, str(error)) from error
    print(f"imported: {summary.imported}")
    print(f"unsigned: {summary.unsigned}")
    print(f"already imported: {summary.already_imported}")
    print(f"output: {summary.output}")
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


def format_analysis(answer: dict, run_dir: str, *, reused: bool) -> str:
    """The human summary of the compact answer, with links into the run directory."""
    lines = [
        f"status: {answer['status']}",
        f"run_id: {answer['run_id']}",
        f"run_dir: {run_dir}"
        + (" (reused: the result of the same earlier request)" if reused else ""),
    ]
    for name, stage in answer["stages"].items():
        lines.append(_state_line(f"stage {name}", stage["status"], stage["reason"]))
    for name, assessment in answer["assessment_status"].items():
        lines.append(_state_line(f"assessment {name}", assessment["status"], assessment["reason"]))
    coverage = answer["coverage"]
    lines.append(f"coverage: reviewed {coverage['reviewed_s']} s of {coverage['requested_s']} s")
    for finding in answer["findings"]:
        lines.append(
            f"finding {finding['id']} ({finding['category']}, {finding['polarity']}, "
            f"{finding['evidence_basis']}) {_seconds(finding['start'])}-"
            f"{_seconds(finding['end'])} s: {_one_line(finding['observation'])}"
        )
    lines.append(
        f"findings: {len(answer['findings'])} shown of {answer['findings_total']}, "
        f"unlocalized: {answer['unlocalized_total']}"
    )
    lines += [f"limitation: {_one_line(text)}" for text in answer["limitations"]]
    lines += [f"error {error['stage'] or '-'}: {error['code']}" for error in answer["errors"]]
    lines.append(f"report: {run_dir}/{answer['links']['report']}")
    lines.append(f"result: {run_dir}/{answer['links']['result']}")
    return "\n".join(lines)


def format_comparison(result: dict, path: str) -> str:
    """The verdict, every reason it is not conclusive, the limitations and the file."""
    lines = [f"verdict: {result['verdict']}", f"criterion: {result['criterion']['raw']}"]
    for label, attempt in result["attempts"].items():
        lines.append(
            _state_line(
                f"attempt {label}: run {attempt['run_id']}, audio review",
                attempt["audio_review"]["status"],
                attempt["audio_review"]["reason"],
            )
        )
    for reason in result["reasons"]:
        attempt = f" ({reason['attempt']})" if reason["attempt"] else ""
        lines.append(f"reason {reason['code']}{attempt}: {_one_line(reason['message'])}")
    lines += [f"limitation: {_one_line(text)}" for text in result["limitations"]]
    lines.append(f"comparison: {path}")
    return "\n".join(lines)


def _state_line(label: str, status: str, reason: str | None) -> str:
    return f"{label}: {status}" + (f" - {_one_line(reason)}" if reason else "")


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _plain(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _one_line(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    sys.exit(main())
