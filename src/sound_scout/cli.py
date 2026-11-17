"""Console entry point `sound-scout`: argument parsing only, no business logic yet."""

import argparse
import sys

from sound_scout import __version__

SUBCOMMANDS = {
    "doctor": "report the state of the decoder, ASR, acoustic analysis and audio backend",
    "analyze": "analyze one recording and write a run directory under the data root",
    "compare": "compare two analyzed recordings",
    "serve": "run the MCP stdio server",
}
NOT_IMPLEMENTED_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound-scout",
        description="Sound analysis of Russian speech for the Hermes agent.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, help_text in SUBCOMMANDS.items():
        subparsers.add_parser(name, help=help_text, description=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"sound-scout {args.command}: not implemented", file=sys.stderr)
    return NOT_IMPLEMENTED_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
