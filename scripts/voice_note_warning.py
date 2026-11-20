"""Insert, find or strip the direction warning in a text of renidere/hermes-voice-note (#117).

    python3 -I scripts/voice_note_warning.py insert readme|release <in> <out>
    python3 -I scripts/voice_note_warning.py has <in>
    python3 -I scripts/voice_note_warning.py strip <in> <out>

Stdlib only. The blocks come from the fenced ``markdown readme-warning`` and
``markdown release-warning`` blocks of docs/hermes-voice-note-warning.md. An `<in>` that ends in
`.json` is the output of `gh release view --json body`: its `body` is the text. `insert` puts the
block and a blank line before the text; a text that already has the block is written unchanged.
`has` exits 0 when the text has the whole block (both markers), 1 when not. `strip` removes the
first block between the markers and the blank line after it; a text without a block is written
unchanged, so `strip` of a backup JSON gives its body. `<out>` may be `<in>`. Line endings are
kept. Exit 2 on a draft without the block, and on a text with a marker but no whole block: neither
`insert` nor `strip` then writes anything.
"""

import argparse
import json
import re
import sys
from pathlib import Path

DRAFT = Path(__file__).resolve().parent.parent / "docs" / "hermes-voice-note-warning.md"
START = "<!-- sound-scout-direction-warning:start -->"
END = "<!-- sound-scout-direction-warning:end -->"
KINDS = ("readme", "release")
_BLOCK = re.compile(re.escape(START) + r".*?" + re.escape(END) + r"(?:\r?\n){0,2}", re.S)


def block(draft: str, kind: str) -> str:
    match = re.search(rf"^```markdown {kind}-warning\n(.*?)\n```$", draft, re.S | re.M)
    if match is None:
        raise ValueError(f"no {kind}-warning block in the draft")
    text = match.group(1)
    if not (text.startswith(START) and text.endswith(END)):
        raise ValueError(f"the {kind}-warning block is not wrapped in the markers")
    return text


def has(text: str) -> bool:
    return _BLOCK.search(text) is not None


def _check_markers(text: str) -> None:
    if (START in text or END in text) and not has(text):
        raise ValueError("the text has a warning marker but no whole block; fix it by hand")


def insert(text: str, warning: str) -> str:
    _check_markers(text)
    return text if has(text) else f"{warning}\n\n{text}"


def strip(text: str) -> str:
    _check_markers(text)
    return _BLOCK.sub("", text, count=1)


def _read(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as f:
        raw = f.read()
    return json.loads(raw)["body"] if path.suffix == ".json" else raw


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_insert = sub.add_parser("insert")
    p_insert.add_argument("kind", choices=KINDS)
    p_insert.add_argument("src", type=Path)
    p_insert.add_argument("dst", type=Path)
    p_has = sub.add_parser("has")
    p_has.add_argument("src", type=Path)
    p_strip = sub.add_parser("strip")
    p_strip.add_argument("src", type=Path)
    p_strip.add_argument("dst", type=Path)
    args = parser.parse_args(argv)

    text = _read(args.src)
    if args.command == "has":
        return 0 if has(text) else 1
    try:
        if args.command == "strip":
            _write(args.dst, strip(text))
            return 0
        published = insert(text, block(DRAFT.read_text(encoding="utf-8"), args.kind))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if has(text):
        print("already present, the text is unchanged", file=sys.stderr)
    _write(args.dst, published)
    return 0


if __name__ == "__main__":
    sys.exit(main())
