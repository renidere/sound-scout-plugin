"""Spike A (#95, decision 0016): check on the station the WAV the probe sent through `file.attach`.

    uv run python -I scripts/probe_a_check_attach.py <path from the log> --sha256 <hex from the log>

Stdlib only. The file is only read; neither its path nor its content is printed. One JSON line goes
to stdout: `ok`, `bytes`, `sha256`, `matches_log` (the probe's digest), `matches_synthetic` (the WAV
this script generates with the probe's algorithm) and the parsed RIFF header `wav`. Exit 0 when
both match, 1 when not, 2 on a refusal (`{"ok": false, "error": <code>}`).
"""

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import struct
import sys

RATE = 48_000
SECONDS = 30
MAX_BYTES = 16 * 1024 * 1024
LEFT = (100, 8000)
RIGHT = (150, 6000)
PERIOD = 300  # LCM of both triangle periods: the signal repeats every 300 frames.
SHA256 = re.compile(r"[0-9a-f]{64}")


def _tri(n: int, period: int, amplitude: int) -> int:
    t = (n % period) * 4 * amplitude // period
    if t < amplitude:
        return t
    if t < 3 * amplitude:
        return 2 * amplitude - t
    return t - 4 * amplitude


def synthetic_wav() -> bytes:
    """The probe's WAV: 48 kHz, stereo, PCM16, 30 s of integer triangles."""
    period = b"".join(struct.pack("<hh", _tri(n, *LEFT), _tri(n, *RIGHT)) for n in range(PERIOD))
    data = period * (RATE * SECONDS // PERIOD)
    fmt = struct.pack("<IHHIIHH", 16, 1, 2, RATE, RATE * 4, 4, 16)
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + fmt
    return header + b"data" + struct.pack("<I", len(data)) + data


def parse_wav(data: bytes) -> dict | None:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    fmt = size = None
    offset = 12
    while offset + 8 <= len(data):
        chunk, length = data[offset : offset + 4], struct.unpack_from("<I", data, offset + 4)[0]
        body = offset + 8
        if chunk == b"fmt " and length >= 16 and body + 16 <= len(data):
            fmt = struct.unpack_from("<HHIIHH", data, body)
        elif chunk == b"data":
            size = min(length, len(data) - body)
        offset = body + length + (length & 1)
    if fmt is None or size is None:
        return None
    format_tag, channels, rate, _, block, bits = fmt
    frames = size // block if block else 0
    return {
        "format": format_tag,
        "channels": channels,
        "sample_rate": rate,
        "bits": bits,
        "frames": frames,
        "seconds": frames / rate if rate else None,
    }


class Refused(Exception):
    pass


def _refusal(error: OSError) -> Refused:
    if isinstance(error, FileNotFoundError):
        return Refused("not_found")
    if error.errno == errno.ELOOP:
        return Refused("symlink")
    return Refused("unreadable")


def read_regular(path: str) -> bytes:
    try:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise Refused("symlink")
        if not stat.S_ISREG(info.st_mode):
            raise Refused("not_regular")
        if info.st_size > MAX_BYTES:
            raise Refused("too_large")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise Refused("not_regular")
            data = handle.read(MAX_BYTES + 1)
    except OSError as error:
        raise _refusal(error) from None
    if len(data) > MAX_BYTES:
        raise Refused("too_large")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the WAV the Spike A probe attached.")
    parser.add_argument("path")
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        expected = args.sha256.strip().lower()
        if not SHA256.fullmatch(expected):
            raise Refused("bad_sha256")
        data = read_regular(args.path)
    except Refused as refusal:
        print(json.dumps({"ok": False, "error": str(refusal)}))
        return 2
    digest = hashlib.sha256(data).hexdigest()
    matches_log = digest == expected
    matches_synthetic = digest == hashlib.sha256(synthetic_wav()).hexdigest()
    ok = matches_log and matches_synthetic
    print(
        json.dumps(
            {
                "ok": ok,
                "bytes": len(data),
                "sha256": digest,
                "matches_log": matches_log,
                "matches_synthetic": matches_synthetic,
                "wav": parse_wav(data),
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
