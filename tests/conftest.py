import functools
import shutil
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

WavFactory = Callable[..., Path]
Encoder = Callable[..., Path]


@functools.cache
def ffmpeg_encoders() -> frozenset[str]:
    """The encoder names the installed ffmpeg lists; empty without ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return frozenset()
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    ).stdout
    _, _, entries = listing.partition(" ------\n")
    return frozenset(line.split()[1] for line in entries.splitlines() if line.strip())


def require_encoders(*names: str) -> None:
    """Skip the test, naming the encoder, when ffmpeg lacks one of `names`."""
    missing = [name for name in names if name not in ffmpeg_encoders()]
    if missing:
        pytest.skip(f"ffmpeg encoder not available: {', '.join(missing)}")


@pytest.fixture
def make_wav(tmp_path: Path) -> WavFactory:
    """Synthesize a 16-bit PCM mono WAV (a sine tone) in the temp dir and return its path."""

    def _make_wav(
        name: str = "tone.wav",
        seconds: float = 1.0,
        sample_rate: int = 16_000,
        frequency: float = 220.0,
    ) -> Path:
        t = np.arange(int(seconds * sample_rate)) / sample_rate
        samples = (0.5 * np.sin(2 * np.pi * frequency * t) * np.iinfo(np.int16).max).astype(
            np.int16
        )
        path = tmp_path / name
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            out.writeframes(samples.tobytes())
        return path

    return _make_wav


@pytest.fixture
def wav_file(make_wav: WavFactory) -> Path:
    return make_wav()


@pytest.fixture
def encode() -> Encoder:
    """Transcode a WAV with ffmpeg: `encode(source, target, "libopus", *more ffmpeg args)`.

    `args` go between the input and the audio encoder, so a second `-i`, `-map` or `-c:v` fit
    there. A missing encoder skips the test naming it; extra encoders named in `args` are the
    test's own `require_encoders` call.
    """

    def _encode(source: Path, target: Path, encoder: str, *args: str) -> Path:
        require_encoders(encoder)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), *args, "-c:a", encoder, str(target)],
            check=True,
            capture_output=True,
        )
        return target

    return _encode
