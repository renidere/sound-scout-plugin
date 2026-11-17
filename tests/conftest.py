import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

WavFactory = Callable[..., Path]


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
