import wave
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import WavFactory


def test_wav_file_is_readable_pcm_mono(wav_file: Path, tmp_path: Path) -> None:
    assert wav_file.parent == tmp_path
    with wave.open(str(wav_file), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
        assert wav.getnframes() == 16_000
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    assert samples.max() > 10_000
    assert samples.min() < -10_000


def test_make_wav_honours_parameters(make_wav: WavFactory) -> None:
    path = make_wav(name="short.wav", seconds=0.25, sample_rate=8_000)
    assert path.name == "short.wav"
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 8_000
        assert wav.getnframes() == 2_000


def test_slow_marker_is_registered(pytestconfig: pytest.Config) -> None:
    assert any(line.startswith("slow:") for line in pytestconfig.getini("markers"))
