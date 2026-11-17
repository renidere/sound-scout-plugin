import shutil
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests import conftest, test_ingest
from tests.conftest import Encoder, WavFactory


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_encoders_lists_what_ffmpeg_reports() -> None:
    encoders = conftest.ffmpeg_encoders()
    assert encoders
    assert all(" " not in name for name in encoders)
    assert "=" not in encoders and "------" not in encoders


def test_a_missing_encoder_skips_the_test_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "ffmpeg_encoders", lambda: frozenset({"aac"}))
    conftest.require_encoders("aac")
    with pytest.raises(pytest.skip.Exception, match="encoder not available: libopus, flac"):
        conftest.require_encoders("aac", "libopus", "flac")


def test_encode_skips_by_name_when_the_encoder_is_missing(
    encode: Encoder, wav_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conftest, "ffmpeg_encoders", lambda: frozenset())
    with pytest.raises(pytest.skip.Exception, match="libmp3lame"):
        encode(wav_file, wav_file.with_suffix(".mp3"), "libmp3lame")
    assert not wav_file.with_suffix(".mp3").exists()


@pytest.mark.parametrize(
    "write_mp3",
    [
        lambda path: test_ingest.write_two_bitrate_mp3(path, head_seconds=0.5, tail_seconds=0.5),
        test_ingest.write_damaged_mp3,
    ],
    ids=["write_two_bitrate_mp3", "write_damaged_mp3"],
)
def test_mp3_helpers_skip_by_name_when_libmp3lame_is_missing(
    write_mp3: Callable[[Path], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conftest, "ffmpeg_encoders", lambda: frozenset())
    with pytest.raises(pytest.skip.Exception, match="encoder not available: libmp3lame"):
        write_mp3(tmp_path / "voice.mp3")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")
def test_encode_writes_the_requested_codec_next_to_the_wav(encode: Encoder, wav_file: Path) -> None:
    encoded = encode(wav_file, wav_file.with_suffix(".m4a"), "aac")
    assert encoded.parent == wav_file.parent
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", encoded],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert codec == "aac"
