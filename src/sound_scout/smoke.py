"""Smoke runs of `decoder` and `acoustics`: the real component on a synthetic signal (R01, R03).

`doctor` reports a component `ready` only when a smoke-receipt matches its current fingerprint,
and a receipt proves something only when the component itself ran: a mock or a test double
proves the contract, never the capability. `decoder` writes a 1 s mono WAV PCM s16 at 48 kHz
and decodes it with the real `ffmpeg`/`ffprobe` through `ingest.decode_generated` under the
operator's subprocess budgets (the input policy for user recordings does not apply to a file
generated under the data root, whose ancestors may carry a denied name like `.hermes`);
`acoustics` passes a Praat KlattGrid formant control at a known F0 and digital silence through
the real `acoustics.measure` (Silero VAD and Praat). A pure tone is not speech for the detector,
so the control is formant synthesis, as in the #18 tests (`docs/decisions/0003`). Only after
every check passes is `doctor.write_receipt` called, with the operator's own configuration, so
the fingerprint is the one `doctor` computes; a failed check raises `SmokeFailed` and leaves an
existing receipt untouched.

Nothing is downloaded and no recording is read. Everything this module writes goes into a
temporary `<data_root>/.smoke-*` directory, removed whatever happens, and into
`<data_root>/receipts/`; onnxruntime, imported for the speech detector, has its telemetry
switched off before the import (`guard.disable_telemetry`), so it writes nothing either.
The network guard is the caller's: the `smoke` subcommand installs it like `doctor` does.
"""

import contextlib
import os
import shutil
import tempfile
import time
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from parselmouth.praat import call

from sound_scout import channels, doctor, ingest, paths, runs
from sound_scout.acoustics import VAD_END_LAG_WINDOWS, VAD_WINDOW_MS, measure
from sound_scout.config import Config

COMPONENTS = ("decoder", "acoustics")
SMOKE_DIR_PREFIX = ".smoke-"
DIR_MODE = 0o700

DECODER_RATE = 48_000
DECODER_SECONDS = 1.0
DECODER_FREQUENCY_HZ = 440.0
DECODER_AMPLITUDE = 0.5
DECODER_MAX_ABS_ERROR = 1e-6
DECODER_CONTAINER = "wav"
DECODER_CODEC = "pcm_s16le"
INPUT_NAME = "input.wav"

ACOUSTICS_RATE = 48_000
CONTROL_F0_HZ = 150.0
CONTROL_START_S = 1.0
CONTROL_SECONDS = 1.2
CONTROL_TOTAL_S = 4.0
SILENCE_SECONDS = 2.0
F0_TOLERANCE_REL = 0.03
SPAN_START_TOLERANCE_S = 0.05
SPAN_END_TOLERANCE_S = VAD_END_LAG_WINDOWS * VAD_WINDOW_MS / 1000
MIN_VOICED_FRAMES = 50

VOWEL_FORMANTS = (
    (300, 2300, 3000, 3400),
    (300, 800, 2300, 2800),
    (500, 1800, 2500, 3000),
    (800, 1200, 2300, 2800),
)
BANDWIDTHS = (50, 60, 100, 120)


class SmokeFailed(Exception):
    """The smoke run of `component` did not pass; `reason` says what was measured and expected."""

    def __init__(self, component: str, reason: str) -> None:
        super().__init__(f"{component} smoke failed: {reason}")
        self.component = component
        self.reason = reason


def speech_like_control(
    seconds: float, rate: int, f0: float = 130.0, syllable_s: float = 0.1
) -> np.ndarray:
    """Formant-synthesised alternating vowels at pitch `f0`, faded 20 ms at both ends."""
    grid = call("Create KlattGrid", "kg", 0, seconds, 6, 1, 1, 6, 1, 1, 1)
    call(grid, "Add pitch point", 0, f0)
    call(grid, "Add voicing amplitude point", 0, 90)
    onset, index = 0.0, 0
    while onset < seconds - 1e-9:
        formants = VOWEL_FORMANTS[index % len(VOWEL_FORMANTS)]
        for number, (frequency, bandwidth) in enumerate(zip(formants, BANDWIDTHS, strict=True), 1):
            call(grid, "Add oral formant frequency point", number, onset, frequency)
            call(grid, "Add oral formant bandwidth point", number, onset, bandwidth)
        onset, index = onset + syllable_s, index + 1
    sound = call(call(grid, "To Sound"), "Resample", rate, 50)
    samples = sound.values[0][: int(seconds * rate)]
    fade = int(0.02 * rate)
    envelope = np.ones(len(samples))
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (0.5 * samples * envelope / np.max(np.abs(samples))).astype(np.float32)


def place(signal: np.ndarray, start_s: float, total_s: float, rate: int) -> np.ndarray:
    """`signal` placed at `start_s` inside `total_s` of digital silence."""
    audio = np.zeros(int(total_s * rate), dtype=np.float32)
    begin = int(start_s * rate)
    audio[begin : begin + len(signal)] = signal
    return audio


def decoder(config: Config) -> Path:
    """Decode a synthetic WAV with the real ffmpeg, check it and save the `decoder` receipt."""
    started = time.monotonic()
    frames = round(DECODER_SECONDS * DECODER_RATE)
    t = np.arange(frames) / DECODER_RATE
    samples = np.round(
        DECODER_AMPLITUDE * 32767 * np.sin(2 * np.pi * DECODER_FREQUENCY_HZ * t)
    ).astype(np.int16)
    with _smoke_dir(config) as work:
        source_path = work / INPUT_NAME
        _write_wav(source_path, samples, DECODER_RATE)
        run_dir = work / "run"
        os.mkdir(run_dir, DIR_MODE)
        run = runs.Run(
            run_id=f"smoke-{uuid.uuid4().hex}",
            path=run_dir,
            request_id="smoke",
            identity_hash="",
            reused=False,
            complete=False,
            created=True,
        )
        try:
            source = ingest.decode_generated(source_path, run, config)
        except ingest.IngestError as error:
            raise SmokeFailed("decoder", str(error)) from None
        except paths.PathDenied as error:
            raise SmokeFailed("decoder", f"path refused: {error.reason}") from None
        try:
            audio = channels.load(source)
        except ValueError as error:
            raise SmokeFailed("decoder", str(error)) from None
    probed = source.probe
    measured = {
        "container": probed.container,
        "codec": probed.codec,
        "sample_rate": probed.sample_rate,
        "channels": probed.channels,
        "frames": source.frames,
        "duration_s": source.duration_s,
    }
    expected = {
        "container": DECODER_CONTAINER,
        "codec": DECODER_CODEC,
        "sample_rate": DECODER_RATE,
        "channels": 1,
        "frames": frames,
    }
    for name, value in expected.items():
        if measured[name] != value:
            raise SmokeFailed("decoder", f"{name} is {measured[name]!r}, expected {value!r}")
    if abs(source.duration_s - DECODER_SECONDS) >= 1e-9:
        raise SmokeFailed(
            "decoder", f"duration is {source.duration_s} s, expected {DECODER_SECONDS} s"
        )
    if audio.shape != (frames, 1):
        raise SmokeFailed(
            "decoder",
            f"decoded array frames is {len(audio)} (shape {audio.shape}), "
            f"expected shape {(frames, 1)}",
        )
    max_abs_error = float(np.max(np.abs(audio[:, 0].astype(np.float64) - samples / 32768)))
    measured["max_abs_error"] = max_abs_error
    if not max_abs_error <= DECODER_MAX_ABS_ERROR:
        raise SmokeFailed(
            "decoder",
            f"decoded samples differ from the written ones by {max_abs_error}, "
            f"the limit is {DECODER_MAX_ABS_ERROR}",
        )
    smoke = {
        "test": "sound_scout.smoke.decoder",
        "input": "synthetic",
        "signal": {
            "format": "wav_pcm_s16le",
            "sample_rate": DECODER_RATE,
            "channels": 1,
            "seconds": DECODER_SECONDS,
            "frequency_hz": DECODER_FREQUENCY_HZ,
            "amplitude": DECODER_AMPLITUDE,
        },
        "measured": measured,
        "elapsed_ms": _elapsed_ms(started),
    }
    return _save("decoder", config, smoke)


def acoustics(config: Config) -> Path:
    """Measure the F0 control and digital silence for real and save the `acoustics` receipt."""
    started = time.monotonic()
    state = doctor.component_status("acoustics", config, config.data_root)
    if state["status"] == "unavailable":
        raise SmokeFailed("acoustics", str(state["reason"]))
    control = place(
        speech_like_control(CONTROL_SECONDS, ACOUSTICS_RATE, f0=CONTROL_F0_HZ),
        CONTROL_START_S,
        CONTROL_TOTAL_S,
        ACOUSTICS_RATE,
    )
    span, f0, voiced = _check_control(_measured(control))
    silence = _measured(np.zeros(round(SILENCE_SECONDS * ACOUSTICS_RATE), dtype=np.float32))
    silence_spans = silence["speech_spans"]["value"]
    if silence_spans != []:
        raise SmokeFailed("acoustics", f"digital silence gave speech spans {silence_spans}")
    if silence["f0_hz"]["value"] is not None:
        raise SmokeFailed("acoustics", f"digital silence gave F0 {silence['f0_hz']['value']}")
    smoke = {
        "test": "sound_scout.smoke.acoustics",
        "input": "synthetic",
        "signal": {
            "control": "praat_klattgrid_formant_synthesis",
            "sample_rate": ACOUSTICS_RATE,
            "f0_hz": CONTROL_F0_HZ,
            "control_start_s": CONTROL_START_S,
            "control_seconds": CONTROL_SECONDS,
            "total_s": CONTROL_TOTAL_S,
            "silence_s": SILENCE_SECONDS,
        },
        "tolerance": {
            "f0_rel": F0_TOLERANCE_REL,
            "span_start_s": SPAN_START_TOLERANCE_S,
            "span_end_s": SPAN_END_TOLERANCE_S,
        },
        "measured": {
            "span": {"start": span["start"], "end": span["end"]},
            "f0_hz": {key: f0[key] for key in ("p10", "p50", "p90")},
            "voiced_speech_frames": voiced,
            "silence_spans": 0,
        },
        "elapsed_ms": _elapsed_ms(started),
    }
    return _save("acoustics", config, smoke)


def run(config: Config, components: tuple[str, ...] = COMPONENTS) -> dict[str, Path | SmokeFailed]:
    """Run the named smoke tests in order; a failure is recorded and the next one still runs."""
    unknown = [name for name in components if name not in COMPONENTS]
    if unknown:
        raise ValueError(f"unknown smoke components {unknown}; known: {list(COMPONENTS)}")
    smoke_tests = {"decoder": decoder, "acoustics": acoustics}
    outcomes: dict[str, Path | SmokeFailed] = {}
    for name in components:
        try:
            outcomes[name] = smoke_tests[name](config)
        except SmokeFailed as error:
            outcomes[name] = error
    return outcomes


def _measured(signal: np.ndarray) -> dict[str, dict]:
    result = measure(signal, ACOUSTICS_RATE)
    if result["status"] != "completed":
        raise SmokeFailed("acoustics", f"stage is {result['status']}: {result['reason']}")
    return {item["name"]: item for item in result["measurements"]}


def _check_control(values: dict[str, dict]) -> tuple[dict, dict, int]:
    spans = values["speech_spans"]["value"]
    if not isinstance(spans, list) or len(spans) != 1:
        raise SmokeFailed("acoustics", f"the F0 control gave spans {spans}, expected one")
    span = spans[0]
    start, end = CONTROL_START_S, CONTROL_START_S + CONTROL_SECONDS
    if abs(span["start"] - start) > SPAN_START_TOLERANCE_S:
        raise SmokeFailed(
            "acoustics",
            f"span start {span['start']} s, expected {start} s ± {SPAN_START_TOLERANCE_S} s",
        )
    if abs(span["end"] - end) > SPAN_END_TOLERANCE_S:
        raise SmokeFailed(
            "acoustics", f"span end {span['end']} s, expected {end} s ± {SPAN_END_TOLERANCE_S} s"
        )
    f0 = values["f0_hz"]["value"]
    if f0 is None:
        raise SmokeFailed("acoustics", f"f0 is null: {values['f0_hz']['reason']}")
    limit = F0_TOLERANCE_REL * CONTROL_F0_HZ
    if abs(f0["p50"] - CONTROL_F0_HZ) > limit:
        raise SmokeFailed(
            "acoustics",
            f"f0 p50 {f0['p50']} Hz is outside {CONTROL_F0_HZ} Hz ± {F0_TOLERANCE_REL:.0%}",
        )
    if not f0["p10"] <= f0["p50"] <= f0["p90"]:
        raise SmokeFailed("acoustics", f"f0 percentiles are out of order: {f0}")
    voiced = values["f0_hz"]["settings"]["voiced_speech_frames"]
    if not isinstance(voiced, int) or voiced <= MIN_VOICED_FRAMES:
        raise SmokeFailed(
            "acoustics", f"{voiced} voiced frames in the control, expected over {MIN_VOICED_FRAMES}"
        )
    return span, f0, voiced


def _save(component: str, config: Config, smoke: dict) -> Path:
    try:
        return doctor.write_receipt(component, config, smoke)
    except doctor.ReceiptRejected as error:
        raise SmokeFailed(component, f"receipt refused: {error}") from None
    except paths.PathDenied as error:
        raise SmokeFailed(component, f"receipt path refused: {error.reason}") from None
    except OSError as error:
        raise SmokeFailed(component, f"cannot write the receipt: {error.strerror}") from None


@contextlib.contextmanager
def _smoke_dir(config: Config) -> Iterator[Path]:
    if not config.data_root.is_dir():
        raise SmokeFailed("decoder", "the data root is not an existing directory")
    try:
        work = Path(tempfile.mkdtemp(prefix=SMOKE_DIR_PREFIX, dir=config.data_root))
    except OSError as error:
        raise SmokeFailed(
            "decoder", f"cannot create a work directory under the data root: {error.strerror}"
        ) from None
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.astype("<i2").tobytes())


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
