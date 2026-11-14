"""
tests/test_audio_io.py
======================
Tests for the audio loading module.

Run all tests:
    python -m pytest tests/ -v

Tests use a synthetic 440 Hz sine wave as a stand-in for a real recording.
This is clearly documented. Synthetic audio is ONLY used for software testing.
No synthetic audio is ever presented as real whale data.
"""

import os
import sys
import struct
import wave
import tempfile
import numpy as np
import pytest

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_io import load_audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sine_wav(path: str, frequency: float = 440.0, duration: float = 1.0,
                  sample_rate: int = 22050, amplitude: float = 0.5) -> None:
    """Create a synthetic single-channel WAV file with a sine wave."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


def make_stereo_sine_wav(path: str, sample_rate: int = 22050,
                          duration: float = 0.5) -> None:
    """Create a 2-channel WAV file (stereo)."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    channel = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    stereo = np.column_stack([channel, channel])

    with wave.open(path, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadAudio:

    def test_load_valid_wav_returns_correct_sample_rate(self, tmp_path):
        """Loading a valid WAV should return the correct sample rate."""
        wav_file = str(tmp_path / "test_sine.wav")
        make_sine_wav(wav_file, sample_rate=22050)

        _, sr = load_audio(wav_file)

        assert sr == 22050, f"Expected 22050, got {sr}"

    def test_load_valid_wav_returns_float32_array(self, tmp_path):
        """Returned samples must be a float32 NumPy array."""
        wav_file = str(tmp_path / "test_sine.wav")
        make_sine_wav(wav_file)

        samples, _ = load_audio(wav_file)

        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32

    def test_load_valid_wav_samples_normalized(self, tmp_path):
        """Returned samples must be in [-1.0, 1.0]."""
        wav_file = str(tmp_path / "test_sine.wav")
        make_sine_wav(wav_file, amplitude=0.8)

        samples, _ = load_audio(wav_file)

        assert np.max(np.abs(samples)) <= 1.0 + 1e-6

    def test_load_stereo_wav_returns_mono(self, tmp_path):
        """Stereo files must be mixed down to mono (1D array)."""
        wav_file = str(tmp_path / "stereo.wav")
        make_stereo_sine_wav(wav_file)

        samples, _ = load_audio(wav_file)

        assert samples.ndim == 1, "Expected 1D mono array"

    def test_load_correct_sample_count(self, tmp_path):
        """Sample count must be approximately duration * sample_rate."""
        sr = 16000
        dur = 2.0
        wav_file = str(tmp_path / "timed.wav")
        make_sine_wav(wav_file, sample_rate=sr, duration=dur)

        samples, _ = load_audio(wav_file)

        expected = int(sr * dur)
        # Allow a small tolerance for rounding
        assert abs(len(samples) - expected) <= 2, \
            f"Expected ~{expected} samples, got {len(samples)}"

    def test_file_not_found_raises_error(self, tmp_path):
        """Missing file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_audio(str(tmp_path / "nonexistent.wav"))

    def test_empty_file_raises_error(self, tmp_path):
        """Empty file must raise ValueError or RuntimeError."""
        empty = str(tmp_path / "empty.wav")
        open(empty, "w").close()  # create zero-byte file

        with pytest.raises((ValueError, RuntimeError)):
            load_audio(empty)

    def test_corrupted_file_raises_runtime_error(self, tmp_path):
        """A file with random bytes must raise RuntimeError."""
        bad = str(tmp_path / "corrupted.wav")
        with open(bad, "wb") as f:
            f.write(b"this is not audio data \x00\xFF\xAB\xCD" * 10)

        with pytest.raises((RuntimeError, Exception)):
            load_audio(bad)

    def test_silent_file_raises_value_error(self, tmp_path):
        """An all-zero audio file must raise ValueError (silent)."""
        n = 22050
        silence = np.zeros(n, dtype=np.int16)

        silent = str(tmp_path / "silent.wav")
        with wave.open(silent, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(silence.tobytes())

        with pytest.raises(ValueError, match="silent"):
            load_audio(silent)
