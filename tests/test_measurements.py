"""
tests/test_measurements.py
==========================
Tests for the measurements module.

Uses synthetic signals with known mathematical properties
so we can verify measurements against expected values.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measurements import compute_measurements, format_summary


class TestComputeMeasurements:

    def _sine(self, freq=440.0, sr=22050, duration=1.0, amplitude=0.5):
        """Create a normalized sine wave as test input."""
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        raw = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Normalize to [-1, 1] (as audio_io.load_audio would do)
        return (raw / np.max(np.abs(raw))).astype(np.float32), sr

    def test_duration_correct(self):
        """Duration should match sample_count / sample_rate."""
        sr = 16000
        duration = 2.5
        n = int(sr * duration)
        samples = np.ones(n, dtype=np.float32)
        m = compute_measurements(samples, sr)
        assert abs(m["duration_s"] - duration) < 0.01

    def test_sample_rate_stored(self):
        """Sample rate must be stored as-is."""
        samples = np.ones(1000, dtype=np.float32)
        m = compute_measurements(samples, 44100)
        assert m["sample_rate_hz"] == 44100

    def test_peak_amplitude_is_one_for_normalized_signal(self):
        """Peak of a normalized signal is 1.0."""
        samples, sr = self._sine()
        m = compute_measurements(samples, sr)
        assert abs(m["peak_amplitude"] - 1.0) < 1e-4

    def test_rms_amplitude_sine_wave(self):
        """RMS of a sine wave with amplitude A = A / sqrt(2)."""
        samples, sr = self._sine(amplitude=1.0)
        m = compute_measurements(samples, sr)
        expected_rms = 1.0 / np.sqrt(2)   # ~0.7071 for a full-scale sine
        assert abs(m["rms_amplitude"] - expected_rms) < 0.01, \
            f"Expected ~{expected_rms:.4f}, got {m['rms_amplitude']:.4f}"

    def test_rms_dbfs_is_negative(self):
        """RMS dBFS should be negative for a non-clipping signal."""
        samples, sr = self._sine()
        m = compute_measurements(samples, sr)
        assert m["rms_dbfs"] < 0.0, "RMS dBFS must be negative for sub-full-scale signal"

    def test_dominant_frequency_matches_sine(self):
        """Dominant frequency of a pure sine should match the sine's frequency."""
        freq = 1000.0
        samples, sr = self._sine(freq=freq, sr=22050, duration=2.0)
        m = compute_measurements(samples, sr)
        # Allow ±50 Hz tolerance due to FFT bin resolution
        assert abs(m["dominant_freq_hz"] - freq) < 50, \
            f"Expected ~{freq} Hz dominant, got {m['dominant_freq_hz']} Hz"

    def test_spectral_centroid_pure_tone(self):
        """Spectral centroid of a pure sine should be close to the sine frequency."""
        freq = 800.0
        samples, sr = self._sine(freq=freq, sr=22050, duration=2.0)
        m = compute_measurements(samples, sr)
        assert abs(m["spectral_centroid_hz"] - freq) < 100, \
            f"Centroid {m['spectral_centroid_hz']:.1f} Hz far from expected {freq} Hz"

    def test_n_samples_correct(self):
        """n_samples field must equal the actual sample count."""
        samples = np.ones(5000, dtype=np.float32)
        m = compute_measurements(samples, 22050)
        assert m["n_samples"] == 5000

    def test_channels_is_one(self):
        """channels must always be 1 (audio_io mixes down before this point)."""
        samples = np.ones(1000, dtype=np.float32)
        m = compute_measurements(samples, 22050)
        assert m["channels"] == 1

    def test_returns_all_required_keys(self):
        """All expected keys must be present in the result dict."""
        required = {
            "duration_s", "sample_rate_hz", "channels", "n_samples",
            "peak_amplitude", "rms_amplitude", "rms_dbfs",
            "spectral_centroid_hz", "dominant_freq_hz", "computed_at",
        }
        samples = np.ones(1000, dtype=np.float32)
        m = compute_measurements(samples, 22050)
        missing = required - set(m.keys())
        assert not missing, f"Missing measurement keys: {missing}"


class TestFormatSummary:

    def test_format_summary_returns_string(self):
        """format_summary must return a non-empty string."""
        samples = np.ones(22050, dtype=np.float32)
        m = compute_measurements(samples, 22050)
        result = format_summary(m, "dummy/test.wav")
        assert isinstance(result, str)
        assert len(result) > 100

    def test_format_summary_contains_species(self):
        """Summary must mention the species name."""
        samples = np.ones(22050, dtype=np.float32)
        m = compute_measurements(samples, 22050)
        result = format_summary(m, "test.wav")
        assert "Physeter macrocephalus" in result

    def test_format_summary_contains_duration(self):
        """Duration value must appear in the formatted summary."""
        samples = np.ones(22050, dtype=np.float32)
        m = compute_measurements(samples, 22050)
        result = format_summary(m, "test.wav")
        assert "1.00" in result  # 22050 samples / 22050 Hz = 1.00 s

