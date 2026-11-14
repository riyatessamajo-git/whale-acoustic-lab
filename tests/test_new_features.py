"""
tests/test_new_features.py
--------------------------
Tests for event detection, coda sequence analysis, and comparative measurements.
"""

import numpy as np
import pytest
from measurements import (
    compute_measurements,
    detect_vocalization_events,
    analyze_coda_sequences,
    compare_measurements
)


def _generate_synthetic_clicks(sr=44100, duration=2.0, click_times=(0.2, 0.4, 0.6, 0.8)):
    """Generate synthetic impulse click train."""
    n = int(sr * duration)
    samples = np.zeros(n, dtype=np.float32)
    # Add clicks as short bursts
    for ct in click_times:
        idx = int(ct * sr)
        click_len = int(0.002 * sr)  # 2ms
        samples[idx:idx + click_len] = np.hanning(click_len).astype(np.float32)
    return samples, sr


def test_detect_vocalization_events():
    samples, sr = _generate_synthetic_clicks()
    events = detect_vocalization_events(samples, sr, min_duration_s=0.01)
    assert isinstance(events, list)
    assert len(events) >= 1
    ev = events[0]
    assert "start_s" in ev
    assert "end_s" in ev
    assert "duration_ms" in ev
    assert "peak_amplitude" in ev
    assert ev["peak_amplitude"] > 0


def test_analyze_coda_sequences():
    click_times = [0.1, 0.2, 0.3, 0.4, 0.5]
    samples, sr = _generate_synthetic_clicks(duration=1.0, click_times=click_times)
    res = analyze_coda_sequences(samples, sr, max_gap_s=0.25, min_clicks=3)
    assert isinstance(res, dict)
    assert "total_clicks" in res
    assert "total_sequences" in res
    assert "mean_ici_ms" in res
    assert "candidate_sequences" in res
    assert "scientific_disclaimer" in res
    assert res["total_clicks"] >= 3
    assert len(res["candidate_sequences"]) >= 1


def test_compare_measurements():
    m_a = {
        "duration_s": 10.0,
        "rms_dbfs": -25.0,
        "spectral_centroid_hz": 8000,
        "dominant_freq_hz": 6000
    }
    m_b = {
        "duration_s": 15.0,
        "rms_dbfs": -20.0,
        "spectral_centroid_hz": 9500,
        "dominant_freq_hz": 7000
    }
    diff = compare_measurements(m_a, m_b)
    assert diff["delta_duration_s"] == 5.0
    assert diff["delta_rms_dbfs"] == 5.0
    assert diff["delta_spectral_centroid_hz"] == 1500.0
    assert diff["delta_dominant_freq_hz"] == 1000.0

