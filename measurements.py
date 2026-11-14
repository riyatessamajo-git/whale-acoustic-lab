"""
measurements.py
---------------
Acoustic measurements for the Whale Acoustic Lab toolkit.

Features:
  - compute_measurements(): Standard time & spectral acoustic metrics.
  - detect_vocalization_events(): Signal-based acoustic event extraction.
  - analyze_coda_sequences(): Candidate sperm whale click sequence / ICI analysis.
  - compare_measurements(): Comparative metrics between two recordings.
  - format_summary(): Plain-text report generation.
"""

import os
import datetime
from itertools import groupby
import numpy as np


def compute_measurements(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Compute basic acoustic measurements from a mono float32 audio array.

    Returns a dict with:
      duration_s        : recording length in seconds
      sample_rate_hz    : samples per second
      n_samples         : total sample count
      channels          : always 1 (audio_io mixes down to mono)
      peak_amplitude    : max(|sample|) — loudest single point
      rms_amplitude     : sqrt(mean(sample^2)) — average loudness
      rms_dbfs          : 20*log10(RMS) — loudness on relative digital scale
      spectral_centroid_hz : frequency-weighted mean of the spectrum
      dominant_freq_hz  : frequency bin with highest energy
      computed_at       : ISO timestamp
    """
    n = len(samples)
    duration = n / sample_rate
    peak = float(np.max(np.abs(samples))) if n > 0 else 0.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if n > 0 else 0.0
    rms_dbfs = float(20.0 * np.log10(rms + 1e-12))
    centroid_hz, dominant_hz = _spectral_features(samples, sample_rate)

    return {
        "duration_s": round(duration, 6),
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "n_samples": n,
        "peak_amplitude": round(peak, 8),
        "rms_amplitude": round(rms, 8),
        "rms_dbfs": round(rms_dbfs, 4),
        "spectral_centroid_hz": round(centroid_hz, 4),
        "dominant_freq_hz": round(dominant_hz, 4),
        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _spectral_features(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """
    Compute spectral centroid and dominant frequency using the FFT.
    Uses at most 4 seconds of audio to keep calculation fast and responsive.
    """
    if len(samples) == 0:
        return 0.0, 0.0

    chunk = samples[: min(len(samples), sample_rate * 4)]
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)

    spec_sum = np.sum(spectrum)
    if spec_sum == 0:
        return 0.0, 0.0

    centroid = float(np.sum(freqs * spectrum) / spec_sum)
    dominant = float(freqs[np.argmax(spectrum)])
    return centroid, dominant


def detect_vocalization_events(
    samples: np.ndarray,
    sample_rate: int,
    min_duration_s: float = 0.02,
    max_events: int = 60
) -> list[dict]:
    """
    Detect potential acoustic / vocalization events using adaptive RMS envelope thresholding.
    
    Returns a list of event dictionaries with:
      id, start_s, end_s, duration_ms, peak_amplitude, snr_db
    """
    if len(samples) == 0:
        return []

    frame_size = max(16, int(0.020 * sample_rate))  # 20 ms frames
    hop_size = max(8, int(0.008 * sample_rate))     # 8 ms hop

    if len(samples) < frame_size:
        return []

    n_frames = (len(samples) - frame_size) // hop_size
    if n_frames <= 0:
        return []

    # Vectorized sliding-window RMS envelope
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_size)[::hop_size]
    rms_env = np.sqrt(np.mean(frames ** 2, axis=1))
    times = np.arange(len(rms_env)) * hop_size / sample_rate

    # Adaptive noise floor calculation
    noise_floor = float(np.percentile(rms_env, 30))
    std_val = float(np.std(rms_env))
    threshold = max(0.012, noise_floor + 1.2 * std_val)

    active = rms_env > threshold

    raw_events = []
    idx = 0
    for k, g in groupby(active):
        length = sum(1 for _ in g)
        if k:
            t_start = float(times[idx])
            t_end = float(times[min(idx + length - 1, len(times) - 1)]) + (frame_size / sample_rate)
            dur = t_end - t_start
            if dur >= min_duration_s:
                s_idx = max(0, int(t_start * sample_rate))
                e_idx = min(len(samples), int(t_end * sample_rate))
                sub = samples[s_idx:e_idx]
                peak = float(np.max(np.abs(sub))) if len(sub) > 0 else 0.0
                event_rms = float(np.sqrt(np.mean(sub ** 2))) if len(sub) > 0 else 0.0
                snr = 20.0 * np.log10((event_rms + 1e-8) / (noise_floor + 1e-8))
                raw_events.append({
                    "start_s": round(t_start, 3),
                    "end_s": round(t_end, 3),
                    "duration_ms": round(dur * 1000.0, 1),
                    "peak_amplitude": round(peak, 4),
                    "snr_db": round(max(0.0, float(snr)), 1),
                })
        idx += length

    # Merge closely spaced segments (within 35 ms)
    merged = []
    for ev in raw_events:
        if not merged:
            merged.append(ev)
        else:
            prev = merged[-1]
            if ev["start_s"] - prev["end_s"] <= 0.035:
                new_start = prev["start_s"]
                new_end = max(prev["end_s"], ev["end_s"])
                dur = new_end - new_start
                s_idx = max(0, int(new_start * sample_rate))
                e_idx = min(len(samples), int(new_end * sample_rate))
                sub = samples[s_idx:e_idx]
                peak = float(np.max(np.abs(sub))) if len(sub) > 0 else 0.0
                merged[-1] = {
                    "start_s": round(new_start, 3),
                    "end_s": round(new_end, 3),
                    "duration_ms": round(dur * 1000.0, 1),
                    "peak_amplitude": round(peak, 4),
                    "snr_db": max(prev["snr_db"], ev["snr_db"]),
                }
            else:
                merged.append(ev)

    # Sort and cap
    if len(merged) > max_events:
        top_by_peak = sorted(merged, key=lambda x: x["peak_amplitude"], reverse=True)[:max_events]
        merged = sorted(top_by_peak, key=lambda x: x["start_s"])

    for i, ev in enumerate(merged):
        ev["id"] = i + 1

    return merged


def analyze_coda_sequences(
    samples: np.ndarray,
    sample_rate: int,
    max_gap_s: float = 0.22,
    min_clicks: int = 3
) -> dict:
    """
    Explore candidate sperm whale click sequences and inter-click intervals (ICIs).

    IMPORTANT SCIENTIFIC LABEL:
    These are algorithmically detected candidate sequences, NOT confirmed codas.
    """
    import scipy.signal as sig

    if len(samples) == 0:
        return {
            "total_clicks": 0,
            "total_sequences": 0,
            "mean_ici_ms": 0.0,
            "candidate_sequences": [],
            "click_timeline": [],
            "scientific_disclaimer": "No signal detected."
        }

    # Minimum distance between clicks: 12 ms
    min_dist = max(10, int(0.012 * sample_rate))
    abs_samples = np.abs(samples)
    noise_floor = float(np.percentile(abs_samples, 40))
    height_thresh = max(0.12, noise_floor + 2.5 * float(np.std(abs_samples)))

    peaks, _ = sig.find_peaks(abs_samples, height=height_thresh, distance=min_dist)
    click_times = peaks / sample_rate
    total_clicks = len(click_times)

    disclaimer_text = (
        "Candidate Click Sequences: Detected via automated impulse thresholding. "
        "These represent acoustic pulse clusters and must be verified against field "
        "annotations before being classified as behavioral codas or communicative units."
    )

    if total_clicks < min_clicks:
        return {
            "total_clicks": total_clicks,
            "total_sequences": 0,
            "mean_ici_ms": 0.0,
            "candidate_sequences": [],
            "click_timeline": [round(float(t), 3) for t in click_times[:120]],
            "scientific_disclaimer": disclaimer_text
        }

    # Group clicks into candidate click trains / codas
    sequences = []
    current = []

    for t in click_times:
        t_val = float(t)
        if not current:
            current.append(t_val)
        else:
            gap = t_val - current[-1]
            if gap <= max_gap_s:
                current.append(t_val)
            else:
                if len(current) >= min_clicks:
                    sequences.append(current)
                current = [t_val]

    if len(current) >= min_clicks:
        sequences.append(current)

    all_icis = []
    seq_dicts = []
    for idx, seq in enumerate(sequences):
        seq_icis = [(seq[k] - seq[k - 1]) * 1000.0 for k in range(1, len(seq))]
        all_icis.extend(seq_icis)
        dur = seq[-1] - seq[0]
        seq_dicts.append({
            "sequence_id": idx + 1,
            "click_count": len(seq),
            "start_s": round(seq[0], 3),
            "end_s": round(seq[-1], 3),
            "duration_s": round(dur, 3),
            "avg_ici_ms": round(float(np.mean(seq_icis)), 1) if seq_icis else 0.0,
            "std_ici_ms": round(float(np.std(seq_icis)), 1) if len(seq_icis) > 1 else 0.0,
            "icis_ms": [round(ici, 1) for ici in seq_icis[:15]],
            "click_times": [round(ct, 3) for ct in seq[:25]]
        })

    overall_mean_ici = round(float(np.mean(all_icis)), 1) if all_icis else 0.0

    return {
        "total_clicks": total_clicks,
        "total_sequences": len(seq_dicts),
        "mean_ici_ms": overall_mean_ici,
        "candidate_sequences": seq_dicts[:35],
        "click_timeline": [round(float(t), 3) for t in click_times[:150]],
        "scientific_disclaimer": disclaimer_text
    }


def compare_measurements(m_a: dict, m_b: dict) -> dict:
    """
    Calculate numerical delta differences between two acoustic measurements.
    """
    def _val(m, *keys, default=0.0):
        for k in keys:
            if k in m:
                return float(m[k])
        return default

    dur_a = _val(m_a, "duration_s")
    dur_b = _val(m_b, "duration_s")
    rms_a = _val(m_a, "rms_dbfs")
    rms_b = _val(m_b, "rms_dbfs")
    cen_a = _val(m_a, "spectral_centroid_hz", "spectral_centroid")
    cen_b = _val(m_b, "spectral_centroid_hz", "spectral_centroid")
    dom_a = _val(m_a, "dominant_freq_hz", "dominant_freq")
    dom_b = _val(m_b, "dominant_freq_hz", "dominant_freq")

    return {
        "delta_duration_s": round(dur_b - dur_a, 2),
        "delta_rms_dbfs": round(rms_b - rms_a, 2),
        "delta_spectral_centroid_hz": round(cen_b - cen_a, 1),
        "delta_dominant_freq_hz": round(dom_b - dom_a, 1),
    }


def format_summary(measurements: dict, audio_path: str) -> str:
    """Return a short plain-text summary of the measurements."""
    m = measurements
    sr = m.get("sample_rate_hz", m.get("sample_rate", 0))
    cen = m.get("spectral_centroid_hz", m.get("spectral_centroid", 0))
    dom = m.get("dominant_freq_hz", m.get("dominant_freq", 0))
    lines = [
        "============================================================",
        " WHALE ACOUSTIC LAB — ACOUSTIC ANALYSIS REPORT",
        " Computational Analysis of Whale Vocalizations",
        "============================================================",
        f"File              : {os.path.basename(audio_path)}",
        f"Species / Ref.    : Physeter macrocephalus (Sperm Whale recording)",
        f"Duration          : {float(m.get('duration_s', 0)):.2f} s",
        f"Sample rate       : {sr:,} Hz",
        "",
        "--- Amplitude & Energy (Digital Full Scale) ---",
        f"Peak amplitude    : {float(m.get('peak_amplitude', 0)):.4f}  (normalized, 1.0 = full scale)",
        f"RMS amplitude     : {float(m.get('rms_amplitude', 0)):.4f}",
        f"RMS level         : {float(m.get('rms_dbfs', 0)):.2f} dBFS (relative to digital clipping)",
        "",
        "--- Spectral Characteristics ---",
        f"Spectral centroid : {float(cen):.1f} Hz (energy center of gravity)",
        f"Dominant freq.    : {float(dom):.1f} Hz (peak energy frequency bin)",
        "",
        "--- Scientific Calibration & Accuracy Note ---",
        "• Levels are in digital dBFS. Conversion to calibrated underwater sound pressure",
        "  levels (dB re 1 µPa) requires specific hydrophone sensitivity and pre-amp gain.",
        "• These measurements describe physical acoustic wave dynamics and do NOT infer",
        "  communicative meaning, linguistic content, or species identity without validated data.",
        "============================================================",
    ]
    return "\n".join(lines)
