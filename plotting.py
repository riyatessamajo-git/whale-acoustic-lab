"""
plotting.py
-----------
Waveform and spectrogram plotting utilities for the Whale Acoustic Lab toolkit.

Features:
  - plot_waveform(): Visualizes time-domain amplitude and RMS envelope.
  - plot_spectrogram(): Short-Time Fourier Transform (STFT) with configurable
    window size, overlap, window functions (Hann, Hamming, Blackman), and time slicing.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(
    ax,
    samples: np.ndarray,
    sample_rate: int,
    t_start: float = None,
    t_end: float = None,
    color: str = "#2563eb",
    title: str = "Waveform"
) -> None:
    """
    Draw amplitude versus time onto ax.
    Supports optional time range slicing (t_start, t_end).
    """
    total_duration = len(samples) / sample_rate

    # Apply time slicing if requested
    if t_start is not None or t_end is not None:
        t0 = max(0.0, float(t_start or 0.0))
        t1 = min(total_duration, float(t_end or total_duration))
        if t1 > t0:
            idx0 = int(t0 * sample_rate)
            idx1 = int(t1 * sample_rate)
            samples = samples[idx0:idx1]
            time_offset = t0
            duration = t1 - t0
        else:
            time_offset = 0.0
            duration = total_duration
    else:
        time_offset = 0.0
        duration = total_duration

    if len(samples) == 0:
        ax.text(0.5, 0.5, "No signal data", ha="center", va="center")
        return

    time_axis = np.linspace(time_offset, time_offset + duration, num=len(samples))

    max_display = 60_000
    if len(samples) > max_display:
        step = len(samples) // max_display
        time_axis = time_axis[::step]
        samples = samples[::step]

    rms = float(np.sqrt(np.mean(samples ** 2)))

    ax.plot(time_axis, samples, color=color, linewidth=0.6, alpha=0.9, label="Signal")
    ax.axhline(0, color="#94a3b8", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.axhspan(-rms, rms, alpha=0.15, color="#60a5fa",
               label=f"RMS band ({rms:.4f})")

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude (normalized)", fontsize=9)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlim(time_offset, time_offset + duration)
    ax.set_ylim(-1.08, 1.08)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)


def plot_spectrogram(
    ax,
    samples: np.ndarray,
    sample_rate: int,
    window_size: int = 512,
    overlap: float = 0.75,
    window_type: str = "hann",
    t_start: float = None,
    t_end: float = None,
    max_display_cols: int = 1200,
    cmap: str = "viridis"
) -> None:
    """
    Draw an STFT spectrogram (magnitude in dB) onto ax.

    Parameters:
      window_size      : FFT window in samples (256, 512, 1024, 2048)
      overlap          : fraction of window overlap (0.50, 0.75, 0.875)
      window_type      : windowing function ('hann', 'hamming', 'blackman')
      t_start, t_end   : optional time boundaries for zoom inspection
    """
    total_duration = len(samples) / sample_rate

    # Time slicing
    if t_start is not None or t_end is not None:
        t0 = max(0.0, float(t_start or 0.0))
        t1 = min(total_duration, float(t_end or total_duration))
        if t1 > t0:
            idx0 = int(t0 * sample_rate)
            idx1 = int(t1 * sample_rate)
            samples = samples[idx0:idx1]
            time_offset = t0
            duration = t1 - t0
        else:
            time_offset = 0.0
            duration = total_duration
    else:
        time_offset = 0.0
        duration = total_duration

    if len(samples) < window_size:
        ax.text(0.5, 0.5, "Signal too short for selected FFT size", ha="center", va="center")
        return

    # Choose window function
    wtype = str(window_type).lower()
    if "hamm" in wtype:
        window = np.hamming(window_size)
    elif "black" in wtype:
        window = np.blackman(window_size)
    else:
        window = np.hanning(window_size)

    hop = max(1, int(window_size * (1.0 - overlap)))
    stft = _compute_stft(samples, window_size, hop, window)

    magnitude_db = 20.0 * np.log10(np.abs(stft) + 1e-8)
    n_frames = stft.shape[1]

    if n_frames > max_display_cols:
        step = n_frames // max_display_cols
        magnitude_db = magnitude_db[:, ::step]

    vmin = np.percentile(magnitude_db, 5)
    vmax = np.percentile(magnitude_db, 99)

    im = ax.imshow(
        magnitude_db,
        origin="lower",
        aspect="auto",
        extent=[time_offset, time_offset + duration, 0, sample_rate / 2],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_title(f"Spectrogram (STFT, {window_type.capitalize()} {window_size}, {int(overlap*100)}% overlap)", fontsize=10, weight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("dBFS", fontsize=8)


def _compute_stft(samples, window_size, hop, window):
    """
    Compute STFT using NumPy.
    Returns complex matrix of shape (n_freqs, n_frames).
    """
    n_frames = 1 + (len(samples) - window_size) // hop
    n_freqs = window_size // 2 + 1
    stft = np.zeros((n_freqs, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop
        stft[:, i] = np.fft.rfft(samples[start: start + window_size] * window)
    return stft
