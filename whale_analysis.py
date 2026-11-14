"""
whale_analysis.py
-----------------
Analyses a whale audio recording: loads it, computes basic measurements,
and saves a waveform + spectrogram figure to the outputs/ folder.

Usage:
    python whale_analysis.py                      # default sample
    python whale_analysis.py path/to/file.wav     # your own file
"""

import sys
import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from audio_io import load_audio
from measurements import compute_measurements, format_summary
from plotting import plot_waveform, plot_spectrogram

DEFAULT_AUDIO = os.path.join("data", "sperm_whale_clicks.ogg")
OUTPUT_DIR = "outputs"


def run_analysis(audio_path: str, output_dir: str = OUTPUT_DIR) -> dict:
    """Load audio, compute measurements, save figure and summary."""

    # 1. Load
    print(f"\nLoading: {audio_path}")
    samples, sample_rate = load_audio(audio_path)
    print(f"  {len(samples):,} samples  |  {sample_rate:,} Hz  |  "
          f"{len(samples)/sample_rate:.2f} s")

    # 2. Measure
    m = compute_measurements(samples, sample_rate)

    # 3. Plot
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(audio_path)

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(
        f"Sperm Whale Vocalization — {basename}",
        fontsize=12, y=0.98
    )

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42,
                           top=0.93, bottom=0.09, left=0.09, right=0.96)
    ax_wave = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])

    plot_waveform(ax_wave, samples, sample_rate)
    plot_spectrogram(ax_spec, samples, sample_rate)

    # 4. Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(basename)[0]

    plot_path = os.path.join(output_dir, f"{stem}_{timestamp}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved   : {plot_path}")

    summary_text = format_summary(m, audio_path)
    summary_path = os.path.join(output_dir, f"{stem}_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Summary saved: {summary_path}")

    print("\n" + summary_text)
    return m


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO

    if not os.path.isfile(path):
        print(
            f"\nError: file not found — {path}\n"
            "  Run  python download_sample.py  to get the default recording.\n"
            "  Or pass your own file:  python whale_analysis.py recording.wav\n"
        )
        sys.exit(1)

    run_analysis(path)
