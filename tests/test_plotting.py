"""
tests/test_plotting.py
======================
Tests for the plotting module.

These tests verify that the functions run without errors and produce
the expected plot elements. They do NOT render to screen (Agg backend).
"""

import os
import sys
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plotting import plot_waveform, plot_spectrogram


def make_sine_samples(sr=22050, duration=1.0, freq=440.0):
    """Return normalized float32 mono sine samples."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    s = (np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return s / np.max(np.abs(s))


class TestPlotWaveform:

    def test_waveform_runs_without_error(self):
        """plot_waveform must not raise any exception."""
        samples = make_sine_samples()
        fig, ax = plt.subplots()
        plot_waveform(ax, samples, 22050)
        plt.close(fig)

    def test_waveform_has_time_label(self):
        """Waveform x-axis label must mention 'Time'."""
        samples = make_sine_samples()
        fig, ax = plt.subplots()
        plot_waveform(ax, samples, 22050)
        label = ax.get_xlabel()
        assert "Time" in label or "time" in label
        plt.close(fig)

    def test_waveform_has_amplitude_label(self):
        """Waveform y-axis label must mention 'Amplitude'."""
        samples = make_sine_samples()
        fig, ax = plt.subplots()
        plot_waveform(ax, samples, 22050)
        label = ax.get_ylabel()
        assert "Amplitude" in label or "amplitude" in label
        plt.close(fig)

    def test_waveform_saves_to_file(self, tmp_path):
        """Figure containing waveform must save as a PNG without error."""
        samples = make_sine_samples()
        fig, ax = plt.subplots()
        plot_waveform(ax, samples, 22050)
        out = str(tmp_path / "waveform_test.png")
        fig.savefig(out, dpi=72)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000  # non-trivial file
        plt.close(fig)

    def test_waveform_handles_short_signal(self):
        """Waveform must work even with very short signals (10 ms)."""
        sr = 22050
        n = int(sr * 0.01)  # 10 ms
        samples = np.sin(np.linspace(0, 1, n)).astype(np.float32)
        fig, ax = plt.subplots()
        plot_waveform(ax, samples, sr)
        plt.close(fig)


class TestPlotSpectrogram:

    def test_spectrogram_runs_without_error(self):
        """plot_spectrogram must not raise any exception."""
        samples = make_sine_samples(duration=2.0)
        fig, ax = plt.subplots()
        plot_spectrogram(ax, samples, 22050)
        plt.close(fig)

    def test_spectrogram_has_frequency_label(self):
        """Spectrogram y-axis label must mention 'Frequency'."""
        samples = make_sine_samples(duration=2.0)
        fig, ax = plt.subplots()
        plot_spectrogram(ax, samples, 22050)
        label = ax.get_ylabel()
        assert "Freq" in label or "freq" in label
        plt.close(fig)

    def test_spectrogram_saves_to_file(self, tmp_path):
        """Spectrogram figure must save as a PNG without error."""
        samples = make_sine_samples(duration=2.0)
        fig, ax = plt.subplots()
        plot_spectrogram(ax, samples, 22050)
        out = str(tmp_path / "spectrogram_test.png")
        fig.savefig(out, dpi=72)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000
        plt.close(fig)


class TestOutputFiles:

    def test_full_pipeline_saves_png(self, tmp_path):
        """Integration: run_analysis should save a PNG to the output folder."""
        import wave

        # Write a synthetic WAV
        sr = 22050
        duration = 1.0
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        wav_path = str(tmp_path / "test_sine.wav")
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(signal.tobytes())

        # Run the full pipeline
        output_dir = str(tmp_path / "outputs")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from whale_analysis import run_analysis
        run_analysis(wav_path, output_dir=output_dir)

        # Check outputs
        files = os.listdir(output_dir)
        pngs = [f for f in files if f.endswith(".png")]
        txts = [f for f in files if f.endswith(".txt")]
        assert len(pngs) >= 1, "Expected at least one PNG output"
        assert len(txts) >= 1, "Expected at least one TXT summary output"
