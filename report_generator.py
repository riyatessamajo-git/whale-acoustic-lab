"""
report_generator.py
-------------------
Generates publication-quality PDF and CSV bioacoustic research reports
for the Whale Acoustic Lab toolkit.
"""

import io
import csv
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plotting import plot_waveform, plot_spectrogram


def generate_csv_report(data: dict) -> str:
    """
    Generate a structured CSV report containing recording metadata,
    acoustic measurements, STFT parameters, detected vocalization events,
    and candidate coda sequences.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # 1. Header
    writer.writerow(["WHALE ACOUSTIC LAB — RESEARCH REPORT"])
    writer.writerow(["Computational Analysis of Whale Vocalizations"])
    writer.writerow(["Generated At", data.get("analysed_at", datetime.datetime.now().isoformat())])
    writer.writerow([])

    # 2. Metadata
    writer.writerow(["--- RECORDING METADATA ---"])
    writer.writerow(["Filename", data.get("filename", "audio.wav")])
    writer.writerow(["Duration (seconds)", data.get("duration_s", 0)])
    writer.writerow(["Sample Rate (Hz)", data.get("sample_rate", 0)])
    writer.writerow(["Total Samples", data.get("num_samples", 0)])
    writer.writerow([])

    # 3. Global Measurements
    writer.writerow(["--- ACOUSTIC MEASUREMENTS ---"])
    writer.writerow(["Measurement", "Value", "Unit", "Notes"])
    writer.writerow(["Peak Amplitude", data.get("peak_amplitude", 0), "normalized", "1.0 = full scale"])
    writer.writerow(["RMS Amplitude", data.get("rms_amplitude", 0), "normalized", "Root mean square power"])
    writer.writerow(["RMS Level", data.get("rms_dbfs", 0), "dBFS", "Relative digital scale (0 dBFS = digital clipping)"])
    writer.writerow(["Spectral Centroid", data.get("spectral_centroid", 0), "Hz", "Energy-weighted center of gravity"])
    writer.writerow(["Dominant Frequency", data.get("dominant_freq", 0), "Hz", "Peak energy spectral bin"])
    writer.writerow([])

    # 4. STFT Parameters
    writer.writerow(["--- STFT ANALYSIS PARAMETERS ---"])
    stft_p = data.get("stft_params", {})
    writer.writerow(["FFT Window Size", stft_p.get("window_size", 512), "samples"])
    writer.writerow(["Hop Overlap", f"{int(float(stft_p.get('overlap', 0.75)) * 100)}%", "fraction"])
    writer.writerow(["Window Function", stft_p.get("window_type", "hann").capitalize(), ""])
    writer.writerow([])

    # 5. Detected Vocalization Events
    events = data.get("events", [])
    writer.writerow(["--- VOCALIZATION EVENT DETECTIONS ---"])
    writer.writerow(["Total Detected Events", len(events)])
    if events:
        writer.writerow(["Event ID", "Start Time (s)", "End Time (s)", "Duration (ms)", "Peak Amplitude", "SNR (dB)"])
        for ev in events:
            writer.writerow([
                ev.get("id"),
                ev.get("start_s"),
                ev.get("end_s"),
                ev.get("duration_ms"),
                ev.get("peak_amplitude"),
                ev.get("snr_db", "")
            ])
    writer.writerow([])

    # 6. Candidate Coda Sequences
    coda = data.get("coda_analysis", {})
    seqs = coda.get("candidate_sequences", [])
    writer.writerow(["--- CANDIDATE CODA SEQUENCES (UNVERIFIED) ---"])
    writer.writerow(["Total Clicks Detected", coda.get("total_clicks", 0)])
    writer.writerow(["Candidate Sequences Detected", coda.get("total_sequences", 0)])
    writer.writerow(["Overall Mean ICI (ms)", coda.get("mean_ici_ms", 0)])
    if seqs:
        writer.writerow(["Sequence ID", "Click Count", "Start Time (s)", "End Time (s)", "Duration (s)", "Mean ICI (ms)", "Std ICI (ms)"])
        for s in seqs:
            writer.writerow([
                s.get("sequence_id"),
                s.get("click_count"),
                s.get("start_s"),
                s.get("end_s"),
                s.get("duration_s"),
                s.get("avg_ici_ms"),
                s.get("std_ici_ms")
            ])
    writer.writerow([])

    # 7. Scientific Disclaimers
    writer.writerow(["--- SCIENTIFIC CALIBRATION & ACCURACY NOTICE ---"])
    writer.writerow(["Underwater Pressure Disclaimer", "Measurements in dBFS represent relative uncalibrated digital scale. Calibration to absolute underwater sound pressure levels (dB re 1 µPa) requires verified hydrophone sensitivity and preamplifier gain."])
    writer.writerow(["Non-Semantic DSP Disclaimer", "Automated click and event detections quantify physical acoustic wave properties. They do not infer semantic meaning or species classification without validated reference data."])

    return output.getvalue()


def generate_pdf_report(
    data: dict,
    samples: np.ndarray,
    sample_rate: int,
    stft_params: dict = None
) -> bytes:
    """
    Generate a 2-page publication-quality bioacoustic research report PDF.
    """
    if stft_params is None:
        stft_params = data.get("stft_params", {"window_size": 512, "overlap": 0.75, "window_type": "hann"})

    w_size = int(stft_params.get("window_size", 512))
    overlap = float(stft_params.get("overlap", 0.75))
    w_type = str(stft_params.get("window_type", "hann"))

    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # =====================================================================
        # PAGE 1: Overview, Measurements, Waveform & Spectrogram
        # =====================================================================
        fig1 = plt.figure(figsize=(8.5, 11), facecolor="#ffffff")
        
        # Header banner
        fig1.text(0.08, 0.94, "Whale Acoustic Lab", fontsize=18, weight="bold", color="#1e3a8a")
        fig1.text(0.08, 0.915, "Computational Analysis of Whale Vocalizations — Bioacoustic Research Report", fontsize=10, color="#475569")
        fig1.text(0.08, 0.895, f"File: {data.get('filename', 'Audio')} | Date: {data.get('analysed_at', '')}", fontsize=8.5, color="#64748b")
        
        # Separator line
        line_ax = fig1.add_axes([0.08, 0.88, 0.84, 0.002])
        line_ax.patch.set_facecolor("#2563eb")
        line_ax.axis("off")

        # Table 1: Metadata & Measurements Summary
        meta_table_ax = fig1.add_axes([0.08, 0.74, 0.84, 0.12])
        meta_table_ax.axis("off")
        
        col_labels = ["Parameter / Metric", "Value", "Parameter / Metric", "Value"]
        table_data = [
            ["Duration", f"{data.get('duration_s', 0):.2f} s", "Peak Amplitude", f"{data.get('peak_amplitude', 0):.4f}"],
            ["Sample Rate", f"{data.get('sample_rate', 0):,} Hz", "RMS Amplitude", f"{data.get('rms_amplitude', 0):.4f}"],
            ["Total Samples", f"{data.get('num_samples', 0):,}", "RMS Level", f"{data.get('rms_dbfs', 0):.2f} dBFS"],
            ["Spectral Centroid", f"{data.get('spectral_centroid', 0):,} Hz", "Dominant Frequency", f"{data.get('dominant_freq', 0):,} Hz"],
            ["STFT Window", f"{w_type.capitalize()} ({w_size} samples)", "STFT Overlap", f"{int(overlap * 100)}% ({int(w_size * (1 - overlap))} hop)"]
        ]
        
        t = meta_table_ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="left",
            loc="center",
            colWidths=[0.28, 0.22, 0.28, 0.22]
        )
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1.0, 1.25)
        
        # Style table headers
        for (row, col), cell in t.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if row == 0:
                cell.set_facecolor("#eff6ff")
                cell.set_text_props(weight="bold", color="#1e40af")
            else:
                cell.set_facecolor("#ffffff" if row % 2 == 0 else "#f8fafc")

        # Figure 1: Waveform
        ax_wave = fig1.add_axes([0.08, 0.44, 0.84, 0.24])
        plot_waveform(ax_wave, samples, sample_rate, color="#2563eb", title="Time-Domain Waveform")
        ax_wave.tick_params(colors="#475569", labelsize=8)
        for s in ax_wave.spines.values():
            s.set_color("#cbd5e1")
        ax_wave.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")

        # Figure 2: Spectrogram
        ax_spec = fig1.add_axes([0.08, 0.12, 0.84, 0.26])
        plot_spectrogram(
            ax_spec,
            samples,
            sample_rate,
            window_size=w_size,
            overlap=overlap,
            window_type=w_type,
            cmap="viridis"
        )
        ax_spec.tick_params(colors="#475569", labelsize=8)
        for s in ax_spec.spines.values():
            s.set_color("#cbd5e1")

        fig1.text(0.5, 0.04, "Page 1 of 2 — Whale Acoustic Lab", fontsize=8, ha="center", color="#94a3b8")
        pdf.savefig(fig1, dpi=200)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2: Event Detections, Coda Analysis & Scientific Disclaimers
        # =====================================================================
        fig2 = plt.figure(figsize=(8.5, 11), facecolor="#ffffff")

        fig2.text(0.08, 0.94, "Whale Acoustic Lab — Detailed Analysis", fontsize=15, weight="bold", color="#1e3a8a")
        fig2.text(0.08, 0.918, f"Detected Events & Candidate Coda Sequences | {data.get('filename', '')}", fontsize=9, color="#475569")

        # Section 1: Vocalization Events
        events = data.get("events", [])
        fig2.text(0.08, 0.88, f"1. Vocalization Event Detections (Total: {len(events)})", fontsize=11, weight="bold", color="#0f172a")

        ev_ax = fig2.add_axes([0.08, 0.65, 0.84, 0.21])
        ev_ax.axis("off")

        ev_cols = ["ID", "Start (s)", "End (s)", "Duration (ms)", "Peak Amp", "SNR (dB)"]
        ev_rows = []
        for ev in events[:8]:  # Show top 8 events
            ev_rows.append([
                f"#{ev.get('id')}",
                f"{ev.get('start_s'):.3f}",
                f"{ev.get('end_s'):.3f}",
                f"{ev.get('duration_ms'):.1f}",
                f"{ev.get('peak_amplitude'):.4f}",
                f"{ev.get('snr_db', 0):.1f}"
            ])
        if not ev_rows:
            ev_rows = [["None", "—", "—", "—", "—", "—"]]

        t_ev = ev_ax.table(
            cellText=ev_rows,
            colLabels=ev_cols,
            cellLoc="center",
            loc="center",
            colWidths=[0.10, 0.18, 0.18, 0.20, 0.18, 0.16]
        )
        t_ev.auto_set_font_size(False)
        t_ev.set_fontsize(7.5)
        t_ev.scale(1.0, 1.25)

        for (row, col), cell in t_ev.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if row == 0:
                cell.set_facecolor("#eff6ff")
                cell.set_text_props(weight="bold", color="#1e40af")
            else:
                cell.set_facecolor("#ffffff" if row % 2 == 0 else "#f8fafc")

        # Section 2: Coda Analysis
        coda = data.get("coda_analysis", {})
        seqs = coda.get("candidate_sequences", [])
        total_clicks = coda.get("total_clicks", 0)
        mean_ici = coda.get("mean_ici_ms", 0)

        fig2.text(
            0.08, 0.61,
            f"2. Coda Analysis — Candidate Click Sequences (Clicks: {total_clicks:,} | Mean ICI: {mean_ici:.1f} ms)",
            fontsize=11, weight="bold", color="#0f172a"
        )

        coda_ax = fig2.add_axes([0.08, 0.38, 0.84, 0.21])
        coda_ax.axis("off")

        coda_cols = ["Seq ID", "Clicks", "Start (s)", "End (s)", "Duration (s)", "Mean ICI (ms)"]
        coda_rows = []
        for s in seqs[:8]:  # Show top 8 candidate sequences
            coda_rows.append([
                f"Seq {s.get('sequence_id')}",
                str(s.get("click_count")),
                f"{s.get('start_s'):.3f}",
                f"{s.get('end_s'):.3f}",
                f"{s.get('duration_s'):.3f}",
                f"{s.get('avg_ici_ms'):.1f}"
            ])
        if not coda_rows:
            coda_rows = [["None", "0", "—", "—", "—", "—"]]

        t_coda = coda_ax.table(
            cellText=coda_rows,
            colLabels=coda_cols,
            cellLoc="center",
            loc="center",
            colWidths=[0.14, 0.14, 0.18, 0.18, 0.18, 0.18]
        )
        t_coda.auto_set_font_size(False)
        t_coda.set_fontsize(7.5)
        t_coda.scale(1.0, 1.25)

        for (row, col), cell in t_coda.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if row == 0:
                cell.set_facecolor("#eff6ff")
                cell.set_text_props(weight="bold", color="#1e40af")
            else:
                cell.set_facecolor("#ffffff" if row % 2 == 0 else "#f8fafc")

        # Section 3: Scientific Calibration & Disclaimers Box
        disc_ax = fig2.add_axes([0.08, 0.10, 0.84, 0.23])
        disc_ax.patch.set_facecolor("#f8fafc")
        disc_ax.patch.set_edgecolor("#bfdbfe")
        disc_ax.patch.set_linewidth(1)
        disc_ax.set_xticks([])
        disc_ax.set_yticks([])

        disc_text = (
            "SCIENTIFIC CALIBRATION & ACCURACY NOTICE\n\n"
            "1. Digital Full Scale (dBFS) vs Underwater Sound Pressure (dB re 1 µPa):\n"
            "   Measurements in this report are given in dBFS (decibels relative to full scale, where 0 dBFS\n"
            "   represents digital clipping). They reflect uncalibrated digital audio. Conversion to physical\n"
            "   underwater sound pressure levels (dB re 1 µPa) requires verified hydrophone voltage\n"
            "   sensitivity (dBV/µPa) and amplifier gain curves.\n\n"
            "2. Signal DSP vs Linguistic / Communication Claims:\n"
            "   The metrics, impulse detections, and candidate click sequences quantify physical wave dynamics\n"
            "   (energy distribution, temporal spacing, and spectral centroid). They do NOT infer communicative\n"
            "   intent, semantic meaning, or species identification without validated acoustic reference models\n"
            "   and visual ground-truth confirmation.\n\n"
            "3. Candidate Sequences Disclaimer:\n"
            "   Click sequences are grouped algorithmically based on temporal inter-click intervals (ICIs).\n"
            "   They represent candidate pulse clusters and must be validated before classification as codas."
        )
        disc_ax.text(0.03, 0.94, disc_text, transform=disc_ax.transAxes, fontsize=7.2, va="top", color="#334155", family="sans-serif")

        fig2.text(0.5, 0.04, "Page 2 of 2 — Whale Acoustic Lab", fontsize=8, ha="center", color="#94a3b8")
        pdf.savefig(fig2, dpi=200)
        plt.close(fig2)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

