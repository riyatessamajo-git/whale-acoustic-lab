"""
app.py
------
Whale Acoustic Lab — Computational Analysis of Whale Vocalizations.
Backend server providing audio signal processing, interactive spectrograms,
vocalization event detection, coda sequence analysis, recording comparison,
and PDF/CSV research report generation.
"""

import os
import io
import base64
import datetime
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_from_directory, Response
from werkzeug.utils import secure_filename

from audio_io import load_audio
from measurements import (
    compute_measurements,
    detect_vocalization_events,
    analyze_coda_sequences,
    compare_measurements,
    format_summary,
)
from plotting import plot_waveform, plot_spectrogram
from report_generator import generate_csv_report, generate_pdf_report

app = Flask(__name__, static_folder="static")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
STATIC_DIR  = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DEFAULT_OGG = os.path.join(DATA_DIR, "sperm_whale_clicks.ogg")

ALLOWED_EXTENSIONS = {".wav", ".ogg", ".mp3", ".flac", ".aiff"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=140,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none"
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _resolve_audio_path(filename_or_flag: str) -> tuple[str, str]:
    """Resolve audio path and display name."""
    if not filename_or_flag or filename_or_flag in ("preset", "default", "sperm_whale_clicks.ogg"):
        return DEFAULT_OGG, "sperm_whale_clicks.ogg"

    safe_name = os.path.basename(filename_or_flag)
    uploaded_path = os.path.join(UPLOADS_DIR, safe_name)
    if os.path.isfile(uploaded_path):
        return uploaded_path, safe_name

    data_path = os.path.join(DATA_DIR, safe_name)
    if os.path.isfile(data_path):
        return data_path, safe_name

    # Check if exact path given
    if os.path.isfile(filename_or_flag):
        return filename_or_flag, os.path.basename(filename_or_flag)

    return DEFAULT_OGG, "sperm_whale_clicks.ogg"


def _render_plots(
    samples: np.ndarray,
    sample_rate: int,
    window_size: int = 512,
    overlap: float = 0.75,
    window_type: str = "hann",
    t_start: float = None,
    t_end: float = None,
) -> tuple[str, str]:
    """Render waveform and spectrogram figures into base64 PNGs."""
    BG_FIG = "#ffffff"
    BG_AX = "#f8fafc"
    TICK = "#475569"
    SPINE = "#cbd5e1"
    TITLE_COLOR = "#0f172a"
    GRID_COLOR = "#e2e8f0"

    # 1. Waveform figure
    fig_w, ax_w = plt.subplots(figsize=(11.5, 2.9))
    fig_w.patch.set_facecolor(BG_FIG)
    ax_w.set_facecolor(BG_AX)
    plot_waveform(ax_w, samples, sample_rate, t_start=t_start, t_end=t_end, color="#2563eb", title="Waveform — Amplitude vs Time")
    ax_w.tick_params(colors=TICK, labelsize=8.5)
    for s in ax_w.spines.values():
        s.set_color(SPINE)
    ax_w.xaxis.label.set_color(TITLE_COLOR)
    ax_w.yaxis.label.set_color(TITLE_COLOR)
    ax_w.title.set_color(TITLE_COLOR)
    ax_w.grid(True, linestyle=":", alpha=0.6, color=GRID_COLOR)
    fig_w.tight_layout(pad=0.5)
    waveform_b64 = _fig_to_b64(fig_w)
    plt.close(fig_w)

    # 2. Spectrogram figure
    fig_s, ax_s = plt.subplots(figsize=(11.5, 3.5))
    fig_s.patch.set_facecolor(BG_FIG)
    ax_s.set_facecolor(BG_AX)
    plot_spectrogram(
        ax_s,
        samples,
        sample_rate,
        window_size=window_size,
        overlap=overlap,
        window_type=window_type,
        t_start=t_start,
        t_end=t_end,
        cmap="viridis"
    )
    ax_s.tick_params(colors=TICK, labelsize=8.5)
    for s in ax_s.spines.values():
        s.set_color(SPINE)
    ax_s.xaxis.label.set_color(TITLE_COLOR)
    ax_s.yaxis.label.set_color(TITLE_COLOR)
    ax_s.title.set_color(TITLE_COLOR)
    fig_s.tight_layout(pad=0.5)
    spectrogram_b64 = _fig_to_b64(fig_s)
    plt.close(fig_s)

    return waveform_b64, spectrogram_b64


def _run_full_analysis(
    audio_path: str,
    display_name: str = None,
    window_size: int = 512,
    overlap: float = 0.75,
    window_type: str = "hann",
    t_start: float = None,
    t_end: float = None,
) -> dict:
    """Run full acoustic measurements, event detection, coda analysis, and plot rendering."""
    if display_name is None:
        display_name = os.path.basename(audio_path)

    samples, sample_rate = load_audio(audio_path)

    # 1. Global measurements
    m = compute_measurements(samples, sample_rate)

    # 2. Event detection
    events = detect_vocalization_events(samples, sample_rate)

    # 3. Coda analysis
    coda_data = analyze_coda_sequences(samples, sample_rate)

    # 4. Plots
    waveform_b64, spectrogram_b64 = _render_plots(
        samples,
        sample_rate,
        window_size=window_size,
        overlap=overlap,
        window_type=window_type,
        t_start=t_start,
        t_end=t_end,
    )

    duration_s = round(float(m.get("duration_s", 0)), 2)
    sample_rate_val = int(m.get("sample_rate_hz", m.get("sample_rate", sample_rate)))
    num_samples_val = int(m.get("n_samples", m.get("num_samples", len(samples))))
    peak_amp = round(float(m.get("peak_amplitude", 1.0)), 4)
    rms_amp = round(float(m.get("rms_amplitude", 0.0)), 4)
    rms_dbfs = round(float(m.get("rms_dbfs", 0.0)), 2)
    spectral_centroid = int(round(float(m.get("spectral_centroid_hz", m.get("spectral_centroid", 0)))))
    dominant_freq = int(round(float(m.get("dominant_freq_hz", m.get("dominant_freq", 0)))))

    summary_text = format_summary(m, audio_path)

    is_default = os.path.abspath(audio_path) == os.path.abspath(DEFAULT_OGG)
    audio_stream_url = "/api/audio/preset" if is_default else f"/api/audio/uploaded/{os.path.basename(audio_path)}"

    stft_params = {
        "window_size": int(window_size),
        "overlap": float(overlap),
        "window_type": str(window_type).lower(),
        "hop_length": int(window_size * (1.0 - overlap)),
        "t_start": t_start,
        "t_end": t_end,
    }

    return {
        "filename": display_name,
        "internal_name": os.path.basename(audio_path),
        "is_preset": is_default,
        "audio_url": audio_stream_url,
        "duration_s": duration_s,
        "sample_rate": sample_rate_val,
        "num_samples": num_samples_val,
        "peak_amplitude": peak_amp,
        "rms_amplitude": rms_amp,
        "rms_dbfs": rms_dbfs,
        "spectral_centroid": spectral_centroid,
        "dominant_freq": dominant_freq,
        "stft_params": stft_params,
        "events": events,
        "total_events": len(events),
        "coda_analysis": coda_data,
        "waveform_png": waveform_b64,
        "spectrogram_png": spectrogram_b64,
        "summary_text": summary_text,
        "analysed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scientific_notices": {
            "calibration_notice": (
                "Decibels relative to Full Scale (dBFS): Levels reflect relative digital values where "
                "0 dBFS is the maximum digital clipping ceiling. Calibrated underwater sound pressure "
                "levels (dB re 1 µPa) require known hydrophone sensitivity (dBV/µPa) and amplifier gain."
            ),
            "communication_notice": (
                "Physical DSP Analysis: Signal properties and candidate click sequences quantify physical "
                "wave mechanics only. They do not infer semantic meaning or species identity without "
                "validated acoustic models and ground-truth biological annotations."
            )
        }
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/audio/preset")
def stream_preset():
    """Stream default sperm whale sample."""
    if not os.path.isfile(DEFAULT_OGG):
        return jsonify({"error": "Sample file not found on disk"}), 404
    return send_from_directory(DATA_DIR, "sperm_whale_clicks.ogg", mimetype="audio/ogg")


@app.route("/api/audio/uploaded/<path:filename>")
def stream_uploaded(filename):
    """Stream uploaded audio file."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(UPLOADS_DIR, safe_name)
    if not os.path.isfile(file_path):
        return jsonify({"error": "Uploaded file not found"}), 404
    return send_from_directory(UPLOADS_DIR, safe_name)


@app.route("/api/analyse", methods=["GET", "POST"])
def api_analyse():
    """
    Run full acoustic analysis on an uploaded file or default sample.
    Accepts STFT parameters (window_size, overlap, window_type, t_start, t_end).
    """
    try:
        # Extract optional STFT parameters
        window_size = int(request.form.get("window_size", 512) if request.form else 512)
        overlap = float(request.form.get("overlap", 0.75) if request.form else 0.75)
        window_type = str(request.form.get("window_type", "hann") if request.form else "hann")
        t_start = request.form.get("t_start") if request.form else None
        t_end = request.form.get("t_end") if request.form else None

        t_start = float(t_start) if t_start not in (None, "", "null") else None
        t_end = float(t_end) if t_end not in (None, "", "null") else None

        # Check for uploaded audio file
        if request.method == "POST" and "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                return jsonify({"error": "No file was selected for upload."}), 400

            original_name = file.filename
            _, ext = os.path.splitext(original_name)
            ext = ext.lower()

            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({
                    "error": f"Unsupported format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }), 400

            unique_id = uuid.uuid4().hex[:8]
            safe_base = secure_filename(os.path.splitext(original_name)[0]) or "audio"
            saved_filename = f"{safe_base}_{unique_id}{ext}"
            saved_path = os.path.join(UPLOADS_DIR, saved_filename)
            file.save(saved_path)

            return jsonify(_run_full_analysis(
                saved_path,
                display_name=original_name,
                window_size=window_size,
                overlap=overlap,
                window_type=window_type,
                t_start=t_start,
                t_end=t_end,
            ))

        # Check for JSON payload
        if request.is_json and request.json:
            j = request.json
            window_size = int(j.get("window_size", window_size))
            overlap = float(j.get("overlap", overlap))
            window_type = str(j.get("window_type", window_type))
            t_start = float(j["t_start"]) if j.get("t_start") is not None else None
            t_end = float(j["t_end"]) if j.get("t_end") is not None else None

            target = j.get("sample", "preset")
            audio_path, display_name = _resolve_audio_path(target)
            return jsonify(_run_full_analysis(
                audio_path,
                display_name=display_name,
                window_size=window_size,
                overlap=overlap,
                window_type=window_type,
                t_start=t_start,
                t_end=t_end,
            ))

        # Default fallback sample
        if not os.path.isfile(DEFAULT_OGG):
            return jsonify({"error": "Default audio sample not found. Please upload an audio file."}), 404

        return jsonify(_run_full_analysis(
            DEFAULT_OGG,
            display_name="sperm_whale_clicks.ogg",
            window_size=window_size,
            overlap=overlap,
            window_type=window_type,
            t_start=t_start,
            t_end=t_end,
        ))

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/spectrogram", methods=["POST"])
def api_recompute_spectrogram():
    """
    Fast interactive re-rendering of spectrogram & waveform slice with custom
    STFT parameters (window_size, overlap, window_type) and zoom boundaries (t_start, t_end).
    """
    try:
        data = request.get_json(force=True) or {}
        filename = data.get("filename", "preset")
        audio_path, _ = _resolve_audio_path(filename)

        window_size = int(data.get("window_size", 512))
        overlap = float(data.get("overlap", 0.75))
        window_type = str(data.get("window_type", "hann"))

        t_start = float(data["t_start"]) if data.get("t_start") is not None else None
        t_end = float(data["t_end"]) if data.get("t_end") is not None else None

        samples, sample_rate = load_audio(audio_path)
        waveform_b64, spectrogram_b64 = _render_plots(
            samples,
            sample_rate,
            window_size=window_size,
            overlap=overlap,
            window_type=window_type,
            t_start=t_start,
            t_end=t_end,
        )

        return jsonify({
            "waveform_png": waveform_b64,
            "spectrogram_png": spectrogram_b64,
            "stft_params": {
                "window_size": window_size,
                "overlap": overlap,
                "window_type": window_type,
                "hop_length": int(window_size * (1.0 - overlap)),
                "t_start": t_start,
                "t_end": t_end,
            }
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Compare two audio recordings using their waveforms, spectrograms,
    dominant frequencies, RMS levels, and spectral centroids.
    """
    try:
        path_a, name_a = None, None
        path_b, name_b = None, None

        # Check for uploaded files
        if "audio_a" in request.files and request.files["audio_a"].filename:
            f_a = request.files["audio_a"]
            name_a = f_a.filename
            _, ext_a = os.path.splitext(name_a)
            uid_a = uuid.uuid4().hex[:8]
            safe_a = f"cmp_a_{uid_a}{ext_a}"
            path_a = os.path.join(UPLOADS_DIR, safe_a)
            f_a.save(path_a)

        if "audio_b" in request.files and request.files["audio_b"].filename:
            f_b = request.files["audio_b"]
            name_b = f_b.filename
            _, ext_b = os.path.splitext(name_b)
            uid_b = uuid.uuid4().hex[:8]
            safe_b = f"cmp_b_{uid_b}{ext_b}"
            path_b = os.path.join(UPLOADS_DIR, safe_b)
            f_b.save(path_b)

        # Fallback to defaults or form params if not uploaded
        if not path_a:
            target_a = request.form.get("target_a", "preset")
            path_a, name_a = _resolve_audio_path(target_a)

        if not path_b:
            target_b = request.form.get("target_b", "preset")
            path_b, name_b = _resolve_audio_path(target_b)

        samples_a, sr_a = load_audio(path_a)
        samples_b, sr_b = load_audio(path_b)

        m_a = compute_measurements(samples_a, sr_a)
        m_b = compute_measurements(samples_b, sr_b)

        diff = compare_measurements(m_a, m_b)

        wave_a, spec_a = _render_plots(samples_a, sr_a, window_size=512, overlap=0.75)
        wave_b, spec_b = _render_plots(samples_b, sr_b, window_size=512, overlap=0.75)

        return jsonify({
            "recording_a": {
                "filename": name_a,
                "duration_s": round(float(m_a.get("duration_s", 0)), 2),
                "sample_rate": m_a.get("sample_rate_hz", sr_a),
                "peak_amplitude": round(float(m_a.get("peak_amplitude", 0)), 4),
                "rms_amplitude": round(float(m_a.get("rms_amplitude", 0)), 4),
                "rms_dbfs": round(float(m_a.get("rms_dbfs", 0)), 2),
                "spectral_centroid": int(round(float(m_a.get("spectral_centroid_hz", 0)))),
                "dominant_freq": int(round(float(m_a.get("dominant_freq_hz", 0)))),
                "waveform_png": wave_a,
                "spectrogram_png": spec_a,
            },
            "recording_b": {
                "filename": name_b,
                "duration_s": round(float(m_b.get("duration_s", 0)), 2),
                "sample_rate": m_b.get("sample_rate_hz", sr_b),
                "peak_amplitude": round(float(m_b.get("peak_amplitude", 0)), 4),
                "rms_amplitude": round(float(m_b.get("rms_amplitude", 0)), 4),
                "rms_dbfs": round(float(m_b.get("rms_dbfs", 0)), 2),
                "spectral_centroid": int(round(float(m_b.get("spectral_centroid_hz", 0)))),
                "dominant_freq": int(round(float(m_b.get("dominant_freq_hz", 0)))),
                "waveform_png": wave_b,
                "spectrogram_png": spec_b,
            },
            "comparison": diff,
            "compared_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/export-csv", methods=["GET", "POST"])
def export_csv():
    """Download research report CSV."""
    try:
        filename = request.args.get("filename") or (request.json.get("filename") if request.is_json else "preset")
        audio_path, display_name = _resolve_audio_path(filename)
        analysis_data = _run_full_analysis(audio_path, display_name=display_name)
        csv_content = generate_csv_report(analysis_data)

        stem = os.path.splitext(display_name)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"whale_acoustic_report_{stem}_{timestamp}.csv"

        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/export-pdf", methods=["GET", "POST"])
def export_pdf():
    """Download publication-quality research report PDF."""
    try:
        filename = request.args.get("filename") or (request.json.get("filename") if request.is_json else "preset")
        audio_path, display_name = _resolve_audio_path(filename)

        samples, sample_rate = load_audio(audio_path)
        analysis_data = _run_full_analysis(audio_path, display_name=display_name)

        pdf_bytes = generate_pdf_report(
            analysis_data,
            samples,
            sample_rate,
            stft_params=analysis_data.get("stft_params")
        )

        stem = os.path.splitext(display_name)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"whale_acoustic_report_{stem}_{timestamp}.pdf"

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    print("\n  Whale Acoustic Lab — Computational Analysis of Whale Vocalizations")
    print("  Serving at: http://localhost:5000\n")
    app.run(debug=False, port=5000)
