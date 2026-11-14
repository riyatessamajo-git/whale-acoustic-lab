# Whale Acoustic Lab
### Computational Analysis of Whale Vocalizations

A professional, lightweight bioacoustic research toolkit for analyzing marine mammal vocalizations using digital signal processing (DSP) in Python and an interactive web interface.

---

## Overview

**Whale Acoustic Lab** processes underwater hydrophone recordings to compute time-domain waveforms, Short-Time Fourier Transform (STFT) spectrograms, acoustic loudness metrics, discrete vocalization events, and sperm whale candidate click sequences (codas).

Unlike heavy, desktop-only workstations (such as Raven Pro or PAMGuard), this application runs natively in any desktop or mobile browser with zero client installation, providing 1-click automated extraction and publication-quality research report exports (PDF and CSV).

---

## Core Capabilities

1. **Audio Ingestion & Normalization**:
   - Supports `WAV`, `OGG`, `MP3`, `FLAC`, and `AIFF`.
   - Mixes multi-channel hydrophone data to mono.
   - Normalizes amplitude to a floating-point range of `[-1.0, 1.0]`.

2. **Interactive Spectrogram & STFT Customization**:
   - Configurable FFT window sizes: `256`, `512` (default), `1024`, and `2048` samples.
   - Configurable overlap: `50%`, `75%` (default), and `87.5%`.
   - Windowing functions: `Hann`, `Hamming`, and `Blackman`.
   - Time-window zooming (`t_start` to `t_end`).
   - Real-time cursor inspection for instantaneous time ($s$) and frequency ($\text{Hz}$) readouts.

3. **Vocalization Event Detection**:
   - Signal-based adaptive energy envelope thresholding.
   - Extracts event start/end times, temporal duration (ms), peak amplitude, and signal-to-noise ratio (SNR).
   - 1-click inspection: instantly zooms the spectrogram and cues audio playback to any detected event.

4. **Sperm Whale Coda Analysis**:
   - Impulse peak detection to identify individual echolocation clicks.
   - Calculates Inter-Click Intervals (ICIs in ms), mean interval, and click counts.
   - Visual click timeline mapping pulses across the recording.
   - Clearly labeled as candidate sequences requiring behavioral field validation.

5. **Recording Comparison**:
   - Side-by-side comparative analysis of two audio recordings.
   - Comparative waveform and spectrogram displays.
   - Automated delta metrics table ($\Delta\text{ Duration}$, $\Delta\text{ RMS dBFS}$, $\Delta\text{ Centroid}$, $\Delta\text{ Dominant Frequency}$).

6. **Research Reports**:
   - **Vector PDF Export (`/api/export-pdf`)**: Multi-page publication-quality document containing recording metadata, measurement tables, STFT figures, event tables, and calibration disclaimers.
   - **CSV Export (`/api/export-csv`)**: Machine-readable data for external analysis in R, Python, or Excel.
   - **TXT Summary (`.txt`)**: Quick terminal/text report.

7. **Scientific Calibration & Accuracy**:
   - Explicitly distinguishes relative digital levels ($\text{dBFS}$) from calibrated underwater sound pressure levels ($\text{dB re } 1\,\mu\text{Pa}$).
   - Disclaims semantic/linguistic interpretation, restricting claims to physical wave acoustics.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web Server
```bash
python app.py
```
Open **`http://localhost:5000`** in any web browser (accessible on phones and tablets on the local network).

### 3. (Optional) Run via Command-Line
```bash
# Analyze default sample and save outputs
python whale_analysis.py

# Analyze your own recording
python whale_analysis.py path/to/recording.wav
```

### 4. Run Automated Tests
```bash
pytest
```
*(All 34 unit tests cover audio loading, math accuracy, event detection, coda analysis, comparative metrics, and plot generation).*

---

## Project Structure

```
acoustic/
├── app.py                  # Flask backend server & REST API endpoints
├── audio_io.py             # Audio loader, mono mixdown, [-1, 1] normalizer
├── measurements.py         # Acoustic metrics, event detector, coda analyzer
├── plotting.py             # Waveform and configurable STFT spectrogram renderer
├── report_generator.py     # PDF & CSV scientific report generator
├── download_sample.py      # Sample audio downloader (Wikimedia / NOAA)
├── whale_analysis.py       # Standalone CLI analysis script
├── requirements.txt        # Minimal Python dependencies
├── .gitignore              # Version control ignore rules
├── static/
│   └── index.html          # Web application UI (white-and-blue theme)
├── data/
│   └── sperm_whale_clicks.ogg  # Built-in reference recording (71.2s, 44.1 kHz)
├── uploads/                # Temporary directory for user uploads (.gitkeep)
├── outputs/                # Directory for generated CLI outputs (.gitkeep)
└── tests/                  # Automated test suite (34 tests)
    ├── test_audio_io.py
    ├── test_measurements.py
    ├── test_new_features.py
    └── test_plotting.py
```

---

## Acoustic Measurements Explained

| Measurement | Formula / Concept | Scientific Interpretation |
|---|---|---|
| **Duration** | $N / f_s$ | Total length of the audio recording in seconds. |
| **Sample Rate ($f_s$)** | Samples per second ($\text{Hz}$) | Determines Nyquist bandwidth ($f_{max} = f_s / 2$). E.g., 44.1 kHz captures up to 22.05 kHz. |
| **Peak Amplitude** | $\max(\|x[n]\|)$ | Loudest single instantaneous pressure sample ($1.0$ in normalized audio). |
| **RMS Amplitude** | $\sqrt{\frac{1}{N} \sum x[n]^2}$ | Quadratic mean representing effective continuous acoustic power. |
| **RMS Level (dBFS)** | $20 \log_{10}(\text{RMS})$ | Logarithmic loudness relative to digital clipping ($0\text{ dBFS}$). |
| **Spectral Centroid** | $\frac{\sum f \cdot \|X[f]\|}{\sum \|X[f]\|}$ | Center of spectral mass ("acoustic brightness"). Click energy is concentrated above 8–10 kHz. |
| **Dominant Frequency** | $\operatorname{argmax}(\|X[f]\|)$ | Frequency bin with peak energy. |
| **Inter-Click Interval (ICI)** | $t_{i} - t_{i-1}$ | Time gap between consecutive sperm whale clicks (used in biosonar echolocation and coda identification). |

---

## Scientific Calibration Notice

* **Relative Digital Scale ($\text{dBFS}$)**: All measurements are calculated on digital audio data. Converting digital dBFS to absolute underwater sound pressure levels ($\text{dB re } 1\,\mu\text{Pa}$) requires physical hydrophone voltage sensitivity ($\text{dBV}/\mu\text{Pa}$) and pre-amplifier gain curves.
* **Physical Signal Mechanics**: Signal metrics, event boundaries, and candidate click sequences quantify physical wave acoustics. They do **not** claim to decode semantic whale language or assert species identity without ground-truth biological annotations and validated reference models.

---

## License & Attribution

* **Default Recording**: *Physeter macrocephalus* (Sperm Whale) Normal Echolocation Clicks recorded via hydrophone. Source: Wikimedia Commons / NOAA (Public Domain).
* **Technology**: Python 3.12, NumPy, SciPy, Matplotlib, SoundFile, Flask.
