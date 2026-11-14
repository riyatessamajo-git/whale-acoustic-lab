# Whale Acoustic Lab
### Computational Analysis of Whale Vocalizations

A Python-based bioacoustic signal-processing toolkit for exploring and analysing marine mammal vocalizations using digital signal processing (DSP) and an interactive web interface.

## Overview

Whale Acoustic Lab is an experimental platform designed to process underwater audio recordings and extract acoustic features for preliminary analysis.

The application provides waveform visualization, Short-Time Fourier Transform (STFT) spectrograms, signal-level measurements, vocalization event detection, and candidate sperm whale click-sequence analysis.

It combines Python-based audio processing with a Flask web interface, allowing users to upload recordings, inspect visualizations, compare audio signals, and export analysis reports.

**Note:** This is an exploratory signal-processing project. Its outputs are not validated biological classifications or interpretations of whale communication.

## Core Capabilities

### 1. Audio Ingestion and Processing
- Supports WAV, OGG, MP3, FLAC, and AIFF formats.
- Converts multichannel recordings to mono for analysis.
- Converts audio samples to floating-point representation and normalizes amplitude where applicable.

### 2. Waveform and Spectrogram Visualization
- Generates time-domain waveform plots.
- Produces STFT-based spectrograms.
- Configurable FFT sizes: 256, 512, 1024, and 2048 samples.
- Configurable overlap: 50%, 75%, and 87.5%.
- Window functions: Hann, Hamming, and Blackman.
- Supports time-window selection and cursor-based inspection of time and frequency.

### 3. Vocalization Event Detection
- Uses an adaptive energy-envelope thresholding approach.
- Extracts candidate event start and end times.
- Calculates event duration and peak amplitude.
- Provides signal-to-noise ratio estimates.
- Allows users to inspect detected events through the interface.

### 4. Candidate Sperm Whale Click-Sequence Analysis
- Detects candidate impulse peaks in audio recordings.
- Calculates inter-click intervals (ICIs).
- Reports click counts and mean inter-click intervals.
- Provides a visual timeline of detected candidate clicks.

These outputs represent signal-based detections only. Confirming sperm whale codas or their behavioural significance requires validated recordings, biological annotations, and expert interpretation.

### 5. Recording Comparison
- Compares two audio recordings.
- Displays comparative waveform and spectrogram visualizations.
- Reports differences in selected measurements, including duration, RMS level (dBFS), spectral centroid, and dominant frequency.

### 6. Research Report Exports
- PDF reports containing recording information, measurements, visualizations, and analysis results.
- CSV exports for further analysis using Python, R, or spreadsheet software.
- Text-based summaries for quick reference.

### 7. Scientific Calibration Considerations
- Reports digital signal levels in dBFS rather than calibrated underwater sound pressure levels.
- Distinguishes signal measurements from biological interpretation.
- Does not claim to decode whale language or independently identify species.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web framework | Flask |
| Numerical processing | NumPy |
| Signal processing | SciPy |
| Visualization | Matplotlib |
| Audio handling | SoundFile |
| Testing | Pytest |
| Frontend | HTML, CSS, JavaScript |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/riyatessamajo-git/whale-acoustic-lab.git
cd whale-acoustic-lab
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Application

```bash
python app.py
```

Open the following address in your browser:

```text
http://localhost:5000
```

### 4. Run Command-Line Analysis

```bash
python whale_analysis.py
```

To analyse a specific recording:

```bash
python whale_analysis.py path/to/recording.wav
```

### 5. Run Automated Tests

```bash
pytest
```

The project includes automated tests for audio loading, measurement calculations, event detection, click analysis, comparative metrics, and plotting.

## Project Structure

```text
acoustic/
├── app.py
├── audio_io.py
├── measurements.py
├── plotting.py
├── report_generator.py
├── download_sample.py
├── whale_analysis.py
├── requirements.txt
├── METHODOLOGY.md
├── README.md
├── static/
│   └── index.html
├── data/
├── uploads/
├── outputs/
└── tests/
    ├── __init__.py
    ├── test_audio_io.py
    ├── test_measurements.py
    ├── test_new_features.py
    └── test_plotting.py
```

## Acoustic Measurements Explained

| Measurement | Formula / Concept | Interpretation |
|---|---|---|
| Duration | \(N/f_s\) | Total recording duration in seconds. |
| Sample Rate | Samples per second (Hz) | Determines the Nyquist frequency, \(f_s/2\). |
| Peak Amplitude | \(\max |x[n]|\) | Maximum absolute digital sample amplitude. |
| RMS Amplitude | \(\sqrt{\frac{1}{N}\sum x[n]^2}\) | Effective amplitude measure of the digital signal. |
| RMS Level | \(20\log_{10}(\mathrm{RMS})\) | Digital signal level relative to full scale, expressed in dBFS. |
| Spectral Centroid | \(\frac{\sum f_k|X[k]|}{\sum |X[k]|}\) | Magnitude-weighted centre of the frequency spectrum. |
| Dominant Frequency | Frequency at maximum spectral magnitude | Frequency bin with the highest spectral magnitude. |
| Inter-Click Interval | \(t_i-t_{i-1}\) | Time difference between consecutive detected candidate clicks. |

## Scientific Calibration Notice

### Digital Levels and Underwater Sound Pressure

The application calculates measurements from digital audio samples. RMS levels are reported in dBFS, which represents signal amplitude relative to digital full scale.

These values cannot be directly interpreted as absolute underwater sound pressure levels in dB re 1 µPa.

Conversion requires appropriate calibration information, including hydrophone sensitivity, recording-system gain, and relevant frequency-response characteristics.

### Interpretation Limitations

Signal measurements, detected events, and candidate click sequences describe properties of the recorded audio.

They do not independently establish:
- The biological identity of a vocalization.
- The behavioural context of a detected sound.
- The meaning or semantic content of whale vocalizations.
- The classification of a detected sequence as a confirmed sperm whale coda.

Such interpretations require suitable reference data, biological annotations, and expert validation.

## License and Attribution

This project uses open-source Python libraries under their respective licences.

Any sample audio recordings used with the project should retain their original source attribution and applicable licence information. Check the specific recording's licence before redistribution.

## Disclaimer

Whale Acoustic Lab is an educational and exploratory bioacoustic signal-processing project. It is intended to demonstrate audio processing, acoustic feature extraction, visualization, and preliminary signal analysis. Its outputs should not be treated as validated scientific findings without further verification.