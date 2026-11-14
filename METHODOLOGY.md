# Scientific Methodology & Bioacoustics Notes

Whale Acoustic Lab — Computational Analysis of Whale Vocalizations

---

## 1. Acoustic Signal Processing

### Normalization
All input files (`.wav`, `.ogg`, `.mp3`, `.flac`, `.aiff`) are converted to mono floating-point values in the range $[-1.0, 1.0]$:
$$x_{\text{norm}}[n] = \frac{x[n]}{\max_{k} |x[k]|}$$
This eliminates arbitrary bit-depth and recording volume offsets, enabling consistent numerical processing across disparate hydrophone formats.

### Short-Time Fourier Transform (STFT)
The spectrogram decomposes the non-stationary audio signal into discrete time-frequency bins:
$$X[m, k] = \sum_{n=0}^{N-1} x[m \cdot H + n] \cdot w[n] \cdot e^{-j 2\pi k n / N}$$
- **Window Size ($N$)**: Selectable between 256, 512, 1024, and 2048 samples. Smaller sizes provide higher time resolution (ideal for sub-millisecond clicks); larger sizes provide higher frequency resolution.
- **Hop Size ($H$)**: Governed by the overlap fraction $H = \lfloor N(1 - \text{overlap}) \rfloor$.
- **Window Function ($w[n]$)**: Default **Hann** window:
  $$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right)$$
  Smoothly tapers segment boundaries to zero, minimizing spectral leakage artifacts. Alternate Hamming and Blackman windows are also supported.

---

## 2. Event Detection & Coda Analysis

### Adaptive Energy Thresholding
To detect discrete vocalization events in variable marine noise environments, an adaptive envelope threshold is computed across rolling 20 ms frames:
$$\text{Threshold} = P_{30}(\text{RMS}) + 1.2 \cdot \sigma(\text{RMS})$$
Contiguous active segments are extracted, and segments separated by $< 35\text{ ms}$ are merged to form unified vocalization event boundaries.

### Candidate Coda Sequences (Inter-Click Intervals)
Sperm whale clicks are impulsive broadband transients ($< 2\text{ ms}$). Peaks in the high-pass rectified signal are identified with a minimum refractory period of 12 ms:
$$\text{ICI}_{i} = t_{i} - t_{i-1}$$
Clicks separated by $\le 0.22\text{ s}$ are grouped into candidate sequences. Coda statistics include:
- Total click count ($K$)
- Mean Inter-Click Interval: $\mu_{\text{ICI}} = \frac{1}{K-1} \sum \text{ICI}_{i}$
- ICI standard deviation ($\sigma_{\text{ICI}}$)

---

## 3. Recording Comparison Methodology

When evaluating two recordings ($A$ and $B$), numerical deltas are calculated:
$$\Delta\text{Duration} = T_B - T_A$$
$$\Delta\text{RMS}_{\text{dBFS}} = \text{RMS}_{\text{dBFS}, B} - \text{RMS}_{\text{dBFS}, A}$$
$$\Delta\text{Centroid} = f_{c, B} - f_{c, A}$$
$$\Delta\text{Dominant} = f_{\text{dom}, B} - f_{\text{dom}, A}$$
Synchronized dual waveforms and dual spectrograms visually highlight relative acoustic energy distributions and bandwidth discrepancies.

---

## 4. Scientific Calibration Disclaimers

1. **Digital Full Scale ($\text{dBFS}$)**:
   Measurements in this toolkit reflect relative digital audio levels ($0\text{ dBFS} = \text{digital maximum}$). Underwater sound pressure levels ($\text{dB re } 1\,\mu\text{Pa}$) require hydrophone voltage sensitivity curves ($\text{dBV}/\mu\text{Pa}$) and pre-amplifier gain calibration.

2. **Physical Wave Acoustics vs. Semantic Communication**:
   Digital signal processing quantifies physical sound wave dynamics. These metrics do **not** claim to decode whale language, communicative meaning, or identify species without validated acoustic reference models and biological ground truth.

---

## References

- Whitehead, H. (2003). *Sperm Whales: Social Evolution in the Ocean*. University of Chicago Press.
- Rendell, L. E., & Whitehead, H. (2003). Vocal clans in sperm whales. *Proceedings of the Royal Society B*, 270(1512), 225–231.
- Møhl, B., Wahlberg, M., Madsen, P. T., Heerfordt, A., & Lund, A. (2003). The monopulsed nature of sperm whale clicks. *Journal of the Acoustical Society of America*, 114(2), 1143–1154.
- Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall.
