"""
audio_io.py
===========
Audio loading utilities for the whale analysis project.

Supports WAV (natively via SciPy) and OGG (via SoundFile).
Returns audio as a normalized float32 mono NumPy array.

Design decision: We normalize to float32 in [-1, 1] so that all downstream
measurements work on a consistent scale regardless of the original bit depth.
"""

import os
import numpy as np


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (samples, sample_rate).

    The returned samples array is:
      - float32 dtype
      - normalized to the range [-1.0, 1.0]
      - mono (stereo files are mixed down by averaging channels)

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the audio file.
        Supported formats: WAV, OGG, FLAC (anything SoundFile handles).

    Returns
    -------
    samples : np.ndarray, shape (n_samples,)
    sample_rate : int

    Raises
    ------
    FileNotFoundError  : if the file does not exist.
    ValueError         : if the file is empty or silent.
    RuntimeError       : if the file format is unsupported or corrupted.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".wav":
            samples, sr = _load_wav(filepath)
        else:
            # SoundFile handles OGG, FLAC, AIFF, etc.
            samples, sr = _load_soundfile(filepath)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read '{filepath}'. "
            f"Is the file corrupted or unsupported? Details: {exc}"
        ) from exc

    # Convert to float32
    samples = samples.astype(np.float32)

    # Mix stereo/multi-channel down to mono
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    # Normalize to [-1.0, 1.0]
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val
    else:
        raise ValueError(
            "The audio file appears to be completely silent (all samples are zero)."
        )

    return samples, int(sr)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_wav(filepath: str):
    """Load a WAV file using SciPy (no extra dependency for WAV)."""
    from scipy.io import wavfile
    sr, data = wavfile.read(filepath)
    return data, sr


def _load_soundfile(filepath: str):
    """Load OGG/FLAC/etc. using SoundFile."""
    import soundfile as sf
    data, sr = sf.read(filepath, dtype="float32", always_2d=False)
    return data, sr
