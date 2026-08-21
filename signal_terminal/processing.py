from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Spectrum:
    frequencies_hz: np.ndarray
    magnitude: np.ndarray


@dataclass(frozen=True)
class SpectrogramData:
    frequencies_hz: np.ndarray
    times_s: np.ndarray
    power_db: np.ndarray


def detrend(values: np.ndarray) -> np.ndarray:
    return signal.detrend(np.asarray(values, dtype=float), type="linear")


def bandpass_filter(
    values: np.ndarray,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Band-pass limits must satisfy 0 < low < high.")
    nyquist = 0.5 * sample_rate_hz
    if high_hz >= nyquist:
        raise ValueError("High cutoff must be below Nyquist frequency.")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=float))


def fft_spectrum(values: np.ndarray, sample_rate_hz: float) -> Spectrum:
    x = detrend(values)
    window = np.hanning(x.size)
    spec = np.fft.rfft(x * window)
    freq = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    scale = max(np.sum(window) / 2.0, 1e-12)
    mag = np.abs(spec) / scale
    return Spectrum(freq, mag)


def welch_psd(values: np.ndarray, sample_rate_hz: float) -> Spectrum:
    x = detrend(values)
    nperseg = min(1024, max(32, x.size // 4))
    freq, psd = signal.welch(x, fs=sample_rate_hz, nperseg=nperseg)
    return Spectrum(freq, psd)


def spectrogram(values: np.ndarray, sample_rate_hz: float) -> SpectrogramData:
    x = detrend(values)
    nperseg = min(256, max(32, x.size // 8))
    noverlap = int(0.75 * nperseg)
    freq, times, sxx = signal.spectrogram(
        x,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    db = 10.0 * np.log10(np.maximum(sxx, 1e-18))
    return SpectrogramData(freq, times, db)


def dominant_frequencies(values: np.ndarray, sample_rate_hz: float, top_k: int = 5) -> list[tuple[float, float]]:
    spec = fft_spectrum(values, sample_rate_hz)
    if spec.frequencies_hz.size <= 1:
        return []
    freq = spec.frequencies_hz[1:]
    mag = spec.magnitude[1:]
    peaks, _ = signal.find_peaks(mag)
    if peaks.size == 0:
        peaks = np.arange(mag.size)
    ranked = peaks[np.argsort(mag[peaks])[::-1]][:top_k]
    return [(float(freq[i]), float(mag[i])) for i in ranked]
