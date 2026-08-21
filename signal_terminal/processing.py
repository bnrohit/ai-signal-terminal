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


@dataclass(frozen=True)
class BandPower:
    low_hz: float
    high_hz: float
    absolute_power: float
    relative_power: float


def _as_signal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if x.size < 8:
        raise ValueError("At least 8 samples are required.")
    if not np.isfinite(x).all():
        raise ValueError("Signal contains non-finite values.")
    return x


def _validate_sample_rate(sample_rate_hz: float) -> float:
    fs = float(sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample rate must be a positive finite number.")
    return fs


def detrend(values: np.ndarray) -> np.ndarray:
    return signal.detrend(_as_signal(values), type="linear")


def _apply_sos(values: np.ndarray, sos: np.ndarray) -> np.ndarray:
    x = _as_signal(values)
    if x.size < max(16, 3 * (2 * sos.shape[0] + 1)):
        raise ValueError("Signal is too short for the selected zero-phase filter.")
    return signal.sosfiltfilt(sos, x)


def bandpass_filter(values: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    fs = _validate_sample_rate(sample_rate_hz)
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Band-pass limits must satisfy 0 < low < high.")
    nyquist = 0.5 * fs
    if high_hz >= nyquist:
        raise ValueError("High cutoff must be below Nyquist frequency.")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return _apply_sos(values, sos)


def lowpass_filter(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    fs = _validate_sample_rate(sample_rate_hz)
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        raise ValueError("Low-pass cutoff must satisfy 0 < cutoff < Nyquist.")
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    return _apply_sos(values, sos)


def highpass_filter(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    fs = _validate_sample_rate(sample_rate_hz)
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        raise ValueError("High-pass cutoff must satisfy 0 < cutoff < Nyquist.")
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return _apply_sos(values, sos)


def notch_filter(values: np.ndarray, sample_rate_hz: float, notch_hz: float, quality_factor: float = 30.0) -> np.ndarray:
    fs = _validate_sample_rate(sample_rate_hz)
    if notch_hz <= 0 or notch_hz >= fs / 2:
        raise ValueError("Notch frequency must satisfy 0 < notch < Nyquist.")
    if quality_factor <= 0:
        raise ValueError("Notch quality factor must be positive.")
    b, a = signal.iirnotch(notch_hz, quality_factor, fs=fs)
    x = _as_signal(values)
    if x.size < 16:
        raise ValueError("Signal is too short for zero-phase notch filtering.")
    return signal.filtfilt(b, a, x)


def fft_spectrum(values: np.ndarray, sample_rate_hz: float) -> Spectrum:
    fs = _validate_sample_rate(sample_rate_hz)
    x = detrend(values)
    window = np.hanning(x.size)
    spec = np.fft.rfft(x * window)
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs)
    scale = max(np.sum(window) / 2.0, 1e-12)
    mag = np.abs(spec) / scale
    return Spectrum(freq, mag)


def welch_psd(values: np.ndarray, sample_rate_hz: float) -> Spectrum:
    fs = _validate_sample_rate(sample_rate_hz)
    x = detrend(values)
    nperseg = min(1024, max(32, x.size // 4))
    nperseg = min(nperseg, x.size)
    freq, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    return Spectrum(freq, psd)


def spectrogram(values: np.ndarray, sample_rate_hz: float) -> SpectrogramData:
    fs = _validate_sample_rate(sample_rate_hz)
    x = detrend(values)
    nperseg = min(256, max(32, x.size // 8))
    nperseg = min(nperseg, x.size)
    noverlap = min(nperseg - 1, int(0.75 * nperseg))
    freq, times, sxx = signal.spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, scaling="density", mode="psd")
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
    ranked = peaks[np.argsort(mag[peaks])[::-1]][: max(1, int(top_k))]
    return [(float(freq[i]), float(mag[i])) for i in ranked]


def band_power(values: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> BandPower:
    fs = _validate_sample_rate(sample_rate_hz)
    nyquist = fs / 2.0
    if low_hz < 0 or high_hz <= low_hz or high_hz > nyquist:
        raise ValueError("Band limits must satisfy 0 <= low < high <= Nyquist.")
    psd = welch_psd(values, fs)
    total = float(np.trapezoid(psd.magnitude, psd.frequencies_hz))
    mask = (psd.frequencies_hz >= low_hz) & (psd.frequencies_hz <= high_hz)
    absolute = float(np.trapezoid(psd.magnitude[mask], psd.frequencies_hz[mask])) if np.count_nonzero(mask) >= 2 else 0.0
    relative = absolute / total if total > 1e-18 else 0.0
    return BandPower(float(low_hz), float(high_hz), absolute, relative)


def coherence_spectrum(first: np.ndarray, second: np.ndarray, sample_rate_hz: float) -> Spectrum:
    fs = _validate_sample_rate(sample_rate_hz)
    x = _as_signal(first)
    y = _as_signal(second)
    n = min(x.size, y.size)
    if n < 32:
        raise ValueError("At least 32 aligned samples are required for coherence analysis.")
    x = x[:n]
    y = y[:n]
    nperseg = min(1024, max(32, n // 4), n)
    freq, coh = signal.coherence(x, y, fs=fs, nperseg=nperseg)
    return Spectrum(freq, coh)
