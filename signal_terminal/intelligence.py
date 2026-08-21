from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from .processing import dominant_frequencies, welch_psd


@dataclass(frozen=True)
class SignalFeatures:
    mean: float
    median: float
    rms: float
    std: float
    peak_to_peak: float
    crest_factor: float
    skewness: float
    kurtosis: float
    zero_crossing_rate: float
    spectral_entropy: float
    dominant_frequency_hz: float | None
    spectral_centroid_hz: float | None


@dataclass(frozen=True)
class SignalQuality:
    score: float
    grade: str
    clipping_fraction: float
    drift_fraction: float
    high_frequency_fraction: float


@dataclass(frozen=True)
class AnomalyEvent:
    start_s: float
    end_s: float
    duration_s: float
    peak_score: float
    sample_count: int


@dataclass(frozen=True)
class ChannelComparison:
    pearson_correlation: float
    lag_samples: int
    lag_seconds: float


def extract_features(values: np.ndarray, sample_rate_hz: float) -> SignalFeatures:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 8 or not np.isfinite(x).all():
        raise ValueError("Feature extraction requires at least 8 finite one-dimensional samples.")
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    dominant = dominant_frequencies(x, sample_rate_hz, top_k=1)
    centered = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sample_rate_hz))
    total = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / total) if total > 0 else None
    psd = np.square(spectrum)
    psd_sum = float(np.sum(psd))
    if psd_sum > 0:
        p = psd / psd_sum
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)) / np.log2(max(2, psd.size)))
    else:
        entropy = 0.0
    signs = np.signbit(centered)
    zero_crossings = float(np.count_nonzero(signs[1:] != signs[:-1]) / max(1, x.size - 1))
    return SignalFeatures(mean=float(np.mean(x)), median=float(np.median(x)), rms=rms, std=float(np.std(x)), peak_to_peak=float(np.ptp(x)), crest_factor=(peak / rms) if rms > 1e-12 else 0.0, skewness=float(stats.skew(x, bias=False)) if x.size >= 8 else 0.0, kurtosis=float(stats.kurtosis(x, fisher=True, bias=False)) if x.size >= 8 else 0.0, zero_crossing_rate=zero_crossings, spectral_entropy=entropy, dominant_frequency_hz=dominant[0][0] if dominant else None, spectral_centroid_hz=centroid)


def detect_anomalies(values: np.ndarray, sample_rate_hz: float, contamination: float = 0.02, window_ms: float = 100.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 8 or not np.isfinite(x).all():
        raise ValueError("Anomaly detection requires finite one-dimensional samples.")
    window = max(8, int((window_ms / 1000.0) * float(sample_rate_hz)))
    if x.size < window * 4:
        window = max(8, x.size // 4)
    starts = np.arange(0, x.size - window + 1, window)
    if starts.size < 4:
        return np.zeros(x.size, dtype=bool), np.zeros(x.size, dtype=float)
    feats = []
    for start in starts:
        segment = x[start : start + window]
        feats.append([np.mean(segment), np.std(segment), np.sqrt(np.mean(segment**2)), np.max(segment) - np.min(segment), stats.kurtosis(segment, fisher=True, bias=False) if segment.size >= 8 else 0.0])
    feats_arr = np.nan_to_num(np.asarray(feats, dtype=float))
    contamination = float(np.clip(contamination, 0.001, 0.25))
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    labels = model.fit_predict(feats_arr)
    scores = -model.score_samples(feats_arr)
    sample_flags = np.zeros(x.size, dtype=bool)
    sample_scores = np.zeros(x.size, dtype=float)
    for start, label, score in zip(starts, labels, scores):
        sample_flags[start : start + window] = label == -1
        sample_scores[start : start + window] = float(score)
    return sample_flags, sample_scores


def anomaly_events(flags: np.ndarray, scores: np.ndarray, sample_rate_hz: float) -> list[AnomalyEvent]:
    mask = np.asarray(flags, dtype=bool)
    score_arr = np.asarray(scores, dtype=float)
    if mask.shape != score_arr.shape:
        raise ValueError("Anomaly flags and scores must have the same shape.")
    if mask.ndim != 1:
        raise ValueError("Anomaly flags must be one-dimensional.")
    if not mask.any():
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    fs = float(sample_rate_hz)
    events = []
    for start, end in zip(starts, ends):
        segment_scores = score_arr[start:end]
        events.append(AnomalyEvent(start_s=float(start / fs), end_s=float(end / fs), duration_s=float((end - start) / fs), peak_score=float(np.max(segment_scores)) if segment_scores.size else 0.0, sample_count=int(end - start)))
    return events


def assess_signal_quality(values: np.ndarray, sample_rate_hz: float) -> SignalQuality:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 32 or not np.isfinite(x).all():
        raise ValueError("Signal quality assessment requires at least 32 finite samples.")
    span = float(np.ptp(x))
    if span <= 1e-12:
        return SignalQuality(20.0, "poor", 1.0, 0.0, 0.0)
    tol = max(span * 1e-3, 1e-12)
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    clipping = float(np.mean((x <= min_v + tol) | (x >= max_v - tol)))
    psd = welch_psd(x, sample_rate_hz)
    total = float(np.trapezoid(psd.magnitude, psd.frequencies_hz))
    nyquist = float(sample_rate_hz) / 2.0
    if total > 1e-18:
        drift_mask = psd.frequencies_hz <= max(0.5, nyquist * 0.02)
        high_mask = psd.frequencies_hz >= nyquist * 0.80
        drift = float(np.trapezoid(psd.magnitude[drift_mask], psd.frequencies_hz[drift_mask]) / total) if np.count_nonzero(drift_mask) >= 2 else 0.0
        high = float(np.trapezoid(psd.magnitude[high_mask], psd.frequencies_hz[high_mask]) / total) if np.count_nonzero(high_mask) >= 2 else 0.0
    else:
        drift = high = 0.0
    penalty = min(45.0, clipping * 500.0) + min(25.0, drift * 45.0) + min(20.0, high * 50.0)
    score = float(np.clip(100.0 - penalty, 0.0, 100.0))
    grade = "excellent" if score >= 90 else "good" if score >= 75 else "fair" if score >= 55 else "poor"
    return SignalQuality(score, grade, clipping, drift, high)


def compare_channels(first: np.ndarray, second: np.ndarray, sample_rate_hz: float) -> ChannelComparison:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    n = min(x.size, y.size)
    if n < 16:
        raise ValueError("At least 16 aligned samples are required for channel comparison.")
    x = x[:n]
    y = y[:n]
    corr = 0.0 if np.std(x) <= 1e-12 or np.std(y) <= 1e-12 else float(np.corrcoef(x, y)[0, 1])
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    cross = np.correlate(xc, yc, mode="full")
    lag = int(np.argmax(np.abs(cross)) - (n - 1))
    return ChannelComparison(corr, lag, float(lag / float(sample_rate_hz)))


def interpretation_payload(features: SignalFeatures, quality: SignalQuality, events: list[AnomalyEvent]) -> dict:
    return {"features": asdict(features), "quality": asdict(quality), "anomaly_events": [asdict(event) for event in events]}


def interpret_signal(features: SignalFeatures, anomaly_fraction: float, quality: SignalQuality | None = None) -> list[dict[str, str]]:
    insights = []
    if features.dominant_frequency_hz is not None:
        insights.append({"severity": "info", "title": "Dominant periodic component", "message": f"The strongest detected periodic component is approximately {features.dominant_frequency_hz:.2f} Hz."})
    if features.crest_factor >= 4.0:
        insights.append({"severity": "warning", "title": "Impulsive behavior", "message": f"Crest factor is {features.crest_factor:.2f}, which can indicate short high-amplitude transients relative to RMS energy."})
    if abs(features.kurtosis) >= 3.0:
        insights.append({"severity": "warning", "title": "Non-Gaussian amplitude distribution", "message": f"Excess kurtosis is {features.kurtosis:.2f}; inspect transient events and acquisition artifacts."})
    if features.spectral_entropy >= 0.85:
        insights.append({"severity": "info", "title": "Broad spectral distribution", "message": f"Normalized spectral entropy is {features.spectral_entropy:.2f}, indicating energy is relatively spread across frequencies."})
    if anomaly_fraction >= 0.05:
        insights.append({"severity": "warning", "title": "Elevated anomaly occupancy", "message": f"Approximately {100 * anomaly_fraction:.1f}% of samples fall inside windows flagged by the anomaly model."})
    elif anomaly_fraction > 0:
        insights.append({"severity": "info", "title": "Localized anomalies detected", "message": f"Anomalous windows cover about {100 * anomaly_fraction:.1f}% of samples; review the highlighted regions."})
    else:
        insights.append({"severity": "info", "title": "No strong anomaly windows", "message": "The bounded anomaly detector did not flag windows at the current sensitivity setting."})
    if quality is not None:
        level = "warning" if quality.score < 55 else "info"
        insights.append({"severity": level, "title": "Signal quality heuristic", "message": f"Quality score is {quality.score:.0f}/100 ({quality.grade}). This is a transparent acquisition-quality heuristic, not a calibration result."})
    insights.append({"severity": "note", "title": "Interpretation boundary", "message": "These outputs are engineering decision support, not a diagnosis or safety certification. Validate against domain-specific limits and calibrated instrumentation."})
    return insights
