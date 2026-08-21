from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from .processing import dominant_frequencies


@dataclass(frozen=True)
class SignalFeatures:
    mean: float
    rms: float
    std: float
    peak_to_peak: float
    crest_factor: float
    skewness: float
    kurtosis: float
    dominant_frequency_hz: float | None
    spectral_centroid_hz: float | None


def extract_features(values: np.ndarray, sample_rate_hz: float) -> SignalFeatures:
    x = np.asarray(values, dtype=float)
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    dominant = dominant_frequencies(x, sample_rate_hz, top_k=1)

    centered = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate_hz)
    total = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / total) if total > 0 else None

    return SignalFeatures(
        mean=float(np.mean(x)),
        rms=rms,
        std=float(np.std(x)),
        peak_to_peak=float(np.ptp(x)),
        crest_factor=(peak / rms) if rms > 1e-12 else 0.0,
        skewness=float(stats.skew(x, bias=False)) if x.size >= 8 else 0.0,
        kurtosis=float(stats.kurtosis(x, fisher=True, bias=False)) if x.size >= 8 else 0.0,
        dominant_frequency_hz=dominant[0][0] if dominant else None,
        spectral_centroid_hz=centroid,
    )


def detect_anomalies(
    values: np.ndarray,
    sample_rate_hz: float,
    contamination: float = 0.02,
    window_ms: float = 100.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed Isolation Forest anomaly scoring returned at sample resolution."""
    x = np.asarray(values, dtype=float)
    window = max(8, int((window_ms / 1000.0) * sample_rate_hz))
    if x.size < window * 4:
        window = max(8, x.size // 4)
    starts = np.arange(0, x.size - window + 1, window)
    if starts.size < 4:
        return np.zeros(x.size, dtype=bool), np.zeros(x.size, dtype=float)

    feats = []
    for start in starts:
        segment = x[start : start + window]
        feats.append([
            np.mean(segment),
            np.std(segment),
            np.sqrt(np.mean(segment**2)),
            np.max(segment) - np.min(segment),
            stats.kurtosis(segment, fisher=True, bias=False) if segment.size >= 8 else 0.0,
        ])
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


def interpret_signal(features: SignalFeatures, anomaly_fraction: float) -> list[dict[str, str]]:
    """Transparent rule-based interpretation layer; no hidden external model is required."""
    insights: list[dict[str, str]] = []

    if features.dominant_frequency_hz is not None:
        insights.append({
            "severity": "info",
            "title": "Dominant periodic component",
            "message": f"The strongest detected periodic component is approximately {features.dominant_frequency_hz:.2f} Hz.",
        })
    if features.crest_factor >= 4.0:
        insights.append({
            "severity": "warning",
            "title": "Impulsive behavior",
            "message": f"Crest factor is {features.crest_factor:.2f}, which can indicate short high-amplitude transients relative to RMS energy.",
        })
    if abs(features.kurtosis) >= 3.0:
        insights.append({
            "severity": "warning",
            "title": "Non-Gaussian amplitude distribution",
            "message": f"Excess kurtosis is {features.kurtosis:.2f}; inspect transient events and acquisition artifacts.",
        })
    if anomaly_fraction >= 0.05:
        insights.append({
            "severity": "warning",
            "title": "Elevated anomaly occupancy",
            "message": f"Approximately {100 * anomaly_fraction:.1f}% of samples fall inside windows flagged by the anomaly model.",
        })
    elif anomaly_fraction > 0:
        insights.append({
            "severity": "info",
            "title": "Localized anomalies detected",
            "message": f"Anomalous windows cover about {100 * anomaly_fraction:.1f}% of samples; review the highlighted regions.",
        })
    else:
        insights.append({
            "severity": "info",
            "title": "No strong anomaly windows",
            "message": "The bounded anomaly detector did not flag windows at the current sensitivity setting.",
        })

    insights.append({
        "severity": "note",
        "title": "Interpretation boundary",
        "message": "These outputs are engineering decision support, not a diagnosis or safety certification. Validate against domain-specific limits and calibrated instrumentation.",
    })
    return insights
