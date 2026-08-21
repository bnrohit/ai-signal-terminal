import io

import numpy as np
import pandas as pd

from signal_terminal.intelligence import anomaly_events, assess_signal_quality, compare_channels, detect_anomalies, extract_features
from signal_terminal.io import load_csv_multi, synthetic_signal
from signal_terminal.processing import band_power, bandpass_filter, coherence_spectrum, fft_spectrum, highpass_filter, lowpass_filter, notch_filter, spectrogram


def test_dominant_frequency_close_to_primary():
    frame = synthetic_signal(duration_s=6, sample_rate_hz=500, base_frequency_hz=12, secondary_frequency_hz=45, noise_std=0.03, anomaly=False)
    features = extract_features(frame.values, frame.sample_rate_hz)
    assert features.dominant_frequency_hz is not None
    assert abs(features.dominant_frequency_hz - 12) < 0.5
    assert 0 <= features.spectral_entropy <= 1.05


def test_fft_and_spectrogram_shapes():
    frame = synthetic_signal(duration_s=2, sample_rate_hz=200, base_frequency_hz=12, secondary_frequency_hz=45, anomaly=False)
    fft = fft_spectrum(frame.values, frame.sample_rate_hz)
    spec = spectrogram(frame.values, frame.sample_rate_hz)
    assert fft.frequencies_hz.shape == fft.magnitude.shape
    assert spec.power_db.shape == (spec.frequencies_hz.size, spec.times_s.size)


def test_filters_and_band_power():
    frame = synthetic_signal(duration_s=4, sample_rate_hz=500, anomaly=False)
    low = lowpass_filter(frame.values, frame.sample_rate_hz, 80)
    high = highpass_filter(frame.values, frame.sample_rate_hz, 5)
    band = bandpass_filter(frame.values, frame.sample_rate_hz, 5, 80)
    notch = notch_filter(frame.values, frame.sample_rate_hz, 60)
    assert low.shape == high.shape == band.shape == notch.shape == frame.values.shape
    bp = band_power(frame.values, frame.sample_rate_hz, 8, 16)
    assert bp.absolute_power >= 0
    assert 0 <= bp.relative_power <= 1.01


def test_bandpass_rejects_invalid_cutoff():
    frame = synthetic_signal(duration_s=2, sample_rate_hz=200, base_frequency_hz=12, secondary_frequency_hz=45, anomaly=False)
    try:
        bandpass_filter(frame.values, frame.sample_rate_hz, 5, 120)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected cutoff above Nyquist to fail")


def test_anomaly_detector_and_events():
    frame = synthetic_signal(duration_s=5, sample_rate_hz=500, anomaly=True)
    flags, scores = detect_anomalies(frame.values, frame.sample_rate_hz)
    assert flags.shape == frame.values.shape
    assert scores.shape == frame.values.shape
    assert np.isfinite(scores).all()
    events = anomaly_events(flags, scores, frame.sample_rate_hz)
    assert all(event.duration_s > 0 for event in events)


def test_quality_score_bounds():
    frame = synthetic_signal(duration_s=5, sample_rate_hz=500, anomaly=False)
    quality = assess_signal_quality(frame.values, frame.sample_rate_hz)
    assert 0 <= quality.score <= 100
    assert quality.grade in {"excellent", "good", "fair", "poor"}


def test_multichannel_load_and_compare():
    fs = 200.0
    t = np.arange(1000) / fs
    a = np.sin(2 * np.pi * 10 * t)
    b = 0.8 * np.sin(2 * np.pi * 10 * t + 0.1)
    csv = pd.DataFrame({"time": t, "a": a, "b": b}).to_csv(index=False).encode()
    multi = load_csv_multi(io.BytesIO(csv), ["a", "b"], "time", None)
    assert set(multi.channels) == {"a", "b"}
    comparison = compare_channels(multi.channels["a"], multi.channels["b"], multi.sample_rate_hz)
    assert comparison.pearson_correlation > 0.9
    coh = coherence_spectrum(multi.channels["a"], multi.channels["b"], multi.sample_rate_hz)
    assert np.max(coh.magnitude) > 0.8


def test_irregular_timestamps_are_rejected():
    t = np.array([0, 0.01, 0.02, 0.10] + [0.11 + 0.01 * i for i in range(20)], dtype=float)
    y = np.arange(t.size, dtype=float)
    csv = pd.DataFrame({"time": t, "signal": y}).to_csv(index=False).encode()
    try:
        load_csv_multi(io.BytesIO(csv), ["signal"], "time", None)
    except ValueError as exc:
        assert "irregular" in str(exc).lower()
    else:
        raise AssertionError("Expected irregular timestamps to be rejected")
