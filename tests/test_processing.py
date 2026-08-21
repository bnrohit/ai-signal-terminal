import numpy as np

from signal_terminal.intelligence import detect_anomalies, extract_features
from signal_terminal.io import synthetic_signal
from signal_terminal.processing import bandpass_filter, fft_spectrum, spectrogram


def test_dominant_frequency_close_to_primary():
    frame = synthetic_signal(duration_s=6, sample_rate_hz=500, base_frequency_hz=12, secondary_frequency_hz=45, noise_std=0.03, anomaly=False)
    features = extract_features(frame.values, frame.sample_rate_hz)
    assert features.dominant_frequency_hz is not None
    assert abs(features.dominant_frequency_hz - 12) < 0.5


def test_fft_and_spectrogram_shapes():
    frame = synthetic_signal(duration_s=2, sample_rate_hz=200, anomaly=False)
    fft = fft_spectrum(frame.values, frame.sample_rate_hz)
    spec = spectrogram(frame.values, frame.sample_rate_hz)
    assert fft.frequencies_hz.shape == fft.magnitude.shape
    assert spec.power_db.shape == (spec.frequencies_hz.size, spec.times_s.size)


def test_bandpass_rejects_invalid_cutoff():
    frame = synthetic_signal(duration_s=2, sample_rate_hz=200, anomaly=False)
    try:
        bandpass_filter(frame.values, frame.sample_rate_hz, 5, 120)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected cutoff above Nyquist to fail")


def test_anomaly_detector_returns_sample_resolution():
    frame = synthetic_signal(duration_s=5, sample_rate_hz=500, anomaly=True)
    flags, scores = detect_anomalies(frame.values, frame.sample_rate_hz)
    assert flags.shape == frame.values.shape
    assert scores.shape == frame.values.shape
    assert np.isfinite(scores).all()
