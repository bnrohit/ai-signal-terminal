from __future__ import annotations

import io
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from signal_terminal import __version__
from signal_terminal.intelligence import anomaly_events, assess_signal_quality, compare_channels, detect_anomalies, extract_features, interpretation_payload, interpret_signal
from signal_terminal.io import SignalFrame, load_csv_multi, synthetic_signal
from signal_terminal.processing import band_power, bandpass_filter, coherence_spectrum, dominant_frequencies, fft_spectrum, highpass_filter, lowpass_filter, notch_filter, spectrogram, welch_psd

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="AI Signal Terminal", page_icon="📡", layout="wide")
st.title("📡 AI Signal Terminal")
st.caption(f"v{__version__} · Interactive signal processing + transparent AI-assisted interpretation")

with st.sidebar:
    st.header("Signal source")
    mode = st.radio("Input", ["Synthetic demo", "Upload CSV"], index=0)
    secondary_values: np.ndarray | None = None
    secondary_name: str | None = None

    if mode == "Synthetic demo":
        sample_rate = st.number_input("Sample rate (Hz)", min_value=20.0, max_value=100000.0, value=500.0)
        duration = st.number_input("Duration (s)", min_value=1.0, max_value=120.0, value=10.0)
        nyquist = sample_rate / 2.0
        f1 = st.number_input("Primary frequency (Hz)", min_value=0.1, max_value=max(0.2, nyquist * 0.99), value=min(12.0, nyquist * 0.4))
        f2 = st.number_input("Secondary frequency (Hz)", min_value=0.1, max_value=max(0.2, nyquist * 0.99), value=min(45.0, nyquist * 0.7))
        include_anomaly = st.checkbox("Inject demo transient", value=True)
        try:
            frame = synthetic_signal(duration, sample_rate, f1, f2, anomaly=include_anomaly)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        primary_name = "synthetic"
    else:
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to begin.")
            st.stop()
        raw = uploaded.getvalue()
        if len(raw) > MAX_UPLOAD_BYTES:
            st.error("CSV is larger than the 10 MB interactive-analysis limit.")
            st.stop()
        try:
            preview = pd.read_csv(io.BytesIO(raw), nrows=50)
        except Exception as exc:
            st.error(f"Unable to parse CSV: {exc}")
            st.stop()
        numeric_cols = [c for c in preview.columns if pd.api.types.is_numeric_dtype(preview[c])]
        if not numeric_cols:
            st.error("No numeric columns were detected in the CSV preview.")
            st.stop()
        selected = st.multiselect("Signal columns", numeric_cols, default=numeric_cols[:1], max_selections=8)
        if not selected:
            st.info("Select at least one signal column.")
            st.stop()
        primary_name = st.selectbox("Primary analysis channel", selected)
        time_options = ["<none>"] + [c for c in numeric_cols if c not in selected]
        time_col = st.selectbox("Time column (seconds)", time_options)
        sr = None
        if time_col == "<none>":
            sr = st.number_input("Sample rate (Hz)", min_value=0.001, value=1000.0)
        try:
            multi = load_csv_multi(io.BytesIO(raw), selected, None if time_col == "<none>" else time_col, sr)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        frame = SignalFrame(multi.time, multi.channels[primary_name], multi.sample_rate_hz, "csv")
        comparison_options = [c for c in selected if c != primary_name]
        if comparison_options:
            secondary_name = st.selectbox("Compare with channel", ["<none>"] + comparison_options)
            if secondary_name != "<none>":
                secondary_values = multi.channels[secondary_name]
            else:
                secondary_name = None

    st.divider()
    st.header("Processing")
    filter_mode = st.selectbox("Filter", ["None", "Band-pass", "Low-pass", "High-pass", "50 Hz notch", "60 Hz notch", "Custom notch"])
    anomaly_sensitivity = st.slider("Anomaly sensitivity", 0.005, 0.10, 0.02, 0.005)
    anomaly_window_ms = st.slider("Anomaly window (ms)", 20, 1000, 100, 20)

values = frame.values.copy()
nyquist = frame.sample_rate_hz / 2.0
try:
    if filter_mode == "Band-pass":
        col1, col2 = st.sidebar.columns(2)
        low_default = max(0.1, nyquist * 0.02)
        high_default = max(low_default * 2, nyquist * 0.8)
        low = col1.number_input("Low Hz", min_value=0.001, value=min(low_default, nyquist * 0.3))
        high = col2.number_input("High Hz", min_value=low + 0.001, value=min(high_default, nyquist * 0.95))
        values = bandpass_filter(values, frame.sample_rate_hz, low, high)
    elif filter_mode == "Low-pass":
        cutoff = st.sidebar.number_input("Cutoff Hz", min_value=0.001, value=max(0.01, nyquist * 0.4))
        values = lowpass_filter(values, frame.sample_rate_hz, cutoff)
    elif filter_mode == "High-pass":
        cutoff = st.sidebar.number_input("Cutoff Hz", min_value=0.001, value=max(0.01, nyquist * 0.02))
        values = highpass_filter(values, frame.sample_rate_hz, cutoff)
    elif filter_mode in {"50 Hz notch", "60 Hz notch", "Custom notch"}:
        notch_hz = 50.0 if filter_mode == "50 Hz notch" else 60.0 if filter_mode == "60 Hz notch" else st.sidebar.number_input("Notch Hz", min_value=0.001, value=min(60.0, nyquist * 0.5))
        q = st.sidebar.slider("Notch Q", 5.0, 100.0, 30.0, 5.0)
        values = notch_filter(values, frame.sample_rate_hz, notch_hz, q)
except ValueError as exc:
    st.sidebar.error(str(exc))
    st.stop()

features = extract_features(values, frame.sample_rate_hz)
quality = assess_signal_quality(values, frame.sample_rate_hz)
anomaly_flags, anomaly_scores = detect_anomalies(values, frame.sample_rate_hz, contamination=anomaly_sensitivity, window_ms=float(anomaly_window_ms))
events = anomaly_events(anomaly_flags, anomaly_scores, frame.sample_rate_hz)
anomaly_fraction = float(np.mean(anomaly_flags))
peaks = dominant_frequencies(values, frame.sample_rate_hz, top_k=5)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample rate", f"{frame.sample_rate_hz:.2f} Hz")
m2.metric("Samples", f"{values.size:,}")
m3.metric("RMS", f"{features.rms:.4f}")
m4.metric("Dominant frequency", f"{features.dominant_frequency_hz:.2f} Hz" if features.dominant_frequency_hz is not None else "n/a")
m5.metric("Quality", f"{quality.score:.0f}/100", quality.grade)
m6.metric("Anomaly events", str(len(events)), f"{100 * anomaly_fraction:.1f}% coverage")

overview_tab, frequency_tab, tf_tab, intelligence_tab, compare_tab = st.tabs(["Overview", "Frequency", "Time-Frequency", "AI Interpretation", "Channel Compare"])

with overview_tab:
    wave = go.Figure()
    wave.add_trace(go.Scattergl(x=frame.time, y=values, mode="lines", name=primary_name))
    if anomaly_flags.any():
        wave.add_trace(go.Scattergl(x=frame.time[anomaly_flags], y=values[anomaly_flags], mode="markers", name="Anomaly window", marker={"size": 4}))
    wave.update_layout(title="Time-domain signal", xaxis_title="Time (s)", yaxis_title="Amplitude", height=430)
    st.plotly_chart(wave, use_container_width=True)
    if events:
        st.subheader("Detected anomaly events")
        st.dataframe(pd.DataFrame([asdict(event) for event in events]), use_container_width=True, hide_index=True)
    else:
        st.info("No anomaly events were detected at the current sensitivity/window settings.")

with frequency_tab:
    left, right = st.columns(2)
    with left:
        fft = fft_spectrum(values, frame.sample_rate_hz)
        fig = go.Figure(go.Scatter(x=fft.frequencies_hz, y=fft.magnitude, mode="lines"))
        fig.update_layout(title="FFT magnitude", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        psd = welch_psd(values, frame.sample_rate_hz)
        fig = go.Figure(go.Scatter(x=psd.frequencies_hz, y=10 * np.log10(np.maximum(psd.magnitude, 1e-18)), mode="lines"))
        fig.update_layout(title="Welch power spectral density", xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB/Hz)", height=360)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Dominant components")
    st.dataframe(pd.DataFrame(peaks, columns=["frequency_hz", "magnitude"]), use_container_width=True, hide_index=True)
    st.subheader("Custom band power")
    b1, b2 = st.columns(2)
    band_low = b1.number_input("Band low (Hz)", min_value=0.0, max_value=float(nyquist), value=0.0)
    default_high = min(float(nyquist), max(1.0, float(nyquist) * 0.2))
    band_high = b2.number_input("Band high (Hz)", min_value=0.001, max_value=float(nyquist), value=default_high)
    if band_high > band_low:
        bp = band_power(values, frame.sample_rate_hz, band_low, band_high)
        c1, c2 = st.columns(2)
        c1.metric("Absolute band power", f"{bp.absolute_power:.6g}")
        c2.metric("Relative band power", f"{100 * bp.relative_power:.2f}%")
    else:
        st.warning("Band high must be greater than band low.")

with tf_tab:
    spec = spectrogram(values, frame.sample_rate_hz)
    heat = go.Figure(data=go.Heatmap(x=spec.times_s, y=spec.frequencies_hz, z=spec.power_db))
    heat.update_layout(title="Spectrogram", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=520)
    st.plotly_chart(heat, use_container_width=True)

with intelligence_tab:
    st.subheader("Transparent AI-assisted interpretation")
    for item in interpret_signal(features, anomaly_fraction, quality):
        if item["severity"] == "warning":
            st.warning(f"**{item['title']}** — {item['message']}")
        elif item["severity"] == "note":
            st.caption(f"**{item['title']}** — {item['message']}")
        else:
            st.info(f"**{item['title']}** — {item['message']}")
    q1, q2, q3 = st.columns(3)
    q1.metric("Clipping proxy", f"{100 * quality.clipping_fraction:.2f}%")
    q2.metric("Low-frequency drift power", f"{100 * quality.drift_fraction:.2f}%")
    q3.metric("High-frequency power", f"{100 * quality.high_frequency_fraction:.2f}%")
    st.caption("Quality metrics are transparent heuristics for acquisition review, not calibrated instrument specifications.")
    with st.expander("Feature table", expanded=True):
        st.dataframe(pd.DataFrame([asdict(features)]), use_container_width=True, hide_index=True)
    report = interpretation_payload(features, quality, events)
    report.update({"version": __version__, "source": frame.source, "channel": primary_name, "sample_rate_hz": frame.sample_rate_hz, "sample_count": int(values.size), "filter": filter_mode, "anomaly_sensitivity": anomaly_sensitivity, "anomaly_window_ms": anomaly_window_ms, "dominant_components": [{"frequency_hz": f, "magnitude": m} for f, m in peaks]})
    st.download_button("Download engineering interpretation JSON", data=json.dumps(report, indent=2).encode("utf-8"), file_name="ai_signal_terminal_report.json", mime="application/json")

with compare_tab:
    if secondary_values is None or secondary_name is None:
        st.info("Upload a CSV with at least two selected signal columns to enable aligned channel comparison.")
    else:
        try:
            comparison = compare_channels(values, secondary_values, frame.sample_rate_hz)
            coh = coherence_spectrum(values, secondary_values, frame.sample_rate_hz)
        except ValueError as exc:
            st.error(str(exc))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Pearson correlation", f"{comparison.pearson_correlation:.3f}")
            c2.metric("Strongest lag", f"{comparison.lag_samples} samples")
            c3.metric("Lag time", f"{comparison.lag_seconds:.6f} s")
            fig = go.Figure(go.Scatter(x=coh.frequencies_hz, y=coh.magnitude, mode="lines"))
            fig.update_layout(title=f"Magnitude-squared coherence: {primary_name} vs {secondary_name}", xaxis_title="Frequency (Hz)", yaxis_title="Coherence", yaxis_range=[0, 1], height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Correlation and coherence are descriptive relationships; they do not establish causation.")

export = pd.DataFrame({"time_s": frame.time, "signal": values, "anomaly": anomaly_flags.astype(int), "anomaly_score": anomaly_scores})
st.download_button("Download interpreted signal CSV", data=export.to_csv(index=False).encode("utf-8"), file_name="ai_signal_terminal_analysis.csv", mime="text/csv")
st.divider()
st.caption("Software demonstrator associated with UK Registered Design application no. 6536215. The repository does not define or expand the legal scope of the registered design.")
