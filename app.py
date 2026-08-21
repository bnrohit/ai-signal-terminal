from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from signal_terminal.intelligence import detect_anomalies, extract_features, interpret_signal
from signal_terminal.io import load_csv, synthetic_signal
from signal_terminal.processing import bandpass_filter, fft_spectrum, spectrogram, welch_psd


st.set_page_config(page_title="AI Signal Terminal", page_icon="📡", layout="wide")
st.title("📡 AI Signal Terminal")
st.caption("Interactive signal processing + transparent AI-assisted interpretation")

with st.sidebar:
    st.header("Signal source")
    mode = st.radio("Input", ["Synthetic demo", "Upload CSV"], index=0)
    apply_filter = st.checkbox("Apply band-pass filter", value=False)
    anomaly_sensitivity = st.slider("Anomaly sensitivity", 0.005, 0.10, 0.02, 0.005)

    if mode == "Synthetic demo":
        sample_rate = st.number_input("Sample rate (Hz)", min_value=20.0, max_value=100000.0, value=500.0)
        duration = st.number_input("Duration (s)", min_value=1.0, max_value=120.0, value=10.0)
        f1 = st.number_input("Primary frequency (Hz)", min_value=0.1, value=12.0)
        f2 = st.number_input("Secondary frequency (Hz)", min_value=0.1, value=45.0)
        include_anomaly = st.checkbox("Inject demo transient", value=True)
        frame = synthetic_signal(duration, sample_rate, f1, f2, anomaly=include_anomaly)
    else:
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to begin.")
            st.stop()
        raw = uploaded.getvalue()
        preview = pd.read_csv(io.BytesIO(raw), nrows=20)
        numeric_cols = [c for c in preview.columns if pd.api.types.is_numeric_dtype(preview[c])]
        if not numeric_cols:
            st.error("No numeric columns were detected in the CSV preview.")
            st.stop()
        value_col = st.selectbox("Signal value column", numeric_cols)
        time_options = ["<none>"] + [c for c in numeric_cols if c != value_col]
        time_col = st.selectbox("Time column (seconds)", time_options)
        sr = None
        if time_col == "<none>":
            sr = st.number_input("Sample rate (Hz)", min_value=0.001, value=1000.0)
        try:
            frame = load_csv(io.BytesIO(raw), value_col, None if time_col == "<none>" else time_col, sr)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

values = frame.values.copy()
if apply_filter:
    nyquist = frame.sample_rate_hz / 2.0
    col1, col2 = st.sidebar.columns(2)
    low = col1.number_input("Low Hz", min_value=0.001, value=max(0.1, nyquist * 0.02))
    high_default = max(low * 2, nyquist * 0.8)
    high = col2.number_input("High Hz", min_value=low + 0.001, value=min(high_default, nyquist * 0.95))
    try:
        values = bandpass_filter(values, frame.sample_rate_hz, low, high)
    except ValueError as exc:
        st.sidebar.error(str(exc))

features = extract_features(values, frame.sample_rate_hz)
anomaly_flags, anomaly_scores = detect_anomalies(values, frame.sample_rate_hz, contamination=anomaly_sensitivity)
anomaly_fraction = float(np.mean(anomaly_flags))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Sample rate", f"{frame.sample_rate_hz:.2f} Hz")
m2.metric("Samples", f"{values.size:,}")
m3.metric("RMS", f"{features.rms:.4f}")
m4.metric("Dominant frequency", f"{features.dominant_frequency_hz:.2f} Hz" if features.dominant_frequency_hz else "n/a")
m5.metric("Anomaly coverage", f"{100 * anomaly_fraction:.1f}%")

wave = go.Figure()
wave.add_trace(go.Scattergl(x=frame.time, y=values, mode="lines", name="Signal"))
if anomaly_flags.any():
    wave.add_trace(go.Scattergl(
        x=frame.time[anomaly_flags],
        y=values[anomaly_flags],
        mode="markers",
        name="Anomaly window",
        marker={"size": 4},
    ))
wave.update_layout(title="Time-domain signal", xaxis_title="Time (s)", yaxis_title="Amplitude", height=380)
st.plotly_chart(wave, use_container_width=True)

left, right = st.columns(2)
with left:
    fft = fft_spectrum(values, frame.sample_rate_hz)
    fig = go.Figure(go.Scatter(x=fft.frequencies_hz, y=fft.magnitude, mode="lines"))
    fig.update_layout(title="FFT magnitude", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=350)
    st.plotly_chart(fig, use_container_width=True)
with right:
    psd = welch_psd(values, frame.sample_rate_hz)
    fig = go.Figure(go.Scatter(x=psd.frequencies_hz, y=10 * np.log10(np.maximum(psd.magnitude, 1e-18)), mode="lines"))
    fig.update_layout(title="Welch power spectral density", xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB/Hz)", height=350)
    st.plotly_chart(fig, use_container_width=True)

spec = spectrogram(values, frame.sample_rate_hz)
heat = go.Figure(data=go.Heatmap(x=spec.times_s, y=spec.frequencies_hz, z=spec.power_db))
heat.update_layout(title="Spectrogram", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=420)
st.plotly_chart(heat, use_container_width=True)

st.subheader("AI-assisted interpretation")
for item in interpret_signal(features, anomaly_fraction):
    if item["severity"] == "warning":
        st.warning(f"**{item['title']}** — {item['message']}")
    elif item["severity"] == "note":
        st.caption(f"**{item['title']}** — {item['message']}")
    else:
        st.info(f"**{item['title']}** — {item['message']}")

with st.expander("Feature table"):
    st.dataframe(pd.DataFrame([features.__dict__]), use_container_width=True)

export = pd.DataFrame({
    "time_s": frame.time,
    "signal": values,
    "anomaly": anomaly_flags.astype(int),
    "anomaly_score": anomaly_scores,
})
st.download_button(
    "Download interpreted signal CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="ai_signal_terminal_analysis.csv",
    mime="text/csv",
)

st.divider()
st.caption("Software demonstrator associated with UK Registered Design application no. 6536215. The repository does not define or expand the legal scope of the registered design.")
