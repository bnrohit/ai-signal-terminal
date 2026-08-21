from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalFrame:
    time: np.ndarray
    values: np.ndarray
    sample_rate_hz: float
    source: str


@dataclass(frozen=True)
class MultiSignalFrame:
    time: np.ndarray
    channels: dict[str, np.ndarray]
    sample_rate_hz: float
    source: str


def synthetic_signal(duration_s: float = 10.0, sample_rate_hz: float = 500.0, base_frequency_hz: float = 12.0, secondary_frequency_hz: float = 45.0, noise_std: float = 0.15, anomaly: bool = True, seed: int = 7) -> SignalFrame:
    if duration_s <= 0 or sample_rate_hz <= 0:
        raise ValueError("Duration and sample rate must be positive.")
    nyquist = sample_rate_hz / 2.0
    if base_frequency_hz <= 0 or secondary_frequency_hz <= 0:
        raise ValueError("Synthetic frequencies must be positive.")
    if base_frequency_hz >= nyquist or secondary_frequency_hz >= nyquist:
        raise ValueError("Synthetic frequencies must be below Nyquist frequency.")
    rng = np.random.default_rng(seed)
    n = max(16, int(duration_s * sample_rate_hz))
    t = np.arange(n, dtype=float) / sample_rate_hz
    y = np.sin(2 * np.pi * base_frequency_hz * t) + 0.45 * np.sin(2 * np.pi * secondary_frequency_hz * t) + rng.normal(0.0, noise_std, size=n)
    if anomaly and n > 100:
        center = int(0.62 * n)
        width = max(4, int(0.015 * n))
        idx = np.arange(n)
        y += 2.2 * np.exp(-0.5 * ((idx - center) / width) ** 2)
    return SignalFrame(time=t, values=y, sample_rate_hz=sample_rate_hz, source="synthetic")


def _infer_sample_rate(time_values: np.ndarray) -> float:
    deltas = np.diff(time_values)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        raise ValueError("Cannot infer sample rate from the selected time column.")
    dt = float(np.median(deltas))
    if deltas.size >= 4:
        jitter = float(np.std(deltas) / max(np.mean(deltas), 1e-18))
        if jitter > 0.05:
            raise ValueError("Time samples are too irregular (>5% interval jitter). Resample before frequency analysis.")
    return 1.0 / dt


def load_csv(file_obj: BinaryIO | bytes, value_column: str, time_column: str | None = None, sample_rate_hz: float | None = None) -> SignalFrame:
    multi = load_csv_multi(file_obj, [value_column], time_column, sample_rate_hz)
    return SignalFrame(multi.time, multi.channels[value_column], multi.sample_rate_hz, multi.source)


def load_csv_multi(file_obj: BinaryIO | bytes, value_columns: list[str], time_column: str | None = None, sample_rate_hz: float | None = None) -> MultiSignalFrame:
    if isinstance(file_obj, bytes):
        file_obj = BytesIO(file_obj)
    df = pd.read_csv(file_obj)
    if not value_columns:
        raise ValueError("Select at least one signal column.")
    channels = {}
    for column in value_columns:
        if column not in df.columns:
            raise ValueError(f"Missing value column: {column}")
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Signal column '{column}' contains missing or non-numeric values.")
        if values.size < 16:
            raise ValueError("At least 16 signal samples are required.")
        channels[column] = values
    if time_column:
        if time_column not in df.columns:
            raise ValueError(f"Missing time column: {time_column}")
        time_values = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(time_values).all():
            raise ValueError("Time column contains missing or non-numeric values.")
        if np.any(np.diff(time_values) <= 0):
            raise ValueError("Time column must be strictly increasing.")
        sr = _infer_sample_rate(time_values)
    else:
        if not sample_rate_hz or sample_rate_hz <= 0:
            raise ValueError("Provide a positive sample rate when no time column is selected.")
        sr = float(sample_rate_hz)
        time_values = np.arange(len(df), dtype=float) / sr
    if sr <= 0 or not np.isfinite(sr):
        raise ValueError("Invalid sample rate.")
    return MultiSignalFrame(time=time_values, channels=channels, sample_rate_hz=sr, source="csv")
