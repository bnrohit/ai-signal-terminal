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


def synthetic_signal(
    duration_s: float = 10.0,
    sample_rate_hz: float = 500.0,
    base_frequency_hz: float = 12.0,
    secondary_frequency_hz: float = 45.0,
    noise_std: float = 0.15,
    anomaly: bool = True,
    seed: int = 7,
) -> SignalFrame:
    """Generate a repeatable synthetic waveform for demonstrations and tests."""
    rng = np.random.default_rng(seed)
    n = max(16, int(duration_s * sample_rate_hz))
    t = np.arange(n, dtype=float) / sample_rate_hz
    y = (
        np.sin(2 * np.pi * base_frequency_hz * t)
        + 0.45 * np.sin(2 * np.pi * secondary_frequency_hz * t)
        + rng.normal(0.0, noise_std, size=n)
    )
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
    return 1.0 / dt


def load_csv(
    file_obj: BinaryIO | bytes,
    value_column: str,
    time_column: str | None = None,
    sample_rate_hz: float | None = None,
) -> SignalFrame:
    """Load one numeric signal from CSV, using time or an explicit sample rate."""
    if isinstance(file_obj, bytes):
        file_obj = BytesIO(file_obj)
    df = pd.read_csv(file_obj)
    if value_column not in df.columns:
        raise ValueError(f"Missing value column: {value_column}")
    values = pd.to_numeric(df[value_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Signal column contains missing or non-numeric values.")
    if values.size < 16:
        raise ValueError("At least 16 signal samples are required.")

    if time_column:
        if time_column not in df.columns:
            raise ValueError(f"Missing time column: {time_column}")
        time_values = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(time_values).all():
            raise ValueError("Time column contains missing or non-numeric values.")
        sr = _infer_sample_rate(time_values)
    else:
        if not sample_rate_hz or sample_rate_hz <= 0:
            raise ValueError("Provide a positive sample rate when no time column is selected.")
        sr = float(sample_rate_hz)
        time_values = np.arange(values.size, dtype=float) / sr

    if sr <= 0 or not np.isfinite(sr):
        raise ValueError("Invalid sample rate.")
    return SignalFrame(time=time_values, values=values, sample_rate_hz=sr, source="csv")
