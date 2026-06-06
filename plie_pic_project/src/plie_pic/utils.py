from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def to_utc_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to timezone-aware UTC timestamps."""
    return pd.to_datetime(s, utc=True, errors="coerce")


def stable_softplus(x: pd.Series | np.ndarray) -> np.ndarray:
    """Numerically stable softplus transformation."""
    arr = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)


def rolling_mad(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Past-only rolling median absolute deviation.

    pandas rolling windows are right-closed by default, so the value at t uses
    observations up to and including t and never uses observations after t.
    """
    return values.rolling(window=window, min_periods=min_periods).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
    )


def robust_zscore_past_only(
    values: pd.Series,
    window: int,
    min_periods: int,
    eps: float,
) -> pd.Series:
    """Past-only rolling median/MAD robust z-score."""
    med = values.rolling(window=window, min_periods=min_periods).median()
    mad = rolling_mad(values, window=window, min_periods=min_periods)
    z = (values - med) / (1.4826 * mad + eps)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Average pinball loss for quantile regression."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    loss = np.maximum(quantile * err, (quantile - 1.0) * err)
    return float(np.nanmean(loss))


def weighted_pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    loss = np.maximum(quantile * err, (quantile - 1.0) * err)
    if sample_weight is None:
        return float(np.nanmean(loss))
    w = np.asarray(sample_weight, dtype=float)
    mask = np.isfinite(loss) & np.isfinite(w)
    if not mask.any():
        return math.nan
    return float(np.sum(loss[mask] * w[mask]) / np.sum(w[mask]))


def chronological_split_indices(
    n: int, train_ratio: float, validation_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chronological train/validation/test indices."""
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    train_end = int(np.floor(n * train_ratio))
    val_end = int(np.floor(n * (train_ratio + validation_ratio)))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def finite_ratio(s: pd.Series) -> float:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(arr).mean()) if len(arr) else 0.0


def compact_float(x: float | int | None, digits: int = 4) -> float | None:
    if x is None or not np.isfinite(float(x)):
        return None
    return round(float(x), digits)


def sample_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Evenly sample a DataFrame for browser-friendly visualization."""
    if len(df) <= max_points:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].copy()


def safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    joined = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method=method))


def ensure_list(x: Iterable[int] | int) -> list[int]:
    if isinstance(x, int):
        return [x]
    return list(x)
