from __future__ import annotations

import numpy as np
import pandas as pd


def _quantile_boundaries(history: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    bounds = np.quantile(history, qs)
    if bounds.size > 1:
        bounds = np.maximum.accumulate(bounds)
    return bounds


def point_in_time_quantile_binning(
    df: pd.DataFrame,
    *,
    n_bins: int,
    col: str,
    min_history: int = 24 * 3,
    refit_every: int = 24,
    neutral_bin: int | None = None,
) -> pd.DataFrame:
    if n_bins <= 1:
        raise ValueError("n_bins must be > 1")

    neutral = neutral_bin if neutral_bin is not None else n_bins // 2
    out = df.copy().reset_index(drop=True)
    values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)

    bins = np.full(len(out), neutral, dtype=int)
    last_bounds = None
    last_fit_i = None

    for i, value in enumerate(values):
        hist = values[:i]
        hist = hist[np.isfinite(hist)]
        if hist.size < max(min_history, n_bins):
            bins[i] = neutral
            continue

        if last_bounds is None or last_fit_i is None or (i - last_fit_i) >= refit_every:
            last_bounds = _quantile_boundaries(hist, n_bins=n_bins)
            last_fit_i = i

        if np.isfinite(value):
            bins[i] = int(np.searchsorted(last_bounds, value, side="right"))
        else:
            bins[i] = neutral

    out["bin_index"] = np.clip(bins, 0, n_bins - 1).astype(int)
    return out


def cal_is_extreme_from_bins(
    df: pd.DataFrame,
    *,
    bin_col: str = "bin_index",
    col_new: str = "is_rpn_extreme",
    window_size: int = 30 * 24,
) -> pd.DataFrame:
    out = df.copy()
    bins = pd.to_numeric(out[bin_col], errors="coerce").fillna(4).astype(int)

    def _flag(window: pd.Series) -> float:
        arr = window.astype(int).tolist()
        if 8 in arr:
            last_8 = len(arr) - 1 - arr[::-1].index(8)
            if 4 not in arr[last_8 + 1 :]:
                return -1.0
        if 0 in arr:
            last_0 = len(arr) - 1 - arr[::-1].index(0)
            if 5 not in arr[last_0 + 1 :]:
                return 1.0
        return 0.0

    out[col_new] = bins.rolling(window=window_size, min_periods=1).apply(_flag, raw=False)
    return out


def get_bin_stats(df: pd.DataFrame, *, value_col: str = "risk_priority_number", bin_col: str = "bin_index") -> pd.DataFrame:
    n = len(df)
    stats = (
        df.groupby(bin_col)[value_col]
        .agg(
            interval=lambda x: (x.max() - x.min()),
            count="count",
            median="median",
            mean="mean",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )
    stats["proportion"] = stats["count"] / max(n, 1)
    stats["acc_prop"] = stats["proportion"].cumsum().shift(1).fillna(0)
    stats = stats.rename(columns={"proportion": "prop_q_high", "acc_prop": "prop_q_low"})
    stats["prop_q_high"] = stats["prop_q_low"] + stats["prop_q_high"]
    return stats
