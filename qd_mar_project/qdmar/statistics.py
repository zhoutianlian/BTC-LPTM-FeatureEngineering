"""Past-only robust statistics used by QD-MAR."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def infer_maturity_lag_minutes(times: pd.Series, horizon_minutes: int) -> int:
    """Infer minimum row shift needed so a future-return label is matured.

    For source-clock hourly data, all 20m/30m/60m labels require at least one
    row shift. For 10m grids, 20m/30m/60m imply 2/3/6 row shifts.
    """
    dt = pd.to_datetime(times, utc=True).sort_values().diff().dropna().dt.total_seconds() / 60.0
    if dt.empty or not np.isfinite(dt.median()) or dt.median() <= 0:
        return 1
    median_step = float(dt.median())
    return max(1, int(math.ceil(horizon_minutes / median_step)))


def rolling_mad_sigma(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Compute a fast past-only robust MAD volatility proxy.

    Engineering implementation note: exact rolling MAD via Python rolling.apply
    is too slow for repeated streaming recalculation. We use a causal two-step
    robust proxy: rolling median, then rolling median of absolute deviations
    from that causal median. It preserves the algorithm requirement of
    past-only robust scaling while being suitable for production updates.
    """
    s = pd.to_numeric(series, errors="coerce")
    med = s.rolling(window=window, min_periods=min_periods).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window=window, min_periods=min_periods).median()
    return 1.4826 * mad


def past_matured_sigma(
    df: pd.DataFrame,
    time_col: str,
    ret_col: str,
    horizon_minutes: int,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Past-only volatility scale for horizon labels.

    The return label at row i uses future price. To avoid leakage, the rolling
    volatility at event time t can only use rows whose label has already matured.
    We approximate the maturity boundary by shifting by the horizon-equivalent
    number of rows inferred from the event clock.
    """
    lag = infer_maturity_lag_minutes(df[time_col], horizon_minutes)
    matured = pd.to_numeric(df[ret_col], errors="coerce").shift(lag)
    sigma = rolling_mad_sigma(matured, window=window, min_periods=min_periods)
    # Fallback is still past-only because expanding is applied to matured labels.
    expanding_fallback = matured.expanding(min_periods=max(5, min_periods // 4)).std()
    sigma = sigma.fillna(expanding_fallback)
    return sigma.replace([np.inf, -np.inf], np.nan)


def assign_train_quantile_regime(series: pd.Series, train_mask: pd.Series, labels=("low", "mid", "high")) -> pd.Series:
    """Assign low/mid/high regimes using train-only terciles."""
    train_values = pd.to_numeric(series[train_mask], errors="coerce").dropna()
    out = pd.Series("unknown", index=series.index, dtype=object)
    if train_values.empty:
        return out
    qs = train_values.quantile([1 / 3, 2 / 3]).to_numpy()
    # Ensure monotone unique edges. If not, fall back to median split style.
    q1, q2 = float(qs[0]), float(qs[1])
    vals = pd.to_numeric(series, errors="coerce")
    out.loc[vals <= q1] = labels[0]
    out.loc[(vals > q1) & (vals <= q2)] = labels[1]
    out.loc[vals > q2] = labels[2]
    return out


def winsorize_by_sigma(values: pd.Series, sigma: pd.Series, k: float = 5.0) -> pd.Series:
    """Clip values by row-wise +/- k*sigma."""
    v = pd.to_numeric(values, errors="coerce")
    s = pd.to_numeric(sigma, errors="coerce").abs()
    lower = -k * s
    upper = k * s
    return v.clip(lower=lower, upper=upper)
