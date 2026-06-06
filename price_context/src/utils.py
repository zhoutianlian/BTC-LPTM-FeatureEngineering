from __future__ import annotations

import math
import warnings
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

EPSILON = 1e-12


def parse_window_to_timedelta(window: str | pd.Timedelta) -> pd.Timedelta:
    """Parse a config window such as '1h', '20min', or '7d'."""
    if isinstance(window, pd.Timedelta):
        return window
    try:
        return pd.to_timedelta(window)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Invalid window string: {window!r}") from exc


def window_to_label(window: str | pd.Timedelta) -> str:
    """Convert a window to a stable feature suffix."""
    if isinstance(window, str):
        return window.replace(" ", "").replace("min", "m")
    td = parse_window_to_timedelta(window)
    minutes = int(td.total_seconds() // 60)
    if minutes % (24 * 60) == 0:
        return f"{minutes // (24 * 60)}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def window_to_hours(window: str | pd.Timedelta) -> float:
    return parse_window_to_timedelta(window).total_seconds() / 3600.0


def window_to_bars(window: str | pd.Timedelta, bar_minutes: int | float) -> int:
    td = parse_window_to_timedelta(window)
    bars = int(round(td.total_seconds() / 60.0 / float(bar_minutes)))
    if bars <= 0:
        raise ValueError(f"Window {window!r} is shorter than one configured bar.")
    return bars


def min_periods_for_window(window: str | pd.Timedelta, bar_minutes: int | float, min_obs_ratio: float) -> int:
    return max(1, int(math.ceil(window_to_bars(window, bar_minutes) * min_obs_ratio)))


def squash(x: pd.Series | np.ndarray | float, c: float = 2.0):
    """Monotone bounded transform: 1 - exp(-x / c)."""
    return 1.0 - np.exp(-np.asarray(x, dtype="float64") / max(float(c), EPSILON))


def safe_divide(numer, denom, eps: float = EPSILON):
    return np.asarray(numer, dtype="float64") / (np.asarray(denom, dtype="float64") + eps)


def clip_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    return s.clip(lower=lower, upper=upper)


def sorted_unique_windows(*window_groups: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in window_groups:
        for w in group:
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


def infer_regular_frequency_minutes(time: pd.Series) -> Optional[float]:
    if len(time) < 3:
        return None
    diffs = time.sort_values().diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return None
    return float(diffs.mode().iloc[0])


def past_rolling_percentile(
    series: pd.Series,
    lookback: str,
    min_periods: int = 20,
) -> pd.Series:
    """Past-only empirical percentile of current value against a rolling history.

    Implementation detail: pandas Rolling.rank ranks the current observation inside
    the right-closed rolling window. We transform rank/count to an estimate of the
    percentile versus previous observations only: (rank - 1) / (count - 1). This
    uses the current value only as the score being ranked and never uses future data.
    """
    roll = series.rolling(lookback, min_periods=max(2, min_periods))
    rank = roll.rank(method="average", pct=False)
    count = roll.count()
    pct = (rank - 1.0) / (count - 1.0)
    pct = pct.where(count > 1)
    return pct.clip(0.0, 1.0)


def rolling_iqr_scale_past(
    series: pd.Series,
    lookback: str,
    min_periods: int = 20,
    scale_to_sigma: bool = True,
) -> pd.Series:
    """Past-only robust scale using rolling IQR, excluding current observation."""
    hist = series.shift(1)
    roll = hist.rolling(lookback, min_periods=min_periods)
    q75 = roll.quantile(0.75)
    q25 = roll.quantile(0.25)
    iqr = q75 - q25
    if scale_to_sigma:
        return iqr / 1.349
    return iqr


def rolling_median_past(series: pd.Series, lookback: str, min_periods: int = 20) -> pd.Series:
    return series.shift(1).rolling(lookback, min_periods=min_periods).median()


def robust_zscore_past_rolling(
    series: pd.Series,
    lookback: str,
    clip: Sequence[float] = (-5.0, 5.0),
    min_periods: int = 20,
    eps: float = EPSILON,
) -> pd.Series:
    """Past-only robust z-score using rolling median and IQR."""
    hist = series.shift(1)
    roll = hist.rolling(lookback, min_periods=min_periods)
    median = roll.median()
    q75 = roll.quantile(0.75)
    q25 = roll.quantile(0.25)
    iqr = q75 - q25
    z = (series - median) / (iqr + eps)
    return z.clip(float(clip[0]), float(clip[1]))


def robust_zscore_train_or_past(
    series: pd.Series,
    cfg: dict,
    default_lookback: str = "30d",
    eps: float = EPSILON,
) -> pd.Series:
    """Compute robust z-score without leakage.

    If train_split.start_time/end_time are set, median/IQR are estimated on that
    training slice only. Otherwise, the function falls back to past-only rolling
    median/IQR so that no full-sample statistic is used.
    """
    clip = cfg.get("zscore_clip", [-5, 5])
    lookback = cfg.get("zscore_lookback", default_lookback)
    train = cfg.get("train_split", {}) or {}
    start, end = train.get("start_time"), train.get("end_time")
    method = cfg.get("zscore_method", "past_rolling_robust")
    fallback = cfg.get("fallback_when_train_missing", "past_rolling_robust")
    if method == "past_rolling_robust":
        return robust_zscore_past_rolling(series, lookback=lookback, clip=clip, min_periods=20, eps=eps)
    if method == "train_robust" and start is not None and end is not None:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        train_values = series.loc[(series.index >= start_ts) & (series.index <= end_ts)].dropna()
        if train_values.empty:
            warnings.warn(
                "Configured train split has no non-missing observations; falling back to past rolling robust z-score.",
                RuntimeWarning,
            )
        else:
            med = train_values.median()
            iqr = train_values.quantile(0.75) - train_values.quantile(0.25)
            return ((series - med) / (iqr + eps)).clip(float(clip[0]), float(clip[1]))
    if method == "train_robust" and fallback != "past_rolling_robust":
        raise ValueError("Only fallback_when_train_missing='past_rolling_robust' is supported.")
    return robust_zscore_past_rolling(series, lookback=lookback, clip=clip, min_periods=20, eps=eps)


def robust_sigma_past(
    series: pd.Series,
    lookback: str,
    bar_minutes: int,
    estimator: str = "mad",
    min_periods: Optional[int] = None,
    eps: float = EPSILON,
) -> pd.Series:
    """Past-only robust return scale.

    estimator='mad' follows the requirement formula sigma=1.4826*MAD. It uses
    pandas+numba fixed-bar rolling when numba is available. If numba is missing,
    it falls back to IQR/1.349 and emits a warning.
    """
    estimator = (estimator or "mad").lower()
    if min_periods is None:
        min_periods = max(20, min_periods_for_window(lookback, bar_minutes, 0.5))
    if estimator == "iqr":
        return rolling_iqr_scale_past(series, lookback=lookback, min_periods=min_periods, scale_to_sigma=True).clip(lower=eps)

    if estimator != "mad":
        warnings.warn(f"Unknown robust sigma estimator {estimator!r}; using MAD.", RuntimeWarning)

    window_bars = window_to_bars(lookback, bar_minutes)
    shifted = series.shift(1)
    try:
        from numba import njit  # type: ignore

        @njit(cache=False)
        def _mad_sigma_numba(arr):  # pragma: no cover - exercised by integration run
            valid = arr[~np.isnan(arr)]
            if valid.shape[0] == 0:
                return np.nan
            med = np.median(valid)
            mad = np.median(np.abs(valid - med))
            return 1.4826 * mad

        sigma = shifted.rolling(window_bars, min_periods=min_periods).apply(
            _mad_sigma_numba,
            raw=True,
            engine="numba",
        )
        return sigma.clip(lower=eps)
    except Exception as exc:  # pragma: no cover - depends on optional numba
        warnings.warn(
            f"MAD rolling sigma requires a working numba installation; falling back to IQR scale. Details: {exc}",
            RuntimeWarning,
        )
        return rolling_iqr_scale_past(series, lookback=lookback, min_periods=min_periods, scale_to_sigma=True).clip(lower=eps)


def signed_max_abs(values: np.ndarray) -> float:
    """Return the signed value with largest absolute magnitude in a rolling window."""
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    idx = int(np.nanargmax(np.abs(values)))
    return float(values[idx])


def ensure_datetime_index(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df.set_index(time_col, drop=False)
    return df


def as_float_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").astype("float64")
