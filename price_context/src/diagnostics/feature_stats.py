from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (int,)):
        return int(value)
    return None


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def infer_time_profile(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    if "time" not in df.columns:
        return {
            "has_time": False,
            "sample_count": int(len(df)),
            "start": None,
            "end": None,
            "duplicate_time_count": None,
            "is_monotonic_increasing": None,
            "median_interval_minutes": None,
            "abnormal_interval_count": None,
            "max_interval_minutes": None,
            "interval_threshold_minutes": None,
        }

    time = pd.to_datetime(df["time"], errors="coerce")
    valid_time = time.dropna()
    diff_min = time.diff().dt.total_seconds() / 60.0
    positive_diff = diff_min[diff_min > 0]
    median_interval = json_number(positive_diff.median()) if not positive_diff.empty else None
    configured_tol = float(cfg.get("quality", {}).get("gap_tolerance_minutes", cfg.get("data", {}).get("bar_minutes", 10)))
    if median_interval is None:
        threshold = configured_tol
    else:
        threshold = max(configured_tol, float(median_interval) * 3.0)
    abnormal = diff_min > threshold

    return {
        "has_time": True,
        "sample_count": int(len(df)),
        "start": _ts_to_str(valid_time.min()) if not valid_time.empty else None,
        "end": _ts_to_str(valid_time.max()) if not valid_time.empty else None,
        "duplicate_time_count": int(time.duplicated().sum()),
        "is_monotonic_increasing": bool(time.is_monotonic_increasing),
        "median_interval_minutes": median_interval,
        "abnormal_interval_count": int(abnormal.sum()),
        "max_interval_minutes": json_number(diff_min.max()),
        "interval_threshold_minutes": json_number(threshold),
    }


def compute_numeric_stats(series: pd.Series) -> dict[str, Any]:
    numeric = to_numeric_series(series)
    finite = numeric[np.isfinite(numeric)]
    total = int(len(series))
    nan_count = int(numeric.isna().sum())
    inf_count = int(np.isinf(numeric).sum())
    valid_count = int(len(finite))
    missing_ratio = float(nan_count / total) if total else 1.0
    inf_ratio = float(inf_count / total) if total else 0.0

    stats: dict[str, Any] = {
        "count": valid_count,
        "nan_count": nan_count,
        "nan_ratio": missing_ratio,
        "inf_count": inf_count,
        "inf_ratio": inf_ratio,
        "missing_count": nan_count,
        "missing_ratio": missing_ratio,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "median": None,
        "p01": None,
        "p05": None,
        "p25": None,
        "p75": None,
        "p95": None,
        "p99": None,
        "skew": None,
        "kurtosis": None,
        "zero_ratio": None,
        "positive_ratio": None,
        "negative_ratio": None,
        "unique_count": 0,
        "constant_flag": False,
    }
    if valid_count == 0:
        return stats

    quantiles = finite.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    std = finite.std(ddof=1) if valid_count > 1 else 0.0
    unique_count = int(finite.nunique(dropna=True))
    tolerance = max(1e-12, abs(float(finite.mean())) * 1e-12)
    constant_flag = unique_count <= 1 or float(std) <= tolerance

    stats.update(
        {
            "mean": json_number(finite.mean()),
            "std": json_number(std),
            "min": json_number(finite.min()),
            "max": json_number(finite.max()),
            "median": json_number(quantiles.loc[0.5]),
            "p01": json_number(quantiles.loc[0.01]),
            "p05": json_number(quantiles.loc[0.05]),
            "p25": json_number(quantiles.loc[0.25]),
            "p75": json_number(quantiles.loc[0.75]),
            "p95": json_number(quantiles.loc[0.95]),
            "p99": json_number(quantiles.loc[0.99]),
            "skew": json_number(finite.skew()),
            "kurtosis": json_number(finite.kurt()),
            "zero_ratio": json_number((finite == 0).sum() / valid_count),
            "positive_ratio": json_number((finite > 0).sum() / valid_count),
            "negative_ratio": json_number((finite < 0).sum() / valid_count),
            "unique_count": unique_count,
            "constant_flag": bool(constant_flag),
        }
    )
    return stats


def compute_outlier_diagnostics(
    series: pd.Series,
    time: pd.Series | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    report_cfg = cfg.get("report", {})
    z_threshold = float(report_cfg.get("zscore_threshold", 5.0))
    iqr_multiplier = float(report_cfg.get("iqr_multiplier", 1.5))
    q_low = float(report_cfg.get("extreme_quantile_low", 0.001))
    q_high = float(report_cfg.get("extreme_quantile_high", 0.999))
    max_points = int(report_cfg.get("max_outlier_timestamps", 50))

    numeric = to_numeric_series(series)
    finite_mask = np.isfinite(numeric)
    finite = numeric[finite_mask]
    if finite.empty:
        return {
            "zscore_outlier_count": 0,
            "iqr_outlier_count": 0,
            "quantile_extreme_count": 0,
            "cliff_jump_count": 0,
            "explosion_flag": False,
            "outlier_points": [],
            "zscore_threshold": z_threshold,
            "iqr_multiplier": iqr_multiplier,
            "extreme_quantile_low": q_low,
            "extreme_quantile_high": q_high,
        }

    mean = finite.mean()
    std = finite.std(ddof=1)
    if len(finite) > 1 and np.isfinite(std) and std > 0:
        z = ((numeric - mean) / std).abs()
        z_mask = z > z_threshold
    else:
        z = pd.Series(np.nan, index=numeric.index, dtype="float64")
        z_mask = pd.Series(False, index=numeric.index)

    q25 = finite.quantile(0.25)
    q75 = finite.quantile(0.75)
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        iqr_low = q25 - iqr_multiplier * iqr
        iqr_high = q75 + iqr_multiplier * iqr
        iqr_mask = (numeric < iqr_low) | (numeric > iqr_high)
    else:
        iqr_mask = pd.Series(False, index=numeric.index)

    low_bound = finite.quantile(q_low)
    high_bound = finite.quantile(q_high)
    extreme_mask = (numeric < low_bound) | (numeric > high_bound)

    diff_abs = numeric.diff().abs()
    diff_finite = diff_abs[np.isfinite(diff_abs)]
    if diff_finite.empty:
        cliff_mask = pd.Series(False, index=numeric.index)
    else:
        diff_q99 = diff_finite.quantile(0.99)
        diff_q25 = diff_finite.quantile(0.25)
        diff_q75 = diff_finite.quantile(0.75)
        diff_iqr = diff_q75 - diff_q25
        cliff_threshold = max(float(diff_q99), float(diff_q75 + 10.0 * diff_iqr))
        cliff_mask = diff_abs > cliff_threshold if cliff_threshold > 0 else pd.Series(False, index=numeric.index)

    combined = (z_mask.fillna(False) | cliff_mask.fillna(False)) & finite_mask
    points = _top_outlier_points(numeric, z, combined, time, max_points)
    max_abs = finite.abs().max()
    p99_abs = finite.abs().quantile(0.99)
    explosion_flag = bool(np.isfinite(max_abs) and np.isfinite(p99_abs) and p99_abs > 0 and max_abs > p99_abs * 100.0)

    return {
        "zscore_outlier_count": int(z_mask.fillna(False).sum()),
        "iqr_outlier_count": int(iqr_mask.fillna(False).sum()),
        "quantile_extreme_count": int(extreme_mask.fillna(False).sum()),
        "cliff_jump_count": int(cliff_mask.fillna(False).sum()),
        "explosion_flag": explosion_flag,
        "outlier_points": points,
        "zscore_threshold": z_threshold,
        "iqr_multiplier": iqr_multiplier,
        "extreme_quantile_low": q_low,
        "extreme_quantile_high": q_high,
    }


def compute_temporal_diagnostics(series: pd.Series, rolling_window: int) -> dict[str, Any]:
    numeric = to_numeric_series(series)
    finite = np.isfinite(numeric)
    values = numeric.where(finite)
    changed = values.ne(values.shift(1)) | values.isna() | values.shift(1).isna()
    groups = changed.cumsum()
    constant_lengths = values.notna().groupby(groups).sum()
    zero_lengths = (values == 0).groupby(((values != values.shift(1)) | values.isna()).cumsum()).sum()
    null_lengths = values.isna().groupby(values.notna().cumsum()).sum()

    max_constant = int(constant_lengths.max()) if not constant_lengths.empty else 0
    max_zero = int(zero_lengths.max()) if not zero_lengths.empty else 0
    max_null = int(null_lengths.max()) if not null_lengths.empty else 0
    long_threshold = max(rolling_window, 20)
    return {
        "rolling_window": int(rolling_window),
        "max_consecutive_constant": max_constant,
        "max_consecutive_zero": max_zero,
        "max_consecutive_null": max_null,
        "long_constant_flag": bool(max_constant >= long_threshold),
        "long_zero_flag": bool(max_zero >= long_threshold),
        "long_null_flag": bool(max_null >= long_threshold),
        "long_segment_threshold": int(long_threshold),
    }


def rolling_window_from_config(cfg: dict[str, Any], row_count: int) -> int:
    configured = cfg.get("report", {}).get("rolling_window_bars")
    if configured is not None:
        return max(2, int(configured))
    bar_minutes = float(cfg.get("data", {}).get("bar_minutes", 10))
    daily_bars = max(2, int(round(24.0 * 60.0 / bar_minutes)))
    if row_count <= daily_bars:
        return max(2, min(row_count, int(round(row_count / 4)) or 2))
    return daily_bars


def _top_outlier_points(
    numeric: pd.Series,
    z: pd.Series,
    mask: pd.Series,
    time: pd.Series | None,
    max_points: int,
) -> list[dict[str, Any]]:
    if max_points <= 0 or not mask.any():
        return []
    score = z.where(mask).fillna(numeric.where(mask).abs())
    idx = score.sort_values(ascending=False).head(max_points).index
    points: list[dict[str, Any]] = []
    for i in idx:
        item = {
            "index": int(i) if isinstance(i, (int, np.integer)) else str(i),
            "value": json_number(numeric.loc[i]),
            "zscore": json_number(z.loc[i]) if i in z.index else None,
        }
        if time is not None:
            item["time"] = _ts_to_str(time.loc[i])
        points.append(item)
    return points


def _ts_to_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).isoformat(sep=" ")
    except Exception:
        return str(value)
