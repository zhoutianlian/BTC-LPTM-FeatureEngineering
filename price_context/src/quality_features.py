from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import min_periods_for_window, window_to_bars, window_to_label


def compute_quality_features(df: pd.DataFrame, cfg: dict[str, Any], windows: Iterable[str]) -> pd.DataFrame:
    """Compute observation, missing, gap, and outlier quality features.

    A valid observation requires finite positive OHLC prices and basic OHLC
    consistency. No rows are forward-filled. Gaps are flagged rather than repaired.
    """
    out = pd.DataFrame(index=df.index)
    price_valid = (
        df[["open", "high", "low", "close"]].notna().all(axis=1)
        & np.isfinite(df[["open", "high", "low", "close"]]).all(axis=1)
        & (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )
    price_valid_num = price_valid.astype("float64")

    gap_tol = float(cfg["quality"].get("gap_tolerance_minutes", cfg["data"].get("bar_minutes", 10)))
    time_diff_min = df["time"].diff().dt.total_seconds() / 60.0
    single_gap = (time_diff_min > gap_tol).astype("float64")
    single_outlier = df.get("single_bar_outlier_flag", pd.Series(0, index=df.index)).fillna(0).astype("float64")

    bar_minutes = float(cfg["data"].get("bar_minutes", 10))
    min_obs_ratio = float(cfg["quality"].get("min_obs_ratio", 0.8))
    for w in windows:
        label = window_to_label(w)
        expected = window_to_bars(w, bar_minutes)
        obs_count = price_valid_num.rolling(w, min_periods=1).sum()
        missing_ratio = (1.0 - obs_count / expected).clip(lower=0.0, upper=1.0)
        gap_by_interval = single_gap.rolling(w, min_periods=1).max().fillna(0.0)
        gap_by_missing = (obs_count < expected).astype("float64")
        outlier_flag = single_outlier.rolling(w, min_periods=1).max().fillna(0.0)

        out[f"price_obs_count_{label}"] = obs_count
        out[f"price_expected_count_{label}"] = expected
        out[f"price_missing_ratio_{label}"] = missing_ratio
        out[f"price_gap_flag_{label}"] = ((gap_by_interval > 0) | (gap_by_missing > 0)).astype("int8")
        out[f"price_outlier_flag_{label}"] = (outlier_flag > 0).astype("int8")
        out[f"price_valid_window_{label}"] = (obs_count >= min_obs_ratio * expected).astype("int8")
    return out


def get_valid_window_mask(quality: pd.DataFrame, window: str, cfg: dict[str, Any]) -> pd.Series:
    label = window_to_label(window)
    col = f"price_valid_window_{label}"
    if col not in quality.columns:
        raise KeyError(f"Missing quality column {col}; compute quality for window {window} first.")
    return quality[col].astype(bool)
