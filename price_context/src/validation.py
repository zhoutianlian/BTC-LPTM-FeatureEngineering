from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    rows_before: int
    rows_after: int
    duplicate_time_count: int = 0
    missing_value_counts: dict[str, int] = field(default_factory=dict)
    non_positive_price_count: int = 0
    ohlc_inconsistency_count: int = 0
    non_monotonic_before_sort: bool = False
    gap_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "duplicate_time_count": self.duplicate_time_count,
            "missing_value_counts": self.missing_value_counts,
            "non_positive_price_count": self.non_positive_price_count,
            "ohlc_inconsistency_count": self.ohlc_inconsistency_count,
            "non_monotonic_before_sort": self.non_monotonic_before_sort,
            "gap_count": self.gap_count,
        }


def validate_required_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    inp = cfg["input"]
    required = [inp["time_column"], inp["open_column"], inp["high_column"], inp["low_column"], inp["close_column"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def parse_and_validate_time(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    inp = cfg["input"]
    time_col = inp["time_column"]
    fmt = inp.get("datetime_format")
    tz = inp.get("timezone")
    out[time_col] = pd.to_datetime(out[time_col], format=fmt, errors="coerce")
    if out[time_col].isna().any():
        bad_count = int(out[time_col].isna().sum())
        raise ValueError(f"Failed to parse {bad_count} values in time column {time_col!r}.")
    if tz:
        if out[time_col].dt.tz is None:
            out[time_col] = out[time_col].dt.tz_localize(tz)
        else:
            out[time_col] = out[time_col].dt.tz_convert(tz)
    return out


def standardize_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    inp = cfg["input"]
    rename_map = {
        inp["time_column"]: "time",
        inp["open_column"]: "open",
        inp["high_column"]: "high",
        inp["low_column"]: "low",
        inp["close_column"]: "close",
    }
    out = df.rename(columns=rename_map).copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["time", "open", "high", "low", "close"]]


def check_price_values(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[int, int, dict[str, int]]:
    price_cols = ["open", "high", "low", "close"]
    missing_counts = {c: int(df[c].isna().sum()) for c in price_cols}
    if any(v > 0 for v in missing_counts.values()):
        warnings.warn(f"Missing OHLC values detected and kept as NaN for quality flags: {missing_counts}", RuntimeWarning)

    finite_prices = df[price_cols].where(np.isfinite(df[price_cols]))
    non_positive_count = int((finite_prices <= 0).sum().sum())
    if non_positive_count > 0 and cfg["data"].get("fail_on_invalid_price", True):
        raise ValueError(f"Found {non_positive_count} non-positive OHLC price values.")
    if non_positive_count > 0:
        warnings.warn(f"Found {non_positive_count} non-positive OHLC price values.", RuntimeWarning)

    if cfg["data"].get("validate_ohlc", True):
        valid_or_nan = df[price_cols].notna().all(axis=1)
        inconsistent = valid_or_nan & (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        inconsistency_count = int(inconsistent.sum())
        if inconsistency_count > 0 and cfg["data"].get("fail_on_ohlc_inconsistency", True):
            raise ValueError(f"Found {inconsistency_count} OHLC consistency violations.")
        if inconsistency_count > 0:
            warnings.warn(f"Found {inconsistency_count} OHLC consistency violations.", RuntimeWarning)
    else:
        inconsistency_count = 0
    return non_positive_count, inconsistency_count, missing_counts


def sort_and_deduplicate(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, int, bool]:
    out = df.copy()
    non_monotonic = not out["time"].is_monotonic_increasing
    duplicate_count = int(out["time"].duplicated().sum())
    if duplicate_count > 0:
        msg = f"Found {duplicate_count} duplicate timestamps."
        if cfg["data"].get("drop_duplicate_time", True):
            warnings.warn(msg + f" Keeping {cfg['data'].get('duplicate_keep', 'last')!r} occurrence.", RuntimeWarning)
            out = out.drop_duplicates(subset=["time"], keep=cfg["data"].get("duplicate_keep", "last"))
        else:
            raise ValueError(msg + " Set data.drop_duplicate_time=true or remove duplicates.")
    if cfg["data"].get("sort_by_time", True):
        out = out.sort_values("time")
    elif non_monotonic:
        raise ValueError("time column is not monotonic increasing and data.sort_by_time=false.")
    out = out.reset_index(drop=True)
    return out, duplicate_count, non_monotonic


def count_time_gaps(df: pd.DataFrame, cfg: dict[str, Any]) -> int:
    tol_min = float(cfg["quality"].get("gap_tolerance_minutes", cfg["data"].get("bar_minutes", 10)))
    diff_min = df["time"].diff().dt.total_seconds() / 60.0
    return int((diff_min > tol_min).sum())


def validate_and_prepare_raw(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate schema, parse time, standardize columns, sort, and handle duplicates."""
    rows_before = len(df)
    validate_required_columns(df, cfg)
    parsed = parse_and_validate_time(df, cfg)
    standardized = standardize_columns(parsed, cfg)
    sorted_df, duplicate_count, non_monotonic = sort_and_deduplicate(standardized, cfg)
    non_positive, inconsistent, missing_counts = check_price_values(sorted_df, cfg)
    gap_count = count_time_gaps(sorted_df, cfg)
    if gap_count > 0:
        warnings.warn(f"Detected {gap_count} time gaps greater than configured tolerance.", RuntimeWarning)
    report = ValidationReport(
        rows_before=rows_before,
        rows_after=len(sorted_df),
        duplicate_time_count=duplicate_count,
        missing_value_counts=missing_counts,
        non_positive_price_count=non_positive,
        ohlc_inconsistency_count=inconsistent,
        non_monotonic_before_sort=non_monotonic,
        gap_count=gap_count,
    )
    return sorted_df, report


def assert_required_output_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise AssertionError(f"Output is missing required feature columns: {missing}")
