from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_stats import (
    compute_numeric_stats,
    compute_outlier_diagnostics,
    compute_temporal_diagnostics,
    rolling_window_from_config,
    to_numeric_series,
)
from .metadata import FeatureDefinition, safe_feature_filename


def evaluate_feature(
    df: pd.DataFrame,
    feature: FeatureDefinition,
    time_profile: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    report_cfg = cfg.get("report", {})
    min_valid_count = int(report_cfg.get("min_valid_count", 30))
    warn_missing_ratio = float(report_cfg.get("warn_missing_ratio", 0.20))
    fail_missing_ratio = float(report_cfg.get("fail_missing_ratio", 0.95))
    fail_inf_ratio = float(report_cfg.get("fail_inf_ratio", 0.05))
    rolling_window = rolling_window_from_config(cfg, len(df))

    result: dict[str, Any] = {
        "feature_name": feature.name,
        "category": feature.category,
        "description": feature.description,
        "documented": feature.documented,
        "actual_output": feature.actual_output or feature.name in df.columns,
        "important": feature.important,
        "exists": feature.name in df.columns,
        "detail_link": f"features/{safe_feature_filename(feature.name)}.html",
        "dtype": None,
        "is_numeric": False,
        "status": "PASS",
        "issues": [],
        "stats": {},
        "outliers": {},
        "temporal": {},
        "time_checks": time_profile,
    }

    if feature.name not in df.columns:
        result["dtype"] = "missing"
        result["status"] = "FAIL"
        result["issues"].append("Missing documented output column.")
        return result

    series = df[feature.name]
    result["dtype"] = str(series.dtype)
    numeric = to_numeric_series(series)
    numeric_non_na = int(numeric.notna().sum())
    original_non_na = int(series.notna().sum())
    numeric_ratio = numeric_non_na / original_non_na if original_non_na else 0.0
    result["is_numeric"] = bool(numeric_ratio >= 0.80)

    if not result["is_numeric"] and feature.expected_numeric:
        result["status"] = "FAIL"
        result["issues"].append(f"Expected numeric values but only {numeric_ratio:.1%} of non-null values are numeric.")
        return result

    time = pd.to_datetime(df["time"], errors="coerce") if "time" in df.columns else None
    stats = compute_numeric_stats(series)
    outliers = compute_outlier_diagnostics(series, time, cfg)
    temporal = compute_temporal_diagnostics(series, rolling_window)
    result["stats"] = stats
    result["outliers"] = outliers
    result["temporal"] = temporal

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    if time_profile.get("duplicate_time_count", 0):
        fail_reasons.append("Duplicate timestamps exist in the feature output.")
    if time_profile.get("is_monotonic_increasing") is False:
        fail_reasons.append("Time column is not monotonic increasing.")
    if stats["count"] == 0:
        fail_reasons.append("Column has no finite numeric observations.")
    elif stats["count"] < min_valid_count:
        fail_reasons.append(f"Valid sample count is below configured minimum ({min_valid_count}).")
    if stats["missing_ratio"] >= fail_missing_ratio:
        fail_reasons.append(f"Missing ratio is at least {fail_missing_ratio:.0%}.")
    elif stats["missing_ratio"] > warn_missing_ratio:
        warn_reasons.append(f"Missing ratio is above {warn_missing_ratio:.0%}.")
    if stats["inf_ratio"] >= fail_inf_ratio:
        fail_reasons.append(f"Inf ratio is at least {fail_inf_ratio:.0%}.")
    elif stats["inf_count"] > 0:
        warn_reasons.append("Inf or -inf values were found.")
    if stats["constant_flag"] and stats["count"] >= min_valid_count:
        warn_reasons.append("Column is approximately constant.")
    if outliers["zscore_outlier_count"] > 0:
        warn_reasons.append("Z-score outliers were detected.")
    if outliers["cliff_jump_count"] > 0:
        warn_reasons.append("Large one-step jumps were detected.")
    if outliers["explosion_flag"]:
        warn_reasons.append("Max absolute value is far above the 99th percentile scale.")
    if temporal["long_constant_flag"]:
        warn_reasons.append("A long unchanged segment was detected.")
    if temporal["long_zero_flag"]:
        warn_reasons.append("A long all-zero segment was detected.")
    if temporal["long_null_flag"]:
        warn_reasons.append("A long null segment was detected.")
    if time_profile.get("abnormal_interval_count", 0):
        warn_reasons.append("Abnormal time interval jumps exist in the output time axis.")

    if _looks_like_probability_feature(feature.name, stats):
        warn_reasons.append("Value range extends outside expected [0, 1] bounds.")
    if _looks_like_direction_feature(feature.name, series):
        warn_reasons.append("Direction feature contains values outside {-1, 0, 1}.")
    if _looks_like_flag_feature(feature.name, series):
        warn_reasons.append("Flag feature contains values outside {0, 1}.")

    if fail_reasons:
        result["status"] = "FAIL"
        result["issues"].extend(fail_reasons + warn_reasons)
    elif warn_reasons:
        result["status"] = "WARN"
        result["issues"].extend(warn_reasons)
    else:
        result["status"] = "PASS"

    return result


def compact_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    stats = result.get("stats", {})
    outliers = result.get("outliers", {})
    return {
        "feature_name": result["feature_name"],
        "category": result["category"],
        "exists": result["exists"],
        "dtype": result["dtype"],
        "valid_count": stats.get("count"),
        "missing_ratio": stats.get("missing_ratio"),
        "inf_count": stats.get("inf_count"),
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "min": stats.get("min"),
        "max": stats.get("max"),
        "p01": stats.get("p01"),
        "p50": stats.get("median"),
        "p99": stats.get("p99"),
        "skew": stats.get("skew"),
        "kurtosis": stats.get("kurtosis"),
        "outlier_count": outliers.get("zscore_outlier_count"),
        "constant_flag": stats.get("constant_flag"),
        "status": result["status"],
        "detail_link": result["detail_link"],
    }


def _looks_like_probability_feature(name: str, stats: dict[str, Any]) -> bool:
    bounded_tokens = [
        "compression",
        "efficiency",
        "strength",
        "consistency",
        "align",
        "r2",
        "ratio_bv",
        "proxy",
        "missing_ratio",
    ]
    if not any(token in name for token in bounded_tokens):
        return False
    min_v = stats.get("min")
    max_v = stats.get("max")
    if min_v is None or max_v is None:
        return False
    return bool(float(min_v) < -1e-9 or float(max_v) > 1.0 + 1e-9)


def _looks_like_direction_feature(name: str, series: pd.Series) -> bool:
    if "direction" not in name:
        return False
    values = set(np.unique(to_numeric_series(series).dropna().to_numpy()))
    return not values.issubset({-1.0, 0.0, 1.0})


def _looks_like_flag_feature(name: str, series: pd.Series) -> bool:
    if not name.endswith("_flag") and "_flag_" not in name:
        return False
    values = set(np.unique(to_numeric_series(series).dropna().to_numpy()))
    return not values.issubset({0.0, 1.0})
