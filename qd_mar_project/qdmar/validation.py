"""Data validation and no-future-leakage checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    name: str
    passed: bool
    details: str


def _result(name: str, passed: bool, details: str = "") -> ValidationResult:
    return ValidationResult(name=name, passed=bool(passed), details=details)


def check_timestamp_monotonicity(df: pd.DataFrame, time_col: str = "time") -> ValidationResult:
    times = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    ok = times.notna().all() and times.is_monotonic_increasing and not times.duplicated().any()
    details = f"rows={len(df)}, missing={times.isna().sum()}, duplicates={times.duplicated().sum()}"
    return _result("timestamp_monotonicity", ok, details)


def check_source_clock_alignment(
    df: pd.DataFrame,
    time_col: str = "time",
    source_time_col: str = "liq_feature_time",
) -> ValidationResult:
    if source_time_col not in df.columns:
        return _result("source_clock_alignment", True, f"{source_time_col} not present; skipped")
    event_time = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    liq_time = pd.to_datetime(df[source_time_col], utc=True, errors="coerce")
    ok = (liq_time <= event_time).fillna(False).all()
    max_age = ((event_time - liq_time).dt.total_seconds() / 60.0).max()
    return _result("source_clock_alignment", ok, f"max_age_min={max_age}")


def check_split_order(df: pd.DataFrame, split_col: str = "split", expected=("train", "validation", "test")) -> ValidationResult:
    if split_col not in df.columns:
        return _result("split_order", False, "split column missing")
    code = {s: i for i, s in enumerate(expected)}
    mapped = df[split_col].map(code)
    ok = mapped.notna().all() and mapped.is_monotonic_increasing
    details = str(df[split_col].value_counts(dropna=False).to_dict())
    return _result("split_order", ok, details)


def check_available_time(abs_df: pd.DataFrame) -> ValidationResult:
    if abs_df.empty:
        return _result("available_time", False, "absorption table is empty")
    event_time = pd.to_datetime(abs_df["event_time"], utc=True, errors="coerce")
    available_time = pd.to_datetime(abs_df["available_time"], utc=True, errors="coerce")
    ok = (available_time > event_time).fillna(False).all()
    return _result("available_time", ok, f"rows={len(abs_df)}")


def check_agent_inputs_no_future(agent_inputs: Iterable[str], forbidden_fragments: Iterable[str] | None = None) -> ValidationResult:
    fragments = list(forbidden_fragments or [
        "ret_", "future", "residual_raw", "absorption_raw", "transmission_ratio_raw",
        "available_time", "event_time", "plie_residual", "plie_absorption",
    ])
    bad = [c for c in agent_inputs if any(f in c for f in fragments)]
    return _result("agent_inputs_no_future_columns", len(bad) == 0, f"bad={bad}")


def check_memory_asof(memory_df: pd.DataFrame, abs_df: pd.DataFrame, main_horizon: str = "30m") -> ValidationResult:
    """Check that memory timestamps never precede their latest used matured timestamp."""
    latest_col = f"latest_abs_available_time_{main_horizon}"
    if memory_df.empty or abs_df.empty or latest_col not in memory_df.columns:
        return _result("memory_asof", True, "skipped: required columns missing")
    t = pd.to_datetime(memory_df["time"], utc=True, errors="coerce")
    latest = pd.to_datetime(memory_df[latest_col], utc=True, errors="coerce")
    ok = (latest.isna() | (latest <= t)).all()
    return _result("memory_asof", ok, f"memory_rows={len(memory_df)}")


def run_core_validations(
    df: pd.DataFrame,
    time_col: str,
    split_col: str,
    agent_inputs: list[str],
    expected_splits=("train", "validation", "test"),
    source_time_col: str = "liq_feature_time",
    forbidden_agent_input_fragments: Iterable[str] | None = None,
) -> list[ValidationResult]:
    return [
        check_timestamp_monotonicity(df, time_col=time_col),
        check_source_clock_alignment(df, time_col=time_col, source_time_col=source_time_col),
        check_split_order(df, split_col=split_col, expected=tuple(expected_splits)),
        check_agent_inputs_no_future(agent_inputs, forbidden_fragments=forbidden_agent_input_fragments),
    ]


def validation_results_to_frame(results: list[ValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])
