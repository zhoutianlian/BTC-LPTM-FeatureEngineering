from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.validation import validate_and_prepare_raw


def _cfg() -> dict:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    cfg["quality"]["robust_sigma_estimator"] = "iqr"
    cfg["jump"]["robust_sigma_estimator"] = "iqr"
    return cfg


def test_missing_required_column_raises() -> None:
    cfg = _cfg()
    df = pd.DataFrame({"time": ["2024-01-01"], "open": [1.0], "high": [1.0], "low": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_and_prepare_raw(df, cfg)


def test_invalid_ohlc_relationship_raises() -> None:
    cfg = _cfg()
    df = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00:00"],
            "open": [100.0],
            "high": [99.0],
            "low": [98.0],
            "close": [100.0],
        }
    )
    with pytest.raises(ValueError, match="OHLC consistency"):
        validate_and_prepare_raw(df, cfg)


def test_duplicate_timestamp_can_be_dropped() -> None:
    cfg = _cfg()
    df = pd.DataFrame(
        {
            "time": ["2024-01-01 00:10:00", "2024-01-01 00:00:00", "2024-01-01 00:00:00"],
            "open": [101.0, 100.0, 100.5],
            "high": [102.0, 101.0, 101.0],
            "low": [100.0, 99.0, 99.0],
            "close": [101.0, 100.0, 100.8],
        }
    )
    prepared, report = validate_and_prepare_raw(df, cfg)
    assert report.duplicate_time_count == 1
    assert prepared["time"].is_monotonic_increasing
    assert len(prepared) == 2
