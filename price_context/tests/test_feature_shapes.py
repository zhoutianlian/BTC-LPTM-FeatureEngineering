from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.feature_pipeline import REQUIRED_OUTPUT_COLUMNS, build_features_from_dataframe


def make_ohlc(n: int = 240) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=n, freq="10min")
    close = 100 + np.cumsum(np.sin(np.arange(n) / 12.0) * 0.05 + 0.01)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    return pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close})


def _cfg() -> dict:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    cfg["quality"]["robust_sigma_estimator"] = "iqr"
    cfg["jump"]["robust_sigma_estimator"] = "iqr"
    cfg["realized_vol"]["zscore_lookback"] = "1d"
    cfg["range"]["percentile_lookback"] = "1d"
    return cfg


def test_output_shape_and_required_columns() -> None:
    cfg = _cfg()
    raw = make_ohlc(240)
    features, report = build_features_from_dataframe(raw, cfg)
    assert len(features) == len(raw)
    assert report["rows_after"] == len(raw)
    assert all(c in features.columns for c in REQUIRED_OUTPUT_COLUMNS)
    assert features["time"].iloc[0] == raw["time"].iloc[0]
    assert features["price_feature_age_min"].eq(0).all()


def test_initial_long_window_return_is_missing_then_available() -> None:
    cfg = _cfg()
    raw = make_ohlc(240)
    features, _ = build_features_from_dataframe(raw, cfg)
    assert pd.isna(features.loc[0, "past_return_24h_bps"])
    assert pd.notna(features.loc[145, "past_return_24h_bps"])


def test_gap_flag_triggers_on_missing_bar() -> None:
    cfg = _cfg()
    raw = make_ohlc(80).drop(index=[20]).reset_index(drop=True)
    features, _ = build_features_from_dataframe(raw, cfg)
    assert features["price_gap_flag_1h"].sum() > 0
