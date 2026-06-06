from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.config import load_config
from src.feature_pipeline import REQUIRED_OUTPUT_COLUMNS, build_features_from_dataframe


def make_ohlc(n: int = 220) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=n, freq="10min")
    close = 100 + np.cumsum(0.02 + 0.1 * np.sin(np.arange(n) / 15.0))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    return pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close})


def _cfg() -> dict:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    cfg["quality"]["robust_sigma_estimator"] = "iqr"
    cfg["jump"]["robust_sigma_estimator"] = "iqr"
    cfg["realized_vol"]["zscore_lookback"] = "1d"
    cfg["range"]["percentile_lookback"] = "1d"
    return cfg


def test_changing_future_rows_does_not_change_past_features() -> None:
    cfg = _cfg()
    raw = make_ohlc(220)
    cutoff = 120
    mutated = raw.copy()
    mutated.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 2.0
    features_a, _ = build_features_from_dataframe(raw, cfg)
    features_b, _ = build_features_from_dataframe(mutated, cfg)
    cols = [c for c in REQUIRED_OUTPUT_COLUMNS if c in features_a.columns]
    assert_frame_equal(
        features_a.loc[:cutoff, cols].reset_index(drop=True),
        features_b.loc[:cutoff, cols].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
