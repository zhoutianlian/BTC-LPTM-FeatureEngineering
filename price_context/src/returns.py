from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality_features import get_valid_window_mask
from .utils import parse_window_to_timedelta, window_to_label


def _anchor_close_asof(df: pd.DataFrame, window: str, tolerance: pd.Timedelta) -> pd.Series:
    """Find close at or just before t-window, never after t-window."""
    target = pd.DataFrame({
        "_row": np.arange(len(df)),
        "anchor_time": df["time"] - parse_window_to_timedelta(window),
    }).sort_values("anchor_time")
    source = df[["time", "close"]].rename(columns={"time": "source_time", "close": "anchor_close"}).sort_values("source_time")
    merged = pd.merge_asof(
        target,
        source,
        left_on="anchor_time",
        right_on="source_time",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    anchor = pd.Series(index=np.arange(len(df)), data=np.nan, dtype="float64")
    anchor.iloc[merged["_row"].to_numpy()] = merged["anchor_close"].to_numpy(dtype="float64")
    return pd.Series(anchor.to_numpy(), index=df.index)


def compute_past_returns(df: pd.DataFrame, quality: pd.DataFrame, cfg: dict[str, Any], windows: Iterable[str]) -> pd.DataFrame:
    """Compute past_return_W_bps = 10000 * log(close_t / close_{t-W})."""
    out = pd.DataFrame(index=df.index)
    tolerance = pd.to_timedelta(float(cfg["data"].get("bar_minutes", 10)) * 1.05, unit="min")
    for w in windows:
        label = window_to_label(w)
        anchor_close = _anchor_close_asof(df, w, tolerance=tolerance)
        ret = 10000.0 * np.log(df["close"] / anchor_close)
        ret = ret.replace([np.inf, -np.inf], np.nan)
        if f"price_valid_window_{label}" in quality.columns:
            ret = ret.where(get_valid_window_mask(quality, w, cfg))
        out[f"past_return_{label}_bps"] = ret
    return out
