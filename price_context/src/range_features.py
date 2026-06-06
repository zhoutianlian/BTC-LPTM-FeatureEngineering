from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality_features import get_valid_window_mask
from .utils import EPSILON, past_rolling_percentile, window_to_label


def compute_range_features(
    df: pd.DataFrame,
    quality: pd.DataFrame,
    realized_vol: pd.DataFrame,
    cfg: dict[str, Any],
    windows: Iterable[str],
) -> pd.DataFrame:
    """Compute OHLC range width, past-only compression, and range-to-vol ratio."""
    out = pd.DataFrame(index=df.index)
    range_cfg = cfg.get("range", {})
    compression_method = range_cfg.get("compression_method", "past_percentile")
    if compression_method != "past_percentile":
        raise ValueError("range.compression_method currently supports only 'past_percentile'.")
    pct_lookback = range_cfg.get("percentile_lookback", "30d")
    for w in windows:
        label = window_to_label(w)
        max_high = df["high"].rolling(w, min_periods=1).max()
        min_low = df["low"].rolling(w, min_periods=1).min()
        width = 10000.0 * np.log(max_high / min_low)
        width = width.replace([np.inf, -np.inf], np.nan).where(get_valid_window_mask(quality, w, cfg))
        out[f"range_width_{label}_bps"] = width

        pct = past_rolling_percentile(width, lookback=pct_lookback, min_periods=20)
        out[f"range_compression_{label}"] = (1.0 - pct).clip(0.0, 1.0)
        if range_cfg.get("output_range_to_vol", True):
            rv_col = f"realized_vol_{label}_bps"
            if rv_col in realized_vol.columns:
                out[f"range_to_vol_{label}"] = width / (realized_vol[rv_col] + EPSILON)
    return out
