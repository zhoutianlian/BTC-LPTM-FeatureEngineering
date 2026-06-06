from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality_features import get_valid_window_mask
from .utils import robust_zscore_train_or_past, window_to_hours, window_to_label


def compute_realized_vol(df: pd.DataFrame, quality: pd.DataFrame, cfg: dict[str, Any], windows: Iterable[str]) -> pd.DataFrame:
    """Compute RV_W = sqrt(sum r_i^2) and per-sqrt-hour RV."""
    out = pd.DataFrame(index=df.index)
    ret2 = df["ret_bps"].pow(2)
    rv_cfg = cfg.get("realized_vol", {})
    for w in windows:
        label = window_to_label(w)
        sum_sq = ret2.rolling(w, min_periods=1).sum()
        rv = np.sqrt(sum_sq)
        rv = rv.where(get_valid_window_mask(quality, w, cfg))
        hours = window_to_hours(w)
        rv_hour = np.sqrt(sum_sq / max(hours, 1e-12)).where(get_valid_window_mask(quality, w, cfg))
        out[f"realized_vol_{label}_bps"] = rv
        out[f"realized_vol_{label}_per_sqrt_hour_bps"] = rv_hour
        if rv_cfg.get("output_zscore", True):
            out[f"realized_vol_{label}_z"] = robust_zscore_train_or_past(rv, rv_cfg, default_lookback=rv_cfg.get("zscore_lookback", "30d"))
    return out
