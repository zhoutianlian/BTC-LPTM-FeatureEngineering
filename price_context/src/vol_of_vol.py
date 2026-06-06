from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality_features import get_valid_window_mask
from .utils import EPSILON, min_periods_for_window, robust_zscore_train_or_past, window_to_label


def compute_vol_of_vol(
    realized_vol: pd.DataFrame,
    quality: pd.DataFrame,
    cfg: dict[str, Any],
    windows: Iterable[str],
) -> pd.DataFrame:
    """Compute coefficient of variation of rolling 1h RV inside longer windows."""
    out = pd.DataFrame(index=realized_vol.index)
    rv1 = realized_vol["realized_vol_1h_bps"]
    bar_minutes = int(cfg["data"].get("bar_minutes", 10))
    min_obs_ratio = float(cfg["quality"].get("min_obs_ratio", 0.8))
    z_cfg = {
        "zscore_method": cfg.get("realized_vol", {}).get("zscore_method", "train_robust"),
        "zscore_clip": cfg.get("realized_vol", {}).get("zscore_clip", [-5, 5]),
        "zscore_lookback": cfg.get("realized_vol", {}).get("zscore_lookback", "30d"),
        "train_split": cfg.get("realized_vol", {}).get("train_split", {}),
        "fallback_when_train_missing": cfg.get("realized_vol", {}).get("fallback_when_train_missing", "past_rolling_robust"),
    }
    for w in windows:
        label = window_to_label(w)
        min_periods = min_periods_for_window(w, bar_minutes, min_obs_ratio)
        roll = rv1.rolling(w, min_periods=1)
        count = roll.count()
        mean = roll.mean()
        std = roll.std(ddof=0)
        valid = (count >= min_periods) & get_valid_window_mask(quality, w, cfg)
        vov_abs = std.where(valid)
        vov = (std / (mean + EPSILON)).where(valid)
        out[f"vol_of_vol_{label}"] = vov
        out[f"vol_of_vol_abs_{label}"] = vov_abs
        out[f"vol_of_vol_{label}_z"] = robust_zscore_train_or_past(vov, z_cfg, default_lookback=z_cfg.get("zscore_lookback", "30d"))
    return out
