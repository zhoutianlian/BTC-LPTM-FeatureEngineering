from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_datetime_index, robust_sigma_past


def add_base_price_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add log close and 10-minute log return in bps.

    r_i = 10000 * log(close_i / close_{i-1}). No future values are used.
    """
    out = ensure_datetime_index(df.copy(), "time")
    out["log_close"] = np.log(out["close"])
    out["ret_bps"] = 10000.0 * out["log_close"].diff()

    quality_cfg = cfg["quality"]
    if quality_cfg.get("outlier_method", "rolling_robust") == "rolling_robust":
        sigma = robust_sigma_past(
            out["ret_bps"],
            lookback=quality_cfg.get("outlier_lookback", "7d"),
            bar_minutes=int(cfg["data"].get("bar_minutes", 10)),
            estimator=quality_cfg.get("robust_sigma_estimator", "mad"),
        )
        out["ret_robust_sigma_bps"] = sigma
        out["ret_outlier_z"] = (out["ret_bps"].abs() / (sigma + 1e-12)).replace([np.inf, -np.inf], np.nan)
        out["single_bar_outlier_flag"] = (out["ret_outlier_z"] > float(quality_cfg.get("outlier_z_threshold", 5.0))).astype("int8")
        if quality_cfg.get("winsorize_outliers", False):
            threshold = float(quality_cfg.get("outlier_z_threshold", 5.0)) * sigma
            out["ret_bps_raw"] = out["ret_bps"]
            out["ret_bps"] = out["ret_bps"].clip(lower=-threshold, upper=threshold)
    else:
        out["ret_robust_sigma_bps"] = np.nan
        out["ret_outlier_z"] = np.nan
        out["single_bar_outlier_flag"] = 0
    return out
