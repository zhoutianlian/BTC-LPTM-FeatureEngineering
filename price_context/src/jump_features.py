from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality_features import get_valid_window_mask
from .utils import EPSILON, robust_sigma_past, signed_max_abs, squash, window_to_label


def _get_jump_sigma(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    jump_cfg = cfg.get("jump", {})
    quality_cfg = cfg.get("quality", {})
    if (
        "ret_robust_sigma_bps" in df.columns
        and jump_cfg.get("robust_sigma_lookback", "7d") == quality_cfg.get("outlier_lookback", "7d")
        and jump_cfg.get("robust_sigma_estimator", "mad") == quality_cfg.get("robust_sigma_estimator", "mad")
    ):
        return df["ret_robust_sigma_bps"]
    return robust_sigma_past(
        df["ret_bps"],
        lookback=jump_cfg.get("robust_sigma_lookback", "7d"),
        bar_minutes=int(cfg["data"].get("bar_minutes", 10)),
        estimator=jump_cfg.get("robust_sigma_estimator", "mad"),
    )


def compute_jump_features(df: pd.DataFrame, quality: pd.DataFrame, cfg: dict[str, Any], windows: Iterable[str]) -> pd.DataFrame:
    """Compute robust jump z-score, jump counts, bipower jump ratio, and composite proxy."""
    out = pd.DataFrame(index=df.index)
    jump_cfg = cfg.get("jump", {})
    sigma = _get_jump_sigma(df, cfg)
    jump_z = (df["ret_bps"].abs() / (sigma + EPSILON)).replace([np.inf, -np.inf], np.nan)
    theta = float(jump_cfg.get("jump_z_threshold", 5.0))
    c = float(jump_cfg.get("squash_c", 2.0))
    weights = jump_cfg.get("proxy_weights", {})

    ret2 = df["ret_bps"].pow(2)
    abs_ret = df["ret_bps"].abs()
    bp_term = abs_ret * abs_ret.shift(1)
    for w in windows:
        label = window_to_label(w)
        valid = get_valid_window_mask(quality, w, cfg)
        max_jump_z = jump_z.rolling(w, min_periods=1).max().where(valid)
        jump_count = (jump_z > theta).astype("float64").rolling(w, min_periods=1).sum().where(valid)
        out[f"max_jump_z_{label}"] = max_jump_z
        out[f"jump_count_{label}"] = jump_count

        rv_sum = ret2.rolling(w, min_periods=1).sum()
        bv = (np.pi / 2.0) * bp_term.rolling(w, min_periods=1).sum()
        jump_var = (rv_sum - bv).clip(lower=0.0)
        jump_ratio = (jump_var / (rv_sum + EPSILON)).clip(0.0, 1.0).where(valid)
        if jump_cfg.get("output_bipower_ratio", True):
            out[f"jump_ratio_bv_{label}"] = jump_ratio

        if jump_cfg.get("output_signed_max_jump", True):
            signed = df["ret_bps"].rolling(w, min_periods=1).apply(signed_max_abs, raw=True).where(valid)
            out[f"signed_max_jump_return_{label}_bps"] = signed

        proxy = (
            float(weights.get("max_jump_z", 0.50)) * pd.Series(squash(max_jump_z.fillna(0.0), c=c), index=df.index)
            + float(weights.get("jump_ratio_bv", 0.30)) * jump_ratio.fillna(0.0).clip(0.0, 1.0)
            + float(weights.get("jump_count", 0.20)) * pd.Series(squash(jump_count.fillna(0.0), c=c), index=df.index)
        )
        out[f"jump_proxy_{label}"] = proxy.where(valid).clip(0.0, 1.0)
    return out
