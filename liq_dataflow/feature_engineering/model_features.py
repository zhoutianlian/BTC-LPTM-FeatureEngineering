from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


EPS = 1e-12

MODEL_FEATURE_COLUMNS: List[str] = [
    "time",
    "price",
    "risk_priority_number",
    "bin_index",
    "dominance",
    "z_logTotalP",
    "z_sdom",
    "z_fll_cwt_kf",
    "z_fsl_cwt_kf",
]


@dataclass(frozen=True)
class LiquidationModelFeatureConfig:
    z_window_bars: int = 24


def _validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise ValueError("Input dataframe must contain 'time'.")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last").set_index("time")
    return out


def robust_rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    w = max(int(window), 1)
    minp = min_periods if min_periods is not None else max(20, w // 3)
    med = s.rolling(w, min_periods=minp).median()
    mad = (s - med).abs().rolling(w, min_periods=minp).median()
    return (s - med) / (1.4826 * mad + EPS)


def build_liquidation_model_features(
    df: pd.DataFrame,
    cfg: LiquidationModelFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build standardized liquidation features for the final delivery table.

    The historical model-layer alias ``RPN`` is intentionally normalized back to
    ``risk_priority_number`` so the final output has one canonical name.
    """
    cfg = cfg or LiquidationModelFeatureConfig()
    out = _validate_and_sort(df)

    required = [
        "price",
        "fll_cwt_kf",
        "fsl_cwt_kf",
        "total_ls_cwt_kf",
        "diff_dom_ls_cwt_kf",
        "risk_priority_number",
        "bin_index",
        "dominance",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns for model features: {missing}")

    price = pd.to_numeric(out["price"], errors="coerce").astype(float)
    fll = pd.to_numeric(out["fll_cwt_kf"], errors="coerce").astype(float).clip(lower=0.0)
    fsl = pd.to_numeric(out["fsl_cwt_kf"], errors="coerce").astype(float).clip(lower=0.0)
    total = pd.to_numeric(out["total_ls_cwt_kf"], errors="coerce").astype(float).clip(lower=0.0)
    sdom = pd.to_numeric(out["diff_dom_ls_cwt_kf"], errors="coerce").astype(float)
    rpn = pd.to_numeric(out["risk_priority_number"], errors="coerce").astype(float)
    bins = pd.to_numeric(out["bin_index"], errors="coerce").fillna(4).astype(int)
    dominance = pd.to_numeric(out["dominance"], errors="coerce").fillna(0).astype(int)

    feat = pd.DataFrame(index=out.index)
    feat["price"] = price
    feat["risk_priority_number"] = np.where(total > EPS, rpn, 0.5)
    feat["bin_index"] = bins.astype(float)
    feat["dominance"] = dominance.astype(float)

    feat["z_logTotalP"] = robust_rolling_zscore(np.log1p(total), cfg.z_window_bars)
    feat["z_sdom"] = robust_rolling_zscore(sdom, cfg.z_window_bars)
    feat["z_fll_cwt_kf"] = robust_rolling_zscore(np.log1p(fll), cfg.z_window_bars)
    feat["z_fsl_cwt_kf"] = robust_rolling_zscore(np.log1p(fsl), cfg.z_window_bars)

    feat = feat.reset_index().rename(columns={"index": "time"})
    return feat[MODEL_FEATURE_COLUMNS].copy()
