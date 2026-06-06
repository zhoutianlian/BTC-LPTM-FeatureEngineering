from __future__ import annotations

import numpy as np
import pandas as pd

from liq_dataflow.feature_engineering.config import DominanceConfig


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    sum_y = s.rolling(window=window, min_periods=window).sum()
    sum_xy = s.rolling(window=window, min_periods=window).apply(lambda x: np.dot(x, np.arange(window)), raw=True)
    numerator = window * sum_xy - (window * (window - 1) / 2) * sum_y
    denominator = (window**2 * (window**2 - 1)) / 12
    return numerator / denominator


def compute_dominance_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dom = pd.to_numeric(out["dominance"], errors="coerce").fillna(0).astype(int)
    change_points = dom != dom.shift(1)
    groups = change_points.cumsum()
    run_length = out.groupby(groups).cumcount() + 1

    out["dominance_duration"] = run_length
    out["dominance_duration_total"] = np.where(dom != 0, run_length, 0)
    out["dominance_last"] = dom.replace(0, np.nan).ffill().shift().fillna(0).astype(int)
    out["dominance_class"] = np.where(dom == -1, "Bear", np.where(dom == 1, "Bull", "Congestion"))
    out["dominance_prev"] = dom.shift(1).fillna(0).astype(int)
    out["is_keep"] = ((dom != 0) & (dom == out["dominance_prev"])).astype(int)
    out["is_strengthen"] = ((dom != 0) & (dom == out["dominance_last"])).astype(int)

    dominance_time = np.zeros(len(out), dtype=int)
    bull_count = 0
    bear_count = 0
    last_nonzero = 0
    prev_dom = 0
    for i, cur in enumerate(dom.to_numpy()):
        if cur == 1:
            if prev_dom != 1:
                if last_nonzero == -1:
                    bull_count = 0
                bull_count += 1
            dominance_time[i] = bull_count
            last_nonzero = 1
        elif cur == -1:
            if prev_dom != -1:
                if last_nonzero == 1:
                    bear_count = 0
                bear_count += 1
            dominance_time[i] = bear_count
            last_nonzero = -1
        prev_dom = cur
    out["dominance_time"] = dominance_time
    return out


def build_dominance_features(df_bins: pd.DataFrame, *, cfg: DominanceConfig) -> pd.DataFrame:
    df = df_bins.copy()
    diff = pd.to_numeric(df["diff_ls_cwt_kf"], errors="coerce").astype(float)
    rpn = pd.to_numeric(df["risk_priority_number"], errors="coerce").astype(float)
    bins = pd.to_numeric(df["bin_index"], errors="coerce").fillna(4).astype(int)
    fll = pd.to_numeric(df["fll_cwt_kf"], errors="coerce").astype(float)
    fsl = pd.to_numeric(df["fsl_cwt_kf"], errors="coerce").astype(float)

    df["thr_diff_pos"] = diff.rolling(window=cfg.rolling_window_bars, min_periods=cfg.rolling_min_periods).quantile(0.8)
    df["thr_diff_neg"] = diff.rolling(window=cfg.rolling_window_bars, min_periods=cfg.rolling_min_periods).quantile(0.2)
    df["thr_diff_pos_base"] = diff.rolling(window=cfg.rolling_window_bars, min_periods=cfg.rolling_min_periods).quantile(0.6)
    df["thr_diff_neg_base"] = diff.rolling(window=cfg.rolling_window_bars, min_periods=cfg.rolling_min_periods).quantile(0.4)
    df["fll_rolling_high"] = fll.rolling(window=cfg.rolling_window_bars, min_periods=cfg.rolling_min_periods).quantile(0.8)
    df["fsl_rolling_high"] = fsl.rolling(window=cfg.rolling_window_bars, min_periods=cfg.rolling_min_periods).quantile(0.8)

    bear = ((bins >= 5) & (diff >= df["thr_diff_pos"])) | ((bins == 4) & (rpn >= 0.52) & (diff >= df["thr_diff_pos_base"]))
    bull = ((bins <= 3) & (diff <= df["thr_diff_neg"])) | ((bins == 4) & (rpn <= 0.48) & (diff <= df["thr_diff_neg_base"]))
    df["dominance"] = np.select([bull, bear], [1, -1], default=0).astype(int)

    df["diff_beta_short"] = rolling_slope(diff, 3)
    df["diff_beta_8"] = rolling_slope(diff, 8)
    df["diff_beta_long"] = rolling_slope(diff, 10)
    df["diff_beta_24"] = rolling_slope(diff, 24)
    df["max_variation_diff"] = diff.diff().abs().rolling(window=24, min_periods=1).max()
    df["thr_diff_var"] = df["thr_diff_pos"] - df["thr_diff_neg"]
    df = compute_dominance_status(df)

    beta_prev = df["diff_beta_short"].shift(1)
    hit_ceiling = (
        (df["dominance"] == 1)
        & (diff <= df["thr_diff_neg"])
        & (fsl >= df["fsl_rolling_high"])
        & (beta_prev < 0)
        & (df["diff_beta_short"] >= 0)
    )
    hit_bottom = (
        (df["dominance"] == -1)
        & (diff >= df["thr_diff_pos"])
        & (fll >= df["fll_rolling_high"])
        & (beta_prev > 0)
        & (df["diff_beta_short"] <= 0)
    )
    df["hit_ceiling_bottom"] = np.select([hit_ceiling, hit_bottom], [-1, 1], default=0).astype(int)

    event_value = diff.where(df["hit_ceiling_bottom"] != 0)
    event_type = df["hit_ceiling_bottom"].replace(0, np.nan)
    last_event_value = event_value.ffill().shift(1)
    last_event_type = event_type.ffill().shift(1)
    event_idx = pd.Series(np.where(df["hit_ceiling_bottom"] != 0, np.arange(len(df)), np.nan), index=df.index).ffill().shift(1)
    bars_since_event = pd.Series(np.arange(len(df)), index=df.index) - event_idx
    within_window = (bars_since_event >= 1) & (bars_since_event <= cfg.reverse_window_bars)
    reverse_from_ceiling = (last_event_type == -1) & within_window & (diff <= (last_event_value - cfg.reverse_diff_threshold))
    reverse_from_bottom = (last_event_type == 1) & within_window & (diff >= (last_event_value + cfg.reverse_diff_threshold))
    df["reverse_ceiling_bottom"] = np.select([reverse_from_ceiling, reverse_from_bottom], [1, -1], default=0).astype(int)

    for col in ["hit_ceiling_bottom", "reverse_ceiling_bottom"]:
        recent_same = (df[col] != 0) & (df[col] == df[col].shift(1))
        recent_same |= (df[col] != 0) & (df[col] == df[col].shift(2))
        recent_same |= (df[col] != 0) & (df[col] == df[col].shift(3))
        df.loc[recent_same.fillna(False), col] = 0

    return df
