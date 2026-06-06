"""Rolling matured QD-MAR memory features for online Agent input.

The first project version used ignore-na EWM for directional absorption scores.
That is mathematically causal, but operationally dangerous: because only about
30% of rows are directional-core, a strong old absorption reading can remain
visible for many hours or days when PLIE is neutral or low quality.  This module
therefore keeps the original diagnostic EWM fields, and additionally produces
staleness-aware Agent inputs that decay directional evidence back to neutral
when no fresh directional-core absorption has matured.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .config import Config


def _rolling_mode_object(series: pd.Series, window: int) -> pd.Series:
    """Rolling mode for object/categorical labels, using only past rows."""
    out = []
    vals = series.to_list()
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        s = pd.Series(vals[start:i + 1]).dropna()
        out.append(s.value_counts().index[0] if not s.empty else None)
    return pd.Series(out, index=series.index, dtype=object)


def _decay_freshness(age_hours: pd.Series, halflife_hours: float) -> pd.Series:
    """Convert age in hours to freshness in [0, 1] using exponential half-life."""
    age = pd.to_numeric(age_hours, errors="coerce")
    if halflife_hours <= 0:
        return pd.Series(np.where(age.notna(), 1.0, 0.0), index=age.index)
    out = np.exp(-math.log(2.0) * age / halflife_hours)
    out = pd.Series(out, index=age.index).clip(lower=0.0, upper=1.0)
    out[age.isna()] = 0.0
    return out


def build_memory_features(base_df: pd.DataFrame, event_df: pd.DataFrame, curve_df: pd.DataFrame, cfg: Config, path_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build online-usable rolling memory from matured absorption rows.

    For current time ``T`` the function uses ``merge_asof`` with
    ``available_time <= T``.  No event-level absorption label can enter Agent
    input before its horizon has matured.

    Two memory families are produced:

    1. Diagnostic raw EWMs such as ``mar_abs_score_q_ewm_6_30m``.  They preserve
       the original absorption series and ignore NaNs.
    2. Staleness-aware Agent fields such as
       ``mar_abs_score_q_staleaware_ewm_6_30m``.  They decay directional
       evidence toward neutral when the last directional-core event is old.
    """
    time_col = cfg.get("data", "time_col", default="time")
    main_h = cfg.get("memory", "main_horizon", default="30m")
    spans = cfg.get("memory", "ewm_spans", default=[3, 6, 12])
    persistence_window = int(cfg.get("memory", "persistence_window", default=6))
    takeover_window = int(cfg.get("memory", "takeover_window", default=12))
    curve_mode_window = int(cfg.get("memory", "curve_mode_window", default=6))
    halflife_hours = float(cfg.get("memory", "directional_decay_halflife_hours", default=24.0))

    current = base_df[[time_col]].copy().rename(columns={time_col: "time"})
    current["time"] = pd.to_datetime(current["time"], utc=True)
    current = current.sort_values("time")

    seq = event_df[event_df["horizon"].eq(main_h)].copy()
    seq["available_time"] = pd.to_datetime(seq["available_time"], utc=True)
    seq = seq.sort_values("available_time").reset_index(drop=True)

    if seq.empty:
        mem = current.copy()
        extra_cols = [c for c in ["price", "hmm_state", "plie_main_bps", "plie_reliability", "plie_direction", "plie_phase", "split"] if c in base_df.columns]
        return mem.merge(base_df[[time_col] + extra_cols].rename(columns={time_col: "time"}), on="time", how="left")

    # Latest matured event-level values.
    seq[f"mar_abs_score_q_last_{main_h}"] = seq["absorption_score_q_0_100"]
    seq[f"mar_active_force_aligned_last_{main_h}"] = seq["active_force_aligned_score"]
    seq[f"mar_active_force_price_last_{main_h}"] = seq["active_force_price_score"]
    seq[f"mar_neutral_active_strength_last_{main_h}"] = seq["neutral_active_strength_score_0_100"]
    seq[f"mar_market_response_label_last_{main_h}"] = seq["market_response_label"]

    # Directional-core freshness: available_time order is the true maturity order.
    seq["_is_directional_core"] = seq["response_context"].eq("directional_core")
    seq[f"last_directional_core_available_time_{main_h}"] = seq["available_time"].where(seq["_is_directional_core"]).ffill()
    seq[f"mar_directional_core_age_hours_{main_h}"] = (
        seq["available_time"] - seq[f"last_directional_core_available_time_{main_h}"]
    ).dt.total_seconds() / 3600.0
    seq[f"mar_directional_core_freshness_{main_h}"] = _decay_freshness(
        seq[f"mar_directional_core_age_hours_{main_h}"], halflife_hours
    )

    # Context quality evidence decays naturally because non-core rows have zero directional evidence.
    seq["_directional_quality_obs"] = np.where(seq["_is_directional_core"], seq["quality_weight"].fillna(0.0), 0.0)

    for span in spans:
        # Original diagnostic fields, kept for audit/backward compatibility.
        seq[f"mar_abs_score_q_ewm_{span}_{main_h}"] = (
            seq["absorption_score_q_0_100"].ewm(span=span, adjust=False, ignore_na=True).mean()
        )
        seq[f"mar_active_force_aligned_ewm_{span}_{main_h}"] = (
            seq["active_force_aligned_score"].ewm(span=span, adjust=False, ignore_na=True).mean()
        )
        seq[f"mar_active_force_price_ewm_{span}_{main_h}"] = (
            seq["active_force_price_score"].ewm(span=span, adjust=False, ignore_na=True).mean()
        )
        seq[f"mar_neutral_active_strength_ewm_{span}_{main_h}"] = (
            seq["neutral_active_strength_score_0_100"].ewm(span=span, adjust=False, ignore_na=True).mean()
        )

        # Staleness-aware fields recommended for Agent input.
        freshness = seq[f"mar_directional_core_freshness_{main_h}"].fillna(0.0)
        seq[f"mar_abs_score_q_staleaware_ewm_{span}_{main_h}"] = (
            50.0 + freshness * (seq[f"mar_abs_score_q_ewm_{span}_{main_h}"] - 50.0)
        )
        seq[f"mar_active_force_aligned_staleaware_ewm_{span}_{main_h}"] = (
            freshness * seq[f"mar_active_force_aligned_ewm_{span}_{main_h}"].fillna(0.0)
        )
        seq[f"mar_active_force_price_staleaware_ewm_{span}_{main_h}"] = (
            freshness * seq[f"mar_active_force_price_ewm_{span}_{main_h}"].fillna(0.0)
        )
        seq[f"mar_directional_quality_ewm_{span}_{main_h}"] = (
            seq["_directional_quality_obs"].ewm(span=span, adjust=False, ignore_na=False).mean()
        )

        # Neutral-active evidence should not persist through directional-core rows.
        neutral_obs = seq["neutral_active_strength_score_0_100"].where(~seq["_is_directional_core"], 0.0).fillna(0.0)
        seq[f"mar_neutral_active_strength_evidence_ewm_{span}_{main_h}"] = (
            neutral_obs.ewm(span=span, adjust=False, ignore_na=False).mean()
        )

    labels_str = seq["market_response_label"].fillna("").astype(str)
    amp = labels_str.str.contains("amplification|cascade", regex=True).astype(float)
    abs_like = labels_str.str.contains("absorption|stall|rejection|takeover", regex=True).astype(float)
    takeover = labels_str.str.contains("takeover|rejection", regex=True).astype(float)
    neutral_strong = labels_str.str.contains("strong_active_move|extreme_active_move", regex=True).astype(float)
    seq[f"mar_amplification_persistence_{persistence_window}_{main_h}"] = amp.rolling(persistence_window, min_periods=1).mean()
    seq[f"mar_absorption_persistence_{persistence_window}_{main_h}"] = abs_like.rolling(persistence_window, min_periods=1).mean()
    seq[f"mar_takeover_count_{takeover_window}_{main_h}"] = takeover.rolling(takeover_window, min_periods=1).sum()
    seq[f"mar_neutral_strong_active_count_{takeover_window}_{main_h}"] = neutral_strong.rolling(takeover_window, min_periods=1).sum()
    seq[f"mar_neutral_context_persistence_{takeover_window}_{main_h}"] = (~seq["_is_directional_core"]).astype(float).rolling(takeover_window, min_periods=1).mean()

    seq[f"latest_abs_available_time_{main_h}"] = seq["available_time"]

    keep_cols = [
        "available_time",
        f"latest_abs_available_time_{main_h}",
        f"last_directional_core_available_time_{main_h}",
        f"mar_directional_core_age_hours_{main_h}",
        f"mar_directional_core_freshness_{main_h}",
        f"mar_abs_score_q_last_{main_h}",
        f"mar_active_force_aligned_last_{main_h}",
        f"mar_active_force_price_last_{main_h}",
        f"mar_neutral_active_strength_last_{main_h}",
        f"mar_market_response_label_last_{main_h}",
        f"mar_amplification_persistence_{persistence_window}_{main_h}",
        f"mar_absorption_persistence_{persistence_window}_{main_h}",
        f"mar_takeover_count_{takeover_window}_{main_h}",
        f"mar_neutral_strong_active_count_{takeover_window}_{main_h}",
        f"mar_neutral_context_persistence_{takeover_window}_{main_h}",
    ]
    for span in spans:
        keep_cols.extend([
            f"mar_abs_score_q_ewm_{span}_{main_h}",
            f"mar_active_force_aligned_ewm_{span}_{main_h}",
            f"mar_active_force_price_ewm_{span}_{main_h}",
            f"mar_neutral_active_strength_ewm_{span}_{main_h}",
            f"mar_abs_score_q_staleaware_ewm_{span}_{main_h}",
            f"mar_active_force_aligned_staleaware_ewm_{span}_{main_h}",
            f"mar_active_force_price_staleaware_ewm_{span}_{main_h}",
            f"mar_directional_quality_ewm_{span}_{main_h}",
            f"mar_neutral_active_strength_evidence_ewm_{span}_{main_h}",
        ])

    mem = pd.merge_asof(
        current,
        seq[keep_cols].sort_values("available_time"),
        left_on="time",
        right_on="available_time",
        direction="backward",
    )
    mem = mem.drop(columns=["available_time"], errors="ignore")

    curve = curve_df.copy().sort_values("available_time")
    if not curve.empty:
        curve["available_time"] = pd.to_datetime(curve["available_time"], utc=True)
        curve["mar_curve_label_last"] = curve["mar_curve_label"]
        curve["mar_curve_label_mode_6"] = _rolling_mode_object(curve["mar_curve_label"], curve_mode_window)
        curve_keep = ["available_time", "mar_curve_label_last", "mar_curve_label_mode_6", "mar_response_conflict_score"]
        mem = pd.merge_asof(
            mem.sort_values("time"),
            curve[curve_keep].sort_values("available_time"),
            left_on="time",
            right_on="available_time",
            direction="backward",
        ).drop(columns=["available_time"], errors="ignore")
    else:
        mem["mar_curve_label_last"] = None
        mem["mar_curve_label_mode_6"] = None
        mem["mar_response_conflict_score"] = np.nan

    # Merge online-available path-level / episode-level absorption features.
    # These rows have available_time == current source-clock time and are causal
    # because they only use historical/current price and PLIE pressure.
    if path_df is not None and not path_df.empty:
        pseq = path_df.copy()
        pseq["available_time"] = pd.to_datetime(pseq["available_time"], utc=True)
        wide = None
        selected = [
            "path_absorption_score_0_100",
            "path_pressure_rejection_score",
            "path_cascade_score",
            "path_active_force_aligned_score",
            "path_active_force_price_score",
            "path_quality",
            "path_data_quality",
            "path_signal_clarity",
            "path_activity_level",
            "path_divergence_score",
            "path_transmission_score",
            "path_direction_consistency",
            "path_liq_neutrality_score",
            "path_active_z",
            "path_active_dominance_score",
            "path_active_dominance_direction",
            "path_active_dominance_price_score",
            "path_active_context_quality",
            "path_context",
            "path_label",
        ]
        for wh, grp in pseq.groupby("window_hours"):
            suffix = f"{int(wh)}h"
            g = grp[["available_time"] + [c for c in selected if c in grp.columns]].copy()
            g = g.rename(columns={
                "path_absorption_score_0_100": f"mar_episode_abs_score_{suffix}",
                "path_pressure_rejection_score": f"mar_episode_pressure_rejection_score_{suffix}",
                "path_cascade_score": f"mar_episode_cascade_score_{suffix}",
                "path_active_force_aligned_score": f"mar_episode_active_force_aligned_{suffix}",
                "path_active_force_price_score": f"mar_episode_active_force_price_{suffix}",
                "path_quality": f"mar_episode_quality_{suffix}",
                "path_data_quality": f"mar_episode_data_quality_{suffix}",
                "path_signal_clarity": f"mar_episode_signal_clarity_{suffix}",
                "path_activity_level": f"mar_episode_activity_level_{suffix}",
                "path_divergence_score": f"mar_episode_divergence_score_{suffix}",
                "path_transmission_score": f"mar_episode_transmission_score_{suffix}",
                "path_direction_consistency": f"mar_episode_direction_consistency_{suffix}",
                "path_liq_neutrality_score": f"mar_episode_liq_neutrality_score_{suffix}",
                "path_active_z": f"mar_episode_active_z_{suffix}",
                "path_active_dominance_score": f"mar_episode_active_dominance_score_{suffix}",
                "path_active_dominance_direction": f"mar_episode_active_dominance_direction_{suffix}",
                "path_active_dominance_price_score": f"mar_episode_active_dominance_price_score_{suffix}",
                "path_active_context_quality": f"mar_episode_active_context_quality_{suffix}",
                "path_context": f"mar_episode_context_{suffix}",
                "path_label": f"mar_episode_label_{suffix}",
            })
            wide = g if wide is None else wide.merge(g, on="available_time", how="outer")
        if wide is not None and not wide.empty:
            mem = pd.merge_asof(
                mem.sort_values("time"),
                wide.sort_values("available_time"),
                left_on="time",
                right_on="available_time",
                direction="backward",
            ).drop(columns=["available_time"], errors="ignore")

    # Keep context columns for visualization and evaluation.
    extra_cols = [c for c in ["price", "hmm_state", "plie_main_bps", "plie_reliability", "plie_direction", "plie_phase", "split"] if c in base_df.columns]
    mem = mem.merge(base_df[[time_col] + extra_cols].rename(columns={time_col: "time"}), on="time", how="left")
    return mem
