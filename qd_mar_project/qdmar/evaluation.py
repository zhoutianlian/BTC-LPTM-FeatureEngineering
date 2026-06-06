"""Evaluation metrics for QD-MAR outputs.

QD-MAR is not a generic return predictor, so evaluation focuses on:
1. no-leakage and maturity correctness,
2. PLIE q65 consistency in valid directional context,
3. calibrated response percentile stability,
4. financial consistency by state/response context,
5. Agent-memory usability, staleness, and redundancy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def q65_coverage(event_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate P(Y <= Braw) by horizon/split/context.

    Because PLIE raw magnitude is a q65 passive reference, directional-core
    samples should have coverage near 0.65 if calibration remains consistent.
    Neutral rows are reported as ``coverage_applicable=False`` because q65
    directional absorption is not meaningful when PLIE has no valid direction.
    """
    df = event_df.copy()
    df["coverage_applicable"] = df["response_context"].isin(["directional_core", "directional_weak", "low_quality_plie"])
    df["covered"] = df["aligned_actual_response_bps"] <= df["plie_reference_raw_bps"]
    group_cols = ["horizon", "split", "response_context"]
    out = df.groupby(group_cols).agg(
        n=("covered", "size"),
        coverage=("covered", "mean"),
        median_snr=("snr_raw", "median"),
        median_reliability=("plie_reliability", "median"),
        coverage_applicable=("coverage_applicable", "first"),
    ).reset_index()
    out.loc[~out["coverage_applicable"], "coverage"] = np.nan
    return out


def context_distribution(event_df: pd.DataFrame) -> pd.DataFrame:
    """Response-context proportions by horizon/split."""
    cols = ["horizon", "split", "response_context"]
    counts = event_df.groupby(cols).size().rename("n").reset_index()
    totals = counts.groupby(["horizon", "split"])["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(["horizon", "split", "proportion"], ascending=[True, True, False])


def label_proportions(event_df: pd.DataFrame) -> pd.DataFrame:
    """Market-response label proportions by horizon/split."""
    cols = ["horizon", "split", "market_response_label"]
    counts = event_df.groupby(cols).size().rename("n").reset_index()
    totals = counts.groupby(["horizon", "split"])["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(["horizon", "split", "proportion"], ascending=[True, True, False])


def directional_label_proportions(event_df: pd.DataFrame) -> pd.DataFrame:
    """Directional-core response-label proportions only.

    This is the cleanest label distribution for absorption quality.  Neutral
    active labels are intentionally excluded.
    """
    core = event_df[event_df["response_context"].eq("directional_core")].copy()
    if core.empty:
        return pd.DataFrame()
    cols = ["horizon", "split", "market_response_label"]
    counts = core.groupby(cols).size().rename("n").reset_index()
    totals = counts.groupby(["horizon", "split"])["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(["horizon", "split", "proportion"], ascending=[True, True, False])


def percentile_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    """Response-percentile stability for directional-core samples."""
    core = event_df[event_df["response_context"].eq("directional_core")]
    if core.empty:
        return pd.DataFrame()
    return core.groupby(["horizon", "split"]).agg(
        n=("response_percentile", "count"),
        mean_u=("response_percentile", "mean"),
        median_u=("response_percentile", "median"),
        std_u=("response_percentile", "std"),
        p10=("response_percentile", lambda s: s.quantile(0.10)),
        p90=("response_percentile", lambda s: s.quantile(0.90)),
    ).reset_index()


def directional_quality_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    """Core signal-to-noise and reliability quality by horizon/split/context."""
    cols = ["horizon", "split", "response_context"]
    return event_df.groupby(cols).agg(
        n=("response_context", "size"),
        median_snr=("snr_raw", "median"),
        p90_snr=("snr_raw", lambda s: s.quantile(0.90)),
        median_reliability=("plie_reliability", "median"),
        median_quality_weight=("quality_weight", "median"),
        mean_quality_weight=("quality_weight", "mean"),
        median_sigma_bps=("volatility_sigma_past_bps", "median"),
        median_braw_bps=("plie_reference_raw_bps", "median"),
    ).reset_index()


def state_exit_rates(base_df: pd.DataFrame, event_df: pd.DataFrame, horizon: str = "30m", windows=(12, 24)) -> pd.DataFrame:
    """Research-only future state exit diagnostic by market response label."""
    if "hmm_state" not in base_df.columns:
        return pd.DataFrame()
    base = base_df[["time", "hmm_state"]].copy().sort_values("time").reset_index(drop=True)
    state = base["hmm_state"].to_numpy()
    rows = []
    for w in windows:
        exits = []
        for i in range(len(state)):
            end = min(len(state), i + w + 1)
            exits.append(bool(np.any(state[i + 1:end] != state[i])) if i + 1 < end else np.nan)
        base[f"state_exit_{w}h"] = exits
    ev = event_df[event_df["horizon"].eq(horizon)][["event_time", "response_context", "market_response_label"]].copy()
    merged = ev.merge(base.rename(columns={"time": "event_time"}), on="event_time", how="left")
    for w in windows:
        col = f"state_exit_{w}h"
        tmp = merged.groupby(["response_context", "market_response_label"]).agg(
            n=(col, "count"),
            exit_rate=(col, "mean"),
        ).reset_index()
        tmp["window_hours"] = w
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def staleness_diagnostics(memory_df: pd.DataFrame, main_horizon: str = "30m") -> pd.DataFrame:
    """Summarize directional-core memory freshness.

    High staleness is not a data leak, but it is a quality risk: directional
    absorption should not remain a strong Agent input when no fresh
    directional-core event has matured.
    """
    age_col = f"mar_directional_core_age_hours_{main_horizon}"
    fresh_col = f"mar_directional_core_freshness_{main_horizon}"
    if age_col not in memory_df.columns:
        return pd.DataFrame([{"metric": "staleness_available", "value": 0.0}])
    age = pd.to_numeric(memory_df[age_col], errors="coerce")
    fresh = pd.to_numeric(memory_df.get(fresh_col), errors="coerce")
    rows = [
        {"metric": "staleness_available", "value": 1.0},
        {"metric": "pct_no_core_yet", "value": float(age.isna().mean())},
        {"metric": "median_core_age_hours", "value": float(age.median())},
        {"metric": "p90_core_age_hours", "value": float(age.quantile(0.90))},
        {"metric": "p99_core_age_hours", "value": float(age.quantile(0.99))},
        {"metric": "pct_core_age_gt_6h", "value": float((age > 6).mean())},
        {"metric": "pct_core_age_gt_12h", "value": float((age > 12).mean())},
        {"metric": "pct_core_age_gt_24h", "value": float((age > 24).mean())},
        {"metric": "median_freshness", "value": float(fresh.median()) if fresh is not None else np.nan},
        {"metric": "p10_freshness", "value": float(fresh.quantile(0.10)) if fresh is not None else np.nan},
    ]
    return pd.DataFrame(rows)


def agent_feature_correlation(memory_df: pd.DataFrame, agent_inputs: list[str]) -> pd.DataFrame:
    """Spearman correlation matrix in long form for numeric Agent inputs."""
    numeric = [c for c in agent_inputs if c in memory_df.columns and pd.api.types.is_numeric_dtype(memory_df[c])]
    if len(numeric) < 2:
        return pd.DataFrame()
    corr = memory_df[numeric].corr(method="spearman")
    out = corr.reset_index().melt(id_vars="index", var_name="feature_b", value_name="spearman_corr")
    out = out.rename(columns={"index": "feature_a"})
    return out


def memory_quality(memory_df: pd.DataFrame, agent_inputs: list[str]) -> pd.DataFrame:
    """Range, missingness, and outlier diagnostics for Agent inputs."""
    rows = []
    for col in agent_inputs:
        if col not in memory_df.columns:
            rows.append({"feature": col, "exists": False, "missing_rate": 1.0, "outlier_rate": np.nan, "expected_range": "missing"})
            continue
        s = memory_df[col]
        expected = "categorical"
        outlier = np.nan
        if pd.api.types.is_numeric_dtype(s):
            # Order matters: conflict contains "score" in older names but is [0, 1].
            if "conflict" in col or "persistence" in col or "freshness" in col or "quality" in col or "rejection_score" in col or "cascade_score" in col or "active_dominance_score" in col or "active_context_quality" in col or "liq_neutrality_score" in col:
                expected = "[0, 1]"
                outlier = ((s < 0) | (s > 1)).mean()
            elif "active_force" in col:
                expected = "[-1, 1]"
                outlier = ((s < -1) | (s > 1)).mean()
            elif "score" in col or "strength" in col:
                expected = "[0, 100]"
                outlier = ((s < 0) | (s > 100)).mean()
            elif "count" in col:
                expected = ">=0"
                outlier = (s < 0).mean()
            elif "age_hours" in col:
                expected = ">=0"
                outlier = (s < 0).mean()
            else:
                expected = "finite"
                outlier = (~np.isfinite(s.dropna())).mean()
        rows.append({
            "feature": col,
            "exists": True,
            "missing_rate": s.isna().mean(),
            "outlier_rate": outlier,
            "expected_range": expected,
        })
    return pd.DataFrame(rows)


def latest_summary(base_df: pd.DataFrame, memory_df: pd.DataFrame, event_df: pd.DataFrame, cfg: Config) -> dict:
    """Latest run summary for index.html."""
    last_base = base_df.sort_values("time").iloc[-1]
    last_mem = memory_df.sort_values("time").iloc[-1]
    main_h = cfg.get("memory", "main_horizon", default="30m")
    main = event_df[event_df["horizon"].eq(main_h)].sort_values("available_time")
    last_event = main.iloc[-1] if not main.empty else {}
    return {
        "latest_data_time": str(last_base.get("time")),
        "latest_price": float(last_base.get("price")) if "price" in last_base else None,
        "latest_hmm_state": float(last_base.get("hmm_state")) if "hmm_state" in last_base else None,
        "latest_plie_main_bps": float(last_base.get("plie_main_bps")) if "plie_main_bps" in last_base else None,
        "latest_plie_reliability": float(last_base.get("plie_reliability")) if "plie_reliability" in last_base else None,
        "latest_matured_absorption_available_time": str(last_event.get("available_time")) if len(main) else None,
        "latest_matured_market_response_label": str(last_event.get("market_response_label")) if len(main) else None,
        "latest_agent_inputs": {c: (None if pd.isna(last_mem.get(c)) else last_mem.get(c)) for c in cfg.agent_inputs if c in memory_df.columns},
    }


def path_context_distribution(path_df: pd.DataFrame) -> pd.DataFrame:
    """Path-level context proportions by window/split."""
    if path_df is None or path_df.empty:
        return pd.DataFrame()
    cols = ["window_hours", "split", "path_context"]
    counts = path_df.groupby(cols).size().rename("n").reset_index()
    totals = counts.groupby(["window_hours", "split"])["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(["window_hours", "split", "proportion"], ascending=[True, True, False])


def path_label_proportions(path_df: pd.DataFrame) -> pd.DataFrame:
    """Path-level response label proportions by window/split."""
    if path_df is None or path_df.empty:
        return pd.DataFrame()
    cols = ["window_hours", "split", "path_label"]
    counts = path_df.groupby(cols).size().rename("n").reset_index()
    totals = counts.groupby(["window_hours", "split"])["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(["window_hours", "split", "proportion"], ascending=[True, True, False])



def path_context_label_combo_counts(path_df: pd.DataFrame) -> pd.DataFrame:
    """Counts for each path_context x path_label combination by split/window.

    This is the main v4 audit table.  Context is PLIE-only, label is price
    response. Sparse combinations indicate the taxonomy is too fragmented.
    """
    if path_df is None or path_df.empty:
        return pd.DataFrame()
    cols = ["window_hours", "split", "path_context", "path_label"]
    counts = path_df.groupby(cols).size().rename("n").reset_index()
    totals = counts.groupby(["window_hours", "split"])["n"].transform("sum")
    counts["proportion_in_split_window"] = counts["n"] / totals
    return counts.sort_values(["window_hours", "split", "n"], ascending=[True, True, False])

def path_quality_summary(path_df: pd.DataFrame) -> pd.DataFrame:
    """Summary of decoupled path data quality, signal clarity, and activity.

    The old ``path_quality`` is retained as a compatibility field but should no
    longer be interpreted as market activity.  This summary explicitly reports
    the three refactored concepts so RC-like quiet markets are not mislabeled as
    bad quality.
    """
    if path_df is None or path_df.empty:
        return pd.DataFrame()
    def q(x, p):
        return x.quantile(p)
    agg = {
        "n": ("path_context", "size"),
        "mean_legacy_quality": ("path_quality", "mean"),
        "median_legacy_quality": ("path_quality", "median"),
        "mean_data_quality": ("path_data_quality", "mean"),
        "median_data_quality": ("path_data_quality", "median"),
        "mean_signal_clarity": ("path_signal_clarity", "mean"),
        "median_signal_clarity": ("path_signal_clarity", "median"),
        "mean_activity_level": ("path_activity_level", "mean"),
        "median_activity_level": ("path_activity_level", "median"),
        "mean_abs_score": ("path_absorption_score_0_100", "mean"),
        "median_abs_score": ("path_absorption_score_0_100", "median"),
        "mean_rejection": ("path_pressure_rejection_score", "mean"),
        "p90_rejection": ("path_pressure_rejection_score", lambda s: q(s, 0.90)),
        "mean_cascade": ("path_cascade_score", "mean"),
        "median_directionality": ("path_directionality", "median"),
        "median_active_z": ("path_active_z", "median"),
        "p90_active_z": ("path_active_z", lambda s: q(s, 0.90)),
        "mean_active_dominance": ("path_active_dominance_score", "mean"),
    }
    existing = {name: spec for name, spec in agg.items() if spec[0] in path_df.columns or spec[0] == "path_context"}
    return path_df.groupby(["window_hours", "split", "path_context"]).agg(**existing).reset_index()


def state_proxy_distribution(multiscale_df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of audit-only six-state proxy labels by split.

    The proxy label is not a production state model.  It is an explainability
    artifact used to verify that path evidence maps to RC/RHA/AMB/VT/HPEM/ST in
    the intended direction.
    """
    if multiscale_df is None or multiscale_df.empty or "state_proxy_label" not in multiscale_df.columns:
        return pd.DataFrame()
    df = multiscale_df.copy()
    if "split" not in df.columns:
        df["split"] = "unknown"
    counts = df.groupby(["split", "state_proxy_label"]).size().rename("n").reset_index()
    totals = counts.groupby("split")["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(["split", "proportion"], ascending=[True, False])


def state_evidence_summary(multiscale_df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for interpretable state evidence fields."""
    if multiscale_df is None or multiscale_df.empty:
        return pd.DataFrame()
    cols = [c for c in multiscale_df.columns if c.startswith("e_") or c.startswith("score_")]
    if not cols:
        return pd.DataFrame()
    if "split" not in multiscale_df.columns:
        df = multiscale_df.assign(split="unknown")
    else:
        df = multiscale_df
    rows = []
    for sp, grp in df.groupby("split"):
        for c in cols:
            s = pd.to_numeric(grp[c], errors="coerce")
            rows.append({
                "split": sp,
                "feature": c,
                "mean": s.mean(),
                "median": s.median(),
                "p90": s.quantile(0.90),
                "missing_rate": s.isna().mean(),
            })
    return pd.DataFrame(rows)


def typical_state_evidence_examples(multiscale_df: pd.DataFrame) -> pd.DataFrame:
    """Representative examples for the requested mechanism checks."""
    if multiscale_df is None or multiscale_df.empty:
        return pd.DataFrame()
    df = multiscale_df.copy()
    examples = []
    checks = {
        "partial_absorption_supports_ST_not_RHA": (df.get("path_label_24h", "").astype(str).eq("path_partial_absorption") if "path_label_24h" in df else pd.Series(False, index=df.index)),
        "quiet_no_pressure_supports_RC": (df.get("path_label_6h", "").astype(str).eq("path_quiet_no_pressure") if "path_label_6h" in df else pd.Series(False, index=df.index)),
        "mixed_breakout_supports_VT": (df.get("path_label_24h", "").astype(str).str.contains("mixed_active_breakout") if "path_label_24h" in df else pd.Series(False, index=df.index)),
        "pressure_rejection_supports_RHA": (df.get("path_label_24h", "").astype(str).eq("path_pressure_rejection") if "path_label_24h" in df else pd.Series(False, index=df.index)),
        "reversal_takeover_supports_RHA": (df.get("path_label_24h", "").astype(str).eq("path_reversal_takeover") if "path_label_24h" in df else pd.Series(False, index=df.index)),
        "mixed_chop_supports_AMB": (df.get("path_label_24h", "").astype(str).eq("path_mixed_pressure_chop") if "path_label_24h" in df else pd.Series(False, index=df.index)),
        "data_quality_bad_supports_AMB": (pd.to_numeric(df.get("e_amb_data_quality_bad", pd.Series(0, index=df.index)), errors="coerce") > 0.5),
        "cross_window_conflict_supports_AMB": (pd.to_numeric(df.get("e_amb_cross_window_conflict", pd.Series(0, index=df.index)), errors="coerce") > 0.5),
    }
    keep_cols = [c for c in [
        "time", "split", "state_proxy_label", "state_proxy_margin",
        "path_context_6h", "path_context_12h", "path_context_24h", "path_context_48h",
        "path_label_6h", "path_label_12h", "path_label_24h", "path_label_48h",
        "path_data_quality_24h", "path_signal_clarity_24h", "path_activity_level_24h",
        "e_rc_quiet_pressure", "e_rha_pressure_rejection", "e_rha_reversal_takeover",
        "e_amb_signal_conflict", "e_amb_data_quality_bad", "e_amb_cross_window_conflict",
        "e_vt_active_dominance", "e_hpem_cascade_transmission", "e_st_partial_absorption",
        "score_rc_proxy", "score_rha_proxy", "score_amb_proxy", "score_vt_proxy",
        "score_hpem_proxy", "score_st_proxy"
    ] if c in df.columns]
    for name, mask in checks.items():
        sub = df[mask.fillna(False)].copy()
        if sub.empty:
            continue
        score_col = {
            "partial_absorption_supports_ST_not_RHA": "score_st_proxy",
            "quiet_no_pressure_supports_RC": "score_rc_proxy",
            "mixed_breakout_supports_VT": "score_vt_proxy",
            "pressure_rejection_supports_RHA": "score_rha_proxy",
            "reversal_takeover_supports_RHA": "score_rha_proxy",
            "mixed_chop_supports_AMB": "score_amb_proxy",
            "data_quality_bad_supports_AMB": "score_amb_proxy",
            "cross_window_conflict_supports_AMB": "score_amb_proxy",
        }.get(name, "state_proxy_margin")
        if score_col in sub.columns:
            sub = sub.sort_values(score_col, ascending=False)
        row = sub.iloc[0][keep_cols].to_dict()
        row["check"] = name
        examples.append(row)
    return pd.DataFrame(examples)


def path_scenario_examples(path_df: pd.DataFrame, main_window: int = 24) -> pd.DataFrame:
    """Representative path-level scenarios for manual mechanism review.

    The report selects one high-information example for each observed
    ``pressure_name x context x label`` combination in the main path window.
    It is not a training target; it is a research/audit aid.
    """
    if path_df is None or path_df.empty:
        return pd.DataFrame()
    df = path_df[path_df["window_hours"].eq(main_window)].copy()
    if df.empty:
        df = path_df.copy()
    # High score for directional quality or active-dominance strength.
    score = pd.Series(0.0, index=df.index)
    for col in ["path_quality", "path_active_dominance_score", "path_pressure_rejection_score", "path_cascade_score"]:
        if col in df.columns:
            score = score + pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "path_transmission_ratio" in df.columns:
        score = score + (pd.to_numeric(df["path_transmission_ratio"], errors="coerce").abs().clip(upper=3.0) / 3.0).fillna(0.0)
    df["_example_score"] = score
    group_cols = ["path_pressure_name", "path_context", "path_label"]
    idx = df.sort_values("_example_score", ascending=False).groupby(group_cols, dropna=False).head(1).index
    keep = [
        "time", "window_hours", "split", "price", "hmm_state",
        "path_pressure_name", "path_context", "path_label",
        "path_return_bps", "path_signed_plie_effective_sum_bps",
        "path_raw_plie_total_bps", "path_net_braw_bps",
        "path_direction_consistency", "path_liq_neutrality_score",
        "path_snr", "path_active_z", "path_transmission_ratio",
        "path_absorption_score_0_100", "path_quality",
        "path_pressure_rejection_score", "path_cascade_score",
        "path_active_dominance_score", "path_active_dominance_price_score",
        "_example_score",
    ]
    keep = [c for c in keep if c in df.columns]
    return df.loc[idx, keep].sort_values(["path_pressure_name", "path_context", "path_label"]).reset_index(drop=True)
