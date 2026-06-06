"""Path-level / episode-level QD-MAR absorption features.

This module computes causal path-level market response features over rolling
6h/12h/24h/48h windows.  The v7 design explicitly separates three concepts:

1. ``path_context``: PLIE / liquidation-pressure context only.  It does not use
   price response.
2. ``path_label``: price response to that context using only the already
   realized path ``[T-W, T]``.
3. ``path_data_quality`` / ``path_signal_clarity`` / ``path_activity_level``:
   separate data reliability, semantic clarity, and market activity.  Low
   activity is not low data quality; low activity can be an RC signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, HorizonConfig


def _sigmoid(x: pd.Series | np.ndarray, clip: float = 20.0) -> pd.Series:
    arr = pd.Series(x).astype(float).clip(lower=-float(clip), upper=float(clip))
    return 1.0 / (1.0 + np.exp(-arr))


def _safe_numeric(x, index=None, default=np.nan) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    if index is None:
        return pd.Series(dtype=float)
    return pd.Series(default, index=index, dtype=float)


def _clip01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)


def _infer_source_minutes(times: pd.Series, default_minutes: float = 60.0) -> float:
    """Infer median source-clock spacing in minutes."""
    ts = pd.to_datetime(times, utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return float(default_minutes)
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float(default_minutes)
    return float(diffs.median())


def _rolling_mad_sigma(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Past-only robust scale for path returns."""
    s = pd.to_numeric(series, errors="coerce").shift(1)
    med = s.rolling(window, min_periods=min_periods).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window, min_periods=min_periods).median()
    return (1.4826 * mad).replace([np.inf, -np.inf], np.nan)


def _train_mask(df: pd.DataFrame, cfg: Config) -> pd.Series:
    split_col = cfg.get("data", "split_col", default="split")
    train_split = str(cfg.get("calibration", "train_split", default="train")).lower()
    use_split_thresholds = bool(cfg.get("path_label", "use_train_split_thresholds", default=True))
    if use_split_thresholds and split_col in df.columns:
        m = df[split_col].astype(str).str.lower().eq(train_split)
        if m.any():
            return m
    train_split_ratio = float(cfg.get("path_label", "train_split_ratio", default=0.70))
    return pd.Series(np.arange(len(df)) < int(len(df) * train_split_ratio), index=df.index)


def _train_quantile(s: pd.Series, mask: pd.Series, q: float, default: float) -> float:
    vals = pd.to_numeric(s[mask], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float(default)
    return float(vals.quantile(q))


def _normalize_by_train_quantile(
    s: pd.Series,
    mask: pd.Series,
    q: float = 0.90,
    default_scale: float = 1.0,
    max_value: float = 3.0,
) -> pd.Series:
    scale = _train_quantile(s.abs(), mask, q, default_scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = default_scale
    return (s / scale).clip(lower=0.0, upper=max_value)


def _main_horizon_config(cfg: Config) -> HorizonConfig:
    main_h = cfg.get("path_absorption", "main_horizon", default=cfg.get("memory", "main_horizon", default="30m"))
    for h in cfg.horizons:
        if h.name == main_h:
            return h
    raise ValueError(f"Configured path_absorption.main_horizon={main_h!r} is not present in horizons")


def _ctx_cfg(cfg: Config) -> dict:
    return cfg.get("path_context", default={}) or cfg.get("path_absorption", "path_context", default={}) or {}


def _label_cfg(cfg: Config) -> dict:
    return cfg.get("path_label", default={}) or cfg.get("path_absorption", "path_label", default={}) or {}


def _quality_cfg(cfg: Config) -> dict:
    return cfg.get("path_quality", default={}) or cfg.get("path_absorption", "path_quality", default={}) or {}


def _price_ctx_cfg(cfg: Config) -> dict:
    return cfg.get("price_context", default={}) or {}


def _window_price_suffix(wh: int, cfg: Config) -> str:
    mapping = (_price_ctx_cfg(cfg).get("window_mapping") or {6: "6h", 12: "6h", 24: "24h", 48: "24h"})
    return str(mapping.get(int(wh), mapping.get(str(int(wh)), "24h")))


def _price_feature(df: pd.DataFrame, base: str, suffix: str, default=np.nan) -> pd.Series:
    col = f"{base}_{suffix}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _price_direction_name(path_ret: float) -> str:
    if not np.isfinite(path_ret) or abs(path_ret) < 1e-12:
        return "flat"
    return "up" if path_ret > 0 else "down"


def _directional_label(tr: float, divergence: float, cascade_score: float, trend_against: bool, thresholds: dict) -> str:
    """Directional context label.  Partial absorption is deliberately not a
    reversal label; it is an orderly-transmission/transition evidence."""
    if not np.isfinite(tr):
        return "path_unavailable"
    amp = float(thresholds.get("amplification_tr", 1.25))
    base = float(thresholds.get("baseline_tr", 0.65))
    part = float(thresholds.get("partial_tr", 0.15))
    near_zero = float(thresholds.get("near_zero_tr", -0.15))
    strong_neg = float(thresholds.get("strong_negative_tr", -0.65))
    # Cascade should be high directional transmission, not merely a large ratio
    # in a quiet market.  cascade_score embeds volatility/jump support.
    if tr >= amp and cascade_score >= float(thresholds.get("cascade_score_min", 0.35)):
        return "path_cascade_transmission"
    if tr >= base:
        return "path_baseline_transmission"
    if tr >= part:
        return "path_partial_absorption"
    if tr >= near_zero:
        return "path_full_absorption_stall"
    if tr <= strong_neg and (divergence > 0.75 or trend_against):
        return "path_reversal_takeover"
    return "path_pressure_rejection"


def _neutral_label(path_ret: float, active_z: float, quiet_score: float, trend_high: bool, thresholds: dict) -> str:
    if not np.isfinite(path_ret) or not np.isfinite(active_z):
        return "path_unavailable"
    direction = _price_direction_name(path_ret)
    if quiet_score >= float(thresholds.get("quiet_score_min", 0.60)):
        return "path_quiet_no_pressure"
    normal = float(thresholds.get("active_z_normal", 0.75))
    strong = float(thresholds.get("active_z_strong", 1.50))
    if direction == "flat" or active_z < normal:
        return "path_quiet_no_pressure"
    if active_z >= strong and trend_high:
        return f"path_active_dominance_{direction}"
    return f"path_normal_active_dominance_{direction}"


def _mixed_label(path_ret: float, active_z: float, chop_score: float, trend_high: bool, thresholds: dict) -> str:
    if not np.isfinite(path_ret) or not np.isfinite(active_z):
        return "path_unavailable"
    direction = _price_direction_name(path_ret)
    if chop_score >= float(thresholds.get("chop_score_min", 0.55)):
        return "path_mixed_pressure_chop"
    normal = float(thresholds.get("active_z_normal", 0.75))
    strong = float(thresholds.get("active_z_strong", 1.50))
    if direction == "flat" or active_z < normal:
        return "path_mixed_pressure_chop"
    if active_z >= strong and trend_high:
        return f"path_mixed_active_breakout_{direction}"
    return f"path_normal_mixed_active_breakout_{direction}"


def compute_path_absorption(base_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute causal multiscale path-level absorption features.

    For time ``T`` and window ``W`` the function uses only observations in
    ``[T-W, T]`` and past-only calibration statistics from the train split.  No
    future price or future PLIE observation is used.
    """
    if not cfg.get("path_absorption", "enabled", default=True):
        return pd.DataFrame()

    time_col = cfg.get("data", "time_col", default="time")
    price_col = cfg.get("data", "price_col", default="price")
    h = _main_horizon_config(cfg)
    eps = float(cfg.get("absorption", "eps", default=1e-9))
    pcfg = cfg.get("path_absorption") or {}
    context_cfg = _ctx_cfg(cfg)
    label_cfg = _label_cfg(cfg)
    quality_cfg = _quality_cfg(cfg)

    windows = list(context_cfg.get("windows", context_cfg.get("windows_hours", pcfg.get("windows_hours", [6, 12, 24, 48]))))
    windows = [int(str(w).replace("h", "")) for w in windows]
    vol_window = int(pcfg.get("vol_window", 168))
    vol_min = int(pcfg.get("vol_min_periods", 48))
    lambda_sigma = float(pcfg.get("lambda_sigma", 0.75))
    source_default_minutes = float(pcfg.get("source_clock_default_minutes", 60.0))
    min_window_points = int(pcfg.get("min_window_points", 2))
    price_return_bps_scale = float(pcfg.get("price_return_bps_scale", 10000.0))
    sigmoid_clip = float(pcfg.get("sigmoid_clip", 20.0))
    active_score_clip_upper = float(pcfg.get("active_score_clip_upper", 2.0))
    intensity_quantile = float(context_cfg.get("intensity_train_quantile", pcfg.get("intensity_train_quantile", 0.90)))
    intensity_default_scale = float(context_cfg.get("intensity_default_scale", pcfg.get("intensity_default_scale", 1.0)))
    intensity_max = float(context_cfg.get("intensity_max", pcfg.get("intensity_max", 3.0)))

    mass_low_q = float(context_cfg.get("mass_low_quantile", 0.30))
    mass_high_q = float(context_cfg.get("mass_high_quantile", 0.70))
    core_thr = float(context_cfg.get("direction_core_threshold", 0.65))
    weak_thr = float(context_cfg.get("direction_weak_threshold", 0.40))
    mixed_thr = float(context_cfg.get("mixed_threshold", 0.35))
    min_obs_share = float(context_cfg.get("min_valid_obs_share", 0.80))
    fallback_context = str(context_cfg.get("fallback_context", "auto"))
    min_dir_consistency = float(pcfg.get("min_direction_consistency", weak_thr))
    core_dir_consistency = float(pcfg.get("core_direction_consistency", core_thr))
    min_net_braw_bps = float(pcfg.get("min_net_braw_bps", 0.0))
    min_total_braw_bps = float(pcfg.get("min_total_braw_bps", 0.0))
    min_nonzero_direction_share = float(pcfg.get("min_nonzero_direction_share", 0.0))
    weak_reliability_min = float(pcfg.get("weak_reliability_min", 0.0))
    core_reliability_min = float(pcfg.get("core_reliability_min", weak_reliability_min))
    weak_snr_min = float(pcfg.get("weak_snr_min", 0.0))
    core_snr_min = float(pcfg.get("core_snr_min", weak_snr_min))

    active_z_low = float(label_cfg.get("active_z_normal", pcfg.get("active_z_low", 0.75)))
    active_z_strong = float(label_cfg.get("active_z_strong", pcfg.get("active_z_strong", 1.50)))
    active_z_extreme = float(label_cfg.get("active_z_extreme", pcfg.get("active_z_extreme", 3.00)))
    thresholds = dict(pcfg.get("label_thresholds", {}) or {})
    thresholds.update({k: v for k, v in label_cfg.items() if isinstance(v, (int, float))})
    thresholds.setdefault("active_z_normal", active_z_low)
    thresholds.setdefault("active_z_strong", active_z_strong)
    thresholds.setdefault("active_z_extreme", active_z_extreme)

    df = base_df.copy().sort_values(time_col).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    source_minutes = _infer_source_minutes(df[time_col], default_minutes=source_default_minutes)
    train = _train_mask(df, cfg)

    price = pd.to_numeric(df[price_col], errors="coerce")
    d = pd.to_numeric(df.get("plie_direction"), errors="coerce").fillna(0.0)
    reliability = pd.to_numeric(df.get("plie_reliability"), errors="coerce").fillna(0.0)
    plie_main = pd.to_numeric(df.get(h.plie_col, df.get("plie_main_bps")), errors="coerce").fillna(0.0)
    b_raw = pd.to_numeric(df.get(h.raw_mag_col), errors="coerce").fillna(0.0)

    # Pressure context uses d * rho * intensity.  If the requested scaled
    # intensity column is absent, derive a train-only quantile scaled version.
    source = str(context_cfg.get("pressure_intensity_source", "plie_intensity_scaled"))
    if source in df.columns:
        intensity = pd.to_numeric(df[source], errors="coerce").fillna(0.0)
    elif "plie_intensity" in df.columns:
        intensity = _normalize_by_train_quantile(
            pd.to_numeric(df["plie_intensity"], errors="coerce").fillna(0.0),
            train,
            intensity_quantile,
            intensity_default_scale,
            intensity_max,
        )
    else:
        intensity = _normalize_by_train_quantile(
            plie_main.abs(),
            train,
            intensity_quantile,
            intensity_default_scale,
            intensity_max,
        )

    pressure_unit = d * reliability * intensity
    valid_pressure_obs = d.notna() & reliability.notna() & intensity.notna()
    signed_raw = d * b_raw

    # Baseline impact: default is signed effective PLIE.  It avoids double
    # reliability multiplication because plie_main_bps is already an effective
    # reliability-weighted passive impact.  The alternative is available through
    # config for audit compatibility.
    baseline_mode = str(label_cfg.get("baseline_impact_mode", "effective_plie"))
    if baseline_mode == "direction_abs_plie_times_reliability":
        baseline_unit = d * plie_main.abs() * reliability
    else:
        baseline_unit = plie_main

    # Price-context normalization thresholds are fitted on train split only.
    def q_price(base: str, suffix: str, q: float, default: float) -> float:
        return _train_quantile(_price_feature(df, base, suffix), train, q, default)

    pieces: list[pd.DataFrame] = []
    for wh in windows:
        suffix = _window_price_suffix(wh, cfg)
        n = max(min_window_points, int(round(float(wh) * 60.0 / source_minutes)))
        path_ret = price_return_bps_scale * np.log(price / price.shift(n))
        sigma = _rolling_mad_sigma(path_ret, vol_window, vol_min)
        sigma = sigma.where(sigma > 0, np.nan)

        pressure_mass = pressure_unit.abs().rolling(n, min_periods=n).sum()
        net_pressure = pressure_unit.rolling(n, min_periods=n).sum()
        obs_count = valid_pressure_obs.astype(float).rolling(n, min_periods=1).sum()
        missing_ratio = (1.0 - obs_count / float(n)).clip(lower=0.0, upper=1.0)
        directionality = (net_pressure.abs() / (pressure_mass + eps)).replace([np.inf, -np.inf], np.nan)
        dominant_direction = np.sign(net_pressure).fillna(0.0)

        raw_signed_sum = signed_raw.rolling(n, min_periods=n).sum()
        raw_total_sum = signed_raw.abs().rolling(n, min_periods=n).sum()
        baseline_impact = baseline_unit.rolling(n, min_periods=n).sum()
        nonzero_direction_share = (d.abs() > 0).astype(float).rolling(n, min_periods=n).mean()
        reliability_mean = reliability.rolling(n, min_periods=n).mean()

        mass_low = _train_quantile(pressure_mass, train, mass_low_q, 0.0)
        mass_high = _train_quantile(pressure_mass, train, mass_high_q, max(mass_low, 1.0))

        mature = path_ret.notna() & pressure_mass.notna() & (obs_count >= (min_obs_share * n))
        has_mass = pressure_mass > mass_low
        high_mass = pressure_mass >= mass_high
        path_snr_pre = pressure_mass / (sigma + eps)
        raw_total_ok = raw_total_sum >= min_total_braw_bps
        raw_net_ok = raw_signed_sum.abs() >= min_net_braw_bps
        direction_share_ok = nonzero_direction_share >= min_nonzero_direction_share
        core_quality_ok = (reliability_mean >= core_reliability_min) & (path_snr_pre >= core_snr_min)
        weak_quality_ok = (reliability_mean >= weak_reliability_min) & (path_snr_pre >= weak_snr_min)

        ctx = pd.Series("path_neutral_pressure", index=df.index, dtype=object)
        core = (
            mature & high_mass & raw_total_ok & raw_net_ok & direction_share_ok &
            core_quality_ok & (directionality >= max(core_thr, core_dir_consistency))
        )
        weak = (
            mature & has_mass & raw_total_ok & raw_net_ok & direction_share_ok &
            weak_quality_ok & (directionality >= max(weak_thr, min_dir_consistency)) & ~core
        )
        mixed = mature & high_mass & raw_total_ok & direction_share_ok & (directionality < mixed_thr)
        ctx.loc[mixed] = "path_mixed_pressure"
        ctx.loc[weak] = "path_directional_weak"
        ctx.loc[core] = "path_directional_core"
        if fallback_context == "auto":
            fallback = mature & has_mass & raw_total_ok & direction_share_ok & ~(core | weak | mixed)
            # Medium activity with unclear net direction is mixed pressure; very
            # low activity remains neutral.
            ctx.loc[fallback] = "path_mixed_pressure"
        elif fallback_context in {"path_mixed_pressure", "path_neutral_pressure"}:
            fallback = mature & has_mass & raw_total_ok & direction_share_ok & ~(core | weak | mixed)
            ctx.loc[fallback] = fallback_context

        path_direction = dominant_direction.where(ctx.isin(["path_directional_core", "path_directional_weak"]), 0.0)
        aligned_return = path_direction * path_ret
        transmission_ratio = aligned_return / (baseline_impact.abs() + eps)
        # Fallback ratio using pressure mass if PLIE baseline is numerically tiny.
        transmission_ratio = transmission_ratio.where(baseline_impact.abs() > eps, aligned_return / (pressure_mass + eps))

        path_sigma = sigma
        scale = np.sqrt(np.square(baseline_impact.abs()) + np.square(lambda_sigma * path_sigma) + eps)
        divergence_score = -aligned_return / scale
        transmission_score = aligned_return / scale
        absorption_score = 100.0 * _sigmoid(divergence_score, clip=sigmoid_clip)

        active_z = path_ret.abs() / (path_sigma + eps)
        active_score_raw = ((active_z - active_z_low) / max(active_z_extreme - active_z_low, eps)).clip(lower=0.0, upper=active_score_clip_upper)
        active_dominance_score = np.tanh(active_score_raw).clip(lower=0.0, upper=1.0)
        active_direction = pd.Series(np.sign(path_ret), index=df.index).fillna(0.0)
        active_dominance_price_score = active_direction * active_dominance_score

        # Price context: use already past-only features from optional price_context.
        rv = _price_feature(df, "realized_vol", suffix)
        rv_q_low = q_price("realized_vol", suffix, float(label_cfg.get("vol_low_quantile", 0.30)), np.nan)
        rv_q_high = q_price("realized_vol", suffix, float(label_cfg.get("vol_high_quantile", 0.70)), np.nan)
        trend = _price_feature(df, "trend_strength", suffix)
        trend_q_low = q_price("trend_strength", suffix, float(label_cfg.get("trend_low_quantile", 0.30)), np.nan)
        trend_q_high = q_price("trend_strength", suffix, float(label_cfg.get("trend_high_quantile", 0.70)), np.nan)
        trend_cons = _price_feature(df, "trend_consistency", suffix).fillna(0.0)
        trend_dir = _price_feature(df, "trend_direction", suffix).fillna(0.0)
        rng = _price_feature(df, "range_compression", suffix)
        rng_q_high = q_price("range_compression", suffix, float(label_cfg.get("range_high_quantile", 0.70)), np.nan)
        jump = _price_feature(df, "jump_proxy", suffix)
        jump_q_high = q_price("jump_proxy", suffix, float(label_cfg.get("jump_high_quantile", 0.70)), np.nan)
        price_missing = _price_feature(df, "price_missing_ratio", suffix, default=0.0).fillna(0.0).clip(0, 1)
        price_gap = _price_feature(df, "price_gap_flag", suffix, default=0.0).fillna(0.0).clip(0, 1)
        price_outlier = _price_feature(df, "price_outlier_flag", suffix, default=0.0).fillna(0.0).clip(0, 1)

        vol_low = (rv <= rv_q_low) if np.isfinite(rv_q_low) else pd.Series(False, index=df.index)
        vol_high = (rv >= rv_q_high) if np.isfinite(rv_q_high) else pd.Series(False, index=df.index)
        trend_low = (trend <= trend_q_low) if np.isfinite(trend_q_low) else pd.Series(False, index=df.index)
        trend_high = (trend >= trend_q_high) if np.isfinite(trend_q_high) else pd.Series(False, index=df.index)
        range_high = (rng >= rng_q_high) if np.isfinite(rng_q_high) else pd.Series(False, index=df.index)
        jump_high = (jump >= jump_q_high) if np.isfinite(jump_q_high) else pd.Series(False, index=df.index)
        trend_against = (trend_high & (trend_dir * dominant_direction < 0)).fillna(False)

        # Label-support scores.
        transmission_boost = np.maximum(0.0, np.tanh(transmission_score.clip(lower=-sigmoid_clip, upper=sigmoid_clip)))
        rejection_boost = np.maximum(0.0, np.tanh(divergence_score.clip(lower=-sigmoid_clip, upper=sigmoid_clip)))
        vol_jump_support = ((vol_high.astype(float) + jump_high.astype(float)) / 2.0).clip(0, 1)
        score_weights = pcfg.get("score_weights", {}) or {}
        cascade_base = float(score_weights.get("cascade_base", 0.60))
        cascade_vol_jump = float(score_weights.get("cascade_vol_jump", 0.40))
        rejection_base = float(score_weights.get("rejection_base", 0.60))
        rejection_trend_against = float(score_weights.get("rejection_trend_against", 0.40))
        cascade_score = (transmission_boost * (cascade_base + cascade_vol_jump * vol_jump_support)).clip(0, 1)
        pressure_rejection_score = (rejection_boost * (rejection_base + rejection_trend_against * trend_against.astype(float))).clip(0, 1)

        # Quiet / chop support for neutral and mixed contexts.  These use price
        # context as supplementary evidence and do not change PLIE context.
        quiet_weights = pcfg.get("quiet_score_weights", {}) or {}
        quiet_score = (
            float(quiet_weights.get("vol_low", 0.35)) * vol_low.astype(float)
            + float(quiet_weights.get("trend_low", 0.30)) * trend_low.astype(float)
            + float(quiet_weights.get("range_high", 0.25)) * range_high.astype(float)
            + float(quiet_weights.get("no_jump", 0.10)) * (1.0 - jump_high.astype(float))
        ).clip(0, 1)
        chop_weights = pcfg.get("chop_score_weights", {}) or {}
        chop_score = (
            float(chop_weights.get("trend_low", 0.35)) * trend_low.astype(float)
            + float(chop_weights.get("low_trend_consistency", 0.25)) * (1.0 - trend_cons.clip(0, 1))
            + float(chop_weights.get("range_high", 0.25)) * range_high.astype(float)
            + float(chop_weights.get("low_directionality", 0.15)) * (directionality < weak_thr).astype(float)
        ).clip(0, 1)

        labels = []
        for c, tr, div, r, az, cas, t_against, qs, cs, th in zip(
            ctx.to_numpy(),
            transmission_ratio.to_numpy(),
            divergence_score.to_numpy(),
            path_ret.to_numpy(),
            active_z.to_numpy(),
            cascade_score.to_numpy(),
            trend_against.to_numpy(),
            quiet_score.to_numpy(),
            chop_score.to_numpy(),
            trend_high.to_numpy(),
        ):
            if c in {"path_directional_core", "path_directional_weak"}:
                labels.append(_directional_label(tr, div, cas, bool(t_against), thresholds))
            elif c == "path_mixed_pressure":
                labels.append(_mixed_label(r, az, cs, bool(th), thresholds))
            elif c == "path_neutral_pressure":
                labels.append(_neutral_label(r, az, qs, bool(th), thresholds))
            else:
                labels.append("path_unavailable")

        pressure_name = pd.Series("neutral_pressure", index=df.index, dtype=object)
        pressure_name.loc[dominant_direction > 0] = "upward_pressure"
        pressure_name.loc[dominant_direction < 0] = "downward_pressure"
        pressure_name.loc[ctx.eq("path_neutral_pressure")] = "neutral_pressure"
        pressure_name.loc[ctx.eq("path_mixed_pressure")] = "mixed_pressure"

        # Decoupled quality fields.
        dq_weights = quality_cfg.get("data_quality_weights", {}) or {}
        w_missing = float(dq_weights.get("missing_ratio", 0.35))
        w_gap = float(dq_weights.get("gap_flag", 0.25))
        w_price_missing = float(dq_weights.get("price_missing", 0.20))
        w_plie_missing = float(dq_weights.get("plie_missing", 0.20))
        outlier_weight = float(dq_weights.get("price_outlier", 0.10))
        bad_data = (
            w_missing * missing_ratio.fillna(1.0)
            + w_gap * price_gap
            + w_price_missing * price_missing
            + w_plie_missing * missing_ratio.fillna(1.0)
            + outlier_weight * price_outlier
        )
        data_quality = (1.0 - bad_data).clip(0, 1)
        data_quality = data_quality.where(mature, 0.0)

        gross_activity_norm = (pressure_mass / (mass_high + eps)).clip(0, 1).fillna(0.0)
        directional_clarity = directionality.clip(0, 1).fillna(0.0)
        label_margin_score = pd.Series(0.0, index=df.index)
        label_margin_score.loc[ctx.isin(["path_directional_core", "path_directional_weak"])] = (
            np.maximum(cascade_score, pressure_rejection_score).loc[ctx.isin(["path_directional_core", "path_directional_weak"])]
        )
        label_margin_score.loc[ctx.eq("path_neutral_pressure")] = np.maximum(active_dominance_score, quiet_score).loc[ctx.eq("path_neutral_pressure")]
        label_margin_score.loc[ctx.eq("path_mixed_pressure")] = np.maximum(active_dominance_score, chop_score).loc[ctx.eq("path_mixed_pressure")]

        # Signal clarity is about semantic clarity, not activity.  Low-pressure
        # quiet windows can have high clarity if data are good and quiet_score is high.
        clarity_weights = quality_cfg.get("signal_clarity_weights", {}) or {}
        signal_clarity = (
            float(clarity_weights.get("directionality", 0.35)) * directional_clarity.where(ctx.isin(["path_directional_core", "path_directional_weak"]), 0.0)
            + float(clarity_weights.get("label_margin", 0.30)) * label_margin_score
            + float(clarity_weights.get("quiet_support", 0.20)) * quiet_score.where(ctx.eq("path_neutral_pressure"), 0.0)
            + float(clarity_weights.get("chop_support", 0.20)) * chop_score.where(ctx.eq("path_mixed_pressure"), 0.0)
            + float(clarity_weights.get("data_quality", 0.15)) * data_quality
        ).clip(0, 1)

        # Activity level is separate; low value can support RC.
        activity_weights = quality_cfg.get("activity_level_weights", {}) or {}
        activity_level = (
            float(activity_weights.get("pressure_mass", 0.30)) * gross_activity_norm
            + float(activity_weights.get("realized_vol", 0.25)) * ((rv / (rv_q_high + eps)).clip(0, 1) if np.isfinite(rv_q_high) else 0.0)
            + float(activity_weights.get("trend_strength", 0.25)) * ((trend / (trend_q_high + eps)).clip(0, 1) if np.isfinite(trend_q_high) else 0.0)
            + float(activity_weights.get("jump_proxy", 0.20)) * ((jump / (jump_q_high + eps)).clip(0, 1) if np.isfinite(jump_q_high) else 0.0)
        )
        if not isinstance(activity_level, pd.Series):
            activity_level = pd.Series(activity_level, index=df.index)
        activity_level = activity_level.clip(0, 1).fillna(0.0)

        # Legacy quality is kept only as a compatibility field.  It no longer
        # embeds activity level and therefore does not punish quiet RC regimes.
        legacy_quality = (data_quality * signal_clarity).clip(0, 1)

        active_force_aligned = legacy_quality * np.tanh(transmission_score.clip(lower=-20, upper=20))
        active_force_price = dominant_direction * active_force_aligned
        active_context_quality = active_dominance_score.where(ctx.isin(["path_neutral_pressure", "path_mixed_pressure"]), 0.0)

        part = pd.DataFrame({
            "time": df[time_col],
            "available_time": df[time_col],
            "window_hours": int(wh),
            "window_n_source": int(n),
            "split": df.get("split", "unknown"),
            "price": price,
            "hmm_state": df.get("hmm_state"),
            "state_severity": df.get("state_severity"),
            "state_severity_bucket": df.get("state_severity_bucket"),
            "plie_phase": df.get("plie_phase"),
            "vol_regime": df.get("vol_regime"),
            "path_pressure_mass": pressure_mass,
            "path_net_pressure": net_pressure,
            "path_directionality": directionality,
            "path_dominant_direction": dominant_direction,
            "path_pressure_obs_count": obs_count,
            "path_pressure_missing_ratio": missing_ratio,
            "path_pressure_direction": dominant_direction,
            "path_pressure_name": pressure_name,
            "path_return_bps": path_ret,
            "path_signed_plie_effective_sum_bps": baseline_impact,
            "path_baseline_impact_bps": baseline_impact,
            "path_signed_raw_plie_sum_bps": raw_signed_sum,
            "path_raw_plie_total_bps": raw_total_sum,
            "path_net_braw_bps": raw_signed_sum.abs(),
            "path_direction_consistency": directionality,
            "path_liq_neutrality_score": (1.0 - directionality.clip(0, 1)).fillna(1.0),
            "path_nonzero_direction_share": nonzero_direction_share,
            "path_reliability_mean": reliability_mean,
            "path_sigma_past_bps": path_sigma,
            "path_snr": (pressure_mass / (path_sigma + eps)),
            "path_active_z": active_z,
            "path_active_dominance_score": active_dominance_score,
            "path_active_dominance_direction": active_direction,
            "path_active_dominance_price_score": active_dominance_price_score,
            "path_aligned_return_bps": aligned_return,
            "path_aligned_response_bps": aligned_return,  # backward compatible alias
            "path_transmission_ratio": transmission_ratio,
            "path_absorption_raw": 1.0 - transmission_ratio,
            "path_divergence_score": divergence_score,
            "path_transmission_score": transmission_score,
            "path_absorption_score_0_100": absorption_score,
            "path_quality": legacy_quality,
            "path_data_quality": data_quality,
            "path_signal_clarity": signal_clarity,
            "path_activity_level": activity_level,
            "path_quiet_score": quiet_score,
            "path_chop_score": chop_score,
            "path_active_context_quality": active_context_quality,
            "path_active_force_aligned_score": active_force_aligned,
            "path_active_force_price_score": active_force_price,
            "path_pressure_rejection_score": pressure_rejection_score,
            "path_cascade_score": cascade_score,
            "path_context": ctx,
            "path_label": labels,
            "realized_vol_used_bps": rv,
            "range_compression_used": rng,
            "trend_strength_used": trend,
            "trend_consistency_used": trend_cons,
            "trend_direction_used": trend_dir,
            "jump_proxy_used": jump,
            "price_missing_ratio_used": price_missing,
            "price_gap_flag_used": price_gap,
            "price_outlier_flag_used": price_outlier,
        })
        pieces.append(part)

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return out.sort_values(["available_time", "window_hours"]).reset_index(drop=True)


def _add_state_evidence(out: pd.DataFrame, windows: list[int], cfg: Config | None = None) -> pd.DataFrame:
    """Add interpretable six-state evidence proxy fields.

    These are not a final six-state model.  They are audit/evidence fields that
    help downstream state engines verify that RC/RHA/AMB/VT/HPEM/ST evidence is
    coming from the intended mechanisms.
    """
    if out.empty:
        return out
    o = out.copy()
    mapping_cfg = cfg.get("state_mapping", default={}) if cfg is not None else {}
    partial_boosts_rha = bool((mapping_cfg or {}).get("partial_absorption_boosts_rha", False))
    mixed_auto_amb = bool((mapping_cfg or {}).get("mixed_pressure_auto_amb", False))
    low_activity_is_bad_quality = bool((mapping_cfg or {}).get("low_activity_is_bad_quality", False))
    quiet_supports_rc = bool((mapping_cfg or {}).get("quiet_no_pressure_supports_rc", True))
    # convenience helpers
    def any_label(substr: str, wins=(6,12,24,48)):
        vals = []
        for wh in wins:
            c = f"path_label_{wh}h"
            if c in o.columns:
                vals.append(o[c].astype(str).str.contains(substr, regex=True).astype(float))
        return sum(vals) / max(len(vals), 1) if vals else pd.Series(0.0, index=o.index)

    def any_ctx(substr: str, wins=(6,12,24,48)):
        vals = []
        for wh in wins:
            c = f"path_context_{wh}h"
            if c in o.columns:
                vals.append(o[c].astype(str).str.contains(substr, regex=True).astype(float))
        return sum(vals) / max(len(vals), 1) if vals else pd.Series(0.0, index=o.index)

    def max_label(substr: str, wins=(6,12,24,48)):
        vals = []
        for wh in wins:
            c = f"path_label_{wh}h"
            if c in o.columns:
                vals.append(o[c].astype(str).str.contains(substr, regex=True).astype(float))
        return pd.concat(vals, axis=1).max(axis=1) if vals else pd.Series(0.0, index=o.index)

    def max_ctx(substr: str, wins=(6,12,24,48)):
        vals = []
        for wh in wins:
            c = f"path_context_{wh}h"
            if c in o.columns:
                vals.append(o[c].astype(str).str.contains(substr, regex=True).astype(float))
        return pd.concat(vals, axis=1).max(axis=1) if vals else pd.Series(0.0, index=o.index)

    q6 = o.get("path_label_6h", pd.Series("", index=o.index)).astype(str)
    q12 = o.get("path_label_12h", pd.Series("", index=o.index)).astype(str)
    q24 = o.get("path_label_24h", pd.Series("", index=o.index)).astype(str)
    ctx24 = o.get("path_context_24h", pd.Series("", index=o.index)).astype(str)

    data_q = pd.concat([o.get(f"path_data_quality_{wh}h", pd.Series(np.nan, index=o.index)) for wh in windows], axis=1).mean(axis=1)
    clarity = pd.concat([o.get(f"path_signal_clarity_{wh}h", pd.Series(np.nan, index=o.index)) for wh in windows], axis=1).mean(axis=1)
    activity = pd.concat([o.get(f"path_activity_level_{wh}h", pd.Series(np.nan, index=o.index)) for wh in windows], axis=1).mean(axis=1)

    # RC evidence: quiet neutral pressure + good data + low activity/volatility.
    o["e_rc_quiet_pressure"] = any_label("path_quiet_no_pressure", wins=(6,12)).clip(0,1) if quiet_supports_rc else pd.Series(0.0, index=o.index)
    o["e_rc_good_data_quality"] = data_q.fillna(0).clip(0,1)
    # low activity is an RC support, not a quality penalty.
    o["e_rc_low_vol"] = (1.0 - activity.fillna(1.0)).clip(0,1)
    o["e_rc_range_compression"] = pd.to_numeric(o.get("path_signal_clarity_6h", 0), errors="coerce").fillna(0).clip(0,1) * o["e_rc_quiet_pressure"]
    o["e_rc_low_trend"] = o["e_rc_low_vol"]

    # RHA evidence: no direct boost from partial_absorption.
    o["e_rha_reversal_takeover"] = max_label("path_reversal_takeover", wins=(12,24,48)).clip(0,1)
    o["e_rha_pressure_rejection"] = max_label("path_pressure_rejection", wins=(12,24)).clip(0,1)
    o["e_rha_full_absorption_stall"] = max_label("path_full_absorption_stall", wins=(12,24)).clip(0,1)
    if partial_boosts_rha:
        o["e_rha_full_absorption_stall"] = pd.concat([
            o["e_rha_full_absorption_stall"],
            0.5 * max_label("path_partial_absorption", wins=(12,24)).clip(0,1),
        ], axis=1).max(axis=1)

    # AMB evidence is conflict/quality, not low activity.
    labels = [o.get(f"path_label_{wh}h", pd.Series("", index=o.index)).astype(str) for wh in windows]
    label_df = pd.concat(labels, axis=1)
    cascade_any = label_df.apply(lambda r: any("cascade" in str(v) for v in r), axis=1).astype(float)
    reject_any = label_df.apply(lambda r: any(("rejection" in str(v) or "takeover" in str(v)) for v in r), axis=1).astype(float)
    active_any = label_df.apply(lambda r: any("active_dominance" in str(v) or "mixed_active_breakout" in str(v) for v in r), axis=1).astype(float)
    quiet_any = label_df.apply(lambda r: any("quiet_no_pressure" in str(v) for v in r), axis=1).astype(float)
    mixed_ctx_any = any_ctx("path_mixed_pressure", wins=tuple(windows)).clip(0,1)
    conflict = (((cascade_any > 0) & (reject_any > 0)) | ((active_any > 0) & (reject_any > 0)) | ((cascade_any > 0) & (quiet_any > 0))).astype(float)
    if mixed_auto_amb:
        conflict = pd.concat([conflict, mixed_ctx_any], axis=1).max(axis=1)
    o["e_amb_cross_window_conflict"] = conflict
    data_bad = (1.0 - data_q.fillna(0)).clip(0,1)
    if low_activity_is_bad_quality:
        data_bad = pd.concat([data_bad, (1.0 - activity.fillna(0)).clip(0,1)], axis=1).max(axis=1)
    o["e_amb_data_quality_bad"] = data_bad
    o["e_amb_signal_conflict"] = ((1.0 - clarity.fillna(0)) * 0.6 + o["e_amb_cross_window_conflict"] * 0.4).clip(0,1)

    # state score margin as ambiguity proxy; computed after all state evidences.
    o["e_vt_active_dominance"] = max_label("active_dominance", wins=(6,12,24)).clip(0,1)
    o["e_vt_trend_breakout"] = max_label("mixed_active_breakout", wins=(6,12,24)).clip(0,1)
    o["e_vt_neutral_plie"] = max_ctx("neutral_pressure|mixed_pressure", wins=(6,12,24)).clip(0,1)

    o["e_hpem_cascade_transmission"] = max_label("path_cascade_transmission", wins=(6,12,24)).clip(0,1)
    o["e_hpem_jump"] = pd.concat([o.get(f"path_cascade_score_{wh}h", pd.Series(0, index=o.index)) for wh in (6,12,24)], axis=1).max(axis=1).fillna(0).clip(0,1)
    o["e_hpem_directional_plie"] = any_ctx("path_directional", wins=(6,12,24)).clip(0,1)

    o["e_st_partial_absorption"] = any_label("path_partial_absorption|path_baseline_transmission", wins=(6,12,24)).clip(0,1)
    o["e_st_orderly_trend"] = (1.0 - o["e_hpem_jump"].fillna(0)).clip(0,1) * o["e_st_partial_absorption"]

    # Composite proxy scores for audit only.
    o["score_rc_proxy"] = (0.25*o["e_rc_quiet_pressure"] + 0.20*o["e_rc_low_vol"] + 0.20*o["e_rc_range_compression"] + 0.15*o["e_rc_low_trend"] + 0.20*o["e_rc_good_data_quality"]).clip(0,1)
    o["score_rha_proxy"] = (0.45*o["e_rha_reversal_takeover"] + 0.35*o["e_rha_pressure_rejection"] + 0.20*o["e_rha_full_absorption_stall"]).clip(0,1)
    o["score_amb_proxy"] = (0.30*o["e_amb_signal_conflict"] + 0.35*o["e_amb_data_quality_bad"] + 0.20*o["e_amb_cross_window_conflict"]).clip(0,1)
    o["score_vt_proxy"] = (0.45*o["e_vt_active_dominance"] + 0.45*o["e_vt_trend_breakout"] + 0.10*o["e_vt_neutral_plie"]).clip(0,1)
    o["score_hpem_proxy"] = (0.40*o["e_hpem_cascade_transmission"] + 0.30*o["e_hpem_jump"] + 0.30*o["e_hpem_directional_plie"]).clip(0,1)
    o["score_st_proxy"] = (0.55*o["e_st_partial_absorption"] + 0.45*o["e_st_orderly_trend"]).clip(0,1)

    score_cols = ["score_st_proxy","score_vt_proxy","score_rc_proxy","score_rha_proxy","score_hpem_proxy","score_amb_proxy"]
    scores = o[score_cols].fillna(0)
    top = scores.idxmax(axis=1).str.replace("score_","").str.replace("_proxy","").str.upper()
    o["state_proxy_label"] = top
    sorted_vals = np.sort(scores.to_numpy(), axis=1)
    if sorted_vals.shape[1] >= 2:
        margin = sorted_vals[:, -1] - sorted_vals[:, -2]
    else:
        margin = np.nan
    o["state_proxy_margin"] = margin
    o["e_amb_label_margin_low"] = pd.Series(1.0 - pd.Series(margin, index=o.index).clip(0,1), index=o.index)
    # Recompute AMB proxy with label margin.
    o["score_amb_proxy"] = (0.25*o["e_amb_signal_conflict"] + 0.30*o["e_amb_data_quality_bad"] + 0.20*o["e_amb_cross_window_conflict"] + 0.15*o["e_amb_label_margin_low"]).clip(0,1)
    scores = o[score_cols].fillna(0)
    o["state_proxy_label"] = scores.idxmax(axis=1).str.replace("score_","").str.replace("_proxy","").str.upper()
    sorted_vals = np.sort(scores.to_numpy(), axis=1)
    o["state_proxy_margin"] = sorted_vals[:, -1] - sorted_vals[:, -2]
    return o


def build_path_absorption_multiscale(
    path_df: pd.DataFrame,
    windows: list[int] | tuple[int, ...] = (6, 12, 24, 48),
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Build a wide, online-ready multiscale path absorption table."""
    requested_bases = [
        "path_context",
        "path_label",
        "path_pressure_mass",
        "path_net_pressure",
        "path_directionality",
        "path_dominant_direction",
        "path_pressure_obs_count",
        "path_pressure_missing_ratio",
        "path_aligned_return",
        "path_baseline_impact",
        "path_absorption_score",
        "path_pressure_rejection_score",
        "path_active_dominance_score",
        "path_active_dominance_price_score",
        "path_transmission_ratio",
        "path_direction_consistency",
        "path_cascade_score",
        "path_data_quality",
        "path_signal_clarity",
        "path_activity_level",
    ]
    if path_df is None or path_df.empty:
        cols = ["time", "available_time"]
        for base in requested_bases:
            for wh in windows:
                cols.append(f"{base}_{int(wh)}h")
        return pd.DataFrame(columns=cols)

    df = path_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["available_time"] = pd.to_datetime(df["available_time"], utc=True, errors="coerce")
    df["window_hours"] = pd.to_numeric(df["window_hours"], errors="coerce").astype("Int64")

    out = (
        df[["time", "available_time"]]
        .drop_duplicates()
        .sort_values("time")
        .reset_index(drop=True)
    )

    source_map = {
        "path_context": "path_context",
        "path_label": "path_label",
        "path_pressure_mass": "path_pressure_mass",
        "path_net_pressure": "path_net_pressure",
        "path_directionality": "path_directionality",
        "path_dominant_direction": "path_dominant_direction",
        "path_pressure_obs_count": "path_pressure_obs_count",
        "path_pressure_missing_ratio": "path_pressure_missing_ratio",
        "path_aligned_return": "path_aligned_return_bps",
        "path_baseline_impact": "path_baseline_impact_bps",
        "path_absorption_score": "path_absorption_score_0_100",
        "path_pressure_rejection_score": "path_pressure_rejection_score",
        "path_active_dominance_score": "path_active_dominance_score",
        "path_active_dominance_price_score": "path_active_dominance_price_score",
        "path_transmission_ratio": "path_transmission_ratio",
        "path_direction_consistency": "path_direction_consistency",
        "path_cascade_score": "path_cascade_score",
        "path_data_quality": "path_data_quality",
        "path_signal_clarity": "path_signal_clarity",
        "path_activity_level": "path_activity_level",
    }

    for wh in windows:
        suffix = f"{int(wh)}h"
        sub = df[df["window_hours"].eq(int(wh))].copy()
        if sub.empty:
            for base in requested_bases:
                out[f"{base}_{suffix}"] = np.nan
            continue

        keep = ["time", "available_time"] + [c for c in source_map.values() if c in sub.columns]
        sub = sub[keep].drop_duplicates(subset=["time", "available_time"])
        rename = {src: f"{base}_{suffix}" for base, src in source_map.items() if src in sub.columns}
        sub = sub.rename(columns=rename)
        out = out.merge(sub, on=["time", "available_time"], how="left")

    ordered = ["time", "available_time"]
    for base in requested_bases:
        for wh in windows:
            col = f"{base}_{int(wh)}h"
            if col not in out.columns:
                out[col] = np.nan
            ordered.append(col)

    out = out[ordered].sort_values("time").reset_index(drop=True)
    out = _add_state_evidence(out, [int(w) for w in windows], cfg)
    return out
