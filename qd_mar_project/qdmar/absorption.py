"""QD-MAR absorption feature computation.

The implementation follows the finalized algorithm design:
context-gated, raw-PLIE-referenced, volatility-denoised, quantile-calibrated,
and matured-memory-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, HorizonConfig
from .statistics import past_matured_sigma, assign_train_quantile_regime, winsorize_by_sigma
from .calibration import BucketCDFCalibrator


def _state_severity_bucket(state_severity: pd.Series) -> pd.Series:
    s = pd.to_numeric(state_severity, errors="coerce")
    out = pd.Series("unknown", index=s.index, dtype=object)
    out.loc[s == 2] = "strong"
    out.loc[s == 1] = "mild"
    out.loc[s == 0] = "neutral"
    return out


def _phase_group(phase: pd.Series) -> pd.Series:
    p = phase.fillna("unknown").astype(str)
    out = p.copy()
    out.loc[~out.isin(["early_strong_entry", "accelerating", "mature", "normal", "neutral"])] = "other"
    return out


def prepare_base_context(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Prepare shared past-only context columns for all horizons."""
    out = df.copy()
    time_col = cfg.get("data", "time_col", default="time")
    split_col = cfg.get("data", "split_col", default="split")
    train_split = cfg.get("calibration", "train_split", default="train")
    train_mask = out[split_col].eq(train_split) if split_col in out else pd.Series(False, index=out.index)

    if "state_severity" in out.columns:
        out["state_severity_bucket"] = _state_severity_bucket(out["state_severity"])
    else:
        out["state_severity_bucket"] = "unknown"
    out["plie_phase_group"] = _phase_group(out.get("plie_phase", pd.Series("unknown", index=out.index)))

    sigma_cfg = cfg.get("absorption", "sigma")
    for h in cfg.horizons:
        sigma = past_matured_sigma(
            out,
            time_col=time_col,
            ret_col=h.ret_col,
            horizon_minutes=h.minutes,
            window=int(sigma_cfg["window"]),
            min_periods=int(sigma_cfg["min_periods"]),
        )
        out[f"mar_sigma_past_{h.name}_bps"] = sigma

    # Main vol regime uses 30m if available; fallback to first horizon.
    main_h = cfg.get("memory", "main_horizon", default="30m")
    sigma_col = f"mar_sigma_past_{main_h}_bps"
    if sigma_col not in out.columns:
        sigma_col = f"mar_sigma_past_{cfg.horizons[0].name}_bps"
    out["vol_regime"] = assign_train_quantile_regime(out[sigma_col], train_mask)
    return out


def classify_response_context(
    direction: pd.Series,
    reliability: pd.Series,
    snr_raw: pd.Series,
    b_raw: pd.Series,
    b_min: float,
    reliability_min: float,
    snr_min: float,
    weak_reliability_min: float = 0.30,
    weak_snr_min: float = 0.10,
) -> pd.Series:
    """Classify rows into directional-core, weak-directional, true-neutral, or low-quality contexts.

    ``directional_core`` is the only context where calibrated directional
    absorption labels are considered high-confidence.  ``weak_directional_context``
    keeps the financial fact that PLIE still has a non-zero pressure direction,
    but exposes it only as low-confidence evidence rather than collapsing it into
    neutral active labels.  ``true_neutral_plie`` is reserved for genuinely
    directionless PLIE.
    """
    d = pd.to_numeric(direction, errors="coerce").fillna(0)
    r = pd.to_numeric(reliability, errors="coerce").fillna(0)
    snr = pd.to_numeric(snr_raw, errors="coerce").fillna(0)
    b = pd.to_numeric(b_raw, errors="coerce").fillna(0)

    ctx = pd.Series("low_quality_plie", index=direction.index, dtype=object)
    ctx.loc[d == 0] = "true_neutral_plie"

    core = (d != 0) & (r >= reliability_min) & (snr >= snr_min) & (b >= b_min)
    ctx.loc[core] = "directional_core"

    weak = (
        (d != 0) & ~core &
        ((r >= weak_reliability_min) | (snr >= weak_snr_min) | (b >= b_min))
    )
    ctx.loc[weak] = "weak_directional_context"
    return ctx


def _directional_label(u: float, y: float, b_raw: float, sigma: float, z_amp: float, cfg: Config) -> str:
    labels = cfg.get("absorption", "labels")
    if not np.isfinite(u):
        return "unavailable"
    deadzone_cfg = cfg.get("absorption", "deadzone", default={}) or {}
    b_frac = float(deadzone_cfg.get("b_raw_fraction", 0.15))
    sigma_frac = float(deadzone_cfg.get("sigma_fraction", 0.10))
    deadzone = max(b_frac * float(b_raw), sigma_frac * float(sigma)) if np.isfinite(b_raw) and np.isfinite(sigma) else np.nan
    if u >= labels["u_amplification"] and z_amp > 0:
        return "passive_amplification"
    if labels["u_baseline"] <= u < labels["u_amplification"]:
        return "baseline_transmission"
    if labels["u_normal_low"] <= u < labels["u_baseline"]:
        return "normal_response"
    if u < labels["u_partial_low"] and y < 0:
        return "reversal_takeover"
    if u < labels["u_normal_low"] and np.isfinite(deadzone) and abs(y) < deadzone:
        return "full_absorption_stall"
    if labels["u_partial_low"] <= u < labels["u_normal_low"]:
        return "partial_absorption"
    if u < labels["u_normal_low"]:
        return "partial_absorption"
    return "normal_response"


def _neutral_label(active_z: float, cfg: Config) -> str:
    labels = cfg.get("absorption", "labels")
    if not np.isfinite(active_z):
        return "unavailable"
    if active_z < labels["active_z_low"]:
        return "low_active_move"
    if active_z < labels["active_z_normal"]:
        return "normal_active_move"
    if active_z < labels["active_z_strong"]:
        return "strong_active_move"
    return "extreme_active_move"


def _weak_directional_label(y: float, tr_raw: float, b_raw: float, sigma: float, cfg: Config) -> str:
    """Low-confidence directional response label for non-core directional PLIE.

    These labels are not high-confidence absorption labels.  They preserve the
    financial context that PLIE has a direction while warning downstream Agent
    logic that reliability/SNR was insufficient for a calibrated directional
    classification.
    """
    if not np.isfinite(y):
        return "weak_directional_unavailable"
    weak_cfg = cfg.get("absorption", "weak_directional_labels", default={}) or {}
    deadzone_cfg = cfg.get("absorption", "deadzone", default={}) or {}
    b_frac = float(deadzone_cfg.get("b_raw_fraction", 0.15))
    sigma_frac = float(deadzone_cfg.get("sigma_fraction", 0.10))
    deadzone = max(b_frac * float(b_raw), sigma_frac * float(sigma)) if np.isfinite(b_raw) and np.isfinite(sigma) else np.nan
    if np.isfinite(deadzone) and abs(y) <= deadzone:
        return "weak_directional_stall_candidate"
    if np.isfinite(tr_raw) and tr_raw >= float(weak_cfg.get("amplification_tr", 1.25)):
        return "weak_directional_amplification_candidate"
    if y > 0:
        return "weak_directional_transmission_candidate"
    if y < 0:
        return "weak_directional_rejection_candidate"
    return "weak_directional_uncertain"


def build_calibrator_training_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build a long frame of directional-core rows for response percentile calibration."""
    rows = []
    reliability_min = float(cfg.get("absorption", "reliability_min"))
    snr_min = float(cfg.get("absorption", "snr_min"))
    weak_reliability_min = float(cfg.get("absorption", "weak_reliability_min", default=0.30))
    weak_snr_min = float(cfg.get("absorption", "weak_snr_min", default=0.10))
    eps = float(cfg.get("absorption", "eps"))
    time_col = cfg.get("data", "time_col", default="time")
    split_col = cfg.get("data", "split_col", default="split")
    for h in cfg.horizons:
        d = pd.to_numeric(df["plie_direction"], errors="coerce")
        ret = pd.to_numeric(df[h.ret_col], errors="coerce")
        b_raw = pd.to_numeric(df[h.raw_mag_col], errors="coerce")
        sigma = pd.to_numeric(df[f"mar_sigma_past_{h.name}_bps"], errors="coerce")
        snr = b_raw / (sigma + eps)
        ctx = classify_response_context(
            d, df["plie_reliability"], snr, b_raw, h.b_min_bps, reliability_min, snr_min, weak_reliability_min, weak_snr_min
        )
        y = d * ret
        tr = y / (b_raw + eps)
        part = pd.DataFrame({
            "time": df[time_col],
            "split": df.get(split_col, "unknown"),
            "horizon": h.name,
            "tr_raw": tr,
            "response_context": ctx,
            "state_severity_bucket": df["state_severity_bucket"],
            "plie_phase_group": df["plie_phase_group"],
            "vol_regime": df["vol_regime"],
        })
        rows.append(part[part["response_context"].eq("directional_core")])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_neutral_training_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build a long frame for neutral/weak active move calibration."""
    rows = []
    reliability_min = float(cfg.get("absorption", "reliability_min"))
    snr_min = float(cfg.get("absorption", "snr_min"))
    weak_reliability_min = float(cfg.get("absorption", "weak_reliability_min", default=0.30))
    weak_snr_min = float(cfg.get("absorption", "weak_snr_min", default=0.10))
    eps = float(cfg.get("absorption", "eps"))
    time_col = cfg.get("data", "time_col", default="time")
    split_col = cfg.get("data", "split_col", default="split")
    for h in cfg.horizons:
        d = pd.to_numeric(df["plie_direction"], errors="coerce")
        ret = pd.to_numeric(df[h.ret_col], errors="coerce")
        b_raw = pd.to_numeric(df[h.raw_mag_col], errors="coerce")
        sigma = pd.to_numeric(df[f"mar_sigma_past_{h.name}_bps"], errors="coerce")
        snr = b_raw / (sigma + eps)
        ctx = classify_response_context(
            d, df["plie_reliability"], snr, b_raw, h.b_min_bps, reliability_min, snr_min, weak_reliability_min, weak_snr_min
        )
        active_z = ret.abs() / (sigma + eps)
        part = pd.DataFrame({
            "time": df[time_col],
            "split": df.get(split_col, "unknown"),
            "horizon": h.name,
            "active_z": active_z,
            "response_context": ctx,
            "vol_regime": df["vol_regime"],
        })
        rows.append(part[~part["response_context"].eq("directional_core")])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_calibrators(base_df: pd.DataFrame, cfg: Config) -> dict[str, BucketCDFCalibrator]:
    """Fit train-only directional and neutral empirical CDF calibrators."""
    train_split = cfg.get("calibration", "train_split", default="train")
    min_bucket_size = int(cfg.get("calibration", "min_bucket_size"))

    directional = build_calibrator_training_frame(base_df, cfg)
    neutral = build_neutral_training_frame(base_df, cfg)

    directional_cal = BucketCDFCalibrator(
        value_col="tr_raw",
        bucket_levels=cfg.get("calibration", "bucket_levels"),
        min_bucket_size=min_bucket_size,
    )
    directional_cal.fit(directional, directional["split"].eq(train_split))

    neutral_cal = BucketCDFCalibrator(
        value_col="active_z",
        bucket_levels=cfg.get("calibration", "neutral_bucket_levels"),
        min_bucket_size=min_bucket_size,
    )
    neutral_cal.fit(neutral, neutral["split"].eq(train_split))
    return {"directional": directional_cal, "neutral": neutral_cal}


def compute_absorption_events(base_df: pd.DataFrame, cfg: Config, calibrators: dict[str, BucketCDFCalibrator]) -> pd.DataFrame:
    """Compute event-level matured QD-MAR absorption rows in long format."""
    eps = float(cfg.get("absorption", "eps"))
    reliability_min = float(cfg.get("absorption", "reliability_min"))
    snr_min = float(cfg.get("absorption", "snr_min"))
    weak_reliability_min = float(cfg.get("absorption", "weak_reliability_min", default=0.30))
    weak_snr_min = float(cfg.get("absorption", "weak_snr_min", default=0.10))
    snr_low = float(cfg.get("absorption", "snr_low"))
    snr_full = float(cfg.get("absorption", "snr_full"))
    lambda_sigma = float(cfg.get("absorption", "lambda_sigma"))
    tanh_scale = float(cfg.get("absorption", "neutral_tanh_scale"))
    time_col = cfg.get("data", "time_col", default="time")
    split_col = cfg.get("data", "split_col", default="split")

    pieces = []
    for h in cfg.horizons:
        d = pd.to_numeric(base_df["plie_direction"], errors="coerce").fillna(0)
        ret = pd.to_numeric(base_df[h.ret_col], errors="coerce")
        b_raw = pd.to_numeric(base_df[h.raw_mag_col], errors="coerce")
        plie_eff = pd.to_numeric(base_df[h.plie_col], errors="coerce")
        b_eff = plie_eff.abs()
        reliability = pd.to_numeric(base_df["plie_reliability"], errors="coerce").fillna(0)
        sigma = pd.to_numeric(base_df[f"mar_sigma_past_{h.name}_bps"], errors="coerce")
        snr = b_raw / (sigma + eps)
        w_snr = ((snr - snr_low) / (snr_full - snr_low)).clip(lower=0, upper=1)
        quality_weight = reliability * w_snr
        ctx = classify_response_context(d, reliability, snr, b_raw, h.b_min_bps, reliability_min, snr_min, weak_reliability_min, weak_snr_min)

        y = d * ret
        tr_raw = y / (b_raw + eps)
        absorption_raw = 1.0 - tr_raw
        scale = np.sqrt(np.square(b_raw) + np.square(lambda_sigma * sigma) + eps)
        z_amp = (y - b_raw) / scale
        z_abs = (b_raw - y) / scale
        z_takeover = (-y) / scale
        active_resid_raw = ret - d * b_raw
        active_resid_wins = winsorize_by_sigma(
            active_resid_raw,
            sigma,
            k=float(cfg.get("absorption", "active_residual_winsor_sigma", default=5.0)),
        )
        active_resid_denoised = quality_weight * active_resid_wins

        part = pd.DataFrame({
            "event_time": base_df[time_col],
            "available_time": pd.to_datetime(base_df[time_col], utc=True) + pd.to_timedelta(h.minutes, unit="m"),
            "horizon": h.name,
            "horizon_minutes": h.minutes,
            "split": base_df.get(split_col, "unknown"),
            "price": base_df.get("price"),
            "hmm_state": base_df.get("hmm_state"),
            "state_severity": base_df.get("state_severity"),
            "state_severity_bucket": base_df["state_severity_bucket"],
            "plie_phase": base_df.get("plie_phase"),
            "plie_phase_group": base_df["plie_phase_group"],
            "vol_regime": base_df["vol_regime"],
            "plie_direction": d,
            "plie_reliability": reliability,
            "plie_intensity": base_df.get("plie_intensity"),
            "liq_entropy": base_df.get("liq_entropy"),
            "age_in_state_source": base_df.get("age_in_state_source"),
            "signed_plie_effective_bps": plie_eff,
            "plie_reference_raw_bps": b_raw,
            "plie_effective_abs_bps": b_eff,
            "actual_return_bps": ret,
            "aligned_actual_response_bps": y,
            "snr_raw": snr,
            "volatility_sigma_past_bps": sigma,
            "quality_weight": quality_weight,
            "response_context": ctx,
            "transmission_ratio_raw": tr_raw,
            "absorption_raw": absorption_raw,
            "z_amp": z_amp,
            "z_abs": z_abs,
            "z_takeover": z_takeover,
            "active_residual_raw_bps": active_resid_raw,
            "active_residual_denoised_bps": active_resid_denoised,
        })

        core_mask = part["response_context"].eq("directional_core")
        part["response_percentile"] = np.nan
        part["calibration_bucket_id"] = None
        part["calibration_bucket_n"] = np.nan
        if core_mask.any():
            calib_in = part.loc[core_mask, ["horizon", "state_severity_bucket", "plie_phase_group", "vol_regime", "transmission_ratio_raw"]].rename(
                columns={"transmission_ratio_raw": "tr_raw"}
            )
            cal = calibrators["directional"].transform(calib_in)
            part.loc[core_mask, "response_percentile"] = cal["percentile"].to_numpy()
            part.loc[core_mask, "calibration_bucket_id"] = cal["bucket_id"].to_numpy()
            part.loc[core_mask, "calibration_bucket_n"] = cal["bucket_n"].to_numpy()

        part["absorption_score_q_0_100"] = 100.0 * (1.0 - part["response_percentile"])
        part["active_force_aligned_score"] = quality_weight * (2.0 * part["response_percentile"] - 1.0)
        part["active_force_price_score"] = d * part["active_force_aligned_score"]

        # Neutral / weak branch.
        part["active_z"] = ret.abs() / (sigma + eps)
        part["neutral_active_percentile"] = np.nan
        part["neutral_active_strength_score_0_100"] = np.nan
        part["neutral_active_label"] = None
        non_core_mask = ~core_mask
        if non_core_mask.any():
            neutral_in = part.loc[non_core_mask, ["horizon", "vol_regime", "active_z"]]
            ncal = calibrators["neutral"].transform(neutral_in)
            part.loc[non_core_mask, "neutral_active_percentile"] = ncal["percentile"].to_numpy()
            part.loc[non_core_mask, "neutral_active_strength_score_0_100"] = 100.0 * ncal["percentile"].to_numpy()
            part.loc[non_core_mask, "neutral_active_label"] = [
                _neutral_label(v, cfg) for v in part.loc[non_core_mask, "active_z"].to_numpy()
            ]
        w_neutral = 1.0 - quality_weight.clip(lower=0, upper=1)
        part["neutral_active_force_price_score"] = np.sign(ret) * np.tanh(part["active_z"] / tanh_scale) * w_neutral

        labels = []
        market_labels = []
        for _, row in part.iterrows():
            ctx_row = row["response_context"]
            if ctx_row == "directional_core":
                lab = _directional_label(
                    row["response_percentile"],
                    row["aligned_actual_response_bps"],
                    row["plie_reference_raw_bps"],
                    row["volatility_sigma_past_bps"],
                    row["z_amp"],
                    cfg,
                )
                labels.append(lab)
                market_labels.append(lab)
            elif ctx_row == "weak_directional_context":
                lab = _weak_directional_label(
                    row["aligned_actual_response_bps"],
                    row["transmission_ratio_raw"],
                    row["plie_reference_raw_bps"],
                    row["volatility_sigma_past_bps"],
                    cfg,
                )
                labels.append(lab)
                market_labels.append(lab)
            elif ctx_row == "true_neutral_plie":
                labels.append(None)
                nlab = row["neutral_active_label"] if row["neutral_active_label"] is not None else "unavailable"
                market_labels.append(f"neutral_{nlab}")
            else:
                labels.append(None)
                nlab = row["neutral_active_label"] if row["neutral_active_label"] is not None else "unavailable"
                market_labels.append(f"low_quality_{nlab}")
        part["directional_absorption_label"] = labels
        part["market_response_label"] = market_labels
        pieces.append(part)

    event_df = pd.concat(pieces, ignore_index=True)
    event_df = event_df.sort_values(["available_time", "event_time", "horizon"]).reset_index(drop=True)
    return event_df


def _curve_label(row: pd.Series, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {}
    u20, u30, u60 = row.get("u20"), row.get("u30"), row.get("u60")
    active30 = row.get("neutral_active_strength_30m", np.nan)
    if not (np.isfinite(u20) and np.isfinite(u30) and np.isfinite(u60)):
        if np.isfinite(active30) and active30 >= float(thresholds.get("neutral_active_dominance_percentile", 85.0)):
            return "neutral_active_dominance"
        return "mixed_or_noise"
    low = float(thresholds.get("persistent_low", 0.20))
    high = float(thresholds.get("persistent_high", 0.80))
    fast_high = float(thresholds.get("fast_high", 0.75))
    sustained_high = float(thresholds.get("sustained_high", 0.65))
    delayed_low = float(thresholds.get("delayed_low", 0.35))
    delayed_high = float(thresholds.get("delayed_high", 0.65))
    fast_abs_low_20 = float(thresholds.get("fast_absorption_u20", 0.25))
    fast_abs_low_other = float(thresholds.get("fast_absorption_u30_u60", 0.35))
    if u20 <= low and u30 <= low and u60 <= low:
        return "persistent_reversal_takeover"
    if u20 >= high and u30 >= high and u60 >= high:
        return "persistent_cascade_transmission"
    if u20 >= fast_high and u30 >= sustained_high and u60 >= sustained_high:
        return "fast_to_sustained_transmission"
    if u20 <= delayed_low and u60 >= delayed_high:
        return "delayed_transmission"
    if u20 >= delayed_high and u60 <= delayed_low:
        return "delayed_absorption"
    if u20 <= fast_abs_low_20 and u30 <= fast_abs_low_other and u60 <= fast_abs_low_other:
        return "fast_absorption_or_takeover"
    return "mixed_or_noise"


def compute_absorption_curve(event_df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Build event-level 20/30/60m response percentile curve labels."""
    pivot = event_df.pivot_table(index="event_time", columns="horizon", values="response_percentile", aggfunc="first")
    curve_cfg = cfg.get("absorption", "curve", default={}) if cfg is not None else {}
    horizon_cols = (curve_cfg or {}).get("horizon_percentile_columns", {"20m": "u20", "30m": "u30", "60m": "u60"})
    pivot = pivot.rename(columns=horizon_cols)
    neu = event_df.pivot_table(index="event_time", columns="horizon", values="neutral_active_strength_score_0_100", aggfunc="first")
    if "30m" in neu.columns:
        pivot["neutral_active_strength_30m"] = neu["30m"]
    meta_cols = ["event_time", "split", "hmm_state", "plie_phase", "plie_direction"]
    meta = event_df.sort_values("event_time").drop_duplicates("event_time")[meta_cols].set_index("event_time")
    curve = pivot.join(meta, how="left")
    curve = curve.reset_index()
    maturity_minutes = (curve_cfg or {}).get("maturity_minutes")
    if maturity_minutes in (None, "auto") and cfg is not None:
        maturity_minutes = max(h.minutes for h in cfg.horizons)
    elif maturity_minutes in (None, "auto"):
        maturity_minutes = 60
    curve["available_time"] = pd.to_datetime(curve["event_time"], utc=True) + pd.to_timedelta(float(maturity_minutes), unit="m")
    thresholds = (curve_cfg or {}).get("label_thresholds", {})
    curve["mar_curve_label"] = curve.apply(lambda row: _curve_label(row, thresholds), axis=1)
    percentile_cols = [c for c in ["u20", "u30", "u60"] if c in curve.columns]
    curve["mar_response_conflict_score"] = (curve[percentile_cols].max(axis=1) - curve[percentile_cols].min(axis=1)).astype(float) if percentile_cols else np.nan
    return curve.sort_values("available_time").reset_index(drop=True)
