from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .utils import compact_float, pinball_loss, safe_corr


def evaluation_table_path(cfg: ProjectConfig, name: str):
    filename = cfg.get("evaluation", "table_files", name, default=f"{name}.csv")
    return cfg.path("paths", "evaluation_dir") / str(filename)


def _subset_metrics(df: pd.DataFrame, horizons: list[int], name: str, quantile: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in horizons:
        y = pd.to_numeric(df.get(f"plie_aligned_ret_{h}m_bps"), errors="coerce")
        pred_mag = pd.to_numeric(df.get(f"plie_passive_{h}m_bps_mag"), errors="coerce")
        signed_pred = pd.to_numeric(df.get(f"plie_passive_{h}m_bps"), errors="coerce")
        actual = pd.to_numeric(df.get(f"ret_{h}m_bps"), errors="coerce")
        mask = y.notna() & pred_mag.notna()
        if not mask.any():
            continue
        yv = y[mask].to_numpy(dtype=float)
        pv = pred_mag[mask].to_numpy(dtype=float)
        rows.append(
            {
                "subset": name,
                "horizon_min": h,
                "n": int(mask.sum()),
                "pinball_q": compact_float(pinball_loss(yv, pv, quantile), 6),
                "mae_aligned_vs_plie": compact_float(float(np.mean(np.abs(yv - pv))), 4),
                "mean_aligned_actual_bps": compact_float(float(np.mean(yv)), 4),
                "median_aligned_actual_bps": compact_float(float(np.median(yv)), 4),
                "mean_plie_mag_bps": compact_float(float(np.mean(pv)), 4),
                "transmission_rate": compact_float(float(np.mean(yv > 0.0)), 4),
                "spearman_plie_vs_aligned": compact_float(safe_corr(pred_mag[mask], y[mask], method="spearman"), 5),
                "spearman_signed_plie_vs_return": compact_float(safe_corr(signed_pred[mask], actual[mask], method="spearman"), 5),
            }
        )
    return rows


def _split_names(pred: pd.DataFrame, cfg: ProjectConfig) -> list[str]:
    if "split" not in pred.columns:
        return [str(cfg.get("evaluation", "all_subset_name", default="all"))]
    names = [str(x) for x in cfg.get("evaluation", "split_names", default=["train", "validation", "test"])]
    names.append(str(cfg.get("evaluation", "all_subset_name", default="all")))
    return names


def _split_frame(pred: pd.DataFrame, split: str, all_name: str = "all") -> pd.DataFrame:
    if split in {"all", all_name} or "split" not in pred.columns:
        return pred
    return pred.loc[pred["split"].eq(split)]


def _quantile_null_from_train(pred: pd.DataFrame, horizon: int, quantile: float) -> float:
    col = f"plie_aligned_ret_{horizon}m_bps"
    base = pred.loc[pred["split"].eq("train"), col] if "split" in pred.columns else pred[col]
    return float(np.nanquantile(pd.to_numeric(base, errors="coerce"), quantile))


def _coverage(y: pd.Series, pred_mag: pd.Series) -> float:
    joined = pd.concat([pd.to_numeric(y, errors="coerce"), pd.to_numeric(pred_mag, errors="coerce")], axis=1).dropna()
    if joined.empty:
        return float("nan")
    return float((joined.iloc[:, 0] <= joined.iloc[:, 1]).mean())


def _quantile_calibration_metrics(pred: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Quantile calibration and baseline comparison.

    PLIE-PIC is a q-quantile passive impact baseline, not a conditional mean
    return forecast. The most relevant calibration statistic is therefore
    P(aligned_actual <= PLIE_magnitude), which should be close to q.
    """
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    q = float(cfg.get("features", "quantile", default=0.65))
    all_name = str(cfg.get("evaluation", "all_subset_name", default="all"))
    rows: list[dict[str, Any]] = []
    for h in horizons:
        ycol = f"plie_aligned_ret_{h}m_bps"
        pcol = f"plie_passive_{h}m_bps_mag"
        null_q = _quantile_null_from_train(pred, h, q)
        for split in _split_names(pred, cfg):
            sub = _split_frame(pred, split, all_name)
            y = pd.to_numeric(sub.get(ycol), errors="coerce")
            p = pd.to_numeric(sub.get(pcol), errors="coerce")
            mask = y.notna() & p.notna()
            if not mask.any():
                continue
            yv = y[mask].to_numpy(dtype=float)
            pv = p[mask].to_numpy(dtype=float)
            null_pred = np.full_like(yv, null_q, dtype=float)
            zero_pred = np.zeros_like(yv, dtype=float)
            pin_model = pinball_loss(yv, pv, q)
            pin_null = pinball_loss(yv, null_pred, q)
            pin_zero = pinball_loss(yv, zero_pred, q)
            cov = float(np.mean(yv <= pv))
            rows.append(
                {
                    "split": split,
                    "horizon_min": h,
                    "n": int(mask.sum()),
                    "target_quantile": q,
                    "coverage_actual_le_plie": compact_float(cov, 4),
                    "coverage_error": compact_float(cov - q, 4),
                    "pinball_model": compact_float(pin_model, 6),
                    "pinball_null_train_q": compact_float(pin_null, 6),
                    "pinball_zero": compact_float(pin_zero, 6),
                    "improvement_vs_null_pct": compact_float(100.0 * (pin_null - pin_model) / pin_null if pin_null else np.nan, 4),
                    "improvement_vs_zero_pct": compact_float(100.0 * (pin_zero - pin_model) / pin_zero if pin_zero else np.nan, 4),
                    "train_null_q_bps": compact_float(null_q, 4),
                }
            )
    return pd.DataFrame(rows)


def _conditional_subset_metrics(pred: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Evaluate PLIE only where the mechanism should matter most."""
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    q = float(cfg.get("features", "quantile", default=0.65))
    all_name = str(cfg.get("evaluation", "all_subset_name", default="all"))
    top20 = float(cfg.get("evaluation", "subset_quantiles", "top20", default=0.80))
    top10 = float(cfg.get("evaluation", "subset_quantiles", "top10", default=0.90))
    rows: list[dict[str, Any]] = []

    for split in _split_names(pred, cfg):
        sub0 = _split_frame(pred, split, all_name)
        if sub0.empty:
            continue
        subset_masks: dict[str, pd.Series] = {
            "all": pd.Series(True, index=sub0.index),
            "state_1_5": sub0["hmm_state"].isin([1, 5]) if "hmm_state" in sub0.columns else pd.Series(False, index=sub0.index),
            "strong_entry": sub0["plie_strong_entry"].eq(1) if "plie_strong_entry" in sub0.columns else pd.Series(False, index=sub0.index),
        }
        if "plie_abs_main_bps" in sub0.columns:
            abs_main = pd.to_numeric(sub0["plie_abs_main_bps"], errors="coerce")
            subset_masks["plie_abs_top20"] = abs_main >= abs_main.quantile(top20)
            subset_masks["plie_abs_top10"] = abs_main >= abs_main.quantile(top10)
        if "plie_accel_pos" in sub0.columns:
            accel = pd.to_numeric(sub0["plie_accel_pos"], errors="coerce")
            subset_masks["accel_top10"] = accel >= accel.quantile(top10)
        if "plie_reliability" in sub0.columns:
            rel = pd.to_numeric(sub0["plie_reliability"], errors="coerce")
            subset_masks["reliability_top20"] = rel >= rel.quantile(top20)

        for subset_name, mask in subset_masks.items():
            sub = sub0.loc[mask.fillna(False)]
            if sub.empty:
                continue
            for h in horizons:
                y = pd.to_numeric(sub.get(f"plie_aligned_ret_{h}m_bps"), errors="coerce")
                p = pd.to_numeric(sub.get(f"plie_passive_{h}m_bps_mag"), errors="coerce")
                absr = pd.to_numeric(sub.get(f"plie_absorption_{h}m"), errors="coerce").replace([np.inf, -np.inf], np.nan)
                mask2 = y.notna() & p.notna()
                if not mask2.any():
                    continue
                yv = y[mask2]
                pv = p[mask2]
                rows.append(
                    {
                        "split": split,
                        "subset": subset_name,
                        "horizon_min": h,
                        "n": int(mask2.sum()),
                        "mean_aligned_actual_bps": compact_float(float(yv.mean()), 4),
                        "median_aligned_actual_bps": compact_float(float(yv.median()), 4),
                        "q_aligned_actual_bps": compact_float(float(yv.quantile(q)), 4),
                        "transmission_rate": compact_float(float((yv > 0).mean()), 4),
                        "mean_plie_mag_bps": compact_float(float(pv.mean()), 4),
                        "coverage_actual_le_plie": compact_float(float((yv <= pv).mean()), 4),
                        "mean_absorption": compact_float(float(absr.mean()), 4),
                        "spearman_abs_plie_vs_aligned": compact_float(safe_corr(pv, yv, method="spearman"), 5),
                    }
                )
    return pd.DataFrame(rows)



def _rolling_latest_monitoring(pred: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Latest rolling-window monitoring for live PLIE-PIC health checks.

    These metrics are computed only after the corresponding future-return labels
    have matured. They are not online Agent inputs. The goal is to monitor
    whether the passive-impact baseline remains calibrated on recent data,
    without retraining on every 10m/1h update.
    """
    if pred.empty or "time" not in pred.columns:
        return pd.DataFrame()
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    q = float(cfg.get("features", "quantile", default=0.65))
    windows = cfg.get("monitoring", "rolling_windows_days", default=[7, 14, 30, 60, 90]) or [7, 14, 30, 60, 90]
    top20 = float(cfg.get("evaluation", "subset_quantiles", "top20", default=0.80))
    top10 = float(cfg.get("evaluation", "subset_quantiles", "top10", default=0.90))
    time_col = cfg.get("schema", "time_col")
    df = pred.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if df.empty:
        return pd.DataFrame()
    latest_time = df[time_col].max()

    rows: list[dict[str, Any]] = []

    def subset_masks(sub0: pd.DataFrame) -> dict[str, pd.Series]:
        masks: dict[str, pd.Series] = {"all": pd.Series(True, index=sub0.index)}
        if "hmm_state" in sub0.columns:
            masks["state_1_5"] = sub0["hmm_state"].isin([1, 5])
        if "plie_strong_entry" in sub0.columns:
            masks["strong_entry"] = sub0["plie_strong_entry"].eq(1)
        if "plie_abs_main_bps" in sub0.columns and sub0["plie_abs_main_bps"].notna().any():
            abs_main = pd.to_numeric(sub0["plie_abs_main_bps"], errors="coerce")
            masks["plie_abs_top20"] = abs_main >= abs_main.quantile(top20)
            masks["plie_abs_top10"] = abs_main >= abs_main.quantile(top10)
        if "plie_accel_pos" in sub0.columns and sub0["plie_accel_pos"].notna().any():
            accel = pd.to_numeric(sub0["plie_accel_pos"], errors="coerce")
            masks["accel_top10"] = accel >= accel.quantile(top10)
        if "plie_reliability" in sub0.columns and sub0["plie_reliability"].notna().any():
            rel = pd.to_numeric(sub0["plie_reliability"], errors="coerce")
            masks["reliability_top20"] = rel >= rel.quantile(top20)
        return masks

    for days in windows:
        try:
            days_i = int(days)
        except Exception:
            continue
        start_time = latest_time - pd.Timedelta(days=days_i)
        window = df.loc[df[time_col] >= start_time].copy()
        if window.empty:
            continue
        for subset_name, mask in subset_masks(window).items():
            sub = window.loc[mask.fillna(False)].copy()
            if sub.empty:
                continue
            for h in horizons:
                y = pd.to_numeric(sub.get(f"plie_aligned_ret_{h}m_bps"), errors="coerce")
                p = pd.to_numeric(sub.get(f"plie_passive_{h}m_bps_mag"), errors="coerce")
                actual = pd.to_numeric(sub.get(f"ret_{h}m_bps"), errors="coerce")
                signed_pred = pd.to_numeric(sub.get(f"plie_passive_{h}m_bps"), errors="coerce")
                absr = pd.to_numeric(sub.get(f"plie_absorption_{h}m"), errors="coerce").replace([np.inf, -np.inf], np.nan)
                mask2 = y.notna() & p.notna()
                if not mask2.any():
                    continue
                yv = y[mask2].to_numpy(dtype=float)
                pv = p[mask2].to_numpy(dtype=float)
                null_q = _quantile_null_from_train(pred, h, q)
                null_pred = np.full_like(yv, null_q, dtype=float)
                pin_model = pinball_loss(yv, pv, q)
                pin_null = pinball_loss(yv, null_pred, q)
                coverage = float(np.mean(yv <= pv))
                rows.append(
                    {
                        "latest_time": str(latest_time),
                        "window_days": days_i,
                        "window_start": str(start_time),
                        "subset": subset_name,
                        "horizon_min": h,
                        "n": int(mask2.sum()),
                        "target_quantile": q,
                        "coverage_actual_le_plie": compact_float(coverage, 4),
                        "coverage_error": compact_float(coverage - q, 4),
                        "pinball_q": compact_float(pin_model, 6),
                        "pinball_null_train_q": compact_float(pin_null, 6),
                        "improvement_vs_null_pct": compact_float(100.0 * (pin_null - pin_model) / pin_null if pin_null else np.nan, 4),
                        "mean_aligned_actual_bps": compact_float(float(np.mean(yv)), 4),
                        "median_aligned_actual_bps": compact_float(float(np.median(yv)), 4),
                        "mean_plie_mag_bps": compact_float(float(np.mean(pv)), 4),
                        "transmission_rate": compact_float(float(np.mean(yv > 0.0)), 4),
                        "mean_absorption": compact_float(float(absr[mask2].mean()), 4),
                        "spearman_abs_plie_vs_aligned": compact_float(safe_corr(pd.Series(pv), pd.Series(yv), method="spearman"), 5),
                        "spearman_signed_plie_vs_return": compact_float(safe_corr(signed_pred[mask2], actual[mask2], method="spearman"), 5),
                    }
                )
    return pd.DataFrame(rows)

def _monotonicity_metrics(decile_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decile_metrics.empty:
        return pd.DataFrame()
    for (var, h), sub in decile_metrics.groupby(["variable", "horizon_min"]):
        if len(sub) < 5:
            continue
        rho_plie = safe_corr(sub["decile"], sub["mean_plie_mag_bps"], method="spearman")
        rho_actual = safe_corr(sub["decile"], sub["mean_aligned_actual_bps"], method="spearman")
        rows.append(
            {
                "variable": var,
                "horizon_min": int(h),
                "n_deciles": int(len(sub)),
                "rho_decile_vs_plie": compact_float(rho_plie, 4),
                "rho_decile_vs_aligned_actual": compact_float(rho_actual, 4),
                "plie_shape_pass": bool(rho_plie >= 0.95 if np.isfinite(rho_plie) else False),
                "actual_relation_note": "Actual returns include active flow/absorption; positive monotonicity is useful but not required for PLIE validity.",
            }
        )
    return pd.DataFrame(rows)


def _retrain_monitoring_metrics(calib: pd.DataFrame, rolling: pd.DataFrame | None = None, cfg: ProjectConfig | None = None) -> pd.DataFrame:
    """Convert calibration diagnostics into a simple retraining monitor.

    This is not an automatic trading decision. It is an engineering health check:
    retrain when calibration drifts, not on every new 10m/1h update.
    """
    if calib.empty:
        return pd.DataFrame()
    q = float(calib["target_quantile"].dropna().iloc[0]) if "target_quantile" in calib.columns and calib["target_quantile"].notna().any() else 0.65
    test = calib.loc[calib["split"].eq("test")].copy()
    val = calib.loc[calib["split"].eq("validation")].copy()
    ref = test if not test.empty else val
    if ref.empty:
        ref = calib
    max_cov_error = float(pd.to_numeric(ref["coverage_error"], errors="coerce").abs().max())
    min_improve_null = float(pd.to_numeric(ref["improvement_vs_null_pct"], errors="coerce").min())
    min_improve_zero = float(pd.to_numeric(ref["improvement_vs_zero_pct"], errors="coerce").min())

    rolling_ref = pd.DataFrame() if rolling is None else rolling.copy()
    latest_window_days = None
    rolling_max_cov_error = np.nan
    rolling_min_improve_null = np.nan
    if not rolling_ref.empty:
        monitor_window = 30 if cfg is None else int(cfg.get("monitoring", "primary_window_days", default=30))
        all_rows = rolling_ref.loc[(rolling_ref.get("subset") == "all") & (rolling_ref.get("window_days") == monitor_window)]
        if all_rows.empty:
            all_rows = rolling_ref.loc[rolling_ref.get("subset") == "all"]
        if not all_rows.empty:
            latest_window_days = int(pd.to_numeric(all_rows["window_days"], errors="coerce").max())
            rolling_max_cov_error = float(pd.to_numeric(all_rows["coverage_error"], errors="coerce").abs().max())
            rolling_min_improve_null = float(pd.to_numeric(all_rows["improvement_vs_null_pct"], errors="coerce").min())

    cov_trigger = max_cov_error
    improve_trigger = min_improve_null
    if np.isfinite(rolling_max_cov_error):
        cov_trigger = max(cov_trigger, rolling_max_cov_error)
    if np.isfinite(rolling_min_improve_null):
        improve_trigger = min(improve_trigger, rolling_min_improve_null)

    status = "ok"
    recommendation = "No immediate retraining required; continue online inference and rolling latest monitoring."
    coverage_retrain = 0.08 if cfg is None else float(cfg.get("monitoring", "coverage_error_retrain", default=0.08))
    coverage_watch = 0.05 if cfg is None else float(cfg.get("monitoring", "coverage_error_watch", default=0.05))
    improvement_retrain = -1.0 if cfg is None else float(cfg.get("monitoring", "improvement_vs_null_retrain_pct", default=-1.0))
    improvement_watch = 0.5 if cfg is None else float(cfg.get("monitoring", "improvement_vs_null_watch_pct", default=0.5))

    if cov_trigger > coverage_retrain or improve_trigger < improvement_retrain:
        status = "retrain_now"
        recommendation = "Retrain or recalibrate: quantile coverage or null-baseline performance has drifted beyond tolerance."
    elif cov_trigger > coverage_watch or improve_trigger < improvement_watch:
        status = "watch"
        recommendation = "Acceptable but weak: keep the model live, collect matured labels, and prioritize the next scheduled monthly retrain."
    cadence = "monthly scheduled full retrain; daily rolling monitoring; trigger retrain on calibration drift"
    if cfg is not None:
        cadence = str(cfg.get("retraining", "cadence", default=cadence)) + "; daily rolling monitoring; trigger retrain on calibration drift"
    return pd.DataFrame(
        [
            {
                "status": status,
                "target_quantile": q,
                "max_abs_coverage_error_eval": compact_float(max_cov_error, 4),
                "min_improvement_vs_null_pct_eval": compact_float(min_improve_null, 4),
                "min_improvement_vs_zero_pct_eval": compact_float(min_improve_zero, 4),
                "rolling_primary_window_days": latest_window_days,
                "rolling_max_abs_coverage_error": compact_float(rolling_max_cov_error, 4),
                "rolling_min_improvement_vs_null_pct": compact_float(rolling_min_improve_null, 4),
                "recommended_retrain_cadence": cadence,
                "recommendation": recommendation,
            }
        ]
    )


def evaluate_predictions(pred: pd.DataFrame, cfg: ProjectConfig) -> dict[str, pd.DataFrame]:
    """Evaluate PLIE-PIC with mechanism-aligned metrics."""
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    q = float(cfg.get("features", "quantile", default=0.65))
    results: dict[str, pd.DataFrame] = {}

    rows: list[dict[str, Any]] = []
    if "split" in pred.columns:
        for split in [str(x) for x in cfg.get("evaluation", "split_names", default=["train", "validation", "test"])]:
            rows.extend(_subset_metrics(pred.loc[pred["split"].eq(split)], horizons, split, q))
    rows.extend(_subset_metrics(pred, horizons, str(cfg.get("evaluation", "all_subset_name", default="all")), q))
    results["overall_metrics"] = pd.DataFrame(rows)

    # Quantile calibration and baseline comparison are the main quantitative
    # acceptance checks for this passive-impact baseline.
    results["quantile_calibration_metrics"] = _quantile_calibration_metrics(pred, cfg)

    # By HMM state.
    by_state_rows: list[dict[str, Any]] = []
    for state, sub in pred.groupby("hmm_state", dropna=True):
        for h in horizons:
            y = pd.to_numeric(sub.get(f"plie_aligned_ret_{h}m_bps"), errors="coerce")
            p = pd.to_numeric(sub.get(f"plie_passive_{h}m_bps_mag"), errors="coerce")
            mask = y.notna() & p.notna()
            if not mask.any():
                continue
            by_state_rows.append(
                {
                    "hmm_state": int(state),
                    "horizon_min": h,
                    "n": int(mask.sum()),
                    "mean_aligned_actual_bps": compact_float(float(y[mask].mean()), 4),
                    "median_aligned_actual_bps": compact_float(float(y[mask].median()), 4),
                    "mean_plie_mag_bps": compact_float(float(p[mask].mean()), 4),
                    "transmission_rate": compact_float(float((y[mask] > 0).mean()), 4),
                    "mean_absorption": compact_float(float(pd.to_numeric(sub.get(f"plie_absorption_{h}m"), errors="coerce").replace([np.inf, -np.inf], np.nan).mean()), 4),
                }
            )
    results["by_state_metrics"] = pd.DataFrame(by_state_rows)

    # By transition type.
    by_transition_rows: list[dict[str, Any]] = []
    if "plie_transition_type" in pred.columns:
        counts = pred["plie_transition_type"].value_counts().head(int(cfg.get("evaluation", "top_transition_count", default=30))).index.tolist()
        for tr in counts:
            sub = pred.loc[pred["plie_transition_type"].eq(tr)]
            for h in horizons:
                y = pd.to_numeric(sub.get(f"plie_aligned_ret_{h}m_bps"), errors="coerce")
                p = pd.to_numeric(sub.get(f"plie_passive_{h}m_bps_mag"), errors="coerce")
                mask = y.notna() & p.notna()
                if not mask.any():
                    continue
                by_transition_rows.append(
                    {
                        "transition_type": tr,
                        "horizon_min": h,
                        "n": int(mask.sum()),
                        "mean_aligned_actual_bps": compact_float(float(y[mask].mean()), 4),
                        "mean_plie_mag_bps": compact_float(float(p[mask].mean()), 4),
                        "transmission_rate": compact_float(float((y[mask] > 0).mean()), 4),
                    }
                )
    results["by_transition_metrics"] = pd.DataFrame(by_transition_rows)

    # Decile / monotonicity diagnostics.
    decile_rows: list[dict[str, Any]] = []
    decile_bins = int(cfg.get("evaluation", "decile_bins", default=10))
    for var in list(cfg.get("evaluation", "decile_variables", default=["plie_intensity", "plie_accel_pos", "plie_abs_main_bps"])):
        if var not in pred.columns:
            continue
        numeric = pd.to_numeric(pred[var], errors="coerce")
        try:
            buckets = pd.qcut(numeric.rank(method="first"), decile_bins, labels=False, duplicates="drop")
        except Exception:
            continue
        for b in sorted(buckets.dropna().unique()):
            sub = pred.loc[buckets.eq(b)]
            for h in horizons:
                y = pd.to_numeric(sub.get(f"plie_aligned_ret_{h}m_bps"), errors="coerce")
                p = pd.to_numeric(sub.get(f"plie_passive_{h}m_bps_mag"), errors="coerce")
                mask = y.notna() & p.notna()
                if not mask.any():
                    continue
                decile_rows.append(
                    {
                        "variable": var,
                        "decile": int(b),
                        "horizon_min": h,
                        "n": int(mask.sum()),
                        "mean_variable": compact_float(float(pd.to_numeric(sub[var], errors="coerce").mean()), 5),
                        "mean_plie_mag_bps": compact_float(float(p[mask].mean()), 4),
                        "mean_aligned_actual_bps": compact_float(float(y[mask].mean()), 4),
                        "transmission_rate": compact_float(float((y[mask] > 0).mean()), 4),
                    }
                )
    decile_df = pd.DataFrame(decile_rows)
    results["decile_metrics"] = decile_df
    results["monotonicity_metrics"] = _monotonicity_metrics(decile_df)

    # Conditional mechanism subsets.
    results["conditional_subset_metrics"] = _conditional_subset_metrics(pred, cfg)

    # Rolling latest monitoring for live model health.
    results["rolling_latest_monitoring"] = _rolling_latest_monitoring(pred, cfg)

    # Retraining monitor combines static test/validation calibration with latest rolling windows.
    results["retrain_monitoring"] = _retrain_monitoring_metrics(results["quantile_calibration_metrics"], results["rolling_latest_monitoring"], cfg)

    # Output sanity checks.
    results["output_checks"] = pd.DataFrame(output_sanity_checks(pred, cfg))
    return results


def output_sanity_checks(pred: pd.DataFrame, cfg: ProjectConfig) -> list[dict[str, Any]]:
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    max_bad_ratio = float(cfg.get("evaluation", "output_nan_or_inf_ratio_max", default=0.05))
    direction_pass_ratio = float(cfg.get("evaluation", "state_direction_pass_ratio", default=0.90))
    rows: list[dict[str, Any]] = []
    important = [
        "plie_direction",
        "plie_force_up",
        "plie_intensity",
        "plie_accel_pos",
        "plie_reliability",
        "plie_main_bps",
    ] + [f"plie_passive_{h}m_bps" for h in horizons]
    for col in important:
        if col not in pred.columns:
            rows.append({"check": f"exists::{col}", "passed": False, "message": "Missing output column."})
            continue
        s = pd.to_numeric(pred[col], errors="coerce") if pred[col].dtype.kind not in "OUS" else pred[col]
        if col == "plie_direction":
            valid = set(pd.Series(s).dropna().unique()).issubset({-1.0, 0.0, 1.0})
            rows.append({"check": f"valid_direction::{col}", "passed": bool(valid), "message": "Direction is in {-1,0,1}."})
            continue
        if pd.api.types.is_numeric_dtype(s):
            finite = np.isfinite(s.to_numpy(dtype=float))
            nan_ratio = float(1.0 - finite.mean()) if len(finite) else 1.0
            q01 = float(np.nanquantile(s, 0.01)) if finite.any() else np.nan
            q99 = float(np.nanquantile(s, 0.99)) if finite.any() else np.nan
            rows.append({"check": f"finite::{col}", "passed": nan_ratio < max_bad_ratio, "message": "Finite ratio acceptable.", "nan_or_inf_ratio": nan_ratio, "q01": q01, "q99": q99})
            if col == "plie_reliability":
                rng_ok = bool(((s.dropna() >= -1e-9) & (s.dropna() <= 1 + 1e-9)).all())
                rows.append({"check": "range::plie_reliability", "passed": rng_ok, "message": "Reliability lies in [0,1]."})
    # Financial logic: state 1 should usually have positive direction, state 5 negative.
    if "hmm_state" in pred.columns and "plie_direction" in pred.columns:
        s1 = pred.loc[pred["hmm_state"].eq(1), "plie_direction"]
        s5 = pred.loc[pred["hmm_state"].eq(5), "plie_direction"]
        rows.append({"check": "logic::state1_direction", "passed": bool((s1.dropna() >= 0).mean() > direction_pass_ratio if len(s1.dropna()) else True), "message": "State 1 mostly maps to upward PLIE direction."})
        rows.append({"check": "logic::state5_direction", "passed": bool((s5.dropna() <= 0).mean() > direction_pass_ratio if len(s5.dropna()) else True), "message": "State 5 mostly maps to downward PLIE direction."})
    return rows


def evaluate_hmm_state_structure(source: pd.DataFrame) -> pd.DataFrame:
    """Summarize HMM state duration and transition counts for diagnostics."""
    if source.empty or "hmm_state" not in source.columns:
        return pd.DataFrame()
    df = source.copy()
    df["prev_state"] = df["hmm_state"].shift(1)
    df["transition"] = df["prev_state"].astype("Int64").astype(str) + "->" + df["hmm_state"].astype("Int64").astype(str)
    state_counts = df["hmm_state"].value_counts().sort_index().rename_axis("hmm_state").reset_index(name="count")
    state_counts["share"] = state_counts["count"] / state_counts["count"].sum()
    return state_counts
