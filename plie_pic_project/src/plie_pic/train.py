from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .evaluation import evaluate_predictions, evaluation_table_path
from .features import assign_chronological_split, broadcast_source_predictions_to_10m, build_feature_frame, get_agent_input_columns, select_bar_output_columns
from .io import read_input_frame, save_model, write_csv, write_json
from .model import MultiHorizonPLIEPICModel, model_feature_contribution
from .validation import NoFutureLeakageChecker, summarize_checks


@dataclass
class TrainArtifacts:
    model_path: Path
    source_features_path: Path
    source_predictions_path: Path
    bar_predictions_path: Path
    metrics_paths: dict[str, Path]
    check_path: Path


def setup_logger(cfg: ProjectConfig) -> logging.Logger:
    logger = logging.getLogger("plie_pic.train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_path = cfg.path("paths", "train_log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def train_pipeline(cfg: ProjectConfig) -> TrainArtifacts:
    """Run full offline training pipeline.

    Offline training is strictly separated from online inference. Future prices
    are generated only as labels after source-clock features are finalized.
    """
    cfg.ensure_dirs()
    logger = setup_logger(cfg)
    logger.info("Loading input frame from %s", cfg.path("paths", "input_csv"))
    raw_df = read_input_frame(cfg.path("paths", "input_csv"))

    checker = NoFutureLeakageChecker(cfg)
    pre_checks = checker.run_all(raw_df)
    if not all(r.passed for r in pre_checks if r.severity == "critical"):
        summary = summarize_checks(pre_checks)
        write_json(summary, cfg.path("paths", "pre_feature_checks"))
        raise RuntimeError(f"Critical input leakage/data checks failed. See {cfg.path('paths', 'pre_feature_checks')}")

    logger.info("Building source-clock PLIE feature frame")
    features = build_feature_frame(raw_df, cfg)
    features = assign_chronological_split(features, cfg)
    source_features_path = write_csv(features, cfg.path("paths", "source_features"))
    logger.info("Source feature frame saved: rows=%d cols=%d", len(features), features.shape[1])

    model_feature_names = list(cfg.get("model", "feature_names"))
    agent_columns = [c for c in get_agent_input_columns([int(h) for h in cfg.get("features", "horizons_min")], cfg=cfg) if c in features.columns or c.startswith("plie_passive_") or c == "plie_main_bps"]
    feature_checks = checker.run_all(raw_df, features, features, model_feature_names, agent_columns)
    write_json(summarize_checks(feature_checks), cfg.path("paths", "feature_checks"))

    train_df = features.loc[features["split"].eq("train")].copy()
    min_train = int(cfg.get("model", "min_train_rows"))
    if len(train_df) < min_train:
        raise RuntimeError(f"Training rows {len(train_df)} < min_train_rows {min_train}")

    logger.info("Fitting constrained quantile PLIE-PIC model on %d train rows", len(train_df))
    model = MultiHorizonPLIEPICModel.from_config(cfg)
    model.fit(train_df)
    pred = model.predict(features)
    logger.info("Source predictions generated: rows=%d cols=%d", len(pred), pred.shape[1])

    coef = model_feature_contribution(model)
    coef_path = write_csv(coef, evaluation_table_path(cfg, "model_coefficients"))
    model_path = save_model(
        {
            "model": model,
            "config": cfg.raw,
            "feature_names": model_feature_names,
            "agent_input_columns": get_agent_input_columns([int(h) for h in cfg.get("features", "horizons_min")], cfg=cfg),
            "train_end_time": str(train_df[cfg.get("schema", "time_col")].max()),
            "model_coefficients": coef.to_dict(orient="records"),
        },
        cfg.path("paths", "model_artifact"),
    )
    write_json(model.to_dict(), cfg.path("paths", "model_summary"))

    source_pred_path = write_csv(pred, cfg.path("paths", "train_source_predictions"))
    logger.info("Source predictions saved: %s", source_pred_path)
    bar_pred = broadcast_source_predictions_to_10m(raw_df, pred, cfg)
    logger.info("10m broadcast generated: rows=%d cols=%d", len(bar_pred), bar_pred.shape[1])
    bar_pred_path = write_csv(select_bar_output_columns(bar_pred, cfg), cfg.path("paths", "train_bar_predictions"))
    logger.info("10m compact predictions saved: %s", bar_pred_path)

    logger.info("Evaluating model outputs")
    eval_tables = evaluate_predictions(pred, cfg)
    metric_paths: dict[str, Path] = {"coefficients": coef_path}
    for name, table in eval_tables.items():
        metric_paths[name] = write_csv(table, evaluation_table_path(cfg, name))

    # Walk-forward after main training to keep artifact model trained on configured train split.
    logger.info("Running walk-forward validation")
    wf = run_walk_forward(features, cfg)
    metric_paths["walk_forward"] = write_csv(wf, evaluation_table_path(cfg, "walk_forward"))
    logger.info("Walk-forward metrics saved")

    post_checks = checker.run_all(raw_df, pred, pred, model_feature_names, get_agent_input_columns([int(h) for h in cfg.get("features", "horizons_min")], cfg=cfg))
    check_path = write_json(summarize_checks(post_checks), cfg.path("paths", "post_training_checks"))
    write_json(_latest_summary(pred, cfg), cfg.path("paths", "latest_summary"))

    return TrainArtifacts(
        model_path=model_path,
        source_features_path=source_features_path,
        source_predictions_path=source_pred_path,
        bar_predictions_path=bar_pred_path,
        metrics_paths=metric_paths,
        check_path=check_path,
    )


def run_walk_forward(features: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Rolling walk-forward validation on source-clock features."""
    if not bool(cfg.get("walk_forward", "enabled", default=True)):
        return pd.DataFrame()
    time_col = cfg.get("schema", "time_col")
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    q = float(cfg.get("features", "quantile", default=0.65))
    train_months = int(cfg.get("walk_forward", "train_months"))
    val_months = int(cfg.get("walk_forward", "validation_months"))
    step_months = int(cfg.get("walk_forward", "step_months"))
    max_folds = int(cfg.get("walk_forward", "max_folds"))

    df = features.sort_values(time_col).reset_index(drop=True).copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    start = df[time_col].min()
    end = df[time_col].max()
    rows: list[dict[str, Any]] = []
    fold = 0
    attempts = 0
    train_start = start
    while fold < max_folds and attempts < max_folds * 20:
        attempts += 1
        train_end = train_start + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)
        if val_end > end:
            break
        train = df.loc[(df[time_col] >= train_start) & (df[time_col] < train_end)].copy()
        val = df.loc[(df[time_col] >= train_end) & (df[time_col] < val_end)].copy()
        if len(train) < int(cfg.get("model", "min_train_rows")) or len(val) < 100:
            rows.append({"fold": fold, "train_start": str(train_start), "train_rows": len(train), "validation_rows": len(val), "skipped": True})
            train_start = train_start + pd.DateOffset(months=step_months)
            fold += 1
            continue
        try:
            model = MultiHorizonPLIEPICModel.from_config(cfg).fit(train)
            pred = model.predict(val)
            for h in horizons:
                y = pd.to_numeric(pred[f"plie_aligned_ret_{h}m_bps"], errors="coerce")
                p = pd.to_numeric(pred[f"plie_passive_{h}m_bps_mag"], errors="coerce")
                mask = y.notna() & p.notna()
                if not mask.any():
                    continue
                rows.append(
                    {
                        "fold": fold,
                        "train_start": str(train_start),
                        "train_end": str(train_end),
                        "validation_end": str(val_end),
                        "horizon_min": h,
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "mean_plie_mag_bps": float(p[mask].mean()),
                        "mean_aligned_actual_bps": float(y[mask].mean()),
                        "transmission_rate": float((y[mask] > 0).mean()),
                        "coverage_actual_le_plie": float((y[mask] <= p[mask]).mean()),
                        "pinball_q": float(np.mean(np.maximum(q * (y[mask].to_numpy(dtype=float) - p[mask].to_numpy(dtype=float)), (q - 1.0) * (y[mask].to_numpy(dtype=float) - p[mask].to_numpy(dtype=float))))),
                    }
                )
        except Exception as exc:  # keep diagnostics robust
            rows.append({"fold": fold, "train_start": str(train_start), "error": str(exc)})
        fold += 1
        train_start = train_start + pd.DateOffset(months=step_months)
    return pd.DataFrame(rows)


def _latest_summary(pred: pd.DataFrame, cfg: ProjectConfig) -> dict[str, Any]:
    time_col = cfg.get("schema", "time_col")
    if pred.empty:
        return {}
    row = pred.sort_values(time_col).iloc[-1]
    fields = list(cfg.get("outputs", "latest_summary_fields", default=[])) or [
        "time",
        "price",
        "hmm_state",
        "plie_direction",
        "plie_force_up",
        "plie_intensity",
        "plie_accel_pos",
        "plie_reliability",
        "plie_main_bps",
        "plie_phase",
    ]
    return {k: (None if pd.isna(row.get(k)) else row.get(k)) for k in fields if k in pred.columns}
