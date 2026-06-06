from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .evaluation import evaluate_predictions, evaluation_table_path
from .io import read_json, write_csv, write_json
from .train import train_pipeline
from .visualization import generate_reports


@dataclass
class RetrainDecision:
    """Decision object for scheduled monthly retraining."""

    due: bool
    reason: str
    days_since_model_update: float | None
    monitor_status: str | None
    model_path: Path


def run_monitoring(cfg: ProjectConfig, generate_html: bool | None = None) -> dict[str, Path]:
    """Regenerate evaluation/monitoring tables from existing source predictions.

    This command is intended for daily or intraday monitoring after matured
    20m/30m/60m labels are available. It does not retrain the model.
    """
    cfg.ensure_dirs()
    if generate_html is None:
        generate_html = bool(cfg.get("runtime", "generate_html", default=True))
    pred_path = cfg.path("paths", "train_source_predictions")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}. Run train or infer first.")
    pred = pd.read_csv(pred_path)
    tables = evaluate_predictions(pred, cfg)
    out: dict[str, Path] = {}
    for name, table in tables.items():
        out[name] = write_csv(table, evaluation_table_path(cfg, name))
    if not pred.empty:
        row = pred.sort_values(cfg.get("schema", "time_col")).iloc[-1]
        fields = list(cfg.get("outputs", "latest_summary_fields", default=[]))
        latest = {k: (None if pd.isna(row.get(k)) else row.get(k)) for k in fields if k in pred.columns} if fields else row.to_dict()
        out["latest_summary"] = write_json(latest, cfg.path("paths", "latest_summary"))
    if generate_html:
        out.update(generate_reports(cfg))
    return out


def monthly_retrain_if_due(cfg: ProjectConfig, force: bool | None = None, generate_html: bool | None = None) -> dict[str, Any]:
    """Run a scheduled full retrain only when cadence or drift says it is due.

    This function is safe to call from cron. It is intentionally conservative:
    online 10m/1h updates should call inference/monitoring; full retraining is
    reserved for the monthly cadence or for calibration-drift trigger states.
    """
    if force is None:
        force = bool(cfg.get("runtime", "force_retrain", default=False))
    if generate_html is None:
        generate_html = bool(cfg.get("runtime", "generate_html", default=True))
    decision = should_retrain(cfg, force=force)
    result: dict[str, Any] = {
        "decision": decision.__dict__ | {"model_path": str(decision.model_path)},
        "trained": False,
    }
    if not decision.due:
        path = write_json(result, cfg.path("paths", "scheduled_retrain_decision"))
        result["decision_path"] = str(path)
        return result
    artifacts = train_pipeline(cfg)
    result["trained"] = True
    result["artifacts"] = {
        "model_path": str(artifacts.model_path),
        "source_predictions_path": str(artifacts.source_predictions_path),
        "bar_predictions_path": str(artifacts.bar_predictions_path),
        "check_path": str(artifacts.check_path),
    }
    if generate_html:
        reports = generate_reports(cfg)
        result["reports"] = {k: str(v) for k, v in reports.items()}
    path = write_json(result, cfg.path("paths", "scheduled_retrain_decision"))
    result["decision_path"] = str(path)
    return result


def should_retrain(cfg: ProjectConfig, force: bool = False) -> RetrainDecision:
    """Return whether a monthly/scheduled full retrain should run now."""
    model_path = cfg.path("paths", "model_artifact")
    if force:
        return RetrainDecision(True, "force flag supplied", _days_since_mtime(model_path), _monitor_status(cfg), model_path)
    if not model_path.exists():
        return RetrainDecision(True, "model artifact missing", None, _monitor_status(cfg), model_path)

    monitor_status = _monitor_status(cfg)
    if monitor_status == "retrain_now":
        return RetrainDecision(True, "retrain_monitoring status is retrain_now", _days_since_mtime(model_path), monitor_status, model_path)

    min_days = float(cfg.get("retraining", "min_days_between_full_retrains", default=25))
    days_since = _days_since_mtime(model_path)
    if days_since is not None and days_since >= min_days:
        return RetrainDecision(True, f"monthly cadence due: {days_since:.1f} days since model update >= {min_days:.1f}", days_since, monitor_status, model_path)

    return RetrainDecision(False, "not due: monthly cadence and drift triggers are within tolerance", days_since, monitor_status, model_path)


def _days_since_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - modified).total_seconds() / 86400.0


def _monitor_status(cfg: ProjectConfig) -> str | None:
    path = evaluation_table_path(cfg, "retrain_monitoring")
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or "status" not in df.columns:
            return None
        return str(df["status"].iloc[0])
    except Exception:
        return None
