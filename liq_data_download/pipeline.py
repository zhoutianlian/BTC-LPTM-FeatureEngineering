from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from liq_data_download.config import DownloadConfig, DownloadPaths, DatasetSpec, MetricSpec, load_download_config, pd_timestamp
from liq_data_download.glassnode_client import DataDownloadError, GlassnodeClient, load_glassnode_credentials, series_health
from liq_data_download.logging_utils import append_run_history, log_kv, log_message, log_section, setup_run_logger, utc_run_id, write_run_summary


@dataclass(frozen=True)
class DownloadResult:
    catalog: pd.DataFrame
    paths: DownloadPaths
    run_id: str


class RawDataDownloadProject:
    def __init__(self, root_dir: Path | None = None, *, config: DownloadConfig | None = None, logger=None) -> None:
        self.config = config or load_download_config()
        self.paths = DownloadPaths.from_config(root_dir=root_dir, cfg=self.config)
        self.paths.ensure_dirs()
        self.run_id = utc_run_id()
        run_logger, self.run_context = setup_run_logger(logs_dir=self.paths.logs_dir, run_id=self.run_id, name="liq_data_download")
        self.logger = run_logger
        self.external_logger = logger
        self.credentials = load_glassnode_credentials(self.config.credentials, project_root=self.paths.root_dir)
        self.client = GlassnodeClient(credentials=self.credentials, logger=self.logger)
        self._summary: dict[str, Any] = {"run_id": self.run_id, "root_dir": str(self.paths.root_dir)}
        self._catalog_rows: list[dict[str, Any]] = []

    def _log(self, message: str, level: int = logging.INFO) -> None:
        log_message(self.logger, message, level=level)
        if self.external_logger is not None and self.external_logger is not self.logger:
            log_message(self.external_logger, message, level=level)

    def _dataset_params(self, dataset: DatasetSpec) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        params: dict[str, Any] = {
            "a": dataset.asset,
            "i": dataset.interval,
            "u": int(now.timestamp()),
            "timestamp_format": "unix",
        }
        if dataset.start_time:
            params["s"] = int(pd_timestamp(dataset.start_time).timestamp())
        elif dataset.lookback_days is not None:
            params["s"] = int((now - timedelta(days=int(dataset.lookback_days))).timestamp())
        return params

    @staticmethod
    def _metric_output_names(metric: MetricSpec) -> tuple[str, ...]:
        return tuple(metric.value_fields.values()) if metric.value_fields else (metric.value_name,)

    def _download_dataset(self, dataset: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = self._dataset_params(dataset)
        frames: list[pd.DataFrame] = []
        metric_health_rows: list[dict[str, Any]] = []
        for metric in dataset.metrics:
            frame = self.client.get_series(metric.endpoint, params, value_name=metric.value_name, value_fields=metric.value_fields)
            health = series_health(frame, name=metric.value_name, freq=dataset.interval)
            metric_health_rows.append({
                "dataset": dataset.name,
                "metric": metric.value_name,
                "columns": ", ".join(self._metric_output_names(metric)),
                "endpoint": metric.endpoint,
                "rows": health.rows,
                "start_time": health.start_time,
                "latest_time": health.latest_time,
                "missing_bars": health.missing_bars,
                "missing_ratio_pct": round(health.missing_ratio_pct, 6),
                "stale_hours": None if health.stale_hours is None else round(health.stale_hours, 3),
            })
            log_kv(self.logger, f"Metric download summary | {dataset.name}::{metric.value_name}", metric_health_rows[-1])
            frames.append(frame)
        join_how = dataset.join.lower().strip()
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="time", how=join_how)
        merged = merged.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
        ds_health = series_health(merged, name=dataset.name, freq=dataset.interval)
        dataset_meta = {
            "dataset": dataset.name,
            "asset": dataset.asset,
            "interval": dataset.interval,
            "output_csv": str(self.paths.resolve_output_csv(dataset.output_csv)),
            "description": dataset.description,
            "join": dataset.join,
            "rows": ds_health.rows,
            "start_time": ds_health.start_time,
            "latest_time": ds_health.latest_time,
            "missing_bars": ds_health.missing_bars,
            "missing_ratio_pct": round(ds_health.missing_ratio_pct, 6),
            "stale_hours": None if ds_health.stale_hours is None else round(ds_health.stale_hours, 3),
            "metrics": ", ".join(column for metric in dataset.metrics for column in self._metric_output_names(metric)),
            "endpoints": " | ".join(metric.endpoint for metric in dataset.metrics),
        }
        return merged, {"dataset_meta": dataset_meta, "metric_health": metric_health_rows}

    def _validate_dataset(self, dataset: DatasetSpec, frame: pd.DataFrame, meta: dict[str, Any]) -> None:
        if frame.empty:
            raise DataDownloadError(f"Dataset {dataset.name} is empty")
        if frame["time"].duplicated().any():
            raise DataDownloadError(f"Dataset {dataset.name} contains duplicate timestamps")
        missing_ratio = float(meta["dataset_meta"]["missing_ratio_pct"])
        if self.config.validation.enabled and missing_ratio > self.config.validation.max_missing_ratio_pct:
            raise DataDownloadError(f"Dataset {dataset.name} missing_ratio_pct={missing_ratio:.3f} exceeds configured threshold")
        stale_hours = meta["dataset_meta"].get("stale_hours")
        if stale_hours is not None:
            max_stale = self.config.validation.max_stale_hours_price_10m if dataset.interval == "10m" else self.config.validation.max_stale_hours_hourly
            if float(stale_hours) > float(max_stale):
                self._log(
                    f"Dataset {dataset.name} is stale by {stale_hours:.2f} hours (threshold {max_stale:.2f} hours).",
                    level=logging.WARNING,
                )

    def _write_catalog(self) -> pd.DataFrame:
        catalog = pd.DataFrame(self._catalog_rows)
        self.paths.manifests_root.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(self.paths.catalog_csv, index=False)
        self.paths.catalog_json.write_text(json.dumps(self._catalog_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        lines = ["# Download Catalog", ""]
        if catalog.empty:
            lines.append("No datasets were downloaded.")
        else:
            for row in self._catalog_rows:
                lines.append(f"## {row['dataset']}")
                for key, value in row.items():
                    lines.append(f"- **{key}**: {value}")
                lines.append("")
        self.paths.catalog_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return catalog

    def _finalize_run(self, *, status: str, error_message: str | None = None) -> None:
        self._summary["status"] = status
        if error_message:
            self._summary["error_message"] = error_message
        self._summary["log_file"] = str(self.run_context.log_file)
        self._summary["catalog_csv"] = str(self.paths.catalog_csv)
        self._summary["datasets"] = len(self._catalog_rows)
        write_run_summary(self.run_context.run_dir, self._summary)
        shutil.copy2(self.run_context.log_file, self.paths.latest_log_txt)
        append_run_history(self.paths.run_history_csv, {
            "run_id": self.run_id,
            "status": status,
            "datasets": len(self._catalog_rows),
            "catalog_csv": str(self.paths.catalog_csv),
            "log_file": str(self.run_context.log_file),
            "run_dir": str(self.run_context.run_dir),
        })

    def run(self, *, only: Iterable[str] | None = None, dry_run: bool = False) -> DownloadResult:
        only_set = {x.strip() for x in (only or []) if str(x).strip()}
        self._summary["dry_run"] = bool(dry_run)
        log_section(self.logger, "DOWNLOAD CONFIG")
        log_kv(self.logger, "Project", {
            "project_name": self.config.project.name,
            "base_url": self.credentials.base_url,
            "requests_per_minute": self.credentials.requests_per_minute,
            "api_key_present": bool(self.credentials.api_key),
            "datasets_configured": len(self.config.datasets),
        })
        try:
            if not dry_run and not self.credentials.api_key:
                raise DataDownloadError(
                    f"No Glassnode API key found. Set {self.config.credentials.env_var} or create {self.config.credentials.config_filename}."
                )
            enabled = [d for d in self.config.datasets if d.enabled and (not only_set or d.name in only_set)]
            if not enabled:
                raise ValueError("No enabled datasets selected for download.")
            for dataset in enabled:
                log_section(self.logger, f"DATASET | {dataset.name}")
                log_kv(self.logger, "Dataset spec", {
                    "asset": dataset.asset,
                    "interval": dataset.interval,
                    "output_csv": dataset.output_csv,
                    "join": dataset.join,
                    "start_time": dataset.start_time,
                    "lookback_days": dataset.lookback_days,
                    "metrics": ", ".join(column for metric in dataset.metrics for column in self._metric_output_names(metric)),
                })
                if dry_run:
                    self._catalog_rows.append({
                        "dataset": dataset.name,
                        "asset": dataset.asset,
                        "interval": dataset.interval,
                        "output_csv": str(self.paths.resolve_output_csv(dataset.output_csv)),
                        "description": dataset.description,
                        "join": dataset.join,
                        "rows": None,
                        "start_time": dataset.start_time,
                        "latest_time": None,
                        "missing_bars": None,
                        "missing_ratio_pct": None,
                        "stale_hours": None,
                        "metrics": ", ".join(column for metric in dataset.metrics for column in self._metric_output_names(metric)),
                        "endpoints": " | ".join(metric.endpoint for metric in dataset.metrics),
                        "status": "DRY_RUN",
                    })
                    continue
                frame, meta = self._download_dataset(dataset)
                self._validate_dataset(dataset, frame, meta)
                output_csv = self.paths.resolve_output_csv(dataset.output_csv)
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(output_csv, index=False)
                self._log(f"Saved dataset {dataset.name} -> {output_csv}")
                row = dict(meta["dataset_meta"])
                row["status"] = "OK"
                self._catalog_rows.append(row)
                metric_health = pd.DataFrame(meta["metric_health"])
                metric_health_path = self.paths.manifests_root / f"{dataset.name}_metric_health.csv"
                metric_health.to_csv(metric_health_path, index=False)
                self._log(f"Saved metric health report -> {metric_health_path}")
            catalog = self._write_catalog()
            self._finalize_run(status="SUCCESS")
            self._log("Raw data download pipeline completed successfully.")
            return DownloadResult(catalog=catalog, paths=self.paths, run_id=self.run_id)
        except Exception as exc:
            self._log(f"Download pipeline failed: {exc}", level=logging.ERROR)
            self._log(traceback.format_exc(), level=logging.ERROR)
            self._finalize_run(status="FAILED", error_message=repr(exc))
            raise
