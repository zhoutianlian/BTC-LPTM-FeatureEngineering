from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "BTC Glassnode Raw Data Download"
    description: str = "Download raw BTC market/liquidation datasets from Glassnode into CSV files."


@dataclass(frozen=True)
class CredentialsConfig:
    env_var: str = "GLASSNODE_API_KEY"
    config_filename: str = "glassnode.yaml"
    base_url: str = "https://api.glassnode.com/v1"
    requests_per_minute: int = 10


@dataclass(frozen=True)
class OutputConfig:
    raw_root: str = "data/raw"
    manifests_root: str = "data/manifests"
    logs_root: str = "logs"
    latest_log_filename: str = "latest.log"
    run_history_filename: str = "run_history.csv"
    catalog_csv_filename: str = "download_catalog.csv"
    catalog_md_filename: str = "download_catalog.md"
    catalog_json_filename: str = "download_catalog.json"


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    raise_on_error: bool = True
    max_missing_ratio_pct: float = 5.0
    max_stale_hours_price_10m: float = 3.5
    max_stale_hours_hourly: float = 6.0


@dataclass(frozen=True)
class MetricSpec:
    endpoint: str
    value_name: str
    description: str = ""
    value_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    enabled: bool = True
    asset: str = "BTC"
    interval: str = "1h"
    output_csv: str = ""
    description: str = ""
    join: str = "inner"
    start_time: str | None = None
    lookback_days: int | None = None
    metrics: tuple[MetricSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DownloadConfig:
    project: ProjectConfig = ProjectConfig()
    credentials: CredentialsConfig = CredentialsConfig()
    output: OutputConfig = OutputConfig()
    validation: ValidationConfig = ValidationConfig()
    datasets: tuple[DatasetSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DownloadPaths:
    root_dir: Path
    raw_root: Path
    manifests_root: Path
    logs_dir: Path
    runs_dir: Path
    latest_log_txt: Path
    run_history_csv: Path
    catalog_csv: Path
    catalog_md: Path
    catalog_json: Path

    @classmethod
    def from_config(cls, *, root_dir: Path | None = None, cfg: DownloadConfig) -> "DownloadPaths":
        root = Path(root_dir) if root_dir is not None else PROJECT_ROOT
        raw_root = root / cfg.output.raw_root
        manifests_root = root / cfg.output.manifests_root
        logs_dir = root / cfg.output.logs_root
        runs_dir = logs_dir / "runs"
        return cls(
            root_dir=root,
            raw_root=raw_root,
            manifests_root=manifests_root,
            logs_dir=logs_dir,
            runs_dir=runs_dir,
            latest_log_txt=logs_dir / cfg.output.latest_log_filename,
            run_history_csv=logs_dir / cfg.output.run_history_filename,
            catalog_csv=manifests_root / cfg.output.catalog_csv_filename,
            catalog_md=manifests_root / cfg.output.catalog_md_filename,
            catalog_json=manifests_root / cfg.output.catalog_json_filename,
        )

    def resolve_output_csv(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root_dir / path

    def ensure_dirs(self) -> None:
        for path in [self.raw_root, self.manifests_root, self.logs_dir, self.runs_dir]:
            path.mkdir(parents=True, exist_ok=True)


def _construct_dataclass(dc_cls: type, payload: dict[str, Any] | None):
    return dc_cls(**(payload or {}))


def _parse_metrics(payload: list[dict[str, Any]] | None) -> tuple[MetricSpec, ...]:
    metrics: list[MetricSpec] = []
    for item in payload or []:
        spec = dict(item)
        spec["value_fields"] = dict(spec.get("value_fields") or {})
        metrics.append(MetricSpec(**spec))
    return tuple(metrics)


def _parse_datasets(payload: dict[str, Any] | None) -> tuple[DatasetSpec, ...]:
    datasets: list[DatasetSpec] = []
    for name, spec in (payload or {}).items():
        item = dict(spec or {})
        item.setdefault("name", name)
        item["metrics"] = _parse_metrics(item.get("metrics"))
        datasets.append(DatasetSpec(**item))
    return tuple(datasets)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_download_config(config_path: Path | None = None) -> DownloadConfig:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    payload = load_yaml(path)
    return DownloadConfig(
        project=_construct_dataclass(ProjectConfig, payload.get("project")),
        credentials=_construct_dataclass(CredentialsConfig, payload.get("credentials")),
        output=_construct_dataclass(OutputConfig, payload.get("output")),
        validation=_construct_dataclass(ValidationConfig, payload.get("validation")),
        datasets=_parse_datasets(payload.get("datasets")),
    )


def to_unix_ts(value: str | None) -> int | None:
    if value is None:
        return None
    ts = pd_timestamp(value)
    return int(ts.timestamp())


def pd_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
