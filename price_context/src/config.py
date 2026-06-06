from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing required sections or invalid values."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    cfg = _read_config_file(path)
    validate_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_root"] = str(_infer_project_root(path, cfg))
    return cfg


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            cfg = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            cfg = yaml.safe_load(f)
        else:
            raise ConfigError(f"Unsupported config file extension: {path.suffix}. Use .json, .yaml, or .yml.")
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ConfigError("Config root must be a mapping/object.")
    return cfg


def _infer_project_root(path: Path, cfg: dict[str, Any]) -> Path:
    paths_cfg = cfg.get("paths", {}) or {}
    configured_root = paths_cfg.get("project_root")
    if configured_root:
        root = Path(configured_root)
        if not root.is_absolute():
            root = path.resolve().parent / root
        return root.resolve()
    resolved = path.resolve()
    if resolved.parent.name == "configs":
        return resolved.parent.parent
    return resolved.parent


def validate_config(cfg: dict[str, Any]) -> None:
    required_sections = ["input", "data", "windows", "quality", "realized_vol", "range", "trend", "jump", "output"]
    missing = [sec for sec in required_sections if sec not in cfg]
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    input_required = ["file_path", "time_column", "open_column", "high_column", "low_column", "close_column"]
    missing_input = [k for k in input_required if k not in cfg["input"]]
    if missing_input:
        raise ConfigError(f"Missing input config keys: {missing_input}")

    bar_minutes = cfg["data"].get("bar_minutes")
    if not isinstance(bar_minutes, (int, float)) or bar_minutes <= 0:
        raise ConfigError("data.bar_minutes must be a positive number.")

    min_obs_ratio = cfg["quality"].get("min_obs_ratio", 0.8)
    if not (0 < float(min_obs_ratio) <= 1):
        raise ConfigError("quality.min_obs_ratio must be in (0, 1].")
    if str(cfg["quality"].get("outlier_method", "rolling_robust")) not in {"rolling_robust", "none"}:
        raise ConfigError("quality.outlier_method must be 'rolling_robust' or 'none'.")
    if str(cfg["quality"].get("robust_sigma_estimator", "iqr")) not in {"iqr", "mad"}:
        raise ConfigError("quality.robust_sigma_estimator must be 'iqr' or 'mad'.")

    for key in ["return_windows", "core_windows", "vol_of_vol_windows"]:
        if key not in cfg["windows"] or not isinstance(cfg["windows"][key], list) or not cfg["windows"][key]:
            raise ConfigError(f"windows.{key} must be a non-empty list.")

    rv_cfg = cfg["realized_vol"]
    if str(rv_cfg.get("zscore_method", "train_robust")) not in {"train_robust", "past_rolling_robust"}:
        raise ConfigError("realized_vol.zscore_method must be 'train_robust' or 'past_rolling_robust'.")
    if str(rv_cfg.get("fallback_when_train_missing", "past_rolling_robust")) != "past_rolling_robust":
        raise ConfigError("realized_vol.fallback_when_train_missing currently supports only 'past_rolling_robust'.")

    if str(cfg["range"].get("compression_method", "past_percentile")) != "past_percentile":
        raise ConfigError("range.compression_method currently supports only 'past_percentile'.")

    if str(cfg["trend"].get("regression_time_unit", "hour")) not in {"minute", "hour", "day"}:
        raise ConfigError("trend.regression_time_unit must be one of: minute, hour, day.")

    if str(cfg["jump"].get("robust_sigma_estimator", "iqr")) not in {"iqr", "mad"}:
        raise ConfigError("jump.robust_sigma_estimator must be 'iqr' or 'mad'.")

    output_cfg = cfg["output"]
    if "output_dir" not in output_cfg or "feature_file" not in output_cfg:
        raise ConfigError("output.output_dir and output.feature_file are required.")
    if str(output_cfg.get("csv_writer", "pandas")) not in {"pandas", "fast"}:
        raise ConfigError("output.csv_writer must be either 'pandas' or 'fast'.")
    float_precision = int(output_cfg.get("float_precision", 10))
    if float_precision < 1:
        raise ConfigError("output.float_precision must be a positive integer.")

    execution_cfg = cfg.get("execution", {}) or {}
    level = str(execution_cfg.get("log_level", "INFO")).upper()
    if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
        raise ConfigError("execution.log_level must be a standard Python logging level.")

    report_cfg = cfg.get("report", {}) or {}
    if report_cfg and "output_dir" not in report_cfg:
        raise ConfigError("report.output_dir is required when report is configured.")


def resolve_path(path_value: str | Path, project_root: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(project_root) / path).resolve()
