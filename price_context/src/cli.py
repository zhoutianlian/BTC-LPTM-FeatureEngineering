from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .config import load_config
from .feature_pipeline import REQUIRED_OUTPUT_COLUMNS, run_pipeline


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return package_root().parent


def default_config_path() -> Path:
    return package_root() / "configs" / "feature_price_context.json"


def resolve_config_path(config_arg: str | Path | None) -> Path:
    if config_arg is None:
        return default_config_path()
    path = Path(config_arg)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (repo_root() / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build price_context OHLC feature table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON/YAML config. Defaults to price_context/configs/feature_price_context.json.",
    )
    return parser.parse_args(argv)


def setup_logging(cfg: dict) -> None:
    level_name = str(cfg.get("execution", {}).get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def print_pipeline_result(output_path: Path, features, report: dict, cfg: dict) -> None:
    execution_cfg = cfg.get("execution", {}) or {}
    if execution_cfg.get("print_summary", True):
        print(f"Generated {len(features):,} rows and {len(features.columns):,} columns.")
        print(f"Output file: {output_path}")
        zip_file = report.get("zip_file")
        if zip_file:
            print(f"Zip file: {zip_file}")
        diagnostics = report.get("feature_diagnostics") or {}
        if diagnostics.get("index_html"):
            print(f"Diagnostics report: {diagnostics['index_html']}")

    if execution_cfg.get("print_validation_report", True):
        print("Validation report:")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if execution_cfg.get("print_columns", False):
        mark_required = bool(execution_cfg.get("mark_required_columns", True))
        print("Columns:")
        for col in features.columns:
            required = "*" if mark_required and col in REQUIRED_OUTPUT_COLUMNS else " "
            print(f"{required} {col}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(resolve_config_path(args.config))
    setup_logging(cfg)
    output_path, features, report = run_pipeline(cfg)
    print_pipeline_result(output_path, features, report, cfg)
