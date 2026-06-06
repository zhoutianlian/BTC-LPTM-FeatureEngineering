"""Top-level runner for the liq_dataflow feature engineering project."""

from __future__ import annotations

import argparse
from pathlib import Path

from liq_dataflow.feature_engineering import FeatureEngineeringProject, load_feature_engineering_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BTC liquidation feature engineering from project config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config JSON/YAML path. Defaults to liq_dataflow/configs/feature_engineering.json.",
    )
    args = parser.parse_args()

    cfg = load_feature_engineering_config(args.config)
    project = FeatureEngineeringProject(config=cfg)
    result = project.run()

    print(f"[liq_dataflow] merged features written to: {result.paths.final_features_csv}")
    print(f"[liq_dataflow] feature store written to: {result.paths.feature_store_csv}")
    print(f"[liq_dataflow] validation report written to: {result.paths.validation_report_md}")
    print(f"[liq_dataflow] latest log written to: {result.paths.latest_log_txt}")


if __name__ == "__main__":
    main()
