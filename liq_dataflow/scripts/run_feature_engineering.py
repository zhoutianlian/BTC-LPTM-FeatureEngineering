from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from liq_dataflow.feature_engineering import FeatureEngineeringProject
from liq_dataflow.feature_engineering.config import PROJECT_ROOT


def build_input_frame(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "time" not in df.columns:
        raise ValueError("input csv must contain a time column")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BTC liquidation feature-engineering project once.")
    parser.add_argument("--root-dir", type=Path, default=PROJECT_ROOT, help="Project root directory containing data/, docs/ and logs/.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional input CSV override. Can be raw hourly liquidation data or a preprocessed clean frame.")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip HTML and PNG visualization generation.")
    args = parser.parse_args()

    df_input = build_input_frame(args.input_csv) if args.input_csv else None
    project = FeatureEngineeringProject(root_dir=args.root_dir, df_input=df_input)
    result = project.run(input_csv=args.input_csv if df_input is None else None, build_visualizations=not args.skip_visualization)
    print(f"[liq_dataflow] merged features written to: {result.paths.final_features_csv}")
    print(f"[liq_dataflow] feature store written to: {result.paths.feature_store_csv}")
    print(f"[liq_dataflow] overview html written to: {result.paths.feature_overview_html}")
    print(f"[liq_dataflow] validation report written to: {result.paths.validation_report_md}")
    print(f"[liq_dataflow] latest log written to: {result.paths.latest_log_txt}")
    print(f"[liq_dataflow] run history written to: {result.paths.run_history_csv}")


if __name__ == "__main__":
    main()
