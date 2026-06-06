from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..cli import resolve_config_path, setup_logging
from ..config import load_config, resolve_path
from .report_builder import generate_feature_diagnostics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate price_context feature diagnostics from an existing feature CSV.")
    parser.add_argument("--config", default=None, help="Path to JSON/YAML config.")
    parser.add_argument("--features", default=None, help="Override report.features_file or output.feature_file from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_config_path(args.config))
    setup_logging(cfg)
    if args.features:
        feature_path = Path(args.features)
        if not feature_path.is_absolute():
            feature_path = Path.cwd() / feature_path
    elif cfg.get("report", {}).get("features_file"):
        feature_path = resolve_path(cfg["report"]["features_file"], cfg.get("_project_root", "."))
    else:
        output_dir = resolve_path(cfg["output"]["output_dir"], cfg.get("_project_root", "."))
        feature_path = output_dir / cfg["output"].get("feature_file", "price_context_features.csv")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_path}")

    features = pd.read_csv(feature_path)
    summary = generate_feature_diagnostics_report(features, cfg, output_path=feature_path)
    print(json.dumps({k: summary[k] for k in ["index_html", "summary_json", "feature_total", "fail_count", "warn_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
