#!/usr/bin/env python
"""Run QD-MAR offline pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qdmar.config import Config
from qdmar.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QD-MAR absorption pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML rendering")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = ROOT / config_path
    outputs = run_pipeline(config_path, make_html=False if args.no_html else None)
    cfg = Config.from_yaml(config_path)
    if bool(cfg.run_flag("print_outputs", True)):
        print("QD-MAR pipeline completed. Outputs:")
        for k, v in outputs.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
