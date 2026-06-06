"""Project-level executable entry for QD-MAR feature generation.

Run from the repository root:

    python -m qd_mar_project.feature_qd_mar
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .qdmar.config import Config
from .qdmar.pipeline import run_pipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="QD-MAR absorption feature pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qd_mar_project/configs/default.yaml"),
        help="Path to YAML config. Relative paths are resolved from the repository root.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    outputs = run_pipeline(config_path)
    cfg = Config.from_yaml(config_path)
    if bool(cfg.run_flag("print_outputs", True)):
        print("QD-MAR pipeline completed. Outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
