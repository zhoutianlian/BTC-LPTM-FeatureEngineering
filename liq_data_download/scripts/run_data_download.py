from __future__ import annotations

import argparse
from pathlib import Path

from liq_data_download import RawDataDownloadProject, load_download_config
from liq_data_download.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BTC Glassnode raw-data download project once.")
    parser.add_argument("--root-dir", type=Path, default=PROJECT_ROOT, help="Download-project root directory.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config.yaml override.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional dataset names to download.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write logs/catalog without hitting the API.")
    args = parser.parse_args()

    config = load_download_config(args.config)
    project = RawDataDownloadProject(root_dir=args.root_dir, config=config)
    result = project.run(only=args.only, dry_run=args.dry_run)
    print(f"[liq_data_download] catalog written to: {result.paths.catalog_csv}")
    print(f"[liq_data_download] latest log written to: {result.paths.latest_log_txt}")
    print(f"[liq_data_download] run history written to: {result.paths.run_history_csv}")


if __name__ == "__main__":
    main()
