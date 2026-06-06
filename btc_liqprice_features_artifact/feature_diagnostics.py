"""Standalone config-driven feature diagnostics report entry."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DEFAULT_PIPELINE_CONFIG, load_pipeline_config
from .diagnostics import generate_feature_diagnostics_report
from .execution_logger import setup_run_logger
from .features import load_dataframe, prepare_decision_frame
from .visualization import load_feature_frame


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str | Path, *, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def run(config_path: str | Path | None = None) -> Dict[str, Any]:
    root = _repo_root()
    cfg = load_pipeline_config(config_path or DEFAULT_PIPELINE_CONFIG)
    feature_path = _resolve_path(cfg.paths.output, root=root)
    input_path = _resolve_path(cfg.paths.input, root=root)
    report_dir = _resolve_path(cfg.report.output_dir, root=root)
    doc_path = _resolve_path(cfg.report.feature_doc_path, root=root)
    log_dir = _resolve_path(cfg.paths.log_dir, root=root)

    logger, run_log = setup_run_logger('feature_diagnostics', log_dir)
    try:
        logger.info('开始单独生成输出特征检验与可视化报告')
        logger.info('features=%s', feature_path)
        logger.info('report_dir=%s', report_dir)
        feature_df = load_feature_frame(feature_path)
        price_context: Optional[Any] = None
        if cfg.report.price_context_enabled and input_path.exists():
            raw_df = load_dataframe(input_path)
            decision_df, _ = prepare_decision_frame(raw_df, cfg=cfg.feature_config, columns=cfg.columns)
            price_context = decision_df[['price']].copy() if 'price' in decision_df.columns else None
        elif cfg.report.price_context_enabled:
            logger.warning('price_context_skipped=input path does not exist: %s', input_path)

        result = generate_feature_diagnostics_report(
            feature_df,
            output_dir=report_dir,
            cfg=cfg.report,
            feature_doc_path=doc_path,
            price_context_df=price_context,
            time_col=cfg.columns.output_time_col,
            logger=logger,
        )
        print(f'created={len(result["created_files"])}')
        print(f'report_index={report_dir / "index.html"}')
        print(f'log_file={run_log.resolve()}')
        return result
    except Exception as exc:
        logger.exception('单独生成特征诊断报告失败: %s', exc)
        traceback.print_exc()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate feature diagnostics report from the configured feature output.')
    parser.add_argument(
        '--config',
        default=str(DEFAULT_PIPELINE_CONFIG),
        help='Pipeline JSON config. Defaults to btc_liqprice_features_artifact/configs/feature_liqprice.json.',
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()

