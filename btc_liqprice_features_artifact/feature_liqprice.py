"""Config-driven BTC liquidation-price feature pipeline."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict

from .config import DEFAULT_PIPELINE_CONFIG, PipelineConfig, load_pipeline_config, save_pipeline_config
from .diagnostics import generate_feature_diagnostics_report
from .execution_logger import log_dict, setup_run_logger
from .features import (
    OUTPUT_FEATURE_COLUMNS,
    compute_features,
    feature_matrix_profile,
    load_dataframe,
    prepare_decision_frame,
    save_feature_frame,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str | Path, *, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _write_manifest(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def run(config_path: str | Path | None = None) -> Dict[str, Any]:
    root = _repo_root()
    cfg: PipelineConfig = load_pipeline_config(config_path or DEFAULT_PIPELINE_CONFIG)
    input_path = _resolve_path(cfg.paths.input, root=root)
    output_path = _resolve_path(cfg.paths.output, root=root)
    resolved_config_path = _resolve_path(cfg.paths.resolved_config_output, root=root)
    manifest_path = _resolve_path(cfg.paths.run_manifest, root=root)
    log_dir = _resolve_path(cfg.paths.log_dir, root=root)
    report_dir = _resolve_path(cfg.report.output_dir, root=root)
    feature_doc_path = _resolve_path(cfg.report.feature_doc_path, root=root)

    logger, run_log = setup_run_logger('feature_liqprice', log_dir)
    try:
        logger.info('开始执行 btc_liqprice_features_artifact 配置驱动 pipeline')
        logger.info('project_root=%s', root)
        logger.info('config=%s', Path(config_path or DEFAULT_PIPELINE_CONFIG).resolve())
        logger.info('input=%s', input_path)
        logger.info('output=%s', output_path)
        logger.info('report_enabled=%s', cfg.report.enabled)
        logger.info('report_dir=%s', report_dir)
        log_dict(logger, 'resolved_pipeline_config', cfg.to_dict())

        raw_df = load_dataframe(input_path)
        decision_df, effective_bar_minutes = prepare_decision_frame(raw_df, cfg=cfg.feature_config, columns=cfg.columns)
        feat_df = compute_features(decision_df, bar_minutes=effective_bar_minutes, cfg=cfg.feature_config)

        unknown_outputs = sorted(set(cfg.columns.output_features) - set(OUTPUT_FEATURE_COLUMNS))
        if unknown_outputs:
            raise ValueError(f'columns.output_features contains unknown feature names: {unknown_outputs}')
        feat_df = feat_df[cfg.columns.output_features]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_frame(feat_df, output_path, time_col=cfg.columns.output_time_col)
        save_pipeline_config(cfg, resolved_config_path)

        profile = feature_matrix_profile(feat_df)
        log_dict(logger, 'feature_profile', profile)
        logger.info('effective_bar_minutes=%s', effective_bar_minutes)
        logger.info('feature_output_saved=%s', output_path)

        report_result: Dict[str, Any] | None = None
        if cfg.report.enabled:
            logger.info('开始生成输出特征检验与可视化报告')
            price_context = decision_df[['price']].copy() if 'price' in decision_df.columns else None
            report_result = generate_feature_diagnostics_report(
                feat_df,
                output_dir=report_dir,
                cfg=cfg.report,
                feature_doc_path=feature_doc_path,
                price_context_df=price_context,
                time_col=cfg.columns.output_time_col,
                logger=logger,
            )
            logger.info('feature_diagnostics_index=%s', report_dir / 'index.html')

        manifest = {
            'config': str(Path(config_path or DEFAULT_PIPELINE_CONFIG).resolve()),
            'input': str(input_path),
            'output': str(output_path),
            'resolved_config': str(resolved_config_path),
            'run_manifest': str(manifest_path),
            'log_file': str(run_log.resolve()),
            'effective_bar_minutes': int(effective_bar_minutes),
            'feature_profile': profile,
            'report_enabled': bool(cfg.report.enabled),
            'report_dir': str(report_dir) if cfg.report.enabled else None,
            'report_index': str(report_dir / 'index.html') if cfg.report.enabled and cfg.report.generate_html else None,
            'report_created_files': report_result['created_files'] if report_result else [],
            'report_overview': report_result['summary']['overview'] if report_result else None,
        }
        _write_manifest(manifest_path, manifest)
        logger.info('run_manifest=%s', manifest_path)
        logger.info('btc_liqprice_features_artifact pipeline 执行完成')

        print(f'feature_rows={len(feat_df)}')
        print(f'effective_bar_minutes={effective_bar_minutes}')
        print(f'output={output_path}')
        if cfg.report.enabled and cfg.report.generate_html:
            print(f'report_index={report_dir / "index.html"}')
        print(f'log_file={run_log.resolve()}')
        return manifest
    except Exception as exc:
        logger.exception('btc_liqprice_features_artifact pipeline 失败: %s', exc)
        traceback.print_exc()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the config-driven BTC liquidation-price feature pipeline.')
    parser.add_argument(
        '--config',
        default=str(DEFAULT_PIPELINE_CONFIG),
        help='Pipeline JSON config. Defaults to btc_liqprice_features_artifact/configs/feature_liqprice.json.',
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()

