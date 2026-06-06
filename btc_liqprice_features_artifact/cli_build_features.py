from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

try:
    from .config import DEFAULT_CONFIG_FILENAME, FeatureConfig, ReportConfig, load_config, save_config
    from .diagnostics import generate_feature_diagnostics_report
    from .execution_logger import log_dict, setup_run_logger
    from .features import build_features_from_path, feature_matrix_profile, load_dataframe, prepare_decision_frame, save_feature_frame
except ImportError:  # pragma: no cover - kept for direct script execution
    from config import DEFAULT_CONFIG_FILENAME, FeatureConfig, ReportConfig, load_config, save_config
    from diagnostics import generate_feature_diagnostics_report
    from execution_logger import log_dict, setup_run_logger
    from features import build_features_from_path, feature_matrix_profile, load_dataframe, prepare_decision_frame, save_feature_frame



def _default_report_dir(project_root: Path) -> Path:
    return project_root / 'reports' / 'feature_diagnostics'



def _default_resolved_config_path(output_path: Path) -> Path:
    return output_path.parent / DEFAULT_CONFIG_FILENAME



def _save_run_manifest(output_dir: Path, payload: dict) -> Path:
    path = output_dir / 'run_manifest.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path



def main() -> None:
    parser = argparse.ArgumentParser(description='构建 BTC 清算—价格特征，并自动生成特征检验与可视化报告。')
    parser.add_argument('--input', required=True, help='输入 CSV/Parquet，至少包含 time, price, fll_cwt_kf, fsl_cwt_kf。')
    parser.add_argument('--output', required=True, help='输出特征 CSV/Parquet 路径。')
    parser.add_argument('--config', default=None, help='可选 JSON 配置路径。')
    parser.add_argument('--save-resolved-config', default=None, help='可选：保存解析后配置的路径。默认保存到输出目录。')
    parser.add_argument('--html-dir', default=None, help='兼容旧参数：报告输出目录。默认 btc_liqprice_features_artifact/reports/feature_diagnostics。')
    parser.add_argument('--report-dir', default=None, help='可选：特征诊断报告输出目录。')
    parser.add_argument('--rolling-window-minutes', type=int, default=24 * 60, help='可视化滚动统计窗口，单位分钟。')
    parser.add_argument('--log-dir', default=None, help='日志目录。默认项目根目录/logs。')
    parser.add_argument('--no-visualize', action='store_true', help='仅计算特征，不自动生成 HTML。')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir_arg = args.report_dir or args.html_dir
    report_dir = Path(report_dir_arg).resolve() if report_dir_arg else _default_report_dir(project_root)
    resolved_cfg_path = Path(args.save_resolved_config).resolve() if args.save_resolved_config else _default_resolved_config_path(output_path)
    log_dir = Path(args.log_dir).resolve() if args.log_dir else project_root / 'logs'

    logger, run_log = setup_run_logger('build_features', log_dir)
    try:
        logger.info('开始执行特征工程任务')
        logger.info('project_root=%s', project_root)
        logger.info('input=%s', Path(args.input).resolve())
        logger.info('output=%s', output_path)
        logger.info('report_dir=%s', report_dir)
        logger.info('resolved_config=%s', resolved_cfg_path)
        logger.info('auto_visualize=%s', not args.no_visualize)

        cfg: FeatureConfig = load_config(args.config)
        log_dict(logger, 'resolved_config', cfg.to_dict())

        feat_df, effective_bar_minutes = build_features_from_path(args.input, cfg)
        save_feature_frame(feat_df, output_path)
        save_config(cfg, resolved_cfg_path)

        profile = feature_matrix_profile(feat_df)
        log_dict(logger, 'feature_profile', profile)
        logger.info('effective_bar_minutes=%s', effective_bar_minutes)
        logger.info('feature_output_saved=%s', output_path)

        created_report = []
        if not args.no_visualize:
            logger.info('开始生成输出特征检验与可视化报告')
            report_cfg = ReportConfig(output_dir=str(report_dir), rolling_window_minutes=int(args.rolling_window_minutes))
            raw_df = load_dataframe(args.input)
            decision_df, _ = prepare_decision_frame(raw_df, cfg)
            report_result = generate_feature_diagnostics_report(
                feat_df,
                output_dir=report_dir,
                cfg=report_cfg,
                feature_doc_path=project_root / 'docs' / 'liqprice_feature_engineering.md',
                price_context_df=decision_df[['price']] if 'price' in decision_df.columns else None,
                logger=logger,
            )
            created_report = report_result['created_files']
            logger.info('report_file_count=%s', len(created_report))
            for p in created_report:
                logger.info('report_created=%s', Path(p).resolve())

        manifest = {
            'input': str(Path(args.input).resolve()),
            'output': str(output_path),
            'report_dir': str(report_dir),
            'resolved_config': str(resolved_cfg_path),
            'log_file': str(run_log.resolve()),
            'effective_bar_minutes': int(effective_bar_minutes),
            'auto_visualize': bool(not args.no_visualize),
            'feature_profile': profile,
            'report_outputs': [str(Path(p).resolve()) for p in created_report],
        }
        manifest_path = _save_run_manifest(output_path.parent, manifest)
        logger.info('run_manifest=%s', manifest_path.resolve())
        logger.info('特征工程任务执行完成')

        print(f'feature_rows={len(feat_df)}')
        print(f'effective_bar_minutes={effective_bar_minutes}')
        print(f'output={output_path}')
        if not args.no_visualize:
            print(f'report_index={report_dir / "index.html"}')
        print(f'log_file={run_log.resolve()}')
    except Exception as exc:
        logger.exception('特征工程任务失败: %s', exc)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
