from __future__ import annotations

import argparse
import traceback
from pathlib import Path

try:
    from .config import ReportConfig
    from .diagnostics import generate_feature_diagnostics_report
    from .execution_logger import setup_run_logger
    from .features import OUTPUT_FEATURE_COLUMNS
    from .visualization import load_feature_frame
except ImportError:  # pragma: no cover - kept for direct script execution
    from config import ReportConfig
    from diagnostics import generate_feature_diagnostics_report
    from execution_logger import setup_run_logger
    from features import OUTPUT_FEATURE_COLUMNS
    from visualization import load_feature_frame



def main() -> None:
    parser = argparse.ArgumentParser(description='为已有特征文件单独重建特征检验与可视化报告。')
    parser.add_argument('--input', required=True, help='特征 CSV/Parquet，必须包含 time 列。')
    parser.add_argument('--out-dir', required=True, help='HTML 输出目录。')
    parser.add_argument('--features', nargs='*', default=None, help='可选：仅可视化指定特征。')
    parser.add_argument('--rolling-window-minutes', type=int, default=24 * 60, help='滚动统计窗口，单位分钟。')
    parser.add_argument('--log-dir', default=None, help='日志目录。默认项目根目录/logs。')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    log_dir = Path(args.log_dir).resolve() if args.log_dir else project_root / 'logs'
    logger, run_log = setup_run_logger('visualize_features', log_dir)
    try:
        logger.info('开始重建 HTML 可视化')
        logger.info('input=%s', Path(args.input).resolve())
        logger.info('out_dir=%s', Path(args.out_dir).resolve())
        df = load_feature_frame(args.input)
        features = args.features if args.features else [c for c in OUTPUT_FEATURE_COLUMNS if c in df.columns]
        logger.info('feature_count=%s', len(features))
        report_cfg = ReportConfig(output_dir=str(Path(args.out_dir).resolve()), rolling_window_minutes=int(args.rolling_window_minutes))
        if args.features:
            df = df[['time'] + [c for c in args.features if c in df.columns]].copy()
        result = generate_feature_diagnostics_report(
            df,
            output_dir=args.out_dir,
            cfg=report_cfg,
            feature_doc_path=project_root / 'docs' / 'liqprice_feature_engineering.md',
            price_context_df=df[['time', 'price']] if 'price' in df.columns else None,
            logger=logger,
        )
        created = result['created_files']
        logger.info('created_count=%s', len(created))
        for p in created:
            logger.info('created=%s', Path(p).resolve())
        print(f'created={len(created)}')
        print(f'index={Path(args.out_dir).resolve() / "index.html"}')
        print(f'log_file={run_log.resolve()}')
    except Exception as exc:
        logger.exception('重建 HTML 可视化失败: %s', exc)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
