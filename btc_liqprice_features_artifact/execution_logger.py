from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def setup_run_logger(task_name: str, log_dir: str | Path) -> tuple[logging.Logger, Path]:
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    stamp = utc_now_str()
    run_log = log_root / f'{task_name}_{stamp}.log'
    latest_log = log_root / f'latest_{task_name}.log'

    logger_name = f'btc_liqprice_features.{task_name}.{stamp}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    file_handler = logging.FileHandler(run_log, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    latest_handler = logging.FileHandler(latest_log, mode='w', encoding='utf-8')
    latest_handler.setFormatter(formatter)
    logger.addHandler(latest_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info('日志初始化完成')
    logger.info('task_name=%s', task_name)
    logger.info('run_log=%s', run_log.resolve())
    logger.info('latest_log=%s', latest_log.resolve())
    return logger, run_log


def log_dict(logger: logging.Logger, title: str, payload: Dict[str, Any]) -> None:
    logger.info('%s=%s', title, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def finalize_latest_copy(run_log: str | Path, log_dir: str | Path, task_name: str) -> None:
    src = Path(run_log)
    dst = Path(log_dir) / f'latest_{task_name}.log'
    if src.exists():
        shutil.copyfile(src, dst)

