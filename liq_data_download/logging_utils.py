from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class RunLogContext:
    run_id: str
    run_dir: Path
    log_file: Path


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def setup_run_logger(*, logs_dir: Path, run_id: str, name: str) -> tuple[logging.Logger, RunLogContext]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = logs_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"{name}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    text_log = run_dir / "pipeline.log"
    json_log = run_dir / "pipeline.jsonl"

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    file_handler = logging.FileHandler(text_log, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    json_handler = logging.FileHandler(json_log, encoding="utf-8")
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(JsonLineFormatter())

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.addHandler(json_handler)

    return logger, RunLogContext(run_id=run_id, run_dir=run_dir, log_file=text_log)


def log_message(logger: logging.Logger | None, message: str, level: int = logging.INFO) -> None:
    if logger is not None:
        logger.log(level, message)


def log_section(logger: logging.Logger | None, title: str) -> None:
    log_message(logger, f"{'=' * 16} {title} {'=' * 16}")


def log_kv(logger: logging.Logger | None, title: str, payload: dict[str, Any]) -> None:
    if logger is None:
        return
    log_message(logger, title)
    for key, value in payload.items():
        log_message(logger, f"  - {key}: {value}")


def append_run_history(history_csv: Path, row: dict[str, Any]) -> None:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = history_csv.exists()
    fieldnames = list(row.keys())
    with history_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_run_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Download Run Summary", ""]
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")
    (run_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
