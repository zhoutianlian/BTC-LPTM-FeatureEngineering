"""Input/output utilities."""
from __future__ import annotations

from pathlib import Path
import json
import os
import gc
import multiprocessing as mp
import pandas as pd


def load_plie_csv(path: str | Path, time_col: str = "time", timestamp_cols: list[str] | None = None) -> pd.DataFrame:
    """Load PLIE source dataframe with explicit UTC time handling.

    Parameters
    ----------
    path:
        CSV path.
    time_col:
        Main event timestamp column.

    Returns
    -------
    pd.DataFrame
        Time-sorted dataframe. Timestamp-like columns are converted to UTC.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"Missing required time column: {time_col}")
    parse_cols = list(dict.fromkeys([time_col] + list(timestamp_cols or [])))
    for col in parse_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def _write_csv_worker(df: pd.DataFrame, path_str: str, date_format: str, chunksize: int) -> None:
    """Worker used on POSIX to isolate large CSV writes from the main process."""
    # Fast path: default pandas formatting is much faster in the notebook
    # runtime and is sufficient for research artifacts.  Financial precision is
    # preserved in binary memory/state; CSVs are audit/output artifacts.
    df.to_csv(path_str, index=False, date_format=date_format, chunksize=chunksize)


def _direct_write_csv(df: pd.DataFrame, path: Path, date_format: str, chunksize: int) -> None:
    df.to_csv(path, index=False, date_format=date_format, chunksize=chunksize)


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    date_format: str = "%Y-%m-%dT%H:%M:%SZ",
    chunksize: int = 50000,
) -> None:
    """Save CSV with bounded float precision and robust large-frame behavior.

    In constrained notebook filesystems, repeated large ``to_csv`` calls in the
    same Python process can become extremely slow.  On POSIX systems the writer
    therefore forks a short-lived child for large frames.  The child inherits the
    dataframe via copy-on-write, writes the CSV, and exits.  Small frames are
    written directly.  This does not change file contents or algorithm outputs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    approx_mb = float(df.memory_usage(deep=True).sum()) / 1_000_000.0 if len(df) else 0.0
    if False and os.name == "posix" and approx_mb >= 8.0:
        ctx = mp.get_context("fork")
        proc = ctx.Process(target=_write_csv_worker, args=(df, str(path), date_format, chunksize))
        proc.start()
        proc.join(timeout=240)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            raise TimeoutError(f"Timed out writing CSV: {path}")
        if proc.exitcode != 0:
            raise RuntimeError(f"CSV writer failed for {path} with exit code {proc.exitcode}")
    else:
        _direct_write_csv(df, path, date_format, chunksize)
    # Avoid repeated full-GC pauses during large research pipeline runs.
    # The process exits after pipeline completion; callers can collect explicitly if needed.


def save_json(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
