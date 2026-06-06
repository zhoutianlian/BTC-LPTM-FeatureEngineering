from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def read_input_frame(path: str | Path) -> pd.DataFrame:
    """Read the configured input frame.

    The project accepts ordinary CSV files and ZIP-compressed CSV files. No time
    parsing is performed here; timestamp normalization is centralized in the
    feature and validation modules.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes[-2:] == [".csv", ".zip"] or path.suffix.lower() == ".zip":
        return pd.read_csv(path, compression="zip")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {path}. Expected .csv or .zip/.csv.zip.")


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # Timezone-aware datetime formatting in pandas can be slow for large frames.
    # Convert datetime columns to stable ISO strings before writing.
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(path, index=index)
    return path


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_model(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_model(path: str | Path) -> Any:
    return joblib.load(path)
