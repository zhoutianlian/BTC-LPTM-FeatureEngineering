from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path


def load_ohlc_csv(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load raw OHLC CSV without silently coercing schema errors."""
    project_root = cfg.get("_project_root", ".")
    file_path = resolve_path(cfg["input"]["file_path"], project_root)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input OHLC file not found: {file_path}. Update input.file_path in the config file."
        )
    return pd.read_csv(file_path)
