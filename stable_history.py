from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Mapping

import pandas as pd

_TIME_TEXT_COL = "__stable_history_time_text"


def read_csv_reference(path_spec: str, *, root: Path | None = None) -> pd.DataFrame:
    """Read a CSV from a normal path or a zip member written as zip_path::member."""
    root = root or Path.cwd()
    if "::" in path_spec:
        zip_path_text, member = path_spec.split("::", 1)
        zip_path = Path(zip_path_text)
        if not zip_path.is_absolute():
            zip_path = root / zip_path
        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
            return pd.read_csv(f, low_memory=False)

    path = Path(path_spec)
    if not path.is_absolute():
        path = root / path
    return pd.read_csv(path, low_memory=False)


def _index_by_time(df: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    out = df.copy()
    if time_col not in out.columns:
        raise ValueError(f"Missing time column: {time_col}")
    out[_TIME_TEXT_COL] = out[time_col].astype(str)
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out.dropna(subset=[time_col]).set_index(time_col)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = time_col
    return out


def overlay_history(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    time_col: str = "time",
    reference_rename: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return current rows with overlapping historical cells copied from reference.

    Only columns present in both frames are overlaid. This lets newer schemas
    keep newly added columns while preserving accepted historical values.
    """
    ref = reference.rename(columns=dict(reference_rename or {}))
    cur_idx = _index_by_time(current, time_col=time_col)
    ref_idx = _index_by_time(ref, time_col=time_col)
    common_idx = ref_idx.index.intersection(cur_idx.index)
    common_cols = [c for c in cur_idx.columns if c in ref_idx.columns]
    if len(common_idx) and common_cols:
        cur_idx.loc[common_idx, common_cols] = ref_idx.loc[common_idx, common_cols]

    stats = {
        "current_rows": int(len(cur_idx)),
        "reference_rows": int(len(ref_idx)),
        "preserved_rows": int(len(common_idx)),
        "preserved_columns": int(len(common_cols)),
        "reference_min_time": str(ref_idx.index.min()) if len(ref_idx) else None,
        "reference_max_time": str(ref_idx.index.max()) if len(ref_idx) else None,
        "current_min_time": str(cur_idx.index.min()) if len(cur_idx) else None,
        "current_max_time": str(cur_idx.index.max()) if len(cur_idx) else None,
    }
    result = cur_idx.reset_index()
    if _TIME_TEXT_COL in result.columns:
        result[time_col] = result.pop(_TIME_TEXT_COL)
    return result, stats


def overlay_csv_history(
    current_path: Path,
    reference_spec: str,
    *,
    output_path: Path | None = None,
    root: Path | None = None,
    time_col: str = "time",
    reference_rename: Mapping[str, str] | None = None,
) -> dict[str, object]:
    root = root or Path.cwd()
    output_path = output_path or current_path
    current = pd.read_csv(current_path, low_memory=False)
    reference = read_csv_reference(reference_spec, root=root)
    out, stats = overlay_history(
        current,
        reference,
        time_col=time_col,
        reference_rename=reference_rename,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return {
        **stats,
        "current_path": str(current_path),
        "reference": reference_spec,
        "output_path": str(output_path),
    }


def _parse_rename(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected rename mapping old=new, got: {value}")
        old, new = value.split("=", 1)
        mapping[old] = new
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Overlay accepted historical CSV rows onto a current CSV.")
    parser.add_argument("current_path", type=Path)
    parser.add_argument("reference_spec", help="CSV path or zip_path::member")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--rename", action="append", default=[], help="Reference column rename old=new; may repeat.")
    args = parser.parse_args(argv)

    stats = overlay_csv_history(
        args.current_path,
        args.reference_spec,
        output_path=args.output,
        time_col=args.time_col,
        reference_rename=_parse_rename(args.rename),
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
