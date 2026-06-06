from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from liq_dataflow.feature_engineering.config import FeatureEngineeringConfig
from liq_dataflow.feature_engineering.logging_utils import log_kv, log_message


@dataclass(frozen=True)
class InputFileHealth:
    path: str
    rows: int
    start_time: str | None
    latest_time: str | None
    missing_bars: int
    missing_ratio_pct: float


class CsvInputSource:
    """Load feature-engineering input from a CSV file.

    The feature project deliberately does not download remote data anymore.
    It reads a CSV produced by the sibling raw-data download project, or a local
    fallback sample CSV when the sibling output is unavailable.
    """

    def __init__(self, *, cfg: FeatureEngineeringConfig, root_dir: Path, logger=None) -> None:
        self.cfg = cfg
        self.root_dir = Path(root_dir)
        self.logger = logger

    def _resolve_candidates(self, override_path: Path | None = None) -> list[Path]:
        candidates: list[Path] = []
        for raw in [override_path, self.cfg.input.source_csv, self.cfg.input.fallback_csv]:
            if raw is None:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = (self.root_dir / path).resolve()
            candidates.append(path)
        # preserve order, drop duplicates
        out: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key not in seen:
                out.append(path)
                seen.add(key)
        return out

    @staticmethod
    def frame_health(df: pd.DataFrame, *, path: Path, time_col: str = "time", freq: str = "1h") -> InputFileHealth:
        ordered = df.copy()
        ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
        ordered = ordered.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
        if ordered.empty:
            return InputFileHealth(str(path), 0, None, None, 0, 0.0)
        expected = pd.date_range(start=ordered[time_col].iloc[0], end=ordered[time_col].iloc[-1], freq=freq)
        missing = expected.difference(pd.DatetimeIndex(ordered[time_col]))
        return InputFileHealth(
            path=str(path),
            rows=int(len(ordered)),
            start_time=str(ordered[time_col].iloc[0]),
            latest_time=str(ordered[time_col].iloc[-1]),
            missing_bars=int(len(missing)),
            missing_ratio_pct=float(len(missing) / max(len(expected), 1) * 100.0),
        )

    def load(self, *, override_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
        candidates = self._resolve_candidates(override_path=override_path)
        if not candidates:
            raise FileNotFoundError("No input CSV candidates were configured.")
        for path in candidates:
            if not path.exists():
                log_message(self.logger, f"Input CSV not found, skipping candidate: {path}")
                continue
            df = pd.read_csv(path)
            time_col = self.cfg.columns.time_col
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                health = self.frame_health(df, path=path, time_col=time_col)
            elif "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                health = self.frame_health(df, path=path, time_col="time")
            else:
                health = InputFileHealth(str(path), int(len(df)), None, None, 0, 0.0)
            log_kv(self.logger, "CSV input summary", {
                "path": health.path,
                "rows": health.rows,
                "start_time": health.start_time,
                "latest_time": health.latest_time,
                "missing_bars": health.missing_bars,
                "missing_ratio_pct": round(health.missing_ratio_pct, 6),
                "columns": ", ".join(df.columns.tolist()),
            })
            return df, path
        raise FileNotFoundError("No configured input CSV exists. Checked: " + ", ".join(str(x) for x in candidates))
