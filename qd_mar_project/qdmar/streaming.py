"""Streaming-oriented QD-MAR updater.

This module separates online inference from offline calibration/training. It is
intended as a production skeleton: new price bars and new source-clock PLIE
rows are appended, pending events mature when price reaches event_time+h, and
only matured absorption memory is emitted to the Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd

from .config import Config
from .calibration import load_calibrators
from .absorption import prepare_base_context, compute_absorption_events, compute_absorption_curve
from .memory import build_memory_features


@dataclass
class StreamingQDMarState:
    """Minimal streaming state container.

    In a real deployment, these frames should be persisted to object storage or
    a database after each update. The class keeps the algorithm contract clear:
    calibration is loaded from offline state, while online updates never refit
    thresholds or use not-yet-matured labels.
    """

    cfg: Config
    calibrators_path: Path
    raw_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    matured_absorption: pd.DataFrame = field(default_factory=pd.DataFrame)
    curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    memory: pd.DataFrame = field(default_factory=pd.DataFrame)

    def load_calibrators(self) -> dict[str, Any]:
        return load_calibrators(self.calibrators_path)

    def append_plie_events(self, new_events: pd.DataFrame) -> None:
        """Append new PLIE source-clock events and maintain time order."""
        self.raw_events = pd.concat([self.raw_events, new_events], ignore_index=True)
        self.raw_events["time"] = pd.to_datetime(self.raw_events["time"], utc=True)
        self.raw_events = self.raw_events.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    def recompute_matured_from_available_labels(self, current_time: pd.Timestamp) -> pd.DataFrame:
        """Recompute matured event rows available at current_time.

        Engineering assumption: the input event rows include actual return labels
        only after they have become available. In production, these labels should
        be computed from the price grid once price at event_time+h exists.
        """
        if self.raw_events.empty:
            return pd.DataFrame()
        current_time = pd.Timestamp(current_time, tz="UTC") if pd.Timestamp(current_time).tzinfo is None else pd.Timestamp(current_time).tz_convert("UTC")
        base = prepare_base_context(self.raw_events, self.cfg)
        event_df = compute_absorption_events(base, self.cfg, self.load_calibrators())
        event_df = event_df[pd.to_datetime(event_df["available_time"], utc=True) <= current_time].copy()
        self.matured_absorption = event_df
        self.curve = compute_absorption_curve(event_df) if not event_df.empty else pd.DataFrame()
        self.memory = build_memory_features(base[base["time"] <= current_time], self.matured_absorption, self.curve, self.cfg) if not event_df.empty else pd.DataFrame()
        return self.memory.tail(1)
