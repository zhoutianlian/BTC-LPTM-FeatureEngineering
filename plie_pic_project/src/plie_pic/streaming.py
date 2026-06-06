from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .features import broadcast_source_predictions_to_10m, build_feature_frame
from .inference import build_latest_agent_payload
from .io import load_model, write_csv
from .utils import to_utc_series


@dataclass
class OnlinePLIEEngine:
    """Streaming PLIE-PIC inference engine.

    Engineering assumption: HMM state/posterior columns are supplied by the
    upstream HMM regime system. This class consumes them causally; it does not
    retrain or smooth HMM states online.
    """

    cfg: ProjectConfig
    model_artifact_path: Path | None = None
    price_store: pd.DataFrame = field(default_factory=pd.DataFrame)
    liq_state_store: pd.DataFrame = field(default_factory=pd.DataFrame)
    output_store: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if self.model_artifact_path is None:
            self.model_artifact_path = self.cfg.path("paths", "model_artifact")
        artifact = load_model(self.model_artifact_path)
        self.model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact

    def update_price_data(self, new_price_data: pd.DataFrame) -> dict[str, Any] | None:
        """Append 10m price data and emit latest Agent payload if possible."""
        self.price_store = _append_and_sort(self.price_store, new_price_data, self.cfg.get("schema", "time_col"))
        return self._infer_latest()

    def update_liquidation_state_data(self, new_liq_state_data: pd.DataFrame) -> dict[str, Any] | None:
        """Append hourly liquidation/HMM source data and emit latest payload.

        The incoming frame must include current HMM posterior/state fields. If
        HMM columns are missing, inference is refused instead of inventing a
        state estimate.
        """
        if bool(self.cfg.get("streaming", "require_hmm_columns", default=True)):
            required = [self.cfg.get("schema", "hmm_state_col")] + list(self.cfg.get("schema", "posterior_cols"))
            missing = [c for c in required if c not in new_liq_state_data.columns]
            if missing:
                raise ValueError(f"Streaming liquidation update is missing HMM columns: {missing}")
        key = self.cfg.get("schema", "liq_time_col")
        self.liq_state_store = _append_and_sort(self.liq_state_store, new_liq_state_data, key)
        return self._infer_latest()

    def _merged_current_frame(self) -> pd.DataFrame:
        """Build a current 10m-like frame from price bars and latest available source states."""
        if self.price_store.empty or self.liq_state_store.empty:
            return pd.DataFrame()
        s = self.cfg.get("schema")
        price = self.price_store.copy()
        liq = self.liq_state_store.copy()
        price[s["time_col"]] = to_utc_series(price[s["time_col"]])
        liq[s["liq_time_col"]] = to_utc_series(liq[s["liq_time_col"]])
        if s["time_col"] not in liq.columns:
            liq[s["time_col"]] = liq[s["liq_time_col"]]
        liq = liq.sort_values(s["liq_time_col"])
        price = price.sort_values(s["time_col"])
        merged = pd.merge_asof(
            price,
            liq.drop(columns=[s["time_col"]], errors="ignore"),
            left_on=s["time_col"],
            right_on=s["liq_time_col"],
            direction="backward",
            tolerance=pd.Timedelta(minutes=float(self.cfg.get("streaming", "max_liq_feature_age_min"))),
        )
        merged[s["liq_age_col"]] = (merged[s["time_col"]] - merged[s["liq_time_col"]]).dt.total_seconds() / 60.0
        return merged

    def _infer_latest(self) -> dict[str, Any] | None:
        merged = self._merged_current_frame()
        if merged.empty or merged[self.cfg.get("schema", "liq_time_col")].isna().all():
            return None
        # Feature builder will only use source rows with age == 0. For a new 10m
        # price without a new source snapshot, we infer on all available source
        # rows and then broadcast to the latest 10m bar.
        features = build_feature_frame(merged, self.cfg)
        pred_source = self.model.predict(features)
        pred_10m = broadcast_source_predictions_to_10m(merged, pred_source, self.cfg)
        latest = build_latest_agent_payload(pred_10m, self.cfg)
        self.output_store = pd.concat([self.output_store, pd.DataFrame([latest])], ignore_index=True)
        out_path = Path(self.cfg.get("streaming", "output_store"))
        if not out_path.is_absolute():
            out_path = self.cfg.project_root / out_path
        write_csv(self.output_store, out_path)
        return latest


def _append_and_sort(existing: pd.DataFrame, new: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if new is None or new.empty:
        return existing
    out = pd.concat([existing, new], ignore_index=True)
    if time_col in out.columns:
        out[time_col] = to_utc_series(out[time_col])
        out = out.drop_duplicates(time_col, keep="last").sort_values(time_col).reset_index(drop=True)
    return out
