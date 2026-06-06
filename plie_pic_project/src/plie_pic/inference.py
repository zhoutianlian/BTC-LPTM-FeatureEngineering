from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .features import broadcast_source_predictions_to_10m, build_feature_frame, get_agent_input_columns, select_bar_output_columns
from .io import load_model, read_input_frame, write_csv, write_json


def run_batch_inference(cfg: ProjectConfig) -> dict[str, Path]:
    """Run batch inference using a saved PLIE-PIC model.

    This function rebuilds source-clock causal features from the supplied input,
    applies the saved model, and broadcasts outputs to the 10m execution grid.
    """
    cfg.ensure_dirs()
    model_path = cfg.path("paths", "model_artifact")
    input_path = cfg.path("paths", "input_csv")
    artifact = load_model(model_path)
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    raw = read_input_frame(input_path)
    features = build_feature_frame(raw, cfg)
    pred = model.predict(features)
    source_path = write_csv(pred, cfg.path("paths", "latest_source_predictions"))
    bars = broadcast_source_predictions_to_10m(raw, pred, cfg)
    bar_path = write_csv(select_bar_output_columns(bars, cfg), cfg.path("paths", "latest_bar_predictions"))
    latest_agent = build_latest_agent_payload(bars, cfg)
    agent_path = write_json(latest_agent, cfg.path("paths", "latest_agent_payload"))
    return {"source_predictions": source_path, "bar_predictions": bar_path, "latest_agent_payload": agent_path}


def build_latest_agent_payload(pred_10m: pd.DataFrame, cfg: ProjectConfig) -> dict[str, Any]:
    """Build the latest Agent-facing PLIE evidence package."""
    if pred_10m.empty:
        return {}
    time_col = cfg.get("schema", "time_col")
    pred = pred_10m.sort_values(time_col)
    row = pred.iloc[-1]
    cols = [c for c in get_agent_input_columns([int(h) for h in cfg.get("features", "horizons_min")], cfg=cfg) if c in pred.columns]
    payload = {c: (None if pd.isna(row[c]) else row[c]) for c in cols}
    payload["agent_note"] = make_agent_note(payload)
    return payload


def make_agent_note(payload: dict[str, Any]) -> str:
    direction = payload.get("plie_direction")
    state = payload.get("hmm_state")
    main = payload.get("plie_main_bps")
    phase = payload.get("plie_phase")
    if direction == 1 or direction == 1.0:
        d = "upward short-liquidation forced-buy pressure"
    elif direction == -1 or direction == -1.0:
        d = "downward long-liquidation forced-sell pressure"
    else:
        d = "neutral liquidation pressure"
    return f"state={state}, {d}, phase={phase}, PLIE main passive baseline={main} bps. Compare realized price move against this baseline for absorption/residual analysis."
