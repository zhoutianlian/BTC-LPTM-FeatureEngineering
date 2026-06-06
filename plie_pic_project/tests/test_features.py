from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_no_future_leakage import _synthetic_raw
from plie_pic.config import load_config
from plie_pic.features import build_feature_frame


def test_plie_direction_matches_state_semantics() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    feats = build_feature_frame(_synthetic_raw(), cfg)
    state1 = feats.loc[feats["hmm_state"].eq(1), "plie_direction"]
    state5 = feats.loc[feats["hmm_state"].eq(5), "plie_direction"]
    assert (state1 >= 0).all()
    assert (state5 <= 0).all()


def test_feature_frame_contains_multi_horizon_labels() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    feats = build_feature_frame(_synthetic_raw(), cfg)
    for h in cfg.get("features", "horizons_min"):
        assert f"ret_{h}m_bps" in feats.columns
        assert f"plie_aligned_ret_{h}m_bps" in feats.columns
