from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12

# ---------------------------------------------------------------------
# Stable state semantics
# ---------------------------------------------------------------------
# NOTE ON TERMINOLOGY
# - Short liquidation means shorts are forced to buy back -> upward liquidation pressure.
# - Long liquidation means longs are forced to sell -> downward liquidation pressure.
# - We therefore avoid ambiguous names such as "bull/bear dominance" in emitted
#   columns.  Legacy aliases p_bull/p_bear are still emitted for backward
#   compatibility, but the preferred names are p_up_pressure/p_down_pressure.

STATE_ROLE_MAP: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "short_liq_strong",
        "name_en": "Short-liquidation strong dominance",
        "name_cn": "空头清算强势占优",
        "liq_side": "short_liquidation",
        "liq_side_cn": "空头清算",
        "pressure": "up",
        "pressure_en": "strong upward liquidation pressure",
        "pressure_cn": "清算量向上压力强势",
        "pressure_direction": 1,
        "severity": 2,
    },
    2: {
        "name": "short_liq_mild",
        "name_en": "Short-liquidation mild dominance",
        "name_cn": "空头清算轻度占优",
        "liq_side": "short_liquidation",
        "liq_side_cn": "空头清算",
        "pressure": "up",
        "pressure_en": "mild upward liquidation pressure",
        "pressure_cn": "清算量向上压力轻度",
        "pressure_direction": 1,
        "severity": 1,
    },
    3: {
        "name": "balanced_liq",
        "name_en": "Balanced long/short liquidations",
        "name_cn": "空头/多头清算均衡",
        "liq_side": "balanced",
        "liq_side_cn": "空头/多头清算均衡",
        "pressure": "neutral",
        "pressure_en": "no clear liquidation pressure",
        "pressure_cn": "清算量对价格没有明显压力",
        "pressure_direction": 0,
        "severity": 0,
    },
    4: {
        "name": "long_liq_mild",
        "name_en": "Long-liquidation mild dominance",
        "name_cn": "多头清算轻度占优",
        "liq_side": "long_liquidation",
        "liq_side_cn": "多头清算",
        "pressure": "down",
        "pressure_en": "mild downward liquidation pressure",
        "pressure_cn": "清算量向下压力轻度",
        "pressure_direction": -1,
        "severity": 1,
    },
    5: {
        "name": "long_liq_strong",
        "name_en": "Long-liquidation strong dominance",
        "name_cn": "多头清算强势占优",
        "liq_side": "long_liquidation",
        "liq_side_cn": "多头清算",
        "pressure": "down",
        "pressure_en": "strong downward liquidation pressure",
        "pressure_cn": "清算量向下压力强势",
        "pressure_direction": -1,
        "severity": 2,
    },
}


def _state_cols(one_based_states: bool = True) -> List[str]:
    return [f"p_state_{i}" for i in (range(1, 6) if one_based_states else range(5))]


def _normalize_posteriors(df: pd.DataFrame, p_cols: Iterable[str]) -> pd.DataFrame:
    p_cols = list(p_cols)
    p = df[p_cols].apply(pd.to_numeric, errors="coerce").astype(float).clip(lower=0.0).fillna(0.0)
    denom = p.sum(axis=1)
    valid = denom > 0
    if valid.any():
        p.loc[valid, :] = p.loc[valid, :].div(denom.loc[valid], axis=0)
    return p


def _pick_semantic_perm_from_summary(summary: pd.DataFrame) -> np.ndarray:
    """Pick new->old state permutation with the fixed five-state semantics.

    Semantic axis is short-liquidation dominance:
        short_dom = S - L
    where S is short liquidation proxy and L is long liquidation proxy.

    Positive short_dom -> forced buying / upward liquidation pressure.
    Negative short_dom -> forced selling / downward liquidation pressure.
    """
    summary = summary.copy()
    summary["abs_short_dom_med"] = summary["short_dom_med"].abs()

    # Neutral anchor is the old state whose median short-liq dominance is closest to zero.
    # This respects the user's interpretation that neutral is relative/balanced, not
    # necessarily exactly zero in a market where one side may have a persistent baseline skew.
    neutral_row = summary.sort_values(
        ["abs_short_dom_med", "total_med", "state"], ascending=[True, True, True]
    ).iloc[0]
    neutral_state = int(neutral_row["state"])

    others = summary[summary["state"] != neutral_state].copy()
    up = others[others["short_dom_med"] > 0].sort_values(
        ["short_dom_med", "total_med", "state"], ascending=[False, False, True]
    )
    down = others[others["short_dom_med"] <= 0].sort_values(
        ["short_dom_med", "total_med", "state"], ascending=[False, False, True]
    )

    # Ideal case: two upward-pressure states and two downward-pressure states.
    if len(up) == 2 and len(down) == 2:
        return np.array([
            int(up.iloc[0]["state"]),       # S1: strongest short-liq/up pressure
            int(up.iloc[1]["state"]),       # S2: mild short-liq/up pressure
            neutral_state,                  # S3: balanced
            int(down.iloc[0]["state"]),     # S4: mild long-liq/down pressure
            int(down.iloc[1]["state"]),     # S5: strongest long-liq/down pressure
        ], dtype=int)

    # Robust fallback for imperfect HMM fits: keep the axis monotone from high
    # short-dom to low short-dom, insert the neutral anchor in the middle.
    # The resulting summary will flag sign_ok=False for semantically weak states.
    ordered = others.sort_values(["short_dom_med", "total_med", "state"], ascending=[False, False, True])
    if len(ordered) != 4:
        raise ValueError(f"Expected 4 non-neutral states for K=5, got {len(ordered)}")
    return np.array([
        int(ordered.iloc[0]["state"]),
        int(ordered.iloc[1]["state"]),
        neutral_state,
        int(ordered.iloc[2]["state"]),
        int(ordered.iloc[3]["state"]),
    ], dtype=int)


def compute_state_order_liq_pressure_centered(
    df: pd.DataFrame,
    feat_index: pd.DatetimeIndex,
    states: np.ndarray,
    col_L: str,
    col_S: str,
    neutral_eps: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return perm(new->old) with fixed liquidation-pressure semantics.

    New one-based states are:
        1. short-liquidation strong dominance -> strong upward pressure
        2. short-liquidation mild dominance   -> mild upward pressure
        3. balanced long/short liquidation    -> no clear liquidation pressure
        4. long-liquidation mild dominance    -> mild downward pressure
        5. long-liquidation strong dominance  -> strong downward pressure

    The ordering anchor uses short_dom = S - L.  Neutral is the old state whose
    median short_dom is closest to zero.  This makes state 3 a *relative*
    neutral/balanced state, which is robust when the long-run BTC liquidation
    baseline is not exactly symmetric.
    """
    tmp = df.loc[feat_index, [col_L, col_S]].copy()
    tmp[col_L] = pd.to_numeric(tmp[col_L], errors="coerce").astype(float)
    tmp[col_S] = pd.to_numeric(tmp[col_S], errors="coerce").astype(float)
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    aligned_states = pd.Series(states, index=feat_index).reindex(tmp.index)
    tmp["state"] = aligned_states.astype(int)

    tmp["total"] = tmp[col_L] + tmp[col_S]
    tmp["short_dom"] = tmp[col_S] - tmp[col_L]
    tmp["long_dom"] = -tmp["short_dom"]
    tmp["rpn_short"] = tmp[col_S] / (tmp["total"] + EPS)

    summary = tmp.groupby("state").agg(
        n=("total", "size"),
        total_med=("total", "median"),
        total_p90=("total", lambda x: float(np.nanquantile(x, 0.90))),
        short_dom_med=("short_dom", "median"),
        short_dom_mean=("short_dom", "mean"),
        long_dom_med=("long_dom", "median"),
        L_med=(col_L, "median"),
        S_med=(col_S, "median"),
        rpn_short_med=("rpn_short", "median"),
    ).reset_index()

    perm = _pick_semantic_perm_from_summary(summary)
    neutral_state = int(perm[2])
    neutral_abs = float(summary.loc[summary["state"] == neutral_state, "short_dom_med"].abs().iloc[0])

    role_rows = []
    semantic_clean = True
    total_n = float(summary["n"].sum()) if len(summary) else 0.0
    for new_state, old_state in enumerate(perm, start=1):
        row = summary.loc[summary["state"] == old_state].iloc[0].to_dict()
        role = STATE_ROLE_MAP[new_state]
        short_dom = float(row["short_dom_med"])
        expected_sign = int(role["pressure_direction"])
        if expected_sign > 0:
            sign_ok = short_dom > 0
        elif expected_sign < 0:
            sign_ok = short_dom < 0
        else:
            sign_ok = abs(short_dom) <= neutral_eps if neutral_eps > 0 else True
        semantic_clean = semantic_clean and bool(sign_ok)
        role_rows.append({
            "new_state": new_state,
            "old_state": int(old_state),
            "state_name": role["name"],
            "state_name_en": role["name_en"],
            "state_name_cn": role["name_cn"],
            "liq_side": role["liq_side"],
            "pressure_direction": role["pressure_direction"],
            "pressure_cn": role["pressure_cn"],
            "severity": role["severity"],
            "sign_ok": bool(sign_ok),
            "neutral_anchor_abs_short_dom": neutral_abs,
            "state_count_share": float(row.get("n", 0.0)) / total_n if total_n > 0 else np.nan,
            **row,
        })

    ordered = pd.DataFrame(role_rows)
    ordered["semantic_clean"] = bool(semantic_clean)
    return perm, ordered


# Backward-compatible alias.  The old name used "bull" but the preferred axis
# is short-liquidation/up-pressure dominance.
def compute_state_order_bull_centered(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, pd.DataFrame]:
    return compute_state_order_liq_pressure_centered(*args, **kwargs)


def add_state_metadata_columns(
    df: pd.DataFrame,
    *,
    state_col: str = "hmm_state",
) -> pd.DataFrame:
    out = df.copy()
    if state_col not in out.columns:
        return out
    s = pd.to_numeric(out[state_col], errors="coerce").astype("Int64")
    out["state_name"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("name") if pd.notna(v) else pd.NA)
    out["state_name_en"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("name_en") if pd.notna(v) else pd.NA)
    out["state_name_cn"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("name_cn") if pd.notna(v) else pd.NA)
    out["state_liq_side"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("liq_side") if pd.notna(v) else pd.NA)
    out["state_liq_side_cn"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("liq_side_cn") if pd.notna(v) else pd.NA)
    out["state_pressure"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("pressure") if pd.notna(v) else pd.NA)
    out["state_pressure_cn"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("pressure_cn") if pd.notna(v) else pd.NA)
    out["state_pressure_direction"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("pressure_direction") if pd.notna(v) else pd.NA)
    out["state_severity"] = s.map(lambda v: STATE_ROLE_MAP.get(int(v), {}).get("severity") if pd.notna(v) else pd.NA)
    return out


def _compute_episode_age(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if not valid.any():
        return out
    sv = s[valid].astype(int)
    episode_id = (sv != sv.shift(1)).cumsum()
    age = sv.groupby(episode_id).cumcount().astype(float)
    out.loc[valid] = age
    return out


def add_state_summary_features(
    df: pd.DataFrame,
    *,
    one_based_states: bool = True,
    state_col: str = "hmm_state",
    conf_col: str = "hmm_state_conf",
    source_time_col: Optional[str] = "liq_feature_time",
    emit_legacy_aliases: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    p_cols = [c for c in _state_cols(one_based_states) if c in out.columns]
    maxp_series: Optional[pd.Series] = None

    if p_cols:
        p = _normalize_posteriors(out, p_cols)
        out[p_cols] = p
        if one_based_states:
            p1, p2, p3, p4, p5 = [p.get(f"p_state_{i}", 0.0) for i in range(1, 6)]
        else:
            p1, p2, p3, p4, p5 = [p.get(f"p_state_{i}", 0.0) for i in range(5)]

        out["p_short_liq"] = p1 + p2
        out["p_up_pressure"] = out["p_short_liq"]
        out["p_neutral"] = p3
        out["p_long_liq"] = p4 + p5
        out["p_down_pressure"] = out["p_long_liq"]
        out["dir_expect"] = out["p_up_pressure"] - out["p_down_pressure"]

        if emit_legacy_aliases:
            out["p_bull"] = out["p_up_pressure"]
            out["p_bear"] = out["p_down_pressure"]

        ent = -(p * np.log(p + EPS)).sum(axis=1)
        out["liq_entropy"] = ent
        out["hmm_entropy"] = ent
        maxp_series = p.max(axis=1)
        out["hmm_maxp"] = maxp_series
    elif state_col in out.columns:
        s_int = pd.to_numeric(out[state_col], errors="coerce").astype("Int64")
        if one_based_states:
            out["p_short_liq"] = np.where(s_int.isin([1, 2]), 1.0, 0.0)
            out["p_neutral"] = np.where(s_int == 3, 1.0, 0.0)
            out["p_long_liq"] = np.where(s_int.isin([4, 5]), 1.0, 0.0)
            out["dir_expect"] = np.where(s_int.isin([1, 2]), 1.0, np.where(s_int.isin([4, 5]), -1.0, 0.0))
        else:
            out["p_short_liq"] = np.where(s_int.isin([0, 1]), 1.0, 0.0)
            out["p_neutral"] = np.where(s_int == 2, 1.0, 0.0)
            out["p_long_liq"] = np.where(s_int.isin([3, 4]), 1.0, 0.0)
            out["dir_expect"] = np.where(s_int.isin([0, 1]), 1.0, np.where(s_int.isin([3, 4]), -1.0, 0.0))
        out["p_up_pressure"] = out["p_short_liq"]
        out["p_down_pressure"] = out["p_long_liq"]
        if emit_legacy_aliases:
            out["p_bull"] = out["p_up_pressure"]
            out["p_bear"] = out["p_down_pressure"]
        out["liq_entropy"] = 0.0
        out["hmm_entropy"] = 0.0
        out["hmm_maxp"] = 1.0
        maxp_series = pd.Series(1.0, index=out.index)

    if conf_col in out.columns:
        out["hmm_conf"] = pd.to_numeric(out[conf_col], errors="coerce")
    elif maxp_series is not None:
        out["hmm_conf"] = maxp_series

    if state_col in out.columns:
        out["age_in_state"] = _compute_episode_age(out[state_col])

        if source_time_col and source_time_col in out.columns:
            tmp = out[[source_time_col, state_col]].copy()
            tmp[source_time_col] = pd.to_datetime(tmp[source_time_col], errors="coerce")
            tmp = tmp.dropna(subset=[source_time_col]).sort_values(source_time_col)
            tmp = tmp.drop_duplicates(subset=[source_time_col], keep="last")
            tmp = tmp.set_index(source_time_col)
            tmp["age_in_state_source"] = _compute_episode_age(tmp[state_col])
            out = out.join(tmp[["age_in_state_source"]], on=source_time_col)

    out = add_state_metadata_columns(out, state_col=state_col)
    return out
