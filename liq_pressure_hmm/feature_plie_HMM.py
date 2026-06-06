
"""feature_plie_HMM.py

HMM regime classification driven by futures liquidation pressure.

What this refactor fixes vs your current script
- Reduces 1-bar state flipping in *online-safe* mode via debounced decoding (no future data).
- Makes labels stable across retrains by enforcing a fixed 5-state semantic contract:
  short-liquidation strong, short-liquidation mild, balanced, long-liquidation mild, long-liquidation strong.
- Moves mixed-frequency alignment into an explicit feature-availability rule.
- Builds HMM observations on the liquidation source clock, then broadcasts outputs to the 10m grid.
- Adds downstream summary features and diagnostic visualizations inside the base package.

Core constraints (respected)
- Feature engineering is past-only (rolling quantile ranks computed on (t-lookback, t) then add current).
- "filtered" inference is past-only (P(s_t | x_1:t)).
- Avoids future leakage by default; any offline-only methods are explicitly labeled.

Dependencies
- hmmlearn (GaussianHMM)
- sortedcontainers, scipy, sklearn, joblib, pandas, numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from collections import deque

import numpy as np
import pandas as pd

from sortedcontainers import SortedList
from scipy.stats import norm
from scipy.special import logsumexp

from sklearn.cluster import KMeans

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover
    GaussianHMM = None  # type: ignore

import joblib

from liq_pressure_hmm.alignment import AlignmentConfig, merge_features_asof
from liq_pressure_hmm.state_summary import add_state_summary_features, compute_state_order_liq_pressure_centered
from liq_pressure_hmm.diagnostics import generate_diagnostic_report, write_report_index_html

EPS = 1e-12


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logger(name: str = "liq_hmm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
def ensure_datetime_index(df: pd.DataFrame, timestamp_col: Optional[str], logger: logging.Logger) -> pd.DataFrame:
    """Ensure sorted DatetimeIndex. Drops duplicate timestamps (keeps last)."""
    out = df.copy()
    if timestamp_col is not None:
        if timestamp_col not in out.columns:
            raise ValueError(f"timestamp_col='{timestamp_col}' not in df.columns")
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=False)
        out = out.sort_values(timestamp_col).set_index(timestamp_col)
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex if timestamp_col is None")
        out = out.sort_index()

    if out.index.has_duplicates:
        ndup = int(out.index.duplicated(keep="last").sum())
        logger.warning(f"Found {ndup} duplicate timestamps; keeping last occurrence.")
        out = out[~out.index.duplicated(keep="last")]
    return out


def merge_to_10min_with_ffill(
    df_10m: pd.DataFrame,
    df_other: pd.DataFrame,
    time_col: str = "time",
    price_col: str = "price",
) -> pd.DataFrame:
    """Backward-compatible wrapper around the explicit availability merge."""
    return merge_features_asof(
        df_10m,
        df_other,
        cfg=AlignmentConfig(time_col=time_col, price_col=price_col),
    )


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    # Time handling
    timestamp_col: Optional[str] = "time"

    # Required liquidation columns
    col_L: str = "fll_cwt_kf"
    col_S: str = "fsl_cwt_kf"

    # Optional precomputed columns (if missing, computed from L/S)
    col_rpn: str = "risk_priority_number"
    col_diff: str = "diff_ls_cwt_kf"

    # Observation clock / mixed-frequency handling
    source_time_col: Optional[str] = "liq_feature_time"
    use_source_clock: bool = True
    output_bar_minutes: int = 10

    # Rolling rank settings (past-only)
    lookback: str = "365D"
    min_obs: int = 5000
    clip: float = 1e-4

    # Which transforms to use
    use_quantile_for_rpn: bool = True
    use_quantile_for_imb: bool = True
    use_quantile_for_total: bool = True

    # Optional EWMA smoothing of raw L/S (past-only)
    ewm_span: Optional[int] = None

    add_is_zero: bool = True
    progress_every: int = 20000


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 5
    covariance_type: str = "diag"  # "diag" is usually more stable than "full" here
    n_iter: int = 300
    tol: float = 1e-3

    # Sticky initialization (diag of transition matrix)
    p_stay: float = 0.993  # expected duration (bars) ≈ 1/(1-p)

    # Regularization: Dirichlet pseudo-counts for transition matrix.
    # Larger diag_conc relative to offdiag_conc => fewer switches.
    transmat_diag_conc: float = 50.0
    transmat_offdiag_conc: float = 1.0
    startprob_conc: float = 1.0

    # Training control
    n_restarts: int = 10
    random_state_base: int = 0
    verbose_em: bool = False
    min_covar: float = 1e-6


@dataclass(frozen=True)
class InferenceConfig:
    # filtered: past-only (online-safe)
    # viterbi/smoothed: full-sequence (offline only)
    mode: str = "filtered"  # filtered | viterbi | smoothed
    allow_future: bool = False  # required for mode in {"viterbi","smoothed"}; never enable for trading

    # How to produce discrete states in online-safe fashion
    # - argmax: raw argmax (can flip)
    # - debounce: require persistence before switching (recommended)
    # - viterbi_online: prefix-Viterbi (past-only, smoother than argmax)
    decision: str = "debounce"  # argmax | debounce | viterbi_online

    confirm_bars: int = 3       # debounce: new state must persist this many bars
    switch_prob: float = 0.55   # debounce: candidate's posterior must exceed this
    min_duration_bars: int = 0  # debounce: enforce minimum bars in current state (0 disables)
    one_based_states: bool = True


@dataclass
class ModelBundle:
    model: Any  # GaussianHMM
    feature_cfg: FeatureConfig
    hmm_cfg: HMMConfig
    feature_columns: Tuple[str, ...]
    state_order_note: str
    train_info: Dict[str, Any]


# ---------------------------------------------------------------------
# Rolling quantile rank (past-only)
# ---------------------------------------------------------------------
def rolling_quantile_rank_timebased(
    series: pd.Series,
    lookback: str,
    min_obs: int,
    logger: Optional[logging.Logger] = None,
    progress_every: int = 0,
    name: str = "",
) -> pd.Series:
    """Past-only rolling quantile rank in [0,1].

    Rank at time t is computed vs window (t-lookback, t), then current value is added.
    Exact method using SortedList: O(N log W).
    """
    lb = pd.Timedelta(lookback)
    idx = series.index
    x = series.to_numpy(dtype=float)

    window: deque[tuple[pd.Timestamp, float]] = deque()
    sl = SortedList()
    out = np.full(len(series), np.nan, dtype=float)

    for i, (ts, v) in enumerate(zip(idx, x)):
        cutoff = ts - lb
        while window and window[0][0] < cutoff:
            _, vold = window.popleft()
            sl.remove(vold)

        if np.isnan(v):
            continue

        if len(sl) >= min_obs:
            left = sl.bisect_left(v)
            right = sl.bisect_right(v)
            out[i] = (left + right) / (2.0 * len(sl))

        window.append((ts, float(v)))
        sl.add(float(v))

        if logger and progress_every and (i > 0) and (i % progress_every == 0):
            logger.info(f"[rolling-rank] {name}: processed {i:,}/{len(x):,} points; window={len(sl):,}")

    return pd.Series(out, index=idx, name=f"q_{series.name or name}")


def gaussianize_unit_interval(u: pd.Series, clip: float) -> pd.Series:
    u2 = u.clip(clip, 1.0 - clip)
    return pd.Series(norm.ppf(u2.to_numpy(dtype=float)), index=u.index)


def _arcsinh_scale(x: pd.Series) -> pd.Series:
    med = float(np.nanmedian(np.abs(x.to_numpy(dtype=float))))
    return pd.Series(np.arcsinh(x.to_numpy(dtype=float) / (med + EPS)), index=x.index)


# ---------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------
def make_observation_view(df: pd.DataFrame, cfg: FeatureConfig, logger: logging.Logger) -> pd.DataFrame:
    """Return the dataframe on the liquidation-source observation clock.

    If ``cfg.use_source_clock`` is enabled and ``cfg.source_time_col`` exists,
    duplicate target bars that share the same source feature timestamp are
    collapsed to one observation. This prevents repeated hourly liquidation
    values from being re-counted six times on a 10m grid.
    """
    if getattr(cfg, "timestamp_col", None) is not None and cfg.timestamp_col in df.columns:
        df_idx = ensure_datetime_index(df, cfg.timestamp_col, logger)
    elif isinstance(df.index, pd.DatetimeIndex):
        df_idx = ensure_datetime_index(df, None, logger)
    else:
        df_idx = ensure_datetime_index(df, cfg.timestamp_col, logger)
    source_time_col = getattr(cfg, "source_time_col", None)
    use_source_clock = bool(getattr(cfg, "use_source_clock", False))
    if not (use_source_clock and source_time_col and source_time_col in df_idx.columns):
        return df_idx

    obs = df_idx.reset_index().rename(columns={df_idx.index.name or "index": "_bar_time"})
    obs[source_time_col] = pd.to_datetime(obs[source_time_col], utc=False, errors="coerce")
    obs = obs.dropna(subset=[source_time_col]).sort_values([source_time_col, "_bar_time"])
    obs = obs.drop_duplicates(subset=[source_time_col], keep="last")
    obs = obs.set_index(source_time_col, drop=False).sort_index()

    if obs.index.has_duplicates:
        obs = obs[~obs.index.duplicated(keep="last")]
    logger.info(
        f"Observation clock: collapsed {len(df_idx):,} target bars to {len(obs):,} unique source updates "
        f"using source_time_col='{source_time_col}'."
    )
    return obs


def _build_hmm_features_on_obs(obs: pd.DataFrame, cfg: FeatureConfig, logger: logging.Logger) -> pd.DataFrame:
    for c in [cfg.col_L, cfg.col_S]:
        if c not in obs.columns:
            raise ValueError(f"Missing required column: {c}")

    L = pd.to_numeric(obs[cfg.col_L], errors="coerce").astype(float)
    S = pd.to_numeric(obs[cfg.col_S], errors="coerce").astype(float)

    if cfg.ewm_span is not None and cfg.ewm_span >= 2:
        L = L.ewm(span=cfg.ewm_span, adjust=False).mean()
        S = S.ewm(span=cfg.ewm_span, adjust=False).mean()

    total = L + S

    if cfg.col_diff in obs.columns:
        imb = pd.to_numeric(obs[cfg.col_diff], errors="coerce").astype(float)
        if cfg.ewm_span is not None and cfg.ewm_span >= 2:
            imb = imb.ewm(span=cfg.ewm_span, adjust=False).mean()
    else:
        imb = L - S

    if cfg.col_rpn in obs.columns:
        rpn = pd.to_numeric(obs[cfg.col_rpn], errors="coerce").astype(float)
        if cfg.ewm_span is not None and cfg.ewm_span >= 2:
            rpn = rpn.ewm(span=cfg.ewm_span, adjust=False).mean()
    else:
        rpn = L / (total + EPS)

    logger.info("Building rolling lookback quantile ranks on the observation clock (past-only)...")

    if cfg.use_quantile_for_total:
        qT = rolling_quantile_rank_timebased(total, cfg.lookback, cfg.min_obs, logger, cfg.progress_every, name="Total")
        zT = gaussianize_unit_interval(qT, cfg.clip)
    else:
        zT = _arcsinh_scale(total)

    if cfg.use_quantile_for_imb:
        qImb = rolling_quantile_rank_timebased(imb, cfg.lookback, cfg.min_obs, logger, cfg.progress_every, name="Imb")
        zImb = gaussianize_unit_interval(qImb, cfg.clip)
    else:
        zImb = _arcsinh_scale(imb)

    if cfg.use_quantile_for_rpn:
        qRPN = rolling_quantile_rank_timebased(rpn, cfg.lookback, cfg.min_obs, logger, cfg.progress_every, name="RPN")
        zRPN = gaussianize_unit_interval(qRPN, cfg.clip)
    else:
        zRPN = gaussianize_unit_interval(rpn, cfg.clip)

    feat = pd.DataFrame({"zRPN": zRPN, "zImb": zImb, "zT": zT}, index=obs.index)
    if cfg.add_is_zero:
        feat["is_zero"] = (total == 0).astype(float)

    before = len(feat)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    after = len(feat)
    logger.info(f"Feature rows available on observation clock: {after:,}/{before:,} (dropped {before-after:,} warmup/NaNs)")
    return feat


def build_hmm_features(df: pd.DataFrame, cfg: FeatureConfig, logger: logging.Logger) -> pd.DataFrame:
    obs = make_observation_view(df, cfg, logger)
    return _build_hmm_features_on_obs(obs, cfg, logger)


# ---------------------------------------------------------------------
# HMM init / training / state permutation
# ---------------------------------------------------------------------
def _apply_transition_priors(hmm: Any, cfg: HMMConfig) -> None:
    """Attach Dirichlet priors for transitions/start probs if supported."""
    K = hmm.n_components
    try:
        hmm.startprob_prior = np.full(K, float(cfg.startprob_conc))
    except Exception:
        pass
    try:
        prior = np.full((K, K), float(cfg.transmat_offdiag_conc))
        np.fill_diagonal(prior, float(cfg.transmat_diag_conc))
        hmm.transmat_prior = prior
    except Exception:
        pass


def _init_sticky_hmm(hmm: Any, X: np.ndarray, K: int, cfg: HMMConfig, seed: int) -> None:
    """Sticky init for start/trans + kmeans init for means."""
    d = X.shape[1]

    p = float(cfg.p_stay)
    A = np.full((K, K), (1.0 - p) / (K - 1))
    np.fill_diagonal(A, p)

    hmm.startprob_ = np.full(K, 1.0 / K)
    hmm.transmat_ = A

    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X)
    hmm.means_ = km.cluster_centers_

    if cfg.covariance_type == "diag":
        v = np.var(X, axis=0) + 1e-3
        hmm.covars_ = np.tile(v, (K, 1))
    else:
        C = np.cov(X.T) + 1e-3 * np.eye(d)
        hmm.covars_ = np.tile(C, (K, 1, 1))

    hmm.init_params = ""   # don't overwrite init
    hmm.params = "stmc"    # update start/trans/means/covars


def _init_from_previous(hmm: Any, prev: Any) -> None:
    """Warm-start from previous fitted model."""
    hmm.startprob_ = prev.startprob_.copy()
    hmm.transmat_ = prev.transmat_.copy()
    hmm.means_ = prev.means_.copy()
    if hasattr(prev, "covars_") and prev.covars_ is not None:
        hmm.covars_ = np.array(prev.covars_, copy=True)
    if hasattr(prev, "_covars_") and prev._covars_ is not None:
        hmm._covars_ = np.array(prev._covars_, copy=True)
    hmm.init_params = ""
    hmm.params = "stmc"


def _permute_states_inplace(hmm: Any, perm: np.ndarray) -> None:
    """Permute states in-place (new_state_k == old_state_perm[k])."""
    perm = np.asarray(perm, dtype=int)

    hmm.startprob_ = hmm.startprob_[perm]
    hmm.transmat_ = hmm.transmat_[perm][:, perm]
    hmm.means_ = hmm.means_[perm]

    if hasattr(hmm, "_covars_") and hmm._covars_ is not None:
        if hmm.covariance_type in ("diag", "full", "spherical"):
            hmm._covars_ = hmm._covars_[perm]
        elif hmm.covariance_type == "tied":
            pass
        else:
            raise ValueError(f"Unsupported covariance_type={hmm.covariance_type}")
    elif hasattr(hmm, "covars_") and hmm.covars_ is not None:
        cov = np.asarray(hmm.covars_)
        if hmm.covariance_type == "diag":
            if cov.ndim == 3:
                cov = np.diagonal(cov, axis1=1, axis2=2)
            hmm.covars_ = cov[perm]
        elif hmm.covariance_type in ("full", "spherical"):
            hmm.covars_ = cov[perm]
        elif hmm.covariance_type == "tied":
            pass
        else:
            raise ValueError(f"Unsupported covariance_type={hmm.covariance_type}")


# ---------------------------------------------------------------------
# Past-only inference helpers
# ---------------------------------------------------------------------
def hmm_filtered_posteriors(hmm: Any, X: np.ndarray, init_alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """Filtered posteriors P(s_t | x_1:t). Past-only and online-safe."""
    logB = hmm._compute_log_likelihood(X)  # (T, K)
    K = hmm.n_components
    T = X.shape[0]
    logA = np.log(hmm.transmat_ + 1e-300)

    log_alpha = np.empty((T, K), dtype=float)
    if init_alpha is None:
        log_alpha[0] = np.log(hmm.startprob_ + 1e-300) + logB[0]
    else:
        init_alpha = np.asarray(init_alpha, dtype=float)
        init_alpha = init_alpha / np.maximum(EPS, init_alpha.sum())
        log_alpha[0] = np.log(init_alpha + 1e-300) + logB[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, T):
        log_alpha[t] = logB[t] + logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def viterbi_online_states(hmm: Any, X: np.ndarray) -> np.ndarray:
    """Prefix-Viterbi (past-only): returns argmax end-state of the best path for each prefix."""
    logB = hmm._compute_log_likelihood(X)  # (T, K)
    K = hmm.n_components
    T = X.shape[0]
    logA = np.log(hmm.transmat_ + 1e-300)

    delta = np.empty((T, K), dtype=float)
    delta[0] = np.log(hmm.startprob_ + 1e-300) + logB[0]

    states = np.empty(T, dtype=int)
    states[0] = int(np.argmax(delta[0]))

    for t in range(1, T):
        prev = delta[t - 1][:, None] + logA
        delta[t] = logB[t] + np.max(prev, axis=0)
        states[t] = int(np.argmax(delta[t]))

    return states


def debounce_states(
    post: np.ndarray,
    confirm_bars: int = 3,
    switch_prob: float = 0.55,
    min_duration_bars: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Debounce a posterior matrix (T,K) into stable states (past-only).

    age convention is 0-based: the first bar in an episode has age 0.
    """
    T, _ = post.shape
    states = np.empty(T, dtype=int)
    conf = np.empty(T, dtype=float)

    cur = int(np.argmax(post[0]))
    states[0] = cur
    conf[0] = float(post[0, cur])

    cand = None
    cand_count = 0
    age = 0
    min_duration_bars = int(max(0, min_duration_bars))

    for t in range(1, T):
        raw = int(np.argmax(post[t]))
        raw_p = float(post[t, raw])

        if raw == cur:
            cand = None
            cand_count = 0
            age += 1
            states[t] = cur
            conf[t] = float(post[t, cur])
            continue

        if raw_p < switch_prob:
            cand = None
            cand_count = 0
            age += 1
            states[t] = cur
            conf[t] = float(post[t, cur])
            continue

        if cand != raw:
            cand = raw
            cand_count = 1
        else:
            cand_count += 1

        bars_in_state = age + 1
        if cand_count >= confirm_bars and (min_duration_bars == 0 or bars_in_state >= min_duration_bars):
            cur = raw
            cand = None
            cand_count = 0
            age = 0
        else:
            age += 1

        states[t] = cur
        conf[t] = float(post[t, cur])

    return states, conf


# ---------------------------------------------------------------------
# Stable relabeling: bull dominance centered semantic ordering
# ---------------------------------------------------------------------
def compute_state_order_by_net_total(
    df: pd.DataFrame,
    feat_index: pd.DatetimeIndex,
    states: np.ndarray,
    col_L: str,
    col_S: str,
    neutral_eps: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    return compute_state_order_liq_pressure_centered(
        df=df,
        feat_index=feat_index,
        states=states,
        col_L=col_L,
        col_S=col_S,
        neutral_eps=neutral_eps,
    )


def order_states_by_net_total_inplace(
    hmm: Any,
    df: pd.DataFrame,
    feat_index: pd.DatetimeIndex,
    states_for_ordering: np.ndarray,
    col_L: str,
    col_S: str,
    neutral_eps: float = 0.0,
) -> Tuple[str, pd.DataFrame]:
    perm, summary = compute_state_order_by_net_total(df, feat_index, states_for_ordering, col_L, col_S, neutral_eps)
    _permute_states_inplace(hmm, perm)
    note = (
        "States reordered into stable liquidation-pressure semantics: "
        "1=空头清算强势占优/强向上压力, 2=空头清算轻度占优/轻向上压力, "
        "3=空头多头清算均衡/无明显压力, 4=多头清算轻度占优/轻向下压力, "
        "5=多头清算强势占优/强向下压力.\n"
        "Ordering anchor uses short-liquidation dominance median = median(S-L): choose the closest-to-zero state as neutral, "
        "then sort remaining states from stronger short-liquidation/up-pressure -> mild short-liquidation/up-pressure -> "
        "mild long-liquidation/down-pressure -> stronger long-liquidation/down-pressure.\n"
        f"perm(new->old)={perm.tolist()}"
    )
    return note, summary


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def time_split_features(feat: pd.DataFrame, val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(feat)
    if n < 1000:
        raise ValueError("Not enough feature rows to split/train robustly.")
    cut = int(np.floor(n * (1.0 - val_fraction)))
    return feat.iloc[:cut], feat.iloc[cut:]


def train_hmm_with_restarts(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    hmm_cfg: HMMConfig,
    logger: logging.Logger,
    warm_start_model: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Any]]:
    if GaussianHMM is None:  # pragma: no cover
        raise ImportError("hmmlearn is required (pip install hmmlearn)")

    best = None
    best_info: Dict[str, Any] = {}

    K = hmm_cfg.n_states
    logger.info(
        f"Training HMM: K={K}, cov={hmm_cfg.covariance_type}, restarts={hmm_cfg.n_restarts}, p_stay={hmm_cfg.p_stay}"
    )

    for r in range(hmm_cfg.n_restarts):
        seed = hmm_cfg.random_state_base + r
        logger.info(f"Restart {r+1}/{hmm_cfg.n_restarts} (seed={seed})")

        hmm = GaussianHMM(
            n_components=K,
            covariance_type=hmm_cfg.covariance_type,
            n_iter=hmm_cfg.n_iter,
            tol=hmm_cfg.tol,
            verbose=hmm_cfg.verbose_em,
            random_state=seed,
            min_covar=hmm_cfg.min_covar,
        )
        _apply_transition_priors(hmm, hmm_cfg)

        if warm_start_model is not None and r == 0:
            _init_from_previous(hmm, warm_start_model)
        else:
            _init_sticky_hmm(hmm, X_train, K, hmm_cfg, seed)

        hmm.fit(X_train)

        mon = getattr(hmm, "monitor_", None)
        if mon is not None:
            iters = getattr(mon, "iter", None)
            converged = getattr(mon, "converged", None)
            hist = getattr(mon, "history", [])
            last = hist[-1] if len(hist) else None
            logger.info(f"  EM iters={iters} converged={converged} last_logprob={last}")

        train_ll = float(hmm.score(X_train))
        val_ll = float(hmm.score(X_val)) if X_val is not None and len(X_val) else np.nan

        diag = np.diag(hmm.transmat_)
        avg_stay = float(np.mean(diag))
        exp_dur_bars = 1.0 / np.maximum(1e-9, (1.0 - diag))

        logger.info(f"  train_ll={train_ll:,.2f}  val_ll={val_ll:,.2f}  avg_stay={avg_stay:.4f}")
        logger.info(
            "  expected duration (bars): "
            f"min={exp_dur_bars.min():.1f}, median={np.median(exp_dur_bars):.1f}, max={exp_dur_bars.max():.1f}"
        )

        score = val_ll if X_val is not None and len(X_val) else train_ll
        if (best is None) or (score > best_info.get("score", -np.inf)):
            best = hmm
            best_info = {
                "seed": seed,
                "train_ll": train_ll,
                "val_ll": val_ll,
                "score": score,
                "avg_stay": avg_stay,
            }

    assert best is not None
    logger.info("Selected best restart:")
    logger.info(f"  seed={best_info['seed']}  train_ll={best_info['train_ll']:,.2f}  val_ll={best_info['val_ll']:,.2f}")
    return best, best_info


def train_liquidation_hmm(
    df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    hmm_cfg: HMMConfig,
    val_fraction: float = 0.2,
    logger: Optional[logging.Logger] = None,
    warm_start_bundle: Optional[ModelBundle] = None,
    relabel_neutral_eps: float = 0.0,
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """End-to-end: observation clock features -> train -> stable semantic relabel."""
    if logger is None:
        logger = setup_logger(level=logging.INFO)

    logger.info("=== HMM Training Pipeline Start ===")
    df_idx = ensure_datetime_index(df, feature_cfg.timestamp_col, logger)
    obs_df = make_observation_view(df_idx, feature_cfg, logger)
    feat = _build_hmm_features_on_obs(obs_df, feature_cfg, logger)
    train_feat, val_feat = time_split_features(feat, val_fraction=val_fraction)

    X_train = train_feat.to_numpy(dtype=float)
    X_val = val_feat.to_numpy(dtype=float) if len(val_feat) else None

    warm_model = warm_start_bundle.model if warm_start_bundle is not None else None
    hmm, best_info = train_hmm_with_restarts(X_train, X_val, hmm_cfg, logger, warm_start_model=warm_model)

    post_train = hmm_filtered_posteriors(hmm, X_train)
    states_train, _ = debounce_states(post_train, confirm_bars=3, switch_prob=0.55)

    note, summary = order_states_by_net_total_inplace(
        hmm,
        obs_df,
        train_feat.index,
        states_train,
        feature_cfg.col_L,
        feature_cfg.col_S,
        neutral_eps=relabel_neutral_eps,
    )

    bundle = ModelBundle(
        model=hmm,
        feature_cfg=feature_cfg,
        hmm_cfg=hmm_cfg,
        feature_columns=tuple(train_feat.columns),
        state_order_note=note,
        train_info={
            **best_info,
            "val_fraction": val_fraction,
            "feature_cfg": asdict(feature_cfg),
            "hmm_cfg": asdict(hmm_cfg),
            "n_feat_rows": int(len(feat)),
            "train_rows": int(len(train_feat)),
            "observation_rows": int(len(obs_df)),
            "state_order_summary": summary.to_dict(orient="list"),
        },
    )

    logger.info("=== HMM Training Pipeline Done ===")
    logger.info(note)
    logger.info(f"State ordering summary (train):\n{summary}")
    return bundle, bundle.train_info


# ---------------------------------------------------------------------
# Apply model to dataframe (offline backtest / analysis)
# ---------------------------------------------------------------------
def apply_model_bundle(
    df: pd.DataFrame,
    bundle: ModelBundle,
    inference: Optional[InferenceConfig] = None,
    add_prob_cols: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if logger is None:
        logger = setup_logger(level=logging.INFO)
    if inference is None:
        inference = InferenceConfig()

    if (inference.mode in ("viterbi", "smoothed")) and (not inference.allow_future):
        raise ValueError(
            "InferenceConfig.mode in {'viterbi','smoothed'} uses future observations (full-sequence) and is NOT tradable. "
            "Use mode='filtered' (or decision='viterbi_online') for production, or set allow_future=True for offline labeling only."
        )

    full_out = ensure_datetime_index(df, bundle.feature_cfg.timestamp_col, logger)
    obs_df = make_observation_view(full_out, bundle.feature_cfg, logger)
    feat = _build_hmm_features_on_obs(obs_df, bundle.feature_cfg, logger)
    X = feat.to_numpy(dtype=float)
    hmm = bundle.model

    if inference.mode == "viterbi":
        states = hmm.predict(X)
        post = None
        conf = None
    elif inference.mode == "smoothed":
        post = hmm.predict_proba(X)
        if inference.decision == "debounce":
            states, conf = debounce_states(post, inference.confirm_bars, inference.switch_prob, inference.min_duration_bars)
        else:
            states = np.argmax(post, axis=1)
            conf = np.max(post, axis=1)
    elif inference.mode == "filtered":
        post = hmm_filtered_posteriors(hmm, X)
        if inference.decision == "viterbi_online":
            states = viterbi_online_states(hmm, X)
            conf = post[np.arange(len(states)), states]
        elif inference.decision == "debounce":
            states, conf = debounce_states(post, inference.confirm_bars, inference.switch_prob, inference.min_duration_bars)
        else:
            states = np.argmax(post, axis=1)
            conf = np.max(post, axis=1)
    else:
        raise ValueError("InferenceConfig.mode must be one of: filtered, viterbi, smoothed")

    states_out = states + 1 if inference.one_based_states else states

    obs_signals = pd.DataFrame(index=feat.index)
    obs_signals["hmm_state"] = states_out
    obs_signals["hmm_state_conf"] = conf if conf is not None else np.nan
    if add_prob_cols and post is not None:
        for k in range(hmm.n_components):
            col = f"p_state_{k+1}" if inference.one_based_states else f"p_state_{k}"
            obs_signals[col] = post[:, k]

    source_time_col = getattr(bundle.feature_cfg, "source_time_col", None)
    use_source_clock = bool(getattr(bundle.feature_cfg, "use_source_clock", False))

    if use_source_clock and source_time_col and source_time_col in full_out.columns:
        full_out[source_time_col] = pd.to_datetime(full_out[source_time_col], errors="coerce")
        obs_signals[source_time_col] = obs_signals.index
        full_out = full_out.join(obs_signals.drop(columns=[source_time_col]), on=source_time_col)
    else:
        for col in obs_signals.columns:
            full_out[col] = np.nan
            full_out.loc[feat.index, col] = obs_signals[col]

    full_out = add_state_summary_features(
        full_out,
        one_based_states=inference.one_based_states,
        state_col="hmm_state",
        conf_col="hmm_state_conf",
        source_time_col=source_time_col,
    )
    return full_out


# ---------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------
def save_bundle(bundle: ModelBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> ModelBundle:
    try:
        return joblib.load(path)
    except AttributeError:
        # Backward compatibility: bundles pickled when this file was run as a script
        # are stored under module "__main__" and won't load when imported.
        import __main__ as _main

        for _name, _obj in {
            "FeatureConfig": FeatureConfig,
            "HMMConfig": HMMConfig,
            "InferenceConfig": InferenceConfig,
            "ModelBundle": ModelBundle,
        }.items():
            if not hasattr(_main, _name):
                setattr(_main, _name, _obj)
        return joblib.load(path)


# ---------------------------------------------------------------------
# Stable historical output overlay
# ---------------------------------------------------------------------
def _read_csv_reference(path_spec: str, *, root: Path) -> pd.DataFrame:
    """Read a CSV reference from a path or a zip member written as zip_path::member."""
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


def _index_state_frame(df: pd.DataFrame, *, time_col: str = "time") -> pd.DataFrame:
    out = df.copy()
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
        out = out.dropna(subset=[time_col]).set_index(time_col)
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~pd.isna(out.index)]
    else:
        raise ValueError(f"State dataframe must contain '{time_col}' or use a DatetimeIndex.")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = time_col
    return out


def overlay_stable_history(
    computed: pd.DataFrame,
    *,
    references: List[Tuple[str, pd.DataFrame]],
    time_col: str = "time",
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Preserve accepted historical rows while keeping computed rows for new timestamps.

    References are applied in order; later references take precedence on overlap.
    This is intentionally an output-level stability guard. It prevents a retrained
    or accidentally changed HMM bundle from rewriting rows that have already been
    accepted by downstream PLIE/MAR features.
    """
    out = _index_state_frame(computed, time_col=time_col)
    stats: List[Dict[str, Any]] = []

    for name, ref in references:
        ref_idx = _index_state_frame(ref, time_col=time_col)
        common_idx = ref_idx.index.intersection(out.index)
        if len(common_idx) == 0:
            stats.append({
                "name": name,
                "reference_rows": int(len(ref_idx)),
                "preserved_rows": 0,
                "reference_min_time": None,
                "reference_max_time": None,
            })
            continue

        common_cols = [c for c in out.columns if c in ref_idx.columns]
        out.loc[common_idx, common_cols] = ref_idx.loc[common_idx, common_cols]
        stat = {
            "name": name,
            "reference_rows": int(len(ref_idx)),
            "preserved_rows": int(len(common_idx)),
            "reference_min_time": str(ref_idx.index.min()),
            "reference_max_time": str(ref_idx.index.max()),
            "preserved_min_time": str(common_idx.min()),
            "preserved_max_time": str(common_idx.max()),
        }
        stats.append(stat)
        if logger is not None:
            logger.info(
                "Stable history overlay: %s preserved %s rows (%s -> %s).",
                name,
                stat["preserved_rows"],
                stat["preserved_min_time"],
                stat["preserved_max_time"],
            )

    return out, stats


# ---------------------------------------------------------------------
# Online helpers (streaming features + filtering + debounced decoding)
# ---------------------------------------------------------------------
class RollingQuantileRanker:
    """Incremental past-only rolling quantile rank for streaming."""

    def __init__(self, lookback: str, min_obs: int):
        self.lb = pd.Timedelta(lookback)
        self.min_obs = int(min_obs)
        self.window: deque[tuple[pd.Timestamp, float]] = deque()
        self.sl = SortedList()

    def update(self, ts: pd.Timestamp, v: float) -> float:
        cutoff = ts - self.lb
        while self.window and self.window[0][0] < cutoff:
            _, vold = self.window.popleft()
            self.sl.remove(vold)

        if np.isnan(v):
            return np.nan

        rank = np.nan
        if len(self.sl) >= self.min_obs:
            left = self.sl.bisect_left(v)
            right = self.sl.bisect_right(v)
            rank = (left + right) / (2.0 * len(self.sl))

        self.window.append((ts, float(v)))
        self.sl.add(float(v))
        return float(rank)


class OnlineFeatureBuilder:
    """Streaming feature builder matching build_hmm_features() (for quantile-based features)."""

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.rank_T = RollingQuantileRanker(cfg.lookback, cfg.min_obs) if cfg.use_quantile_for_total else None
        self.rank_Imb = RollingQuantileRanker(cfg.lookback, cfg.min_obs) if cfg.use_quantile_for_imb else None
        self.rank_RPN = RollingQuantileRanker(cfg.lookback, cfg.min_obs) if cfg.use_quantile_for_rpn else None

    def update(self, ts: pd.Timestamp, L: float, S: float, rpn: Optional[float] = None, imb: Optional[float] = None) -> Optional[np.ndarray]:
        cfg = self.cfg
        total = float(L + S)
        if imb is None:
            imb = float(L - S)
        if rpn is None:
            rpn = float(L / (total + EPS))

        # Total
        if cfg.use_quantile_for_total:
            uT = self.rank_T.update(ts, total)  # type: ignore[union-attr]
            if np.isnan(uT):
                return None
            zT = float(norm.ppf(np.clip(uT, cfg.clip, 1.0 - cfg.clip)))
        else:
            zT = float(np.arcsinh(total / (abs(total) + 1.0)))

        # Imb
        if cfg.use_quantile_for_imb:
            uI = self.rank_Imb.update(ts, imb)  # type: ignore[union-attr]
            if np.isnan(uI):
                return None
            zImb = float(norm.ppf(np.clip(uI, cfg.clip, 1.0 - cfg.clip)))
        else:
            zImb = float(np.arcsinh(imb / (abs(imb) + 1.0)))

        # RPN
        if cfg.use_quantile_for_rpn:
            uR = self.rank_RPN.update(ts, rpn)  # type: ignore[union-attr]
            if np.isnan(uR):
                return None
            zRPN = float(norm.ppf(np.clip(uR, cfg.clip, 1.0 - cfg.clip)))
        else:
            zRPN = float(norm.ppf(np.clip(rpn, cfg.clip, 1.0 - cfg.clip)))

        is_zero = 1.0 if (cfg.add_is_zero and total == 0.0) else 0.0

        feats = [zRPN, zImb, zT]
        if cfg.add_is_zero:
            feats.append(is_zero)

        return np.asarray(feats, dtype=float)


class OnlineHMMFilter:
    """Incremental filtered posterior update: alpha_t = P(s_t | x_1:t)."""

    def __init__(self, hmm: Any):
        self.hmm = hmm
        self.logA = np.log(hmm.transmat_ + 1e-300)
        self.log_alpha: Optional[np.ndarray] = None

    def step(self, x_t: np.ndarray) -> np.ndarray:
        x_t = np.asarray(x_t, dtype=float).reshape(1, -1)
        logB_t = self.hmm._compute_log_likelihood(x_t)[0]

        if self.log_alpha is None:
            self.log_alpha = np.log(self.hmm.startprob_ + 1e-300) + logB_t
        else:
            self.log_alpha = logB_t + logsumexp(self.log_alpha[:, None] + self.logA, axis=0)

        self.log_alpha -= logsumexp(self.log_alpha)
        return np.exp(self.log_alpha)


class DebouncedStateDecoder:
    """Online debounced decoder (no future, 0-based age convention)."""

    def __init__(self, confirm_bars: int = 3, switch_prob: float = 0.55, min_duration_bars: int = 0):
        self.confirm_bars = int(confirm_bars)
        self.switch_prob = float(switch_prob)
        self.min_duration_bars = int(max(0, min_duration_bars))
        self.cur_state: Optional[int] = None
        self.cand_state: Optional[int] = None
        self.cand_count: int = 0
        self.age_in_state: int = 0

    def step(self, post: np.ndarray) -> Tuple[int, float]:
        post = np.asarray(post, dtype=float)
        raw = int(np.argmax(post))
        raw_p = float(post[raw])

        if self.cur_state is None:
            self.cur_state = raw
            self.age_in_state = 0
            return self.cur_state, raw_p

        if raw == self.cur_state:
            self.cand_state = None
            self.cand_count = 0
            self.age_in_state += 1
            return self.cur_state, float(post[self.cur_state])

        if raw_p < self.switch_prob:
            self.cand_state = None
            self.cand_count = 0
            self.age_in_state += 1
            return self.cur_state, float(post[self.cur_state])

        if self.cand_state != raw:
            self.cand_state = raw
            self.cand_count = 1
        else:
            self.cand_count += 1

        bars_in_state = self.age_in_state + 1
        if self.cand_count >= self.confirm_bars and (self.min_duration_bars == 0 or bars_in_state >= self.min_duration_bars):
            self.cur_state = raw
            self.cand_state = None
            self.cand_count = 0
            self.age_in_state = 0
        else:
            self.age_in_state += 1

        return self.cur_state, float(post[self.cur_state])


class OnlineRegimeEngine:
    """Streaming engine: features -> filtered posteriors -> stable discrete state.

    For retraining:
      - retrain on a schedule (e.g., daily) using warm_start_bundle to stabilize parameters.
      - then swap in the new bundle (and reset filter state if desired).
    """

    def __init__(self, bundle: ModelBundle, inference: Optional[InferenceConfig] = None):
        self.bundle = bundle
        self.inference = inference or InferenceConfig(mode="filtered", decision="debounce")
        self.fe = OnlineFeatureBuilder(bundle.feature_cfg)
        self.hmm_filter = OnlineHMMFilter(bundle.model)
        self.decoder = DebouncedStateDecoder(
            self.inference.confirm_bars,
            self.inference.switch_prob,
            self.inference.min_duration_bars,
        )

    def step(
        self,
        ts: pd.Timestamp,
        L: float,
        S: float,
        rpn: Optional[float] = None,
        imb: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        x = self.fe.update(ts, L, S, rpn=rpn, imb=imb)
        if x is None:
            return None  # warmup not complete

        post = self.hmm_filter.step(x)
        st0, conf = self.decoder.step(post)

        st = st0 + 1 if self.inference.one_based_states else st0
        result = {"time": ts, "hmm_state": st, "hmm_state_conf": conf, "post": post, "age_in_state_source": self.decoder.age_in_state}
        if self.inference.one_based_states:
            for k in range(len(post)):
                result[f"p_state_{k+1}"] = float(post[k])
        else:
            for k in range(len(post)):
                result[f"p_state_{k}"] = float(post[k])
        return result

# ---------------------------------------------------------------------
# Continuous training (warm-start periodic retrain + incremental features)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ContinuousTrainingConfig:
    """
    Controls when/how we retrain while new data arrives.

    NOTE:
    - hmmlearn does not provide partial_fit; "continuous training" here means
      periodic refits on a rolling window, warm-started from the last model.
    """
    # Trigger
    retrain_every: Optional[str] = "7D"        # wall-clock cadence; set None to disable
    min_new_bars: int = 144                    # require this many *new feature rows* since last retrain

    # Training window (for speed + non-stationarity)
    train_window: Optional[str] = "730D"       # last N days of FEATURES for refit; None => use all stored
    max_train_rows: Optional[int] = None       # hard cap rows (applied after train_window)

    # Keep in-memory history (raw is used for state ordering; features for training)
    keep_raw_window: Optional[str] = None      # None => max(feature_cfg.lookback, train_window)
    keep_feat_window: Optional[str] = None     # None => train_window

    # Make periodic refits cheaper than the initial search
    retrain_n_iter: Optional[int] = 120
    retrain_n_restarts: Optional[int] = 2      # warm-start restart + one random restart

    # State relabeling / semantic diagnostics
    relabel_neutral_eps: float = 0.0

    # Online inference behavior when swapping models
    reset_filter_on_swap: bool = True

    # Prime online filter/decoder with recent history on init (to avoid "cold start" states)
    prime_history_bars: int = 256


def ensure_datetime_index_flexible(
    df: pd.DataFrame,
    preferred_timestamp_col: Optional[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Like ensure_datetime_index(), but also accepts:
      - df already indexed by DatetimeIndex even if preferred_timestamp_col is set.
    """
    if preferred_timestamp_col is not None and preferred_timestamp_col in df.columns:
        return ensure_datetime_index(df, preferred_timestamp_col, logger)
    if isinstance(df.index, pd.DatetimeIndex):
        return ensure_datetime_index(df, None, logger)
    return ensure_datetime_index(df, preferred_timestamp_col, logger)


def _select_time_window(df: pd.DataFrame, window: Optional[str]) -> pd.DataFrame:
    if window is None or len(df) == 0:
        return df
    lb = pd.Timedelta(window)
    cutoff = df.index[-1] - lb
    return df.loc[df.index >= cutoff]


def _cap_rows(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.iloc[-int(max_rows):]


def _compute_components_like_offline(df_idx: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Reproduce the *raw* (pre-quantile) series used by build_hmm_features(), for seeding."""
    L = df_idx[cfg.col_L].astype(float)
    S = df_idx[cfg.col_S].astype(float)

    if cfg.ewm_span is not None and cfg.ewm_span >= 2:
        L = L.ewm(span=cfg.ewm_span, adjust=False).mean()
        S = S.ewm(span=cfg.ewm_span, adjust=False).mean()

    total = L + S

    if cfg.col_diff in df_idx.columns:
        imb = df_idx[cfg.col_diff].astype(float)
        if cfg.ewm_span is not None and cfg.ewm_span >= 2:
            imb = imb.ewm(span=cfg.ewm_span, adjust=False).mean()
    else:
        imb = L - S

    if cfg.col_rpn in df_idx.columns:
        rpn = df_idx[cfg.col_rpn].astype(float)
        if cfg.ewm_span is not None and cfg.ewm_span >= 2:
            rpn = rpn.ewm(span=cfg.ewm_span, adjust=False).mean()
    else:
        rpn = L / (total + EPS)

    return total, imb, rpn


def seed_online_feature_builder(
    builder: OnlineFeatureBuilder,
    df_history: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.Timestamp:
    """
    Seed OnlineFeatureBuilder's rolling rankers so streaming inference/training can start immediately.

    We only replay the last lookback window (not the whole history) for efficiency.
    """
    if logger is None:
        logger = setup_logger(level=logging.INFO)

    cfg = builder.cfg
    df_idx = ensure_datetime_index_flexible(df_history, cfg.timestamp_col, logger)
    df_idx = make_observation_view(df_idx, cfg, logger)
    if len(df_idx) == 0:
        raise ValueError("df_history is empty")

    end_ts = df_idx.index[-1]
    cutoff = end_ts - pd.Timedelta(cfg.lookback)

    total, imb, rpn = _compute_components_like_offline(df_idx, cfg)
    mask = df_idx.index >= cutoff

    def _replay(rank_obj: Optional[RollingQuantileRanker], s: pd.Series, name: str) -> None:
        if rank_obj is None:
            return
        idx = s.index[mask]
        vals = s.to_numpy(dtype=float)[mask]
        n = 0
        for ts, v in zip(idx, vals):
            if np.isnan(v):
                continue
            rank_obj.update(ts, float(v))
            n += 1
        logger.info(f"[seed] {name}: loaded {n:,} points into rolling ranker (lookback={cfg.lookback})")

    _replay(builder.rank_T, total, "Total")
    _replay(builder.rank_Imb, imb, "Imb")
    _replay(builder.rank_RPN, rpn, "RPN")
    return end_ts


def build_features_incremental(
    df_new: pd.DataFrame,
    feature_cfg: FeatureConfig,
    builder: OnlineFeatureBuilder,
    feature_columns: Tuple[str, ...],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Compute features for *new rows only* using OnlineFeatureBuilder (past-only),
    returning a DataFrame indexed by time with columns == feature_columns.
    """
    if logger is None:
        logger = setup_logger(level=logging.INFO)

    df_idx = ensure_datetime_index_flexible(df_new, feature_cfg.timestamp_col, logger)
    df_idx = make_observation_view(df_idx, feature_cfg, logger)
    if len(df_idx) == 0:
        return pd.DataFrame(columns=list(feature_columns), index=pd.DatetimeIndex([]))

    has_rpn = feature_cfg.col_rpn in df_idx.columns
    has_diff = feature_cfg.col_diff in df_idx.columns

    rows: List[np.ndarray] = []
    idxs: List[pd.Timestamp] = []

    colL = feature_cfg.col_L
    colS = feature_cfg.col_S
    colR = feature_cfg.col_rpn
    colD = feature_cfg.col_diff
    upd = builder.update

    for ts, row in df_idx.iterrows():
        L = float(row[colL])
        S = float(row[colS])
        rpn = float(row[colR]) if has_rpn and pd.notna(row[colR]) else None
        imb = float(row[colD]) if has_diff and pd.notna(row[colD]) else None
        x = upd(ts, L, S, rpn=rpn, imb=imb)
        if x is None:
            continue
        idxs.append(ts)
        rows.append(x)

    if not rows:
        return pd.DataFrame(columns=list(feature_columns), index=pd.DatetimeIndex([]))

    X = np.vstack(rows)
    feat = pd.DataFrame(X, index=pd.DatetimeIndex(idxs), columns=list(feature_columns))
    return feat[~feat.index.duplicated(keep="last")].sort_index()


def train_liquidation_hmm_from_features(
    raw_df_idx: pd.DataFrame,
    feat: pd.DataFrame,
    feature_cfg: FeatureConfig,
    hmm_cfg: HMMConfig,
    val_fraction: float,
    logger: Optional[logging.Logger] = None,
    warm_start_bundle: Optional[ModelBundle] = None,
    relabel_neutral_eps: float = 0.0,
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """
    Same as train_liquidation_hmm(), but assumes features are already computed & clean.
    Key speed-up: we *don't* recompute rolling ranks.
    """
    if logger is None:
        logger = setup_logger(level=logging.INFO)

    feat = feat.sort_index()
    train_feat, val_feat = time_split_features(feat, val_fraction=val_fraction)

    X_train = train_feat.to_numpy(dtype=float)
    X_val = val_feat.to_numpy(dtype=float) if len(val_feat) else None

    warm_model = warm_start_bundle.model if warm_start_bundle is not None else None
    hmm, best_info = train_hmm_with_restarts(X_train, X_val, hmm_cfg, logger, warm_start_model=warm_model)

    post_train = hmm_filtered_posteriors(hmm, X_train)
    states_train, _ = debounce_states(post_train, confirm_bars=3, switch_prob=0.55)

    raw_obs = make_observation_view(raw_df_idx, feature_cfg, logger)

    note, summary = order_states_by_net_total_inplace(
        hmm,
        raw_obs,
        train_feat.index,
        states_train,
        feature_cfg.col_L,
        feature_cfg.col_S,
        neutral_eps=relabel_neutral_eps,
    )

    bundle = ModelBundle(
        model=hmm,
        feature_cfg=feature_cfg,
        hmm_cfg=hmm_cfg,
        feature_columns=tuple(train_feat.columns),
        state_order_note=note,
        train_info={
            **best_info,
            "val_fraction": val_fraction,
            "feature_cfg": asdict(feature_cfg),
            "hmm_cfg": asdict(hmm_cfg),
            "n_feat_rows": int(len(feat)),
            "train_rows": int(len(train_feat)),
            "state_order_summary": summary.to_dict(orient="list"),
        },
    )

    logger.info(note)
    logger.info(f"State ordering summary (train):\n{summary}")
    return bundle, bundle.train_info


class OnlineViterbiDecoder:
    """Incremental prefix-Viterbi end-state (past-only)."""
    def __init__(self, hmm: Any):
        self.hmm = hmm
        self.logA = np.log(hmm.transmat_ + 1e-300)
        self.delta: Optional[np.ndarray] = None

    def step(self, x_t: np.ndarray) -> int:
        x_t = np.asarray(x_t, dtype=float).reshape(1, -1)
        logB_t = self.hmm._compute_log_likelihood(x_t)[0]
        if self.delta is None:
            self.delta = np.log(self.hmm.startprob_ + 1e-300) + logB_t
        else:
            self.delta = logB_t + np.max(self.delta[:, None] + self.logA, axis=0)
        return int(np.argmax(self.delta))


class ContinuousRegimeTrainer:
    """
    Stream -> online inference, and periodically warm-start refit on a rolling window.

    This is additive: your existing train/apply/online code continues to work unchanged.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        history_df: pd.DataFrame,
        ct_cfg: ContinuousTrainingConfig = ContinuousTrainingConfig(),
        inference: Optional[InferenceConfig] = None,
        logger: Optional[logging.Logger] = None,
        feature_store: Optional[pd.DataFrame] = None,
        val_fraction: float = 0.2,
    ):
        self.logger = logger or setup_logger(level=logging.INFO)
        self.ct_cfg = ct_cfg
        self.val_fraction = float(val_fraction)
        self.inference = inference or InferenceConfig(mode="filtered", decision="debounce")

        self._base_feature_cfg = bundle.feature_cfg
        self._base_hmm_cfg = bundle.hmm_cfg
        self.bundle = bundle

        # stores
        raw_idx = ensure_datetime_index_flexible(history_df, self._base_feature_cfg.timestamp_col, self.logger)
        keep_cols = [c for c in [self._base_feature_cfg.col_L, self._base_feature_cfg.col_S,
                                self._base_feature_cfg.col_rpn, self._base_feature_cfg.col_diff] if c in raw_idx.columns]
        self.raw_store = raw_idx[keep_cols].copy()
        self.feature_store = build_hmm_features(history_df, self._base_feature_cfg, self.logger) if feature_store is None \
            else feature_store.sort_index().copy()

        # online components (seed features to avoid warmup gap)
        self.fe_builder = OnlineFeatureBuilder(self._base_feature_cfg)
        seed_online_feature_builder(self.fe_builder, history_df, self.logger)

        self.hmm_filter = OnlineHMMFilter(self.bundle.model)
        self.decoder = DebouncedStateDecoder(
            self.inference.confirm_bars,
            self.inference.switch_prob,
            self.inference.min_duration_bars,
        )
        self.viterbi = OnlineViterbiDecoder(self.bundle.model)

        self.last_seen_ts = self.raw_store.index[-1] if len(self.raw_store) else None
        self.last_retrain_ts = self.last_seen_ts
        self.new_feat_rows_since_retrain = 0

        self._trim_stores()
        self._prime_online_state()

    def ingest_bar(self, ts: pd.Timestamp, L: float, S: float, rpn: Optional[float] = None, imb: Optional[float] = None) -> Optional[Dict[str, Any]]:
        ts = pd.to_datetime(ts)
        if self.last_seen_ts is not None and ts <= self.last_seen_ts:
            return None  # assume append-only stream for efficiency

        # raw store
        row = {self._base_feature_cfg.col_L: float(L), self._base_feature_cfg.col_S: float(S)}
        if self._base_feature_cfg.col_rpn in self.raw_store.columns:
            row[self._base_feature_cfg.col_rpn] = np.nan if rpn is None else float(rpn)
        if self._base_feature_cfg.col_diff in self.raw_store.columns:
            row[self._base_feature_cfg.col_diff] = np.nan if imb is None else float(imb)
        self.raw_store.loc[ts, list(row.keys())] = list(row.values())
        self.raw_store = self.raw_store.sort_index()
        self.last_seen_ts = ts

        # feature update
        x = self.fe_builder.update(ts, float(L), float(S), rpn=rpn, imb=imb)
        if x is None:
            self._trim_stores()
            return None

        self.feature_store.loc[ts, list(self.bundle.feature_columns)] = x
        self.feature_store = self.feature_store.sort_index()
        self.new_feat_rows_since_retrain += 1

        # inference
        post = self.hmm_filter.step(x)
        if self.inference.decision == "viterbi_online":
            st0 = self.viterbi.step(x)
            conf = float(post[st0])
        elif self.inference.decision == "debounce":
            st0, conf = self.decoder.step(post)
        else:
            st0 = int(np.argmax(post))
            conf = float(post[st0])
        st = st0 + 1 if self.inference.one_based_states else st0

        self.maybe_retrain(now_ts=ts)
        self._trim_stores()
        return {"time": ts, "hmm_state": st, "hmm_state_conf": conf, "post": post}

    def ingest_frame(self, df_new: pd.DataFrame) -> pd.DataFrame:
        df_idx = ensure_datetime_index_flexible(df_new, self._base_feature_cfg.timestamp_col, self.logger)
        if self.last_seen_ts is not None:
            df_idx = df_idx.loc[df_idx.index > self.last_seen_ts]
        if len(df_idx) == 0:
            return pd.DataFrame(columns=["hmm_state", "hmm_state_conf"], index=pd.DatetimeIndex([]))

        # raw store (reindex allows missing optional cols)
        keep_cols = self.raw_store.columns.tolist()
        self.raw_store = pd.concat([self.raw_store, df_idx.reindex(columns=keep_cols)], axis=0)
        self.raw_store = self.raw_store[~self.raw_store.index.duplicated(keep="last")].sort_index()
        self.last_seen_ts = self.raw_store.index[-1]

        # incremental features
        feat_new = build_features_incremental(df_idx, self._base_feature_cfg, self.fe_builder, self.bundle.feature_columns, logger=self.logger)
        if len(feat_new):
            self.feature_store = pd.concat([self.feature_store, feat_new], axis=0)
            self.feature_store = self.feature_store[~self.feature_store.index.duplicated(keep="last")].sort_index()
            self.new_feat_rows_since_retrain += int(len(feat_new))

        # inference for new rows
        out = []
        for ts, x in feat_new.iterrows():
            x_arr = x.to_numpy(dtype=float)
            post = self.hmm_filter.step(x_arr)
            if self.inference.decision == "viterbi_online":
                st0 = self.viterbi.step(x_arr)
                conf = float(post[st0])
            elif self.inference.decision == "debounce":
                st0, conf = self.decoder.step(post)
            else:
                st0 = int(np.argmax(post))
                conf = float(post[st0])
            st = st0 + 1 if self.inference.one_based_states else st0
            out.append((ts, st, conf))

        res = pd.DataFrame(out, columns=["time", "hmm_state", "hmm_state_conf"]).set_index("time")
        self.maybe_retrain(now_ts=self.last_seen_ts)
        self._trim_stores()
        return res

    def maybe_retrain(self, now_ts: Optional[pd.Timestamp] = None) -> bool:
        now_ts = now_ts or self.last_seen_ts
        if now_ts is None or self.ct_cfg.retrain_every is None:
            return False
        if self.last_retrain_ts is None:
            self.last_retrain_ts = now_ts
            self.new_feat_rows_since_retrain = 0
            return False
        if self.new_feat_rows_since_retrain < self.ct_cfg.min_new_bars:
            return False
        if (now_ts - self.last_retrain_ts) < pd.Timedelta(self.ct_cfg.retrain_every):
            return False
        return self.retrain()

    def retrain(self) -> bool:
        if len(self.feature_store) < 1200:
            self.logger.warning("Skip retrain: not enough feature rows yet.")
            return False

        feat_win = _cap_rows(_select_time_window(self.feature_store, self.ct_cfg.train_window), self.ct_cfg.max_train_rows)
        if len(feat_win) < 1200:
            self.logger.warning("Skip retrain: training window too small after trimming.")
            return False

        missing = feat_win.index.difference(self.raw_store.index)
        if len(missing):
            self.logger.warning(f"Skip retrain: raw_store missing {len(missing):,} timestamps (increase keep_raw_window).")
            return False
        raw_win = self.raw_store.loc[feat_win.index]

        import dataclasses as _dc
        hmm_cfg_used = self._base_hmm_cfg
        if self.ct_cfg.retrain_n_iter is not None:
            hmm_cfg_used = _dc.replace(hmm_cfg_used, n_iter=int(self.ct_cfg.retrain_n_iter))
        if self.ct_cfg.retrain_n_restarts is not None:
            hmm_cfg_used = _dc.replace(hmm_cfg_used, n_restarts=int(self.ct_cfg.retrain_n_restarts))

        self.logger.info(f"=== Continuous retrain: feat_rows={len(feat_win):,}, n_iter={hmm_cfg_used.n_iter}, restarts={hmm_cfg_used.n_restarts} ===")
        try:
            new_bundle, _ = train_liquidation_hmm_from_features(
                raw_df_idx=raw_win,
                feat=feat_win,
                feature_cfg=self._base_feature_cfg,
                hmm_cfg=hmm_cfg_used,
                val_fraction=self.val_fraction,
                logger=self.logger,
                warm_start_bundle=self.bundle,
                relabel_neutral_eps=self.ct_cfg.relabel_neutral_eps,
            )
        except Exception as e:
            self.logger.exception(f"Continuous retrain failed; keeping existing model. Error: {e}")
            return False

        self.bundle = new_bundle

        if self.ct_cfg.reset_filter_on_swap:
            self._prime_online_state()
        else:
            self.hmm_filter = OnlineHMMFilter(self.bundle.model)
            self.viterbi = OnlineViterbiDecoder(self.bundle.model)

        self.last_retrain_ts = self.last_seen_ts
        self.new_feat_rows_since_retrain = 0
        self.logger.info("=== Continuous retrain done; model swapped ===")
        return True

    def _prime_online_state(self) -> None:
        n = int(max(0, self.ct_cfg.prime_history_bars))
        self.hmm_filter = OnlineHMMFilter(self.bundle.model)
        self.decoder = DebouncedStateDecoder(
            self.inference.confirm_bars,
            self.inference.switch_prob,
            self.inference.min_duration_bars,
        )
        self.viterbi = OnlineViterbiDecoder(self.bundle.model)

        if n <= 0 or len(self.feature_store) == 0:
            return
        tail = self.feature_store.iloc[-min(n, len(self.feature_store)):]
        for _ts, x in tail.iterrows():
            x_arr = x.to_numpy(dtype=float)
            post = self.hmm_filter.step(x_arr)
            if self.inference.decision == "viterbi_online":
                _ = self.viterbi.step(x_arr)
            elif self.inference.decision == "debounce":
                _ = self.decoder.step(post)

    def _trim_stores(self) -> None:
        keep_raw = self.ct_cfg.keep_raw_window
        if keep_raw is None:
            candidates = [self._base_feature_cfg.lookback]
            if self.ct_cfg.train_window is not None:
                candidates.append(self.ct_cfg.train_window)
            keep_raw = max((pd.Timedelta(x) for x in candidates)).isoformat()
        keep_feat = self.ct_cfg.keep_feat_window if self.ct_cfg.keep_feat_window is not None else self.ct_cfg.train_window

        if len(self.raw_store):
            self.raw_store = _select_time_window(self.raw_store, keep_raw)
        if len(self.feature_store) and keep_feat is not None:
            self.feature_store = _select_time_window(self.feature_store, keep_feat)

        self.raw_store = self.raw_store[~self.raw_store.index.duplicated(keep="last")].sort_index()
        self.feature_store = self.feature_store[~self.feature_store.index.duplicated(keep="last")].sort_index()


@dataclass
class ContinuousTrainerState:
    """Serializable snapshot so you can stop/restart without re-computing features."""
    bundle: ModelBundle
    raw_store: pd.DataFrame
    feature_store: pd.DataFrame
    last_seen_ts: Optional[pd.Timestamp]
    last_retrain_ts: Optional[pd.Timestamp]
    new_feat_rows_since_retrain: int
    ct_cfg: ContinuousTrainingConfig
    val_fraction: float
    inference: InferenceConfig


def save_continuous_state(trainer: ContinuousRegimeTrainer, path: str) -> None:
    state = ContinuousTrainerState(
        bundle=trainer.bundle,
        raw_store=trainer.raw_store,
        feature_store=trainer.feature_store,
        last_seen_ts=trainer.last_seen_ts,
        last_retrain_ts=trainer.last_retrain_ts,
        new_feat_rows_since_retrain=trainer.new_feat_rows_since_retrain,
        ct_cfg=trainer.ct_cfg,
        val_fraction=trainer.val_fraction,
        inference=trainer.inference,
    )
    joblib.dump(state, path)


def load_continuous_state(path: str, logger: Optional[logging.Logger] = None) -> ContinuousRegimeTrainer:
    logger = logger or setup_logger(level=logging.INFO)
    state: ContinuousTrainerState = joblib.load(path)

    trainer = ContinuousRegimeTrainer(
        bundle=state.bundle,
        history_df=state.raw_store,      # already indexed; handled by ensure_datetime_index_flexible()
        ct_cfg=state.ct_cfg,
        inference=state.inference,
        logger=logger,
        feature_store=state.feature_store,
        val_fraction=state.val_fraction,
    )
    trainer.last_seen_ts = state.last_seen_ts
    trainer.last_retrain_ts = state.last_retrain_ts
    trainer.new_feat_rows_since_retrain = state.new_feat_rows_since_retrain
    return trainer

if __name__ == "__main__":
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(
        description="Liquidity pressure HMM pipeline (prepare -> train -> apply -> visualize).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("liq_pressure_hmm/configs/feature_plie_HMM.json"),
        help="Path to JSON config (relative paths resolved from repo root).",
    )
    parser.add_argument("--skip-prepare-input", action="store_true", help="Skip building merged input CSV.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain and overwrite the HMM bundle. By default, an existing bundle is reused to keep historical states stable.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Deprecated alias for the default reuse-existing-bundle behavior.")
    parser.add_argument("--skip-apply", action="store_true", help="Skip applying model and load existing state CSV.")
    parser.add_argument("--skip-vis", action="store_true", help="Skip HTML visualization.")
    args = parser.parse_args()

    if args.retrain and args.skip_train:
        parser.error("--retrain and --skip-train cannot be used together.")

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    paths = cfg.get("paths", {})

    def rpath(p: str, default: str) -> Path:
        v = Path(paths.get(p, default))
        return v if v.is_absolute() else (root / v)

    features_rpn_path = rpath("features_rpn", "liq_dataflow/data/features/features_rpn.csv")
    price_10m_path = rpath("price_10m", "liq_dataflow/data/raw/BTC_price_10m.csv")
    merged_input_path = rpath("merged_output", "liq_pressure_hmm/input/BTC_price_lld_10m.csv")
    bundle_output_path = rpath("bundle_output", "liq_pressure_hmm/model/liq_hmm_bundle.joblib")
    state_output_path = rpath("states_output", "liq_pressure_hmm/output/hmm_state.csv")
    vis_output_path = rpath("vis_output", "liq_pressure_hmm/output/hmm_price_states.html")

    feature_cfg = FeatureConfig(**cfg.get("feature_config", {}))
    hmm_cfg = HMMConfig(**cfg.get("hmm_config", {}))
    inference_cfg = InferenceConfig(**cfg.get("inference_config", {}))
    alignment_cfg = AlignmentConfig(**cfg.get("alignment_config", {}))

    train_cfg = cfg.get("train", {})
    val_fraction = float(train_cfg.get("val_fraction", 0.2))
    relabel_neutral_eps = float(train_cfg.get("relabel_neutral_eps", 0.0))

    vis_cfg = cfg.get("visualization", {})
    tail_bars = vis_cfg.get("tail_bars", None)
    tail_days = vis_cfg.get("tail_days", None)
    diagnostic_dir = vis_cfg.get("diagnostic_dir", "liq_pressure_hmm/output/diagnostics")
    diagnostic_event_quantile = float(vis_cfg.get("event_quantile", 0.99))
    price_plot_max_points = vis_cfg.get("price_plot_max_points", 60000)
    dashboard_max_points = int(vis_cfg.get("dashboard_max_points", 30000))
    state_boxplot_max_points_per_state = int(vis_cfg.get("state_boxplot_max_points_per_state", 6000))
    max_shape_segments = int(vis_cfg.get("max_shape_segments", 1200))
    state_background_max_points = int(vis_cfg.get("state_background_max_points", 8000))
    state_background_opacity = float(vis_cfg.get("state_background_opacity", 0.20))
    write_index = bool(vis_cfg.get("write_index", True))

    stability_cfg = cfg.get("stability", {})
    stability_enabled = bool(stability_cfg.get("enabled", False))
    stability_preserve_existing = bool(stability_cfg.get("preserve_existing_output", True))
    stability_history_reference = stability_cfg.get("history_reference", None)
    stability_report_path_cfg = stability_cfg.get("report_output", "liq_pressure_hmm/output/stability_overlay_report.json")
    stability_report_path = Path(stability_report_path_cfg)
    if not stability_report_path.is_absolute():
        stability_report_path = root / stability_report_path

    logger = setup_logger(level=logging.INFO)

    if not args.skip_prepare_input:
        from liq_pressure_hmm.feature.build_input import build_btc_price_lld_10m

        build_btc_price_lld_10m(
            features_rpn_path=features_rpn_path,
            price_10m_path=price_10m_path,
            output_path=merged_input_path,
            alignment_cfg=alignment_cfg,
        )
        logger.info(f"Wrote merged input: {merged_input_path}")

    df_data = pd.read_csv(merged_input_path)
    df_data["time"] = pd.to_datetime(df_data["time"])
    df_data = df_data.sort_values("time").reset_index(drop=True)

    should_train = bool(args.retrain or not bundle_output_path.is_file())

    if should_train:
        bundle, _info = train_liquidation_hmm(
            df_data,
            feature_cfg,
            hmm_cfg,
            val_fraction=val_fraction,
            logger=logger,
            warm_start_bundle=None,
            relabel_neutral_eps=relabel_neutral_eps,
        )
        bundle_output_path.parent.mkdir(parents=True, exist_ok=True)
        save_bundle(bundle, str(bundle_output_path))
        logger.info(f"Saved model bundle: {bundle_output_path}")
    else:
        bundle = load_bundle(str(bundle_output_path))
        logger.info(f"Loaded existing model bundle: {bundle_output_path}")

    if not args.skip_apply:
        stable_references: List[Tuple[str, pd.DataFrame]] = []
        if stability_enabled and stability_preserve_existing and state_output_path.is_file():
            stable_references.append((f"existing:{state_output_path}", pd.read_csv(state_output_path, low_memory=False)))
        if stability_enabled and stability_history_reference:
            stable_references.append((
                f"history_reference:{stability_history_reference}",
                _read_csv_reference(str(stability_history_reference), root=root),
            ))

        df_with_states = apply_model_bundle(
            df_data,
            bundle,
            inference=inference_cfg,
            add_prob_cols=True,
            logger=logger,
        )
        overlay_stats: List[Dict[str, Any]] = []
        if stable_references:
            df_with_states, overlay_stats = overlay_stable_history(
                df_with_states,
                references=stable_references,
                time_col="time",
                logger=logger,
            )
            stability_report_path.parent.mkdir(parents=True, exist_ok=True)
            stability_report_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "state_output": str(state_output_path),
                        "computed_rows": int(len(df_with_states)),
                        "references": overlay_stats,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info(f"Wrote stability overlay report: {stability_report_path}")
        state_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_states.reset_index().to_csv(state_output_path, index=False)
        logger.info(f"Wrote states: {state_output_path}")
    else:
        df_with_states = pd.read_csv(state_output_path, low_memory=False)
        df_with_states["time"] = pd.to_datetime(df_with_states["time"])
        df_with_states = df_with_states.set_index("time").sort_index()
        logger.info(f"Loaded states: {state_output_path}")

    if not args.skip_vis:
        from liq_pressure_hmm.vis_HMM import HMMVisStyle, plot_price_with_hmm_states_html

        df_vis = df_with_states
        if isinstance(tail_days, (int, float)) and tail_days > 0:
            df_vis = df_vis[df_vis.index >= (df_vis.index.max() - pd.Timedelta(days=float(tail_days)))]
        elif tail_bars is not None:
            df_vis = df_vis.iloc[-int(tail_bars) :]

        plot_price_with_hmm_states_html(
            df_vis,
            output_path=vis_output_path,
            max_points=int(price_plot_max_points) if price_plot_max_points is not None else None,
            max_shape_segments=max_shape_segments,
            style=HMMVisStyle(
                band_alpha=state_background_opacity,
                background_max_points=state_background_max_points,
            ),
        )
        logger.info(f"Wrote visualization: {vis_output_path}")

        diag_path = Path(diagnostic_dir)
        if not diag_path.is_absolute():
            diag_path = root / diag_path
        diag_outputs = generate_diagnostic_report(
            df_vis,
            out_dir=diag_path,
            event_quantile=diagnostic_event_quantile,
            dashboard_max_points=dashboard_max_points,
            state_boxplot_max_points_per_state=state_boxplot_max_points_per_state,
            max_shape_segments=max_shape_segments,
            background_max_points=state_background_max_points,
            background_opacity=state_background_opacity,
            write_index=write_index,
        )
        logger.info(f"Wrote diagnostic report: {diag_outputs}")

        if write_index:
            output_index = vis_output_path.parent / "index.html"
            links = {"Main price/state chart": vis_output_path}
            links.update({f"Diagnostics - {k}": v for k, v in diag_outputs.items()})
            write_report_index_html(
                output_path=output_index,
                title="BTC liquidation HMM output index",
                links=links,
                notes=[
                    "This page links to all HTML diagnostics generated under the output directory.",
                    "Visualization plots may be downsampled for speed; CSV/model outputs are not downsampled.",
                ],
            )
            logger.info(f"Wrote output index: {output_index}")
