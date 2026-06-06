# Code design — liq_pressure_hmm v2

## Layers

1. `alignment.py`
   - Causal mixed-frequency alignment.
   - Handles timestamp offset, merge tolerance, raw/source timestamps, and feature age.

2. `feature_plie_HMM.py`
   - Main training/inference CLI.
   - Builds source-clock HMM features.
   - Trains sticky Gaussian HMM.
   - Applies filtered posterior inference and debounced hard decoding.
   - Writes model outputs and visual diagnostics.

3. `state_summary.py`
   - Stable state semantics.
   - Corrected liquidation-side role map.
   - Adds `p_short_liq`, `p_up_pressure`, `p_long_liq`, `p_down_pressure`, `p_neutral`, `dir_expect`, `liq_entropy`, `hmm_conf`, `age_in_state`, `age_in_state_source`.

4. `diagnostics.py`
   - Efficient Plotly diagnostic reports.
   - State count distribution.
   - Regime dashboard.
   - State feature boxplots.
   - Transition/duration diagnostics.
   - Event-window diagnostics.
   - HTML index page generation.

5. `vis_HMM.py`
   - Main price/state chart.
   - Optimized with optional downsampling, `Scattergl`, hover state metadata, and an always-visible compact heatmap state background.

6. `tests/`
   - Alignment tests.
   - State semantic and summary feature tests.

## State relabeling contract

The HMM states are relabeled using `short_dom = S - L`, where:

- `S = fsl_cwt_kf`: short-liquidation proxy, forced buying, upward liquidation pressure.
- `L = fll_cwt_kf`: long-liquidation proxy, forced selling, downward liquidation pressure.

Relabeling rules:

1. Choose the old state whose median `short_dom` is closest to zero as new `state 3`.
2. Among the remaining states, the highest median `short_dom` becomes `state 1`.
3. The second-highest median `short_dom` becomes `state 2`.
4. The less-negative median `short_dom` becomes `state 4`.
5. The most-negative median `short_dom` becomes `state 5`.

This enforces monotone liquidation-pressure semantics but does not force equal state counts.

## Visualization performance policy

HTML plots are diagnostics, not model inputs. Large time-series plots are downsampled only for rendering speed. The model output CSV remains full-resolution.

## V3 visualization background design

The visualization layer now uses a bounded heatmap background for regime colors.  The earlier rectangle-band implementation was exact but could become slow when the number of contiguous state episodes was large; the optimized V2 implementation skipped backgrounds above a threshold, which made the chart harder to inspect.  V3 keeps the background visible while bounding rendering cost.

Design choice:

- Price and state hover are carried by the price trace.
- State color background is carried by one low-opacity heatmap trace.
- The heatmap x-axis is uniformly thinned by `state_background_max_points`.
- The layer is added before the price trace, so the price line remains readable.

This is a visual approximation only; model outputs remain exact in `hmm_state.csv`.
