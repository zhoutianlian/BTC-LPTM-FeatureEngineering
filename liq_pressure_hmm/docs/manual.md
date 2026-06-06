# Manual — liq_pressure_hmm v2

## Run

```bash
python -m liq_pressure_hmm.feature_plie_HMM
```

If `liq_pressure_hmm/model/liq_hmm_bundle.joblib` already exists, the command reuses it by default so historical states remain stable. Use `--retrain` only when you intentionally want to fit and overwrite a new HMM bundle.

When `stability.enabled` is true in `configs/feature_plie_HMM.json`, the generated state CSV preserves overlapping rows from the configured historical reference and keeps computed values only for new timestamps.

The pipeline writes:

- `liq_pressure_hmm/input/BTC_price_lld_10m.csv`
- `liq_pressure_hmm/model/liq_hmm_bundle.joblib`
- `liq_pressure_hmm/output/hmm_state.csv`
- `liq_pressure_hmm/output/hmm_price_states.html`
- `liq_pressure_hmm/output/index.html`
- `liq_pressure_hmm/output/diagnostics/index.html`
- `liq_pressure_hmm/output/diagnostics/state_distribution.html`
- `liq_pressure_hmm/output/diagnostics/regime_dashboard.html`
- `liq_pressure_hmm/output/diagnostics/state_feature_boxplots.html`
- `liq_pressure_hmm/output/diagnostics/transition_duration.html`
- `liq_pressure_hmm/output/diagnostics/event_windows.html`
- `liq_pressure_hmm/output/diagnostics/state_semantic_summary.csv`

## State semantics

- `state 1`: short-liquidation strong dominance / 空头清算强势占优 / strong upward liquidation pressure
- `state 2`: short-liquidation mild dominance / 空头清算轻度占优 / mild upward liquidation pressure
- `state 3`: balanced long/short liquidations / 空头/多头清算均衡 / no clear liquidation pressure
- `state 4`: long-liquidation mild dominance / 多头清算轻度占优 / mild downward liquidation pressure
- `state 5`: long-liquidation strong dominance / 多头清算强势占优 / strong downward liquidation pressure

## Visualization speed controls

Configure in `configs/feature_plie_HMM.json`:

```json
"visualization": {
  "price_plot_max_points": 60000,
  "dashboard_max_points": 30000,
  "state_boxplot_max_points_per_state": 6000,
  "max_shape_segments": 1200,
  "state_background_max_points": 8000,
  "state_background_opacity": 0.20,
  "write_index": true
}
```

These parameters only affect HTML size and plotting speed. They do not downsample `hmm_state.csv` or model outputs.

## Recommended inspection order

1. Open `output/index.html`.
2. Check `state_distribution.html` for state counts and shares.
3. Check `state_feature_boxplots.html` for semantic separation across states.
4. Check `regime_dashboard.html` for time-series behavior, hover details, confidence, entropy, age, and staleness.
5. Check `event_windows.html` for price response after extreme liquidation events.

Background performance note:

- `hmm_price_states.html` and the first row of `regime_dashboard.html` now use a compact categorical heatmap background instead of per-segment rectangle shapes.
- `state_background_max_points` only controls visual background resolution; it does not modify `hmm_state.csv` or any model output.
- Increase it, for example to `12000`, if you want finer state-band detail; decrease it, for example to `4000`, if HTML size is more important.

## Visualization background bands after V3

The state-colored background in `hmm_price_states.html` and `diagnostics/regime_dashboard.html` is now rendered as a compact heatmap layer rather than one rectangle per state segment.  This fixes the case where the background disappeared when the state path had too many transitions.

Relevant config:

```json
"visualization": {
  "state_background_max_points": 8000,
  "state_background_opacity": 0.20,
  "price_plot_max_points": 60000,
  "dashboard_max_points": 30000
}
```

Operational notes:

- Increase `state_background_max_points` if you want finer background boundaries in long samples.
- Decrease it if HTML size or browser rendering becomes heavy.
- This setting affects only the visual background layer; it does not change model output or feature CSVs.
