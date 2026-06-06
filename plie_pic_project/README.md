# PLIE-PIC

Source-clock, mechanism-constrained passive liquidation impact curve for BTC futures liquidation pressure.

## Quick start

```bash
pip install -r plie_pic_project/requirements.txt
# Run from the repository root, the parent directory of plie_pic_project.
python -m plie_pic_project.feature_plie_pic
```

Open:

```text
reports/html/index.html
```

## Common commands

```bash
# Full training + evaluation artifacts
python -m plie_pic_project.feature_plie_pic train

# Batch inference with existing model
python -m plie_pic_project.feature_plie_pic infer

# Refresh monitoring/evaluation without retraining
python -m plie_pic_project.feature_plie_pic monitor

# Monthly scheduled retraining check
python -m plie_pic_project.feature_plie_pic scheduled-retrain

# Force full retrain by setting runtime.force_retrain: true in config/config.yaml,
# then run scheduled-retrain.

# Generate HTML only
python -m plie_pic_project.feature_plie_pic report

# Tests
PYTHONPATH=plie_pic_project/src pytest -q plie_pic_project/tests
```

## Key docs

- `docs/manual.md`
- `docs/algorithm.md`
- `docs/code_design.md`
- `docs/plie_feature_engineering.md`
