# BTC-LPTM-FeatureEngineering
BTC Liquidation–Price Transmission Regime Identification and Dominant Mechanism Explanation Feature Engineering


## download
python -m liq_data_download.scripts.run_data_download

## liq_dataflow
python -m liq_dataflow.feature_liq_dataflow

## hmm_pressure
python -m liq_pressure_hmm.feature_plie_HMM

默认命令会在已有 `liq_pressure_hmm/model/liq_hmm_bundle.joblib` 时复用该 bundle，避免历史状态因重训发生漂移；只有 bundle 不存在时才会训练新模型。若需要显式重训 HMM，请使用：

```bash
python -m liq_pressure_hmm.feature_plie_HMM --retrain
```

总入口 `run_readme_pipeline.py` 默认复用已有 HMM bundle；需要显式重训时运行：

```bash
python run_readme_pipeline.py --retrain-hmm
```

总入口的新数据模式会运行：

```bash
python -m liq_pressure_hmm.feature_plie_HMM --skip-train
```

如果 `liq_pressure_hmm/model/liq_hmm_bundle.joblib` 不存在，默认模式会直接停止，避免无意中训练出新 HMM。

为保持历史 model input 稳定，总入口会在每个步骤运行前快照已有输出，步骤结束后先恢复已验收行，再用 `input.zip` 中的基线历史表覆盖初始历史，只保留真正新追加的时间点。覆盖逻辑在 `stable_history.py`，HMM 输出层也会按 `liq_pressure_hmm/configs/feature_plie_HMM.json` 的 `stability.history_reference` 锁定历史状态。

## plie
python -m plie_pic_project.feature_plie_pic

## price context
python -m price_context.feature_price_context

## abs
python -m qd_mar_project.feature_qd_mar

## liqprice
python -m btc_liqprice_features_artifact.feature_liqprice

## result
plie_pic_project/outputs/predictions/plie_predictions_source.csv

qd_mar_project/output/features/absorption_memory.csv
qd_mar_project/output/features/base_context.csv

qd_mar_project/output/features/path_absorption_multiscale.csv
qd_mar_project/output/features/path_absorption.csv

price_context/output/price_context_features.csv

btc_liqprice_features_artifact/output/liqprice_features.csv

liq_dataflow/data/features/features_liq_dataflow.csv
