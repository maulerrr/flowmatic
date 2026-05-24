# Flowmatic Q1 Experiment Report

## Scope
This report summarizes neural forecasting, imputation, anomaly detection, classification, graph forecasting, multi-seed robustness, ablations, cross-dataset transfer, and streaming benchmarks.

## Best Forecasting/Reconstruction Runs
| name                                         | kind                | dataset    | test_rmse | test_mae |
| -------------------------------------------- | ------------------- | ---------- | --------- | -------- |
| q1_astana_tranad_anomaly_detector_seed42     | tranad_anomaly      | astana     | 0.0308    | 0.0226   |
| q1_hf_weather_timesblock_forecaster_seed2026 | timesblock_forecast | hf_weather | 0.0332    | 0.0254   |
| q1_hf_weather_timesblock_forecaster_seed42   | timesblock_forecast | hf_weather | 0.0335    | 0.0264   |
| q1_astana_tranad_anomaly_detector_seed7      | tranad_anomaly      | astana     | 0.0354    | 0.0263   |
| q1_hf_weather_timesblock_forecaster_seed7    | timesblock_forecast | hf_weather | 0.0356    | 0.0279   |

## Best Classification Runs
| name                                             | kind                   | dataset | test_accuracy | test_macro_f1 |
| ------------------------------------------------ | ---------------------- | ------- | ------------- | ------------- |
| q1_astana_transformer_severity_classifier_seed7  | transformer_classifier | astana  | 0.9909        | 0.9252        |
| q1_astana_transformer_severity_classifier        | transformer_classifier | astana  | 0.9892        | 0.9177        |
| q1_astana_transformer_severity_classifier_seed42 | transformer_classifier | astana  | 0.9893        | 0.9119        |

## Figures
- `figures/forecast_rmse_leaderboard.png`
- `figures/classification_macro_f1.png`
- `figures/latency_vs_parameters.png`
- `figures/validation_curves_q1_suite.png`
- `figures/multi_seed_rmse_boxplot.png`
- `figures/sequence_length_ablation.png`
- `figures/cross_dataset_transfer.png`
- `figures/streaming_throughput.png`
- `figures/streaming_latency.png`

## Publication Caveat
These are now reproducible research artifacts. For submission, run exact official repositories with matched dataset splits where licensing/dependencies allow, then cite the local implementation as the production backbone.