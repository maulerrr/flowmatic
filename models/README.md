# Flowmatic Q1 Smart-City ML Research Backbone

This folder now contains a real PyTorch research stack for the Flowmatic core processing unit.

It includes:

- trainable neural architectures,
- reproducible dataset loaders,
- time-aware train/validation/test splits,
- checkpointed `.pt` models,
- model cards,
- metrics reports,
- inference samples,
- lightweight JSON preprocessing artifacts retained for backend portability.

## Trained Neural Checkpoints

Saved in `models/checkpoints/`:

- `astana_tcn_density_forecaster`
  - TCN forecasting model for Astana traffic density.
- `astana_gru_speed_forecaster`
  - GRU forecasting model for Astana vehicle speed.
- `hf_weather_transformer_forecaster`
  - Transformer forecaster for weather telemetry.
- `hf_ett_tcn_energy_forecaster`
  - TCN forecaster for ETT energy telemetry.
- `astana_autoencoder_anomaly_repair`
  - Sequence autoencoder for anomaly scoring and reconstruction-based repair.
- `astana_transformer_severity_classifier`
  - Transformer classifier for event severity.
- `hf_traffic_stgcn_forecaster`
  - Spatio-temporal graph forecaster for traffic sensor networks.
- `q1_astana_patchtst_density_forecaster`
  - PatchTST-inspired channel-independent patch transformer.
- `q1_astana_itransformer_speed_forecaster`
  - iTransformer-inspired variate-token forecaster.
- `q1_hf_weather_timesblock_forecaster`
  - TimesNet/TimesBlock-inspired temporal 2D variation forecaster.
- `q1_hf_ett_dlinear_energy_forecaster`
  - DLinear trend/seasonal decomposition forecaster.
- `q1_hf_ett_nlinear_energy_forecaster`
  - NLinear normalization-linear baseline.
- `q1_astana_saits_imputer`
  - SAITS-inspired self-attention masked imputer.
- `q1_astana_tranad_anomaly_detector`
  - TranAD-inspired residual-conditioned transformer anomaly model.
- `q1_astana_transformer_severity_classifier`
  - Production severity classifier with TorchScript export.
- `q1_hf_traffic_stgcn_forecaster`
  - Production spatio-temporal graph forecaster.

Each run contains:

```text
model.pt
model.safetensors
model.torchscript.pt
metadata.json
model_card.md
```

Suite metrics are saved in:

```text
models/reports/q1_core_suite_metrics.json
models/reports/production_model_registry.json
models/checkpoints/registry.json
```

## Datasets Used

- `data/astana_synthetic_data.csv`
- Hugging Face `pkr7098/time-series-forecasting-datasets`
  - `ETTh1.csv`
  - `weather.csv`
  - `traffic.csv`

The Hugging Face token is read only from `HF_TOKEN` when downloading. It must not be committed.

## Upload checkpoints to Hugging Face

Best checkpoint per model family (no seed suffixes, no `q1_` prefix):

```bash
python models/upload_checkpoints_to_hf.py
```

Reads `HF_TOKEN` from the repo root `.env`. Writes `models/reports/huggingface_model_manifest.json` with Hub repo IDs (`pushthetempo/flowmatic-...`).

Also save the same token in Flowmatic **Settings → Hugging Face integration** so the Core Unit catalogue can list your models.

## Phase 2 — Production portfolio & dataset prep

From repo root:

```bash
python models/paper/run_phase2_prep.py
```

Individual steps:

```bash
python models/paper/prepare_benchmark_datasets.py
python models/paper/select_production_portfolio.py
python models/paper/enrich_production_capabilities.py
python models/build_registry.py
```

Phase 3 multi-seed validation uses:

```bash
python models/paper_experiments.py
# or retrain production suite:
python models/train.py --config models/configs/phase3_multiseed_suite.yaml
```

## Train The Q1 Core Suite

From repo root:

```bash
python models/train.py --config models/configs/q1_core_suite.yaml
```

Enhanced Q1/prod suite:

```bash
python models/train.py --config models/configs/q1_enhanced_suite.yaml
python models/build_registry.py
```

## Run Inference

```bash
python models/infer.py --run astana_tcn_density_forecaster
python models/infer.py --run hf_traffic_stgcn_forecaster
python models/infer.py --run q1_astana_patchtst_density_forecaster
python models/infer.py --run q1_astana_saits_imputer
```

## Architecture Map

The framework code lives in:

```text
models/flowml/
  data.py       # dataset loading, schema inference, sliding windows, graph datasets
  models.py     # TCN, GRU, Transformer, Autoencoder, STGCN-style models
  training.py   # train/eval loops, metrics, checkpoint writing
```

Production exports include:

- PyTorch checkpoint: `model.pt`
- safetensors weights: `model.safetensors`
- TorchScript bundle: `model.torchscript.pt`
- metadata with input schema, scaler, latency, parameter count, and test metrics
- SHA256 checksums in `production_model_registry.json`

## Research Roadmap

For a Q1 paper, this is the backbone, not the final result. The next research-grade steps are:

- add stronger baselines: PatchTST, TimesNet, iTransformer, DLinear/NLinear;
- add foundation adapters: Chronos, MOMENT, TimesFM, Lag-Llama;
- add imputation baselines: SAITS and BRITS;
- add anomaly baselines: TranAD, MTAD-GAT, Anomaly Transformer, USAD;
- add traffic graph baselines: DCRNN, STGCN, Graph WaveNet, STAEformer;
- run larger seeds and ablations;
- report latency, throughput, memory, and drift-recovery metrics;
- validate on multiple smart-city domains and cross-city transfer.

## Production/S3 Plan

For production, upload each checkpoint folder to:

```text
s3://<bucket>/<basePrefix>/models/<organizationId>/<runName>/<version>/
```

The backend processing unit should load:

- `metadata.json` for schema and task metadata,
- `model.pt` for neural weights,
- `model_card.md` for traceability.
