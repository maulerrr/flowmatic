# Research Protocol and Official Baseline Parity

## Completed Locally
- Full-row enhanced training protocol over Astana, ETT, Weather, and Traffic datasets.
- Three-seed robustness protocol with seeds `7`, `42`, and `2026`.
- Sequence-length ablations for PatchTST-style and DLinear-style models.
- Cross-dataset zero-shot transfer tests for DLinear-style and PatchTST-style models.
- Streaming throughput and latency benchmarks over batch sizes `1`, `16`, and `64`.
- Production bundles with `.pt`, `.safetensors`, TorchScript, metadata, checksums, and model cards.

## Local Parity Implementations
| Family | Local run kind | Paper/reference intent | Status |
| --- | --- | --- | --- |
| PatchTST | `patchtst_forecast` | Channel-independent patch transformer | Local reproduction implemented |
| iTransformer | `itransformer_forecast` | Variables as tokens, temporal history as features | Local reproduction implemented |
| TimesNet | `timesblock_forecast` | Multi-period 2D temporal variation modeling | Local inspired reproduction implemented |
| DLinear/NLinear | `dlinear_forecast`, `nlinear_forecast` | Linear decomposition and normalization baselines | Local reproduction implemented |
| SAITS | `saits_imputer` | Self-attention masked time-series imputation | Local inspired reproduction implemented |
| TranAD | `tranad_anomaly` | Transformer residual-conditioned anomaly reconstruction | Local inspired reproduction implemented |
| STGCN | `stgcn_forecast` | Spatio-temporal graph convolution forecasting | Local inspired reproduction implemented |

## Exact Official Baseline Step Before Submission
For a Q1 submission, exact official upstream repositories should be run with identical splits and budgets, then included as a separate comparison block. The current codebase intentionally keeps local implementations small enough for production use and backend integration; that is not the same as claiming official SOTA parity.

Recommended official checks:
- PatchTST official implementation against ETT/Weather/Traffic splits.
- Time-Series-Library implementations of TimesNet, iTransformer, DLinear, and NLinear.
- SAITS/BRITS official imputation baselines on masked smart-city streams.
- TranAD/Anomaly Transformer/USAD anomaly baselines with injected and real anomalies.
- STGCN/DCRNN/Graph WaveNet/STAEformer on traffic graph datasets.

## Paper Figures
Use SVG files for vector publication where possible, PNG files for quick manuscript previews.