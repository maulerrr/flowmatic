# Flowmatic model inference service

Local PyTorch inference for Flowmatic smart-city checkpoints. Models are **downloaded from Hugging Face once**, cached on disk, loaded into memory, and run on **CPU or GPU** — no Hugging Face serverless API.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health + device info |
| POST | `/v1/prefetch` | Download/cache weights (`modelId` or `localRun`) |
| POST | `/v1/infer` | Run inference on cached weights |
| GET | `/v1/cache` | List HF cache + in-memory loaded models |

### Infer request

```json
{
  "modelId": "pushthetempo/flowmatic-astana-tranad-anomaly-detector",
  "token": "hf_...",
  "inputs": { "averageSpeedKph": 48, "vehicleCount": 92 }
}
```

Or for bundled local checkpoints:

```json
{
  "localRun": "astana_tcn_density_forecaster",
  "inputs": { "Traffic_Density": 74 }
}
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8093` | HTTP port |
| `INFER_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `HF_HOME` | `/cache/huggingface` | Hugging Face hub cache root |
| `CHECKPOINTS_DIR` | `/app/models/checkpoints` | Local checkpoint runs (read-only mount) |

## Docker (with GPU)

From repo root:

```bash
docker compose up -d --build model-inference backend
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host. Your RTX 4060 Ti 8GB is more than enough — Flowmatic checkpoints are typically **tens of MB**, not multi‑GB LLMs.

Set `INFER_DEVICE=cpu` in `.env` if GPU passthrough is unavailable; inference still works.

## Local dev (native GPU)

```bash
cd services/model-inference
pip install -r requirements.txt
export PYTHONPATH=../../models:./src
export CHECKPOINTS_DIR=../../models/checkpoints
export INFER_DEVICE=cuda
python -m uvicorn server:app --host 0.0.0.0 --port 8093
```

Point the backend at `MODEL_INFERENCE_URL=http://127.0.0.1:8093`.
