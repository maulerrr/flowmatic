# Flowmatic

Intelligent data preparation and adaptive smart-city pipeline platform.

| Doc | Purpose |
|-----|---------|
| **[HANDOFF.md](./HANDOFF.md)** | **Agent handoff (2026-05-21)** — platform OK, thesis NOT acceptable, HF links, experiment state |
| [ADAPTIVE_PLATFORM_ROADMAP.md](./ADAPTIVE_PLATFORM_ROADMAP.md) | Phased roadmap (Phases 1–4) |

**Stack:** NestJS (Fastify) + Vue 3 + PostgreSQL + MinIO + RabbitMQ + Python inference  
**Demo login:** `thesis.demo@flowmatic.local` / `ThesisDemo2025!`

---

## Quickstart (existing machine, ~5 min)

```bash
git clone https://github.com/maulerrr/flowmatic.git
cd flowmatic
cp .env.example .env
# Optional: set HUGGINGFACE_TOKEN, OPENAI_API_KEY in .env

docker compose up -d --build
```

| URL | Service |
|-----|---------|
| http://localhost | Frontend |
| http://localhost:8080/api/v1 | Backend API |
| http://localhost:8091 | Sensor simulator |
| http://localhost:8093 | Model inference |
| http://localhost:9101 | MinIO console |

Open **Connectors** → create/use smart-city pipeline → set core unit to **Auto** (needs `models/checkpoints` or HF models — see below).

---

## New device setup (full)

### 1. Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Latest | Run full stack |
| [Git](https://git-scm.com/) | Any | Clone repo |
| [Bun](https://bun.sh/) | 1.x | Optional local backend/frontend dev |
| Python | 3.10+ | ML scripts, dataset prep |
| NVIDIA driver + CUDA | Optional | GPU training / `docker-compose.gpu.yml` |

**Disk:** ~5 GB for Docker images; additional space for `models/checkpoints/` (~50–200 MB) and `models/datasets/` (~500 MB+ after download).

### 2. Clone and configure

```bash
git clone https://github.com/maulerrr/flowmatic.git
cd flowmatic

cp .env.example .env
cp backend/.env.example backend/.env   # if present; else backend reads root .env via compose
```

Edit `.env`:

```bash
# Required for production-like auth
JWT_SECRET=change_me_to_random_string

# Strongly recommended
HUGGINGFACE_TOKEN=hf_...          # HF dataset download + model catalogue + export
# OPENAI_API_KEY=sk-...           # Upload summaries + Insights copilot (optional)

# Defaults work for local Docker
POSTGRES_PASSWORD=secure_password
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
```

**Never commit `.env` files.**

### 3. Demo data (Astana CSV)

The sensor simulator expects:

```
data/astana_synthetic_data.csv
```

This path is **gitignored**. Obtain the file from:

- Your team backup / previous machine, or
- Regenerate per thesis data spec (30k rows), or
- Copy from thesis experiment artifacts if available.

Without it, the simulator may fail health checks; batch upload experiments also need this file.

### 4. ML checkpoints (Auto routing & inference)

`models/checkpoints/` is **gitignored** (too large for GitHub). On a new device, choose one:

**Option A — Restore from Hugging Face (recommended)**

```bash
pip install huggingface_hub
# Optional: set HF_TOKEN or HUGGINGFACE_TOKEN in .env (avoids rate limits)
python models/download_checkpoints_from_hf.py --skip-existing
```

Downloads all 8 production models into `models/checkpoints/<sourceRun>/` using `models/reports/huggingface_model_manifest.json`. See [HANDOFF.md §3](./HANDOFF.md#3-hugging-face-models-published-weights) for individual repo URLs.

**Option B — Train locally**

```bash
pip install torch pandas pyyaml huggingface_hub  # + deps from models/README.md

python models/paper/run_phase2_prep.py          # downloads HF datasets to models/datasets/
python models/train.py --config models/configs/phase3_multiseed_suite.yaml
```

**Option C — Copy from old machine**

Copy the entire `models/checkpoints/` directory into the repo.

### 5. Start the platform

```bash
docker compose up -d --build
```

Wait until healthy:

```bash
docker compose ps
# backend, model-inference, sensor-simulator should be healthy
```

**Verify API:**

```bash
curl -s http://localhost:8080/api/v1/health
curl -s http://localhost:8093/health
```

**Verify frontend:** http://localhost → login → **Connectors** (smart-city workbench).

### 6. Enable Auto core unit

1. Ensure `models/reports/production_portfolio.json` exists (committed in repo).
2. Ensure checkpoint dirs referenced in portfolio exist under `models/checkpoints/`.
3. Backend mounts `./models:/app/models:ro` — restart after adding checkpoints:

   ```bash
   docker compose restart backend
   ```

4. In workbench → Processing → Core unit → **Auto** → Save.

### 7. Optional: GPU inference

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d model-inference
```

Set `INFER_DEVICE=cuda` in `.env` if supported.

### 8. Optional: export adapter tests

```bash
docker compose --profile export-test up -d postgres-export-test mongo-export-test
python backend/scripts/e2e/continuous-exports.e2e.py
```

---

## ML / research quickstart

```bash
# Phase 2 — datasets + 8-model portfolio metadata
python models/paper/run_phase2_prep.py

# Phase 3 — aggregate multi-seed tables + streaming benchmark + HF upload
python models/paper/run_phase3_core.py

# Full retrain + ablation (hours, GPU recommended)
python models/paper_experiments.py
```

**Key outputs:**

| Path | Content |
|------|---------|
| `models/reports/production_portfolio.json` | Official 8 models |
| `models/paper/tables/multi_seed_aggregate.md` | Test metrics ± CI |
| `models/reports/huggingface_model_manifest.json` | HF URLs |

**HF model links:** see [HANDOFF.md §3](./HANDOFF.md#3-hugging-face-models-published-weights).

---

## Local development (without full Docker)

Run infra in Docker, apps locally:

```bash
docker compose up -d postgres minio rabbitmq sensor-simulator model-inference

cd backend
bun install
# Set DATABASE_URL=postgres://admin:secure_password@localhost:5432/flowmatic in backend/.env
bun run prisma:migrate
bun run dev    # :8080

cd frontend
bun install
bun run dev    # :5173, proxy to API
```

---

## Project layout

```
flowmatic/
├── HANDOFF.md              ← start here for agent continuity
├── backend/                NestJS API, smart-city module, Auto routing
├── frontend/               Vue 3 workbench + insights
├── models/                 Training, checkpoints (local), paper tables
├── services/               sensor-simulator, model-inference, federated demo
├── thesis/                 Gap analysis, LaTeX draft sources (thesis quality: poor)
├── docker-compose.yml
└── data/                   Local Astana CSV (gitignored)
```

---

## Commands

```bash
# Backend
cd backend && bun run typecheck && bun run build && bun run test

# Frontend
cd frontend && bun run type-check && bun run build-only

# Router unit tests
cd backend && bun run test -- src/modules/smart-city/__tests__/pipeline-model-router.spec.ts
```

---

## Thesis & paper status

**Thesis draft status: NOT ACCEPTABLE** for submission. See [HANDOFF.md](./HANDOFF.md).

Useful references:

- `thesis/GAP_ANALYSIS.md` — manuscript vs codebase
- `thesis/THESIS_SUPPLEMENT_V2.md` — platform evolution narrative
- `thesis/latex/chapters/` — draft LaTeX (needs full rewrite)
- `models/paper/tables/checkpoint_leaderboard.md` — neural leaderboard

Do **not** rely on local `memoirthesis-v5-phase4.pdf` without rework.

---

## License

MIT
