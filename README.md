# Flowmatic

Intelligent data preparation platform for batch uploads and live smart-city pipelines.

**Stack:** NestJS (Fastify) + Vue 3 + PostgreSQL + MinIO + RabbitMQ + Bun  
**Handoff / roadmap:** [`ADAPTIVE_PLATFORM_ROADMAP.md`](./ADAPTIVE_PLATFORM_ROADMAP.md)

---

## Quick start (Docker — recommended)

```bash
cp .env.example .env
# Optional: set OPENAI_API_KEY, HUGGINGFACE_TOKEN in .env

docker compose up -d --build
```

| URL | Service |
|-----|---------|
| http://localhost | Frontend |
| http://localhost:8080/api/v1 | Backend API |
| http://localhost:8091 | Sensor simulator (Astana WS/HTTP) |
| http://localhost:8093 | Model inference |
| http://localhost:9101 | MinIO console |

**Demo login:** `thesis.demo@flowmatic.local` / `ThesisDemo2025!` (register if DB was reset)

---

## Local development (without Docker)

```bash
# Backend
cd backend
bun install
cp .env.example .env   # edit DATABASE_URL, etc.
bun run prisma:migrate
bun run dev            # http://localhost:8080

# Frontend (separate terminal)
cd frontend
bun install
bun run dev            # http://localhost:5173
```

Requires local Postgres, MinIO, and RabbitMQ — or use Docker for infra only.

---

## Main features

- **Batch pipeline** — CSV/JSON upload → quality → clean → export (HF, Postgres, MongoDB, files)
- **Smart City workbench** (`/connectors`) — sources → core unit (Manual/Auto routing) → medallion lake → export → federated demo
- **Insight Engine** (`/insights`) — scheduled analysis, adaptive charts, copilot chat
- **Model registry** — `models/checkpoints/` + `models/reports/huggingface_model_manifest.json` power auto routing

---

## Optional compose profiles

**Export adapter smoke tests** (Postgres/Mongo test DBs — not started by default):

```bash
docker compose --profile export-test up -d postgres-export-test mongo-export-test
python backend/scripts/e2e/continuous-exports.e2e.py
```

**GPU inference overlay:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d model-inference
```

---

## Project layout

```
flowmatic/
├── backend/           NestJS API, Prisma, queues
├── frontend/          Vue 3 SPA
├── models/            Checkpoints, training scripts, paper tables
├── services/          sensor-simulator, model-inference, federated-coordinator-demo
├── thesis/            Manuscript assets, figures, experiments
├── data/              Local demo CSV (gitignored)
├── docker-compose.yml
└── ADAPTIVE_PLATFORM_ROADMAP.md
```

---

## Commands

```bash
# Backend
cd backend && bun run typecheck && bun run build && bun run test

# Frontend
cd frontend && bun run type-check && bun run build-only
```

---

## Environment

Copy [`.env.example`](./.env.example) for Docker Compose. For local backend dev, see [`backend/.env.example`](./backend/.env.example) (points to the same variables with local hostnames).

---

## Thesis & research

- Gap analysis: `thesis/GAP_ANALYSIS.md`
- Q1 supplement: `thesis/THESIS_SUPPLEMENT_V2.md`
- Experiment leaderboard: `models/paper/tables/checkpoint_leaderboard.md`

Archived planning docs: `docs/archive/`

---

## License

MIT
