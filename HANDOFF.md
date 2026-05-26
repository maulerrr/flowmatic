# Flowmatic Agent Handoff

**Date:** 2026-05-21  
**Repository:** [github.com/maulerrr/flowmatic](https://github.com/maulerrr/flowmatic)  
**Branch at handoff:** `main`

---

## Status summary (read this first)

| Area | Status | Notes |
|------|--------|--------|
| **Platform (Flowmatic app)** | **OK** | Docker stack runs; Auto/Manual core unit, smart-city workbench, insights, batch pipeline, HF integration. Suitable for demo and further product work. |
| **ML research (Phases 2–3)** | **OK (core done)** | 8-model production portfolio, multi-seed metrics, datasets ingested, checkpoints on Hugging Face. Gaps: router ablations, Astana→PeMS holdout eval, CRPS/AUROC. |
| **Thesis / paper (Phase 4)** | **NOT ACCEPTABLE** | Draft PDF `memoirthesis-v5-phase4.pdf` (local only, **do not treat as submission-ready**) has structural rewrites but missing UI screenshots, broken Unicode in Ch.1, proxy figures, incomplete classical Ch.5 integration, and no defensible single narrative. **Requires full rewrite**, not patch. |

**Primary handoff doc:** this file. **Roadmap:** [`ADAPTIVE_PLATFORM_ROADMAP.md`](./ADAPTIVE_PLATFORM_ROADMAP.md). **Setup:** [`README.md`](./README.md).

---

## 1. What Flowmatic is

Production smart-city + data-prep platform:

- **Batch path:** CSV/JSON upload → quality → clean → export (Postgres, Mongo, HF, files).
- **Smart City path:** sensor sources → **adaptive core unit** (Manual/Auto model routing) → medallion lake → export → optional federated demo.
- **Insight Engine:** scheduled analysis, charts, copilot chat (`/insights`).
- **ML backbone:** PyTorch research under `models/`; inference via `model-inference` service; registry drives Auto routing.

**Stack:** NestJS (Fastify) + Vue 3 + PostgreSQL + MinIO + RabbitMQ + Bun backend; Python inference sidecar; Docker Compose.

**Demo login:** `thesis.demo@flowmatic.local` / `ThesisDemo2025!` (re-register if DB volume was reset).

---

## 2. Phased delivery state

### Phase 1 — Adaptive core unit ✅

Implemented and unit-tested:

| Component | Path |
|-----------|------|
| Model registry | `backend/src/modules/smart-city/pipeline-model-registry.service.ts` |
| Rule router | `backend/src/modules/smart-city/pipeline-model-router.service.ts` |
| Optional LLM policy | `pipeline-auto-routing.service.ts` (on Save Auto policy, not per event) |
| Workbench UI | `frontend/src/modules/smart-city/components/pipeline-processing-stage.vue` |
| Tests | `backend/src/modules/smart-city/__tests__/pipeline-model-router.spec.ts` |

**Requires for Auto mode:** `models/` mounted in backend container (`docker-compose.yml` volume `./models:/app/models:ro`) and `models/reports/production_portfolio.json` or checkpoints present.

### Phase 2 — Portfolio & datasets ✅

**Run:** `python models/paper/run_phase2_prep.py`

| Artifact | Purpose |
|----------|---------|
| `models/reports/production_portfolio.json` | 8 official models + capabilities + priorities |
| `models/paper/reports/dataset_manifest.json` | 6 dataset families ingested |
| `models/configs/phase3_multiseed_suite.yaml` | Train/validation config |

**Production portfolio (8 models):**

| Slot | Checkpoint run | Kind |
|------|----------------|------|
| Traffic forecast | `q1_astana_patchtst_density_forecaster` | patchtst_forecast |
| Traffic anomaly | `q1_astana_tranad_anomaly_detector` | tranad_anomaly |
| Weather forecast | `q1_hf_weather_timesblock_forecaster` | timesblock_forecast |
| Graph traffic | `q1_hf_traffic_stgcn_forecaster` | stgcn_forecast |
| Imputation | `q1_astana_saits_imputer` | saits_imputer |
| Severity classification | `q1_astana_transformer_severity_classifier` | transformer_classifier |
| Energy forecast | `q1_hf_ett_dlinear_energy_forecaster` | dlinear_forecast |
| Speed forecast | `q1_astana_itransformer_speed_forecaster` | itransformer_forecast |

### Phase 3 — Experiments (partial) ⚠️

**Run:** `python models/paper/run_phase3_core.py`

Done:

- Multi-seed aggregate (seeds 42, 7, 2026) → `models/paper/tables/multi_seed_aggregate.{md,csv}`
- Streaming latency benchmark → `models/paper/tables/streaming_benchmark.csv`
- HF upload of portfolio → `models/reports/huggingface_model_manifest.json`
- Summary → `models/paper/reports/phase3_core_summary.json`

**Not done** (do not claim in paper):

- Auto router vs oracle ablation
- Astana → PeMS/METR-LA external holdout
- CRPS, AUROC, calibration metrics
- Full `paper_experiments.py` refresh (ablation + transfer retrain)

**Evaluation protocol (neural):** 70% train / 15% val / **15% temporal test** (`models/flowml/data.py` → `split_dataset`). Test metrics only on held-out tail.

### Phase 4 — Thesis rewrite ❌ UNACCEPTABLE

Attempted via `scripts/build-thesis-phase4.ps1` + LaTeX under `thesis/latex/chapters/`.

**Problems with draft:**

- Local PDF only; not reviewed for submission
- Missing real UI screenshots (`thesis/figures/screenshots/` not reliably in repo)
- Architecture PNGs may be missing on fresh clone
- Unicode errors in legacy Introduction text
- Ch.5 mixes new neural narrative with removed classical figures/tables inconsistently
- Does not meet Q1 bar for experiment section (missing router ablation, external validation, probabilistic metrics)

**LaTeX sources worth keeping:** `thesis/latex/chapters/*.tex`, `model_formulas.tex`, `tables_phase4.tex`, `figures_phase4.tex`.

**Canonical memoir project (external):** `C:\Users\BG\Desktop\master-thesis\dissertation_latex_v5\` — build script syncs into this tree.

---

## 3. Hugging Face models (published weights)

**Hub user:** `@pushthetempo`  
**Manifest:** `models/reports/huggingface_model_manifest.json`  
**Upload command:** `python models/upload_checkpoints_to_hf.py --portfolio`

| Model | Hugging Face URL |
|-------|------------------|
| PatchTST traffic density | https://huggingface.co/pushthetempo/flowmatic-astana-patchtst-density-forecaster |
| TranAD anomaly | https://huggingface.co/pushthetempo/flowmatic-astana-tranad-anomaly-detector |
| TimesBlock weather | https://huggingface.co/pushthetempo/flowmatic-weather-timesblock-forecaster |
| STGCN graph traffic | https://huggingface.co/pushthetempo/flowmatic-traffic-stgcn-forecaster |
| SAITS imputer | https://huggingface.co/pushthetempo/flowmatic-astana-saits-imputer |
| Transformer severity classifier | https://huggingface.co/pushthetempo/flowmatic-astana-transformer-severity-classifier |
| DLinear energy (ETT) | https://huggingface.co/pushthetempo/flowmatic-ett-dlinear-energy-forecaster |
| iTransformer speed | https://huggingface.co/pushthetempo/flowmatic-astana-itransformer-speed-forecaster |

**Note:** Git does **not** contain checkpoint binaries (`.gitignore`). On a new machine, restore weights from HF:

```bash
pip install huggingface_hub
python models/download_checkpoints_from_hf.py --skip-existing
```

Requires `HF_TOKEN` or `HUGGINGFACE_TOKEN` in `.env` (public repos work without token; token avoids rate limits).

---

## 4. Key experiment numbers (held-out test, multi-seed)

From `models/paper/tables/multi_seed_aggregate.md` (n=3 seeds):

| Model | Metric | Mean ± 95% CI |
|-------|--------|----------------|
| PatchTST (Astana density) | RMSE | 0.561 ± 0.004 |
| TranAD (Astana) | RMSE | 0.035 ± 0.004 |
| TimesBlock (weather) | RMSE | 0.034 ± 0.002 |
| STGCN (HF traffic) | RMSE | 0.479 ± 0.002 |
| SAITS imputer | Masked MSE | 0.640 ± 0.009 |
| Transformer severity | Macro-F1 | **0.911 ± 0.017** |
| DLinear (ETT) | RMSE | 0.153 ± 0.011 |
| iTransformer (speed) | RMSE | 0.999 ± 0.001 |

Classical thesis baseline (original Ch.5, not re-run on platform): XGBoost **0.9683 accuracy** on prepared Astana tabular features.

---

## 5. Repository map (what matters)

```
flowmatic/
├── HANDOFF.md                    ← this file
├── ADAPTIVE_PLATFORM_ROADMAP.md  ← phased plan
├── README.md                     ← setup + quickstart
├── docker-compose.yml            ← default stack (models mount on backend)
├── .env.example
├── backend/                      ← NestJS API, Prisma, smart-city module
├── frontend/                     ← Vue SPA
├── services/
│   ├── sensor-simulator/         ← Astana traffic + weather HTTP/WS
│   ├── model-inference/          ← TorchScript / HF inference
│   └── federated-coordinator-demo/
├── models/
│   ├── checkpoints/              ← GITIGNORED — train or restore from HF
│   ├── datasets/                 ← GITIGNORED — run prepare_benchmark_datasets.py
│   ├── configs/phase3_multiseed_suite.yaml
│   ├── paper/                    ← experiment scripts + tables + figures
│   ├── reports/
│   │   ├── production_portfolio.json
│   │   ├── huggingface_model_manifest.json
│   │   └── phase2_prep_summary.json
│   ├── upload_checkpoints_to_hf.py
│   └── paper/run_phase{2,3}_*.py
├── thesis/
│   ├── GAP_ANALYSIS.md
│   ├── THESIS_SUPPLEMENT_V2.md
│   └── latex/                    ← Phase 4 rewrite sources (draft quality)
├── data/                         ← GITIGNORED — astana_synthetic_data.csv locally
└── scripts/build-thesis-phase4.ps1
```

---

## 6. Commands cheat sheet

### Platform

```bash
cp .env.example .env
docker compose up -d --build
# UI: http://localhost  |  API: http://localhost:8080/api/v1
```

### ML / experiments

```bash
# Python 3.10+; PyTorch with CUDA optional
pip install torch pandas pyyaml huggingface_hub  # see models/README.md for full deps

python models/download_checkpoints_from_hf.py --skip-existing  # new device
python models/paper/run_phase2_prep.py      # datasets + portfolio
python models/paper/run_phase3_core.py      # aggregate + streaming + HF upload
python models/paper_experiments.py          # full retrain (hours) — optional

python models/upload_checkpoints_to_hf.py --portfolio
```

### Backend tests

```bash
cd backend && bun run test -- src/modules/smart-city/__tests__/pipeline-model-router.spec.ts
```

---

## 7. Environment secrets (never commit)

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Dataset download, HF export, model catalogue |
| `OPENAI_API_KEY` | Upload summaries, insights copilot, optional Auto policy LLM |
| `JWT_SECRET` | Auth |
| Postgres/MinIO/RabbitMQ passwords | Infra |

Set in root `.env` and/or `backend/.env` (Docker reads both for backend service).

---

## 8. Known gaps & recommended next work

### Platform (low priority — OK)

- Human QA checklist in roadmap §Phase 1 verification still unchecked
- Capture fresh workbench screenshots into `thesis/figures/screenshots/` and commit

### Experiments (medium)

1. Run Astana→PeMS transfer eval using ingested `pems_metr_la`
2. Router ablation: auto vs oracle vs wrong manual
3. Regenerate `paper_experiments.py` tables if retraining

### Thesis (high — blocked on quality)

1. **Reject** `memoirthesis-v5-phase4.pdf` as baseline
2. Restore/commit UI screenshots + architecture PNGs (from `.mmd` via mermaid-cli)
3. Rewrite Ch.5 as single coherent experiment chapter (classical study → platform validation → neural portfolio)
4. Fix Introduction Unicode; align abstract with one classifier story (RF 96.62% vs XGBoost vs Transformer)
5. External validation subsection with honest limitations
6. Only then rebuild PDF

---

## 9. Docker / compose notes

- **Default `up`:** postgres, minio, rabbitmq, sensor-simulator, model-inference, backend, frontend. **No** export-test DBs.
- **Export-test profile:** `docker compose --profile export-test up -d postgres-export-test mongo-export-test`
- **GPU inference:** `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d model-inference`
- Backend **`depends_on` model-inference** — inference must be healthy before API starts
- **`./models:/app/models:ro`** on backend — required for Auto routing registry

---

## 10. Git commit at this handoff

This snapshot commits:

- Handoff + README refresh
- Phase 2–3 scripts, configs, reports, experiment tables
- Thesis LaTeX draft sources (not the PDF)
- Platform registry changes for production portfolio
- Paper figures under `models/paper/figures/` (for reproducibility)

**Not committed:** `data/`, `models/checkpoints/`, `models/datasets/`, `.env*`, local thesis PDF.

---

## 11. Contact context for next agent

- User language: English; thesis may be Russian/English mix in external memoir project
- User rejected Phase 4 PDF quality — prioritize platform + experiments over thesis patches until a clear rewrite plan exists
- Models on HF to avoid GitHub 100MB limit
- Prior conversation arc: Phase 1 Auto routing → Phase 2 portfolio/datasets → Phase 3 core experiments + HF push → Phase 4 thesis attempt (failed quality bar)

**Start here:** read this file → run `docker compose up` → verify Auto mode with demo login → read `production_portfolio.json` → decide thesis vs experiment next steps with user.
