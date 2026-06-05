# Flowmatic Agent Handoff

**Date:** 2026-05-27  
**Repository:** [github.com/maulerrr/flowmatic](https://github.com/maulerrr/flowmatic)  
**Branch:** `main`  
**Last updated from:** Laptop (`c:\Users\kurai\OneDrive\Desktop\projects\flowmatic`) — push before continuing on desktop PC.

---

## Status summary (read this first)

| Area | Status | Notes |
|------|--------|--------|
| **Platform (Flowmatic app)** | **OK** | Docker stack runs on laptop. Astana Geospatial Demo pipeline configured. Demo login works. |
| **ML research (Phases 2–3)** | **OK (core done)** | 8-model portfolio, multi-seed metrics, HF checkpoints. Gaps: router ablations, Astana→PeMS holdout, CRPS/AUROC. |
| **Thesis (AITU template)** | **SUBMISSION-DRAFT** | Full 50-page PDF rebuilt on laptop. Citations, TOC/LOF/LOT, figures, metadata fixed. **Not final** — see Q1 review for remaining revisions before antiplagiarism/defense. |
| **Q1 paper review** | **DONE** | `paper-review-q1.md` — defense-ready with revisions; journal-not-ready. |

**Primary handoff doc:** this file. **Thesis review:** [`paper-review-q1.md`](./paper-review-q1.md). **Gap analysis:** [`thesis/GAP_ANALYSIS.md`](./thesis/GAP_ANALYSIS.md). **Setup:** [`README.md`](./README.md).

---

## 1. What changed on the laptop (2026-05-21 → 2026-05-27)

### Thesis — new canonical source tree

The old desktop path `C:\Users\BG\Desktop\master-thesis\dissertation_latex_v5\` is **not on the laptop**. The submission thesis now lives in:

```
thesis/AITU Thesis Template/     ← PRIMARY LaTeX source (6 chapters + appendix)
memoirthesis-flowmatic.pdf       ← Compiled output (repo root, copy of template build)
thesis/figures/                  ← Shared figures (architecture, results, screenshots)
paper-review-q1.md               ← Strict Q1/defense review (2026-05-27)
```

**Build command (requires Docker):**

```powershell
.\scripts\build-thesis-laptop.ps1
```

Output: `thesis/AITU Thesis Template/memoirthesis.pdf` + copy to `memoirthesis-flowmatic.pdf`.

### Thesis fixes applied (2026-05-27)

- BibTeX + 4-pass LaTeX build (citations no longer `?`)
- Table of Contents, List of Figures, List of Tables populated
- `listings` / `fancyvrb` for Appendix A code blocks
- Figure sizing via `\thesisfigure` + `adjustbox` (screenshots fit page)
- Title metadata: **Supervisor Aivar Sakhipov Aituarovich**, program **7M06105**, **School of Software Engineering**
- Neural formulations moved to **Chapter 3** (`sec:neural_formulations`)
- UI screenshots committed under `thesis/figures/screenshots/` (login, workbench, core unit, workflow graph)
- Architecture Mermaid PNGs regenerated via `scripts/render-thesis-figures.ps1`

### Platform demo on laptop

- `.env` + `backend/.env` created from examples
- `data/astana_synthetic_data.csv` generated (30k rows): `node scripts/generate-astana-dataset.js`
- Astana Geospatial Demo pipeline via `scripts/setup-astana-geospatial-demo.ps1`
- Stack: `docker compose up -d --build` → http://localhost

**Demo login:** `thesis.demo@flowmatic.local` / `ThesisDemo2025!`

---

## 2. What Flowmatic is

Production smart-city + data-prep platform:

- **Batch path:** CSV/JSON upload → quality → clean → export (Postgres, Mongo, HF, files).
- **Smart City path:** sensor sources → **adaptive core unit** (Manual/Auto model routing) → medallion lake → export → optional federated demo.
- **Insight Engine:** scheduled analysis, charts, copilot chat (`/insights`).
- **ML backbone:** PyTorch research under `models/`; inference via `model-inference` service; registry drives Auto routing.

**Stack:** NestJS (Fastify) + Vue 3 + PostgreSQL + MinIO + RabbitMQ + Bun backend; Python inference sidecar; Docker Compose.

---

## 3. Phased delivery state

### Phase 1 — Adaptive core unit ✅

| Component | Path |
|-----------|------|
| Model registry | `backend/src/modules/smart-city/pipeline-model-registry.service.ts` |
| Rule router | `backend/src/modules/smart-city/pipeline-model-router.service.ts` |
| Optional LLM policy | `pipeline-auto-routing.service.ts` |
| Workbench UI | `frontend/src/modules/smart-city/components/pipeline-processing-stage.vue` |
| Tests | `backend/src/modules/smart-city/__tests__/pipeline-model-router.spec.ts` |

**Requires for Auto mode:** `./models:/app/models:ro` on backend + `models/reports/production_portfolio.json`.

### Phase 2 — Portfolio & datasets ✅

**Run:** `python models/paper/run_phase2_prep.py`

### Phase 3 — Experiments (partial) ⚠️

**Run:** `python models/paper/run_phase3_core.py`

**Not done** (do not claim as answered in thesis):

- Auto router vs oracle ablation (**RQ4 gap** — see `paper-review-q1.md` Weakness 2)
- Astana → PeMS/METR-LA external holdout
- CRPS, AUROC, calibration metrics

**Evaluation protocol (neural):** 70% train / 15% val / **15% temporal test**; seeds `{42, 7, 2026}`.

### Phase 4 — Thesis ✅ draft / ⚠️ revisions before final submit

| Item | Status |
|------|--------|
| AITU template 6 chapters + appendix | ✅ In repo |
| Compiled PDF (~50 pp) | ✅ `memoirthesis-flowmatic.pdf` |
| Citations, cross-refs, TOC/LOF/LOT | ✅ Fixed |
| UI screenshots in Ch. 4 | ✅ Committed |
| Q1 review document | ✅ `paper-review-q1.md` |
| Classical baseline reproducibility | ❌ Numbers cited from prior manuscript; no script in repo |
| RQ4 routing metrics | ❌ Implemented in UI only; not quantified |
| Legacy XGBoost / archival narrative | ✅ Removed entirely; thesis = Flowmatic + HF portfolio only |
| Stale text in conclusion §6.4 (screenshots pending) | ✅ Removed |
| Tier 0–1 thesis narrative (two tracks) | ✅ AITU Template updated May 2026 |

**Legacy sources (keep for reference, not primary build):**

- `thesis/latex/` — Phase 4 snippet sources
- `scripts/build-thesis-phase4.ps1` — old build path

---

## 4. Key experiment numbers

From `models/paper/tables/multi_seed_aggregate.md` (n=3 seeds, held-out test):

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

**Production classifier:** Transformer severity **0.911 ± 0.017** macro-F1 (HF, `run_phase3_core.py`). No legacy tabular baselines in thesis text.

**Platform batch experiment (Table 4.1):** 30,000 records, 406 ms processing, quality 100/100.

---

## 5. Repository map (what matters)

```
flowmatic/
├── HANDOFF.md                         ← this file
├── paper-review-q1.md                 ← Q1/defense review (2026-05-27)
├── memoirthesis-flowmatic.pdf         ← compiled thesis (laptop build)
├── scripts/
│   ├── build-thesis-laptop.ps1        ← Docker TeX build (USE THIS)
│   ├── render-thesis-figures.ps1      ← Mermaid → PNG
│   ├── setup-astana-geospatial-demo.ps1
│   ├── generate-astana-dataset.js
│   └── validate-thesis-references.ps1
├── thesis/
│   ├── AITU Thesis Template/          ← PRIMARY LaTeX (memoirthesis.tex)
│   ├── figures/
│   │   ├── architecture/*.png
│   │   ├── results/*.png
│   │   └── screenshots/*.png          ← UI captures for Ch. 4
│   ├── GAP_ANALYSIS.md
│   └── latex/                         ← legacy Phase 4 snippets
├── backend/                           ← NestJS + smart-city module
├── frontend/                          ← Vue SPA
├── models/                            ← ML scripts, configs, reports
├── services/                          ← sensor-simulator, model-inference
└── data/                              ← GITIGNORED — generate locally
```

---

## 6. Commands cheat sheet

### Platform (PC or laptop)

```bash
cp .env.example .env
cp backend/.env.example backend/.env   # if missing
docker compose up -d --build
# UI: http://localhost  |  API: http://localhost:8080/api/v1
```

**Generate Astana CSV (if `data/` empty):**

```bash
node scripts/generate-astana-dataset.js
```

**Astana Geospatial Demo pipeline:**

```powershell
.\scripts\setup-astana-geospatial-demo.ps1
```

### Thesis build

```powershell
.\scripts\build-thesis-laptop.ps1
```

Requires Docker (uses `texlive/texlive:latest`). Syncs figures from `thesis/figures/` into template tree.

### ML / experiments

```bash
python models/download_checkpoints_from_hf.py --skip-existing
python models/paper/run_phase2_prep.py
python models/paper/run_phase3_core.py
```

### Backend tests

```bash
cd backend && bun run test -- src/modules/smart-city/__tests__/pipeline-model-router.spec.ts
```

---

## 7. Hugging Face models

**Hub user:** `@pushthetempo`  
**Manifest:** `models/reports/huggingface_model_manifest.json`

Checkpoints are **not in Git** — restore with `python models/download_checkpoints_from_hf.py --skip-existing` and `HF_TOKEN` in `.env`.

---

## 8. Recommended next work (priority order)

From `paper-review-q1.md` — do these on **desktop PC** before antiplagiarism submission:

1. **Critical:** Add RQ4 routing evaluation table OR reframe RQ4 as implementation-only in conclusions.
2. **Critical:** Harmonize 96.83% vs 96.62% classifier numbers in abstract / Table 4.2 / §5.3.
3. **Critical:** Add classical baseline reproduction script OR explicit "prior study" disclaimer with frozen artifact.
4. **High:** Split Table 4.2 by task; remove suspicious 1.88 GB/s throughput row or measure properly.
5. **High:** Remove stale "screenshots will be added" from `conclusion.tex` §6.4.
6. **Medium:** Discuss iTransformer RMSE ≈ 1.0 failure in Results/Discussion.

### Platform (optional)

- Re-run demo screenshots after UI changes
- Router ablation experiment for thesis Table 5.X

---

## 9. Environment secrets (never commit)

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | HF download/upload, model catalogue |
| `OPENAI_API_KEY` | Upload summaries, insights copilot, optional Auto policy LLM |
| `JWT_SECRET` | Auth |
| Postgres/MinIO/RabbitMQ passwords | Infra |

---

## 10. Git / what is committed vs ignored

**Committed in this push:**

- Fresh `HANDOFF.md`, `paper-review-q1.md`
- Full `thesis/AITU Thesis Template/` LaTeX sources + `thesisbiblio.bib`
- `thesis/figures/` (architecture, results, screenshots)
- `memoirthesis-flowmatic.pdf` (compiled thesis)
- Build scripts (`build-thesis-laptop.ps1`, etc.)
- LaTeX build aux files may be present — safe to delete locally; rebuild regenerates

**Not committed (`.gitignore`):**

- `data/`, `models/checkpoints/`, `models/datasets/`
- `.env`, `backend/.env`
- Old rejected drafts: `memoirthesis-v5-phase4.pdf`, `memoirthesis-v5-condensed.pdf`

---

## 11. Continue on desktop PC

```bash
git clone https://github.com/maulerrr/flowmatic.git
cd flowmatic
git pull   # if already cloned
```

Then:

1. Read this file + `paper-review-q1.md`
2. Open `memoirthesis-flowmatic.pdf` — verify formatting
3. `cp .env.example .env` and configure secrets
4. `docker compose up -d --build`
5. Pick a critical fix from §8 and continue thesis polish

**Old desktop thesis path** (`dissertation_latex_v5`) is superseded by `thesis/AITU Thesis Template/` unless user explicitly merges content back.

---

## 12. Contact context for next agent

- User goal: submit thesis to **antiplagiarism** soon, then defense
- Author: **Ramazan Bakytuly**; supervisor: **Aivar Sakhipov Aituarovich**; program **7M06105**; School of Software Engineering, Astana IT University
- Thesis title: *Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems*
- Work spans **classical preparation study** + **Flowmatic platform** + **neural portfolio** — dual narrative; see Table 4.4 and Q1 review
- Models on HF to avoid GitHub 100MB limit
- Laptop session completed PDF build + review; PC session should focus on **critical revisions** in §8, not re-scaffolding

**Start here:** `HANDOFF.md` → `paper-review-q1.md` → `memoirthesis-flowmatic.pdf` → user picks next fix.
