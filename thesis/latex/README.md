# LaTeX integration guide

Paste-ready blocks for updating the memoir thesis source (v5 condensed or v6).

## Files

| File | Contents |
|------|----------|
| `chapters/chapter01_addendum.tex` | Revised objectives and research questions |
| `chapters/chapter03_methodology.tex` | Methodology (classical + adaptive routing) |
| `chapters/chapter04_implementation.tex` | Implementation chapter |
| `chapters/chapter05_results.tex` | Results and discussion |
| `chapters/chapter06_conclusion.tex` | Conclusion |
| `figures.tex` | Architecture + UI + neural result figures |
| `figures_phase4.tex` | Figures used via `\input` in Chapter 5 |
| `tables.tex` | Upload experiment, classical vs neural, demo tables |
| `tables_phase4.tex` | Dataset, split, multi-seed, ablation tables |
| `frontmatter/abstract.tex` | Revised abstract |

## Usage

```latex
\input{thesis/latex/figures}
\input{thesis/latex/tables}
```

## Render architecture figures

From repo root:

```powershell
.\scripts\render-thesis-figures.ps1
```

Sources: `thesis/figures/architecture/*.mmd` → PNG via Docker `minlag/mermaid-cli`.

## Build full PDF (external memoir project)

```powershell
.\scripts\build-thesis-phase4.ps1
```

Requires memoir project at `C:\Users\BG\Desktop\master-thesis\dissertation_latex_v5\`.

## Figure mapping

| Manuscript figure | Asset |
|-------------------|-------|
| Batch architecture (Fig. 4.1) | `01-batch-upload-pipeline.png` |
| Smart-city pipeline (Fig. 4.2) | `02-smart-city-streaming.png` |
| Deployment topology | `03-microservices-topology.png` |
| Medallion lake | `04-medallion-data-lake.png` |
| UI screenshots | `figures/screenshots/*.png` (capture after platform run) |
| Neural results | `figures/results/*.png` |

## Reproducing experiments

```powershell
docker compose up -d
python thesis/experiments/_run_upload_experiment.py
python thesis/experiments/_setup_smart_city_pipeline.py
```

Use `http://localhost/api/v1` (port 80 via Nginx), not `127.0.0.1:8080` on Windows.
