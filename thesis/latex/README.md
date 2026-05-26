# LaTeX integration guide

Paste-ready blocks for updating `memoirthesis-v5-condensed` (or a v6 LaTeX source).

## Files

| File | Contents |
|------|----------|
| `figures.tex` | `\includegraphics` blocks for architecture, UI, and neural result figures |
| `tables.tex` | Upload experiment, classical vs neural, smart-city demo, gap analysis |

## Usage

```latex
\input{thesis/latex/figures}
\input{thesis/latex/tables}
```

Adjust paths if figures live under a different directory relative to your `.tex` root. Recommended layout:

```
thesis/
  figures/
    architecture/*.png
    screenshots/*.png
    results/*.png
  latex/
    figures.tex
    tables.tex
```

## Packages required

```latex
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{float}
\usepackage{booktabs}
```

## Figure mapping (manuscript update)

| Old | New replacement |
|-----|-----------------|
| Fig. 4.1 Batch architecture | `01-batch-upload-pipeline.png` |
| Fig. 4.2 Streaming architecture | `02-smart-city-streaming.png` + `08-smart-city-live-workbench.png` |
| (new) | `03-microservices-topology.png`, UI screenshots `00`–`10` |
| Ch. 5 neural results | `figures/results/*.png` + Table `tab:classical-vs-neural` |

## Reproducing experiments

```powershell
docker compose up -d
python thesis/experiments/_run_upload_experiment.py
python thesis/experiments/_setup_smart_city_pipeline.py
```

Use `http://localhost/api/v1` (port 80 via Nginx), **not** `127.0.0.1:8080` — another process may bind 8080 on Windows.
