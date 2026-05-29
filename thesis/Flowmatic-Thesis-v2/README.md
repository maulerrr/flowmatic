# Flowmatic Thesis v2 (submission manuscript)

**Output PDF:** `memoirthesis-flowmatic-v2.pdf` at repository root.

The original AITU template under `thesis/AITU Thesis Template/` is **not modified**.

## University template

This build uses the full AITU `memoir` driver: trimmed A4 layout, custom headers, roman front matter, list of tables/figures, theorem environments, and `unsrt` bibliography.

## Build

```powershell
.\scripts\build-thesis-v2.ps1
```

## Content (May 2026)

- **57 bibliography entries** (verified `.bib` sources; clickable DOI/URL links via `hyperref`)
- **Expanded literature review** (Ch.~2) with comparative table
- **Baselines, corruption stress test, routing pilot** (Ch.~5)
- **Formal discussion** of scope boundaries, baselines, and validity threats (Ch.~5)
- **Scientific vs.\ engineering** scope (Ch.~1); platform vs.\ neural validation layers
- **Appendices:** reproducibility + Astana dataset construction

## Evidence scripts

`thesis/experiments/v2/run_thesis_v2_experiments.py` → `thesis_v2_evidence.json`, `generated_*.tex`
