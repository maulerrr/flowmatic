# Q1 Dissertation Reference

## Q1 indexation — what it means

**Q1** (Clarivate JCR / Scimago SJR Q1, or equivalent top quartile) implies:
- Rigorous peer review with high rejection rates
- Expectation of **clear novelty** and **field impact**
- Reproducibility and methodological transparency
- Positioning against strongest baselines, not strawmen

Master's theses rarely publish directly in Q1 journals, but **Q1-level rigor** means the work could be extended into a journal submission with ≤6 months additional experiments.

## STEM CS/ML chapter specifications

### Abstract (150–300 words)
- Problem (1–2 sentences)
- Method (1–2 sentences)
- Key results with **numbers**
- Contribution statement
- **Cross-check:** every number in abstract appears identically in body

### Chapter 1 — Introduction
- Context and motivation (why now, why important)
- Problem statement
- Research questions / objectives
- Contributions (bulleted, verifiable)
- Thesis structure roadmap

### Chapter 2 — Literature Review
- Thematic organization (not chronological-only)
- Seminal works + recent advances (≤3 years)
- Comparative table optional: approach × dataset × metric × limitation
- Explicit gap paragraph ending with "Therefore, this thesis..."

### Chapter 3 — Methodology
- Research design and philosophy ( positivist / design-science for systems theses)
- Dataset description: source, size, splits, ethics, preprocessing
- Model/algorithm specification with notation
- Baseline selection **justification**
- Evaluation metrics and statistical tests
- Threats to validity subsection
- Reproducibility statement (configs, seeds, code location)

### Chapter 4 — Implementation / System Design
- Architecture with verified diagrams
- Design decisions tied to requirements
- Technology choices justified (not "we used X because popular")
- Deployment and operational constraints

### Chapter 5 — Results and Discussion
- Results per RQ, in RQ order
- Tables/figures referenced before prose
- Discussion interprets; does not repeat numbers
- Comparison to baselines and prior published work
- Ablation or sensitivity analysis
- Negative results

### Chapter 6 — Conclusion
- Summary per contribution (not copy-paste abstract)
- Limitations (honest, specific)
- Future work (actionable, not generic)

## Experimental reporting checklist

```
Pre-experiment:
- [ ] Hypothesis pre-registered in methodology
- [ ] Baselines identified with citation
- [ ] Hyperparameter budget equal or justified
- [ ] Data splits fixed and saved (seed documented)

Execution:
- [ ] ≥3 random seeds for stochastic methods
- [ ] Hardware logged (GPU model, batch size)
- [ ] Failed runs logged (not discarded silently)

Reporting:
- [ ] Mean ± std or median [IQR] reported
- [ ] Effect size or practical significance discussed
- [ ] Confidence intervals where appropriate
- [ ] All tables have units and decimal precision consistent
- [ ] Leaderboard includes parameter count / latency if systems claim
```

## Literature matrix columns

| Column | Content |
|--------|---------|
| ID | L1, L2, ... |
| Authors (Year) | Full citation key |
| Problem | One line |
| Method | One line |
| Data | Datasets used |
| Metrics | Primary metrics |
| Result | Best reported number |
| Limitation | Author-stated or your assessment |
| Relevance | High/Med/Low + which RQ |
| Gap link | How it relates to your gap |

## Venue positioning (CS/ML adjacent)

When targeting Q1 extension, map thesis chapters to journal expectations:

| Venue type | Extra expectations beyond thesis |
|------------|----------------------------------|
| Systems (e.g., TOS, TPDS) | Scale, failure modes, deployment metrics |
| ML (e.g., JMLR, ML) | Theory or strong empirical + ablations |
| Applied AI (e.g., KBS, EAAI) | Real-world dataset, case study depth |
| Transport/ITS journals | Domain validation, policy relevance |

Search: `"[venue name]" aims and scope` before claiming fit.

## AITU / institutional alignment

This repo uses Astana IT University memoir template under `thesis/AITU Thesis Template*/`.

Institutional requirements typically include:
- Declaration, abstract (Kazakh/Russian/English per guidelines)
- Methodological consistency with approved research plan
- Plagiarism originality threshold (verify current university policy via official PDF)

Consult: `thesis/Appendix 16 Methodological Guidelines...pdf` for local requirements.

## LaTeX and figure hygiene

- Vector figures preferred (SVG/PDF); PNG ≥300 DPI for raster
- Mermaid sources: `thesis/figures/architecture/*.mmd`
- Render: `.\scripts\render-thesis-figures.ps1`
- No Unicode artifacts in PDF (verify with `pdftotext` or visual pass)
- `\ref{}` and `\cite{}` must compile without warnings

## Citation quality bar

- Primary sources > surveys > blogs
- arXiv acceptable if no peer-reviewed version exists; note preprint status
- Self-citation ≤15% of bibliography unless meta-analysis
- Every figure/table adapted from prior work: cite in caption

## Red flags that block Q1-level acceptance

- Single-seed neural results presented as definitive
- Missing baseline or unfair tuning
- Claims of "real-time" without latency methodology
- Architecture figure contradicts code
- Abstract numbers ≠ body numbers
- Related work missing obvious competitors in subfield
- No limitations section
- Reproducibility artifacts absent when code/data can be shared
