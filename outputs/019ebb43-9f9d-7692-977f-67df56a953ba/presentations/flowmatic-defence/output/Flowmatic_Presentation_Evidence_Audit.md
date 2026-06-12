# Flowmatic Presentation Evidence Audit

This audit was built from thesis v3, generated experiment artifacts, code modules, review comments, and the official recommendations/template inventory.

| Slide | Claim | Source | Verified against code | Verified against thesis | Limitation | Status |
| 1 | Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems | Thesis v3 title page; official AITU template | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 2 | Dirty transport data breaks the path from operations to analytics. | Thesis v3 Sec. 1.1; recommendations: problem -> method -> result | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 3 | The gap is measurement linkage, not a claim that no tools exist. | Thesis v3 Sec. 2.12-2.17; review Q1 risk S03/S07 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 4 | The aim is implemented through six objectives and four scoped questions. | Thesis v3 Sec. 1.2, 3.3, 5.9 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 5 | The scientific claim is the evaluated chain, not the existence of the platform. | Thesis v3 Sec. 1.3, 2.13.1, Appendix A; review W1 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 6 | The research workflow tests the artefact under controlled, reproducible conditions. | Thesis v3 Ch. 3, Ch. 5; recommendations: research pipeline | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 7 | The evidence is reproducible, but not all data are externally real-world validated. | Thesis v3 Sec. 3.4, 5.2.1, Appendix C | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 8 | DQI makes preparation auditable before data reach models. | Thesis v3 Sec. 3.6-3.7; quality.service.ts; cleaning.service.ts | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 9 | Flowmatic connects auditable preparation with task-specific inference. | docker-compose.yml; Prisma schema; backend smart-city/model registry modules | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 10 | The system is intelligent as a bounded hybrid assistant, not as general AI. | pipeline-model-router.service.ts; model registry; thesis v3 Sec. 3.11 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 11 | The experiment design separates leakage control from statistical strength. | Thesis v3 Sec. 5.2.2; review W3 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 12 | Preparation recovered measurable quality, partly by rejecting invalid records. | thesis_v2_evidence.json; generated_tables.tex | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 13 | Downstream results support capability, with explicit task limits. | multi_seed_aggregate.md; generated_baselines.tex; thesis_v2_evidence.json | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 14 | Preparation improved forecasting under a controlled upper-bound experiment. | thesis_v2_evidence.json; generated_prep_ablation.tex | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 15 | The prototype works, but the measured boundary matters. | UPLOAD_PIPELINE_REPORT.md; streaming_benchmark.csv; thesis screenshots | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 16 | Each research question has evidence and an explicit scope limit. | Thesis v3 Table 5.19 / Sec. 5.9 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 17 | My contribution is the integrated implementation and thesis-new evaluation package. | Thesis v3 Appendix A; review W1/S05 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 18 | Flowmatic is an evaluated research prototype for auditable transport-data preparation. | Thesis v3 Ch. 6; review final verdict | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Scoped in slide notes | Included |
| 20 | DQI definitions and formula | Thesis Sec. 3.6 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 21 | Clean 100/100 vs stress-test 0.833 | Table 4.1; Table 5.9; evidence JSON | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 22 | Dataset facts | Appendix C; Sec. 5.2.1 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 23 | Split, leakage, uncertainty | Sec. 5.2.2 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 24 | Baseline table | generated_baselines.tex | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 25 | Neural portfolio | production_portfolio.json | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 26 | iTransformer interpretation | generated_itransformer_retrain.tex | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 27 | Routing scope | generated_tables.tex | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 28 | Oracle restoration ablation | generated_prep_ablation.tex | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 29 | Prior publication vs thesis-new | Appendix A; Sec. 2.13.1 | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 30 | Personal contribution matrix | Appendix A; repository evidence | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 31 | Technology roles | docker-compose.yml; Prisma schema | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 32 | Codebase/module map | repo map | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 33 | Failure handling limits | code audit | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 34 | Security and production roadmap | Prisma schema; security middleware | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |
| 35 | Reproducibility and exact numbers | evidence JSON; tables | Yes - repo/code audited where applicable | Yes - v3 thesis and generated tables | Backup-level detail | Included |

## Unresolved / deliberately excluded claims

- No live municipal deployment, production SLA, HA, disaster recovery, penetration test, or operator study is claimed.
- LLM narration is treated as optional and unevaluated, not as the basis for the intelligent-assistant claim.
- Routing is framed as policy conformance, not expert-labelled routing accuracy.
- Preparation ablation is labelled as oracle-restoration upper-bound evidence.
- Three-seed values are treated as descriptive variability, not strong statistical significance.
- DQI clean 100/100 and stress-test 0.833 are presented as different scoring contexts.
