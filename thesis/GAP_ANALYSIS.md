# Thesis Manuscript versus Flowmatic Codebase — Gap Analysis

This note compares **memoir thesis v5 (condensed)**—*Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems* (Ramazan, Astana IT University, June 2025)—with the **current Flowmatic repository** as of the supplement authoring date. The manuscript emphasizes a **Python research stack** (pandas, scikit-learn, XGBoost, SHAP) and **tabular traffic preparation** with classical classifiers; the repository has evolved into a **full-stack data-preparation platform** with **Neural Q1 artifacts**, **Dockerized services**, and a **Smart City workbench** that partially instantiates the thesis narrative but changes several implementation assumptions.

## Concise comparison table

| Topic | Thesis manuscript (PDF) | Flowmatic codebase / ops |
| --- | --- | --- |
| Core runtime | Python scripts, reproducible experiment tables; batch and “streaming” as *design* | **NestJS 11** backend, **Vue** SPA, **Prisma** on PostgreSQL, **typed configuration** (`AppConfigService`) |
| Figures 4.1–4.2 (batch/stream) | High-level modular pipeline + **Kafka** topic graph (`traffic.raw` → … → `traffic.predictions`) | **Operational diagrams** aligned with ingestion → queue → worker → storage; streaming path implemented via **HTTP sensor simulator**, **persistent events**, optional **Boss/PgBoss** + **`smart-city-stage-export`**, not Kafka-first |
| Classification / ML story | Archival offline pilot: **XGBoost** (0.9683 accuracy, June 2025 memoir; not in repo/HF) | **Production track**: HF Transformer severity checkpoints; **tabular upload pipeline** does QC/clean only—no XGBoost in Nest |
| Anomaly detection | **Ensemble**: Isolation Forest, LOF, One-Class SVM | Not mirrored as a sklearn ensemble inside the Nest cleaning path; anomalies addressed in neural suite (e.g., **TranAD**) and platform QC (outliers via **Z-score** in `PROJECT_GUIDE`) |
| Feature engineering | Domain-specific temporal/spatial/traffic aggregates (16 derived from 9 attributes) | **Generic** profiling/clean for CSV/JSON; Smart City pipelines add **research models** & lake-oriented outputs rather than thesis FE table |
| Data quality metrics | Six-dimension composite **DQI** (chapter 5 tables) | **Practical QA**: completeness, duplicates, column typing, **Z-score** outliers (`QualityService`), LLM narrative summary |
| Explainability | **SHAP** for classifier attribution | Summaries via LLM; no SHAP path wired into production API described in guide |
| Datasets emphasized | Semi-synthetic **Astana** + benchmarks **METR-LA**, **PEMS-BAY**, **PeMSD4** | Neural leaderboard adds **HF-derived** bundles (`hf_weather`, `hf_ett`, `hf_traffic`); ingestion platform is dataset-agnostic |
| Export / deployment | Thesis positions streaming for smart-city deployments; acknowledges prototype scope | **Multi-adapter export** (PostgreSQL, MongoDB, Hugging Face, files), **`model-inference`** microservice, **docker-compose** stack |
| Auth & multi-tenancy | Outside thesis core | **Organizations**, **roles**, cookie sessions (`PROJECT_GUIDE`); HF token per org |

## Interpretation for examiners

The manuscript remains a coherent **methods-and-evaluation thesis** grounded in classical ML and a Kafka-oriented architectural sketch. Flowmatic demonstrates **engineering closure**: the same scientific intent—automating preparation and bridging batch to operational use—is realized with **different middleware choices**, a **typed service architecture**, and a **parallel neural benchmarking line** absent from the original evaluation chapter. Updating Chapter 4 figures and extending results with **`checkpoint_leaderboard.md`**/`Q1` artifacts closes the scholarly loop without rewriting the foundational motivation.
