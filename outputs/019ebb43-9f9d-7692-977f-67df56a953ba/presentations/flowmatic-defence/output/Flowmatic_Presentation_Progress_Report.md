# Flowmatic Presentation Progress Report

## Task

Create a final master's thesis defence presentation for:

**Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems**

The implemented platform is **Flowmatic**. The presentation had to follow the official AITU master's presentation template and recommendations, while also pre-empting weaknesses from the Q1-style review and defence question bank.

## Inputs Used

- Official presentation template: `C:\Users\BG\Downloads\Telegram Desktop\Шаблон_презентации_магистранта_1.pptx`
- Presentation recommendations: `C:\Users\BG\Downloads\Telegram Desktop\Recoms for Masters (копия).pdf`
- Defence question bank: `C:\Users\BG\Downloads\MASTER_THESIS_DEFENCE_QUESTIONS.md`
- Q1-style review: `C:\Users\BG\Downloads\paper-review-q1.md`
- Active thesis source: `C:\Users\BG\Desktop\flowmatic\thesis\Flowmatic-Thesis-v3`
- Compiled thesis PDF: `C:\Users\BG\Desktop\flowmatic\memoirthesis-flowmatic-v3.pdf`
- Flowmatic codebase: backend, frontend, services, Docker Compose, model reports, generated experiment artifacts

## Discovery Work

I checked the current Flowmatic repository instead of relying only on previous context. The audit covered:

- Docker topology: PostgreSQL, MinIO, RabbitMQ, backend, frontend, sensor simulator, model-inference service, federated demo
- Backend modules: ingestion, quality, cleaning, export, storage, auth, smart-city model registry, smart-city router
- Prisma schema: users, organisations, pipeline runs, storage files, smart-city pipelines, sensor events, model artifacts, insight runs
- Frontend evidence: upload flow, smart-city workbench, workflow graph screenshots
- Model artifacts: production portfolio, Hugging Face manifest, multi-seed aggregate metrics, streaming benchmark
- Thesis v3 LaTeX and generated tables
- Review and question-bank defence risks

## Key Evidence Confirmed

- Active thesis version is v3 under `thesis\Flowmatic-Thesis-v3`.
- Official template uses 16:9 wide slide size: `12192000 x 6858000 EMU`.
- Clean Astana batch upload is a platform/latency result, not the main quality-improvement result.
- 20% corruption stress test:
  - DQI before: `0.788`
  - DQI after: `0.833`
  - Delta: `+0.045`
  - Dropped rows: `987 / 10,000`
- Preparation ablation:
  - PatchTST prep-off RMSE: `0.9045`
  - PatchTST prep-on RMSE: `0.5802`
  - Delta: `0.3243`
  - Framed as oracle-restoration upper-bound, not deployable imputation performance.
- Severity classification:
  - Transformer macro-F1: `0.911`
  - Baselines: majority `0.317`, logistic regression `0.449`, random forest `0.627`
  - Framed as rule-label recovery, not verified real incident-severity prediction.
- TranAD:
  - Injected-anomaly AUROC: `0.8411`
  - LOF baseline AUROC: `0.6607`
  - Framed as injected-window evidence, not municipal incident validation.
- Routing:
  - `140 / 140` simulator events matched scenario-aligned policy oracles.
  - Framed as policy conformance / implementation verification, not independent routing accuracy.

## Defence Strategy Applied

The deck was structured to answer the commission's likely questions before they are asked:

- What is scientifically new?
- Is this research or only software development?
- Why is the system called intelligent?
- What is your personal contribution?
- What is new compared with prior publications?
- Why trust semi-synthetic data?
- Did severity classification just learn a rule?
- Is routing conformance the same as accuracy?
- Did the ablation use pristine reference values?
- Are the reported timings production SLAs?
- Is Flowmatic production-ready?

The main framing is:

**Flowmatic is a functioning research prototype. The scientific value is not a new neural architecture or a new universal DQI theory; it is the measured, reproducible preparation-to-downstream evaluation chain.**

## Deck Structure Produced

- 18 main defence slides
- 17 backup slides, including the backup divider
- 35 total slides
- Speaker notes embedded for every slide
- Separate defence script created
- PDF preview exported directly from the final PPTX through PowerPoint COM

Main-deck arc:

1. Title
2. Operational problem
3. Existing solutions, limitations, gap, contribution
4. Aim, object, subject, RQs
5. Scientific novelty and contribution boundaries
6. Research methodology
7. Data and validation scope
8. DQI and preparation method
9. Flowmatic architecture
10. Why it is an intelligent assistant
11. Experimental protocol
12. DQI stress-test result
13. Downstream model evidence
14. Preparation-to-forecasting ablation
15. Working implementation and timing boundaries
16. RQ closure
17. Personal contribution and publications
18. Conclusions, limitations, next steps

Backup slides cover DQI definitions, DQI reconciliation, dataset facts, split/leakage/statistics, baselines, model portfolio, iTransformer weakness, routing scope, oracle-restoration caveat, prior-publication overlap, personal contribution, technology roles, module map, failure handling, production/security roadmap, and exact numbers.

## How The PPTX Was Built

The presentation was generated programmatically using the bundled Node runtime and `pptxgenjs`.

Reason:

- The official Presentations helper expected a Unix `unzip` binary that was not available in the Windows workspace.
- `python-pptx` was not installed in the bundled Python environment.
- LibreOffice was not available in PATH.
- PowerPoint COM was available and was used for final PDF export and render verification.

The generated deck preserves the official template requirements by using:

- the same 16:9 wide dimensions;
- AITU logo;
- academic blue/white visual identity;
- required AITU section flow;
- title, aim/objectives, hypothesis/RQ-equivalent, object/subject, novelty, methods/data, architecture, results, conclusions, and publications sections.

## Deliverables Created

All final files are in:

`C:\Users\BG\Desktop\flowmatic\outputs\019ebb43-9f9d-7692-977f-67df56a953ba\presentations\flowmatic-defence\output`

Files:

- `Flowmatic_Master_Thesis_Defence_Final.pptx`
- `Flowmatic_Master_Thesis_Defence_Final.pdf`
- `Flowmatic_Defence_Slide_Plan.md`
- `Flowmatic_Presentation_Evidence_Audit.md`
- `Flowmatic_Defence_Script.md`
- `Flowmatic_Presentation_QA.md`

This progress report was added afterward:

- `Flowmatic_Presentation_Progress_Report.md`

## Verification Performed

Package-level verification:

- PPTX slides: `35`
- Speaker-note pages: `35`
- Media files: `38`
- Empty media files: `0`
- PDF pages: `35`

Visual verification:

- Exported the final PPTX to PDF via PowerPoint COM.
- Exported rendered slide PNGs through PowerPoint.
- Built and inspected a contact sheet of all 35 rendered slides.
- Confirmed the main deck has readable, varied layouts and backup slides are dense but usable.

Content verification:

- Values were cross-checked against thesis v3 generated tables and `thesis_v2_evidence.json`.
- Code-backed claims were checked against the relevant backend services, Docker Compose, Prisma schema, and model registry/router files.
- Review risks were mapped into main-slide framing or backup-slide answers.

## Important Caveats

- The deck is strong for defence framing, but it has not been live-rehearsed for exact timing.
- The timing estimate is `10-12 minutes` for the 18-slide main deck.
- The official template was followed in dimensions, identity, and section structure, but the final slide layouts were rebuilt programmatically for clarity and evidence density rather than edited manually inside the original PowerPoint file.
- Claims deliberately excluded:
  - live municipal deployment;
  - production SLA;
  - high availability;
  - penetration-tested security;
  - independently labelled routing accuracy;
  - real-world incident severity prediction;
  - deployable imputation performance equal to the oracle ablation.

## Final State

The presentation package is ready for review and rehearsal. It is designed to sell the work confidently while avoiding the dangerous overclaims identified in the review and question bank.
