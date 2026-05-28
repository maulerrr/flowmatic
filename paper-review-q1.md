# Q1-Level Academic Paper Review

**Manuscript reviewed:** *Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems* (Master's thesis, Astana IT University, 50 pp., compiled PDF `memoirthesis-flowmatic.pdf`; LaTeX source in `thesis/AITU Thesis Template/`).

**Review basis:** Full read of Abstract, Chapters 1–6, Appendix A, Tables 4.1–5.5, figure captions, and cross-check against repository artifacts (`models/configs/phase3_multiseed_suite.yaml`, `thesis/GAP_ANALYSIS.md`, smart-city routing services).

---

## 1. Executive Verdict

**Readiness level:** **Defense-ready with revisions** (as a master's thesis); **Not ready** (as a Q1 journal submission).

**Risk level:** **Medium** for thesis defense; **High** for journal submission without major restructuring.

The dissertation presents a credible engineering integration (Flowmatic platform + neural portfolio + reproducible scripts) and honestly flags several gaps. However, it still reads as **two partially merged studies**—a classical tabular preparation thesis and a separate neural/platform track—without a single unified experimental narrative. RQ4 is asserted but not measured. Classical headline numbers (96.83% / 96.62%) are not reproducible from the supplied artifact pipeline. A strict Q1 reviewer would reject for insufficient novelty isolation, missing baselines, and incomparable metrics across tasks.

For **thesis defense**, the work is defensible if the author can explain the manuscript–implementation evolution, justify semi-synthetic Astana use, and acknowledge that routing and classical baselines need sharper evidence.

---

## 2. Core Strengths

1. **Clear operational problem framing (Ch. 1, §1.1–1.2).** The motivation—preparation as a first-class, auditable step in UTMS—is well grounded in data-quality and ITS literature with appropriate citations (`wang1996beyond`, `pipino2002data`, `its_bigdata2018`).

2. **Structured research questions (Ch. 1, §1.4).** RQ1–RQ4 map to distinct evaluation layers in Ch. 5 (classical tabular, platform latency, neural portfolio, routing), which aids examiner navigation even when evidence is uneven.

3. **Focused literature review (Ch. 2).** The review is scoped to preparation, missing data, feature engineering, anomaly detection, deployment, and recent time-series models—not a generic ITS survey. Section 2.9 explicitly states what was actually completed.

4. **Documented neural evaluation protocol (Ch. 3, §3.5–3.6; Table 5.2).** Temporal 70/15/15 splits, test-only reporting, z-score fit on train only, and three seeds with CI aggregation are appropriate minimum discipline for stochastic deep models.

5. **Reproducibility artifacts (Ch. 5 overview; Appendix A; `phase3_multiseed_suite.yaml`).** Eight-model portfolio, Hugging Face publication, TorchScript latency figures, and a one-command reproduction path (`run_phase3_core.py`) exceed typical prototype-thesis standards.

6. **Implementation evidence (Ch. 4; Table 4.1; Figures 4.5–4.8).** Screenshots and measured batch upload latency (30k rows, 406 ms processing) demonstrate that the platform is real, not diagram-only.

7. **Explicit limitations (Ch. 6, §6.3; Ch. 5, §5.9).** The text admits missing router ablations, unused METR-LA/PEMS-BAY hold-outs, and unvalidated LLM explainability—this honesty helps defense credibility.

---

## 3. Critical Weaknesses

### Weakness 1: Two merged research lines without a unified scientific claim

**Problem:** The thesis combines (a) a classical Python/scikit-learn preparation study with XGBoost/ensemble baselines and (b) a NestJS/Vue Flowmatic platform with eight neural checkpoints. Table 4.4 (`tab:gap-analysis`) documents Kafka→WebSocket, SHAP→LLM, and RF/XGBoost→Transformer shifts, but the abstract and contributions list present them as one coherent system.

**Why it matters:** Examiners and Q1 reviewers will ask whether the scientific contribution is *preparation methodology*, *platform engineering*, or *neural benchmarking*. Without a single claim, novelty appears incremental on all three fronts.

**How to mitigate:** Add a **Contribution Matrix** table (1 page) mapping each contribution → evidence artifact → research question → limitation. Rewrite Abstract and §1.5 to lead with one sentence: *"Scientifically, we show X; engineering-wise, we deliver Y; we do not claim Z."* Move platform-only features (federated demo, copilot) to a clearly labeled engineering appendix.

**Priority:** Critical

---

### Weakness 2: RQ4 (routing) is implemented but not evaluated

**Problem:** Ch. 5, §5.9 states: *"Manual/Auto routing is implemented and demonstrated in the workbench (RQ4). Formal router-vs-oracle ablations … are not yet reported."* Algorithm 1 (Ch. 3) defines rule-based routing, but no table reports routing accuracy, task-match rate, latency impact, or comparison to an oracle that picks the best checkpoint per sensor profile.

**Why it matters:** RQ4 is listed as a research question and included in the summary conclusions (Ch. 5, §5.10 item 4) as if answered. This is the most likely **defense attack point**.

**How to mitigate:** Add **Table 5.X: Routing evaluation** with columns: sensor profile, gold task, Auto-selected model, Manual default, match (Y/N), inference latency. Run on ≥100 simulated events from the Astana Geospatial Demo. Report **profile-match accuracy** and **modality-safe selection rate**. If oracle comparison is infeasible before submission, downgrade RQ4 to an *implementation objective* and remove it from "answered" conclusions.

**Priority:** Critical

---

### Weakness 3: Classical preparation results are cited, not re-derived or linked to artifacts

**Problem:** Ch. 5, §5.3 reports 0.9683 accuracy, 0.7823 raw baseline, 0.9234 AutoML baseline, and anomaly F1 = 0.858 with the phrase *"The original dissertation compared…"* No script, notebook, split protocol, or table in the appendix reproduces these numbers. The repository's ingestion path does not re-run XGBoost (`thesis/GAP_ANALYSIS.md`, row on classification).

**Why it matters:** A thesis claiming reproducibility cannot treat its strongest classical claim as inherited prose. Antiplagiarism and examiners care less; **scientific auditability** cares greatly.

**How to mitigate:** Either (a) add `models/paper/run_classical_baselines.py` + **Table 5.X** with fold-wise CV, feature list, and class balance, or (b) reframe §5.3 as *"Prior pilot study (2024 manuscript)"* with a frozen CSV snapshot and commit hash, or (c) demote classical numbers to illustrative and center RQ1 on platform QC outcomes only. **Do not leave both 96.83% (abstract) and 96.62% (Table 4.2) unexplained.**

**Priority:** Critical

---

### Weakness 4: Internal metric inconsistency (XGBoost vs Random Forest; accuracy vs F1)

**Problem:** Abstract cites **XGBoost 96.83%** accuracy; Table 4.2 lists **Random Forest 96.62%** and **XGBoost 96.62%**; §5.3 cites **0.9683** for XGBoost. §5.8 acknowledges the discrepancy but does not resolve it. Macro-F1 for the neural classifier (0.911 ± 0.017 in abstract; 0.925 for seed 7 in Table 4.2) is reported without clarifying seed aggregation rules.

**Why it matters:** Numeric inconsistency undermines trust in all other tables. Q1 reviewers treat this as a red flag for sloppy experimentation.

**How to mitigate:** Pick **one canonical classifier story** for the abstract (recommend: XGBoost, since §5.3 and RQ1 emphasize it). Add a footnote to Table 4.2: *"RF and XGBoost differ by <0.3 pp on the same split; abstract reports XGBoost."* Report neural classification only as **multi-seed mean ± CI** everywhere, not seed-7 point estimates mixed with aggregates.

**Priority:** Critical

---

### Weakness 5: Neural portfolio lacks external baselines and SOTA comparison

**Problem:** Table 5.3 reports PatchTST, TranAD, STGCN, etc., on internal/HF bundles, but there is **no comparison** to published leaderboard numbers on METR-LA, PEMS-BAY, or standard ETT/Weather benchmarks. METR-LA and PEMS-BAY appear in Table 5.1 as ingested but unused. iTransformer speed forecasting RMSE = **0.999 ± 0.001** (Table 5.3) indicates near-failure but receives no discussion.

**Why it matters:** Training eight known architectures is **engineering replication**, not scientific novelty, unless positioned against fair baselines (ARIMA, naive seasonal, published STGCN/PatchTST numbers) on shared splits.

**How to mitigate:** Add **Table 5.X: External baselines** with at least: naive last-value, seasonal naive, DLinear/PatchTST numbers from cited papers on the same dataset slice. Add a **Failure analysis** subsection for iTransformer and high-RMSE slots. Run one METR-LA hold-out experiment or remove METR-LA from Table 5.1 until results exist.

**Priority:** High

---

### Weakness 6: Anomaly and classification metrics are incomparable across rows

**Problem:** Table 4.2 juxtaposes ensemble **anomaly rate 3.92%**, TranAD **test RMSE 0.0308**, PatchTST **RMSE 0.5593**, and upload **QC score 100/100** without a unified evaluation frame. TranAD uses reconstruction RMSE, not event-level AUROC/F1 against injected anomalies.

**Why it matters:** The table reads as a marketing dashboard, not a scientific comparison. Reviewers will reject composite tables that mix incompatible metrics.

**How to mitigate:** Split Table 4.2 into **Table A (tabular classification)**, **Table B (anomaly detection with AUROC/F1 at fixed FPR)**, **Table C (forecasting RMSE/MAE)**, **Table D (platform QC/latency)**. Inject labeled anomalies into Astana test windows and report **AUROC** for TranAD vs IF/LOF/OCSVM ensemble.

**Priority:** High

---

### Weakness 7: Dataset validity and external generalisation are weak

**Problem:** Primary case study is **semi-synthetic Astana** (Table 5.1). Cross-dataset transfer (Table 5.5) only covers HF ETT↔weather—not traffic graph or Astana→PeMS. Limitations acknowledge this, but the abstract still implies broad UTMS relevance.

**Why it matters:** Claims about "urban transportation management systems" require at least one real agency dataset or a widely used traffic benchmark with documented generalisation gap.

**How to mitigate:** Add **one real-world or standard benchmark result** (even a single METR-LA 15% test RMSE row). Qualify abstract language: *"semi-synthetic Astana case study"* in the first sentence. Expand §5.7.2 with Astana→hf_traffic transfer if PeMS experiments are not feasible.

**Priority:** High

---

### Weakness 8: Classical methodology is under-specified

**Problem:** DQI Equation (3.1) defines six dimensions but **no weights \(w_i\)**, thresholds, or validation against human labels appear. Feature selection ("mutual information and model-based importance," Ch. 2) is not operationalized in Ch. 3. Ensemble weights \(\beta_k\) in Equation (3.3) are unspecified. Tabular CV protocol (folds, stratification, leak prevention) is absent.

**Why it matters:** Without hyperparameter and weight disclosure, classical results cannot be replicated or defended under questioning.

**How to mitigate:** Add **Table 3.1: DQI dimension weights and formulas**; **Table 3.2: Classical experiment protocol** (train/val/test or CV, class distribution, hyperparameters for XGBoost/IF/LOF/OCSVM). Reference config file or appendix listing.

**Priority:** High

---

### Weakness 9: Training budget and statistical power are thin for deep models

**Problem:** `phase3_multiseed_suite.yaml` sets **`epochs: 8`**, **`batch_size: 128`**, **`lr: 0.0008`** for all eight architectures. Only **n = 3 seeds** feed the 95% CI. No learning-curve saturation analysis justifies 8 epochs (Figure 5.7 shows curves but interpretation is minimal).

**Why it matters:** Undertrained Transformers and wide CIs weaken claims about portfolio readiness. Q1 venues expect justification of compute budget or early-stopping criteria.

**How to mitigate:** Report **validation-loss convergence** per model; increase epochs until val loss plateaus for at least PatchTST and Transformer classifier, or cite early-stopping patience. Add **5–10 seeds** for the two headline models (severity classifier, TranAD) or bootstrap CIs over test windows.

**Priority:** Medium

---

### Weakness 10: Explainability claims outpace evidence

**Problem:** Ch. 3, §3.7 and Ch. 5, §5.9 replace SHAP with **LLM-generated pipeline summaries** without user study, faithfulness metric, or hallucination audit. Table 4.4 lists this as a deliberate design shift but no evaluation follows.

**Why it matters:** For transportation agencies, unvalidated LLM narration is a **liability**, not a contribution. Q1 reviewers in XAI will reject unsubstantiated replacement of SHAP.

**How to mitigate:** Either run a **small expert review** (n≥3 analysts rate summary correctness vs ground-truth pipeline metadata on 20 runs) or downgrade language to *"optional narrative prototype, not evaluated for factual fidelity."* Keep SHAP results for classical models in an appendix table.

**Priority:** Medium

---

### Weakness 11: Stale and suspicious operational metrics

**Problem:** Ch. 6, §6.4 still states *"Final UI screenshots … will be added after validation runs"* although Figures 4.5–4.8 already exist. Table 4.3 reports **throughput 1.88 GB/s** from a UI estimate without measurement methodology—likely erroneous for a demo simulator.

**Why it matters:** Stale future-work text signals incomplete revision; implausible throughput damages credibility.

**How to mitigate:** Delete outdated screenshot sentence in §6.4. Replace throughput with **events/s** from backend logs or remove the row. Add footnote on measurement environment (Docker host spec, single pipeline, date).

**Priority:** Medium

---

### Weakness 12: Scientific novelty is engineering integration, not a new method

**Problem:** DQI weighting, ensemble anomaly scoring, and rule-based routing (Algorithm 1) are standard compositions. Neural models are off-the-shelf architectures trained on prepared bundles. The novel element is **integration into Flowmatic**, which is valuable engineering but weak for Q1 novelty bars.

**Why it matters:** Journal submission requires a crisp *"what is new vs. AutoML prep + model registry + time-series zoo."*

**How to mitigate:** Reframe novelty as **(i) unified preparation-to-inference audit trail for UTMS**, **(ii) modality-safe registry routing policy**, **(iii) reproducible multi-task portfolio under one temporal protocol**. Support (ii) with the routing table from Weakness 2. Cite closest systems (Feurer Auto-sklearn, MLflow, BentoML, urban digital twin platforms) and state differences in a **Related Systems** subsection.

**Priority:** Medium

---

## 4. Fast Improvement Notes

| Fix | Location | Effort |
|-----|----------|--------|
| Harmonize 96.83% vs 96.62%; pick XGBoost as canonical | Abstract, Table 4.2, §5.8 | 30 min |
| Remove "screenshots will be added" from §6.4 | `conclusion.tex` | 5 min |
| Remove or justify 1.88 GB/s throughput | Table 4.3 | 15 min |
| Add one paragraph on iTransformer RMSE ≈ 1.0 failure | §5.5 or §5.9 | 20 min |
| Label Table 4.2 rows by **task** and split into two tables | `tables.tex` | 1 hr |
| Add DQI weights table with default \(w_i = 1/6\) if equal | Ch. 3 | 30 min |
| Cross-reference every figure in text before it appears | Ch. 4–5 | 1 hr |
| Ensure classical baseline script path or "prior work" disclaimer | Appendix A | 2 hr |
| Add routing match-rate table (even 50-event pilot) | Ch. 5 | 4 hr |

---

## 5. Scientific Novelty Assessment

**Current state:** Novelty is **implicit and fragmented**. The strongest defensible claim is an **integrated, reproducible UTMS preparation platform** that connects batch QC, streaming smart-city pipelines, and a multi-task neural registry—not a new learning algorithm.

**Weakness:** The text sometimes implies novelty from assembling known models (PatchTST, TranAD, SAITS) without outperforming published baselines or introducing a new preparation algorithm.

**How to strengthen:**

1. State the **gap** narrowly: *"Prior work evaluates preparation, forecasting, and anomaly detection separately; none provide auditable DQI-driven batch prep **and** modality-safe streaming inference with published artifacts."*

2. Support with **one ablation only you can run**: preparation OFF vs ON for downstream severity classification **within the same neural pipeline** (not cross-pipeline XGBoost vs Transformer).

3. Position neural portfolio as **operational capability evidence**, not SOTA chasing—unless METR-LA numbers are added.

---

## 6. Methodology Review

| Area | Assessment | Mitigation |
|------|------------|------------|
| **Datasets** | Astana semi-synthetic primary; HF bundles for weather/energy/graph; METR-LA/PEMS unused | Add one standard benchmark result or demote unused datasets from Table 5.1 |
| **Models** | Classical ensemble + XGBoost; eight neural architectures | Document hyperparameters; add naive/ARIMA baselines for forecasting slots |
| **Architecture** | Flowmatic microservices credible (Ch. 4) | Add deployment diagram with data-flow numbering matching §4.2–4.4 |
| **Experiments** | Three-layer structure is sound | Unify metrics; add routing eval; reproduce classical numbers |
| **Baselines** | Raw vs simple vs AutoML for classical; weak for neural | FeatureTools config details; published STGCN/PatchTST numbers on same slice |
| **Metrics** | RMSE, masked MSE, macro-F1, latency | Add AUROC for anomaly; MAPE/CRPS for forecasting; separate platform SLO metrics |
| **Reproducibility** | Strong for Phase 3 neural | Extend appendix with classical + Docker demo IDs + HF repo URLs |
| **Limitations** | Well written | Tie each limitation to a specific RQ impact statement |

---

## 7. Results and Evaluation Review

**Convincing elements:**
- Batch upload experiment (Table 4.1) is concrete and verifiable.
- Multi-seed aggregation (Table 5.3) with held-out temporal test is appropriate.
- Sequence-length ablation (Table 5.4) shows non-monotonic context benefit—scientifically interesting.
- Streaming latency figures (Figures 5.5–5.6) support deployment claims.

**Unconvincing or missing elements:**
- No statistical test for classical preparation gains (McNemar or paired bootstrap on F1).
- No routing quantitative results for RQ4.
- Cross-dataset transfer (Table 5.5) too narrow (2 architectures × 2 HF sets only).
- **Classification:** comparing 99.09% Transformer accuracy to 96.62% XGBoost when feature pipelines differ overstates neural superiority.
- **Anomaly:** reconstruction RMSE does not validate operational anomaly detection.
- **Forecasting:** PatchTST Astana RMSE 0.561 and iTransformer 0.999 need context (target scale, naive baseline).

**Suggested additions:**
1. Table: **Preparation ablation** (raw → DQI-guided → full FE) on **one** downstream task with identical splits.
2. Table: **Routing accuracy** on simulated multi-sensor demo.
3. Figure: **AUROC curves** TranAD vs ensemble on injected anomalies.
4. Row: **METR-LA or PeMS** test RMSE for STGCN or remove from dataset table.

---

## 8. Structure and Logic Review

| Section | Verdict | Notes |
|---------|---------|-------|
| **Abstract** | Needs revision | Strong overview but overclaims UTMS breadth; fix classifier number; qualify Astana as semi-synthetic |
| **Introduction** | Good | RQs and objectives align with content; scope/limitations upfront helps |
| **Literature** | Good | Focused; §2.9 honestly scopes contribution |
| **Methodology** | Uneven | Neural formulations solid; classical prep and DQI under-specified; figures referenced before definition (Fig 3.1 refs Ch. 4 figures) |
| **Implementation** | Strong for thesis | Good screenshots; gap table (4.4) is valuable but should move earlier as "design evolution" |
| **Results** | Fragmented | Three layers correct; conclusions overstate RQ4; classical section reads as legacy paste |
| **Discussion** | Adequate | Honest about gaps; needs failure analysis (iTransformer, high RMSE) |
| **Conclusion** | Needs cleanup | Stale future-work on screenshots; repeats numbers without reconciling inconsistencies |

**Flow issue:** Methodology references batch/smart-city figures that appear in Chapter 4—acceptable but forward references should note *"implementation figures in Chapter 4"* to avoid circular dependency feel.

---

## 9. Academic Style Review

**Promotional / unsupported phrasing:**
- *"clearly above raw-data and generic AutoML baselines"* (Abstract)—add effect size or confidence interval.
- *"Together these results answer RQ3"* (§5.5)—RQ3 asks *how models perform*, not whether they beat SOTA; soften to *"provide initial multi-task evidence."*
- *"registry-based routing enables modality-aware inference"* (§5.10)—change to *"implements"* until evaluated.

**Repetition:** Contributions in Ch. 1, findings in Ch. 6, and summary in Ch. 5 overlap heavily—condense one block.

**Representative rewrites:**

| Before | After |
|--------|-------|
| *"clearly above raw-data and generic AutoML baselines"* | *"improves accuracy by 18.6 pp over raw features and 4.5 pp over generic AutoML features on the same Astana split (Table X)."* |
| *"Together these results answer RQ3"* | *"These held-out test metrics provide a first multi-task comparison under a shared temporal protocol (RQ3); external SOTA comparison remains future work."* |
| *"The dissertation links classical preparation research with a working multimodal platform"* | *"The dissertation connects a classical preparation study with a separately engineered multimodal platform and documents where they diverge (Table 4.4)."* |

---

## 10. Reviewer Questions for Defense

### Novelty
- **Q:** What is scientifically new here beyond integrating existing models and services?
  **Prepare:** One-sentence gap statement; emphasize auditable preparation + modality-safe routing policy + reproducible artifact bundle, not new architecture invention.

- **Q:** Why are Kafka and SHAP in the literature but not in the final system?
  **Prepare:** Table 4.4 walkthrough; institutional DevOps and production constraints; do not claim equivalence without evidence.

### Dataset
- **Q:** Why should examiners trust semi-synthetic Astana data for UTMS claims?
  **Prepare:** Data generation script (`generate-astana-dataset.js`), column schema, known limitations; plan for PeMS/METR-LA validation.

- **Q:** METR-LA and PEMS-BAY are listed but not evaluated—why include them?
  **Prepare:** Ingestion readiness vs evaluation scope; commit to one benchmark row or remove from Table 5.1.

### Methods
- **Q:** What are the DQI weights and who validated the six dimensions?
  **Prepare:** Default weights, sensitivity analysis (even ±20% weight perturbation on one run), or cite `bieberstein2006data` composite index precedent.

- **Q:** How does Auto routing choose among compatible models—what is \(\pi(m \mid \mathbf{f})\)?
  **Prepare:** Walk through Algorithm 1 with a concrete sensor example from the Astana demo; show `pipeline-model-router.service.ts` priority rules.

### Architecture
- **Q:** Does the batch upload path actually run the classical XGBoost preparation pipeline?
  **Prepare:** Honest answer: QC/cleaning yes, XGBoost training no—neural inference is separate; classical results are from prior offline study.

### Evaluation
- **Q:** Why compare TranAD RMSE to ensemble anomaly rate in one table?
  **Prepare:** Acknowledge metric mismatch; propose split tables and AUROC-based anomaly evaluation.

- **Q:** iTransformer RMSE is 0.999—is the portfolio actually production-ready?
  **Prepare:** Discuss failure; explain slot assignment vs performance threshold; consider dropping or retraining iTransformer slot.

- **Q:** Three seeds—is that enough for confidence intervals?
  **Prepare:** Justify as exploratory; report variance per model; offer to expand seeds for classifier and TranAD.

### Application
- **Q:** Has any transportation agency used this system on real data?
  **Prepare:** No—simulated demo only; defense focuses on reproducible prototype and institutional deployability (Docker, auth, lake export).

### Limitations
- **Q:** You claim LLM explainability replaces SHAP—is that safe for operators?
  **Prepare:** Not validated; optional; deterministic Insight Engine metrics are primary; LLM is narrative layer only.

---

## 11. Prioritized Revision Plan

| Priority | Issue | Action | Expected Impact |
|----------|-------|--------|-----------------|
| Critical | RQ4 unanswered | Add routing match-rate table on demo events; revise §5.10 conclusion | High — prevents defense failure on false claim |
| Critical | Classical numbers unreproduced | Add script or "prior study" disclaimer + frozen artifact | High — auditability |
| Critical | 96.83% vs 96.62% inconsistency | Harmonize abstract, Table 4.2, §5.3 | High — trust |
| Critical | Dual narrative confusion | Contribution matrix + abstract rewrite | High — clarity |
| High | Incomparable Table 4.2 | Split by task; add AUROC for anomaly | High — scientific rigor |
| High | No external neural baselines | Naive + one published benchmark row | High — Q1 credibility |
| High | DQI/classical protocol missing | Tables 3.1–3.2 with weights and CV | Medium — methodology defense |
| Medium | iTransformer failure ignored | Failure analysis paragraph | Medium — honesty |
| Medium | Stale conclusion text | Remove screenshot future-work | Low — polish |
| Medium | 1.88 GB/s throughput | Remove or measure properly | Medium — credibility |
| Low | LLM explainability | Downgrade claims or add mini user check | Medium — XAI scrutiny |

---

## 12. Final Recommendation

**Major revision required** (if evaluated as a Q1 journal paper).

**Acceptable for defense with revisions** (as a master's thesis)—provided the author:
1. Fixes numeric inconsistencies and stale text before antiplagiarism/final submission.
2. Does not claim RQ4 is *answered* until routing metrics exist, or reframes RQ4 as implementation-only.
3. Can verbally explain the classical-vs-platform split using Table 4.4.

The work is **not publication-ready** for a Q1 venue without routing evaluation, baseline comparisons, dataset generalisation, and a unified scientific claim. For **Astana IT University master's defense**, it is **defensible** after targeted revisions above—especially the RQ4 and classical reproducibility gaps—because the platform demonstration, neural artifact bundle, and honest limitations provide sufficient engineering and methodological substance for a software-engineering-oriented program.

---

*Review completed: 2026-05-27. Reviewer stance: strict Q1 / professor simulation per project skill `q1-paper-reviewer`.*
