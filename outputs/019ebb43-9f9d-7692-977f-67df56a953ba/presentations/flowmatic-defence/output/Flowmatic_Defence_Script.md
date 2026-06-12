# Flowmatic Defence Script

Estimated main script: 10-12 minutes. Short emergency version: 6 minutes by skipping detailed verbal explanation on slides 6, 11, 13, and 15.

## Opening statement

My thesis develops and evaluates Flowmatic, a research prototype that makes urban-transport data preparation measurable, auditable, and connected to downstream analytical models.

## Slide 1: Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems

Takeaway: Flowmatic - master's thesis defence | 7M06105 Computer Science and Engineering

Speaker script: Takeaway: Flowmatic is presented as an evaluated research prototype, not as a production-certified municipal system. Explain the title, candidate, programme, supervisor, and year. Transition: the defence begins with the operational problem that motivates the platform.

## Slide 2: Dirty transport data breaks the path from operations to analytics.

Takeaway: The research problem is not just missing values; it is the missing auditable preparation-to-inference chain.

Speaker script: Takeaway: Flowmatic addresses the preparation chain, not generic traffic prediction. Explain heterogeneous sources, defects, and the downstream consequence. Limitation: no external accident statistics are used because the thesis evidence is about data preparation and analytics reliability. Transition to the gap.

## Slide 3: The gap is measurement linkage, not a claim that no tools exist.

Takeaway: Existing systems solve parts of the workflow; the thesis evaluates the chain end to end within a prototype.

Speaker script: Takeaway: the novelty is framed against nearby systems without pretending they do not exist. Explain the four columns. This pre-empts 'why not pandas, Airflow, MLflow, or AutoML'. Transition to aim and research questions.

## Slide 4: The aim is implemented through six objectives and four scoped questions.

Takeaway: Each research question is tied to measurable evidence and a visible limitation.

Speaker script: Takeaway: the thesis is research because objectives are tested through measurable protocols. State aim in one sentence. Explain object and subject. Transition to novelty boundaries.

## Slide 5: The scientific claim is the evaluated chain, not the existence of the platform.

Takeaway: Prior publications and standard components are separated from thesis-new evidence.

Speaker script: Takeaway: this slide answers 'what is new compared with your prior publication'. Explain that platform concepts are continued, while the defence rests on new evaluation: stress tests, baselines, ablation, routing conformance, and reproducibility. Transition to method.

## Slide 6: The research workflow tests the artefact under controlled, reproducible conditions.

Takeaway: The research workflow tests the artefact under controlled, reproducible conditions.

Speaker script: Takeaway: design-build-evaluate is made scientific through testable protocols. Walk from literature to artifact to stress tests, baselines, ablations, and reproducibility. Transition to data scope.

## Slide 7: The evidence is reproducible, but not all data are externally real-world validated.

Takeaway: This slide makes semi-synthetic and simulated data visible before results appear.

Speaker script: Takeaway: trust comes from transparent provenance, not pretending the data are fully public municipal logs. Explain Astana semi-synthetic, HF datasets, simulated feeds, injected anomalies, and rule labels. Transition to DQI method.

## Slide 8: DQI makes preparation auditable before data reach models.

Takeaway: Six dimensions are operationalised through profiling, rules, cleaning, and export records.

Speaker script: Takeaway: the DQI is an operational adaptation of known data-quality dimensions. Explain six dimensions and the raw->profile->DQI->repair/reject->prepared->audit flow. Transition to architecture.

## Slide 9: Flowmatic connects auditable preparation with task-specific inference.

Takeaway: Flowmatic connects auditable preparation with task-specific inference.

Speaker script: Takeaway: this is a real multi-service prototype. Walk left to right: frontend, backend modules, queue/storage/db, simulator, inference, registry. Clarify optional LLM and federated demo are not central evaluated claims. Transition to intelligence definition.

## Slide 10: The system is intelligent as a bounded hybrid assistant, not as general AI.

Takeaway: Automation combines deterministic diagnostics, policy routing, learned models, and operator control.

Speaker script: Takeaway: answer 'why intelligent assistant?' precisely. Deterministic checks and rules are legitimate automation; learned components are task-specific. Operator remains in control. Transition to experimental protocol.

## Slide 11: The experiment design separates leakage control from statistical strength.

Takeaway: Temporal hold-out is strong; three-seed variability should be read descriptively.

Speaker script: Takeaway: the split prevents direct row leakage, but n=3 is limited. Say values are multi-seed descriptive variability, not broad statistical significance. Transition to stress test results.

## Slide 12: Preparation recovered measurable quality, partly by rejecting invalid records.

Takeaway: At 20% corruption, DQI improved from 0.788 to 0.833 while 987 rows were dropped.

Speaker script: Takeaway: the best preparation evidence is the stress test, not the clean 100/100 upload. Explain red/green bars and row drops. Scope boundary: row rejection can remove potentially useful rare records. Transition to model evidence.

## Slide 13: Downstream results support capability, with explicit task limits.

Takeaway: Metrics are separated by task to avoid pretending RMSE, F1, AUROC, and MSE are comparable.

Speaker script: Takeaway: the portfolio is evidence for integrated task slots, not SOTA benchmarking. Mention key comparators and caveats: rule labels, injected anomalies, z-score windows. Transition to ablation.

## Slide 14: Preparation improved forecasting under a controlled upper-bound experiment.

Takeaway: PatchTST RMSE dropped from 0.9045 to 0.5802 after oracle-style restoration on 20% corrupted data.

Speaker script: Takeaway: this is the strongest preparation-to-downstream linkage, but it is not deployable imputation performance. State 'oracle restoration upper-bound' before the examiner does. Transition to implementation/performance.

## Slide 15: The prototype works, but the measured boundary matters.

Takeaway: Use core processing and inference-latency numbers without turning them into production SLA claims.

Speaker script: Takeaway: Flowmatic is real and measurable, but not production certified. Explain the timing boundaries: 406 ms batch processing, TorchScript latency from benchmark, not full wall-clock SLA. Transition to RQ closure.

## Slide 16: Each research question has evidence and an explicit scope limit.

Takeaway: Each research question has evidence and an explicit scope limit.

Speaker script: Takeaway: this is the defence consolidation slide. Walk each row quickly: supported, evidence, scope limit. Transition to personal contribution and publications.

## Slide 17: My contribution is the integrated implementation and thesis-new evaluation package.

Takeaway: Prior publications are supporting context; thesis claims are tied to new or extended artefacts.

Speaker script: Takeaway: do not say 'I did everything'. Say exactly what belongs to the thesis: platform integration, scripts, experiments, analysis, and defensible writing. Transition to conclusion.

## Slide 18: Flowmatic is an evaluated research prototype for auditable transport-data preparation.

Takeaway: The strongest claim is narrow and defensible: controlled preparation evidence can be linked to downstream model behaviour.

Speaker script: Takeaway: finish with the honest thesis-level statement. What was achieved: platform, methodology, evidence, reproducibility. What remains: real external data, independent labels, stronger baselines, operator study, security and scaling validation. Invite questions.

## Closing statement

Flowmatic should be judged as an evaluated research prototype. The strongest defensible result is the linked preparation-to-downstream evidence chain under controlled, reproducible conditions, with limitations made explicit rather than hidden.
