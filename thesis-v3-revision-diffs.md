# Flowmatic Thesis V3 Revision Diff Log

This file records the old/now text changes made after `paper-review-codex-v3.md`. The old PDF `memoirthesis-flowmatic-v3.pdf` is intentionally preserved; the revised PDF is compiled separately.

## 1. Abstract clean-upload framing

File: `thesis/Flowmatic-Thesis-v3/frontmatter/abstract.tex`

Old:

> On clean Astana batch uploads the pipeline reports composite quality 100/100 in 406 ms; under controlled corruptions up to 20%, composite DQI recovers by up to 0.045 while invalid rows are removed.

Now:

> In batch platform validation, a 30,000-row Astana upload completes in 406 ms; under controlled corruptions up to 20%, composite DQI recovers by up to 0.045 while invalid rows are removed.

Reason: clean 100/100 is a latency/platform result, not the strongest quality-improvement evidence.

## 2. Abstract routing framing

File: `thesis/Flowmatic-Thesis-v3/frontmatter/abstract.tex`

Old:

> Rule-based Auto routing over 140 simulator events achieves 100% scenario-aligned policy consistency under documented oracles.

Now:

> Rule-based Auto routing over 140 simulator events achieves 100% scenario-aligned policy conformance under documented oracles, not independent expert-labelled routing accuracy.

Reason: the routing oracles are aligned with the documented routing policy, so the result is policy conformance rather than independent generalisation.

## 3. Chapter 1 novelty framing

File: `thesis/Flowmatic-Thesis-v3/chapters/chapter01/introduction.tex`

Old:

> An integrated preparation-to-inference lifecycle that combines data quality assessment, automated cleaning, export, model registry usage, and downstream inference in one workflow.

Now:

> A prototype preparation-to-inference lifecycle that combines data quality assessment, automated cleaning, export, model registry usage, and downstream inference in one reproducible workflow.

Old:

> A six-dimensional Data Quality Index applied to urban transportation data and evaluated under controlled data corruption scenarios.

Now:

> An operational adaptation of a six-dimensional Data Quality Index to urban transportation data, evaluated through controlled corruption and recovery scenarios.

Old:

> A model-oriented routing mechanism that connects prepared traffic and sensor streams to task-specific analytical models.

Now:

> A rule-based model-routing policy that connects prepared traffic and sensor streams to task-specific analytical models and is evaluated as policy conformance.

Reason: avoid implying new DQI theory or learned routing novelty.

## 4. RQ4 methodology wording

File: `thesis/Flowmatic-Thesis-v3/chapters/chapter03/methodology.tex`

Old:

> To what extent does rule-based Auto routing select scenario-aligned checkpoints on a labelled simulator corpus...

Now:

> To what extent does rule-based Auto routing conform to documented scenario policies on a labelled simulator corpus...

Old:

> Accuracy is the fraction of events for which the selected checkpoint kind equals the scenario oracle.

Now:

> Policy conformance is the fraction of events for which the selected checkpoint kind equals the scenario oracle; because these oracles are aligned with the documented routing policy, the metric verifies implementation consistency rather than independent routing generalisation.

Reason: explicitly defines the RQ4 metric as conformance.

## 5. Chapter 2 routing positioning

File: `thesis/Flowmatic-Thesis-v3/chapters/chapter02/literature.tex`

Old:

> routing accuracy

Now:

> routing policy-conformance checks

Old:

> routing accuracy on labelled simulator events

Now:

> routing policy conformance on labelled simulator events

Reason: keeps the literature-positioning chapter aligned with the revised RQ4 interpretation.

## 6. iTransformer speed slot

File: `thesis/Flowmatic-Thesis-v3/chapters/chapter04/tables.tex`

Old:

> Speed forecast | iTransformer | Astana | flowmatic-astana-itransformer-speed-forecaster

Now:

> Speed forecast (auxiliary; deployment-parity slot) | iTransformer | Astana | flowmatic-astana-itransformer-speed-forecaster

Added note:

> The iTransformer speed slot is retained as an auxiliary deployment-parity artifact; Chapter 5 reports its weak level-speed RMSE and first-difference retune.

Reason: the model remains in the registry, but it should not read as a fully successful production forecaster.

## 7. Chapter 5 routing result

File: `thesis/Flowmatic-Thesis-v3/chapters/chapter05/results.tex`

Old:

> Routing (RQ4) - rule-based Auto router on 140 labelled simulator events.

Now:

> Routing (RQ4) - rule-based Auto router policy conformance on 140 labelled simulator events.

Old:

> Auto routing achieves 100% scenario-aligned policy consistency...

Now:

> Auto routing achieves 100% scenario-aligned policy conformance...

Added:

> The evaluation tests agreement under documented routing rules and scenario labels co-designed with those rules; it should be read as implementation verification, not independent expert-labelled routing accuracy.

Reason: same result, narrower interpretation.

## 8. RQ closure table

File: `thesis/Flowmatic-Thesis-v3/chapters/chapter05/results.tex`

Old:

> RQ4 | Supported: 100% scenario-aligned policy consistency (140 events)

Now:

> RQ4 | Supported as policy conformance: 100% agreement with scenario oracles co-designed with the routing policy (140 events)

Reason: closes RQ4 with the same scope boundary stated in the body.

## 9. Generated routing table wording

Files:

- `thesis/Flowmatic-Thesis-v3/chapters/chapter05/generated_tables.tex`
- `thesis/Flowmatic-Thesis-v3/chapters/chapter05/generated_routing_mismatch.tex`
- `thesis/experiments/v2/run_thesis_v2_experiments.py`

Old:

> Auto routing evaluation...

> Routing accuracy by simulator scenario profile...

> Correct / Acc.

> scenario-aligned policy consistency; task-family accuracy

Now:

> Auto routing policy-conformance evaluation...

> Routing policy conformance by simulator scenario profile...

> Matches / Conf.

> scenario-aligned policy conformance; task-family agreement

Added generated table note:

> Oracles are co-designed with the documented routing policy, so the result verifies implementation consistency rather than independent routing generalisation.

Reason: keeps the table labels consistent with the revised interpretation and makes the future generator output durable.

## 10. Chapter 6 routing conclusion

File: `thesis/Flowmatic-Thesis-v3/chapters/chapter06/conclusion.tex`

Old:

> rule-based Auto routing evaluated on labelled simulator events

Now:

> rule-based Auto routing evaluated for policy conformance on labelled simulator events

Old:

> routing accuracy with documented geo-policy effects for RQ4

Now:

> routing policy-conformance checks with documented geo-policy effects for RQ4

Old:

> Auto routing achieves 100% scenario-aligned policy consistency on 140 simulator events.

Now:

> Auto routing achieves 100% scenario-aligned policy conformance on 140 simulator events whose oracles are co-designed with the documented routing policy. This verifies implementation consistency for the prototype rather than independent expert-labelled routing accuracy.

Reason: conclusion now matches the limitations and RQ closure table.

## 11. Appendix C reproducibility claim

File: `thesis/Flowmatic-Thesis-v3/chapters/appendices/astana_dataset.tex`

Old:

> The thesis does not rely on that manuscript for external verification: all equations and schema in this appendix are self-contained for reproducibility.

Now:

> The released CSV, schema, and experiment scripts are sufficient to reproduce thesis tables; exact calibration constants for the NDA-derived generator remain part of the companion study and are not required to rerun the reported Flowmatic experiments.

Reason: Appendix C contains symbolic generator equations but not every NDA-derived calibration constant.
