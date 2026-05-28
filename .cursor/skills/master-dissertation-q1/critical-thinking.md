# Critical Thinking Frameworks

Use these during the REACT loop. Do not skip the Challenge phase.

## Toulmin argument decomposition

For every major claim in the dissertation:

```
Claim:      [What you assert]
Grounds:    [Data, citation, or artifact]
Warrant:    [Why grounds support the claim]
Backing:    [Theory or prior work supporting the warrant]
Qualifier:  [Under what conditions the claim holds]
Rebuttal:   [Strongest counterargument and your response]
```

**Example (STEM):**
- Claim: Adaptive model routing reduces p95 latency vs. static assignment.
- Grounds: Table 5.3, 3 seeds, streaming benchmark CSV.
- Warrant: Router selects smaller models for low-complexity windows.
- Backing: Model selection literature (cascade/routing surveys).
- Qualifier: Holds for Astana-derived traffic at 100 Hz ingest; not validated cross-city.
- Rebuttal: Gains may be routing overhead artifact → report overhead ms explicitly.

## Hypothesis lifecycle

```
1. Formulate  → H0 (null) and H1 (alternative), both falsifiable
2. Operationalize → metric, dataset, procedure, stopping rule
3. Pre-specify → document before seeing results (methodology chapter)
4. Execute    → run; log failures and anomalies
5. Interpret  → accept/reject/inconclusive with effect size, not p-value alone
6. Revise     → update future work; do not retroactively change hypotheses
```

## Adversarial personas

Rotate through these during Challenge phase:

### The Methods Reviewer
- "Your baseline had unequal tuning budget."
- "Single seed is anecdote, not evidence."
- "Train/test leakage possible via temporal overlap?"
- "Hardware confound: did baseline run on same GPU class?"

### The Related-Work Reviewer
- "You omitted [seminal 2019 paper in this exact subproblem]."
- "Your 'gap' was solved in [industry report / arXiv preprint]."
- "Novelty claim is incremental combination, not new insight."

### The Systems Reviewer
- "Architecture diagram does not match deployed code path."
- "Latency numbers exclude cold-start / queue wait."
- "Reproducibility: where is docker-compose + seed + config hash?"

### The Examiner (Master's defense)
- "Explain this equation to a non-specialist in 60 seconds."
- "What would you do differently with 6 more months?"
- "Which result are you least confident in, and why?"

For each persona attack: **fix**, **qualify with limitation**, or **remove claim**.

## Inference discipline

| Statement type | Allowed without citation? | Example |
|----------------|---------------------------|---------|
| Definition (standard) | Yes, if textbook-standard | "RMSE measures forecast error." |
| Empirical fact from your work | Yes, with artifact path | "Macro-F1 = 0.87 (Table 5.2)." |
| Empirical fact from literature | No | Must cite primary source |
| Causal claim | No, without design support | "X causes Y" needs experiment or explicit causal model |
| Future impact | Qualify as speculation | "[SPECULATIVE] Could enable..." |

## Synthesis patterns (not laundry lists)

**Thematic synthesis paragraph structure:**
1. **Theme** — sub-problem or approach family
2. **Consensus** — what most papers agree on
3. **Controversy** — where results or assumptions diverge
4. **Gap** — what remains unsolved (linked to your RQ)

**Bad:** "Smith (2020) used LSTM. Jones (2021) used Transformer. ..."
**Good:** "Sequence models dominate traffic forecasting (Smith, 2020; Jones, 2021), yet routing across heterogeneous model classes under streaming latency constraints remains under-evaluated — motivating RQ3."

## Cognitive bias guards

| Bias | Guard |
|------|-------|
| Confirmation | Actively search for disconfirming evidence before writing Discussion |
| HARKing | Separate pre-specified hypotheses from exploratory findings |
| Cherry-picking | Report all metrics in protocol, not only best |
| Authority | Cite primary experiments, not survey-of-survey |
| Complexity bias | Prefer simplest explanation that fits evidence |

## Reasoning chain template

Use internally before major decisions:

```markdown
## Decision: [e.g., include TranAD vs. classical IF in Ch.5]

**Options:** A, B, C
**Criteria:** narrative coherence, evidence strength, examiner familiarity, gap alignment
**Evidence gathered:** [paths, searches performed]
**Trade-offs:** ...
**Decision:** ... because ...
**Residual risk:** ... mitigated by ...
```
