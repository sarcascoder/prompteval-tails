# Research Log

## Project: Reproduction & adversarial stress-test of PromptEval (arXiv 2405.17202, NeurIPS 2024)

### Domain-selection history (Phase 1–3)
- **Chosen domain:** data-centric / eval methodology (after efficient-LLM-inference was found brutally saturated).
- Efficient-inference kill list: KV-eviction/quant + speculative decoding are saturated (~200 abstracts scanned). Two promising angles collapsed under novelty verification:
  - "Apple-Silicon efficiency benchmark" → owned by Benazir&Lin (2508.08531), POMACS characterization, Silicon Showdown (2605.00519), Open-TQ-Metal (2604.16957).
  - "Dispatch-term on batch-1 decode roofline" → owned by 2605.30571 and TaxBreak (2603.12465).
- Eval-methodology also saturated for *clean ideas* (SOTA-audit 2605.17273, Answer-Matching 2507.02856, Rank-Intervals 2606.08679, PromptEval-budget studies). Decision: pivot from "novel method" to "novel *finding* via reproduction + adversarial stress-test".

### Target chosen: PromptEval
- Public artifacts: `github.com/felipemaiapolo/prompteval`; HF `PromptEval/PromptEval_MMLU_correctness` & `_full`.
- $0, pure re-analysis on M5. Estimator = **Extended Rasch** `logit(Y_ij)=θ_i(prompt)+β_j(item)`, additive, fit by L2-regularized logistic regression (C=100, liblinear, no intercept). Quantiles taken over 100 templates' estimated accuracies, per (subject × LLM). Budgets = seen cells [200,400,800,1600]; 200 ≈ "2 single-prompt evals" (the headline).
- Data: MMLU = 57 subjects × 15 LLMs × (100 templates × ~100–150 items). Also BBH (15 tasks × 11 LLMs) and LMentry (10 × 16).

### Findings so far
1. **Reproduction faithful.** My re-run matches their released `processed_results_MMLU_quantiles`. Same magnitude + same U-shape across quantiles.
2. **Center-vs-tail gap (their own data).** At budget 200, PromptEval abs error: q50≈0.053 but q5≈0.27, q95≈0.19 — tails **4–5× worse than the median**. The headline "accurate quantile estimation" is a *center* phenomenon; the practically-important extreme quantiles (worst/best prompt) are far less reliable. They report a single averaged error, which masks this.
3. **Pre-registered hypothesis (compression) REFUTED by data.** Signed error shows est(P5)<true(P5) and est(P95)>true(P95): the estimator *over-disperses* the tails (spread ratio >1), i.e. it makes the prompt-sensitivity distribution look **wider** than it is at low budget — it over-estimates how bad the worst prompt is and how good the best prompt is. (Opposite of my prior; recorded honestly.)

### Next
- Confirm full-grid cell-by-cell fidelity vs their array (running).
- Quantify center-vs-tail reliability gap with seeds + CIs; test significance.
- Mechanism: is tail error driven by template×item (subject) interaction / Rasch misspecification? Is over-dispersion a low-budget small-sample percentile artifact?
- Fix: better tail estimation (e.g. shrinkage-corrected quantiles, or interaction-aware model). Evaluate.
