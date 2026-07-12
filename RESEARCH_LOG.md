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

### Confirmed robust findings (full grid, 57×15×5, bootstrap CIs)
- **Fidelity: EXACT.** My harness reproduces their released `processed_results_MMLU` quantile-error array to 4 decimals (max cell diff = 0.0000).
- **Directional over-dispersion (the crux).** PromptEval's estimated prompt-accuracy distribution is over-dispersed at EVERY budget: signed P5 error <0 (worst prompt estimated too low) and signed P95 error >0 (best prompt too high), 100% of cells, tight CIs. Median spread ratio est(P95−P5)/true = **8.4×** @200, still **3.1×** @1600. Counterintuitive: a regularized estimator *inflates*, not compresses, the tail spread.
- **Decision-relevant.** @ headline budget 200: estimated worst-case (P5) prompt accuracy is **0.265 too low** → tells a practitioner their worst prompt is 26 pts worse than reality; prompt-sensitivity spread inflated ~8×. Best-prompt selection regret ≈ 0.035 (picks a prompt ~3.5 pts below true best), barely improves with budget.
- **More budget doesn't close the gap.** tail/center abs-error ratio *grows* 4.2×→10× as budget rises (center error collapses faster than tail). So it's not merely a tiny-budget artifact.
- **Mechanism (hypothesis):** percentiles of per-template accuracy estimated from ~2 seen items/template are noise-inflated; Rasch borrowing-strength helps (8.4× vs baseline ~20×) but doesn't deconvolve the sampling noise from the tail.

### FIX validated (subset, MMLU @200)
- Fixed-point parametric-bootstrap deconvolution (src/fix_deconv.py) estimates the true-
  spread shrink factor s from budgeted data ALONE (never sees true quantiles), then linearly
  shrinks per-template accuracy estimates before taking quantiles.
- Result: tail |err| 0.189 -> 0.068 (**64% lower**), improves ALL quantiles incl. median;
  recovered shrink s≈0.36 (~2.8x deflation). Honestly under-corrects vs ideal (can't fully
  recover a tiny true spread from ~2 items/template) — itself a useful cautionary point.

### Novelty verdict (this specific finding+fix): CLEAR
- No paper reports PromptEval/IRT multi-prompt quantile over-dispersion + deconvolution fix.
- Closest: "Tail-Shape Estimation in LLM Evaluation Is Fragile" (2606.16511) — ORTHOGONAL
  (toxicity reward-model extreme-value tails; not PromptEval/prompt-performance quantiles).
  Good related-work cite, not a scoop. Also cite Madaan "Quantifying Variance" (2406.10229).

### Synthetic control (known truth) — validates mechanism, corrected the FIX
- Simulated Rasch worlds (P=I=100) with controlled true prompt-spread. Over-dispersion
  appears exactly as predicted and is LARGER when true spread is small (spread ratio 6.1x at
  sigma_theta=0.2, matching MMLU's ~8x). Confirms mechanism with zero dependence on released data.
- The earlier bootstrap "fix" was UNRELIABLE (synthetic: only 4-6% vs 64% on real data) — the
  parametric bootstrap simulates from the already-inflated fit, so it can't see the full
  inflation. DROPPED it. Good catch by the control.

### FINAL fix = split-half reliability disattenuation (consistent, bootstrap-free)
- Split budget into 2 disjoint stratified halves -> independent noisy per-template estimates;
  V_signal = Cov(aA,aB); shrink factor s = clip(V_signal/Var(abar),0,1); shrink toward mean.
- Validated vs synthetic ORACLE (known-truth shrink): oracle recovers 83-94% tail error at all
  spreads; split-half matches oracle when true spread is SMALL (83% @ sigma=0.2) but OVER-shrinks
  when true spread is large (-96% @ sigma=1.0).
- Real data @ budget 200: MMLU tail |err| 0.226->0.041 (**82% lower**, P5 84%, P95 78%). BUT
  BBH +42% (P5 slightly worse), LMentry HURTS -57%. Reason: at budget<~2 items/template
  (LMentry 258 tmpl, BBH 187 tmpl vs 100 cells/half) reliability estimate collapses; and
  LMentry's over-dispersion is ASYMMETRIC (only high tail inflated) so symmetric shrink harms P5.
- Honest conclusion: over-dispersion is correctable in principle (oracle) and a simple
  reliability correction recovers most tail error in the severe symmetric regime (MMLU/MCQ),
  but robust budget-only correction across benchmarks is an OPEN problem — a genuine finding.

### Universal across all 3 benchmarks
- Center-tail gap: tail 2.4-4.2x center @200 (Wilcoxon p<=1e-21). High tail (P95/best-prompt)
  OVER-estimated everywhere (correction helps P95 by 52-78% on all three).

### Next: finalize paper (honest fix framing), figures, repro package, reviewer-2 pass.
