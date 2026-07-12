# Reviewer #2 pass (adversarial self-review)

Goal: try to reject the paper. Each attack is followed by how the paper answers it and any
residual weakness kept honestly.

### A1. "Extreme quantiles are harder to estimate than the median — this is trivial."
**Answer.** The claim is not "tails are noisier." It is that the error is (i) *directional*
(systematic over-dispersion: worst prompt too low, best too high — a practitioner is actively
misled, not just uncertain), (ii) *decision-relevant* (worst-case, spread, and best-prompt are
the entire reason to estimate a distribution rather than a mean), (iii) *hidden* by the single
averaged error metric prior work reports, and (iv) *relatively worse with more budget* (ratio
4.2×→10×). None of these follow from "extreme quantiles are harder." **Residual:** the raw
existence of larger tail error is unsurprising; our contribution is its structure and
consequence.

### A2. "Your ‘truth’ is itself a finite-item estimate, so the comparison is biased."
**Answer.** We adopt PromptEval's own ground-truth convention (full-data accuracy over all
items). Finite-item truth would if anything *over*-state the true spread, making our
over-dispersion estimate **conservative**. Decisively, the synthetic study (§6) uses an
*exactly known* true spread and reproduces the effect. **Residual:** none material.

### A3. "One data source (the authors' own release)."
**Answer.** Three benchmarks (MMLU/MCQ, BBH/free-form, LMentry), 42 model×task families, five
seeds, *plus* a from-scratch Rasch simulator that shares none of the released data. Direction is
universal. **Residual:** we do not generate fresh LLM outputs over new template sets; the
synthetic generator + three benchmarks are the mitigation.

### A4. "The ‘fix’ is cherry-picked to MMLU; it fails elsewhere."
**Answer.** Correct, and we say so prominently (§7, Fig. 5): it helps on MMLU (82%) and the high
tail everywhere, but over-corrects on LMentry. We present it with an oracle upper bound and a
mechanistic account of *when* it fails (asymmetry; budget < 1 item/template). Non-universality
is reported as a finding, not hidden. **Residual:** we do not deliver a universally-safe
correction; we scope it and flag reliability-gating as future work.

### A5. "p ≈ 1e-141 is meaningless at n=855."
**Answer.** We lead with effect sizes (tail/center 4.2×; signed bias −0.265; spread ratio 8.4×;
best-prompt regret 3.5 pts) and bootstrap CIs; p-values are secondary.

### A6. "Is this PromptEval-specific or true of any estimator?"
**Answer.** It is a property of the regularized additive-Rasch estimator under finite sampling:
the naive baseline is far worse (spread ratio ~20× vs ~8×), the oracle shows it is correctable,
and the well-specified synthetic (no interactions) still over-disperses — isolating finite-sample
inflation as the cause.

### A7. "Maybe the additive model ignoring prompt×item interactions is the real cause."
**Answer.** We attribute the tail error to finite-sample inflation, evidenced by (i) monotone
decay of the spread ratio toward 1 as budget→full and (ii) the interaction-free synthetic
generator still exhibiting the effect. Interactions are a *complementary* source we explicitly
leave in Limitations rather than conflate.

### A8. "Novelty is thin — known quantile bias applied to a method."
**Answer.** The contribution is a faithful (exact) reproduction of a widely-reused NeurIPS
method plus a specific, decision-relevant, previously-unreported failure mode, its mechanism,
and the boundaries of a simple correction. We target a reproducibility / findings venue and do
not claim new estimation theory. Novelty search found no prior report of PromptEval/multi-prompt
quantile over-dispersion or this correction; the closest work (arXiv:2606.16511) concerns
toxicity extreme-value tails, a different setting.

## Verdict
The **diagnosis** (reproduction + directional over-dispersion + decision-relevance + mechanism
+ synthetic confirmation) is robust and defensible. The **correction** is honestly scoped as
partial. Overall: a solid reproducibility/findings-tier contribution; not a new-method
main-track claim, and not presented as one.

## Known residual weaknesses (kept honest)
- No fresh LLM inference over new template sets (mitigated by 3 benchmarks + simulation).
- Correction not robust across benchmarks; reliability-gated/asymmetry-aware version is future work.
- Only the released additive-Rasch variant analyzed; covariate-augmented variants untested.
