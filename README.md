# Reliable in the Center, Unreliable at the Edges
### Auditing the tail fragility of multi-prompt LLM evaluation

A reproduction and adversarial stress-test of **PromptEval** (Maia Polo et al., *Efficient
multi-prompt evaluation of LLMs*, NeurIPS 2024, [arXiv:2405.17202](https://arxiv.org/abs/2405.17202)).
Everything here is a **re-analysis of the authors' public data plus simulation**, and runs on a
laptop (CPU only, minutes) at **$0**.

## TL;DR findings
1. **Exact reproduction.** Our re-implementation matches the authors' released MMLU
   quantile-error array to **4 decimals** (max cell difference `0.0000`).
2. **A center–tail reliability gap.** At the advertised "2-eval" budget, median-quantile error
   is `0.05` but the 5th/95th-percentile error is **4.2× larger** (95% CI [4.05, 4.45];
   paired Wilcoxon `p≈1e-141`, n=855 subject×model cells). The gap **grows** with budget
   (4.2×→10×) because the center converges faster.
3. **Directional over-dispersion.** The bias is not symmetric noise: the worst prompt is
   estimated too low and the best too high (100% of cells over-dispersed; median spread ratio
   **8.4×** at the headline budget). A practitioner is told their worst prompt is ~26 accuracy
   points worse than it is, and that their model is far more prompt-sensitive than it is.
4. **Generalizes** to BBH (free-form) and LMentry (tail/center 2.4–4.2×; over-dispersion in
   85–100% of cells; all `p ≤ 1e-21`).
5. **Mechanism + correctability.** A controlled synthetic study with *known* truth reproduces
   the effect (worst when true spread is small). It is correctable in principle (oracle removes
   83–94% of tail error); a label-free **split-half reliability correction** removes **82%** of
   MMLU tail error — but we chart where it fails (asymmetric distributions; budget < 1 item per
   template), leaving robust budget-only correction open.

## Install
```bash
conda create -y -n research python=3.12 && conda activate research
pip install -r requirements.txt
# optional, for the PDF:
conda install -y -c conda-forge tectonic
```

## Reproduce everything
```bash
bash run_all.sh
```
This clones the upstream code+data into `external/`, runs the reproduction, diagnosis,
synthetic study, and correction, regenerates all figures, and (if `tectonic` is present) builds
`paper/main.pdf`.

## Repository layout
```
src/repro_core.py     # faithful re-implementation harness (uses authors' estimator);
                      #   records signed + absolute quantile error, spread, best-prompt regret
src/mechanism.py      # exact-match fidelity check + center-vs-tail CIs and significance tests
src/synthetic.py      # controlled Rasch simulation with known truth (mechanism + oracle)
src/fix_splithalf.py  # split-half reliability correction + grid runner
src/fix_deconv.py     # (superseded) bootstrap correction — kept for transparency; see log
src/figures.py        # Fig 1–3 (diagnosis)
src/figures_fix.py    # Fig 4–5 (correction, incl. honest cross-benchmark panel)
paper/main.tex        # arXiv-style manuscript -> paper/main.pdf
RESEARCH_LOG.md       # full chronological research log incl. dropped hypotheses
external/prompteval/  # upstream code + released data (git-ignored; cloned by run_all.sh)
```

## Data & provenance
All correctness matrices are the authors' public release
(`PromptEval/PromptEval_MMLU_correctness`, `PromptEval/PromptEval_MMLU_full` on the Hugging Face
Hub, MIT-licensed) and ship inside the upstream repo (`external/prompteval/data/Ys.pickle`).
MMLU: 57 subjects × 15 LLMs × 100 templates. BBH: 15 tasks × 11 LLMs × 187 templates.
LMentry: 10 tasks × 16 LLMs × 258 templates. We add no new model inference.

## Honesty notes
- The bootstrap correction in `src/fix_deconv.py` was found unreliable by the synthetic control
  (worked on real data, failed under known truth) and was **replaced** by the split-half
  correction. It is retained for transparency; see `RESEARCH_LOG.md`.
- We report effect sizes alongside p-values (large n makes p-values uninformative on their own).

## License
Code: MIT (`LICENSE`). Upstream PromptEval code/data are MIT-licensed by their authors.
