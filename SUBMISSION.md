# Submission guide — arXiv, GitHub, then a workshop / Findings

Everything needed to publish this work. The paper source is `paper/main.tex`; a ready-to-upload
arXiv package is `arxiv_submission.tar.gz` (verified to compile standalone).

---
## 0. Fill in 3 placeholders first
In `paper/main.tex` replace:
- `YOUR NAME` → your name (and co-authors, if any)
- `YOUR AFFILIATION \quad your.email@example.com` → affiliation (or "Independent Researcher") + email
- `REPLACE-WITH-YOUR-USERNAME/prompteval-tails` (two places) → your GitHub repo path

Then rebuild the tarball:
```
cd paper && tectonic main.tex && cd ..
rm -rf arxiv_submission && mkdir -p arxiv_submission/figures
cp paper/main.tex arxiv_submission/
cp paper/figures/fig{1_center_vs_tail,2_overdispersion,3_tail_center_ratio,4_correction_mmlu,5_crossbench_fix}.png arxiv_submission/figures/
tar -czf arxiv_submission.tar.gz -C arxiv_submission .
```

---
## 1. GitHub (do this first so the paper can link to it)
```
# create an empty public repo named e.g. prompteval-tails on github.com, then:
git remote add origin https://github.com/<username>/prompteval-tails.git
git push -u origin master     # (or: git branch -M main && git push -u origin main)
```
The repo already has README, LICENSE (MIT), requirements.txt, run_all.sh, code, figures, and
RESEARCH_LOG. `external/` (upstream code+data) is git-ignored and re-cloned by `run_all.sh`.

---
## 2. arXiv (immediate; free)
1. Create an account at arxiv.org with an **institutional email if you have one** (helps skip
   endorsement). Personal email is fine but first-time cs submissions may need an **endorsement**
   from an existing arXiv author — see the note below.
2. Start a new submission → upload **`arxiv_submission.tar.gz`** (arXiv compiles the LaTeX).
3. Metadata:
   - **Primary category:** `cs.CL` (Computation and Language)
   - **Cross-list:** `cs.LG`, `stat.ML`
   - **Title:** Reliable in the Center, Unreliable at the Edges: Auditing the Tail Fragility of Multi-Prompt LLM Evaluation
   - **Authors:** <your name(s)>
   - **Abstract:** paste the plain-text abstract below
   - **Comments:** e.g. "8 pages, 5 figures. Code: https://github.com/<username>/prompteval-tails"
   - **License:** CC BY 4.0 recommended (or arXiv non-exclusive).
4. Choose a license, preview the compiled PDF, submit. It appears after the next announcement cycle.

**Endorsement note:** if arXiv asks for an endorsement for cs.CL, you need one existing cs.CL
author to endorse you (arXiv gives you a code to send them). An institutional email often avoids
this. This is the one step I cannot do for you.

### Abstract (plain text, paste into arXiv)
> Because large language model (LLM) accuracy varies sharply with the exact wording of a prompt,
> a growing line of work replaces single-prompt scores with an estimate of the distribution of
> accuracy across many prompt templates. PromptEval, a NeurIPS 2024 method built on a regularized
> Rasch model, is a prominent and widely reused instance: it estimates performance quantiles
> across 100 templates at "the budget of two single-prompt evaluations." We first reproduce
> PromptEval exactly---our re-implementation matches the authors' released MMLU quantile-error
> array to four decimal places. We then stress-test the claim that matters most in practice: how
> well does it estimate the tails of the prompt-accuracy distribution (worst- and best-case
> prompts), which are precisely why one runs a multi-prompt evaluation? We find a large,
> systematic, previously unreported center-tail reliability gap. At the advertised budget,
> median-quantile error is a negligible 0.05, but the 5th/95th-percentile error is 4.2x larger,
> and the error is directional: the worst prompt is estimated far too low and the best too high,
> so the estimated prompt-sensitivity spread is inflated. The gap does not close with budget. A
> controlled simulation with known ground truth reproduces the over-dispersion, and freshly
> generated evaluations from two local models confirm it on data outside the original pipeline. We
> show the bias is correctable in principle and give a label-free split-half correction that
> removes ~82% of the tail error where over-dispersion is severe, while charting where it fails.
> The rule: trust the center; correct the tails only when reliability can be estimated.

---
## 3. Workshop / reproducibility track (after arXiv)
Good fits once arXiv is up:
- **ICLR Blogposts / Tiny Papers**, or a **reproducibility workshop** (e.g. MLRC / ML Reproducibility).
- **NeurIPS/ICLR workshops** on evaluation (e.g. "Evaluating Foundation Models", "Data-centric ML").
- Cite the arXiv id in the submission; most workshops are non-archival and allow arXiv preprints.

## 4. ACL/EMNLP Findings (with a polish pass)
Feasible with: a short "how to use this in practice" section, one more model family in the
fresh-data table, and an anonymized version (ARR requires double-blind — strip author block and
the repo URL for review). Timeline depends on the ARR cycle.
