"""Fix figures: split-half correction on MMLU, and honest cross-benchmark where-it-helps view."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
OUT = "paper/figures"; QS = [5, 25, 50, 75, 95]


def load(b, bud=200):
    return pd.DataFrame(json.load(open(f"results/splithalf_{b}_b{bud}.json")))

# Fig 4: MMLU orig vs corrected abs error by quantile (grouped bars)
df = load("MMLU")
g = df.groupby("quantile").agg(orig=("orig_abs", "mean"), corr=("corr_abs", "mean"))
x = np.arange(len(QS)); w = 0.38
fig, ax = plt.subplots(figsize=(6.0, 3.7))
ax.bar(x - w/2, g["orig"].values, w, label="PromptEval", color="#c1121f")
ax.bar(x + w/2, g["corr"].values, w, label="+ split-half correction", color="#2a9d8f")
ax.set_xticks(x); ax.set_xticklabels([f"$q_{{{q}}}$" for q in QS])
ax.set_ylabel("mean |estimate − truth|"); ax.set_xlabel("quantile")
ax.set_title("Split-half correction cuts MMLU tail error ~82% (budget=200)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_correction_mmlu.pdf"); fig.savefig(f"{OUT}/fig4_correction_mmlu.png")
plt.close(fig)

# Fig 5: cross-benchmark tail-error reduction (honest: helps MMLU, mixed/hurts elsewhere)
rows = []
for b, bud in [("MMLU", 200), ("BBH", 200), ("BBH", 1600), ("LMentry", 200), ("LMentry", 1600)]:
    try:
        d = load(b, bud)
    except FileNotFoundError:
        continue
    tl = d[d["quantile"].isin([5, 95])]
    red = 100 * (tl["orig_abs"].mean() - tl["corr_abs"].mean()) / tl["orig_abs"].mean()
    rows.append((f"{b}\nB={bud}", red))
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
colors = ["#2a9d8f" if v > 0 else "#c1121f" for v in vals]
fig, ax = plt.subplots(figsize=(6.0, 3.5))
ax.bar(range(len(vals)), vals, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("tail-error reduction (%)")
ax.set_title("Correction helps where over-dispersion is severe/symmetric — not universally")
for i, v in enumerate(vals):
    ax.annotate(f"{v:+.0f}%", (i, v), ha="center", va="bottom" if v > 0 else "top", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_crossbench_fix.pdf"); fig.savefig(f"{OUT}/fig5_crossbench_fix.png")
plt.close(fig)
print("wrote fig4_correction_mmlu, fig5_crossbench_fix")
