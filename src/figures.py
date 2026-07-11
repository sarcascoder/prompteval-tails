"""Publication figures for the PromptEval tail-reliability study. Real data only."""
import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
OUT = "paper/figures"; os.makedirs(OUT, exist_ok=True)
BUD = [200, 400, 800, 1600]; QS = [5, 25, 50, 75, 95]
COL = {200: "#c1121f", 400: "#e07a00", 800: "#0077b6", 1600: "#023047"}

d = json.load(open("results/repro_full_MMLU_tNone_lNone_s5.json"))
df = pd.DataFrame(d["recs"])


def bootci(x, f=np.mean, n=1000, seed=0):
    r = np.random.RandomState(seed); x = np.asarray(x); idx = np.arange(len(x))
    b = [f(x[r.choice(idx, len(idx), True)]) for _ in range(n)]
    return f(x), np.percentile(b, 2.5), np.percentile(b, 97.5)


# ---- Fig 1: center-vs-tail U-curve (PromptEval), abs error vs quantile per budget ----
fig, ax = plt.subplots(figsize=(6.0, 3.7))
pe = df[df.method == "prompteval"]
for b in BUD:
    means, los, his = [], [], []
    for q in QS:
        v = pe[(pe.budget == b) & (pe["quantile"] == q)]["absol"].values
        m, lo, hi = bootci(v); means.append(m); los.append(lo); his.append(hi)
    ax.plot(QS, means, "o-", color=COL[b], label=f"budget={b}", lw=1.8, ms=5)
    ax.fill_between(QS, los, his, color=COL[b], alpha=0.15)
ax.set_xlabel("quantile of the prompt-accuracy distribution")
ax.set_ylabel("mean |estimate − truth|  (accuracy)")
ax.set_title("Tails are 4–10x harder than the center (PromptEval, MMLU)")
ax.set_xticks(QS); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_center_vs_tail.pdf"); fig.savefig(f"{OUT}/fig1_center_vs_tail.png")
plt.close(fig)

# ---- Fig 2: directional over-dispersion (signed error vs quantile) ----
fig, ax = plt.subplots(figsize=(6.0, 3.7))
for b in BUD:
    means, los, his = [], [], []
    for q in QS:
        v = pe[(pe.budget == b) & (pe["quantile"] == q)]["signed"].values
        m, lo, hi = bootci(v); means.append(m); los.append(lo); his.append(hi)
    ax.plot(QS, means, "o-", color=COL[b], label=f"budget={b}", lw=1.8, ms=5)
    ax.fill_between(QS, los, his, color=COL[b], alpha=0.15)
ax.axhline(0, color="k", lw=0.8)
ax.annotate("worst prompt estimated TOO LOW", (5, -0.20), fontsize=8, color="#7a0010")
ax.annotate("best prompt estimated TOO HIGH", (58, 0.16), fontsize=8, color="#7a0010", ha="right")
ax.set_xlabel("quantile of the prompt-accuracy distribution")
ax.set_ylabel("signed error  (estimate − truth)")
ax.set_title("Directional bias: the estimated spread is inflated at every budget")
ax.set_xticks(QS); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_overdispersion.pdf"); fig.savefig(f"{OUT}/fig2_overdispersion.png")
plt.close(fig)

# ---- Fig 3: tail/center ratio vs budget (does more budget fix it? no) ----
fig, ax = plt.subplots(figsize=(6.0, 3.7))
for meth, c, mk in [("prompteval", "#0077b6", "o"), ("baseline", "#8d99ae", "s")]:
    ratios = []
    for b in BUD:
        s = df[(df.method == meth) & (df.budget == b)]
        g = s.groupby(["task", "llm", "quantile"])["absol"].mean().unstack("quantile")
        ratios.append(((g[5] + g[95]) / 2).mean() / g[50].mean())
    ax.plot(BUD, ratios, mk + "-", color=c, label=meth, lw=1.8, ms=6)
ax.set_xscale("log"); ax.set_xticks(BUD); ax.set_xticklabels(BUD)
ax.set_xlabel("evaluation budget (seen cells)")
ax.set_ylabel("tail / center  mean-|error| ratio")
ax.set_title("More budget does not close the tail/center gap")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_tail_center_ratio.pdf"); fig.savefig(f"{OUT}/fig3_tail_center_ratio.png")
plt.close(fig)

print("wrote fig1_center_vs_tail, fig2_overdispersion, fig3_tail_center_ratio to", OUT)
