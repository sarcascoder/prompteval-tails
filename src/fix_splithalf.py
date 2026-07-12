"""
Split-half reliability correction for PromptEval tail over-dispersion (consistent,
bootstrap-free). Validated against known synthetic truth before use on real data.

Model: each per-template accuracy estimate a = a_true + e with independent noise e.
Split the budget into two disjoint stratified halves, fit the estimator on each ->
a_A, a_B (independent noise). Then
    V_signal = Cov(a_A, a_B)                         (noise cancels in the cross-moment)
    a_bar    = (a_A + a_B) / 2
    s        = clip( V_signal / Var(a_bar), 0, 1 )   (disattenuation / linear shrinkage)
    a_corr   = mean(a_bar) + s * (a_bar - mean(a_bar))
Return quantiles of a_corr. Uses only budgeted data; never sees the true quantiles.
"""
import sys, os, numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "external", "prompteval")
sys.path.insert(0, os.path.join(REPO, "prompteval"))
import types
_stub = types.ModuleType("utils"); _stub.check_multicolinearity = lambda *a, **k: None
sys.modules["utils"] = _stub
from methods import ExtendedRaschModel, StratSample  # noqa: E402

QS = [5, 25, 50, 75, 95]


def _fit_acc(seen, Y):
    m = ExtendedRaschModel(); m.fit(seen, Y)
    return m.get_Y_hat().mean(-1)


def splithalf_quantiles(Y, budget, seed):
    """Return (orig_quantiles, corrected_quantiles, shrink_s)."""
    # full budget estimate (what PromptEval reports)
    seen_full = StratSample(np.zeros(Y.shape, bool), budget, seed)
    a_full = _fit_acc(seen_full, Y)
    orig_q = np.percentile(a_full, QS)

    # two disjoint stratified halves of the SAME budget
    seenA = StratSample(np.zeros(Y.shape, bool), budget // 2, seed)
    # sample B from the complement by masking already-seen cells as "unavailable":
    # StratSample only sets unseen cells, so start from seenA and grow, then B = new-seen
    seen_both = StratSample(seenA.copy(), budget, seed + 10_000)
    seenB = seen_both & (~seenA)
    aA = _fit_acc(seenA, Y)
    aB = _fit_acc(seenB, Y)

    abar = 0.5 * (aA + aB)
    v_signal = np.cov(aA, aB)[0, 1]
    v_bar = np.var(abar)
    s = float(np.clip(v_signal / v_bar, 0.0, 1.0)) if v_bar > 1e-9 else 1.0
    mu = abar.mean()
    a_corr = mu + s * (abar - mu)
    return orig_q, np.percentile(a_corr, QS), s


def run_grid(bench="MMLU", budget=200, n_tasks=None, n_llms=None, seeds=3):
    """Evaluate split-half correction across a benchmark grid; return long-form rows."""
    import pickle
    Ys = pickle.load(open(os.path.join(REPO, "data", "Ys.pickle"), "rb"))
    tasks = list(Ys[bench].keys())[: (n_tasks or None)]
    rows = []
    for t in tasks:
        nl = len(Ys[bench][t]) if n_llms is None else min(n_llms, len(Ys[bench][t]))
        for llm in range(nl):
            Y = np.asarray(Ys[bench][t][llm], float)
            tq = np.percentile(Y.mean(-1), QS)
            for seed in range(seeds):
                oq, cq, s = splithalf_quantiles(Y, budget, seed)
                for i, q in enumerate(QS):
                    rows.append(dict(bench=bench, task=t, llm=llm, seed=seed, budget=budget,
                                     quantile=q, true=float(tq[i]),
                                     orig_abs=float(abs(oq[i] - tq[i])),
                                     corr_abs=float(abs(cq[i] - tq[i])), s=s))
        print(f"[{bench}] {t} done", flush=True)
    return rows


if __name__ == "__main__":
    import argparse, json, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="MMLU")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--n_tasks", type=int, default=None)
    ap.add_argument("--n_llms", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    rows = run_grid(a.bench, a.budget, a.n_tasks, a.n_llms, a.seeds)
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(f"results/splithalf_{a.bench}_b{a.budget}.json", "w"))
    df = pd.DataFrame(rows)
    g = df.groupby("quantile").agg(orig=("orig_abs", "mean"), corr=("corr_abs", "mean"))
    g["red_pct"] = (100 * (g["orig"] - g["corr"]) / g["orig"]).round(1)
    print(g.round(4).to_string())
    tl = df[df["quantile"].isin([5, 95])]
    print(f"TAIL: {tl['orig_abs'].mean():.4f} -> {tl['corr_abs'].mean():.4f} "
          f"({100*(tl['orig_abs'].mean()-tl['corr_abs'].mean())/tl['orig_abs'].mean():.1f}% lower)")
