"""
Reproduce PromptEval's core distribution-estimation pipeline on the released
MMLU data, but additionally record SIGNED error (not just absolute) at each
quantile and the full estimated per-template accuracy vectors.

Hypothesis (pre-registered): the regularized additive Rasch estimator systematically
COMPRESSES the estimated per-template accuracy distribution (regression to the mean),
over-estimating the low quantiles and under-estimating the high quantiles, so it
understates worst-case prompt degradation. Their published metric is |est-true|,
which masks this directional bias.

Everything here is re-analysis of the authors' released data + their own estimator
code (external/prompteval). No new model inference.
"""
import sys, os, pickle, argparse, json
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "external", "prompteval")
sys.path.insert(0, os.path.join(REPO, "prompteval"))
# The authors' utils.py imports torch for feature code we never touch (we use the
# plain additive Rasch model, X=Z=None, so check_multicolinearity is never called).
# Stub that single symbol to load their exact estimator without the torch dependency.
import types  # noqa: E402
_stub = types.ModuleType("utils")
_stub.check_multicolinearity = lambda *a, **k: None
sys.modules["utils"] = _stub
from methods import Baseline, PromptEval  # noqa: E402

QUANTILES = [5, 25, 50, 75, 95]
BUDGETS = [200, 400, 800, 1600]


def load_data():
    p = os.path.join(REPO, "data")
    with open(os.path.join(p, "Ys.pickle"), "rb") as f:
        Ys = pickle.load(f)
    with open(os.path.join(p, "Xs.pickle"), "rb") as f:
        Xs = pickle.load(f)
    return Ys, Xs


def run(bench="MMLU", n_tasks=None, n_llms=None, seeds=(0, 1, 2), verbose=True):
    Ys, Xs = load_data()
    tasks = list(Ys[bench].keys())
    if n_tasks:
        tasks = tasks[:n_tasks]
    recs = []  # one row per (task, llm, seed, budget, method, quantile)
    spread = []  # spread-ratio + best-prompt regret at each budget
    for ti, task in enumerate(tasks):
        n_llm_task = len(Ys[bench][task]) if n_llms is None else min(n_llms, len(Ys[bench][task]))
        for llm in range(n_llm_task):
            Y = np.asarray(Ys[bench][task][llm], dtype=float)
            true_acc = Y.mean(-1)                      # true per-template accuracy
            true_q = np.percentile(true_acc, QUANTILES)
            true_spread = true_q[-1] - true_q[0]       # P95 - P5
            best_true = true_acc.max()
            for seed in seeds:
                for name, M in [("baseline", Baseline()), ("prompteval", PromptEval())]:
                    M.fit(Y, quantiles=QUANTILES, rounds_eval=BUDGETS, random_seed=seed)
                    est_q_all = M.estimates["pirt" if name == "prompteval" else "estimates"]
                    accs_all = M.estimates["accs_hat"]
                    for bi, budget in enumerate(BUDGETS):
                        est_q = np.asarray(est_q_all[bi])
                        for qi, q in enumerate(QUANTILES):
                            recs.append(dict(task=task, llm=llm, seed=seed, budget=budget,
                                             method=name, quantile=q,
                                             est=float(est_q[qi]), true=float(true_q[qi]),
                                             signed=float(est_q[qi] - true_q[qi]),
                                             absol=float(abs(est_q[qi] - true_q[qi]))))
                        # spread ratio + best-prompt regret from estimated per-template accs
                        est_acc = np.asarray(accs_all[bi])
                        est_spread = np.percentile(est_acc, 95) - np.percentile(est_acc, 5)
                        best_hat_idx = int(np.argmax(est_acc))
                        spread.append(dict(task=task, llm=llm, seed=seed, budget=budget, method=name,
                                           true_spread=float(true_spread),
                                           est_spread=float(est_spread),
                                           spread_ratio=float(est_spread / true_spread) if true_spread > 1e-9 else np.nan,
                                           best_true=float(best_true),
                                           best_regret=float(best_true - true_acc[best_hat_idx])))
        if verbose:
            print(f"[{bench}] task {ti+1}/{len(tasks)} done ({task})", flush=True)
    return recs, spread


def summarize(recs, spread):
    import pandas as pd
    df = pd.DataFrame(recs); sp = pd.DataFrame(spread)
    print("\n=== ABSOLUTE error by method x quantile x budget (mean) ===")
    piv = df.groupby(["method", "budget", "quantile"])["absol"].mean().unstack("quantile")
    print(piv.round(4).to_string())
    print("\n=== SIGNED error (est - true) by method x quantile x budget (mean) — the key test ===")
    piv2 = df.groupby(["method", "budget", "quantile"])["signed"].mean().unstack("quantile")
    print(piv2.round(4).to_string())
    print("\n=== SPREAD RATIO est(P95-P5)/true(P95-P5) [<1 => compression] & best-prompt regret ===")
    piv3 = sp.groupby(["method", "budget"]).agg(
        spread_ratio=("spread_ratio", "mean"),
        best_regret=("best_regret", "mean")).round(4)
    print(piv3.to_string())
    return df, sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="MMLU")
    ap.add_argument("--n_tasks", type=int, default=None)
    ap.add_argument("--n_llms", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/repro_core")
    args = ap.parse_args()
    recs, spread = run(args.bench, args.n_tasks, args.n_llms, tuple(range(args.seeds)))
    os.makedirs("results", exist_ok=True)
    tag = f"{args.bench}_t{args.n_tasks}_l{args.n_llms}_s{args.seeds}"
    with open(f"{args.out}_{tag}.json", "w") as f:
        json.dump({"recs": recs, "spread": spread}, f)
    df, sp = summarize(recs, spread)
    print(f"\nsaved -> {args.out}_{tag}.json  (n_rec={len(recs)})")
