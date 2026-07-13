"""
Apply the PromptEval estimator + center-vs-tail / over-dispersion analysis to the FRESH,
locally-generated correctness matrices (results/fresh/Y_*.json). If the finding replicates on
data untouched by the authors' pipeline, over-dispersion is a property of the estimator.
"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join("external", "prompteval", "prompteval"))
import types
_stub = types.ModuleType("utils"); _stub.check_multicolinearity = lambda *a, **k: None
sys.modules["utils"] = _stub
from methods import PromptEval  # noqa: E402

QS = [5, 25, 50, 75, 95]
BUDGETS = [200, 400, 800]


def analyze(files, seeds=5):
    rows, spread = [], []
    for f in files:
        model = os.path.basename(f)[2:-5]
        data = json.load(open(f))
        for subj, Ylist in data.items():
            Y = np.asarray(Ylist, float)
            if Y.shape[0] < 10 or Y.mean(1).std() < 1e-9:
                continue
            true_q = np.percentile(Y.mean(1), QS)
            true_spread = true_q[-1] - true_q[0]
            for seed in range(seeds):
                M = PromptEval(); M.fit(Y, quantiles=QS, rounds_eval=BUDGETS, random_seed=seed)
                for bi, b in enumerate(BUDGETS):
                    est_q = np.asarray(M.estimates["pirt"][bi])
                    for qi, q in enumerate(QS):
                        rows.append(dict(model=model, subj=subj, seed=seed, budget=b, quantile=q,
                                         signed=float(est_q[qi] - true_q[qi]),
                                         absol=float(abs(est_q[qi] - true_q[qi]))))
                    est_acc = np.asarray(M.estimates["accs_hat"][bi])
                    es = np.percentile(est_acc, 95) - np.percentile(est_acc, 5)
                    spread.append(dict(model=model, subj=subj, seed=seed, budget=b,
                                       true_spread=float(true_spread), est_spread=float(es),
                                       ratio=float(es / true_spread) if true_spread > 1e-6 else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(spread)


if __name__ == "__main__":
    files = sorted(glob.glob("results/fresh/Y_*.json"))
    files = [f for f in files if "smoke" not in f]
    print("fresh matrices:", [os.path.basename(f) for f in files])
    df, sp = analyze(files)
    n_cells = df[["model", "subj"]].drop_duplicates().shape[0]
    print(f"\n{n_cells} fresh (model x subject) cells\n")
    for b in BUDGETS:
        s = df[df.budget == b]
        g = s.groupby(["model", "subj", "quantile"])["absol"].mean().unstack("quantile")
        center = g[50].values; tail = (g[5].values + g[95].values) / 2
        ratio = tail.mean() / center.mean()
        p5 = s[s["quantile"] == 5]["signed"].mean(); p95 = s[s["quantile"] == 95]["signed"].mean()
        spb = sp[sp.budget == b]
        od = (spb.est_spread > spb.true_spread).mean()
        print(f"budget {b:>4}: center|err|={center.mean():.3f}  tail|err|={tail.mean():.3f}  "
              f"tail/center={ratio:.2f}x  signed P5={p5:+.3f} P95={p95:+.3f}  "
              f"over-disp={100*od:.0f}%  median spread ratio={spb.ratio.median():.1f}x")
    # significance at headline budget
    s = df[df.budget == 200]
    g = s.groupby(["model", "subj", "quantile"])["absol"].mean().unstack("quantile")
    tail = (g[5].values + g[95].values) / 2; center = g[50].values
    _, p = stats.wilcoxon(tail, center)
    print(f"\nWilcoxon tail>center @200 across {len(center)} fresh cells: p={p:.2e}")
    json.dump({"rows": df.to_dict("records"), "spread": sp.to_dict("records")},
              open("results/fresh/analysis.json", "w"))
    print("saved -> results/fresh/analysis.json")
