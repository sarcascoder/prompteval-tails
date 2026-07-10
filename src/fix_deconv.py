"""
Fix + mechanism proof: variance-deconvolution correction for PromptEval tail quantiles.

Diagnosis: PromptEval's per-template accuracy estimates a_hat = a_true + estimation
noise; percentiles of a_hat therefore over-state the spread (tails pushed outward).

Correction (parametric bootstrap, model-agnostic w.r.t. the estimator internals):
  1. Fit PromptEval at the given budget -> smooth prob matrix P = sigmoid(logits),
     and estimated per-template accuracy a_hat = Yhat.mean(-1).
  2. Treat P as a synthetic ground truth (known per-template accuracy P.mean(-1)).
     Repeatedly simulate Y_sim ~ Bernoulli(P), re-run the SAME budgeted sampling +
     estimator, and measure the spread-inflation factor infl = std(a_sim_hat)/std(P.mean-1).
  3. Deflate the real estimate around its mean by infl (linear shrinkage), recompute
     quantiles. If the mechanism is right, (a) infl ~ the observed 8x, and (b) deflating
     reduces tail error without harming the center.
"""
import sys, os, json, argparse
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "external", "prompteval")
sys.path.insert(0, os.path.join(REPO, "prompteval"))
import types
_stub = types.ModuleType("utils"); _stub.check_multicolinearity = lambda *a, **k: None
sys.modules["utils"] = _stub
from methods import ExtendedRaschModel, StratSample, sigmoid  # noqa: E402

QUANTILES = [5, 25, 50, 75, 95]


def fit_rasch(Y, budget, seed):
    """Fit the authors' ExtendedRasch at a given seen-cell budget.
    Return (a_hat, thetas, betas, intercept-free logits builder pieces)."""
    seen = np.zeros(Y.shape, bool)
    seen = StratSample(seen, budget, seed)
    m = ExtendedRaschModel()
    m.fit(seen, Y)
    a_hat = m.get_Y_hat().mean(-1)
    return a_hat, m.thetas.copy(), m.betas.copy()


def _P_from(thetas, betas):
    return sigmoid(thetas[:, None] + betas[None, :])


def infl_at_spread(thetas, betas, s, budget, B, rng):
    """Inflation factor of the estimator when the TRUE prompt-effect spread is
    deflated by factor s (thetas shrunk toward their mean by s)."""
    th_mu = thetas.mean()
    th_defl = th_mu + s * (thetas - th_mu)
    P = _P_from(th_defl, betas)
    truth = P.mean(-1)
    s_truth = truth.std()
    if s_truth < 1e-6:
        return 1.0
    facs = []
    for _ in range(B):
        Ysim = (rng.random(P.shape) < P).astype(float)
        a_sim, _, _ = fit_rasch(Ysim, budget, int(rng.integers(1 << 30)))
        facs.append(a_sim.std() / s_truth)
    return float(np.median(facs))


def deconv_shrink(a_hat, thetas, betas, budget, rng, B=12, iters=4):
    """Fixed-point deconvolution: find true-spread deflation s* such that simulating
    from a truth with that spread and re-estimating reproduces the observed inflated
    spread. Estimator inflates by infl(s*); we want infl(s*)*s* ~ 1 (obs spread = a_hat
    spread), i.e. s* = 1/infl(s*). Iterate s_{k+1} = 1/infl(s_k)."""
    s = 1.0
    for _ in range(iters):
        infl = infl_at_spread(thetas, betas, s, budget, B, rng)
        s = float(np.clip(1.0 / max(infl, 1e-6), 1e-3, 1.0))
    return s


def corrected_quantiles(a_hat, s):
    mu = a_hat.mean()
    a_corr = mu + s * (a_hat - mu)
    return np.percentile(a_corr, QUANTILES), a_corr


def run(bench="MMLU", n_tasks=6, n_llms=4, budgets=(200,), seeds=(0, 1, 2), B=15):
    import pickle
    Ys = pickle.load(open(os.path.join(REPO, "data", "Ys.pickle"), "rb"))
    tasks = list(Ys[bench].keys())[:n_tasks]
    rows = []
    for task in tasks:
        nl = min(n_llms, len(Ys[bench][task]))
        for llm in range(nl):
            Y = np.asarray(Ys[bench][task][llm], float)
            true_q = np.percentile(Y.mean(-1), QUANTILES)
            for budget in budgets:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    a_hat, thetas, betas = fit_rasch(Y, budget, seed)
                    orig_q = np.percentile(a_hat, QUANTILES)
                    s = deconv_shrink(a_hat, thetas, betas, budget, rng, B=B)
                    corr_q, _ = corrected_quantiles(a_hat, s)
                    for qi, q in enumerate(QUANTILES):
                        rows.append(dict(task=task, llm=llm, budget=budget, seed=seed,
                                         quantile=q, true=float(true_q[qi]),
                                         orig=float(orig_q[qi]), corr=float(corr_q[qi]),
                                         shrink=s))
        print(f"[{bench}] {task} done", flush=True)
    return rows


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    df["orig_abs"] = (df["orig"] - df["true"]).abs()
    df["corr_abs"] = (df["corr"] - df["true"]).abs()
    print("\n=== mean |error|: ORIG vs CORRECTED, by budget x quantile ===")
    g = df.groupby(["budget", "quantile"]).agg(orig=("orig_abs", "mean"),
                                               corr=("corr_abs", "mean"),
                                               shrink=("shrink", "mean"))
    g["reduction_%"] = (100 * (g["orig"] - g["corr"]) / g["orig"]).round(1)
    print(g.round(4).to_string())
    tail = df[df["quantile"].isin([5, 95])]
    print(f"\nTAIL (P5,P95) mean|err|: orig={tail.orig_abs.mean():.4f} -> "
          f"corr={tail.corr_abs.mean():.4f}  "
          f"({100*(tail.orig_abs.mean()-tail.corr_abs.mean())/tail.orig_abs.mean():.1f}% lower)")
    print(f"mean deconvolved shrink factor s = {df['shrink'].mean():.3f} "
          f"(=> implied spread deflation ~{1/df['shrink'].mean():.1f}x; cf. observed ~8x @200)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="MMLU")
    ap.add_argument("--n_tasks", type=int, default=6)
    ap.add_argument("--n_llms", type=int, default=4)
    ap.add_argument("--budgets", type=int, nargs="+", default=[200])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--B", type=int, default=15)
    ap.add_argument("--out", default="results/fix_deconv")
    a = ap.parse_args()
    rows = run(a.bench, a.n_tasks, a.n_llms, tuple(a.budgets), tuple(range(a.seeds)), a.B)
    os.makedirs("results", exist_ok=True)
    tag = f"{a.bench}_t{a.n_tasks}_l{a.n_llms}_b{'_'.join(map(str,a.budgets))}_s{a.seeds}"
    json.dump(rows, open(f"{a.out}_{tag}.json", "w"))
    summarize(rows)
    print(f"saved -> {a.out}_{tag}.json")
