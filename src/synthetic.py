"""
Controlled synthetic experiment: simulate from a KNOWN Rasch ground truth with a
controlled true prompt-spread, run the authors' PromptEval estimator, and confirm the
over-dispersion appears and scales as predicted. Because the true spread is known
exactly (no noisy empirical ground truth), this isolates the estimator's finite-sample
tail inflation from any property of the released data, and lets us check the
deconvolution fix against ground truth.
"""
import sys, os, json, numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "external", "prompteval")
sys.path.insert(0, os.path.join(REPO, "prompteval"))
import types
_stub = types.ModuleType("utils"); _stub.check_multicolinearity = lambda *a, **k: None
sys.modules["utils"] = _stub
from methods import ExtendedRaschModel, StratSample, sigmoid  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from fix_deconv import deconv_shrink  # noqa: E402

QS = [5, 25, 50, 75, 95]


def make_world(P, I, sigma_theta, sigma_beta, rng):
    theta = rng.normal(0, sigma_theta, P)
    beta = rng.normal(0, sigma_beta, I)
    Pmat = sigmoid(theta[:, None] + beta[None, :])
    a_true = Pmat.mean(1)                    # population per-template accuracy (known truth)
    return Pmat, a_true


def fit_est(Yfull, budget, seed):
    seen = StratSample(np.zeros(Yfull.shape, bool), budget, seed)
    m = ExtendedRaschModel(); m.fit(seen, Yfull)
    return m.get_Y_hat().mean(-1), m.thetas.copy(), m.betas.copy()


def run(P=100, I=100, sigma_beta=1.0, budget=200,
        sigma_thetas=(0.2, 0.5, 1.0), n_worlds=30, B_boot=10):
    rows = []
    for st in sigma_thetas:
        for w in range(n_worlds):
            rng = np.random.default_rng(1000 * int(st * 100) + w)
            Pmat, a_true = make_world(P, I, st, sigma_beta, rng)
            true_q = np.percentile(a_true, QS)
            true_spread = true_q[-1] - true_q[0]
            Yfull = (rng.random((P, I)) < Pmat).astype(float)
            a_hat, th, be = fit_est(Yfull, budget, w)
            est_q = np.percentile(a_hat, QS)
            est_spread = est_q[-1] - est_q[0]
            s = deconv_shrink(a_hat, th, be, budget, rng, B=B_boot)
            mu = a_hat.mean(); a_corr = mu + s * (a_hat - mu)
            corr_q = np.percentile(a_corr, QS)
            rows.append(dict(sigma_theta=st, world=w,
                             true_spread=float(true_spread), est_spread=float(est_spread),
                             spread_ratio=float(est_spread / true_spread) if true_spread > 1e-6 else np.nan,
                             corr_spread=float(corr_q[-1] - corr_q[0]),
                             tail_err_orig=float((abs(est_q[0] - true_q[0]) + abs(est_q[-1] - true_q[-1])) / 2),
                             tail_err_corr=float((abs(corr_q[0] - true_q[0]) + abs(corr_q[-1] - true_q[-1])) / 2),
                             center_err=float(abs(est_q[2] - true_q[2])),
                             shrink=float(s)))
    return rows


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby("sigma_theta").agg(
        true_spread=("true_spread", "mean"),
        spread_ratio=("spread_ratio", "median"),
        corr_spread_ratio=("corr_spread", "mean"),
        tail_orig=("tail_err_orig", "mean"),
        tail_corr=("tail_err_corr", "mean"),
        center=("center_err", "mean"),
        shrink=("shrink", "mean"))
    g["corr_spread_ratio"] = (g["corr_spread_ratio"] / g["true_spread"])
    g["tail_reduction_%"] = (100 * (g["tail_orig"] - g["tail_corr"]) / g["tail_orig"]).round(1)
    print("KNOWN-truth synthetic study (P=100,I=100,budget=200); over-dispersion vs true spread")
    print(g.round(3).to_string())
    return df


if __name__ == "__main__":
    rows = run()
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/synthetic.json", "w"))
    summarize(rows)
