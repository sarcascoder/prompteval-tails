"""
Mechanism + decision-relevance analysis of the center-vs-tail gap.

Questions:
 (Q1) Direction of tail bias, robustly (median, not mean; guard tiny denominators).
 (Q2) Is the estimated prompt-accuracy distribution over- or under-dispersed, and
      does it invert PromptEval's value proposition (worst-case / spread / best-prompt)?
 (Q3) Decision-relevant errors a practitioner actually makes:
        - worst-case (P5) mis-estimate  -> over/under-pessimistic?
        - prompt-sensitivity spread (P95-P5) mis-estimate
        - best-prompt selection regret (true acc gap of the prompt PromptEval calls best)
All from the released MMLU data + authors' estimator (re-analysis only).
"""
import json, numpy as np, pandas as pd
from scipy import stats

d = json.load(open("results/repro_full_MMLU_tNone_lNone_s5.json"))
df = pd.DataFrame(d["recs"]); sp = pd.DataFrame(d["spread"])

def boot_ci(x, f=np.mean, n=2000, seed=0):
    r = np.random.RandomState(seed); x = np.asarray(x); idx = np.arange(len(x))
    b = [f(x[r.choice(idx, len(idx), True)]) for _ in range(n)]
    return f(x), np.percentile(b, 2.5), np.percentile(b, 97.5)

print("="*78)
print("Q1/Q2: SIGNED tail error & spread (PromptEval), by budget — is the estimated")
print("       prompt-accuracy distribution over-dispersed? (robust)")
print("="*78)
for b in [200, 400, 800, 1600]:
    s = df[(df.method=="prompteval") & (df.budget==b)]
    q5 = s[s["quantile"]==5]["signed"]; q95 = s[s["quantile"]==95]["signed"]
    m5,l5,u5 = boot_ci(q5.values); m95,l95,u95 = boot_ci(q95.values)
    print(f" budget {b:>4}:  signed P5 = {m5:+.3f} [{l5:+.3f},{u5:+.3f}]   "
          f"signed P95 = {m95:+.3f} [{l95:+.3f},{u95:+.3f}]")
print(" (P5<0 => estimated worst-prompt too LOW = over-pessimistic; "
      "P95>0 => best-prompt too HIGH = over-optimistic; both => spread inflated)")

# robust spread ratio: restrict to cells whose TRUE spread is non-trivial (>=2 acc pts)
print("\n" + "="*78)
print("Q2 (robust): estimated spread / true spread, restricted to cells with")
print("             true (P95-P5) >= 0.02, using MEDIAN ratio (+ boot CI)")
print("="*78)
for b in [200, 400, 800, 1600]:
    s = sp[(sp.method=="prompteval") & (sp.budget==b)].copy()
    s = s[s.true_spread >= 0.02]
    ratio = (s.est_spread / s.true_spread).values
    med, lo, hi = boot_ci(ratio, np.median)
    frac_over = float((ratio > 1).mean())
    print(f" budget {b:>4}:  median spread ratio = {med:.2f} [{lo:.2f},{hi:.2f}]   "
          f"(% cells over-dispersed = {100*frac_over:.0f}%, n={len(s)})")

print("\n" + "="*78)
print("Q3: DECISION-RELEVANT errors at the headline budget (200) vs a full single-")
print("    prompt eval's worth of extra budget. PromptEval vs Baseline (empirical acc).")
print("="*78)
for b in [200, 1600]:
    print(f"\n --- budget {b} ---")
    for meth in ["prompteval", "baseline"]:
        s = df[(df.method==meth) & (df.budget==b)]
        # worst-case (P5) signed error
        wc = s[s["quantile"]==5]["signed"].values
        # spread error via sp
        ss = sp[(sp.method==meth) & (sp.budget==b)]
        spread_err = (ss.est_spread - ss.true_spread).values
        regret = ss.best_regret.values
        mwc,lwc,uwc = boot_ci(wc)
        mre,lre,ure = boot_ci(regret)
        print(f"  {meth:<11}: worst-case(P5) signed err = {mwc:+.3f} [{lwc:+.3f},{uwc:+.3f}] | "
              f"best-prompt regret = {mre:.3f} [{lre:.3f},{ure:.3f}]")

# Is tail error significantly reduced but still dominant? center vs tail across budgets
print("\n" + "="*78)
print("Q: tail/center abs-error ratio across budgets (PromptEval) — does more budget")
print("   fix it? (paired Wilcoxon tail>center per subjectxLLM)")
print("="*78)
for b in [200, 400, 800, 1600]:
    s = df[(df.method=="prompteval") & (df.budget==b)]
    g = s.groupby(["task","llm","quantile"])["absol"].mean().unstack("quantile")
    tail = (g[5]+g[95]).values/2; center = g[50].values
    ratio = tail.mean()/center.mean()
    _,p = stats.wilcoxon(tail, center)
    print(f" budget {b:>4}: tail/center = {ratio:.2f}x   (median center={np.median(center):.3f}, "
          f"median tail={np.median(tail):.3f}, p={p:.1e})")
