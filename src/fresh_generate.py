"""
Generate FRESH template x item correctness matrices locally, independent of the PromptEval
release, to test whether the center-vs-tail over-dispersion is a property of the estimator
(reappears on our own data) rather than an artifact of the authors' data pipeline.

- Models: small open LLMs runnable on M5 unified memory (Qwen2.5 family).
- Prompts: an independently-constructed bank of format templates (instruction, question tag,
  option format, separators, answer cue, few-shot count).
- Scoring: MMLU multiple-choice via next-token log-probs of the answer letters (A-D).
Everything runs on MPS, CPU-only fallback, $0.
"""
import os, sys, json, argparse, itertools, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LETTERS = ["A", "B", "C", "D"]

# ---- independent template bank -------------------------------------------------
INSTRUCTIONS = [
    "",
    "The following is a multiple choice question. Choose the correct answer.\n\n",
    "Answer the multiple-choice question below by selecting the best option.\n\n",
    "You are an expert. Read the question and pick the single best answer.\n\n",
    "Solve the following problem.\n\n",
]
Q_TAGS = ["", "Question: ", "Q: ", "Problem: "]
OPT_FMT = ["{L}. {t}", "({L}) {t}", "{L}) {t}", "{L}: {t}"]
SEP = ["\n", "\n\n"]
CUES = ["\nAnswer:", "\nThe answer is:", "\nCorrect option:", "\nAnswer (A-D):"]
SHOTS = [0, 1, 3]


def build_templates(n, seed=0):
    combos = list(itertools.product(range(len(INSTRUCTIONS)), range(len(Q_TAGS)),
                                    range(len(OPT_FMT)), range(len(SEP)),
                                    range(len(CUES)), range(len(SHOTS))))
    rng = random.Random(seed); rng.shuffle(combos)
    return combos[:n]


def render(item, combo, shots_pool):
    ins, qt, of, sp, cue, sh = (INSTRUCTIONS[combo[0]], Q_TAGS[combo[1]], OPT_FMT[combo[2]],
                                SEP[combo[3]], CUES[combo[4]], SHOTS[combo[5]])
    def one(q, choices):
        opts = sp.join(of.format(L=L, t=t) for L, t in zip(LETTERS, choices))
        return f"{qt}{q}{sp}{opts}{cue}"
    prefix = ins
    for ex in shots_pool[:sh]:
        prefix += one(ex["question"], ex["choices"]) + " " + LETTERS[ex["answer"]] + "\n\n"
    return prefix + one(item["question"], item["choices"])


@torch.no_grad()
def score_batch(model, tok, prompts, letter_ids):
    """Batched next-token letter log-probs. Left-padded so last position is index -1."""
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    logits = model(**enc).logits[:, -1, :]                 # (B, vocab)
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp[:, letter_ids].argmax(dim=-1).cpu().numpy()  # (B,) predicted letter index


def run(model_name, subjects, n_templates, n_items, seed=0, smoke=False, batch=32, out_path=None):
    # Resume support: load any already-completed subjects and skip them.
    out = {}
    if out_path and os.path.exists(out_path):
        out = json.load(open(out_path))
        done = [s for s in subjects if s in out]
        if done:
            print(f"resuming: {len(done)} subjects already done, skipping {done}", flush=True)
    todo = [s for s in subjects if s not in out]
    if not todo:
        print("all subjects already generated; nothing to do", flush=True)
        return out
    print(f"loading {model_name} on {DEVICE} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if DEVICE == "mps" else torch.float32).to(DEVICE).eval()
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids[0] for L in LETTERS]
    templates = build_templates(n_templates, seed)
    if smoke:
        templates = templates[:4]
    for subj in todo:
        ds = load_dataset("cais/mmlu", subj)
        test, dev = ds["test"], ds["dev"]
        items = [test[i] for i in range(min(n_items, len(test)))]
        golds = np.array([it["answer"] for it in items])
        shots_pool = [dev[i] for i in range(len(dev))]
        Y = np.zeros((len(templates), len(items)), dtype=np.int8)
        for ti, combo in enumerate(templates):
            prompts = [render(it, combo, shots_pool) for it in items]
            preds = np.concatenate([score_batch(model, tok, prompts[i:i+batch], letter_ids)
                                    for i in range(0, len(prompts), batch)])
            Y[ti] = (preds == golds).astype(np.int8)
            if ti % 10 == 0:
                print(f"  {subj}: template {ti+1}/{len(templates)} "
                      f"(acc {Y[ti].mean():.3f})", flush=True)
        out[subj] = Y.tolist()
        acc = Y.mean(1)
        print(f"[{model_name}] {subj}: Y {Y.shape}, mean {Y.mean():.3f}, "
              f"per-template acc [{acc.min():.3f},{acc.max():.3f}] spread(P95-P5) "
              f"{np.percentile(acc,95)-np.percentile(acc,5):.3f}", flush=True)
        if out_path:  # checkpoint after every subject so a pause never loses work
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            json.dump(out, open(out_path, "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--subjects", nargs="+",
                    default=["marketing", "high_school_psychology", "nutrition", "sociology"])
    ap.add_argument("--n_templates", type=int, default=40)
    ap.add_argument("--n_items", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs("results/fresh", exist_ok=True)
    tag = a.model.split("/")[-1] + ("_smoke" if a.smoke else "")
    out_path = f"results/fresh/Y_{tag}.json"
    out = run(a.model, a.subjects, a.n_templates, a.n_items,
              smoke=a.smoke, batch=a.batch, out_path=out_path)
    json.dump(out, open(out_path, "w"))
    print(f"saved -> {out_path}  ({len(out)} subjects)")
