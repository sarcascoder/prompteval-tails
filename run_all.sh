#!/usr/bin/env bash
# One-command reproduction of all results and figures.
# Usage: bash run_all.sh   (activate the conda `research` env first, or it will try to)
set -euo pipefail
cd "$(dirname "$0")"

if ! python -c "import mlx" 2>/dev/null && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate research || true
fi

# 0. Fetch the authors' code + released data (idempotent).
if [ ! -d external/prompteval ]; then
  git clone --depth 1 https://github.com/felipemaiapolo/prompteval external/prompteval
fi

mkdir -p results paper/figures

echo "== 1. Exact reproduction + signed/spread instrumentation (MMLU, BBH, LMentry) =="
for B in MMLU BBH LMentry; do
  python src/repro_core.py --bench "$B" --seeds 5 --out results/repro_full
done

echo "== 2. Fidelity check + center-vs-tail diagnosis (prints exact-match + CIs/tests) =="
python src/mechanism.py

echo "== 3. Controlled synthetic study (known-truth over-dispersion + oracle) =="
python src/synthetic.py

echo "== 4. Split-half correction across benchmarks/budgets =="
python src/fix_splithalf.py --bench MMLU    --budget 200  --seeds 3
python src/fix_splithalf.py --bench BBH     --budget 200  --seeds 3
python src/fix_splithalf.py --bench BBH     --budget 1600 --seeds 3
python src/fix_splithalf.py --bench LMentry --budget 200  --seeds 3
python src/fix_splithalf.py --bench LMentry --budget 1600 --seeds 3

echo "== 4b. Fresh-data replication (optional; needs torch+transformers+datasets, ~1.5h on MPS) =="
echo "   Skips automatically if torch is not installed."
if python -c "import torch, transformers, datasets" 2>/dev/null; then
  python src/fresh_generate.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --subjects marketing high_school_psychology nutrition sociology management human_aging \
    --n_templates 40 --n_items 100 --batch 32
  python src/fresh_analyze.py
else
  echo "   (torch/transformers/datasets not installed -> skipping fresh-data step)"
fi

echo "== 5. Figures =="
python src/figures.py
python src/figures_fix.py

echo "== 6. Paper (requires tectonic: conda install -c conda-forge tectonic) =="
if command -v tectonic >/dev/null 2>&1; then (cd paper && tectonic main.tex) && echo "paper/main.pdf built"; else echo "tectonic not found; skipping PDF"; fi

echo "ALL DONE. Key outputs: results/*.json, paper/figures/*.png, paper/main.pdf"
