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

echo "== 5. Figures =="
python src/figures.py
python src/figures_fix.py

echo "== 6. Paper (requires tectonic: conda install -c conda-forge tectonic) =="
if command -v tectonic >/dev/null 2>&1; then (cd paper && tectonic main.tex) && echo "paper/main.pdf built"; else echo "tectonic not found; skipping PDF"; fi

echo "ALL DONE. Key outputs: results/*.json, paper/figures/*.png, paper/main.pdf"
