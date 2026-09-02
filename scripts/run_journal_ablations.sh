#!/usr/bin/env bash
# Run SEASON ablation conditions sequentially, then regenerate all journal tables.
set -euo pipefail

ROOT="/DATA/vaneet_2221cs15/vlm_new_gen"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' '{print $2, $1}' | sort -nr | head -1 | awk '{print $2}'
  )"
fi

PY="$ROOT/.venv/bin/python"
LOG="$ROOT/results/logs/journal_ablations.log"
mkdir -p "$ROOT/results/logs" "$ROOT/results/tables"

echo "[$(date -Is)] GPU=$CUDA_VISIBLE_DEVICES — starting SEASON ablations" | tee -a "$LOG"

"$PY" -m crashx.run_journal_experiments \
  --run-ablations \
  --test-jsonl crashx/data/splits/test.jsonl \
  --lora-path outputs/crashlogic_7b_lora \
  --results-dir results \
  --tables-dir results/tables \
  --num-frames 8 \
  --max-new-tokens 256 \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] DONE — tables at results/tables/all_tables.md" | tee -a "$LOG"
