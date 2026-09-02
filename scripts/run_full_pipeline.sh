#!/usr/bin/env bash
# Full CrashX pipeline: train CrashLogic-7B then run all experiment conditions.
set -euo pipefail

ROOT="/DATA/vaneet_2221cs15/vlm_new_gen"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pick GPU with the most free memory (physical index).
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' '{print $2, $1}' | sort -nr | head -1 | awk '{print $2}'
  )"
fi

PY="$ROOT/.venv/bin/python"
LOGDIR="$ROOT/results/logs"
mkdir -p "$LOGDIR" "$ROOT/outputs/crashlogic_7b_lora" "$ROOT/results"

echo "[$(date -Is)] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.free,utilization.gpu --format=csv || true

# Ensure splits exist
if [[ ! -f crashx/data/splits/test.jsonl ]]; then
  echo "[$(date -Is)] Building splits..."
  "$PY" -m crashx.data.process_ccd \
    --excel Car_Crash_Text_Dataset_ground_truth.xlsx \
    --video-dir video1500 \
    --out-dir crashx/data/splits
fi

# -------- Stage 1: QLoRA SFT (skip if adapter already present) --------
if [[ ! -f outputs/crashlogic_7b_lora/adapter_config.json ]]; then
  echo "[$(date -Is)] Starting QLoRA training..."
  "$PY" -m crashx.models.train_qlora \
    --train-jsonl crashx/data/splits/train.jsonl \
    --val-jsonl crashx/data/splits/val.jsonl \
    --output-dir outputs/crashlogic_7b_lora \
    --epochs 5 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --num-frames 8 \
    --max-side 224 \
    --max-seq-len 768 \
    --num-workers 0 \
    2>&1 | tee "$LOGDIR/train_qlora.log"
  echo "[$(date -Is)] Training finished."
else
  echo "[$(date -Is)] Found existing LoRA at outputs/crashlogic_7b_lora — skipping train."
fi

# -------- Stage 2: Full experiment table --------
echo "[$(date -Is)] Starting run_experiments (3 conditions + metrics)..."
"$PY" -m crashx.run_experiments \
  --test-jsonl crashx/data/splits/test.jsonl \
  --lora-path outputs/crashlogic_7b_lora \
  --results-dir results \
  --num-frames 8 \
  --max-new-tokens 256 \
  --alpha 1.0 \
  --neg-mode reverse \
  2>&1 | tee "$LOGDIR/run_experiments.log"

echo "[$(date -Is)] DONE. Tables:"
echo "---- markdown ----"
cat results/comparison.md || true
echo "---- latex ----"
cat results/comparison.tex || true
echo "[$(date -Is)] Summary JSON: results/comparison_summary.json"
