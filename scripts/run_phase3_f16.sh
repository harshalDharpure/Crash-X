#!/usr/bin/env bash
# Phase 3 fast path: 16-frame QLoRA (3 epochs) → Greedy+TCD eval → tables
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
PY="${ROOT}/.venv/bin/python"
LOGDIR="${ROOT}/results/logs"
mkdir -p "$LOGDIR"
OUT="${ROOT}/outputs/crashlogic_7b_lora_f16"
LOG="${LOGDIR}/phase3_f16_retrain.log"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '{gsub(/ /,"",$2); gsub(/ /,"",$3); print $1","$2","$3}' \
    | sort -t',' -k2,2nr -k3,3n | head -1 | cut -d',' -f1
}

GPU="$(pick_gpu)"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "Launching Phase 3 on GPU ${GPU}; log → ${LOG}"

nohup bash -c "
set -euo pipefail
cd '$ROOT'
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU
PY='$PY'
OUT='$OUT'
echo '=== Phase 3 start '\$(date -Is)' GPU=$GPU ==='
echo '>>> Train 16-frame CrashLogic (3 epochs, seed 42)'
\"\$PY\" -m crashx.models.train_qlora \
  --train-jsonl crashx/data/splits/train.jsonl \
  --val-jsonl crashx/data/splits/val.jsonl \
  --output-dir \"\$OUT\" \
  --num-frames 16 --max-side 224 --max-seq-len 1024 \
  --epochs 3 --batch-size 1 --grad-accum 8 --seed 42
echo '>>> Infer Greedy-f16'
\"\$PY\" -m crashx.infer_named --lora-path \"\$OUT\" --name CrashLogic-7B-Greedy-f16 \
  --decode greedy --num-frames 16
echo '>>> Infer TCD-a0.5-f16'
\"\$PY\" -m crashx.infer_named --lora-path \"\$OUT\" --name CrashLogic-7B-SEASON-a0.5-f16 \
  --decode season --alpha 0.5 --num-frames 16
echo '>>> Tables'
\"\$PY\" -m crashx.run_journal_experiments --tables-only --n-bootstrap 1000
echo '=== Phase 3 done '\$(date -Is)' ==='
" >> "$LOG" 2>&1 &

echo $! > "${LOGDIR}/phase3_f16.pid"
echo "PID $(cat ${LOGDIR}/phase3_f16.pid)"
echo "Monitor: tail -f ${LOG}"
