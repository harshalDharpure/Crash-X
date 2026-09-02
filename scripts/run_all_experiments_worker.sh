#!/usr/bin/env bash
# CrashX Option A — worker: sequential foundation baselines + table regen on one GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.
PY="${ROOT}/.venv/bin/python"

echo "============================================================"
echo "CrashX Option A — worker started: $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
echo "============================================================"

BASELINES=(
  "ZeroShot-Qwen2-VL-2B"
  "ZeroShot-LLaVA-Video-7B"
  "ZeroShot-Qwen2.5-VL-3B"
)

EXPECTED_TEST_VIDEOS=150

is_complete_predictions() {
  local pred_file="$1"
  [[ -f "$pred_file" ]] || return 1
  local count
  count="$("$PY" -c "import json; print(len(json.load(open(r'''${pred_file}'''))))")"
  [[ "$count" -ge "$EXPECTED_TEST_VIDEOS" ]]
}

for COND in "${BASELINES[@]}"; do
  PRED="${ROOT}/results/${COND}_predictions.json"
  if is_complete_predictions "$PRED"; then
    echo "[skip] ${COND} — complete predictions exist (${EXPECTED_TEST_VIDEOS} videos)"
    continue
  fi
  if [[ -f "$PRED" ]]; then
    echo "[rerun] ${COND} — incomplete predictions found, removing partial file"
    rm -f "$PRED" "${ROOT}/results/${COND}_metrics.json"
  fi
  echo ""
  echo ">>> $(date -Is) Running: ${COND}"
  if "$PY" -m crashx.run_foundation_baselines \
      --only-condition "$COND" \
      --results-dir "${ROOT}/results"; then
    echo ">>> $(date -Is) SUCCESS: ${COND}"
  else
    echo ">>> $(date -Is) FAILED: ${COND} (continuing)"
  fi
done

echo ""
echo ">>> $(date -Is) Regenerating journal tables (bootstrap N=1000)"
"$PY" -m crashx.run_journal_experiments --tables-only --n-bootstrap 1000

echo ""
echo "============================================================"
echo "Worker finished: $(date -Is)"
echo "Tables: ${ROOT}/results/tables/all_tables.md"
echo "============================================================"
