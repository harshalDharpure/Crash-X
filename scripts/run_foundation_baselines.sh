#!/usr/bin/env bash
# CrashX Option A — Phase 1: Foundation model baseline blitz (parallel GPUs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.
PY="${ROOT}/.venv/bin/python"
RESULTS="${ROOT}/results"
LOGDIR="${RESULTS}/logs"
mkdir -p "$LOGDIR"

echo "[Option A Phase 1] Launching foundation baselines on 150-video test split"

# InternVL on GPU 0
CUDA_VISIBLE_DEVICES=0 "$PY" -m crashx.run_foundation_baselines \
  --only-condition ZeroShot-InternVL2.5-8B \
  --results-dir "$RESULTS" \
  > "${LOGDIR}/baseline_internvl.log" 2>&1 &
PID_INTERNVL=$!
echo "  InternVL2.5-8B → GPU 0 (PID $PID_INTERNVL)"

# LLaVA-Video on GPU 1
CUDA_VISIBLE_DEVICES=1 "$PY" -m crashx.run_foundation_baselines \
  --only-condition ZeroShot-LLaVA-Video-7B \
  --results-dir "$RESULTS" \
  > "${LOGDIR}/baseline_llava_video.log" 2>&1 &
PID_LLAVA=$!
echo "  LLaVA-NeXT-Video-7B → GPU 1 (PID $PID_LLAVA)"

echo ""
echo "Monitor:"
echo "  tail -f ${LOGDIR}/baseline_internvl.log"
echo "  tail -f ${LOGDIR}/baseline_llava_video.log"
echo ""
echo "When complete, regenerate tables:"
echo "  PYTHONPATH=. $PY -m crashx.run_journal_experiments --tables-only"

wait "$PID_INTERNVL" && echo "InternVL baseline done." || echo "InternVL baseline failed (see ${LOGDIR}/baseline_internvl.log)."
wait "$PID_LLAVA" && echo "LLaVA-Video baseline done." || echo "LLaVA-Video baseline failed (see ${LOGDIR}/baseline_llava_video.log)."

# Regenerate tables only if at least one new baseline succeeded
if [[ -f "${RESULTS}/ZeroShot-InternVL2.5-8B_predictions.json" ]] \
   || [[ -f "${RESULTS}/ZeroShot-LLaVA-Video-7B_predictions.json" ]] \
   || [[ -f "${RESULTS}/ZeroShot-Qwen2-VL-2B_predictions.json" ]]; then
  echo "[Option A Phase 1] Regenerating tables with bootstrap CIs..."
  "$PY" -m crashx.run_journal_experiments --tables-only
else
  echo "[Option A Phase 1] Skipping table regen — no new baseline predictions found."
fi
echo "Done. See results/tables/"
