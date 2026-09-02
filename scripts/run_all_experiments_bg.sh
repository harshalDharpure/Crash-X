#!/usr/bin/env bash
# CrashX Option A — launcher: run ALL experiments in background on freest GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${ROOT}/results/logs"
mkdir -p "$LOGDIR"
MASTER_LOG="${LOGDIR}/option_a_all_experiments.log"
PIDFILE="${LOGDIR}/option_a_all_experiments.pid"
WORKER="${ROOT}/scripts/run_all_experiments_worker.sh"

pick_freest_gpu() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '{gsub(/ /,"",$2); gsub(/ /,"",$3); print $1","$2","$3}' \
    | sort -t',' -k2,2nr -k3,3n \
    | head -1 \
    | cut -d',' -f1
}

if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(cat "$PIDFILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Experiment bag already running (PID ${OLD_PID})"
    echo "Monitor: tail -f ${MASTER_LOG}"
    exit 0
  fi
fi

GPU="$(pick_freest_gpu)"
echo "Starting Option A experiment bag on GPU ${GPU} (most free VRAM)"
nohup env CUDA_VISIBLE_DEVICES="${GPU}" bash "$WORKER" >> "$MASTER_LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "  PID: $(cat "$PIDFILE")"
echo "  Log: ${MASTER_LOG}"
echo "  Monitor: tail -f ${MASTER_LOG}"
