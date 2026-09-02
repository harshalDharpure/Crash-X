#!/usr/bin/env bash
# CrashX Option A — Phase 1+2: Bootstrap stats + foundation baselines
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.
PY="${ROOT}/.venv/bin/python"

echo "=== Phase 2: Regenerate tables (TCD α=0.5 primary + bootstrap CIs) ==="
"$PY" -m crashx.run_journal_experiments --tables-only --n-bootstrap 1000

echo ""
echo "=== Phase 1: Launch foundation baselines (if not already run) ==="
bash "${ROOT}/scripts/run_foundation_baselines.sh"
