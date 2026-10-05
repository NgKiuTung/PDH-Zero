#!/usr/bin/env bash
# Measure environment and batched-policy throughput for the Phase 2 capacity decision.
set -euo pipefail

MODE="smoke"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || { echo "mode must be smoke or full" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
RESULTS_ROOT="${PDH_RESULTS_ROOT:-$ROOT/results}/gate-c/runs"
ENGINE_GAMES=1000
ITERATIONS=30
BATCHES="256,512,1024,2048"
if [[ "$MODE" == "full" ]]; then
  ENGINE_GAMES=20000
  ITERATIONS=100
  BATCHES="256,512,1024,2048,4096,8192,16384"
fi

$PY scripts/run_step.py --gate C --results-root "$RESULTS_ROOT" --experiment-id C01_engine_throughput --timeout-seconds 7200 -- \
  $PY scripts/07_engine_benchmark.py --games "$ENGINE_GAMES" --bot random
$PY scripts/run_step.py --gate C --results-root "$RESULTS_ROOT" --experiment-id C02_policy_batch_scan --timeout-seconds 7200 -- \
  $PY scripts/08_policy_benchmark.py --device auto --batch-sizes "$BATCHES" --iterations "$ITERATIONS"

cat <<'EOF'
Gate C automatic measurements finished.
Use docs/P1-07_decision_report_template.md to combine C01/C02 with B04 metrics.jsonl.
EOF
