#!/usr/bin/env bash
# Execute Mini-PDH correctness, information isolation, PPO training, and evaluation.
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
RESULTS_ROOT="${PDH_RESULTS_ROOT:-$ROOT/results}/gate-b/runs"
ENGINE_GAMES=500
EVAL_SEEDS=50
TIMEOUT=3600
if [[ "$MODE" == "full" ]]; then
  ENGINE_GAMES=5000
  EVAL_SEEDS=5000
  TIMEOUT=43200
fi

$PY scripts/run_step.py --gate B --results-root "$RESULTS_ROOT" --experiment-id B01_unit_tests --timeout-seconds 1800 -- \
  $PY -m unittest discover -s tests -v
$PY scripts/run_step.py --gate B --results-root "$RESULTS_ROOT" --experiment-id B02_engine_invariants --timeout-seconds 1800 -- \
  $PY scripts/07_engine_benchmark.py --games "$ENGINE_GAMES" --bot random
$PY scripts/run_step.py --gate B --results-root "$RESULTS_ROOT" --experiment-id B03_hidden_info_isolation --timeout-seconds 900 -- \
  $PY scripts/11_check_hidden_info_isolation.py --device auto

run_training_and_evaluation() {
  local label="$1"
  local config="$2"
  local seed="$3"
  local train_run
  train_run="$($PY scripts/run_step.py --gate B --results-root "$RESULTS_ROOT" \
    --experiment-id "B04_${label}_seed${seed}" --timeout-seconds "$TIMEOUT" -- \
    $PY scripts/09_train_smoke.py --config "$config" --seed "$seed")"
  local checkpoint="$train_run/artifacts/training/checkpoint_latest.pt"
  [[ -f "$checkpoint" ]] || { echo "Training checkpoint missing: $checkpoint" >&2; exit 4; }
  $PY scripts/run_step.py --gate B --results-root "$RESULTS_ROOT" \
    --experiment-id "B05_eval_${label}_seed${seed}" --timeout-seconds "$TIMEOUT" -- \
    $PY scripts/10_evaluate.py --checkpoint "$checkpoint" --device auto \
      --wall-seeds "$EVAL_SEEDS" --opponent heuristic
  echo "Completed $label seed=$seed: $train_run"
}

if [[ "$MODE" == "smoke" ]]; then
  run_training_and_evaluation "privileged_smoke" "configs/ppo_smoke.yaml" 20260929
else
  for seed in 20260929 3407 9102; do
    run_training_and_evaluation "plain" "configs/ppo_phase1_plain_critic.yaml" "$seed"
    run_training_and_evaluation "privileged" "configs/ppo_phase1.yaml" "$seed"
  done
fi

echo "Gate B finished. Preserve every failed and successful run directory."
