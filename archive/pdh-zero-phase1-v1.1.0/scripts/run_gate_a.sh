
#!/usr/bin/env bash
# Execute Gate A automatic checks. Stop/restart persistence remains a manual two-step gate.
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
RESULTS_ROOT="${PDH_RESULTS_ROOT:-$ROOT/results}/gate-a/runs"
PERSIST_BASE="${PDH_PERSIST_ROOT:-$ROOT/state/persistence}"
SOAK_SECONDS=5
[[ "$MODE" == "full" ]] && SOAK_SECONDS=3600
BACKEND="$($PY - <<'PY'
import torch
if not torch.cuda.is_available():
    print("cpu")
elif torch.version.hip is not None:
    print("rocm")
else:
    print("cuda")
PY
)"
PERSIST_DIR="$PERSIST_BASE/$BACKEND"
SMOKE_BATCH=4096
SOAK_BATCH=4096
if [[ "$BACKEND" == "cpu" ]]; then
  SMOKE_BATCH=256
  SOAK_BATCH=64
fi

if [[ "$BACKEND" == "cuda" ]]; then
  $PY scripts/run_step.py --gate A --experiment-id A00_cuda_blackwell_preflight     --results-root "$RESULTS_ROOT" --timeout-seconds 600 --     $PY scripts/00_cuda_blackwell_preflight.py
fi
$PY scripts/run_step.py --gate A --experiment-id A01_system_info   --results-root "$RESULTS_ROOT" --timeout-seconds 300 --   bash scripts/01_collect_system_info.sh
$PY scripts/run_step.py --gate A --experiment-id A02_fp32_smoke   --results-root "$RESULTS_ROOT" --timeout-seconds 900 --   $PY scripts/03_torch_backend_smoke.py --device auto --dtype float32 --steps 50 --batch-size "$SMOKE_BATCH"
$PY scripts/run_step.py --gate A --experiment-id A02_bf16_smoke   --results-root "$RESULTS_ROOT" --timeout-seconds 900 --   $PY scripts/03_torch_backend_smoke.py --device auto --dtype bfloat16 --steps 30 --batch-size "$SMOKE_BATCH"
$PY scripts/run_step.py --gate A --experiment-id A03_precision   --results-root "$RESULTS_ROOT" --timeout-seconds 900 --   $PY scripts/04_precision_test.py --device auto
$PY scripts/run_step.py --gate A --experiment-id A04_checkpoint   --results-root "$RESULTS_ROOT" --timeout-seconds 900 --   $PY scripts/05_checkpoint_test.py --device auto
if [[ ! -e "$PERSIST_DIR/storage-sentinel/sentinel.txt" ]]; then
  $PY scripts/run_step.py --gate A --experiment-id A05_storage_create     --results-root "$RESULTS_ROOT" --timeout-seconds 120 --     bash scripts/02_check_storage.sh create "$PERSIST_DIR/storage-sentinel"
else
  echo "Existing persistence sentinel retained: $PERSIST_DIR"
fi
$PY scripts/run_step.py --gate A --experiment-id A06_stability_soak   --results-root "$RESULTS_ROOT" --timeout-seconds "$((SOAK_SECONDS + 300))" --   $PY scripts/06_stability_soak.py --device auto --duration-seconds "$SOAK_SECONDS" --batch-size "$SOAK_BATCH"

echo "Gate A automatic checks finished."
echo "A05 remains UNVERIFIED until the instance is stopped, restarted, and this command passes:"
echo "  bash scripts/02_check_storage.sh verify $PERSIST_DIR/storage-sentinel"
