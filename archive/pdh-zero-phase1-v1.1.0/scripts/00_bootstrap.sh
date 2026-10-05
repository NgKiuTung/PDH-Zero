
#!/usr/bin/env bash
# Create a backend-specific activation layer without replacing the image Torch build.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
BASE_PYTHON="$(command -v python3)"

"$BASE_PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(
        f"Python 3.12+ is required by this release; observed {sys.version.split()[0]}"
    )
PY

BACKEND="$($BASE_PYTHON - <<'PY'
import torch
if not torch.cuda.is_available():
    print("cpu")
elif torch.version.hip is not None:
    print("rocm")
else:
    print("cuda")
PY
)"
ENV_DIR=".pdh-env-${BACKEND}"
CACHE_ROOT="${PDH_CACHE_ROOT:-$PROJECT_ROOT/.cache}"
TMP_ROOT="${PDH_TMP_ROOT:-$PROJECT_ROOT/tmp/runtime}"
mkdir -p "$ENV_DIR/bin" "$CACHE_ROOT/xdg" "$CACHE_ROOT/torch" "$CACHE_ROOT/pip" "$TMP_ROOT"
cat > "$ENV_DIR/bin/activate" <<EOF
# PDH-Zero backend-specific activation; source this file from Bash.
export PDH_PROJECT_ROOT="$PROJECT_ROOT"
export PDH_BACKEND="$BACKEND"
export PDH_CACHE_ROOT="$CACHE_ROOT"
export PDH_TMP_ROOT="$TMP_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$TMP_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
ln -sfn "$ENV_DIR" .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python - <<'PY'
import torch
import pdh_phase1
backend = "cpu" if not torch.cuda.is_available() else ("rocm" if torch.version.hip else "cuda")
payload = {
    "package_version": pdh_phase1.__version__,
    "torch": torch.__version__,
    "backend": backend,
    "hip": torch.version.hip,
    "cuda_runtime": torch.version.cuda,
    "accelerator_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
print(payload)
PY
printf '\nBootstrap complete for backend=%s. Activate with: source .venv/bin/activate\n' "$BACKEND"
