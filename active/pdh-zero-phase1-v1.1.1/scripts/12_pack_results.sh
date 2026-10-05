#!/usr/bin/env bash
# Package one immutable result directory using only the preinstalled Python runtime.
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
[[ "$SCRIPT_DIR" != "$SCRIPT_PATH" ]] || SCRIPT_DIR="."
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/12_pack_results.py" "$@"
