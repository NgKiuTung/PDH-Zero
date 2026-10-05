#!/usr/bin/env bash
# Create or verify a sentinel used to test stop/restart persistence.
set -euo pipefail

MODE="${1:-}"
DEFAULT_ROOT="${PDH_PERSIST_ROOT:-${PDH_PROJECT_ROOT:-$(pwd)}/state/persistence}"
TARGET_DIR="${2:-$DEFAULT_ROOT/storage-sentinel}"
mkdir -p "$TARGET_DIR"
SENTINEL="$TARGET_DIR/sentinel.txt"
DIGEST="$TARGET_DIR/sentinel.sha256"

case "$MODE" in
  create)
    if [[ -e "$SENTINEL" || -e "$DIGEST" ]]; then
      echo "Refusing to overwrite an existing persistence sentinel: $TARGET_DIR" >&2
      exit 2
    fi
    python - "$SENTINEL" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import secrets
import sys

Path(sys.argv[1]).write_text(
    f"created_at={datetime.now(timezone.utc).isoformat()}\n"
    f"nonce={secrets.token_hex(32)}\n",
    encoding="utf-8",
)
PY
    sha256sum "$SENTINEL" > "$DIGEST"
    sync
    echo "CREATED $SENTINEL"
    ;;
  verify)
    [[ -f "$SENTINEL" && -f "$DIGEST" ]] || {
      echo "Persistence sentinel is missing: $TARGET_DIR" >&2
      exit 3
    }
    (cd "$TARGET_DIR" && sha256sum -c "$(basename "$DIGEST")")
    echo "VERIFIED $SENTINEL"
    ;;
  *)
    echo "Usage: $0 create|verify [target-dir]" >&2
    exit 2
    ;;
esac
