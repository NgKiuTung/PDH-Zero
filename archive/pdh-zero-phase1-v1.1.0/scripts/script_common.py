"""Shared path and JSON helpers for Phase 1 command-line experiments."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def artifact_dir(explicit: str | None = None, child: str | None = None) -> Path:
    """Resolve and create an artifact directory without overwriting a prior run."""
    if explicit:
        target = Path(explicit)
    elif os.environ.get("PDH_RUN_DIR"):
        target = Path(os.environ["PDH_RUN_DIR"]) / "artifacts"
    else:
        target = PROJECT_ROOT / "results" / "manual"
    if child:
        target = target / child
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: Path, payload: Any) -> None:
    """Write readable UTF-8 JSON for experiment evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def synchronize(torch_module: Any) -> None:
    """Synchronize the active accelerator when one exists."""
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
