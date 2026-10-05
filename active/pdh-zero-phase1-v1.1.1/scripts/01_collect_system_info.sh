#!/usr/bin/env bash
# Collect non-secret hardware, software, and mounted-storage facts.
set -euo pipefail

OUT_DIR="${1:-${PDH_RUN_DIR:-results/manual_system_info}/artifacts}"
mkdir -p "$OUT_DIR"
{
  echo "===== UTC ====="
  date -u --iso-8601=seconds
  echo "===== UNAME ====="
  uname -a
  echo "===== CPU ====="
  nproc
  lscpu
  echo "===== MEMORY ====="
  free -h
  echo "===== FILESYSTEM ====="
  for target in . "${PDH_HOME:-}" /root/rivermind-data /root/rivermind-fs; do
    [[ -n "$target" && -e "$target" ]] && df -hT "$target" || true
  done
  echo "===== PCI DISPLAY ====="
  lspci -nn 2>/dev/null | grep -Ei 'AMD|ATI|NVIDIA|VGA|Display' || true
  echo "===== AMD SMI ====="
  command -v amd-smi >/dev/null && amd-smi static --asic | sed -E 's/(ASIC_SERIAL:).*/\1 REDACTED/' || true
  echo "===== ROCMINFO ====="
  command -v rocminfo >/dev/null && rocminfo | grep -E 'Marketing Name|Name:|gfx[0-9]+|Compute Unit' | head -n 120 || true
  echo "===== NVIDIA SMI ====="
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader || nvidia-smi || true
  fi
} >"$OUT_DIR/system.txt" 2>&1
python - "$OUT_DIR" <<'PY'
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch

out = Path(sys.argv[1])
props = None
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    props = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(p.total_memory),
        "multiprocessor_count": int(p.multi_processor_count),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "arch_list": list(torch.cuda.get_arch_list()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def disk(path: str) -> dict[str, int] | None:
    target = Path(path)
    if not target.exists():
        return None
    usage = shutil.disk_usage(target)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "accelerator_available": torch.cuda.is_available(),
    "accelerator_count": torch.cuda.device_count(),
    "accelerator": props,
    "cpu_count": os.cpu_count(),
    "storage": {
        "root": disk("/"),
        "data_disk": disk("/root/rivermind-data"),
        "file_storage": disk("/root/rivermind-fs"),
    },
}
(out / "software.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False))
PY
