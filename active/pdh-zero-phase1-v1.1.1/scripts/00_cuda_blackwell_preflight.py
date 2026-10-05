
#!/usr/bin/env python3
"""Reject incompatible CUDA images before consuming RTX 50-series GPU time."""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

from script_common import artifact_dir, synchronize, write_json
from pdh_phase1.cuda_compat import validate_cuda_architecture


def parse_args() -> argparse.Namespace:
    """Parse the bounded preflight options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-capability", default="12.0")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--output")
    return parser.parse_args()


def parse_capability(value: str) -> tuple[int, int]:
    """Parse a dotted compute capability such as ``12.0``."""
    parts = value.split(".", maxsplit=1)
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid compute capability: {value!r}")
    return int(parts[0]), int(parts[1])


def query_driver() -> dict[str, str | None]:
    """Read non-sensitive NVIDIA driver metadata when nvidia-smi is available."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {"nvidia_smi": None}
    return {"nvidia_smi": result.stdout.strip() if result.returncode == 0 else None}


def benchmark(dtype: torch.dtype, size: int) -> dict[str, float | bool]:
    """Run one bounded matrix multiplication and return finite-output evidence."""
    device = torch.device("cuda:0")
    left = torch.randn(size, size, device=device, dtype=dtype)
    right = torch.randn(size, size, device=device, dtype=dtype)
    synchronize(torch)
    started = time.perf_counter()
    output = left @ right
    synchronize(torch)
    elapsed = time.perf_counter() - started
    finite = bool(torch.isfinite(output).all().item())
    checksum = float(output.float().mean().item())
    del left, right, output
    return {
        "elapsed_seconds": elapsed,
        "finite": finite,
        "mean_checksum": checksum,
        "tflops_estimate": (2.0 * size**3 / max(elapsed, 1e-12)) / 1e12,
    }


def main() -> int:
    """Validate the CUDA runtime, architecture kernels, BF16, and basic math."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA accelerator is unavailable")
    if torch.version.hip is not None:
        raise RuntimeError("This preflight requires CUDA, but the Torch build uses ROCm/HIP")

    expected_capability = parse_capability(args.require_capability)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    arch_list = list(torch.cuda.get_arch_list())
    errors = validate_cuda_architecture(torch.version.cuda, capability, arch_list)
    if capability != expected_capability:
        errors.append(f"Expected compute capability {expected_capability}, observed {capability}")

    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if not bf16_supported:
        errors.append("CUDA device does not report BF16 support")

    fp32 = benchmark(torch.float32, args.matrix_size)
    bf16 = benchmark(torch.bfloat16, args.matrix_size) if bf16_supported else None
    if not fp32["finite"]:
        errors.append("FP32 matrix multiplication produced non-finite values")
    if bf16 is not None and not bf16["finite"]:
        errors.append("BF16 matrix multiplication produced non-finite values")

    props = torch.cuda.get_device_properties(0)
    summary = {
        "status": "PASS" if not errors else "FAIL",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "torch_arch_list": arch_list,
        "total_memory_bytes": int(props.total_memory),
        "multiprocessor_count": int(props.multi_processor_count),
        "bf16_supported": bf16_supported,
        "fp32_matmul": fp32,
        "bf16_matmul": bf16,
        "driver": query_driver(),
        "errors": errors,
    }
    write_json(artifact_dir(args.output) / "cuda_blackwell_preflight.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError("CUDA Blackwell preflight failed: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
