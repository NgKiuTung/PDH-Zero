#!/usr/bin/env python3
"""Verify bounded forward, backward, and AdamW updates on CPU, CUDA, or ROCm."""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn

from script_common import artifact_dir, synchronize, write_json
from pdh_phase1.device import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = resolve_device(args.device)
    seed_everything(20260929)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 128),
        nn.GELU(),
        nn.Linear(128, 34),
    ).to(device=info.device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    synchronize(torch)
    started = time.perf_counter()
    for _ in range(args.steps):
        inputs = torch.randn(args.batch_size, 256, device=info.device, dtype=dtype)
        targets = torch.randint(0, 34, (args.batch_size,), device=info.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    synchronize(torch)
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    summary = {
        "status": "PASS",
        "backend": info.backend,
        "device_name": info.name,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "samples_per_second": args.steps * args.batch_size / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "finite": all(math.isfinite(item) for item in losses),
        "peak_memory_bytes": peak_memory,
    }
    out = artifact_dir(args.output)
    write_json(out / f"torch_smoke_{args.dtype}.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
