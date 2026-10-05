#!/usr/bin/env python3
"""Run a bounded accelerator soak test with finite-loss and memory checks."""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn

from script_common import artifact_dir, synchronize, write_json
from pdh_phase1.device import resolve_device, seed_everything
from pdh_phase1.io_utils import append_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-interval-seconds", type=int, default=10)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_seconds <= 0 or args.duration_seconds > 7200:
        raise ValueError("duration must be in [1, 7200] seconds")
    info = resolve_device(args.device)
    seed_everything(20260929)
    model = nn.Sequential(
        nn.Linear(256, 1024),
        nn.GELU(),
        nn.Linear(1024, 256),
        nn.GELU(),
        nn.Linear(256, 64),
    ).to(info.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = artifact_dir(args.output)
    metrics_path = out / "soak_metrics.jsonl"
    started = time.perf_counter()
    next_sample = started
    iterations = 0
    losses: list[float] = []
    while time.perf_counter() - started < args.duration_seconds:
        inputs = torch.randn(args.batch_size, 256, device=info.device)
        targets = torch.randint(0, 64, (args.batch_size,), device=info.device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during soak test")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        iterations += 1
        now = time.perf_counter()
        if now >= next_sample:
            synchronize(torch)
            append_jsonl(
                metrics_path,
                {
                    "elapsed_seconds": now - started,
                    "iterations": iterations,
                    "loss": losses[-1],
                    "memory_allocated_bytes": int(torch.cuda.memory_allocated())
                    if torch.cuda.is_available()
                    else 0,
                    "memory_reserved_bytes": int(torch.cuda.memory_reserved())
                    if torch.cuda.is_available()
                    else 0,
                },
            )
            next_sample = now + args.sample_interval_seconds
    synchronize(torch)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "PASS",
        "backend": info.backend,
        "device_name": info.name,
        "duration_seconds": elapsed,
        "iterations": iterations,
        "iterations_per_second": iterations / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "finite": all(math.isfinite(value) for value in losses),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
    }
    write_json(out / "soak_summary.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
