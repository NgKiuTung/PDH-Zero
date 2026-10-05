#!/usr/bin/env python3
"""Scan batched actor inference throughput on the selected Torch backend."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from script_common import artifact_dir, synchronize, write_json
from pdh_phase1.device import resolve_device, seed_everything
from pdh_phase1.mini_pdh import ACTION_COUNT, ACTOR_OBSERVATION_DIM
from pdh_phase1.rl.model import RecurrentActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--batch-sizes", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--output")
    return parser.parse_args()


def actor_forward(model: RecurrentActorCritic, observations: torch.Tensor) -> torch.Tensor:
    hidden = torch.zeros(1, observations.shape[0], model.hidden_size, device=observations.device)
    encoded = model.actor_encoder(observations.unsqueeze(1))
    output, _ = model.actor_gru(encoded, hidden)
    return model.policy_head(output[:, 0])


def main() -> int:
    args = parse_args()
    info = resolve_device(args.device)
    seed_everything(20260929)
    model = RecurrentActorCritic(
        ACTOR_OBSERVATION_DIM,
        ACTOR_OBSERVATION_DIM,
        args.hidden_size,
    ).to(info.device)
    model.eval()
    rows: list[dict[str, float | int]] = []
    for batch_size in [int(value) for value in args.batch_sizes.split(",") if value.strip()]:
        observations = torch.randn(batch_size, ACTOR_OBSERVATION_DIM, device=info.device)
        with torch.no_grad():
            for _ in range(args.warmup):
                logits = actor_forward(model, observations)
                if logits.shape != (batch_size, ACTION_COUNT):
                    raise RuntimeError("Unexpected policy output shape")
            synchronize(torch)
            started = time.perf_counter()
            for _ in range(args.iterations):
                actor_forward(model, observations)
            synchronize(torch)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "batch_size": batch_size,
                "iterations": args.iterations,
                "elapsed_seconds": elapsed,
                "observations_per_second": batch_size * args.iterations / elapsed,
                "latency_ms_per_batch": elapsed * 1000.0 / args.iterations,
                "memory_allocated_bytes": int(torch.cuda.memory_allocated())
                if torch.cuda.is_available()
                else 0,
            }
        )
    best = max(rows, key=lambda row: float(row["observations_per_second"]))
    out = artifact_dir(args.output)
    with (out / "policy_benchmark.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "PASS",
        "backend": info.backend,
        "device_name": info.name,
        "hidden_size": args.hidden_size,
        "results": rows,
        "best": best,
    }
    write_json(out / "policy_benchmark.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
