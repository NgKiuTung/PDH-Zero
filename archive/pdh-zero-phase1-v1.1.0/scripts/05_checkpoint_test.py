#!/usr/bin/env python3
"""Prove model, optimizer, and RNG continuation from an atomic checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from script_common import artifact_dir, write_json
from pdh_phase1.checkpoint import load_checkpoint, save_checkpoint
from pdh_phase1.device import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--output")
    return parser.parse_args()


def make_model(device: torch.device) -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 5)).to(device)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    inputs = torch.randn(64, 16, device=device)
    targets = torch.randint(0, 5, (64,), device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def main() -> int:
    args = parse_args()
    info = resolve_device(args.device)
    seed_everything(42017, deterministic=True)
    out = artifact_dir(args.output)
    checkpoint = out / "checkpoint_test.pt"
    model = make_model(info.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first_loss = train_step(model, optimizer, info.device)
    save_checkpoint(checkpoint, model, optimizer, {"step": 1, "backend": info.backend})
    expected_loss = train_step(model, optimizer, info.device)
    expected_parameters = [parameter.detach().cpu().clone() for parameter in model.parameters()]

    restored_model = make_model(info.device)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    metadata = load_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        info.device,
        restore_rng=True,
    )
    actual_loss = train_step(restored_model, restored_optimizer, info.device)
    parameter_error = max(
        float((expected - actual.detach().cpu()).abs().max().item())
        for expected, actual in zip(expected_parameters, restored_model.parameters())
    )
    passed = abs(expected_loss - actual_loss) <= 1e-7 and parameter_error <= 1e-7
    summary = {
        "status": "PASS" if passed else "FAIL",
        "backend": info.backend,
        "first_loss": first_loss,
        "expected_continuation_loss": expected_loss,
        "restored_continuation_loss": actual_loss,
        "parameter_max_abs_error": parameter_error,
        "metadata": metadata,
        "checkpoint": str(checkpoint),
    }
    write_json(out / "checkpoint_test.json", summary)
    print(summary)
    if not passed:
        raise RuntimeError("Checkpoint continuation did not reproduce the expected state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
