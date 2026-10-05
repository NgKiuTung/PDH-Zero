#!/usr/bin/env python3
"""Compare deterministic FP32 forward/backward results against a CPU reference."""

from __future__ import annotations

import argparse
import copy

import torch
from torch import nn

from script_common import artifact_dir, synchronize, write_json
from pdh_phase1.device import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--output")
    parser.add_argument("--forward-tolerance", type=float, default=1e-4)
    parser.add_argument("--gradient-tolerance", type=float, default=5e-4)
    return parser.parse_args()


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 11))


def run_once(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = nn.functional.cross_entropy(output, targets)
    loss.backward()
    gradients = [parameter.grad.detach().cpu().clone() for parameter in model.parameters()]
    return output.detach().cpu(), gradients


def main() -> int:
    args = parse_args()
    seed_everything(1907, deterministic=True)
    inputs_cpu = torch.randn(128, 32)
    targets_cpu = torch.randint(0, 11, (128,))
    reference = make_model().float()
    candidate = copy.deepcopy(reference)
    reference_output, reference_gradients = run_once(reference, inputs_cpu, targets_cpu)
    info = resolve_device(args.device)
    candidate = candidate.to(info.device)
    candidate_output, candidate_gradients = run_once(
        candidate,
        inputs_cpu.to(info.device),
        targets_cpu.to(info.device),
    )
    synchronize(torch)
    forward_error = float((reference_output - candidate_output).abs().max().item())
    gradient_error = max(
        float((expected - actual).abs().max().item())
        for expected, actual in zip(reference_gradients, candidate_gradients)
    )
    passed = forward_error <= args.forward_tolerance and gradient_error <= args.gradient_tolerance
    summary = {
        "status": "PASS" if passed else "FAIL",
        "backend": info.backend,
        "device_name": info.name,
        "forward_max_abs_error": forward_error,
        "gradient_max_abs_error": gradient_error,
        "forward_tolerance": args.forward_tolerance,
        "gradient_tolerance": args.gradient_tolerance,
    }
    write_json(artifact_dir(args.output) / "precision_test.json", summary)
    print(summary)
    if not passed:
        raise RuntimeError("Numerical comparison exceeded the declared tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
