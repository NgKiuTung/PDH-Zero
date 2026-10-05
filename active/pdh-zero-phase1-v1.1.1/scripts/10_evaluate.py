#!/usr/bin/env python3
"""Evaluate a checkpoint with fixed wall seeds and four-seat rotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from script_common import artifact_dir, write_json
from pdh_phase1.config import load_config
from pdh_phase1.device import resolve_device
from pdh_phase1.evaluation import evaluate_model, evaluate_rule_bot
from pdh_phase1.mini_pdh import ACTOR_OBSERVATION_DIM, PRIVILEGED_OBSERVATION_DIM
from pdh_phase1.rl.model import RecurrentActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--seed-start", type=int, default=950000)
    parser.add_argument("--wall-seeds", type=int, default=500)
    parser.add_argument("--opponent", choices=("random", "heuristic"), default="heuristic")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = resolve_device(args.device)
    payload = torch.load(Path(args.checkpoint), map_location=info.device, weights_only=False)
    training_config = payload["metadata"]["config"]
    rules = load_config(training_config["ruleset"])
    critic_dim = (
        PRIVILEGED_OBSERVATION_DIM
        if bool(training_config["privileged_critic"])
        else ACTOR_OBSERVATION_DIM
    )
    model = RecurrentActorCritic(
        ACTOR_OBSERVATION_DIM,
        critic_dim,
        int(training_config["hidden_size"]),
    ).to(info.device)
    model.load_state_dict(payload["model"])
    seeds = [args.seed_start + index for index in range(args.wall_seeds)]
    summary = {
        "status": "PASS",
        "backend": info.backend,
        "checkpoint": str(args.checkpoint),
        "model": evaluate_model(model, rules, info.device, seeds, args.opponent),
        "random_baseline": evaluate_rule_bot(rules, seeds, "random", args.opponent),
        "heuristic_vs_random": evaluate_rule_bot(rules, seeds, "heuristic", "random"),
    }
    write_json(artifact_dir(args.output) / "evaluation.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
