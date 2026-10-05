#!/usr/bin/env python3
"""Verify that hidden hands and wall order cannot change Actor input or output."""

from __future__ import annotations

import argparse

import torch

from script_common import artifact_dir, write_json
from pdh_phase1.config import load_config
from pdh_phase1.device import resolve_device, seed_everything
from pdh_phase1.mini_pdh import (
    ACTION_COUNT,
    ACTOR_OBSERVATION_DIM,
    PRIVILEGED_OBSERVATION_DIM,
    MiniPDHEnv,
    encode_actor,
    encode_critic,
)
from pdh_phase1.rl.model import RecurrentActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="configs/mini_pdh_v0.yaml")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "rocm"))
    parser.add_argument("--output")
    return parser.parse_args()


def actor_logits(model: RecurrentActorCritic, observation: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    hidden = torch.zeros(1, 1, model.hidden_size, device=observation.device)
    encoded = model.actor_encoder(observation.view(1, 1, -1))
    output, _ = model.actor_gru(encoded, hidden)
    logits = model.policy_head(output)[0, 0]
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def main() -> int:
    args = parse_args()
    seed_everything(20260929, deterministic=True)
    info = resolve_device(args.device)
    rules = load_config(args.rules)
    original = MiniPDHEnv(rules)
    original.reset(20260929)
    altered = original.clone()
    altered.hands[1], altered.hands[2] = altered.hands[2], altered.hands[1]
    altered.wall.reverse()
    altered.assert_invariants()
    seat = original.current_seat
    actor_original = encode_actor(original, seat)
    actor_altered = encode_actor(altered, seat)
    critic_original = encode_critic(original, seat, True)
    critic_altered = encode_critic(altered, seat, True)
    if not torch.equal(actor_original, actor_altered):
        raise AssertionError("Actor observation changed after hidden-state permutation")
    if torch.equal(critic_original, critic_altered):
        raise AssertionError("Privileged critic failed to observe the hidden-state permutation")
    model = RecurrentActorCritic(
        ACTOR_OBSERVATION_DIM,
        PRIVILEGED_OBSERVATION_DIM,
        64,
    ).to(info.device)
    legal_original = torch.tensor(original.legal_actions(), dtype=torch.bool, device=info.device)
    legal_altered = torch.tensor(altered.legal_actions(), dtype=torch.bool, device=info.device)
    if not torch.equal(legal_original, legal_altered):
        raise AssertionError("Legal action mask changed after hidden-state permutation")
    output_original = actor_logits(model, actor_original.to(info.device), legal_original)
    output_altered = actor_logits(model, actor_altered.to(info.device), legal_altered)
    max_error = float((output_original - output_altered).abs().max().item())
    passed = max_error == 0.0
    summary = {
        "status": "PASS" if passed else "FAIL",
        "backend": info.backend,
        "actor_observation_equal": True,
        "critic_observation_changed": True,
        "legal_mask_equal": True,
        "actor_logit_max_abs_error": max_error,
        "action_count": ACTION_COUNT,
    }
    write_json(artifact_dir(args.output) / "hidden_info_isolation.json", summary)
    print(summary)
    if not passed:
        raise RuntimeError("Actor output changed after hidden-state permutation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
