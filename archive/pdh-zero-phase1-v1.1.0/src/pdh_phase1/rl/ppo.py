"""Recurrent PPO update over complete learner-seat trajectories."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Categorical

from .model import RecurrentActorCritic
from .trajectory import EpisodeTrajectory, pad_trajectories


@dataclass(frozen=True)
class PPOMetrics:
    """Mean optimizer statistics for one update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    gradient_norm: float


def ppo_update(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: list[EpisodeTrajectory],
    config: dict[str, Any],
    device: torch.device,
    rng: random.Random,
) -> PPOMetrics:
    """Optimize PPO objective while preserving episode-level recurrent boundaries."""
    advantages = [value for episode in episodes for value in episode.advantages]
    mean_advantage = sum(advantages) / max(len(advantages), 1)
    variance = sum((value - mean_advantage) ** 2 for value in advantages) / max(len(advantages), 1)
    scale = math.sqrt(variance + 1e-8)
    for episode in episodes:
        episode.advantages = [(value - mean_advantage) / scale for value in episode.advantages]

    totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "grad": 0.0}
    optimization_steps = 0
    minibatch_size = max(1, int(config["sequence_minibatch_size"]))
    indices = list(range(len(episodes)))
    model.train()
    for _ in range(int(config["ppo_epochs"])):
        rng.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            selected = [episodes[index] for index in indices[start : start + minibatch_size]]
            batch = pad_trajectories(selected, device)
            logits, values, _ = model.forward_sequences(
                batch.actor_observations,
                batch.critic_observations,
                batch.legal_masks,
            )
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(batch.actions)
            entropy = distribution.entropy()
            valid = batch.valid
            ratio = torch.exp(new_log_probs - batch.old_log_probs)
            unclipped = ratio * batch.advantages
            clipped = torch.clamp(
                ratio,
                1.0 - float(config["clip_ratio"]),
                1.0 + float(config["clip_ratio"]),
            ) * batch.advantages
            policy_loss = -torch.minimum(unclipped, clipped)[valid].mean()
            value_loss = torch.square(values - batch.returns)[valid].mean()
            entropy_mean = entropy[valid].mean()
            loss = (
                policy_loss
                + float(config["value_coef"]) * value_loss
                - float(config["entropy_coef"]) * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["max_grad_norm"])
            )
            optimizer.step()
            approximate_kl = (batch.old_log_probs - new_log_probs)[valid].mean().abs()
            totals["policy"] += float(policy_loss.detach().item())
            totals["value"] += float(value_loss.detach().item())
            totals["entropy"] += float(entropy_mean.detach().item())
            totals["kl"] += float(approximate_kl.detach().item())
            totals["grad"] += float(gradient_norm.detach().item())
            optimization_steps += 1
    divisor = max(optimization_steps, 1)
    return PPOMetrics(
        policy_loss=totals["policy"] / divisor,
        value_loss=totals["value"] / divisor,
        entropy=totals["entropy"] / divisor,
        approximate_kl=totals["kl"] / divisor,
        gradient_norm=totals["grad"] / divisor,
    )
