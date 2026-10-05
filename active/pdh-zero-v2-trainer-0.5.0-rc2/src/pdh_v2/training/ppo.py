"""按完整手局序列训练的 Recurrent PPO。"""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from pdh_v2.policy import CandidateActorCritic
from pdh_v2.training.buffer import Episode


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO 更新参数。"""

    epochs: int = 4
    minibatch_decisions: int = 4096
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 1.0


@dataclass(frozen=True, slots=True)
class PPOMetrics:
    """一次更新的聚合训练指标。"""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    gradient_norm: float
    minibatches: int
    decisions: int


def _episode_groups(
    episodes: list[Episode],
    maximum_decisions: int,
    rng: random.Random,
) -> list[list[Episode]]:
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    groups: list[list[Episode]] = []
    current: list[Episode] = []
    count = 0
    for episode in shuffled:
        if current and count + episode.decisions > maximum_decisions:
            groups.append(current)
            current = []
            count = 0
        current.append(episode)
        count += episode.decisions
    if current:
        groups.append(current)
    return groups


def _sample_statistics(
    logits: Tensor, counts: list[int], selected: Tensor
) -> tuple[Tensor, Tensor]:
    max_actions = max(counts)
    padded = torch.full(
        (len(counts), max_actions),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    cursor = 0
    for row, count in enumerate(counts):
        padded[row, :count] = logits[cursor : cursor + count]
        cursor += count
    distribution = Categorical(logits=padded)
    return distribution.log_prob(selected), distribution.entropy()


def _evaluate_sequences(
    model: CandidateActorCritic,
    episodes: list[Episode],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    hidden_states = [model.initial_state(1, device).squeeze(0) for _ in episodes]
    log_probs: list[Tensor] = []
    entropies: list[Tensor] = []
    values: list[Tensor] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    advantages: list[Tensor] = []
    returns: list[Tensor] = []
    max_length = max(episode.decisions for episode in episodes)

    for step in range(max_length):
        active = [index for index, episode in enumerate(episodes) if step < episode.decisions]
        if not active:
            continue
        transitions = [episodes[index].transitions[step] for index in active]
        observations = torch.stack([item.observation for item in transitions]).to(device)
        hidden = torch.stack([hidden_states[index] for index in active])
        action_parts = [item.action_features for item in transitions]
        counts = [part.shape[0] for part in action_parts]
        action_features = torch.cat(action_parts).to(device)
        owners = torch.repeat_interleave(
            torch.arange(len(active), device=device),
            torch.tensor(counts, device=device),
        )
        logits, step_values, next_hidden = model.forward_step(
            observations, hidden, action_features, owners
        )
        selected = torch.tensor(
            [item.action_index for item in transitions], dtype=torch.long, device=device
        )
        step_log_probs, step_entropies = _sample_statistics(logits, counts, selected)
        for row, episode_index in enumerate(active):
            hidden_states[episode_index] = next_hidden[row]
        log_probs.append(step_log_probs)
        entropies.append(step_entropies)
        values.append(step_values)
        old_log_probs.extend(item.old_log_prob for item in transitions)
        old_values.extend(item.old_value for item in transitions)
        for episode_index in active:
            advantage = episodes[episode_index].advantage
            target = episodes[episode_index].returns
            if advantage is None or target is None:
                raise RuntimeError("episode advantages are not ready")
            advantages.append(advantage[step])
            returns.append(target[step])

    return (
        torch.cat(log_probs),
        torch.cat(entropies),
        torch.cat(values),
        torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        torch.tensor(old_values, dtype=torch.float32, device=device),
        torch.stack(advantages).to(device),
        torch.stack(returns).to(device),
    )


def update_ppo(
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: tuple[Episode, ...],
    device: torch.device,
    config: PPOConfig,
    seed: int,
) -> PPOMetrics:
    """执行完整序列 PPO 更新。"""
    if not episodes:
        raise RuntimeError("no episodes collected")
    model.train()
    rng = random.Random(seed)
    totals = {
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "grad": 0.0,
        "batches": 0,
    }
    episode_list = list(episodes)

    for _ in range(config.epochs):
        for group in _episode_groups(episode_list, config.minibatch_decisions, rng):
            (
                log_probs,
                entropy,
                values,
                old_log_probs,
                old_values,
                advantages,
                targets,
            ) = _evaluate_sequences(
                model, group, device
            )
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
            ) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            clipped_values = old_values + torch.clamp(
                values - old_values, -config.value_clip, config.value_clip
            )
            value_loss = 0.5 * torch.maximum(
                (values - targets).pow(2), (clipped_values - targets).pow(2)
            ).mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = (old_log_probs - log_probs).mean().abs()
            totals["policy"] += float(policy_loss.detach().cpu())
            totals["value"] += float(value_loss.detach().cpu())
            totals["entropy"] += float(entropy_mean.detach().cpu())
            totals["kl"] += float(approximate_kl.detach().cpu())
            totals["grad"] += float(torch.as_tensor(gradient_norm).detach().cpu())
            totals["batches"] += 1

    batches = int(totals["batches"])
    decisions = sum(episode.decisions for episode in episode_list)
    return PPOMetrics(
        totals["policy"] / batches,
        totals["value"] / batches,
        totals["entropy"] / batches,
        totals["kl"] / batches,
        totals["grad"] / batches,
        batches,
        decisions,
    )
