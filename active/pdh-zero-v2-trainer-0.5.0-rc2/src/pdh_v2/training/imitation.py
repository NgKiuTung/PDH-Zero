"""规则启发式教师的短时行为克隆暖启动。"""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from pdh_v2.config import PLAYERS
from pdh_v2.policy import CandidateActorCritic, choose_action
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.buffer import Episode, Transition


@dataclass(frozen=True, slots=True)
class ImitationMetrics:
    """行为克隆训练指标。"""

    loss: float
    accuracy: float
    entropy: float
    decisions: int
    episodes: int


def collect_teacher_episodes(
    environments: BatchedSessions,
    minimum_hands: int,
) -> tuple[tuple[Episode, ...], int]:
    """运行确定性启发式并排空所有进行中的手牌。"""
    environments.reactivate_all()
    current = [[Episode() for _ in range(PLAYERS)] for _ in range(environments.count)]
    completed: list[Episode] = []
    hands = 0
    draining = False
    drain_trigger = max(1, minimum_hands - environments.count)
    while not environments.all_inactive():
        requests = environments.requests()
        choices: dict[int, int] = {}
        for request in requests:
            session = environments.sessions[request.env_index]
            action_index = choose_action(session, request.view.actions)
            choices[request.env_index] = action_index
            current[request.env_index][request.view.player].transitions.append(
                Transition(
                    torch.tensor(request.view.observation, dtype=torch.float32),
                    torch.tensor(request.view.action_features, dtype=torch.float32),
                    action_index,
                    0.0,
                    0.0,
                )
            )
        terminals = environments.step(choices, drain=draining)
        for terminal in terminals:
            hands += 1
            for seat in range(PLAYERS):
                episode = current[terminal.env_index][seat]
                if episode.transitions:
                    completed.append(episode)
                current[terminal.env_index][seat] = Episode()
        if not draining and hands >= drain_trigger:
            draining = True
    return tuple(completed), hands


def _groups(episodes: list[Episode], maximum: int, rng: random.Random) -> list[list[Episode]]:
    rng.shuffle(episodes)
    result: list[list[Episode]] = []
    current: list[Episode] = []
    count = 0
    for episode in episodes:
        if current and count + episode.decisions > maximum:
            result.append(current)
            current = []
            count = 0
        current.append(episode)
        count += episode.decisions
    if current:
        result.append(current)
    return result


def _sequence_loss(
    model: CandidateActorCritic,
    episodes: list[Episode],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    hidden = [model.initial_state(1, device).squeeze(0) for _ in episodes]
    losses: list[Tensor] = []
    accuracies: list[Tensor] = []
    entropies: list[Tensor] = []
    decisions = 0
    for step in range(max(episode.decisions for episode in episodes)):
        active = [index for index, episode in enumerate(episodes) if step < episode.decisions]
        if not active:
            continue
        transitions = [episodes[index].transitions[step] for index in active]
        observations = torch.stack([item.observation for item in transitions]).to(device)
        hidden_batch = torch.stack([hidden[index] for index in active])
        action_parts = [item.action_features for item in transitions]
        counts = [part.shape[0] for part in action_parts]
        features = torch.cat(action_parts).to(device)
        owners = torch.repeat_interleave(
            torch.arange(len(active), device=device),
            torch.tensor(counts, device=device),
        )
        logits, _, next_hidden = model.forward_step(
            observations, hidden_batch, features, owners
        )
        padded = torch.full(
            (len(active), max(counts)), -torch.inf, dtype=logits.dtype, device=device
        )
        cursor = 0
        for row, count in enumerate(counts):
            padded[row, :count] = logits[cursor : cursor + count]
            cursor += count
        targets = torch.tensor(
            [item.action_index for item in transitions], dtype=torch.long, device=device
        )
        distribution = Categorical(logits=padded)
        losses.append(-distribution.log_prob(targets))
        accuracies.append((padded.argmax(dim=-1) == targets).float())
        entropies.append(distribution.entropy())
        decisions += len(active)
        for row, episode_index in enumerate(active):
            hidden[episode_index] = next_hidden[row]
    return (
        torch.cat(losses).mean(),
        torch.cat(accuracies).mean(),
        torch.cat(entropies).mean(),
        decisions,
    )


def behavior_clone(
    model: CandidateActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: tuple[Episode, ...],
    device: torch.device,
    epochs: int,
    minibatch_decisions: int,
    seed: int,
) -> ImitationMetrics:
    """用完整手局序列拟合规则启发式动作。"""
    model.train()
    rng = random.Random(seed)
    loss_total = accuracy_total = entropy_total = 0.0
    batches = 0
    decision_total = 0
    for _ in range(epochs):
        for group in _groups(list(episodes), minibatch_decisions, rng):
            loss, accuracy, entropy, decisions = _sequence_loss(model, group, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            accuracy_total += float(accuracy.detach().cpu())
            entropy_total += float(entropy.detach().cpu())
            decision_total += decisions
            batches += 1
    return ImitationMetrics(
        loss_total / batches,
        accuracy_total / batches,
        entropy_total / batches,
        decision_total,
        len(episodes),
    )
