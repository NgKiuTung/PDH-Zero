"""Episode trajectory containers and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EpisodeTrajectory:
    """All learner-seat decisions from one terminal four-player game."""

    actor_observations: list[torch.Tensor] = field(default_factory=list)
    critic_observations: list[torch.Tensor] = field(default_factory=list)
    legal_masks: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    old_values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    learner_seat: int = 0
    outcome_type: str = ""
    final_score: float = 0.0

    def append(
        self,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        legal_mask: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
    ) -> None:
        """Append one learner decision; reward is assigned after the game terminates."""
        self.actor_observations.append(actor_observation.cpu())
        self.critic_observations.append(critic_observation.cpu())
        self.legal_masks.append(legal_mask.cpu())
        self.actions.append(int(action))
        self.old_log_probs.append(float(log_prob))
        self.old_values.append(float(value))
        self.rewards.append(0.0)

    def finalize(
        self,
        final_score: float,
        outcome_type: str,
        reward_normalizer: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Assign terminal reward and compute normalized GAE/returns."""
        if not self.actions:
            raise ValueError("Cannot finalize an empty learner trajectory")
        self.final_score = float(final_score)
        self.outcome_type = str(outcome_type)
        self.rewards[-1] = self.final_score / float(reward_normalizer)
        advantages = [0.0] * len(self.actions)
        next_value = 0.0
        gae = 0.0
        for index in range(len(self.actions) - 1, -1, -1):
            delta = self.rewards[index] + gamma * next_value - self.old_values[index]
            gae = delta + gamma * gae_lambda * gae
            advantages[index] = gae
            next_value = self.old_values[index]
        self.advantages = advantages
        self.returns = [advantage + value for advantage, value in zip(advantages, self.old_values)]

    @property
    def length(self) -> int:
        """Return the number of learner decisions in the game."""
        return len(self.actions)


@dataclass(frozen=True)
class PaddedBatch:
    """Padded recurrent trajectories and their valid-position mask."""

    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    valid: torch.Tensor


def pad_trajectories(episodes: list[EpisodeTrajectory], device: torch.device) -> PaddedBatch:
    """Pad complete episodes without joining recurrent state across game boundaries."""
    if not episodes:
        raise ValueError("At least one trajectory is required")
    batch_size = len(episodes)
    max_length = max(item.length for item in episodes)
    actor_dim = episodes[0].actor_observations[0].numel()
    critic_dim = episodes[0].critic_observations[0].numel()
    action_dim = episodes[0].legal_masks[0].numel()
    actor = torch.zeros(batch_size, max_length, actor_dim, device=device)
    critic = torch.zeros(batch_size, max_length, critic_dim, device=device)
    legal = torch.ones(batch_size, max_length, action_dim, dtype=torch.bool, device=device)
    actions = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
    old_log_probs = torch.zeros(batch_size, max_length, device=device)
    old_values = torch.zeros(batch_size, max_length, device=device)
    returns = torch.zeros(batch_size, max_length, device=device)
    advantages = torch.zeros(batch_size, max_length, device=device)
    valid = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
    for row, episode in enumerate(episodes):
        length = episode.length
        actor[row, :length] = torch.stack(episode.actor_observations).to(device)
        critic[row, :length] = torch.stack(episode.critic_observations).to(device)
        legal[row, :length] = torch.stack(episode.legal_masks).to(device)
        actions[row, :length] = torch.tensor(episode.actions, dtype=torch.long, device=device)
        old_log_probs[row, :length] = torch.tensor(episode.old_log_probs, device=device)
        old_values[row, :length] = torch.tensor(episode.old_values, device=device)
        returns[row, :length] = torch.tensor(episode.returns, device=device)
        advantages[row, :length] = torch.tensor(episode.advantages, device=device)
        valid[row, :length] = True
    return PaddedBatch(actor, critic, legal, actions, old_log_probs, old_values, returns, advantages, valid)
