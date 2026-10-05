"""Compact recurrent actor-critic network used by Phase 1 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from pdh_phase1.mini_pdh import ACTION_COUNT


@dataclass(frozen=True)
class RecurrentState:
    """Actor and critic GRU hidden states for one decision stream."""

    actor: torch.Tensor
    critic: torch.Tensor


@dataclass(frozen=True)
class ActionOutput:
    """Sampled action, old-policy statistics, and updated recurrent state."""

    action: int
    log_prob: float
    value: float
    state: RecurrentState


class RecurrentActorCritic(nn.Module):
    """Separate actor and critic GRUs with a fixed legal-action policy head."""

    def __init__(self, actor_dim: int, critic_dim: int, hidden_size: int):
        """Initialize a small network suitable for CPU, CUDA, and ROCm."""
        super().__init__()
        self.actor_dim = int(actor_dim)
        self.critic_dim = int(critic_dim)
        self.hidden_size = int(hidden_size)
        self.actor_encoder = nn.Sequential(
            nn.Linear(self.actor_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.actor_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, ACTION_COUNT)
        self.critic_encoder = nn.Sequential(
            nn.Linear(self.critic_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.critic_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> RecurrentState:
        """Create zero GRU state for a new set of independent episodes."""
        shape = (1, int(batch_size), self.hidden_size)
        return RecurrentState(
            actor=torch.zeros(shape, device=device),
            critic=torch.zeros(shape, device=device),
        )

    def forward_sequences(
        self,
        actor_observations: torch.Tensor,
        critic_observations: torch.Tensor,
        legal_masks: torch.Tensor,
        state: RecurrentState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        """Evaluate padded decision sequences; loss masking is handled by the caller."""
        if actor_observations.ndim != 3 or critic_observations.ndim != 3:
            raise ValueError("Sequence observations must have shape [batch, time, features]")
        batch_size = actor_observations.shape[0]
        recurrent = state or self.initial_state(batch_size, actor_observations.device)
        actor_features = self.actor_encoder(actor_observations)
        actor_output, actor_hidden = self.actor_gru(actor_features, recurrent.actor)
        logits = self.policy_head(actor_output)
        if legal_masks.shape != logits.shape:
            raise ValueError(f"Legal mask shape {legal_masks.shape} != logits shape {logits.shape}")
        if not torch.all(legal_masks.any(dim=-1)):
            raise ValueError("Every sequence position must expose at least one legal action")
        masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
        critic_features = self.critic_encoder(critic_observations)
        critic_output, critic_hidden = self.critic_gru(critic_features, recurrent.critic)
        values = self.value_head(critic_output).squeeze(-1)
        return masked_logits, values, RecurrentState(actor_hidden, critic_hidden)

    @torch.no_grad()
    def act(
        self,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        legal_mask: torch.Tensor,
        state: RecurrentState,
        deterministic: bool = False,
    ) -> ActionOutput:
        """Choose one legal action and advance both actor and critic recurrent states."""
        logits, values, next_state = self.forward_sequences(
            actor_observation.view(1, 1, -1),
            critic_observation.view(1, 1, -1),
            legal_mask.view(1, 1, -1),
            state,
        )
        distribution = Categorical(logits=logits[0, 0])
        action_tensor = torch.argmax(logits[0, 0]) if deterministic else distribution.sample()
        return ActionOutput(
            action=int(action_tensor.item()),
            log_prob=float(distribution.log_prob(action_tensor).item()),
            value=float(values[0, 0].item()),
            state=RecurrentState(next_state.actor.detach(), next_state.critic.detach()),
        )

    @torch.no_grad()
    def actor_action(
        self,
        actor_observation: torch.Tensor,
        legal_mask: torch.Tensor,
        actor_hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor]:
        """Choose an action for a frozen opponent without evaluating privileged critic input."""
        encoded = self.actor_encoder(actor_observation.view(1, 1, -1))
        output, next_hidden = self.actor_gru(encoded, actor_hidden)
        logits = self.policy_head(output)[0, 0]
        masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        distribution = Categorical(logits=masked)
        action = torch.argmax(masked) if deterministic else distribution.sample()
        return int(action.item()), next_hidden.detach()
