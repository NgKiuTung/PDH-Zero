"""Neural opponent controller used by historical self-play collection."""

from __future__ import annotations

import torch

from pdh_phase1.mini_pdh import MiniPDHEnv, encode_actor

from .model import RecurrentActorCritic


class FrozenPolicyController:
    """One seat's recurrent state backed by a read-only policy snapshot."""

    def __init__(self, model: RecurrentActorCritic, device: torch.device, deterministic: bool = False):
        """Bind a frozen model and create episode-local hidden state on reset."""
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.actor_hidden = torch.empty(0)
        self.reset()

    def reset(self) -> None:
        """Reset the opponent's GRU stream at the start of a game."""
        self.actor_hidden = torch.zeros(
            1,
            1,
            self.model.hidden_size,
            device=self.device,
        )

    def choose_action(self, env: MiniPDHEnv) -> int:
        """Choose from public information only and update this seat's hidden state."""
        actor_observation = encode_actor(env, env.current_seat).to(self.device)
        legal_mask = torch.tensor(env.legal_actions(), dtype=torch.bool, device=self.device)
        action, self.actor_hidden = self.model.actor_action(
            actor_observation,
            legal_mask,
            self.actor_hidden,
            deterministic=self.deterministic,
        )
        return action
