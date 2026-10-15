"""Public-only per-seat beliefs and learned, non-additive multi-route action fusion.

SAFETY: No game/session, label, replay, scorer or wall API is imported here. Beliefs
are predictions, not hidden observations. Auxiliary fitting is detached from the
original Actor; the residual is optimized exclusively by true-net-score PPO.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from pdh_v2.policy import CandidateActorCritic, ModelConfig

ROUTES = slice(0, 8)
READY = slice(8, 9)
WIN = slice(9, 43)
PONG = slice(43, 77)
KONG = slice(77, 111)
NEXT_KIND = slice(111, 118)
NEXT_DISCARD = slice(118, 152)
VALUE = slice(152, 153)
BELIEF_DIM = 153
MODES = ("baseline", "diagnostic", "joint")


def probabilities(raw: Tensor) -> Tensor:
    """Decode compatible routes, current capabilities, and behavior-policy forecasts."""
    return torch.cat((raw[..., ROUTES].tanh(), raw[..., 8:111].sigmoid(),
                      raw[..., NEXT_KIND].softmax(-1), raw[..., NEXT_DISCARD].softmax(-1),
                      torch.nn.functional.softplus(raw[..., VALUE])), dim=-1)


class SeatBelief(nn.Module):
    """Shared head with explicit relative-seat identity; no closed-hand gating."""

    def __init__(self, hidden: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden + 622 + 102 + 3, 128), nn.SiLU(),
                                  nn.Linear(128, BELIEF_DIM))
        self.register_buffer("seat_ids", torch.eye(3))
        self.register_buffer("qualified", torch.zeros(3, BELIEF_DIM))
        self.register_buffer("prior", torch.zeros(3, BELIEF_DIM))
        self.prior[:, NEXT_KIND] = 1/7
        self.prior[:, NEXT_DISCARD] = 1/34

    def forward(self, observations: Tensor, hidden: Tensor) -> Tensor:
        """Return [decision, next/opposite/previous seat, target] from lawful inputs."""
        context = torch.cat((observations.detach(), hidden.detach()), -1)
        shared = context[:, None, :].expand(-1, 3, -1)
        # NOTE: an explicit lawful per-opponent public block makes seat correspondence testable.
        public = observations.detach()[:, 170:476].reshape(-1, 3, 102)
        seats = self.seat_ids[None, :, :].expand(context.shape[0], -1, -1)
        return self.head(torch.cat((shared, public, seats), -1))

    def forecast(self, observations: Tensor, hidden: Tensor) -> Tensor:
        """Use only independently qualified heads; all other tasks revert to TRAIN priors."""
        estimated = probabilities(self(observations, hidden))
        return estimated*self.qualified + self.prior*(1-self.qualified)


class TableActorCritic(CandidateActorCritic):
    """Original recurrent policy plus a zero-initialized nonlinear table residual."""

    def __init__(self, config: ModelConfig, mode: str):
        if mode not in MODES:
            raise ValueError("rc5.invalid_mode")
        super().__init__(config)
        self.mode = mode
        self.belief = SeatBelief(config.hidden_size)
        # 74 actions + 8 own routes + 459 beliefs + 192 route interactions + 12 tile responses.
        self.fusion = nn.Sequential(nn.Linear(745, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    @classmethod
    def from_baseline(cls, baseline: CandidateActorCritic, mode: str):
        """Copy every existing tensor; additive modules never rename legacy weights."""
        model = cls(baseline.config, mode).to(next(baseline.parameters()).device)
        state = model.state_dict()
        state.update(baseline.state_dict())
        model.load_state_dict(state, strict=True)
        return model

    def fusion_inputs(self, observations: Tensor, features: Tensor,
                      owners: Tensor, belief: Tensor) -> Tensor:
        """Couple candidate post-action routes with each opponent, not a fixed risk sum."""
        owned = belief.index_select(0, owners)
        own_routes = observations[:, 572:580].index_select(0, owners)
        after_routes = features[:, 46:54]
        interactions = (after_routes[:, None, :, None]
                        * owned[..., ROUTES][:, :, None, :]).flatten(1)
        tiles = features[:, 7:41]
        tiles = tiles / tiles.sum(-1, keepdim=True).clamp_min(1e-9)
        responses = torch.stack([(owned[..., part] * tiles[:, None, :]).sum(-1)
                                 for part in (WIN, PONG, KONG, NEXT_DISCARD)], -1).flatten(1)
        return torch.cat((features, own_routes, owned.flatten(1), interactions, responses), -1)

    def forward(self, *args):
        """Expose ordinary module invocation for isolated tests and inference clients."""
        return self.forward_step(*args)

    def forward_step(self, observations, hidden, action_features, action_owners):
        """Do not persist belief caches or cross-hand memory; callers retain only GRU."""
        logits, risk, values, state = super().forward_step(
            observations, hidden, action_features, action_owners)
        if self.mode == "baseline":
            return logits, risk, values, state
        belief = self.belief.forecast(observations, state).detach()
        if self.mode == "joint":
            inputs = self.fusion_inputs(observations, action_features, action_owners, belief)
            logits = logits + self.fusion(inputs).squeeze(-1)
        return logits, risk, values, state