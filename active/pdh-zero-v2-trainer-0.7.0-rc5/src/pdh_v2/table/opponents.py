"""Frozen, public-view-only opponent conditions for controlled evaluation, not training."""
from __future__ import annotations

import numpy as np
import torch

from pdh_v2.training.collector import RolloutCollector

OPPONENTS = ('heuristic', 'route_mix', 'frozen_rc41')


class FrozenOpponent:
    """Maintain separate within-hand states for every environment/seat; factory resets waves."""

    def __init__(self, model, count: int, device):
        self.model, self.device = model, device
        self.hidden = model.initial_state(count * 4, device)

    @torch.no_grad()
    def __call__(self, backend, requests):
        """The backend is deliberately not inspected for game state or hidden tiles."""
        if not requests:
            return {}
        observations = torch.from_numpy(np.stack([
            np.asarray(r.view.observation, dtype=np.float32) for r in requests])).to(self.device)
        features, owners, counts = RolloutCollector._pack_action_features(requests)
        ids = torch.tensor([4*r.env_index + r.view.player for r in requests], device=self.device)
        logits, _, _, state = self.model.forward_step(observations,
            self.hidden.index_select(0, ids), features.to(self.device), owners.to(self.device))
        self.hidden.index_copy_(0, ids, state)
        values, choices, cursor = logits.cpu(), {}, 0
        for request, count in zip(requests, counts):
            choices[request.env_index] = int(values[cursor:cursor+count].argmax())
            cursor += count
        return choices


def route_mix(backend, requests):
    """Distinct deterministic public-feature policy: two best routes plus action option value."""
    choices = {}
    for request in requests:
        features = np.asarray(request.view.action_features)
        best = np.sort(features[:, 46:54], axis=-1)
        scores = best[:, -1] + .65 * best[:, -2] + .15 * features[:, 73]
        # WIN is always legal here; this opponent never deliberately declines it.
        scores = scores + 100 * features[:, 5]
        choices[request.env_index] = int(scores.argmax())
    return choices


def opponent_factory(name: str, baseline, device):
    """Resolve a frozen condition; unknown opponents cannot silently fall back to the teacher."""
    if name == 'heuristic':
        return None
    if name == 'route_mix':
        return lambda count: route_mix
    if name == 'frozen_rc41':
        return lambda count: FrozenOpponent(baseline, count, device)
    raise ValueError('rc5.invalid_opponent')