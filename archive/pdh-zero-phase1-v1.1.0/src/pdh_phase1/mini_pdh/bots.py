"""Deterministic baseline controllers for Mini-PDH v0."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .actions import DISCARD_ACTIONS, Action
from .env import MiniPDHEnv
from .types import Phase, TERMINAL_HONOR_TYPES


class Controller(Protocol):
    """Decision interface shared by rule bots and neural controllers."""

    def reset(self) -> None:
        """Reset episode-local controller state."""

    def choose_action(self, env: MiniPDHEnv) -> int:
        """Return one action that is legal in the current environment state."""


@dataclass
class RandomLegalBot:
    """Uniformly sample from the current legal action set."""

    seed: int

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        """Keep the seeded stream continuous across games for reproducibility."""

    def choose_action(self, env: MiniPDHEnv) -> int:
        """Sample one legal action without using hidden information."""
        legal = [index for index, allowed in enumerate(env.legal_actions()) if allowed]
        return self._rng.choice(legal)


@dataclass
class GoalDistanceBot:
    """Simple public-rule heuristic used as a stronger non-learning baseline."""

    seed: int

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        """The bot has no episode-local state."""

    def choose_action(self, env: MiniPDHEnv) -> int:
        """Prefer immediate wins, useful PONGs, and discards with high structural potential."""
        legal = env.legal_actions()
        if legal[Action.HU]:
            return int(Action.HU)
        if env.phase == Phase.RESPONSE_PONG:
            tile = env.last_discard
            if tile is not None and self._should_pong(env, tile):
                return int(Action.PONG)
            return int(Action.PASS)
        if env.phase == Phase.RESPONSE_HU:
            return int(Action.PASS)
        candidates = [tile for tile in DISCARD_ACTIONS if legal[tile]]
        scored = [(self._post_discard_score(env, tile), -tile, tile) for tile in candidates]
        best_score = max(item[0] for item in scored)
        best = [tile for score, _, tile in scored if score == best_score]
        return self._rng.choice(best)

    def _should_pong(self, env: MiniPDHEnv, tile: int) -> bool:
        seat = env.current_seat
        honors_or_terminals = tile in TERMINAL_HONOR_TYPES
        pair_density = sum(amount >= 2 for amount in env.hands[seat])
        return honors_or_terminals or pair_density >= 2

    def _post_discard_score(self, env: MiniPDHEnv, discard: int) -> float:
        counts = list(env.hands[env.current_seat])
        counts[discard] -= 1
        pairs = sum(value // 2 for value in counts)
        triplets = sum(value // 3 for value in counts)
        adjacent = sum(min(counts[index], counts[index + 1]) for index in range(8))
        honors = sum(counts[index] for index in range(9, 13))
        numbers = sum(counts[:9])
        pure_bias = max(honors, numbers) * 0.15
        terminal_bias = sum(counts[index] for index in TERMINAL_HONOR_TYPES) * 0.05
        return 2.0 * pairs + 3.0 * triplets + 0.4 * adjacent + pure_bias + terminal_bias
