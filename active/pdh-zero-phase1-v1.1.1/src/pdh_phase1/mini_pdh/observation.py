"""Actor and privileged-critic observation encoding for Mini-PDH v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .types import PHASES, PLAYERS, TILE_TYPES


ACTOR_OBSERVATION_DIM = 159
PRIVILEGED_EXTRA_DIM = 65
PRIVILEGED_OBSERVATION_DIM = ACTOR_OBSERVATION_DIM + PRIVILEGED_EXTRA_DIM


class ObservableState(Protocol):
    """Minimal state contract required by the observation encoder."""

    hands: list[list[int]]
    open_pongs: list[list[int]]
    discards: list[list[int]]
    wall: list[int]
    phase: object
    current_seat: int
    dealer_seat: int
    last_discard: int | None
    discard_source: int | None
    decision_count: int
    max_decisions: int


def _relative_seats(seat: int) -> tuple[int, int, int, int]:
    """Return self, next, opposite, and previous absolute seats."""
    return tuple((seat + offset) % PLAYERS for offset in range(PLAYERS))


def _tile_histogram(tiles: list[int]) -> list[float]:
    """Encode a public tile list as normalized counts."""
    counts = [0.0] * TILE_TYPES
    for tile in tiles:
        counts[tile] += 0.25
    return counts


def encode_actor(state: ObservableState, seat: int) -> torch.Tensor:
    """Encode only the seat's private hand and publicly observable game state."""
    relative = _relative_seats(seat)
    values: list[float] = [value / 4.0 for value in state.hands[seat]]
    values.extend(_tile_histogram(state.open_pongs[seat]))
    for absolute in relative:
        values.extend(_tile_histogram(state.discards[absolute]))
    for absolute in relative:
        values.extend(_tile_histogram(state.open_pongs[absolute]))
    values.extend(1.0 if state.phase == phase else 0.0 for phase in PHASES)
    dealer_relative = (state.dealer_seat - seat) % PLAYERS
    values.extend(1.0 if index == dealer_relative else 0.0 for index in range(PLAYERS))
    values.extend(
        1.0 if state.last_discard == index else 0.0 for index in range(TILE_TYPES)
    )
    values.append(1.0 if state.last_discard is None else 0.0)
    source_relative = None if state.discard_source is None else (state.discard_source - seat) % PLAYERS
    values.extend(1.0 if source_relative == index else 0.0 for index in range(PLAYERS))
    values.append(1.0 if source_relative is None else 0.0)
    values.append(len(state.wall) / 52.0)
    values.append(min(state.decision_count / max(state.max_decisions, 1), 1.0))
    if len(values) != ACTOR_OBSERVATION_DIM:
        raise RuntimeError(f"Actor observation dimension mismatch: {len(values)}")
    return torch.tensor(values, dtype=torch.float32)


def encode_critic(state: ObservableState, seat: int, privileged: bool) -> torch.Tensor:
    """Append all hidden hands and wall counts only for the training critic."""
    actor = encode_actor(state, seat)
    if not privileged:
        return actor
    relative = _relative_seats(seat)
    extras: list[float] = []
    for absolute in relative:
        extras.extend(value / 4.0 for value in state.hands[absolute])
    wall_counts = [0.0] * TILE_TYPES
    for tile in state.wall:
        wall_counts[tile] += 0.25
    extras.extend(wall_counts)
    if len(extras) != PRIVILEGED_EXTRA_DIM:
        raise RuntimeError(f"Privileged observation dimension mismatch: {len(extras)}")
    return torch.cat((actor, torch.tensor(extras, dtype=torch.float32)))
