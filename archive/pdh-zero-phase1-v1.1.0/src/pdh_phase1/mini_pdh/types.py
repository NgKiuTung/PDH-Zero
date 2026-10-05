"""Core immutable types and constants for Mini-PDH v0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


TILE_TYPES = 13
PLAYERS = 4
COPIES_PER_TILE = 4
NUMBER_TILE_TYPES = tuple(range(9))
HONOR_TILE_TYPES = tuple(range(9, 13))
TERMINAL_HONOR_TYPES = (0, 8, 9, 10, 11, 12)
TILE_NAMES = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "E", "S", "W", "N")


class Phase(str, Enum):
    """Decision phases exposed through the actor observation."""

    SELF_ACTION = "SELF_ACTION"
    DISCARD_AFTER_PONG = "DISCARD_AFTER_PONG"
    RESPONSE_HU = "RESPONSE_HU"
    RESPONSE_PONG = "RESPONSE_PONG"


PHASES = tuple(Phase)


@dataclass(frozen=True)
class WinResult:
    """Validated winning structure and its maximum Mini-PDH multiplier."""

    multiplier: int
    patterns: tuple[str, ...]
    selected_pattern: str


@dataclass(frozen=True)
class StepResult:
    """One environment transition returned by :meth:`MiniPDHEnv.step`."""

    done: bool
    rewards: tuple[float, float, float, float]
    event: dict[str, Any]
