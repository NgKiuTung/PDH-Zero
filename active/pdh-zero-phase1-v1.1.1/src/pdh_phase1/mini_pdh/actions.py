"""Action constants for the fixed Mini-PDH v0 action space."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Semantic actions shared by the environment and policy network."""

    PASS = 13
    HU = 14
    PONG = 15


ACTION_COUNT = 16
DISCARD_ACTIONS = tuple(range(13))


def action_name(action: int) -> str:
    """Return a human-readable action name for logs and replay artifacts."""
    if action in DISCARD_ACTIONS:
        return f"DISCARD_{action}"
    try:
        return Action(action).name
    except ValueError as exc:
        raise ValueError(f"Unknown action: {action}") from exc
