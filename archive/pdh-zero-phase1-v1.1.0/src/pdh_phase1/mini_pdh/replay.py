"""Deterministic replay verification for Mini-PDH v0."""

from __future__ import annotations

from typing import Any

from .env import MiniPDHEnv


def replay_game(config: dict[str, Any], record: dict[str, Any]) -> MiniPDHEnv:
    """Replay one exported record and verify all expected terminal evidence."""
    if record.get("ruleset_id") != config.get("ruleset_id"):
        raise ValueError("Replay ruleset does not match the supplied configuration")
    env = MiniPDHEnv(config)
    env.reset(int(record["seed"]))
    for action in record["actions"]:
        env.step(int(action))
    if not env.done:
        raise AssertionError("Replay action list did not reach a terminal state")
    if list(env.rewards) != list(record["expected_rewards"]):
        raise AssertionError("Replay rewards differ from the recorded evidence")
    if env.outcome != record["expected_outcome"]:
        raise AssertionError("Replay outcome differs from the recorded evidence")
    if env.state_digest() != record["expected_state_digest"]:
        raise AssertionError("Replay state digest differs from the recorded evidence")
    return env
