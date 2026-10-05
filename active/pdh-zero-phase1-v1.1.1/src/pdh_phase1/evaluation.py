"""Fixed-wall, four-seat-rotation evaluation for Mini-PDH policies."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch

from .mini_pdh import MiniPDHEnv, encode_actor
from .mini_pdh.bots import GoalDistanceBot, RandomLegalBot
from .rl.model import RecurrentActorCritic


def evaluate_model(
    model: RecurrentActorCritic,
    rules_config: dict[str, Any],
    device: torch.device,
    seeds: list[int],
    opponent_kind: str = "heuristic",
) -> dict[str, Any]:
    """Evaluate one model at every seat for each deterministic wall seed."""
    model.eval()
    scores: list[float] = []
    wins = 0
    draws = 0
    outcomes: Counter[str] = Counter()
    seat_scores: dict[int, list[float]] = defaultdict(list)
    for seed in seeds:
        for model_seat in range(4):
            env = MiniPDHEnv(rules_config)
            env.reset(seed)
            actor_hidden = torch.zeros(1, 1, model.hidden_size, device=device)
            opponents = {
                seat: _make_rule_bot(opponent_kind, seed * 101 + model_seat * 7 + seat)
                for seat in range(4)
                if seat != model_seat
            }
            while not env.done:
                if env.current_seat == model_seat:
                    observation = encode_actor(env, model_seat).to(device)
                    legal = torch.tensor(env.legal_actions(), dtype=torch.bool, device=device)
                    action, actor_hidden = model.actor_action(
                        observation,
                        legal,
                        actor_hidden,
                        deterministic=True,
                    )
                else:
                    action = opponents[env.current_seat].choose_action(env)
                env.step(action)
            score = float(env.rewards[model_seat])
            scores.append(score)
            seat_scores[model_seat].append(score)
            outcomes[str(env.outcome.get("type", "UNKNOWN"))] += 1
            wins += int(score > 0)
            draws += int(env.outcome.get("type") == "DRAW")
    games = len(scores)
    return {
        "games": games,
        "wall_seeds": len(seeds),
        "opponent_kind": opponent_kind,
        "mean_score": sum(scores) / max(games, 1),
        "win_rate": wins / max(games, 1),
        "draw_rate": draws / max(games, 1),
        "outcomes": dict(sorted(outcomes.items())),
        "seat_mean_scores": {
            str(seat): sum(values) / len(values) for seat, values in sorted(seat_scores.items())
        },
    }


def evaluate_rule_bot(
    rules_config: dict[str, Any],
    seeds: list[int],
    candidate_kind: str,
    opponent_kind: str,
) -> dict[str, Any]:
    """Evaluate a rule bot with the same fixed-wall, four-seat protocol."""
    scores: list[float] = []
    wins = 0
    for seed in seeds:
        for candidate_seat in range(4):
            env = MiniPDHEnv(rules_config)
            env.reset(seed)
            bots = {
                seat: _make_rule_bot(
                    candidate_kind if seat == candidate_seat else opponent_kind,
                    seed * 101 + candidate_seat * 7 + seat,
                )
                for seat in range(4)
            }
            while not env.done:
                env.step(bots[env.current_seat].choose_action(env))
            score = float(env.rewards[candidate_seat])
            scores.append(score)
            wins += int(score > 0)
    return {
        "games": len(scores),
        "candidate_kind": candidate_kind,
        "opponent_kind": opponent_kind,
        "mean_score": sum(scores) / max(len(scores), 1),
        "win_rate": wins / max(len(scores), 1),
    }


def _make_rule_bot(kind: str, seed: int) -> object:
    if kind == "random":
        return RandomLegalBot(seed)
    if kind == "heuristic":
        return GoalDistanceBot(seed)
    raise ValueError(f"Unsupported rule-bot kind: {kind}")
