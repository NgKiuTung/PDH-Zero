"""One-learner-seat trajectory collection against frozen and rule-based opponents."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from pdh_phase1.mini_pdh import MiniPDHEnv, encode_actor, encode_critic
from pdh_phase1.mini_pdh.bots import GoalDistanceBot, RandomLegalBot

from .controllers import FrozenPolicyController
from .model import RecurrentActorCritic
from .trajectory import EpisodeTrajectory


@dataclass(frozen=True)
class CollectionStats:
    """Measured work and outcomes for one self-play collection batch."""

    episodes: int
    attempts: int
    environment_decisions: int
    mean_score: float
    win_rate: float
    draw_rate: float
    elapsed_seconds: float


def clone_frozen_model(model: RecurrentActorCritic, device: torch.device) -> RecurrentActorCritic:
    """Create a read-only snapshot with identical architecture and parameters."""
    frozen = copy.deepcopy(model).to(device)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def collect_episodes(
    model: RecurrentActorCritic,
    frozen_model: RecurrentActorCritic,
    rules_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
    update_index: int,
) -> tuple[list[EpisodeTrajectory], CollectionStats]:
    """Collect terminal trajectories while updating only one rotating learner seat per game."""
    target = int(training_config["games_per_update"])
    base_seed = int(training_config["seed"]) + update_index * 1_000_003
    rng = random.Random(base_seed)
    episodes: list[EpisodeTrajectory] = []
    attempts = 0
    decisions = 0
    start = time.perf_counter()
    model.eval()
    max_attempts = max(target * 20, 100)
    while len(episodes) < target and attempts < max_attempts:
        game_seed = base_seed + attempts * 17
        learner_seat = attempts % 4
        env = MiniPDHEnv(rules_config)
        env.reset(game_seed)
        trajectory = EpisodeTrajectory(learner_seat=learner_seat)
        learner_state = model.initial_state(1, device)
        controllers = _build_opponents(
            frozen_model,
            device,
            training_config["opponent_mix"],
            rng,
            game_seed,
            learner_seat,
        )
        while not env.done:
            seat = env.current_seat
            if seat == learner_seat:
                actor_observation = encode_actor(env, seat)
                critic_observation = encode_critic(
                    env,
                    seat,
                    bool(training_config["privileged_critic"]),
                )
                legal_mask = torch.tensor(env.legal_actions(), dtype=torch.bool)
                output = model.act(
                    actor_observation.to(device),
                    critic_observation.to(device),
                    legal_mask.to(device),
                    learner_state,
                    deterministic=False,
                )
                learner_state = output.state
                trajectory.append(
                    actor_observation,
                    critic_observation,
                    legal_mask,
                    output.action,
                    output.log_prob,
                    output.value,
                )
                action = output.action
            else:
                action = controllers[seat].choose_action(env)
            env.step(action)
            decisions += 1
        if trajectory.length:
            trajectory.finalize(
                env.rewards[learner_seat],
                str(env.outcome.get("type", "UNKNOWN")),
                float(training_config["reward_normalizer"]),
                float(training_config["gamma"]),
                float(training_config["gae_lambda"]),
            )
            episodes.append(trajectory)
        attempts += 1
    if len(episodes) != target:
        raise RuntimeError(
            f"Collected {len(episodes)}/{target} learner trajectories after {attempts} games"
        )
    elapsed = time.perf_counter() - start
    scores = [episode.final_score for episode in episodes]
    wins = sum(score > 0 for score in scores)
    draws = sum(episode.outcome_type == "DRAW" for episode in episodes)
    return episodes, CollectionStats(
        episodes=len(episodes),
        attempts=attempts,
        environment_decisions=decisions,
        mean_score=sum(scores) / len(scores),
        win_rate=wins / len(scores),
        draw_rate=draws / len(scores),
        elapsed_seconds=elapsed,
    )


def _build_opponents(
    frozen_model: RecurrentActorCritic,
    device: torch.device,
    mix: dict[str, float],
    rng: random.Random,
    game_seed: int,
    learner_seat: int,
) -> dict[int, object]:
    controllers: dict[int, object] = {}
    for seat in range(4):
        if seat == learner_seat:
            continue
        kind = _sample_kind(mix, rng)
        seed = game_seed * 31 + seat
        if kind == "frozen_policy":
            controllers[seat] = FrozenPolicyController(frozen_model, device)
        elif kind == "random":
            controllers[seat] = RandomLegalBot(seed)
        elif kind == "heuristic":
            controllers[seat] = GoalDistanceBot(seed)
        else:
            raise ValueError(f"Unsupported opponent kind: {kind}")
    return controllers


def _sample_kind(mix: dict[str, float], rng: random.Random) -> str:
    names = list(mix)
    weights = [float(mix[name]) for name in names]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"Invalid opponent mix: {mix}")
    return rng.choices(names, weights=weights, k=1)[0]
