"""固定牌序下候选模型对三家启发式的健康评估。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

import torch
from torch import Tensor

from pdh_v2.config import PLAYERS
from pdh_v2.policy import CandidateActorCritic, choose_action
from pdh_v2.rollout import BatchedSessions


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """长训练期间的固定开发集评估配置。"""

    hands: int = 2048
    environments: int = 64
    seed: int = 0xECA12027


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """候选物理座位轮换后的攻防与收益指标。"""

    hands: int
    mean_score: float
    score_std: float
    cvar_5: float
    win_rate: float
    self_draw_rate: float
    discard_win_rate: float
    point_shoot_rate: float
    draw_rate: float
    minimum_score: int
    maximum_score: int

    def to_dict(self) -> dict[str, int | float]:
        """返回可直接写入 JSON 的稳定字段。"""
        return dict(asdict(self))


def _pack(requests) -> tuple[Tensor, Tensor, list[int]]:
    features: list[tuple[float, ...]] = []
    owners: list[int] = []
    counts: list[int] = []
    for owner, request in enumerate(requests):
        count = len(request.view.action_features)
        counts.append(count)
        features.extend(request.view.action_features)
        owners.extend([owner] * count)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(owners, dtype=torch.long),
        counts,
    )


def _greedy(logits: Tensor, counts: list[int]) -> list[int]:
    result: list[int] = []
    cursor = 0
    for count in counts:
        result.append(int(logits[cursor : cursor + count].argmax().item()))
        cursor += count
    return result


@torch.no_grad()
def evaluate_vs_heuristic(
    model: CandidateActorCritic,
    device: torch.device,
    config: EvaluationConfig,
) -> EvaluationResult:
    """四座位均衡地评估模型；其他三个固定玩家使用确定性启发式。"""
    if config.hands < 1 or config.environments < PLAYERS:
        raise ValueError("evaluation budget is too small")
    model.eval()
    environments = BatchedSessions(config.environments, config.seed)
    candidate_seats = tuple(index % PLAYERS for index in range(environments.count))
    hidden = model.initial_state(environments.count, device)
    scores: list[int] = []
    wins = self_draws = discard_wins = point_shoots = draws = 0
    draining = False
    drain_trigger = max(1, config.hands - environments.count)

    while not environments.all_inactive():
        requests = environments.requests()
        choices: dict[int, int] = {}
        model_requests = [
            request
            for request in requests
            if request.view.player == candidate_seats[request.env_index]
        ]
        if model_requests:
            observations = torch.tensor(
                [request.view.observation for request in model_requests],
                dtype=torch.float32,
                device=device,
            )
            action_cpu, owners_cpu, counts = _pack(model_requests)
            action_features = action_cpu.to(device)
            owners = owners_cpu.to(device)
            indices = torch.tensor(
                [request.env_index for request in model_requests],
                dtype=torch.long,
                device=device,
            )
            logits, _, _, next_hidden = model.forward_step(
                observations,
                hidden.index_select(0, indices),
                action_features,
                owners,
            )
            selected = _greedy(logits, counts)
            hidden.index_copy_(0, indices, next_hidden)
            for request, action_index in zip(model_requests, selected):
                choices[request.env_index] = action_index
        for request in requests:
            if request.env_index in choices:
                continue
            session = environments.sessions[request.env_index]
            choices[request.env_index] = choose_action(session, request.view.actions)

        terminals = environments.step(choices, drain=draining)
        for terminal in terminals:
            if sum(terminal.receipt) != 0:
                raise RuntimeError("evaluation terminal is not zero-sum")
            seat = candidate_seats[terminal.env_index]
            score = int(terminal.receipt[seat])
            scores.append(score)
            draws += int(terminal.draw)
            candidate_won = seat in terminal.winners
            wins += int(candidate_won)
            self_draws += int(candidate_won and terminal.method == "SELF_DRAW")
            discard_wins += int(candidate_won and terminal.discarder is not None)
            point_shoots += int(terminal.discarder == seat)
            hidden[terminal.env_index].zero_()
        if not draining and len(scores) >= drain_trigger:
            draining = True

    if not scores:
        raise RuntimeError("evaluation produced no hands")
    ordered = sorted(scores)
    tail_count = max(1, math.ceil(len(ordered) * 0.05))
    return EvaluationResult(
        hands=len(scores),
        mean_score=statistics.fmean(scores),
        score_std=statistics.pstdev(scores),
        cvar_5=statistics.fmean(ordered[:tail_count]),
        win_rate=wins / len(scores),
        self_draw_rate=self_draws / len(scores),
        discard_win_rate=discard_wins / len(scores),
        point_shoot_rate=point_shoots / len(scores),
        draw_rate=draws / len(scores),
        minimum_score=min(scores),
        maximum_score=max(scores),
    )
