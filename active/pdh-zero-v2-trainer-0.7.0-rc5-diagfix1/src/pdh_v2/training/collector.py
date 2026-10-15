"""固定策略版本的混合对手多桌批量采样。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import numpy as np
from torch import Tensor
from torch.distributions import Categorical

from pdh_v2.config import PLAYERS
from pdh_v2.game.actions import ActionKind
from pdh_v2.observation.encoder import ACTION_KIND_INDEX
from pdh_v2.policy import CandidateActorCritic
from pdh_v2.rollout import BatchedSessions, TerminalHand
from pdh_v2.training.buffer import Episode, Transition, normalize_advantages
from pdh_v2.training.curriculum import SeatMix


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """一次 PPO 采样批次配置。"""

    minimum_hands: int = 64
    reward_scale: float = 32.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    seat_mix: SeatMix = SeatMix((1.0, 0.0, 0.0, 0.0))
    seat_seed: int = 0x51EA7


@dataclass(frozen=True, slots=True)
class CollectionResult:
    """已排空手局的训练数据和原始业务指标。"""

    episodes: tuple[Episode, ...]
    terminals: tuple[TerminalHand, ...]
    decisions: int
    environment_decisions: int
    heuristic_decisions: int
    hands: int
    draws: int
    wins: int
    point_shoots: int
    learner_wins: int
    learner_point_shoots: int
    learner_score_sum: int
    raw_score_sum: tuple[int, ...]
    learner_seat_histogram: tuple[int, ...]
    action_counts: tuple[int, ...]
    win_opportunities: int
    wins_taken: int
    missed_wins: int
    risk_samples: int
    risk_positives: int
    learner_seat_hands: int = 0
    learner_win_count: int = 0
    pong_opportunities: int = 0
    declined_pongs: int = 0


class RolloutCollector:
    """对同一策略版本采样；规则对手行动不进入当前策略 PPO。"""

    def __init__(
        self,
        environments: BatchedSessions,
        model: CandidateActorCritic,
        device: torch.device,
        config: CollectionConfig,
        progress: Callable[[int, int], None] | None = None,
        on_terminal: Callable[[TerminalHand], None] | None = None,
    ) -> None:
        self.environments = environments
        self.model = model
        self.device = device
        self.config = config
        self.progress = progress
        self.on_terminal = on_terminal
        self.seat_mix = config.seat_mix
        hidden = model.config.hidden_size
        self.hidden = torch.zeros(
            environments.count,
            PLAYERS,
            hidden,
            device=device,
            dtype=torch.float32,
        )
        self.current = [
            [Episode() for _ in range(PLAYERS)] for _ in range(environments.count)
        ]

    def set_seat_mix(self, seat_mix: SeatMix) -> None:
        """在完整 PPO 更新边界切换下一批课程。"""
        self.seat_mix = seat_mix

    @staticmethod
    def _hand_index(hand_id: str) -> int:
        if not hand_id.startswith("H"):
            raise ValueError(f"invalid hand id: {hand_id}")
        return int(hand_id[1:])

    def _learner_seats(self, env_index: int, hand_id: str) -> tuple[int, ...]:
        return self.seat_mix.learner_seats(
            self.config.seat_seed,
            env_index,
            self._hand_index(hand_id),
        )

    @staticmethod
    def _pack_action_features(requests) -> tuple[Tensor, Tensor, list[int]]:
        features = []
        owners: list[int] = []
        counts: list[int] = []
        for owner, request in enumerate(requests):
            count = len(request.view.action_features)
            counts.append(count)
            features.append(np.asarray(request.view.action_features, dtype=np.float32))
            owners.extend([owner] * count)
        return (
            torch.from_numpy(np.concatenate(features, axis=0)),
            torch.tensor(owners, dtype=torch.long),
            counts,
        )

    @staticmethod
    def _sample(logits: Tensor, counts: list[int]) -> tuple[Tensor, Tensor, Tensor]:
        max_actions = max(counts)
        padded = torch.full(
            (len(counts), max_actions),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
        cursor = 0
        for row, count in enumerate(counts):
            padded[row, :count] = logits[cursor : cursor + count]
            cursor += count
        distribution = Categorical(logits=padded)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy()

    def _finalize_hand(
        self,
        terminal: TerminalHand,
        completed: list[Episode],
    ) -> tuple[int, int, int, int, int]:
        env_index = terminal.env_index
        learner_seats = self._learner_seats(env_index, terminal.hand_id)
        risk_samples = 0
        risk_positives = 0
        discard_id = ACTION_KIND_INDEX[ActionKind.DISCARD]
        # NOTE: 抢加杠的责任牌不是上一张弃牌，不能回填成弃牌风险正样本。
        if terminal.method == "DISCARD_WIN" and terminal.discarder in learner_seats:
            episode = self.current[env_index][terminal.discarder]
            for transition in reversed(episode.transitions):
                if transition.action_kind == discard_id:
                    transition.risk_target = 1.0
                    risk_positives += 1
                    break

        score_sum = sum(terminal.receipt[seat] for seat in learner_seats)
        for seat in range(PLAYERS):
            episode = self.current[env_index][seat]
            if seat in learner_seats and episode.transitions:
                for transition in episode.transitions:
                    risk_samples += int(transition.risk_target is not None)
                episode.finish(
                    terminal.receipt[seat] / self.config.reward_scale,
                    self.config.gamma,
                    self.config.gae_lambda,
                )
                episode.group_id = f"{env_index}:{terminal.hand_id}"
                completed.append(episode)
            self.current[env_index][seat] = Episode()
        self.hidden[env_index].zero_()
        learner_won = int(any(seat in learner_seats for seat in terminal.winners))
        learner_shot = int(terminal.discarder in learner_seats)
        return learner_won, learner_shot, score_sum, risk_samples, risk_positives

    @torch.no_grad()
    def collect(self) -> CollectionResult:
        """采集至少指定手数，并完整排空当时所有进行中的手牌。"""
        self.model.eval()
        self.environments.reactivate_all()
        completed: list[Episode] = []
        terminals: list[TerminalHand] = []
        draining = False
        drain_trigger = max(1, self.config.minimum_hands - self.environments.count)
        learner_decisions = 0
        environment_decisions = 0
        heuristic_decisions = 0
        learner_wins = 0
        learner_point_shoots = 0
        learner_score_sum = 0
        seat_histogram = [0] * (PLAYERS + 1)
        action_counts = [0] * len(ActionKind)
        win_opportunities = 0
        wins_taken = 0
        risk_samples = 0
        risk_positives = 0
        learner_win_count = 0
        pong_opportunities = 0
        declined_pongs = 0

        while not self.environments.all_inactive():
            if self.progress is not None:
                self.progress(len(terminals), learner_decisions)
            requests = self.environments.requests()
            if not requests:
                break
            environment_decisions += len(requests)
            model_requests = [
                request
                for request in requests
                if request.view.player
                in self._learner_seats(request.env_index, request.hand_id)
            ]
            choices: dict[int, int] = {}

            if model_requests:
                observations_cpu = torch.from_numpy(np.stack(
                    [np.asarray(request.view.observation, dtype=np.float32)
                     for request in model_requests]))
                action_cpu, owners_cpu, counts = self._pack_action_features(model_requests)
                env_indices = torch.tensor(
                    [request.env_index for request in model_requests],
                    dtype=torch.long,
                    device=self.device,
                )
                players = torch.tensor(
                    [request.view.player for request in model_requests],
                    dtype=torch.long,
                    device=self.device,
                )
                observations = observations_cpu.to(self.device, non_blocking=True)
                action_features = action_cpu.to(self.device, non_blocking=True)
                owners = owners_cpu.to(self.device, non_blocking=True)
                hidden = self.hidden[env_indices, players]
                logits, _, values, next_hidden = self.model.forward_step(
                    observations,
                    hidden,
                    action_features,
                    owners,
                )
                selected, log_probs, _ = self._sample(logits, counts)
                self.hidden[env_indices, players] = next_hidden

                log_prob_values = log_probs.cpu().tolist()
                value_values = values.cpu().tolist()
                selected_values = selected.cpu().tolist()
                for row, request in enumerate(model_requests):
                    selected_index = int(selected_values[row])
                    action = request.view.actions[selected_index]
                    choices[request.env_index] = selected_index
                    has_win = any(
                        candidate.kind is ActionKind.WIN
                        for candidate in request.view.actions
                    )
                    took_win = action.kind is ActionKind.WIN
                    has_pong = any(a.kind is ActionKind.PONG for a in request.view.actions)
                    pong_opportunities += int(has_pong)
                    declined_pongs += int(has_pong and action.kind is not ActionKind.PONG)
                    win_opportunities += int(has_win)
                    wins_taken += int(took_win)
                    action_counts[ACTION_KIND_INDEX[action.kind]] += 1
                    risk_target = 0.0 if action.kind is ActionKind.DISCARD else None
                    episode = self.current[request.env_index][request.view.player]
                    episode.transitions.append(
                        Transition(
                            observations_cpu[row].clone(),
                            torch.tensor(
                                request.view.action_features,
                                dtype=torch.float32,
                            ),
                            selected_index,
                            float(log_prob_values[row]),
                            float(value_values[row]),
                            action_kind=ACTION_KIND_INDEX[action.kind],
                            risk_target=risk_target,
                            win_available=has_win,
                        )
                    )
                learner_decisions += len(model_requests)

            heuristic_requests = tuple(r for r in requests if r.env_index not in choices)
            choices.update(self.environments.heuristic_actions(heuristic_requests))
            heuristic_decisions += len(heuristic_requests)

            finished = self.environments.step(choices, drain=draining)
            for terminal in finished:
                terminals.append(terminal)
                if self.on_terminal is not None:
                    self.on_terminal(terminal)
                learner_seats = self._learner_seats(terminal.env_index, terminal.hand_id)
                seat_histogram[len(learner_seats)] += 1
                learner_win_count += sum(seat in terminal.winners for seat in learner_seats)
                won, shot, score, hand_risk_samples, hand_risk_positives = (
                    self._finalize_hand(terminal, completed)
                )
                learner_wins += won
                learner_point_shoots += shot
                learner_score_sum += score
                risk_samples += hand_risk_samples
                risk_positives += hand_risk_positives
            if not draining and len(terminals) >= drain_trigger:
                draining = True

        if not completed:
            raise RuntimeError("collection produced no learner episodes")
        normalize_advantages(completed)
        raw_scores = [0] * PLAYERS
        for terminal in terminals:
            for seat, value in enumerate(terminal.receipt):
                raw_scores[seat] += value
        return CollectionResult(
            tuple(completed),
            tuple(terminals),
            learner_decisions,
            environment_decisions,
            heuristic_decisions,
            len(terminals),
            sum(item.draw for item in terminals),
            sum(not item.draw for item in terminals),
            sum(item.discarder is not None for item in terminals),
            learner_wins,
            learner_point_shoots,
            learner_score_sum,
            tuple(raw_scores),
            tuple(seat_histogram),
            tuple(action_counts),
            win_opportunities,
            wins_taken,
            win_opportunities - wins_taken,
            risk_samples,
            risk_positives,
            sum(i * n for i, n in enumerate(seat_histogram)),
            learner_win_count,
            pong_opportunities,
            declined_pongs,
        )
