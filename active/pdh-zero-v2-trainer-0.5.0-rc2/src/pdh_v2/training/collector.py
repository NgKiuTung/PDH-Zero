"""固定策略版本的多桌批量采样。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from pdh_v2.config import PLAYERS
from pdh_v2.policy import CandidateActorCritic
from pdh_v2.rollout import BatchedSessions, TerminalHand
from pdh_v2.training.buffer import Episode, Transition, normalize_advantages


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """一次 PPO 采样批次配置。"""

    minimum_hands: int = 64
    reward_scale: float = 32.0
    gamma: float = 1.0
    gae_lambda: float = 0.95


@dataclass(frozen=True, slots=True)
class CollectionResult:
    """已排空手局的训练数据和原始业务指标。"""

    episodes: tuple[Episode, ...]
    terminals: tuple[TerminalHand, ...]
    decisions: int
    hands: int
    draws: int
    wins: int
    point_shoots: int
    raw_score_sum: tuple[int, ...]


class RolloutCollector:
    """对同一策略版本采样，并在更新前排空所有进行中的手牌。"""

    def __init__(
        self,
        environments: BatchedSessions,
        model: CandidateActorCritic,
        device: torch.device,
        config: CollectionConfig,
    ) -> None:
        self.environments = environments
        self.model = model
        self.device = device
        self.config = config
        hidden = model.config.hidden_size
        self.hidden = torch.zeros(
            environments.count, PLAYERS, hidden, device=device, dtype=torch.float32
        )
        self.current = [
            [Episode() for _ in range(PLAYERS)] for _ in range(environments.count)
        ]

    @staticmethod
    def _pack_action_features(requests) -> tuple[Tensor, Tensor, list[int]]:
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
    ) -> None:
        env_index = terminal.env_index
        for seat in range(PLAYERS):
            episode = self.current[env_index][seat]
            if episode.transitions:
                episode.finish(
                    terminal.receipt[seat] / self.config.reward_scale,
                    self.config.gamma,
                    self.config.gae_lambda,
                )
                completed.append(episode)
            self.current[env_index][seat] = Episode()
        self.hidden[env_index].zero_()

    @torch.no_grad()
    def collect(self) -> CollectionResult:
        """采集至少指定手数，并完整排空当时所有进行中的手牌。"""
        self.model.eval()
        self.environments.reactivate_all()
        completed: list[Episode] = []
        terminals: list[TerminalHand] = []
        draining = False
        drain_trigger = max(1, self.config.minimum_hands - self.environments.count)
        decisions = 0

        while not self.environments.all_inactive():
            requests = self.environments.requests()
            if not requests:
                break
            observations_cpu = torch.tensor(
                [request.view.observation for request in requests], dtype=torch.float32
            )
            action_cpu, owners_cpu, counts = self._pack_action_features(requests)
            env_indices = torch.tensor(
                [request.env_index for request in requests], dtype=torch.long, device=self.device
            )
            players = torch.tensor(
                [request.view.player for request in requests], dtype=torch.long, device=self.device
            )
            observations = observations_cpu.to(self.device, non_blocking=True)
            action_features = action_cpu.to(self.device, non_blocking=True)
            owners = owners_cpu.to(self.device, non_blocking=True)
            hidden = self.hidden[env_indices, players]
            logits, values, next_hidden = self.model.forward_step(
                observations, hidden, action_features, owners
            )
            selected, log_probs, _ = self._sample(logits, counts)
            self.hidden[env_indices, players] = next_hidden

            choices = {
                request.env_index: int(action)
                for request, action in zip(requests, selected.cpu().tolist())
            }
            log_prob_values = log_probs.cpu().tolist()
            value_values = values.cpu().tolist()
            selected_values = selected.cpu().tolist()
            for row, request in enumerate(requests):
                episode = self.current[request.env_index][request.view.player]
                episode.transitions.append(
                    Transition(
                        observations_cpu[row].clone(),
                        torch.tensor(request.view.action_features, dtype=torch.float32),
                        int(selected_values[row]),
                        float(log_prob_values[row]),
                        float(value_values[row]),
                    )
                )
            decisions += len(requests)
            finished = self.environments.step(choices, drain=draining)
            for terminal in finished:
                terminals.append(terminal)
                self._finalize_hand(terminal, completed)
            if not draining and len(terminals) >= drain_trigger:
                draining = True

        normalize_advantages(completed)
        raw_scores = [0] * PLAYERS
        for terminal in terminals:
            for seat, value in enumerate(terminal.receipt):
                raw_scores[seat] += value
        result = CollectionResult(
            tuple(completed),
            tuple(terminals),
            decisions,
            len(terminals),
            sum(item.draw for item in terminals),
            sum(not item.draw for item in terminals),
            sum(item.discarder is not None for item in terminals),
            tuple(raw_scores),
        )
        return result
