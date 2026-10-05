"""按手局组织的循环 PPO 训练数据。"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(slots=True)
class Transition:
    """一次实际策略决策；候选动作集合保持当时顺序。"""

    observation: Tensor
    action_features: Tensor
    action_index: int
    old_log_prob: float
    old_value: float
    reward: float = 0.0


@dataclass(slots=True)
class Episode:
    """一个座位在一手内的完整决策序列。"""

    transitions: list[Transition] = field(default_factory=list)
    advantage: Tensor | None = None
    returns: Tensor | None = None

    @property
    def decisions(self) -> int:
        """返回实际策略决策数。"""
        return len(self.transitions)

    def finish(self, reward: float, gamma: float, gae_lambda: float) -> None:
        """写入终局净分并计算终止序列 GAE。"""
        if not self.transitions:
            return
        self.transitions[-1].reward += reward
        advantages = torch.zeros(len(self.transitions), dtype=torch.float32)
        next_value = 0.0
        next_advantage = 0.0
        for index in range(len(self.transitions) - 1, -1, -1):
            item = self.transitions[index]
            delta = item.reward + gamma * next_value - item.old_value
            next_advantage = delta + gamma * gae_lambda * next_advantage
            advantages[index] = next_advantage
            next_value = item.old_value
        values = torch.tensor(
            [item.old_value for item in self.transitions], dtype=torch.float32
        )
        self.advantage = advantages
        self.returns = advantages + values


def normalize_advantages(episodes: list[Episode]) -> None:
    """对本轮全部有效决策统一标准化优势。"""
    tensors = [episode.advantage for episode in episodes if episode.advantage is not None]
    if not tensors:
        raise RuntimeError("no advantages available")
    merged = torch.cat(tensors)
    mean = merged.mean()
    std = merged.std(unbiased=False).clamp_min(1e-8)
    for episode in episodes:
        if episode.advantage is not None:
            episode.advantage = (episode.advantage - mean) / std
