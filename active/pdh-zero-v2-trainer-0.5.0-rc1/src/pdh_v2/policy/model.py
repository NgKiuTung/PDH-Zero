"""候选动作式循环 Actor-Critic。

模型只对规则层给出的合法动作评分，因此不会通过大离散动作表产生非法动作。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from pdh_v2.observation import ACTION_FEATURE_DIM, OBSERVATION_DIM


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """最小训练模型配置。"""

    hidden_size: int = 256
    observation_size: int = OBSERVATION_DIM
    action_size: int = ACTION_FEATURE_DIM


class CandidateActorCritic(nn.Module):
    """共享 Actor-Critic；每桌每座位由调用方维护独立 GRU 隐状态。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.observation_size, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.memory = nn.GRUCell(hidden, hidden)
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.policy_query = nn.Linear(hidden, hidden)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # NOTE: 初始策略略偏向胡，避免随机初始化在合法胡牌时几乎总是错过。
        self.kind_bias = nn.Parameter(torch.zeros(7))
        with torch.no_grad():
            self.kind_bias[5] = 1.5  # WIN
            self.kind_bias[6] = -0.1  # PASS

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        """返回零初始化的独立记忆。"""
        return torch.zeros(batch_size, self.config.hidden_size, device=device)

    def forward_step(
        self,
        observations: Tensor,
        hidden: Tensor,
        action_features: Tensor,
        action_owners: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """批量处理若干决策和展平候选动作。

        Args:
            observations: `[B, observation_size]`。
            hidden: `[B, hidden_size]`。
            action_features: `[A, action_size]`。
            action_owners: 每个候选动作对应的决策索引，形状 `[A]`。

        Returns:
            展平动作 logits、每个决策的 value、更新后 hidden。
        """
        encoded = self.observation_encoder(observations)
        next_hidden = self.memory(encoded, hidden)
        action_encoded = self.action_encoder(action_features)
        query = self.policy_query(next_hidden).index_select(0, action_owners)
        logits = (query * action_encoded).sum(dim=-1) / math.sqrt(self.config.hidden_size)
        kind_ids = action_features[:, :7].argmax(dim=-1)
        logits = logits + self.kind_bias.index_select(0, kind_ids)
        values = self.value_head(next_hidden).squeeze(-1)
        return logits, values, next_hidden
