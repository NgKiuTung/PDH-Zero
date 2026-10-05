"""候选动作式循环 Actor-Critic。

模型只对规则层提供的合法动作评分。策略头显式组合状态与动作表示，并使用仅基于
合法观察训练出的弃牌风险辅助头，避免将对手暗牌或未来牌墙泄漏给 Actor。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pdh_v2.game.actions import ActionKind
from pdh_v2.observation import ACTION_FEATURE_DIM, OBSERVATION_DIM
from pdh_v2.observation.encoder import ACTION_KIND_INDEX


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """训练模型配置；结构变化会由检查点身份校验阻止误加载。"""

    hidden_size: int = 384
    observation_size: int = OBSERVATION_DIM
    action_size: int = ACTION_FEATURE_DIM
    risk_policy_weight: float = 0.5


class CandidateActorCritic(nn.Module):
    """共享 Actor-Critic；每桌每座位由调用方维护独立 GRU 隐状态。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.observation_size, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.memory = nn.GRUCell(hidden, hidden)
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_size, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        pair_size = hidden * 3
        self.policy_pair = nn.Sequential(
            nn.Linear(pair_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.risk_pair = nn.Sequential(
            nn.Linear(pair_size, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.kind_bias = nn.Parameter(torch.zeros(len(ActionKind)))
        with torch.no_grad():
            # 初始策略优先执行已合法的胡牌，避免冷启动期间错过稀有终局。
            self.kind_bias[ACTION_KIND_INDEX[ActionKind.WIN]] = 2.0
            self.kind_bias[ACTION_KIND_INDEX[ActionKind.PASS]] = -0.15
            # 风险头零初始化，使初始策略不受未训练风险预测扰动。
            final = self.risk_pair[-1]
            if not isinstance(final, nn.Linear):
                raise RuntimeError("risk head final layer must be linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        """返回零初始化的独立记忆。"""
        return torch.zeros(batch_size, self.config.hidden_size, device=device)

    def forward_step(
        self,
        observations: Tensor,
        hidden: Tensor,
        action_features: Tensor,
        action_owners: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """批量处理若干决策和展平候选动作。

        Returns:
            展平策略 logits、每个候选动作的风险 logits、每个决策的 value、
            更新后 hidden。
        """
        encoded = self.observation_encoder(observations)
        next_hidden = self.memory(encoded, hidden)
        action_encoded = self.action_encoder(action_features)
        owned_state = next_hidden.index_select(0, action_owners)
        pair = torch.cat(
            (owned_state, action_encoded, owned_state * action_encoded),
            dim=-1,
        )
        policy_logits = self.policy_pair(pair).squeeze(-1)
        risk_logits = self.risk_pair(pair).squeeze(-1)
        kind_ids = action_features[:, : len(ActionKind)].argmax(dim=-1)
        policy_logits = policy_logits + self.kind_bias.index_select(0, kind_ids)

        discard_id = ACTION_KIND_INDEX[ActionKind.DISCARD]
        discard_mask = (kind_ids == discard_id).to(policy_logits.dtype)
        centered_risk = torch.sigmoid(risk_logits.detach()) - 0.5
        policy_logits = (
            policy_logits
            - discard_mask * centered_risk * self.config.risk_policy_weight
        )
        values = self.value_head(next_hidden).squeeze(-1)
        return policy_logits, risk_logits, values, next_hidden
