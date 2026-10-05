"""单写入者共享特征缓冲；只放合法策略投影，牌墙/他家暗牌不进入缓冲。"""
from __future__ import annotations

import numpy as np

from pdh_v2.observation import ACTION_FEATURE_DIM, OBSERVATION_DIM, DecisionView
from pdh_v2.rollout.vector_env import EnvDecision


class SharedViews:
    """每Worker独享RawArray；同步响应确认完成写入，主进程复制后才允许下一次写。"""

    def __init__(self, context, count: int, capacity: int = 64) -> None:
        if count < 1 or capacity < 1:
            raise ValueError('shared buffer dimensions must be positive')
        self.count = count
        self.capacity = capacity
        self.observations = context.RawArray('f', count * OBSERVATION_DIM)
        self.features = context.RawArray('f', count * capacity * ACTION_FEATURE_DIM)

    def _arrays(self):
        observations = np.frombuffer(self.observations, dtype=np.float32)
        features = np.frombuffer(self.features, dtype=np.float32)
        return (observations.reshape(self.count, OBSERVATION_DIM),
                features.reshape(self.count, self.capacity, ACTION_FEATURE_DIM))

    def write(self, requests: tuple[EnvDecision, ...]) -> tuple:
        """Worker写浮点数到共享区域；Pipe只传身份及小型合法动作元数据。"""
        observations, features = self._arrays()
        result = []
        for request in requests:
            index, view = request.env_index, request.view
            n = len(view.actions)
            if not 0 <= index < self.count or not 1 <= n <= self.capacity:
                raise ValueError('shared action capacity/index exceeded; no truncation allowed')
            if len(view.observation) != OBSERVATION_DIM or len(view.action_features) != n:
                raise ValueError('shared observation dimensions mismatch')
            observations[index] = view.observation
            features[index, :n] = view.action_features
            result.append((index, request.hand_id, view.player, view.actions))
        return tuple(result)

    def read(self, records: tuple) -> tuple[EnvDecision, ...]:
        """主进程复制当前批次，避免下一步覆写污染已保存的GRU轨迹。"""
        observations, features = self._arrays()
        result = []
        for index, hand_id, player, actions in records:
            if not 0 <= index < self.count or not 1 <= len(actions) <= self.capacity:
                raise ValueError('invalid shared view metadata')
            obs = observations[index].copy()
            acts = features[index, :len(actions)].copy()
            if not np.isfinite(obs).all() or not np.isfinite(acts).all():
                raise ValueError('non-finite shared features')
            result.append(EnvDecision(index, hand_id, DecisionView(player, obs, actions, acts)))
        return tuple(result)