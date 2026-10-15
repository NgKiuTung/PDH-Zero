"""完整历史重算与合法候选分布；冻结参考模型也使用自己的GRU历史。"""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch.distributions import Categorical

from pdh_v2.policy import CandidateActorCritic
from pdh_v2.training.buffer import Episode


@dataclass
class SequenceValues:
    """同一时间优先顺序的张量；有效掩码排除候选padding。"""
    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    risk: Tensor
    risk_target: Tensor
    old_log_prob: Tensor
    old_value: Tensor
    advantage: Tensor
    returns: Tensor
    log_distribution: Tensor
    valid: Tensor
    chosen: Tensor
    choices: Tensor


def sequence_groups(episodes: tuple[Episode, ...], budget: int, seed: int) -> list[list[Episode]]:
    """整段序列装箱，不把GRU历史截断后冒充零初始。"""
    if not episodes or budget < 1:
        raise ValueError("non-empty episodes and positive budget required")
    items = list(episodes)
    random.Random(seed).shuffle(items)
    groups: list[list[Episode]] = []
    batch: list[Episode] = []
    count = 0
    for episode in items:
        if batch and count + episode.decisions > budget:
            groups.append(batch)
            batch, count = [], 0
        batch.append(episode)
        count += episode.decisions
    if batch:
        groups.append(batch)
    return groups


def sequence_values(model: CandidateActorCritic, episodes: list[Episode],
                    device: torch.device) -> SequenceValues:
    """所有模型按相同完整序列重算；不得沿用被其他优化器修改前的隐状态。"""
    if not episodes or any(not e.transitions for e in episodes):
        raise ValueError("empty sequence")
    width = max(t.action_features.shape[0] for e in episodes for t in e.transitions)
    hidden = [model.initial_state(1, device)[0] for _ in episodes]
    columns: list[list[Tensor]] = [[] for _ in range(13)]
    for step in range(max(e.decisions for e in episodes)):
        active = [i for i, e in enumerate(episodes) if step < e.decisions]
        items = [episodes[i].transitions[step] for i in active]
        counts = [len(t.action_features) for t in items]
        features = torch.cat([t.action_features for t in items]).to(device)
        owners = torch.repeat_interleave(torch.arange(len(active), device=device),
                                        torch.tensor(counts, device=device))
        logits, risks, values, next_hidden = model.forward_step(
            torch.stack([t.observation for t in items]).to(device),
            torch.stack([hidden[i] for i in active]), features, owners)
        # NOTE: padding仅供批量分布；KL中必须同时屏蔽，防止0*(-inf)产生NaN。
        padded = logits.new_full((len(items), width), -torch.inf)
        mask = torch.zeros_like(padded, dtype=torch.bool)
        cursor = 0
        risk_selected = []
        for row, (count, item, i) in enumerate(zip(counts, items, active)):
            padded[row, :count] = logits[cursor:cursor + count]
            mask[row, :count] = True
            risk_selected.append(risks[cursor + item.action_index])
            hidden[i] = next_hidden[row]
            cursor += count
        selected = torch.tensor([t.action_index for t in items], device=device)
        dist = Categorical(logits=padded)
        def _floats(values):
            return torch.tensor(values, dtype=torch.float32, device=device)
        row_values = (
            dist.log_prob(selected), dist.entropy(), values, torch.stack(risk_selected),
            _floats([float('nan') if t.risk_target is None else t.risk_target for t in items]),
            _floats([t.old_log_prob for t in items]), _floats([t.old_value for t in items]),
            _floats([float(episodes[i].advantage[step]) if episodes[i].advantage is not None
                    else 0.0 for i in active]),
            _floats([float(episodes[i].returns[step]) if episodes[i].returns is not None
                    else 0.0 for i in active]),
            dist.logits, mask, selected, torch.tensor(counts, device=device),
        )
        for column, value in zip(columns, row_values):
            column.append(value)
    return SequenceValues(*(torch.cat(column) for column in columns))


def categorical_kl(reference: Tensor, candidate: Tensor, valid: Tensor) -> Tensor:
    """逐决策KL(reference||candidate)，只累计合法动作。"""
    left = torch.where(valid, reference, torch.zeros_like(reference))
    right = torch.where(valid, candidate, torch.zeros_like(candidate))
    probabilities = torch.where(valid, left.exp(), torch.zeros_like(left))
    return (probabilities * (left - right)).sum(-1).clamp_min(0.0)


def choice_mean(values: Tensor, choices: Tensor) -> Tensor:
    """只在真正有选择的决策上评估策略漂移，避免大量强制PASS稀释KL。"""
    mask = choices > 1
    return values[mask].mean() if bool(mask.any()) else values.sum() * 0.0
