"""固定教师KL锚与更新后全批KL保护；不在PPO批次外执行突发BC更新。"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from pdh_v2.policy import CandidateActorCritic
from pdh_v2.training.buffer import Episode
from pdh_v2.training.sequence_batch import (
    categorical_kl, choice_mean, sequence_groups, sequence_values,
)


@dataclass(frozen=True)
class StablePPOConfig:
    """工程保护候选参数，效果需要固定评估；不是已经验证的最优超参。"""
    epochs: int = 2
    minibatch_decisions: int = 8192
    clip_ratio: float = 0.1
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.003
    risk_coefficient: float = 0.05
    anchor_coefficient: float = 0.05
    target_kl: float = 0.01
    max_gradient_norm: float = 0.5
    behavior_tolerance: float = 0.0005

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.minibatch_decisions < 1:
            raise ValueError("positive PPO sizes required")
        if not 0 < self.clip_ratio < 1 or self.target_kl <= 0:
            raise ValueError("invalid policy trust limits")


def _snapshot(model, optimizer):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def _restore(model, optimizer, snapshot) -> None:
    model.load_state_dict(snapshot[0])
    optimizer.load_state_dict(snapshot[1])


def guarded_update(
    model: CandidateActorCritic, optimizer: torch.optim.Optimizer,
    anchor: CandidateActorCritic, episodes: tuple[Episode, ...],
    device: torch.device, config: StablePPOConfig, seed: int, check_stop=None,
) -> dict[str, int | float | str | bool]:
    """固定行为策略与独立冻结参考历史；超限恢复整个更新和优化器。"""
    if model.config.risk_policy_weight != 0:
        raise ValueError("refinement requires diagnostic-only risk head")
    if model is anchor or any(p.requires_grad for p in anchor.parameters()):
        raise ValueError("anchor must be a separate frozen model")
    snapshot = _snapshot(model, optimizer)
    groups = sequence_groups(episodes, config.minibatch_decisions, seed)
    references = []
    originals = []
    choice_count = 0
    behavior_error = 0.0
    model.eval()
    with torch.no_grad():
        for group in groups:
            if check_stop is not None:
                check_stop()
            old = sequence_values(model, group, device)
            fixed = sequence_values(anchor, group, device)
            behavior_error = max(behavior_error, float(
                (old.log_prob - old.old_log_prob).abs().max()))
            originals.append(old.log_distribution.detach())
            references.append(fixed.log_distribution.detach())
            choice_count += int((old.choices > 1).sum())
    if behavior_error > config.behavior_tolerance:
        raise RuntimeError(f"on-policy replay mismatch: {behavior_error:.6g}")
    totals = dict(policy_loss=0.0, value_loss=0.0, risk_loss=0.0, entropy=0.0,
                  anchor_kl=0.0, gradient_norm=0.0)
    batches = 0
    early_stop = False
    model.train()
    try:
        for _ in range(config.epochs):
            for group, old_dist, ref_dist in zip(groups, originals, references):
                if check_stop is not None:
                    check_stop()
                current = sequence_values(model, group, device)
                kl = choice_mean(categorical_kl(old_dist, current.log_distribution,
                                                current.valid), current.choices)
                if not torch.isfinite(kl):
                    raise FloatingPointError("non-finite policy KL")
                if float(kl.detach()) > config.target_kl:
                    early_stop = True
                    break
                ratio = (current.log_prob - current.old_log_prob).exp()
                surrogate = torch.minimum(ratio * current.advantage,
                    ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio) * current.advantage)
                policy_loss = -choice_mean(surrogate, current.choices)
                value_loss = 0.5 * (current.value - current.returns).square().mean()
                mask = torch.isfinite(current.risk_target)
                # NOTE: 未加类别权重，输出可解释为当前选中弃牌的经验风险；暂不接入策略。
                risk_loss = (F.binary_cross_entropy_with_logits(current.risk[mask],
                             current.risk_target[mask]) if bool(mask.any())
                             else current.risk.sum() * 0.0)
                entropy = choice_mean(current.entropy, current.choices)
                anchor_kl = choice_mean(categorical_kl(ref_dist, current.log_distribution,
                                                       current.valid), current.choices)
                loss = (policy_loss + config.value_coefficient*value_loss
                        - config.entropy_coefficient*entropy
                        + config.anchor_coefficient*anchor_kl
                        + config.risk_coefficient*risk_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),
                          config.max_gradient_norm, error_if_nonfinite=True)
                optimizer.step()
                values = dict(policy_loss=policy_loss, value_loss=value_loss,
                              risk_loss=risk_loss, entropy=entropy, anchor_kl=anchor_kl,
                              gradient_norm=grad)
                for key, value in values.items():
                    totals[key] += float(value.detach())
                batches += 1
            if early_stop:
                break
        # NOTE: 均值不能只看最后一个minibatch；重新检查完整采样批次，且只统计真实选择。
        total_kl = 0.0
        with torch.no_grad():
            for group, old_dist in zip(groups, originals):
                if check_stop is not None:
                    check_stop()
                current = sequence_values(model, group, device)
                kl_rows = categorical_kl(old_dist, current.log_distribution, current.valid)
                total_kl += float(kl_rows[current.choices > 1].sum())
        final_kl = total_kl / max(1, choice_count)
        if not torch.isfinite(torch.tensor(final_kl)):
            raise FloatingPointError("non-finite post-update KL")
        accepted = final_kl <= config.target_kl
        if not accepted:
            _restore(model, optimizer, snapshot)
        return {**{key: value/max(1,batches) for key,value in totals.items()},
                "post_update_kl": final_kl, "kl_target": config.target_kl,
                "update_accepted": accepted, "early_stop": early_stop,
                "optimizer_steps_attempted": batches,
                "behavior_replay_max_abs_error": behavior_error,
                "unique_actor_decisions": sum(e.decisions for e in episodes),
                "decisions_with_choice": choice_count}
    except BaseException:
        _restore(model, optimizer, snapshot)
        raise
