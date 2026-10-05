"""混合对手课程与确定性座位分配。

每个采样批次固定一个课程阶段。当前策略控制的座位才进入 on-policy PPO，规则
启发式座位只提供更密集、可解释的对局响应与终局信号。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

from pdh_v2.config import PLAYERS


@dataclass(frozen=True, slots=True)
class SeatMix:
    """每手由1至4个学习座位控制的概率权重。"""

    weights: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.weights):
            raise ValueError("seat-mix weights must not be negative")
        if sum(self.weights) <= 0:
            raise ValueError("seat-mix must contain a positive weight")

    def learner_count(self, seed: int, env_index: int, hand_index: int) -> int:
        """按稳定随机键选择本手学习座位数量。"""
        rng = random.Random(seed ^ (env_index * 1_000_003) ^ (hand_index * 97_409))
        draw = rng.random() * sum(self.weights)
        cursor = 0.0
        for count, weight in enumerate(self.weights, start=1):
            cursor += weight
            if draw < cursor:
                return count
        return PLAYERS

    def learner_seats(self, seed: int, env_index: int, hand_index: int) -> tuple[int, ...]:
        """均衡选择物理座位组合；同一环境恢复后结果不依赖进程调度。"""
        count = self.learner_count(seed, env_index, hand_index)
        candidates = tuple(combinations(range(PLAYERS), count))
        rng = random.Random(
            (seed ^ 0x5EA7) + env_index * 104_729 + hand_index * 130_363
        )
        return candidates[rng.randrange(len(candidates))]


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """一个固定策略版本批次使用的对手课程。"""

    name: str
    minimum_decisions: int
    seat_mix: SeatMix


DEFAULT_STAGES = (
    CurriculumStage("GUIDED_SOLO", 0, SeatMix((1.0, 0.0, 0.0, 0.0))),
    CurriculumStage("GUIDED_DUO", 250_000, SeatMix((0.55, 0.45, 0.0, 0.0))),
    CurriculumStage("GUIDED_TRIO", 750_000, SeatMix((0.15, 0.45, 0.40, 0.0))),
    CurriculumStage(
        "ANCHORED_SELF_PLAY",
        1_500_000,
        SeatMix((0.10, 0.20, 0.30, 0.40)),
    ),
)


@dataclass(frozen=True, slots=True)
class CurriculumSchedule:
    """按累计有效 Actor 决策选择课程阶段。"""

    stages: tuple[CurriculumStage, ...] = DEFAULT_STAGES

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("curriculum requires at least one stage")
        starts = [stage.minimum_decisions for stage in self.stages]
        if starts[0] != 0 or starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("curriculum stage thresholds must start at zero and increase")

    def stage(self, decisions: int) -> CurriculumStage:
        """返回累计决策对应的最后一个已生效阶段。"""
        if decisions < 0:
            raise ValueError("decisions must not be negative")
        selected = self.stages[0]
        for stage in self.stages:
            if decisions < stage.minimum_decisions:
                break
            selected = stage
        return selected
