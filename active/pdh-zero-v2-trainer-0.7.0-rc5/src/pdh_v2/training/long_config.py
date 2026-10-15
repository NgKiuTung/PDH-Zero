"""长训练任务配置与恢复身份。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LongRunConfig:
    """首轮完整房规基线的预算与容量配置。"""

    output: Path
    smoke_dir: Path
    max_wall_seconds: int = 48 * 60 * 60
    target_decisions: int = 50_000_000
    environments: int = 96
    hands_per_update: int = 192
    hidden_size: int = 256
    learning_rate: float = 3e-4
    reward_scale: float = 32.0
    ppo_epochs: int = 3
    minibatch_decisions: int = 8192
    seed: int = 20261001
    device: str = "cuda"
    allow_shared_gpu: bool = False
    evaluation_hands: int = 2048
    evaluation_environments: int = 64
    evaluation_interval_decisions: int = 5_000_000
    snapshot_interval_decisions: int = 5_000_000
    min_free_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        positive = {
            "max_wall_seconds": self.max_wall_seconds,
            "target_decisions": self.target_decisions,
            "environments": self.environments,
            "hands_per_update": self.hands_per_update,
            "hidden_size": self.hidden_size,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_decisions": self.minibatch_decisions,
            "evaluation_hands": self.evaluation_hands,
            "evaluation_environments": self.evaluation_environments,
            "evaluation_interval_decisions": self.evaluation_interval_decisions,
            "snapshot_interval_decisions": self.snapshot_interval_decisions,
            "min_free_bytes": self.min_free_bytes,
        }
        invalid = [name for name, value in positive.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"long run values must be positive: {invalid}")
        if self.learning_rate <= 0 or self.reward_scale <= 0:
            raise ValueError("learning_rate and reward_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        """返回可持久化配置。"""
        return {
            **asdict(self),
            "output": str(self.output),
            "smoke_dir": str(self.smoke_dir),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongRunConfig":
        """从任务目录中的冻结配置恢复。"""
        values = dict(payload)
        values["output"] = Path(values["output"])
        values["smoke_dir"] = Path(values["smoke_dir"])
        return cls(**values)

    def checkpoint_identity(self) -> dict[str, Any]:
        """返回不得在恢复时改变的训练契约字段。"""
        values = self.to_dict()
        values.pop("output")
        return values
