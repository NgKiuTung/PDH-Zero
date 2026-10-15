"""受控调优配置、时钟及进度；手工停止与程序故障分别记录。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

from pdh_v2.training.events import JsonlWriter


class StopRequested(Exception):
    """主动取消或总预算耗尽；不把它写成未知训练故障。"""


@dataclass(frozen=True)
class RefinementConfig:
    """本轮冻结SOLO并控制更新幅度；时间预算包含初始化和所有评估。"""
    output: Path
    warmup_checkpoint: Path | None = None
    workers: int = 4
    environments: int = 64
    hands_per_update: int = 256
    max_seconds: int = 10800
    target_decisions: int = 500_000
    evaluation_hands: int = 4096
    evaluation_interval: int = 125_000
    evaluation_seed: int | None = None
    learning_rate: float = 0.00005
    reward_scale: float = 32.0
    minibatch_decisions: int = 8192
    seed: int | None = None
    seed_origin: str = "auto"
    device: str = 'cuda'
    transport: str = 'shared'
    max_anchor_kl: float = .12
    max_cvar_drop: float = 2.0
    selection_patience: int = 2
    initial_selection: Path | None = None
    feature_backend: str = "python"

    def __post_init__(self) -> None:
        from pdh_v2.training.random_streams import derive_seed, new_seed
        generated = self.seed is None
        seed = new_seed() if generated else self.seed
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('data_protocol.invalid_seed')
        object.__setattr__(self, 'seed', seed)
        object.__setattr__(self, 'seed_origin', 'generated' if generated else 'explicit')
        evaluation_seed = (derive_seed(seed, 'development') if self.evaluation_seed is None
                           else self.evaluation_seed)
        if type(evaluation_seed) is not int or not 0 <= evaluation_seed < 2**63:
            raise ValueError('data_protocol.invalid_evaluation_seed')
        object.__setattr__(self, 'evaluation_seed', evaluation_seed)
        if (self.warmup_checkpoint is None) == (self.initial_selection is None):
            raise ValueError("requires exactly one warmup or selected candidate")
        if self.feature_backend not in {"python", "cpp"}:
            raise ValueError("unknown feature backend")
        if self.workers < 0 or self.workers > self.environments:
            raise ValueError('workers must be between zero and environments')
        if min(self.environments, self.hands_per_update, self.max_seconds,
               self.target_decisions, self.evaluation_interval, self.minibatch_decisions) < 1:
            raise ValueError('budgets must be positive')
        if self.evaluation_hands < 4 or self.evaluation_hands % 4:
            raise ValueError('evaluation hands must be divisible by four')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('invalid learning rate')
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError('invalid reward scale')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        if self.transport not in ('pipe', 'shared') or self.selection_patience < 1:
            raise ValueError('invalid transport/patience')
        if not math.isfinite(self.max_cvar_drop) or self.max_cvar_drop < 0:
            raise ValueError('invalid tail risk margin')
        if not math.isfinite(self.max_anchor_kl) or self.max_anchor_kl <= 0:
            raise ValueError('invalid anchor limit')

    def to_dict(self) -> dict:
        """持久化实际生效配置；不依赖交互终端环境变量。"""
        return {**asdict(self), 'output':str(self.output),
                 'warmup_checkpoint': (str(self.warmup_checkpoint)
                                       if self.warmup_checkpoint else None),
                'initial_selection': (str(self.initial_selection)
                                      if self.initial_selection else None)}


def utc() -> str:
    """事件使用UTC，终端可以按本地时区显示。"""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    """同目录原子提交；禁止JSON NaN污染验收文件。"""
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                   encoding='utf-8')
    os.replace(tmp, path)


class Progress:
    """有界前台任务进度；信号只置标记，在可清理边界退出。"""

    def __init__(self, output: Path, maximum_seconds: int) -> None:
        self.output = output
        self.started = time.monotonic()
        self.deadline = self.started + maximum_seconds
        self.requested: str | None = None
        self.events = JsonlWriter(output/'events.jsonl')
        self.phase = 'initializing'
        self._last = 0.0
        self._handlers: dict = {}
        self.current: dict = {}

    @property
    def elapsed(self) -> float:
        """包括取样、更新、评估和I/O的总墙钟秒。"""
        return time.monotonic()-self.started

    def install_signals(self) -> None:
        """仅主线程安装；测试可不安装。"""
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)

    def restore_signals(self) -> None:
        """退出后不污染调用进程的原信号处理。"""
        for sig, handler in self._handlers.items():
            signal.signal(sig, handler)

    def _signal(self, signum, frame) -> None:
        self.requested = f'signal_{signum}'

    def check(self) -> None:
        """最大预算不以额外隐形评估延长；未完成评估保留为partial。"""
        if self.requested:
            raise StopRequested(self.requested)
        if time.monotonic() >= self.deadline:
            raise StopRequested('wall_budget_reached')

    def event(self, name: str, **fields) -> None:
        """只写业务边界；高频进度由tick限流。"""
        self.events.write({'timestamp':utc(), 'event':name, **fields})

    def stage(self, name: str, **fields) -> None:
        """阶段切换立即可见，不让GPU空闲阶段成为无日志黑盒。"""
        self.phase = name
        self.current = dict(fields)
        self.event('phase_started', phase=name, **fields)
        self.tick(0, 0, force=True)

    def tick(self, completed: int, target: int, force: bool = False) -> None:
        """最多每10秒写一次心跳，CPU阶段也有可观察进度。"""
        self.check()
        now = time.monotonic()
        if force or now-self._last >= 10:
            self._last = now
            atomic_json(self.output/'status.json', {
                'status':'running', 'phase':self.phase, 'pid':os.getpid(),
                'updated_at_utc':utc(), 'elapsed_wall_seconds':self.elapsed,
                'phase_completed':completed, 'phase_target':target, **self.current})
