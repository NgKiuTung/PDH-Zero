"""Frozen finite experiment budgets, data roles and protected-source receipts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

from pdh_v2.table.opponents import OPPONENTS
from pdh_v2.training.candidate_store import file_digest


@dataclass(frozen=True)
class StudyConfig:
    """Equal natural hands/updates across arms; time caps are safety limits, not early selection."""
    seeds: tuple[int, ...]
    updates: int = 1
    hands_per_update: int = 8
    evaluation_hands: int = 8
    workers: int = 0
    evaluation_environments: int = 8
    fit_steps: int = 400
    fit_batch: int = 64
    learning_rate: float = .00005
    minibatch_decisions: int = 2048
    max_training_seconds: int = 600
    max_evaluation_seconds: int = 600
    device: str = 'cpu'
    backend: str = 'cpp'
    transport: str = 'pipe'
    opponents: tuple[str, ...] = OPPONENTS
    smoke_only: bool = True

    def __post_init__(self):
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('rc5.independent_seeds_required')
        if any(type(s) is not int or not 0 <= s < 2**55 for s in self.seeds):
            raise ValueError('rc5.invalid_seed')
        if min(self.updates, self.hands_per_update, self.evaluation_environments,
               self.fit_steps, self.fit_batch, self.minibatch_decisions,
               self.max_training_seconds, self.max_evaluation_seconds) < 1:
            raise ValueError('rc5.invalid_budget')
        if self.evaluation_hands < 4 or self.evaluation_hands % 4:
            raise ValueError('rc5.incomplete_wall_groups')
        if not 0 <= self.workers <= min(self.hands_per_update, self.evaluation_environments):
            raise ValueError('rc5.invalid_workers')
        if (self.device not in {'cpu', 'cuda'} or self.backend not in {'python', 'cpp'}
                or self.transport not in {'pipe', 'shared'}
                or set(self.opponents) != set(OPPONENTS) or len(self.opponents) != 3):
            raise ValueError('rc5.invalid_experiment_condition')
        if not 0 < self.learning_rate <= .0001:
            raise ValueError('rc5.invalid_learning_rate')
        if not self.smoke_only and (len(self.seeds) < 3 or self.evaluation_hands < 4096):
            raise ValueError('rc5.confirmatory_scale_not_met')

    def to_dict(self):
        """JSON serializable, complete effective configuration."""
        return asdict(self)


def derived_seed(seed: int, domain: str) -> int:
    """Separate train/development/auxiliary streams; no ad hoc date-based random defaults."""
    return int.from_bytes(hashlib.sha256(f'rc5:{seed}:{domain}'.encode()).digest()[:6], 'big')


def source_identity(root: Path) -> dict[str, str]:
    """Protect every source module and resource, along with the frozen rule text."""
    suffixes = {'.py', '.json', '.cpp', '.h', '.hpp', '.cc'}
    code = {p for folder in ('src', 'native', 'tools')
            for p in (root/folder).rglob('*') if p.suffix in suffixes}
    paths = sorted(code | set((root/'rules').rglob('*.*')))
    return {str(path.relative_to(root)): file_digest(path) for path in paths if path.is_file()}


def check_identity(root: Path, expected: dict) -> None:
    """Any in-scope source/evaluator addition, removal or mutation invalidates the running trial."""
    if source_identity(root) != expected:
        raise RuntimeError('rc5.protected_source_changed')


class BudgetClock:
    """A hard deadline never turns an unfinished hand into a PPO sample or a success."""
    def __init__(self, seconds: int):
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self, *args):
        """Raise before unfinished work can be accepted as a completed fixed-budget trial."""
        if time.monotonic()-self.started > self.seconds:
            raise TimeoutError('rc5.fixed_budget_exhausted')


def read_config(path: Path) -> StudyConfig:
    """No unknown config keys or silent defaults from an old runtime plan."""
    data = json.loads(path.read_text())
    data['seeds'] = tuple(data['seeds'])
    if 'opponents' in data:
        data['opponents'] = tuple(data['opponents'])
    return StudyConfig(**data)