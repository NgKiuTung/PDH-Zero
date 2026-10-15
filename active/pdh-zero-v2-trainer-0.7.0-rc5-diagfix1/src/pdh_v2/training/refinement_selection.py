"""开发集候选选择和有限预算早停；不把重复开发集筛选包装成独立验证。"""
from __future__ import annotations

from dataclasses import replace
import math

from pdh_v2.training.candidate_store import CandidateStore
from pdh_v2.training.paired_metrics import development_gate, paired_comparison
from pdh_v2.training.refinement_control import atomic_json


def adapt_update_config(original, report: dict, initial_lr: float):
    """按实测累计偏移平滑增加锚强度；范围是工程候选，不是通用最优阈值。"""
    drift = report['effective_anchor_kl']
    scale = min(8.0, max(1.0, drift / .025))
    penalty = min(.8, .05 * scale)
    rate = max(initial_lr / 4, initial_lr / math.sqrt(scale))
    return replace(original, anchor_coefficient=penalty), rate


class DevelopmentSelection:
    """选择名义收益更好且未超尾部预算的候选；qualified需更严格的区间门禁。"""

    def __init__(self, run, baseline_rows, baseline_summary, max_cvar_drop=2.0):
        self.run = run
        self.baseline_rows = baseline_rows
        self.baseline_summary = baseline_summary
        self.best_summary = baseline_summary
        self.selected = 'baseline'
        self.qualified = None
        self.best_qualified_mean = float('-inf')
        self.stale_evaluations = 0
        self.max_cvar_drop = max_cvar_drop
        self.store = CandidateStore(run)

    def observe(self, label: str, rows: list[dict], summary: dict, clock) -> str:
        """所有候选完成后写独立记录；INCONCLUSIVE不回填PASS，也不会丢权重。"""
        comparison = paired_comparison(self.baseline_rows, rows)
        gate = development_gate(comparison, max_cvar_drop=self.max_cvar_drop)
        self.store.complete(label, summary, comparison, gate)
        safe_point = (summary['point_shoot'] <= self.baseline_summary['point_shoot'] + .01
                      and summary['cvar_5'] >=
                      self.baseline_summary['cvar_5'] - self.max_cvar_drop)
        improved = (safe_point and gate != 'REGRESSION' and
                    summary['mean_score'] >= self.best_summary['mean_score'] + .1)
        if improved:
            self.selected = label
            self.best_summary = summary
            self.store.select(label)
        self.stale_evaluations = 0 if improved else self.stale_evaluations + 1
        if gate == 'PASS' and summary['mean_score'] > self.best_qualified_mean:
            self.qualified = label
            self.best_qualified_mean = summary['mean_score']
            self.store.select(label, qualified=True)
        fields = dict(decision=gate, comparison=comparison, candidate=label,
                      selected_development=self.selected, qualified=self.qualified,
                      stale_evaluations=self.stale_evaluations,
                      selection_is_confirmatory=False, long_training_allowed=False)
        atomic_json(self.run / 'gate.json', {'gate': gate, **fields})
        clock.event('development_comparison', **fields)
        return gate