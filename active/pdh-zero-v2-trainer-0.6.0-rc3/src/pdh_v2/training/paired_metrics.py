"""同牌墙四座位分组的开发集配对比较；不把四座位当独立样本。"""
from __future__ import annotations

import math
import numpy as np

from pdh_v2.training.loss_attribution import summarize_losses


def summarize(rows: list[dict]) -> dict:
    """所有实际评估局都入分母；胡牌按身份，不用score>0猜。"""
    if not rows:
        raise ValueError("no evaluation rows")
    values = np.array([r['score'] for r in rows], dtype=float)
    return {"hands":len(rows), "mean_score":float(values.mean()),
            "score_std":float(values.std()),
            "median_score":float(np.median(values)),
            "quantiles": {str(q): float(np.quantile(values, q))
                          for q in (.01, .05, .1, .5, .9, .95, .99)},
            "cvar_1":float(np.sort(values)[:max(1, math.ceil(len(rows)*.01))].mean()),
            "cvar_5":float(np.sort(values)[:max(1, math.ceil(len(rows)*.05))].mean()),
            **{k:float(np.mean([r[k] for r in rows])) for k in
               ('win', 'point_shoot', 'draw', 'self_draw', 'discard_win')},
            "minimum_score":int(values.min()), "maximum_score":int(values.max()),
            **summarize_losses(rows)}


def _groups(rows: list[dict]) -> dict[int, dict[int, dict]]:
    groups: dict[int, dict[int, dict]] = {}
    for row in rows:
        seed, seat = row['wall_seed'], row['candidate_seat']
        if type(seed) is not int or type(seat) is not int or seat not in range(4):
            raise ValueError("invalid evaluation key")
        group = groups.setdefault(seed, {})
        if seat in group:
            raise ValueError("duplicate paired evaluation key")
        if not np.isfinite(row['score']):
            raise ValueError("non-finite score")
        group[seat] = row
    if any(set(g) != set(range(4)) for g in groups.values()):
        raise ValueError("incomplete four-seat wall group")
    return groups


def paired_comparison(reference: list[dict], candidate: list[dict],
                      bootstrap_samples: int = 2000, seed: int = 7107) -> dict:
    """重采样牌墙组的开发集区间；反复看同一开发集不构成最终盲测证据。"""
    a, b = _groups(reference), _groups(candidate)
    if not a or a.keys() != b.keys() or bootstrap_samples < 1:
        raise ValueError("paired evaluation key sets differ")
    keys = sorted(a)
    metrics = ('score', 'point_shoot', 'win', 'draw')
    differences = np.array([[np.mean([float(b[k][s][m])-float(a[k][s][m])
                                     for s in range(4)]) for m in metrics] for k in keys])
    rng = np.random.default_rng(seed)
    base_scores = np.array([[a[k][s]['score'] for s in range(4)] for k in keys], dtype=float)
    new_scores = np.array([[b[k][s]['score'] for s in range(4)] for k in keys], dtype=float)
    tail_size = max(1, math.ceil(len(reference) * .05))
    # NOTE: 本地分批采样，避免扩到大评估集时一次分配巨量bootstrap矩阵。
    samples = []
    tail_samples = []
    for start in range(0, bootstrap_samples, 100):
        n = min(100, bootstrap_samples-start)
        indices = rng.integers(0, len(keys), size=(n, len(keys)))
        samples.append(differences[indices].mean(axis=1))
        # NOTE: CVaR是非线性统计量，重采样整墙后重新求尾部，不能平均逐墙CVaR。
        left = np.sort(base_scores[indices].reshape(n, -1), axis=1)[:, :tail_size].mean(1)
        right = np.sort(new_scores[indices].reshape(n, -1), axis=1)[:, :tail_size].mean(1)
        tail_samples.append(right - left)
    quantiles = np.quantile(np.concatenate(samples), [.025, .975], axis=0)
    result = {m: {"delta":float(differences[:,i].mean()),
                  "ci95":[float(quantiles[0,i]),float(quantiles[1,i])]}
              for i,m in enumerate(metrics)}
    result['cvar_5'] = {
        'delta': float(np.sort(new_scores.ravel())[:tail_size].mean()
                       - np.sort(base_scores.ravel())[:tail_size].mean()),
        'ci95': [float(v) for v in np.quantile(np.concatenate(tail_samples), [.025, .975])],
    }
    return {"wall_groups":len(keys), "hands":len(reference),
            "scope":"repeated_development_set_not_confirmatory", **result}


def development_gate(comparison: dict, minimum_delta: float = .1,
                     max_cvar_drop: float = 2.0) -> str:
    """三态门禁：证据不足为INCONCLUSIVE，不把微小均值变化说成显著。"""
    gain = comparison['score']; risk = comparison['point_shoot']
    tail = comparison.get('cvar_5')
    if (gain['ci95'][1] < 0 or risk['ci95'][0] > .01
            or (tail is not None and tail['ci95'][1] < -max_cvar_drop)):
        return 'REGRESSION'
    if (gain['delta'] >= minimum_delta and gain['ci95'][0] > 0 and risk['ci95'][1] <= .01
            and (tail is None or tail['ci95'][0] >= -max_cvar_drop)):
        return 'PASS'
    return 'INCONCLUSIVE'
