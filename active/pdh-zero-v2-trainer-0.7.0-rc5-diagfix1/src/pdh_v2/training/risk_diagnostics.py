"""选中弃牌的风险审计；稀有事件缺样本时返回None，不能伪造完美AUC。"""
from __future__ import annotations

import numpy as np
import torch

from pdh_v2.training.sequence_batch import sequence_groups, sequence_values


def probability_metrics(probabilities: list[float], labels: list[int]) -> dict:
    """返回未加权Brier、log loss、AP和固定阈值召回；相同分数按组累计。"""
    if len(probabilities) != len(labels) or any(y not in (0, 1) for y in labels):
        raise ValueError("invalid probability-label pairs")
    if not labels:
        return {"samples": 0, "positives": 0, "average_precision": None}
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probability")
    clipped = np.clip(p, 1e-9, 1-1e-9)
    positives = int(y.sum())
    order = np.argsort(-p, kind='stable')
    sy, sp = y[order], p[order]
    ends = np.r_[np.flatnonzero(sp[:-1] != sp[1:]), len(sp)-1]
    tp = np.cumsum(sy)[ends]
    precision = tp/(ends+1)
    ap = float(np.sum(np.diff(np.r_[0, tp])*precision)/positives) if positives else None
    positive_predictions = p >= 0.05
    hits = int(y[positive_predictions].sum())
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(float(np.abs(p[bins == i].mean()-y[bins == i].mean())
                    * np.mean(bins == i)) for i in np.unique(bins))
    return {"samples":len(y), "positives":positives, "prevalence":float(y.mean()),
            "average_precision":ap, "brier":float(np.mean((p-y)**2)),
            "log_loss":float(np.mean(-y*np.log(clipped)-(1-y)*np.log(1-clipped))),
            "ece_10_equal_width":ece,
            "mean_predicted_risk":float(p.mean()),
            "threshold":0.05,
            "recall_at_threshold":hits/positives if positives else None,
            "precision_at_threshold":hits/int(positive_predictions.sum())
            if positive_predictions.any() else None}


@torch.no_grad()
def audit_risk(model, episodes, device, batch_decisions: int) -> dict:
    """每个已选弃牌只统计一次；这是本批诊断，不当作独立验证集结论。"""
    probabilities: list[float] = []
    labels: list[int] = []
    for group in sequence_groups(episodes, batch_decisions, 0):
        data = sequence_values(model, group, device)
        mask = torch.isfinite(data.risk_target)
        probabilities.extend(data.risk[mask].sigmoid().cpu().tolist())
        labels.extend(data.risk_target[mask].int().cpu().tolist())
    return {"scope":"same_rollout_diagnostic_not_holdout",
            **probability_metrics(probabilities, labels)}
