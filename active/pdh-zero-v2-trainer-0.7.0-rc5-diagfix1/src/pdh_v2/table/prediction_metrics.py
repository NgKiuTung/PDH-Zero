"""Per-seat calibration against a fitted TRAIN-only empirical-prior baseline."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from pdh_v2.table.model import (KONG, NEXT_DISCARD, NEXT_KIND, PONG, READY, ROUTES,
                                VALUE, WIN, probabilities)


def empirical_prior(target, mask):
    """Relative-seat marginal probabilities; never estimated from the validation split."""
    prior = (target * mask).sum(0) / mask.sum(0).clamp_min(1)
    prior[..., 8:152] = prior[..., 8:152].clamp(1e-6, 1-1e-6)
    prior[..., NEXT_KIND] /= prior[..., NEXT_KIND].sum(-1, keepdim=True)
    prior[..., NEXT_DISCARD] /= prior[..., NEXT_DISCARD].sum(-1, keepdim=True)
    return prior


def masked_mean(value, mask):
    """All-censored terms have zero loss and no fictitious samples."""
    return (value * mask).sum() / mask.sum().clamp_min(1)


def auxiliary_loss(raw, target, mask):
    """Independent auxiliary objective, detached from Actor and real-score PPO."""
    result = masked_mean(F.smooth_l1_loss(raw[..., ROUTES].tanh(), target[..., ROUTES],
                                        reduction='none'), mask[..., ROUTES])
    for part in (READY, WIN, PONG, KONG):
        result = result + masked_mean(F.binary_cross_entropy_with_logits(
            raw[..., part], target[..., part], reduction='none'), mask[..., part])
    for part in (NEXT_KIND, NEXT_DISCARD):
        ce = -(target[..., part] * raw[..., part].log_softmax(-1)).sum(-1)
        result = result + masked_mean(ce, mask[..., part].amax(-1))
    value = F.softplus(raw[..., VALUE])
    return result + masked_mean(F.smooth_l1_loss(value, target[..., VALUE],
                                                reduction='none'), mask[..., VALUE])


def _binary_metrics(prediction, target, mask):
    valid = mask.bool()
    p, y = prediction[valid].clamp(1e-7, 1-1e-7), target[valid]
    if not p.numel():
        return dict(count=0, positives=0, brier=None, log_loss=None, ece=None)
    ece = 0.0
    for index in range(10):
        members = (p >= index/10) & (p < (index+1)/10)
        if members.any():
            ece += float(members.float().mean() * (p[members].mean()-y[members].mean()).abs())
    return dict(count=p.numel(), positives=int(y.sum()), brier=float((p-y).square().mean()),
                log_loss=float(F.binary_cross_entropy(p, y)), ece=ece,
                prevalence=float(y.mean()), predicted_mean=float(p.mean()))


def _categorical_metrics(prediction, target, mask):
    valid = mask.amax(-1).bool()
    p, y = prediction[valid], target[valid]
    if not p.shape[0]:
        return dict(count=0, log_loss=None, accuracy=None)
    return dict(count=p.shape[0], log_loss=float(-(y*p.clamp_min(1e-7).log()).sum(-1).mean()),
                accuracy=float((p.argmax(-1) == y.argmax(-1)).float().mean()))


def seat_metrics(prediction, target, mask):
    """Metric definitions separate structural progress, capabilities, and observed actions."""
    return dict(route_mae=float(masked_mean((prediction[..., ROUTES]-target[..., ROUTES]).abs(),
                                            mask[..., ROUTES])),
                nominal_value_mae=float(masked_mean(
                    (prediction[..., VALUE]-target[..., VALUE]).abs(), mask[..., VALUE])),
                nominal_value_count=int(mask[..., VALUE].sum()),
                **{name: _binary_metrics(prediction[..., part], target[..., part], mask[..., part])
                   for name, part in (('ready', READY), ('win_on_discard', WIN),
                                      ('pong_capability', PONG), ('kong_capability', KONG))},
                **{name: _categorical_metrics(prediction[..., part], target[..., part],
                                              mask[..., part])
                   for name, part in (('next_action', NEXT_KIND), ('next_discard', NEXT_DISCARD))})


def evaluation_report(raw, target, mask, closed, prior):
    """Always report each relative opponent, including all closed-hand-only subsets."""
    predicted = probabilities(raw)
    baseline = prior[None].expand_as(predicted)
    return {str(seat + 1): {
        'model': seat_metrics(predicted[:, seat], target[:, seat], mask[:, seat]),
        'train_prior': seat_metrics(baseline[:, seat], target[:, seat], mask[:, seat]),
        'closed_model': seat_metrics(predicted[:, seat], target[:, seat],
                                      mask[:, seat] * closed[:, seat, None]),
        'closed_train_prior': seat_metrics(baseline[:, seat], target[:, seat],
                                            mask[:, seat] * closed[:, seat, None])}
        for seat in range(3)}