"""Predeclared exploratory usefulness gate; no unvalidated rare-risk head enters policy."""
from __future__ import annotations

import torch

from pdh_v2.table.model import (BELIEF_DIM, KONG, NEXT_DISCARD, NEXT_KIND, PONG,
                                READY, ROUTES, VALUE, WIN)


def qualify_heads(metrics: dict) -> tuple[torch.Tensor, dict]:
    """Compare each task/seat to TRAIN-only priors on withheld training-session groups.

    This finite development criterion is not a significance claim. It prevents
    treating a lower fitting loss or a rare-event constant predictor as useful.
    """
    qualified = torch.zeros(3, BELIEF_DIM)
    report = {}
    for seat in range(3):
        candidate = metrics[str(seat+1)]['model']
        prior = metrics[str(seat+1)]['train_prior']
        decisions = {'routes': candidate['route_mae'] < .95*prior['route_mae']}
        qualified[seat, ROUTES] = float(decisions['routes'])
        for name, part in (('ready', READY), ('win_on_discard', WIN),
                            ('pong_capability', PONG), ('kong_capability', KONG)):
            current, baseline = candidate[name], prior[name]
            accepted = (current['count'] >= 100 and current['positives'] >= 20
                and current['count'] - current['positives'] >= 20
                and current['brier'] < baseline['brier']
                and current['log_loss'] < baseline['log_loss'] and current['ece'] <= .05)
            decisions[name] = accepted
            qualified[seat, part] = float(accepted)
        for name, part in (('next_action', NEXT_KIND), ('next_discard', NEXT_DISCARD)):
            current, baseline = candidate[name], prior[name]
            accepted = (current['count'] >= 100
                        and current['log_loss'] < .98*baseline['log_loss'])
            decisions[name] = accepted
            qualified[seat, part] = float(accepted)
        value = (candidate['nominal_value_count'] >= 20
                 and candidate['nominal_value_mae'] < .95*prior['nominal_value_mae'])
        decisions['nominal_value'] = value
        qualified[seat, VALUE] = float(value)
        report[str(seat+1)] = decisions
    return qualified, dict(scope='exploratory_auxiliary_development_gate_not_confirmatory',
                            tasks=report, active_channels_per_seat=qualified.sum(-1).tolist(),
                            unqualified_action='fallback_to_frozen_train_prior')