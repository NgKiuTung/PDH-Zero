"""Belief gates are task/seat specific and can never promote censored or worse predictors."""
import copy
import unittest

import torch

from pdh_v2.table.model import SeatBelief, NEXT_KIND, NEXT_DISCARD, PONG, ROUTES
from pdh_v2.table.prediction_gate import qualify_heads
from pdh_v2.table.prediction_metrics import auxiliary_loss, empirical_prior, evaluation_report


def gate_fixture():
    binary = dict(count=200, positives=30, brier=.1, log_loss=.2, ece=.01)
    category = dict(count=200, log_loss=.5)
    candidate = dict(route_mae=.1, ready=binary, win_on_discard=binary,
                     pong_capability=binary, kong_capability=binary,
                     next_action=category, next_discard=category,
                     nominal_value_count=30, nominal_value_mae=.1)
    prior = copy.deepcopy(candidate)
    prior.update(route_mae=.2, nominal_value_mae=.2)
    for name in ('ready', 'win_on_discard', 'pong_capability', 'kong_capability'):
        prior[name] = dict(binary, brier=.2, log_loss=.3)
    prior.update(next_action=dict(category, log_loss=.6), next_discard=dict(category, log_loss=.6))
    return {str(s): dict(model=copy.deepcopy(candidate), train_prior=copy.deepcopy(prior))
            for s in (1, 2, 3)}


class Rc5PredictionTests(unittest.TestCase):
    def test_perfectly_qualified_all_seats(self):
        qualified, report = qualify_heads(gate_fixture())
        self.assertEqual(qualified.sum().item(), 459)
        self.assertEqual(report['active_channels_per_seat'], [153., 153., 153.])

    def test_one_seat_does_not_suppress_other_seats(self):
        data = gate_fixture()
        data['2']['model']['route_mae'] = .3
        qualified, _ = qualify_heads(data)
        self.assertEqual(qualified[1, ROUTES].sum().item(), 0)
        self.assertEqual(qualified[0, ROUTES].sum().item(), 8)

    def test_rare_positive_head_is_not_qualified(self):
        data = gate_fixture()
        data['1']['model']['pong_capability'] = dict(data['1']['model']['pong_capability'],
                                                    positives=19)
        qualified, _ = qualify_heads(data)
        self.assertEqual(qualified[0, PONG].sum().item(), 0)

    def test_failed_heads_return_training_prior_exactly(self):
        model = SeatBelief(384)
        model.prior.fill_(.123)
        result = model.forecast(torch.randn(2, 622), torch.randn(2, 384))
        torch.testing.assert_close(result, model.prior.expand_as(result), atol=0, rtol=0)

    def test_censored_loss_has_zero_gradient(self):
        raw = torch.zeros(2, 3, 153, requires_grad=True)
        loss = auxiliary_loss(raw, torch.zeros_like(raw), torch.zeros_like(raw))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(raw.grad.abs().sum().item(), 0)

    def test_censored_metrics_do_not_invent_events(self):
        raw = torch.zeros(2, 3, 153)
        prior = empirical_prior(raw, torch.zeros_like(raw))
        report = evaluation_report(raw, raw, torch.zeros_like(raw),
                                   torch.ones(2, 3, dtype=torch.bool), prior)
        self.assertEqual(report['1']['model']['ready']['count'], 0)
        self.assertIsNone(report['1']['model']['ready']['brier'])

    def test_empty_categorical_prior_remains_normalized(self):
        target = torch.zeros(2, 3, 153)
        prior = empirical_prior(target, torch.zeros_like(target))
        torch.testing.assert_close(prior[:, NEXT_KIND].sum(-1), torch.ones(3))
        torch.testing.assert_close(prior[:, NEXT_DISCARD].sum(-1), torch.ones(3))