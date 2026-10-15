"""Controlled architecture: zero residual and diagnostic mode cannot alter the Actor."""
import copy
import unittest

import torch

from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.table.model import TableActorCritic


def fixture():
    torch.manual_seed(31)
    base = CandidateActorCritic(ModelConfig(hidden_size=32, risk_policy_weight=0.0))
    inputs = (torch.randn(2, 622), torch.zeros(2, 32), torch.randn(5, 74),
              torch.tensor([0, 0, 1, 1, 1]))
    return base, inputs


class Rc5ModelTests(unittest.TestCase):
    def test_diagnostic_logits_and_hidden_are_exact(self):
        base, inputs = fixture()
        model = TableActorCritic.from_baseline(base, "diagnostic")

        expected, actual = base.forward_step(*inputs), model(*inputs)

        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)

    def test_initial_joint_is_exact_and_zero_residual_is_trainable(self):
        base, inputs = fixture()
        model = TableActorCritic.from_baseline(base, "joint")

        expected, actual = base.forward_step(*inputs), model(*inputs)
        actual[0].sum().backward()

        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        self.assertGreater(model.fusion[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.belief.head[0].weight.grad)

    def test_auxiliary_gradient_cannot_reach_actor(self):
        base, inputs = fixture()
        model = TableActorCritic.from_baseline(base, "diagnostic")
        _, _, _, hidden = model(*inputs)

        model.belief(inputs[0], hidden).sum().backward()

        self.assertIsNone(model.memory.weight_hh.grad)
        self.assertIsNotNone(model.belief.head[0].weight.grad)

    def test_modes_reject_unknown_value(self):
        base, _ = fixture()
        with self.assertRaises(ValueError):
            TableActorCritic.from_baseline(base, "latest")

    def test_roundtrip_state_is_exact(self):
        base, inputs = fixture()
        model = TableActorCritic.from_baseline(base, "joint")
        clone = TableActorCritic.from_baseline(base, "joint")

        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)

        torch.testing.assert_close(model(*inputs)[0], clone(*inputs)[0], rtol=0, atol=0)