"""候选动作循环策略测试。"""
from __future__ import annotations

import unittest

import torch

from pdh_v2.observation import ACTION_FEATURE_DIM, OBSERVATION_DIM
from pdh_v2.policy import CandidateActorCritic, ModelConfig


class PolicyTests(unittest.TestCase):
    def test_forward_shapes_and_gradients(self) -> None:
        model = CandidateActorCritic(ModelConfig(hidden_size=32))
        observations = torch.randn(3, OBSERVATION_DIM)
        hidden = model.initial_state(3, torch.device("cpu"))
        actions = torch.randn(9, ACTION_FEATURE_DIM)
        actions[:, :7] = 0
        actions[torch.arange(9), torch.arange(9) % 7] = 1
        owners = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
        logits, values, next_hidden = model.forward_step(
            observations, hidden, actions, owners
        )
        self.assertEqual(tuple(logits.shape), (9,))
        self.assertEqual(tuple(values.shape), (3,))
        self.assertEqual(tuple(next_hidden.shape), (3, 32))
        (logits.mean() + values.mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
