"""Recurrent policy shape and legal-mask tests."""

from __future__ import annotations

import unittest

import torch

from pdh_phase1.mini_pdh import ACTION_COUNT, ACTOR_OBSERVATION_DIM
from pdh_phase1.rl.model import RecurrentActorCritic


class ModelTests(unittest.TestCase):
    """Check recurrent interfaces used by collection and PPO."""

    def test_masked_action_is_always_legal(self) -> None:
        model = RecurrentActorCritic(ACTOR_OBSERVATION_DIM, ACTOR_OBSERVATION_DIM, 32)
        actor = torch.randn(ACTOR_OBSERVATION_DIM)
        legal = torch.zeros(ACTION_COUNT, dtype=torch.bool)
        legal[3] = True
        state = model.initial_state(1, torch.device("cpu"))
        output = model.act(actor, actor, legal, state)
        self.assertEqual(output.action, 3)

    def test_sequence_shapes(self) -> None:
        model = RecurrentActorCritic(ACTOR_OBSERVATION_DIM, ACTOR_OBSERVATION_DIM, 32)
        actor = torch.randn(2, 5, ACTOR_OBSERVATION_DIM)
        legal = torch.ones(2, 5, ACTION_COUNT, dtype=torch.bool)
        logits, values, state = model.forward_sequences(actor, actor, legal)
        self.assertEqual(tuple(logits.shape), (2, 5, ACTION_COUNT))
        self.assertEqual(tuple(values.shape), (2, 5))
        self.assertEqual(tuple(state.actor.shape), (1, 2, 32))


if __name__ == "__main__":
    unittest.main()
