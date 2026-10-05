"""Tests for public Actor isolation and privileged Critic visibility."""

from __future__ import annotations

import unittest

import torch

from pdh_phase1.config import load_config
from pdh_phase1.mini_pdh import MiniPDHEnv, encode_actor, encode_critic


class ObservationTests(unittest.TestCase):
    """Prove hidden-state permutations cannot enter the Actor observation."""

    def test_hidden_permutation_changes_only_privileged_observation(self) -> None:
        rules = load_config("configs/mini_pdh_v0.yaml")
        original = MiniPDHEnv(rules)
        original.reset(20260929)
        altered = original.clone()
        altered.hands[1], altered.hands[2] = altered.hands[2], altered.hands[1]
        altered.wall.reverse()
        altered.assert_invariants()
        seat = original.current_seat
        self.assertTrue(torch.equal(encode_actor(original, seat), encode_actor(altered, seat)))
        self.assertFalse(
            torch.equal(encode_critic(original, seat, True), encode_critic(altered, seat, True))
        )


if __name__ == "__main__":
    unittest.main()
