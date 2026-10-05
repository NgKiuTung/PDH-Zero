"""State-machine, legality, conservation, and zero-sum tests."""

from __future__ import annotations

import unittest

from pdh_phase1.config import load_config
from pdh_phase1.mini_pdh import IllegalActionError, MiniPDHEnv
from pdh_phase1.mini_pdh.bots import RandomLegalBot


class EnvironmentTests(unittest.TestCase):
    """Exercise complete random games and explicit failure paths."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_config("configs/mini_pdh_v0.yaml")

    def test_random_games_preserve_tiles_and_zero_sum(self) -> None:
        for seed in range(200):
            env = MiniPDHEnv(self.rules)
            env.reset(seed)
            bot = RandomLegalBot(seed + 99)
            while not env.done:
                env.step(bot.choose_action(env))
            env.assert_invariants()
            self.assertAlmostEqual(sum(env.rewards), 0.0)

    def test_illegal_action_raises(self) -> None:
        env = MiniPDHEnv(self.rules)
        env.reset(7)
        illegal = next(index for index, allowed in enumerate(env.legal_actions()) if not allowed)
        with self.assertRaises(IllegalActionError):
            env.step(illegal)

    def test_same_seed_and_action_policy_is_deterministic(self) -> None:
        digests = []
        for _ in range(2):
            env = MiniPDHEnv(self.rules)
            env.reset(12345)
            while not env.done:
                action = next(index for index, allowed in enumerate(env.legal_actions()) if allowed)
                env.step(action)
            digests.append(env.state_digest())
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
