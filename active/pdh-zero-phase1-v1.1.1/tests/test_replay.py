"""Deterministic replay artifact tests."""

from __future__ import annotations

import unittest

from pdh_phase1.config import load_config
from pdh_phase1.mini_pdh import MiniPDHEnv
from pdh_phase1.mini_pdh.bots import GoalDistanceBot
from pdh_phase1.mini_pdh.replay import replay_game


class ReplayTests(unittest.TestCase):
    """Verify a compact seed/action record reproduces terminal state exactly."""

    def test_exported_game_replays_exactly(self) -> None:
        rules = load_config("configs/mini_pdh_v0.yaml")
        env = MiniPDHEnv(rules)
        env.reset(444)
        bot = GoalDistanceBot(777)
        while not env.done:
            env.step(bot.choose_action(env))
        record = env.export_replay()
        replayed = replay_game(rules, record)
        self.assertEqual(replayed.state_digest(), env.state_digest())


if __name__ == "__main__":
    unittest.main()
