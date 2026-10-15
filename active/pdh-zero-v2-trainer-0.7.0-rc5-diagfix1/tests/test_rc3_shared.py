"""共享缓冲不得改变动作或结算；写入下一步不得修改之前的Actor观察。"""
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pdh_v2.game.session import GameSession
from pdh_v2.observation import decision_view
from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions
from pdh_v2.rollout.shared_views import SharedViews
from pdh_v2.rollout.vector_env import EnvDecision
from pdh_v2.training.paired_evaluation import paired_evaluate
from pdh_v2.training.collector import RolloutCollector, CollectionConfig
from tests.test_refinement import fixture_model


class SharedTransportTests(unittest.TestCase):
    def test_shared_view_survives_next_write(self):
        transport = SharedViews(mp.get_context('spawn'), 1)
        view = decision_view(GameSession(31))
        metadata = transport.write((EnvDecision(0, 'H', view),))
        original = transport.read(metadata)[0].view.observation
        before = original.copy()
        transport.write((EnvDecision(0, 'G', decision_view(GameSession(73))),))
        np.testing.assert_array_equal(original, before)

    def test_capacity_overflow_is_not_silent_truncation(self):
        transport = SharedViews(mp.get_context('spawn'), 1, capacity=1)
        with self.assertRaises(ValueError):
            transport.write((EnvDecision(0, 'H', decision_view(GameSession(31))),))

    def test_fixed_actions_shared_and_serial_match(self):
        serial = BatchedSessions(4, 73)
        shared = ParallelSessions(4, 73, 2, transport='shared')
        try:
            while not serial.all_inactive():
                left, right = serial.requests(), shared.requests()
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    self.assertEqual(a.view.actions, b.view.actions)
                    np.testing.assert_array_equal(np.asarray(a.view.observation, np.float32),
                                                  b.view.observation)
                choices = {r.env_index: 0 for r in left}
                self.assertEqual(serial.step(choices, True), shared.step(choices, True))
        finally:
            shared.close()
        self.assertTrue(all(not process.is_alive() for process in shared._processes))

    def test_shared_paired_evaluation_identical(self):
        model = fixture_model()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            left, _ = paired_evaluate(model, torch.device('cpu'), root / 'serial', 8, 4, 0, 71)
            right, _ = paired_evaluate(model, torch.device('cpu'), root / 'shared', 8, 4, 2, 71,
                                      transport='shared')
            self.assertEqual(left, right)

    def test_shared_learner_collector_runs(self):
        model = fixture_model()
        env = ParallelSessions(2, 41, 2, transport='shared')
        try:
            data = RolloutCollector(env, model, torch.device('cpu'),
                                    CollectionConfig(minimum_hands=2)).collect()
            self.assertGreater(data.decisions, 0)
            self.assertEqual(sum(data.raw_score_sum), 0)
        finally:
            env.close()