"""CPU子进程生命周期、观察缓存与串行差分。"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions
from pdh_v2.observation import decision_view
from pdh_v2.training.paired_evaluation import paired_evaluate
from tests.test_refinement import fixture_model


class ParallelRefinementTests(unittest.TestCase):
    def test_cached_view_not_computed_again_for_step(self):
        env=BatchedSessions(1,31)
        with patch('pdh_v2.rollout.vector_env.decision_view',wraps=decision_view) as call:
            env.requests(); env.requests(); env.step({0:0})
            self.assertEqual(call.call_count,1)

    def test_negative_action_index_rejected(self):
        with self.assertRaises(ValueError):
            BatchedSessions(1,31).step({0:-1})

    def test_worker_and_serial_fixed_actions_match(self):
        serial=BatchedSessions(4,73)
        parallel=ParallelSessions(4,73,2)
        try:
            while not serial.all_inactive():
                self.assertEqual(serial.requests(),parallel.requests())
                choices={r.env_index:0 for r in serial.requests()}
                self.assertEqual(serial.step(choices,True),parallel.step(choices,True))
            self.assertTrue(parallel.all_inactive())
            self.assertEqual(len(parallel.snapshot()),4)
        finally:
            parallel.close()
        self.assertTrue(all(not p.is_alive() for p in parallel._processes))

    def test_heuristic_actions_match(self):
        serial=BatchedSessions(2,71)
        parallel=ParallelSessions(2,71,2)
        try:
            self.assertEqual(serial.heuristic_actions(serial.requests()),
                             parallel.heuristic_actions(parallel.requests()))
        finally:
            parallel.close()

    def test_dead_worker_fails_bounded_not_hang(self):
        env=ParallelSessions(1,13,1,timeout=5)
        env._processes[0].terminate(); env._processes[0].join(timeout=3)
        try:
            with self.assertRaises((RuntimeError,EOFError,BrokenPipeError)):
                env.requests()
        finally:
            env.close()

    def test_serial_parallel_paired_evaluation_same_records(self):
        model=fixture_model()
        with tempfile.TemporaryDirectory() as d:
            a,_=paired_evaluate(model,torch.device('cpu'),Path(d)/'a',8,4,0,71)
            b,_=paired_evaluate(model,torch.device('cpu'),Path(d)/'b',8,4,2,71)
            self.assertEqual(a,b)
            self.assertEqual(len(a),8)
