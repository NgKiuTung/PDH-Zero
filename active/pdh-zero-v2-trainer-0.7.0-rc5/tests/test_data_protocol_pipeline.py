"""完整手局审计与运行级隔离；重复评估不能被当作新的独立样本。"""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pdh_v2.training.data_audit import audit_run
from pdh_v2.training.refinement_control import RefinementConfig
from pdh_v2.training.refinement_runner import run_refinement
from tests.test_refinement import write_warmup


class DataProtocolPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.warmup = write_warmup(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def config(self, name, **overrides):
        """用完整自然牌局小预算覆盖真实运行入口，而不是模拟PPO返回成功。"""
        values = dict(output=self.root/name, warmup_checkpoint=self.warmup, workers=0,
                      environments=2, hands_per_update=2, max_seconds=180,
                      target_decisions=1, evaluation_hands=4, minibatch_decisions=512,
                      device='cpu')
        return RefinementConfig(**dict(values, **overrides))

    def test_distinct_runs_record_different_default_seed(self):
        first, second = self.config('first'), self.config('second')
        self.assertEqual(run_refinement(first, False), 0)
        self.assertEqual(run_refinement(second, False), 0)
        a = json.loads((first.output/'data-integrity.json').read_text())
        b = json.loads((second.output/'data-integrity.json').read_text())
        self.assertNotEqual(a['run_seed'], b['run_seed'])
        self.assertNotEqual(a['development_seed'], b['development_seed'])
        self.assertGreater(a['completed_training_action_events'], 2)
        self.assertFalse(a['generalization_verified'])
        self.assertEqual(a['lineage']['historical_coverage'], 'partial')

    def test_reused_explicit_development_seed_blocks_before_optimizer(self):
        first = self.config('first', seed=814)
        self.assertEqual(run_refinement(first, False), 0)
        second = self.config('second', seed=814)
        with patch('pdh_v2.training.refinement_runner.guarded_update') as optimizer:
            with self.assertRaisesRegex(ValueError, 'reused'):
                run_refinement(second, False)
        optimizer.assert_not_called()
        manifest = json.loads((second.output/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertFalse(manifest['long_training_allowed'])

    def test_audit_reads_complete_rollouts_and_fixed_pair_without_overlap(self):
        config = self.config('run')
        self.assertEqual(run_refinement(config, False), 0)
        report = audit_run(config.output, replay_samples=1)
        self.assertEqual(report['status'], 'PASS')
        self.assertFalse(any(report['training_development_starting_wall_overlap'].values()))
        self.assertTrue(all(r['sampled_full_rule_replays_verified'] == 1
                            for r in report['replays'].values()))
        self.assertTrue(all(r['action_events_min'] > 1
                            for r in report['replays'].values()))

    def test_audit_detects_actual_cross_split_wall_reuse(self):
        config = self.config('run')
        self.assertEqual(run_refinement(config, False), 0)
        with gzip.open(config.output/'evaluations/baseline/hands.jsonl.gz', 'rt') as stream:
            row = stream.readline()
        with gzip.open(config.output/'replays/duplicate.jsonl.gz', 'wt') as stream:
            stream.write(row)
        report = audit_run(config.output, replay_samples=0)
        self.assertEqual(report['status'], 'OVERLAP')

    def test_audit_missing_replays_refuses_pass(self):
        with self.assertRaisesRegex(ValueError, 'no_replays'):
            audit_run(self.root)

    def test_audit_negative_sample_count_refused(self):
        with self.assertRaisesRegex(ValueError, 'negative'):
            audit_run(self.root, -1)
