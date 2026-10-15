"""局部KL、累计KL与开发选择各有独立门禁，不混为统计结论。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.refinement_control import Progress, RefinementConfig
from pdh_v2.training.refinement_runner import run_refinement
from pdh_v2.training.refinement_gate import review_run
from pdh_v2.training.refinement_selection import DevelopmentSelection, adapt_update_config
from pdh_v2.training.stable_ppo import StablePPOConfig, guarded_update
from tests.test_refinement import fixture_model, paired_rows, write_warmup
from tests.test_rc3_evidence import source


class AnchorGuardTests(unittest.TestCase):
    def test_small_local_step_still_rejected_by_anchor_limit(self):
        model = fixture_model()
        anchor = copy.deepcopy(model).eval().requires_grad_(False)
        data = RolloutCollector(BatchedSessions(1, 317), model, torch.device('cpu'),
                               CollectionConfig(minimum_hands=1)).collect()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00005, weight_decay=0)
        before = copy.deepcopy(model.state_dict())
        report = guarded_update(model, optimizer, anchor, data.episodes, torch.device('cpu'),
            StablePPOConfig(epochs=1, max_anchor_kl=1e-10), 31)
        self.assertFalse(report['update_accepted'])
        self.assertEqual(report['rejection_reason'], 'anchor_kl')
        self.assertLess(report['post_update_kl'], .01)
        self.assertEqual(report['effective_anchor_kl'], 0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertEqual(len(optimizer.state), 0)

    def test_adaptation_is_bounded_not_repeated_halving(self):
        config = StablePPOConfig()
        next_config, lr = adapt_update_config(config, {'effective_anchor_kl': .1}, 5e-5)
        self.assertGreater(next_config.anchor_coefficient, config.anchor_coefficient)
        self.assertLess(lr, 5e-5)
        self.assertGreaterEqual(lr, 5e-5 / 4)
        again, again_lr = adapt_update_config(next_config, {'effective_anchor_kl': .1}, 5e-5)
        self.assertEqual(lr, again_lr)
        self.assertEqual(next_config, again)

    def test_invalid_tail_budget_rejected(self):
        with self.assertRaises(ValueError):
            RefinementConfig(Path('/x'), Path('/y'), max_cvar_drop=float('nan'))


class SelectionIntegrationTests(unittest.TestCase):
    def test_full_run_keeps_final_candidate_before_gate(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = RefinementConfig(output=root / 'run', warmup_checkpoint=write_warmup(root),
                workers=0, environments=2, hands_per_update=2, max_seconds=180,
                target_decisions=1, evaluation_hands=4, minibatch_decisions=512, device='cpu')
            self.assertEqual(run_refinement(config, install_signals=False), 0)
            saved = list((config.output / 'checkpoints/candidates').glob('*/model.pt'))
            self.assertEqual(len(saved), 2)
            selected = json.loads((config.output / 'selected_development.json').read_text())
            self.assertTrue((config.output / selected['checkpoint']).is_file())
            self.assertFalse(review_run(config.output)['long_training_allowed'])

    def test_nominal_tail_regression_not_selected_even_positive_mean(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            summary = {'mean_score': 0, 'point_shoot': .05, 'cvar_5': -10}
            manager = DevelopmentSelection(root, paired_rows(), summary)
            manager.store.register(source(root), 'd000000000001', 1, 1)
            bad = {'mean_score': 1, 'point_shoot': .04, 'cvar_5': -14}
            manager.observe('d000000000001', paired_rows(1), bad, Progress(root, 60))
            self.assertEqual(manager.selected, 'baseline')
            self.assertEqual(manager.stale_evaluations, 1)