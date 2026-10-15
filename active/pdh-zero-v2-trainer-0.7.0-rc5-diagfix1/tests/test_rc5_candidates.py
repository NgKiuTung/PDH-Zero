"""Exact saved weights and pre-rollback proposal receipts; fixtures are not trained results."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.table.checkpoints import save_candidate, load_candidate, tensor_digest
from pdh_v2.table.model import TableActorCritic
from pdh_v2.training.collector import RolloutCollector, CollectionConfig
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.stable_ppo import guarded_update, StablePPOConfig
from tests.test_refinement import fixture_model


class Rc5CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(813)
        cls.model = TableActorCritic(ModelConfig(risk_policy_weight=0.), 'joint')

    def test_exact_tensor_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.pt'
            receipt = save_candidate(self.model, path, {'fixture': True})
            loaded, _ = load_candidate(path, torch.device('cpu'))
            self.assertEqual(tensor_digest(loaded.state_dict()), receipt['tensor_sha256'])
            self.assertFalse(loaded.belief.head[0].weight.requires_grad)

    def test_candidate_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.pt'
            path.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                save_candidate(self.model, path, {})
            self.assertEqual(path.read_bytes(), b'preserve')

    def test_tampered_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.pt'
            save_candidate(self.model, path, {})
            path.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'digest'):
                load_candidate(path, torch.device('cpu'))

    def test_toy_width_not_accepted_as_real_rc5_candidate(self):
        toy = TableActorCritic.from_baseline(fixture_model(), 'joint')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.pt'
            save_candidate(toy, path, {'fixture': True})
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_candidate(path, torch.device('cpu'))

    def test_actual_proposal_seen_before_model_and_optimizer_rollback(self):
        model = fixture_model()
        base = copy.deepcopy(model).requires_grad_(False)
        before = tensor_digest(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.5)
        data = RolloutCollector(BatchedSessions(1, 834), model, torch.device('cpu'),
                                CollectionConfig(minimum_hands=1)).collect()
        seen = []
        def receipt(candidate, statistics):
            seen.append((tensor_digest(candidate.state_dict()), statistics['update_accepted']))
        result = guarded_update(model, optimizer, base, data.episodes, torch.device('cpu'),
            StablePPOConfig(epochs=1, target_kl=1e-8, minibatch_decisions=512), 91,
            on_proposal=receipt)
        self.assertFalse(result['update_accepted'])
        self.assertEqual(len(seen), 1)
        self.assertNotEqual(seen[0][0], before)
        self.assertEqual(tensor_digest(model.state_dict()), before)
        self.assertEqual(len(optimizer.state), 0)