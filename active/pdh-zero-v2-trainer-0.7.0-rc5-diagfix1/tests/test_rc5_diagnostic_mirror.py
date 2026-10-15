"""Diagnostic control mirrors baseline Actor updates without independent PPO drift."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.table.checkpoints import load_candidate, save_candidate, tensor_digest
from pdh_v2.table.data import contract
from pdh_v2.table.model import SeatBelief, TableActorCritic
from pdh_v2.table.study_training import mirror_diagnostic_arm
from pdh_v2.training.candidate_store import file_digest


class _Batch:
    hands = 1
    decisions = 2
    environment_decisions = 3
    learner_score_sum = 4
    raw_score_sum = (0, 0, 0, 0)


def _belief(path: Path) -> None:
    model = SeatBelief(384)
    payload = dict(schema='rc5-belief-v2', state=model.state_dict(), seed=1,
                   steps=1, batch=1, dataset_sha256='fixture',
                   source_sha256=contract()['training_replay_sha256'],
                   baseline_sha256=contract()['baseline']['sha256'],
                   train_prior=torch.zeros(3, 153))
    torch.save(payload, path)
    path.with_suffix('.pt.sha256').write_text(f'{file_digest(path)}  belief.pt\n')


class Rc5DiagnosticMirrorTests(unittest.TestCase):
    def test_each_diagnostic_update_exactly_inherits_baseline_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(7)
            baseline = CandidateActorCritic(
                ModelConfig(risk_policy_weight=0.)).requires_grad_(False)
            source = root/'baseline'
            source.mkdir()
            actor = TableActorCritic.from_baseline(baseline, 'baseline')
            initial = source/'candidates/initial.pt'
            save_candidate(actor, initial, {})
            updates = []
            for update in (1, 2):
                with torch.no_grad():
                    next(actor.parameters()).add_(update / 1000)
                candidate = source/f'candidates/update-{update:04d}-proposal.pt'
                save_candidate(actor, candidate, {})
                updates.append(dict(update_accepted=True,
                                    effective_checkpoint=str(candidate.relative_to(source))))
            belief = root/'belief.pt'
            _belief(belief)
            output = root/'diagnostic'
            config = SimpleNamespace(updates=2, max_training_seconds=60)
            with patch('pdh_v2.table.study_training._collect', return_value=_Batch()):
                model, candidate, report = mirror_diagnostic_arm(
                    baseline, belief, config, 99, output, object(), 'owner', source,
                    {'updates': updates})
            source_model, _ = load_candidate(
                source/'candidates/update-0002-proposal.pt', torch.device('cpu'))
            keys = tuple(baseline.state_dict())
            expected = tensor_digest({k: source_model.state_dict()[k] for k in keys})
            actual = tensor_digest({k: model.state_dict()[k] for k in keys})
            self.assertEqual(actual, expected)
            self.assertFalse(report['independent_ppo'])
            self.assertEqual(report['complete_hands'], 2)
            self.assertTrue(all(row['actor_tensor_exact'] for row in report['updates']))
            loaded, _ = load_candidate(candidate, torch.device('cpu'))
            roundtrip = tensor_digest({k: loaded.state_dict()[k] for k in keys})
            self.assertEqual(roundtrip, expected)
