"""稳定训练改动的复现、边界和无泄漏回归；全部可在CPU执行。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from pdh_v2.game.session import GameSession
from pdh_v2.observation import decision_view
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.checkpoint import save_checkpoint
from pdh_v2.training.paired_metrics import paired_comparison, development_gate
from pdh_v2.training.refinement_artifacts import load_warmup
from pdh_v2.training.refinement_control import RefinementConfig, StopRequested
from pdh_v2.training.refinement_runner import run_refinement
from pdh_v2.training.risk_diagnostics import probability_metrics
from pdh_v2.training.stable_ppo import StablePPOConfig, guarded_update


def fixture_model(width=16):
    torch.manual_seed(101)
    torch.set_num_threads(1)
    return CandidateActorCritic(ModelConfig(hidden_size=width,risk_policy_weight=0.0))


def paired_rows(gain=0):
    return [dict(wall_seed=seed,candidate_seat=seat,score=seed%7+gain,
                 point_shoot=0,win=0,draw=1) for seed in range(3) for seat in range(4)]


def write_warmup(root,update=0):
    model=fixture_model()
    path=root/'warmup.pt'
    save_checkpoint(path,model,torch.optim.AdamW(model.parameters()),update,{'teacher_hands':8})
    return path


class RefinementTests(unittest.TestCase):
    def test_risk_supervision_does_not_modify_actor_encoder_or_logits(self):
        model=fixture_model()
        view=decision_view(GameSession(41))
        obs=torch.tensor([view.observation]); features=torch.tensor(view.action_features)
        owners=torch.zeros(len(view.actions),dtype=torch.long)
        hidden=model.initial_state(1,torch.device('cpu'))
        old=copy.deepcopy(model.state_dict())
        logits_before=model.forward_step(obs,hidden,features,owners)[0].detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
        for _ in range(3):
            risk=model.forward_step(obs,hidden,features,owners)[1]
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(risk,torch.zeros_like(risk)).backward()
            optimizer.step()
        self.assertTrue(torch.equal(old['observation_encoder.0.weight'],
                                    model.state_dict()['observation_encoder.0.weight']))
        after_logits = model.forward_step(obs, hidden, features, owners)[0]
        self.assertTrue(torch.equal(logits_before, after_logits))
        self.assertFalse(torch.equal(
            old['risk_pair.2.bias'], model.state_dict()['risk_pair.2.bias']))

    def test_paired_gain_and_gate(self):
        result=paired_comparison(paired_rows(),paired_rows(1),100)
        self.assertEqual(result['score']['ci95'],[1.0,1.0])
        self.assertEqual(development_gate(result),'PASS')

    def test_paired_regression(self):
        self.assertEqual(development_gate(paired_comparison(paired_rows(),paired_rows(-1),100)),
                         'REGRESSION')

    def test_zero_difference_is_inconclusive(self):
        self.assertEqual(development_gate(paired_comparison(paired_rows(),paired_rows(),100)),
                         'INCONCLUSIVE')

    def test_missing_paired_seat_refused(self):
        with self.assertRaises(ValueError):
            paired_comparison(paired_rows(),paired_rows()[:-1])

    def test_duplicate_paired_seat_refused(self):
        with self.assertRaises(ValueError):
            paired_comparison(paired_rows(),paired_rows()+paired_rows()[:1])

    def test_AP_ties_not_artificially_perfect(self):
        result=probability_metrics([.1,.1,.1,.1],[1,0,0,0])
        self.assertEqual(result['average_precision'],.25)

    def test_AP_without_positives_is_unknown(self):
        self.assertIsNone(probability_metrics([.01,.02],[0,0])['average_precision'])

    def test_AP_ordered_perfect(self):
        self.assertEqual(probability_metrics([.8,.7,.2],[1,1,0])['average_precision'],1)

    def test_probability_bounds(self):
        with self.assertRaises(ValueError):
            probability_metrics([float('nan')],[0])

    def test_warmup_load_and_freeze_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            model,anchor,report=load_warmup(write_warmup(Path(d)),torch.device('cpu'))
            self.assertEqual(model.config.risk_policy_weight,0)
            self.assertFalse(any(p.requires_grad for p in anchor.parameters()))
            self.assertIsNot(model,anchor)
            self.assertEqual(len(report['checkpoint_sha256']),64)

    def test_latest_checkpoint_is_not_warmup(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_warmup(write_warmup(Path(d),16),torch.device('cpu'))

    def test_controlled_cpu_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); warmup=write_warmup(root)
            config=RefinementConfig(output=root/'run',warmup_checkpoint=warmup,workers=0,
                environments=2,hands_per_update=2,max_seconds=180,target_decisions=1,
                evaluation_hands=4,minibatch_decisions=512,device='cpu')
            self.assertEqual(run_refinement(config,install_signals=False),0)
            result=json.loads((config.output/'manifest.json').read_text())
            self.assertEqual(result['status'],'completed')
            self.assertGreater(result['total_learner_decisions'],0)
            self.assertFalse(result['long_training_allowed'])
            self.assertTrue((config.output/'checkpoints/checkpoint_best.pt').is_file())
            self.assertTrue((config.output/'evaluations/baseline/results.jsonl').is_file())

    def test_manual_stop_is_not_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); warmup=write_warmup(root)
            config=RefinementConfig(output=root/'run',warmup_checkpoint=warmup,device='cpu')
            with patch('pdh_v2.training.refinement_runner._run',
                       side_effect=StopRequested('signal_2')):
                self.assertEqual(run_refinement(config,install_signals=False),0)
            result=json.loads((config.output/'manifest.json').read_text())
            self.assertEqual(result['status'],'stopped')
            self.assertEqual(result['stop_reason'],'signal_2')

    def test_existing_run_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); warmup=write_warmup(root); out=root/'run'; out.mkdir()
            (out/'sentinel').write_text('unchanged')
            with self.assertRaises(FileExistsError):
                config = RefinementConfig(output=out, warmup_checkpoint=warmup, device='cpu')
                run_refinement(config, False)
            self.assertEqual((out/'sentinel').read_text(),'unchanged')


class PPOGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=fixture_model()
        cls.data=RolloutCollector(BatchedSessions(1,317),cls.model,torch.device('cpu'),
                                  CollectionConfig(minimum_hands=1)).collect()

    def run_update(self,lr=.00005,kl=.01):
        model=copy.deepcopy(self.model); anchor=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0)
        before=copy.deepcopy(model.state_dict())
        result=guarded_update(model,optimizer,anchor,self.data.episodes,torch.device('cpu'),
                             StablePPOConfig(epochs=1,minibatch_decisions=512,target_kl=kl),31)
        return result,before,model,optimizer

    def test_collected_policy_recomputes_same_history(self):
        result,_,_,_=self.run_update()
        self.assertLess(result['behavior_replay_max_abs_error'],.0005)
        self.assertGreater(result['unique_actor_decisions'],0)

    def test_excessive_step_rolls_back_weights_and_optimizer(self):
        result,before,model,optimizer=self.run_update(.5,.00000001)
        self.assertFalse(result['update_accepted'])
        self.assertTrue(all(torch.equal(t,model.state_dict()[k]) for k,t in before.items()))
        self.assertEqual(len(optimizer.state),0)

    def test_stale_policy_rejected_before_update(self):
        model=copy.deepcopy(self.model)
        episodes=copy.deepcopy(self.data.episodes)
        episodes[0].transitions[0].old_log_prob += .1
        anchor=copy.deepcopy(model).eval().requires_grad_(False)
        with self.assertRaises(RuntimeError):
            guarded_update(model,torch.optim.AdamW(model.parameters()),anchor,episodes,
                           torch.device('cpu'),StablePPOConfig(),1)
