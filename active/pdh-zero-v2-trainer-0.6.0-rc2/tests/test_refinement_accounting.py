"""统计分母、抢杠标签与可重放账本的业务回归。"""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from pdh_v2.game.actions import ActionKind
from pdh_v2.observation.encoder import ACTION_KIND_INDEX
from pdh_v2.replay.audit import audit_terminal_record
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.buffer import Transition
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.curriculum import SeatMix
from pdh_v2.training.events import ReplayWriter
from pdh_v2.training.refinement_metrics import collection_metrics
from tests.test_refinement import fixture_model


class AccountingTests(unittest.TestCase):
    def test_rob_kong_does_not_label_previous_discard_as_positive(self):
        model=fixture_model()
        collector=RolloutCollector(BatchedSessions(1,311),model,torch.device('cpu'),
            CollectionConfig(seat_mix=SeatMix((0,0,0,1))))
        transition=Transition(torch.zeros(1),torch.zeros(1,1),0,0,0,
            action_kind=ACTION_KIND_INDEX[ActionKind.DISCARD],risk_target=0.0)
        collector.current[0][0].transitions.append(transition)
        terminal=SimpleNamespace(env_index=0,hand_id='H00000000',discarder=0,
                                 method='ROB_KONG',winners=(1,),receipt=(-3,3,0,0))
        completed=[]
        collector._finalize_hand(terminal,completed)
        self.assertEqual(transition.risk_target,0.0)

    def test_point_shoot_denominator_is_learner_seat_hands(self):
        model=fixture_model()
        result=RolloutCollector(BatchedSessions(1,313),model,torch.device('cpu'),
            CollectionConfig(minimum_hands=1,seat_mix=SeatMix((0,1,0,0)))).collect()
        metrics=collection_metrics(result,1.0)
        self.assertEqual(metrics['learner_seat_hands'],2*metrics['hands'])
        self.assertEqual(metrics['learner_point_shoot_rate_per_seat'],
                         result.learner_point_shoots/(2*result.hands))

    def test_terminal_replay_verifies_settlement_receipt(self):
        env=BatchedSessions(2,311)
        terminals=[]
        while not env.all_inactive():
            terminals.extend(env.step(env.heuristic_actions(env.requests()),True))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'hands.gz'
            ReplayWriter(path).write_many(tuple(terminals),1)
            with gzip.open(path,'rt') as stream:
                rows=[json.loads(line) for line in stream]
            self.assertTrue(all(audit_terminal_record(r)['status']=='PASS' for r in rows))
            bad=copy.deepcopy(rows[0]);bad['receipt'][0]+=1
            with self.assertRaises(ValueError):
                audit_terminal_record(bad)

    def test_hidden_reveal_initial_state_can_be_replayed(self):
        env=BatchedSessions(1,313)
        hand=env.sessions[0].current_hand
        hand.players[0].revealed_retained.add(hand.players[0].concealed[0])
        terminals=[]
        while not env.all_inactive():
            terminals.extend(env.step(env.heuristic_actions(env.requests()),True))
        terminal=terminals[0]
        record={'transcript':json.loads(terminal.transcript.to_json()),
                'audit':terminal.audit,'receipt':list(terminal.receipt)}
        self.assertEqual(audit_terminal_record(record)['status'],'PASS')
