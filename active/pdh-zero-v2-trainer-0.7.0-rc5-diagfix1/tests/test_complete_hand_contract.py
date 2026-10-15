"""整局采样、局内GRU历史、完整终局结算与四座位配对的实际回归。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pdh_v2.game.session import GameSession
from pdh_v2.observation import decision_view
from pdh_v2.replay import replay_hand
from pdh_v2.rollout import BatchedSessions
from pdh_v2.training.buffer import Episode
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.paired_evaluation import paired_evaluate
from pdh_v2.training.sequence_batch import sequence_groups, sequence_values
from pdh_v2.training.wall_registry import wall_fingerprint
from tests.test_refinement import fixture_model


class CompleteHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fixture_model()
        cls.backend = BatchedSessions(2, 317)
        cls.collector = RolloutCollector(cls.backend, cls.model, torch.device('cpu'),
                                        CollectionConfig(minimum_hands=2))
        cls.first = cls.collector.collect()
        cls.second = cls.collector.collect()

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_each_record_replays_to_real_terminal(self):
        results = [replay_hand(t.transcript).outcome for t in self.first.terminals]
        self.assertTrue(all(item is not None for item in results))
        self.assertTrue(all(len(t.transcript.actions) > 1 for t in self.first.terminals))

    def test_new_update_continues_session_and_uses_new_walls(self):
        first = {wall_fingerprint(t.transcript.wall) for t in self.first.terminals}
        second = {wall_fingerprint(t.transcript.wall) for t in self.second.terminals}
        self.assertFalse(first & second)
        self.assertTrue(all(t.hand_id != 'H00000000' for t in self.second.terminals))

    def test_dealer_state_is_preserved_between_hands(self):
        endings = {t.env_index: t.audit['after_dealer'] for t in self.first.terminals}
        starts = {t.env_index: t.audit['before_dealer'] for t in self.second.terminals}
        self.assertEqual(endings, starts)

    def test_terminal_net_score_is_the_reward(self):
        rewards = sum(sum(t.reward for t in e.transitions) for e in self.first.episodes)
        self.assertAlmostEqual(rewards * 32, self.first.learner_score_sum)
        self.assertTrue(all(t.reward == 0 for e in self.first.episodes
                            for t in e.transitions[:-1]))

    def test_batching_keeps_entire_episode_objects(self):
        groups = sequence_groups(self.first.episodes, 8, 10)
        flattened = [e for group in groups for e in group]
        self.assertEqual({id(e) for e in flattened}, {id(e) for e in self.first.episodes})
        self.assertEqual(sum(e.decisions for e in flattened), self.first.decisions)

    def test_policy_replays_whole_history_not_isolated_questions(self):
        with torch.no_grad():
            values = sequence_values(self.model, list(self.first.episodes), torch.device('cpu'))
        self.assertLess(float((values.log_prob - values.old_log_prob).abs().max()), 1e-4)

    def test_dropping_prefix_changes_decision_prediction(self):
        episode = self.first.episodes[0]
        index = next(i for i, t in enumerate(episode.transitions)
                     if i > 5 and len(t.action_features) > 1)
        prefix = Episode(transitions=episode.transitions[:index+1])
        isolated = Episode(transitions=[episode.transitions[index]])
        with torch.no_grad():
            full = sequence_values(self.model, [prefix], torch.device('cpu')).log_prob[-1]
            cut = sequence_values(self.model, [isolated], torch.device('cpu')).log_prob[-1]
        self.assertGreater(float((full-cut).abs()), 1e-7)

    def test_actor_projection_does_not_expose_future_wall_order(self):
        session = GameSession(713)
        original = decision_view(session)
        altered = copy.deepcopy(session)
        order = list(altered.current_hand.wall.order)
        order[90], order[95] = order[95], order[90]
        altered.current_hand.wall.order = tuple(order)
        changed = decision_view(altered)
        self.assertEqual(original.observation, changed.observation)
        self.assertEqual(original.action_features, changed.action_features)

    def test_four_seat_evaluation_is_four_complete_games_not_four_cut_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            before = copy.deepcopy(self.model.state_dict())
            rows, summary = paired_evaluate(self.model, torch.device('cpu'), Path(directory)/'eval',
                                            hands=4, environments=2, workers=0, seed=991)
            self.assertEqual(summary['hands'], 4)
            self.assertEqual(summary['unique_starting_walls'], 1)
            self.assertEqual({r['candidate_seat'] for r in rows}, {0, 1, 2, 3})
            self.assertGreater(summary['action_events_total'], 4)
            self.assertTrue(all(torch.equal(v, self.model.state_dict()[k])
                                for k, v in before.items()))
            self.assertEqual(summary['unit'], 'complete_natural_hand_not_isolated_decision')
