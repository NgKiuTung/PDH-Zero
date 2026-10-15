"""Labels distinguish current capability, observed behavior, and censored events."""
import unittest

import numpy as np

from pdh_v2.game.session import GameSession
from pdh_v2.table.labels import future_behavior, state_targets, waiting_capabilities
from pdh_v2.table.model import NEXT_DISCARD, NEXT_KIND, ROUTES


class Rc5LabelTests(unittest.TestCase):
    def test_state_targets_have_separate_masks(self):
        session = GameSession(8293)

        target, mask = state_targets(session, 0)

        self.assertEqual(target.shape, (3, 153))
        np.testing.assert_array_equal(mask[:, ROUTES], np.ones((3, 8)))
        self.assertEqual(mask[:, NEXT_KIND].sum(), 0)

    def test_missing_future_event_is_censored_not_pass(self):
        target, mask = np.zeros((3, 153)), np.zeros((3, 153))

        future_behavior(target, mask, (), 0, 0)

        self.assertEqual(mask[:, NEXT_KIND].sum(), 0)
        self.assertEqual(mask[:, NEXT_DISCARD].sum(), 0)

    def test_future_event_is_relative_seat_and_strictly_after_snapshot(self):
        target, mask = np.zeros((3, 153)), np.zeros((3, 153))
        actions = ({"player": 1, "kind": "DISCARD", "tiles": [12], "choices": 2},
                   {"player": 1, "kind": "DISCARD", "tiles": [20], "choices": 2})

        future_behavior(target, mask, actions, 0, 0)

        self.assertEqual(target[0, NEXT_DISCARD.start + 3], 0)
        self.assertEqual(target[0, NEXT_DISCARD.start + 5], 1)
        self.assertEqual(mask[1:, NEXT_DISCARD].sum(), 0)

    def test_forced_pass_is_not_a_voluntary_action_target(self):
        target, mask = np.zeros((3, 153)), np.zeros((3, 153))
        actions = ({"player": 0, "kind": "DISCARD", "tiles": [8], "choices": 12},
                   {"player": 1, "kind": "PASS", "tiles": [], "choices": 1})

        future_behavior(target, mask, actions, 0, 0)

        self.assertEqual(mask[0, NEXT_KIND].sum(), 0)

    def test_nonwaiting_size_is_censored(self):
        session = GameSession(19)

        _, _, _, _, _, valid = waiting_capabilities(session.current_hand.players[0], 83, False)

        self.assertFalse(valid)