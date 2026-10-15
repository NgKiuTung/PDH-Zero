"""Public/private intervention fixtures must be legal and causally interpretable."""
import unittest

import numpy as np

from pdh_v2.game.session import GameSession
from pdh_v2.observation import decision_view
from pdh_v2.table.audit import private_intervention, public_history_pair


class Rc5AuditTests(unittest.TestCase):
    def test_private_intervention_preserves_actor_input(self):
        session = GameSession(93822)
        original = decision_view(session)

        shadow = decision_view(private_intervention(session, original.player))

        np.testing.assert_array_equal(original.observation, shadow.observation)
        np.testing.assert_array_equal(original.action_features, shadow.action_features)
        self.assertEqual(original.actions, shadow.actions)

    def test_public_pair_preserves_own_hand_but_changes_public_history(self):
        seed = 830000001

        initial, left, right = public_history_pair(seed)

        self.assertEqual(initial.player, left.player)
        self.assertEqual(left.actions, right.actions)
        np.testing.assert_array_equal(left.observation[:68], right.observation[:68])
        self.assertFalse(np.array_equal(left.observation, right.observation))