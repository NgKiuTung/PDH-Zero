"""合法观察与动作编码测试。"""
from __future__ import annotations

import copy
import unittest

from pdh_v2.game.session import GameSession
from pdh_v2.observation import (
    ACTION_FEATURE_DIM,
    OBSERVATION_DIM,
    decision_view,
    encode_observation,
)


class ObservationTests(unittest.TestCase):
    def test_dimensions_and_legal_actions(self) -> None:
        session = GameSession(17)
        view = decision_view(session)
        self.assertEqual(len(view.observation), OBSERVATION_DIM)
        self.assertTrue(view.actions)
        self.assertTrue(
            all(len(features) == ACTION_FEATURE_DIM for features in view.action_features)
        )

    def test_other_hidden_tiles_do_not_change_observation(self) -> None:
        session = GameSession(21)
        clone = copy.deepcopy(session)
        player = decision_view(session).player
        other = (player + 1) % 4
        concealed = clone.current_hand.players[other].concealed
        wall = clone.current_hand.wall
        # NOTE: 隐藏实体互换不会改变公开计数与牌墙长度。
        concealed[0], wall.order = wall.order[wall.front], tuple(
            concealed[0] if index == wall.front else card
            for index, card in enumerate(wall.order)
        )
        self.assertEqual(
            encode_observation(session, player), encode_observation(clone, player)
        )

    def test_own_hand_changes_observation(self) -> None:
        session = GameSession(25)
        clone = copy.deepcopy(session)
        player = decision_view(session).player
        clone.current_hand.players[player].concealed.reverse()
        # Reordering identical private counts should not matter.
        self.assertEqual(
            encode_observation(session, player), encode_observation(clone, player)
        )
        own = clone.current_hand.players[player].concealed
        wall = clone.current_hand.wall
        own[0], wall.order = wall.order[wall.front], tuple(
            own[0] if index == wall.front else card
            for index, card in enumerate(wall.order)
        )
        self.assertNotEqual(
            encode_observation(session, player), encode_observation(clone, player)
        )


if __name__ == "__main__":
    unittest.main()
