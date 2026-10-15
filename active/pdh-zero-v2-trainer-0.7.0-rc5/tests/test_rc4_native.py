"""PDH研究代码：native输入边界、不可见信息隔离及混合动作回归。"""
import copy
import unittest

import numpy as np

from pdh_v2.game.session import GameSession
from pdh_v2.native import load_extension
from pdh_v2.native.bridge import pack_state
from pdh_v2.native.validation import compare_view
from pdh_v2.observation import acting_player
from pdh_v2.observation.encoder import decision_view_reference
from pdh_v2.policy.heuristic import choose_action


def exercise(seed, steps):
    session = GameSession(seed)
    for _ in range(steps):
        compare_view(session)
        view = decision_view_reference(session)
        action = view.actions[choose_action(session, view.actions)]
        result = session.current_hand.step(action)
        if result is not None:
            session.resolve(continue_play=True)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.module = load_extension()
        except FileNotFoundError:
            raise unittest.SkipTest("native.not_built; verify_native required separately")

    def test_initial_state(self):
        self.assertGreater(compare_view(GameSession(17))["actions"], 0)

    def test_repeated_legal_hands(self):
        exercise(710007, 900)

    def test_another_wall(self):
        exercise(910027, 900)

    def test_hidden_swap_does_not_enter_pack(self):
        session = GameSession(21)
        clone = copy.deepcopy(session)
        player = acting_player(session)
        hand = clone.current_hand
        own = hand.players[(player+1)%4].concealed
        old = own[0]
        own[0] = hand.wall.order[hand.wall.front]
        order = list(hand.wall.order)
        order[hand.wall.front] = old
        hand.wall.order = tuple(order)
        np.testing.assert_array_equal(pack_state(session, player), pack_state(clone, player))

    def test_wrong_dtype_rejected(self):
        with self.assertRaises(TypeError):
            self.module.encode_batch(np.zeros((1,1243)), np.zeros((1,7),np.int32),
                                     np.array([0,1],np.int32))

    def test_wrong_state_shape_rejected(self):
        with self.assertRaises(ValueError):
            self.module.encode_batch(np.zeros((1,1242),np.int32),
                np.zeros((1,7),np.int32), np.array([0,1],np.int32))

    def test_wrong_offsets_rejected(self):
        with self.assertRaises(ValueError):
            self.module.encode_batch(np.zeros((1,1243),np.int32),
                np.zeros((1,7),np.int32), np.array([0,2],np.int32))

    def test_action_player_mismatch_rejected(self):
        session = GameSession(17)
        packed = pack_state(session, 0)[None,:]
        with self.assertRaises(ValueError):
            self.module.encode_batch(packed,np.array([[6,1,-1,-1,-1,-1,-1]],np.int32),
                                     np.array([0,1],np.int32))
