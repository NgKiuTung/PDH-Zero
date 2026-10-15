"""成功副露才建立供牌链；普通过牌不伪造贡献。"""
import unittest

from pdh_v2.game.actions import ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.session import GameSession
from pdh_v2.native import load_extension
from pdh_v2.native.validation import compare_view
from pdh_v2.observation.encoder import decision_view_reference
from pdh_v2.replay.supply import before_meld_transition, capture_transition
from pdh_v2.tiles import tile_type
from tests.test_game import allocate, wall_for_hands


def response_session():
    hands = allocate([
        [31] + [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
        [31, 31, 31] + [16, 17, 19, 20, 21, 22, 23, 24, 25, 28],
        [18, 26, 27, 30, 32, 33, 8, 9, 0, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    ])
    session = GameSession(31)
    session.current_hand = MahjongHand("supply", 0, wall_for_hands(0, hands))
    action = next(a for a in session.current_hand.legal_actions(0)
                  if a.kind is ActionKind.DISCARD and tile_type(a.tiles[0]) == 31)
    session.current_hand.step(action)
    return session


def take(session, kind, chain):
    view = decision_view_reference(session)
    action = next(a for a in view.actions if a.kind is kind)
    before = before_meld_transition(session, action)
    session.current_hand.step(action)
    capture_transition(session, before, chain)


class SupplyTests(unittest.TestCase):
    def test_open_kong_feature_differential(self):
        try:
            load_extension()
        except FileNotFoundError:
            self.skipTest("native.not_built")
        session = response_session()
        self.assertIn(ActionKind.OPEN_KONG,
                      [a.kind for a in decision_view_reference(session).actions])
        self.assertGreater(compare_view(session)["actions"], 0)

    def test_completed_pong_supply(self):
        session = response_session()
        chain = []
        take(session, ActionKind.PONG, chain)
        self.assertEqual(chain, [])
        take(session, ActionKind.PASS, chain)
        take(session, ActionKind.PASS, chain)
        self.assertEqual(len(chain), 1)
        self.assertTrue(chain[0]["external_supply"])
        self.assertEqual(chain[0]["provider"], 0)
        self.assertEqual(chain[0]["claimant"], 1)
        self.assertEqual(chain[0]["kind"], "PONG")

    def test_completed_open_kong_supply(self):
        session = response_session()
        chain = []
        take(session, ActionKind.OPEN_KONG, chain)
        take(session, ActionKind.PASS, chain)
        take(session, ActionKind.PASS, chain)
        self.assertEqual(len(chain), 1)
        self.assertTrue(chain[0]["external_supply"])
        self.assertEqual(chain[0]["kind"], "OPEN_KONG")

    def test_pass_does_not_create_supply(self):
        session = response_session()
        chain = []
        take(session, ActionKind.PASS, chain)
        take(session, ActionKind.PASS, chain)
        take(session, ActionKind.PASS, chain)
        self.assertEqual(chain, [])
