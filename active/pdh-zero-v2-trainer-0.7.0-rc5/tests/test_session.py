"""连续会话结算、虚拟散场发牌与责任支付集成测试。"""
import unittest

from pdh_v2.game.session import GameSession
from pdh_v2.game.state import HandOutcome, WinnerResult
from pdh_v2.horses import HorseSource
from pdh_v2.scoring import ScoreResult
from pdh_v2.tiles import tile_type


def score(multiplier: int) -> ScoreResult:
    """构造已由规则层裁定的最小计分结果。"""
    return ScoreResult(multiplier, (), (), "TEST", 0)


class SessionSettlementTests(unittest.TestCase):
    """验证会话只消费已裁定终局，不在集成测试重复测试胡形。"""

    def test_virtual_next_hand_finishes_final_pending_settlement(self) -> None:
        session = GameSession(seed=20260930, dealer=0)
        session.current_hand.outcome = HandOutcome(
            session.current_hand.hand_id,
            "SELF_DRAW",
            (WinnerResult(1, score(6)),),
            None,
            False,
            "FRONT",
            (0, 4, 8),
        )

        result = session.resolve(continue_play=False)

        self.assertTrue(result.virtual_next_hand)
        self.assertIs(result.horse_source, HorseSource.NEXT_HAND)
        self.assertEqual(result.next_dealer.player, 1)
        self.assertEqual(sum(result.receipt), 0)
        self.assertGreater(result.receipt[1], 0)
        virtual = [event for event in session.events if event["event"] == "HAND_DEALT"][-1]
        self.assertTrue(virtual["virtual"])

    def test_next_hand_reveal_marks_only_selected_hit_entities(self) -> None:
        session = GameSession(seed=8, dealer=0)
        session.current_hand.outcome = HandOutcome(
            session.current_hand.hand_id,
            "SELF_DRAW",
            (WinnerResult(1, score(6)),),
            None,
            False,
            "FRONT",
            (0, 4, 8),
        )

        session.resolve(continue_play=True)

        record = session.book.get(f"S:H{0:08d}")
        self.assertIsNotNone(record.selection)
        expected = set(record.selection.hits)
        player = session.current_hand.players[1]
        actual = player.revealed_retained
        selected_types = {tile_type(tile) for tile in expected}
        same_kind_unselected = {
            tile
            for tile in player.concealed
            if tile_type(tile) in selected_types and tile not in expected
        }

        self.assertTrue(same_kind_unselected)
        self.assertEqual(actual, expected)
        self.assertTrue(actual <= set(player.concealed))
        self.assertTrue(actual.isdisjoint(same_kind_unselected))
        reveal = [event for event in session.events if event["event"] == "REVEAL_RETAINED_HORSES"]
        self.assertEqual(len(reveal), 1)
        self.assertNotIn("entity_ids", reveal[0])

    def test_direct_open_kong_liability_overrides_ordinary_self_draw(self) -> None:
        session = GameSession(seed=71, dealer=0)
        session.current_hand.outcome = HandOutcome(
            session.current_hand.hand_id,
            "SELF_DRAW",
            (WinnerResult(1, score(3)),),
            None,
            False,
            "FRONT",
            tuple(range(6)),
            kong_provider=0,
        )

        result = session.resolve(continue_play=False)

        self.assertGreater(result.receipt[1], 0)
        self.assertLess(result.receipt[0], 0)
        self.assertEqual(result.receipt[2:], (0, 0))
        self.assertEqual(result.receipt[1], -result.receipt[0])

    def test_multiwin_short_wall_uses_shared_separate_deck(self) -> None:
        session = GameSession(seed=99, dealer=0)
        session.current_hand.outcome = HandOutcome(
            session.current_hand.hand_id,
            "DISCARD_WIN",
            (WinnerResult(1, score(3)), WinnerResult(2, score(6))),
            3,
            False,
            "FRONT",
            tuple(range(5)),
        )

        result = session.resolve(continue_play=False)

        self.assertIs(result.horse_source, HorseSource.SEPARATE_DECK)
        self.assertEqual(result.next_dealer.player, 1)
        self.assertGreater(result.receipt[1], 0)
        self.assertGreater(result.receipt[2], 0)
        self.assertLess(result.receipt[3], 0)
        self.assertEqual(sum(result.receipt), 0)
        event = next(row for row in session.events if row["event"] == "SEPARATE_HORSE_DECK")
        self.assertEqual(event["required"], 6)

    def test_draw_records_zero_receipt_and_keeps_dealer_state(self) -> None:
        session = GameSession(seed=7, dealer=2)
        before = session.dealer_state
        session.current_hand.outcome = HandOutcome(
            session.current_hand.hand_id,
            "DRAW",
            (),
            None,
            True,
            "FRONT",
            (),
        )

        result = session.resolve(continue_play=False)

        self.assertEqual(result.receipt, (0, 0, 0, 0))
        self.assertEqual(result.next_dealer, before)
        self.assertEqual(session.book.balances, (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
