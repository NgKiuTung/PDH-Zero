"""标准结构与家族等级测试；不替代未封板的完整牌型计分。"""
import unittest
from pdh_v2.contracts import RuleError
from pdh_v2.shapes import four_concealed_multipliers, seven_pairs_multiplier, standard_shapes
from pdh_v2.tiles import MeldKind, WinningHand
from tests.helpers import cards, meld


class ShapesTests(unittest.TestCase):
    """测试完整物理结构和胡牌张作用，不按可获得牌种数量重分档。"""

    def test_standard_sequence_structure_only(self) -> None:
        hand = WinningHand(cards([0,1,2, 3,4,5, 9,10,11, 18,19,20, 27,27]))
        self.assertEqual(len(standard_shapes(hand)), 1)
        self.assertEqual(four_concealed_multipliers(hand, 27*4+1), ())

    def test_invalid_structure_returns_empty(self) -> None:
        self.assertEqual(standard_shapes(WinningHand(tuple(range(0, 56, 4)))), ())

    def test_sequence_cannot_cross_suit(self) -> None:
        hand = WinningHand(cards([7,8,9, 12,12,12, 21,21,21, 30,30,30, 33,33]))
        self.assertEqual(standard_shapes(hand), ())

    def test_honors_cannot_make_sequence(self) -> None:
        hand = WinningHand(cards([27,28,29, 0,0,0, 9,9,9, 18,18,18, 33,33]))
        self.assertEqual(standard_shapes(hand), ())

    def test_multiple_complete_explanations_returned(self) -> None:
        hand = WinningHand(cards([0]*3+[1]*3+[2]*3+[3]*3+[4]*2))
        self.assertGreater(len(standard_shapes(hand)), 1)

    def test_seven_pairs_base(self) -> None:
        hand = WinningHand(cards([0,0, 8,8, 27,27, 28,28, 29,29, 31,31, 32,32]))
        self.assertEqual(seven_pairs_multiplier(hand), 4)

    def test_seven_pairs_one_quad(self) -> None:
        hand = WinningHand(cards([0]*4+[8]*2+[9]*2+[17]*2+[18]*2+[26]*2))
        self.assertEqual(seven_pairs_multiplier(hand), 8)

    def test_seven_pairs_two_quads(self) -> None:
        hand = WinningHand(cards([0]*4+[8]*4+[9]*2+[17]*2+[18]*2))
        self.assertEqual(seven_pairs_multiplier(hand), 16)

    def test_seven_pairs_three_quads(self) -> None:
        hand = WinningHand(cards([0]*4+[8]*4+[9]*4+[17]*2))
        self.assertEqual(seven_pairs_multiplier(hand), 32)

    def test_seven_pairs_rejects_declared_kong(self) -> None:
        hand = WinningHand(cards([9]*3+[18]*3+[27]*3+[31]*2), (meld(0, MeldKind.CLOSED_KONG),))
        self.assertIsNone(seven_pairs_multiplier(hand))

    def test_seven_pairs_rejects_odd_counts(self) -> None:
        hand = WinningHand(cards([0]*3+[9]*3+[18]*3+[27]*3+[31]*2))
        self.assertIsNone(seven_pairs_multiplier(hand))

    def test_four_concealed_double_pair_ten(self) -> None:
        hand = WinningHand(cards([0]*3+[11]*3+[24]*3+[16]*3+[27]*2))
        self.assertEqual(four_concealed_multipliers(hand, 16*4+2), (10,))

    def test_four_concealed_single_wait_thirteen(self) -> None:
        hand = WinningHand(cards([0]*3+[11]*3+[24]*3+[16]*3+[27]*2))
        self.assertEqual(four_concealed_multipliers(hand, 27*4+1), (13,))

    def test_four_concealed_accepts_closed_kong(self) -> None:
        hand = WinningHand(cards([11]*3+[24]*3+[16]*3+[27]*2), (meld(0, MeldKind.CLOSED_KONG),))
        self.assertEqual(four_concealed_multipliers(hand, 16*4+2), (10,))

    def test_four_concealed_rejects_pong(self) -> None:
        hand = WinningHand(cards([9]*3+[18]*3+[27]*3+[31]*2), (meld(0, MeldKind.PONG),))
        self.assertEqual(four_concealed_multipliers(hand, 31*4+1), ())

    def test_four_closed_kongs_single_wait_structure(self) -> None:
        declared = (
            meld(0, MeldKind.CLOSED_KONG),
            meld(9, MeldKind.CLOSED_KONG),
            meld(18, MeldKind.CLOSED_KONG),
            meld(27, MeldKind.CLOSED_KONG),
        )
        hand = WinningHand(cards([31, 31]), declared)
        self.assertEqual(four_concealed_multipliers(hand, 31*4+1), (13,))

    def test_winning_tile_must_be_in_hand(self) -> None:
        hand = WinningHand(cards([0]*3+[9]*3+[18]*3+[27]*3+[31]*2))
        with self.assertRaises(RuleError):
            four_concealed_multipliers(hand, 135)
