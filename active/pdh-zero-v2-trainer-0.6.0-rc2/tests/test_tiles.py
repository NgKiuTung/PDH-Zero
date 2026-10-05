"""实体唯一性、张数和输入边界测试。"""
import unittest
from typing import Any, cast
from pdh_v2.contracts import RuleError
from pdh_v2.tiles import Meld, MeldKind, WinningHand, counts_of, tile_type, validate_tiles
from tests.helpers import cards, meld


class TilesTests(unittest.TestCase):
    """每种四副本、无吃牌声明；测试不认为结构正确就必然能胡。"""

    def test_complete_deck_has_four_per_kind(self) -> None:
        self.assertEqual(counts_of(range(136)), (4,) * 34)

    def test_last_tile_type(self) -> None:
        self.assertEqual(tile_type(135), 33)

    def test_duplicate_entity_rejected(self) -> None:
        with self.assertRaisesRegex(RuleError, "重复"):
            validate_tiles([1, 1])

    def test_boolean_entity_rejected(self) -> None:
        with self.assertRaises(RuleError):
            tile_type(True)

    def test_entity_out_of_range_rejected(self) -> None:
        with self.assertRaises(RuleError):
            tile_type(136)

    def test_meld_cannot_be_sequence(self) -> None:
        with self.assertRaises(RuleError):
            Meld(MeldKind.PONG, cards([0, 1, 2]))

    def test_kong_needs_four(self) -> None:
        with self.assertRaises(RuleError):
            Meld(MeldKind.CLOSED_KONG, cards([0, 0, 0]))

    def test_hand_needs_declared_kong_extra_physical_tile(self) -> None:
        hand = WinningHand(cards([9]*3+[18]*3+[27]*3+[31]*2), (meld(0, MeldKind.CLOSED_KONG),))
        self.assertEqual(len(hand.concealed), 11)

    def test_nonwinning_size_rejected(self) -> None:
        with self.assertRaises(RuleError):
            WinningHand(tuple(range(13)))

    def test_entity_shared_by_hand_and_meld_rejected(self) -> None:
        with self.assertRaises(RuleError):
            WinningHand(tuple(range(11)), (meld(0, MeldKind.PONG),))

    def test_wrong_declared_meld_type_rejected(self) -> None:
        with self.assertRaises(RuleError):
            Meld(cast(Any, "PONG"), (0, 1, 2))
