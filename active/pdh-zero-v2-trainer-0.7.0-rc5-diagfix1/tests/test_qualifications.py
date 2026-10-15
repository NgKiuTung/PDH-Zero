"""资格必要条件与海底摸前边界。"""
import unittest
from pdh_v2.contracts import RuleError
from pdh_v2.qualifications import WinMethod, closed_bonus, route_qualifications, sea_bottom_bonus
from pdh_v2.tiles import MeldKind
from tests.helpers import meld


class QualificationsTests(unittest.TestCase):
    """门清附加不成为独立胡牌；亮马不能伪装成碰杠。"""

    def test_no_base_no_closed_bonus(self) -> None:
        self.assertEqual(closed_bonus(None, False, ()), 0)

    def test_plain_pure_suit_has_bonus(self) -> None:
        self.assertEqual(closed_bonus("PURE_ONE_SUIT", False, ()), 3)

    def test_plain_mixed_suit_has_bonus(self) -> None:
        self.assertEqual(closed_bonus("MIXED_ONE_SUIT", False, ()), 3)

    def test_additional_big_pattern_excludes_bonus(self) -> None:
        self.assertEqual(closed_bonus("PURE_ONE_SUIT", True, ()), 0)

    def test_pong_cancels_bonus(self) -> None:
        self.assertEqual(closed_bonus("MIXED_ONE_SUIT", False, (meld(0,MeldKind.PONG),)), 0)

    def test_closed_kong_cancels_bonus(self) -> None:
        self.assertEqual(closed_bonus("PURE_ONE_SUIT", False, (meld(0,MeldKind.CLOSED_KONG),)), 0)

    def test_closed_kong_preserves_four_but_not_pairs(self) -> None:
        q = route_qualifications((meld(0,MeldKind.CLOSED_KONG),))
        self.assertTrue(q.four_concealed_history_ok)
        self.assertFalse(q.seven_pairs_history_ok)
        self.assertFalse(q.thirteen_orphans_history_ok)
        self.assertFalse(q.closed_bonus_history_ok)

    def test_open_kong_cancels_four(self) -> None:
        result = route_qualifications((meld(0, MeldKind.OPEN_KONG),))
        self.assertFalse(result.four_concealed_history_ok)

    def test_added_kong_does_not_restore_four(self) -> None:
        result = route_qualifications((meld(0, MeldKind.ADDED_KONG),))
        self.assertFalse(result.four_concealed_history_ok)

    def test_unknown_base_rejected(self) -> None:
        with self.assertRaises(RuleError):
            closed_bonus("CLOSED", False, ())

    def test_same_entity_in_two_melds_rejected(self) -> None:
        one = meld(0, MeldKind.PONG)
        with self.assertRaises(RuleError):
            route_qualifications((one, one))

    def test_sea_uses_before_not_after(self) -> None:
        self.assertEqual(sea_bottom_bonus(WinMethod.SELF_DRAW, 5), 0)

    def test_sea_at_four(self) -> None:
        self.assertEqual(sea_bottom_bonus(WinMethod.SELF_DRAW, 4), 3)

    def test_sea_at_three(self) -> None:
        self.assertEqual(sea_bottom_bonus(WinMethod.SELF_DRAW, 3), 3)

    def test_sea_at_one(self) -> None:
        self.assertEqual(sea_bottom_bonus(WinMethod.SELF_DRAW, 1), 3)

    def test_no_sea_on_discard(self) -> None:
        self.assertEqual(sea_bottom_bonus(WinMethod.DISCARD_WIN, 1), 0)

    def test_no_sea_on_rob_kong(self) -> None:
        self.assertEqual(sea_bottom_bonus(WinMethod.ROB_KONG, 1), 0)

    def test_no_draw_from_empty_wall(self) -> None:
        with self.assertRaises(RuleError):
            sea_bottom_bonus(WinMethod.SELF_DRAW, 0)
