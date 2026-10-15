"""十二张落地展示计数、第四碰自摸责任与普通吃胡付款。"""
from dataclasses import replace
import unittest

from pdh_v2.contracts import RuleError
from pdh_v2.ledger import SettlementBook
from pdh_v2.qualifications import WinMethod
from pdh_v2.responsibility import (
    TwelveGroundedLiability,
    grounded_tile_count,
    is_twelve_grounded,
    lock_kong_draw_liability,
    lock_twelve_grounded_liability,
    pong_origin_group_count,
)
from pdh_v2.settlement import cover_three_payments, twelve_grounded_payments
from pdh_v2.tiles import MeldKind
from tests.helpers import cards, meld, ordinary_win


class ResponsibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.three_pongs = (
            meld(0, MeldKind.PONG),
            meld(1, MeldKind.PONG),
            meld(2, MeldKind.PONG),
        )

    def test_three_pongs_then_added_kong_stays_nine_grounded(self) -> None:
        melds = (
            meld(0, MeldKind.PONG),
            meld(1, MeldKind.PONG),
            meld(2, MeldKind.ADDED_KONG),
        )
        self.assertEqual(grounded_tile_count(melds), 9)
        self.assertEqual(pong_origin_group_count(melds), 3)
        self.assertFalse(is_twelve_grounded(melds))

    def test_fourth_open_kong_is_twelve_display_but_not_fourth_pong_trigger(self) -> None:
        melds = self.three_pongs + (meld(3, MeldKind.OPEN_KONG),)
        self.assertEqual(grounded_tile_count(melds), 12)
        self.assertTrue(is_twelve_grounded(melds))
        liability = lock_twelve_grounded_liability(
            "hand-1",
            1,
            2,
            self.three_pongs,
            MeldKind.OPEN_KONG,
        )
        self.assertIsNone(liability)

    def test_closed_kong_is_not_grounded_or_responsibility_trigger(self) -> None:
        melds = self.three_pongs + (meld(3, MeldKind.CLOSED_KONG),)
        self.assertEqual(grounded_tile_count(melds), 9)
        self.assertIsNone(
            lock_twelve_grounded_liability(
                "hand-1",
                1,
                2,
                self.three_pongs,
                MeldKind.CLOSED_KONG,
            )
        )

    def test_added_kong_action_does_not_trigger_responsibility(self) -> None:
        self.assertIsNone(
            lock_twelve_grounded_liability(
                "hand-1",
                1,
                2,
                self.three_pongs,
                MeldKind.ADDED_KONG,
            )
        )

    def test_fourth_pong_locks_provider_for_current_hand(self) -> None:
        liability = lock_twelve_grounded_liability(
            "hand-1",
            claimant=1,
            provider=2,
            existing_melds=self.three_pongs,
            incoming_kind=MeldKind.PONG,
        )
        self.assertEqual(liability, TwelveGroundedLiability("hand-1", 1, 2))

    def test_not_three_pong_origin_groups_does_not_lock(self) -> None:
        melds = (
            meld(0, MeldKind.PONG),
            meld(1, MeldKind.PONG),
            meld(2, MeldKind.OPEN_KONG),
        )
        self.assertIsNone(
            lock_twelve_grounded_liability(
                "hand-1",
                1,
                2,
                melds,
                MeldKind.PONG,
            )
        )

    def test_self_draw_is_covered_three_shares(self) -> None:
        liability = TwelveGroundedLiability("hand-1", 1, 2)
        self.assertEqual(
            twelve_grounded_payments(ordinary_win(), 3, liability),
            (0, 72, -72, 0),
        )

    def test_discard_win_ignores_grounded_cover(self) -> None:
        liability = TwelveGroundedLiability("hand-1", 1, 2)
        win = replace(
            ordinary_win(),
            method=WinMethod.DISCARD_WIN,
            discarder=3,
        )
        self.assertEqual(
            twelve_grounded_payments(win, 3, liability),
            (0, 24, 0, -24),
        )

    def test_liability_does_not_cross_hand(self) -> None:
        liability = TwelveGroundedLiability("hand-1", 1, 2)
        with self.assertRaises(RuleError):
            twelve_grounded_payments(
                replace(ordinary_win(), hand_id="hand-2"),
                3,
                liability,
            )

    def test_cover_three_payment_pays_three_shares(self) -> None:
        self.assertEqual(
            cover_three_payments(1, 2, 1, 6, 3, 0),
            (0, 72, -72, 0),
        )


    def test_only_direct_open_kong_locks_kong_draw_liability(self) -> None:
        direct = lock_kong_draw_liability(
            "hand-1", claimant=1, provider=2, kong_kind=MeldKind.OPEN_KONG
        )
        self.assertEqual((direct.claimant, direct.responsible), (1, 2))
        self.assertIsNone(
            lock_kong_draw_liability(
                "hand-1", claimant=1, provider=None, kong_kind=MeldKind.CLOSED_KONG
            )
        )
        self.assertIsNone(
            lock_kong_draw_liability(
                "hand-1", claimant=1, provider=None, kong_kind=MeldKind.ADDED_KONG
            )
        )

    def test_claimant_cannot_be_provider(self) -> None:
        with self.assertRaises(RuleError):
            lock_twelve_grounded_liability(
                "hand-1",
                1,
                1,
                self.three_pongs,
                MeldKind.PONG,
            )


class ResponsibilityLedgerTests(unittest.TestCase):
    """第四碰责任进入统一结算账本，恢复前后不重复付款。"""

    def test_self_draw_liability_is_paid_by_provider(self) -> None:
        book = SettlementBook()
        liability = TwelveGroundedLiability("hand-1", 1, 2)
        book.begin_twelve_grounded(ordinary_win(), 6, liability)
        book.bind("settlement-1", "wall-1", cards([0, 2, 3, 4, 6, 7]))
        self.assertEqual(book.pay("settlement-1"), (0, 18, -18, 0))

    def test_discard_win_stays_single_payer_in_ledger(self) -> None:
        book = SettlementBook()
        liability = TwelveGroundedLiability("hand-1", 1, 2)
        win = replace(
            ordinary_win(),
            method=WinMethod.DISCARD_WIN,
            discarder=3,
        )
        book.begin_twelve_grounded(win, 6, liability)
        book.bind("settlement-1", "wall-1", cards([0, 2, 3, 4, 6, 7]))
        self.assertEqual(book.pay("settlement-1"), (0, 6, 0, -6))

    def test_begin_after_paid_is_idempotent(self) -> None:
        book = SettlementBook()
        liability = TwelveGroundedLiability("hand-1", 1, 2)
        win = ordinary_win()
        book.begin_twelve_grounded(win, 6, liability)
        book.bind("settlement-1", "wall-1", cards([0, 2, 3, 4, 6, 7]))
        receipt = book.pay("settlement-1")
        restored = book.begin_twelve_grounded(win, 6, liability)
        self.assertEqual(restored.receipt, receipt)
        self.assertEqual(len(book.events), 3)

    def test_begin_rejects_liability_from_another_hand(self) -> None:
        book = SettlementBook()
        with self.assertRaises(RuleError):
            book.begin_twelve_grounded(
                ordinary_win(),
                6,
                TwelveGroundedLiability("hand-other", 1, 2),
            )
