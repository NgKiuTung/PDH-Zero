"""普通付款、待结算、冻结旧位置与公开投影测试。"""
from dataclasses import replace
import unittest
from typing import Any, cast

from pdh_v2.contracts import RuleError
from pdh_v2.horses import HorseSource
from pdh_v2.ledger import SettlementBook
from pdh_v2.qualifications import WinMethod, route_qualifications
from pdh_v2.settlement import ordinary_payments
from tests.helpers import next_hand_cards, ordinary_win


class PaymentsTests(unittest.TestCase):
    """原始整数积分；不假定未批准的封顶或承包。"""

    def test_self_draw_three_payments(self) -> None:
        self.assertEqual(ordinary_payments(ordinary_win(),3), (-24,72,-24,-24))

    def test_discard_one_payment(self) -> None:
        win = replace(ordinary_win(),method=WinMethod.DISCARD_WIN,discarder=2)
        self.assertEqual(ordinary_payments(win,3),(0,24,-24,0))

    def test_zero_horses_still_pays(self) -> None:
        self.assertEqual(ordinary_payments(ordinary_win(),0),(-6,18,-6,-6))

    def test_net_sum_zero(self) -> None:
        self.assertEqual(sum(ordinary_payments(ordinary_win(),6)),0)

    def test_idle_winner_cannot_inherit_continuation(self) -> None:
        with self.assertRaises(RuleError):
            replace(ordinary_win(),continuation=2)

    def test_self_draw_cannot_have_discarder(self) -> None:
        with self.assertRaises(RuleError):
            replace(ordinary_win(),discarder=2)

    def test_discarder_cannot_be_winner(self) -> None:
        with self.assertRaises(RuleError):
            replace(ordinary_win(),method=WinMethod.DISCARD_WIN,discarder=1)

    def test_discard_requires_discarder(self) -> None:
        with self.assertRaises(RuleError):
            replace(ordinary_win(),method=WinMethod.DISCARD_WIN)

    def test_rob_kong_not_assigned_ordinary_payer(self) -> None:
        with self.assertRaises(RuleError):
            replace(ordinary_win(),method=WinMethod.ROB_KONG)

    def test_more_hits_than_candidates_rejected(self) -> None:
        with self.assertRaises(RuleError):
            ordinary_payments(ordinary_win(),7)

    def test_noninteger_base_rejected(self) -> None:
        with self.assertRaises(RuleError):
            replace(ordinary_win(),base_units=cast(Any, 1.5))


class LedgerTests(unittest.TestCase):
    """单写入者权威事件账本，重复请求不产生新收入。"""

    def setUp(self) -> None:
        self.book = SettlementBook()
        self.win = ordinary_win()
        self.book.begin(self.win,3)

    def test_unbound_does_not_pay_zero_horses(self) -> None:
        self.assertEqual(self.book.get(self.win.settlement_id).status,"WAITING_HORSES")
        with self.assertRaises(RuleError):
            self.book.pay(self.win.settlement_id)
        self.assertEqual(self.book.balances,(0,0,0,0))

    def test_next_hand_binds_using_previous_position(self) -> None:
        record = self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        self.assertEqual(record.status,"READY")
        self.assertIsNotNone(record.selection)
        self.assertEqual(self.book.pay(self.win.settlement_id),(-24,72,-24,-24))

    def test_repeat_pay_is_exactly_once(self) -> None:
        self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        first = self.book.pay(self.win.settlement_id)
        self.assertEqual(self.book.pay(self.win.settlement_id),first)
        self.assertEqual(self.book.balances,first)
        self.assertEqual(len(self.book.events),3)

    def test_same_begin_after_paid_is_idempotent(self) -> None:
        self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        self.book.pay(self.win.settlement_id)
        record = self.book.begin(self.win,3)
        self.assertEqual(record.status,"PAID")
        self.assertEqual(len(self.book.events),3)

    def test_changed_win_with_same_id_rejected(self) -> None:
        with self.assertRaises(RuleError):
            self.book.begin(replace(self.win,multiplier=9),3)

    def test_changed_source_rejected_after_bind(self) -> None:
        self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        with self.assertRaises(RuleError):
            self.book.bind(self.win.settlement_id,"hand-3",next_hand_cards())

    def test_changed_unused_suffix_rejected(self) -> None:
        original = next_hand_cards()
        self.book.bind(self.win.settlement_id,"hand-2",original)
        changed = original[:-2] + (original[-1],original[-2])
        with self.assertRaises(RuleError):
            self.book.bind(self.win.settlement_id,"hand-2",changed)

    def test_repeat_bind_after_pay_keeps_paid(self) -> None:
        self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        self.book.pay(self.win.settlement_id)
        record = self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        self.assertEqual(record.status,"PAID")
        self.assertEqual(len(self.book.events),3)

    def test_public_projection_only_contains_hits(self) -> None:
        self.book.bind(self.win.settlement_id,"hand-2",next_hand_cards())
        self.assertEqual(self.book.next_hand_reveal(self.win.settlement_id), {
            "event":"REVEAL_RETAINED_HORSES", "source_hand_id":"hand-2",
            "settles_hand_id":"hand-1", "holder":1,
            "cards":[{"kind":1,"count":1},{"kind":14,"count":1},{"kind":28,"count":1}],
        })

    def test_hidden_nonhit_change_does_not_change_public_projection(self) -> None:
        original = next_hand_cards()
        self.book.bind(self.win.settlement_id,"hand-2",original)
        other = SettlementBook()
        other.begin(self.win,3)
        changed = original[:3] + (132,) + original[4:]
        other.bind(self.win.settlement_id,"hand-2",changed)
        self.assertEqual(other.next_hand_reveal(self.win.settlement_id),
                         self.book.next_hand_reveal(self.win.settlement_id))

    def test_reveal_does_not_declare_meld_or_remove_cards(self) -> None:
        original = next_hand_cards()
        self.book.bind(self.win.settlement_id,"hand-2",original)
        self.book.next_hand_reveal(self.win.settlement_id)
        self.assertEqual(len(original),14)
        self.assertEqual(self.book.events[1]["ordered_source"],list(original))
        self.assertTrue(route_qualifications(()).seven_pairs_history_ok)

    def test_no_reveal_before_binding(self) -> None:
        self.assertIsNone(self.book.next_hand_reveal(self.win.settlement_id))

    def test_old_wall_does_not_become_next_hand_known_tiles(self) -> None:
        book = SettlementBook()
        book.begin(self.win,6)
        book.bind(self.win.settlement_id,"old-wall",next_hand_cards()[:6])
        self.assertIsNone(book.next_hand_reveal(self.win.settlement_id))

    def test_special_deck_not_next_hand_known_tiles(self) -> None:
        win = replace(self.win,winner=0,continuation=5)
        book = SettlementBook()
        record = book.begin(win,15)
        book.bind(win.settlement_id,"special-1",range(136))
        book.pay(win.settlement_id)
        self.assertEqual(record.plan.source,HorseSource.SEPARATE_DECK)
        self.assertIsNone(book.next_hand_reveal(win.settlement_id))
        self.assertEqual(record.win.continuation,5)

    def test_outside_modification_of_events_cannot_change_book(self) -> None:
        external = self.book.events
        external[0]["win"]["winner"] = 3
        self.assertEqual(self.book.get(self.win.settlement_id).win.winner,1)

    def test_missing_record_rejected(self) -> None:
        with self.assertRaises(RuleError):
            self.book.get("missing")


class UncappedAndDrawTests(unittest.TestCase):
    """无封顶与流局零点数的已确认基础结算。"""

    def test_multiplier_is_not_capped_at_twenty(self) -> None:
        win = replace(ordinary_win(), multiplier=10**30)
        receipt = ordinary_payments(win, 0)
        self.assertEqual(receipt, (-(10**30), 3 * 10**30, -(10**30), -(10**30)))

    def test_draw_payments_are_zero(self) -> None:
        from pdh_v2.settlement import draw_payments
        self.assertEqual(draw_payments(), (0, 0, 0, 0))

    def test_draw_is_recorded_without_balance_change(self) -> None:
        from pdh_v2.settlement import DrawRecord
        book = SettlementBook()
        draw = DrawRecord("draw-1", "hand-draw", 0, 2)
        self.assertEqual(book.record_draw(draw), draw)
        self.assertEqual(book.balances, (0, 0, 0, 0))
        self.assertEqual(book.get_draw("draw-1"), draw)
        self.assertEqual(book.record_draw(draw), draw)
        self.assertEqual(len(book.events), 1)
