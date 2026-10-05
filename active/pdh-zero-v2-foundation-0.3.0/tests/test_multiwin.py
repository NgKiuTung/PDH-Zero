"""一炮多响共用马牌、旧墙不足专门开马、各位置中马及下一庄家。"""
import unittest

from pdh_v2.contracts import RuleError
from pdh_v2.dealer import DealerState, horse_requirement
from pdh_v2.horses import (
    HorseScenario,
    HorseSource,
    plan_multiwin_horses,
    select_shared_horses,
)
from pdh_v2.ledger import SettlementBook
from pdh_v2.settlement import MultiDiscardWin, MultiWinner, multi_discard_payments
from tests.helpers import cards, multi_win


class MultiWinTests(unittest.TestCase):
    def test_shared_horses_count_each_winner_by_own_position(self) -> None:
        plan = plan_multiwin_horses(0, 6)
        self.assertIs(plan.source, HorseSource.OLD_WALL)
        selection = select_shared_horses(
            plan,
            "wall-1",
            cards([0, 1, 9, 10, 27, 28]),
        )
        self.assertEqual(selection.hit_count_for(0), 3)
        self.assertEqual(selection.hit_count_for(1), 3)
        self.assertEqual(len(selection.cards), 6)

    def test_short_old_wall_uses_separate_deck_not_next_hand(self) -> None:
        plan = plan_multiwin_horses(0, 5)
        self.assertIs(plan.scenario, HorseScenario.MULTI_WIN)
        self.assertIs(plan.source, HorseSource.SEPARATE_DECK)
        self.assertTrue(plan.reshuffle_after)

    def test_short_old_wall_shared_selection_requires_full_deck(self) -> None:
        plan = plan_multiwin_horses(0, 5)
        with self.assertRaises(RuleError):
            select_shared_horses(plan, "new-deck", range(14))
        selection = select_shared_horses(plan, "new-deck", range(136))
        self.assertEqual(selection.cards, tuple(range(6)))

    def test_old_wall_still_wins_when_it_has_enough(self) -> None:
        plan = plan_multiwin_horses(5, 20)
        self.assertIs(plan.source, HorseSource.OLD_WALL)
        self.assertFalse(plan.reshuffle_after)

    def test_multi_discard_payment_uses_each_winner_hits(self) -> None:
        win = multi_win()
        receipt = multi_discard_payments(win, {0: 2, 1: 1})
        self.assertEqual(receipt, (18, 6, 0, -24))
        self.assertEqual(sum(receipt), 0)

    def test_multi_ledger_separate_deck_is_idempotent(self) -> None:
        book = SettlementBook()
        record = book.begin_multi(multi_win(), old_available=5)
        self.assertIs(record.plan.source, HorseSource.SEPARATE_DECK)
        book.bind_multi("multi-1", "horse-deck-1", range(136))
        first = book.pay_multi("multi-1")
        self.assertEqual(book.pay_multi("multi-1"), first)
        self.assertEqual(book.balances, first)
        self.assertEqual(len(book.events), 3)

    def test_changed_multi_source_rejected_after_bind(self) -> None:
        book = SettlementBook()
        book.begin_multi(multi_win(), old_available=5)
        book.bind_multi("multi-1", "horse-deck-1", range(136))
        changed = tuple(range(135)) + (135,)
        with self.assertRaises(RuleError):
            book.bind_multi("multi-1", "horse-deck-2", changed)

    def test_dealer_winner_keeps_dealer_and_continues(self) -> None:
        state = DealerState(0, True, 2)
        next_state = state.after_multi_win((0, 2))
        self.assertEqual(next_state, DealerState(0, True, 3))
        self.assertEqual(horse_requirement(next_state.continuation), 12)

    def test_lowest_relative_winner_becomes_new_dealer(self) -> None:
        state = DealerState(0, True, 2)
        next_state = state.after_multi_win((3, 1))
        self.assertEqual(next_state, DealerState(1, True, 0))

    def test_larger_co_winner_does_not_get_continuation_next_hand(self) -> None:
        state = DealerState(0, True, 2).after_multi_win((1, 3))
        self.assertEqual(state, DealerState(1, True, 0))
        self.assertEqual(state.after_win(3), DealerState(3, True, 0))

    def test_invalid_shared_plan_rejected(self) -> None:
        ordinary_plan = __import__(
            "pdh_v2.horses",
            fromlist=["plan_horses"],
        ).plan_horses(0, 6)
        with self.assertRaises(RuleError):
            select_shared_horses(ordinary_plan, "wall", range(6))

    def test_non_dealer_multi_winners_cannot_inherit_old_continuation(self) -> None:
        with self.assertRaises(RuleError):
            MultiDiscardWin(
                "bad-continuation",
                "hand",
                previous_dealer=0,
                continuation=2,
                base_units=1,
                discarder=3,
                winners=(MultiWinner(1, 3), MultiWinner(2, 3)),
            )

    def test_multi_win_requires_two_winners(self) -> None:
        with self.assertRaises(RuleError):
            MultiDiscardWin(
                "bad",
                "hand",
                0,
                0,
                1,
                3,
                (MultiWinner(0, 3),),
            )
