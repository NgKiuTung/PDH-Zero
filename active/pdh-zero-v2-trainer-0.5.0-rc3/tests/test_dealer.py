"""连庄增量、保留、归零与相对编号。"""
import unittest
from pdh_v2.contracts import RuleError
from pdh_v2.dealer import DealerState, horse_requirement


class DealerTests(unittest.TestCase):
    """已裁定单赢家事件；不预设多赢家下庄选择。"""

    def test_initial_dealer_first_win_is_zero(self) -> None:
        state = DealerState().after_win(0)
        self.assertEqual(state, DealerState(0, True, 0))
        self.assertEqual(horse_requirement(state.continuation), 6)

    def test_initial_dealer_after_draws_still_first_win(self) -> None:
        state = DealerState().after_draw().after_draw().after_win(0)
        self.assertEqual(state.continuation, 0)

    def test_win_win_draw_win_two_continuations(self) -> None:
        state = DealerState().after_win(0).after_win(0).after_draw().after_win(0)
        self.assertEqual(state.continuation, 2)
        self.assertEqual(horse_requirement(state.continuation), 10)

    def test_repeated_draws_do_not_reset(self) -> None:
        state = DealerState(2, True, 3).after_draw().after_draw()
        self.assertEqual(state, DealerState(2, True, 3))

    def test_new_winner_resets_old_continuation(self) -> None:
        state = DealerState(0, True, 2).after_win(1)
        self.assertEqual(state, DealerState(1, True, 0))

    def test_new_dealer_then_wins_increments(self) -> None:
        state = DealerState(0, True, 2).after_win(1).after_win(1)
        self.assertEqual(horse_requirement(state.continuation), 8)

    def test_relative_position_uses_current_dealer(self) -> None:
        self.assertEqual(DealerState(1).relative_seat(0), 3)

    def test_invalid_initial_continuation(self) -> None:
        with self.assertRaises(RuleError):
            DealerState(0, False, 1)

    def test_winner_boolean_rejected(self) -> None:
        with self.assertRaises(RuleError):
            DealerState().after_win(True)

    def test_negative_continuation_rejected(self) -> None:
        with self.assertRaises(RuleError):
            horse_requirement(-1)

    def test_first_four_horse_requirements(self) -> None:
        self.assertEqual(tuple(map(horse_requirement, (0,1,2,3))), (6,8,10,12))
