"""取马数、三个来源分支与按上一局位置逐张计马。"""
import unittest
from collections import Counter
from pdh_v2.contracts import RuleError
from pdh_v2.horses import (
    HorsePlan,
    HorseScenario,
    HorseSource,
    horse_owner,
    plan_horses,
    plan_multiwin_horses,
    select_horses,
)
from tests.helpers import cards


class HorsesTests(unittest.TestCase):
    """输入来源已有实际顺序；不在测试里假定任何未决墙端。"""

    def test_mapping_covers_all_34_types(self) -> None:
        self.assertEqual(Counter(map(horse_owner, range(0,136,4))), {0:11,1:8,2:8,3:7})

    def test_east_and_red_belong_dealer(self) -> None:
        self.assertEqual((horse_owner(27*4),horse_owner(31*4)), (0,0))

    def test_south_and_green_belong_one(self) -> None:
        self.assertEqual((horse_owner(28*4),horse_owner(32*4)), (1,1))

    def test_west_white_and_north(self) -> None:
        self.assertEqual((horse_owner(29*4),horse_owner(33*4),horse_owner(30*4)), (2,2,3))

    def test_old_wall_exactly_six(self) -> None:
        self.assertEqual(plan_horses(0,6).source, HorseSource.OLD_WALL)

    def test_ten_needed_seven_left_goes_next_hand(self) -> None:
        plan = plan_horses(2,7)
        self.assertEqual((plan.source,plan.required), (HorseSource.NEXT_HAND,10))

    def test_ten_needed_three_left_still_ten(self) -> None:
        self.assertEqual(plan_horses(2,3).required, 10)

    def test_fourteen_is_not_over_hand_limit(self) -> None:
        self.assertEqual(plan_horses(4,13).source, HorseSource.NEXT_HAND)

    def test_fourteen_wall_exactly_enough(self) -> None:
        self.assertEqual(plan_horses(4,14).source, HorseSource.OLD_WALL)

    def test_sixteen_wall_twenty_prefers_old_wall(self) -> None:
        self.assertEqual(plan_horses(5,20).source, HorseSource.OLD_WALL)

    def test_sixteen_wall_exactly_enough(self) -> None:
        self.assertEqual(plan_horses(5,16).source, HorseSource.OLD_WALL)

    def test_sixteen_wall_fifteen_separate_deck(self) -> None:
        self.assertEqual(plan_horses(5,15).source, HorseSource.SEPARATE_DECK)

    def test_largest_single_deck_capacity(self) -> None:
        selection = select_horses(plan_horses(65,0), "special-65", range(136), 0)
        self.assertEqual(len(selection.cards),136)
        self.assertEqual(selection.hit_count,44)

    def test_over_capacity_uses_separate_deck_cycles(self) -> None:
        plan = plan_horses(66, 0)
        self.assertEqual((plan.source, plan.required), (HorseSource.SEPARATE_DECK, 138))
        source = tuple(range(136)) + (0, 1)
        selection = select_horses(plan, "special-cycle", source, 0)
        self.assertEqual(len(selection.cards), 138)

    def test_next_hand_selects_raw_prefix_not_sorted(self) -> None:
        original = tuple(reversed(range(14)))
        selection = select_horses(plan_horses(2,7), "next-1", original, 1)
        self.assertEqual(selection.cards,original[:10])
        self.assertEqual(original,tuple(reversed(range(14))))

    def test_selection_counts_repeated_matching_entities(self) -> None:
        original = cards([1,1,14,28,3,27,8,9,10,11,12,13,15,16])
        selection = select_horses(plan_horses(0,3), "next-2", original, 1)
        self.assertEqual(selection.hit_count,4)

    def test_first_six_example_uses_old_seat(self) -> None:
        original = cards([1,14,28,21,27,8,0,2,3,4,5,6,7,9])
        selection = select_horses(plan_horses(0,3), "next-3", original, 1)
        self.assertEqual(selection.hit_count,3)

    def test_source_must_be_full_next_dealer_hand(self) -> None:
        with self.assertRaises(RuleError):
            select_horses(plan_horses(2,7), "next-4", range(10), 1)

    def test_separate_source_requires_enough_unique_cards_per_cycle(self) -> None:
        plan = plan_horses(5, 15)
        with self.assertRaises(RuleError):
            select_horses(plan, "special-4", range(15), 0)
        selection = select_horses(plan, "special-4", range(16), 0)
        self.assertEqual(len(selection.cards), 16)

    def test_corrupt_plan_rejected(self) -> None:
        with self.assertRaises(RuleError):
            HorsePlan(5,20,16,HorseSource.SEPARATE_DECK)

    def test_empty_source_id_rejected(self) -> None:
        with self.assertRaises(RuleError):
            select_horses(plan_horses(0,6), "", range(6), 0)

    def test_invalid_previous_position_rejected(self) -> None:
        with self.assertRaises(RuleError):
            select_horses(plan_horses(0,6), "old-1", range(6), 4)

    def test_negative_wall_rejected(self) -> None:
        with self.assertRaises(RuleError):
            plan_horses(0,-1)


class MultiHorsePlanningTests(unittest.TestCase):
    """一炮多响旧墙不足时不借任何赢家下局手牌。"""

    def test_multi_short_wall_uses_separate_deck_even_for_six(self) -> None:
        plan = plan_multiwin_horses(0, 5)
        self.assertEqual(
            (plan.scenario, plan.source, plan.required, plan.reshuffle_after),
            (HorseScenario.MULTI_WIN, HorseSource.SEPARATE_DECK, 6, True),
        )

    def test_multi_enough_wall_keeps_old_wall(self) -> None:
        plan = plan_multiwin_horses(0, 6)
        self.assertEqual(
            (plan.source, plan.reshuffle_after),
            (HorseSource.OLD_WALL, False),
        )
