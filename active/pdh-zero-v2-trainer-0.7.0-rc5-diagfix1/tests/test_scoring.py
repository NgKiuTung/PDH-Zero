"""完整牌型评分、去重叠加与多解释最高分选择。"""
from collections import Counter
import unittest

from pdh_v2.qualifications import WinMethod
from pdh_v2.scoring import Pattern, ScoreContext, score_hand
from pdh_v2.tiles import Meld, MeldKind, WinningHand


def build_hand(
    concealed_kinds: list[int],
    meld_specs: list[tuple[int, MeldKind]] | None = None,
) -> WinningHand:
    """在整手范围内分配唯一实体，避免测试构造掩盖牌数错误。"""
    used: Counter[int] = Counter()

    def take(kind: int, count: int) -> tuple[int, ...]:
        start = used[kind]
        if start + count > 4:
            raise ValueError(kind)
        used[kind] += count
        return tuple(kind * 4 + offset for offset in range(start, start + count))

    concealed = tuple(card for kind in concealed_kinds for card in take(kind, 1))
    melds = []
    for kind, category in meld_specs or []:
        count = 3 if category is MeldKind.PONG else 4
        melds.append(Meld(category, take(kind, count)))
    return WinningHand(concealed, tuple(melds))


def result(
    kinds: list[int],
    winning_index: int = -1,
    method: WinMethod = WinMethod.SELF_DRAW,
    wall_before: int = 20,
    melds: list[tuple[int, MeldKind]] | None = None,
    kong_draw: bool = False,
):
    hand = build_hand(kinds, melds)
    scored = score_hand(
        ScoreContext(
            hand,
            hand.concealed[winning_index],
            method,
            wall_before,
            kong_draw,
        )
    )
    return scored


class ScoringTests(unittest.TestCase):
    def test_plain_mixed_suit_closed_is_six(self) -> None:
        scored = result([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 27, 27])
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 6)
        self.assertEqual(
            set(scored.patterns),
            {Pattern.MIXED_ONE_SUIT, Pattern.CLOSED_BONUS},
        )

    def test_pure_closed_straight_is_twelve(self) -> None:
        scored = result([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 8, 8])
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 12)
        self.assertEqual(
            set(scored.patterns),
            {Pattern.PURE_ONE_SUIT, Pattern.CLOSED_BONUS, Pattern.ONE_STRAIGHT},
        )

    def test_plain_structure_without_big_pattern_cannot_win(self) -> None:
        self.assertIsNone(result([0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 27, 27]))

    def test_four_concealed_pure_does_not_repeat_pong(self) -> None:
        scored = result([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4], 11)
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 16)
        self.assertEqual(
            set(scored.patterns),
            {Pattern.FOUR_CONCEALED, Pattern.PURE_ONE_SUIT},
        )
        self.assertEqual(scored.suppressed[0].pattern, Pattern.PONG_PONG)

    def test_four_concealed_single_big_winds_all_honors_is_fifty_seven(self) -> None:
        scored = result([27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31])
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 57)
        self.assertIn(Pattern.FOUR_CONCEALED_SINGLE, scored.patterns)
        self.assertIn(Pattern.ALL_HONORS, scored.patterns)
        self.assertIn(Pattern.BIG_WINDS, scored.patterns)

    def test_seven_pairs_pure_is_ten(self) -> None:
        scored = result([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 10)
        self.assertEqual(set(scored.patterns), {Pattern.SEVEN_PAIRS, Pattern.PURE_ONE_SUIT})

    def test_pure_terminals_luxury_seven_pairs_is_twenty_eight(self) -> None:
        scored = result([0, 0, 0, 0, 8, 8, 9, 9, 17, 17, 18, 18, 26, 26])
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 28)
        self.assertEqual(
            set(scored.patterns),
            {Pattern.LUXURY_SEVEN_PAIRS, Pattern.PURE_TERMINALS},
        )

    def test_small_dragons_can_add_pong_pong(self) -> None:
        scored = result(
            [31, 31, 31, 32, 32, 32, 33, 33, 10, 10, 10],
            melds=[(1, MeldKind.PONG)],
        )
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 12)
        self.assertEqual(set(scored.patterns), {Pattern.SMALL_DRAGONS, Pattern.PONG_PONG})

    def test_big_winds_all_honors_suppresses_pong(self) -> None:
        scored = result([27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31])
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 57)
        self.assertEqual(
            set(scored.patterns),
            {Pattern.BIG_WINDS, Pattern.ALL_HONORS, Pattern.FOUR_CONCEALED_SINGLE},
        )

    def test_thirteen_sided_is_sixteen(self) -> None:
        kinds = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 0]
        scored = result(kinds)
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 16)
        self.assertEqual(scored.patterns, (Pattern.THIRTEEN_SIDED,))

    def test_ordinary_thirteen_orphans_is_thirteen(self) -> None:
        kinds = [0, 0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        scored = result(kinds)
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 13)
        self.assertEqual(scored.patterns, (Pattern.THIRTEEN_ORPHANS,))

    def test_sea_and_kong_draw_stack_on_real_self_draw(self) -> None:
        scored = result(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
            11,
            wall_before=4,
            kong_draw=True,
        )
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 22)
        self.assertIn(Pattern.SEA_BOTTOM, scored.patterns)
        self.assertIn(Pattern.KONG_DRAW, scored.patterns)

    def test_multiple_interpretations_choose_highest_complete_one(self) -> None:
        scored = result([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 10)
        self.assertTrue(scored.interpretation.startswith("SEVEN_PAIRS"))

    def test_terminal_family_is_mutually_classified(self) -> None:
        yao = result(
            [28, 28],
            melds=[
                (0, MeldKind.PONG),
                (8, MeldKind.PONG),
                (9, MeldKind.PONG),
                (27, MeldKind.PONG),
            ],
        )
        mixed = result(
            [28, 28],
            melds=[
                (0, MeldKind.PONG),
                (8, MeldKind.PONG),
                (27, MeldKind.PONG),
                (31, MeldKind.PONG),
            ],
        )
        honors = result(
            [32, 32],
            melds=[
                (27, MeldKind.PONG),
                (28, MeldKind.PONG),
                (29, MeldKind.PONG),
                (31, MeldKind.PONG),
            ],
        )

        self.assertEqual(yao.multiplier, 9)
        self.assertEqual(yao.patterns, (Pattern.TERMINALS_HONORS,))
        self.assertEqual(mixed.multiplier, 12)
        self.assertEqual(mixed.patterns, (Pattern.MIXED_ONE_SUIT_TERMINALS,))
        self.assertEqual(honors.multiplier, 20)
        self.assertEqual(honors.patterns, (Pattern.ALL_HONORS,))

    def test_small_winds_all_triplets_adds_pong_pong(self) -> None:
        scored = result(
            [1, 1, 1, 30, 30],
            melds=[
                (27, MeldKind.PONG),
                (28, MeldKind.PONG),
                (29, MeldKind.PONG),
            ],
        )
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 22)
        self.assertEqual(
            set(scored.patterns),
            {Pattern.SMALL_WINDS, Pattern.PONG_PONG, Pattern.MIXED_ONE_SUIT},
        )

    def test_thirteen_sided_discard_win_is_allowed(self) -> None:
        kinds = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 0]
        scored = result(kinds, method=WinMethod.DISCARD_WIN)
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 16)
        self.assertEqual(scored.patterns, (Pattern.THIRTEEN_SIDED,))

    def test_eighteen_arhats_four_closed_kongs_stacks_single_wait(self) -> None:
        hand = build_hand([31, 31], [(0, MeldKind.CLOSED_KONG), (1, MeldKind.CLOSED_KONG),
                                     (2, MeldKind.CLOSED_KONG), (3, MeldKind.CLOSED_KONG)])
        scored = score_hand(
            ScoreContext(hand, hand.concealed[-1], WinMethod.SELF_DRAW, 20)
        )
        self.assertIsNotNone(scored)
        self.assertEqual(scored.multiplier, 48)
        self.assertEqual(
            set(scored.patterns),
            {
                Pattern.EIGHTEEN_ARHATS,
                Pattern.FOUR_CONCEALED_SINGLE,
                Pattern.MIXED_ONE_SUIT,
            },
        )


if __name__ == "__main__":
    unittest.main()
