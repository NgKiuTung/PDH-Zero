"""标准胡形和七对等级的参考分解；不是完整房规胡牌或计分裁决器。"""
from dataclasses import dataclass
from functools import lru_cache

from pdh_v2.config import RULES
from pdh_v2.contracts import RuleError
from pdh_v2.tiles import MeldKind, WinningHand, counts_of, tile_type


@dataclass(frozen=True, slots=True)
class Group:
    """未副露面子；sequence中的kind是顺子最小牌种，triplet中是刻子牌种。"""

    category: str
    kind: int


@dataclass(frozen=True, slots=True)
class StandardShape:
    """一个完整四面子一将解释；多个解释分别返回，不擅自混合评分。"""

    pair: int
    concealed_groups: tuple[Group, ...]


def _remove(counts: tuple[int, ...], positions: tuple[int, ...]) -> tuple[int, ...]:
    updated = list(counts)
    for position in positions:
        updated[position] -= 1
    return tuple(updated)


@lru_cache(maxsize=8192)
def _groups(counts: tuple[int, ...]) -> tuple[tuple[Group, ...], ...]:
    first = next((i for i, count in enumerate(counts) if count), None)
    if first is None:
        return ((),)
    results = []
    if counts[first] >= 3:
        remainder = _remove(counts, (first, first, first))
        results.extend((Group("triplet", first),) + suffix for suffix in _groups(remainder))
    # NOTE: 序数牌每9种一门，风箭不能组成顺子；同门末两种不能起顺。
    if first < 27 and first % 9 <= 6 and counts[first + 1] and counts[first + 2]:
        remainder = _remove(counts, (first, first + 1, first + 2))
        results.extend((Group("sequence", first),) + suffix for suffix in _groups(remainder))
    return tuple(results)


def standard_shapes(hand: WinningHand) -> tuple[StandardShape, ...]:
    """枚举完整标准结构；返回空表示无该结构，非空不代表有基础大胡。"""
    counts = counts_of(hand.concealed)
    results = []
    for pair, count in enumerate(counts):
        if count < 2:
            continue
        for groups in _groups(_remove(counts, (pair, pair))):
            if len(groups) == 4 - len(hand.melds):
                results.append(StandardShape(pair, groups))
    return tuple(results)


def seven_pairs_multiplier(hand: WinningHand) -> int | None:
    """仅判七对家族结构和4/8/16/32档；无副露，未声明四同牌计两对。"""
    if hand.melds:
        return None
    counts = counts_of(hand.concealed)
    if any(count % 2 for count in counts):
        return None
    quads = counts.count(4)
    return int(RULES["seven_pairs_multipliers"][quads])


def four_concealed_multipliers(hand: WinningHand, winning_tile_id: int) -> tuple[int, ...]:
    """按实际胡牌张补刻/补将返回10/13候选；允许接炮，暗杠不排除。"""
    if winning_tile_id not in hand.concealed:
        raise RuleError("shape.winning_tile")
    if any(meld.kind is not MeldKind.CLOSED_KONG for meld in hand.melds):
        return ()
    winning_kind = tile_type(winning_tile_id)
    values: set[int] = set()
    for shape in standard_shapes(hand):
        if all(group.category == "triplet" for group in shape.concealed_groups):
            wait = "single_wait" if winning_kind == shape.pair else "double_pair"
            values.add(int(RULES["four_concealed"][wait]))
    # NOTE: 不依据未见有效牌种数改成单骑，也不在此替Q6选择多解释最高计分。
    return tuple(sorted(values))
