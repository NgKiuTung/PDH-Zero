"""依据已冻结房规枚举完整胡形并选择最高计分解释。"""
from dataclasses import dataclass
from typing import Iterable

from pdh_v2.config import RULES
from pdh_v2.contracts import RuleError, integer
from pdh_v2.qualifications import WinMethod, sea_bottom_bonus
from pdh_v2.scoring.patterns import Pattern, ScoreItem, SuppressedItem
from pdh_v2.shapes import Group, StandardShape, standard_shapes
from pdh_v2.tiles import MeldKind, WinningHand, counts_of, tile_type

TERMINAL_TYPES = frozenset((0, 8, 9, 17, 18, 26))
HONOR_TYPES = frozenset(range(27, 34))
ORPHAN_TYPES = TERMINAL_TYPES | HONOR_TYPES
WIND_TYPES = frozenset(range(27, 31))
DRAGON_TYPES = frozenset(range(31, 34))


@dataclass(frozen=True, slots=True)
class ScoreContext:
    """评分所需事实；winning_tile必须是最终未副露牌中的真实实体。"""

    hand: WinningHand
    winning_tile: int
    method: WinMethod
    wall_before_draw: int
    kong_draw: bool = False

    def __post_init__(self) -> None:
        if self.winning_tile not in self.hand.concealed:
            raise RuleError("shape.winning_tile")
        if not isinstance(self.method, WinMethod):
            raise RuleError("input.type", field="win_method")
        integer(self.wall_before_draw, "wall_before_draw", 0, 136)
        if type(self.kong_draw) is not bool:
            raise RuleError("input.type", field="kong_draw")


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """单个完整解释的计分结果；selected由score_hand按倍数和固定优先级选择。"""

    multiplier: int
    items: tuple[ScoreItem, ...]
    suppressed: tuple[SuppressedItem, ...]
    interpretation: str
    priority: int

    @property
    def patterns(self) -> tuple[Pattern, ...]:
        """返回实际计分牌型，不含被包含而抑制的标签。"""
        return tuple(item.pattern for item in self.items)


def _all_types(hand: WinningHand) -> frozenset[int]:
    return frozenset(
        tile_type(card)
        for card in hand.concealed
        + tuple(card for meld in hand.melds for card in meld.tiles)
    )


def _suit_pattern(types: frozenset[int]) -> Pattern | None:
    suits = {kind // 9 for kind in types if kind < 27}
    honors = any(kind >= 27 for kind in types)
    if len(suits) != 1:
        return None
    return Pattern.MIXED_ONE_SUIT if honors else Pattern.PURE_ONE_SUIT


def _terminal_pattern(types: frozenset[int]) -> Pattern | None:
    if not types or not types <= ORPHAN_TYPES:
        return None
    has_honor = bool(types & HONOR_TYPES)
    has_terminal = bool(types & TERMINAL_TYPES)
    if has_honor and not has_terminal:
        return Pattern.ALL_HONORS
    if has_terminal and not has_honor:
        return Pattern.PURE_TERMINALS
    terminal_suits = {kind // 9 for kind in types & TERMINAL_TYPES}
    if len(terminal_suits) == 1:
        return Pattern.MIXED_ONE_SUIT_TERMINALS
    return Pattern.TERMINALS_HONORS


def _item(pattern: Pattern) -> ScoreItem:
    value = int(RULES["patterns"][pattern.value])
    return ScoreItem(pattern, value)


def _event_items(context: ScoreContext) -> tuple[ScoreItem, ...]:
    result: list[ScoreItem] = []
    if sea_bottom_bonus(context.method, context.wall_before_draw):
        result.append(_item(Pattern.SEA_BOTTOM))
    if context.kong_draw:
        if context.method is not WinMethod.SELF_DRAW:
            raise RuleError("bonus.kong_draw")
        result.append(_item(Pattern.KONG_DRAW))
    return tuple(result)


def _thirteen_orphans(context: ScoreContext) -> ScoreResult | None:
    if context.hand.melds:
        return None
    counts = counts_of(context.hand.concealed)
    if sum(counts) != 14 or any(counts[kind] == 0 for kind in ORPHAN_TYPES):
        return None
    if any(counts[kind] for kind in range(34) if kind not in ORPHAN_TYPES):
        return None
    if sorted(counts[kind] for kind in ORPHAN_TYPES) != [1] * 12 + [2]:
        return None
    pre = list(counts)
    pre[tile_type(context.winning_tile)] -= 1
    thirteen_sided = all(pre[kind] == 1 for kind in ORPHAN_TYPES)
    pattern = Pattern.THIRTEEN_SIDED if thirteen_sided else Pattern.THIRTEEN_ORPHANS
    items = (_item(pattern),) + _event_items(context)
    return ScoreResult(sum(item.multiplier for item in items), items, (), pattern.value, 400)


def _seven_pairs(context: ScoreContext) -> ScoreResult | None:
    hand = context.hand
    if hand.melds:
        return None
    counts = counts_of(hand.concealed)
    if any(value % 2 for value in counts) or sum(value // 2 for value in counts) != 7:
        return None
    quads = counts.count(4)
    family = (
        Pattern.SEVEN_PAIRS,
        Pattern.LUXURY_SEVEN_PAIRS,
        Pattern.DOUBLE_LUXURY_SEVEN_PAIRS,
        Pattern.TRIPLE_LUXURY_SEVEN_PAIRS,
    )[quads]
    items = [_item(family)]
    types = _all_types(hand)
    terminal = _terminal_pattern(types)
    suit = _suit_pattern(types)
    # NOTE: 终端家族比普通花色描述更完整；混一色幺九已包含混一色3。
    if terminal is not None:
        items.append(_item(terminal))
    elif suit is not None:
        items.append(_item(suit))
    items.extend(_event_items(context))
    return ScoreResult(
        sum(item.multiplier for item in items),
        tuple(items),
        (),
        f"SEVEN_PAIRS:{quads}",
        200,
    )


def _declared_group_types(hand: WinningHand) -> tuple[int, ...]:
    return tuple(tile_type(meld.tiles[0]) for meld in hand.melds)


def _all_group_types(hand: WinningHand, shape: StandardShape) -> tuple[int, ...]:
    return _declared_group_types(hand) + tuple(group.kind for group in shape.concealed_groups)


def _all_triplets(hand: WinningHand, shape: StandardShape) -> bool:
    return all(group.category == "triplet" for group in shape.concealed_groups)


def _straight(shape: StandardShape) -> bool:
    starts = {group.kind for group in shape.concealed_groups if group.category == "sequence"}
    return any({base, base + 3, base + 6} <= starts for base in (0, 9, 18))


def _dragon_wind_items(
    hand: WinningHand,
    shape: StandardShape,
) -> tuple[list[ScoreItem], set[Pattern]]:
    group_types = _all_group_types(hand, shape)
    triplet_types = {
        kind for kind in _declared_group_types(hand)
    } | {
        group.kind for group in shape.concealed_groups if group.category == "triplet"
    }
    items: list[ScoreItem] = []
    includes_pong: set[Pattern] = set()
    dragons = len(triplet_types & DRAGON_TYPES)
    if dragons == 3:
        items.append(_item(Pattern.BIG_DRAGONS))
    elif dragons == 2 and shape.pair in DRAGON_TYPES:
        items.append(_item(Pattern.SMALL_DRAGONS))
    winds = len(triplet_types & WIND_TYPES)
    if winds == 4:
        items.append(_item(Pattern.BIG_WINDS))
        includes_pong.add(Pattern.BIG_WINDS)
    elif winds == 3 and shape.pair in WIND_TYPES:
        items.append(_item(Pattern.SMALL_WINDS))
    # Defensive assertion: a standard interpretation must always have four groups.
    if len(group_types) != 4:
        raise RuleError("shape.size")
    return items, includes_pong


def _four_concealed_item(
    context: ScoreContext,
    shape: StandardShape,
) -> ScoreItem | None:
    if any(meld.kind is not MeldKind.CLOSED_KONG for meld in context.hand.melds):
        return None
    if not _all_triplets(context.hand, shape):
        return None
    winning_kind = tile_type(context.winning_tile)
    pattern = (
        Pattern.FOUR_CONCEALED_SINGLE
        if winning_kind == shape.pair
        else Pattern.FOUR_CONCEALED
    )
    return _item(pattern)


def _eighteen_arhats(hand: WinningHand) -> bool:
    return len(hand.melds) == 4 and all(
        meld.kind is not MeldKind.PONG for meld in hand.melds
    )


def _standard_result(
    context: ScoreContext,
    shape: StandardShape,
    index: int,
) -> ScoreResult | None:
    hand = context.hand
    items: list[ScoreItem] = []
    suppressed: list[SuppressedItem] = []
    types = _all_types(hand)
    terminal = _terminal_pattern(types)
    suit = _suit_pattern(types)
    if terminal is not None:
        items.append(_item(terminal))
    elif suit is not None:
        items.append(_item(suit))

    relation_items, includes_pong = _dragon_wind_items(hand, shape)
    items.extend(relation_items)
    four = _four_concealed_item(context, shape)
    if four is not None:
        items.append(four)
        includes_pong.add(four.pattern)
    if _eighteen_arhats(hand):
        items.append(_item(Pattern.EIGHTEEN_ARHATS))
        includes_pong.add(Pattern.EIGHTEEN_ARHATS)

    all_triplets = _all_triplets(hand, shape)
    if all_triplets:
        if terminal is not None:
            includes_pong.add(terminal)
        if Pattern.BIG_WINDS in {item.pattern for item in items}:
            includes_pong.add(Pattern.BIG_WINDS)
        if includes_pong:
            suppressed.append(SuppressedItem(Pattern.PONG_PONG, "included"))
        else:
            items.append(_item(Pattern.PONG_PONG))

    if _straight(shape):
        items.append(_item(Pattern.ONE_STRAIGHT))

    base_patterns = {
        item.pattern
        for item in items
        if item.pattern not in {Pattern.ONE_STRAIGHT, Pattern.SEA_BOTTOM, Pattern.KONG_DRAW}
    }
    if suit is not None and not hand.melds and base_patterns == {suit}:
        items.append(_item(Pattern.CLOSED_BONUS))

    items.extend(_event_items(context))
    base_items = [
        item for item in items
        if item.pattern not in {
            Pattern.CLOSED_BONUS,
            Pattern.ONE_STRAIGHT,
            Pattern.SEA_BOTTOM,
            Pattern.KONG_DRAW,
        }
    ]
    if not base_items:
        return None
    priority = 300 if four is not None or _eighteen_arhats(hand) else 100
    return ScoreResult(
        sum(item.multiplier for item in items),
        tuple(items),
        tuple(suppressed),
        f"STANDARD:{index}:PAIR={shape.pair}",
        priority,
    )


def score_interpretations(context: ScoreContext) -> tuple[ScoreResult, ...]:
    """返回全部完整合法大胡解释；事件附加不能单独创造胡牌资格。"""
    results: list[ScoreResult] = []
    orphan = _thirteen_orphans(context)
    if orphan is not None:
        results.append(orphan)
    pairs = _seven_pairs(context)
    if pairs is not None:
        results.append(pairs)
    for index, shape in enumerate(standard_shapes(context.hand)):
        result = _standard_result(context, shape, index)
        if result is not None:
            results.append(result)
    return tuple(results)


def score_hand(context: ScoreContext) -> ScoreResult | None:
    """按最终倍数最高、固定解释优先级和稳定牌型序选择唯一计分解释。"""
    results = score_interpretations(context)
    if not results:
        return None
    return max(
        results,
        key=lambda result: (
            result.multiplier,
            result.priority,
            tuple(item.pattern.value for item in result.items),
            result.interpretation,
        ),
    )
