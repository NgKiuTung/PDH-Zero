"""已确认资格与海底条件；许可由项目所有者决定。不输出完整胡牌结论。"""
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from pdh_v2.config import DECK_SIZE, RULES
from pdh_v2.contracts import RuleError, integer
from pdh_v2.tiles import Meld, MeldKind, validate_tiles


class WinMethod(str, Enum):
    """真实胡牌来源；抢杠不是自摸，其责任支付仍是未决专项。"""

    SELF_DRAW = "SELF_DRAW"
    DISCARD_WIN = "DISCARD_WIN"
    ROB_KONG = "ROB_KONG"


@dataclass(frozen=True, slots=True)
class RouteQualifications:
    """仅表示历史未破坏必要资格；不代表手牌结构已成立或能最终完成。"""

    closed_bonus_history_ok: bool
    seven_pairs_history_ok: bool
    four_concealed_history_ok: bool
    thirteen_orphans_history_ok: bool


def route_qualifications(melds: Iterable[Meld]) -> RouteQualifications:
    """输入真实声明面子；亮马不是面子，不得放入此序列。"""
    declared = tuple(melds)
    if not all(isinstance(meld, Meld) for meld in declared):
        raise RuleError("input.type", field="melds")
    integer(len(declared), "meld_count", 0, 4)
    validate_tiles(card for meld in declared for card in meld.tiles)
    no_melds = not declared
    return RouteQualifications(
        closed_bonus_history_ok=no_melds,
        seven_pairs_history_ok=no_melds,
        four_concealed_history_ok=all(
            meld.kind is MeldKind.CLOSED_KONG for meld in declared
        ),
        thirteen_orphans_history_ok=no_melds,
    )


def closed_bonus(
    simple_suit: str | None,
    other_big_pattern: bool,
    melds: Iterable[Meld],
) -> int:
    """调用方提供已检测基础项；仅单纯清/混且无声明碰杠加3，事件项不传成基础项。"""
    if simple_suit not in (None, "MIXED_ONE_SUIT", "PURE_ONE_SUIT"):
        raise RuleError("bonus.base")
    if type(other_big_pattern) is not bool:
        raise RuleError("input.type", field="other_big_pattern")
    history_ok = route_qualifications(melds).closed_bonus_history_ok
    if simple_suit and history_ok and not other_big_pattern:
        return int(RULES["bonuses"]["closed"])
    return 0


def sea_bottom_bonus(method: WinMethod, wall_before_draw: int) -> int:
    """只判事件+3；摸前1..4且自摸。无基础大胡时调用方仍必须拒绝胡。"""
    if not isinstance(method, WinMethod):
        raise RuleError("input.type", field="win_method")
    low = 1 if method is WinMethod.SELF_DRAW else 0
    count = integer(wall_before_draw, "wall_before_draw", low, DECK_SIZE)
    eligible = method is WinMethod.SELF_DRAW and count <= int(RULES["sea_max_before_draw"])
    return int(RULES["bonuses"]["sea_bottom"]) if eligible else 0
