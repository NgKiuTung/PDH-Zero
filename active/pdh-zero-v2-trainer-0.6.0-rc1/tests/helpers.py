"""测试实体牌构造器；不参与正式洗牌或发牌规则。"""
from collections import Counter
from typing import Iterable

from pdh_v2.qualifications import WinMethod
from pdh_v2.settlement import MultiDiscardWin, MultiWinner, OrdinaryWin
from pdh_v2.tiles import Meld, MeldKind


def cards(kinds: Iterable[int]) -> tuple[int, ...]:
    """按给定牌种顺序分配四副本中的实体ID；超过四张即失败。"""
    used: Counter[int] = Counter()
    result = []
    for kind in kinds:
        if not 0 <= kind < 34 or used[kind] >= 4:
            raise ValueError(kind)
        result.append(kind * 4 + used[kind])
        used[kind] += 1
    return tuple(result)


def meld(kind: int, category: MeldKind) -> Meld:
    """生成单种牌的合法声明面子；跨面子重复由被测输入校验。"""
    size = 3 if category is MeldKind.PONG else 4
    return Meld(category, cards([kind] * size))


def ordinary_win() -> OrdinaryWin:
    """已裁定的单赢家普通自摸算例；不作为胡牌判定替代。"""
    return OrdinaryWin(
        "settlement-1",
        "hand-1",
        1,
        0,
        0,
        6,
        1,
        WinMethod.SELF_DRAW,
    )


def multi_win() -> MultiDiscardWin:
    """已裁定的一炮双响算例；共用一批马牌。"""
    return MultiDiscardWin(
        "multi-1",
        "hand-1",
        previous_dealer=0,
        continuation=0,
        base_units=1,
        discarder=3,
        winners=(MultiWinner(0, 6), MultiWinner(1, 3)),
    )


def next_hand_cards() -> tuple[int, ...]:
    """前六张命中1号的前三张；后八张只作为实际完整起手牌。"""
    return cards([1, 14, 28, 21, 27, 8, 0, 2, 3, 4, 5, 6, 7, 9])
