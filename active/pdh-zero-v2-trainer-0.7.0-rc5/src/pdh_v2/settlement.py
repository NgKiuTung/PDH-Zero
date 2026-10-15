"""已裁定结算的支付算术；许可由项目所有者决定。"""
from dataclasses import dataclass
from typing import Mapping

from pdh_v2.config import PLAYERS
from pdh_v2.contracts import RuleError, integer, text
from pdh_v2.dealer import horse_requirement
from pdh_v2.qualifications import WinMethod
from pdh_v2.responsibility import KongDrawLiability, TwelveGroundedLiability


def _positive(value: object, field: str) -> int:
    """正整数且无房规封顶；Python整数不在此引入人为上限。"""
    if type(value) is not int or value < 1:
        raise RuleError("input.positive", field=field)
    return value


@dataclass(frozen=True, slots=True)
class DrawRecord:
    """流局审计记录；无点数流动，庄位/连庄变化由DealerState.after_draw处理。"""

    draw_id: str
    hand_id: str
    dealer: int
    continuation: int

    def __post_init__(self) -> None:
        text(self.draw_id, "draw_id")
        text(self.hand_id, "hand_id")
        integer(self.dealer, "dealer", 0, PLAYERS - 1)
        horse_requirement(self.continuation)


@dataclass(frozen=True, slots=True)
class OrdinaryWin:
    """调用方已核定的普通单赢家胡事件快照；倍数是最终F输入。"""

    settlement_id: str
    hand_id: str
    winner: int
    previous_dealer: int
    continuation: int
    multiplier: int
    base_units: int
    method: WinMethod
    discarder: int | None = None

    def __post_init__(self) -> None:
        text(self.settlement_id, "settlement_id")
        text(self.hand_id, "hand_id")
        integer(self.winner, "winner", 0, PLAYERS - 1)
        integer(self.previous_dealer, "previous_dealer", 0, PLAYERS - 1)
        horse_requirement(self.continuation)
        _positive(self.multiplier, "multiplier")
        _positive(self.base_units, "base_units")
        if self.winner != self.previous_dealer and self.continuation != 0:
            raise RuleError("settlement.context")
        if self.method is WinMethod.SELF_DRAW:
            if self.discarder is not None:
                raise RuleError("settlement.payer")
        elif self.method in (WinMethod.DISCARD_WIN, WinMethod.ROB_KONG):
            integer(self.discarder, "discarder", 0, PLAYERS - 1)
            if self.discarder == self.winner:
                raise RuleError("settlement.payer")
        else:
            raise RuleError("settlement.method")

    @property
    def previous_relative_seat(self) -> int:
        """本局相对庄位，跨局发牌不改变这一冻结快照。"""
        return (self.winner - self.previous_dealer) % PLAYERS


@dataclass(frozen=True, slots=True)
class MultiWinner:
    """一炮多响中的一个赢家；各赢家可有不同最终倍数和中马数。"""

    winner: int
    multiplier: int

    def __post_init__(self) -> None:
        integer(self.winner, "winner", 0, PLAYERS - 1)
        _positive(self.multiplier, "multiplier")


@dataclass(frozen=True, slots=True)
class MultiDiscardWin:
    """一张弃牌同时被多家胡的快照；共用一批马，各自按本局位置算m。"""

    settlement_id: str
    hand_id: str
    previous_dealer: int
    continuation: int
    base_units: int
    discarder: int
    winners: tuple[MultiWinner, ...]

    def __post_init__(self) -> None:
        text(self.settlement_id, "settlement_id")
        text(self.hand_id, "hand_id")
        integer(self.previous_dealer, "previous_dealer", 0, PLAYERS - 1)
        horse_requirement(self.continuation)
        _positive(self.base_units, "base_units")
        integer(self.discarder, "discarder", 0, PLAYERS - 1)
        winners = tuple(self.winners)
        if len(winners) < 2 or not all(isinstance(item, MultiWinner) for item in winners):
            raise RuleError("settlement.multi")
        ids = tuple(item.winner for item in winners)
        if len(set(ids)) != len(ids) or self.discarder in ids:
            raise RuleError("settlement.multi")
        if self.previous_dealer not in ids and self.continuation != 0:
            raise RuleError("settlement.context")
        object.__setattr__(self, "winners", winners)

    def relative_seat(self, winner: int) -> int:
        """赢家各自用计分所属局的位置算马。"""
        integer(winner, "winner", 0, PLAYERS - 1)
        if winner not in {item.winner for item in self.winners}:
            raise RuleError("settlement.multi")
        return (winner - self.previous_dealer) % PLAYERS


def ordinary_payments(win: OrdinaryWin, horse_hits: int) -> tuple[int, ...]:
    """普通单赢家付款；m=0仍付款，无封顶。"""
    hits = integer(horse_hits, "horse_hits", 0, horse_requirement(win.continuation))
    unit = win.base_units * win.multiplier * (hits + 1)
    result = [0] * PLAYERS
    if win.method is WinMethod.SELF_DRAW:
        result = [-unit] * PLAYERS
        result[win.winner] = unit * (PLAYERS - 1)
    else:
        if win.discarder is None:
            raise RuleError("settlement.payer")
        result[win.discarder] = -unit
        result[win.winner] = unit
    return tuple(result)


def multi_discard_payments(
    win: MultiDiscardWin,
    horse_hits: Mapping[int, int],
) -> tuple[int, ...]:
    """一炮多响：点炮者分别向每个赢家支付一份，各赢家m独立。"""
    expected = {item.winner for item in win.winners}
    if set(horse_hits) != expected:
        raise RuleError("settlement.multi")
    result = [0] * PLAYERS
    total = 0
    for item in win.winners:
        hits = integer(
            horse_hits[item.winner],
            "horse_hits",
            0,
            horse_requirement(win.continuation),
        )
        unit = win.base_units * item.multiplier * (hits + 1)
        result[item.winner] += unit
        total += unit
    result[win.discarder] -= total
    return tuple(result)


def cover_three_payments(
    winner: int,
    responsible: int,
    base_units: int,
    multiplier: int,
    horse_hits: int,
    continuation: int,
) -> tuple[int, ...]:
    """已裁定承包三家时的纯算术：责任者独付三份S。"""
    integer(winner, "winner", 0, PLAYERS - 1)
    integer(responsible, "responsible", 0, PLAYERS - 1)
    if winner == responsible:
        raise RuleError("settlement.payer")
    base = _positive(base_units, "base_units")
    factor = _positive(multiplier, "multiplier")
    hits = integer(horse_hits, "horse_hits", 0, horse_requirement(continuation))
    unit = base * factor * (hits + 1)
    result = [0] * PLAYERS
    result[winner] = unit * (PLAYERS - 1)
    result[responsible] = -unit * (PLAYERS - 1)
    return tuple(result)


def twelve_grounded_payments(
    win: OrdinaryWin,
    horse_hits: int,
    liability: TwelveGroundedLiability,
) -> tuple[int, ...]:
    """第四碰责任只覆盖同局该玩家自摸；吃胡仍由实际点炮者支付一份。"""
    if win.hand_id != liability.hand_id or win.winner != liability.claimant:
        raise RuleError("responsibility.context")
    if liability.applies_to(win.hand_id, win.winner, win.method):
        return cover_three_payments(
            win.winner,
            liability.responsible,
            win.base_units,
            win.multiplier,
            horse_hits,
            win.continuation,
        )
    return ordinary_payments(win, horse_hits)



def kong_draw_payments(
    win: OrdinaryWin,
    horse_hits: int,
    liability: KongDrawLiability,
) -> tuple[int, ...]:
    """直接点杠补牌自摸责任优先于十二张落地，只执行一套三家承包。"""
    if win.hand_id != liability.hand_id or win.winner != liability.claimant:
        raise RuleError("responsibility.context")
    if win.method is not WinMethod.SELF_DRAW:
        raise RuleError("responsibility.context")
    return cover_three_payments(
        win.winner, liability.responsible, win.base_units, win.multiplier,
        horse_hits, win.continuation,
    )

def draw_payments() -> tuple[int, ...]:
    """流局无事发生：点数不流动；事件本身仍应写入牌谱/账本。"""
    return (0,) * PLAYERS
