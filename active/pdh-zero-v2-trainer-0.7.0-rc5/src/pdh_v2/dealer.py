"""庄家转换与连庄状态；许可由项目所有者决定。"""
from dataclasses import dataclass
from sys import maxsize
from typing import Iterable

from pdh_v2.config import PLAYERS, RULES
from pdh_v2.contracts import RuleError, integer


def _winners(values: Iterable[int], minimum: int = 1) -> tuple[int, ...]:
    """规范化赢家ID；拒绝重复、布尔值和空集合。"""
    result = tuple(values)
    if len(result) < minimum:
        raise RuleError("dealer.winners")
    for player in result:
        integer(player, "winner", 0, PLAYERS - 1)
    if len(set(result)) != len(result):
        raise RuleError("dealer.winners")
    return result


@dataclass(frozen=True, slots=True)
class DealerState:
    """固定玩家编号+首次胡资格+累计连庄；流局不可清除首次胡资格。"""

    player: int = 0
    has_won: bool = False
    continuation: int = 0

    def __post_init__(self) -> None:
        integer(self.player, "dealer", 0, PLAYERS - 1)
        integer(self.continuation, "continuation", 0, maxsize)
        if type(self.has_won) is not bool:
            raise RuleError("input.type", field="has_won")
        if not self.has_won and self.continuation != 0:
            raise RuleError("dealer.state")

    def after_win(self, winner: int) -> "DealerState":
        """单赢家：旧庄再胡才递增；闲家胡成为新庄且本次n=0。"""
        integer(winner, "winner", 0, PLAYERS - 1)
        count = self.continuation + 1 if winner == self.player and self.has_won else 0
        return DealerState(winner, True, count)

    def after_multi_win(self, winners: Iterable[int]) -> "DealerState":
        """一炮多响后的下庄：庄家胡则庄家续庄，否则本局相对位置最小的赢家坐庄。"""
        seats = _winners(winners, minimum=2)
        if self.player in seats:
            return self.after_win(self.player)
        next_dealer = min(seats, key=self.relative_seat)
        # NOTE: 其他同时胡牌但未成为庄家的玩家不会获得下一局连庄资格。
        return DealerState(next_dealer, True, 0)

    def after_draw(self) -> "DealerState":
        """流局庄位/累计数/首次胡资格全部保留；点数由结算层记录为零。"""
        return self

    def relative_seat(self, player: int) -> int:
        """固定ID按逆时针0..3编码；返回相对本局庄家的编号。"""
        return (integer(player, "player", 0, PLAYERS - 1) - self.player) % PLAYERS


def horse_requirement(continuation: int) -> int:
    """根据结算已确定的n计算N=2n+6；不擅自封顶。"""
    count = integer(continuation, "continuation", 0, maxsize)
    return int(RULES["horse"]["initial_count"]) + count * int(RULES["horse"]["per_continuation"])
