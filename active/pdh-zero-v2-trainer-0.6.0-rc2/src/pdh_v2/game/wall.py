"""固定牌序、传统发牌、墙头摸牌与墙尾杠补牌。"""
from dataclasses import dataclass

from pdh_v2.config import DECK_SIZE, PLAYERS
from pdh_v2.contracts import RuleError, integer
from pdh_v2.tiles import validate_tiles


@dataclass(frozen=True, slots=True)
class DrawnTile:
    """一次真实摸牌；wall_before用于海底判定。"""

    player: int
    tile: int
    source: str
    wall_before: int


class Wall:
    """单手牌墙；front和back之间的实体尚未离墙。"""

    def __init__(self, order: tuple[int, ...]) -> None:
        self.order = validate_tiles(order, DECK_SIZE)
        self.front = 0
        self.back = DECK_SIZE - 1

    @property
    def remaining(self) -> int:
        """返回仍在牌墙中的实体牌张数。"""
        return max(0, self.back - self.front + 1)

    def _draw(self, player: int, source: str) -> DrawnTile:
        integer(player, "player", 0, PLAYERS - 1)
        if self.remaining == 0:
            raise RuleError("wall.empty")
        before = self.remaining
        if source == "FRONT":
            tile = self.order[self.front]
            self.front += 1
        elif source == "BACK":
            tile = self.order[self.back]
            self.back -= 1
        else:
            raise RuleError("wall.source")
        return DrawnTile(player, tile, source, before)

    def draw_front(self, player: int) -> DrawnTile:
        """普通摸牌。"""
        return self._draw(player, "FRONT")

    def draw_back(self, player: int) -> DrawnTile:
        """杠后补牌。"""
        return self._draw(player, "BACK")

    def available_after(self, source: str) -> tuple[int, ...]:
        """从最后摸牌同一墙端的后继方向返回可用旧墙牌序。"""
        if source == "FRONT":
            return self.order[self.front : self.back + 1]
        if source == "BACK":
            return tuple(reversed(self.order[self.front : self.back + 1]))
        raise RuleError("wall.source")

    def deal(self, dealer: int) -> tuple[tuple[int, ...], ...]:
        """按4张×3轮，再庄2闲1的原始顺序发牌；返回每家到手顺序。"""
        integer(dealer, "dealer", 0, PLAYERS - 1)
        hands: list[list[int]] = [[] for _ in range(PLAYERS)]
        seats = tuple((dealer + offset) % PLAYERS for offset in range(PLAYERS))
        for _ in range(3):
            for player in seats:
                for _ in range(4):
                    hands[player].append(self.draw_front(player).tile)
        for index, player in enumerate(seats):
            count = 2 if index == 0 else 1
            for _ in range(count):
                hands[player].append(self.draw_front(player).tile)
        return tuple(tuple(hand) for hand in hands)
