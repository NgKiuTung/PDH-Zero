"""136张实体与面子契约；许可由项目所有者决定。牌序由调用方提供。"""
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from pdh_v2.config import COPIES, DECK_SIZE, TILE_TYPES
from pdh_v2.contracts import RuleError, integer


class MeldKind(str, Enum):
    """已声明面子类型；本房规无吃牌，顺子只能出现在未副露结构分解中。"""

    PONG = "PONG"
    OPEN_KONG = "OPEN_KONG"
    CLOSED_KONG = "CLOSED_KONG"
    ADDED_KONG = "ADDED_KONG"


def tile_type(tile_id: int) -> int:
    """实体ID为0..135，类型为0..33；拒绝非法或布尔ID，O(1)。"""
    return integer(tile_id, "tile_id", 0, DECK_SIZE - 1) // COPIES


def validate_tiles(values: Iterable[int], expected: int | None = None) -> tuple[int, ...]:
    """校验实体唯一及可选精确张数，返回不可变序列；不排序，保留发牌次序。"""
    cards = tuple(values)
    for card in cards:
        tile_type(card)
    if len(set(cards)) != len(cards):
        raise RuleError("tile.duplicate")
    if expected is not None and len(cards) != expected:
        raise RuleError("tile.count", field="expected")
    return cards


def counts_of(values: Iterable[int]) -> tuple[int, ...]:
    """将实体序列转为34种计数；重复实体不作为同种的另一副本。"""
    counts = [0] * TILE_TYPES
    for card in validate_tiles(values):
        counts[tile_type(card)] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class Meld:
    """一个真实声明的碰/杠；同牌种，实体不重复，适合跨线程只读共享。"""

    kind: MeldKind
    tiles: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MeldKind):
            raise RuleError("input.type", field="meld.kind")
        cards = validate_tiles(self.tiles, 3 if self.kind is MeldKind.PONG else 4)
        if len({tile_type(card) for card in cards}) != 1:
            raise RuleError("meld.invalid")
        object.__setattr__(self, "tiles", cards)


@dataclass(frozen=True, slots=True)
class WinningHand:
    """胡牌候选的完整物理输入；只保证张数一致，不代表该手有合法大胡。"""

    concealed: tuple[int, ...]
    melds: tuple[Meld, ...] = ()

    def __post_init__(self) -> None:
        melds = tuple(self.melds)
        integer(len(melds), "meld_count", 0, 4)
        if not all(isinstance(meld, Meld) for meld in melds):
            raise RuleError("input.type", field="hand.melds")
        concealed = validate_tiles(self.concealed)
        all_tiles = concealed + tuple(card for meld in melds for card in meld.tiles)
        validate_tiles(all_tiles)
        kongs = sum(meld.kind is not MeldKind.PONG for meld in melds)
        if len(all_tiles) != 14 + kongs:
            raise RuleError("shape.size")
        object.__setattr__(self, "concealed", concealed)
        object.__setattr__(self, "melds", melds)
