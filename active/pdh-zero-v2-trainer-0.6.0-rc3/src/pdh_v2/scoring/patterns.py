"""牌型标识与计分解释数据结构。"""
from dataclasses import dataclass
from enum import Enum


class Pattern(str, Enum):
    """计分项；组合名称只用于展示，不重复计分。"""

    MIXED_ONE_SUIT = "MIXED_ONE_SUIT"
    PURE_ONE_SUIT = "PURE_ONE_SUIT"
    PONG_PONG = "PONG_PONG"
    SEVEN_PAIRS = "SEVEN_PAIRS"
    LUXURY_SEVEN_PAIRS = "LUXURY_SEVEN_PAIRS"
    DOUBLE_LUXURY_SEVEN_PAIRS = "DOUBLE_LUXURY_SEVEN_PAIRS"
    TRIPLE_LUXURY_SEVEN_PAIRS = "TRIPLE_LUXURY_SEVEN_PAIRS"
    TERMINALS_HONORS = "TERMINALS_HONORS"
    MIXED_ONE_SUIT_TERMINALS = "MIXED_ONE_SUIT_TERMINALS"
    PURE_TERMINALS = "PURE_TERMINALS"
    SMALL_DRAGONS = "SMALL_DRAGONS"
    BIG_DRAGONS = "BIG_DRAGONS"
    SMALL_WINDS = "SMALL_WINDS"
    BIG_WINDS = "BIG_WINDS"
    ALL_HONORS = "ALL_HONORS"
    EIGHTEEN_ARHATS = "EIGHTEEN_ARHATS"
    THIRTEEN_ORPHANS = "THIRTEEN_ORPHANS"
    THIRTEEN_SIDED = "THIRTEEN_SIDED"
    FOUR_CONCEALED = "FOUR_CONCEALED"
    FOUR_CONCEALED_SINGLE = "FOUR_CONCEALED_SINGLE"
    CLOSED_BONUS = "CLOSED_BONUS"
    ONE_STRAIGHT = "ONE_STRAIGHT"
    SEA_BOTTOM = "SEA_BOTTOM"
    KONG_DRAW = "KONG_DRAW"


@dataclass(frozen=True, slots=True)
class ScoreItem:
    """实际计分项。"""

    pattern: Pattern
    multiplier: int


@dataclass(frozen=True, slots=True)
class SuppressedItem:
    """已识别但因包含或同族等级不重复计分的项目。"""

    pattern: Pattern
    reason: str
