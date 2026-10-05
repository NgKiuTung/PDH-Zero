"""玩家动作契约。"""
from dataclasses import dataclass
from enum import Enum

from pdh_v2.config import PLAYERS
from pdh_v2.contracts import RuleError, integer
from pdh_v2.tiles import validate_tiles


class ActionKind(str, Enum):
    """无吃牌房规下的全部策略动作。"""

    DISCARD = "DISCARD"
    PONG = "PONG"
    OPEN_KONG = "OPEN_KONG"
    CLOSED_KONG = "CLOSED_KONG"
    ADDED_KONG = "ADDED_KONG"
    WIN = "WIN"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class Action:
    """动作携带真实实体牌，便于回放和牌数守恒审计。"""

    player: int
    kind: ActionKind
    tiles: tuple[int, ...] = ()
    meld_index: int | None = None

    def __post_init__(self) -> None:
        integer(self.player, "player", 0, PLAYERS - 1)
        if not isinstance(self.kind, ActionKind):
            raise RuleError("input.type", field="action.kind")
        cards = validate_tiles(self.tiles)
        expected = {
            ActionKind.DISCARD: 1,
            ActionKind.PONG: 2,
            ActionKind.OPEN_KONG: 3,
            ActionKind.CLOSED_KONG: 4,
            ActionKind.ADDED_KONG: 1,
            ActionKind.WIN: 0,
            ActionKind.PASS: 0,
        }[self.kind]
        if len(cards) != expected:
            raise RuleError("action.tiles")
        if self.kind is ActionKind.ADDED_KONG:
            integer(self.meld_index, "meld_index", 0, 3)
        elif self.meld_index is not None:
            raise RuleError("action.meld_index")
        object.__setattr__(self, "tiles", cards)
