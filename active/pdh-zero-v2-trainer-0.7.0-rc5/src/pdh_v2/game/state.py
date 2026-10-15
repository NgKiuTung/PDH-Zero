"""完整手局状态数据结构。"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pdh_v2.config import PLAYERS
from pdh_v2.contracts import RuleError, integer, text
from pdh_v2.qualifications import WinMethod
from pdh_v2.responsibility import TwelveGroundedLiability
from pdh_v2.scoring import ScoreResult
from pdh_v2.tiles import Meld, validate_tiles


class Phase(str, Enum):
    """策略交互阶段。"""

    ACTION = "ACTION"
    DISCARD_RESPONSE = "DISCARD_RESPONSE"
    ADDED_KONG_RESPONSE = "ADDED_KONG_RESPONSE"
    TERMINAL = "TERMINAL"


@dataclass(slots=True)
class MeldRecord:
    """声明面子及来源；provider仅对他人弃牌形成的公开面子有效。"""

    meld: Meld
    provider: int | None = None


@dataclass(slots=True)
class PlayerState:
    """单手玩家私有与公开状态。"""

    concealed: list[int] = field(default_factory=list)
    melds: list[MeldRecord] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)
    revealed_retained: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        validate_tiles(self.concealed)


@dataclass(frozen=True, slots=True)
class PendingDiscard:
    """等待其他玩家响应的弃牌。"""

    player: int
    tile: int


@dataclass(frozen=True, slots=True)
class PendingAddedKong:
    """加杠在抢杠窗口关闭前不得修改真实面子。"""

    player: int
    tile: int
    meld_index: int


@dataclass(frozen=True, slots=True)
class KongContext:
    """最近一次直接补牌对应的杠；点杠责任只覆盖紧接补牌自摸。"""

    kind: str
    declarer: int
    provider: int | None


@dataclass(frozen=True, slots=True)
class WinnerResult:
    """一个赢家的最终解释。"""

    player: int
    score: ScoreResult

    def __post_init__(self) -> None:
        integer(self.player, "winner", 0, PLAYERS - 1)
        if not isinstance(self.score, ScoreResult):
            raise RuleError("input.type", field="winner.score")


@dataclass(frozen=True, slots=True)
class HandOutcome:
    """手局终局；draw为真时winners为空。"""

    hand_id: str
    method: str
    winners: tuple[WinnerResult, ...]
    discarder: int | None
    draw: bool
    wall_source: str
    old_wall: tuple[int, ...]
    twelve_grounded: TwelveGroundedLiability | None = None
    kong_provider: int | None = None

    def __post_init__(self) -> None:
        text(self.hand_id, "hand_id")
        if type(self.draw) is not bool:
            raise RuleError("input.type", field="outcome.draw")
        winners = tuple(self.winners)
        if not all(isinstance(item, WinnerResult) for item in winners):
            raise RuleError("input.type", field="outcome.winners")
        if len({item.player for item in winners}) != len(winners):
            raise RuleError("outcome.invalid")
        if self.wall_source not in {"FRONT", "BACK"}:
            raise RuleError("wall.source")
        old_wall = validate_tiles(self.old_wall)
        object.__setattr__(self, "winners", winners)
        object.__setattr__(self, "old_wall", old_wall)
        if self.draw:
            if self.method != "DRAW" or winners or self.discarder is not None:
                raise RuleError("outcome.invalid")
            if self.twelve_grounded is not None or self.kong_provider is not None:
                raise RuleError("outcome.invalid")
            return
        try:
            method = WinMethod(self.method)
        except ValueError as exc:
            raise RuleError("outcome.invalid") from exc
        if not winners:
            raise RuleError("outcome.invalid")
        winner_ids = {item.player for item in winners}
        if method is WinMethod.SELF_DRAW:
            if len(winners) != 1 or self.discarder is not None:
                raise RuleError("outcome.invalid")
        else:
            integer(self.discarder, "discarder", 0, PLAYERS - 1)
            if self.discarder in winner_ids:
                raise RuleError("outcome.invalid")
        if self.twelve_grounded is not None:
            if method is not WinMethod.SELF_DRAW:
                raise RuleError("outcome.invalid")
            if self.twelve_grounded.hand_id != self.hand_id:
                raise RuleError("outcome.invalid")
            if self.twelve_grounded.claimant not in winner_ids:
                raise RuleError("outcome.invalid")
        if self.kong_provider is not None:
            if method is not WinMethod.SELF_DRAW or self.twelve_grounded is not None:
                raise RuleError("outcome.invalid")
            integer(self.kong_provider, "kong_provider", 0, PLAYERS - 1)
            if self.kong_provider in winner_ids:
                raise RuleError("outcome.invalid")


@dataclass(slots=True)
class ResponseState:
    """响应队列与已提交动作。"""

    source_player: int
    queue: list[int]
    responses: dict[int, Any] = field(default_factory=dict)
