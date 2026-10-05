"""十二张落地与杠上开花的责任锁定。"""
from dataclasses import dataclass
from typing import Iterable

from pdh_v2.config import PLAYERS
from pdh_v2.contracts import RuleError, integer, text
from pdh_v2.qualifications import WinMethod
from pdh_v2.tiles import Meld, MeldKind, validate_tiles


def _melds(values: Iterable[Meld]) -> tuple[Meld, ...]:
    declared = tuple(values)
    if not all(isinstance(meld, Meld) for meld in declared):
        raise RuleError("input.type", field="melds")
    integer(len(declared), "meld_count", 0, 4)
    validate_tiles(card for meld in declared for card in meld.tiles)
    return declared


def grounded_tile_count(melds: Iterable[Meld]) -> int:
    """公开刻/杠组数×3；展示张数与包牌触发分开。"""
    declared = _melds(melds)
    groups = sum(meld.kind is not MeldKind.CLOSED_KONG for meld in declared)
    return groups * 3


def is_twelve_grounded(melds: Iterable[Meld]) -> bool:
    """四组公开刻/杠在展示层属于十二张落地。"""
    return grounded_tile_count(melds) == 12


def pong_origin_group_count(melds: Iterable[Meld]) -> int:
    """碰及由碰升级的加杠各算一次碰出组；直接明杠和暗杠不冒充碰。"""
    declared = _melds(melds)
    return sum(meld.kind in (MeldKind.PONG, MeldKind.ADDED_KONG) for meld in declared)


@dataclass(frozen=True, slots=True)
class TwelveGroundedLiability:
    """第四碰提供者承担同一手内该玩家自摸的三家份额。"""

    hand_id: str
    claimant: int
    responsible: int

    def __post_init__(self) -> None:
        text(self.hand_id, "hand_id")
        integer(self.claimant, "claimant", 0, PLAYERS - 1)
        integer(self.responsible, "responsible", 0, PLAYERS - 1)
        if self.claimant == self.responsible:
            raise RuleError("responsibility.party")

    def applies_to(self, hand_id: str, winner: int, method: WinMethod) -> bool:
        """责任仅限触发当局、触发玩家自摸。"""
        text(hand_id, "hand_id")
        integer(winner, "winner", 0, PLAYERS - 1)
        if not isinstance(method, WinMethod):
            raise RuleError("input.type", field="win_method")
        return hand_id == self.hand_id and winner == self.claimant and method is WinMethod.SELF_DRAW


@dataclass(frozen=True, slots=True)
class KongDrawLiability:
    """点杠直接补牌自摸时，点杠者承担三家份额。"""

    hand_id: str
    claimant: int
    responsible: int

    def __post_init__(self) -> None:
        text(self.hand_id, "hand_id")
        integer(self.claimant, "claimant", 0, PLAYERS - 1)
        integer(self.responsible, "responsible", 0, PLAYERS - 1)
        if self.claimant == self.responsible:
            raise RuleError("responsibility.party")


def lock_twelve_grounded_liability(
    hand_id: str,
    claimant: int,
    provider: int,
    existing_melds: Iterable[Meld],
    incoming_kind: MeldKind,
) -> TwelveGroundedLiability | None:
    """仅三组碰源面子后的第四次碰锁责。"""
    text(hand_id, "hand_id")
    integer(claimant, "claimant", 0, PLAYERS - 1)
    integer(provider, "provider", 0, PLAYERS - 1)
    if claimant == provider:
        raise RuleError("responsibility.party")
    if not isinstance(incoming_kind, MeldKind):
        raise RuleError("input.type", field="incoming_kind")
    declared = _melds(existing_melds)
    if incoming_kind is not MeldKind.PONG:
        return None
    if len(declared) != 3 or pong_origin_group_count(declared) != 3:
        return None
    return TwelveGroundedLiability(hand_id, claimant, provider)


def lock_kong_draw_liability(
    hand_id: str,
    claimant: int,
    provider: int | None,
    kong_kind: MeldKind,
) -> KongDrawLiability | None:
    """只有他人弃牌形成的直接明杠可在紧接补牌自摸时锁定杠爆责任。"""
    text(hand_id, "hand_id")
    integer(claimant, "claimant", 0, PLAYERS - 1)
    if not isinstance(kong_kind, MeldKind):
        raise RuleError("input.type", field="kong_kind")
    if kong_kind is not MeldKind.OPEN_KONG:
        return None
    integer(provider, "provider", 0, PLAYERS - 1)
    if provider == claimant:
        raise RuleError("responsibility.party")
    return KongDrawLiability(hand_id, claimant, provider)
