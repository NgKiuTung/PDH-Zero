"""马牌来源、逐张归属与多响专门开马；物理顺序由已记录输入提供。"""
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from pdh_v2.config import DECK_SIZE, PLAYERS, RULES
from pdh_v2.contracts import RuleError, integer, text
from pdh_v2.dealer import horse_requirement
from pdh_v2.tiles import tile_type, validate_tiles


class HorseSource(str, Enum):
    """来源选择不混补；SEPARATE_DECK结算后必须另行洗牌开打。"""

    OLD_WALL = "OLD_WALL"
    NEXT_HAND = "NEXT_HAND"
    SEPARATE_DECK = "SEPARATE_DECK"


class HorseScenario(str, Enum):
    """单赢家与一炮多响在旧墙不足时使用不同后备牌源。"""

    ORDINARY = "ORDINARY"
    MULTI_WIN = "MULTI_WIN"


@dataclass(frozen=True, slots=True)
class HorsePlan:
    """不可变计划；数量始终等于要求N，未知超容量输入明确失败。"""

    continuation: int
    old_available: int
    required: int
    source: HorseSource
    scenario: HorseScenario = HorseScenario.ORDINARY

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, HorseScenario):
            raise RuleError("input.type", field="horse.scenario")
        expected = _source_for(self.continuation, self.old_available, self.scenario)
        if self.required != horse_requirement(self.continuation) or self.source is not expected:
            raise RuleError("horse.source")

    @property
    def reshuffle_after(self) -> bool:
        """专门开马牌序只用于上局结算，之后必须重新洗牌且不增加连庄。"""
        return self.source is HorseSource.SEPARATE_DECK


def _source_for(
    continuation: int,
    old_available: int,
    scenario: HorseScenario,
) -> HorseSource:
    required = horse_requirement(continuation)
    remaining = integer(old_available, "old_available", 0, DECK_SIZE)
    if remaining >= required:
        return HorseSource.OLD_WALL
    if scenario is HorseScenario.MULTI_WIN:
        # NOTE: 多响旧墙不足时统一新开一副牌，不借任一赢家的下一把手牌。
        return HorseSource.SEPARATE_DECK
    if required <= int(RULES["horse"]["next_dealer_hand_size"]):
        return HorseSource.NEXT_HAND
    return HorseSource.SEPARATE_DECK


def plan_horses(continuation: int, old_available: int) -> HorsePlan:
    """单赢家：旧墙足够优先；不足时按14张边界选择下局手牌或专门开马。"""
    scenario = HorseScenario.ORDINARY
    source = _source_for(continuation, old_available, scenario)
    return HorsePlan(
        continuation,
        old_available,
        horse_requirement(continuation),
        source,
        scenario,
    )


def plan_multiwin_horses(continuation: int, old_available: int) -> HorsePlan:
    """一炮多响：旧墙不足即统一使用新牌序开马，结算后重新洗牌。"""
    scenario = HorseScenario.MULTI_WIN
    source = _source_for(continuation, old_available, scenario)
    return HorsePlan(
        continuation,
        old_available,
        horse_requirement(continuation),
        source,
        scenario,
    )


def horse_owner(card: int) -> int:
    """返回该实体牌对应的相对庄位，重复牌种各实体独立计马。"""
    kind = tile_type(card)
    if kind < 27:
        return int(RULES["horse"]["rank_owners"][kind % 9])
    return int(RULES["horse"]["honor_owners"][kind - 27])


@dataclass(frozen=True, slots=True)
class HorseSelection:
    """绑定后的真实前N张；未命中牌仍包含在审计序列，不继续抽至中够N张。"""

    plan: HorsePlan
    source_id: str
    cards: tuple[int, ...]
    winner_relative_seat: int

    def __post_init__(self) -> None:
        text(self.source_id, "source_id")
        integer(self.winner_relative_seat, "winner_relative_seat", 0, PLAYERS - 1)
        cards = tuple(self.cards)
        if len(cards) != self.plan.required:
            raise RuleError("horse.source")
        if self.plan.source is HorseSource.SEPARATE_DECK:
            for start in range(0, len(cards), DECK_SIZE):
                validate_tiles(cards[start : start + DECK_SIZE])
        else:
            validate_tiles(cards)
        object.__setattr__(self, "cards", cards)

    @property
    def hits(self) -> tuple[int, ...]:
        """仅命中实体；按上局位置匹配，不改变手牌归属。"""
        return tuple(
            card for card in self.cards
            if horse_owner(card) == self.winner_relative_seat
        )

    @property
    def hit_count(self) -> int:
        """返回m，不按牌种去重；数值范围0..N。"""
        return len(self.hits)


def _source_cards(plan: HorsePlan, ordered_source: Iterable[int]) -> tuple[int, ...]:
    """校验真实牌源；专门开马可按136张分轮重洗，因此不同轮允许复用实体ID。"""
    raw = tuple(ordered_source)
    if plan.source is HorseSource.OLD_WALL:
        return validate_tiles(raw, plan.old_available)
    if plan.source is HorseSource.NEXT_HAND:
        return validate_tiles(raw, int(RULES["horse"]["next_dealer_hand_size"]))
    if len(raw) < plan.required:
        raise RuleError("horse.source")
    for start in range(0, len(raw), DECK_SIZE):
        chunk = raw[start : start + DECK_SIZE]
        # NOTE: 每轮来自一次完整独立洗牌；末轮只需提供实际使用部分。
        validate_tiles(chunk)
    return raw


def select_horses(
    plan: HorsePlan,
    source_id: str,
    ordered_source: Iterable[int],
    winner_relative_seat: int,
) -> HorseSelection:
    """从完整已记录来源选前N张；不洗牌、不排序、不从两个牌源混补。"""
    cards = _source_cards(plan, ordered_source)
    return HorseSelection(
        plan,
        source_id,
        cards[:plan.required],
        winner_relative_seat,
    )


@dataclass(frozen=True, slots=True)
class SharedHorseSelection:
    """一炮多响共用同一批马牌；每个赢家按自己的本局相对位置计算命中。"""

    plan: HorsePlan
    source_id: str
    cards: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.plan.scenario is not HorseScenario.MULTI_WIN:
            raise RuleError("horse.scenario")
        text(self.source_id, "source_id")
        cards = tuple(self.cards)
        if len(cards) != self.plan.required:
            raise RuleError("horse.source")
        if self.plan.source is HorseSource.SEPARATE_DECK:
            for start in range(0, len(cards), DECK_SIZE):
                validate_tiles(cards[start : start + DECK_SIZE])
        else:
            validate_tiles(cards)
        object.__setattr__(self, "cards", cards)

    def hits_for(self, winner_relative_seat: int) -> tuple[int, ...]:
        """返回指定赢家的命中实体；相同牌种多张分别计数。"""
        seat = integer(winner_relative_seat, "winner_relative_seat", 0, PLAYERS - 1)
        return tuple(card for card in self.cards if horse_owner(card) == seat)

    def hit_count_for(self, winner_relative_seat: int) -> int:
        """返回指定赢家的m。"""
        return len(self.hits_for(winner_relative_seat))


def select_shared_horses(
    plan: HorsePlan,
    source_id: str,
    ordered_source: Iterable[int],
) -> SharedHorseSelection:
    """选择多响统一马牌；旧墙不足时来源必须是完整新牌序而非下局手牌。"""
    cards = _source_cards(plan, ordered_source)
    return SharedHorseSelection(plan, source_id, cards[:plan.required])
