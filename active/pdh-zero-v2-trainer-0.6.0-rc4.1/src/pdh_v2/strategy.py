"""合法信息范围内的多路线与公开风险特征。

本模块只使用当前玩家私有手牌和已经公开的事件。规则资格、结构潜力和动作造成的
不可逆变化被显式编码，避免策略只能依赖极稀疏终局回报反推碰杠代价。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Iterable

from pdh_v2.config import PLAYERS, TILE_TYPES
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.session import GameSession
from pdh_v2.qualifications import RouteQualifications
from pdh_v2.tiles import Meld, MeldKind, tile_type

ROUTE_NAMES = (
    "SEVEN_PAIRS",
    "PONG_PONG",
    "MIXED_ONE_SUIT",
    "PURE_ONE_SUIT",
    "TERMINALS_HONORS",
    "THIRTEEN_ORPHANS",
    "DRAGONS",
    "WINDS",
)
ROUTE_FEATURE_DIM = len(ROUTE_NAMES)
QUALIFICATION_FEATURE_DIM = 4
ACTION_STRATEGY_FEATURE_DIM = (
    ROUTE_FEATURE_DIM * 2 + QUALIFICATION_FEATURE_DIM * 2 + 4
)
OBSERVATION_STRATEGY_FEATURE_DIM = (
    ROUTE_FEATURE_DIM + QUALIFICATION_FEATURE_DIM + TILE_TYPES + PLAYERS
)

_ORPHAN_TYPES = frozenset((0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33))


@dataclass(frozen=True, slots=True)
class RouteState:
    """当前牌路结构潜力与历史资格。"""

    potentials: tuple[float, ...]
    qualifications: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """一次决策内可复用的合法信息，避免为每个候选动作重复扫描公开区。"""

    player: int
    before: RouteState
    known_fractions: tuple[float, ...]
    threats: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SimulatedActionState:
    """候选动作执行后的牌路状态与公开风险摘要。"""

    route: RouteState
    known_fraction: float
    public_risk: float
    is_terminal_or_honor: float
    option_value: float


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _structural_counts(cards: Iterable[int], meld_types: Iterable[int]) -> list[int]:
    counts = [0] * TILE_TYPES
    for card in cards:
        counts[tile_type(card)] += 1
    # 杠在胡形中仍只占一个刻子面子，结构特征按三张计。
    for kind in meld_types:
        counts[kind] += 3
    return counts


def _qualification_values(meld_kinds: Iterable[MeldKind]) -> tuple[float, ...]:
    kinds = tuple(meld_kinds)
    no_melds = not kinds
    return (
        float(no_melds),
        float(no_melds),
        float(all(kind is MeldKind.CLOSED_KONG for kind in kinds)),
        float(no_melds),
    )


def qualification_values(value: RouteQualifications) -> tuple[float, ...]:
    """将规则资格对象转换为稳定的模型特征顺序。"""
    return (
        float(value.closed_bonus_history_ok),
        float(value.seven_pairs_history_ok),
        float(value.four_concealed_history_ok),
        float(value.thirteen_orphans_history_ok),
    )


def route_potentials(cards: Iterable[int], meld_types: Iterable[int]) -> tuple[float, ...]:
    """返回八条可并存路线的结构潜力，范围约为[-1, 1]。

    这些值是规则一致的结构特征，不是精确完成概率或向听数。未见牌仍可能在对手
    手牌中，因此本函数不会把公开剩余上界伪装为牌墙真实余量。
    """
    meld_types_tuple = tuple(meld_types)
    counts = _structural_counts(cards, meld_types_tuple)
    pairs = sum(value // 2 for value in counts)
    triplets = sum(value // 3 for value in counts)
    quads = sum(value == 4 for value in counts)
    declared_groups = len(meld_types_tuple)

    suits = [sum(counts[start : start + 9]) for start in (0, 9, 18)]
    honors = sum(counts[27:])
    dominant = max(suits)
    off_suit = sum(suits) - dominant
    terminal_honor_count = sum(counts[index] for index in _ORPHAN_TYPES)
    other_numbers = sum(counts) - terminal_honor_count

    seven_pairs = -1.0
    orphans = -1.0
    if declared_groups == 0:
        seven_pairs = _clamp((pairs + 0.35 * quads) / 7.0)
        unique_orphans = sum(counts[index] > 0 for index in _ORPHAN_TYPES)
        orphan_pair = any(counts[index] >= 2 for index in _ORPHAN_TYPES)
        orphans = _clamp((unique_orphans / 13.0) * 0.9 + float(orphan_pair) * 0.1)

    pong_pong = _clamp((min(triplets, 4) / 4.0) * 0.85 + float(pairs > 0) * 0.15)
    mixed = tanh((dominant + honors - 2.0 * off_suit) / 8.0)
    pure = tanh((dominant - 2.0 * off_suit - 2.0 * honors) / 8.0)
    terminals = tanh((terminal_honor_count - 2.0 * other_numbers) / 8.0)
    dragons = sum(min(counts[index], 3) / 3.0 for index in (31, 32, 33)) / 3.0
    winds = sum(min(counts[index], 3) / 3.0 for index in (27, 28, 29, 30)) / 4.0
    return (
        seven_pairs,
        pong_pong,
        _clamp(mixed),
        _clamp(pure),
        _clamp(terminals),
        orphans,
        _clamp(dragons),
        _clamp(winds),
    )


def _state_parts(
    session: GameSession,
    player: int,
) -> tuple[list[int], list[int], list[MeldKind]]:
    state = session.current_hand.players[player]
    cards = list(state.concealed)
    meld_types = [tile_type(record.meld.tiles[0]) for record in state.melds]
    meld_kinds = [record.meld.kind for record in state.melds]
    return cards, meld_types, meld_kinds


def route_state(cards: Iterable[int], melds: Iterable[Meld]) -> RouteState:
    """从显式手牌和面子构建路线状态，供规则测试与离线诊断使用。"""
    meld_tuple = tuple(melds)
    meld_types = [tile_type(meld.tiles[0]) for meld in meld_tuple]
    meld_kinds = [meld.kind for meld in meld_tuple]
    return RouteState(route_potentials(cards, meld_types), _qualification_values(meld_kinds))


def current_route_state(session: GameSession, player: int) -> RouteState:
    """读取当前玩家的合法路线状态。"""
    cards, meld_types, meld_kinds = _state_parts(session, player)
    return RouteState(route_potentials(cards, meld_types), _qualification_values(meld_kinds))


def known_tile_fractions(session: GameSession, player: int) -> tuple[float, ...]:
    """返回当前玩家可知的各牌种实体占比，公开手牌实体只计一次。"""
    hand = session.current_hand
    known: set[int] = set(hand.players[player].concealed)
    for state in hand.players:
        known.update(state.discards)
        known.update(state.revealed_retained)
        for record in state.melds:
            known.update(record.meld.tiles)
    counts = [0] * TILE_TYPES
    for card in known:
        counts[tile_type(card)] += 1
    return tuple(value / 4.0 for value in counts)


def opponent_threats(session: GameSession, player: int) -> tuple[float, ...]:
    """返回相对座位的公开威胁量；仅基于副露与公开留手。"""
    hand = session.current_hand
    result: list[float] = []
    for offset in range(PLAYERS):
        seat = (player + offset) % PLAYERS
        if seat == player:
            result.append(0.0)
            continue
        state = hand.players[seat]
        meld_pressure = len(state.melds) / 4.0
        reveal_pressure = min(1.0, len(state.revealed_retained) / 6.0)
        result.append(_clamp(0.8 * meld_pressure + 0.2 * reveal_pressure))
    return tuple(result)


def strategy_context(session: GameSession, player: int) -> StrategyContext:
    """构建一次决策内共享的多路线与公开信息上下文。"""
    return StrategyContext(
        player,
        current_route_state(session, player),
        known_tile_fractions(session, player),
        opponent_threats(session, player),
    )


def _simulate_action(
    session: GameSession,
    action: Action,
) -> tuple[list[int], list[int], list[MeldKind]]:
    cards, meld_types, meld_kinds = _state_parts(session, action.player)
    if action.kind is ActionKind.DISCARD:
        cards.remove(action.tiles[0])
    elif action.kind in (ActionKind.PONG, ActionKind.OPEN_KONG):
        for card in action.tiles:
            cards.remove(card)
        pending = session.current_hand.pending_discard
        if pending is None:
            raise RuntimeError("discard claim has no pending discard")
        meld_types.append(tile_type(pending.tile))
        meld_kinds.append(
            MeldKind.PONG if action.kind is ActionKind.PONG else MeldKind.OPEN_KONG
        )
    elif action.kind is ActionKind.CLOSED_KONG:
        for card in action.tiles:
            cards.remove(card)
        meld_types.append(tile_type(action.tiles[0]))
        meld_kinds.append(MeldKind.CLOSED_KONG)
    elif action.kind is ActionKind.ADDED_KONG:
        cards.remove(action.tiles[0])
        if action.meld_index is None:
            raise RuntimeError("added kong has no meld index")
        meld_kinds[action.meld_index] = MeldKind.ADDED_KONG
    elif action.kind not in (ActionKind.WIN, ActionKind.PASS):
        raise RuntimeError(f"unsupported action kind: {action.kind}")
    return cards, meld_types, meld_kinds


def public_discard_risk(
    session: GameSession,
    player: int,
    kind: int,
    known_fractions: tuple[float, ...] | None = None,
) -> float:
    """按公开副露和已知实体给出弃牌风险先验，不读取任何暗牌。"""
    known = (known_fractions or known_tile_fractions(session, player))[kind]
    unknown_fraction = 1.0 - known
    tile_suit = kind // 9 if kind < 27 else 3
    threat = 0.0
    for seat, state in enumerate(session.current_hand.players):
        if seat == player or not state.melds:
            continue
        meld_types = [tile_type(record.meld.tiles[0]) for record in state.melds]
        aligned = sum(
            (value // 9 if value < 27 else 3) == tile_suit for value in meld_types
        ) / len(meld_types)
        pressure = len(state.melds) / 4.0
        threat = max(threat, pressure * (0.5 + 0.5 * aligned))
    return max(0.0, min(1.0, unknown_fraction * threat))


def simulate_action_state(
    session: GameSession,
    action: Action,
    context: StrategyContext | None = None,
) -> SimulatedActionState:
    """构建候选动作后的路线、资格损失与公开风险摘要。"""
    shared = context or strategy_context(session, action.player)
    if shared.player != action.player:
        raise ValueError("strategy context belongs to another player")
    cards, meld_types, meld_kinds = _simulate_action(session, action)
    after = RouteState(route_potentials(cards, meld_types), _qualification_values(meld_kinds))
    selected_type = tile_type(action.tiles[0]) if action.tiles else None
    known_fraction = 0.0 if selected_type is None else shared.known_fractions[selected_type]
    risk = 0.0
    terminal_or_honor = 0.0
    if selected_type is not None:
        risk = public_discard_risk(
            session,
            action.player,
            selected_type,
            shared.known_fractions,
        )
        terminal_or_honor = float(selected_type >= 27 or selected_type % 9 in (0, 8))
    ordered = sorted(after.potentials, reverse=True)
    option_value = ordered[0] + 0.35 * ordered[1] + 0.15 * ordered[2]
    return SimulatedActionState(
        after,
        known_fraction,
        risk,
        terminal_or_honor,
        _clamp(option_value / 1.5),
    )


def action_strategy_features(
    session: GameSession,
    action: Action,
    context: StrategyContext | None = None,
) -> tuple[float, ...]:
    """返回候选动作的多路线变化、资格损失和公开风险特征。"""
    shared = context or strategy_context(session, action.player)
    simulated = simulate_action_state(session, action, shared)
    delta = tuple(
        after - prior
        for prior, after in zip(shared.before.potentials, simulated.route.potentials)
    )
    lost = tuple(
        max(0.0, prior - after)
        for prior, after in zip(
            shared.before.qualifications,
            simulated.route.qualifications,
        )
    )
    result = (
        simulated.route.potentials
        + delta
        + simulated.route.qualifications
        + lost
        + (
            simulated.known_fraction,
            simulated.public_risk,
            simulated.is_terminal_or_honor,
            simulated.option_value,
        )
    )
    if len(result) != ACTION_STRATEGY_FEATURE_DIM:
        raise RuntimeError("action strategy feature dimension mismatch")
    return result


def observation_strategy_features(
    session: GameSession,
    player: int,
    context: StrategyContext | None = None,
) -> tuple[float, ...]:
    """返回当前玩家的路线、资格、已知实体和公开威胁特征。"""
    shared = context or strategy_context(session, player)
    if shared.player != player:
        raise ValueError("strategy context belongs to another player")
    result = (
        shared.before.potentials
        + shared.before.qualifications
        + shared.known_fractions
        + shared.threats
    )
    if len(result) != OBSERVATION_STRATEGY_FEATURE_DIM:
        raise RuntimeError("observation strategy feature dimension mismatch")
    return result
