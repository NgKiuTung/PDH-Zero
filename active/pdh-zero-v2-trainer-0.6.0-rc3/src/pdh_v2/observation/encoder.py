"""合法 Actor 观察与候选动作特征。

观察只包含当前玩家私有手牌以及截至当前时刻的公开状态。审计真值中的其他玩家
暗牌、未来牌墙和未来抓马结果不会进入该模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pdh_v2.config import PLAYERS, TILE_TYPES
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.session import GameSession
from pdh_v2.game.state import Phase
from pdh_v2.strategy import (
    ACTION_STRATEGY_FEATURE_DIM,
    OBSERVATION_STRATEGY_FEATURE_DIM,
    StrategyContext,
    action_strategy_features,
    observation_strategy_features,
    strategy_context,
)
from pdh_v2.tiles import tile_type

ACTION_KINDS = tuple(ActionKind)
ACTION_KIND_INDEX = {kind: index for index, kind in enumerate(ACTION_KINDS)}
PHASES = (Phase.ACTION, Phase.DISCARD_RESPONSE, Phase.ADDED_KONG_RESPONSE)
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}

# 牌种计数特征均按每种最多四张归一化。
_BASE_OBSERVATION_DIM = 34 + 34 + PLAYERS * 34 * 3 + 4 + 3 + 4 + 34 + 34 + 17
_BASE_ACTION_FEATURE_DIM = len(ACTION_KINDS) + TILE_TYPES + 1 + 4
OBSERVATION_DIM = _BASE_OBSERVATION_DIM + OBSERVATION_STRATEGY_FEATURE_DIM
ACTION_FEATURE_DIM = _BASE_ACTION_FEATURE_DIM + ACTION_STRATEGY_FEATURE_DIM


def _counts(cards: Iterable[int]) -> list[float]:
    values = [0.0] * TILE_TYPES
    for card in cards:
        values[tile_type(card)] += 0.25
    return values


def _one_hot(index: int | None, size: int) -> list[float]:
    values = [0.0] * size
    if index is not None:
        values[index] = 1.0
    return values


def acting_player(session: GameSession) -> int:
    """返回当前唯一需要提交动作的玩家。"""
    hand = session.current_hand
    if hand.phase is Phase.ACTION:
        return hand.current_player
    if hand.response is None or not hand.response.queue:
        raise RuntimeError("response phase has no responder")
    return hand.response.queue[0]


def encode_observation(
    session: GameSession,
    player: int,
    context: StrategyContext | None = None,
) -> tuple[float, ...]:
    """构建固定长度合法观察。

    座位相关公开信息以当前玩家为相对0号旋转，减少固定玩家编号带来的无意义
    非对称；庄家位置仍单独编码。
    """
    hand = session.current_hand
    own = hand.players[player]
    features: list[float] = []
    features.extend(_counts(own.concealed))
    features.extend(_counts(card for card in own.concealed if card in own.revealed_retained))

    for offset in range(PLAYERS):
        seat = (player + offset) % PLAYERS
        state = hand.players[seat]
        features.extend(_counts(state.discards))
        features.extend(_counts(card for record in state.melds for card in record.meld.tiles))
        features.extend(_counts(state.revealed_retained))

    dealer_relative = (session.dealer_state.player - player) % PLAYERS
    features.extend(_one_hot(dealer_relative, PLAYERS))
    features.extend(_one_hot(PHASE_INDEX.get(hand.phase), len(PHASES)))

    source_player: int | None = None
    if hand.response is not None:
        source_player = (hand.response.source_player - player) % PLAYERS
    features.extend(_one_hot(source_player, PLAYERS))

    pending_discard_type = (
        None if hand.pending_discard is None else tile_type(hand.pending_discard.tile)
    )
    features.extend(_one_hot(pending_discard_type, TILE_TYPES))
    pending_kong_type = (
        None if hand.pending_added_kong is None else tile_type(hand.pending_added_kong.tile)
    )
    features.extend(_one_hot(pending_kong_type, TILE_TYPES))

    # 这些标量都是当前玩家可知的公开事实或自身状态。
    features.extend(
        [
            hand.wall.remaining / 136.0,
            min(session.dealer_state.continuation, 16) / 16.0,
            float(session.dealer_state.has_won),
            float(hand.pass_win_locked[player]),
            float(hand.last_draw.source == "FRONT"),
            float(hand.last_draw.source == "BACK"),
            len(own.melds) / 4.0,
            len(own.discards) / 30.0,
            len(own.revealed_retained) / 14.0,
        ]
    )
    for offset in range(PLAYERS):
        seat = (player + offset) % PLAYERS
        features.append(len(hand.players[seat].melds) / 4.0)
        features.append(len(hand.players[seat].discards) / 30.0)

    features.extend(observation_strategy_features(session, player, context))
    if len(features) != OBSERVATION_DIM:
        raise RuntimeError(f"observation dimension mismatch: {len(features)}")
    return tuple(features)


def encode_action(
    session: GameSession,
    action: Action,
    context: StrategyContext | None = None,
) -> tuple[float, ...]:
    """把一个真实实体动作编码为固定长度候选特征。"""
    kind_features = _one_hot(ACTION_KIND_INDEX[action.kind], len(ACTION_KINDS))
    tile_features = _counts(action.tiles)
    selected_revealed = sum(
        card in session.current_hand.players[action.player].revealed_retained
        for card in action.tiles
    )
    revealed_fraction = selected_revealed / max(1, len(action.tiles))
    meld_index = _one_hot(action.meld_index, 4)
    result = tuple(
        kind_features
        + tile_features
        + [revealed_fraction]
        + meld_index
        + list(action_strategy_features(session, action, context))
    )
    if len(result) != ACTION_FEATURE_DIM:
        raise RuntimeError(f"action dimension mismatch: {len(result)}")
    return result


@dataclass(frozen=True, slots=True)
class DecisionView:
    """一次策略决策所需的合法观察、候选动作及其特征。"""

    player: int
    observation: tuple[float, ...]
    actions: tuple[Action, ...]
    action_features: tuple[tuple[float, ...], ...]


def decision_view(session: GameSession) -> DecisionView:
    """读取当前策略决策；终局状态不允许请求动作。"""
    player = acting_player(session)
    actions = session.current_hand.legal_actions(player)
    if not actions:
        raise RuntimeError("active player has no legal action")
    context = strategy_context(session, player)
    return DecisionView(
        player,
        encode_observation(session, player, context),
        actions,
        tuple(encode_action(session, action, context) for action in actions),
    )
