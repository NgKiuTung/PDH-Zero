"""无真人牌谱的确定性启发式教师。

教师只使用自身手牌和公开信息。基础打牌强度来自可解释的多路线结构势能；新增
资格与风险项只作小幅校正，避免防守权重过大导致四家长期不成牌。
"""
from __future__ import annotations

from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.session import GameSession
from pdh_v2.strategy import StrategyContext, simulate_action_state, strategy_context
from pdh_v2.tiles import tile_type


def _potential(cards: list[int], meld_types: list[int], declared_groups: int) -> float:
    counts = [0] * 34
    for card in cards:
        counts[tile_type(card)] += 1
    for kind in meld_types:
        counts[kind] += 3

    pairs = sum(value // 2 for value in counts)
    triplets = sum(value // 3 for value in counts) + declared_groups
    quads = sum(value == 4 for value in counts)
    suit_counts = [sum(counts[start : start + 9]) for start in (0, 9, 18)]
    honors = sum(counts[27:])
    dominant = max(suit_counts)
    off_suit = sum(suit_counts) - dominant
    terminals_honors = honors + sum(counts[index] for index in (0, 8, 9, 17, 18, 26))
    other_numbers = sum(counts) - terminals_honors

    seven_pairs = -1000.0 if declared_groups else pairs * 4.0 + quads * 2.0
    pong_pong = triplets * 5.0 + pairs * 1.2
    mixed_suit = dominant * 1.8 + honors * 1.4 - off_suit * 3.0
    pure_suit = dominant * 2.0 - off_suit * 4.0 - honors * 2.5
    terminals = terminals_honors * 2.2 - other_numbers * 4.0
    return max(seven_pairs, pong_pong, mixed_suit, pure_suit, terminals)


def _state_parts(session: GameSession, player: int) -> tuple[list[int], list[int], int]:
    state = session.current_hand.players[player]
    cards = list(state.concealed)
    meld_types = [tile_type(record.meld.tiles[0]) for record in state.melds]
    return cards, meld_types, len(state.melds)


def score_action(
    session: GameSession,
    action: Action,
    context: StrategyContext | None = None,
) -> float:
    """返回候选动作的确定性教师分数。"""
    if action.kind is ActionKind.WIN:
        return 1_000_000.0
    shared = context or strategy_context(session, action.player)
    simulated = simulate_action_state(session, action, shared)
    hand = session.current_hand
    state = hand.players[action.player]
    cards, meld_types, declared_groups = _state_parts(session, action.player)

    if action.kind is ActionKind.PASS:
        return _potential(cards, meld_types, declared_groups)
    if action.kind is ActionKind.DISCARD:
        cards.remove(action.tiles[0])
        reveal_bonus = 0.2 if action.tiles[0] in state.revealed_retained else 0.0
        return (
            _potential(cards, meld_types, declared_groups)
            + reveal_bonus
            + 0.08 * simulated.option_value
            + 0.08 * simulated.known_fraction
            - 0.35 * simulated.public_risk
        )
    if action.kind in (ActionKind.PONG, ActionKind.OPEN_KONG):
        for card in action.tiles:
            cards.remove(card)
        pending = hand.pending_discard
        incoming = pending.tile if pending is not None else action.tiles[0]
        meld_types.append(tile_type(incoming))
        qualification_loss = sum(
            max(0.0, before - after)
            for before, after in zip(
                shared.before.qualifications,
                simulated.route.qualifications,
            )
        )
        return (
            _potential(cards, meld_types, declared_groups + 1)
            + 0.5
            - 0.12 * qualification_loss
        )
    if action.kind is ActionKind.CLOSED_KONG:
        for card in action.tiles:
            cards.remove(card)
        meld_types.append(tile_type(action.tiles[0]))
        return _potential(cards, meld_types, declared_groups + 1) + 0.1
    if action.kind is ActionKind.ADDED_KONG:
        cards.remove(action.tiles[0])
        return _potential(cards, meld_types, declared_groups) + 0.2
    raise RuntimeError(f"unsupported action kind: {action.kind}")


def choose_action(session: GameSession, actions: tuple[Action, ...]) -> int:
    """选择最高分动作；平分时保持规则层候选顺序以确保可重放。"""
    if not actions:
        raise ValueError("actions must not be empty")
    context = strategy_context(session, actions[0].player)
    scores = [score_action(session, action, context) for action in actions]
    return max(range(len(actions)), key=scores.__getitem__)
