"""更新边界的连续会话快照与恢复。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pdh_v2.dealer import DealerState
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.session import GameSession
from pdh_v2.game.state import Phase
from pdh_v2.ledger import SettlementBook
from pdh_v2.rollout.vector_env import BatchedSessions


def _assert_update_boundary(session: GameSession) -> None:
    """验证会话停在已发好下一手、尚未产生策略动作的边界。"""
    hand = session.current_hand
    if hand.phase is not Phase.ACTION or hand.current_player != hand.dealer:
        raise RuntimeError("session is not at an update boundary")
    if hand.dealer != session.dealer_state.player:
        raise RuntimeError("session dealer and current hand dealer differ")
    if hand.pending_discard is not None or hand.pending_added_kong is not None:
        raise RuntimeError("pending response cannot be checkpointed")
    if hand.response is not None or hand.outcome is not None:
        raise RuntimeError("terminal or response state cannot be checkpointed")
    if any(player.melds or player.discards for player in hand.players):
        raise RuntimeError("started hand cannot be checkpointed as a clean boundary")
    if any(hand.pass_win_locked) or hand.twelve_liabilities:
        raise RuntimeError("hand liabilities must be empty at an update boundary")


def snapshot_session(session: GameSession) -> dict[str, Any]:
    """保存可无损恢复的更新边界会话状态。"""
    _assert_update_boundary(session)
    hand = session.current_hand
    return {
        "schema_version": 1,
        "seed": session.seed,
        "wall_rng_state": session.wall_rng.getstate(),
        "horse_rng_state": session.horse_rng.getstate(),
        "dealer_state": asdict(session.dealer_state),
        "scores": list(session.scores),
        "hand_index": session.hand_index,
        "current_hand": {
            "hand_id": hand.hand_id,
            "dealer": hand.dealer,
            "wall_order": list(hand.wall.order),
            "revealed_retained": [
                sorted(player.revealed_retained) for player in hand.players
            ],
        },
    }


def restore_session(payload: dict[str, Any]) -> GameSession:
    """从受信任的长任务检查点恢复一个连续会话。"""
    if payload.get("schema_version") != 1:
        raise RuntimeError("session snapshot schema mismatch")
    dealer_payload = dict(payload["dealer_state"])
    dealer = DealerState(
        int(dealer_payload["player"]),
        bool(dealer_payload["has_won"]),
        int(dealer_payload["continuation"]),
    )
    session = GameSession(int(payload["seed"]), dealer.player)
    session.wall_rng.setstate(payload["wall_rng_state"])
    session.horse_rng.setstate(payload["horse_rng_state"])
    session.dealer_state = dealer
    session.scores = [int(value) for value in payload["scores"]]
    session.book = SettlementBook()
    session.hand_index = int(payload["hand_index"])
    session.events = []

    current = dict(payload["current_hand"])
    hand = MahjongHand(
        str(current["hand_id"]),
        int(current["dealer"]),
        tuple(int(value) for value in current["wall_order"]),
    )
    reveals = list(current["revealed_retained"])
    if len(reveals) != len(hand.players):
        raise RuntimeError("revealed retained seat count mismatch")
    for player, cards in zip(hand.players, reveals):
        player.revealed_retained.update(int(card) for card in cards)
    hand.validate_invariants()
    session.current_hand = hand
    _assert_update_boundary(session)
    return session


def snapshot_batched(environments: BatchedSessions) -> tuple[dict[str, Any], ...]:
    """保存全部牌桌；只允许在采样已排空时调用。"""
    if not environments.all_inactive():
        raise RuntimeError("all environments must be inactive before checkpoint")
    return tuple(snapshot_session(session) for session in environments.sessions)


def restore_batched(payload: tuple[dict[str, Any], ...]) -> BatchedSessions:
    """恢复全部牌桌，并保持暂停状态直到下一次采样。"""
    if not payload:
        raise RuntimeError("empty environment snapshot")
    restored = object.__new__(BatchedSessions)
    restored.sessions = [restore_session(dict(item)) for item in payload]
    restored.active = [False] * len(restored.sessions)
    restored._views = {}
    return restored
