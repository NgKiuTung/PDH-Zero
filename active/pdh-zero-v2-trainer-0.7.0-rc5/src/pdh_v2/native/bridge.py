"""PDH研究代码：批量打包合法投影；不读取他家暗牌或未来牌墙。"""
from __future__ import annotations

import numpy as np

from pdh_v2.game.actions import ActionKind
from pdh_v2.game.state import Phase
from pdh_v2.native.runtime import load_extension
from pdh_v2.tiles import MeldKind

PACKED_STATE = 1243
_ACTION_INDEX = {kind: i for i, kind in enumerate(ActionKind)}
_MELD_INDEX = {kind: i for i, kind in enumerate(MeldKind)}
_PHASE_INDEX = {Phase.ACTION: 0, Phase.DISCARD_RESPONSE: 1, Phase.ADDED_KONG_RESPONSE: 2}


def pack_state(session, player: int) -> np.ndarray:
    """固定布局仅承载当前玩家可见状态；长度超限拒绝而非截断。"""
    hand = session.current_hand
    result = np.full(PACKED_STATE, -1, dtype=np.int32)
    result[:11] = (player, session.dealer_state.player, _PHASE_INDEX[hand.phase],
        hand.response.source_player if hand.response is not None else -1,
        hand.pending_discard.tile if hand.pending_discard is not None else -1,
        hand.pending_added_kong.tile if hand.pending_added_kong is not None else -1,
        hand.wall.remaining, min(session.dealer_state.continuation, 16),
        int(session.dealer_state.has_won), int(hand.pass_win_locked[player]),
        {"FRONT": 0, "BACK": 1}.get(hand.last_draw.source, 2))
    own = hand.players[player].concealed
    if len(own) > 14:
        raise ValueError("native.input.own_size")
    result[16] = len(own)
    result[17:17 + len(own)] = own
    for seat, state in enumerate(hand.players):
        start = 31 + 303 * seat
        discards, reveals = state.discards, tuple(state.revealed_retained)
        if max(len(discards), len(reveals)) > 136 or len(state.melds) > 4:
            raise ValueError("native.input.public_size")
        result[start] = len(discards)
        result[start+1:start+1+len(discards)] = discards
        result[start+137] = len(reveals)
        result[start+138:start+138+len(reveals)] = reveals
        result[start+274] = len(state.melds)
        for j, record in enumerate(state.melds):
            offset = start + 275 + 7*j
            meld = record.meld
            result[offset:offset+3] = (_MELD_INDEX[meld.kind], -1, len(meld.tiles))
            result[offset+3:offset+3+len(meld.tiles)] = meld.tiles
    return result


def encode_many(sessions, players, action_groups):
    """一个native调用处理所有待决策牌桌，返回float64以保留参考计算精度。"""
    from pdh_v2.observation.encoder import DecisionView
    if not sessions or not len(sessions) == len(players) == len(action_groups):
        raise ValueError("native.batch.empty_or_mismatch")
    states = np.stack([pack_state(s, p) for s, p in zip(sessions, players)])
    counts = [len(actions) for actions in action_groups]
    offsets = np.array([0, *np.cumsum(counts)], dtype=np.int32)
    actions = np.full((int(offsets[-1]), 7), -1, dtype=np.int32)
    row = 0
    for group in action_groups:
        for action in group:
            actions[row, :3] = (_ACTION_INDEX[action.kind], action.player,
                               -1 if action.meld_index is None else action.meld_index)
            actions[row, 3:3+len(action.tiles)] = action.tiles
            row += 1
    observations, features, scores = load_extension().encode_batch(states, actions, offsets)
    # NOTE: 独立输出所有权；下一次调用不会复写当前GRU轨迹引用。
    return tuple(DecisionView(player, observations[i], group,
                              features[offsets[i]:offsets[i+1]],
                              scores[offsets[i]:offsets[i+1]])
                 for i, (player, group) in enumerate(zip(players, action_groups)))
