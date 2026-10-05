"""参考手局的规范状态摘要；供牌谱哈希与差分测试使用。"""
from dataclasses import asdict
from typing import Any

from pdh_v2.game.actions import Action


def _action_data(action: Action) -> dict[str, object]:
    """把动作转换为稳定的JSON兼容结构。"""
    return {
        "player": action.player,
        "kind": action.kind.value,
        "tiles": list(action.tiles),
        "meld_index": action.meld_index,
    }


def hand_state_digest(hand: Any) -> dict[str, object]:
    """返回影响后续合法行为或结算的完整状态摘要。"""
    response = None
    if hand.response is not None:
        response = {
            "source_player": hand.response.source_player,
            "queue": list(hand.response.queue),
            "responses": {
                str(player): _action_data(action)
                for player, action in sorted(hand.response.responses.items())
            },
        }
    return {
        "hand_id": hand.hand_id,
        "dealer": hand.dealer,
        "phase": hand.phase.value,
        "current_player": hand.current_player,
        "wall_front": hand.wall.front,
        "wall_back": hand.wall.back,
        "last_draw": asdict(hand.last_draw),
        "pending_discard": (
            None if hand.pending_discard is None else asdict(hand.pending_discard)
        ),
        "pending_added_kong": (
            None
            if hand.pending_added_kong is None
            else asdict(hand.pending_added_kong)
        ),
        "response": response,
        "kong_context": (
            None if hand.kong_context is None else asdict(hand.kong_context)
        ),
        "twelve_liabilities": {
            str(player): asdict(liability)
            for player, liability in sorted(hand.twelve_liabilities.items())
        },
        "pass_win_locked": list(hand.pass_win_locked),
        "players": [
            {
                "concealed": list(player.concealed),
                "melds": [
                    {
                        "kind": item.meld.kind.value,
                        "tiles": list(item.meld.tiles),
                        "provider": item.provider,
                    }
                    for item in player.melds
                ],
                "discards": list(player.discards),
                "revealed_retained": sorted(player.revealed_retained),
            }
            for player in hand.players
        ],
        "outcome": None
        if hand.outcome is None
        else {
            "method": hand.outcome.method,
            "winners": [
                {
                    "player": item.player,
                    "score": {
                        "multiplier": item.score.multiplier,
                        "items": [
                            {
                                "pattern": score_item.pattern.value,
                                "multiplier": score_item.multiplier,
                            }
                            for score_item in item.score.items
                        ],
                        "suppressed": [
                            {
                                "pattern": suppressed.pattern.value,
                                "reason": suppressed.reason,
                            }
                            for suppressed in item.score.suppressed
                        ],
                        "interpretation": item.score.interpretation,
                        "priority": item.score.priority,
                    },
                }
                for item in hand.outcome.winners
            ],
            "discarder": hand.outcome.discarder,
            "draw": hand.outcome.draw,
            "wall_source": hand.outcome.wall_source,
            "old_wall": list(hand.outcome.old_wall),
            "twelve_grounded": (
                None
                if hand.outcome.twelve_grounded is None
                else asdict(hand.outcome.twelve_grounded)
            ),
            "kong_provider": hand.outcome.kong_provider,
        },
    }
