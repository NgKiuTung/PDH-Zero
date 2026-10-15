"""PDH研究代码：成功形成的碰杠供牌链；只记录事实关联，不宣称因果责任。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from pdh_v2.dealer import DealerState
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.session import GameSession
from pdh_v2.game.state import Phase
from pdh_v2.strategy import current_route_state


def meld_signature(state) -> tuple:
    """公开面子身份；加杠是原碰的延续而不是第二次外部供牌。"""
    return tuple((m.meld.kind.value, m.meld.tiles, m.provider) for m in state.melds)


def before_meld_transition(session, action) -> dict | None:
    """只在可能实际落地的边界快照，避免每次普通弃牌重复编码牌型。"""
    hand = session.current_hand
    possible = (hand.phase is Phase.ACTION and action.kind is ActionKind.CLOSED_KONG)
    possible |= hand.response is not None and len(hand.response.queue) == 1
    if not possible:
        return None
    return {"players": tuple((meld_signature(state), tuple(state.concealed),
             asdict(current_route_state(session, i))) for i, state in enumerate(hand.players)),
            "draw_count": sum(e["event"] in {"DRAW", "SUPPLEMENT_DRAW"} for e in hand.events)}


def capture_transition(session, before: dict | None, chain: list[dict]) -> None:
    """成功状态差异才记录；被胡优先截断的碰/杠声明不会进入供牌链。"""
    if before is None:
        return
    hand = session.current_hand
    for seat, state in enumerate(hand.players):
        previous, concealed, routes = before["players"][seat]
        now = meld_signature(state)
        for index, meld in enumerate(now):
            if index < len(previous) and previous[index] == meld:
                continue
            kind, tiles, provider = meld
            added = index < len(previous)
            supplied = [t for t in tiles if t not in concealed] if not added else []
            original = next((x for x in reversed(chain)
                             if x["claimant"] == seat and x["meld_index"] == index), None)
            chain.append({"claimant": seat, "meld_index": index, "kind": kind,
                "provider": provider, "external_supply": provider is not None and not added,
                "supplied_tile": supplied[0] if provider is not None and not added else None,
                "root_supply_event": (original["root_supply_event"] if original else len(chain)),
                "transition_index": len(chain), "hand_event_index": len(hand.events)-1,
                "draw_count_before": before["draw_count"],
                "routes_before": routes, "routes_after": asdict(current_route_state(session, seat)),
                "scope": "observed_meld_association_not_causal_contribution"})


def trace_record(record: dict) -> dict:
    """对旧真实牌谱离线重放成功副露，并校验原终局摘要；不改原文件。"""
    t = record["transcript"]
    session = GameSession(0)
    session.dealer_state = DealerState(**record["audit"]["before_dealer"])
    session.current_hand = MahjongHand(t["hand_id"], t["dealer"], tuple(t["wall"]))
    for player, revealed in zip(session.current_hand.players,
                               record["audit"]["initial_revealed_retained"]):
        player.revealed_retained.update(revealed)
    chain = []
    for item in t["actions"]:
        action = Action(item["player"], ActionKind(item["kind"]),
                        tuple(item["tiles"]), item["meld_index"])
        before = before_meld_transition(session, action)
        session.current_hand.step(action)
        capture_transition(session, before, chain)
    digest = hashlib.sha256(json.dumps(session.current_hand.state_digest(),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if digest != t["state_sha256"]:
        raise ValueError("supply.replay.digest_mismatch")
    winners = record["winners"]
    return {"state_sha256": digest, "chain": chain,
        "winner_supply_events": [r for r in chain if r["external_supply"]
                                 and r["claimant"] in winners],
        "scope": "verified_replay_association_not_causal_contribution"}
