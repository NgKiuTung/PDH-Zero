"""PDH研究代码：真实/合成牌谱逐决策差分；数值容差不能代替动作一致性。"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from pdh_v2.dealer import DealerState
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.session import GameSession
from pdh_v2.native.bridge import encode_many
from pdh_v2.observation.encoder import decision_view_reference
from pdh_v2.policy.heuristic import score_action
from pdh_v2.replay.audit import audit_terminal_record
from pdh_v2.strategy import strategy_context


def compare_view(session) -> dict:
    """同一合法状态对照622/74维特征、全部教师分数及平分选择。"""
    left = decision_view_reference(session)
    right = encode_many((session,), (left.player,), (left.actions,))[0]
    context = strategy_context(session, left.player)
    expected = np.array([score_action(session, a, context) for a in left.actions])
    for a, b in ((left.observation, right.observation),
                 (left.action_features, right.action_features), (expected, right.teacher_scores)):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-12)
    if int(expected.argmax()) != int(right.teacher_scores.argmax()):
        raise AssertionError("native.heuristic.action_mismatch")
    return {"actions": len(left.actions), "max_abs_error": max(
        float(np.max(np.abs(np.asarray(left.observation)-right.observation))),
        float(np.max(np.abs(np.asarray(left.action_features)-right.action_features))),
        float(np.max(np.abs(expected-right.teacher_scores))))}


def verify_records(path: Path, limit: int = 16) -> dict:
    """只读gzip牌谱；有限样本逐步重放，账本/终局摘要另做独立完整校验。"""
    if limit < 1:
        raise ValueError("native.validation.positive_limit")
    hands = steps = actions = 0
    error = 0.0
    kinds = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            t = record["transcript"]
            session = GameSession(0)
            session.dealer_state = DealerState(**record["audit"]["before_dealer"])
            session.current_hand = MahjongHand(t["hand_id"], t["dealer"], tuple(t["wall"]))
            for player, revealed in zip(session.current_hand.players,
                                       record["audit"]["initial_revealed_retained"]):
                player.revealed_retained.update(revealed)
            for item in t["actions"]:
                result = compare_view(session)
                steps += 1
                actions += result["actions"]
                error = max(error, result["max_abs_error"])
                action = Action(item["player"], ActionKind(item["kind"]),
                                tuple(item["tiles"]), item["meld_index"])
                kinds.add(action.kind.value)
                session.current_hand.step(action)
            digest = hashlib.sha256(json.dumps(session.current_hand.state_digest(),
                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if digest != t["state_sha256"]:
                raise AssertionError("native.replay.final_digest_mismatch")
            audit_terminal_record(record)
            hands += 1
            if hands == limit:
                break
    if not hands:
        raise ValueError("native.validation.empty")
    return {"status": "PASS", "hands": hands, "decisions": steps,
            "candidate_actions": actions, "max_abs_error": error,
            "observed_action_kinds": sorted(kinds), "atol": 2e-12,
            "scope": "sampled_replays_not_all_possible_states"}
