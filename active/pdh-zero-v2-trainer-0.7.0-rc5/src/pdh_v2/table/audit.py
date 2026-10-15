"""PDH research code: actual-weight noninterference and legal public-history pairs.

No loss function, checkpoint selection, or training is run by this audit.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import numpy as np

from pdh_v2.game.actions import ActionKind
from pdh_v2.game.session import GameSession
from pdh_v2.game.state import Phase
from pdh_v2.observation import decision_view
from pdh_v2.policy import CandidateActorCritic
from pdh_v2.strategy import public_discard_risk, strategy_context
from pdh_v2.table.data import infer, message, records, replay_views
from pdh_v2.training.refinement_control import atomic_json


def private_intervention(session: GameSession, observer: int) -> GameSession:
    """Swap unexposed opponent cards and reverse the future wall, preserving public state."""
    clone = copy.deepcopy(session)
    hand = clone.current_hand
    available = []
    for offset in (1, 2, 3):
        player = hand.players[(observer + offset) % 4]
        indices = [i for i, card in enumerate(player.concealed)
                   if card not in player.revealed_retained and card != hand.last_draw.tile]
        if indices:
            available.append((player, indices[0]))
    if len(available) >= 2:
        left, li = available[0]
        right, ri = available[1]
        left.concealed[li], right.concealed[ri] = right.concealed[ri], left.concealed[li]
    wall = hand.wall
    wall.order = (wall.order[:wall.front]
                  + tuple(reversed(wall.order[wall.front:wall.back + 1]))
                  + wall.order[wall.back + 1:])
    hand.validate_invariants()
    return clone


def _advance_without_claims(session: GameSession, target: int,
                            alternative_discard: int = 0) -> None:
    """Reach the target draw using legal discards/PASS, without consulting hidden opponents."""
    for _ in range(40):
        view = decision_view(session)
        if session.current_hand.phase is Phase.ACTION and view.player == target:
            return
        kind = (ActionKind.DISCARD if session.current_hand.phase is Phase.ACTION
                else ActionKind.PASS)
        choices = [a for a in view.actions if a.kind is kind]
        selected = min(alternative_discard, len(choices) - 1)
        session.current_hand.step(choices[selected])
        alternative_discard = 0
    raise RuntimeError(message("audit_bound"))


def public_history_pair(seed: int):
    """Two reachable histories: same own hand/draw, one opponent legally discards differently."""
    session = GameSession(seed)
    initial = decision_view(session)
    session.current_hand.step(next(a for a in initial.actions if a.kind is ActionKind.DISCARD))
    _advance_without_claims(session, 1)
    view = decision_view(session)
    by_type = {a.tiles[0] // 4: a for a in view.actions if a.kind is ActionKind.DISCARD}
    if len(by_type) < 2:
        raise ValueError(message("audit_pair_insufficient"))
    left, right = copy.deepcopy(session), copy.deepcopy(session)
    alternatives = list(by_type.values())
    left.current_hand.step(alternatives[0])
    right.current_hand.step(alternatives[1])
    _advance_without_claims(left, 0)
    _advance_without_claims(right, 0)
    return initial, decision_view(left), decision_view(right)


def run_audit(model: CandidateActorCritic, replay: Path, output: Path,
              hands: int = 16, pairs: int = 32, stride: int = 8) -> dict:
    """Audit real replays and legal paired fixtures; do not infer strength from sensitivity."""
    if min(hands, pairs, stride) < 1:
        raise ValueError(message("positive_sizes"))
    device = next(model.parameters()).device
    checks = total_actions = completed = 0
    max_logit_difference = 0.0
    closed_checks = closed_zero = 0
    for record in records(replay, hands):
        hidden = [model.initial_state(1, device) for _ in range(4)]
        for index, session, view in replay_views(record):
            original, state = infer(model, view, hidden[view.player], device)
            if index % stride == 0:
                shadow = decision_view(private_intervention(session, view.player))
                if (not np.array_equal(view.observation, shadow.observation)
                        or not np.array_equal(view.action_features, shadow.action_features)
                        or view.actions != shadow.actions):
                    raise AssertionError(message("private_input_leak"))
                changed, shadow_state = infer(model, shadow, hidden[view.player], device)
                difference = float((original - changed).abs().max())
                max_logit_difference = max(max_logit_difference, difference)
                if not torch.equal(original, changed) or not torch.equal(state, shadow_state):
                    raise AssertionError(message("private_actor_leak"))
                checks += 1
                if all(not p.melds for i, p in enumerate(session.current_hand.players)
                       if i != view.player):
                    context = strategy_context(session, view.player)
                    values = [public_discard_risk(session, view.player, k, context.known_fractions)
                              for k in range(34)]
                    closed_checks += 1
                    closed_zero += int(all(v == 0.0 for v in values))
            hidden[view.player] = state
            total_actions += 1
        completed += 1
    differences = []
    for index in range(pairs):
        initial, left, right = public_history_pair(830000001 + index)
        if left.actions != right.actions or not np.array_equal(
                left.observation[:68], right.observation[:68]):
            raise AssertionError(message("public_pair_invalid"))
        _, history = infer(model, initial, model.initial_state(1, device), device)
        a, _ = infer(model, left, history, device)
        b, _ = infer(model, right, history, device)
        pa, pb = a.softmax(-1), b.softmax(-1)
        differences.append({"pair": index,
                            "total_variation": float((pa - pb).abs().sum() / 2),
                            "argmax_changed": bool(a.argmax() != b.argmax())})
    report = {
        "status": "PASS", "complete_real_hands_replayed": completed,
        "real_action_events": total_actions, "private_interventions": checks,
        "max_private_logit_difference": max_logit_difference,
        "closed_opponent_explicit_risk_checks": closed_checks,
        "closed_opponent_explicit_risk_all_zero": closed_zero,
        "public_history_legal_fixture_pairs": differences,
        "mean_public_total_variation": sum(r["total_variation"] for r in differences) / pairs,
        "public_argmax_changes": sum(r["argmax_changed"] for r in differences),
        "strength_inference": False, "target_5090_tested": False,
        "memory_scope": "within_hand_observer_decision_snapshots_not_lossless_event_stream",
    }
    atomic_json(output, report)
    return report