"""Training/audit-only current-state labels; NEVER imported by the normal Actor.

Hidden cards define supervised targets, not inference inputs. Action forecasts are
the next non-forced behavior-policy decision, censored at hand end/horizon. They are
not intention labels and not causal responses to an unplayed candidate action.
"""
from __future__ import annotations

import math

import numpy as np

from pdh_v2.game.actions import ActionKind
from pdh_v2.observation.encoder import ACTION_KIND_INDEX
from pdh_v2.qualifications import WinMethod
from pdh_v2.scoring import ScoreContext, score_hand
from pdh_v2.strategy import route_state
from pdh_v2.tiles import WinningHand
from pdh_v2.table.model import (BELIEF_DIM, KONG, NEXT_DISCARD, NEXT_KIND, PONG,
                                READY, ROUTES, VALUE, WIN)


def waiting_capabilities(player, wall_remaining: int, pass_locked: bool):
    """Current structural big-hand waits and hypothetical discard-claim capabilities.

    This does not look for cards in the wall. Nominal score excludes unknown horses,
    dealer continuation, and other future settlement effects. Non-waiting-size
    snapshots are censored rather than falsely labeled as unable to win.
    """
    melds = tuple(record.meld for record in player.melds)
    cards = tuple(player.concealed)
    counts = np.bincount([c // 4 for c in cards], minlength=34)
    pong = (counts >= 2).astype(np.float32)
    kong = ((counts >= 3) & (wall_remaining > 0)).astype(np.float32)
    waits = np.zeros(34, dtype=np.float32)
    if len(cards) != 13 - 3 * len(melds):
        return waits, pong, kong, False, 0.0, False
    used = set(cards).union(c for meld in melds for c in meld.tiles)
    maximum = 0
    for kind in range(34):
        incoming = next((c for c in range(kind * 4, kind * 4 + 4) if c not in used), None)
        if incoming is None:
            continue
        score = score_hand(ScoreContext(WinningHand(cards + (incoming,), melds), incoming,
                                       WinMethod.DISCARD_WIN, min(136, wall_remaining + 1)))
        if score is not None:
            waits[kind] = 1.0
            maximum = max(maximum, score.multiplier)
    ready = bool(waits.any())
    if pass_locked:
        waits.fill(0.0)
    return waits, pong, kong, ready, math.log1p(maximum) / 8.0, True


def state_targets(session, observer: int):
    """Label the three relative seats; masks keep unknown/not-applicable targets out."""
    targets = np.zeros((3, BELIEF_DIM), dtype=np.float32)
    mask = np.zeros_like(targets)
    hand = session.current_hand
    for index, offset in enumerate((1, 2, 3)):
        seat = (observer + offset) % 4
        player = hand.players[seat]
        targets[index, ROUTES] = route_state(
            player.concealed, (r.meld for r in player.melds)).potentials
        mask[index, ROUTES] = 1
        waits, pong, kong, ready, value, valid = waiting_capabilities(
            player, hand.wall.remaining, hand.pass_win_locked[seat])
        targets[index, WIN], targets[index, PONG], targets[index, KONG] = waits, pong, kong
        targets[index, READY], targets[index, VALUE] = ready, value
        if valid:
            mask[index, PONG], mask[index, KONG] = 1, 1
            mask[index, READY], mask[index, WIN] = 1, 1
            mask[index, VALUE] = float(ready)
    return targets, mask


def future_behavior(targets, mask, actions: tuple, index: int, observer: int,
                    horizon: int = 24) -> None:
    """Predict the next non-forced decision and next discard; censor absent observations.

    Forced PASS dominates raw event counts but is not a strategic decision. Choice
    counts are computed by the replay's authoritative legal-action enumerator and
    used only to select targets, never appended to the current Actor input.
    """
    suffix = actions[index + 1:index + 1 + horizon]
    for relative, offset in enumerate((1, 2, 3)):
        seat = (observer + offset) % 4
        event = next((a for a in suffix if a["player"] == seat
                      and a.get('choices', 0) > 1), None)
        if event is not None:
            kind = ACTION_KIND_INDEX[ActionKind(event["kind"])]
            targets[relative, NEXT_KIND.start + kind] = 1
            mask[relative, NEXT_KIND] = 1
        discard = next((a for a in suffix if a["player"] == seat
                        and a["kind"] == ActionKind.DISCARD.value), None)
        if discard is not None:
            targets[relative, NEXT_DISCARD.start + discard["tiles"][0] // 4] = 1
            mask[relative, NEXT_DISCARD] = 1