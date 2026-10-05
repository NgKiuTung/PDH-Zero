"""Deterministic four-player Mini-PDH v0 state machine."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any, Iterable

from .actions import ACTION_COUNT, DISCARD_ACTIONS, Action, action_name
from .rules import evaluate_win
from .types import COPIES_PER_TILE, PLAYERS, TILE_TYPES, Phase, StepResult, WinResult


class IllegalActionError(ValueError):
    """Raised when a controller selects an action outside the legal mask."""


class MiniPDHEnv:
    """Small deterministic Mahjong-like environment for Phase 1 feasibility tests.

    The environment exposes only one decision maker at a time. Automatic transitions
    such as drawing the next tile are deterministic and recorded in the full event log.
    """

    def __init__(self, config: dict[str, Any]):
        """Create an environment from the validated Mini-PDH configuration."""
        self.config = copy.deepcopy(config)
        self.max_decisions = int(config["max_decisions"])
        self.dealer_seat = int(config["dealer_seat"])
        self.pattern_multipliers = {
            str(name): int(value) for name, value in config["patterns"].items()
        }
        self.seed = 0
        self._rng = random.Random()
        self.hands: list[list[int]] = []
        self.open_pongs: list[list[int]] = []
        self.discards: list[list[int]] = []
        self.wall: list[int] = []
        self.phase = Phase.SELF_ACTION
        self.current_seat = self.dealer_seat
        self.last_discard: int | None = None
        self.discard_source: int | None = None
        self.pending_responders: list[int] = []
        self.response_index = 0
        self.decision_count = 0
        self.done = False
        self.rewards = [0.0] * PLAYERS
        self.outcome: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.action_history: list[int] = []

    def reset(self, seed: int) -> None:
        """Reset to a deterministic deal generated from ``seed``."""
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        deck = [tile for tile in range(TILE_TYPES) for _ in range(COPIES_PER_TILE)]
        self._rng.shuffle(deck)
        self.hands = [[0] * TILE_TYPES for _ in range(PLAYERS)]
        self.open_pongs = [[] for _ in range(PLAYERS)]
        self.discards = [[] for _ in range(PLAYERS)]
        for _ in range(int(self.config["initial_concealed_tiles"])):
            for seat in range(PLAYERS):
                self.hands[seat][deck.pop()] += 1
        self.hands[self.dealer_seat][deck.pop()] += 1
        self.wall = deck
        self.phase = Phase.SELF_ACTION
        self.current_seat = self.dealer_seat
        self.last_discard = None
        self.discard_source = None
        self.pending_responders = []
        self.response_index = 0
        self.decision_count = 0
        self.done = False
        self.rewards = [0.0] * PLAYERS
        self.outcome = {}
        self.events = [{"type": "DEAL", "seed": self.seed, "dealer": self.dealer_seat}]
        self.action_history = []
        self.assert_invariants()

    def clone(self) -> "MiniPDHEnv":
        """Return an independent deep copy for isolation and replay tests."""
        return copy.deepcopy(self)

    def legal_actions(self) -> tuple[bool, ...]:
        """Return the fixed-size legal action mask for the current decision."""
        legal = [False] * ACTION_COUNT
        if self.done:
            return tuple(legal)
        if self.phase in {Phase.SELF_ACTION, Phase.DISCARD_AFTER_PONG}:
            for tile, amount in enumerate(self.hands[self.current_seat]):
                legal[tile] = amount > 0
            if self.phase == Phase.SELF_ACTION and self._win_for_self(self.current_seat) is not None:
                legal[Action.HU] = True
        elif self.phase == Phase.RESPONSE_HU:
            legal[Action.PASS] = True
            legal[Action.HU] = True
        elif self.phase == Phase.RESPONSE_PONG:
            legal[Action.PASS] = True
            legal[Action.PONG] = True
        return tuple(legal)

    def step(self, action: int) -> StepResult:
        """Apply one legal controller decision and return the transition outcome."""
        if self.done:
            raise IllegalActionError("Cannot step a finished game")
        action_int = int(action)
        if action_int < 0 or action_int >= ACTION_COUNT or not self.legal_actions()[action_int]:
            raise IllegalActionError(
                f"Illegal action {action_name(action_int) if 0 <= action_int < ACTION_COUNT else action_int} "
                f"for seat={self.current_seat}, phase={self.phase.value}"
            )
        decision_event = {
            "type": "ACTION",
            "seat": self.current_seat,
            "phase": self.phase.value,
            "action": action_int,
            "action_name": action_name(action_int),
            "decision": self.decision_count,
        }
        self.events.append(decision_event)
        self.action_history.append(action_int)
        self.decision_count += 1
        if action_int in DISCARD_ACTIONS:
            self._discard(action_int)
        elif action_int == Action.HU:
            self._hu()
        elif action_int == Action.PONG:
            self._pong()
        else:
            self._pass()
        if not self.done and self.decision_count >= self.max_decisions:
            self._finish_draw("MAX_DECISIONS")
        self.assert_invariants()
        return StepResult(self.done, tuple(self.rewards), decision_event)

    def export_replay(self) -> dict[str, Any]:
        """Return the minimal deterministic replay record and expected final digest."""
        if not self.done:
            raise RuntimeError("Replay export requires a finished game")
        return {
            "schema_version": "1.0",
            "ruleset_id": self.config["ruleset_id"],
            "seed": self.seed,
            "actions": list(self.action_history),
            "expected_rewards": list(self.rewards),
            "expected_outcome": copy.deepcopy(self.outcome),
            "expected_state_digest": self.state_digest(),
        }

    def state_digest(self) -> str:
        """Hash all gameplay state needed to prove deterministic replay."""
        payload = {
            "hands": self.hands,
            "open_pongs": self.open_pongs,
            "discards": self.discards,
            "wall": self.wall,
            "phase": self.phase.value,
            "current_seat": self.current_seat,
            "last_discard": self.last_discard,
            "discard_source": self.discard_source,
            "decision_count": self.decision_count,
            "done": self.done,
            "rewards": self.rewards,
            "outcome": self.outcome,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_invariants(self) -> None:
        """Raise immediately when tile conservation or zero-sum state is violated."""
        if len(self.hands) != PLAYERS or len(self.discards) != PLAYERS or len(self.open_pongs) != PLAYERS:
            raise AssertionError("Expected four player state arrays")
        aggregate = [0] * TILE_TYPES
        for seat in range(PLAYERS):
            if len(self.hands[seat]) != TILE_TYPES:
                raise AssertionError(f"Seat {seat} has malformed hand vector")
            if any(amount < 0 or amount > 4 for amount in self.hands[seat]):
                raise AssertionError(f"Seat {seat} has invalid concealed count")
            if len(self.open_pongs[seat]) > 2:
                raise AssertionError(f"Seat {seat} exceeds the two-meld Mini-PDH limit")
            for tile, amount in enumerate(self.hands[seat]):
                aggregate[tile] += amount
            for tile in self.discards[seat]:
                aggregate[tile] += 1
            for tile in self.open_pongs[seat]:
                aggregate[tile] += 3
        for tile in self.wall:
            aggregate[tile] += 1
        if aggregate != [COPIES_PER_TILE] * TILE_TYPES:
            raise AssertionError(f"Tile conservation failed: {aggregate}")
        if self.done and abs(sum(self.rewards)) > 1e-9:
            raise AssertionError(f"Terminal rewards are not zero-sum: {self.rewards}")
        if not 0 <= self.current_seat < PLAYERS:
            raise AssertionError(f"Invalid current seat: {self.current_seat}")

    def _discard(self, tile: int) -> None:
        seat = self.current_seat
        self.hands[seat][tile] -= 1
        self.discards[seat].append(tile)
        self.last_discard = tile
        self.discard_source = seat
        self.events.append({"type": "DISCARD", "seat": seat, "tile": tile})
        responders = self._ordered_other_seats(seat)
        hu_responders = [candidate for candidate in responders if self._win_on_discard(candidate, tile)]
        if hu_responders:
            self._start_response(Phase.RESPONSE_HU, hu_responders)
            return
        self._start_pong_or_advance(responders)

    def _hu(self) -> None:
        winner = self.current_seat
        if self.phase == Phase.SELF_ACTION:
            result = self._win_for_self(winner)
            if result is None:
                raise AssertionError("Legal mask exposed an invalid self-draw win")
            self._finish_win(winner, winner, result, "SELF_DRAW")
            return
        if self.phase != Phase.RESPONSE_HU or self.last_discard is None or self.discard_source is None:
            raise AssertionError("HU reached outside a valid response window")
        counts = list(self.hands[winner])
        counts[self.last_discard] += 1
        result = evaluate_win(counts, self.open_pongs[winner], self.pattern_multipliers)
        if result is None:
            raise AssertionError("Legal mask exposed an invalid discard win")
        self._finish_win(winner, self.discard_source, result, "DISCARD_WIN")

    def _pong(self) -> None:
        if self.phase != Phase.RESPONSE_PONG or self.last_discard is None or self.discard_source is None:
            raise AssertionError("PONG reached outside a valid response window")
        seat = self.current_seat
        tile = self.last_discard
        if self.hands[seat][tile] < 2:
            raise AssertionError("PONG responder no longer owns two matching tiles")
        self.hands[seat][tile] -= 2
        claimed = self.discards[self.discard_source].pop()
        if claimed != tile:
            raise AssertionError("Claimed discard does not match pending PONG tile")
        self.open_pongs[seat].append(tile)
        self.events.append(
            {"type": "PONG", "seat": seat, "tile": tile, "source": self.discard_source}
        )
        self.current_seat = seat
        self.phase = Phase.DISCARD_AFTER_PONG
        self._clear_response()

    def _pass(self) -> None:
        phase = self.phase
        self.events.append({"type": "PASS", "seat": self.current_seat, "phase": phase.value})
        self.response_index += 1
        if self.response_index < len(self.pending_responders):
            self.current_seat = self.pending_responders[self.response_index]
            return
        if phase == Phase.RESPONSE_HU:
            if self.discard_source is None:
                raise AssertionError("Missing discard source after HU response window")
            self._start_pong_or_advance(self._ordered_other_seats(self.discard_source))
            return
        if phase == Phase.RESPONSE_PONG:
            self._advance_after_unclaimed_discard()
            return
        raise AssertionError(f"PASS reached in unsupported phase {phase.value}")

    def _start_pong_or_advance(self, responders: Iterable[int]) -> None:
        if self.last_discard is None:
            raise AssertionError("PONG search requires a pending discard")
        pong_responders = [
            seat
            for seat in responders
            if self.hands[seat][self.last_discard] >= 2 and len(self.open_pongs[seat]) < 2
        ]
        if pong_responders:
            self._start_response(Phase.RESPONSE_PONG, pong_responders)
        else:
            self._advance_after_unclaimed_discard()

    def _start_response(self, phase: Phase, responders: list[int]) -> None:
        if not responders:
            raise AssertionError("Cannot start an empty response window")
        self.phase = phase
        self.pending_responders = responders
        self.response_index = 0
        self.current_seat = responders[0]
        self.events.append({"type": "RESPONSE_WINDOW", "phase": phase.value, "seats": responders})

    def _advance_after_unclaimed_discard(self) -> None:
        if self.discard_source is None:
            raise AssertionError("Turn advance requires a discard source")
        next_seat = (self.discard_source + 1) % PLAYERS
        self._clear_response()
        self._draw(next_seat)

    def _draw(self, seat: int) -> None:
        if not self.wall:
            self._finish_draw("WALL_EMPTY")
            return
        tile = self.wall.pop()
        self.hands[seat][tile] += 1
        self.current_seat = seat
        self.phase = Phase.SELF_ACTION
        self.events.append({"type": "DRAW", "seat": seat, "tile": tile})

    def _clear_response(self) -> None:
        self.last_discard = None
        self.discard_source = None
        self.pending_responders = []
        self.response_index = 0

    def _ordered_other_seats(self, source: int) -> list[int]:
        return [(source + offset) % PLAYERS for offset in range(1, PLAYERS)]

    def _win_for_self(self, seat: int) -> WinResult | None:
        return evaluate_win(self.hands[seat], self.open_pongs[seat], self.pattern_multipliers)

    def _win_on_discard(self, seat: int, tile: int) -> bool:
        counts = list(self.hands[seat])
        counts[tile] += 1
        return evaluate_win(counts, self.open_pongs[seat], self.pattern_multipliers) is not None

    def _finish_win(self, winner: int, payer: int, result: WinResult, method: str) -> None:
        multiplier = float(result.multiplier)
        rewards = [0.0] * PLAYERS
        if method == "SELF_DRAW":
            for seat in range(PLAYERS):
                rewards[seat] = 3.0 * multiplier if seat == winner else -multiplier
        else:
            rewards[winner] = 3.0 * multiplier
            rewards[payer] = -3.0 * multiplier
        self.rewards = rewards
        self.done = True
        self.outcome = {
            "type": method,
            "winner": winner,
            "payer": payer,
            "multiplier": result.multiplier,
            "patterns": list(result.patterns),
            "selected_pattern": result.selected_pattern,
            "decisions": self.decision_count,
        }
        self.events.append({"type": "TERMINAL", **self.outcome, "rewards": rewards})

    def _finish_draw(self, reason: str) -> None:
        self.rewards = [0.0] * PLAYERS
        self.done = True
        self.outcome = {"type": "DRAW", "reason": reason, "decisions": self.decision_count}
        self.events.append({"type": "TERMINAL", **self.outcome, "rewards": self.rewards})
