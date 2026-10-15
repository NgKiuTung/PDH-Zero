"""PDH research code: verified rc4.1 inputs and full, time-causal replay views.

NOTE: Private records are accepted here for labels/audit, never by the Actor API.
The original archives and manifests are read-only; path remapping is explicit.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import numpy as np

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.dealer import DealerState
from pdh_v2.game.actions import Action, ActionKind
from pdh_v2.game.engine import MahjongHand
from pdh_v2.game.session import GameSession
from pdh_v2.observation import DecisionView, decision_view
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.replay import HandTranscript
from pdh_v2.replay.audit import audit_terminal_record
from pdh_v2.training.candidate_store import file_digest

RESOURCE = Path(__file__).resolve().parents[1] / "resources" / "rc5.json"


def contract() -> dict[str, Any]:
    """Read versioned identities and user-facing text; no environment-derived defaults."""
    return json.loads(RESOURCE.read_text(encoding="utf-8"))


def message(key: str) -> str:
    """Return the required resource string; an unknown key is a programming error."""
    return str(contract()["messages"][key])


def load_baseline(path: Path, device: torch.device) -> CandidateActorCritic:
    """Load only the exact authorized d266137 bytes, on CPU before moving to device."""
    if file_digest(path) != contract()["baseline"]["sha256"]:
        raise ValueError(message("baseline_mismatch"))
    # SAFETY(rc5): SHA256 is checked against the bundled, frozen identity before pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("rule_source_sha256") != SOURCE_SHA256:
        raise ValueError(message("rule_mismatch"))
    config = ModelConfig(**payload["model_config"])
    if (config.hidden_size, config.observation_size, config.action_size,
            config.risk_policy_weight) != (384, 622, 74, 0.0):
        raise ValueError(message("baseline_mismatch"))
    model = CandidateActorCritic(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def records(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream complete terminal records; limit selects whole hands, not partial games."""
    if limit is not None and limit < 1:
        raise ValueError(message("positive_sizes"))
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            if "transcript" not in row or "audit" not in row or "receipt" not in row:
                raise ValueError(message("incomplete_record"))
            yield row


def restore_record(record: dict[str, Any]) -> tuple[GameSession, HandTranscript]:
    """Rebuild decision-time public state, including previous-hand retained reveals."""
    transcript = HandTranscript.from_json(json.dumps(record["transcript"]).encode())
    audit = record["audit"]
    session = GameSession(audit["session_seed"])
    session.dealer_state = DealerState(**audit["before_dealer"])
    session.current_hand = MahjongHand(transcript.hand_id, transcript.dealer, transcript.wall)
    for player, revealed in zip(session.current_hand.players,
                                audit["initial_revealed_retained"], strict=True):
        if not set(revealed).issubset(player.concealed):
            raise ValueError(message("incomplete_record"))
        player.revealed_retained.update(revealed)
    return session, transcript


def replay_views(record: dict[str, Any]) -> Iterator[tuple[int, GameSession, DecisionView]]:
    """Yield pre-action views, then execute the original legal action and verify the end.

    The yielded session is mutable and must not be retained without copying. The
    caller must consume the iterator to exhaustion before calling the hand verified.
    """
    session, transcript = restore_record(record)
    for index, data in enumerate(transcript.actions):
        view = decision_view(session)
        yield index, session, view
        action = Action(data["player"], ActionKind(data["kind"]),
                        tuple(data["tiles"]), data["meld_index"])
        session.current_hand.step(action)
    if HandTranscript.from_hand(session.current_hand).state_sha256 != transcript.state_sha256:
        raise ValueError(message("replay_mismatch"))
    audit_terminal_record(record)


def infer(model: CandidateActorCritic, view: DecisionView,
          hidden: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the public-only Actor interface; private session/record objects are excluded."""
    features = torch.as_tensor(np.asarray(view.action_features), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _, _, next_hidden = model.forward_step(
            torch.as_tensor(np.asarray(view.observation), dtype=torch.float32,
                            device=device).unsqueeze(0), hidden,
            features, torch.zeros(len(view.actions), dtype=torch.long, device=device))
    return logits, next_hidden