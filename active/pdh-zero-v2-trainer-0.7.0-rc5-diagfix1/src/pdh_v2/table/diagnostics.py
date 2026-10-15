"""Post-freeze, read-only prediction diagnostics on complete evaluation trajectories."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pdh_v2.table.checkpoints import tensor_digest
from pdh_v2.table.data import infer, records, replay_views
from pdh_v2.table.labels import state_targets, future_behavior
from pdh_v2.table.prediction_metrics import evaluation_report
from pdh_v2.training.refinement_control import atomic_json


def diagnose_replays(model, replay: Path, output: Path, prior: torch.Tensor,
                     hands: int = 4, stride: int = 16):
    """No optimizer or training dataset writes; labels remain separate from all Actor calls."""
    before = tensor_digest(model.state_dict())
    device = next(model.parameters()).device
    raws, targets, masks, closed = [], [], [], []
    completed = actions = 0
    for row in records(replay, hands):
        hidden = [model.initial_state(1, device) for _ in range(4)]
        behavior, pending = [], []
        for index, session, view in replay_views(row):
            behavior.append({**row['transcript']['actions'][index], 'choices': len(view.actions)})
            _, state = infer(model, view, hidden[view.player], device)
            hidden[view.player] = state
            actions += 1
            if index % stride:
                continue
            obs = torch.as_tensor(np.asarray(view.observation), dtype=torch.float32,
                                  device=device).unsqueeze(0)
            with torch.no_grad():
                raws.append(model.belief(obs, state).cpu())
            target, mask = state_targets(session, view.player)
            pending.append((target, mask, index, view.player))
            targets.append(target)
            masks.append(mask)
            closed.append([not session.current_hand.players[(view.player+k) % 4].melds
                           for k in (1, 2, 3)])
        for target, mask, index, observer in pending:
            future_behavior(target, mask, tuple(behavior), index, observer)
        completed += 1
    if not raws or tensor_digest(model.state_dict()) != before:
        raise RuntimeError('rc5.diagnostic_mutated_or_empty')
    metrics = evaluation_report(torch.cat(raws), torch.from_numpy(np.stack(targets)),
        torch.from_numpy(np.stack(masks)), torch.tensor(closed), prior.cpu())
    report = dict(status='PASS', complete_hands=completed, action_events=actions,
                  snapshots=len(raws), per_relative_seat=metrics,
                  role='exposed_development_diagnostics_no_training_no_selection',
                  actor_before_sha256=before, actor_after_sha256=before)
    atomic_json(output, report)
    return report