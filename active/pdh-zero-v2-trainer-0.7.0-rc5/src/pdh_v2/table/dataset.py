"""Causal feature snapshots plus separate private labels, split by entire sessions."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from pdh_v2.table.data import contract, infer, records, replay_views
from pdh_v2.table.exposure import ExposureStore
from pdh_v2.table.labels import future_behavior, state_targets
from pdh_v2.table.tensor_checks import validate_dataset_tensors
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.refinement_control import atomic_json


def session_split(session_seed: int) -> tuple[str, bool]:
    """Hold out entire sessions from auxiliary fitting, including adjacent predeals."""
    identity = hashlib.sha256(f'rc5-aux-session:{session_seed}'.encode()).hexdigest()
    return identity, int(identity[:16], 16) % 5 == 0


def make_dataset(model, replay: Path, registry: Path, output: Path,
                 hands: int = 96, stride: int = 12) -> dict:
    """Consume every selected hand to its audited terminal; sample features, not rewards."""
    if min(hands, stride) < 1 or output.exists():
        raise ValueError('rc5.invalid_dataset_request')
    started = time.monotonic()
    device = next(model.parameters()).device
    features, histories, targets, masks, splits, closed = [], [], [], [], [], []
    groups, hand_receipts = [], []
    action_count = 0
    with ExposureStore(registry) as store:
        source_hash = store.require_training_source(replay)
        for row in records(replay, hands):
            store.check_label_wall(row['transcript']['wall'])
            group, validation = session_split(row['audit']['session_seed'])
            hidden = [model.initial_state(1, device) for _ in range(4)]
            behavior, pending = [], []
            for index, session, view in replay_views(row):
                behavior.append({**row['transcript']['actions'][index],
                                 'choices': len(view.actions)})
                _, state = infer(model, view, hidden[view.player], device)
                hidden[view.player] = state
                action_count += 1
                if index % stride:
                    continue
                target, mask = state_targets(session, view.player)
                pending.append((target, mask, index, view.player))
                features.append(np.asarray(view.observation, dtype=np.float32))
                histories.append(state.squeeze(0).cpu().numpy())
                targets.append(target)
                masks.append(mask)
                splits.append(validation)
                groups.append(group)
                closed.append([not session.current_hand.players[(view.player + k) % 4].melds
                               for k in (1, 2, 3)])
            # SAFETY: only labels are filled from the now-verified future suffix.
            for target, mask, index, observer in pending:
                future_behavior(target, mask, tuple(behavior), index, observer)
            hand_receipts.append(row['transcript']['state_sha256'])
    if not features or not any(splits) or all(splits) or len(hand_receipts) != hands:
        raise ValueError('rc5.auxiliary_split_or_hand_count_incomplete')
    payload = dict(schema='rc5-auxiliary-v2', baseline=contract()['baseline']['sha256'],
                   source_sha256=source_hash, groups=groups,
                   features=dict(observation=torch.from_numpy(np.stack(features)),
                                 history=torch.from_numpy(np.stack(histories))),
                   labels=dict(target=torch.from_numpy(np.stack(targets)),
                               mask=torch.from_numpy(np.stack(masks))),
                   validation=torch.tensor(splits), closed=torch.tensor(closed))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = dict(status='PASS', complete_natural_hands=hands, action_events=action_count,
                  snapshots=len(features), train_snapshots=splits.count(False),
                  validation_snapshots=splits.count(True), session_groups=len(set(groups)),
                  train_groups=len({g for g, v in zip(groups, splits) if not v}),
                  validation_groups=len({g for g, v in zip(groups, splits) if v}),
                  split='whole_session_from_archived_train_only',
                  validation_claim='unseen_to_new_auxiliary_fit_not_unseen_to_rc4_1_actor',
                  future_label_horizon_events=24, source_sha256=source_hash,
                  action_label='next_nonforced_legal_decision_not_next_forced_pass',
                  baseline_sha256=payload['baseline'], dataset_sha256=file_digest(output),
                  audited_terminal_digests=hand_receipts, elapsed_seconds=time.monotonic()-started)
    atomic_json(output.with_suffix('.json'), report)
    return report


def load_dataset(path: Path) -> dict:
    """Integrity and schema checks before tensor-only loading; never accepts heldout archives."""
    receipt = __import__('json').loads(path.with_suffix('.json').read_text())
    if file_digest(path) != receipt['dataset_sha256']:
        raise ValueError('rc5.dataset_digest_mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if (data['schema'] != 'rc5-auxiliary-v2'
            or data['source_sha256'] != contract()['training_replay_sha256']
            or data['baseline'] != contract()['baseline']['sha256']):
        raise ValueError('rc5.dataset_provenance_mismatch')
    validate_dataset_tensors(data)
    train = {g for g, valid in zip(data['groups'], data['validation'].tolist()) if not valid}
    valid = {g for g, valid in zip(data['groups'], data['validation'].tolist()) if valid}
    if not train or not valid or train & valid:
        raise ValueError('rc5.auxiliary_session_overlap')
    return data