"""Fixed-step auxiliary fitting; validation is read only after the predeclared budget."""
from __future__ import annotations

from pathlib import Path
import time

import torch

from pdh_v2.table.dataset import load_dataset
from pdh_v2.table.model import SeatBelief
from pdh_v2.table.prediction_gate import qualify_heads
from pdh_v2.table.prediction_metrics import auxiliary_loss, empirical_prior, evaluation_report
from pdh_v2.table.tensor_checks import finite_state
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.events import JsonlWriter
from pdh_v2.training.refinement_control import atomic_json


def fit_belief(dataset: Path, output: Path, seed: int, steps: int = 400,
               batch: int = 64, device: torch.device = torch.device('cpu')) -> dict:
    """Fit one independently seeded head; no heldout file, PPO reward or Actor gradient."""
    if min(steps, batch) < 1:
        raise ValueError('rc5.invalid_fit_budget')
    output.mkdir(parents=True, exist_ok=False)
    data = load_dataset(dataset)
    torch.manual_seed(seed)
    model = SeatBelief(data['features']['history'].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    train = torch.where(~data['validation'])[0]
    valid = torch.where(data['validation'])[0]
    rng = torch.Generator().manual_seed(seed)
    obs = data['features']['observation'].to(device)
    history = data['features']['history'].to(device)
    target, mask = data['labels']['target'].to(device), data['labels']['mask'].to(device)
    prior = empirical_prior(target[train], mask[train])
    events = JsonlWriter(output / 'updates.jsonl')
    started = time.monotonic()
    for step in range(steps):
        ids = train[torch.randint(len(train), (batch,), generator=rng)]
        raw = model(obs[ids], history[ids])
        loss = auxiliary_loss(raw, target[ids], mask[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('rc5.nonfinite_auxiliary_loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        events.write(dict(step=step+1, loss=float(loss.detach()), gradient_norm=float(grad),
                          elapsed_seconds=time.monotonic()-started))
    fit_seconds = time.monotonic()-started
    model.eval()
    with torch.no_grad():
        raw = model(obs[valid], history[valid])
        metrics = evaluation_report(raw, target[valid], mask[valid],
                                    data['closed'][valid].to(device), prior)
        qualified, qualification = qualify_heads(metrics)
        model.qualified.copy_(qualified.to(device))
        model.prior.copy_(prior)
    target_path = output / 'belief.pt'
    payload = dict(schema='rc5-belief-v2', state=model.cpu().state_dict(), seed=seed,
                   steps=steps, batch=batch, dataset_sha256=file_digest(dataset),
                   source_sha256=data['source_sha256'], baseline_sha256=data['baseline'],
                   train_prior=prior.detach().cpu())
    torch.save(payload, target_path)
    digest = file_digest(target_path)
    target_path.with_suffix('.pt.sha256').write_text(digest+'  belief.pt\n')
    report = dict(status='PASS', scientific_status='UNVALIDATED_SMALL_AUXILIARY_SAMPLE',
                  seed=seed, steps=steps, batch=batch, fit_seconds=fit_seconds,
                  dataset_sha256=payload['dataset_sha256'], checkpoint_sha256=digest,
                  train_snapshots=len(train), validation_snapshots=len(valid),
                  validation_scope='whole_archived_train_sessions_unseen_only_to_new_head',
                  per_relative_seat=metrics, usefulness_gate=qualification,
                  actor_modified=False, target_5090_tested=False)
    atomic_json(output / 'report.json', report)
    return report


def load_belief(path: Path, model) -> dict:
    """Load a saved fitted head only after hash and baseline/source binding checks."""
    from pdh_v2.table.data import contract
    expected = path.with_suffix('.pt.sha256').read_text().split()[0]
    if file_digest(path) != expected:
        raise ValueError('rc5.belief_digest_mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['schema'] != 'rc5-belief-v2'
            or payload['baseline_sha256'] != contract()['baseline']['sha256']
            or payload['source_sha256'] != contract()['training_replay_sha256']):
        raise ValueError('rc5.belief_provenance_mismatch')
    finite_state(payload['state'])
    model.belief.load_state_dict(payload['state'], strict=True)
    model.belief.requires_grad_(False)
    return payload