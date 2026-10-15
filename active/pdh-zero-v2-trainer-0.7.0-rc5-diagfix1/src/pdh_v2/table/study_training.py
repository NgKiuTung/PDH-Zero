"""Exact complete-hand training budgets, immutable proposals, and existing PPO rollback."""
from __future__ import annotations

from pathlib import Path
import time

import torch

from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions
from pdh_v2.table.checkpoints import load_candidate, save_candidate, tensor_digest
from pdh_v2.table.fit import load_belief
from pdh_v2.table.model import TableActorCritic
from pdh_v2.table.study_contract import BudgetClock, derived_seed
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.events import JsonlWriter, ReplayWriter
from pdh_v2.training.refinement_control import atomic_json
from pdh_v2.training.stable_ppo import StablePPOConfig, guarded_update


class NaturalBatch:
    """Stop each table only at its natural terminal, so one batch is exactly N hands."""
    def __init__(self, backend):
        self.backend = backend

    @property
    def count(self):
        """Number of predeclared full natural hands in the batch."""
        return self.backend.count

    def reactivate_all(self):
        """Start the predeclared full-hand batch; no arbitrary truncation."""
        self.backend.reactivate_all()

    def requests(self):
        """Return only the original public DecisionViews."""
        return self.backend.requests()

    def heuristic_actions(self, requests):
        """Use the identical frozen training opponent in all three arms."""
        return self.backend.heuristic_actions(requests)

    def step(self, choices, drain=False):
        """Always drain completed tables, independent of which policy finishes first."""
        return self.backend.step(choices, drain=True)

    def all_inactive(self):
        """An unfinished table never counts as a completed batch."""
        return self.backend.all_inactive()

    def close(self):
        """Close only the owned serial/parallel backend."""
        self.backend.close()


def _collect(model, config, seed: int, update: int, store, owner: str,
             arm: str, clock, replay):
    root_seed = derived_seed(seed, f'train-update:{update}')
    count = config.hands_per_update
    seeds = tuple(derived_seed(root_seed, f'table:{i}') for i in range(count))
    store.reserve_initial(seeds, owner+':train', 'train')
    base = (ParallelSessions(count, root_seed, config.workers, seeds,
            check_stop=clock.check, transport=config.transport) if config.workers
            else BatchedSessions(count, root_seed, seeds))
    environment = NaturalBatch(base)
    register = store.terminal_callback(owner+':train', 'train', arm)
    def terminal_callback(terminal):
        """Commit wall identity and terminal evidence before updating the Actor."""
        register(terminal)
        # NOTE: completed evidence survives a later timeout; partial hands never enter PPO.
        replay.write_many((terminal,), update)
    try:
        collector = RolloutCollector(environment, model, next(model.parameters()).device,
            CollectionConfig(minimum_hands=count, seat_seed=derived_seed(seed, 'learner-seat')),
            progress=clock.check, on_terminal=terminal_callback)
        result = collector.collect()
        if result.hands != count or not environment.all_inactive():
            raise RuntimeError('rc5.natural_hand_budget_mismatch')
        return result
    finally:
        environment.close()


def train_arm(baseline, belief: Path, config, seed: int, mode: str,
              output: Path, store, owner: str):
    """Same initialization/sampling/PPO budgets; only diagnostic/fusion mode differs."""
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(derived_seed(seed, 'model-initialization'))
    model = TableActorCritic.from_baseline(baseline, mode)
    load_belief(belief, model)
    if mode == 'joint' and not bool((model.belief.qualified.sum(-1) > 0).all()):
        raise ValueError('rc5.joint_blocked_no_useful_belief_for_every_opponent')
    anchor = baseline
    if any(p.requires_grad for p in anchor.parameters()):
        raise ValueError('rc5.anchor_must_be_frozen')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config.learning_rate)
    current = output / 'candidates/initial.pt'
    save_candidate(model, current, dict(seed=seed, arm=mode, update=0,
                   role='initial_reproducibility_not_promoted'))
    events, replay = JsonlWriter(output/'updates.jsonl'), ReplayWriter(output/'hands.jsonl.gz')
    torch.manual_seed(derived_seed(seed, 'actor-sampling'))
    clock = BudgetClock(config.max_training_seconds)
    updates = []
    for update in range(1, config.updates+1):
        clock.check()
        started = time.monotonic()
        batch = _collect(model, config, seed, update, store, owner, f'{owner}:{seed}:{mode}',
                         clock, replay)
        collection_seconds = time.monotonic()-started
        proposal = output / f'candidates/update-{update:04d}-proposal.pt'
        def preserve(candidate, statistics):
            """Record the proposed parameters even when the existing KL gate rejects them."""
            save_candidate(candidate, proposal, dict(seed=seed, arm=mode, update=update,
                           learner_decisions=batch.decisions, **statistics))
        started = time.monotonic()
        metrics = guarded_update(model, optimizer, anchor, batch.episodes,
            next(model.parameters()).device,
            StablePPOConfig(minibatch_decisions=config.minibatch_decisions),
            derived_seed(seed, f'ppo:{update}'), check_stop=clock.check, on_proposal=preserve)
        update_seconds = time.monotonic()-started
        if metrics['update_accepted']:
            current = proposal
        row = dict(update=update, hands=batch.hands, learner_decisions=batch.decisions,
                   environment_decisions=batch.environment_decisions,
                   raw_learner_net_score=batch.learner_score_sum,
                   raw_all_seat_sum=sum(batch.raw_score_sum),
                   collection_seconds=collection_seconds, update_seconds=update_seconds,
                   effective_checkpoint=str(current.relative_to(output)),
                   proposal=str(proposal.relative_to(output)), **metrics)
        events.write(row)
        updates.append(row)
    report = dict(status='PASS', seed=seed, mode=mode, updates=updates,
                   elapsed_training_seconds=time.monotonic()-clock.started,
                   complete_hands=sum(r['hands'] for r in updates),
                   learner_decisions=sum(r['learner_decisions'] for r in updates),
                   effective_checkpoint=str(current.relative_to(output)),
                   reward='true_net_settlement_divided_by_32_gamma_1',
                   training_start='fresh_seed_per_update_dealer_0_no_continuation',
                   recurrent_scope='within_natural_hand_not_cross_hand_return')
    atomic_json(output / 'report.json', report)
    return model.eval(), current, report


def _legacy_digest(model, keys) -> str:
    """Hash only the frozen rc4.1 Actor tensors used by the control gate."""
    state = model.state_dict()
    return tensor_digest({key: state[key] for key in keys})


def _mirror_actor(target, source, keys) -> None:
    """Copy only legacy Actor tensors; fitted belief state remains diagnostic-only."""
    target_state, source_state = target.state_dict(), source.state_dict()
    for key in keys:
        target_state[key] = source_state[key]
    target.load_state_dict(target_state, strict=True)
    if _legacy_digest(target, keys) != _legacy_digest(source, keys):
        raise RuntimeError('rc5.diagnostic_actor_mirror_not_exact')


def _mirror_update(model, baseline, config, seed, update, source_row, source_arm,
                   output, store, owner, clock, replay) -> tuple[Path, dict]:
    """Collect the declared hands, then inherit that baseline update exactly."""
    started = time.monotonic()
    batch = _collect(model, config, seed, update, store, owner,
                     f'{owner}:{seed}:diagnostic', clock, replay)
    collection_seconds = time.monotonic()-started
    source_path = source_arm / source_row['effective_checkpoint']
    source_model, _ = load_candidate(source_path, next(model.parameters()).device)
    keys = tuple(baseline.state_dict())
    _mirror_actor(model, source_model, keys)
    proposal = output / f'candidates/update-{update:04d}-mirrored.pt'
    source_sha = file_digest(source_path)
    save_candidate(model, proposal, dict(seed=seed, arm='diagnostic', update=update,
        role='exact_baseline_actor_mirror_not_promoted', source_checkpoint_sha256=source_sha))
    row = dict(update=update, hands=batch.hands, learner_decisions=batch.decisions,
               environment_decisions=batch.environment_decisions,
               raw_learner_net_score=batch.learner_score_sum,
               raw_all_seat_sum=sum(batch.raw_score_sum),
               collection_seconds=collection_seconds, update_seconds=0.0,
               optimizer_steps_attempted=0, independent_ppo=False, actor_tensor_exact=True,
               source_update_accepted=source_row['update_accepted'],
               source_effective_checkpoint=str(source_path.relative_to(source_arm)),
               source_checkpoint_sha256=source_sha,
               effective_checkpoint=str(proposal.relative_to(output)),
               proposal=str(proposal.relative_to(output)))
    del source_model
    return proposal, row


def mirror_diagnostic_arm(baseline, belief: Path, config, seed: int, output: Path,
                          store, owner: str, source_arm: Path, source_training: dict):
    """Run the negative control with exact baseline Actor updates and equal hand budgets.

    NOTE: Independent CUDA PPO runs need not be bit-exact. Diagnostic instrumentation
    therefore mirrors each already-produced baseline Actor update instead of treating
    numerical reproducibility as evidence that the belief head is non-intervening.
    """
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(derived_seed(seed, 'model-initialization'))
    model = TableActorCritic.from_baseline(baseline, 'diagnostic')
    load_belief(belief, model)
    current = output / 'candidates/initial.pt'
    save_candidate(model, current, dict(seed=seed, arm='diagnostic', update=0,
                   role='initial_reproducibility_not_promoted'))
    events, replay = JsonlWriter(output/'updates.jsonl'), ReplayWriter(output/'hands.jsonl.gz')
    torch.manual_seed(derived_seed(seed, 'actor-sampling'))
    clock, updates = BudgetClock(config.max_training_seconds), []
    source_updates = source_training['updates']
    if len(source_updates) != config.updates:
        raise RuntimeError('rc5.diagnostic_source_budget_mismatch')
    for update, source_row in enumerate(source_updates, 1):
        clock.check()
        current, row = _mirror_update(model, baseline, config, seed, update, source_row,
            source_arm, output, store, owner, clock, replay)
        events.write(row)
        updates.append(row)
    report = dict(status='PASS', seed=seed, mode='diagnostic', updates=updates,
        elapsed_training_seconds=time.monotonic()-clock.started,
        complete_hands=sum(r['hands'] for r in updates),
        learner_decisions=sum(r['learner_decisions'] for r in updates),
        effective_checkpoint=str(current.relative_to(output)),
        reward='true_net_settlement_divided_by_32_gamma_1',
        training_start='fresh_seed_per_update_dealer_0_no_continuation',
        recurrent_scope='within_natural_hand_not_cross_hand_return',
        control='exact_same_seed_baseline_actor_mirror', independent_ppo=False)
    atomic_json(output / 'report.json', report)
    return model.eval(), current, report
