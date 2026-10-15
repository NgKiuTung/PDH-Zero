"""Frozen score evaluation and causal-free, post-evaluation prediction diagnostics."""
from __future__ import annotations

from pathlib import Path

from pdh_v2.table.opponents import opponent_factory
from pdh_v2.table.outcome_diagnostics import summarize_candidate_outcomes
from pdh_v2.table.study_contract import BudgetClock, derived_seed
from pdh_v2.training.paired_evaluation import paired_evaluate
from pdh_v2.training.paired_metrics import paired_comparison, development_gate
from pdh_v2.training.refinement_control import atomic_json


def evaluate_conditions(model, baseline, config, seed, output: Path, store, owner, label):
    """Same fresh development wall groups and all four seats, separately for each opponent."""
    rows, summaries = {}, {}
    clock = BudgetClock(config.max_evaluation_seconds)
    evaluation_seed = derived_seed(seed, 'development')
    store.reserve_initial((evaluation_seed+i*1_000_003
                           for i in range(config.evaluation_hands//4)),
                          owner+':development', 'development')
    for opponent in config.opponents:
        clock.check()
        callback = store.terminal_callback(owner+':development', 'development',
                                           f'{owner}:{seed}:{label}:{opponent}')
        result, summary = paired_evaluate(model, next(model.parameters()).device,
            output / opponent, hands=config.evaluation_hands,
            environments=config.evaluation_environments, workers=config.workers,
            seed=evaluation_seed, check_stop=clock.check,
            transport=config.transport, on_terminal=callback,
            opponent_factory=opponent_factory(opponent, baseline, next(model.parameters()).device))
        summarize_candidate_outcomes(output / opponent)
        rows[opponent], summaries[opponent] = result, summary
    return rows, summaries


def compare_conditions(reference, candidate, config):
    """Development score/tail gates remain unchanged; no tuning against old heldout."""
    result = {}
    for opponent in config.opponents:
        comparison = paired_comparison(reference[opponent], candidate[opponent])
        gate = development_gate(comparison)
        result[opponent] = dict(comparison=comparison, raw_development_gate=gate,
                               gate='INCONCLUSIVE_SMOKE' if config.smoke_only else gate)
    return result


def diagnostic_control(reference: list[dict], diagnostic: list[dict]) -> bool:
    """Exact completed trajectories/receipts, not merely similar averages."""
    identity = lambda row: (row['wall_seed'], row['candidate_seat'])
    left = {identity(r): (r['state_sha256'], r['score'], r['action_events']) for r in reference}
    right = {identity(r): (r['state_sha256'], r['score'], r['action_events']) for r in diagnostic}
    return left == right


def decision_record(config, comparisons: dict, checkpoint: Path, rollback: dict):
    """Retain every artifact; continue to use frozen rc4.1 unless sufficient evidence exists."""
    gates = [row['gate'] for row in comparisons.values()]
    if config.smoke_only:
        status = 'inconclusive'
    elif any(gate == 'REGRESSION' for gate in gates):
        status = 'discard'
    elif all(gate == 'PASS' for gate in gates):
        status = 'keep_for_independent_confirmation'
    else:
        status = 'inconclusive'
    return dict(status=status, candidate=str(checkpoint), deployment_promoted=False,
                rollback=rollback, reason='frozen_development_gate_all_opponent_conditions',
                final_unseen_confirmation_run=False)