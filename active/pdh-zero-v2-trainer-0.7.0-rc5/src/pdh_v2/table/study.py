"""Finite autoresearch-style three-arm study; all outcomes and candidates are retained."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import platform
import shutil
import time
import uuid

import torch

from pdh_v2.table.checkpoints import tensor_digest, load_candidate
from pdh_v2.table.data import contract, load_baseline
from pdh_v2.table.diagnostics import diagnose_replays
from pdh_v2.table.exposure import ExposureStore
from pdh_v2.table.fit import fit_belief
from pdh_v2.table.process_control import RunSignals, validate_device
from pdh_v2.table.study_contract import check_identity, derived_seed, source_identity
from pdh_v2.table.study_evaluation import (compare_conditions, decision_record,
                                           diagnostic_control, evaluate_conditions)
from pdh_v2.table.study_training import train_arm
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.events import JsonlWriter
from pdh_v2.training.refinement_control import StopRequested, atomic_json, utc


def _write_trial(output, record, events):
    events.write(record)
    with (output / 'results.tsv').open('a') as stream:
        for opponent, result in record['comparisons'].items():
            score = result['comparison']['score']
            stream.write('\t'.join(map(str, (record['seed'], record['mode'], opponent,
                record['checkpoint_sha256'], score['delta'], score['ci95'][0], score['ci95'][1],
                record['decision']['status'], record['training']['complete_hands'],
                record['training']['learner_decisions'])))+'\n')


def _seed_study(root, baseline, dataset, config, seed, output, store, manifest, events):
    run = output / f'seed-{seed}'
    run.mkdir()
    owner = manifest['study_id']
    check_identity(root, manifest['protected_source'])
    # NOTE: Establish the untouched reference before any new-head fitting or PPO trial.
    reference, reference_summary = evaluate_conditions(
        baseline, baseline, config, seed, run/'frozen_reference', store, owner, 'frozen_reference')
    atomic_json(run/'frozen_reference.json', reference_summary)
    events.write(dict(event='frozen_reference_established', seed=seed,
                      baseline_sha256=manifest['baseline_sha256'], summaries=reference_summary))
    with (output/'results.tsv').open('a') as stream:
        for opponent in config.opponents:
            stream.write('\t'.join(map(str, (seed, 'frozen_reference', opponent,
                manifest['baseline_sha256'], 0, 0, 0, 'baseline', 0, 0)))+'\n')
    fit_report = fit_belief(dataset, run/'auxiliary', derived_seed(seed, 'auxiliary-fit'),
                            config.fit_steps, config.fit_batch, next(baseline.parameters()).device)
    belief = run/'auxiliary/belief.pt'
    baseline_rows, baseline_tensor = None, None
    trials = []
    for mode in ('baseline', 'diagnostic', 'joint'):
        check_identity(root, manifest['protected_source'])
        arm_dir = run / mode
        model, candidate, training = train_arm(baseline, belief, config, seed, mode,
                                               arm_dir, store, owner)
        expected_tensor = tensor_digest(model.state_dict())
        model, _ = load_candidate(candidate, next(baseline.parameters()).device)
        if tensor_digest(model.state_dict()) != expected_tensor:
            raise RuntimeError('rc5.saved_candidate_not_exact')
        rows, summaries = evaluate_conditions(model, baseline, config, seed,
            arm_dir/'evaluation', store, owner, mode)
        comparisons = compare_conditions(reference, rows, config)
        original_keys = baseline.state_dict().keys()
        legacy_digest = tensor_digest({k: model.state_dict()[k] for k in original_keys})
        if mode == 'baseline':
            baseline_rows, baseline_tensor = rows, legacy_digest
        control = None
        if mode == 'diagnostic':
            control = dict(actor_tensor_exact=(legacy_digest == baseline_tensor),
                all_opponent_trajectories_exact=all(diagnostic_control(baseline_rows[o], rows[o])
                                                    for o in config.opponents))
            if not all(control.values()):
                raise RuntimeError('rc5.diagnostic_control_changed_policy')
        if mode == 'joint':
            payload = torch.load(belief, map_location='cpu', weights_only=True)
            for opponent in config.opponents:
                diagnose_replays(model, arm_dir/'evaluation'/opponent/'hands.jsonl.gz',
                    arm_dir/'evaluation'/opponent/'prediction-diagnostics.json',
                    payload['train_prior'], hands=min(4, config.evaluation_hands))
        controlled = compare_conditions(baseline_rows, rows, config)
        decision = decision_record(config, controlled, candidate.relative_to(output),
                                   manifest['rollback'])
        report = dict(seed=seed, mode=mode, checkpoint=str(candidate.relative_to(output)),
                      checkpoint_sha256=file_digest(candidate), training=training,
                      summaries=summaries, comparisons=comparisons,
                      versus_same_budget_original=controlled, decision=decision,
                      diagnostic_control=control,
                      fit_report_sha256=file_digest(run/'auxiliary/report.json'))
        atomic_json(arm_dir/'trial.json', report)
        check_identity(root, manifest['protected_source'])
        _write_trial(output, report, events)
        logging.info(json.dumps(dict(event='trial_finished', seed=seed, mode=mode,
                                    status=decision['status']), ensure_ascii=False))
        trials.append(report)
        del model
    return trials


def run_study(root: Path, baseline_path: Path, dataset: Path, registry: Path,
              output: Path, config) -> dict:
    """No background loop, implicit GPU purchase, automatic deployment or silent resume."""
    output.mkdir(parents=True, exist_ok=False)
    owner = 'rc5-study-' + uuid.uuid4().hex
    manifest = dict(schema='rc5-study-v1', study_id=owner, created_at=utc(),
                    config=config.to_dict(), baseline_sha256=contract()['baseline']['sha256'],
                    dataset_sha256=file_digest(dataset), protected_source=source_identity(root),
                    rollback=dict(baseline_sha256=contract()['baseline']['sha256'],
                                  label='d000000266137',
                                  action='use_unchanged_original_checkpoint'),
                    hardware=dict(python=platform.python_version(), torch=torch.__version__,
                                  cuda_available=torch.cuda.is_available(),
                                  requested_device=config.device),
                    fixed_budget='equal_full_natural_hands_and_updates_plus_equal_time_safety_caps',
                    historical_coverage='partial', final_heldout_used_for_training=False)
    atomic_json(output/'manifest.json', manifest)
    (output/'results.tsv').write_text(
        'seed\tarm\topponent\tcheckpoint_sha256\tdelta_vs_frozen\tci_low\tci_high'
        '\tstatus\thands\tdecisions\n')
    events = JsonlWriter(output/'events.jsonl')
    started = time.monotonic()
    trials = []
    signals = RunSignals(output)
    try:
        device = validate_device(config.device)
        baseline = load_baseline(baseline_path, device)
        baseline.requires_grad_(False)
        with ExposureStore(registry) as store:
            store.require_known_history()
            for seed in config.seeds:
                trials.extend(_seed_study(root, baseline, dataset, config, seed, output,
                                          store, manifest, events))
        shutil.copyfile(registry, output/'exposure.sqlite3')
        result = dict(status='PASS', scientific_status='INCONCLUSIVE' if config.smoke_only
                       else 'DEVELOPMENT_ONLY', completed_seeds=len(config.seeds),
                       completed_arms=len(trials), opponent_conditions=list(config.opponents),
                       elapsed_seconds=time.monotonic()-started, deployment_promoted=False,
                       target_5090_tested=False, rollback=manifest['rollback'],
                       diagnostic_controls=[t['diagnostic_control'] for t in trials
                                            if t['diagnostic_control'] is not None],
                       dataset_sha256=manifest['dataset_sha256'],
                       registry_sha256=file_digest(output/'exposure.sqlite3'))
        atomic_json(output/'report.json', result)
        return result
    except BaseException as error:
        status = 'stopped' if isinstance(error, (StopRequested, KeyboardInterrupt)) else 'crash'
        failure = dict(status=status, error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.monotonic()-started, partial_evidence_retained=True,
                       deployment_promoted=False, rollback=manifest['rollback'])
        events.write(failure)
        atomic_json(output/'failure.json', failure)
        raise
    finally:
        signals.close()