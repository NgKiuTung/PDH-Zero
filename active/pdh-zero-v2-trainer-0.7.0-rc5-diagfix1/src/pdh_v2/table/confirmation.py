"""PDH research code: one-use independent confirmation, never a feedback path into training."""
from __future__ import annotations

import json
from pathlib import Path
import secrets

from pdh_v2.table.checkpoints import load_candidate
from pdh_v2.table.data import load_baseline
from pdh_v2.table.exposure import ExposureStore
from pdh_v2.table.freeze import confined_file, load_freeze
from pdh_v2.table.opponents import opponent_factory
from pdh_v2.table.process_control import RunSignals, validate_device
from pdh_v2.table.study_contract import BudgetClock, check_identity
from pdh_v2.training.paired_evaluation import paired_evaluate
from pdh_v2.training.paired_metrics import paired_comparison, development_gate
from pdh_v2.training.refinement_control import atomic_json


def consume_freeze(store, freeze_id: str, output: Path) -> None:
    """An aborted confirmation consumes its freeze ID too; it cannot be relabeled unseen."""
    with store.connection:
        store.connection.execute('''CREATE TABLE IF NOT EXISTS rc5_final_use
            (freeze_id TEXT PRIMARY KEY, output TEXT NOT NULL)''')
        store.connection.execute('INSERT INTO rc5_final_use VALUES (?,?)',
                                 (freeze_id, str(output.resolve())))


def confirm_frozen(root: Path, frozen_dir: Path, baseline_path: Path, registry: Path,
                   output: Path, device_name: str = 'cpu', workers: int = 0,
                   environments: int = 8) -> dict:
    """Use all frozen seeds/conditions once; report failures, retain evidence, never deploy."""
    if not 0 <= workers <= environments or environments < 1:
        raise ValueError('rc5.invalid_workers')
    frozen = load_freeze(root, frozen_dir)
    device = validate_device(device_name)
    baseline = load_baseline(baseline_path, device)
    output.mkdir(parents=True, exist_ok=False)
    signals, clock = RunSignals(output), BudgetClock(frozen['max_seconds'])
    owner = 'rc5-final-' + frozen['freeze_id']
    new_seeds = {str(s): secrets.randbits(47) for s in frozen['seeds']}
    report = dict(status='INCOMPLETE', freeze_id=frozen['freeze_id'],
                  fresh_wall_seeds=new_seeds, historical_coverage='partial',
                  final_scope='new_walls_known_opponent_families_not_unseen_opponent_family',
                  compared_training_seeds=frozen['seeds'], results=[],
                  deployment_promoted=False, training_performed=False)
    atomic_json(output/'manifest.json', {**report, 'freeze': frozen})
    try:
        with ExposureStore(registry) as store:
            store.require_known_history()
            consume_freeze(store, frozen['freeze_id'], output)
            for training_seed in frozen['seeds']:
                seed = new_seeds[str(training_seed)]
                store.reserve_initial((seed+i*1_000_003
                    for i in range(frozen['hands_per_condition']//4)), owner, 'heldout')
                for opponent in frozen['opponents']:
                    rows = {}
                    for mode in ('frozen_reference', 'baseline', 'joint'):
                        check_identity(root, frozen['protected_source'])
                        clock.check()
                        model = baseline
                        if mode != 'frozen_reference':
                            item = next(c for c in frozen['candidates']
                                        if c['seed'] == training_seed and c['mode'] == mode)
                            candidate = confined_file(frozen_dir, item['file'])
                            model, _ = load_candidate(candidate, device)
                        arm = f'{training_seed}/{opponent}/{mode}'
                        rows[mode], _ = paired_evaluate(model, device, output/arm,
                            hands=frozen['hands_per_condition'], environments=environments,
                            workers=workers, seed=seed, transport='pipe', check_stop=clock.check,
                            on_terminal=store.terminal_callback(owner, 'heldout', owner+arm),
                            opponent_factory=opponent_factory(opponent, baseline, device))
                        if mode != 'frozen_reference':
                            del model
                    comparisons = {mode: paired_comparison(rows[mode], rows['joint'])
                                   for mode in ('frozen_reference', 'baseline')}
                    item = dict(seed=training_seed, opponent=opponent, comparisons=comparisons,
                                gates={k: development_gate(v) for k, v in comparisons.items()})
                    report['results'].append(item)
                    atomic_json(output/f'{training_seed}/{opponent}/comparison.json', item)
        check_identity(root, frozen['protected_source'])
        gates = [g for item in report['results'] for g in item['gates'].values()]
        elapsed = __import__('time').monotonic()-clock.started
        report.update(status='PASS', gate='PASS' if all(g == 'PASS' for g in gates)
                      else 'NOT_PASSED', elapsed_seconds=elapsed)
    except BaseException as error:
        report.update(status='FAILED_OR_STOPPED', error=str(error), gate='INCOMPLETE',
                      exposed_test_never_reusable_as_unseen=True)
        raise
    finally:
        atomic_json(output/'report.json', report)
        signals.close()
    return report