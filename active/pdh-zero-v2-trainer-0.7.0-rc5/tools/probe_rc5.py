"""PDH research code: real rc5 candidate, serial/parallel trajectory equivalence and timing."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch

from pdh_v2.native import configure
from pdh_v2.runtime import detect_resources, worker_candidates
from pdh_v2.table.checkpoints import load_candidate
from pdh_v2.table.data import load_baseline, message
from pdh_v2.table.exposure import ExposureStore
from pdh_v2.table.opponents import opponent_factory, OPPONENTS
from pdh_v2.table.process_control import RunSignals, validate_device
from pdh_v2.table.study_contract import BudgetClock, source_identity, check_identity
from pdh_v2.table.study_evaluation import diagnostic_control
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.paired_evaluation import paired_evaluate
from pdh_v2.training.refinement_control import atomic_json


def main() -> int:
    """Benchmark only owned processes; a mismatch is failure, not a faster recommendation."""
    parser = argparse.ArgumentParser(description=message('probe_help'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--environments', type=int, default=8)
    parser.add_argument('--hands', type=int, default=8)
    parser.add_argument('--transport', choices=('pipe', 'shared'), default='pipe')
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    resources = detect_resources()
    allowed = worker_candidates(resources, requested=tuple(range(1, 13)))
    if (args.workers not in allowed or args.workers > args.environments
            or args.hands < 4 or args.hands % 4 or args.max_seconds < 1):
        raise ValueError('rc5.probe_invalid_budget_or_resource_limit')
    torch.set_num_threads(1)
    configure('cpp')
    device = validate_device(args.device)
    baseline = load_baseline(args.baseline, device)
    model, _ = load_candidate(args.candidate, device)
    args.output.mkdir(parents=True, exist_ok=False)
    protected = source_identity(ROOT)
    signals, clock = RunSignals(args.output), BudgetClock(args.max_seconds)
    owner, seed = 'rc5-probe-'+secrets.token_hex(12), secrets.randbits(47)
    result = dict(status='INCOMPLETE', seed=seed, resources=resources, device=args.device,
                  torch=torch.__version__, checkpoint_sha256=file_digest(args.candidate),
                  gpu_name=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  backend='cpp', transport=args.transport, training_performed=False,
                  protected_source=protected, profiles=[])
    try:
        with ExposureStore(args.registry) as store:
            store.require_known_history()
            store.reserve_initial((seed+i*1_000_003 for i in range(args.hands//4)),
                                  owner, 'development')
            serial = {}
            for workers in (0, args.workers):
                started = time.monotonic()
                if args.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                for opponent in OPPONENTS:
                    arm = f'w{workers}/{opponent}'
                    rows, summary = paired_evaluate(model, device, args.output/arm,
                        hands=args.hands, environments=args.environments, workers=workers,
                        seed=seed, transport=args.transport, check_stop=clock.check,
                        on_terminal=store.terminal_callback(owner, 'development', owner+arm),
                        opponent_factory=opponent_factory(opponent, baseline, device))
                    if not workers:
                        serial[opponent] = rows
                    elif not diagnostic_control(serial[opponent], rows):
                        raise RuntimeError('rc5.serial_parallel_trajectory_mismatch')
                elapsed = time.monotonic()-started
                result['profiles'].append(dict(workers=workers, environments=args.environments,
                    natural_hands=args.hands*3, elapsed_seconds=elapsed,
                    hands_per_second=args.hands*3/elapsed,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
                        if args.device == 'cuda' else None))
        check_identity(ROOT, protected)
        best = max(result['profiles'], key=lambda p: p['hands_per_second'])
        result.update(status='PASS', all_opponent_trajectories_exact=True,
                      recommendation=dict(workers=best['workers'],
                                          environments=args.environments),
                      recommendation_scope='only_two_measured_profiles_not_global_optimum')
    except BaseException as error:
        result.update(status='FAILED_OR_STOPPED', error=str(error))
        raise
    finally:
        atomic_json(args.output/'report.json', result)
        signals.close()
    logging.info(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())