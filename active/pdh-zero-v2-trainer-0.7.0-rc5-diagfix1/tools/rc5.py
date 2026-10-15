"""PDH research code: bounded rc5 audit/research command line (no automatic paid GPU start)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from pdh_v2.native import configure
from pdh_v2.table.audit import run_audit
from pdh_v2.table.data import load_baseline, message


def main() -> int:
    """Validate CLI sizes/identities before running; return nonzero on a real failure."""
    parser = argparse.ArgumentParser(description=message("cli_description"))
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help=message("audit_help"))
    audit_source = audit.add_mutually_exclusive_group(required=True)
    audit_source.add_argument("--baseline", type=Path)
    audit_source.add_argument("--candidate", type=Path)
    audit.add_argument("--replays", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--hands", type=int, default=16)
    audit.add_argument("--pairs", type=int, default=32)
    audit.add_argument("--backend", choices=("python", "cpp"), default="python")
    prepare = commands.add_parser("prepare", help=message("prepare_help"))
    prepare.add_argument("--train-run", type=Path, required=True)
    prepare.add_argument("--heldout-run", type=Path, required=True)
    prepare.add_argument("--registry", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    dataset = commands.add_parser("dataset", help=message("dataset_help"))
    dataset.add_argument("--baseline", type=Path, required=True)
    dataset.add_argument("--replays", type=Path, required=True)
    dataset.add_argument("--registry", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--hands", type=int, default=96)
    dataset.add_argument("--stride", type=int, default=12)
    dataset.add_argument("--backend", choices=("python", "cpp"), default="python")
    fit = commands.add_parser("fit", help=message("fit_help"))
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--steps", type=int, default=400)
    fit.add_argument("--batch", type=int, default=64)
    fit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    study = commands.add_parser("study", help=message("study_help"))
    study.add_argument("--baseline", type=Path, required=True)
    study.add_argument("--dataset", type=Path, required=True)
    study.add_argument("--registry", type=Path, required=True)
    study.add_argument("--config", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan", help=message("plan_help"))
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    plan.add_argument("--backend", choices=("python", "cpp"), default="cpp")
    plan.add_argument("--workers", type=int, default=0)
    freeze = commands.add_parser('freeze', help=message('freeze_help'))
    freeze.add_argument('--study', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    freeze.add_argument('--hands', type=int, default=4096)
    freeze.add_argument('--max-seconds', type=int, default=21600)
    final = commands.add_parser('confirm', help=message('confirm_help'))
    final.add_argument('--freeze', type=Path, required=True)
    final.add_argument('--baseline', type=Path, required=True)
    final.add_argument('--registry', type=Path, required=True)
    final.add_argument('--output', type=Path, required=True)
    final.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    final.add_argument('--workers', type=int, default=0)
    final.add_argument('--environments', type=int, default=8)
    final.add_argument('--backend', choices=('python', 'cpp'), default='cpp')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(1)
    configure(getattr(args, 'backend', 'python'))
    if args.output.exists():
        raise FileExistsError(message("output_exists"))
    if args.command == 'freeze':
        from pdh_v2.table.freeze import freeze_study
        report = freeze_study(ROOT, args.study, args.output, args.hands, args.max_seconds)
    elif args.command == 'confirm':
        from pdh_v2.table.confirmation import confirm_frozen
        report = confirm_frozen(ROOT, args.freeze, args.baseline, args.registry, args.output,
                                args.device, args.workers, args.environments)
    elif args.command == 'plan':
        import secrets
        from pdh_v2.table.study_contract import StudyConfig
        from pdh_v2.training.refinement_control import atomic_json
        config = StudyConfig(seeds=tuple(secrets.randbits(47) for _ in range(3)),
                             device=args.device, backend=args.backend, workers=args.workers)
        report = config.to_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
    elif args.command == 'study':
        from pdh_v2.table.study import run_study
        from pdh_v2.table.study_contract import read_config
        config = read_config(args.config)
        configure(config.backend)
        report = run_study(ROOT, args.baseline, args.dataset, args.registry, args.output, config)
    elif args.command == 'prepare':
        from pdh_v2.table.exposure import prepare_registry
        report = prepare_registry(args.train_run, args.heldout_run, args.registry, args.output)
    elif args.command == 'fit':
        from pdh_v2.table.fit import fit_belief
        report = fit_belief(args.dataset, args.output, args.seed, args.steps,
                            args.batch, torch.device(args.device))
    else:
        if args.command == 'audit' and args.candidate is not None:
            from pdh_v2.table.checkpoints import load_candidate
            model, _ = load_candidate(args.candidate, torch.device('cpu'))
        else:
            model = load_baseline(args.baseline, torch.device("cpu"))
        if args.command == 'dataset':
            from pdh_v2.table.dataset import make_dataset
            report = make_dataset(model, args.replays, args.registry, args.output,
                                   args.hands, args.stride)
        else:
            report = run_audit(model, args.replays, args.output, args.hands, args.pairs)
    logging.info(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())