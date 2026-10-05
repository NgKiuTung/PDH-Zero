"""冻结一个候选后在新牌墙上与baseline比较，不训练、不替换源实验权重。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pdh_v2.training.heldout_evaluation import evaluate_frozen_pair


def main() -> int:
    """默认选selected_development；rc2历史模型须显式指定实际checkpoint，不猜265k。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--candidate-checkpoint', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--hands', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--max-seconds', type=int, default=5400)
    parser.add_argument('--max-cvar-drop', type=float, default=2.0)
    args = parser.parse_args()
    name = 'HELDOUT_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_root.resolve() / name
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        logging.info(json.dumps({'event': 'evaluation_requested', 'output': str(output)}))
        report = evaluate_frozen_pair(args.run_dir, output, args.candidate_checkpoint,
            args.hands, args.seed, args.workers, args.device, args.max_seconds, args.max_cvar_drop)
        logging.info(json.dumps(report, ensure_ascii=False, indent=2))
        return {'PASS': 0, 'REGRESSION': 2}.get(report['gate'], 3)
    except Exception as exc:
        logging.exception(json.dumps({'status': 'failed', 'error': str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())