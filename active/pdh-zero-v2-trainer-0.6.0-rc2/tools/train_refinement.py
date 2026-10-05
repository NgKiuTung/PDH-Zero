"""从已保存warmup开始受控SOLO调优；不重跑2048手教师、不自动开启长任务。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdh_v2.training.refinement_control import RefinementConfig
from pdh_v2.training.refinement_runner import run_refinement


def main() -> int:
    """默认总墙钟90分钟包含评估；SIGINT/SIGTERM记录stopped并回收本任务worker。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup-checkpoint',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--environments',type=int,default=64)
    parser.add_argument('--hands-per-update',type=int,default=256)
    parser.add_argument('--max-seconds',type=int,default=5400)
    parser.add_argument('--target-decisions',type=int,default=750000)
    parser.add_argument('--evaluation-hands',type=int,default=1024)
    parser.add_argument('--evaluation-interval',type=int,default=250000)
    parser.add_argument('--learning-rate',type=float,default=.00005)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    name='REFINE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output_root.resolve()/name
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    try:
        config=RefinementConfig(output=output,warmup_checkpoint=args.warmup_checkpoint.resolve(),
            workers=args.workers,environments=args.environments,
            hands_per_update=args.hands_per_update,
            max_seconds=args.max_seconds,target_decisions=args.target_decisions,
            evaluation_hands=args.evaluation_hands,evaluation_interval=args.evaluation_interval,
            learning_rate=args.learning_rate,device=args.device)
        logging.info(json.dumps({'event':'run_requested','output':str(output)},ensure_ascii=False))
        return run_refinement(config)
    except Exception as exc:
        logging.exception(json.dumps({'status':'failed','error':str(exc)},ensure_ascii=False))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
