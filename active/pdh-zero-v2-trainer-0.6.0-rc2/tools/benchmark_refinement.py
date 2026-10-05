"""相同种子与确定性启发式的串并行参考基准；不冒充GPU端到端加速比。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions


def measure(workers: int, count: int) -> dict:
    """每桌恰好一手；结果摘要必须与串行模式完全相同。"""
    start=time.monotonic()
    env=ParallelSessions(count,37041,workers) if workers else BatchedSessions(count,37041)
    initialized=time.monotonic()
    records=[]; actions=0
    try:
        while not env.all_inactive():
            requests=env.requests()
            choices=env.heuristic_actions(requests)
            actions+=len(choices)
            for terminal in env.step(choices,True):
                records.append((terminal.env_index,terminal.hand_id,terminal.receipt,
                                terminal.transcript.state_sha256))
        sampled=time.monotonic()
    finally:
        env.close()
    elapsed=time.monotonic()-start
    digest=hashlib.sha256(json.dumps(sorted(records),sort_keys=True).encode()).hexdigest()
    return {'workers':workers,'tables':count,'hands':len(records),'decisions':actions,
            'startup_seconds':initialized-start,'collection_seconds':sampled-initialized,
            'total_seconds_including_shutdown':elapsed,
            'collection_decisions_per_second':actions/(sampled-initialized),
            'terminal_digest':digest}


def main() -> int:
    """只写新报告；重复运行须使用新的输出文件。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',default='0,2,4')
    parser.add_argument('--tables',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results=[measure(int(w),args.tables) for w in args.workers.split(',')]
    same=len({row['terminal_digest'] for row in results})==1
    payload={'scope':'CPU_rule_environment_and_IPC_not_GPU_training',
             'semantic_equivalence':same,'results':results}
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
    return 0 if same else 1


if __name__=='__main__':
    raise SystemExit(main())
