"""PDH研究代码：固定实际模型的全链路采样计时，不把函数微基准称作训练加速。"""
from __future__ import annotations

import gc
import hashlib
import json
import time

import torch

from pdh_v2.native import configure
from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.curriculum import SeatMix
from pdh_v2.training.selection_input import load_selection
from pdh_v2.runtime.resources import cgroup_counters


def measure_trial(selection, device: str, feature_backend: str, workers: int,
                  tables: int = 64, repeats: int = 2) -> dict:
    """同seed/模型/桌数；两次测量复用worker但从同一输入重置，保证横向可比。"""
    if tables < 1 or repeats < 2 or repeats > 5:
        raise ValueError("benchmark.config.invalid")
    configure(feature_backend)
    torch.set_num_threads(1)
    model, _, provenance = load_selection(selection, torch.device(device))
    model.eval()
    seeds = tuple(73100+i*1000003 for i in range(tables))
    start = time.monotonic()
    env = (ParallelSessions(tables,73100,workers,seeds,transport="shared")
           if workers else BatchedSessions(tables,73100,seeds))
    startup = time.monotonic()-start
    measures = []
    child_pids = list(getattr(env,"pids",()))
    try:
        for repeat in range(repeats):
            if repeat:
                env.reset(seeds)
            torch.manual_seed(63127)
            collector = RolloutCollector(env,model,torch.device(device),CollectionConfig(
                minimum_hands=tables, seat_mix=SeatMix((1.0,0.0,0.0,0.0)),seat_seed=3331))
            counter = cgroup_counters()
            then = time.monotonic()
            data = collector.collect()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic()-then
            after = cgroup_counters()
            terminals = sorted((t.env_index,t.hand_id,t.receipt,t.transcript.state_sha256)
                               for t in data.terminals)
            digest = hashlib.sha256(json.dumps(terminals,sort_keys=True).encode()).hexdigest()
            measures.append({"seconds":elapsed,"hands":len(terminals),
                "learner_decisions":data.decisions,
                "environment_decisions":data.environment_decisions,
                "environment_decisions_per_second":data.environment_decisions/elapsed,
                "learner_decisions_per_second":data.decisions/elapsed,"digest":digest,
                "cgroup_delta":{k:after[k]-v for k,v in counter.items()
                                if k in after and k!="memory_current_bytes"},
                "cgroup_memory_current_bytes":after.get("memory_current_bytes")})
            del data, collector
            gc.collect()
    finally:
        env.close()
    if len({x["digest"] for x in measures})!=1:
        raise RuntimeError("benchmark.repeat_semantics_changed")
    return {"status":"PASS","backend":feature_backend,"workers":workers,
        "tables":tables,"startup_seconds":startup,"repeats":measures,
        "worker_pids_closed":child_pids,"checkpoint_sha256":provenance["checkpoint_sha256"],
        "scope":"complete_rollout_inference_heuristic_step_audit_no_PPO_no_worker_startup"}
