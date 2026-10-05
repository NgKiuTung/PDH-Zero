"""独立SOLO稳定性实验：复用warmup、受限PPO、固定配对评估，不自动开长训。"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.rollout import BatchedSessions
from pdh_v2.rollout.parallel_env import ParallelSessions
from pdh_v2.rollout.snapshot import snapshot_batched
from pdh_v2.training.collector import CollectionConfig, RolloutCollector
from pdh_v2.training.curriculum import SeatMix
from pdh_v2.training.events import JsonlWriter, ReplayWriter
from pdh_v2.training.paired_evaluation import paired_evaluate
from pdh_v2.training.paired_metrics import development_gate, paired_comparison
from pdh_v2.training.refinement_artifacts import commit_best, load_warmup, save_model
from pdh_v2.training.refinement_control import (
    Progress, RefinementConfig, StopRequested, atomic_json, utc,
)
from pdh_v2.training.refinement_metrics import collection_metrics
from pdh_v2.training.risk_diagnostics import audit_risk
from pdh_v2.training.stable_ppo import StablePPOConfig, guarded_update


def _assert_device_free(config: RefinementConfig) -> torch.device:
    device = torch.device(config.device)
    if device.type == 'cuda':
        check = subprocess.run(['nvidia-smi','--query-compute-apps=pid',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                 check=True, timeout=10)
        if check.stdout.strip():
            raise RuntimeError('GPU occupied; no process was killed')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
    return device


def _evaluate(model, device, config, clock, label):
    clock.stage('evaluation', evaluation=label, total_hands=config.evaluation_hands)
    rows, summary = paired_evaluate(model, device, config.output/'evaluations'/label,
        hands=config.evaluation_hands, environments=min(64,config.evaluation_hands),
        workers=config.workers, seed=config.evaluation_seed,
        progress=clock.tick, check_stop=clock.check)
    clock.event('evaluation_completed', evaluation=label, **summary)
    return rows, summary


def _run(config, device, clock, manifest):
    clock.stage('load_warmup')
    model, anchor, provenance = load_warmup(config.warmup_checkpoint, device)
    manifest['initialization'] = provenance
    ppo_config = StablePPOConfig(minibatch_decisions=config.minibatch_decisions)
    manifest['ppo_config'] = asdict(ppo_config)
    atomic_json(config.output/'manifest.json', manifest)
    clock.event('warmup_loaded', **provenance)
    # NOTE: 不继承BC的Adam动量；不再每10次PPO在同优化器上突然做一次教师复习。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    checkpoints = config.output/'checkpoints'; checkpoints.mkdir()
    initial = checkpoints/'checkpoint_initial.pt'
    save_model(initial,model,optimizer,0,{'stage':'SOLO_STABILITY','source':provenance})
    commit_best(initial,checkpoints/'checkpoint_best.pt')
    base_rows, base_summary = _evaluate(model,device,config,clock,'baseline')
    clock.stage('start_environment_workers', requested_workers=config.workers)
    backend = (ParallelSessions(config.environments,config.seed^0x51A71E,
                 config.workers,check_stop=clock.check) if config.workers
               else BatchedSessions(config.environments,config.seed^0x51A71E))
    worker_ids = backend.pids if isinstance(backend,ParallelSessions) else ()
    clock.event('workers_started', pids=list(worker_ids), start_method='spawn')
    writer = ReplayWriter(config.output/'replays'/'hands.jsonl.gz')
    update = decisions = accepted_updates = 0
    rejected = 0
    next_eval = config.evaluation_interval
    last_evaluated = 0
    last_gate = 'INCONCLUSIVE'
    collector = RolloutCollector(backend,model,device,CollectionConfig(
        minimum_hands=config.hands_per_update, reward_scale=config.reward_scale,
        seat_mix=SeatMix((1.0,0.0,0.0,0.0)), seat_seed=config.seed^0x5EA7),
        progress=lambda hands,steps:clock.tick(hands,config.hands_per_update),
        on_terminal=lambda terminal:writer.write_many((terminal,),update+1))
    log = JsonlWriter(config.output/'metrics.jsonl')
    stop_reason = 'target_decisions'
    best_mean = base_summary['mean_score']
    try:
        while decisions < config.target_decisions:
            clock.check()
            if shutil.disk_usage(config.output).free < 8*1024**3:
                raise StopRequested('disk_below_8GiB')
            # NOTE: 留出一次实测评估耗时与余量；总预算仍由check硬性终止等待。
            if clock.elapsed + 1.25*base_summary['seconds'] + 30 >= config.max_seconds:
                stop_reason = 'reserved_final_evaluation_budget'
                break
            clock.stage('collecting', update=update+1, total_learner_decisions=decisions,
                        worker_pids=list(worker_ids))
            start = time.monotonic()
            data = collector.collect()
            collected_seconds = time.monotonic()-start
            if sum(data.raw_score_sum) != 0:
                raise RuntimeError('non-zero-sum rollout')
            clock.stage('ppo_update', update=update+1, learner_decisions=data.decisions)
            start = time.monotonic()
            trained = guarded_update(model,optimizer,anchor,data.episodes,device,ppo_config,
                                     config.seed+update, check_stop=clock.check)
            update_seconds = time.monotonic()-start
            update += 1; decisions += data.decisions
            accepted_updates += int(trained['update_accepted'])
            rejected = 0 if trained['update_accepted'] else rejected+1
            if not trained['update_accepted']:
                for group in optimizer.param_groups:
                    group['lr'] *= .5
            clock.check()
            clock.stage('risk_diagnostics', update=update)
            risk = audit_risk(model,data.episodes,device,config.minibatch_decisions)
            record = {'timestamp':utc(),'update':update,'stage':'SOLO_STABILITY',
                      'total_learner_decisions':decisions,'elapsed_wall_seconds':clock.elapsed,
                      'ppo_seconds':update_seconds, 'learning_rate':optimizer.param_groups[0]['lr'],
                      **collection_metrics(data,collected_seconds),**trained,'risk':risk}
            if device.type == 'cuda':
                record['cuda_peak_memory_bytes'] = torch.cuda.max_memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            snapshots = (backend.snapshot() if isinstance(backend,ParallelSessions)
                         else snapshot_batched(backend))
            candidate = checkpoints/'checkpoint_latest.pt'
            save_model(candidate,model,optimizer,update,{
                'total_decisions':decisions,'stage':'SOLO_STABILITY',
                'environment_snapshots':snapshots,'run_config':config.to_dict(),
                'source':provenance,'ppo_config':asdict(ppo_config),
                'resume_supported_by_this_cli':False})
            log.write(record); clock.event('update_committed',**record)
            manifest.update(updates=update,accepted_updates=accepted_updates,
                            total_learner_decisions=decisions)
            atomic_json(config.output/'manifest.json',manifest)
            if rejected >= 3:
                stop_reason = 'three_consecutive_KL_rejections'
                break
            if decisions >= next_eval:
                rows, summary = _evaluate(model,device,config,clock,f'd{decisions:012d}')
                comparison = paired_comparison(base_rows,rows)
                last_gate = development_gate(comparison); last_evaluated=decisions
                clock.event('development_comparison',decision=last_gate,comparison=comparison)
                atomic_json(config.output/'gate.json',{'gate':last_gate,'comparison':comparison,
                            'long_training_allowed':False})
                if last_gate == 'PASS' and summary['mean_score'] > best_mean:
                    commit_best(candidate,checkpoints/'checkpoint_best.pt')
                    best_mean=summary['mean_score']
                if last_gate == 'REGRESSION':
                    stop_reason = 'paired_development_regression'
                    break
                next_eval += config.evaluation_interval
        if decisions and last_evaluated != decisions:
            rows, summary = _evaluate(model,device,config,clock,f'final-d{decisions:012d}')
            comparison=paired_comparison(base_rows,rows)
            last_gate=development_gate(comparison)
            atomic_json(config.output/'gate.json',{'gate':last_gate,'comparison':comparison,
                        'long_training_allowed':False})
            clock.event('development_comparison',decision=last_gate,comparison=comparison)
            if last_gate == 'PASS' and summary['mean_score'] > best_mean:
                commit_best(checkpoints/'checkpoint_latest.pt',checkpoints/'checkpoint_best.pt')
        if not decisions:
            last_gate='INCONCLUSIVE'
        manifest.update(status='completed',stop_reason=stop_reason,updates=update,
            accepted_updates=accepted_updates,total_learner_decisions=decisions,
            development_gate=last_gate,long_training_allowed=False)
        return 0
    finally:
        backend.close()
        clock.event('workers_closed',pids=list(worker_ids))


def run_refinement(config: RefinementConfig, install_signals: bool = True) -> int:
    """不覆盖旧运行；CPU测试与目标GPU效果门禁分开。"""
    if not config.warmup_checkpoint.is_file():
        raise FileNotFoundError(config.warmup_checkpoint)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (config.output.parent/'.refinement.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('a refinement task already holds this experiment lock')
    try:
        backend_device = _assert_device_free(config)
        config.output.mkdir(exist_ok=False)
        clock = Progress(config.output,config.max_seconds)
        if install_signals:
            clock.install_signals()
        random.seed(config.seed); torch.manual_seed(config.seed); torch.set_num_threads(1)
        manifest={'schema_version':3,'status':'running','created_at_utc':utc(),
            'pid':os.getpid(),'scope':'solo_stability_refinement_candidate',
            'rule_source_sha256':SOURCE_SHA256,'config':config.to_dict(),
            'runtime':{'torch':torch.__version__,'device':str(backend_device)},
            'long_training_allowed':False}
        atomic_json(config.output/'manifest.json',manifest)
        try:
            result = _run(config,backend_device,clock,manifest)
        except (StopRequested,KeyboardInterrupt) as exc:
            manifest.update(status='stopped',stop_reason=str(exc) or 'KeyboardInterrupt',
                            development_gate='INCOMPLETE',long_training_allowed=False)
            clock.event('training_stopped',reason=manifest['stop_reason'])
            result = 0
        except BaseException as exc:
            manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            clock.event('training_failed',error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            manifest.update(finished_at_utc=utc(),elapsed_wall_seconds=clock.elapsed)
            atomic_json(config.output/'manifest.json',manifest)
            atomic_json(config.output/'status.json',manifest)
            clock.restore_signals()
        return result
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
