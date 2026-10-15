"""PDH研究代码：只选择实测且语义一致的负载计划，拒绝旧机器/旧源码计划。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sysconfig

from pdh_v2.native.runtime import ROOT
from pdh_v2.runtime.resources import detect_resources, worker_candidates
from pdh_v2.training.selection_input import resolve_selection


def code_digest() -> str:
    """绑定可执行源，不把日志、二进制和测试报告的时间戳算作源代码。"""
    h = hashlib.sha256()
    for folder in ("src", "tools", "native"):
        for path in sorted((ROOT/folder).rglob("*")):
            if path.suffix in {".py", ".cpp", ".hpp"}:
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def choose_trial(trials: list[dict], near_best: float = .05) -> dict:
    """5%内选较少worker；cpp不足5%实测优势时保留Python参考配置。"""
    if not 0 <= near_best < 1:
        raise ValueError("benchmark.near_best.invalid")
    reference = next(t for t in trials if t["backend"]=="python" and t["status"]=="PASS")
    ref_digest = reference["repeats"][0]["digest"]
    for trial in trials:
        if (trial["status"] == "PASS"
                and any(r["digest"] != ref_digest for r in trial["repeats"])):
            trial["status"] = "SEMANTIC_MISMATCH"
    valid = [t for t in trials if t["status"] == "PASS"]
    for trial in valid:
        trial["median_rollout_dps"] = statistics.median(
            r["environment_decisions_per_second"] for r in trial["repeats"])
    fastest = max(t["median_rollout_dps"] for t in valid)
    close = [t for t in valid if t["median_rollout_dps"] >= fastest*(1-near_best)]
    choice = min(close,key=lambda t:(t["workers"],-t["median_rollout_dps"]))
    if (choice["backend"] == "cpp"
            and choice["median_rollout_dps"] < reference["median_rollout_dps"]*1.05):
        choice=reference
    return {"workers":choice["workers"],"feature_backend":choice["backend"],
        "environments":choice["tables"],"transport":"shared", "torch_num_threads":1,
        "measured_rollout_speedup_vs_reference": (
            choice["median_rollout_dps"]/reference["median_rollout_dps"]),
        "selection_scope":"same_model_same_walls_rollout_only_not_PPO_or_learning_quality"}


def validate_plan(path: Path, device: str) -> dict:
    """重启后再次核配额/ABI/权重；降低资源或代码更新需重新测量，不盲用旧计划。"""
    import torch
    plan=json.loads(path.read_text())
    if (plan.get("status")!="PASS" or plan["code_sha256"]!=code_digest()
            or plan["soabi"]!=sysconfig.get_config_var("SOABI")
            or plan["torch"]!=torch.__version__ or plan["device"]!=device):
        raise ValueError("runtime_plan.identity_mismatch")
    _, record=resolve_selection(Path(plan["selection_path"]))
    if record["checkpoint_sha256"]!=plan["checkpoint_sha256"]:
        raise ValueError("runtime_plan.checkpoint_mismatch")
    current=detect_resources()
    recommendation=plan["recommendation"]
    if recommendation["workers"] not in worker_candidates(current):
        raise ValueError("runtime_plan.cpu_budget_changed")
    previous=plan["resources"]["memory_limit_bytes"]
    if previous is not None and current["memory_limit_bytes"] is not None:
        if current["memory_limit_bytes"] < previous:
            raise ValueError("runtime_plan.memory_budget_changed")
    if device=="cuda" and torch.cuda.get_device_name(0)!=plan["gpu_name"]:
        raise ValueError("runtime_plan.gpu_changed")
    return plan
