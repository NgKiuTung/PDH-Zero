"""PDH研究代码：加载已冻结开发候选；不是恢复优化器，也不冒充已独立验证。"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.training.candidate_store import file_digest, validate_label


def resolve_selection(selection_path: Path) -> tuple[Path, dict]:
    """权重路径限定在本run候选目录，指针、候选元数据和权重哈希必须相符。"""
    selection_path = selection_path.resolve(strict=True)
    if selection_path.stat().st_size > 1024**2:
        raise ValueError("selection.manifest.too_large")
    selection = json.loads(selection_path.read_text())
    if (selection.get("selection_scope") != "selected_development"
            or selection.get("status") != "evaluated"
            or selection.get("long_training_allowed") is not False):
        raise ValueError("selection.scope.invalid")
    label = selection["label"]
    validate_label(label)
    run = selection_path.parent
    expected = run/"checkpoints"/"candidates"/label/"model.pt"
    source = (run/selection["checkpoint"]).resolve(strict=True)
    if source != expected.resolve() or run not in source.parents:
        raise ValueError("selection.path.invalid")
    record = json.loads((expected.parent/"candidate.json").read_text())
    for name in ("label", "status", "checkpoint", "checkpoint_sha256", "update", "gate"):
        if selection.get(name) != record.get(name):
            raise ValueError("selection.record.mismatch")
    if not source.is_file() or source.stat().st_size > 1024**3:
        raise ValueError("selection.checkpoint.invalid")
    if file_digest(source) != selection["checkpoint_sha256"]:
        raise ValueError("selection.hash.mismatch")
    if selection.get("summary") != record.get("summary"):
        raise ValueError("selection.summary.mismatch")
    summary = selection["summary"]
    results = run/"evaluations"/label/"results.jsonl"
    if file_digest(results) != summary["results_sha256"]:
        raise ValueError("selection.evaluation.hash_mismatch")
    return source, selection


def load_selection(path: Path, device: torch.device):
    """导入模型后新建优化器，锚固定为本轮输入模型；不继承旧BC/PPO动量。"""
    source, selection = resolve_selection(path)
    data = torch.load(source, map_location="cpu", weights_only=True)
    if data.get("schema_version") != 1 or data.get("rule_source_sha256") != SOURCE_SHA256:
        raise ValueError("selection.rules_or_schema.invalid")
    if data.get("update") != selection["update"]:
        raise ValueError("selection.update.mismatch")
    config = ModelConfig(**data["model_config"])
    if config.risk_policy_weight != 0:
        raise ValueError("selection.requires_diagnostic_only_risk")
    model = CandidateActorCritic(config).to(device)
    model.load_state_dict(data["model"], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("selection.non_finite_weights")
    anchor = copy.deepcopy(model).eval()
    anchor.requires_grad_(False)
    return model, anchor, {"checkpoint_sha256": file_digest(source),
        "selection_sha256": file_digest(path), "source_candidate": selection["label"],
        "source_update": selection["update"], "source_gate": selection["gate"],
        "source_is_confirmatory": False, "effective_model_config": asdict(config),
        "optimizer_policy": "fresh_AdamW_no_prior_momentum",
        "risk_policy": "diagnostic_only_unchanged", "anchor_scope": "frozen_rc4_input_candidate"}
