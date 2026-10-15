"""旧warmup只读导入、风险模式显式迁移、模型与优化器原子检查点。"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import CandidateActorCritic, ModelConfig
from pdh_v2.training.checkpoint import save_checkpoint


def load_warmup(path: Path, device: torch.device):
    """只接收本项目update=0教师检查点；不读取失败任务latest作为起点。"""
    if not path.is_file() or path.stat().st_size > 1024**3:
        raise ValueError('warmup checkpoint missing or exceeds 1GiB')
    # NOTE: 本项目检查点只有Tensor/基础容器，使用受限加载器，不执行任意pickle对象。
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('schema_version') != 1 or data.get('rule_source_sha256') != SOURCE_SHA256:
        raise ValueError('checkpoint schema or rule identity mismatch')
    if data.get('update') != 0 or data.get('metadata',{}).get('teacher_hands',0) < 1:
        raise ValueError('requires verified update-zero teacher warmup checkpoint')
    original_config = ModelConfig(**data['model_config'])
    config = replace(original_config, risk_policy_weight=0.0)
    model = CandidateActorCritic(config).to(device)
    model.load_state_dict(data['model'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('checkpoint contains non-finite model weights')
    anchor = copy.deepcopy(model).eval()
    anchor.requires_grad_(False)
    return model, anchor, {'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'original_model_config':asdict(original_config), 'effective_model_config':asdict(config),
        'optimizer_policy':'fresh_AdamW_no_BC_momentum',
        'risk_policy':'diagnostic_only_rebaseline_required'}


def save_model(path: Path, model, optimizer, update: int, metadata: dict) -> None:
    """使用现有原子检查点写入，额外生成旁路文件校验。"""
    save_checkpoint(path, model, optimizer, update, metadata)
    path.with_suffix(path.suffix+'.sha256').write_text(
        hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name+'\n')


def commit_best(source: Path, destination: Path) -> None:
    """仅由通过开发集门禁的候选触发；不中断替换已提交best。"""
    temporary = destination.with_suffix('.tmp')
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, destination)
    destination.with_suffix('.pt.sha256').write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest()+'  '+destination.name+'\n')
