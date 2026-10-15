"""Versioned rc5 candidate artifacts, immutable identities and explicit original-policy rollback."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from pdh_v2.config import SOURCE_SHA256
from pdh_v2.policy import ModelConfig
from pdh_v2.table.data import contract
from pdh_v2.table.model import TableActorCritic
from pdh_v2.table.tensor_checks import finite_state
from pdh_v2.training.candidate_store import file_digest
from pdh_v2.training.refinement_control import atomic_json


def tensor_digest(state: dict) -> str:
    """Canonical tensor content hash avoids serializer metadata confusing equality tests."""
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str((str(value.dtype), tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def save_candidate(model, path: Path, metadata: dict, optimizer=None):
    """Save before evaluation; never overwrite or publish through latest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    finite_state(state)
    payload = dict(schema='pdh-rc5-candidate-v1', model_config=asdict(model.config),
                   mode=model.mode, model=state, baseline_sha256=contract()['baseline']['sha256'],
                   rule_source_sha256=SOURCE_SHA256, metadata=metadata,
                   optimizer=None if optimizer is None else optimizer.state_dict())
    # NOTE: State is sufficient for inspection/replay; exact process resume is not advertised.
    with path.open('xb') as stream:
        torch.save(payload, stream)
    digest = file_digest(path)
    path.with_suffix('.pt.sha256').write_text(digest+'  '+path.name+'\n')
    receipt = dict(checkpoint_sha256=digest, tensor_sha256=tensor_digest(state), mode=model.mode,
                   baseline_sha256=payload['baseline_sha256'], metadata=metadata,
                   exact_resume_supported=False)
    atomic_json(path.with_suffix('.json'), receipt)
    return receipt


def load_candidate(path: Path, device):
    """Tensor-only load with companion digest, frozen-rule identity and strict state keys."""
    if file_digest(path) != path.with_suffix('.pt.sha256').read_text().split()[0]:
        raise ValueError('rc5.candidate_digest_mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['schema'] != 'pdh-rc5-candidate-v1'
            or payload['baseline_sha256'] != contract()['baseline']['sha256']
            or payload['rule_source_sha256'] != SOURCE_SHA256):
        raise ValueError('rc5.candidate_identity_mismatch')
    config = ModelConfig(**payload['model_config'])
    if (config.observation_size, config.action_size, config.hidden_size) != (622, 74, 384):
        raise ValueError('rc5.candidate_configuration_mismatch')
    if config.risk_policy_weight != 0.:
        raise ValueError('rc5.candidate_configuration_mismatch')
    finite_state(payload['model'])
    model = TableActorCritic(config, payload['mode'])
    model.load_state_dict(payload['model'], strict=True)
    model.belief.requires_grad_(False)
    return model.to(device).eval(), payload