"""PDH研究代码：新实验随机种子和用途分离的可复现派生。"""
from __future__ import annotations

import hashlib
import secrets


def new_seed() -> int:
    """生成可存档的63位实验种子；生成后记录，允许显式种子复现实验。"""
    return secrets.randbits(63)


def derive_seed(root: int, purpose: str, index: int = 0) -> int:
    """确定性派生独立用途流，不复用逐版本的日期默认值。"""
    if type(root) is not int or not 0 <= root < 2**63 or index < 0 or not purpose:
        raise ValueError('data_protocol.invalid_seed')
    raw = f'pdh-data-v1:{root}:{purpose}:{index}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') & (2**63 - 1)

