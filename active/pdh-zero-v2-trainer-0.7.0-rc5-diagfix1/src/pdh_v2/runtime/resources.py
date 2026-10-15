"""PDH研究代码：cgroup v1/v2只读资源发现，支持可见祖先的更紧上限。"""
from __future__ import annotations

import math
import os
from pathlib import Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None


def parse_cpu_max(text: str) -> float | None:
    """cpu.max是总CPU时间/周期，不是专属物理核心数量。"""
    quota, period = text.split()
    p = int(period)
    if p <= 0:
        raise ValueError("resources.cpu_period.invalid")
    if quota == "max":
        return None
    q = int(quota)
    if q <= 0:
        raise ValueError("resources.cpu_quota.invalid")
    return q/p


def _directories(root: Path, membership: str) -> list[Path]:
    """namespace根和可见成员祖先均检查，不可见祖先明确不作推测。"""
    found = {root}
    for line in membership.splitlines():
        pieces = line.split(":", 2)
        if len(pieces) != 3 or ".." in Path(pieces[2]).parts:
            continue
        relative = pieces[2].lstrip("/")
        bases = [root] if not pieces[1] else [root/c for c in pieces[1].split(",")]
        bases += [root/"cpu,cpuacct"] if "cpu" in pieces[1].split(",") else []
        for base in bases:
            current = base/relative
            while current == root or root in current.parents:
                found.add(current)
                if current == root:
                    break
                current = current.parent
    return sorted(found, key=str)


def detect_resources(root: Path = Path("/sys/fs/cgroup"),
                     membership: str | None = None, affinity_count: int | None = None,
                     declared_vcpus: int = 14) -> dict:
    """输出有效可见配额；缺少配额时以用户声明14vCPU作为保守上限。"""
    if declared_vcpus < 1:
        raise ValueError("resources.declared_vcpus.invalid")
    affinity = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1) if affinity_count is None else affinity_count
    if affinity < 1:
        raise ValueError("resources.affinity.invalid")
    member = _read(Path("/proc/self/cgroup")) if membership is None else membership
    cpu, memory, evidence = [], [], []
    for directory in _directories(root, member or ""):
        text = _read(directory/"cpu.max")
        if text is not None:
            value = parse_cpu_max(text)
            evidence.append({"path": str(directory/"cpu.max"), "value": text})
            if value is not None:
                cpu.append(value)
        q, p = _read(directory/"cpu.cfs_quota_us"), _read(directory/"cpu.cfs_period_us")
        if q is not None and p is not None and int(q) > 0:
            cpu.append(parse_cpu_max(q+" "+p))
        for name in ("memory.max", "memory.limit_in_bytes"):
            value = _read(directory/name)
            if value is not None and value != "max" and 0 < int(value) < 2**60:
                memory.append(int(value))
                evidence.append({"path": str(directory/name), "value": value})
    effective = min([float(affinity), *cpu]) if cpu else min(affinity, declared_vcpus)
    return {"logical_cpus_visible": os.cpu_count(), "affinity_cpus": affinity,
        "quota_cpu_equivalents": min(cpu) if cpu else None,
        "effective_cpu_equivalents": effective,
        "memory_limit_bytes": min(memory) if memory else None,
        "declared_vcpus_fallback": declared_vcpus, "quota_known": bool(cpu),
        "evidence": evidence, "scope": "visible_cgroups_and_affinity_not_dedicated_cores"}


def worker_candidates(resources: dict, requested=(2, 4, 6, 8, 10, 12)) -> list[int]:
    """保留2核供主进程/系统；低配至少1worker，初次搜索上限12而非128。"""
    ceiling = max(1, math.floor(resources["effective_cpu_equivalents"])-2)
    memory = resources["memory_limit_bytes"]
    if memory is not None:
        # NOTE: 初始搜索预留2GiB；实测峰值仍必须在每个配置后检查。
        ceiling = min(ceiling, max(1, (memory-2*1024**3)//(768*1024**2)))
    ceiling = min(12, ceiling)
    choices = sorted({n for n in requested if type(n) is int and 1 <= n <= ceiling})
    return choices or [1]


def cgroup_counters() -> dict:
    """压测前后采样本cgroup计数；不是单一进程资源，也不是精确停顿比例。"""
    values = {}
    for name in ("cpu.stat", "memory.current", "memory.events"):
        text = _read(Path("/sys/fs/cgroup")/name)
        if text is None:
            continue
        if name == "memory.current":
            values["memory_current_bytes"] = int(text)
        else:
            for line in text.splitlines():
                key, value = line.split()
                values[name+"."+key] = int(value)
    return values
