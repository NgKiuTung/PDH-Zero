"""PDH研究代码：配额感知的运行策略，不把宿主逻辑CPU当容器配额。"""
from pdh_v2.runtime.resources import detect_resources, worker_candidates

__all__ = ["detect_resources", "worker_candidates"]
