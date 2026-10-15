"""PDH研究代码：显式选择native后端，禁止静默降级或加载过期构建。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[3]
_MODULE = None


def source_digest() -> str:
    """绑定全部本项目native源码；构建机路径不参与哈希。"""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "native").glob("*")):
        if path.suffix in {".cpp", ".hpp"}:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def backend_name() -> str:
    """子进程继承显式后端，不根据装没装扩展自行改变实验。"""
    name = os.environ.get("PDH_FEATURE_BACKEND", "python")
    if name not in {"python", "cpp"}:
        raise ValueError("native.backend.invalid")
    return name


def configure(name: str) -> None:
    """只影响当前进程及其后代；C++不可用则报错，不偷偷换实现。"""
    if name not in {"python", "cpp"}:
        raise ValueError("native.backend.invalid")
    if name == "cpp":
        load_extension()
    os.environ["PDH_FEATURE_BACKEND"] = name


def load_extension():
    """检验本Python ABI、源码及二进制哈希后，加载项目内扩展。"""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    directory = ROOT / "build" / "native"
    manifest = json.loads((directory / "build.json").read_text())
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    path = directory / ("_pdh_native" + suffix)
    if (manifest["source_sha256"] != source_digest()
            or manifest["soabi"] != sysconfig.get_config_var("SOABI")
            or manifest["binary_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("native.build.identity_mismatch")
    spec = importlib.util.spec_from_file_location("_pdh_native", path)
    if spec is None or spec.loader is None:
        raise ImportError("native.build.import_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCHEMA_VERSION != 1:
        raise ValueError("native.schema_mismatch")
    sys.modules["_pdh_native"] = module
    _MODULE = module
    return module
