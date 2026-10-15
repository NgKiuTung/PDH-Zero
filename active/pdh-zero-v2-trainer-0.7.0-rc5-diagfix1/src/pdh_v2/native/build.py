"""PDH研究代码：单进程编译CPU扩展，不安装软件、不链接libtorch/CUDA。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

from pdh_v2.native.runtime import ROOT, source_digest


def build() -> dict:
    """有限时编译，日志持久化；原子发布通过链接的二进制及身份清单。"""
    import torch
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        raise RuntimeError("native.compiler.not_found")
    include = Path(torch.__file__).parent / "include"
    pyinclude = Path(sysconfig.get_paths()["include"])
    if not (include/"pybind11/pybind11.h").is_file() or not (pyinclude/"Python.h").is_file():
        raise RuntimeError("native.headers.missing")
    directory = ROOT / "build" / "native"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/".build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = directory / ("_pdh_native" + sysconfig.get_config_var("EXT_SUFFIX"))
        temporary = directory / ("_pdh_native.tmp" + sysconfig.get_config_var("EXT_SUFFIX"))
        command = [compiler, "-std=c++20", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                   "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror", "-I"+str(include),
                   "-I"+str(pyinclude), *map(str, sorted((ROOT/"native").glob("*.cpp"))),
                   "-o", str(temporary)]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=180, check=False)
            (directory/"compile.log").write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError("native.compile.failed: " + str(directory/"compile.log"))
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            report = {"status": "PASS", "source_sha256": source_digest(),
                      "binary_sha256": digest, "soabi": sysconfig.get_config_var("SOABI"),
                      "compiler": compiler, "command": command,
                      "cpu_only": True, "fast_math": False, "threads": 1}
            os.replace(temporary, target)
            temp_json = directory/"build.json.tmp"
            temp_json.write_text(json.dumps(report, indent=2)+"\n")
            os.replace(temp_json, directory/"build.json")
            return report
        finally:
            temporary.unlink(missing_ok=True)
