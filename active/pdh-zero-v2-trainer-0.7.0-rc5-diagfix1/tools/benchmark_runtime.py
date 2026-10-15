"""固定真实模型，顺序测试容器配额内的Python/C++负载，生成可验证运行计划。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import sysconfig

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pdh_v2.runtime import detect_resources, worker_candidates
from pdh_v2.runtime.measure import measure_trial
from pdh_v2.runtime.plan import choose_trial, code_digest
from pdh_v2.runtime.subprocesses import execute
from pdh_v2.training.selection_input import resolve_selection


def benchmark(selection: Path, output: Path, device="cuda", tables=64, repeats=2,
              enable_cpp: bool = True) -> dict:
    """一次仅运行一个负载组；预期14核依次测2/4/6/8/10/12，不使用128并发。"""
    import torch
    import subprocess
    if device=="cuda":
        probe=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10,check=True)
        if any(int(x.strip())!=os.getpid() for x in probe.stdout.splitlines() if x.strip()):
            raise RuntimeError("benchmark.gpu_occupied")
    source, record=resolve_selection(selection)
    resources=detect_resources()
    choices=worker_candidates(resources)
    base=min(choices,key=lambda n:abs(n-4))
    profiles=[("python",base)]+[("cpp" if enable_cpp else "python",n) for n in choices
                                if enable_cpp or n!=base]
    output.mkdir(parents=True,exist_ok=False)
    trials=[]
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             NUMEXPR_NUM_THREADS="1")
    for backend,workers in profiles:
        if workers not in worker_candidates(detect_resources()):
            raise RuntimeError("benchmark.resources_changed")
        label=f"{backend}-w{workers}"
        result_path=output/(label+".json")
        logging.info(json.dumps({"event":"benchmark_profile_started","profile":label},
                                ensure_ascii=False))
        command=[sys.executable,str(ROOT/"tools/benchmark_runtime.py"),"--trial",
                 "--selection",str(selection.resolve()),"--output",str(result_path),
                 "--device",device,"--backend",backend,"--workers",str(workers),
                 "--tables",str(tables),"--repeats",str(repeats)]
        code=execute(command,output/(label+"-process"),env,timeout=240)
        if code!=0 or not result_path.is_file():
            raise RuntimeError("benchmark.profile.failed: "+label)
        trial=json.loads(result_path.read_text())
        if trial["checkpoint_sha256"]!=record["checkpoint_sha256"]:
            raise RuntimeError("benchmark.checkpoint_changed")
        trial["status"]="PASS"
        trials.append(trial)
        memory=resources["memory_limit_bytes"]
        if memory and any((m.get("cgroup_memory_current_bytes") or 0)>memory*.85
                          for m in trial["repeats"]):
            raise RuntimeError("benchmark.memory_safety_margin")
    recommendation=choose_trial(trials)
    plan={"status":"PASS","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "selection_path":str(selection.resolve()),"checkpoint_sha256":record["checkpoint_sha256"],
        "code_sha256":code_digest(),"soabi":sysconfig.get_config_var("SOABI"),
        "torch":torch.__version__,"device":device,
        "gpu_name":torch.cuda.get_device_name(0) if device=="cuda" else None,
        "resources":resources,"tested_profiles":profiles,"trials":trials,
        "recommendation":recommendation,"training_started":False,
        "long_training_allowed":False}
    (output/"runtime-plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    return plan


def main() -> int:
    """JSON仅报告实测，后台子进程有身份记录且退出回收。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--tables",type=int,default=64)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--trial",action="store_true")
    parser.add_argument("--backend",choices=["python","cpp"],default="python")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device=="cuda" and not args.trial:
        import subprocess
        probe=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10,check=True)
        if probe.stdout.strip():
            raise RuntimeError("benchmark.gpu_occupied")
    if args.trial:
        report=measure_trial(args.selection,args.device,args.backend,args.workers,
                             args.tables,args.repeats)
        with args.output.open("x") as stream:
            json.dump(report,stream,indent=2)
    else:
        report=benchmark(args.selection,args.output,args.device,args.tables,args.repeats)
    logging.info(json.dumps(report.get("recommendation", {"status":"PASS"})))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
