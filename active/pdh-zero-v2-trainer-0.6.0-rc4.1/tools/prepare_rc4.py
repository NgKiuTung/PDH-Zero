"""rc4唯一准备入口：核验源码/候选、编译/差分、限额负载实测后输出训练计划。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"tools"))
from pdh_v2.native.build import build
from pdh_v2.native.validation import verify_records
from pdh_v2.runtime import detect_resources
from pdh_v2.runtime.subprocesses import execute
from pdh_v2.training.selection_input import resolve_selection
from benchmark_runtime import benchmark


def main() -> int:
    """不启动训练；新目录输出可复核计划，失败明确保留阶段与日志。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--tables",type=int,default=64)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--python-only",action="store_true")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root=args.output_root or args.source_run.resolve().parent.parent/"v2-rc4-prepare"
    output=root.resolve()/("PREPARE_"+stamp)
    output.mkdir(parents=True,exist_ok=False)
    report={"status":"running","output":str(output),"training_started":False}
    try:
        selection=args.source_run.resolve()/"selected_development.json"
        _, candidate=resolve_selection(selection)
        report.update(source_candidate=candidate["label"],resources=detect_resources())
        (output/"resources.json").write_text(json.dumps(report["resources"],indent=2)+"\n")
        report["phase"]="build_native"
        logging.info(json.dumps(report,ensure_ascii=False))
        if not args.python_only:
            report["native_build"]=build()
        report["phase"]="verify_package"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PDH_FEATURE_BACKEND="python",
                 OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        code=execute([sys.executable,str(ROOT/"tools/verify_refinement.py")],
                     output/"package-tests",env,600)
        if code:
            raise RuntimeError("prepare.package_verification_failed")
        if not args.python_only:
            report["phase"]="native_real_replay_differential"
            replay=args.source_run/"evaluations"/candidate["label"]/"hands.jsonl.gz"
            report["differential"]=verify_records(replay,16)
        report["phase"]="benchmark"
        logging.info(json.dumps({"phase":"benchmark","resources":report["resources"]},
                                ensure_ascii=False))
        plan=benchmark(selection,output/"benchmark",args.device,args.tables,args.repeats,
                       enable_cpp=not args.python_only)
        path=output/"benchmark/runtime-plan.json"
        report.update(status="PASS",phase="ready",runtime_plan=str(path),
                      recommendation=plan["recommendation"],long_training_allowed=False)
        logging.info(json.dumps(report,ensure_ascii=False,indent=2))
        return 0
    except Exception as exc:
        report.update(status="FAIL",error_type=type(exc).__name__,error=str(exc))
        logging.exception(json.dumps(report,ensure_ascii=False))
        return 1
    finally:
        (output/"preparation.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    raise SystemExit(main())
