"""PDH研究代码：限时、单任务、可定位子进程组；只清理本次创建的进程。"""
import json
import os
from pathlib import Path
import signal
import subprocess


def execute(command: list[str], directory: Path, env: dict, timeout: float) -> int:
    """输出/PID先登记；超时先TERM再KILL本组，绝不按进程名称全局清理。"""
    directory.mkdir(parents=True,exist_ok=False)
    with (directory/"stdout.log").open("w") as out, (directory/"stderr.log").open("w") as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
        identity={"pid":process.pid,"pgid":process.pid,"command":command,"status":"running"}
        (directory/"process.json").write_text(json.dumps(identity,indent=2)+"\n")
        try:
            code=process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=5)
            identity["status"]="interrupted_or_timeout"
            raise
        finally:
            # NOTE: 异常退出的父进程也可能留下worker；PGID只属于本次创建的会话。
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            identity["exit_code"]=process.poll()
            if identity["status"]=="running":
                identity["status"]="exited"
            (directory/"process.json").write_text(json.dumps(identity,indent=2)+"\n")
    return code
