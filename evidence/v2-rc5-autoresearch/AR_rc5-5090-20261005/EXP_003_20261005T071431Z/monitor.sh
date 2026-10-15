#!/usr/bin/env bash

BASE="/root/rivermind-data/pdh-zero"
EXP3="$(cat "$BASE/state/current_rc5_smoke_exp.txt")"
LPID="$(cat "$EXP3/launcher.pid")"

while true; do
    clear

    echo "============================================================"
    echo " PDH-Zero rc5 / EXP_003 live monitor"
    echo " $(date '+%F %T %Z')"
    echo "============================================================"
    echo
    echo "EXP3=$EXP3"
    echo "LAUNCHER_PID=$LPID"

    echo
    echo "===== PROCESS STATE ====="

    if kill -0 "$LPID" 2>/dev/null; then
        echo "LAUNCHER_STATE=RUNNING"
        ps -o pid=,ppid=,pgid=,sid=,stat=,etime=,%cpu=,%mem=,rss=,cmd= \
            -p "$LPID" 2>/dev/null || true
    else
        echo "LAUNCHER_STATE=EXITED"
    fi

    TOOL_PID=""

    if [ -f "$EXP3/study/process.json" ]; then
        TOOL_PID="$(
            python3 - <<PY 2>/dev/null
import json
from pathlib import Path

p = Path("$EXP3/study/process.json")
try:
    print(json.loads(p.read_text())["pid"])
except Exception:
    pass
PY
        )"
    fi

    if [ -n "$TOOL_PID" ]; then
        echo
        echo "TOOL_PID=$TOOL_PID"

        if kill -0 "$TOOL_PID" 2>/dev/null; then
            echo "TOOL_STATE=RUNNING"
            ps -o pid=,ppid=,pgid=,sid=,stat=,etime=,%cpu=,%mem=,rss=,cmd= \
                -p "$TOOL_PID" 2>/dev/null || true
        else
            echo "TOOL_STATE=EXITED"
        fi

        echo
        echo "===== PROCESS TREE ====="
        ps -eo pid,ppid,pgid,stat,etime,%cpu,%mem,cmd \
            --forest \
            | grep -E "PID|$LPID|$TOOL_PID|pdh_v2|tools/rc5.py" \
            | grep -v grep || true
    else
        echo "TOOL_PROCESS_JSON=NOT_READY"
    fi

    echo
    echo "===== GPU ====="
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader 2>/dev/null || true

    echo
    echo "===== ARTIFACT ACTIVITY ====="
    for f in \
        "$EXP3/study/events.jsonl" \
        "$EXP3/study/results.tsv" \
        "$EXP3/study/report.json" \
        "$EXP3/study.stdout.log" \
        "$EXP3/launcher.stdout.log"
    do
        if [ -e "$f" ]; then
            stat -c '%y  %s bytes  %n' "$f"
        else
            echo "MISSING  $f"
        fi
    done

    echo
    echo "===== LAST 12 EVENTS ====="
    tail -n 12 "$EXP3/study/events.jsonl" 2>/dev/null || echo "(no events yet)"

    echo
    echo "===== LAST 20 STUDY LOG LINES ====="
    tail -n 20 "$EXP3/study.stdout.log" 2>/dev/null || echo "(no stdout yet)"

    echo
    echo "===== EXIT / REPORT ====="

    if [ -f "$EXP3/launcher.exitcode" ]; then
        printf 'EXIT_CODE='
        cat "$EXP3/launcher.exitcode"
    else
        echo "EXIT_CODE=NOT_WRITTEN_YET"
    fi

    if [ -f "$EXP3/study/report.json" ]; then
        echo
        echo "REPORT=AVAILABLE"
        python3 - <<PY
import json
from pathlib import Path

r = json.loads(Path("$EXP3/study/report.json").read_text())

for k in (
    "status",
    "scientific_status",
    "elapsed_seconds",
    "completed_seeds",
    "completed_arms",
    "deployment_promoted",
):
    if k in r:
        print(f"{k} = {r[k]}")
PY
    else
        echo "REPORT=NOT_READY"
    fi

    # launcher 已结束时再等两秒，让 exitcode/report 完成落盘，然后退出监控。
    if ! kill -0 "$LPID" 2>/dev/null; then
        sleep 2

        echo
        echo "============================================================"
        echo " EXP_003 launcher has exited"
        echo "============================================================"

        if [ -f "$EXP3/launcher.exitcode" ]; then
            printf 'FINAL_EXIT_CODE='
            cat "$EXP3/launcher.exitcode"
        else
            echo "FINAL_EXIT_CODE=UNKNOWN"
        fi

        if [ -f "$EXP3/study/report.json" ]; then
            echo
            echo "===== FINAL REPORT ====="
            python3 -m json.tool "$EXP3/study/report.json"
        else
            echo "FINAL_REPORT=MISSING"
        fi

        echo
        echo "===== FINAL LOG ====="
        tail -n 120 "$EXP3/study.stdout.log" 2>/dev/null || true

        break
    fi

    echo
    echo "refresh: 10 seconds | Ctrl+C only stops monitor, not experiment"
    sleep 10
done
