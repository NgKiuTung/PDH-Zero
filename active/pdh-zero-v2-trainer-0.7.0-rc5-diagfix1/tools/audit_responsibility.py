"""只读回算真实评估的自责分/防御率，并抽样重放实际尾部供牌链。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pdh_v2.replay.audit import audit_terminal_record
from pdh_v2.replay.supply import trace_record
from pdh_v2.training.responsibility_metrics import responsibility_row, summarize_responsibility


def audit_run(run: Path, output: Path, tail_hands: int = 20) -> dict:
    """不更改旧训练产物；全量核对行账本，供牌链仅覆盖明确列出的样本。"""
    if tail_hands < 0 or tail_hands > 256:
        raise ValueError("audit.tail_limit.invalid")
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for directory in sorted((run/"evaluations").iterdir()):
        if not (directory/"results.jsonl").is_file():
            continue
        summary = json.loads((directory/"summary.json").read_text())
        content = (directory/"results.jsonl").read_bytes()
        if hashlib.sha256(content).hexdigest() != summary["results_sha256"]:
            raise ValueError("audit.evaluation.hash_mismatch")
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) != summary["hands"]:
            raise ValueError("audit.evaluation.row_count_mismatch")
        report = summarize_responsibility(rows)
        expected = sum(row["score"] for row in rows)
        if expected != report["total_income"]-report["total_outgoing"]:
            raise ValueError("audit.net_score_mismatch")
        report["results_sha256"] = summary["results_sha256"]
        worst = sorted(rows, key=lambda r: (r["score"], r["wall_seed"], r["candidate_seat"]))
        selected = {r["state_sha256"] for r in worst[:tail_hands]}
        traced = []
        with gzip.open(directory/"hands.jsonl.gz", "rt") as stream:
            for line in stream:
                record = json.loads(line)
                if record["transcript"]["state_sha256"] not in selected:
                    continue
                audit_terminal_record(record)
                traced.append(trace_record(record))
        if set(r["state_sha256"] for r in traced) != selected:
            raise ValueError("audit.tail_transcript.missing")
        report["tail_supply_replays_verified"] = len(traced)
        report["supply_scope"] = "sampled_worst_net_hands_only_not_all_hands"
        (output/(directory.name+"-supply.json")).write_text(json.dumps(traced, indent=2)+"\n")
        reports[directory.name] = report
    if not reports:
        raise ValueError("audit.no_evaluations")
    result = {"status": "PASS", "source_run_name": run.name,
              "reward_modified": False, "evaluations": reports,
              "causal_contribution": "NOT_ESTIMATED"}
    (output/"responsibility-report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main() -> int:
    """新目录保存报告，失败退出非零；聚合不变更旧评估门禁。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-hands", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    report = audit_run(args.run_dir,args.output,args.tail_hands)
    logging.info(json.dumps({"status":report["status"],"output":str(args.output),
                            "evaluations":len(report["evaluations"])},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
