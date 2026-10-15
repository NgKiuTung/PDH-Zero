"""根据课程学习产物判断是否值得生成下一版正式长训练包。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean


def _read_json(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                result.append(dict(json.loads(line)))
    return result


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def main() -> int:
    """输出可审计门禁结论；失败不会修改训练产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = _read_json(root / "manifest.json")
    metrics = _read_jsonl(root / "metrics.jsonl")
    events = _read_jsonl(root / "events.jsonl")
    evaluations = [
        _read_json(path)
        for path in sorted((root / "evaluations").glob("evaluation-d*.json"))
    ]
    reasons: list[str] = []
    if manifest.get("status") != "completed":
        reasons.append("manifest.status is not completed")
    if len(metrics) < 3:
        reasons.append("fewer than three PPO updates")
    if len(evaluations) < 2:
        reasons.append("fewer than two fixed evaluations")

    tail = metrics[-min(5, len(metrics)) :]
    tail_draw = fmean(float(item["draw_rate"]) for item in tail) if tail else 1.0
    tail_opportunities = sum(int(item["win_opportunities"]) for item in tail)
    tail_wins_taken = sum(int(item["wins_taken"]) for item in tail)
    tail_capture = (
        tail_wins_taken / tail_opportunities if tail_opportunities else 0.0
    )
    numeric = [
        float(item[key])
        for item in tail
        for key in (
            "policy_loss",
            "value_loss",
            "risk_loss",
            "entropy",
            "approximate_kl",
            "gradient_norm",
        )
    ]
    if not _finite(numeric):
        reasons.append("non-finite training metric")
    if tail_draw > 0.90:
        reasons.append(f"tail mixed-opponent draw rate remains too high: {tail_draw:.4f}")
    if tail_opportunities == 0:
        reasons.append("tail updates contain no legal win opportunities")
    elif tail_capture < 0.98:
        reasons.append(f"legal win capture rate is below 98%: {tail_capture:.4f}")

    warmups = [item for item in events if item.get("event") == "heuristic_warmup_completed"]
    warmup_accuracy = None
    warmup_improvement = None
    if not warmups:
        reasons.append("heuristic warmup event is missing")
    else:
        warmup = warmups[-1]
        before = dict(warmup["validation_before"])
        after = dict(warmup["validation_after"])
        warmup_accuracy = float(after["accuracy"])
        warmup_improvement = warmup_accuracy - float(before["accuracy"])
        if warmup_accuracy < 0.65:
            reasons.append(
                f"teacher validation accuracy below 65%: {warmup_accuracy:.4f}"
            )
        if warmup_improvement < -0.01:
            reasons.append(
                "teacher validation accuracy degraded by more than one point: "
                f"{warmup_improvement:.4f}"
            )

    final_stage = str(tail[-1]["curriculum_stage"]) if tail else None
    if final_stage == "GUIDED_SOLO":
        reasons.append("curriculum never advanced beyond the solo guided stage")

    improvement = None
    latest_point_shoot = None
    if len(evaluations) >= 2:
        first = evaluations[0]
        latest = evaluations[-1]
        improvement = float(latest["mean_score"]) - float(first["mean_score"])
        latest_point_shoot = float(latest["point_shoot_rate"])
        if improvement < 0.10:
            reasons.append(f"evaluation mean-score improvement below 0.10: {improvement:.4f}")
        if latest_point_shoot > float(first["point_shoot_rate"]) + 0.01:
            reasons.append("evaluation point-shoot rate worsened by more than one point")

    report = {
        "schema_version": 1,
        "run_dir": str(root),
        "curriculum_gate": "PASS" if not reasons else "FAIL",
        "updates": len(metrics),
        "evaluations": len(evaluations),
        "tail_draw_rate": tail_draw,
        "tail_win_opportunities": tail_opportunities,
        "tail_win_capture_rate": tail_capture,
        "teacher_validation_accuracy": warmup_accuracy,
        "teacher_validation_improvement": warmup_improvement,
        "final_curriculum_stage": final_stage,
        "evaluation_mean_score_improvement": improvement,
        "latest_point_shoot_rate": latest_point_shoot,
        "reasons": reasons,
        "next_action": (
            "build_long_run_package" if not reasons else "iterate_curriculum_or_model"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
