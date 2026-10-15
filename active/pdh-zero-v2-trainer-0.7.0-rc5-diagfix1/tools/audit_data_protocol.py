"""只读审计完整局长、牌墙重复、训练开发重叠与抽样规则重放。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pdh_v2.training.data_audit import audit_run


def main() -> int:
    """JSON报告写到新文件；只做离线审计，不占GPU、不更改源实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--replay-samples', type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        report = audit_run(args.run_dir, args.replay_samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        logging.info(json.dumps({'status': report['status'], 'report': str(args.output)}))
        return 0 if report['status'] == 'PASS' else 2
    except Exception as exc:
        logging.exception(json.dumps({'status': 'FAIL', 'error': str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
