"""只读重算稳定调优门禁：0=PASS，2=REGRESSION，3=INCONCLUSIVE/INCOMPLETE。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from pdh_v2.training.refinement_gate import review_run


def main() -> int:
    """保留原始结果；不原地生成或覆盖用户证据文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        result = review_run(args.run_dir)
        logging.info(json.dumps(result, ensure_ascii=False, indent=2))
        return {'PASS':0, 'REGRESSION':2}.get(result['gate'], 3)
    except (OSError, ValueError, KeyError) as exc:
        logging.error(json.dumps({'gate':'ERROR', 'error':str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
