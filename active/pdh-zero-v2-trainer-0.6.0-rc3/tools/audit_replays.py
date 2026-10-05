"""只读验证gzip牌谱中的手局与付款，不修改原始记录。"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdh_v2.replay.audit import audit_terminal_record


def main() -> int:
    """返回首个失败行；不跳过坏记录后声称全部通过。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    args=parser.parse_args()
    count=0
    try:
        with gzip.open(args.replay,'rt',encoding='utf-8') as stream:
            for count,line in enumerate(stream,1):
                audit_terminal_record(json.loads(line))
        if count==0:
            raise ValueError('empty replay')
        print(json.dumps({'status':'PASS','hands':count}))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'FAIL','line':count,'error':str(exc)}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
