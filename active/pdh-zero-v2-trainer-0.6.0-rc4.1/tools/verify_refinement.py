"""一次执行本候选包的CPU回归、文件校验及基础静态审计；不占用GPU。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def _hash_errors() -> list[str]:
    failures = []
    for line in (ROOT / 'FILES.sha256').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            failures.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(relative)
    return failures


def main() -> int:
    """验收日志放新建tmp目录，不覆盖发布证据；退出码0仅代表CPU程序门禁。"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'tmp' / ('verify-refinement-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    failures = _hash_errors()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT/'src'), str(ROOT))),
               CUDA_VISIBLE_DEVICES='', PYTHONWARNINGS='error')
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.', '-v'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
    raw = result.stdout + result.stderr
    (output/'tests.log').write_text(raw, encoding='utf-8')
    audit = subprocess.run([sys.executable, str(ROOT/'tools/static_audit.py'),
        '--output', str(output/'static.json')], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60, check=False)
    (output/'static.log').write_text(audit.stdout+audit.stderr, encoding='utf-8')
    matched = re.search(r'Ran (\d+) tests?', raw)
    passed = not failures and result.returncode == 0 and audit.returncode == 0
    report = dict(refinement_candidate_check='PASS' if passed else 'FAIL',
                  python=sys.version.split()[0], tests_run=int(matched[1]) if matched else None,
                  tests_exit_code=result.returncode, hash_errors=failures,
                  basic_static_exit_code=audit.returncode, refinement_entry_ready=passed,
                  formal_long_run_ready=False, target_cuda_verified=False,
                  ruff='NOT_RUN', mypy='NOT_RUN', report_directory=str(output))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    logging.info(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
