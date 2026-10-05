"""编译本项目C++20 CPU特征核；无需nvcc或独立pybind11安装。"""
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pdh_v2.native.build import build


def main() -> int:
    """退出码0只代表扩展构建通过；不代表正确性或目标机性能通过。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        logging.info(json.dumps(build(), ensure_ascii=False, indent=2))
        return 0
    except Exception:
        logging.exception("native.build.failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
