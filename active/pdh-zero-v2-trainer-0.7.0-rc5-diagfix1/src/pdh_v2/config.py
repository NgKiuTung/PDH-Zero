"""只加载本包确认常量；许可由项目所有者决定。"""
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any


@lru_cache(maxsize=None)
def resource(name: str) -> dict[str, Any]:
    """读取内部JSON资源；返回对象只供读取，不接受外部路径。"""
    return dict(json.loads(files("pdh_v2.resources").joinpath(name).read_text("utf-8")))


def message(code: str, **details: object) -> str:
    """根据消息键格式化中文错误；缺失键作为编程错误直接暴露。"""
    template = str(resource("messages.json")[code])
    return template.format(**details)


RULES = resource("confirmed.json")
SOURCE_SHA256: str = str(RULES["source_sha256"])
TILE_TYPES: int = int(RULES["tile_types"])
COPIES: int = int(RULES["copies_per_type"])
DECK_SIZE: int = TILE_TYPES * COPIES
PLAYERS: int = int(RULES["players"])
