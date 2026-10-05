"""输入边界契约；许可由项目所有者决定。异常用稳定代码供测试和日志消费。"""
from typing import Any
from pdh_v2.config import message


class RuleError(ValueError):
    """确定的输入错误或明确未决房规；不在内部替换为默认行为。"""

    def __init__(self, code: str, **details: object) -> None:
        self.code = code
        self.details = details
        super().__init__(message(code, **details))


def integer(value: Any, field: str, low: int, high: int) -> int:
    """验证有界整数；显式拒绝bool，防止True被当作一个玩家或一张牌。"""
    if type(value) is not int or not low <= value <= high:
        raise RuleError("input.integer", field=field, low=low, high=high)
    return value


def text(value: Any, field: str) -> str:
    """验证持久化标识；不将标识解释为文件路径或可执行内容。"""
    if not isinstance(value, str) or not value.strip():
        raise RuleError("input.text", field=field)
    return value
