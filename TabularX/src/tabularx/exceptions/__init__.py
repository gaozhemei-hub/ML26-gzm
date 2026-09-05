"""TabularX 自定义异常。

分层设计：
- TabularXError（基类）：所有异常的父类，pipeline 只需 catch 这一个。
- DataError（基类）：数据类问题（用户可修复——改数据/改参数）。
- 具体异常：按阶段和可修复性归类。

使用方式（在 core 中抛出）：
    from tabularx.exceptions import InvalidDataError
    raise InvalidDataError("X 必须是 pd.DataFrame ...")
"""

from __future__ import annotations

# 注意：子模块导入必须放在 TabularXError 定义之后（避免循环导入）
# 顺序：先定义基类 → 再导入子模块 → 最后导出所有


class TabularXError(Exception):
    """所有 TabularX 异常的基类。

    pipeline 统一捕获本类，将错误信息记录到 run_metadata.json。
    """

    def __init__(self, message: str = "", *, stage: str | None = None):
        super().__init__(message)
        self.message = message
        self.stage = stage  # 错误发生的阶段，如 "profile"

    def __str__(self) -> str:
        if self.stage:
            return f"[{self.stage}] {self.message}"
        return self.message


class DataError(TabularXError):
    """数据类错误：用户可以通过修改数据或参数修复。

    例如：输入类型错误、数据为空、X/y 长度不一致、目标全空、任务类型非法。
    """


# 在基类定义之后导入子阶段异常
from tabularx.exceptions.profile import (  # noqa: E402
    EmptyDataError,
    EmptyTargetError,
    InvalidDataError,
    MismatchedLengthError,
    ProfileError,
    TaskTypeError,
)


__all__ = [
    # 基类
    "TabularXError",
    "DataError",
    # profile 阶段
    "ProfileError",
    "InvalidDataError",
    "EmptyDataError",
    "MismatchedLengthError",
    "EmptyTargetError",
    "TaskTypeError",
]

