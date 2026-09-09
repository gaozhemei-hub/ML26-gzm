"""Split 阶段的异常定义。

这些异常在 core/split.py 中抛出，pipeline 统一捕获。
全部继承 SplitError，同时继承 DataError（用户可通过改数据/参数修复）。
"""

from __future__ import annotations

from tabularx.exceptions import DataError


class SplitError(DataError):
    """Split 阶段基础异常。"""

    def __init__(self, message: str = "", *, stage: str = "split"):
        super().__init__(message, stage=stage)


class InvalidSplitError(SplitError):
    """划分参数无效（如 test_size + validation_size 超过 1.0、子集样本数不足等）。"""


class StratifyError(SplitError):
    """分层采样失败（如某个类别在子集中样本数不足）。"""


__all__ = [
    "SplitError",
    "InvalidSplitError",
    "StratifyError",
]