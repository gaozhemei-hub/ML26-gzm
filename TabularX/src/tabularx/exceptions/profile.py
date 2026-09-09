"""Profile 阶段的异常定义。

这些异常在 core/profile.py 中抛出，pipeline 统一捕获。
全部继承 ProfileError，同时继承 DataError（用户可通过改数据/参数修复）。
"""

from __future__ import annotations

from tabularx.exceptions import DataError


class ProfileError(DataError):
    """Profile 阶段基础异常。"""

    def __init__(self, message: str = "", *, stage: str = "profile"):
        super().__init__(message, stage=stage)


class InvalidDataError(ProfileError):
    """输入类型不正确（X 非 DataFrame / y 非 Series）。"""


class EmptyDataError(ProfileError):
    """数据为空（X 0 行 0 列）。"""


class MismatchedLengthError(ProfileError):
    """X 与 y 行数不一致。"""


class EmptyTargetError(ProfileError):
    """y 全为空，无法推断任务类型。"""


class TaskTypeError(ProfileError):
    """任务类型非法或自动推断失败。"""


class InsufficientDataError(ProfileError):
    """数据量不足，无法进行有意义的划分。

    例如：回归任务样本数 < min_samples_regression，
          分类任务某类别样本数 < min_samples_per_class。
    """


__all__ = [
    "ProfileError",
    "InvalidDataError",
    "EmptyDataError",
    "MismatchedLengthError",
    "EmptyTargetError",
    "TaskTypeError",
    "InsufficientDataError",
]
