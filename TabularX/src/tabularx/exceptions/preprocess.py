"""Preprocess 阶段的异常定义。

这些异常在 core/preprocess.py 中抛出，pipeline 统一捕获。
全部继承 PreprocessError，同时继承 DataError（用户可通过改数据/参数修复）。
"""

from __future__ import annotations

from tabularx.exceptions import DataError


class PreprocessError(DataError):
    """Preprocess 阶段基础异常。"""

    def __init__(self, message: str = "", *, stage: str = "preprocess"):
        super().__init__(message, stage=stage)


class TextColumnNotSupportedError(PreprocessError):
    """发现文本列但 handle_text="keep"，暂不支持。"""


class TransformError(PreprocessError):
    """数据变换执行失败（如填补/编码/缩放出错）。"""


__all__ = [
    "PreprocessError",
    "TextColumnNotSupportedError",
    "TransformError",
]