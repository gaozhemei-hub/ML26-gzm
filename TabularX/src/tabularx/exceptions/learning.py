"""Learning 阶段的异常定义。

这些异常在 core/learning 中抛出，pipeline 统一捕获。
全部继承 ModelSearchError。
"""

from __future__ import annotations

from tabularx.exceptions import DataError


class ModelSearchError(DataError):
    """Learning 阶段基础异常。"""

    def __init__(self, message: str = "", *, stage: str = "model_search"):
        super().__init__(message, stage=stage)


class ModelNotAvailableError(ModelSearchError):
    """模型不可用：未注册、依赖缺失、或任务类型不匹配。"""


class ModelTrainingError(ModelSearchError):
    """模型训练或评估执行失败。"""


class OptunaError(ModelSearchError):
    """Optuna 调参过程出错。"""


__all__ = [
    "ModelSearchError",
    "ModelNotAvailableError",
    "ModelTrainingError",
    "OptunaError",
]