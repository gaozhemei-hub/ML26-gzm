"""Schema 数据类型定义 - 各阶段间的数据契约。"""

from tabularx.schema.preprocess import PreprocessedData, TransformRecord
from tabularx.schema.profile import ColumnFact, DataProfile, Evidence, TargetFact
from tabularx.schema.split import SplitResult
from tabularx.schema.learning import (
    ModelSpec,
    TrialRecord,
    OptunaResult,
    ModelRunResult,
    ModelSearchResult,
    FairnessConfig,
)

__all__ = [
    "DataProfile",
    "ColumnFact",
    "TargetFact",
    "Evidence",
    "SplitResult",
    "PreprocessedData",
    "TransformRecord",
    "ModelSpec",
    "TrialRecord",
    "OptunaResult",
    "ModelRunResult",
    "ModelSearchResult",
    "FairnessConfig",
]