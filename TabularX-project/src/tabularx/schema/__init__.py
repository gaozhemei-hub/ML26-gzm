"""Schema 数据类型定义 - 各阶段间的数据契约。"""

from tabularx.schema.preprocess import ColumnMapping, PreprocessedData, TransformRecord
from tabularx.schema.profile import (
    AuditCheckResult,
    AuditFinding,
    AuditSeverity,
    ColumnFact,
    DataProfile,
    Evidence,
    ProfileAuditResult,
    TargetFact,
)
from tabularx.schema.split import SplitResult
from tabularx.schema.learning import (
    ModelSpec,
    TrialRecord,
    OptunaResult,
    ModelRunResult,
    ModelSearchResult,
)

__all__ = [
    "DataProfile",
    "ColumnFact",
    "TargetFact",
    "Evidence",
    "AuditFinding",
    "AuditCheckResult",
    "AuditSeverity",
    "ProfileAuditResult",
    "SplitResult",
    "PreprocessedData",
    "TransformRecord",
    "ColumnMapping",
    "ModelSpec",
    "TrialRecord",
    "OptunaResult",
    "ModelRunResult",
    "ModelSearchResult",
]