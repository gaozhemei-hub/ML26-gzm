"""TabularX 工具包 - 面向可解释性与可复现性的自动化表格建模工具包。"""

from tabularx.core.params import (
    ModelSearchConfig,
    PreprocessConfig,
    ProfileConfig,
    SplitConfig,
    TabularXParams,
)
from tabularx.core.preprocess import preprocess
from tabularx.core.profile import (
    build_profile,
    build_column_fact,
    build_target_fact,
    ProfileUtils,
)
from tabularx.core.split import split_data
from tabularx.core.learning.model_registry import ModelRegistry, get_default_registry
from tabularx.core.learning.model_search import search_models
from tabularx.core.learning.ensemble_search import build_voting
from tabularx.core.learning.metrics import (
    resolve_primary_metric,
    compute_all_metrics,
    compute_regression_metrics,
    compute_classification_metrics,
)
from tabularx.schema.preprocess import PreprocessedData, TransformRecord
from tabularx.schema.profile import DataProfile, ColumnFact, TargetFact, Evidence
from tabularx.schema.split import SplitResult
from tabularx.schema.learning import (
    ModelSpec,
    ModelRunResult,
    ModelSearchResult,
    FairnessConfig,
)
from tabularx.exceptions import (
    TabularXError,
    DataError,
    ProfileError,
    InvalidDataError,
    EmptyDataError,
    MismatchedLengthError,
    EmptyTargetError,
    TaskTypeError,
    InsufficientDataError,
    SplitError,
    InvalidSplitError,
    StratifyError,
    PreprocessError,
    TextColumnNotSupportedError,
    TransformError,
    ModelSearchError,
    ModelNotAvailableError,
    ModelTrainingError,
    OptunaError,
)

__all__ = [
    # 核心函数
    "build_profile",
    "build_column_fact",
    "build_target_fact",
    "ProfileUtils",
    "split_data",
    "preprocess",
    # 模型搜索
    "ModelRegistry",
    "get_default_registry",
    "search_models",
    "build_voting",
    "resolve_primary_metric",
    "compute_all_metrics",
    "compute_regression_metrics",
    "compute_classification_metrics",
    # schema 类型
    "DataProfile",
    "ColumnFact",
    "TargetFact",
    "Evidence",
    "SplitResult",
    "PreprocessedData",
    "TransformRecord",
    "ModelSpec",
    "ModelRunResult",
    "ModelSearchResult",
    "FairnessConfig",
    # 参数配置
    "ProfileConfig",
    "SplitConfig",
    "PreprocessConfig",
    "ModelSearchConfig",
    "TabularXParams",
    # 异常
    "TabularXError",
    "DataError",
    "ProfileError",
    "InvalidDataError",
    "EmptyDataError",
    "MismatchedLengthError",
    "EmptyTargetError",
    "TaskTypeError",
    "InsufficientDataError",
    "SplitError",
    "InvalidSplitError",
    "StratifyError",
    "PreprocessError",
    "TextColumnNotSupportedError",
    "TransformError",
    "ModelSearchError",
    "ModelNotAvailableError",
    "ModelTrainingError",
    "OptunaError",
]