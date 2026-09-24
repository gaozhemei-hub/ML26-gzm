"""TabularX 工具包 - 面向可解释性与可复现性的自动化表格建模工具包。"""

from tabularx.core.params import (
    ModelSearchConfig,
    PreprocessConfig,
    ProfileConfig,
    SplitConfig,
    TabularXParams,
    ShapConfig,
    FairnessConfig,
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
from tabularx.core.learning import explain_model, evaluate_fairness
from tabularx.core.learning.metrics import (
    resolve_primary_metric,
    compute_all_metrics,
    compute_regression_metrics,
    compute_classification_metrics,
)
from tabularx.core.reports import (
    ReportType,
    render_report,
    render_profile_report,
    render_split_report,
    render_preprocess_report,
    render_model_search_report,
    render_model_detail_report,
    render_voting_report,
    render_explanation_report,
    render_full_report,
)
from tabularx.schema.preprocess import PreprocessedData, TransformRecord
from tabularx.schema.profile import DataProfile, ColumnFact, TargetFact, Evidence
from tabularx.schema.split import SplitResult
from tabularx.schema.learning import (
    ModelSpec,
    ModelRunResult,
    ModelSearchResult,
    FairnessResult,
    FairnessFeatureResult,
    FairnessGroupResult,
    ShapResult,
)
from tabularx.schema.estimator import AutoResult
from tabularx.estimator import AutoEstimator
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
    ShapError,
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
    "explain_model",
    "evaluate_fairness",
    "resolve_primary_metric",
    "compute_all_metrics",
    "compute_regression_metrics",
    "compute_classification_metrics",
    # 报告渲染
    "ReportType",
    "render_report",
    "render_profile_report",
    "render_split_report",
    "render_preprocess_report",
    "render_model_search_report",
    "render_model_detail_report",
    "render_voting_report",
    "render_explanation_report",
    "render_full_report",
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
    "FairnessResult",
    "FairnessFeatureResult",
    "FairnessGroupResult",
    "ShapResult",
    # 高阶封装
    "AutoEstimator",
    "AutoResult",
    # 参数配置
    "ProfileConfig",
    "SplitConfig",
    "PreprocessConfig",
    "ModelSearchConfig",
    "ShapConfig",
    "FairnessConfig",
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
    "ShapError",
    "FairnessError",
]