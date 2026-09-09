"""TabularX 工具包 - 面向可解释性与可复现性的自动化表格建模工具包。"""

from tabularx.core.params import (
    PreprocessConfig,
    ProfileConfig,
    SplitConfig,
    TabularXParams,
)
from tabularx.core.profile import (
    build_profile,
    build_column_fact,
    build_target_fact,
    ProfileUtils,
)
from tabularx.schema.profile import DataProfile, ColumnFact, TargetFact, Evidence
from tabularx.exceptions import (
    TabularXError,
    DataError,
    ProfileError,
    InvalidDataError,
    EmptyDataError,
    MismatchedLengthError,
    EmptyTargetError,
    TaskTypeError,
)

__all__ = [
    # 核心函数
    "build_profile",
    "build_column_fact",
    "build_target_fact",
    "ProfileUtils",
    # schema 类型
    "DataProfile",
    "ColumnFact",
    "TargetFact",
    "Evidence",
    # 参数配置
    "ProfileConfig",
    "SplitConfig",
    "PreprocessConfig",
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
]