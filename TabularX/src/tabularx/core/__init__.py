"""TabularX core 模块 - 纯函数逻辑，零 IO。"""

from tabularx.core.params import (
    PreprocessConfig,
    ProfileConfig,
    SplitConfig,
    TabularXParams,
)
from tabularx.core.preprocess import preprocess
from tabularx.core.split import split_data

__all__ = [
    # 参数配置
    "ProfileConfig",
    "SplitConfig",
    "PreprocessConfig",
    "TabularXParams",
    # 核心函数
    "preprocess",
    "split_data",
]