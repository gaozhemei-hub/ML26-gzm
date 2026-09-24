"""TabularX core 模块 - 纯函数逻辑，零 IO。"""

from tabularx.core.params import (
    ModelSearchConfig,
    PreprocessConfig,
    ProfileConfig,
    SplitConfig,
    TabularXParams,
)
from tabularx.core.preprocess import preprocess
from tabularx.core.split import split_data
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
    render_template,
)


__all__ = [
    # 参数配置
    "ProfileConfig",
    "SplitConfig",
    "PreprocessConfig",
    "ModelSearchConfig",
    "TabularXParams",
    # 核心函数
    "preprocess",
    "split_data",
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
    "render_template",
]