"""Preprocess 阶段的数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TransformRecord:
    """记录一列在预处理过程中的变换。"""

    col_name: str
    """原始列名"""
    original_dtype: str
    """原始 pandas dtype"""
    original_n_unique: int
    """原始唯一值数"""
    action_sequence: list[str]
    """变换动作序列，如 ["median_impute", "standard_scale"]"""
    dropped: bool = False
    """是否被丢弃"""
    n_features_out: int = 1
    """变换后展成的特征数（OneHot 后会 >1）"""
    note: str | None = None
    """补充说明"""


@dataclass
class ColumnMapping:
    """记录原始列 → 输出列的映射关系。

    用于追踪敏感特征经过预处理后的变换情况：
      - 丢弃 → output_names 为空列表
      - 原样保留或数值变换 → output_names 为 [原列名]
      - OneHot → output_names 为 [col_A, col_B, ...]
      - 日期提取 → output_names 为 [col_year, col_month, col_day]
    """

    original_name: str
    """原始列名"""
    output_names: list[str]
    """变换后的输出列名列表（空列表表示被丢弃）"""
    is_dropped: bool = False
    """是否被丢弃"""
    n_features_out: int = 1
    """输出特征数"""


@dataclass
class PreprocessedData:
    """预处理后的完整数据集。

    包含变换后的各子集以及每列的变换记录，
    不持有任何 sklearn transformer 对象，确保可序列化。
    """

    # ── 变换后的数据 ────────────────────────────────────────
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    # ── 元信息 ──────────────────────────────────────────────
    n_features_in: int
    """原始特征数（预处理前）"""
    n_features_out: int
    """预处理后的特征数（OneHot 展开后可能更多）"""
    dropped_columns: list[str]
    """被丢弃的列名"""
    feature_names_out: list[str]
    """预处理后的特征名列表（对下游 ModelSearch 重要：解释特征重要性时需要知道名字）"""
    transforms: list[TransformRecord] = field(default_factory=list)
    """每列的变换记录"""
    column_mappings: list[ColumnMapping] = field(default_factory=list)
    """原始列到输出列的映射（用于追踪敏感特征等）"""


__all__ = [
    "TransformRecord",
    "ColumnMapping",
    "PreprocessedData",
]