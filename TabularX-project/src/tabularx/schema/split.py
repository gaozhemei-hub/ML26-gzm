"""Split 阶段的数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tabularx.schema.audit import SplitAuditResult
from tabularx.schema.profile import Evidence


@dataclass
class SplitResult:
    """划分结果：只存索引和元信息，不存实际数据。

    Preprocess 阶段通过 train_indices / valid_indices / test_indices
    从原始 X, y 中切片获取各子集数据。
    """

    # ── 划分索引（下游 Preprocess 使用）──
    train_indices: np.ndarray
    """训练集在原始 X/y 中的行索引"""
    valid_indices: np.ndarray
    """验证集在原始 X/y 中的行索引"""
    test_indices: np.ndarray
    """测试集在原始 X/y 中的行索引"""

    # ── 划分元信息（复现用）──
    test_size: float
    """测试集占原始总样本的比例"""
    validation_size: float
    """验证集占去掉测试集后剩余部分的比例"""
    random_state: int
    """随机种子"""
    stratify: bool
    """是否使用了分层采样"""

    # ── 各集大小统计（方便查看，无需重新计算）──
    n_train: int
    n_valid: int
    n_test: int
    train_ratio: float
    """训练集占原始总样本的实际比例"""
    valid_ratio: float
    """验证集占原始总样本的实际比例"""
    test_ratio: float
    """测试集占原始总样本的实际比例"""

    # ── 判断依据（报告用）──
    evidences: list[Evidence] = field(default_factory=list)

    # ── 切分策略与结构信息（阶段 2 新增，带默认值保证向后兼容）──
    strategy: str = "stratified"
    """实际使用的切分策略：stratified / random / group / temporal。"""
    group_column: str | None = None
    """声明的实体分组列（未声明为 None）。"""
    time_column: str | None = None
    """声明的时间列（未声明为 None）。"""
    audit: SplitAuditResult | None = None
    """划分完整性审计结果；未启用审计时为 None。"""

    @property
    def n_total(self) -> int:
        """原始总样本数。"""
        return self.n_train + self.n_valid + self.n_test

    @property
    def class_distribution_train(self) -> dict[str, float]:
        """训练集类别分布（仅当 stratify=True 时可用）。
        注意：此属性在 SplitResult 中只存索引，不在类内持有 y，
        因此调用方需自行从 y 计算。保留此属性占位。
        """
        raise AttributeError(
            "SplitResult 不存 y 数据。调用方请通过 "
            "y.iloc[result.train_indices].value_counts(normalize=True) 计算。"
        )


__all__ = [
    "SplitResult",
    "SplitAuditResult",
]