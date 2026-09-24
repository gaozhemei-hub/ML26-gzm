"""Profile 阶段的数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Evidence:
    """判断依据的运行时数据快照（规则文本在模板中，此处只存数据）。"""

    metric: str  # 证据名，如 "n_unique"
    value: float | int | str  # 证据值，如 2
    note: str | None = None  # 补充说明，如 "目标列去空后的唯一值个数"


@dataclass
class ColumnFact:
    """单列的事实记录（只描述，不决策）。"""

    col_name: str
    dtype: str  # pandas dtype 字符串
    n_missing: int
    missing_rate: float
    n_unique: int
    tags: list[str] = field(default_factory=list)
    # 数值列统计（非数值列留空）
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    skewness: float | None = None  # 偏度，|skew|>1 表示偏斜明显
    # 判断依据数据快照
    evidences: list[Evidence] = field(default_factory=list)


@dataclass
class TargetFact:
    """目标变量的事实记录。"""

    task_type: Literal["binary", "multiclass", "regression"]
    n_classes: int | None = None  # 分类时
    class_balance: dict[str, float] | None = None  # 分类时：{"0": 0.6, "1": 0.4}
    mean: float | None = None  # 回归时
    std: float | None = None
    missing_rate: float = 0.0
    tags: list[str] = field(default_factory=list)  # 如 ["imbalanced"]
    # 判断依据数据快照
    evidences: list[Evidence] = field(default_factory=list)


@dataclass
class DataProfile:
    """数据集的完整事实记录。

    纯观测，不做任何决策。后续步骤（split/preprocess/search）根据这些事实做决策。
    """

    n_rows: int
    n_cols: int
    columns: list[ColumnFact]
    target: TargetFact
    fingerprint: str  # 数据指纹（sha256），用于复现校验