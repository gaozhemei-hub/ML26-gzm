"""数据概况模块：观测数据事实，不做任何决策。

角色等价于 MLJAR 的 DataInfo.compute，但输出强类型 dataclass。
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from tabularx.schema.profile import (
    ColumnFact,
    DataProfile,
    TargetFact,
)


class ProfileUtils:
    """列类型推断、统计量、scale 判据等工具方法集。"""

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"

    # ── 列类型推断 ──────────────────────────────────────────

    @staticmethod
    def infer_column_type(col: pd.Series) -> str:
        """判断列的类型（MLJAR 式逐列分析）。

        规则（参考 MLJAR PreprocessingUtils.get_type）：
        1. 全空列 → "categorical"（保守标记，build_column_fact 会加 empty_column 标记）
        2. datetime → "datetime"
        3. bool     → "boolean"
        4. object / str / category → 看唯一值数量：
           - >200 且 >50% 行唯一 → "text"
           - 否则 → "categorical"
        5. int / uint → "categorical"（如果唯一值 ≤ 20） 否则 → "numeric"
        6. float → "numeric"
        """
        col_clean = col.dropna()

        # 全空列：无法判断类型，保守标记为 categorical
        if col_clean.empty:
            return ProfileUtils.CATEGORICAL

        if pd.api.types.is_datetime64_any_dtype(col):
            return ProfileUtils.DATETIME

        if pd.api.types.is_bool_dtype(col):
            return ProfileUtils.BOOLEAN

        dtype = col.dtype

        # 类别型 / 文本型（pandas 3.0 字符串 dtype 为 StringDtype/str，不再是 object）
        is_string_like = (
            pd.api.types.is_object_dtype(col)
            or pd.api.types.is_string_dtype(col)
            or isinstance(dtype, pd.CategoricalDtype)
        )
        if is_string_like:
            unique_cnt = col_clean.nunique()
            if unique_cnt > 200 and unique_cnt > int(0.5 * len(col_clean)):
                return ProfileUtils.TEXT
            return ProfileUtils.CATEGORICAL

        # 整数型
        if pd.api.types.is_integer_dtype(col):
            unique_cnt = col_clean.nunique()
            if unique_cnt <= 20:
                return ProfileUtils.CATEGORICAL
            return ProfileUtils.NUMERIC

        # 浮点型
        return ProfileUtils.NUMERIC

    # ── scale 判据 ──────────────────────────────────────────

    @staticmethod
    def is_scale_needed(col: pd.Series) -> bool:
        """判断数值列是否需要标准化（MLJAR 判据：|mean|>0.5 or std>1.5）。"""
        col_clean = col.dropna()
        if col_clean.empty:
            return False
        abs_mean = abs(np.mean(col_clean))
        std = np.std(col_clean)
        return bool(abs_mean > 0.5 or std > 1.5)

    # ── 任务类型推断 ─────────────────────────────────────────

    @staticmethod
    def infer_task_type(y: pd.Series) -> Literal["binary", "multiclass", "regression"]:
        """推断任务类型，显式支持 'binary'/'multiclass'/'regression'。

        用 sklearn 的 type_of_target 做基础判断，对边界情况做兜底。
        """
        y_clean = y.dropna()

        if y_clean.empty:
            raise ValueError("y 为空，无法推断任务类型")

        # ── 第 1 层：type_of_target 直接能判定的 ──────────────
        raw = type_of_target(y_clean)

        # 明确是回归的（continuous），直接返回
        if raw in ("continuous", "continuous-multioutput"):
            return "regression"

        # 明确是二分类的，直接返回
        if raw == "binary":
            return "binary"

        # ── 第 2 层：type_of_target 判为 multiclass 但实际可能是回归 ──
        # type_of_target 对浮点连续值会误判为 multiclass（sklearn 已知局限）
        if raw == "multiclass" and pd.api.types.is_float_dtype(y_clean):
            unique_cnt = y_clean.nunique()
            # 唯一值多（>20）→ 回归；唯一值少（≤20）→ 尊重 type_of_target 的 multiclass 判断
            if unique_cnt > 20:
                return "regression"
            return "multiclass"

        # ── 第 3 层：type_of_target 判为 multiclass 且是整数 ──
        if raw == "multiclass":
            # 整数型少量唯一值 → 分类
            if pd.api.types.is_integer_dtype(y_clean):
                unique_cnt = y_clean.nunique()
                if unique_cnt <= 20:
                    return "binary" if unique_cnt == 2 else "multiclass"
            # 其他情况保持 multiclass
            return "multiclass"

        # ── 第 4 层：兜底 ────────────────────────────────────
        raise ValueError(
            f"无法推断任务类型，type_of_target 返回: {raw}，"
            f"dtype={y_clean.dtype}，unique_cnt={y_clean.nunique()}"
        )

    # ── 统计量 ──────────────────────────────────────────────

    @staticmethod
    def compute_basic_stats(col: pd.Series) -> dict:
        """计算数值列的基本统计量。"""
        col_clean = col.dropna()
        if col_clean.empty:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(col_clean)),
            "std": float(np.std(col_clean)),
            "min": float(np.min(col_clean)),
            "max": float(np.max(col_clean)),
        }

    # ── 数据指纹 ─────────────────────────────────────────────

    @staticmethod
    def compute_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
        """对 X+y 计算 sha256 指纹，用于复现校验。"""
        hasher = hashlib.sha256()
        # 用 pandas 的哈希方法得到统一哈希
        x_hash = pd.util.hash_pandas_object(X).values
        y_hash = pd.util.hash_pandas_object(y).values
        hasher.update(x_hash.tobytes())
        hasher.update(y_hash.tobytes())
        return hasher.hexdigest()[:16]  # 取前 16 位，足够区分


# ── 顶层函数 ─────────────────────────────────────────────────


def build_column_fact(col: pd.Series) -> ColumnFact:
    """构建单列的事实记录。"""
    col_type = ProfileUtils.infer_column_type(col)
    n_missing = int(col.isna().sum())
    missing_rate = n_missing / len(col) if len(col) > 0 else 0.0
    col_clean = col.dropna()
    n_unique = int(col_clean.nunique())

    # 确定标签 tags
    tags = [col_type]
    if col_type == ProfileUtils.NUMERIC:
        if ProfileUtils.is_scale_needed(col):
            tags.append("scale_needed")
    if n_missing > 0:
        tags.append("has_missing")

    # 数值列额外统计
    stats = ProfileUtils.compute_basic_stats(col) if col_type == ProfileUtils.NUMERIC else {}

    return ColumnFact(
        col_name=col.name if col.name is not None else "",
        dtype=str(col.dtype),
        n_missing=n_missing,
        missing_rate=missing_rate,
        n_unique=n_unique,
        tags=tags,
        **stats,
    )


def build_target_fact(y: pd.Series) -> TargetFact:
    """构建目标变量的事实记录。"""
    task_type = ProfileUtils.infer_task_type(y)
    y_clean = y.dropna()
    missing_rate = (len(y) - len(y_clean)) / len(y) if len(y) > 0 else 0.0

    if task_type in ("binary", "multiclass"):
        value_counts = y_clean.value_counts(normalize=True)
        class_balance = {str(k): float(v) for k, v in value_counts.items()}
        return TargetFact(
            task_type=task_type,
            n_classes=int(y_clean.nunique()),
            class_balance=class_balance,
            mean=None,
            std=None,
            missing_rate=missing_rate,
        )
    else:
        # 回归
        return TargetFact(
            task_type="regression",
            n_classes=None,
            class_balance=None,
            mean=float(np.mean(y_clean)) if not y_clean.empty else None,
            std=float(np.std(y_clean)) if not y_clean.empty else None,
            missing_rate=missing_rate,
        )


def build_profile(X: pd.DataFrame, y: pd.Series) -> DataProfile:
    """纯观测，零决策：构建数据集的完整事实记录。

    Args:
        X: 特征 DataFrame，必须有列名。
        y: 目标 Series。

    Returns:
        DataProfile：包含所有列和目标的观测事实。
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X 必须是 pd.DataFrame，收到 {type(X)}")
    if not isinstance(y, pd.Series):
        raise TypeError(f"y 必须是 pd.Series，收到 {type(y)}")

    columns = [build_column_fact(X[col]) for col in X.columns]
    target = build_target_fact(y)
    fingerprint = ProfileUtils.compute_fingerprint(X, y)

    return DataProfile(
        n_rows=len(X) if len(X.columns) > 0 else len(y),
        n_cols=len(X.columns),
        columns=columns,
        target=target,
        fingerprint=fingerprint,
    )