"""数据概况模块：观测数据事实，不做任何决策。

角色等价于 MLJAR 的 DataInfo.compute，但输出强类型 dataclass。
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from tabularx.exceptions import (
    EmptyDataError,
    EmptyTargetError,
    InvalidDataError,
    MismatchedLengthError,
    TaskTypeError,
)
from tabularx.schema.profile import (
    ColumnFact,
    DataProfile,
    Evidence,
    TargetFact,
)

# 合法的任务类型取值
TASK_TYPES = ("binary", "multiclass", "regression")


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
            raise EmptyTargetError("y 为空，无法推断任务类型")

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
        raise TaskTypeError(
            f"无法推断任务类型，type_of_target 返回: {raw}，"
            f"dtype={y_clean.dtype}，unique_cnt={y_clean.nunique()}。"
            "建议：使用 build_profile(X, y, task_type=...) 手动指定。"
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
    col_clean = col.dropna()
    n_missing = int(col.isna().sum())
    missing_rate = n_missing / len(col) if len(col) > 0 else 0.0
    n_unique = int(col_clean.nunique())
    evidences: list[Evidence] = []

    # 全空列：保守标记 categorical，并记录证据
    if col_clean.empty:
        evidences.append(
            Evidence("n_non_missing", 0, "非缺失值数量为 0，判定为全空列")
        )
        return ColumnFact(
            col_name=col.name if col.name is not None else "",
            dtype=str(col.dtype),
            n_missing=n_missing,
            missing_rate=missing_rate,
            n_unique=0,
            tags=["categorical", "empty_column"],
            evidences=evidences,
        )

    # 常数列：所有值相同，对下游无信息量，preprocess 阶段应丢弃
    if n_unique == 1:
        evidences.append(Evidence("n_unique", 1, "仅 1 个唯一值，判定为常数列"))
        return ColumnFact(
            col_name=col.name if col.name is not None else "",
            dtype=str(col.dtype),
            n_missing=n_missing,
            missing_rate=missing_rate,
            n_unique=n_unique,
            tags=["constant_column"],
            evidences=evidences,
        )

    # 正常列：推断列类型
    col_type = ProfileUtils.infer_column_type(col)
    evidences.append(
        Evidence("col_type", col_type, f"列类型判定为 {col_type}")
    )

    # 确定标签 tags
    tags = [col_type]
    if col_type == ProfileUtils.NUMERIC:
        scale_needed = ProfileUtils.is_scale_needed(col)
        if scale_needed:
            tags.append("scale_needed")
            evidences.append(
                Evidence(
                    "scale_rule",
                    "|mean|>0.5 or std>1.5",
                    f"|mean|={abs(float(np.mean(col_clean))):.4f}, "
                    f"std={float(np.std(col_clean)):.4f}，满足判据 → scale_needed",
                )
            )
        else:
            evidences.append(
                Evidence(
                    "scale_rule",
                    "|mean|>0.5 or std>1.5",
                    f"|mean|={abs(float(np.mean(col_clean))):.4f}, "
                    f"std={float(np.std(col_clean)):.4f}，不满足判据 → 无需 scale",
                )
            )
    if n_missing > 0:
        tags.append("has_missing")
        evidences.append(
            Evidence("n_missing", n_missing, f"存在 {n_missing} 个缺失值（{missing_rate:.2%}）")
        )

    # 数值列额外统计
    stats = ProfileUtils.compute_basic_stats(col) if col_type == ProfileUtils.NUMERIC else {}

    return ColumnFact(
        col_name=col.name if col.name is not None else "",
        dtype=str(col.dtype),
        n_missing=n_missing,
        missing_rate=missing_rate,
        n_unique=n_unique,
        tags=tags,
        evidences=evidences,
        **stats,
    )


def build_target_fact(y: pd.Series, task_type: str = "auto") -> TargetFact:
    """构建目标变量的事实记录。

    Args:
        y: 目标 Series。
        task_type: "auto"（自动推断）| "classification" | "regression"。
            手动指定时跳过自动推断（classification 会继续细分 binary/multiclass）。
    """
    if task_type not in ("auto", "classification", "regression"):
        raise TaskTypeError(
            f"task_type 必须是 'auto'/'classification'/'regression'，收到 {task_type!r}。"
            "建议：使用 'auto'（自动推断）或 'classification'/'regression'（手动指定）。"
        )

    y_clean = y.dropna()
    missing_rate = (len(y) - len(y_clean)) / len(y) if len(y) > 0 else 0.0
    n_unique = int(y_clean.nunique())
    evidences: list[Evidence] = []

    # 确定任务类型
    if task_type == "regression":
        inferred = "regression"
        evidences.append(Evidence("task_type_source", "user", "用户手动指定为回归任务"))
    elif task_type == "classification":
        # 已知分类，细分 binary / multiclass
        inferred = "binary" if n_unique == 2 else "multiclass"
        evidences.append(
            Evidence(
                "task_type_source",
                "user+auto",
                f"用户手动指定分类任务；按唯一值数 {n_unique} 细分为 "
                f"{inferred}",
            )
        )
    else:  # auto
        inferred = ProfileUtils.infer_task_type(y_clean)
        evidences.append(
            Evidence(
                "task_type_source",
                "auto",
                f"自动推断：唯一值 {n_unique} 个，dtype={y_clean.dtype}，"
                f"type_of_target 判定为 {inferred}",
            )
        )

    if inferred in ("binary", "multiclass"):
        value_counts = y_clean.value_counts(normalize=True)
        class_balance = {str(k): float(v) for k, v in value_counts.items()}
        return TargetFact(
            task_type=inferred,  # type: ignore[arg-type]
            n_classes=n_unique,
            class_balance=class_balance,
            mean=None,
            std=None,
            missing_rate=missing_rate,
            evidences=evidences,
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
            evidences=evidences,
        )


def build_profile(
    X: pd.DataFrame, y: pd.Series, task_type: str = "auto"
) -> DataProfile:
    """纯观测，零决策：构建数据集的完整事实记录。

    Args:
        X: 特征 DataFrame，必须有列名。
        y: 目标 Series。
        task_type: "auto"（默认，自动推断）| "classification" | "regression"。
            手动指定时跳过自动推断。

    Returns:
        DataProfile：包含所有列和目标的观测事实。
    """
    if not isinstance(X, pd.DataFrame):
        raise InvalidDataError(f"X 必须是 pd.DataFrame，收到 {type(X)}。建议：使用 pd.DataFrame(X, columns=[...]) 转换。")
    if not isinstance(y, pd.Series):
        raise InvalidDataError(f"y 必须是 pd.Series，收到 {type(y)}。建议：使用 pd.Series(y) 转换。")

    # 数据非空检查
    if len(X) == 0 or len(X.columns) == 0:
        raise EmptyDataError(
            f"X 为空（shape={X.shape}）。请提供至少 1 行 1 列的特征数据。"
        )
    if len(y) == 0:
        raise EmptyDataError("y 为空。请提供至少 1 个样本的目标值。")

    # X 与 y 长度对齐检查
    if len(X) != len(y):
        raise MismatchedLengthError(
            f"X 行数（{len(X)}）与 y 长度（{len(y)}）不一致。"
            "请确保 X 与 y 对应同一批样本。"
        )

    columns = [build_column_fact(X[col]) for col in X.columns]
    target = build_target_fact(y, task_type=task_type)
    fingerprint = ProfileUtils.compute_fingerprint(X, y)

    return DataProfile(
        n_rows=len(X),
        n_cols=len(X.columns),
        columns=columns,
        target=target,
        fingerprint=fingerprint,
    )