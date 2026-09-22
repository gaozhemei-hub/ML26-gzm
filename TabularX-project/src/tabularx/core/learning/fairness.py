"""公平性评估模块。

对训练好的模型执行公平性分析，计算：
- Demographic Parity：各子组正类预测率的最大差值
- Equalized Odds：各子组 TPR（真正率）和 FPR（假正率）的最大差值

设计原则：
- 零外部依赖（仅用 sklearn.metrics + numpy + pandas，均为项目已有依赖）
- 对每个敏感特征独立评估
- 通过 column_mappings 处理预处理后的列变换（丢弃、原样保留、OneHot 展开）
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from tabularx.schema.preprocess import ColumnMapping
from tabularx.schema.learning import (
    FairnessGroupResult,
    FairnessFeatureResult,
    FairnessResult,
)
from tabularx.core.params import FairnessConfig


# ═══════════════════════════════════════════════════════════════
# 内部：解析敏感特征在预处理后的实际列名
# ═══════════════════════════════════════════════════════════════


def _resolve_sensitive_column(
    original_name: str,
    column_mappings: list[ColumnMapping],
    X_test: pd.DataFrame,
) -> tuple[pd.Series | None, bool, str | None]:
    """解析敏感特征在预处理后的实际数据列。

    Args:
        original_name: 原始列名。
        column_mappings: 列映射列表。
        X_test: 预处理后的测试集。

    Returns:
        (group_series, is_dropped, dropped_reason)
        group_series: 每个样本所属的子组标签 Series。如果被丢弃，则为 None。
        is_dropped: 是否被丢弃。
        dropped_reason: 丢弃原因说明。
    """
    # 查找 column_mappings
    mapping = None
    for cm in column_mappings:
        if cm.original_name == original_name:
            mapping = cm
            break

    if mapping is None:
        # column_mappings 中没有记录，尝试直接从 X_test 中取
        if original_name in X_test.columns:
            return X_test[original_name], False, None
        return None, True, f"原始列 {original_name!r} 在 column_mappings 中未找到"

    if mapping.is_dropped or not mapping.output_names:
        return None, True, f"敏感特征 {original_name!r} 在预处理中被丢弃（常数列/全空列）"

    if len(mapping.output_names) == 1:
        out_name = mapping.output_names[0]
        if out_name in X_test.columns:
            return X_test[out_name], False, None
        return None, True, f"预处理后的列 {out_name!r} 在测试集中不存在"

    # OneHot 展开：把多列还原为原始类别标签
    if all(c in X_test.columns for c in mapping.output_names):
        # 每行取值为 1 的列名作为类别标签
        encoded = X_test[mapping.output_names]
        if not (encoded.isin([0, 1]).all(axis=1) & encoded.sum(axis=1).eq(1)).all():
            return None, True, "独热编码存在未知或不明确的类别，无法可靠还原敏感分组"
        group_labels = encoded.idxmax(axis=1)
        return group_labels, False, None

    return None, True, f"OneHot 展开列 {mapping.output_names} 在测试集中不完整"


# ═══════════════════════════════════════════════════════════════
# 内部：计算单个子组的指标
# ═══════════════════════════════════════════════════════════════


def _compute_group_metrics(
    group_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    task_type: str,
) -> FairnessGroupResult:
    """计算单个子组的公平性指标。

    Args:
        group_name: 子组名称。
        y_true: 真实标签。
        y_pred: 预测标签。
        y_prob: 预测概率（保留兼容参数；正类预测率使用离散预测标签）。
        task_type: 任务类型。

    Returns:
        FairnessGroupResult 对象。
    """
    if task_type != "binary":
        raise ValueError("DP/EO currently requires a binary task with 0/1 labels")
    n = len(y_true)
    support = n
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    positive_rate = float(np.mean(yp == 1))
    acc = float(accuracy_score(yt, yp))
    positives, negatives = np.sum(yt == 1), np.sum(yt == 0)
    tpr = float(np.sum((yt == 1) & (yp == 1)) / positives) if positives else None
    fpr = float(np.sum((yt == 0) & (yp == 1)) / negatives) if negatives else None

    return FairnessGroupResult(
        group_name=str(group_name),
        n_samples=n,
        positive_rate=positive_rate,
        accuracy=acc,
        tpr=tpr,
        fpr=fpr,
        support=support,
    )


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════


def evaluate_fairness(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    column_mappings: list[ColumnMapping],
    sensitive_features: list[str],
    task_type: str,
    config: FairnessConfig | None = None,
) -> FairnessResult:
    """执行公平性评估。

    对每个指定的敏感特征：
      1. 通过 column_mappings 解析预处理后的实际列
      2. 按子组拆分测试集
      3. 计算各子组的 positive_rate / accuracy / TPR / FPR
      4. 计算 Demographic Parity 和 Equalized Odds 差异值
      5. 根据配置的阈值标记是否违反公平性标准

    Args:
        X_test: 预处理后的测试集特征。
        y_test: 测试集真实标签。
        y_pred: 模型预测标签。
        y_prob: 模型预测概率（保留兼容参数；不用于正类预测率）。
        column_mappings: 列映射列表（来自 PreprocessedData）。
        sensitive_features: 用户指定的敏感特征原始列名列表。
        task_type: 任务类型。
        config: FairnessConfig 对象（可选，提供后标记违反字段）。

    Returns:
        FairnessResult 对象。
    """
    if not sensitive_features:
        return FairnessResult(
            enabled=False,
            sensitive_features=[],
        )

    if task_type != "binary" or not set(np.asarray(y_test)) <= {0, 1} or not set(np.asarray(y_pred)) <= {0, 1}:
        return FairnessResult(enabled=True, sensitive_features=sensitive_features,
            error="DP/EO 仅适用于当前实现支持的 0/1 二分类任务，正类为 1；本次不适用。")
    if len(X_test) != len(y_test) or len(y_test) != len(y_pred):
        raise ValueError("Fairness inputs must have equal row counts")
    if isinstance(y_test, pd.Series) and not X_test.index.equals(y_test.index):
        raise ValueError("Fairness features and labels must have aligned indices")
    y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
    _cfg = config or FairnessConfig()

    feature_results: list[FairnessFeatureResult] = []

    for feat_name in sensitive_features:
        # 解析敏感特征
        group_series, is_dropped, reason = _resolve_sensitive_column(
            feat_name, column_mappings, X_test
        )

        if is_dropped or group_series is None:
            feature_results.append(
                FairnessFeatureResult(
                    feature_name=feat_name,
                    is_dropped=True,
                    dropped_reason=reason or "未知原因",
                )
            )
            continue

        # 按子组拆分
        unique_groups = group_series.unique()
        group_results: list[FairnessGroupResult] = []

        for g in unique_groups:
            mask = (group_series.isna() if pd.isna(g) else group_series == g).to_numpy()
            grp = _compute_group_metrics(
                group_name=str(g),
                y_true=y_test[mask],
                y_pred=y_pred[mask],
                y_prob=y_prob[mask] if y_prob is not None else None,
                task_type=task_type,
            )
            group_results.append(grp)

        def gap(attribute):
            rates = [getattr(g, attribute) for g in group_results]
            return max(rates) - min(rates) if len(rates) >= 2 and all(v is not None for v in rates) else None
        dp_diff, eo_tpr_diff, eo_fpr_diff = gap("positive_rate"), gap("tpr"), gap("fpr")
        dp_violation = dp_diff > _cfg.dp_threshold if dp_diff is not None else None
        eo_tpr_violation = eo_tpr_diff > _cfg.eo_tpr_threshold if eo_tpr_diff is not None else None
        eo_fpr_violation = eo_fpr_diff > _cfg.eo_fpr_threshold if eo_fpr_diff is not None else None

        feature_results.append(
            FairnessFeatureResult(
                feature_name=feat_name,
                groups=group_results,
                demographic_parity_diff=dp_diff,
                equalized_odds_tpr_diff=eo_tpr_diff,
                equalized_odds_fpr_diff=eo_fpr_diff,
                dp_violation=dp_violation,
                eo_tpr_violation=eo_tpr_violation,
                eo_fpr_violation=eo_fpr_violation,
            )
        )

    return FairnessResult(
        enabled=True,
        sensitive_features=sensitive_features,
        feature_results=feature_results,
    )


__all__ = [
    "evaluate_fairness",
]