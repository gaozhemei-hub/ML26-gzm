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
        group_labels = X_test[mapping.output_names].idxmax(axis=1)
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
        y_prob: 预测概率（用于计算 positive_rate）。
        task_type: 任务类型。

    Returns:
        FairnessGroupResult 对象。
    """
    n = len(y_true)
    support = n

    # 正类预测率
    if task_type != "regression" and y_prob is not None:
        # 二分类：取正类（索引 1）的概率均值
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            positive_rate = float(np.mean(y_prob[:, 1]))
        else:
            positive_rate = float(np.mean(y_pred))
    else:
        positive_rate = float(np.mean(y_pred))

    # 准确率（仅分类任务有效）
    acc = 0.0
    if task_type != "regression":
        try:
            acc = float(accuracy_score(y_true, y_pred))
        except Exception:
            acc = 0.0

    # TPR / FPR（仅分类任务有意义）
    tpr = 0.0
    fpr = 0.0
    if task_type != "regression":
        # 将 y_true 和 y_pred 视为二值
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

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
        y_prob: 模型预测概率（可选，用于更精确的 positive_rate）。
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
            mask = group_series == g
            grp = _compute_group_metrics(
                group_name=str(g),
                y_true=y_test[mask],
                y_pred=y_pred[mask],
                y_prob=y_prob[mask] if y_prob is not None else None,
                task_type=task_type,
            )
            group_results.append(grp)

        # 计算公平性差异
        if len(group_results) >= 2:
            pos_rates = [g.positive_rate for g in group_results]
            tprs = [g.tpr for g in group_results]
            fprs = [g.fpr for g in group_results]
            dp_diff = max(pos_rates) - min(pos_rates)
            eo_tpr_diff = max(tprs) - min(tprs)
            eo_fpr_diff = max(fprs) - min(fprs)
        else:
            dp_diff = 0.0
            eo_tpr_diff = 0.0
            eo_fpr_diff = 0.0

        # 根据配置的阈值判断是否违反公平性标准
        dp_violation = dp_diff > _cfg.dp_threshold
        eo_tpr_violation = eo_tpr_diff > _cfg.eo_tpr_threshold
        eo_fpr_violation = eo_fpr_diff > _cfg.eo_fpr_threshold

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