"""评估指标工具集。

统一封装 sklearn 的指标计算，提供：
- 自动选择主指标（按 task_type 和 tags）
- 回归 / 分类指标批量计算
- 优化方向判断
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from tabularx.schema.profile import DataProfile


# ═══════════════════════════════════════════════════════════════
# 回归指标
# ═══════════════════════════════════════════════════════════════

REGRESSION_METRICS = {
    "rmse": {
        "func": lambda y_true, y_pred, **kwargs: float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "direction": "minimize",
        "label": "RMSE",
    },
    "mae": {
        "func": lambda y_true, y_pred, **kwargs: float(
            mean_absolute_error(y_true, y_pred)
        ),
        "direction": "minimize",
        "label": "MAE",
    },
    "r2": {
        "func": lambda y_true, y_pred, **kwargs: float(r2_score(y_true, y_pred)),
        "direction": "maximize",
        "label": "R²",
    },
}


# ═══════════════════════════════════════════════════════════════
# 分类指标
# ═══════════════════════════════════════════════════════════════

CLASSIFICATION_METRICS = {
    "accuracy": {
        "func": lambda y_true, y_pred, **kwargs: float(accuracy_score(y_true, y_pred)),
        "direction": "maximize",
        "label": "准确率",
    },
    "balanced_accuracy": {
        "func": lambda y_true, y_pred, **kwargs: float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "direction": "maximize",
        "label": "平衡准确率",
    },
    "f1": {
        "func": lambda y_true, y_pred, **kwargs: float(
            f1_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "direction": "maximize",
        "label": "F1",
    },
    "f1_macro": {
        "func": lambda y_true, y_pred, **kwargs: float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "direction": "maximize",
        "label": "F1 Macro",
    },
    "roc_auc": {
        "func": lambda y_true, y_pred, y_prob=None, **kwargs: (
            float(roc_auc_score(y_true, y_prob[:, 1] if y_prob is not None and y_prob.shape[1] >= 2 else y_pred))
            if y_prob is not None
            else 0.0
        ),
        "direction": "maximize",
        "label": "ROC-AUC",
    },
    "precision": {
        "func": lambda y_true, y_pred, **kwargs: float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "direction": "maximize",
        "label": "精确率",
    },
    "recall": {
        "func": lambda y_true, y_pred, **kwargs: float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "direction": "maximize",
        "label": "召回率",
    },
}


# ═══════════════════════════════════════════════════════════════
# 指标工具函数
# ═══════════════════════════════════════════════════════════════


def resolve_primary_metric(
    task_type: str,
    primary_metric: str = "auto",
    profile: DataProfile | None = None,
) -> tuple[str, str]:
    """解析主指标名称，返回 (metric_name, direction)。

    Args:
        task_type: "regression" / "binary" / "multiclass"。
        primary_metric: "auto" 或具体指标名。
        profile: 可选，用于检测 imbalanced 标签。

    Returns:
        (metric_name, "maximize" | "minimize")

    Raises:
        ValueError: 不支持的指标或不支持的任务组合。
    """
    if primary_metric != "auto":
        # 用户指定了指标，验证是否合法
        if task_type == "regression":
            if primary_metric not in REGRESSION_METRICS:
                raise ValueError(
                    f"回归任务不支持主指标 {primary_metric!r}。"
                    f"  可选：{list(REGRESSION_METRICS)}"
                )
        else:
            if primary_metric not in CLASSIFICATION_METRICS:
                raise ValueError(
                    f"分类任务不支持主指标 {primary_metric!r}。"
                    f"  可选：{list(CLASSIFICATION_METRICS)}"
                )
        direction = _get_metric_direction(primary_metric)
        return primary_metric, direction

    # auto 模式：按 task_type + profile tags 自动选择
    if task_type == "regression":
        return "rmse", "minimize"

    # 分类任务：检查是否 imbalanced
    is_imbalanced = False
    if profile is not None and "imbalanced" in profile.target.tags:
        is_imbalanced = True

    if task_type == "binary":
        if is_imbalanced:
            return "roc_auc", "maximize"
        return "roc_auc", "maximize"

    # multiclass
    if is_imbalanced:
        return "f1_macro", "maximize"
    return "f1_macro", "maximize"


def _get_metric_direction(metric_name: str) -> str:
    """返回指标的优化方向。"""
    if metric_name in REGRESSION_METRICS:
        return REGRESSION_METRICS[metric_name]["direction"]
    if metric_name in CLASSIFICATION_METRICS:
        return CLASSIFICATION_METRICS[metric_name]["direction"]
    return "maximize"


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """计算所有回归指标。

    Args:
        y_true: 真实值。
        y_pred: 预测值。

    Returns:
        指标名字典，如 {"rmse": 1.23, "mae": 0.87, "r2": 0.91}。
    """
    metrics = {}
    for name, spec in REGRESSION_METRICS.items():
        try:
            metrics[name] = spec["func"](y_true, y_pred)
        except Exception:
            metrics[name] = float("nan")
    return metrics


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """计算所有分类指标。

    Args:
        y_true: 真实标签。
        y_pred: 预测标签。
        y_prob: 预测概率（二分类时用于 ROC-AUC）。

    Returns:
        指标名字典。
    """
    metrics = {}
    for name, spec in CLASSIFICATION_METRICS.items():
        try:
            if name == "roc_auc":
                metrics[name] = spec["func"](y_true, y_pred, y_prob=y_prob)
            else:
                metrics[name] = spec["func"](y_true, y_pred)
        except Exception:
            metrics[name] = float("nan")
    return metrics


def compute_all_metrics(
    task_type: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """根据任务类型计算全部指标。

    Args:
        task_type: "regression" / "binary" / "multiclass"。
        y_true: 真实值。
        y_pred: 预测值。
        y_prob: 可选，分类时的预测概率。

    Returns:
        指标名字典。
    """
    if task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    return compute_classification_metrics(y_true, y_pred, y_prob=y_prob)


__all__ = [
    "REGRESSION_METRICS",
    "CLASSIFICATION_METRICS",
    "resolve_primary_metric",
    "compute_regression_metrics",
    "compute_classification_metrics",
    "compute_all_metrics",
]