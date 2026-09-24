"""Learning 子模块 - 模型搜索、调参与评估。

包含：
- model_registry: 模型注册表（所有分类/回归模型的定义）
- metrics: 评估指标计算
- optuna_tuner: Optuna 超参数调优
- model_search: 多模型并行搜索编排
- ensemble_search: 模型组合（Voting）
- plots: 模型评估图（混淆矩阵/ROC/调参轨迹）
"""

from tabularx.core.learning.model_registry import ModelRegistry, get_default_registry
from tabularx.core.learning.model_search import search_models
from tabularx.core.learning.metrics import (
    resolve_primary_metric,
    compute_all_metrics,
    compute_regression_metrics,
    compute_classification_metrics,
)
from tabularx.core.learning.optuna_tuner import tune_model
from tabularx.core.learning.ensemble_search import build_voting
from tabularx.core.learning.shap_explainer import explain_model
from tabularx.core.learning.fairness import evaluate_fairness
from tabularx.core.learning.plots import (
    save_model_plots,
    save_voting_plots,
    save_trial_trajectory_plot,
)


__all__ = [
    "ModelRegistry",
    "get_default_registry",
    "search_models",
    "tune_model",
    "build_voting",
    "explain_model",
    "evaluate_fairness",
    "resolve_primary_metric",
    "compute_all_metrics",
    "compute_regression_metrics",
    "compute_classification_metrics",
    "save_model_plots",
    "save_voting_plots",
    "save_trial_trajectory_plot",
]