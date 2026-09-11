"""Learning 阶段的数据类型定义。

包含模型规约、搜索结果、公平性配置等数据类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 参数定义
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParamDef:
    """声明式搜索空间参数定义。

    替代之前的 Callable 搜索空间函数，每个参数用数据描述，
    使 Optuna 调参器可以统一遍历并生成 trial.suggest_*() 调用。
    """

    name: str
    """参数名，如 "n_estimators"、"max_depth" """
    type: Literal["int", "float", "categorical"]
    """参数类型"""
    low: float | int | None = None
    """int/float 时的搜索下限"""
    high: float | int | None = None
    """int/float 时的搜索上限"""
    step: float | int | None = None
    """int 的步长（可选）"""
    log: bool = False
    """float 是否使用对数尺度采样"""
    choices: list[Any] | None = None
    """categorical 的候选值列表"""
    description: str | None = None
    """参数说明（用于报告和文档）"""


# ═══════════════════════════════════════════════════════════════
# 模型规约
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelSpec:
    """描述一个可注册的模型。

    不包含模型实例，只包含如何创建、调参、推断的相关信息。
    """

    name: str
    """模型名，如 "random_forest"、"xgboost" """
    task_type: Literal["regression", "classification"]
    """适用的任务类型"""
    factory: Callable[..., Any]
    """模型工厂函数，接收 **params 返回模型实例"""
    search_space_defs: list[ParamDef]
    """声明式搜索空间定义列表（替代 search_space 函数）"""
    default_params: dict[str, Any] | None = None
    """默认参数（始终注入到模型中），如 {"random_state": 42, "n_jobs": 1}"""
    supports_predict_proba: bool = False
    """是否支持 predict_proba（分类模型使用）"""
    supports_feature_importance: bool = False
    """是否有原生的 feature_importances_ 属性"""
    optional_dependency: str | None = None
    """可选依赖名，如 "xgboost"，用于在不可用时给出清晰的提示信息"""


# ═══════════════════════════════════════════════════════════════
# 调参相关的数据类型
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrialRecord:
    """记录一次 Optuna trial 的结果。"""

    trial_number: int
    params: dict[str, Any]
    value: float
    state: str
    duration_seconds: float | None = None


@dataclass
class OptunaResult:
    """单个模型 Optuna 调参的结果。"""

    model_name: str
    task_type: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    trials: list[TrialRecord] = field(default_factory=list)
    study_name: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


# ═══════════════════════════════════════════════════════════════
# 模型搜索结果
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelRunResult:
    """单个模型从调参到评估的完整结果。"""

    model_name: str
    task_type: str
    status: Literal["success", "failed"]

    # ── 调参结果 ────────────────────────────────────────────
    best_params: dict[str, Any]
    n_trials: int
    optuna_duration_seconds: float | None = None
    optuna_trials: list[TrialRecord] | None = None
    """Optuna 每次 trial 的完整记录（用于画调参轨迹图）"""

    # ── 验证集指标 ──────────────────────────────────────────
    validation_metrics: dict[str, float] = field(default_factory=dict)

    # ── 最终（测试集）指标 ──────────────────────────────────
    test_metrics: dict[str, float] | None = None

    # ── 保存路径 ────────────────────────────────────────────
    model_path: str | None = None
    """模型文件保存路径（.joblib）"""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    """工件路径，如 {"confusion_matrix": "artifacts/confusion_matrix.png"} """

    # ── 原因为记录 ──────────────────────────────────────────
    error: str | None = None
    note: str | None = None


@dataclass
class ModelSearchResult:
    """所有模型搜索的完整结果。"""

    task_type: Literal["binary", "multiclass", "regression"]
    primary_metric: str
    """模型选择的主指标名，如 "rmse"、"roc_auc" """
    optimization_direction: Literal["maximize", "minimize"]
    """主指标的优化方向"""

    # ── 各模型结果 ──────────────────────────────────────────
    model_results: list[ModelRunResult] = field(default_factory=list)

    # ── 最佳单模型 ──────────────────────────────────────────
    best_model_name: str | None = None
    best_model_params: dict[str, Any] | None = None
    best_validation_metrics: dict[str, float] = field(default_factory=dict)
    best_test_metrics: dict[str, float] | None = None

    # ── Voting / Stacking 组合模型 ──────────────────────────
    ensemble_result: ModelRunResult | None = None

    # ── 元信息 ──────────────────────────────────────────────
    n_models_total: int = 0
    n_models_succeeded: int = 0
    n_models_failed: int = 0

    # ── 报告保存路径 ────────────────────────────────────────
    report_dir: str | None = None

    # ── 辅助属性 ────────────────────────────────────────────

    @property
    def succeeded_results(self) -> list[ModelRunResult]:
        """返回成功的模型结果列表。"""
        return [r for r in self.model_results if r.status == "success"]

    @property
    def failed_results(self) -> list[ModelRunResult]:
        """返回失败的模型结果列表。"""
        return [r for r in self.model_results if r.status == "failed"]

    @property
    def ranking(self) -> list[ModelRunResult]:
        """按验证集主指标排序的模型列表（最优在前）。"""
        if not self.succeeded_results:
            return []
        reverse = self.optimization_direction == "maximize"
        return sorted(
            self.succeeded_results,
            key=lambda r: r.validation_metrics.get(self.primary_metric, 0.0),
            reverse=reverse,
        )


# ═══════════════════════════════════════════════════════════════
# 公平性配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class FairnessConfig:
    """公平性感知评估的配置。"""

    enabled: bool = False
    """是否启用公平性评估"""
    sensitive_features: list[str] = field(default_factory=list)
    """敏感特征列名列表（必须存在于原始 X 中）"""
    metrics: list[str] = field(
        default_factory=lambda: ["demographic_parity", "equalized_odds"]
    )
    """需要计算的公平性指标列表"""


__all__ = [
    "ParamDef",
    "ModelSpec",
    "TrialRecord",
    "OptunaResult",
    "ModelRunResult",
    "ModelSearchResult",
    "FairnessConfig",
]