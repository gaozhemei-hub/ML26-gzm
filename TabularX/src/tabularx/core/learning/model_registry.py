"""模型注册表。

统一管理所有可用模型，每个模型提供：
- 工厂函数（惰性导入，处理可选依赖）
- Optuna 搜索空间函数
- 默认参数
- 元信息（任务类型、是否支持 predict_proba 等）

设计原则：
- 可选依赖在函数内部惰性导入，避免导入时崩溃
- ModelSpec 是 frozen dataclass，注册表只读
- 按 task_type 筛选可用模型
"""

from __future__ import annotations

from typing import Any, Callable

import optuna

from tabularx.exceptions.learning import ModelNotAvailableError
from tabularx.schema.learning import ModelSpec


# ═══════════════════════════════════════════════════════════════
# 搜索空间定义
# ═══════════════════════════════════════════════════════════════


def _rf_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Random Forest 搜索空间。"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30, step=3),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05, step=0.005),
    }


def _xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """XGBoost 搜索空间。"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.3, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lightgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """LightGBM 搜索空间。"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.3, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _catboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """CatBoost 搜索空间。"""
    return {
        "iterations": trial.suggest_int("iterations", 50, 500, step=50),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.3, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.5, 1.0
        ),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "random_strength": trial.suggest_float(
            "random_strength", 1e-8, 10.0, log=True
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.0, 1.0
        ),
    }


def _lr_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """线性模型（Ridge / LogisticRegression）搜索空间。"""
    return {
        "C" if _is_logistic else "alpha": trial.suggest_float(
            "C", 1e-4, 1e2, log=True
        ),
    }


# ═══════════════════════════════════════════════════════════════
# 工厂函数（惰性导入）
# ═══════════════════════════════════════════════════════════════


def _make_random_forest_regressor(**params) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params)


def _make_random_forest_classifier(**params) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params)


def _make_xgboost_regressor(**params) -> Any:
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    except ImportError as e:
        raise ModelNotAvailableError(
            "XGBoost 未安装。请通过 pip install xgboost 安装。"
        ) from e


def _make_xgboost_classifier(**params) -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    except ImportError as e:
        raise ModelNotAvailableError(
            "XGBoost 未安装。请通过 pip install xgboost 安装。"
        ) from e


def _make_lightgbm_regressor(**params) -> Any:
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(**params)
    except ImportError as e:
        raise ModelNotAvailableError(
            "LightGBM 未安装。请通过 pip install lightgbm 安装。"
        ) from e


def _make_lightgbm_classifier(**params) -> Any:
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(**params)
    except ImportError as e:
        raise ModelNotAvailableError(
            "LightGBM 未安装。请通过 pip install lightgbm 安装。"
        ) from e


def _make_catboost_regressor(**params) -> Any:
    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**params)
    except ImportError as e:
        raise ModelNotAvailableError(
            "CatBoost 未安装。请通过 pip install catboost 安装。"
        ) from e


def _make_catboost_classifier(**params) -> Any:
    try:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params)
    except ImportError as e:
        raise ModelNotAvailableError(
            "CatBoost 未安装。请通过 pip install catboost 安装。"
        ) from e


# ═══════════════════════════════════════════════════════════════
# 默认参数
# ═══════════════════════════════════════════════════════════════

_DEFAULT_PARAMS_RF = {"random_state": 42, "n_jobs": 1, "verbose": 0}
_DEFAULT_PARAMS_XGB = {
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0,
}
_DEFAULT_PARAMS_LGBM = {
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}
_DEFAULT_PARAMS_CATB = {
    "random_seed": 42,
    "thread_count": 1,
    "verbose": 0,
    "allow_writing_files": False,
}


# ═══════════════════════════════════════════════════════════════
# 完整注册列表
# ═══════════════════════════════════════════════════════════════

_MODELS: list[ModelSpec] = [
    # ── 回归 ────────────────────────────────────────────────
    ModelSpec(
        name="random_forest",
        task_type="regression",
        factory=_make_random_forest_regressor,
        search_space=_rf_search_space,
        default_params=_DEFAULT_PARAMS_RF,
        supports_feature_importance=True,
    ),
    ModelSpec(
        name="xgboost",
        task_type="regression",
        factory=_make_xgboost_regressor,
        search_space=_xgboost_search_space,
        default_params=_DEFAULT_PARAMS_XGB,
        supports_feature_importance=True,
        optional_dependency="xgboost",
    ),
    ModelSpec(
        name="lightgbm",
        task_type="regression",
        factory=_make_lightgbm_regressor,
        search_space=_lightgbm_search_space,
        default_params=_DEFAULT_PARAMS_LGBM,
        supports_feature_importance=True,
        optional_dependency="lightgbm",
    ),
    ModelSpec(
        name="catboost",
        task_type="regression",
        factory=_make_catboost_regressor,
        search_space=_catboost_search_space,
        default_params=_DEFAULT_PARAMS_CATB,
        supports_feature_importance=True,
        optional_dependency="catboost",
    ),
    # ── 分类 ────────────────────────────────────────────────
    ModelSpec(
        name="random_forest",
        task_type="classification",
        factory=_make_random_forest_classifier,
        search_space=_rf_search_space,
        default_params=_DEFAULT_PARAMS_RF,
        supports_predict_proba=True,
        supports_feature_importance=True,
    ),
    ModelSpec(
        name="xgboost",
        task_type="classification",
        factory=_make_xgboost_classifier,
        search_space=_xgboost_search_space,
        default_params=_DEFAULT_PARAMS_XGB,
        supports_predict_proba=True,
        supports_feature_importance=True,
        optional_dependency="xgboost",
    ),
    ModelSpec(
        name="lightgbm",
        task_type="classification",
        factory=_make_lightgbm_classifier,
        search_space=_lightgbm_search_space,
        default_params=_DEFAULT_PARAMS_LGBM,
        supports_predict_proba=True,
        supports_feature_importance=True,
        optional_dependency="lightgbm",
    ),
    ModelSpec(
        name="catboost",
        task_type="classification",
        factory=_make_catboost_classifier,
        search_space=_catboost_search_space,
        default_params=_DEFAULT_PARAMS_CATB,
        supports_predict_proba=True,
        supports_feature_importance=True,
        optional_dependency="catboost",
    ),
]


# ═══════════════════════════════════════════════════════════════
# ModelRegistry
# ═══════════════════════════════════════════════════════════════


class ModelRegistry:
    """模型注册表，管理所有可用模型的查询与访问。"""

    def __init__(self) -> None:
        self._models: list[ModelSpec] = list(_MODELS)

    def list_models(self, task_type: str) -> list[str]:
        """返回指定任务类型的可用模型名称列表（去重）。"""
        seen: set[str] = set()
        result: list[str] = []
        for spec in self._models:
            if spec.task_type == task_type and spec.name not in seen:
                seen.add(spec.name)
                result.append(spec.name)
        return result

    def get(self, name: str, task_type: str) -> ModelSpec:
        """按名称和任务类型获取模型规约。

        Args:
            name: 模型名，如 "random_forest"、"xgboost"。
            task_type: "regression" 或 "classification"。

        Returns:
            ModelSpec 实例。

        Raises:
            ModelNotAvailableError: 模型不可用（未注册或任务类型不匹配）。
        """
        for spec in self._models:
            if spec.name == name and spec.task_type == task_type:
                return spec
        raise ModelNotAvailableError(
            f"模型 {name!r} 在 {task_type} 任务中不可用。"
            f"  可用回归模型：{self.list_models('regression')}。"
            f"  可用分类模型：{self.list_models('classification')}。"
        )

    def get_all(self, task_type: str) -> list[ModelSpec]:
        """获取指定任务类型的所有模型规约列表。"""
        return [spec for spec in self._models if spec.task_type == task_type]

    def register(self, spec: ModelSpec) -> None:
        """注册一个自定义模型（扩展用）。"""
        self._models.append(spec)


# 全局单例
_DEFAULT_REGISTRY: ModelRegistry | None = None


def get_default_registry() -> ModelRegistry:
    """获取全局默认模型注册表（单例）。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ModelRegistry()
    return _DEFAULT_REGISTRY


__all__ = [
    "ModelRegistry",
    "get_default_registry",
]