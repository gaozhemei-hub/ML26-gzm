"""模型注册表。

统一管理所有可用模型，每个模型提供：
- 工厂函数（惰性导入，处理可选依赖）
- 声明式 ParamDef 搜索空间（可被配置层覆盖）
- 默认参数
- 元信息（任务类型、是否支持 predict_proba 等）

设计原则：
- 可选依赖在函数内部惰性导入，避免导入时崩溃
- ModelSpec 是 frozen dataclass，注册表只读
- 按 task_type 筛选可用模型
"""

from __future__ import annotations

from typing import Any

from tabularx.exceptions.learning import ModelNotAvailableError
from tabularx.schema.learning import ModelSpec, ParamDef


# ═══════════════════════════════════════════════════════════════
# 搜索空间定义（声明式 ParamDef 列表）
# ═══════════════════════════════════════════════════════════════


_RF_SEARCH_SPACE: list[ParamDef] = [
    ParamDef("n_estimators", "int", low=50, high=500, step=50,
             description="树的数量"),
    ParamDef("max_depth", "int", low=3, high=30, step=3,
             description="树的最大深度"),
    ParamDef("min_samples_split", "int", low=2, high=20,
             description="内部节点再划分所需最小样本数"),
    ParamDef("min_samples_leaf", "int", low=1, high=10,
             description="叶子节点最少样本数"),
    ParamDef("max_features", "categorical", choices=["sqrt", "log2", None],
             description="每棵树随机采样的特征数策略"),
    ParamDef("ccp_alpha", "float", low=0.0, high=0.05, step=0.005,
             description="剪枝参数"),
]


_XGBOOST_SEARCH_SPACE: list[ParamDef] = [
    ParamDef("n_estimators", "int", low=50, high=500, step=50,
             description="弱学习器数量"),
    ParamDef("max_depth", "int", low=3, high=15,
             description="树的最大深度"),
    ParamDef("learning_rate", "float", low=1e-3, high=0.3, log=True,
             description="学习率"),
    ParamDef("subsample", "float", low=0.5, high=1.0,
             description="行采样比例"),
    ParamDef("colsample_bytree", "float", low=0.5, high=1.0,
             description="每棵树的列采样比例"),
    ParamDef("min_child_weight", "int", low=1, high=10,
             description="叶子节点最小权重和"),
    ParamDef("gamma", "float", low=0.0, high=5.0,
             description="分裂最小损失减少"),
    ParamDef("reg_alpha", "float", low=1e-8, high=10.0, log=True,
             description="L1 正则化系数"),
    ParamDef("reg_lambda", "float", low=1e-8, high=10.0, log=True,
             description="L2 正则化系数"),
]


_LIGHTGBM_SEARCH_SPACE: list[ParamDef] = [
    ParamDef("n_estimators", "int", low=50, high=500, step=50,
             description="弱学习器数量"),
    ParamDef("max_depth", "int", low=3, high=15,
             description="树的最大深度"),
    ParamDef("learning_rate", "float", low=1e-3, high=0.3, log=True,
             description="学习率"),
    ParamDef("subsample", "float", low=0.5, high=1.0,
             description="行采样比例"),
    ParamDef("colsample_bytree", "float", low=0.5, high=1.0,
             description="列采样比例"),
    ParamDef("min_child_samples", "int", low=5, high=100,
             description="叶子节点最少样本数"),
    ParamDef("num_leaves", "int", low=15, high=127,
             description="叶子节点数量（控制模型复杂度）"),
    ParamDef("reg_alpha", "float", low=1e-8, high=10.0, log=True,
             description="L1 正则化系数"),
    ParamDef("reg_lambda", "float", low=1e-8, high=10.0, log=True,
             description="L2 正则化系数"),
]


_CATBOOST_SEARCH_SPACE: list[ParamDef] = [
    ParamDef("iterations", "int", low=50, high=500, step=50,
             description="迭代轮数"),
    ParamDef("depth", "int", low=3, high=10,
             description="树深度"),
    ParamDef("learning_rate", "float", low=1e-3, high=0.3, log=True,
             description="学习率"),
    ParamDef("subsample", "float", low=0.5, high=1.0,
             description="行采样比例"),
    ParamDef("colsample_bylevel", "float", low=0.5, high=1.0,
             description="每层列采样比例"),
    ParamDef("min_data_in_leaf", "int", low=1, high=50,
             description="叶子节点最少样本数"),
    ParamDef("l2_leaf_reg", "float", low=1e-8, high=10.0, log=True,
             description="L2 正则化系数"),
    ParamDef("random_strength", "float", low=1e-8, high=10.0, log=True,
             description="随机噪声强度"),
    
]


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
    "bootstrap_type": "Bernoulli",
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
        search_space_defs=_RF_SEARCH_SPACE,
        default_params=_DEFAULT_PARAMS_RF,
        supports_feature_importance=True,
    ),
    ModelSpec(
        name="xgboost",
        task_type="regression",
        factory=_make_xgboost_regressor,
        search_space_defs=_XGBOOST_SEARCH_SPACE,
        default_params=_DEFAULT_PARAMS_XGB,
        supports_feature_importance=True,
        optional_dependency="xgboost",
    ),
    ModelSpec(
        name="lightgbm",
        task_type="regression",
        factory=_make_lightgbm_regressor,
        search_space_defs=_LIGHTGBM_SEARCH_SPACE,
        default_params=_DEFAULT_PARAMS_LGBM,
        supports_feature_importance=True,
        optional_dependency="lightgbm",
    ),
    ModelSpec(
        name="catboost",
        task_type="regression",
        factory=_make_catboost_regressor,
        search_space_defs=_CATBOOST_SEARCH_SPACE,
        default_params=_DEFAULT_PARAMS_CATB,
        supports_feature_importance=True,
        optional_dependency="catboost",
    ),
    # ── 分类 ────────────────────────────────────────────────
    ModelSpec(
        name="random_forest",
        task_type="classification",
        factory=_make_random_forest_classifier,
        search_space_defs=_RF_SEARCH_SPACE,
        default_params=_DEFAULT_PARAMS_RF,
        supports_predict_proba=True,
        supports_feature_importance=True,
    ),
    ModelSpec(
        name="xgboost",
        task_type="classification",
        factory=_make_xgboost_classifier,
        search_space_defs=_XGBOOST_SEARCH_SPACE,
        default_params=_DEFAULT_PARAMS_XGB,
        supports_predict_proba=True,
        supports_feature_importance=True,
        optional_dependency="xgboost",
    ),
    ModelSpec(
        name="lightgbm",
        task_type="classification",
        factory=_make_lightgbm_classifier,
        search_space_defs=_LIGHTGBM_SEARCH_SPACE,
        default_params=_DEFAULT_PARAMS_LGBM,
        supports_predict_proba=True,
        supports_feature_importance=True,
        optional_dependency="lightgbm",
    ),
    ModelSpec(
        name="catboost",
        task_type="classification",
        factory=_make_catboost_classifier,
        search_space_defs=_CATBOOST_SEARCH_SPACE,
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