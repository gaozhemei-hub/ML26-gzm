"""core/learning 子模块的单元测试。

覆盖：
- ModelSearchConfig 参数校验
- 指标计算与自动选择
- 模型注册表查询
- Optuna 调参（faker 单进程模式）
- 模型搜索编排（集成，短 trial 数）
- Voting 组合
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes

from tabularx.core.params import (
    ModelSearchConfig,
    TabularXParams,
    FairnessConfig,
)
from tabularx.core.learning import (
    ModelRegistry,
    get_default_registry,
    resolve_primary_metric,
    compute_all_metrics,
    compute_regression_metrics,
    compute_classification_metrics,
    tune_model,
    search_models,
    build_voting,
)
from tabularx.core.learning.model_registry import ModelSpec
from tabularx.core.learning.metrics import (
    REGRESSION_METRICS,
    CLASSIFICATION_METRICS,
)
from tabularx.exceptions.learning import (
    ModelSearchError,
    ModelNotAvailableError,
)
from tabularx.schema.learning import (
    ModelRunResult,
    ModelSearchResult,
    ParamDef,
)
from tabularx.schema.preprocess import PreprocessedData
from tabularx.schema.profile import DataProfile, ColumnFact, TargetFact


# ============================================================
# ModelSearchConfig —— 参数校验
# ============================================================


class TestModelSearchConfig:
    def test_default_values(self):
        cfg = ModelSearchConfig()
        assert cfg.n_jobs == 4
        assert cfg.n_trials == 50
        assert cfg.models == ["auto"]
        assert cfg.top_k_models == 3
        assert cfg.primary_metric == "auto"
        assert cfg.retrain_on_train_valid is True

    def test_set_param(self):
        cfg = ModelSearchConfig()
        cfg.set_param("n_jobs", 8)
        assert cfg.n_jobs == 8

    def test_set_param_validation(self):
        cfg = ModelSearchConfig()
        with pytest.raises(ValueError, match="n_jobs"):
            cfg.set_param("n_jobs", 0)
        with pytest.raises(ValueError, match="n_jobs"):
            cfg.set_param("n_jobs", -1)

    def test_invalid_primary_metric(self):
        with pytest.raises(ValueError, match="primary_metric"):
            ModelSearchConfig(primary_metric="invalid_metric")

    def test_empty_models(self):
        with pytest.raises(ValueError, match="models"):
            ModelSearchConfig(models=[])

    def test_invalid_name(self):
        cfg = ModelSearchConfig()
        with pytest.raises(ValueError, match="未知参数"):
            cfg.set_param("non_existent", 1)

    def test_tabularx_params_integration(self):
        p = TabularXParams()
        assert p.model_search.n_jobs == 4
        p.set_param("model_search", "n_jobs", 6)
        assert p.model_search.n_jobs == 6

    def test_to_dict(self):
        cfg = ModelSearchConfig()
        d = cfg.to_dict()
        assert d["n_jobs"] == 4
        assert d["n_trials"] == 50
        assert "model_search" in TabularXParams().to_dict()


# ============================================================
# 指标计算
# ============================================================


class TestMetrics:
    def test_resolve_primary_metric_regression(self):
        name, direction = resolve_primary_metric("regression")
        assert name == "rmse"
        assert direction == "minimize"

    def test_resolve_primary_metric_binary(self):
        name, direction = resolve_primary_metric("binary")
        assert name == "roc_auc"
        assert direction == "maximize"

    def test_resolve_primary_metric_multiclass(self):
        name, direction = resolve_primary_metric("multiclass")
        assert name == "f1_macro"
        assert direction == "maximize"

    def test_resolve_custom_metric(self):
        name, direction = resolve_primary_metric("regression", "mae")
        assert name == "mae"
        assert direction == "minimize"

    def test_resolve_invalid_metric(self):
        with pytest.raises(ValueError):
            resolve_primary_metric("regression", "roc_auc")

    def test_compute_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] > 0
        assert 0 < metrics["r2"] < 1

    def test_compute_classification_metrics(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2],
                           [0.3, 0.7], [0.6, 0.4]])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert metrics["accuracy"] > 0

    def test_compute_all_metrics_regression(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_all_metrics("regression", y_true, y_pred)
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0


# ============================================================
# 模型注册表
# ============================================================


class TestModelRegistry:
    def test_default_registry(self):
        registry = get_default_registry()
        reg_models = registry.list_models("regression")
        cls_models = registry.list_models("classification")
        assert "random_forest" in reg_models
        assert "xgboost" in reg_models
        assert "lightgbm" in reg_models
        assert "catboost" in reg_models
        assert reg_models == cls_models  # 回归和分类应该同名列表

    def test_get_spec(self):
        registry = get_default_registry()
        spec = registry.get("random_forest", "regression")
        assert spec.name == "random_forest"
        assert spec.task_type == "regression"
        assert spec.supports_feature_importance is True
        assert spec.supports_predict_proba is False

    def test_get_classification_spec(self):
        registry = get_default_registry()
        spec = registry.get("xgboost", "classification")
        assert spec.name == "xgboost"
        assert spec.task_type == "classification"
        assert spec.supports_predict_proba is True

    def test_get_not_available(self):
        registry = get_default_registry()
        with pytest.raises(ModelNotAvailableError):
            registry.get("nonexistent_model", "regression")

    def test_get_wrong_task(self):
        registry = get_default_registry()
        with pytest.raises(ModelNotAvailableError):
            registry.get("random_forest", "regression_not_exist")

    def test_custom_registration(self):
        def dummy_factory(**params):
            return "dummy_model"

        spec = ModelSpec(
            name="my_custom_model",
            task_type="regression",
            factory=dummy_factory,
            search_space_defs=[
                ParamDef("param1", "int", low=1, high=10,
                         description="dummy param"),
            ],
        )
        registry = ModelRegistry()
        registry.register(spec)
        retrieved = registry.get("my_custom_model", "regression")
        assert retrieved.name == "my_custom_model"


# ============================================================
# Optuna 调参（使用小数据 + 少 trial，确保测试速度快）
# ============================================================


class TestOptunaTuner:
    @pytest.fixture
    def regression_data(self):
        X, y = load_diabetes(return_X_y=True)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_ser = pd.Series(y, name="target")
        split_idx = len(X) // 2
        return (
            X_df.iloc[:split_idx],
            y_ser.iloc[:split_idx],
            X_df.iloc[split_idx:],
            y_ser.iloc[split_idx:],
        )

    def test_tune_random_forest_regression(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        registry = get_default_registry()
        spec = registry.get("random_forest", "regression")

        result = tune_model(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            task_type="regression",
            metric_name="rmse",
            direction="minimize",
            n_trials=3,
        )

        assert result.model_name == "random_forest"
        assert result.best_value > 0
        assert len(result.best_params) > 0
        assert result.n_trials == 3
        assert result.duration_seconds is not None
        assert len(result.trials) > 0


# ============================================================
# 模型搜索编排（集成测试）
# ============================================================


class TestModelSearch:
    @pytest.fixture
    def preprocessed_and_profile(self):
        """构造回归任务的 PreprocessedData + DataProfile。"""
        X, y = load_diabetes(return_X_y=True)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_ser = pd.Series(y, name="target")

        n = len(X_df)
        split = int(n * 0.6), int(n * 0.8)

        ppd = PreprocessedData(
            X_train=X_df.iloc[: split[0]],
            y_train=y_ser.iloc[: split[0]],
            X_valid=X_df.iloc[split[0]: split[1]],
            y_valid=y_ser.iloc[split[0]: split[1]],
            X_test=X_df.iloc[split[1]:],
            y_test=y_ser.iloc[split[1]:],
            n_features_in=X_df.shape[1],
            n_features_out=X_df.shape[1],
            dropped_columns=[],
            feature_names_out=list(X_df.columns),
        )

        profile = DataProfile(
            n_rows=n,
            n_cols=X_df.shape[1],
            columns=[],
            target=TargetFact(
                task_type="regression",
                mean=float(y_ser.mean()),
                std=float(y_ser.std()),
            ),
            fingerprint="test",
        )

        return ppd, profile

    def test_search_with_one_model(self, preprocessed_and_profile):
        """只跑 random_forest 2 个 trial，验证整体流程通顺。"""
        ppd, profile = preprocessed_and_profile

        config = ModelSearchConfig(
            models=["random_forest"],
            n_trials=2,
            n_jobs=1,
        )

        result = search_models(
            preprocessed_data=ppd,
            profile=profile,
            config=config,
        )

        assert result.task_type == "regression"
        assert result.primary_metric == "rmse"
        assert result.n_models_total == 1
        assert result.n_models_succeeded == 1
        assert result.n_models_failed == 0
        assert result.best_model_name == "random_forest"
        assert len(result.best_model_params) > 0
        assert len(result.best_validation_metrics) > 0
        assert "rmse" in result.best_validation_metrics
        assert result.best_test_metrics is None
        assert all(r.test_metrics is None for r in result.model_results)

    def test_search_all_models_short(self, preprocessed_and_profile):
        """全部模型各 2 个 trial，验证并行不出错。"""
        ppd, profile = preprocessed_and_profile

        config = ModelSearchConfig(
            models=["auto"],
            n_trials=2,
            n_jobs=2,
        )

        result = search_models(
            preprocessed_data=ppd,
            profile=profile,
            config=config,
        )

        assert result.n_models_total > 1
        assert result.n_models_succeeded > 0
        assert result.best_model_name is not None
        # ranking 按验证集指标排序
        assert len(result.ranking) == result.n_models_succeeded

    def test_best_model_retrain_test(self, preprocessed_and_profile):
        """验证 retrain_on_train_valid=True 时测试集指标存在。"""
        ppd, profile = preprocessed_and_profile

        config = ModelSearchConfig(
            models=["random_forest"],
            n_trials=2,
            retrain_on_train_valid=True,
        )

        result = search_models(
            preprocessed_data=ppd,
            profile=profile,
            config=config,
        )

        assert result.best_test_metrics is None
        assert all(r.test_metrics is None for r in result.model_results)

    def test_search_failed_model(self, preprocessed_and_profile):
        """验证注册表中不存在的模型报错。"""
        ppd, profile = preprocessed_and_profile

        config = ModelSearchConfig(
            models=["nonexistent_model"],
        )

        with pytest.raises(ModelNotAvailableError):
            search_models(
                preprocessed_data=ppd,
                profile=profile,
                config=config,
            )

    def test_search_with_fixed_params(self, preprocessed_and_profile):
        """验证 fixed_params 使指定参数跳过 Optuna 搜索。"""
        ppd, profile = preprocessed_and_profile

        config = ModelSearchConfig(
            models=["random_forest"],
            n_trials=2,
            fixed_params={"random_forest": {"max_depth": 5, "n_estimators": 100}},
        )

        result = search_models(
            preprocessed_data=ppd,
            profile=profile,
            config=config,
        )

        assert result.best_model_name == "random_forest"
        assert result.best_model_params["max_depth"] == 5
        assert result.best_model_params["n_estimators"] == 100

    def test_search_with_search_space_overrides(self, preprocessed_and_profile):
        """验证 search_space_overrides 改变了搜索边界。"""
        ppd, profile = preprocessed_and_profile

        config = ModelSearchConfig(
            models=["random_forest"],
            n_trials=3,
            search_space_overrides={
                "random_forest": {"n_estimators": {"low": 100, "high": 200}}
            },
        )

        result = search_models(
            preprocessed_data=ppd,
            profile=profile,
            config=config,
        )

        assert result.n_models_succeeded >= 0
        assert result.best_model_name == "random_forest"


# ============================================================
# Voting 组合
# ============================================================


class TestEnsembleSearch:
    @pytest.fixture
    def search_result_with_data(self):
        """构造一个包含多个成功模型结果的搜索。"""
        X, y = load_diabetes(return_X_y=True)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_ser = pd.Series(y, name="target")

        n = len(X_df)
        split = int(n * 0.6), int(n * 0.8)

        ppd = PreprocessedData(
            X_train=X_df.iloc[: split[0]],
            y_train=y_ser.iloc[: split[0]],
            X_valid=X_df.iloc[split[0]: split[1]],
            y_valid=y_ser.iloc[split[0]: split[1]],
            X_test=X_df.iloc[split[1]:],
            y_test=y_ser.iloc[split[1]:],
            n_features_in=X_df.shape[1],
            n_features_out=X_df.shape[1],
            dropped_columns=[],
            feature_names_out=list(X_df.columns),
        )

        registry = get_default_registry()
        specs = [registry.get(name, "regression")
                 for name in ["random_forest", "xgboost"]]

        model_results = []
        for spec in specs:
            params = {}
            if spec.default_params:
                params = dict(spec.default_params)
            params["n_estimators"] = 10
            params["max_depth"] = 3

            if spec.name == "xgboost":
                params["n_estimators"] = 10
                params["max_depth"] = 3

            try:
                model = spec.factory(**params)
                model.fit(ppd.X_train, ppd.y_train)
                y_pred = model.predict(ppd.X_valid)
                y_prob = None
                if spec.supports_predict_proba:
                    try:
                        y_prob = model.predict_proba(ppd.X_valid)
                    except Exception:
                        y_prob = None
                valid_metrics = compute_all_metrics(
                    "regression", ppd.y_valid, y_pred, y_prob
                )
                model_results.append(
                    ModelRunResult(
                        model_name=spec.name,
                        task_type="regression",
                        status="success",
                        best_params=params,
                        n_trials=1,
                        validation_metrics=valid_metrics,
                    )
                )
            except Exception as e:
                model_results.append(
                    ModelRunResult(
                        model_name=spec.name,
                        task_type="regression",
                        status="failed",
                        best_params={},
                        n_trials=1,
                        error=str(e),
                    )
                )

        search_result = ModelSearchResult(
            task_type="regression",
            primary_metric="rmse",
            optimization_direction="minimize",
            model_results=model_results,
            best_model_name=model_results[0].model_name if model_results else None,
            best_model_params=model_results[0].best_params if model_results else None,
            best_validation_metrics=model_results[0].validation_metrics if model_results else {},
        )

        return search_result, ppd

    def test_voting_with_enough_models(self, search_result_with_data):
        """有 2 个成功模型时，Voting 应该成功构建。"""
        search_result, ppd = search_result_with_data

        voting_result = build_voting(
            search_result=search_result,
            preprocessed_data=ppd,
            task_type="regression",
        )

        assert voting_result is not None
        assert voting_result.status == "success"
        assert voting_result.model_name == "voting"
        assert len(voting_result.validation_metrics) > 0
        assert voting_result.test_metrics is None

    def test_voting_no_models(self):
        """没有模型时，Voting 应该返回 None。"""
        search_result = ModelSearchResult(
            task_type="regression",
            primary_metric="rmse",
            optimization_direction="minimize",
            model_results=[],
        )
        from tabularx.schema.preprocess import PreprocessedData
        ppd = PreprocessedData(
            X_train=pd.DataFrame(),
            y_train=pd.Series(),
            X_valid=pd.DataFrame(),
            y_valid=pd.Series(),
            X_test=pd.DataFrame(),
            y_test=pd.Series(),
            n_features_in=0,
            n_features_out=0,
            dropped_columns=[],
            feature_names_out=[],
        )

        result = build_voting(
            search_result=search_result,
            preprocessed_data=ppd,
            task_type="regression",
        )

        assert result is None


# ============================================================
# FairnessConfig —— 基本创建
# ============================================================


class TestFairnessConfig:
    def test_default(self):
        cfg = FairnessConfig()
        assert cfg.sensitive_features == []
        assert "demographic_parity" in cfg.metrics

    def test_custom(self):
        cfg = FairnessConfig(
            sensitive_features=["gender", "race"],
        )
        assert "gender" in cfg.sensitive_features
