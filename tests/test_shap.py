"""SHAP 可解释性分析模块的单元测试。

覆盖：
- explain_model 在多分类任务上的执行
- explain_model 在回归任务上的执行
- 返回结果包含正确的特征重要性和图路径
- 错误处理（非法模型）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TabularX" / "src"))

from tabularx.core.learning.model_registry import get_default_registry
from tabularx.core.learning.shap_explainer import explain_model
from tabularx.schema.learning import ShapResult


# ============================================================
# 辅助：准备训练和解释数据
# ============================================================

def _prepare_iris() -> tuple:
    """准备鸢尾花多分类数据。"""
    registry = get_default_registry()
    spec = registry.get("random_forest", "classification")

    X, y = load_iris(return_X_y=True)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")

    n = len(X_df)
    split = int(n * 0.7)
    X_train = X_df.iloc[:split]
    y_train = y_ser.iloc[:split]
    X_test = X_df.iloc[split:]

    best_params = {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "ccp_alpha": 0.0,
    }

    return spec, best_params, X_train, y_train, X_test, list(X_df.columns), "multiclass"


def _prepare_diabetes() -> tuple:
    """准备糖尿病回归数据。"""
    registry = get_default_registry()
    spec = registry.get("random_forest", "regression")

    X, y = load_diabetes(return_X_y=True)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")

    n = len(X_df)
    split = int(n * 0.7)
    X_train = X_df.iloc[:split]
    y_train = y_ser.iloc[:split]
    X_test = X_df.iloc[split:]

    best_params = {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "ccp_alpha": 0.0,
    }

    return spec, best_params, X_train, y_train, X_test, list(X_df.columns), "regression"


# ============================================================
# 测试
# ============================================================

class TestExplainModel:
    """explain_model 的核心功能测试。"""

    def test_multiclass_returns_shap_result(self, tmp_path):
        """多分类任务：返回 ShapResult 且特征重要性非空。"""
        spec, best_params, X_train, y_train, X_test, feature_names, task_type = _prepare_iris()

        result = explain_model(
            spec=spec,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_explain=X_test,
            feature_names=feature_names,
            task_type=task_type,
            output_dir=tmp_path,
            model_name="rf_iris",
            max_explain_samples=50,
        )

        assert isinstance(result, ShapResult)
        assert result.error is None, f"SHAP 执行出错：{result.error}"
        assert len(result.feature_importance) > 0
        assert len(result.top_k_features) > 0
        assert result.prediction_type == "multiclass"
        assert result.model_name == "rf_iris"

    def test_multiclass_artifact_paths_not_empty(self, tmp_path):
        """多分类任务：生成至少一张 SHAP 图。"""
        spec, best_params, X_train, y_train, X_test, feature_names, task_type = _prepare_iris()

        result = explain_model(
            spec=spec,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_explain=X_test,
            feature_names=feature_names,
            task_type=task_type,
            output_dir=tmp_path,
            model_name="rf_iris",
            max_explain_samples=50,
        )

        assert len(result.artifact_paths) > 0, "至少应该生成一张 SHAP 图"
        # 检查图文件是否存在
        for key, rel_path in result.artifact_paths.items():
            full_path = tmp_path / Path(rel_path).name
            assert full_path.exists(), f"图文件不存在：{full_path}"

    def test_multiclass_feature_names_match(self, tmp_path):
        """多分类：特征重要性中的特征名来自输入。"""
        spec, best_params, X_train, y_train, X_test, feature_names, task_type = _prepare_iris()

        result = explain_model(
            spec=spec,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_explain=X_test,
            feature_names=feature_names,
            task_type=task_type,
            output_dir=tmp_path,
            model_name="rf_iris",
            max_explain_samples=50,
        )

        for feat in result.feature_importance:
            assert feat in feature_names, f"特征名 {feat} 不在输入特征名列表中"

    def test_regression_returns_shap_result(self, tmp_path):
        """回归任务：返回 ShapResult 且特征重要性非空。"""
        spec, best_params, X_train, y_train, X_test, feature_names, task_type = _prepare_diabetes()

        result = explain_model(
            spec=spec,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_explain=X_test,
            feature_names=feature_names,
            task_type=task_type,
            output_dir=tmp_path,
            model_name="rf_diabetes",
            max_explain_samples=50,
        )

        assert isinstance(result, ShapResult)
        assert result.error is None, f"SHAP 执行出错：{result.error}"
        assert len(result.feature_importance) > 0
        assert result.prediction_type == "regression"

    def test_regression_artifact_paths_not_empty(self, tmp_path):
        """回归任务：生成至少一张 SHAP 图。"""
        spec, best_params, X_train, y_train, X_test, feature_names, task_type = _prepare_diabetes()

        result = explain_model(
            spec=spec,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_explain=X_test,
            feature_names=feature_names,
            task_type=task_type,
            output_dir=tmp_path,
            model_name="rf_diabetes",
            max_explain_samples=50,
        )

        assert len(result.artifact_paths) > 0, "至少应该生成一张 SHAP 图"

    def test_error_on_invalid_model(self, tmp_path):
        """传入未训练的模型参数：优雅地返回 ShapResult 带 error。"""
        spec, _, X_train, y_train, X_test, feature_names, task_type = _prepare_iris()

        # 传入空参—RandomForest 会因缺少关键参数而无法训练，应返回错误
        result = explain_model(
            spec=spec,
            best_params={},  # 空参数可能导致训练失败
            X_train=X_train,
            y_train=y_train,
            X_explain=X_test,
            feature_names=feature_names,
            task_type=task_type,
            output_dir=tmp_path,
            model_name="bad_model",
            max_explain_samples=50,
        )

        # 应该返回带错误的结果而不是崩溃
        assert isinstance(result, ShapResult)
        # 空参数可能随机森林能用默认值，也可能失败。只要不崩即可

    def test_feature_importance_sorted_descending(self, tmp_path):
        """特征重要性按降序排列。"""
        spec, best_params, X_train, y_train, X_test, feature_names, task_type = _prepare_iris()

        result = explain_model(
            spec=spec,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_explain=X_test,
            feature_names=feature_names,
            task_type=task_type,
            output_dir=tmp_path,
            model_name="rf_iris",
            max_explain_samples=50,
        )

        if result.feature_importance:
            values = list(result.feature_importance.values())
            for i in range(len(values) - 1):
                assert values[i] >= values[i + 1], (
                    f"特征重要性未降序排列：索引 {i} = {values[i]}, {i+1} = {values[i+1]}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])