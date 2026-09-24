"""公平性评估模块的单元测试。

覆盖：
- evaluate_fairness 在未指定敏感特征时返回 enabled=False
- evaluate_fairness 在指定敏感特征时正确计算各子组指标
- 特征被丢弃时的处理
- Single/multiple sensitive features
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TabularX" / "src"))

from tabularx.core.learning.fairness import evaluate_fairness
from tabularx.schema.learning import (
    FairnessResult,
    FairnessFeatureResult,
    FairnessGroupResult,
)
from tabularx.schema.preprocess import ColumnMapping


# ============================================================
# 测试
# ============================================================

class TestEvaluateFairness:
    """evaluate_fairness 的核心功能测试。"""

    def test_no_sensitive_features(self):
        """未指定敏感特征时返回 enabled=False。"""
        n = 50
        result = evaluate_fairness(
            X_test=pd.DataFrame({"x": np.random.randn(n)}),
            y_test=pd.Series(np.random.randint(0, 2, n)),
            y_pred=np.random.randint(0, 2, n),
            y_prob=np.random.rand(n, 2),
            column_mappings=[],
            sensitive_features=[],
            task_type="binary",
        )
        assert isinstance(result, FairnessResult)
        assert result.enabled is False
        assert len(result.feature_results) == 0

    def test_single_sensitive_binary(self):
        """单个二值敏感特征：正确计算各子组指标。"""
        n = 100
        X_test = pd.DataFrame({
            "gender": np.random.randint(0, 2, n),
            "f1": np.random.randn(n),
        })
        y_test = pd.Series(np.random.randint(0, 2, n))
        y_pred = np.random.randint(0, 2, n)
        y_prob = np.random.rand(n, 2)

        mappings = [
            ColumnMapping(original_name="gender", output_names=["gender"], n_features_out=1),
            ColumnMapping(original_name="f1", output_names=["f1"], n_features_out=1),
        ]

        result = evaluate_fairness(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            column_mappings=mappings,
            sensitive_features=["gender"],
            task_type="binary",
        )

        assert result.enabled is True
        assert len(result.feature_results) == 1
        feat = result.feature_results[0]
        assert feat.is_dropped is False
        assert len(feat.groups) == 2  # 0 和 1 两个子组
        assert feat.demographic_parity_diff is not None
        assert feat.equalized_odds_tpr_diff is not None
        assert feat.equalized_odds_fpr_diff is not None

    def test_dropped_feature(self):
        """被丢弃的敏感特征：记录 is_dropped。"""
        n = 50
        X_test = pd.DataFrame({"x": np.random.randn(n)})
        y_test = pd.Series(np.random.randint(0, 2, n))
        y_pred = np.random.randint(0, 2, n)

        mappings = [
            ColumnMapping(original_name="race", output_names=[], is_dropped=True, n_features_out=0),
        ]

        result = evaluate_fairness(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=None,
            column_mappings=mappings,
            sensitive_features=["race"],
            task_type="binary",
        )

        assert result.enabled is True
        assert len(result.feature_results) == 1
        feat = result.feature_results[0]
        assert feat.is_dropped is True
        assert feat.dropped_reason is not None
        assert len(feat.groups) == 0

    def test_multiple_sensitive_features(self):
        """多个敏感特征：对每个独立分析。"""
        n = 100
        X_test = pd.DataFrame({
            "gender": np.random.randint(0, 2, n),
            "age_demo": np.random.randint(0, 3, n),  # 三个子组
            "f1": np.random.randn(n),
        })
        y_test = pd.Series(np.random.randint(0, 2, n))
        y_pred = np.random.randint(0, 2, n)

        mappings = [
            ColumnMapping(original_name="gender", output_names=["gender"], n_features_out=1),
            ColumnMapping(original_name="age_demo", output_names=["age_demo"], n_features_out=1),
        ]

        result = evaluate_fairness(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=None,
            column_mappings=mappings,
            sensitive_features=["gender", "age_demo"],
            task_type="binary",
        )

        assert result.enabled is True
        assert len(result.feature_results) == 2
        # 第一个特征有 2 个子组
        assert len(result.feature_results[0].groups) == 2
        # 第二个特征有 3 个子组
        assert len(result.feature_results[1].groups) == 3

    def test_onehot_expanded_feature(self):
        """OneHot 展开的特征：正确还原为原始类别标签。"""
        n = 100
        # 模拟 OneHot 展开：3 个输出列
        group_assignments = np.random.choice(["A", "B", "C"], n)
        X_test = pd.DataFrame({
            "group_A": (group_assignments == "A").astype(int),
            "group_B": (group_assignments == "B").astype(int),
            "group_C": (group_assignments == "C").astype(int),
            "f1": np.random.randn(n),
        })
        y_test = pd.Series(np.random.randint(0, 2, n))
        y_pred = np.random.randint(0, 2, n)

        mappings = [
            ColumnMapping(
                original_name="group",
                output_names=["group_A", "group_B", "group_C"],
                n_features_out=3,
            ),
        ]

        result = evaluate_fairness(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=None,
            column_mappings=mappings,
            sensitive_features=["group"],
            task_type="binary",
        )

        assert result.enabled is True
        assert len(result.feature_results) == 1
        feat = result.feature_results[0]
        assert feat.is_dropped is False
        # OneHot 展开后应该能正确还原为 A/B/C 三个子组
        assert len(feat.groups) == 3

    def test_demographic_parity_zero_when_identical(self):
        """所有子组预测率相同时，DP 应为 0。"""
        n = 60
        y_test = pd.Series(np.zeros(n))
        y_pred = np.zeros(n)  # 全预测 0

        X_test = pd.DataFrame({"gender": [0] * 30 + [1] * 30})

        mappings = [
            ColumnMapping(original_name="gender", output_names=["gender"], n_features_out=1),
        ]

        result = evaluate_fairness(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=None,
            column_mappings=mappings,
            sensitive_features=["gender"],
            task_type="binary",
        )

        assert result.feature_results[0].demographic_parity_diff == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])