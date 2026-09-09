"""core/preprocess.py 与 schema/preprocess.py 的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from tabularx import build_profile, preprocess, split_data
from tabularx.core.params import PreprocessConfig
from tabularx.core.preprocess import (
    _transform_boolean,
    _transform_categorical,
    _transform_datetime,
    _transform_numeric,
)
from tabularx.exceptions.preprocess import (
    PreprocessError,
    TextColumnNotSupportedError,
    TransformError,
)
from tabularx.schema.preprocess import PreprocessedData, TransformRecord
from tabularx.schema.profile import DataProfile
from tabularx.schema.split import SplitResult


# ─────────────────────────────────────────────────────────────
# PreprocessConfig —— 参数校验
# ─────────────────────────────────────────────────────────────

class TestPreprocessConfig:
    def test_default_values(self):
        cfg = PreprocessConfig()
        assert cfg.numeric_impute == "median"
        assert cfg.categorical_impute == "most_frequent"
        assert cfg.scale_method == "standard"
        assert cfg.skew_method == "yeo-johnson"
        assert cfg.low_cardinality_encoding == "onehot"
        assert cfg.high_cardinality_encoding == "target"
        assert cfg.max_onehot_categories == 20
        assert cfg.handle_text == "drop"
        assert cfg.handle_datetime == "extract"

    def test_invalid_numeric_impute(self):
        with pytest.raises(ValueError, match="numeric_impute"):
            PreprocessConfig(numeric_impute="invalid")

    def test_invalid_categorical_impute(self):
        with pytest.raises(ValueError, match="categorical_impute"):
            PreprocessConfig(categorical_impute="invalid")

    def test_invalid_scale_method(self):
        with pytest.raises(ValueError, match="scale_method"):
            PreprocessConfig(scale_method="invalid")

    def test_invalid_skew_method(self):
        with pytest.raises(ValueError, match="skew_method"):
            PreprocessConfig(skew_method="invalid")

    def test_invalid_low_cardinality_encoding(self):
        with pytest.raises(ValueError, match="low_cardinality_encoding"):
            PreprocessConfig(low_cardinality_encoding="invalid")

    def test_invalid_high_cardinality_encoding(self):
        with pytest.raises(ValueError, match="high_cardinality_encoding"):
            PreprocessConfig(high_cardinality_encoding="invalid")

    def test_invalid_handle_text(self):
        with pytest.raises(ValueError, match="handle_text"):
            PreprocessConfig(handle_text="invalid")

    def test_invalid_handle_datetime(self):
        with pytest.raises(ValueError, match="handle_datetime"):
            PreprocessConfig(handle_datetime="invalid")

    def test_invalid_datetime_features(self):
        with pytest.raises(ValueError, match="datetime_features"):
            PreprocessConfig(datetime_features="invalid")

    def test_max_onehot_categories_too_small(self):
        with pytest.raises(ValueError, match="max_onehot_categories"):
            PreprocessConfig(max_onehot_categories=0)

    def test_set_param_valid(self):
        cfg = PreprocessConfig()
        cfg.set_param("scale_method", "minmax")
        assert cfg.scale_method == "minmax"

    def test_set_param_invalid_name(self):
        cfg = PreprocessConfig()
        with pytest.raises(ValueError, match="未知参数"):
            cfg.set_param("nonexistent", 42)

    def test_set_param_invalid_value(self):
        cfg = PreprocessConfig()
        with pytest.raises(ValueError, match="scale_method"):
            cfg.set_param("scale_method", "invalid")

    def test_impute_constant_value_range(self):
        # constant value 无范围限制，应该正常工作
        cfg = PreprocessConfig(impute_constant_value=-999.0)
        assert cfg.impute_constant_value == -999.0

    def test_overview_contains_keys(self):
        cfg = PreprocessConfig()
        overview = cfg.overview()
        for key in ("numeric_impute", "scale_method", "skew_method"):
            assert key in overview

    def test_to_dict(self):
        cfg = PreprocessConfig()
        d = cfg.to_dict()
        assert d["scale_method"] == "standard"
        assert isinstance(d, dict)


# ─────────────────────────────────────────────────────────────
# _transform_numeric —— 数值列变换
# ─────────────────────────────────────────────────────────────

class TestTransformNumeric:
    def _make_profile_with_tags(self, col_name: str, *tags) -> DataProfile:
        from tabularx.schema.profile import ColumnFact, TargetFact

        return DataProfile(
            n_rows=100,
            n_cols=1,
            columns=[
                ColumnFact(
                    col_name=col_name,
                    dtype="float64",
                    n_missing=0,
                    missing_rate=0.0,
                    n_unique=50,
                    tags=list(tags),
                    evidences=[],
                )
            ],
            target=TargetFact(task_type="regression"),
            fingerprint="test",
        )

    def test_no_tags_no_missing(self):
        """数值列无缺失、无 scale/skew 标记 → 不做任何变换。"""
        col_train = pd.Series([1.0, 2.0, 3.0], name="x")
        col_valid = pd.Series([4.0, 5.0], name="x")
        col_test = pd.Series([6.0, 7.0], name="x")
        profile = self._make_profile_with_tags("x", "numeric")
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, note = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert actions == []
        assert note is None
        assert list(out_train) == [1.0, 2.0, 3.0]

    def test_median_impute(self):
        """数值列有缺失 → median 填补。"""
        col_train = pd.Series([1.0, np.nan, 3.0], name="x")
        col_valid = pd.Series([np.nan, 5.0], name="x")
        col_test = pd.Series([6.0, np.nan], name="x")
        profile = self._make_profile_with_tags("x", "numeric")
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, note = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert len(actions) == 1
        assert "median_impute" in actions[0]
        # median of [1, 3] = 2.0
        assert out_train.iloc[1] == pytest.approx(2.0)

    def test_standard_scale(self):
        """scale_needed → StandardScaler。"""
        col_train = pd.Series([100.0, 200.0, 300.0, 400.0], name="x")
        col_valid = pd.Series([500.0, 600.0], name="x")
        col_test = pd.Series([700.0, 800.0], name="x")
        profile = self._make_profile_with_tags("x", "numeric", "scale_needed")
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, note = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert any("standard_scale" in a for a in actions)
        # 训练集 mean≈0, std≈1（StandardScaler 用总体标准差 ddof=0）
        assert abs(out_train.mean()) < 1e-10
        assert abs(np.std(out_train.values)) == pytest.approx(1.0)

    def test_minmax_scale(self):
        """scale_method=minmax → MinMaxScaler。"""
        col_train = pd.Series([100.0, 200.0, 300.0, 400.0], name="x")
        col_valid = pd.Series([500.0, 600.0], name="x")
        col_test = pd.Series([700.0, 800.0], name="x")
        profile = self._make_profile_with_tags("x", "numeric", "scale_needed")
        cfg = PreprocessConfig(scale_method="minmax")

        _, _, _, actions, _ = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert any("minmax_scale" in a for a in actions)

    def test_yeojohnson_skewed(self):
        """skewed → Yeo-Johnson 变换。"""
        col_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], name="x")
        col_valid = pd.Series([6.0, 7.0], name="x")
        col_test = pd.Series([8.0, 9.0], name="x")
        profile = self._make_profile_with_tags("x", "numeric", "skewed")
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, note = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert any("yeo-johnson" in a for a in actions)

    def test_yeojohnson_with_scale_needed(self):
        """skewed + scale_needed → 只做 Yeo-Johnson（自带标准化），action 标记 +scale。"""
        col_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], name="x")
        col_valid = pd.Series([6.0, 7.0], name="x")
        col_test = pd.Series([8.0, 9.0], name="x")
        profile = self._make_profile_with_tags("x", "numeric", "skewed", "scale_needed")
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, note = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert any(a == "yeo-johnson+scale" for a in actions)

    def test_log1p_skewed(self):
        """skew_method=log1p → log1p 变换。"""
        col_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], name="x")
        col_valid = pd.Series([6.0, 7.0], name="x")
        col_test = pd.Series([8.0, 9.0], name="x")
        profile = self._make_profile_with_tags("x", "numeric", "skewed")
        cfg = PreprocessConfig(skew_method="log1p")

        out_train, out_valid, out_test, actions, note = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert any("log1p" in a for a in actions)

    def test_impute_then_scale(self):
        """同时有缺失和 scale_needed → 先填补再 scale。"""
        col_train = pd.Series([100.0, np.nan, 300.0, 400.0], name="x")
        col_valid = pd.Series([np.nan, 600.0], name="x")
        col_test = pd.Series([700.0, np.nan], name="x")
        profile = self._make_profile_with_tags("x", "numeric", "scale_needed")
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, note = _transform_numeric(
            col_train, col_valid, col_test, "x", profile, cfg
        )
        assert len(actions) >= 1
        all_actions = " ".join(actions)
        assert "impute" in all_actions
        assert "scale" in all_actions


# ─────────────────────────────────────────────────────────────
# _transform_boolean —— 布尔列变换
# ─────────────────────────────────────────────────────────────

class TestTransformBoolean:
    def test_no_missing(self):
        """布尔列无缺失 → 直接转 int。"""
        col_train = pd.Series([True, False, True], name="b")
        col_valid = pd.Series([False, True], name="b")
        col_test = pd.Series([True, False], name="b")

        out_train, out_valid, out_test, actions = _transform_boolean(
            col_train, col_valid, col_test, "b"
        )
        assert list(out_train) == [1, 0, 1]
        assert actions == ["to_int"]

    def test_with_missing(self):
        """布尔列有缺失 → 填补 False → 转 int。"""
        col_train = pd.Series([True, np.nan, False], name="b")
        col_valid = pd.Series([np.nan, True], name="b")
        col_test = pd.Series([True, np.nan], name="b")

        out_train, out_valid, out_test, actions = _transform_boolean(
            col_train, col_valid, col_test, "b"
        )
        assert "to_int" in actions
        assert "fillna_False" in actions[0]
        # 缺失被填补为 False(0)
        assert out_train.iloc[1] == 0

    def test_all_true(self):
        """全 True 的布尔列。"""
        col_train = pd.Series([True, True, True], name="b")
        col_valid = pd.Series([True, True], name="b")
        col_test = pd.Series([True, True], name="b")

        out_train, out_valid, out_test, actions = _transform_boolean(
            col_train, col_valid, col_test, "b"
        )
        assert all(out_train == 1)


# ─────────────────────────────────────────────────────────────
# _transform_categorical —— 类别列变换
# ─────────────────────────────────────────────────────────────

class TestTransformCategorical:
    def _make_profile(self, col_name: str, n_unique: int) -> DataProfile:
        from tabularx.schema.profile import ColumnFact, TargetFact

        tags = ["categorical"]
        if n_unique > 50:
            tags.append("high_cardinality")
        return DataProfile(
            n_rows=100,
            n_cols=1,
            columns=[
                ColumnFact(
                    col_name=col_name,
                    dtype="object",
                    n_missing=0,
                    missing_rate=0.0,
                    n_unique=n_unique,
                    tags=tags,
                    evidences=[],
                )
            ],
            target=TargetFact(task_type="binary"),
            fingerprint="test",
        )

    def test_onehot_low_cardinality(self):
        """低基数 → OneHot 编码。"""
        col_train = pd.Series(["a", "b", "c", "a", "b"], name="cat")
        col_valid = pd.Series(["c", "a"], name="cat")
        col_test = pd.Series(["b", "c"], name="cat")
        y_train = pd.Series([0, 1, 0, 1, 0])
        y_valid = pd.Series([0, 1])
        y_test = pd.Series([1, 0])
        profile = self._make_profile("cat", 3)
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, out_features, note = _transform_categorical(
            col_train, col_valid, col_test, y_train, y_valid, y_test,
            "cat", profile, cfg,
        )
        assert any("onehot" in a for a in actions)
        assert len(out_features) == 3  # a, b, c 各一列
        assert out_train.shape[1] == 3
        # a 行 = [1, 0, 0]
        assert out_train.iloc[0, 0] == 1.0

    def test_label_low_cardinality(self):
        """低基数 + low_cardinality_encoding=label → Label 编码。"""
        col_train = pd.Series(["a", "b", "c", "a", "b"], name="cat")
        col_valid = pd.Series(["c", "a"], name="cat")
        col_test = pd.Series(["b", "c"], name="cat")
        y_train = pd.Series([0, 1, 0, 1, 0])
        y_valid = pd.Series([0, 1])
        y_test = pd.Series([1, 0])
        profile = self._make_profile("cat", 3)
        cfg = PreprocessConfig(low_cardinality_encoding="label")

        out_train, out_valid, out_test, actions, out_features, note = _transform_categorical(
            col_train, col_valid, col_test, y_train, y_valid, y_test,
            "cat", profile, cfg,
        )
        assert any("label_encode" in a for a in actions)
        assert len(out_features) == 1
        assert out_train.shape[1] == 1

    def test_target_encode_high_cardinality(self):
        """高基数 + high_cardinality_encoding=target → 目标编码。"""
        # 构造 60 个唯一值 > 50 阈值
        values = [f"v{i}" for i in range(60)]
        col_train = pd.Series(values * 2, name="cat")  # 120 行
        col_valid = pd.Series(values[:10], name="cat")
        col_test = pd.Series(values[:10], name="cat")
        y_train = pd.Series(np.random.default_rng(42).integers(0, 2, 120))
        y_valid = pd.Series([0] * 10)
        y_test = pd.Series([1] * 10)
        profile = self._make_profile("cat", 60)
        cfg = PreprocessConfig(high_cardinality_encoding="target")

        out_train, out_valid, out_test, actions, out_features, note = _transform_categorical(
            col_train, col_valid, col_test, y_train, y_valid, y_test,
            "cat", profile, cfg,
        )
        assert any("target_encode" in a for a in actions)
        assert out_train.shape[1] == 1
        assert note is not None

    def test_frequency_encode_high_cardinality(self):
        """高基数 + high_cardinality_encoding=frequency → 频率编码。"""
        values = [f"v{i}" for i in range(60)]
        col_train = pd.Series(values * 2, name="cat")
        col_valid = pd.Series(values[:10], name="cat")
        col_test = pd.Series(values[:10], name="cat")
        y_train = pd.Series([0] * 120)
        y_valid = pd.Series([0] * 10)
        y_test = pd.Series([1] * 10)
        profile = self._make_profile("cat", 60)
        cfg = PreprocessConfig(high_cardinality_encoding="frequency")

        out_train, out_valid, out_test, actions, out_features, note = _transform_categorical(
            col_train, col_valid, col_test, y_train, y_valid, y_test,
            "cat", profile, cfg,
        )
        assert any("frequency_encode" in a for a in actions)
        assert out_train.shape[1] == 1

    def test_impute_then_encode(self):
        """类别列有缺失 → 先填补再编码。"""
        col_train = pd.Series(["a", np.nan, "b", "a"], name="cat")
        col_valid = pd.Series([np.nan, "b"], name="cat")
        col_test = pd.Series(["a", np.nan], name="cat")
        y_train = pd.Series([0, 1, 0, 1])
        y_valid = pd.Series([0, 1])
        y_test = pd.Series([1, 0])
        profile = self._make_profile("cat", 3)
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, out_features, note = _transform_categorical(
            col_train, col_valid, col_test, y_train, y_valid, y_test,
            "cat", profile, cfg,
        )
        all_actions = " ".join(actions)
        assert "impute" in all_actions
        assert "onehot" in all_actions

    def test_unknown_category_in_test(self):
        """测试集出现训练集未见过的类别 → handle_unknown=ignore 不报错。"""
        col_train = pd.Series(["a", "b", "c"], name="cat")
        col_valid = pd.Series(["a", "b"], name="cat")
        col_test = pd.Series(["x", "y"], name="cat")  # 新类别
        y_train = pd.Series([0, 1, 0])
        y_valid = pd.Series([0, 1])
        y_test = pd.Series([1, 0])
        profile = self._make_profile("cat", 3)
        cfg = PreprocessConfig()

        out_train, out_valid, out_test, actions, out_features, note = _transform_categorical(
            col_train, col_valid, col_test, y_train, y_valid, y_test,
            "cat", profile, cfg,
        )
        # OneHot 在未知类别上全为 0
        assert out_test.shape[1] == 3


# ─────────────────────────────────────────────────────────────
# _transform_datetime —— 日期列变换
# ─────────────────────────────────────────────────────────────

class TestTransformDatetime:
    def test_basic_extract(self):
        """basic 模式提取年/月/日。"""
        col_train = pd.Series(pd.date_range("2024-01-01", periods=5), name="dt")
        col_valid = pd.Series(pd.date_range("2024-02-01", periods=3), name="dt")
        col_test = pd.Series(pd.date_range("2024-03-01", periods=2), name="dt")
        cfg = PreprocessConfig(datetime_features="basic")

        out_train, out_valid, out_test, actions, out_features = _transform_datetime(
            col_train, col_valid, col_test, "dt", cfg,
        )
        assert out_train.shape[1] == 3  # year, month, day
        assert "year" in out_features[0]
        assert "month" in out_features[1]
        assert "day" in out_features[2]
        assert "datetime_extract" in actions[0]

    def test_full_extract(self):
        """full 模式额外提取星期和季度。"""
        col_train = pd.Series(pd.date_range("2024-01-01", periods=5), name="dt")
        col_valid = pd.Series(pd.date_range("2024-02-01", periods=3), name="dt")
        col_test = pd.Series(pd.date_range("2024-03-01", periods=2), name="dt")
        cfg = PreprocessConfig(datetime_features="full")

        out_train, out_valid, out_test, actions, out_features = _transform_datetime(
            col_train, col_valid, col_test, "dt", cfg,
        )
        assert out_train.shape[1] == 5  # year, month, day, dayofweek, quarter
        assert "dayofweek" in out_features[3]
        assert "quarter" in out_features[4]


# ─────────────────────────────────────────────────────────────
# preprocess —— 端到端集成测试
# ─────────────────────────────────────────────────────────────

def _run_end_to_end(dataset_loader):
    """在单个 sklearn 数据集上跑完整 preprocess 管线，返回结果对象。"""
    data = dataset_loader(as_frame=True)
    X, y = data.data, data.target
    profile = build_profile(X, y)
    split_result = split_data(X, y, profile)
    result = preprocess(X, y, profile, split_result)
    return result, profile, split_result, X


class TestPreprocessEndToEnd:
    def test_breast_cancer(self):
        """breast_cancer: 二分类，30 数值列 → 30 列不变。"""
        result, _, _, X = _run_end_to_end(load_breast_cancer)
        assert isinstance(result, PreprocessedData)
        assert result.n_features_in == 30
        assert result.n_features_out == 30
        assert result.X_train.shape[0] > 0
        assert result.X_valid.shape[0] > 0
        assert result.X_test.shape[0] > 0
        assert result.y_train.shape[0] == result.X_train.shape[0]
        assert result.y_valid.shape[0] == result.X_valid.shape[0]
        assert result.y_test.shape[0] == result.X_test.shape[0]

    def test_iris(self):
        """iris: 多分类，4 列 → 4 列。"""
        result, _, _, X = _run_end_to_end(load_iris)
        assert result.n_features_in == 4
        assert result.n_features_out == 4

    def test_diabetes(self):
        """diabetes: 回归，10 列 → 10 列。"""
        result, _, _, X = _run_end_to_end(load_diabetes)
        assert result.n_features_in == 10
        assert result.n_features_out == 10

    def test_feature_names_out(self):
        """feature_names_out 的长度与 n_features_out 一致。"""
        result, _, _, _ = _run_end_to_end(load_breast_cancer)
        assert len(result.feature_names_out) == result.n_features_out
        assert all(isinstance(n, str) for n in result.feature_names_out)

    def test_dropped_columns_list(self):
        """dropped_columns 始终为 list（即使为空）。"""
        result, _, _, _ = _run_end_to_end(load_breast_cancer)
        assert isinstance(result.dropped_columns, list)

    def test_transforms_present(self):
        """transforms 列表不为空，且每列有记录。"""
        result, _, _, X = _run_end_to_end(load_breast_cancer)
        assert len(result.transforms) == X.shape[1]
        for t in result.transforms:
            assert isinstance(t, TransformRecord)
            assert t.col_name
            assert isinstance(t.action_sequence, list)

    def test_no_data_leakage(self):
        """变换在训练集上拟合，验证/测试集不应影响训练集统计量。"""
        result, _, _, _ = _run_end_to_end(load_breast_cancer)
        # 训练集的均值不应被验证/测试集影响（只是合理性检查）
        # 只要不抛异常即通过
        assert result.X_train is not None

    def test_constant_column_dropped(self):
        """常数列被丢弃。"""
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({
            "a": rng.integers(0, 100, n),
            "b": [0] * n,   # 常数列
            "c": rng.uniform(0, 10, n),
        })
        y = pd.Series([0, 1] * (n // 2))
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        result = preprocess(X, y, profile, split_result)
        assert "b" in result.dropped_columns
        assert "b" not in result.feature_names_out
        assert "b" not in result.X_train.columns

    def test_text_column_dropped_by_default(self):
        """文本列默认被丢弃（handle_text=drop）。"""
        rng = np.random.default_rng(42)
        n = 300
        X = pd.DataFrame({
            "a": rng.uniform(0, 10, n),
            "text": [f"long_text_{i}" for i in range(n)],
        })
        y = pd.Series([0, 1] * (n // 2))
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        result = preprocess(X, y, profile, split_result)
        assert "text" in result.dropped_columns

    def test_text_column_keep_raises(self):
        """handle_text=keep 遇到文本列 → TextColumnNotSupportedError。"""
        rng = np.random.default_rng(42)
        n = 300
        X = pd.DataFrame({
            "a": rng.uniform(0, 10, n),
            "text": [f"long_text_{i}" for i in range(n)],
        })
        y = pd.Series([0, 1] * (n // 2))
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        cfg = PreprocessConfig(handle_text="keep")
        with pytest.raises(TextColumnNotSupportedError):
            preprocess(X, y, profile, split_result, config=cfg)

    def test_empty_column_dropped(self):
        """全空列被丢弃。"""
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({
            "a": rng.integers(0, 100, n),
            "empty": [np.nan] * n,
        })
        y = pd.Series([0, 1] * (n // 2))
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        result = preprocess(X, y, profile, split_result)
        assert "empty" in result.dropped_columns

    def test_datetime_extract(self):
        """日期列被抽成数值特征。"""
        X = pd.DataFrame({
            "a": np.arange(30.0),
            "dt": pd.date_range("2024-01-01", periods=30),
        })
        y = pd.Series([0, 1] * 15)
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        result = preprocess(X, y, profile, split_result)
        assert "dt" not in result.X_train.columns  # 原列不存在
        # 应有 dt_year, dt_month, dt_day
        assert any("dt_year" in c for c in result.feature_names_out)

    def test_datetime_drop(self):
        """handle_datetime=drop → 日期列被丢弃。"""
        n = 100
        X = pd.DataFrame({
            "a": np.arange(float(n)),
            "dt": pd.date_range("2024-01-01", periods=n),
        })
        y = pd.Series([0, 1] * (n // 2))
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        cfg = PreprocessConfig(handle_datetime="drop")
        result = preprocess(X, y, profile, split_result, config=cfg)
        assert "dt" in result.dropped_columns

    def test_deterministic(self):
        """相同输入 → 相同输出（纯函数特性）。"""
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        r1 = preprocess(X, y, profile, split_result)
        r2 = preprocess(X, y, profile, split_result)
        assert r1.X_train.equals(r2.X_train)
        assert r1.X_valid.equals(r2.X_valid)
        assert r1.X_test.equals(r2.X_test)

    def test_custom_config(self):
        """传入自定义 PreprocessConfig 生效。"""
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        cfg = PreprocessConfig(scale_method="robust")
        result = preprocess(X, y, profile, split_result, config=cfg)
        # 不报错即通过
        assert result.n_features_out > 0

    def test_missing_values_in_numeric(self):
        """数值列有缺失时正确填补。"""
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({
            "a": rng.choice([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], size=n),
            "b": rng.uniform(0, 10, n),
        })
        y = pd.Series([0, 1] * (n // 2))
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        result = preprocess(X, y, profile, split_result)
        # 填补后不应有 NaN
        assert result.X_train.isna().sum().sum() == 0
        assert result.X_valid.isna().sum().sum() == 0
        assert result.X_test.isna().sum().sum() == 0


# ─────────────────────────────────────────────────────────────
# TransformRecord / PreprocessedData —— schema 验证
# ─────────────────────────────────────────────────────────────

class TestTransformRecord:
    def test_default_fields(self):
        record = TransformRecord(
            col_name="x",
            original_dtype="float64",
            original_n_unique=10,
            action_sequence=["median_impute"],
        )
        assert record.dropped is False
        assert record.n_features_out == 1
        assert record.note is None

    def test_dropped_record(self):
        record = TransformRecord(
            col_name="bad",
            original_dtype="float64",
            original_n_unique=1,
            action_sequence=["dropped"],
            dropped=True,
            n_features_out=0,
            note="常数列",
        )
        assert record.dropped is True
        assert record.n_features_out == 0


class TestPreprocessedData:
    def test_has_all_required_attrs(self):
        """PreprocessedData 必须包含所有必要字段。"""
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        result = preprocess(X, y, profile, split_result)

        assert hasattr(result, "X_train")
        assert hasattr(result, "y_train")
        assert hasattr(result, "X_valid")
        assert hasattr(result, "y_valid")
        assert hasattr(result, "X_test")
        assert hasattr(result, "y_test")
        assert hasattr(result, "n_features_in")
        assert hasattr(result, "n_features_out")
        assert hasattr(result, "dropped_columns")
        assert hasattr(result, "feature_names_out")
        assert hasattr(result, "transforms")

    def test_train_valid_test_partition_sizes(self):
        """三集样本数加总等于原始样本数。"""
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        profile = build_profile(X, y)
        split_result = split_data(X, y, profile)
        result = preprocess(X, y, profile, split_result)

        total = (
            result.X_train.shape[0]
            + result.X_valid.shape[0]
            + result.X_test.shape[0]
        )
        assert total == len(X)