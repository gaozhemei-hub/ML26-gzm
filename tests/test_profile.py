"""core/profile.py 与 schema/profile.py 的单元测试。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from tabularx import (
    ProfileUtils,
    build_column_fact,
    build_profile,
    build_target_fact,
)
from tabularx.schema.profile import ColumnFact, DataProfile, TargetFact


# ─────────────────────────────────────────────────────────────
# ProfileUtils.infer_column_type —— 列类型推断
# ─────────────────────────────────────────────────────────────

class TestInferColumnType:
    def test_numeric_float(self):
        col = pd.Series([1.5, 2.5, 3.5, 4.5], name="x")
        assert ProfileUtils.infer_column_type(col) == "numeric"

    def test_numeric_int_many_unique(self):
        # 整数但唯一值多（>20）→ numeric
        col = pd.Series(range(100), name="x")
        assert ProfileUtils.infer_column_type(col) == "numeric"

    def test_categorical_int_few_unique(self):
        # 整数但唯一值少（≤20）→ categorical
        col = pd.Series([0, 1, 2, 1, 0, 2] * 10, name="x")
        assert ProfileUtils.infer_column_type(col) == "categorical"

    def test_categorical_object(self):
        col = pd.Series(["a", "b", "c", "a", "b"], name="x")
        assert ProfileUtils.infer_column_type(col) == "categorical"

    def test_text_high_cardinality(self):
        # object 列，唯一值 >200 且 >50% 行唯一 → text
        n = 500
        col = pd.Series([f"id_{i}" for i in range(n)], name="x")
        assert ProfileUtils.infer_column_type(col) == "text"

    def test_boolean(self):
        col = pd.Series([True, False, True, False], name="x")
        assert ProfileUtils.infer_column_type(col) == "boolean"

    def test_datetime(self):
        col = pd.Series(pd.date_range("2024-01-01", periods=10), name="x")
        assert ProfileUtils.infer_column_type(col) == "datetime"

    def test_categorical_dtype(self):
        col = pd.Series(pd.Categorical(["a", "b", "a", "c"]), name="x")
        assert ProfileUtils.infer_column_type(col) == "categorical"

    def test_empty_column(self):
        col = pd.Series([np.nan] * 10, name="x")
        # 全空列：clean 为空，unique 为 0，按 object 处理 → categorical
        assert ProfileUtils.infer_column_type(col) == "categorical"


# ─────────────────────────────────────────────────────────────
# ProfileUtils.is_scale_needed —— scale 判据（MLJAR 式）
# ─────────────────────────────────────────────────────────────

class TestIsScaleNeeded:
    def test_small_range_no_scale(self):
        col = pd.Series([0.1, 0.2, 0.3, 0.4])
        assert ProfileUtils.is_scale_needed(col) is False

    def test_large_mean_scale(self):
        # |mean|>0.5 → True
        col = pd.Series([10.0, 11.0, 12.0, 13.0])
        assert ProfileUtils.is_scale_needed(col) is True

    def test_large_std_scale(self):
        # std>1.5 → True
        col = pd.Series([100.0, 200.0, 300.0, 400.0])
        assert ProfileUtils.is_scale_needed(col) is True

    def test_01_binary_no_scale(self):
        # 0/1 值：|mean|=0.5, std=0.5 → False
        col = pd.Series([0, 1, 0, 1, 0, 1])
        assert ProfileUtils.is_scale_needed(col) is False

    def test_empty_column(self):
        col = pd.Series([np.nan] * 5)
        assert ProfileUtils.is_scale_needed(col) is False

    def test_matches_mljar_threshold(self):
        # MLJAR: |mean|>0.5 or std>1.5
        # mean=0.4, std=0.1 → False
        col = pd.Series([0.3, 0.4, 0.5, 0.4, 0.4, 0.4])
        assert ProfileUtils.is_scale_needed(col) is False
        # mean=0.6, std=0.1 → True
        col = pd.Series([0.5, 0.6, 0.7, 0.6, 0.6, 0.6])
        assert ProfileUtils.is_scale_needed(col) is True


# ─────────────────────────────────────────────────────────────
# ProfileUtils.infer_task_type —— 任务类型推断
# ─────────────────────────────────────────────────────────────

class TestInferTaskType:
    def test_binary_int(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        assert ProfileUtils.infer_task_type(y) == "binary"

    def test_binary_str(self):
        y = pd.Series(["cat", "dog", "cat", "dog"])
        assert ProfileUtils.infer_task_type(y) == "binary"

    def test_binary_float_values(self):
        # 浮点但只有 2 个唯一值 → binary（type_of_target 直接判 binary）
        y = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0])
        assert ProfileUtils.infer_task_type(y) == "binary"

    def test_multiclass_int(self):
        y = pd.Series([0, 1, 2, 1, 0, 2])
        assert ProfileUtils.infer_task_type(y) == "multiclass"

    def test_multiclass_str(self):
        y = pd.Series(["a", "b", "c", "a", "b", "c"])
        assert ProfileUtils.infer_task_type(y) == "multiclass"

    def test_multiclass_float_few_unique(self):
        # 浮点但唯一值少（≤20）且 >2 个 → 尊重 type_of_target 的 multiclass
        y = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        assert ProfileUtils.infer_task_type(y) == "multiclass"

    def test_regression_float_many_unique(self):
        # 浮点连续值（唯一值 >20）→ regression
        y = pd.Series([float(i) for i in range(100)])
        assert ProfileUtils.infer_task_type(y) == "regression"

    def test_regression_int_many_unique(self):
        # 整数但唯一值多（>20）→ 仍视为 multiclass（type_of_target 判断）
        y = pd.Series(list(range(100)))
        # 目前行为：整数唯一值多 → multiclass（尊重 sklearn 语义）
        assert ProfileUtils.infer_task_type(y) == "multiclass"

    def test_missing_values_in_y(self):
        # 带缺失值的 y
        y = pd.Series([0, 1, np.nan, 0, 1, 0])
        assert ProfileUtils.infer_task_type(y) == "binary"

    def test_empty_y_raises(self):
        with pytest.raises(ValueError):
            ProfileUtils.infer_task_type(pd.Series([], dtype=float))

    def test_standard_datasets(self):
        # sklearn 标准数据集
        data = load_breast_cancer(as_frame=True)
        assert ProfileUtils.infer_task_type(data.target) == "binary"

        data = load_iris(as_frame=True)
        assert ProfileUtils.infer_task_type(data.target) == "multiclass"

        data = load_diabetes(as_frame=True)
        assert ProfileUtils.infer_task_type(data.target) == "regression"


# ─────────────────────────────────────────────────────────────
# ProfileUtils.compute_basic_stats —— 基本统计量
# ─────────────────────────────────────────────────────────────

class TestComputeBasicStats:
    def test_numeric_stats(self):
        col = pd.Series([1.0, 2.0, 3.0, 4.0])
        stats = ProfileUtils.compute_basic_stats(col)
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(4.0)

    def test_nan_handling(self):
        col = pd.Series([1.0, np.nan, 3.0, np.nan])
        stats = ProfileUtils.compute_basic_stats(col)
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["n_missing"] if "n_missing" in stats else True

    def test_empty_returns_none(self):
        col = pd.Series([np.nan] * 5)
        stats = ProfileUtils.compute_basic_stats(col)
        assert stats["mean"] is None
        assert stats["std"] is None


# ─────────────────────────────────────────────────────────────
# ProfileUtils.compute_fingerprint —— 数据指纹
# ─────────────────────────────────────────────────────────────

class TestComputeFingerprint:
    def test_same_data_same_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df2 = df1.copy()
        y1 = pd.Series([0, 1, 0])
        y2 = y1.copy()
        assert ProfileUtils.compute_fingerprint(df1, y1) == ProfileUtils.compute_fingerprint(df2, y2)

    def test_diff_data_diff_hash(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        y = pd.Series([0, 1, 0])
        assert ProfileUtils.compute_fingerprint(df1, y) != ProfileUtils.compute_fingerprint(df2, y)

    def test_hash_format(self):
        df = pd.DataFrame({"a": [1, 2]})
        y = pd.Series([0, 1])
        fp = ProfileUtils.compute_fingerprint(df, y)
        assert isinstance(fp, str)
        assert len(fp) == 16  # 前 16 位 hex


# ─────────────────────────────────────────────────────────────
# build_column_fact —— 单列事实构建
# ─────────────────────────────────────────────────────────────

class TestBuildColumnFact:
    def test_numeric_col(self):
        col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="feat")
        fact = build_column_fact(col)
        assert isinstance(fact, ColumnFact)
        assert fact.col_name == "feat"
        assert fact.dtype == "float64"
        assert fact.n_missing == 0
        assert fact.missing_rate == 0.0
        assert fact.n_unique == 5
        assert "numeric" in fact.tags
        assert "scale_needed" in fact.tags  # |mean|=3.0>0.5 → scale
        assert fact.mean == pytest.approx(3.0)

    def test_categorical_col(self):
        col = pd.Series(["a", "b", "a", "c", "b"], name="cat")
        fact = build_column_fact(col)
        assert "categorical" in fact.tags
        assert fact.mean is None  # 非数值列无统计
        assert fact.n_unique == 3

    def test_missing_values_tag(self):
        col = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name="x")
        fact = build_column_fact(col)
        assert "has_missing" in fact.tags
        assert fact.n_missing == 2
        assert fact.missing_rate == pytest.approx(0.4)

    def test_missing_rate_zero(self):
        col = pd.Series([1.0, 2.0], name="x")
        fact = build_column_fact(col)
        assert fact.missing_rate == 0.0


# ─────────────────────────────────────────────────────────────
# build_target_fact —— 目标事实构建
# ─────────────────────────────────────────────────────────────

class TestBuildTargetFact:
    def test_binary_target(self):
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        fact = build_target_fact(y)
        assert isinstance(fact, TargetFact)
        assert fact.task_type == "binary"
        assert fact.n_classes == 2
        assert fact.class_balance is not None
        # 类别平衡（各 50%）
        assert abs(fact.class_balance["0"] - 0.5) < 1e-9
        assert abs(fact.class_balance["1"] - 0.5) < 1e-9

    def test_multiclass_target(self):
        y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])
        fact = build_target_fact(y)
        assert fact.task_type == "multiclass"
        assert fact.n_classes == 3
        assert set(fact.class_balance.keys()) == {"0", "1", "2"}

    def test_regression_target(self):
        y = pd.Series([float(i) for i in range(100)])
        fact = build_target_fact(y)
        assert fact.task_type == "regression"
        assert fact.n_classes is None
        assert fact.class_balance is None
        assert fact.mean is not None
        assert fact.std is not None

    def test_target_missing_rate(self):
        y = pd.Series([0, 1, np.nan, 0, np.nan, 1])
        fact = build_target_fact(y)
        assert fact.missing_rate == pytest.approx(2 / 6)


# ─────────────────────────────────────────────────────────────
# build_profile —— 顶层函数
# ─────────────────────────────────────────────────────────────

class TestBuildProfile:
    def test_breast_cancer(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        assert isinstance(profile, DataProfile)
        assert profile.n_rows == 569
        assert profile.n_cols == 30
        assert len(profile.columns) == 30
        assert profile.target.task_type == "binary"
        assert isinstance(profile.fingerprint, str)
        assert len(profile.fingerprint) == 16

    def test_iris(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        assert profile.n_rows == 150
        assert profile.n_cols == 4
        assert profile.target.task_type == "multiclass"
        assert profile.target.n_classes == 3

    def test_diabetes(self):
        data = load_diabetes(as_frame=True)
        profile = build_profile(data.data, data.target)
        assert profile.n_rows == 442
        assert profile.n_cols == 10
        assert profile.target.task_type == "regression"

    def test_all_columns_have_facts(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        col_names = [c.col_name for c in profile.columns]
        assert col_names == list(data.data.columns)

    def test_reproducible_fingerprint(self):
        data = load_breast_cancer(as_frame=True)
        p1 = build_profile(data.data, data.target)
        p2 = build_profile(data.data.copy(), data.target.copy())
        assert p1.fingerprint == p2.fingerprint

    def test_invalid_X_type(self):
        with pytest.raises(TypeError):
            build_profile(np.zeros((10, 3)), pd.Series([0] * 10))

    def test_invalid_y_type(self):
        with pytest.raises(TypeError):
            build_profile(pd.DataFrame({"a": [1] * 10}), np.zeros(10))

    def test_empty_dataframe(self):
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0, 1])
        profile = build_profile(X, y)
        assert profile.n_rows == 4
        assert profile.n_cols == 0
        assert profile.columns == []


# ─────────────────────────────────────────────────────────────
# 新增：常数列检测 + 手动 task_type + Evidence 证据收集
# ─────────────────────────────────────────────────────────────

class TestConstantColumn:
    def test_constant_column_tagged(self):
        """所有值相同的列标记为 constant_column。"""
        col = pd.Series([5, 5, 5, 5, 5], name="const")
        fact = build_column_fact(col)
        assert "constant_column" in fact.tags
        assert fact.n_unique == 1

    def test_constant_column_evidences(self):
        """常数列的证据记录了唯一值数。"""
        col = pd.Series(["x"] * 10, name="const")
        fact = build_column_fact(col)
        assert any(ev.metric == "n_unique" and ev.value == 1 for ev in fact.evidences)

    def test_constant_numeric_has_no_scale_tag(self):
        """常数列不再判断 scale。"""
        col = pd.Series([7.0] * 10, name="const")
        fact = build_column_fact(col)
        assert "scale_needed" not in fact.tags

    def test_empty_column_has_evidences(self):
        """全空列的证据记录了非缺失值数。"""
        col = pd.Series([np.nan] * 5, name="empty")
        fact = build_column_fact(col)
        assert "empty_column" in fact.tags
        assert any(ev.metric == "n_non_missing" and ev.value == 0 for ev in fact.evidences)


class TestManualTaskType:
    def test_manual_regression(self):
        """手动指定 regression 覆盖自动推断。"""
        # 这个数据自动推断会是 binary，但手动指定为 regression
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        fact = build_target_fact(y, task_type="regression")
        assert fact.task_type == "regression"

    def test_manual_classification_binary(self):
        """手动指定 classification，自动细分 binary。"""
        y = pd.Series([0, 1, 0, 1, 0, 1])
        fact = build_target_fact(y, task_type="classification")
        assert fact.task_type == "binary"

    def test_manual_classification_multiclass(self):
        """手动指定 classification，自动细分 multiclass。"""
        y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])
        fact = build_target_fact(y, task_type="classification")
        assert fact.task_type == "multiclass"

    def test_invalid_task_type_raises(self):
        """非法 task_type 抛 ValueError。"""
        y = pd.Series([0, 1, 0, 1])
        with pytest.raises(ValueError):
            build_target_fact(y, task_type="invalid")

    def test_manual_task_type_in_profile(self):
        """build_profile 支持手动传入 task_type。"""
        data = load_iris(as_frame=True)
        # iris 自动推断是 multiclass，手动强制为 regression 验证覆盖
        profile = build_profile(data.data, data.target, task_type="regression")
        assert profile.target.task_type == "regression"


class TestEvidenceCollection:
    def test_numeric_col_has_scale_evidence(self):
        """数值列的 scale 证据存在。"""
        col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="feat")
        fact = build_column_fact(col)
        assert any(ev.metric == "scale_rule" for ev in fact.evidences)
        # 证据里有 mean/std 数值
        scale_ev = [ev for ev in fact.evidences if ev.metric == "scale_rule"][0]
        assert "mean" in scale_ev.note and "std" in scale_ev.note

    def test_target_has_task_type_evidence(self):
        """目标的 task_type 证据存在。"""
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        fact = build_target_fact(y)
        assert any(ev.metric == "task_type_source" for ev in fact.evidences)

    def test_missing_col_has_missing_evidence(self):
        """含缺失值的列有缺失证据。"""
        col = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name="x")
        fact = build_column_fact(col)
        assert any(ev.metric == "n_missing" and ev.value == 2 for ev in fact.evidences)


class TestReportThreeLayerStructure:
    """报告三层结构：摘要 / 判断流程与证据 / 详细统计。"""

    def test_summary_section(self):
        """第 1 部分摘要存在，含结论。"""
        from tabularx.reports import render_profile_report

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        html = render_profile_report(profile)
        assert "① 摘要" in html
        assert "关键发现" in html
        assert "判定为" in html

    def test_evidence_section(self):
        """第 2 部分判断流程与证据存在，含规则文本。"""
        from tabularx.reports import render_profile_report

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        html = render_profile_report(profile)
        assert "② 判断流程与证据" in html
        assert "2.2 目标变量判断依据" in html
        assert "规则" in html

    def test_detailed_stats_section(self):
        """第 3 部分详细统计存在，含特征列表格。"""
        from tabularx.reports import render_profile_report

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        html = render_profile_report(profile)
        assert "③ 详细描述性统计" in html
        assert "特征列" in html
        assert "<table>" in html

    def test_mermaid_flowchart_in_report(self):
        """报告含判断流程图（Mermaid）。"""
        from tabularx.reports import render_profile_report

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        html = render_profile_report(profile)
        assert "mermaid" in html
        assert "flowchart" in html


# ─────────────────────────────────────────────────────────────
# reports —— 报告渲染（jinja2 模板 → HTML 字符串）
# ─────────────────────────────────────────────────────────────

class TestRenderReport:
    """测试 reports.py 的模板渲染功能（纯函数，无 IO）。"""

    def test_render_profile_html(self):
        """渲染 DataProfile → HTML，应包含关键信息。"""
        from tabularx.reports import ReportType, render_report

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        html = render_report(ReportType.PROFILE, profile)

        # 基础结构
        assert html.startswith("<!DOCTYPE html>")
        assert "TabularX 数据概况报告" in html
        # 关键数据出现在 HTML 中
        assert "569" in html                       # n_rows
        assert "30" in html                        # n_cols
        assert "binary" in html                    # task_type
        assert "mean radius" in html               # 第一列列名
        assert profile.fingerprint in html         # 数据指纹
        # 表头结构
        assert "<table>" in html
        assert "列名" in html

    def test_render_profile_returns_str(self):
        """返回值必须是字符串（供上层写文件）。"""
        from tabularx.reports import render_profile_report

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        html = render_profile_report(profile)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_render_deterministic(self):
        """相同输入 → 相同输出（纯函数特性）。"""
        from tabularx.reports import render_report, ReportType

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        h1 = render_report(ReportType.PROFILE, profile)
        h2 = render_report(ReportType.PROFILE, profile)
        assert h1 == h2

    def test_render_regression_profile(self):
        """回归任务的 profile 也能正常渲染（无 n_classes/class_balance 的分支）。"""
        from tabularx.reports import render_profile_report

        data = load_diabetes(as_frame=True)
        profile = build_profile(data.data, data.target)
        html = render_profile_report(profile)
        assert "regression" in html
        # 回归分支显示均值/标准差，不显示类别数/类别分布
        assert "n_classes" not in html
        assert "class_balance" not in html

    def test_render_injected_template_dir(self):
        """支持注入自定义模板目录（可测性：不依赖包内模板）。"""
        import tempfile
        from pathlib import Path

        from tabularx.reports import render_template

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_template = Path(tmpdir) / "custom.html.jinja2"
            custom_template.write_text(
                "<p>n_rows={{ data.n_rows }}</p>", encoding="utf-8"
            )
            data = load_breast_cancer(as_frame=True)
            profile = build_profile(data.data, data.target)
            # 用底层 render_template + 注入目录
            html = render_template("custom.html.jinja2", profile, template_dir=tmpdir)
            assert "<p>n_rows=569</p>" in html

    def test_template_not_found_raises(self):
        """模板不存在时抛 TemplateNotFound。"""
        import tempfile

        from tabularx.reports import render_template

        with tempfile.TemporaryDirectory() as tmpdir:
            data = load_breast_cancer(as_frame=True)
            profile = build_profile(data.data, data.target)
            with pytest.raises(Exception):
                render_template("missing.html.jinja2", profile, template_dir=tmpdir)

    def test_autoescape_prevents_html_injection(self):
        """autoescape 开启：数据中的 HTML 不被执行。"""
        import tempfile
        from pathlib import Path

        from tabularx.reports import render_template

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_template = Path(tmpdir) / "x.html.jinja2"
            custom_template.write_text(
                "<p>{{ data.name }}</p>", encoding="utf-8"
            )
            # 模拟恶意列名
            html = render_template(
                "x.html.jinja2",
                {"name": "<script>alert('xss')</script>"},
                template_dir=tmpdir,
            )
            assert "<script>alert('xss')</script>" not in html
            assert "&lt;script&gt;" in html

    def test_profile_report_end_to_end(self):
        """端到端：build_profile → render_profile_report，全流程无异常。"""
        from tabularx.reports import render_profile_report

        for loader in (load_breast_cancer, load_iris, load_diabetes):
            data = loader(as_frame=True)
            profile = build_profile(data.data, data.target)
            html = render_profile_report(profile)
            assert isinstance(html, str)
            assert len(html) > 0
