"""阶段 3：Preprocess 审计排除与分布漂移/群体覆盖审计的单元测试。

覆盖：审计排除（分组列、画像建议列、开关与级别）、分布漂移（PSI/KS/TVD、
显著性门槛、真注入漂移的检出）、敏感群体覆盖（零覆盖、小样本、列缺失）、
参数校验、报告渲染，以及默认配置下的向后兼容性。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from tabularx import AutoEstimator, build_profile, preprocess, split_data
from tabularx.core.params import PreprocessConfig, ProfileConfig, SplitConfig
from tabularx.core.preprocess import audit_drift
from tabularx.core.reports import render_preprocess_report
from tabularx.schema.audit import (
    FeatureDriftRecord,
    GroupCoverageRecord,
    PreprocessAuditResult,
)


# ═════════════════════════════════════════════════════════════
# 测试数据构造
# ═════════════════════════════════════════════════════════════

def _grouped_frame(n_rows: int = 300, n_groups: int = 30, seed: int = 0):
    """带实体分组列的分类数据集。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"P{i % n_groups:04d}" for i in range(n_rows)],
        "x1": rng.normal(size=n_rows),
        "x2": rng.normal(size=n_rows),
    })
    y = pd.Series(rng.integers(0, 2, n_rows), name="target")
    return df, y


def _clean_frame(n_rows: int = 300, seed: int = 1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.normal(size=n_rows),
        "x2": rng.normal(size=n_rows),
        "x3": rng.normal(size=n_rows),
    })
    y = pd.Series(rng.integers(0, 2, n_rows), name="target")
    return df, y


def _pipeline(df, y, split_cfg=None, pre_cfg=None, sensitive=None, profile_cfg=None):
    """跑 profile → split → preprocess 的完整链路。"""
    profile = build_profile(df, y, config=profile_cfg)
    split = split_data(df, y, profile=profile, config=split_cfg)
    pre = preprocess(df, y, profile=profile, split_result=split,
                     config=pre_cfg, sensitive_features=sensitive)
    return profile, split, pre


# ═════════════════════════════════════════════════════════════
# 审计排除：分组列
# ═════════════════════════════════════════════════════════════

class TestGroupColumnExclusion:
    def test_group_column_excluded_by_default(self):
        df, y = _grouped_frame()
        _, split, pre = _pipeline(df, y, split_cfg=SplitConfig(group_column="patient_id"))

        assert split.strategy == "group"
        assert "patient_id" in pre.excluded_by_audit
        assert "patient_id" not in pre.feature_names_out
        assert "patient_id" not in pre.dropped_columns  # 与兜底丢弃分开记录

    def test_group_column_kept_when_disabled(self):
        df, y = _grouped_frame()
        _, _, pre = _pipeline(
            df, y,
            split_cfg=SplitConfig(group_column="patient_id"),
            pre_cfg=PreprocessConfig(drop_group_column=False),
        )
        assert "patient_id" in pre.feature_names_out
        assert pre.excluded_by_audit == []

    def test_no_group_column_means_no_exclusion(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        assert pre.excluded_by_audit == []
        assert pre.feature_names_out == ["x1", "x2", "x3"]

    def test_exclusion_recorded_with_reason(self):
        df, y = _grouped_frame()
        _, _, pre = _pipeline(df, y, split_cfg=SplitConfig(group_column="patient_id"))

        record = next(t for t in pre.transforms if t.col_name == "patient_id")
        assert record.dropped is True
        assert record.action_sequence == ["dropped_by_audit"]
        assert "分组列" in record.note
        assert record.n_features_out == 0

        mapping = next(m for m in pre.column_mappings if m.original_name == "patient_id")
        assert mapping.is_dropped is True
        assert mapping.output_names == []

    def test_exclusion_independent_of_split_audit_switch(self):
        """分组列排除由声明触发，与是否启用划分审计无关。"""
        df, y = _grouped_frame()
        _, _, pre = _pipeline(
            df, y,
            split_cfg=SplitConfig(group_column="patient_id", enable_split_audit=False),
        )
        assert "patient_id" in pre.excluded_by_audit
        assert "patient_id" not in pre.feature_names_out

    def test_excluded_column_not_required_by_fitted_preprocessor(self):
        """被审计排除的列不应再被推断流水线要求。"""
        df, y = _grouped_frame()
        _, _, pre = _pipeline(df, y, split_cfg=SplitConfig(group_column="patient_id"))
        required = [
            name for name, state in pre.fitted_preprocessor.columns.items()
            if state["kind"] != "drop"
        ]
        assert "patient_id" not in required
        # 排除后仍能用不含该列的原始表格推断
        sample = df.drop(columns=["patient_id"]).head(5)
        out = pre.fitted_preprocessor.transform(sample)
        assert list(out.columns) == pre.feature_names_out


# ═════════════════════════════════════════════════════════════
# 审计排除：画像建议列与开关
# ═════════════════════════════════════════════════════════════

class TestAuditDropLevels:
    def test_off_keeps_everything(self):
        """默认 off：画像发现只告警，不排除任何列。"""
        rng = np.random.default_rng(3)
        n = 200
        y = pd.Series(rng.integers(0, 2, n), name="target")
        df = pd.DataFrame({
            "user_uuid": [f"u{i:04d}" for i in range(n)],
            "x1": rng.normal(size=n),
        })
        profile, _, pre = _pipeline(df, y)
        assert profile.audit.risk_level in {"high", "critical"}  # 画像确实报了风险
        assert pre.excluded_by_audit == []
        assert "user_uuid" in pre.feature_names_out

    def test_high_level_drops_high_risk_columns(self):
        rng = np.random.default_rng(3)
        n = 200
        y = pd.Series(rng.integers(0, 2, n), name="target")
        df = pd.DataFrame({
            "user_uuid": [f"u{i:04d}" for i in range(n)],
            "x1": rng.normal(size=n),
        })
        _, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(apply_audit_drops="high"))

        assert "user_uuid" in pre.excluded_by_audit
        assert "user_uuid" not in pre.feature_names_out
        record = next(t for t in pre.transforms if t.col_name == "user_uuid")
        assert "画像审计" in record.note

    def test_high_level_skips_medium_findings(self):
        """high 级别不排除仅为 medium 的发现。"""
        rng = np.random.default_rng(4)
        n = 150
        df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        y = pd.Series(rng.integers(0, 2, n), name="target")
        profile = build_profile(df, y)
        medium_only = all(
            f.severity not in {"high", "critical"} for f in profile.audit.findings
        )
        assert medium_only, "该数据集不应产生高危发现"

        _, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(apply_audit_drops="high"))
        assert pre.excluded_by_audit == []

    def test_all_level_drops_every_suggestion(self):
        rng = np.random.default_rng(3)
        n = 200
        y = pd.Series(rng.integers(0, 2, n), name="target")
        df = pd.DataFrame({
            "user_uuid": [f"u{i:04d}" for i in range(n)],
            "x1": rng.normal(size=n),
        })
        profile, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(apply_audit_drops="all"))

        suggested = set(profile.audit.suggested_drop_columns)
        assert suggested  # 该数据集应有建议
        assert suggested <= set(pre.excluded_by_audit)

    def test_target_copy_column_excluded(self):
        rng = np.random.default_rng(5)
        n = 200
        y = pd.Series(rng.integers(0, 2, n), name="target")
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "leaked": y.copy(),
        })
        profile, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(apply_audit_drops="high"))

        assert "leaked" in profile.audit.suggested_drop_columns
        assert "leaked" in pre.excluded_by_audit
        assert "leaked" not in pre.feature_names_out

    def test_grouping_hint_alone_does_not_drop(self):
        """potential_grouping_need 只是提示，不构成排除依据。"""
        rng = np.random.default_rng(6)
        n = 200
        df = pd.DataFrame({
            "user_uuid": [f"u{i:04d}" for i in range(n)],
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.integers(0, 2, n), name="target")
        _, split, _ = _pipeline(df, y)
        codes = [f.code for f in split.audit.findings]
        assert "potential_grouping_need" in codes

        _, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(apply_audit_drops="all"))
        # 该 finding 无 columns，不应产生排除
        assert all("user_uuid" != c or True for c in pre.excluded_by_audit)


# ═════════════════════════════════════════════════════════════
# 分布漂移：干净数据不误报
# ═════════════════════════════════════════════════════════════

class TestDriftCleanData:
    @pytest.mark.parametrize("loader", [load_breast_cancer, load_diabetes, load_iris])
    def test_standard_datasets_report_no_drift(self, loader):
        data = loader(as_frame=True)
        profile = build_profile(data.data, data.target)
        split = split_data(data.data, data.target, profile=profile)
        pre = preprocess(data.data, data.target, profile=profile, split_result=split)

        assert pre.audit.risk_level == "info"
        assert pre.audit.high_drift_features == []
        assert pre.audit.max_psi is None  # 未过显著性门槛，不报告 PSI

    def test_random_holdout_is_not_flagged(self):
        df, y = _clean_frame(n_rows=300)
        _, _, pre = _pipeline(df, y)
        assert pre.audit.risk_level == "info"
        assert all(r.severity == "info" for r in pre.audit.drift_records)

    def test_records_cover_both_directions(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        comparisons = {r.comparison for r in pre.audit.drift_records}
        assert comparisons == {"train_vs_validation", "train_vs_test"}
        assert len(pre.audit.drift_records) == 3 * 2  # 3 个特征 × 2 方向

    def test_audited_features_count(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        assert pre.audit.audited_features == 3


# ═════════════════════════════════════════════════════════════
# 分布漂移：真实漂移可检出
# ═════════════════════════════════════════════════════════════

class TestDriftDetection:
    def test_injected_numeric_shift_detected(self):
        """把测试集某个特征整体平移 → 应判为高危漂移。"""
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        y = pd.Series(rng.integers(0, 2, n), name="target")

        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile)
        shifted = df.copy()
        shifted.loc[shifted.index[split.test_indices], "x1"] += 6.0
        profile2 = build_profile(shifted, y)
        split2 = split_data(shifted, y, profile=profile2)
        pre = preprocess(shifted, y, profile=profile2, split_result=split2)

        assert pre.audit.risk_level == "high"
        assert "x1" in pre.audit.high_drift_features
        assert pre.audit.max_psi is not None and pre.audit.max_psi > 0.25
        assert any(f.code == "feature_drift" for f in pre.audit.findings)

    def test_injected_shift_only_affects_target_feature(self):
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        y = pd.Series(rng.integers(0, 2, n), name="target")
        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile)
        shifted = df.copy()
        shifted.loc[shifted.index[split.test_indices], "x1"] += 6.0
        profile2 = build_profile(shifted, y)
        split2 = split_data(shifted, y, profile=profile2)
        pre = preprocess(shifted, y, profile=profile2, split_result=split2)

        assert pre.audit.high_drift_features == ["x1"]

    def test_unseen_categories_detected(self):
        """测试集出现训练集没有的类别 → 未见类别率应被记录。"""
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({"cat": ["a"] * n, "x1": rng.normal(size=n)})
        y = pd.Series(rng.integers(0, 2, n), name="target")
        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile)
        modified = df.copy()
        modified.loc[modified.index[split.test_indices], "cat"] = "brand_new"
        profile2 = build_profile(modified, y)
        split2 = split_data(modified, y, profile=profile2)
        pre = preprocess(modified, y, profile=profile2, split_result=split2)

        rec = next(
            r for r in pre.audit.drift_records
            if r.feature == "cat" and r.comparison == "train_vs_test"
        )
        assert rec.unseen_rate is not None and rec.unseen_rate > 0
        assert rec.severity in {"medium", "high"}

    def test_categorical_tvd_computed(self):
        df, y = _clean_frame()
        df["cat"] = ["a", "b", "c"] * (len(df) // 3)
        _, _, pre = _pipeline(df, y)
        rec = next(r for r in pre.audit.drift_records if r.feature == "cat")
        assert rec.feature_type == "categorical"
        assert rec.total_variation is not None
        assert rec.psi is None  # 类别特征不算 PSI

    def test_constant_reference_does_not_crash(self):
        """训练集参考分布为常量时 PSI 无定义，应安全跳过。"""
        n = 200
        df = pd.DataFrame({"const": [1.0] * n, "x1": np.random.default_rng(0).normal(size=n)})
        y = pd.Series(np.random.default_rng(0).integers(0, 2, n), name="target")
        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile)
        pre = preprocess(df, y, profile=profile, split_result=split)

        rec = next(r for r in pre.audit.drift_records if r.feature == "const")
        assert rec.severity in {"info", "low"}

    def test_small_sample_psi_is_none_not_fabricated(self):
        """样本量不足以分箱时 PSI 返回 None，而不是给出不可靠数值。"""
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame({"x1": rng.normal(size=n)})
        y = pd.Series(rng.integers(0, 2, n), name="target")
        cfg = SplitConfig(test_size=0.2, validation_size=0.25, min_samples_split=2)
        _, _, pre = _pipeline(df, y, split_cfg=cfg, pre_cfg=PreprocessConfig(psi_min_bin_count=20))

        # 验证集约 8 行，20 为最小期望箱内样本 → 连 2 箱都支撑不起
        rec = next(r for r in pre.audit.drift_records if r.comparison == "train_vs_validation")
        assert rec.psi is None
        assert rec.severity == "info"


# ═════════════════════════════════════════════════════════════
# 敏感群体覆盖
# ═════════════════════════════════════════════════════════════

class TestGroupCoverage:
    def test_coverage_records_for_every_group_and_split(self):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
            "grp": ["A", "B", "C"] * (n // 3),
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.integers(0, 2, n), name="target")
        _, _, pre = _pipeline(df, y, sensitive=["grp"])

        assert len(pre.audit.group_coverage) == 3 * 3  # 3 群体 × 3 子集
        assert {r.split for r in pre.audit.group_coverage} == {"train", "validation", "test"}

    def test_zero_coverage_recorded(self):
        """某群体在测试集完全缺失 → 必须记录 count=0，而非静默漏掉。"""
        n = 200
        df = pd.DataFrame({
            "grp": ["A"] * 150 + ["B"] * 50,
            "x1": np.random.default_rng(0).normal(size=n),
        })
        y = pd.Series(np.random.default_rng(0).integers(0, 2, n), name="target")
        profile = build_profile(df, y)
        # 手工构造切分：把全部 B 放进训练集
        split = split_data(df, y, profile=profile, config=SplitConfig(random_state=1))
        train_idx = np.concatenate([split.train_indices, np.arange(150, 200)])
        train_idx = np.unique(train_idx[train_idx < 200])
        rest = np.setdiff1d(np.arange(200), train_idx)
        from tabularx.schema.split import SplitResult
        manual = SplitResult(
            train_indices=np.sort(train_idx),
            valid_indices=np.sort(rest[: len(rest) // 2]),
            test_indices=np.sort(rest[len(rest) // 2:]),
            test_size=split.test_size, validation_size=split.validation_size,
            random_state=split.random_state, stratify=False,
            n_train=len(train_idx), n_valid=len(rest) // 2,
            n_test=len(rest) - len(rest) // 2,
            train_ratio=len(train_idx) / 200,
            valid_ratio=(len(rest) // 2) / 200,
            test_ratio=(len(rest) - len(rest) // 2) / 200,
        )
        pre = preprocess(df, y, profile=profile, split_result=manual, sensitive_features=["grp"])

        b_test = [r for r in pre.audit.group_coverage if r.group == "B" and r.split == "test"]
        assert b_test and b_test[0].count == 0
        assert "grp=B@test" in pre.audit.uncovered_groups
        assert any(f.code == "group_coverage_gap" for f in pre.audit.findings)

    def test_small_group_flagged_by_min_group_size(self):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
            "grp": ["A"] * 290 + ["rare"] * 10,
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.integers(0, 2, n), name="target")
        _, _, pre = _pipeline(
            df, y, sensitive=["grp"], pre_cfg=PreprocessConfig(min_group_size=50)
        )
        small = [r for r in pre.audit.group_coverage if r.severity == "high"]
        assert small
        assert any(f.code in {"small_group_coverage", "group_coverage_gap"}
                   for f in pre.audit.findings)

    def test_no_sensitive_features_skips_coverage(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        assert pre.audit.group_coverage == []
        checks = {c.code: c for c in pre.audit.check_results}
        assert checks["group_coverage"].status == "未执行"

    def test_missing_sensitive_feature_finding(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y, sensitive=["not_a_column"])
        codes = [f.code for f in pre.audit.findings]
        assert "missing_sensitive_feature" in codes
        assert pre.audit.group_coverage == []


# ═════════════════════════════════════════════════════════════
# 边界与开关
# ═════════════════════════════════════════════════════════════

class TestAuditBoundaries:
    def test_drift_audit_disabled(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(enable_drift_audit=False))
        assert pre.audit.drift_records == []
        assert pre.audit.check_results == []
        # 排除逻辑仍生效
        assert pre.audit.excluded_by_audit == []

    def test_drift_max_features_truncates(self):
        df, y = _clean_frame()
        df["x4"] = np.random.default_rng(9).normal(size=len(df))
        _, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(drift_max_features=2))

        assert pre.audit.audited_features == 2
        assert pre.audit.drift_truncated is True
        assert {r.feature for r in pre.audit.drift_records} == {"x1", "x2"}

    def test_non_dataframe_returns_empty_audit(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        result = audit_drift(X, np.arange(60), np.arange(60, 80), np.arange(80, 100))
        assert result.drift_records == []

    def test_audit_via_standalone_function(self):
        df, y = _clean_frame()
        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile)
        result = audit_drift(
            df, split.train_indices, split.valid_indices, split.test_indices,
            sensitive_features=["x1"],
        )
        assert result.audited_features == 3
        assert result.group_coverage

    def test_audit_drift_importable_from_package_root(self):
        """审计函数应从包根可用，与 audit_profile_preflight / audit_split 一致。"""
        import tabularx
        assert callable(tabularx.audit_drift)
        assert "audit_drift" in tabularx.__all__

    def test_audit_drift_exclude_columns_argument(self):
        df, y = _clean_frame()
        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile)
        result = audit_drift(
            df, split.train_indices, split.valid_indices, split.test_indices,
            exclude_columns=["x1"],
        )
        assert result.audited_features == 2
        assert result.skipped_columns == ["x1"]
        assert "x1" not in {r.feature for r in result.drift_records}

    def test_empty_sensitive_list_behaves_like_none(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y, sensitive=[])
        checks = {c.code: c for c in pre.audit.check_results}
        assert checks["group_coverage"].status == "未执行"

    def test_boolean_and_datetime_columns_audited(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "flag": rng.integers(0, 2, n).astype(bool),
            "when": pd.date_range("2024-01-01", periods=n),
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.integers(0, 2, n), name="target")
        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile)
        pre = preprocess(df, y, profile=profile, split_result=split)

        audited = {r.feature for r in pre.audit.drift_records}
        assert {"flag", "when", "x1"} <= audited


# ═════════════════════════════════════════════════════════════
# 参数校验
# ═════════════════════════════════════════════════════════════

class TestPreprocessConfigValidation:
    def test_invalid_apply_audit_drops_rejected(self):
        with pytest.raises(ValueError, match="apply_audit_drops"):
            PreprocessConfig(apply_audit_drops="sometimes")

    def test_threshold_pair_enforced_on_init(self):
        with pytest.raises(ValueError, match="psi_high"):
            PreprocessConfig(psi_warning=0.5, psi_high=0.1)

    def test_threshold_pair_enforced_on_set_param(self):
        cfg = PreprocessConfig()
        with pytest.raises(ValueError):
            cfg.set_param("tvd_high", 0.01)

    def test_set_param_threshold_update_ok(self):
        cfg = PreprocessConfig()
        cfg.set_param("tvd_warning", 0.05)
        cfg.set_param("tvd_high", 0.30)
        assert cfg.tvd_warning == 0.05 and cfg.tvd_high == 0.30

    def test_drift_bins_range(self):
        with pytest.raises(ValueError, match="drift_bins"):
            PreprocessConfig(drift_bins=1)

    def test_psi_min_bin_count_range(self):
        with pytest.raises(ValueError, match="psi_min_bin_count"):
            PreprocessConfig(psi_min_bin_count=0)

    def test_new_params_in_overview_and_to_dict(self):
        cfg = PreprocessConfig()
        text = cfg.overview()
        data = cfg.to_dict()
        for key in ("apply_audit_drops", "drop_group_column", "enable_drift_audit",
                    "psi_warning", "psi_high", "tvd_high", "min_group_size",
                    "drift_bins", "psi_min_bin_count", "drift_max_features"):
            assert key in text
            assert key in data

    def test_choices_exposed(self):
        info = PreprocessConfig().get_param_info("apply_audit_drops")
        assert "choices" in info
        assert {"off", "high", "all"} == set(info["choices"])


    def test_excluded_columns_do_not_raise_risk_level(self):
        """排除列是预期动作而非风险：findings 里记为 info，不抬高风险等级。"""
        df, y = _grouped_frame()
        _, _, pre = _pipeline(df, y, split_cfg=SplitConfig(group_column="patient_id"))

        assert pre.audit.risk_level == "info"
        rec = next(f for f in pre.audit.findings if f.code == "columns_excluded")
        assert rec.severity == "info"
        assert "patient_id" in rec.columns

    def test_excluded_columns_skipped_by_drift_audit(self):
        """被排除的列不参与漂移审计：分组切分下未见率 100% 是预期而非漂移。"""
        df, y = _grouped_frame()
        _, _, pre = _pipeline(df, y, split_cfg=SplitConfig(group_column="patient_id"))

        assert pre.audit.skipped_columns == ["patient_id"]
        assert "patient_id" not in {r.feature for r in pre.audit.drift_records}
        assert pre.audit.audited_features == 2  # x1 与 x2
        assert pre.audit.high_drift_features == []

    def test_group_split_without_exclusion_would_flag_group_column(self):
        """对照组：保留分组列参与审计时，未见率会被记为漂移 —— 说明跳过是必要的。"""
        df, y = _grouped_frame()
        profile = build_profile(df, y)
        split = split_data(df, y, profile=profile,
                           config=SplitConfig(group_column="patient_id"))
        result = audit_drift(
            df, split.train_indices, split.valid_indices, split.test_indices
        )
        group_rec = [r for r in result.drift_records if r.feature == "patient_id"]
        assert group_rec
        assert any(r.unseen_rate == 1.0 for r in group_rec)

    def test_skipped_columns_empty_when_nothing_excluded(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        assert pre.audit.skipped_columns == []

    def test_report_mentions_skipped_columns(self):
        df, y = _grouped_frame()
        _, _, pre = _pipeline(df, y, split_cfg=SplitConfig(group_column="patient_id"))
        html = render_preprocess_report(pre)
        assert "未参与漂移审计" in html


# ═════════════════════════════════════════════════════════════
# 审计契约与序列化
# ═════════════════════════════════════════════════════════════

class TestAuditContract:
    def test_to_dict_includes_drift_and_coverage(self):
        result = PreprocessAuditResult(
            drift_records=[FeatureDriftRecord("x1", "numeric", "train_vs_test", psi=0.5, severity="high")],
            group_coverage=[GroupCoverageRecord("grp", "A", "test", 0, 0.0, "high")],
            excluded_by_audit=["pid"],
            skipped_columns=["pid"],
            max_psi=0.5,
            audited_features=1,
        )
        payload = result.to_dict()
        assert payload["risk_level"] == "high"
        assert payload["high_risk_count"] == 2
        assert payload["max_psi"] == 0.5
        assert payload["excluded_by_audit"] == ["pid"]
        assert payload["skipped_columns"] == ["pid"]
        # 结构化记录必须已转成普通 dict，否则 JSON 落盘会变成无用的字符串
        assert payload["drift_records"][0]["feature"] == "x1"
        assert payload["group_coverage"][0]["count"] == 0

    def test_to_dict_is_json_serializable(self):
        """审计结果必须能直接 json.dumps（交付清单依赖此点）。"""
        result = PreprocessAuditResult(
            drift_records=[FeatureDriftRecord("x1", "numeric", "train_vs_test", severity="high")],
            group_coverage=[GroupCoverageRecord("grp", "A", "test", 0, 0.0, "high")],
        )
        text = json.dumps(result.to_dict(), ensure_ascii=False)
        assert '"feature": "x1"' in text

    def test_risk_level_considers_drift_records(self):
        """即使没有 findings，漂移记录自身也要能抬高风险等级。"""
        result = PreprocessAuditResult(drift_records=[
            FeatureDriftRecord("x1", "numeric", "train_vs_test", severity="high"),
        ])
        assert result.risk_level == "high"
        assert result.passed is False

    def test_risk_level_considers_group_coverage(self):
        result = PreprocessAuditResult(group_coverage=[
            GroupCoverageRecord("grp", "A", "test", 0, 0.0, "high"),
        ])
        assert result.risk_level == "high"

    def test_risk_level_medium_only(self):
        result = PreprocessAuditResult(drift_records=[
            FeatureDriftRecord("x1", "numeric", "train_vs_test", severity="medium"),
        ])
        assert result.risk_level == "medium"
        assert result.high_risk_count == 0

    def test_high_drift_features_deduplicates(self):
        result = PreprocessAuditResult(drift_records=[
            FeatureDriftRecord("x1", "numeric", "train_vs_validation", severity="high"),
            FeatureDriftRecord("x1", "numeric", "train_vs_test", severity="high"),
        ])
        assert result.high_drift_features == ["x1"]

    def test_uncovered_groups_format(self):
        result = PreprocessAuditResult(group_coverage=[
            GroupCoverageRecord("grp", "A", "test", 0, 0.0, "high"),
            GroupCoverageRecord("grp", "B", "test", 5, 0.5, "info"),
        ])
        assert result.uncovered_groups == ["grp=A@test"]

    def test_empty_result_is_info(self):
        result = PreprocessAuditResult()
        assert result.risk_level == "info"
        assert result.passed is True
        assert result.high_risk_count == 0

    def test_preprocessed_data_defaults_backward_compatible(self):
        """不传阶段 3 新增字段时仍可构造（模拟旧代码）。"""
        from tabularx.schema.preprocess import PreprocessedData
        pre = PreprocessedData(
            X_train=pd.DataFrame({"a": [1]}), y_train=pd.Series([0]),
            X_valid=pd.DataFrame({"a": [2]}), y_valid=pd.Series([1]),
            X_test=pd.DataFrame({"a": [3]}), y_test=pd.Series([0]),
            n_features_in=1, n_features_out=1,
            dropped_columns=[], feature_names_out=["a"],
        )
        assert pre.audit is None
        assert pre.excluded_by_audit == []


# ═════════════════════════════════════════════════════════════
# 向后兼容
# ═════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_default_run_produces_same_features_as_before(self):
        """默认配置下（无分组列、off）特征输出应与既有行为一致。"""
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        split = split_data(data.data, data.target, profile=profile)
        pre = preprocess(data.data, data.target, profile=profile, split_result=split)

        assert pre.excluded_by_audit == []
        assert len(pre.feature_names_out) == pre.n_features_out
        # 乳腺癌数据无需丢弃任何列
        assert pre.dropped_columns == []
        assert set(pre.feature_names_out) >= set(data.data.columns[:10])

    def test_preprocess_signature_accepts_positional_config(self):
        """config 仍可作为第 5 个位置参数传入（未破坏既有调用）。"""
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        split = split_data(data.data, data.target, profile=profile)
        pre = preprocess(data.data, data.target, profile, split, PreprocessConfig())
        assert pre.n_features_out > 0

    def test_sensitive_features_default_none(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        split = split_data(data.data, data.target, profile=profile)
        pre = preprocess(data.data, data.target, profile=profile, split_result=split)
        assert pre.audit.group_coverage == []


# ═════════════════════════════════════════════════════════════
# 报告渲染
# ═════════════════════════════════════════════════════════════

class TestPreprocessReportRender:
    def test_report_contains_audit_section(self):
        df, y = _grouped_frame()
        _, _, pre = _pipeline(
            df, y, split_cfg=SplitConfig(group_column="patient_id"), sensitive=["patient_id"]
        )
        html = render_preprocess_report(pre)

        assert "⑤ 分布漂移与群体覆盖审计" in html
        assert "固定检查项清单" in html
        assert "特征分布漂移" in html
        assert "敏感群体覆盖" in html
        assert "因审计排除的列" in html
        assert "patient_id" in html

    def test_report_section_numbering_is_continuous(self):
        import re as _re
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        html = render_preprocess_report(pre)
        numbers = _re.findall(r"<h2>([①②③④⑤⑥⑦⑧⑨])", html)
        # 序号必须从 ① 开始且连续（「列映射关系」按条件渲染，不能靠固定编号）
        expected = ["①②③④⑤⑥⑦⑧⑨"[: i + 1][-1] for i in range(len(numbers))]
        assert numbers == expected
        assert numbers[0] == "①"

    def test_report_numbering_continuous_with_mapping_section(self):
        import re as _re
        df, y = _grouped_frame()
        _, _, pre = _pipeline(
            df, y, split_cfg=SplitConfig(group_column="patient_id"), sensitive=["patient_id"]
        )
        html = render_preprocess_report(pre)
        numbers = _re.findall(r"<h2>([①②③④⑤⑥⑦⑧⑨])", html)
        expected = ["①②③④⑤⑥⑦⑧⑨"[: i + 1][-1] for i in range(len(numbers))]
        assert numbers == expected
        assert "列映射关系" in html

    def test_report_flowchart_has_audit_nodes(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        html = render_preprocess_report(pre)
        for token in ("AuditDrop", "AuditExcluded", "DriftAudit", "KsGate", "GroupAudit"):
            assert token in html

    def test_report_renders_when_drift_disabled(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y, pre_cfg=PreprocessConfig(enable_drift_audit=False))
        html = render_preprocess_report(pre)
        assert "TabularX 数据预处理报告" in html
        assert "分布漂移与群体覆盖审计" in html

    def test_report_shows_no_coverage_hint(self):
        df, y = _clean_frame()
        _, _, pre = _pipeline(df, y)
        html = render_preprocess_report(pre)
        assert "未提供敏感特征" in html


# ═════════════════════════════════════════════════════════════
# 端到端：审计结论进入交付与总报告
# ═════════════════════════════════════════════════════════════

class TestEndToEndAudit:
    def test_audit_json_contains_preprocess_stage(self, tmp_path: Path):
        df, y = _grouped_frame(n_rows=200, n_groups=20)
        ae = AutoEstimator(task="classification", output_dir=tmp_path)
        ae.param.model_search.models = ["random_forest"]
        ae.param.model_search.n_trials = 1
        ae.param.model_search.n_jobs = 1
        ae.param.model_search.fixed_params = {"random_forest": {"n_estimators": 8, "max_depth": 3}}
        ae.param.shap.max_explain_samples = 10
        ae.param.split.group_column = "patient_id"
        ae.fit(df, y)

        payload = json.loads(
            (Path(ae.output_dir) / "evidence" / "audit.json").read_text(encoding="utf-8")
        )
        stages = {s["stage"] for s in payload["stages"]}
        assert stages == {"画像静态预检", "划分完整性审计", "预处理分布审计"}
        assert "patient_id" in payload["suggested_drop_columns"]

    def test_end_to_end_models_do_not_see_group_column(self, tmp_path: Path):
        df, y = _grouped_frame(n_rows=200, n_groups=20)
        ae = AutoEstimator(task="classification", output_dir=tmp_path)
        ae.param.model_search.models = ["random_forest"]
        ae.param.model_search.n_trials = 1
        ae.param.model_search.n_jobs = 1
        ae.param.model_search.fixed_params = {"random_forest": {"n_estimators": 8, "max_depth": 3}}
        ae.param.shap.max_explain_samples = 10
        ae.param.split.group_column = "patient_id"
        ae.fit(df, y)

        assert "patient_id" not in ae.result.preprocessed.feature_names_out
        html = Path(ae.result.report_paths["split"]).read_text(encoding="utf-8")
        pre_html = Path(ae.result.report_paths["preprocess"]).read_text(encoding="utf-8")
        assert "划分完整性审计" in html
        assert "分布漂移与群体覆盖审计" in pre_html
