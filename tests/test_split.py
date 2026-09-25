"""core/split.py 与 schema/split.py 的单元测试。

覆盖：基础切分行为、分层与随机策略、group/temporal 结构感知切分、
划分完整性审计（跨集合重复 / 实体泄漏 / 时间穿越）的命中与未命中、
参数校验、报告渲染，以及不传新参数时的向后兼容性。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from tabularx import (
    audit_split,
    build_profile,
    resolve_split_strategy,
    split_data,
)
from tabularx.core.params import SplitConfig
from tabularx.exceptions import (
    InvalidSplitError,
    SplitColumnMissingError,
    StratifyError,
    TimeParseError,
)
from tabularx.schema.audit import (
    AuditCheckResult,
    AuditFinding,
    SplitAuditResult,
)
from tabularx.schema.split import SplitResult


# ═════════════════════════════════════════════════════════════
# 测试数据构造
# ═════════════════════════════════════════════════════════════

def _clf_frame(n_rows: int = 200, n_groups: int = 20, seed: int = 7):
    """构造带实体列的分类数据集。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"P{i % n_groups:04d}" for i in range(n_rows)],
        "x1": rng.normal(size=n_rows),
        "x2": rng.normal(size=n_rows),
    })
    y = pd.Series(rng.integers(0, 2, n_rows), name="target")
    return df, y


def _temporal_frame(n_rows: int = 200, seed: int = 11):
    """构造带时间列的回归数据集。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "event_time": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
        "x1": rng.normal(size=n_rows),
    })
    y = pd.Series(rng.normal(size=n_rows), name="target")
    return df, y


# ═════════════════════════════════════════════════════════════
# 基础切分行为
# ═════════════════════════════════════════════════════════════

class TestBasicSplit:
    def test_ratios_and_sizes(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        cfg = SplitConfig(test_size=0.2, validation_size=0.25)
        result = split_data(data.data, data.target, profile=profile, config=cfg)

        assert result.n_total == len(data.data)
        assert result.n_train + result.n_valid + result.n_test == len(data.data)
        assert result.test_ratio == pytest.approx(result.n_test / result.n_total)
        assert result.valid_ratio == pytest.approx(result.n_valid / result.n_total)
        assert result.train_ratio == pytest.approx(1 - result.test_ratio - result.valid_ratio)

    def test_indices_disjoint_and_complete(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        train = set(result.train_indices.tolist())
        valid = set(result.valid_indices.tolist())
        test = set(result.test_indices.tolist())

        assert not (train & valid)
        assert not (train & test)
        assert not (valid & test)
        assert train | valid | test == set(range(len(data.data)))

    def test_indices_sorted_ascending(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        for idx in (result.train_indices, result.valid_indices, result.test_indices):
            assert np.all(np.diff(idx) > 0)

    def test_reproducible_with_same_seed(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        r1 = split_data(data.data, data.target, profile=profile, config=SplitConfig(random_state=123))
        r2 = split_data(data.data, data.target, profile=profile, config=SplitConfig(random_state=123))

        assert np.array_equal(r1.train_indices, r2.train_indices)
        assert np.array_equal(r1.valid_indices, r2.valid_indices)
        assert np.array_equal(r1.test_indices, r2.test_indices)

    def test_different_seed_changes_split(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        r1 = split_data(data.data, data.target, profile=profile, config=SplitConfig(random_state=1))
        r2 = split_data(data.data, data.target, profile=profile, config=SplitConfig(random_state=2))

        assert not np.array_equal(r1.test_indices, r2.test_indices)

    def test_accepts_numpy_arrays(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = pd.Series(rng.normal(size=120))
        # 用 DataFrame 建画像，但用 ndarray 切分
        profile = build_profile(pd.DataFrame(X, columns=["a", "b", "c"]), y)
        result = split_data(X, y, profile=profile)

        assert result.n_total == 120
        assert result.strategy == "random"

    def test_default_strategy_is_stratified_for_classification(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        assert result.strategy == "stratified"
        assert result.stratify is True

    def test_default_strategy_is_random_for_regression(self):
        data = load_diabetes(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        assert result.strategy == "random"
        assert result.stratify is False


class TestStratifyBehaviour:
    def test_class_distribution_close_after_stratify(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        overall = data.target.mean()
        for idx in (result.train_indices, result.valid_indices, result.test_indices):
            subset_mean = data.target.iloc[idx].mean()
            assert abs(subset_mean - overall) < 0.05

    def test_stratify_quality_evidence_recorded(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        metrics = {ev.metric for ev in result.evidences}
        assert "stratify_quality" in metrics
        assert "stratify_reason" in metrics
        assert "random_seed" in metrics
        assert "train_ratio" in metrics

    def test_explicit_random_strategy_disables_stratify(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        cfg = SplitConfig(split_strategy="random")
        result = split_data(data.data, data.target, profile=profile, config=cfg)

        assert result.strategy == "random"
        assert result.stratify is False
        # 未分层时仍应记录原因
        reason = next(ev for ev in result.evidences if ev.metric == "stratify_reason")
        assert reason.value == "random"


class TestSplitErrors:
    def test_subset_too_small_raises(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=12)})
        y = pd.Series(rng.normal(size=12))
        profile = build_profile(df, y)
        cfg = SplitConfig(test_size=0.4, validation_size=0.4, min_samples_split=10)

        with pytest.raises(InvalidSplitError):
            split_data(df, y, profile=profile, config=cfg)

    def test_stratify_failure_raises_stratify_error(self):
        # 先分出极大测试集，剩余样本每类只剩 1 个 → 第二阶段分层失败
        df = pd.DataFrame({"a": np.arange(20, dtype=float)})
        y = pd.Series([0] * 10 + [1] * 10)
        profile = build_profile(df, y)
        cfg = SplitConfig(test_size=0.9, validation_size=0.5, min_samples_split=1)

        with pytest.raises(StratifyError):
            split_data(df, y, profile=profile, config=cfg)

    def test_subset_below_minimum_after_stratify(self):
        # 分层可行但子集样本数低于下限 → InvalidSplitError
        df = pd.DataFrame({"a": np.arange(40, dtype=float)})
        y = pd.Series([0] * 20 + [1] * 20)
        profile = build_profile(df, y)
        cfg = SplitConfig(test_size=0.8, validation_size=0.5, min_samples_split=5)

        with pytest.raises(InvalidSplitError):
            split_data(df, y, profile=profile, config=cfg)


# ═════════════════════════════════════════════════════════════
# 策略解析
# ═════════════════════════════════════════════════════════════

class TestResolveSplitStrategy:
    def test_auto_classification_is_stratified(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        assert resolve_split_strategy(profile, SplitConfig()) == "stratified"

    def test_auto_regression_is_random(self):
        data = load_diabetes(as_frame=True)
        profile = build_profile(data.data, data.target)
        assert resolve_split_strategy(profile, SplitConfig()) == "random"

    def test_auto_with_group_column_is_group(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        cfg = SplitConfig(group_column="some_id")
        assert resolve_split_strategy(profile, cfg) == "group"

    def test_auto_with_time_column_is_temporal(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        cfg = SplitConfig(time_column="some_time")
        assert resolve_split_strategy(profile, cfg) == "temporal"

    def test_explicit_strategy_wins_over_declaration(self):
        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        cfg = SplitConfig(split_strategy="random", group_column="some_id")
        assert resolve_split_strategy(profile, cfg) == "random"


# ═════════════════════════════════════════════════════════════
# SplitConfig 参数校验
# ═════════════════════════════════════════════════════════════

class TestSplitConfigValidation:
    def test_group_strategy_requires_group_column(self):
        with pytest.raises(ValueError, match="group_column"):
            SplitConfig(split_strategy="group")

    def test_temporal_strategy_requires_time_column(self):
        with pytest.raises(ValueError, match="time_column"):
            SplitConfig(split_strategy="temporal")

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError, match="split_strategy"):
            SplitConfig(split_strategy="nonsense")

    def test_negative_duplicate_audit_max_rows_rejected(self):
        with pytest.raises(ValueError, match="duplicate_audit_max_rows"):
            SplitConfig(duplicate_audit_max_rows=0)

    def test_set_param_validates_choices(self):
        cfg = SplitConfig()
        with pytest.raises(ValueError):
            cfg.set_param("split_strategy", "unknown")

    def test_set_param_group_requires_column_first(self):
        cfg = SplitConfig()
        with pytest.raises(ValueError, match="group_column"):
            cfg.set_param("split_strategy", "group")
        cfg.set_param("group_column", "pid")
        cfg.set_param("split_strategy", "group")
        assert cfg.split_strategy == "group"

    def test_set_param_cannot_clear_column_in_use(self):
        cfg = SplitConfig(split_strategy="group", group_column="pid")
        with pytest.raises(ValueError):
            cfg.set_param("group_column", None)

    def test_temporal_set_param(self):
        cfg = SplitConfig()
        with pytest.raises(ValueError, match="time_column"):
            cfg.set_param("split_strategy", "temporal")
        cfg.set_param("time_column", "ts")
        cfg.set_param("split_strategy", "temporal")
        assert cfg.time_column == "ts"

    def test_choices_exposed_in_param_info(self):
        info = SplitConfig().get_param_info("split_strategy")
        assert "choices" in info
        assert "auto" in info["choices"]

    def test_new_params_appear_in_to_dict_and_overview(self):
        cfg = SplitConfig()
        data = cfg.to_dict()
        for key in ("split_strategy", "group_column", "time_column",
                    "enable_split_audit", "duplicate_audit_max_rows"):
            assert key in data
            assert key in cfg.overview()


# ═════════════════════════════════════════════════════════════
# 实体分组切分
# ═════════════════════════════════════════════════════════════

class TestGroupSplit:
    def test_entities_never_cross_splits(self):
        df, y = _clf_frame(n_rows=300, n_groups=30)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.strategy == "group"
        assert result.group_column == "patient_id"

        groups = df["patient_id"].astype("string")
        train_groups = set(groups.iloc[result.train_indices])
        valid_groups = set(groups.iloc[result.valid_indices])
        test_groups = set(groups.iloc[result.test_indices])

        assert not (train_groups & valid_groups)
        assert not (train_groups & test_groups)
        assert not (valid_groups & test_groups)

    def test_audit_passes_for_group_split(self):
        df, y = _clf_frame(n_rows=300, n_groups=30)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.audit is not None
        assert result.audit.group_overlap_count == 0
        assert result.audit.risk_level == "info"

    def test_auto_strategy_picks_group_when_declared(self):
        df, y = _clf_frame(n_rows=200, n_groups=20)
        profile = build_profile(df, y)
        cfg = SplitConfig(group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.strategy == "group"

    def test_group_split_records_strategy_evidence(self):
        df, y = _clf_frame(n_rows=200, n_groups=20)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        metrics = {ev.metric for ev in result.evidences}
        assert "split_strategy" in metrics
        assert "random_seed" in metrics

    def test_classification_group_split_drops_stratify_with_evidence(self):
        df, y = _clf_frame(n_rows=300, n_groups=30)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.stratify is False
        assert any(ev.metric == "stratify_tradeoff" for ev in result.evidences)

    def test_missing_group_column_raises(self):
        df, y = _clf_frame(n_rows=100, n_groups=10)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="not_a_column")

        with pytest.raises(SplitColumnMissingError):
            split_data(df, y, profile=profile, config=cfg)

    def test_non_dataframe_with_group_declaration_raises(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y = pd.Series(rng.integers(0, 2, 100))
        profile = build_profile(pd.DataFrame(X, columns=["a", "b", "c"]), y)
        cfg = SplitConfig(split_strategy="group", group_column="a")

        with pytest.raises(SplitColumnMissingError):
            split_data(X, y, profile=profile, config=cfg)

    def test_group_split_reproducible(self):
        df, y = _clf_frame(n_rows=300, n_groups=30)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id", random_state=5)
        r1 = split_data(df, y, profile=profile, config=cfg)
        r2 = split_data(df, y, profile=profile, config=cfg)

        assert np.array_equal(r1.train_indices, r2.train_indices)
        assert np.array_equal(r1.test_indices, r2.test_indices)


# ═════════════════════════════════════════════════════════════
# 时序切分
# ═════════════════════════════════════════════════════════════

class TestTemporalSplit:
    def test_splits_are_time_ordered(self):
        df, y = _temporal_frame(n_rows=200)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="temporal", time_column="event_time")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.strategy == "temporal"
        t = pd.to_datetime(df["event_time"])
        assert t.iloc[result.train_indices].max() <= t.iloc[result.valid_indices].min()
        assert t.iloc[result.valid_indices].max() <= t.iloc[result.test_indices].min()

    def test_audit_reports_ordered(self):
        df, y = _temporal_frame(n_rows=200)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="temporal", time_column="event_time")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.audit is not None
        assert result.audit.temporal_ordered is True
        assert result.audit.risk_level == "info"

    def test_auto_strategy_picks_temporal_when_declared(self):
        df, y = _temporal_frame(n_rows=200)
        profile = build_profile(df, y)
        cfg = SplitConfig(time_column="event_time")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.strategy == "temporal"

    def test_unparseable_time_raises(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "event_time": ["2024-01-01"] * (n - 1) + ["not-a-date"],
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.normal(size=n))
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="temporal", time_column="event_time")

        with pytest.raises(TimeParseError):
            split_data(df, y, profile=profile, config=cfg)

    def test_missing_time_column_raises(self):
        df, y = _temporal_frame(n_rows=100)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="temporal", time_column="missing_col")

        with pytest.raises(SplitColumnMissingError):
            split_data(df, y, profile=profile, config=cfg)

    def test_temporal_split_records_boundaries_evidence(self):
        df, y = _temporal_frame(n_rows=200)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="temporal", time_column="event_time")
        result = split_data(df, y, profile=profile, config=cfg)

        metrics = {ev.metric for ev in result.evidences}
        assert "temporal_boundaries" in metrics


# ═════════════════════════════════════════════════════════════
# 划分完整性审计：命中场景
# ═════════════════════════════════════════════════════════════

class TestSplitAuditHits:
    def test_random_split_with_group_declaration_detects_leakage(self):
        df, y = _clf_frame(n_rows=300, n_groups=20)
        profile = build_profile(df, y)
        # 故意用随机切分，让同一实体跨集合
        cfg = SplitConfig(split_strategy="random", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.audit is not None
        assert result.audit.risk_level == "critical"
        assert result.audit.group_overlap_count > 0
        codes = [f.code for f in result.audit.findings]
        assert "cross_split_groups" in codes
        assert any("group" in a for a in result.audit.suggested_actions)

    def test_random_split_with_time_declaration_detects_overlap(self):
        df, y = _temporal_frame(n_rows=300)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="random", time_column="event_time")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.audit is not None
        assert result.audit.temporal_ordered is False
        codes = [f.code for f in result.audit.findings]
        assert "temporal_overlap" in codes
        assert any("temporal" in a for a in result.audit.suggested_actions)

    def test_cross_split_duplicates_detected(self):
        # 构造含重复行的数据，并把重复行分到不同集合
        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            "b": ["x", "y", "z", "w", "x", "y"],
        })
        train = np.array([0, 1])
        valid = np.array([2, 4])   # 索引 4 与索引 0 行内容相同
        test = np.array([3, 5])    # 索引 5 与索引 1 行内容相同

        result = audit_split(X, train, valid, test)

        assert result.duplicate_count == 2
        codes = [f.code for f in result.findings]
        assert "cross_split_duplicates" in codes
        assert result.risk_level == "critical"
        checks = {c.code: c for c in result.check_results}
        assert checks["cross_split_duplicates"].status == "发现风险"

    def test_duplicates_within_same_split_not_flagged(self):
        X = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0]})
        # 两个重复行都在训练集内 → 不算跨集合重复
        result = audit_split(X, np.array([0, 1]), np.array([2]), np.array([3]))

        assert result.duplicate_count == 0
        assert result.risk_level == "info"

    def test_device_column_name_detected_as_group_name_pattern(self):
        # 列名模式识别属于画像阶段；此处验证声明后审计确实生效
        df, y = _clf_frame(n_rows=200, n_groups=10)
        df = df.rename(columns={"patient_id": "device_code"})
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="random", group_column="device_code")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.audit.group_overlap_count > 0
        assert result.audit.risk_level == "critical"

    def test_missing_group_column_in_audit_records_high_risk(self):
        df, y = _clf_frame(n_rows=60, n_groups=6)
        cfg = SplitConfig(group_column="absent_column")
        result = audit_split(
            df, np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), config=cfg
        )

        codes = [f.code for f in result.findings]
        assert "missing_group_column" in codes
        assert result.risk_level == "high"

    def test_missing_time_column_in_audit_records_high_risk(self):
        df, y = _clf_frame(n_rows=60, n_groups=6)
        cfg = SplitConfig(time_column="absent_time")
        result = audit_split(
            df, np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), config=cfg
        )

        codes = [f.code for f in result.findings]
        assert "missing_time_column" in codes
        assert result.risk_level == "high"

    def test_invalid_time_values_in_audit_records_medium_risk(self):
        df = pd.DataFrame({"t": ["2024-01-01", "bad", "2024-03-01", "2024-04-01",
                                 "2024-05-01", "2024-06-01"]})
        cfg = SplitConfig(time_column="t")
        result = audit_split(
            df, np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), config=cfg
        )

        codes = [f.code for f in result.findings]
        assert "invalid_time_values" in codes
        assert result.temporal_ordered is None

    def test_validation_range_overlapping_test_is_detected(self):
        df = pd.DataFrame({
            "t": pd.to_datetime([
                "2024-01-01", "2024-01-02",   # train
                "2024-01-03", "2024-01-10",   # valid 延伸到测试期
                "2024-01-05", "2024-01-06",   # test 落在验证区间内
            ])
        })
        cfg = SplitConfig(time_column="t")
        result = audit_split(
            df, np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), config=cfg
        )

        assert result.temporal_ordered is False
        assert "temporal_overlap" in {f.code for f in result.findings}


# ═════════════════════════════════════════════════════════════
# 划分完整性审计：未命中与开关
# ═════════════════════════════════════════════════════════════

class TestSplitAuditClean:
    def test_clean_data_no_findings(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        assert result.audit is not None
        assert result.audit.risk_level == "info"
        assert result.audit.findings == []
        assert result.audit.suggested_actions == []
        assert result.audit.duplicate_count == 0

    def test_undeclared_checks_marked_not_executed(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        checks = {c.code: c for c in result.audit.check_results}
        assert checks["cross_split_duplicates"].status == "通过"
        assert checks["cross_split_groups"].status == "未执行"
        assert checks["temporal_overlap"].status == "未执行"

    def test_declared_group_makes_check_executed(self):
        df, y = _clf_frame(n_rows=200, n_groups=20)
        profile = build_profile(df, y)
        cfg = SplitConfig(group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        checks = {c.code: c for c in result.audit.check_results}
        assert checks["cross_split_groups"].status == "通过"
        assert checks["temporal_overlap"].status == "未执行"

    def test_audit_disabled_returns_empty(self):
        df, y = _clf_frame(n_rows=200, n_groups=20)
        profile = build_profile(df, y)
        cfg = SplitConfig(enable_split_audit=False, group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.audit is not None
        assert result.audit.findings == []
        assert result.audit.check_results == []

    def test_group_overlap_count_zero_for_group_split(self):
        df, y = _clf_frame(n_rows=300, n_groups=30)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.audit.group_overlap_count == 0
        assert result.audit.temporal_ordered is None

    def test_sampling_path_when_exceeding_max_rows(self):
        df, y = _clf_frame(n_rows=200, n_groups=20)
        cfg = SplitConfig(duplicate_audit_max_rows=50)
        result = audit_split(
            df,
            np.arange(0, 100),
            np.arange(100, 150),
            np.arange(150, 200),
            config=cfg,
        )
        # 抽样路径不应抛错，重复数应为 0（数据本身无重复行）
        assert result.duplicate_count == 0

    def test_audit_handles_non_dataframe_input(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 2))
        result = audit_split(X, np.arange(0, 30), np.arange(30, 45), np.arange(45, 60))

        checks = {c.code: c for c in result.check_results}
        assert checks["cross_split_duplicates"].status == "未执行"


# ═════════════════════════════════════════════════════════════
# 与画像阶段的联动
# ═════════════════════════════════════════════════════════════

class TestProfileAuditLinkage:
    def test_high_risk_identifier_column_suggests_grouping(self):
        rng = np.random.default_rng(3)
        n = 120
        df = pd.DataFrame({
            "user_uuid": [f"u{i:04d}" for i in range(n)],
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.integers(0, 2, n))
        profile = build_profile(df, y)
        assert profile.audit is not None
        assert profile.audit.risk_level in {"high", "critical"}

        result = split_data(df, y, profile=profile)
        codes = [f.code for f in result.audit.findings]
        assert "potential_grouping_need" in codes

    def test_no_linkage_finding_when_group_declared(self):
        rng = np.random.default_rng(4)
        n = 120
        df = pd.DataFrame({
            "user_uuid": [f"u{i:04d}" for i in range(n)],
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.integers(0, 2, n))
        profile = build_profile(df, y)
        cfg = SplitConfig(group_column="user_uuid")
        result = split_data(df, y, profile=profile, config=cfg)

        codes = [f.code for f in result.audit.findings]
        assert "potential_grouping_need" not in codes

    def test_clean_profile_produces_no_linkage_finding(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        codes = [f.code for f in result.audit.findings]
        assert "potential_grouping_need" not in codes


# ═════════════════════════════════════════════════════════════
# 审计契约与序列化
# ═════════════════════════════════════════════════════════════

class TestSplitAuditContract:
    def test_risk_level_empty_is_info(self):
        assert SplitAuditResult().risk_level == "info"
        assert SplitAuditResult().high_risk_count == 0
        assert SplitAuditResult().passed is True

    def test_risk_level_takes_max_severity(self):
        report = SplitAuditResult(findings=[
            AuditFinding("a", "low", "t", "e", "r"),
            AuditFinding("b", "high", "t", "e", "r"),
        ])
        assert report.risk_level == "high"
        assert report.high_risk_count == 1

    def test_to_dict_includes_subclass_fields(self):
        report = SplitAuditResult(
            findings=[AuditFinding("x", "critical", "t", "e", "r")],
            check_results=[AuditCheckResult("x", "检查", "发现风险", "有风险")],
            suggested_actions=["去重后重新切分"],
            duplicate_count=3,
            group_overlap_count=2,
            temporal_ordered=False,
        )
        payload = report.to_dict()

        assert payload["risk_level"] == "critical"
        assert payload["high_risk_count"] == 1
        assert payload["suggested_actions"] == ["去重后重新切分"]
        assert payload["duplicate_count"] == 3
        assert payload["group_overlap_count"] == 2
        assert payload["temporal_ordered"] is False
        assert payload["findings"][0]["code"] == "x"

    def test_model_info_questions_present(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        assert len(result.audit.model_info_questions) >= 1

    def test_split_result_defaults_backward_compatible(self):
        # 不传阶段 2 新增字段时仍可构造（模拟旧代码）
        legacy = SplitResult(
            train_indices=np.array([0, 1]),
            valid_indices=np.array([2]),
            test_indices=np.array([3]),
            test_size=0.25,
            validation_size=0.33,
            random_state=42,
            stratify=True,
            n_train=2, n_valid=1, n_test=1,
            train_ratio=0.5, valid_ratio=0.25, test_ratio=0.25,
        )
        assert legacy.strategy == "stratified"
        assert legacy.group_column is None
        assert legacy.time_column is None
        assert legacy.audit is None
        assert legacy.n_total == 4

    def test_split_result_exposes_new_fields(self):
        df, y = _clf_frame(n_rows=200, n_groups=20)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)

        assert result.strategy == "group"
        assert result.group_column == "patient_id"
        assert result.time_column is None
        assert isinstance(result.audit, SplitAuditResult)


# ═════════════════════════════════════════════════════════════
# 报告渲染
# ═════════════════════════════════════════════════════════════

class TestSplitReportRender:
    def test_report_contains_audit_section(self):
        from tabularx.core.reports import render_split_report

        df, y = _clf_frame(n_rows=300, n_groups=20)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="random", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)
        html = render_split_report(result)

        assert "划分完整性审计" in html
        assert "固定检查项清单" in html
        assert "风险详情" in html
        assert "建议动作" in html
        assert "patient_id" in html

    def test_report_renders_clean_split(self):
        from tabularx.core.reports import render_split_report

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)
        html = render_split_report(result)

        assert "划分完整性审计" in html
        assert "未发现跨集合重复" in html
        # 干净切分不应出现风险详情区块
        assert "风险详情" not in html

    def test_report_shows_group_strategy_conclusion(self):
        from tabularx.core.reports import render_split_report

        df, y = _clf_frame(n_rows=300, n_groups=30)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="group", group_column="patient_id")
        result = split_data(df, y, profile=profile, config=cfg)
        html = render_split_report(result)

        assert "分组切分" in html
        assert "patient_id" in html

    def test_report_shows_temporal_strategy_conclusion(self):
        from tabularx.core.reports import render_split_report

        df, y = _temporal_frame(n_rows=200)
        profile = build_profile(df, y)
        cfg = SplitConfig(split_strategy="temporal", time_column="event_time")
        result = split_data(df, y, profile=profile, config=cfg)
        html = render_split_report(result)

        assert "时序切分" in html
        assert "event_time" in html

    def test_report_renders_when_audit_disabled(self):
        from tabularx.core.reports import render_split_report

        data = load_iris(as_frame=True)
        profile = build_profile(data.data, data.target)
        cfg = SplitConfig(enable_split_audit=False)
        result = split_data(data.data, data.target, profile=profile, config=cfg)
        html = render_split_report(result)

        # audit 存在但为空 → 渲染基本结构且不报错
        assert "TabularX 数据划分报告" in html


# ═════════════════════════════════════════════════════════════
# 向后兼容：不传新参数时索引与旧版一致
# ═════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_default_config_matches_legacy_random_state_behaviour(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        # 与直接调用 sklearn 的旧两阶段实现对比
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(data.data))
        train_rest, test = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=data.target
        )
        train, valid = train_test_split(
            train_rest, test_size=0.25, random_state=42,
            stratify=data.target.iloc[train_rest],
        )

        assert np.array_equal(result.test_indices, np.sort(test))
        assert np.array_equal(result.valid_indices, np.sort(valid))
        assert np.array_equal(result.train_indices, np.sort(train))

    def test_regression_default_matches_legacy(self):
        data = load_diabetes(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)

        from sklearn.model_selection import train_test_split

        indices = np.arange(len(data.data))
        train_rest, test = train_test_split(indices, test_size=0.2, random_state=42)
        train, valid = train_test_split(train_rest, test_size=0.25, random_state=42)

        assert np.array_equal(result.test_indices, np.sort(test))
        assert np.array_equal(result.train_indices, np.sort(train))

    def test_downstream_preprocess_still_works(self):
        from tabularx import preprocess

        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        split = split_data(data.data, data.target, profile=profile)
        pre = preprocess(data.data, data.target, profile=profile, split_result=split)

        assert pre.n_features_in == data.data.shape[1]
        assert len(pre.X_train) == split.n_train
        assert len(pre.X_valid) == split.n_valid
        assert len(pre.X_test) == split.n_test
