"""可信审计在报告与交付清单中的呈现测试。

覆盖：画像报告流程图的审计阶段、划分报告流程图的审计链路、
总报告的审计总览页与封面色险提示、审计结论的结构化落盘，
以及未启用审计时的降级渲染。
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from tabularx import AutoEstimator, build_profile, split_data
from tabularx.core.params import ProfileConfig, SplitConfig
from tabularx.core.reports import render_full_report, render_profile_report, render_split_report


# ── 数据构造 ─────────────────────────────────────────────────


def _risky_frame(n_rows: int = 120, seed: int = 3):
    """构造同时触发画像与划分风险的分类数据集。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"P{i % 20:04d}" for i in range(n_rows)],
        "x1": rng.normal(size=n_rows),
        "x2": rng.normal(size=n_rows),
    })
    y = pd.Series(rng.integers(0, 2, n_rows), name="target")
    return df, y


def _clean_profile():
    data = load_breast_cancer(as_frame=True)
    return build_profile(data.data, data.target), data


def _small_estimator(tmp_path: Path) -> AutoEstimator:
    ae = AutoEstimator(task="classification", output_dir=tmp_path)
    ae.param.model_search.models = ["random_forest"]
    ae.param.model_search.n_trials = 1
    ae.param.model_search.n_jobs = 1
    ae.param.model_search.fixed_params = {"random_forest": {"n_estimators": 8, "max_depth": 3}}
    ae.param.shap.max_explain_samples = 10
    return ae


# ═════════════════════════════════════════════════════════════
# 画像报告：流程图与摘要
# ═════════════════════════════════════════════════════════════

class TestProfileFlowchart:
    def test_flowchart_covers_static_audit_stage(self):
        profile, _ = _clean_profile()
        html = render_profile_report(profile)

        # 审计阶段的节点与判定分支
        assert "AuditStart" in html
        assert "AuditID" in html
        assert "AuditCopy" in html
        assert "AuditForbid" in html
        assert "RiskSum" in html
        # 审计节点使用独立样式类，和普通观测节点区分
        assert "classDef audit" in html

    def test_flowchart_has_identifier_branch(self):
        profile, _ = _clean_profile()
        html = render_profile_report(profile)
        assert "identifier_like" in html

    def test_flowchart_note_mentions_audit(self):
        profile, _ = _clean_profile()
        html = render_profile_report(profile)
        assert "静态泄漏预检（ID / 目标复制 / 目标代理 / 禁用特征）" in html
        assert "再对整体做静态泄漏预检" in html

    def test_summary_counts_identifier_columns(self):
        df, y = _risky_frame()
        profile = build_profile(df, y)
        html = render_profile_report(profile)

        assert "列疑似标识符" in html
        assert "identifier_like" in html

    def test_summary_reports_audit_risk_level(self):
        profile, _ = _clean_profile()
        html = render_profile_report(profile)
        assert "静态泄漏预检风险等级" in html
        assert "info" in html

    def test_identifier_tag_uses_warning_badge(self):
        """identifier_like 应与其他风险标签一样渲染为警示样式。"""
        df, y = _risky_frame()
        profile = build_profile(df, y)
        html = render_profile_report(profile)

        # 特征详情表里 identifier_like 单独渲染为警示徽标
        assert '<span class="badge badge-warn">identifier_like</span>' in html
        # 摘要计数行存在
        assert "列疑似标识符" in html

    def test_flowchart_renders_when_audit_disabled(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target,
                                config=ProfileConfig(enable_leakage_audit=False))
        html = render_profile_report(profile)
        # audit 存在但为空：流程与摘要仍可渲染
        assert "数据安全与泄漏预检" in html
        assert "① 摘要" in html


# ═════════════════════════════════════════════════════════════
# 划分报告：流程图覆盖审计链路
# ═════════════════════════════════════════════════════════════

class TestSplitFlowchart:
    def test_flowchart_contains_audit_chain(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)
        html = render_split_report(result)

        assert "Audit[" in html
        assert "DupCheck" in html
        assert "GroupCheck" in html
        assert "TimeCheck" in html

    def test_flowchart_contains_strategy_branches(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)
        html = render_split_report(result)

        for token in ("StrategyCheck", "GroupSplit", "TemporalSplit", "Stratify", "Random"):
            assert token in html

    def test_flowchart_risk_nodes_marked_as_error(self):
        data = load_breast_cancer(as_frame=True)
        profile = build_profile(data.data, data.target)
        result = split_data(data.data, data.target, profile=profile)
        html = render_split_report(result)

        assert "DupRisk" in html
        assert "GroupRisk" in html
        assert "TimeRisk" in html


# ═════════════════════════════════════════════════════════════
# 总报告：审计总览页
# ═════════════════════════════════════════════════════════════

class TestFullReportAuditOverview:
    def test_overview_page_and_flowchart_present(self, tmp_path):
        df, y = _risky_frame()
        ae = _small_estimator(tmp_path)
        ae.param.split.split_strategy = "random"
        ae.param.split.group_column = "patient_id"
        ae.fit(df, y)

        html = (Path(ae.output_dir) / "report.html").read_text(encoding="utf-8")
        assert "可信审计总览" in html
        assert "ProfileAudit" in html
        assert "SplitAudit" in html
        assert "审计链路" in html

    def test_cover_shows_overall_risk(self, tmp_path):
        df, y = _risky_frame()
        ae = _small_estimator(tmp_path)
        ae.param.split.split_strategy = "random"
        ae.param.split.group_column = "patient_id"
        ae.fit(df, y)

        html = (Path(ae.output_dir) / "report.html").read_text(encoding="utf-8")
        assert "可信审计综合风险" in html
        # 该运行中划分审计必然给出 critical
        assert "CRITICAL" in html

    def test_findings_table_lists_stage_and_severity(self, tmp_path):
        df, y = _risky_frame()
        ae = _small_estimator(tmp_path)
        ae.param.split.split_strategy = "random"
        ae.param.split.group_column = "patient_id"
        ae.fit(df, y)

        html = (Path(ae.output_dir) / "report.html").read_text(encoding="utf-8")
        assert "风险与发现汇总" in html
        assert "划分完整性审计" in html
        assert "同一实体跨集合出现" in html
        assert "建议动作" in html

    def test_clean_run_shows_no_findings_conclusion(self, tmp_path):
        data = load_breast_cancer(as_frame=True)
        ae = _small_estimator(tmp_path)
        ae.fit(data.data, data.target)

        html = (Path(ae.output_dir) / "report.html").read_text(encoding="utf-8")
        assert "可信审计总览" in html
        assert "均未发现泄漏、重复或非法特征风险" in html
        assert "风险与发现汇总" not in html

    def test_full_report_without_audit_summary_still_renders(self):
        """未提供 audit_summary 时模板降级：不渲染总览页，其余结构正常。"""
        html = render_full_report(dict(sections=[dict(id="a", title="示例", body="<p>x</p>")]))
        assert "TabularX 实验报告" in html
        assert "可信审计总览" not in html
        assert "示例" in html

    def test_audit_flowchart_covers_all_three_stages(self, tmp_path):
        df, y = _risky_frame()
        ae = _small_estimator(tmp_path)
        ae.fit(df, y)

        html = (Path(ae.output_dir) / "report.html").read_text(encoding="utf-8")
        for token in ("ProfileAudit", "SplitAudit", "PreAudit", "ExcludeCols"):
            assert token in html

    def test_audit_overview_does_not_break_paging(self, tmp_path):
        df, y = _risky_frame()
        ae = _small_estimator(tmp_path)
        ae.fit(df, y)

        html = (Path(ae.output_dir) / "report.html").read_text(encoding="utf-8")
        # 每页容器都带唯一 data-index，且审计总览页占用第一页之后的位置
        indices = re.findall(r'<div class="page[^"]*" data-index="([^"]+)"', html)
        assert indices[0] == "0"          # 封面
        assert indices[1] == "1"          # 审计总览
        assert len(indices) == len(set(indices))
        assert "报告结束" in html


# ═════════════════════════════════════════════════════════════
# 审计结论结构化落盘与交付清单
# ═════════════════════════════════════════════════════════════

class TestAuditArtifact:
    def test_audit_json_written_with_structure(self, tmp_path):
        df, y = _risky_frame()
        ae = _small_estimator(tmp_path)
        ae.param.split.split_strategy = "random"
        ae.param.split.group_column = "patient_id"
        ae.fit(df, y)

        payload = json.loads(
            (Path(ae.output_dir) / "evidence" / "audit.json").read_text(encoding="utf-8")
        )
        assert payload["overall_risk_level"] == "critical"
        assert {s["stage"] for s in payload["stages"]} == {
            "画像静态预检", "划分完整性审计", "预处理分布审计",
        }
        assert any(f["title"] == "同一实体跨集合出现" for f in payload["findings"])
        assert payload["suggested_actions"]

    def test_audit_record_registered_in_delivery(self, tmp_path):
        data = load_breast_cancer(as_frame=True)
        ae = _small_estimator(tmp_path)
        ae.fit(data.data, data.target)

        records = json.loads(
            (Path(ae.output_dir) / "evidence" / "delivery.json").read_text(encoding="utf-8")
        )["records"]
        assert "audit" in records
        assert records["audit"]["status"] == "generated"
        assert records["audit"]["required"] is True

    def test_clean_run_audit_json_is_info(self, tmp_path):
        data = load_breast_cancer(as_frame=True)
        ae = _small_estimator(tmp_path)
        ae.fit(data.data, data.target)

        payload = json.loads(
            (Path(ae.output_dir) / "evidence" / "audit.json").read_text(encoding="utf-8")
        )
        assert payload["overall_risk_level"] == "info"
        assert payload["findings"] == []

    def test_audit_summary_aggregates_profile_drop_columns(self, tmp_path):
        """画像判定为高危标识符的列应出现在审计汇总的建议排除列中。"""
        rng = np.random.default_rng(5)
        n = 120
        df = pd.DataFrame({
            "user_uuid": [f"u{i:04d}" for i in range(n)],
            "x1": rng.normal(size=n),
        })
        y = pd.Series(rng.integers(0, 2, n), name="target")

        ae = _small_estimator(tmp_path)
        ae.fit(df, y)

        payload = json.loads(
            (Path(ae.output_dir) / "evidence" / "audit.json").read_text(encoding="utf-8")
        )
        assert "user_uuid" in payload["suggested_drop_columns"]
        assert payload["overall_risk_level"] in {"high", "critical"}

    def test_audit_delivery_label_is_chinese(self, tmp_path):
        data = load_breast_cancer(as_frame=True)
        ae = _small_estimator(tmp_path)
        ae.fit(data.data, data.target)

        html = (Path(ae.output_dir) / "evidence" / "delivery.html").read_text(encoding="utf-8")
        assert "可信审计结论" in html
