import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trustguard.distribution import audit_distribution
from trustguard.ledger import ReproducibilityLedger, verify_artifact_manifest
from trustguard.protocol import EvaluationProtocolError, EvaluationProtocolGuard
from trustguard.reporting import render_trustguard_report
from trustguard.pipeline import TrustGuardConfig, run_trustguard


class ProtocolTests(unittest.TestCase):
    def test_test_access_requires_locked_validation_model(self):
        guard = EvaluationProtocolGuard()
        with self.assertRaises(EvaluationProtocolError):
            guard.authorize_final_test("rf")
        guard.record_validation("rf")
        guard.lock_validation_choice("rf")
        guard.authorize_final_test("rf")
        self.assertTrue(guard.compliant)
        with self.assertRaises(EvaluationProtocolError):
            guard.authorize_final_test("rf")


class LedgerAndDriftTests(unittest.TestCase):
    def test_ledger_manifest_verifies(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"]})
        y = pd.Series([0, 0, 1, 1], name="target")
        with tempfile.TemporaryDirectory() as temp:
            ReproducibilityLedger().capture(temp, X, y, config={"seed": 42})
            self.assertTrue(verify_artifact_manifest(temp)["valid"])

    def test_manifest_detects_unexpected_artifact(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0], name="target")
        with tempfile.TemporaryDirectory() as temp:
            ReproducibilityLedger().capture(temp, X, y)
            Path(temp, "added_after_manifest.txt").write_text("changed", encoding="utf-8")
            verification = verify_artifact_manifest(temp)
            self.assertFalse(verification["valid"])
            self.assertIn("added_after_manifest.txt", verification["unexpected"])

    def test_distribution_detects_categorical_shift_and_group_coverage(self):
        X = pd.DataFrame({"value": [0.0, 0.1, 0.2, 0.3, 10.0, 11.0], "group": ["a", "a", "a", "a", "b", "b"]})
        result = audit_distribution(X, [0, 1, 2, 3], [4], [5], sensitive_features=["group"])
        self.assertTrue(any(record.severity in {"medium", "high"} for record in result.records))
        self.assertTrue(any(record.severity == "high" for record in result.group_coverage))

    def test_group_absent_from_test_is_recorded_with_zero_count(self):
        X = pd.DataFrame({"group": ["a", "b", "a", "b", "a"]})
        result = audit_distribution(
            X,
            [0, 1],
            [2, 3],
            [4],
            sensitive_features=["group"],
        )
        missing = [
            row for row in result.group_coverage
            if row.group == "b" and row.split == "test"
        ]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].count, 0)
        self.assertEqual(missing[0].severity, "high")


class ReportTests(unittest.TestCase):
    def test_report_renderer_is_pure_and_contains_all_sections(self):
        report = render_trustguard_report({
            "leakage": {"risk_level": "low", "findings": []},
            "distribution": {
                "risk_level": "medium",
                "feature_records": [],
                "group_coverage": [{
                    "sensitive_feature": "group",
                    "group": "group-a",
                    "split": "test",
                    "count": 12,
                    "proportion": 0.4,
                    "severity": "high",
                }],
            },
            "protocol": {"compliant": True, "test_access_count": 1},
            "ledger": {"data": {"fingerprint_sha256": "abc123"}},
            "dataset": {"name": "测试数据集", "source": "单元测试", "rows": 30, "features": 2, "task_type": "binary"},
        })
        self.assertIn("TabularX TrustGuard", report)
        self.assertIn("数据泄露审计", report)
        self.assertIn("评估协议", report)
        self.assertIn("可复现实验证据", report)
        self.assertIn("特征漂移与群体覆盖", report)
        self.assertIn("敏感群体覆盖", report)
        self.assertIn("group-a", report)
        self.assertIn("abc123", report)
        self.assertIn("MLJAR 可解释性工件对比实验", report)
        self.assertIn("尚未执行 MLJAR/CC-18", report)
        self.assertIn("检查项结果", report)
        self.assertIn("未指定 group_column", report)
        self.assertIn("主实验数据集", report)
        self.assertIn("测试数据集", report)
        self.assertNotIn("做了什么", report)
        self.assertNotIn("如何解读", report)

    def test_report_renders_real_benchmark_summary_shape(self):
        report = render_trustguard_report({
            "leakage": {"risk_level": "info", "findings": []},
            "distribution": {"risk_level": "info", "feature_records": [], "group_coverage": []},
            "protocol": {"status": "not_started", "test_access_count": 0},
            "ledger": {"data": {}, "split": {}, "environment": {}},
            "model_result": {
                "primary_metric": "roc_auc",
                "direction": "maximize",
                "selected_model_name": "lightgbm",
                "final_test_metrics": {"accuracy": 0.95, "roc_auc": 0.98},
                "candidates": [{"model_name": "lightgbm", "validation_metrics": {"roc_auc": 0.99}, "error": None}],
            },
            "benchmark": {
                "dataset_count": 5,
                "success_count": 5,
                "failure_count": 0,
                "rows": [{
                    "dataset": "openml_task_3",
                    "mljar_explanation": "5/6",
                    "mljar_trust_artifacts": "0/4",
                    "trustguard_artifacts": "4/4",
                    "injected_leakage_detected": True,
                    "interpretation": "注入风险被识别。",
                }],
                "interpretation": "基于实际运行工件汇总。",
            },
        })
        self.assertIn("openml_task_3", report)
        self.assertIn("目标复制和高唯一性 ID 风险均被识别", report)
        self.assertNotIn("尚未执行 MLJAR/CC-18", report)
        self.assertIn("LightGBM 根据验证集", report)
        self.assertNotIn("Result and interpretation", report)


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_independent_report_tree(self):
        X = pd.DataFrame({"a": range(60), "group": ["x", "y"] * 30})
        y = pd.Series([0, 1] * 30, name="target")
        with tempfile.TemporaryDirectory() as temp:
            result = run_trustguard(
                X,
                y,
                temp,
                config=TrustGuardConfig(
                    task_type="classification",
                    sensitive_features=("group",),
                    dataset_name="流水线测试数据",
                    dataset_source="单元测试",
                ),
            )
            self.assertTrue(Path(result["report_html"]).is_file())
            self.assertTrue(Path(temp, "reports", "trustguard.html").is_file())
            html = Path(temp, "reports", "trustguard.html").read_text(encoding="utf-8")
            self.assertIn("流水线测试数据", html)
            self.assertIn("重复数为 0", html)
            self.assertTrue(verify_artifact_manifest(temp)["valid"])


if __name__ == "__main__":
    unittest.main()
