import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trustguard.run_benchmark_suite import run


class BenchmarkSuiteTests(unittest.TestCase):
    def test_suite_aggregates_mljar_and_injected_leakage_without_network(self):
        X = pd.DataFrame({"feature": range(60), "category": ["a", "b"] * 30})
        y = pd.Series([0, 1] * 30, name="target")

        def fake_load_data(*args, **kwargs):
            return X, y, {"source": "unit-test", "task_id": 3}

        def fake_run_mljar(_X, _y, output_dir, _time_limit):
            results = Path(output_dir) / "mljar_results"
            results.mkdir(parents=True)
            for filename in (
                "README.md", "shap_summary.png", "feature_importance.csv",
                "confusion_matrix.png", "learning_curve.png", "fairness.md",
            ):
                (results / filename).write_text("fixture", encoding="utf-8")
            return results

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "suite": "fixture",
                "datasets": [{"name": "task_3", "openml_task_id": 3}],
            }), encoding="utf-8")
            with patch("trustguard.run_benchmark_suite.load_data", fake_load_data), patch(
                "trustguard.run_benchmark_suite.run_mljar", fake_run_mljar
            ):
                summary_path = run(manifest, root / "out", total_time_limit=1)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["success_count"], 1)
            self.assertTrue(summary["rows"][0]["injected_leakage_detected"])
            report = (root / "out" / "reports" / "trustguard.html").read_text(encoding="utf-8")
            self.assertIn("task_3", report)
            self.assertNotIn("尚未执行 MLJAR/CC-18", report)


if __name__ == "__main__":
    unittest.main()
