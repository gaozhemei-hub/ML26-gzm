import unittest

import pandas as pd

from trustguard.leakage import AuditResult, LeakageAuditConfig, LeakageAuditor


class LeakageAuditTests(unittest.TestCase):
    def test_target_copy_is_critical_and_excluded(self):
        y = pd.Series([0, 1] * 10, name="target")
        X = pd.DataFrame({"copied_target": y, "feature": range(20)})
        result = LeakageAuditor().preflight(X, y, "binary")
        codes = {finding.code for finding in result.findings}
        self.assertIn("target_copy", codes)
        self.assertIn("copied_target", result.excluded_columns)
        self.assertEqual(result.risk_level, "critical")

    def test_validation_range_overlapping_test_is_detected(self):
        X = pd.DataFrame({
            "event_time": pd.to_datetime([
                "2025-01-01", "2025-01-02",  # train
                "2025-01-03", "2025-01-10",  # validation
                "2025-01-05", "2025-01-06",  # test overlaps validation
            ])
        })
        auditor = LeakageAuditor(LeakageAuditConfig(time_column="event_time"))
        result = auditor.check_split(X, [0, 1], [2, 3], [4, 5], AuditResult())
        self.assertIn("temporal_overlap", {item.code for item in result.findings})


if __name__ == "__main__":
    unittest.main()
