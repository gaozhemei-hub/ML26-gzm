"""Dependency-light regression tests for the standalone extension."""
import unittest

import pandas as pd

from leakage_audit import LeakageAuditConfig, LeakageAuditor, validate_selection_protocol


class Run:
    def __init__(self, name, valid, test):
        self.model_name = name
        self.status = "success"
        self.validation_metrics = {"roc_auc": valid}
        self.test_metrics = {"roc_auc": test}


class LeakageAuditTests(unittest.TestCase):
    def test_target_copy_and_identifier_are_reported(self):
        X = pd.DataFrame({"record_id": range(20), "outcome_copy": [0, 1] * 10})
        y = pd.Series([0, 1] * 10)
        result = LeakageAuditor().preflight(X, y, "binary")
        self.assertEqual(result.risk_level, "critical")
        self.assertTrue(any(f.code == "target_copy" for f in result.findings))
        self.assertTrue(any(f.code == "identifier_like_column" for f in result.findings))

    def test_group_overlap_is_reported(self):
        X = pd.DataFrame({"patient_id": ["a", "a", "b", "b", "c", "c"], "x": range(6)})
        result = LeakageAuditor(LeakageAuditConfig(group_column="patient_id")).check_split(X, [0, 2], [1, 4], [3, 5])
        self.assertTrue(any(f.code == "cross_split_groups" for f in result.findings))

    def test_test_driven_selection_is_reported(self):
        result = validate_selection_protocol([Run("A", .90, .60), Run("B", .80, .95)], "roc_auc", "maximize")
        self.assertTrue(any(f.code == "test_set_model_selection" for f in result.findings))


if __name__ == "__main__":
    unittest.main()
