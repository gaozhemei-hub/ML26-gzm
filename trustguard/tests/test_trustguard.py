import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trustguard.distribution import audit_distribution
from trustguard.ledger import ReproducibilityLedger, verify_artifact_manifest
from trustguard.protocol import EvaluationProtocolError, EvaluationProtocolGuard


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

    def test_distribution_detects_categorical_shift_and_group_coverage(self):
        X = pd.DataFrame({"value": [0.0, 0.1, 0.2, 0.3, 10.0, 11.0], "group": ["a", "a", "a", "a", "b", "b"]})
        result = audit_distribution(X, [0, 1, 2, 3], [4], [5], sensitive_features=["group"])
        self.assertTrue(any(record.severity in {"medium", "high"} for record in result.records))
        self.assertTrue(any(record.severity == "high" for record in result.group_coverage))


if __name__ == "__main__":
    unittest.main()
