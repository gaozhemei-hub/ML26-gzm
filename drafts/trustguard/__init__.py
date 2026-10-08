"""TrustGuard: independent trustworthiness extensions for TabularX.

This package is intentionally separate from ``src/tabularx``.  Its ``leakage``
subpackage supplies leakage checks and safe model selection; the package root
adds an explicit test-access guard, reproducibility evidence and
distribution/coverage auditing.
"""

from .distribution import DistributionAuditConfig, DistributionAuditResult, audit_distribution, write_distribution_artifacts
from .ledger import ReproducibilityLedger, verify_artifact_manifest
from .protocol import EvaluationProtocolGuard, EvaluationProtocolError, write_protocol_artifacts
from .benchmark import DatasetSpec, create_cc18_manifest, inspect_artifacts, write_benchmark_plan
from .reporting import render_trustguard_report, write_trustguard_report
from .pipeline import TrustGuardConfig, run_trustguard

__all__ = [
    "DistributionAuditConfig", "DistributionAuditResult", "audit_distribution",
    "write_distribution_artifacts", "ReproducibilityLedger", "verify_artifact_manifest",
    "EvaluationProtocolGuard", "EvaluationProtocolError", "write_protocol_artifacts",
    "DatasetSpec", "create_cc18_manifest", "inspect_artifacts", "write_benchmark_plan",
    "render_trustguard_report", "write_trustguard_report",
    "TrustGuardConfig", "run_trustguard",
]
