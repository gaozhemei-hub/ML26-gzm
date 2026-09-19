"""Copy this small adapter into a future integration commit, not into core now."""
from leakage_audit import LeakageAuditConfig, LeakageAuditor, validate_selection_protocol, write_audit_artifacts


def audit_existing_tabularx_run(X, y, task_type, split_result, search_result, output_dir):
    """Audit an existing TabularX run without changing any TabularX source file.

    The caller invokes this immediately after ``split_data`` and again after
    ``search_models``.  It intentionally reports, rather than hides, the
    current test-set exposure until the team accepts a separate safe-search PR.
    """
    auditor = LeakageAuditor(LeakageAuditConfig())
    audit = auditor.preflight(X, y, task_type)
    audit = auditor.check_split(
        X, split_result.train_indices, split_result.valid_indices, split_result.test_indices, audit
    )
    audit = validate_selection_protocol(
        search_result.model_results, search_result.primary_metric,
        search_result.optimization_direction, audit,
    )
    return audit, write_audit_artifacts(audit, output_dir)
