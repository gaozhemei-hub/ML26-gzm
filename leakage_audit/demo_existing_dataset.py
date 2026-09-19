"""Run LeakageAudit against the project's existing breast-cancer tutorial data.

This is intentionally a standalone entry point: it imports TabularX but does
not alter its source, configuration objects, or output conventions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer

from leakage_audit import LeakageAuditConfig, LeakageAuditor, validate_selection_protocol, write_audit_artifacts
from tabularx import build_profile, preprocess, search_models, split_data


def run(output_dir: str | Path, run_model_search: bool = False, inject_leakage: bool = False) -> dict[str, str]:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data.copy(), data.target.rename("target")
    if inject_leakage:
        # Educational switch: verifies that the report catches a known failure.
        X["target_copy_demo"] = y

    output_dir = Path(output_dir)
    auditor = LeakageAuditor(LeakageAuditConfig(mode="warn"))
    profile = build_profile(X, y, task_type="classification")
    audit = auditor.preflight(X, y, profile.target.task_type)
    X_safe = X.drop(columns=audit.excluded_columns, errors="ignore")
    if len(X_safe.columns) != len(X.columns):
        profile = build_profile(X_safe, y, task_type="classification")
    split = split_data(X_safe, y, profile)
    audit = auditor.check_split(X_safe, split.train_indices, split.valid_indices, split.test_indices, audit)

    if run_model_search:
        processed = preprocess(X_safe, y, profile, split)
        searched = search_models(processed, profile)
        audit = validate_selection_protocol(
            searched.model_results, searched.primary_metric, searched.optimization_direction, audit
        )
    return write_audit_artifacts(audit, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./leakage-demo-output")
    parser.add_argument("--run-model-search", action="store_true")
    parser.add_argument("--inject-leakage", action="store_true")
    args = parser.parse_args()
    print(run(args.output_dir, args.run_model_search, args.inject_leakage))
