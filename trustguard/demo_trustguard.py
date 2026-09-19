"""Run the four TrustGuard modules without modifying TabularX source code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from trustguard.leakage import LeakageAuditConfig, LeakageAuditor, write_audit_artifacts
from tabularx import build_profile, split_data
from trustguard.distribution import audit_distribution, write_distribution_artifacts
from trustguard.ledger import ReproducibilityLedger
from trustguard.protocol import EvaluationProtocolGuard, write_protocol_artifacts


def load_data(dataset: str, csv_path: str | None, target: str | None):
    if csv_path:
        if not target:
            raise ValueError("使用 --csv 时必须提供 --target。")
        frame = pd.read_csv(csv_path)
        if target not in frame:
            raise ValueError(f"目标列 {target!r} 不在 CSV 中。")
        return frame.drop(columns=[target]), frame[target], "auto"
    loaders = {"breast_cancer": (load_breast_cancer, "classification"), "iris": (load_iris, "classification"), "diabetes": (load_diabetes, "regression")}
    loader, task = loaders[dataset]
    data = loader(as_frame=True)
    return data.data, data.target.rename("target"), task


def run(output_dir: str | Path, *, dataset: str = "breast_cancer", csv_path: str | None = None, target: str | None = None, group_column: str | None = None, time_column: str | None = None, sensitive_features: list[str] | None = None) -> dict[str, str]:
    X, y, task = load_data(dataset, csv_path, target)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    profile = build_profile(X, y, task_type=task)
    audit = LeakageAuditor(LeakageAuditConfig(group_column=group_column, time_column=time_column)).preflight(X, y, profile.target.task_type)
    split = split_data(X.drop(columns=audit.excluded_columns), y, profile)
    audit = LeakageAuditor(LeakageAuditConfig(group_column=group_column, time_column=time_column)).check_split(X, split.train_indices, split.valid_indices, split.test_indices, audit)
    leakage = write_audit_artifacts(audit, root)
    drift = write_distribution_artifacts(audit_distribution(X, split.train_indices, split.valid_indices, split.test_indices, sensitive_features=sensitive_features or []), root)
    guard = EvaluationProtocolGuard()
    guard.record_validation("safe_model_search", "候选模型只允许使用训练/验证集")
    guard.lock_validation_choice("safe_model_search", "将由安全搜索器按验证集选择最终候选")
    guard.authorize_final_test("safe_model_search", "最终锁定模型的一次测试集评估")
    protocol = write_protocol_artifacts(guard, root)
    # Write the run index before creating the manifest, so the manifest covers it too.
    result = {"leakage_summary": str(leakage["summary"]), "distribution_summary": str(drift["summary"]), "protocol_log": str(protocol), "ledger": str(root / "reproducibility_ledger" / "reproducibility_ledger.json"), "manifest": str(root / "artifact_manifest.json")}
    (root / "trustguard_run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ReproducibilityLedger().capture(root, X, y, split=split, config={"task_type": profile.target.task_type, "group_column": group_column, "time_column": time_column}, command="python trustguard/demo_trustguard.py ...")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustGuard 四模块独立演示")
    parser.add_argument("--dataset", choices=["breast_cancer", "iris", "diabetes"], default="breast_cancer")
    parser.add_argument("--csv")
    parser.add_argument("--target")
    parser.add_argument("--group-column")
    parser.add_argument("--time-column")
    parser.add_argument("--sensitive-features", nargs="*", default=[])
    parser.add_argument("--output-dir", default="./trustguard-output")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, dataset=args.dataset, csv_path=args.csv, target=args.target, group_column=args.group_column, time_column=args.time_column, sensitive_features=args.sensitive_features), ensure_ascii=False, indent=2))
