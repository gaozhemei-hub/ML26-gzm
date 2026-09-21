"""TrustGuard orchestration layer.

Core audit functions remain free of file-system side effects.  This module is
the single place that coordinates TabularX profiling/splitting and writes the
independent TrustGuard artifact tree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from tabularx import build_profile, split_data

from .distribution import audit_distribution, write_distribution_artifacts
from .leakage import LeakageAuditConfig, LeakageAuditor, write_audit_artifacts
from .ledger import ReproducibilityLedger
from .protocol import EvaluationProtocolGuard, write_protocol_artifacts
from .reporting import write_trustguard_report


@dataclass(frozen=True)
class TrustGuardConfig:
    """Central configuration for one TrustGuard run."""

    task_type: str = "auto"
    group_column: str | None = None
    time_column: str | None = None
    sensitive_features: tuple[str, ...] = field(default_factory=tuple)
    pdf_report: bool = False
    command: str | None = None
    dataset_name: str | None = None
    dataset_source: str | None = None
    model_result_path: str | None = None
    benchmark_summary_path: str | None = None


def run_trustguard(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    *,
    config: TrustGuardConfig | None = None,
    protocol_guard: EvaluationProtocolGuard | None = None,
) -> dict[str, str]:
    """Run all four safeguards and write a self-contained artifact directory.

    ``protocol_guard`` may be supplied by the real model-search workflow.  If
    omitted, an initialized (not-yet-complete) guard is recorded; the audit
    pipeline never fabricates a test-set access that did not occur.
    """
    cfg = config or TrustGuardConfig()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    profile = build_profile(X, y, task_type=cfg.task_type)
    leakage_config = LeakageAuditConfig(
        group_column=cfg.group_column,
        time_column=cfg.time_column,
    )
    auditor = LeakageAuditor(leakage_config)
    audit = auditor.preflight(X, y, profile.target.task_type)

    model_X = X.drop(columns=audit.excluded_columns, errors="ignore")
    split = split_data(model_X, y, profile)
    audit = auditor.check_split(
        X,
        split.train_indices,
        split.valid_indices,
        split.test_indices,
        audit,
    )
    leakage_paths = write_audit_artifacts(audit, root)

    distribution = audit_distribution(
        X,
        split.train_indices,
        split.valid_indices,
        split.test_indices,
        sensitive_features=cfg.sensitive_features,
    )
    distribution_paths = write_distribution_artifacts(distribution, root)

    guard = protocol_guard or EvaluationProtocolGuard()
    protocol_path = write_protocol_artifacts(guard, root)

    ledger = ReproducibilityLedger()
    ledger_paths = ledger.capture(
        root,
        X,
        y,
        split=split,
        config=asdict(cfg),
        command=cfg.command,
    )
    ledger_data = json.loads(ledger_paths["ledger"].read_text(encoding="utf-8"))
    report_data = {
        "dataset": {
            "name": cfg.dataset_name or "未命名数据集",
            "source": cfg.dataset_source or "用户提供",
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "task_type": profile.target.task_type,
            "target": str(y.name or "target"),
        },
        "leakage": audit.to_dict(),
        "distribution": distribution.to_dict(),
        "protocol": guard.to_dict(),
        "ledger": ledger_data,
        "benchmark": None,
        "model_result": None,
    }
    if cfg.model_result_path:
        model_path = Path(cfg.model_result_path)
        report_data["model_result"] = json.loads(model_path.read_text(encoding="utf-8"))
    if cfg.benchmark_summary_path:
        benchmark_path = Path(cfg.benchmark_summary_path)
        report_data["benchmark"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    report_paths = write_trustguard_report(
        report_data,
        root,
        pdf_report=cfg.pdf_report,
    )

    result = {
        "leakage_summary": str(leakage_paths["summary"]),
        "distribution_summary": str(distribution_paths["summary"]),
        "protocol_log": str(protocol_path),
        "ledger": str(ledger_paths["ledger"]),
        "manifest": str(root / "artifact_manifest.json"),
        **{f"report_{key}": value for key, value in report_paths.items()},
    }
    (root / "trustguard_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Regenerate last so the manifest covers every final artifact.
    ledger.write_artifact_manifest(root)
    return result


__all__ = ["TrustGuardConfig", "run_trustguard"]
