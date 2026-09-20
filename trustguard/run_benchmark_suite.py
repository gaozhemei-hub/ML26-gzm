"""Run the MLJAR-versus-TrustGuard artifact study on an OpenML manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trustguard.benchmark import inspect_artifacts
from trustguard.pipeline import TrustGuardConfig, run_trustguard
from trustguard.reporting import write_trustguard_report
from trustguard.run_mljar_baseline import load_data, run_mljar


_EXPLANATION_KEYS = (
    "report", "feature_importance", "shap", "confusion_matrix",
    "learning_curve", "fairness",
)
_TRUST_KEYS = (
    "leakage_audit", "test_access_log", "reproducibility_ledger",
    "distribution_audit",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _present(inventory: dict[str, bool], keys: tuple[str, ...]) -> str:
    found = [key for key in keys if inventory.get(key)]
    return f"{len(found)}/{len(keys)}：" + ("、".join(found) if found else "无")


def _load_trustguard_report_data(run_dir: Path) -> dict[str, Any]:
    return {
        "leakage": _read_json(run_dir / "leakage_audit" / "summary.json"),
        "distribution": _read_json(run_dir / "distribution_audit" / "summary.json"),
        "protocol": _read_json(run_dir / "evaluation_protocol" / "test_access_log.json"),
        "ledger": _read_json(run_dir / "reproducibility_ledger" / "reproducibility_ledger.json"),
    }


def run(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    total_time_limit: int = 120,
    pdf_report: bool = False,
) -> Path:
    """Run clean MLJAR, clean TrustGuard and injected-leakage TrustGuard."""
    manifest = _read_json(manifest_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    representative_run: Path | None = None

    for spec in manifest["datasets"]:
        name = spec["name"]
        task_id = spec.get("openml_task_id")
        dataset_root = root / name
        try:
            X, y, source = load_data("breast_cancer", None, None, task_id)

            mljar_root = dataset_root / "mljar"
            mljar_results = run_mljar(X, y, mljar_root, total_time_limit)
            mljar_inventory_path = inspect_artifacts("MLJAR", mljar_results, mljar_root)
            mljar_inventory = _read_json(mljar_inventory_path)["inventory"]

            clean_root = dataset_root / "trustguard_clean"
            run_trustguard(
                X,
                y,
                clean_root,
                config=TrustGuardConfig(
                    task_type="auto",
                    command=f"OpenML task {task_id}: TrustGuard clean audit",
                ),
            )
            trust_inventory_path = inspect_artifacts(
                "TrustGuard", clean_root, dataset_root / "inventories"
            )
            trust_inventory = _read_json(trust_inventory_path)["inventory"]

            injected_X = X.copy()
            injected_X["__leak_target_copy__"] = y.to_numpy()
            injected_X["__leak_record_id__"] = range(len(injected_X))
            injected_root = dataset_root / "trustguard_injected"
            run_trustguard(
                injected_X,
                y,
                injected_root,
                config=TrustGuardConfig(
                    task_type="auto",
                    command=f"OpenML task {task_id}: TrustGuard injected-leakage audit",
                ),
            )
            injected = _read_json(injected_root / "leakage_audit" / "summary.json")
            codes = {item["code"] for item in injected.get("findings", [])}
            detected = {"target_copy", "identifier_like_column"}.issubset(codes)

            rows.append({
                "dataset": name,
                "task_id": task_id,
                "status": "success",
                "source": source,
                "mljar_explanation": _present(mljar_inventory, _EXPLANATION_KEYS),
                "mljar_trust_artifacts": _present(mljar_inventory, _TRUST_KEYS),
                "trustguard_artifacts": _present(trust_inventory, _TRUST_KEYS),
                "injected_leakage_detected": detected,
                "injected_finding_codes": sorted(codes),
                "interpretation": (
                    "MLJAR 解释工件与 TrustGuard 可信性工件互补；注入的目标复制和 ID 风险均被识别。"
                    if detected else "注入风险未全部命中，需要检查数据类型或审计规则。"
                ),
            })
            representative_run = representative_run or clean_root
        except Exception as exc:
            rows.append({
                "dataset": name,
                "task_id": task_id,
                "status": "error",
                "mljar_explanation": "运行失败",
                "mljar_trust_artifacts": "运行失败",
                "trustguard_artifacts": "运行失败",
                "injected_leakage_detected": False,
                "interpretation": f"{type(exc).__name__}: {exc}",
            })

    success_count = sum(row["status"] == "success" for row in rows)
    summary = {
        "suite": manifest.get("suite", "custom"),
        "dataset_count": len(rows),
        "success_count": success_count,
        "failure_count": len(rows) - success_count,
        "total_time_limit_seconds_per_dataset": total_time_limit,
        "comparison_scope": "解释性与可信性工件完整性，不作为模型性能排名。",
        "rows": rows,
        "interpretation": (
            "MLJAR 的 Explain 模式用于提供模型级解释基线；TrustGuard 补充泄露审计、"
            "测试访问日志、复现实验证据和漂移/覆盖审计。泄露注入列验证审计规则是否真正生效。"
        ),
    }
    summary_path = root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if representative_run is not None:
        report_data = _load_trustguard_report_data(representative_run)
        report_data["benchmark"] = summary
        write_trustguard_report(report_data, root, pdf_report=pdf_report)
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLJAR 与 TrustGuard 的 CC-18 工件对比实验")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="./outputs/trustguard-benchmark")
    parser.add_argument("--total-time-limit", type=int, default=120)
    parser.add_argument("--pdf-report", action="store_true")
    args = parser.parse_args()
    print(run(args.manifest, args.output_dir, total_time_limit=args.total_time_limit, pdf_report=args.pdf_report))
