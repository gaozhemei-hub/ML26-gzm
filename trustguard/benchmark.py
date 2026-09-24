"""Reusable metadata and artifact inspection for the MLJAR comparison study."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DatasetSpec:
    name: str
    source: str
    target: str | None = None
    openml_task_id: int | None = None
    notes: str = ""


def create_cc18_manifest(output_path: str | Path, limit: int = 5) -> Path:
    """Fetch a reproducible list of CC-18 tasks; requires optional ``openml``."""
    try:
        import openml
    except ImportError as exc:
        raise ImportError("请安装 openml：python -m pip install openml") from exc
    suite = openml.study.get_suite("OpenML-CC18")
    specs = [DatasetSpec(name=f"openml_task_{int(task_id)}", source="OpenML-CC18", openml_task_id=int(task_id), notes="Use the task ID to preserve dataset/target provenance.") for task_id in suite.tasks[:limit]]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"suite": "OpenML-CC18", "purpose": "Artifact-generation comparison; performance is secondary.", "datasets": [asdict(s) for s in specs]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_benchmark_plan(datasets: Iterable[DatasetSpec], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "benchmark_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "research_question": "Do explanatory AutoML artifacts also document whether an experiment is trustworthy?",
        "tools": ["MLJAR AutoML", "original TabularX", "TabularX + TrustGuard"],
        "datasets": [asdict(s) for s in datasets],
        "required_artifacts": ["leaderboard_or_model_scores", "feature_importance", "shap", "confusion_matrix_or_metrics", "leakage_audit", "test_access_log", "reproducibility_ledger", "distribution_audit"],
        "method_note": "Use the same input table and fixed recorded settings. This compares report artifacts; do not claim identical predictive-score protocols across tools unless their validation strategies are harmonized.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def inspect_artifacts(tool_name: str, run_dir: str | Path, output_dir: str | Path) -> Path:
    """Create a transparent artifact-presence inventory from a completed run."""
    root = Path(run_dir)
    names = [path.name.lower() for path in root.rglob("*") if path.is_file()]
    patterns = {
        "report": ("report", ".html", ".md"), "feature_importance": ("importance",),
        "shap": ("shap",), "confusion_matrix": ("confusion",), "learning_curve": ("learning", "curve"),
        "fairness": ("fair",), "leakage_audit": ("leakage",), "test_access_log": ("test_access", "protocol"),
        "reproducibility_ledger": ("ledger", "manifest", "fingerprint"), "distribution_audit": ("drift", "distribution", "coverage"),
    }
    inventory = {label: any(any(token in name for token in tokens) for name in names) for label, tokens in patterns.items()}
    path = Path(output_dir) / f"{tool_name.lower().replace(' ', '_')}_artifact_inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tool": tool_name, "run_dir": str(root), "inventory": inventory, "files_scanned": len(names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
