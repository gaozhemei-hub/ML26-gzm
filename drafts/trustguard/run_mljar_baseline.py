"""Run MLJAR as the reproducible baseline for the TrustGuard artifact study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from trustguard.benchmark import inspect_artifacts


def load_data(dataset: str, csv_path: str | None, target: str | None, openml_task_id: int | None):
    if csv_path:
        if not target:
            raise ValueError("使用 --csv 时必须提供 --target。")
        frame = pd.read_csv(csv_path)
        if target not in frame:
            raise ValueError(f"目标列 {target!r} 不在 CSV 中。")
        return frame.drop(columns=[target]), frame[target], {"source": str(csv_path), "target": target}
    if openml_task_id is not None:
        try:
            import openml
        except ImportError as exc:
            raise ImportError("OpenML 任务需要安装 openml。") from exc
        task = openml.tasks.get_task(openml_task_id)
        try:
            X, y = task.get_X_and_y(dataset_format="dataframe")
        except TypeError:
            X, y = task.get_X_and_y()
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = y if isinstance(y, pd.Series) else pd.Series(y, name="target")
        return X, y, {"source": "OpenML", "task_id": openml_task_id, "target": y.name}
    loaders = {"breast_cancer": load_breast_cancer, "iris": load_iris, "diabetes": load_diabetes}
    data = loaders[dataset](as_frame=True)
    return data.data, data.target.rename("target"), {"source": f"sklearn:{dataset}", "target": "target"}


def run_mljar(X: pd.DataFrame, y: pd.Series, output_dir: str | Path, total_time_limit: int) -> Path:
    try:
        from supervised.automl import AutoML
    except ImportError as exc:
        raise ImportError("请先安装：python -m pip install -r trustguard/requirements-benchmark.txt") from exc
    destination = Path(output_dir)
    results_path = destination / "mljar_results"
    destination.mkdir(parents=True, exist_ok=True)
    automl = AutoML(results_path=str(results_path), mode="Explain", total_time_limit=total_time_limit)
    automl.fit(X, y)
    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLJAR 基线：可解释性工件对比")
    parser.add_argument("--dataset", choices=["breast_cancer", "iris", "diabetes"], default="breast_cancer")
    parser.add_argument("--csv")
    parser.add_argument("--target")
    parser.add_argument("--openml-task-id", type=int)
    parser.add_argument("--total-time-limit", type=int, default=120, help="单数据集 MLJAR 时间预算（秒）")
    parser.add_argument("--output-dir", default="./mljar-baseline-output")
    args = parser.parse_args()
    X, y, source = load_data(args.dataset, args.csv, args.target, args.openml_task_id)
    results = run_mljar(X, y, args.output_dir, args.total_time_limit)
    out = Path(args.output_dir)
    metadata = {"tool": "MLJAR AutoML", "mode": "Explain", "total_time_limit_seconds": args.total_time_limit, "source": source, "n_rows": len(X), "n_features": X.shape[1]}
    (out / "mljar_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    inventory = inspect_artifacts("MLJAR", results, out)
    print(json.dumps({"results": str(results), "metadata": str(out / "mljar_run_metadata.json"), "inventory": str(inventory)}, ensure_ascii=False))
