"""Batch-run MLJAR artifact baselines for a manifest generated from OpenML-CC18."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustguard.benchmark import inspect_artifacts
from trustguard.run_mljar_baseline import load_data, run_mljar


def run(manifest_path: str | Path, output_dir: str | Path, total_time_limit: int) -> Path:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in manifest["datasets"]:
        task_id = spec.get("openml_task_id")
        name = spec["name"]
        destination = root / name
        try:
            X, y, source = load_data("breast_cancer", None, None, task_id)
            model_dir = run_mljar(X, y, destination, total_time_limit)
            inventory = inspect_artifacts("MLJAR", model_dir, destination)
            results.append({"name": name, "task_id": task_id, "status": "success", "source": source, "inventory": str(inventory)})
        except Exception as exc:  # continue so a single unavailable task does not erase the benchmark.
            results.append({"name": name, "task_id": task_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    path = root / "cc18_mljar_batch_summary.json"
    path.write_text(json.dumps({"manifest": str(manifest_path), "total_time_limit_seconds_per_dataset": total_time_limit, "runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 CC-18 上的 MLJAR 工件基线")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--total-time-limit", type=int, default=120)
    parser.add_argument("--output-dir", default="./cc18-mljar-baseline")
    args = parser.parse_args()
    print(run(args.manifest, args.output_dir, args.total_time_limit))
