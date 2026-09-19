"""Dataset-agnostic CLI for the standalone safe model selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from leakage_audit.safe_model_search import safe_search_models
from tabularx import build_profile, preprocess, split_data


_BUILTINS = {
    "breast_cancer": (load_breast_cancer, "classification"),
    "iris": (load_iris, "classification"),
    "diabetes": (load_diabetes, "regression"),
}


def load_dataset(dataset: str, csv_path: str | None, target: str | None):
    """Load one bundled sklearn dataset or a user-provided CSV table."""
    if csv_path:
        if not target:
            raise ValueError("使用 --csv 时必须同时提供 --target 列名。")
        frame = pd.read_csv(csv_path)
        if target not in frame.columns:
            raise ValueError(f"目标列 {target!r} 不在 CSV 中。可用列：{list(frame.columns)}")
        return frame.drop(columns=[target]), frame[target], "auto"
    loader, task = _BUILTINS[dataset]
    data = loader(as_frame=True)
    return data.data, data.target.rename("target"), task


def run(
    output_dir: str | Path,
    *,
    dataset: str = "breast_cancer",
    csv_path: str | None = None,
    target: str | None = None,
    models: list[str] | None = None,
    n_trials: int = 3,
) -> Path:
    X, y, task = load_dataset(dataset, csv_path, target)
    profile = build_profile(X, y, task_type=task)
    split = split_data(X, y, profile)
    processed = preprocess(X, y, profile, split)
    result = safe_search_models(processed, profile, model_names=models, n_trials=n_trials)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "safe_model_search.json"
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeakageAudit 独立安全模型搜索")
    parser.add_argument("--dataset", choices=sorted(_BUILTINS), default="breast_cancer")
    parser.add_argument("--csv", help="用户 CSV 数据集路径；使用时必须传 --target")
    parser.add_argument("--target", help="CSV 的目标列名")
    parser.add_argument("--models", nargs="+", default=None, help="候选模型；默认 random_forest xgboost lightgbm catboost")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--output-dir", default="./safe-search-demo-output")
    args = parser.parse_args()
    print(run(args.output_dir, dataset=args.dataset, csv_path=args.csv, target=args.target,
              models=args.models, n_trials=args.n_trials))
