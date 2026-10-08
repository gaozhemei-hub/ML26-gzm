"""Run the four TrustGuard modules without modifying TabularX source code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from trustguard.pipeline import TrustGuardConfig, run_trustguard


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


def run(output_dir: str | Path, *, dataset: str = "breast_cancer", csv_path: str | None = None, target: str | None = None, group_column: str | None = None, time_column: str | None = None, sensitive_features: list[str] | None = None, pdf_report: bool = False, model_result_path: str | None = None, benchmark_summary_path: str | None = None) -> dict[str, str]:
    X, y, task = load_data(dataset, csv_path, target)
    dataset_labels = {
        "breast_cancer": "sklearn 乳腺癌数据集",
        "iris": "sklearn 鸢尾花数据集",
        "diabetes": "sklearn 糖尿病数据集",
    }
    config = TrustGuardConfig(
        task_type=task,
        group_column=group_column,
        time_column=time_column,
        sensitive_features=tuple(sensitive_features or ()),
        pdf_report=pdf_report,
        command="python -m trustguard.demo_trustguard ...",
        dataset_name=Path(csv_path).stem if csv_path else dataset_labels[dataset],
        dataset_source=str(Path(csv_path).resolve()) if csv_path else f"scikit-learn:{dataset}",
        model_result_path=model_result_path,
        benchmark_summary_path=benchmark_summary_path,
    )
    return run_trustguard(X, y, output_dir, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustGuard 四模块独立演示")
    parser.add_argument("--dataset", choices=["breast_cancer", "iris", "diabetes"], default="breast_cancer")
    parser.add_argument("--csv")
    parser.add_argument("--target")
    parser.add_argument("--group-column")
    parser.add_argument("--time-column")
    parser.add_argument("--sensitive-features", nargs="*", default=[])
    parser.add_argument("--output-dir", default="./trustguard-output")
    parser.add_argument("--pdf-report", action="store_true", help="同时生成 reports/trustguard.pdf（需要 WeasyPrint）")
    parser.add_argument("--model-result", help="可选：安全模型搜索生成的 model_search_result.json")
    parser.add_argument("--benchmark-summary", help="可选：MLJAR/CC-18 benchmark_summary.json，用于报告第 5 节")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, dataset=args.dataset, csv_path=args.csv, target=args.target, group_column=args.group_column, time_column=args.time_column, sensitive_features=args.sensitive_features, pdf_report=args.pdf_report, model_result_path=args.model_result, benchmark_summary_path=args.benchmark_summary), ensure_ascii=False, indent=2))
