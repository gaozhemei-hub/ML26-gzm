"""一条命令完成四模型安全搜索、四类审计和 HTML/PDF 综合报告。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabularx import build_profile, preprocess, split_data

from trustguard.demo_trustguard import load_data
from trustguard.leakage import safe_search_models
from trustguard.pipeline import TrustGuardConfig, run_trustguard
from trustguard.protocol import EvaluationProtocolGuard


_DATASET_LABELS = {
    "breast_cancer": "sklearn 乳腺癌数据集",
    "iris": "sklearn 鸢尾花数据集",
    "diabetes": "sklearn 糖尿病数据集",
}


def run(
    output_dir: str | Path,
    *,
    dataset: str = "breast_cancer",
    csv_path: str | None = None,
    target: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    sensitive_features: list[str] | None = None,
    n_trials: int = 5,
    pdf_report: bool = False,
    benchmark_summary_path: str | None = None,
) -> dict[str, str]:
    """运行完整实验，并让报告直接消费真实模型与审计结果。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    X, y, task = load_data(dataset, csv_path, target)

    # 复用既有 TabularX 的画像、划分和预处理，不修改其实现。
    profile = build_profile(X, y, task_type=task)
    split = split_data(X, y, profile)
    prepared = preprocess(X, y, profile, split)

    guard = EvaluationProtocolGuard()
    search = safe_search_models(
        prepared,
        profile,
        model_names=["random_forest", "xgboost", "lightgbm", "catboost"],
        n_trials=n_trials,
        protocol_guard=guard,
    )
    model_result_path = root / "model_search_result.json"
    model_result_path.write_text(
        json.dumps(search.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    dataset_name = Path(csv_path).stem if csv_path else _DATASET_LABELS[dataset]
    dataset_source = str(Path(csv_path).resolve()) if csv_path else f"scikit-learn:{dataset}"
    config = TrustGuardConfig(
        task_type=task,
        group_column=group_column,
        time_column=time_column,
        sensitive_features=tuple(sensitive_features or ()),
        pdf_report=pdf_report,
        command="python -m trustguard.run_full_experiment ...",
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        model_result_path=str(model_result_path),
        benchmark_summary_path=benchmark_summary_path,
    )
    paths = run_trustguard(X, y, root, config=config, protocol_guard=guard)
    paths["model_result"] = str(model_result_path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustGuard 四模型与可信性审计完整实验")
    parser.add_argument("--dataset", choices=list(_DATASET_LABELS), default="breast_cancer")
    parser.add_argument("--csv")
    parser.add_argument("--target")
    parser.add_argument("--group-column")
    parser.add_argument("--time-column")
    parser.add_argument("--sensitive-features", nargs="*", default=[])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--benchmark-summary", help="可选：MLJAR 对比实验的 benchmark_summary.json")
    parser.add_argument("--pdf-report", action="store_true")
    parser.add_argument("--output-dir", default="./outputs/trustguard-full-experiment")
    args = parser.parse_args()
    result = run(
        args.output_dir,
        dataset=args.dataset,
        csv_path=args.csv,
        target=args.target,
        group_column=args.group_column,
        time_column=args.time_column,
        sensitive_features=args.sensitive_features,
        n_trials=args.n_trials,
        pdf_report=args.pdf_report,
        benchmark_summary_path=args.benchmark_summary,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
