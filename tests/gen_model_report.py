"""生成模型搜索报告的 Demo 脚本。

读取 sklearn 内置数据集，构造 PreprocessedData + DataProfile，
运行实际模型搜索（含 Optuna 调参），生成总览和详细报告到 .inbox/reports/。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# 确保能找到 tabularx
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TabularX" / "src"))

from tabularx.core.params import ModelSearchConfig
from tabularx.core.learning import (
    search_models,
    build_voting,
)
from tabularx.reports import render_model_search_report, render_model_detail_report
from tabularx.schema.learning import ModelSearchResult
from tabularx.schema.preprocess import PreprocessedData
from tabularx.schema.profile import DataProfile, TargetFact


# ═══════════════════════════════════════════════════════════════
# 辅助：构造 PreprocessedData + DataProfile
# ═══════════════════════════════════════════════════════════════


def _make_regression_data() -> tuple[PreprocessedData, DataProfile]:
    X, y = load_diabetes(return_X_y=True)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")

    n = len(X_df)
    split = int(n * 0.6), int(n * 0.8)

    ppd = PreprocessedData(
        X_train=X_df.iloc[: split[0]],
        y_train=y_ser.iloc[: split[0]],
        X_valid=X_df.iloc[split[0]: split[1]],
        y_valid=y_ser.iloc[split[0]: split[1]],
        X_test=X_df.iloc[split[1]:],
        y_test=y_ser.iloc[split[1]:],
        n_features_in=X_df.shape[1],
        n_features_out=X_df.shape[1],
        dropped_columns=[],
        feature_names_out=list(X_df.columns),
    )

    profile = DataProfile(
        n_rows=n,
        n_cols=X_df.shape[1],
        columns=[],
        target=TargetFact(
            task_type="regression",
            mean=float(y_ser.mean()),
            std=float(y_ser.std()),
        ),
        fingerprint="diabetes_demo",
    )

    return ppd, profile


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    out_dir = Path(__file__).resolve().parent.parent / ".inbox" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 准备数据 ────────────────────────────────────────
    print("加载数据 ...")
    ppd, profile = _make_regression_data()
    print(f"数据: {profile.n_rows} 行, {profile.n_cols} 列, 任务: {profile.target.task_type}")

    # ── 2. 模型搜索 ────────────────────────────────────────
    config = ModelSearchConfig(
        models=["random_forest", "xgboost"],
        n_trials=5,
        n_jobs=1,
        retrain_on_train_valid=True,
        fixed_params={
            "random_forest": {"max_depth": 5},
        },
        search_space_overrides={
            "xgboost": {"n_estimators": {"low": 50, "high": 150}},
        },
    )

    print(f"开始模型搜索（{config.models}，每个 {config.n_trials} trials）...")
    result: ModelSearchResult = search_models(
        preprocessed_data=ppd,
        profile=profile,
        config=config,
    )
    print(f"  成功: {result.n_models_succeeded}, 失败: {result.n_models_failed}")
    print(f"  最佳模型: {result.best_model_name}")
    if result.best_test_metrics:
        print(f"  测试集 {result.primary_metric}: {result.best_test_metrics.get(result.primary_metric):.4f}")

    # ── 3. Voting 组合（如果有多于 1 个成功模型） ─────────
    if len(result.succeeded_results) >= 2:
        print("构建 Voting 组合 ...")
        voting_result = build_voting(
            search_result=result,
            preprocessed_data=ppd,
            task_type=profile.target.task_type,
        )
        result.ensemble_result = voting_result
        if voting_result and voting_result.status == "success":
            print(f"  Voting 验证集 {result.primary_metric}: "
                  f"{voting_result.validation_metrics.get(result.primary_metric):.4f}")

    # ── 4. 渲染总览报告 ────────────────────────────────────
    print("渲染总览报告 ...")
    overview_html = render_model_search_report(result)
    overview_path = out_dir / "model_search.html"
    overview_path.write_text(overview_html, encoding="utf-8")
    print(f"  已保存: {overview_path}")

    # ── 5. 渲染各模型详情报告 ──────────────────────────────
    print("渲染模型详情报告 ...")
    all_runs = result.model_results + ([result.ensemble_result] if result.ensemble_result else [])
    for run in all_runs:
        if run is None:
            continue

        # 构造传递给详情模板的上下文：包含排名和最佳模型信息
        detail_data = {
            "run": run,
            "primary_metric": result.primary_metric,
            "optimization_direction": result.optimization_direction,
            "best_model_name": result.best_model_name,
            "best_validation_metrics": result.best_validation_metrics,
        }

        detail_html = render_model_detail_report(detail_data)
        detail_path = models_dir / f"{run.model_name}.html"
        detail_path.write_text(detail_html, encoding="utf-8")
        status_icon = "✅" if run.status == "success" else "❌"
        print(f"  {status_icon} {run.model_name}.html")

    # ── 6. 生成失败模型的简要报告 ──────────────────────────
    failed_dir = models_dir / "_failed"
    for run in result.failed_results:
        failed_dir.mkdir(parents=True, exist_ok=True)
        html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>失败模型 — {run.model_name}</title><style>
body {{ font-family: sans-serif; margin: 2rem; color: #333; }}
h1 {{ color: #c0392b; border-bottom: 2px solid #e74c3c; }}
.error-box {{ background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 1rem; }}
</style></head><body>
<h1>❌ 模型调参失败：{run.model_name}</h1>
<div class="error-box"><pre>{run.error}</pre></div>
</body></html>"""
        (failed_dir / f"{run.model_name}.html").write_text(html, encoding="utf-8")

    print(f"\n全部完成！报告目录: {out_dir}")
    print(f"  总览: {overview_path}")
    print(f"  详情: {models_dir}/")


if __name__ == "__main__":
    main()