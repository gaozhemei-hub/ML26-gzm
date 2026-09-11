"""生成回归任务模型搜索报告（Diabetes，全部 4 个模型）。

演示：回归 + rmse/mae/r2 指标 + 无分类图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TabularX" / "src"))

from tabularx.core.params import ModelSearchConfig
from tabularx.core.learning import search_models, build_voting
from tabularx.core.learning.model_registry import get_default_registry
from tabularx.reports import render_model_search_report, render_model_detail_report
from tabularx.schema.learning import ModelSearchResult
from tabularx.schema.preprocess import PreprocessedData
from tabularx.schema.profile import DataProfile, TargetFact


def _make_data() -> tuple[PreprocessedData, DataProfile]:
    X, y = load_diabetes(return_X_y=True)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_ser = pd.Series(y, name="target")
    n = len(X_df); s = int(n * 0.6), int(n * 0.8)
    ppd = PreprocessedData(
        X_train=X_df.iloc[:s[0]], y_train=y_ser.iloc[:s[0]],
        X_valid=X_df.iloc[s[0]:s[1]], y_valid=y_ser.iloc[s[0]:s[1]],
        X_test=X_df.iloc[s[1]:], y_test=y_ser.iloc[s[1]:],
        n_features_in=X_df.shape[1], n_features_out=X_df.shape[1],
        dropped_columns=[], feature_names_out=list(X_df.columns),
    )
    profile = DataProfile(n_rows=n, n_cols=X_df.shape[1], columns=[],
        target=TargetFact(task_type="regression", mean=float(y_ser.mean()), std=float(y_ser.std())),
        fingerprint="diabetes_demo")
    return ppd, profile


def main():
    out_dir = Path(__file__).resolve().parent.parent / ".inbox" / "reports"
    models_dir = out_dir / "models"; model_assets = models_dir / "assets"
    for d in [out_dir, models_dir, model_assets]: d.mkdir(parents=True, exist_ok=True)
    report_name = "model_search_regression"

    ppd, profile = _make_data()
    print(f"数据: {profile.n_rows} 行, {profile.n_cols} 列, 任务: {profile.target.task_type}")

    config = ModelSearchConfig(
        models=["random_forest", "xgboost", "lightgbm", "catboost"],
        n_trials=5, n_jobs=2, retrain_on_train_valid=True,
    )
    registry = get_default_registry()
    result: ModelSearchResult = search_models(
        preprocessed_data=ppd, profile=profile, config=config, registry=registry)
    print(f"  成功: {result.n_models_succeeded}, 失败: {result.n_models_failed}, 最佳: {result.best_model_name}")

    if len(result.succeeded_results) >= 2:
        vr = build_voting(search_result=result, preprocessed_data=ppd,
                          task_type=profile.target.task_type, registry=registry)
        result.ensemble_result = vr

    # 回归无分类图，所有模型的 artifact_paths 留空
    overview_data = dict(result.__dict__)
    overview_data["is_classification"] = False
    overview_data["artifact_dir"] = None
    overview_data["has_confusion_matrix"] = False
    overview_data["has_roc_curve"] = False
    overview_data["ranking"] = result.ranking
    overview_data["succeeded_results"] = result.succeeded_results
    overview_data["failed_results"] = result.failed_results

    overview_html = render_model_search_report(overview_data)
    overview_path = out_dir / f"{report_name}.html"
    overview_path.write_text(overview_html, encoding="utf-8")
    print(f"  总览: {overview_path}")

    all_runs = result.model_results + ([result.ensemble_result] if result.ensemble_result else [])
    for run in all_runs:
        if run is None: continue
        detail_data = dict(run=run, primary_metric=result.primary_metric,
            optimization_direction=result.optimization_direction,
            is_classification=False, artifact_dir=None,
            has_confusion_matrix=False, has_roc_curve=False)
        detail_html = render_model_detail_report(detail_data)
        (models_dir / f"{run.model_name}.html").write_text(detail_html, encoding="utf-8")
        print(f"  [OK] {run.model_name}.html")

    print(f"\n完成: {overview_path}")


if __name__ == "__main__":
    main()