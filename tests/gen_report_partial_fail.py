"""生成部分模型失败的场景（Breast Cancer + 故意过少的 trial）。

演示：当一个或多个模型失败时的报告展示。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TabularX" / "src"))

from tabularx.core.params import ModelSearchConfig
from tabularx.core.learning import search_models, build_voting
from tabularx.core.learning.model_registry import get_default_registry
from tabularx.core.reports import render_model_search_report, render_model_detail_report
from tabularx.schema.learning import ModelSearchResult
from tabularx.schema.preprocess import PreprocessedData
from tabularx.schema.profile import DataProfile, TargetFact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def _make_data() -> tuple[PreprocessedData, DataProfile]:
    X, y = load_breast_cancer(return_X_y=True)
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
    classes = sorted(y_ser.unique())
    cb = {str(c): float((y_ser == c).mean()) for c in classes}
    profile = DataProfile(n_rows=n, n_cols=X_df.shape[1], columns=[],
        target=TargetFact(task_type="binary", n_classes=2, class_balance=cb, tags=[]),
        fingerprint="bc_demo")
    return ppd, profile


def _gen_plots(ppd, spec, params, task_type, name, assets_dir):
    params2 = dict(params)
    if spec.default_params:
        for k, v in spec.default_params.items(): params2.setdefault(k, v)
    model = spec.factory(**params2)
    Xf = pd.concat([ppd.X_train, ppd.X_valid]); yf = pd.concat([ppd.y_train, ppd.y_valid])
    model.fit(Xf, yf)
    yp = model.predict(ppd.X_test); yprob = model.predict_proba(ppd.X_test) if hasattr(model, "predict_proba") else None
    assets_dir.mkdir(parents=True, exist_ok=True); paths = {}
    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(ppd.y_test, yp, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} - Confusion Matrix"); fig.tight_layout()
    p = assets_dir / f"{name}_confusion_matrix.png"; fig.savefig(p, dpi=120); plt.close(fig)
    paths["confusion_matrix"] = f"assets/{p.name}"
    if yprob is not None and yprob.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5,4))
        RocCurveDisplay.from_predictions(ppd.y_test, yprob[:,1], ax=ax)
        ax.set_title(f"{name} - ROC Curve"); ax.plot([0,1],[0,1],"k--",alpha=0.3); fig.tight_layout()
        p = assets_dir / f"{name}_roc_curve.png"; fig.savefig(p, dpi=120); plt.close(fig)
        paths["roc_curve"] = f"assets/{p.name}"
    return paths


def main():
    out_dir = Path(__file__).resolve().parent.parent / ".inbox" / "reports"
    models_dir = out_dir / "models"; model_assets = models_dir / "assets"
    for d in [out_dir, models_dir, model_assets]: d.mkdir(parents=True, exist_ok=True)
    report_name = "model_search_partial_fail"

    ppd, profile = _make_data()
    print(f"数据: {profile.n_rows} 行, {profile.n_cols} 列, 任务: {profile.target.task_type}")

    # 故意用极少的 trial 并指定不合理的 fixed_params 使某些模型失败
    config = ModelSearchConfig(
        models=["random_forest", "xgboost", "lightgbm", "catboost"],
        n_trials=2, n_jobs=2, retrain_on_train_valid=True,
        fixed_params={
            "random_forest": {"max_depth": -1},  # 故意给非法值，有些版本会报错
        },
    )
    registry = get_default_registry()
    result: ModelSearchResult = search_models(
        preprocessed_data=ppd, profile=profile, config=config, registry=registry)
    print(f"  成功: {result.n_models_succeeded}, 失败: {result.n_models_failed}, 最佳: {result.best_model_name}")

    if len(result.succeeded_results) >= 2:
        vr = build_voting(search_result=result, preprocessed_data=ppd,
                          task_type=profile.target.task_type, registry=registry)
        result.ensemble_result = vr

    for run in result.succeeded_results:
        spec = registry.get(run.model_name, "classification")
        paths = _gen_plots(ppd, spec, run.best_params, profile.target.task_type, run.model_name, model_assets)
        run.artifact_paths = paths
        print(f"  [图] {run.model_name}")

    has_cm = has_roc = False; artifact_dir = None
    for run in result.succeeded_results:
        if run.model_name == result.best_model_name and run.artifact_paths:
            artifact_dir = "models/assets"
            has_cm = "confusion_matrix" in run.artifact_paths
            has_roc = "roc_curve" in run.artifact_paths
            break
    overview_data = dict(result.__dict__)
    overview_data["is_classification"] = True
    overview_data["artifact_dir"] = artifact_dir
    overview_data["has_confusion_matrix"] = has_cm
    overview_data["has_roc_curve"] = has_roc
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
        ad = "assets" if run.artifact_paths else None
        hc = "confusion_matrix" in run.artifact_paths if run.artifact_paths else False
        hr = "roc_curve" in run.artifact_paths if run.artifact_paths else False
        detail_data = dict(run=run, primary_metric=result.primary_metric,
            optimization_direction=result.optimization_direction,
            is_classification=True, artifact_dir=ad, has_confusion_matrix=hc, has_roc_curve=hr)
        detail_html = render_model_detail_report(detail_data)
        (models_dir / f"{run.model_name}.html").write_text(detail_html, encoding="utf-8")
        print(f"  [OK] {run.model_name}.html")

    print(f"\n完成: {overview_path}")


if __name__ == "__main__":
    main()