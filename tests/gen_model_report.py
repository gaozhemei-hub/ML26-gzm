"""生成模型搜索报告的 Demo 脚本。

读取 sklearn 内置数据集，构造 PreprocessedData + DataProfile，
运行实际模型搜索（含 Optuna 调参），生成总览和详细报告到 .inbox/reports/。
所有成功模型均生成分类评估图（混淆矩阵 + ROC 曲线）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer

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

# ─── matplotlib 禁用 GUI 后端 ───────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


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


def _make_classification_data() -> tuple[PreprocessedData, DataProfile]:
    X, y = load_breast_cancer(return_X_y=True)
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

    classes = sorted(y_ser.unique())
    class_balance = {str(c): float((y_ser == c).mean()) for c in classes}

    profile = DataProfile(
        n_rows=n,
        n_cols=X_df.shape[1],
        columns=[],
        target=TargetFact(
            task_type="binary",
            n_classes=2,
            class_balance=class_balance,
            tags=[],
        ),
        fingerprint="breast_cancer_demo",
    )
    return ppd, profile


# ═══════════════════════════════════════════════════════════════
# 辅助：生成分类评估图
# ═══════════════════════════════════════════════════════════════


def _generate_classification_plots(
    ppd: PreprocessedData,
    spec,
    best_params: dict,
    task_type: str,
    model_name: str,
    assets_dir: Path,
) -> dict[str, str]:
    """生成混淆矩阵和 ROC 曲线图，保存到 assets_dir。

    Returns:
        图文件名到相对路径的映射，如 {"confusion_matrix": "assets/xgb_cm.png"}
    """
    if task_type == "regression":
        return {}

    params = dict(best_params)
    if spec.default_params:
        for k, v in spec.default_params.items():
            params.setdefault(k, v)

    model = spec.factory(**params)
    X_full = pd.concat([ppd.X_train, ppd.X_valid], axis=0)
    y_full = pd.concat([ppd.y_train, ppd.y_valid], axis=0)
    model.fit(X_full, y_full)

    y_test_pred = model.predict(ppd.X_test)
    y_test_prob = None
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(ppd.X_test)

    assets_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        ppd.y_test, y_test_pred, ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()
    cm_path = assets_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    paths["confusion_matrix"] = f"assets/{cm_path.name}"

    # ROC 曲线（二分类）
    if y_test_prob is not None and y_test_prob.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(ppd.y_test, y_test_prob[:, 1], ax=ax)
        ax.set_title(f"{model_name} - ROC Curve")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        fig.tight_layout()
        roc_path = assets_dir / f"{model_name}_roc_curve.png"
        fig.savefig(roc_path, dpi=120)
        plt.close(fig)
        paths["roc_curve"] = f"assets/{roc_path.name}"

    return paths


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════


def main():
    out_dir = Path(__file__).resolve().parent.parent / ".inbox" / "reports"
    models_dir = out_dir / "models"
    model_assets_dir = models_dir / "assets"

    for d in [out_dir, models_dir, model_assets_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. 准备数据 ────────────────────────────────────────
    print("加载数据 ...")
    ppd, profile = _make_classification_data()
    print(f"数据: {profile.n_rows} 行, {profile.n_cols} 列, 任务: {profile.target.task_type}")

    # ── 2. 模型搜索 ────────────────────────────────────────
    config = ModelSearchConfig(
        models=["random_forest", "xgboost"],
        n_trials=5,
        n_jobs=1,
        retrain_on_train_valid=True,
        fixed_params={"random_forest": {"max_depth": 5}},
        search_space_overrides={"xgboost": {"n_estimators": {"low": 50, "high": 150}}},
    )

    from tabularx.core.learning.model_registry import get_default_registry
    registry = get_default_registry()

    print(f"开始模型搜索（{config.models}，每个 {config.n_trials} trials）...")
    result: ModelSearchResult = search_models(
        preprocessed_data=ppd, profile=profile, config=config, registry=registry,
    )
    print(f"  成功: {result.n_models_succeeded}, 失败: {result.n_models_failed}")
    print(f"  最佳模型: {result.best_model_name}")
    if result.best_test_metrics:
        print(f"  测试集 {result.primary_metric}: {result.best_test_metrics.get(result.primary_metric):.4f}")

    # ── 3. Voting ──────────────────────────────────────────
    if len(result.succeeded_results) >= 2:
        print("构建 Voting 组合 ...")
        voting_result = build_voting(
            search_result=result, preprocessed_data=ppd,
            task_type=profile.target.task_type, registry=registry,
        )
        result.ensemble_result = voting_result
        if voting_result and voting_result.status == "success":
            print(f"  Voting 验证集 {result.primary_metric}: "
                  f"{voting_result.validation_metrics.get(result.primary_metric):.4f}")

    # ── 4. 为所有成功模型生成分类评估图 ──────────────────
    is_classification = profile.target.task_type != "regression"
    if is_classification:
        registry_task = "classification"
        print("生成分类评估图 ...")
        for run in result.succeeded_results:
            spec = registry.get(run.model_name, registry_task)
            paths = _generate_classification_plots(
                ppd=ppd, spec=spec, best_params=run.best_params,
                task_type=profile.target.task_type,
                model_name=run.model_name, assets_dir=model_assets_dir,
            )
            run.artifact_paths = paths
            print(f"  [OK] {run.model_name} 图已生成")

    # ── 5. 构造扁平化总览数据（ModelSearchResult 字段 + 额外字段） ──
    is_classification = profile.target.task_type != "regression"

    has_cm = False
    has_roc = False
    artifact_dir = None
    for run in result.succeeded_results:
        if run.model_name == result.best_model_name and run.artifact_paths:
            artifact_dir = "models/assets"
            has_cm = "confusion_matrix" in run.artifact_paths
            has_roc = "roc_curve" in run.artifact_paths
            break

    overview_data = dict(result.__dict__)
    overview_data["is_classification"] = is_classification
    overview_data["artifact_dir"] = artifact_dir
    overview_data["has_confusion_matrix"] = has_cm
    overview_data["has_roc_curve"] = has_roc

    # ── 6. 渲染总览报告 ────────────────────────────────────
    print("渲染总览报告 ...")
    overview_html = render_model_search_report(overview_data)
    overview_path = out_dir / "model_search.html"
    overview_path.write_text(overview_html, encoding="utf-8")
    print(f"  已保存: {overview_path}")

    # ── 7. 渲染各模型详情报告 ──────────────────────────────
    print("渲染模型详情报告 ...")
    all_runs = result.model_results + ([result.ensemble_result] if result.ensemble_result else [])
    for run in all_runs:
        if run is None:
            continue
        artifact_dir_detail = "assets" if run.artifact_paths else None
        has_cm_d = "confusion_matrix" in run.artifact_paths if run.artifact_paths else False
        has_roc_d = "roc_curve" in run.artifact_paths if run.artifact_paths else False

        detail_data = {
            "run": run,
            "primary_metric": result.primary_metric,
            "optimization_direction": result.optimization_direction,
            "is_classification": is_classification,
            "artifact_dir": artifact_dir_detail,
            "has_confusion_matrix": has_cm_d,
            "has_roc_curve": has_roc_d,
        }
        detail_html = render_model_detail_report(detail_data)
        detail_path = models_dir / f"{run.model_name}.html"
        detail_path.write_text(detail_html, encoding="utf-8")
        print(f"  [OK] {run.model_name}.html")

    # ── 8. 失败模型 ────────────────────────────────────────
    failed_dir = models_dir / "_failed"
    for run in result.failed_results:
        failed_dir.mkdir(parents=True, exist_ok=True)
        html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>失败模型 — {run.model_name}</title><style>
body {{ font-family: sans-serif; margin: 2rem; color: #333; }}
h1 {{ color: #c0392b; border-bottom: 2px solid #e74c3c; }}
.error-box {{ background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 1rem; }}
</style></head><body>
<h1>模型调参失败：{run.model_name}</h1>
<div class="error-box"><pre>{run.error}</pre></div>
</body></html>"""
        (failed_dir / f"{run.model_name}.html").write_text(html, encoding="utf-8")

    print(f"\n全部完成！报告目录: {out_dir}")
    print(f"  总览: {overview_path}")
    print(f"  详情: {models_dir}/")


if __name__ == "__main__":
    main()