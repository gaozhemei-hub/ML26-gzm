"""模型评估可视化：混淆矩阵、ROC 曲线、调参轨迹图。

每个函数接收必要的模型和数据，将图片保存到 assets_dir 并更新 artifact_paths。
不直接依赖 AutoEstimator，只依赖 schema 类型和 sklearn 模型。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.ensemble import VotingClassifier, VotingRegressor

from tabularx.core.learning.model_registry import get_default_registry
from tabularx.schema.learning import ModelSearchResult, ModelRunResult
from tabularx.schema.preprocess import PreprocessedData


def save_model_plots(
    run: ModelRunResult,
    assets_dir: str | Path,
    search_result: ModelSearchResult,
    preprocessed: PreprocessedData,
    task_type: str,
    registry=None,
    fitted_model=None,
) -> None:
    """为单个模型生成混淆矩阵和 ROC 曲线，保存到 assets_dir 并写入 run.artifact_paths。

    Args:
        run: 模型运行结果（其 best_params 用于重建模型）。
        assets_dir: 图片保存目录。
        search_result: 模型搜索结果（用于获取 primary_metric 等）。
        preprocessed: 预处理后的数据。
        task_type: 任务类型 "binary" / "multiclass" / "regression"。
        registry: 可选模型注册表，默认使用全局单例。
    """
    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    reg = registry or get_default_registry()
    reg_task = "classification" if task_type in ("binary", "multiclass") else "regression"

    is_classification = task_type in ("binary", "multiclass")
    safe_name = run.model_name.replace(" ", "_")

    try:
        spec = reg.get(run.model_name, reg_task)
    except Exception:
        return

    # 重建模型
    model = fitted_model
    X_full = pd.concat([preprocessed.X_train, preprocessed.X_valid])
    y_full = pd.concat([preprocessed.y_train, preprocessed.y_valid])
    if model is None:
        model = spec.factory(**{**(spec.default_params or {}), **run.best_params})
        model.fit(X_full, y_full)
    y_pred = model.predict(preprocessed.X_test)

    if is_classification:
        # 混淆矩阵
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            preprocessed.y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{run.model_name} - Confusion Matrix")
        fig.tight_layout()
        fig.savefig(assets_path / f"{safe_name}_confusion_matrix.png", dpi=120)
        plt.close(fig)
        run.artifact_paths["confusion_matrix"] = f"assets/{safe_name}_confusion_matrix.png"

        # ROC 曲线（仅二分类）
        if spec.supports_predict_proba and task_type == "binary":
            try:
                y_prob = model.predict_proba(preprocessed.X_test)[:, 1]
            except Exception:
                y_prob = None
            if y_prob is not None:
                fig, ax = plt.subplots(figsize=(5, 4))
                RocCurveDisplay.from_predictions(preprocessed.y_test, y_prob, ax=ax)
                ax.set_title(f"{run.model_name} - ROC Curve")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                fig.tight_layout()
                fig.savefig(assets_path / f"{safe_name}_roc_curve.png", dpi=120)
                plt.close(fig)
                run.artifact_paths["roc_curve"] = f"assets/{safe_name}_roc_curve.png"


def save_voting_plots(
    ensemble_run: ModelRunResult,
    assets_dir: str | Path,
    search_result: ModelSearchResult,
    preprocessed: PreprocessedData,
    task_type: str,
    registry=None,
) -> None:
    """为 Voting 组合模型生成混淆矩阵和 ROC 曲线。

    Note:
        由于 Voting 模型本身未持久化，此处按成功模型重建 VotingClassifier/Regressor。
    """
    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    reg = registry or get_default_registry()
    reg_task = "classification" if task_type in ("binary", "multiclass") else "regression"
    is_classification = task_type in ("binary", "multiclass")

    succeeded = [r for r in search_result.succeeded_results if r.status == "success"]
    if len(succeeded) < 2:
        return

    try:
        estimators = []
        for r in succeeded:
            spec = reg.get(r.model_name, reg_task)
            m = spec.factory(**r.best_params, **(spec.default_params or {}))
            Xf = pd.concat([preprocessed.X_train, preprocessed.X_valid])
            yf = pd.concat([preprocessed.y_train, preprocessed.y_valid])
            m.fit(Xf, yf)
            estimators.append((r.model_name, m))

        if is_classification:
            voting = VotingClassifier(
                estimators=estimators,
                voting="soft" if task_type == "binary" else "hard")
        else:
            voting = VotingRegressor(estimators=estimators)

        X_full = pd.concat([preprocessed.X_train, preprocessed.X_valid])
        y_full = pd.concat([preprocessed.y_train, preprocessed.y_valid])
        voting.fit(X_full, y_full)
        y_pred = voting.predict(preprocessed.X_valid)

        if is_classification:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                preprocessed.y_valid, y_pred, ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Voting - Validation Confusion Matrix")
            fig.tight_layout()
            fig.savefig(assets_path / "voting_confusion_matrix.png", dpi=120)
            plt.close(fig)
            ensemble_run.artifact_paths["confusion_matrix"] = "assets/voting_confusion_matrix.png"

            if task_type == "binary":
                try:
                    y_prob = voting.predict_proba(preprocessed.X_valid)[:, 1]
                except Exception:
                    y_prob = None
                if y_prob is not None:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    RocCurveDisplay.from_predictions(preprocessed.y_valid, y_prob, ax=ax)
                    ax.set_title("Voting - Validation ROC Curve")
                    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(assets_path / "voting_roc_curve.png", dpi=120)
                    plt.close(fig)
                    ensemble_run.artifact_paths["roc_curve"] = "assets/voting_roc_curve.png"

        # 候选模型与 Voting 对比统一使用验证集指标，避免测试/验证口径混用。
        try:
            primary = search_result.primary_metric
            labels = []
            values = []
            colors = []
            for r in search_result.succeeded_results:
                if primary not in r.validation_metrics:
                    continue
                labels.append(r.model_name)
                values.append(r.validation_metrics[primary])
                colors.append("#3498db")
            if primary in ensemble_run.validation_metrics:
                labels.append("Voting")
                values.append(ensemble_run.validation_metrics[primary])
                colors.append("#e67e22")

            if labels:
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
                ax.set_ylabel(primary)
                ax.set_title("Validation Model Comparison")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.4f}", ha="center", va="bottom", fontsize=7)
                fig.tight_layout()
                fig.savefig(assets_path / "voting_comparison.png", dpi=120)
                plt.close(fig)
                ensemble_run.artifact_paths["voting_comparison"] = "assets/voting_comparison.png"
        except Exception:
            pass
    except Exception:
        pass


def save_trial_trajectory_plot(
    run: ModelRunResult,
    assets_dir: str | Path,
    primary_metric: str,
) -> None:
    """为单个模型生成调参轨迹散点图。

    Args:
        run: 模型运行结果（需含 optuna_trials）。
        assets_dir: 图片保存目录。
        primary_metric: 主指标名（用作 y 轴标签）。
    """
    if not run.optuna_trials:
        return
    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    safe_name = run.model_name.replace(" ", "_")

    trials = sorted(run.optuna_trials, key=lambda t: t.trial_number)
    nums = [t.trial_number for t in trials]
    vals = [t.value for t in trials]
    best_val = run.validation_metrics.get(primary_metric, 0)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(nums, vals, c="#3498db", s=20, alpha=0.6, label="Trial")
    ax.axhline(y=best_val, color="#e74c3c", linestyle="--", alpha=0.7,
               label=f"Best: {best_val:.4f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel(primary_metric)
    ax.set_title(f"{run.model_name} - Trial Trajectory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(assets_path / f"{safe_name}_trial_trajectory.png", dpi=120)
    plt.close(fig)
    run.artifact_paths["trial_trajectory"] = f"assets/{safe_name}_trial_trajectory.png"


__all__ = [
    "save_model_plots",
    "save_voting_plots",
    "save_trial_trajectory_plot",
]
