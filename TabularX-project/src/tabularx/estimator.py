"""AutoEstimator：一键建模的高阶封装。

用法：
    ae = AutoEstimator(task="classification", output_dir="./output")
    ae.fit(X, y, sensitive_features=["gender"])
    print(ae.result.profile.overview())
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabularx.core.params import TabularXParams
from tabularx.core.profile import build_profile
from tabularx.core.split import split_data
from tabularx.core.preprocess import preprocess
from tabularx.core.learning import (
    search_models,
    build_voting,
    explain_model,
    evaluate_fairness,
    get_default_registry,
)
from tabularx.core.reports import (
    render_profile_report,
    render_split_report,
    render_preprocess_report,
    render_model_search_report,
    render_model_detail_report,
    render_voting_report,
    render_explanation_report,
)
from tabularx.schema.estimator import AutoResult


class AutoEstimator:
    """一键建模：配置 → 训练 → 解释 → 公平性评估。

    Attributes:
        task: 任务类型 "auto"（默认，自动推断）| "classification" | "regression"。
        output_dir: 报告和工件保存目录。
        param: TabularXParams 实例，配置各阶段参数。
        result_: fit() 完成后自动生成的 AutoResult 实例。
    """

    def __init__(
        self,
        task: str = "auto",
        output_dir: str | Path = "./tabularx_output",
    ):
        self.task = task
        self.output_dir = Path(output_dir)
        self.param = TabularXParams()
        self.result_: AutoResult | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: list[str] | None = None,
    ) -> AutoEstimator:
        """执行完整流水线：profile → split → preprocess → search → [explain] → [fairness]。

        Args:
            X: 特征 DataFrame。
            y: 目标 Series。
            sensitive_features: 可选，敏感特征原始列名列表（用于公平性评估）。

        Returns:
            self，便于链式调用。
        """
        # ── 1. Profile ────────────────────────────────────────
        profile = build_profile(X, y, task_type=self.task, config=self.param.profile)
        self.task = profile.target.task_type  # "auto" → 具体类型

        # ── 2. Split ──────────────────────────────────────────
        split = split_data(X, y, profile=profile, config=self.param.split)

        # ── 3. Preprocess ─────────────────────────────────────
        preprocessed = preprocess(
            X, y,
            profile=profile,
            split_result=split,
            config=self.param.preprocess,
        )

        # ── 4. Model Search ───────────────────────────────────
        searched = search_models(
            preprocessed_data=preprocessed,
            profile=profile,
            config=self.param.model_search,
        )

        # ── 5. Voting 组合 ────────────────────────────────────
        if searched.best_model_name:
            voting = build_voting(
                search_result=searched,
                preprocessed_data=preprocessed,
                task_type=self.task,
            )
            if voting is not None:
                searched.ensemble_result = voting

        # ── 6. SHAP 解释性 ────────────────────────────────────
        shap = None
        if searched.best_model_name:
            shap = self._run_shap(searched, preprocessed)

        # ── 7. 公平性评估 ─────────────────────────────────────
        fairness = None
        if sensitive_features and searched.best_model_name:
            fairness = self._run_fairness(
                searched, preprocessed, sensitive_features,
            )

        # ── 8. 组装结果 ───────────────────────────────────────
        self.result_ = AutoResult(
            profile=profile,
            split=split,
            preprocessed=preprocessed,
            searched=searched,
            shap=shap,
            fairness=fairness,
            output_dir=self.output_dir,
        )

        # ── 9. 保存报告 ───────────────────────────────────────
        self._save_reports()

        return self

    # ── 内部：运行 SHAP ─────────────────────────────────────

    def _run_shap(self, searched, preprocessed):
        """对最优模型执行 SHAP 可解释性分析。"""
        registry = get_default_registry()
        registry_task = (
            "classification"
            if self.task in ("binary", "multiclass")
            else "regression"
        )
        best_spec = registry.get(searched.best_model_name, registry_task)

        X_train_full = pd.concat([preprocessed.X_train, preprocessed.X_valid])
        y_train_full = pd.concat([preprocessed.y_train, preprocessed.y_valid])

        shap_dir = self.output_dir / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)

        return explain_model(
            spec=best_spec,
            best_params=searched.best_model_params,
            X_train=X_train_full,
            y_train=y_train_full,
            X_explain=preprocessed.X_test,
            feature_names=preprocessed.feature_names_out,
            task_type=self.task,
            output_dir=str(shap_dir),
            config=self.param.shap,
        )

    # ── 内部：运行公平性评估 ───────────────────────────────

    def _run_fairness(self, searched, preprocessed, sensitive_features):
        """对最优模型执行公平性评估。"""
        registry = get_default_registry()
        registry_task = (
            "classification"
            if self.task in ("binary", "multiclass")
            else "regression"
        )
        best_spec = registry.get(searched.best_model_name, registry_task)

        # 重建最优模型并预测
        model = best_spec.factory(
            **searched.best_model_params,
            **(best_spec.default_params or {}),
        )
        X_train_full = pd.concat([preprocessed.X_train, preprocessed.X_valid])
        y_train_full = pd.concat([preprocessed.y_train, preprocessed.y_valid])
        model.fit(X_train_full, y_train_full)

        y_pred = model.predict(preprocessed.X_test)
        y_prob = None
        if best_spec.supports_predict_proba and self.task == "binary":
            y_prob = model.predict_proba(preprocessed.X_test)[:, 1]

        return evaluate_fairness(
            X_test=preprocessed.X_test,
            y_test=preprocessed.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            column_mappings=preprocessed.column_mappings,
            sensitive_features=sensitive_features,
            task_type=self.task,
            config=self.param.fairness,
        )

    # ── 内部：保存报告 ─────────────────────────────────────

    def _save_reports(self) -> None:
        """将所有阶段报告渲染为 HTML 并写入 output_dir。"""
        result = self.result_
        output_dir = self.output_dir

        reports_dir = output_dir / "reports"
        models_dir = output_dir / "models"
        reports_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        target_name = _resolve_target_name(result.preprocessed)

        # ── 各阶段报告 ──────────────────────────────────────
        _write_html(reports_dir / "profile.html", render_profile_report(result.profile))
        _write_html(reports_dir / "split.html", render_split_report(result.split))
        _write_html(
            reports_dir / "preprocess.html",
            render_preprocess_report(result.preprocessed),
        )
        _write_html(
            reports_dir / "model_search.html",
            render_model_search_report(result.searched),
        )

        # ── 各模型详情 ──────────────────────────────────────
        for run in result.searched.succeeded_results:
            detail_data = {"run": run, "target_name": target_name}
            safe_name = run.model_name.replace(" ", "_")
            _write_html(
                models_dir / f"{safe_name}.html",
                render_model_detail_report(detail_data),
            )

        # ── Voting 报告 ─────────────────────────────────────
        if result.searched.ensemble_result:
            best_single_val = (result.searched.best_test_metrics or {}).get(
                result.searched.primary_metric, float("nan")
            )
            voting_data = {
                "run": result.searched.ensemble_result,
                "primary_metric": result.searched.primary_metric,
                "optimization_direction": result.searched.optimization_direction,
                "best_single_test": best_single_val,
                "is_classification": self.task in ("binary", "multiclass"),
                "succeeded_results": result.searched.succeeded_results,
            }
            _write_html(
                models_dir / "voting.html",
                render_voting_report(voting_data),
            )

        # ── 可解释性 + 公平性报告 ────────────────────────────
        explanation_data = {
            "shap_result": result.shap,
            "fairness_result": result.fairness,
            "task_type": self.task,
            "target_name": target_name,
        }
        _write_html(
            reports_dir / "explanation.html",
            render_explanation_report(explanation_data),
        )

        # 记录已生成路径
        result.report_paths = {
            "profile": str(reports_dir / "profile.html"),
            "split": str(reports_dir / "split.html"),
            "preprocess": str(reports_dir / "preprocess.html"),
            "model_search": str(reports_dir / "model_search.html"),
            "explanation": str(reports_dir / "explanation.html"),
        }

    # ── 属性访问 ───────────────────────────────────────────

    @property
    def result(self) -> AutoResult:
        """获取 fit() 的完整结果。"""
        if self.result_ is None:
            raise RuntimeError("请先调用 fit() 方法。")
        return self.result_

    def __repr__(self) -> str:
        status = "就绪" if self.result_ is None else "已完成"
        return (
            f"<AutoEstimator task={self.task!r} "
            f"output_dir={self.output_dir!s} {status}>"
        )


# ── 辅助函数 ────────────────────────────────────────────────


def _resolve_target_name(preprocessed) -> str:
    """从预处理数据中推断目标变量名。"""
    try:
        name = preprocessed.y_train.name
        if name is not None:
            return name
    except (AttributeError, IndexError):
        pass
    return "target"


def _write_html(path: Path, html: str) -> None:
    """将 HTML 字符串写入文件。"""
    path.write_text(html, encoding="utf-8")


__all__ = [
    "AutoEstimator",
]