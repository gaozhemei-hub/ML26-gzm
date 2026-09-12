"""AutoEstimator：一键建模的高阶封装。

用法：
    ae = AutoEstimator(task="classification", output_dir="./output")
    ae.fit(X, y, sensitive_features=["gender"])
    print(ae.result.profile.overview())
"""

from __future__ import annotations

import warnings
from datetime import datetime
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
    save_model_plots,
    save_voting_plots,
    save_trial_trajectory_plot,
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
    render_full_report,
)
from tabularx.schema.estimator import AutoResult


class AutoEstimator:
    """一键建模：配置 → 训练 → 解释 → 公平性评估。

    Attributes:
        task: 任务类型 "auto"（默认，自动推断）| "classification" | "regression"。
        output_dir: 报告和工件保存目录。
        pdf_report: 是否额外生成 PDF 版完整报告（需安装 weasyprint）。
        param: TabularXParams 实例，配置各阶段参数。
        result_: fit() 完成后自动生成的 AutoResult 实例。
    """

    def __init__(
        self,
        task: str = "auto",
        output_dir: str | Path = "./tabularx_output",
        pdf_report: bool = False,
    ):
        self.task = task
        self.output_dir = Path(output_dir)
        self.pdf_report = pdf_report
        self.param = TabularXParams()
        self.result_: AutoResult | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: list[str] | None = None,
    ) -> AutoEstimator:
        """执行完整流水线：profile → split → preprocess → search → [explain] → [fairness]。
        """
        profile = build_profile(X, y, task_type=self.task, config=self.param.profile)
        self.task = profile.target.task_type

        split = split_data(X, y, profile=profile, config=self.param.split)
        preprocessed = preprocess(X, y, profile=profile, split_result=split,
                                  config=self.param.preprocess)
        searched = search_models(preprocessed_data=preprocessed, profile=profile,
                                 config=self.param.model_search)

        if searched.best_model_name:
            voting = build_voting(search_result=searched, preprocessed_data=preprocessed,
                                  task_type=self.task)
            if voting is not None:
                searched.ensemble_result = voting

        shap = None
        if searched.best_model_name:
            shap = self._run_shap(searched, preprocessed)

        fairness = None
        if sensitive_features and searched.best_model_name:
            fairness = self._run_fairness(searched, preprocessed, sensitive_features)

        self.result_ = AutoResult(
            profile=profile, split=split, preprocessed=preprocessed,
            searched=searched, shap=shap, fairness=fairness,
            output_dir=self.output_dir,
        )
        self._save_all()
        return self

    # ── 内部：运行 SHAP ─────────────────────────────────────

    def _run_shap(self, searched, preprocessed):
        registry = get_default_registry()
        registry_task = ("classification" if self.task in ("binary", "multiclass")
                         else "regression")
        best_spec = registry.get(searched.best_model_name, registry_task)
        X_train_full = pd.concat([preprocessed.X_train, preprocessed.X_valid])
        y_train_full = pd.concat([preprocessed.y_train, preprocessed.y_valid])

        assets_dir = self.output_dir / "reports" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        return explain_model(
            spec=best_spec, best_params=searched.best_model_params,
            X_train=X_train_full, y_train=y_train_full,
            X_explain=preprocessed.X_test,
            feature_names=preprocessed.feature_names_out,
            task_type=self.task,
            output_dir=str(assets_dir),
            config=self.param.shap,
        )

    # ── 内部：运行公平性评估 ───────────────────────────────

    def _run_fairness(self, searched, preprocessed, sensitive_features):
        registry = get_default_registry()
        registry_task = ("classification" if self.task in ("binary", "multiclass")
                         else "regression")
        best_spec = registry.get(searched.best_model_name, registry_task)
        model = best_spec.factory(**searched.best_model_params,
                                  **(best_spec.default_params or {}))
        X_train_full = pd.concat([preprocessed.X_train, preprocessed.X_valid])
        y_train_full = pd.concat([preprocessed.y_train, preprocessed.y_valid])
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(preprocessed.X_test)
        y_prob = None
        if best_spec.supports_predict_proba and self.task == "binary":
            y_prob = model.predict_proba(preprocessed.X_test)[:, 1]
        return evaluate_fairness(
            X_test=preprocessed.X_test, y_test=preprocessed.y_test,
            y_pred=y_pred, y_prob=y_prob,
            column_mappings=preprocessed.column_mappings,
            sensitive_features=sensitive_features,
            task_type=self.task, config=self.param.fairness,
        )

    # ── 内部：重建最优模型 ─────────────────────────────────

    def _get_best_model(self, searched, preprocessed):
        if not searched.best_model_name:
            return None
        registry = get_default_registry()
        registry_task = ("classification" if self.task in ("binary", "multiclass")
                         else "regression")
        best_spec = registry.get(searched.best_model_name, registry_task)
        model = best_spec.factory(**searched.best_model_params,
                                  **(best_spec.default_params or {}))
        X_train_full = pd.concat([preprocessed.X_train, preprocessed.X_valid])
        y_train_full = pd.concat([preprocessed.y_train, preprocessed.y_valid])
        model.fit(X_train_full, y_train_full)
        return model, best_spec

    # ── 内部：展平 explanation 模板数据 ────────────────────

    def _build_explanation_data(self):
        result = self.result_
        shap = result.shap
        fairness = result.fairness

        data = {
            "task_type": self.task,
            "target_name": _resolve_target_name(result.preprocessed),
            "best_model_name": result.searched.best_model_name or "",
            "shap_available": False,
            "shap_artifact_dir": "assets",
            "shap_n_explain_samples": 0,
            "shap_n_features": 0,
            "shap_top_k_explained_ratio": 0.0,
            "shap_has_summary": False,
            "shap_has_importance": False,
            "shap_has_waterfall": False,
            "shap_feature_importance": {},
            "shap_top_k_stats": {},
            "shap_sum_total": 0.0,
            "shap_error": None,
            "fairness_enabled": False,
            "fairness_error": None,
            "fairness_sensitive_features": [],
            "fairness_feature_results": [],
        }

        if shap is not None and shap.error is None:
            data["shap_available"] = True
            data["shap_n_explain_samples"] = shap.n_explain_samples
            data["shap_n_features"] = shap.n_features
            data["shap_top_k_explained_ratio"] = shap.top_k_explained_ratio
            data["shap_feature_importance"] = shap.feature_importance
            data["shap_top_k_stats"] = shap.top_k_stats
            data["shap_sum_total"] = shap.shap_sum_total
            data["shap_has_summary"] = "shap_summary" in shap.artifact_paths
            data["shap_has_importance"] = "shap_importance" in shap.artifact_paths
            data["shap_has_waterfall"] = "shap_waterfall" in shap.artifact_paths
            data["shap_error"] = shap.error
        elif shap is not None and shap.error:
            data["shap_error"] = shap.error

        if fairness is not None and fairness.enabled:
            data["fairness_enabled"] = True
            data["fairness_sensitive_features"] = fairness.sensitive_features
            data["fairness_feature_results"] = fairness.feature_results
            data["fairness_error"] = fairness.error

        return data

    # ── 内部：构建 model_search 报告数据 ──────────────────

    def _build_model_search_data(self):
        result = self.result_
        searched = result.searched
        is_cls = self.task in ("binary", "multiclass")

        best_run = next((r for r in searched.succeeded_results
                         if r.model_name == searched.best_model_name), None)
        has_cm = bool(best_run and "confusion_matrix" in best_run.artifact_paths)
        has_roc = bool(best_run and "roc_curve" in best_run.artifact_paths)

        data = dict(searched.__dict__)
        data["is_classification"] = is_cls
        data["artifact_dir"] = "assets"
        data["has_confusion_matrix"] = has_cm
        data["has_roc_curve"] = has_roc
        data["ranking"] = searched.ranking
        data["succeeded_results"] = searched.succeeded_results
        data["failed_results"] = searched.failed_results
        data["shap_available"] = (result.shap is not None and result.shap.error is None)
        return data

    # ── 内部：保存全部输出 ─────────────────────────────────

    def _save_all(self) -> None:
        result = self.result_
        output_dir = self.output_dir
        reports_dir = output_dir / "reports"
        models_dir = reports_dir / "models"
        assets_dir = reports_dir / "assets"
        for d in [reports_dir, models_dir, assets_dir]:
            d.mkdir(parents=True, exist_ok=True)

        target_name = _resolve_target_name(result.preprocessed)
        is_cls = self.task in ("binary", "multiclass")
        reg = get_default_registry()

        # ══════════════════════════════════════════════════════
        # 1) 模型评估图 + 调参轨迹
        # ══════════════════════════════════════════════════════
        for run in result.searched.succeeded_results:
            save_model_plots(run, assets_dir, result.searched,
                             result.preprocessed, self.task, reg)
            save_trial_trajectory_plot(run, assets_dir,
                                       result.searched.primary_metric)

        if result.searched.ensemble_result:
            save_voting_plots(result.searched.ensemble_result, assets_dir,
                              result.searched, result.preprocessed, self.task, reg)

        # ══════════════════════════════════════════════════════
        # 2) 各报告 HTML
        # ══════════════════════════════════════════════════════
        search_data = self._build_model_search_data()
        _write_html(reports_dir / "model_search.html",
                    render_model_search_report(search_data))
        _write_html(reports_dir / "profile.html",
                    render_profile_report(result.profile))
        _write_html(reports_dir / "split.html",
                    render_split_report(result.split))
        _write_html(reports_dir / "preprocess.html",
                    render_preprocess_report(result.preprocessed))

        for run in result.searched.succeeded_results:
            safe_name = run.model_name.replace(" ", "_")
            has_cm = "confusion_matrix" in run.artifact_paths
            has_roc = "roc_curve" in run.artifact_paths
            detail_data = {
                "run": run, "target_name": target_name,
                "primary_metric": result.searched.primary_metric,
                "optimization_direction": result.searched.optimization_direction,
                "is_classification": is_cls,
                "artifact_dir": "assets",
                "has_confusion_matrix": has_cm,
                "has_roc_curve": has_roc,
            }
            _write_html(models_dir / f"{safe_name}.html",
                        render_model_detail_report(detail_data))

        if result.searched.ensemble_result:
            er = result.searched.ensemble_result
            has_cm = "confusion_matrix" in er.artifact_paths
            has_roc = "roc_curve" in er.artifact_paths
            best_val = (result.searched.best_test_metrics or {}).get(
                result.searched.primary_metric, float("nan"))
            voting_data = {
                "run": er,
                "primary_metric": result.searched.primary_metric,
                "optimization_direction": result.searched.optimization_direction,
                "best_single_test": best_val,
                "is_classification": is_cls,
                "artifact_dir": "assets",
                "has_confusion_matrix": has_cm,
                "has_roc_curve": has_roc,
                "succeeded_results": result.searched.succeeded_results,
            }
            _write_html(models_dir / "voting.html", render_voting_report(voting_data))

        explanation_data = self._build_explanation_data()
        _write_html(reports_dir / "explanation.html",
                    render_explanation_report(explanation_data))

        # ══════════════════════════════════════════════════════
        # 3) 最优模型文件
        # ══════════════════════════════════════════════════════
        if result.searched.best_model_name:
            model, _ = self._get_best_model(result.searched, result.preprocessed)
            model_path = output_dir / "best_model.joblib"
            try:
                import joblib
                joblib.dump(model, model_path)
            except ImportError:
                import pickle
                with open(output_dir / "best_model.pkl", "wb") as f:
                    pickle.dump(model, f)

        # ══════════════════════════════════════════════════════
        # 4) 完整合并报告
        # ══════════════════════════════════════════════════════
        sections = []
        section_defs = [
            ("profile", "数据概况", result.profile, render_profile_report),
            ("split", "数据划分", result.split, render_split_report),
            ("preprocess", "数据预处理", result.preprocessed, render_preprocess_report),
            ("model_search", "模型搜索", search_data, render_model_search_report),
        ]

        for run in result.searched.succeeded_results:
            safe_name = run.model_name.replace(" ", "_")
            has_cm = "confusion_matrix" in run.artifact_paths
            has_roc = "roc_curve" in run.artifact_paths
            detail_data = {
                "run": run, "target_name": target_name,
                "primary_metric": result.searched.primary_metric,
                "optimization_direction": result.searched.optimization_direction,
                "is_classification": is_cls,
                "artifact_dir": "assets",
                "has_confusion_matrix": has_cm,
                "has_roc_curve": has_roc,
            }
            section_defs.append(
                ("model_" + safe_name, f"模型：{run.model_name}",
                 detail_data, render_model_detail_report))

        if result.searched.ensemble_result:
            er = result.searched.ensemble_result
            has_cm = "confusion_matrix" in er.artifact_paths
            has_roc = "roc_curve" in er.artifact_paths
            best_val = (result.searched.best_test_metrics or {}).get(
                result.searched.primary_metric, float("nan"))
            voting_data = {
                "run": er,
                "primary_metric": result.searched.primary_metric,
                "optimization_direction": result.searched.optimization_direction,
                "best_single_test": best_val,
                "is_classification": is_cls,
                "artifact_dir": "assets",
                "has_confusion_matrix": has_cm,
                "has_roc_curve": has_roc,
                "succeeded_results": result.searched.succeeded_results,
            }
            section_defs.append(
                ("voting", "Voting 组合模型", voting_data, render_voting_report))

        section_defs.append(
            ("explanation", "解释与公平性", explanation_data, render_explanation_report))

        for sec_id, sec_title, sec_data, sec_renderer in section_defs:
            sections.append({
                "id": sec_id,
                "title": sec_title,
                "body": sec_renderer(sec_data, body_only=True),
            })

        dataset_info = {}
        try:
            dataset_info["n_rows"] = result.profile.n_rows
            dataset_info["n_cols"] = result.profile.n_cols
            if result.profile.target.n_classes is not None:
                dataset_info["n_classes"] = result.profile.target.n_classes
        except Exception:
            pass

        full_data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_type": self.task,
            "best_model_name": result.searched.best_model_name,
            "dataset_info": dataset_info,
            "sections": sections,
        }

        html_path = output_dir / "report.html"
        _write_html(html_path, render_full_report(full_data))

        if self.pdf_report:
            try:
                import weasyprint
                weasyprint.HTML(filename=str(html_path)).write_pdf(
                    output_dir / "report.pdf")
            except ImportError:
                warnings.warn(
                    "weasyprint 未安装，跳过 PDF 生成。"
                    " 通过 pip install weasyprint 安装。")

        result.report_paths = {
            "profile": str(reports_dir / "profile.html"),
            "split": str(reports_dir / "split.html"),
            "preprocess": str(reports_dir / "preprocess.html"),
            "model_search": str(reports_dir / "model_search.html"),
            "explanation": str(reports_dir / "explanation.html"),
            "full": str(html_path),
        }

    # ── 属性访问 ───────────────────────────────────────────

    @property
    def result(self) -> AutoResult:
        if self.result_ is None:
            raise RuntimeError("请先调用 fit() 方法。")
        return self.result_

    def __repr__(self) -> str:
        status = "就绪" if self.result_ is None else "已完成"
        return (f"<AutoEstimator task={self.task!r} "
                f"output_dir={self.output_dir!s} {status}>")


# ── 辅助函数 ────────────────────────────────────────────────


def _resolve_target_name(preprocessed) -> str:
    try:
        name = preprocessed.y_train.name
        if name is not None:
            return name
    except (AttributeError, IndexError):
        pass
    return "target"


def _write_html(path: Path, html: str) -> None:
    path.write_text(html, encoding="utf-8")


__all__ = ["AutoEstimator"]